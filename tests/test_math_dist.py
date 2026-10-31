# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DIST-001 (reusable scalar and joint histograms) and DIST-002 (prefix distribution calculus and exact quantiles)
(torbles/math_dist.c, math_dist_gpu.cu, math_dist.py).

Oracles, independent of the implementation: DuckDB SQL over the typed source table of the same Database (GROUP BY
with COUNT(*), COUNT(w), SUM(w) in HUGEINT, NULL as its own group per column; quantile_disc / quantile_cont; COUNT and
SUM with WHERE ranges), the engine's own SQL GROUP BY (db.sql), and Python integers / Fractions recomputed here from
the expanded rows (each value repeated once per row, or once per unit of its integer weight). Every native result is
also computed on the CUDA path (emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared for
equality with the CPU result. Seeds are fixed.
"""
import ctypes as C
import math
import subprocess
import sys
import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_dist as md
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

DDL = "CREATE TABLE f (k BIGINT, s VARCHAR, x BIGINT, y BIGINT, d DECIMAL(10,2), w BIGINT, v BIGINT)"
SCALE = {"x": 0, "y": 0, "d": 2, "w": 0, "v": 0}


def obj(values):
    return pd.Series(values, dtype=object)


def dataset(rng, n, nulls=0.15):
    """small supports with many duplicates, NULLs in every column, an all-NULL group (k = 7), an absent key value
    (k = 6), a nonnegative weight and a wide-range column"""
    k = [int(a) for a in rng.integers(0, 6, n)]
    s = [str(a) for a in rng.choice(["a", "b", "c"], n)]
    x = [int(a) for a in rng.integers(-3, 4, n)]
    y = [int(a) for a in rng.integers(0, 5, n)]
    d = [f"{int(a) / 100:.2f}" for a in rng.choice([-250, -5, 0, 125, 999], n)]
    w = [int(a) for a in rng.integers(0, 50, n)]
    v = [int(a) for a in rng.integers(-10**12, 10**12, n)]
    for col in (k, x, y, d, w, v):
        for i in np.flatnonzero(rng.random(n) < nulls):
            col[int(i)] = None
    for i in rng.choice(n, size=min(6, n), replace=False):                 # group k = 7: every value NULL
        k[int(i)], x[int(i)], y[int(i)], d[int(i)], v[int(i)] = 7, None, None, None, None
    frame = pd.DataFrame({"k": obj(k), "s": obj(s), "x": obj(x), "y": obj(y), "d": obj(d), "w": obj(w), "v": obj(v)})
    return torbles.from_ddl(DDL, {"f": frame})


def logical(name, alias="f"):
    expr, scale = f'{alias}."{name}"', SCALE.get(name, 0)
    return f"CAST({expr} * {10 ** scale} AS HUGEINT)" if scale else f"CAST({expr} AS HUGEINT)"


def cell_order(key):
    return tuple((v is None, 0 if v is None else v) for v in key)


def where(con, rows):
    if rows is None:
        return "TRUE"
    con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(np.asarray(rows)).astype(np.int64)}))
    return "f.rowid IN (SELECT rid FROM sel_rows)"


def duck_cells(db, cols, keys, weight=None, rows=None):
    """{group key tuple: (rows, [(cell key, count[, wcount, wsum])] in cell order)} by DuckDB over the source table"""
    con = db.con
    sel = where(con, rows)
    gk = [f'f."{k}"' for k in keys]
    cells = [logical(c) for c in cols]
    agg = ["COUNT(*)"] + ([f'COUNT(f."{weight}")', f'SUM(CAST(f."{weight}" AS HUGEINT))'] if weight else [])
    sql = (f"SELECT {', '.join(gk + cells + agg)} FROM f WHERE {sel} GROUP BY " +
           ", ".join(str(i + 1) for i in range(len(gk) + len(cells))))
    out = {}
    for r in con.execute(sql).fetchall():
        g, key, a = tuple(r[:len(gk)]), tuple(None if v is None else int(v) for v in r[len(gk):len(gk) + len(cols)]), r[len(gk) + len(cols):]
        cell = (key, int(a[0])) + ((int(a[1]), 0 if a[2] is None else int(a[2])) if weight else ())
        rows_g, lst = out.get(g, (0, []))
        out[g] = (rows_g + int(a[0]), lst + [cell])
    return {g: (r, sorted(c, key=lambda t: cell_order(t[0]))) for g, (r, c) in out.items()}


def native_cells(h):
    return {row["key"]: (row["rows"], row["cells"]) for row in h.table()}


def mask(rng, n, p):
    return rng.random(n) < p


def expanded(values, weights=None):
    """the expanded multiset: each present value once per row, or once per unit of its integer weight"""
    out = []
    for i, x in enumerate(values):
        if x is None:
            continue
        rep = 1 if weights is None else weights[i]
        if rep is None or rep <= 0:
            continue
        out.extend([x] * rep)
    return sorted(out)


def same_array(u, v):
    if isinstance(u, list) and u and isinstance(u[0], np.ndarray):
        return len(u) == len(v) and all(np.array_equal(a, b) for a, b in zip(u, v))
    return np.array_equal(np.asarray(u, dtype=object), np.asarray(v, dtype=object))


class Base(unittest.TestCase):
    def both(self, scope, cols, **kw):
        """the CPU histogram, checked equal (natively and array by array) to the CUDA one (emulated, scrambled)"""
        cpu = md.histogram(scope, cols, **kw)
        gpu = md.histogram(scope, cols, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu), (cols, kw))
        a, b = cpu.arrays(), gpu.arrays()
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(same_array(a[key], b[key]), key)
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    def assert_absent_empty(self, h):
        for g in range(h.groups):
            st = h.group(g)
            if not st["present"]:
                self.assertEqual((st["rows"], st["cells"]), (0, []), g)


# ====================================================================== DIST-001
class Histograms(Base):
    # ------------------------------------------------------------------ acceptance 1: grouped counts
    def test_dist001_grouped_counts_match_duckdb_with_duplicates_and_null_categories(self):
        rng = np.random.default_rng(1001)
        db = dataset(rng, 2500)
        t = db["f"]
        for keys in (["k"], ["k", "s"], []):
            for rows in (None, mask(rng, t.n, 0.6)):
                scope = mbk.Scope.from_table(t, ["x", "y", "d", "w", "v"], group_by=keys, rows=rows)
                for cols in (["x"], ["d"], ["x", "y"], ["x", "y", "d"], ["v"], ["v", "x"]):
                    for weight in (None, "w"):
                        expect = duck_cells(db, cols, keys, weight, rows)
                        for repr_ in ("sparse", "dense"):
                            if "v" in cols:                       # a 2 * 10^12 wide domain: dense and the device
                                with self.assertRaises(AlgebraBudgetError):   # counting scratch are refused
                                    md.histogram(scope, cols, weight=weight, repr=repr_, backend="cuda")
                                if repr_ == "dense":
                                    continue
                                h = md.histogram(scope, cols, weight=weight)  # the CPU sorts instead
                            else:
                                h = self.both(scope, cols, weight=weight, repr=repr_)
                            self.assertEqual(native_cells(h), expect, (keys, cols, weight, repr_))
                            self.assertEqual(h.present, len(expect))
                            self.assertEqual(h.rows, scope.selected)
                            self.assert_absent_empty(h)
                        if "v" in cols:
                            for strategy in ("sort", "count"):
                                if strategy == "count":
                                    with self.assertRaises(AlgebraBudgetError):    # the counting scratch is too big
                                        md.histogram(scope, cols, weight=weight, strategy=strategy)
                                    continue
                                self.assertEqual(native_cells(md.histogram(scope, cols, weight=weight,
                                                                           strategy=strategy)), expect)
        # the engine's own GROUP BY (db.sql) agrees on counts, NULL groups included
        scope = mbk.Scope.from_table(t, ["x", "y", "w"], group_by=["k"])
        h = md.histogram(scope, ["x", "y"], weight="w")
        rel = db.sql('SELECT k, x, y, COUNT(*), COUNT(w), SUM(w) FROM f GROUP BY k, x, y')
        engine = {}
        for r in rel.rows:
            g, key = (r[0],), (r[1], r[2])
            engine.setdefault(g, []).append((key, int(r[3]), int(r[4]), 0 if r[5] is None else int(r[5])))
        got = {g: cells for g, (_, cells) in native_cells(h).items()}
        self.assertEqual(got, {g: sorted(c, key=lambda c: cell_order(c[0])) for g, c in engine.items()})

    def test_dist001_null_category_is_never_merged_and_presence_is_explicit(self):
        # value 0 and NULL are distinct cells; joint NULL patterns are distinct; an all-NULL group is present
        x = [0, None, 0, None, 5, 5, 5, None, 0, 2]
        y = [1, 1, None, None, 1, 1, None, None, 1, 3]
        w = [3, 4, None, 6, 1, 1, 1, 2, None, 9]
        g = [0, 0, 0, 0, 0, 2, 2, 3, 3, 3]
        valid = {n: [v is not None for v in col] for n, col in (("x", x), ("y", y), ("w", w))}
        vals = {n: [0 if v is None else v for v in col] for n, col in (("x", x), ("y", y), ("w", w))}
        scope = mbk.Scope.from_arrays(vals, valid=valid, groups=g, ngroups=5)
        for repr_ in ("sparse", "dense"):
            h = self.both(scope, ["x", "y"], weight="w", repr=repr_)
            self.assertEqual(h.cells(0), [((0, 1), 1, 1, 3), ((0, None), 1, 0, 0), ((5, 1), 1, 1, 1),
                                          ((None, 1), 1, 1, 4), ((None, None), 1, 1, 6)])
            self.assertEqual(h.group(1), dict(present=False, rows=0, cells=[]))           # absent
            self.assertEqual(h.group(4), dict(present=False, rows=0, cells=[]))
            self.assertEqual(h.group(2)["cells"], [((5, 1), 1, 1, 1), ((5, None), 1, 1, 1)])
            one = self.both(scope, ["x"], weight="w", repr=repr_)
            self.assertEqual(one.cells(3), [((0,), 1, 0, 0), ((2,), 1, 1, 9), ((None,), 1, 1, 2)])
            self.assertEqual(one.cells(0), [((0,), 2, 1, 3), ((5,), 1, 1, 1), ((None,), 2, 2, 10)])
        # a group whose values are all NULL is present, with its rows in the NULL cell
        scope = mbk.Scope.from_arrays({"x": [1, 2, 3, 4]}, valid={"x": [False, False, True, True]}, groups=[0, 0, 1, 1],
                                      ngroups=3)
        for repr_ in ("sparse", "dense"):
            h = self.both(scope, ["x"], repr=repr_)
            self.assertEqual(h.group(0), dict(present=True, rows=2, cells=[((None,), 2)]))
            self.assertEqual(h.group(2), dict(present=False, rows=0, cells=[]))
            self.assertEqual(h.present, 2)
        # duplicates collapse into one cell with their multiplicity
        scope = mbk.Scope.from_arrays({"x": [7] * 50 + [-7] * 3})
        for repr_ in ("sparse", "dense"):
            self.assertEqual(self.both(scope, ["x"], repr=repr_).cells(0), [((-7,), 3), ((7,), 50)])

    def test_dist001_empty_selections(self):
        rng = np.random.default_rng(1002)
        db = dataset(rng, 600)
        t = db["f"]
        for keys in (["k"], [], ["k", "s"]):
            scope = mbk.Scope.from_table(t, ["x", "y", "w"], group_by=keys, rows=np.zeros(t.n, bool))
            for repr_ in ("sparse", "dense"):
                for weight in (None, "w"):
                    h = self.both(scope, ["x", "y"], weight=weight, repr=repr_)
                    self.assertEqual((h.present, h.rows, h.table()), (0, 0, []))
                    self.assertEqual(h.info["occupied"], 0)
                    self.assertEqual(native_cells(h), duck_cells(db, ["x", "y"], keys, weight, np.zeros(t.n, bool)))
                    self.assert_absent_empty(h)
        empty = mbk.Scope.from_arrays({"x": np.zeros(0, np.int64)})
        for repr_ in ("sparse", "dense"):
            h = self.both(empty, ["x"], repr=repr_)
            self.assertEqual((h.groups, h.present, h.info["cells"]), (1, 0, 0))
            self.assertEqual(h.to_sparse().info["cells"], 0)
            self.assertEqual(h.to_dense().info["cells"], 0)
        p = md.histogram(empty, ["x"]).prefix()
        self.assertEqual(p.cdf(0), dict(mass=0, total=0, fraction=None))
        self.assertIsNone(p.quantile(Fraction(1, 2)))

    # ------------------------------------------------------------------ acceptance 2: conversions and joint budgets
    def test_dist001_dense_sparse_conversions_preserve_every_cell(self):
        rng = np.random.default_rng(1003)
        n = 4000
        cols = {"a": rng.integers(-20, 21, n), "b": rng.integers(0, 6, n), "c": rng.integers(100, 104, n),
                "w": rng.integers(-1000, 1000, n)}
        valid = {"a": mask(rng, n, 0.9), "b": mask(rng, n, 0.95), "w": mask(rng, n, 0.8)}
        groups = rng.integers(0, 9, n)
        groups[groups == 4] = 5                                      # group 4 absent
        scope = mbk.Scope.from_arrays(cols, valid=valid, groups=groups, ngroups=9, rows=mask(rng, n, 0.8))
        for names in (["a"], ["a", "b"], ["c", "a", "b"], ["b", "c"]):
            for weight in (None, "w"):
                sparse = self.both(scope, names, weight=weight)
                dense = self.both(scope, names, weight=weight, repr="dense")
                self.assertTrue(dense.to_sparse().equal(sparse), names)
                self.assertTrue(sparse.to_dense(dense.info["domain"]).equal(dense), names)
                for g in range(scope.groups):                       # every represented cell, cell by cell
                    self.assertEqual(sparse.cells(g), dense.cells(g))
                hull = sparse.to_dense()                            # the hull of the occupied cells
                self.assertTrue(hull.to_sparse().equal(sparse))
                wider = [(lo - 3, hi + 2, True) for lo, hi, _ in dense.info["domain"]]
                wide = sparse.to_dense(wider)
                self.assertEqual(wide.info["cells"], scope.groups * math.prod(hi - lo + 2 for lo, hi, _ in wider))
                self.assertTrue(wide.to_sparse().equal(sparse))
                self.assertTrue(wide.to_dense(dense.info["domain"]).equal(dense))   # dense to dense, narrower
                # a domain that would drop an occupied cell is refused, never truncated
                lo, hi, null = dense.info["domain"][0]
                for bad in ([(lo + 1, hi, null)] + dense.info["domain"][1:], [(lo, hi - 1, null)] + dense.info["domain"][1:]):
                    with self.assertRaises(DomainError):
                        sparse.to_dense(bad)
                    with self.assertRaises(DomainError):
                        dense.to_dense(bad)
                if any(n in valid for n in names):
                    no_null = [(a, b, False) if names[i] in valid else (a, b, c)
                               for i, (a, b, c) in enumerate(dense.info["domain"])]
                    with self.assertRaises(DomainError):
                        sparse.to_dense(no_null)
                # an explicit domain at build time: the same rule before anything is read
                built = self.both(scope, names, weight=weight, repr="dense", domain=wider)
                self.assertTrue(built.equal(wide))
                with self.assertRaises(DomainError):
                    md.histogram(scope, names, weight=weight, repr="dense",
                                 domain=[(lo + 1, hi, True)] + dense.info["domain"][1:])
        # structural zeros: a dense cell without rows is not a sparse cell
        s = mbk.Scope.from_arrays({"x": [0, 0, 9]})
        d = md.histogram(s, ["x"], repr="dense")
        self.assertEqual((d.info["cells"], d.info["occupied"], d.to_sparse().info["cells"]), (10, 2, 2))

    def test_dist001_over_budget_joint_domains_are_refused_at_the_limit(self):
        rng = np.random.default_rng(1004)
        n = 3000
        cols = {"a": rng.integers(0, 64, n), "b": rng.integers(0, 64, n), "c": rng.integers(0, 64, n),
                "w": rng.integers(0, 9, n)}
        scope = mbk.Scope.from_arrays(cols, valid={"c": mask(rng, n, 0.9)}, groups=rng.integers(0, 3, n), ngroups=3)
        sparse = md.histogram(scope, ["a", "b", "c"], weight="w")
        cells = 64 * 64 * 65                                         # c has a NULL slot
        for backend in ("cpu", "cuda"):
            needs = md.need(scope, ["a", "b", "c"], weight="w", repr="dense", backend=backend)
            self.assertEqual((needs["support"], needs["states"], needs["variables"]), (cells, 3 * cells, 4))
            ref = md.histogram(scope, ["a", "b", "c"], weight="w", repr="dense")
            for resource, value in needs.items():
                ok = md.histogram(scope, ["a", "b", "c"], weight="w", repr="dense", backend=backend,
                                  budget=Budget(**{resource: value}))
                self.assertTrue(ok.equal(ref))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    md.histogram(scope, ["a", "b", "c"], weight="w", repr="dense", backend=backend,
                                 budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        # sparse -> dense of a joint domain: support, states and bytes at the limit and one past
        dense = sparse.to_dense()
        self.assertTrue(dense.equal(md.histogram(scope, ["a", "b", "c"], weight="w", repr="dense")))
        for resource, value in (("support", cells), ("states", 3 * cells), ("bytes", dense.bytes)):
            self.assertTrue(sparse.to_dense(budget=Budget(**{resource: value})).equal(dense))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                sparse.to_dense(budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        # a joint domain past 64 bits is refused whatever the budget; the default support budget refuses a big one
        wide = mbk.Scope.from_arrays({"p": [-(1 << 62), 1 << 62], "q": [-(1 << 62), 1 << 62]})
        unlimited = Budget(**{r: (1 << 64) - 1 for r in ("support", "states", "bytes")})
        for backend in ("cpu", "cuda"):
            with self.assertRaises(AlgebraBudgetError):
                md.histogram(wide, ["p", "q"], repr="dense", backend=backend, budget=unlimited)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            md.histogram(scope, ["a", "b", "c"], repr="dense", domain=[(0, 1023, 0), (0, 1023, 0), (0, 63, 1)])
        self.assertEqual(ctx.exception.resource, "support")
        sp = md.histogram(wide, ["p", "q"])                       # sparse: two cells, any domain
        self.assertEqual(sp.cells(0), [((-(1 << 62), -(1 << 62)), 1), (((1 << 62), (1 << 62)), 1)])
        with self.assertRaises(AlgebraBudgetError):
            sp.to_dense(budget=unlimited)
        # the sparse build: the scratch is checked first, then scratch + final state once the cells are counted
        for strategy in ("count", "sort"):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                md.histogram(scope, ["a", "b"], strategy=strategy, budget=Budget(bytes=0))
            first = ctx.exception.need
            with self.assertRaises(AlgebraBudgetError) as ctx:
                md.histogram(scope, ["a", "b"], strategy=strategy, budget=Budget(bytes=first))
            total = ctx.exception.need
            self.assertGreater(total, first)
            ok = md.histogram(scope, ["a", "b"], strategy=strategy, budget=Budget(bytes=total))
            self.assertTrue(ok.equal(md.histogram(scope, ["a", "b"])))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                md.histogram(scope, ["a", "b"], strategy=strategy, budget=Budget(bytes=total - 1))
            self.assertEqual(ctx.exception.need, total)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                md.histogram(scope, ["a", "b"], strategy=strategy, budget=Budget(support=ok.info["max_cells"] - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("support", ok.info["max_cells"]))
        # the CUDA sparse build counts in a device scratch over the scope domain: refused past its budget
        dev = md.need(scope, ["a", "b"], backend="cuda")
        self.assertEqual(dev["states"], 3 * 64 * 64)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            md.histogram(scope, ["a", "b"], backend="cuda", budget=Budget(states=dev["states"] - 1))
        self.assertEqual(ctx.exception.resource, "states")
        self.assertTrue(md.histogram(scope, ["a", "b"], backend="cuda", budget=Budget(states=dev["states"]))
                        .equal(md.histogram(scope, ["a", "b"])))

    # ------------------------------------------------------------------ CUDA, merges, rollups, marginals
    def test_dist001_cuda_equals_cpu_on_many_groups_scrambled(self):
        rng = np.random.default_rng(1005)
        n = 20000
        cols = {"a": rng.integers(-100, 100, n), "b": rng.integers(0, 7, n),
                "w": rng.integers(-(1 << 62), 1 << 62, n)}              # weight sums past 2^64: exact 128 bits
        valid = {"a": mask(rng, n, 0.9), "w": mask(rng, n, 0.85)}
        for G, order in ((1, None), (13, "scattered"), (1000, "scattered"), (1000, "clustered")):
            groups = None if G == 1 else rng.integers(0, G, n)
            if order == "clustered":
                groups = np.sort(groups)
            scope = mbk.Scope.from_arrays(cols, valid=valid, groups=groups, ngroups=G if G > 1 else None,
                                          rows=mask(rng, n, 0.8))
            for names in (["a"], ["a", "b"], ["b"]):
                for repr_ in ("sparse", "dense"):
                    h = self.both(scope, names, weight="w", repr=repr_)
                    self.assertEqual(h.rows, scope.selected)
                for strategy in ("count", "sort"):
                    self.assertTrue(md.histogram(scope, names, weight="w", strategy=strategy).equal(
                        md.histogram(scope, names, weight="w")))
        # exact 128-bit weight sums against Python integers
        scope = mbk.Scope.from_arrays(cols, valid=valid)
        h = self.both(scope, ["b"], weight="w")
        for key, count, wcount, wsum in h.cells(0):
            sel = [i for i in range(n) if (cols["b"][i] if key[0] is not None else None) == key[0]]
            self.assertEqual(count, len(sel))
            self.assertEqual(wcount, sum(1 for i in sel if valid["w"][i]))
            self.assertEqual(wsum, sum(int(cols["w"][i]) for i in sel if valid["w"][i]))
        self.assertTrue(any(abs(c[3]) > (1 << 64) for c in h.cells(0)))

    def test_dist001_chunked_builders_merges_and_rollups(self):
        rng = np.random.default_rng(1006)
        db = dataset(rng, 3000)
        t = db["f"]
        rows = mask(rng, t.n, 0.75)
        scope = mbk.Scope.from_table(t, ["x", "y", "d", "w", "v"], group_by=["k", "s"], rows=rows)
        for cols, repr_, strategy in ((["x", "y"], "sparse", "count"), (["x", "y"], "sparse", "sort"),
                                      (["x", "d"], "dense", "auto"), (["v"], "sparse", "auto")):
            one = md.histogram(scope, cols, weight="w", repr=repr_, strategy=strategy)
            for trial in range(3):
                cuts = sorted(set(int(c) for c in rng.integers(0, t.n, int(rng.integers(1, 25))))) + [t.n]
                ranges = list(zip([0] + cuts[:-1], cuts))
                parts = [md.Builder(scope, cols, weight="w", repr=repr_, strategy=strategy) for _ in range(3)]
                for i, r in enumerate(rng.permutation(len(ranges))):
                    parts[i % 3].accumulate(*ranges[int(r)])
                parts[0].merge(parts[1]).merge(parts[2])
                self.assertTrue(parts[0].finish().equal(one), (cols, repr_, strategy, trial))
            b = md.Builder(scope, cols, weight="w", repr=repr_, strategy=strategy)
            b.accumulate(0, 100)
            with self.assertRaises(ValueError):
                b.accumulate(50, 150)                                # a row is counted once
            with self.assertRaises(ValueError):
                b.finish()                                           # rows 100.. are not covered
            other = md.Builder(scope, cols, weight="w", repr=repr_, strategy=strategy).accumulate(99, 200)
            with self.assertRaises(ValueError):
                b.merge(other)                                       # overlapping coverage
            with self.assertRaises(ValueError):
                b.merge(md.Builder(scope, cols, repr=repr_, strategy=strategy))   # another spec (no weight)
            twin = mbk.Scope.from_table(t, ["x", "y", "d", "w", "v"], group_by=["k", "s"], rows=rows.copy())
            rest = md.Builder(twin, cols, weight="w", repr=repr_, strategy=strategy).accumulate(100, t.n)
            self.assertTrue(b.merge(rest).finish().equal(one))       # an equal descriptor from another scope object
        # merge of finished states: partitions of the rows (two scopes) add up to the whole
        part = mask(rng, t.n, 0.5)
        for repr_ in ("sparse", "dense"):
            whole = md.histogram(mbk.Scope.from_table(t, ["x", "y", "w"], group_by=["k"]), ["x", "y"], weight="w",
                                 repr=repr_)
            domain = whole.info["domain"]
            a = md.histogram(mbk.Scope.from_table(t, ["x", "y", "w"], group_by=["k"], rows=part), ["x", "y"],
                             weight="w", repr=repr_, domain=domain)
            b = md.histogram(mbk.Scope.from_table(t, ["x", "y", "w"], group_by=["k"], rows=~part), ["x", "y"],
                             weight="w", repr=repr_, domain=domain)
            merged = a.merge(b)
            self.assertTrue(merged.equal(whole), repr_)
            self.assertFalse(merged.info["scoped"])
            if repr_ == "dense":                                     # dense states merge over one domain only
                c = md.histogram(mbk.Scope.from_table(t, ["x", "y", "w"], group_by=["k"], rows=~part), ["x", "y"],
                                 weight="w", repr="dense")
                if c.info["domain"] != domain:
                    with self.assertRaises(DomainError):
                        a.merge(c)
            with self.assertRaises(ValueError):
                a.merge(md.histogram(mbk.Scope.from_table(t, ["x", "y"], group_by=["k"]), ["x", "y"], repr=repr_))
        # rollup along the scope's refinement map equals the coarse build; presence is kept
        for fine_keys, coarse_keys in ((["k", "s"], ["k"]), (["k", "s"], ["s"]), (["k"], [])):
            fine = mbk.Scope.from_table(t, ["x", "d", "w"], group_by=fine_keys, rows=rows)
            coarse = mbk.Scope.from_table(t, ["x", "d", "w"], group_by=coarse_keys, rows=rows)
            for repr_ in ("sparse", "dense"):
                direct = md.histogram(coarse, ["x", "d"], weight="w", repr=repr_)
                up = md.histogram(fine, ["x", "d"], weight="w", repr=repr_).rollup(coarse)
                self.assertTrue(up.equal(direct), (fine_keys, coarse_keys, repr_))
                self.assertTrue(up.info["scoped"])
                self.assertEqual(native_cells(up), duck_cells(db, ["x", "d"], coarse_keys, "w", rows))
                explicit = md.histogram(fine, ["x", "d"], weight="w", repr=repr_).rollup(
                    map=fine.rollup_map(coarse), groups=coarse.groups)
                self.assertTrue(explicit.equal(direct))
                self.assertFalse(explicit.info["scoped"])
        fine = mbk.Scope.from_table(t, ["x", "w"], group_by=["k", "s"], rows=rows)
        coarse = mbk.Scope.from_table(t, ["x", "w"], group_by=["k"], rows=rows)
        up = md.histogram(fine, ["x"]).rollup(coarse)
        by_key = {coarse.key_values(g): up.group(g) for g in range(coarse.groups)}
        self.assertTrue(by_key[(7,)]["present"] and by_key[(7,)]["cells"] == [((None,), by_key[(7,)]["rows"])])
        self.assertEqual(by_key[(6,)], dict(present=False, rows=0, cells=[]))
        with self.assertRaises(DomainError):
            md.histogram(coarse, ["x"]).rollup(fine)                 # not a refinement

    def test_dist001_marginals_equal_direct_builds(self):
        rng = np.random.default_rng(1007)
        db = dataset(rng, 2000)
        scope = mbk.Scope.from_table(db["f"], ["x", "y", "d", "w"], group_by=["k"], rows=mask(rng, db["f"].n, 0.7))
        for repr_ in ("sparse", "dense"):
            joint = md.histogram(scope, ["x", "y", "d"], weight="w", repr=repr_)
            for keep in (["x"], ["y"], ["d"], ["x", "d"], ["y", "d"], ["x", "y"], ["x", "y", "d"]):
                m = joint.marginal(keep)
                self.assertTrue(m.equal(md.histogram(scope, keep, weight="w", repr=repr_)), (repr_, keep))
                self.assertEqual(m.names, tuple(keep))
                self.assertEqual(m.info["scans"], 0)
        with self.assertRaises(ValueError):
            joint.marginal(["w"])

    # ------------------------------------------------------------------ exactness, adoption, refusals
    def test_dist001_wide_values_scales_and_adoption_checks(self):
        # int64 extremes (the SORT strategy), decimals keep their scale, self-extending columns
        ext = [-(1 << 63), (1 << 63) - 1, 0, (1 << 63) - 1, -(1 << 63) + 1]
        scope = mbk.Scope.from_arrays({"e": ext, "m": [1, 2, 3, 4, 5]}, scales={"m": 3})
        h = md.histogram(scope, ["e"])
        self.assertEqual(h.info["strategy"], "sort")
        with self.assertRaises(AlgebraBudgetError):                  # the device counting scratch has 2^64 cells
            md.histogram(scope, ["e"], backend="cuda")
        self.assertEqual(h.cells(0), [((-(1 << 63),), 1), ((-(1 << 63) + 1,), 1), ((0,), 1), (((1 << 63) - 1,), 2)])
        self.assertEqual(md.histogram(scope, ["m"]).info["scales"], (3,))
        rng = np.random.default_rng(1008)
        n = 300
        big = [int(a) for a in rng.integers(-(1 << 40), 1 << 40, n)]
        for i in rng.choice(n, 30, replace=False):
            big[int(i)] = None
        gs = [int(a) for a in rng.integers(0, 3, n)]
        db = torbles.from_ddl("CREATE TABLE f (g BIGINT, b BIGINT)", {"f": pd.DataFrame({"g": obj(gs), "b": obj(big)})})
        t = db["f"]
        self.assertIn("b", t.wide)                                   # a self-extending (64-bit) column
        sc = mbk.Scope.from_table(t, ["b"], group_by=["g"])
        hb = md.histogram(sc, ["b"])
        self.assertEqual(native_cells(hb), duck_cells(db, ["b"], ["g"]))
        expect = {}
        for gi, bi in zip(gs, big):
            expect.setdefault((gi,), {}).setdefault(bi, 0)
            expect[(gi,)][bi] += 1
        for row in hb.table():
            self.assertEqual(row["cells"], sorted([((b,), c) for b, c in expect[row["key"]].items()],
                                                  key=lambda c: cell_order(c[0])))
        # adoption: states that do not describe the scope are refused
        s = mbk.Scope.from_arrays({"x": [1, 2, 2, 5]}, valid={"x": [True, True, True, False]}, groups=[0, 0, 1, 1],
                                  ngroups=2)
        spec, _ = md._spec(s, ["x"], None, "sparse", None, "auto")

        def adopt(rows, offset, values, nulls, count):
            st = md._States()
            keep = [np.asarray(rows, np.uint64), np.asarray(offset, np.uint64), np.asarray(values, np.int64),
                    np.asarray(nulls, np.uint8), np.asarray(count, np.uint64)]
            st.ngroups, st.ncells, st.capacity, st.k, st.repr = 2, len(count), len(count), 1, 2
            st.rows, st.offset, st.nulls, st.count = keep[0].ctypes.data, keep[1].ctypes.data, keep[3].ctypes.data, keep[4].ctypes.data
            st.value[0] = keep[2].ctypes.data
            error, refusal = C.c_int32(), md._Refusal()
            hd = md._call("tr_dist_hist_from_states", s._handle, C.byref(spec), C.byref(st), None, C.byref(refusal), C.byref(error))
            if hd:
                return md.Histogram._wrap(hd, s, ("x",), None, "adopted", "test")
            return error.value
        good = adopt([2, 2], [0, 2, 4], [1, 2, 2, 0], [0, 0, 0, 1], [1, 1, 1, 1])
        self.assertTrue(isinstance(good, md.Histogram) and good.equal(md.histogram(s, ["x"])))
        self.assertEqual(adopt([2, 1], [0, 2, 4], [1, 2, 2, 0], [0, 0, 0, 1], [1, 1, 1, 1]), -1)   # rows != cells
        self.assertEqual(adopt([2, 2], [0, 2, 4], [2, 1, 2, 0], [0, 0, 0, 1], [1, 1, 1, 1]), -1)   # out of order
        self.assertEqual(adopt([2, 2], [0, 2, 4], [1, 2, 2, 7], [0, 0, 0, 1], [1, 1, 1, 1]), -1)   # NULL with a value
        self.assertEqual(adopt([2, 2], [0, 2, 4], [1, 9, 2, 0], [0, 0, 0, 1], [1, 1, 1, 1]), -1)   # outside the domain
        self.assertEqual(adopt([2, 2], [0, 1, 4], [1, 2, 2, 0], [0, 0, 0, 1], [1, 1, 1, 1]), -1)   # offsets
        self.assertEqual(adopt([3, 2], [0, 2, 4], [1, 2, 2, 0], [0, 0, 0, 1], [2, 1, 1, 1]), -1)   # total rows

    def test_dist001_staleness_invalid_inputs_and_opt_in_import(self):
        rng = np.random.default_rng(1009)
        db = dataset(rng, 400)
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["x", "w"], group_by=["k"])
        h = md.histogram(scope, ["x"])
        t.flip(0, 0)
        for fn in (lambda: md.histogram(scope, ["x"]), lambda: md.histogram(scope, ["x"], backend="cuda"),
                   lambda: md.Builder(scope, ["x"]), lambda: h.rollup(mbk.Scope.from_table(t, ["x", "w"]))):
            with self.assertRaises(StaleStateError):
                fn()
        self.assertEqual(h.cells(0), h.cells(0))                     # a built histogram stays readable
        s = mbk.Scope.from_arrays({"x": [1, 2], "y": [3, 4], "z": [5, 6], "u": [7, 8]})
        for bad in (lambda: md.histogram(s, []), lambda: md.histogram(s, ["x", "y", "z", "u"]),
                    lambda: md.histogram(s, ["q"]), lambda: md.histogram(s, ["x"], weight="q"),
                    lambda: md.histogram(s, ["x"], repr="tree"), lambda: md.histogram(s, ["x"], strategy="hash"),
                    lambda: md.histogram(s, ["x"], domain=[(0, 1, 0)]), lambda: md.histogram(s, ["x"], backend="tpu"),
                    lambda: md.histogram(s, ["x"], repr="dense", domain=[(0, 1, 0), (0, 1, 0)])):
            with self.assertRaises(ValueError):
                bad()
        spec, _ = md._spec(s, ["x"], None, "sparse", None, "auto")
        spec.k = 4
        error, refusal = C.c_int32(), md._Refusal()
        self.assertIsNone(md._call("tr_dist_hist_build", s._handle, C.byref(spec), None, C.byref(refusal), C.byref(error)))
        self.assertEqual(error.value, -12)                           # more than 3 coordinates: unsupported
        code = "import sys, torbles; print('torbles.math_dist' in sys.modules, 'torbles.math_resp' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False", "False"])


# ====================================================================== DIST-002
def thresholds(values):
    """every observed value, the midpoints between neighbours, and points below and above"""
    xs = sorted(set(values))
    if not xs:
        return [Fraction(0), Fraction(-1, 3)]
    pts = [Fraction(xs[0]) - 1, Fraction(xs[-1]) + 1, Fraction(2 * xs[0] - 1, 2)]
    for a, b in zip(xs, xs[1:]):
        pts += [Fraction(a + b, 2), Fraction(2 * a + b, 3)]
    return pts + [Fraction(x) for x in xs]


def in_range(x, lo, hi, closed):
    return (lo is None or x > lo or (closed[0] and x == lo)) and (hi is None or x < hi or (closed[1] and x == hi))


def q_oracle(ys, p, rule):
    """PERCENTILE_DISC / PERCENTILE_CONT / lower / higher over a sorted expanded list (Python Fractions)"""
    M = len(ys)
    if not M:
        return None
    if rule == "disc":
        return ys[max(1, math.ceil(p * M)) - 1]
    h = p * (M - 1)
    j = math.floor(h)
    f = h - j
    if rule == "lower" or f == 0:
        return ys[j]
    if rule == "higher":
        return ys[j + 1]
    v = ys[j] + f * (ys[j + 1] - ys[j])
    return v.numerator if v.denominator == 1 else v


class Prefixes(Base):
    def same_on_cuda(self, p, items):
        """the batch of request dicts answered on the CPU equals the CUDA kernel's answers, bit for bit"""
        reqs = md.requests(items)
        self.assertTrue(np.array_equal(p.query_arrays(reqs), p.query_arrays(reqs, backend="cuda")))

    def groups_of(self, scope, values, weights=None, groups=None, rows=None):
        """per group: the expanded sorted multiset and the (value, mass) list"""
        out = {}
        n = len(values)
        for i in range(n):
            if rows is not None and not rows[i]:
                continue
            g = 0 if groups is None else int(groups[i])
            out.setdefault(g, ([], []))
            out[g][0].append(values[i]); out[g][1].append(None if weights is None else weights[i])
        return {g: (expanded(v, None if weights is None else w), v, w) for g, (v, w) in out.items()}

    def make(self, rng, n, G, weighted, span=12, nulls=0.15):
        x = [int(a) for a in rng.integers(-span, span + 1, n)]
        w = [int(a) for a in rng.integers(0, 6, n)]
        vx = mask(rng, n, 1 - nulls)
        vw = mask(rng, n, 0.9)
        groups = rng.integers(0, G, n)
        groups[groups == 1] = 2                                      # group 1 absent
        groups[:5] = G - 1                                           # group G - 1: only NULL values
        vx[:5] = False
        groups[5:] = np.where(groups[5:] == G - 1, 0, groups[5:])
        rows = mask(rng, n, 0.85)
        rows[:5] = True
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, valid={"x": vx, "w": vw}, groups=groups, ngroups=G, rows=rows)
        xs = [x[i] if vx[i] else None for i in range(n)]
        ws = [w[i] if vw[i] else None for i in range(n)] if weighted else None
        return scope, xs, ws, groups, rows

    # ------------------------------------------------------------------ acceptance 1: ranks and thresholds
    def test_dist002_every_threshold_at_and_between_observed_values_matches_expanded_rows(self):
        rng = np.random.default_rng(2001)
        for weighted in (False, True):
            scope, xs, ws, groups, rows = self.make(rng, 600, 6, weighted, span=8)
            for repr_ in ("sparse", "dense"):
                h = md.histogram(scope, ["x"], weight="w" if weighted else None, repr=repr_)
                p = h.prefix("weight" if weighted else "count", 4)
                oracle = self.groups_of(scope, xs, ws, groups, rows)
                items, expect = [], []
                for g in range(scope.groups):
                    ys = oracle.get(g, ([], [], []))[0]
                    vals = sorted(set(ys))
                    for t in thresholds(vals):
                        for lo, hi in ((None, t), (t, None), (t, t), (t - 2, t), (t, t + Fraction(5, 2))):
                            for closed in ((True, True), (True, False), (False, True), (False, False)):
                                items.append(dict(op="range", group=g, lo=lo, hi=hi, closed=closed))
                                m = sum(1 for y in ys if in_range(y, lo, hi, closed))
                                expect.append((m, len(ys), None if not ys else Fraction(m, len(ys))))
                                if closed == (True, False) and lo is not None:
                                    for k in range(5):
                                        items.append(dict(op="moment", group=g, lo=lo, hi=hi, closed=closed, k=k))
                                        expect.append(sum(y ** k for y in ys if in_range(y, lo, hi, closed)))
                cpu = p.query_arrays(md.requests(items))
                gpu = p.query_arrays(md.requests(items), backend="cuda")
                self.assertTrue(np.array_equal(cpu, gpu))
                for q, r, e in zip(items, p.query(items), expect):
                    if q["op"] == "range":
                        self.assertEqual((r["mass"], r["total"], r["fraction"]), e, q)
                    else:
                        self.assertEqual(r["value"], e, q)
                # CDF and tails by name, against DuckDB-style counts of the expanded rows
                for g, (ys, _, _) in oracle.items():
                    for t in thresholds(sorted(set(ys)))[:6]:
                        self.assertEqual(p.cdf(t, group=g)["mass"], sum(1 for y in ys if y <= t))
                        self.assertEqual(p.cdf(t, group=g, closed=False)["mass"], sum(1 for y in ys if y < t))
                        self.assertEqual(p.tail(t, group=g)["mass"], sum(1 for y in ys if y >= t))
                        self.assertEqual(p.tail(t, group=g, closed=False)["mass"], sum(1 for y in ys if y > t))

    def test_dist002_every_rank_and_quantile_matches_expanded_rows(self):
        rng = np.random.default_rng(2002)
        for weighted in (False, True):
            scope, xs, ws, groups, rows = self.make(rng, 900, 5, weighted, span=6)
            p = md.histogram(scope, ["x"], weight="w" if weighted else None).prefix("weight" if weighted else "count", 1)
            oracle = self.groups_of(scope, xs, ws, groups, rows)
            items, expect = [], []
            for g in range(scope.groups):
                ys = oracle.get(g, ([], [], []))[0]
                M = len(ys)
                levels = {Fraction(0), Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(99, 100)}
                for r in range(0, M + 1):
                    if M:
                        levels |= {Fraction(r, M), Fraction(2 * r + 1, 2 * M + 2)}
                    if M > 1 and r < M:
                        levels |= {Fraction(r, M - 1), Fraction(2 * r + 1, 2 * (M - 1))} if r < M - 1 else {Fraction(r, M - 1)}
                for lev in sorted(levels):
                    if not 0 <= lev <= 1:
                        continue
                    for rule in md.RULES:
                        items.append(dict(op="quantile", group=g, p=lev, rule=rule))
                        expect.append(q_oracle(ys, lev, rule))
                for r in range(1, M + 2):
                    items.append(dict(op="rank", group=g, r=r))
                    expect.append(ys[r - 1] if r <= M else None)
            cpu = p.query_arrays(md.requests(items))
            self.assertTrue(np.array_equal(cpu, p.query_arrays(md.requests(items), backend="cuda")))
            for q, r, e in zip(items, p.query(items), expect):
                self.assertEqual(r["value"], e, q)
                self.assertEqual(r["empty"], e is None)
            # named calls agree with the batch
            g = 0
            ys = oracle[g][0]
            self.assertEqual(p.quantile(Fraction(1, 2), "cont", group=g), q_oracle(ys, Fraction(1, 2), "cont"))
            self.assertEqual(p.quantile(Decimal("0.25"), group=g), q_oracle(ys, Fraction(1, 4), "disc"))

    def test_dist002_quantiles_and_range_sums_match_duckdb(self):
        rng = np.random.default_rng(2003)
        db = dataset(rng, 2500)
        t = db["f"]
        rows = mask(rng, t.n, 0.8)
        sel = where(db.con, rows)
        for col in ("x", "d"):
            scope = mbk.Scope.from_table(t, [col, "w"], group_by=["k"], rows=rows)
            p = md.histogram(scope, [col]).prefix("count", 2)
            gmap = {scope.key_values(g): g for g in range(scope.groups)}
            for q in (0, 0.125, 0.25, 0.5, 0.75, 0.875, 1):
                lev = Fraction(q)
                res = db.con.execute(f'SELECT k, quantile_disc({logical(col)}, {q}), quantile_cont({logical(col)}, {q}) '
                                     f'FROM f WHERE {sel} AND f."{col}" IS NOT NULL GROUP BY k').fetchall()
                self.assertTrue(res)
                for k, disc, cont in res:
                    g = gmap[(k,)]
                    self.assertEqual(p.quantile(lev, "disc", group=g), int(disc), (col, q, k))
                    got = p.quantile(lev, "cont", group=g, backend="cuda")
                    self.assertEqual(got, p.quantile(lev, "cont", group=g))
                    self.assertAlmostEqual(float(got), float(cont), delta=1e-9 * max(1.0, abs(float(cont))))
            for lo, hi in ((-1, 2), (None, 0), (1, None)):
                cond = " AND ".join(c for c in (f"{logical(col)} >= {lo}" if lo is not None else "",
                                                f"{logical(col)} <= {hi}" if hi is not None else "") if c) or "TRUE"
                res = db.con.execute(f"SELECT k, COUNT({logical(col)}), SUM({logical(col)}), "
                                     f"SUM({logical(col)} * {logical(col)}) FROM f WHERE {sel} AND {cond} GROUP BY k").fetchall()
                self.assertGreater(len(res), 3)
                for k, c, s1, s2 in res:
                    g = gmap[(k,)]
                    self.assertEqual(p.count(lo, hi, group=g)["mass"], int(c))
                    self.assertEqual(p.moment(1, lo, hi, group=g), int(s1 or 0))
                    self.assertEqual(p.moment(2, lo, hi, group=g), int(s2 or 0))

    def test_dist002_trimmed_sums_and_range_integrals(self):
        rng = np.random.default_rng(2004)
        for weighted in (False, True):
            scope, xs, ws, groups, rows = self.make(rng, 1200, 4, weighted, span=40)
            p = md.histogram(scope, ["x"], weight="w" if weighted else None).prefix(
                "weight" if weighted else "count", 4)
            oracle = self.groups_of(scope, xs, ws, groups, rows)
            items, expect = [], []
            for g, (ys, _, _) in oracle.items():
                M = len(ys)
                pairs = [(1, M), (1, 0), (M + 1, M), (1, 1), (M, M)] + [
                    tuple(sorted(int(a) for a in rng.integers(1, M + 1, 2))) for _ in range(25)] if M else [(1, 0)]
                for r1, r2 in pairs:
                    for k in range(5):
                        items.append(dict(op="trimmed", group=g, r1=r1, r2=r2, k=k))
                        expect.append(sum(y ** k for y in ys[r1 - 1:r2]))
                        items.append(dict(op="trimmed_mean", group=g, r1=r1, r2=r2, k=k))
                        expect.append(Fraction(sum(y ** k for y in ys[r1 - 1:r2]), r2 - r1 + 1) if r2 >= r1 else None)
            cpu = p.query_arrays(md.requests(items))
            self.assertTrue(np.array_equal(cpu, p.query_arrays(md.requests(items), backend="cuda")))
            for q, r, e in zip(items, p.query(items), expect):
                self.assertEqual(r["value"], e, q)
            # a trimmed mean: drop the lowest and highest tenth of the ranks
            g, (ys, _, _) = next((g, o) for g, o in oracle.items() if len(o[0]) >= 10)
            a = len(ys) // 10
            self.assertEqual(p.trimmed_mean(a + 1, len(ys) - a, group=g), Fraction(sum(ys[a:len(ys) - a]), len(ys) - 2 * a))

    # ------------------------------------------------------------------ acceptance 2: the specified conventions
    def test_dist002_specified_interpolation_empty_ties_and_endpoints(self):
        # ties: [5, 5, 5, 7] (DuckDB / SQL: disc and cont as below); empty group 1; one value in group 2
        scope = mbk.Scope.from_arrays({"x": [5, 5, 5, 7, 9, 3]}, valid={"x": [True, True, True, True, True, True]},
                                      groups=[0, 0, 0, 0, 2, 3], ngroups=4,
                                      rows=np.array([True, True, True, True, True, False]))
        p = md.histogram(scope, ["x"]).prefix()
        q = lambda lev, rule="disc", g=0: p.quantile(Fraction(lev), rule, group=g)
        self.assertEqual([q(f) for f in ("0", "1/4", "1/2", "3/4", "1")], [5, 5, 5, 5, 7])
        self.assertEqual([q(f, "cont") for f in ("0", "1/4", "1/2", "3/4", "9/10", "1")],
                         [5, 5, 5, Fraction(11, 2), Fraction(32, 5), 7])
        self.assertEqual([q(f, "lower") for f in ("3/4", "9/10")], [5, 5])
        self.assertEqual([q(f, "higher") for f in ("3/4", "2/3")], [7, 5])     # h = 2 exactly: no step up
        self.assertEqual(p.value_at_rank(3), 5)
        self.assertEqual(p.value_at_rank(4), 7)
        self.assertIsNone(p.value_at_rank(5))
        # endpoints: closed and open ends at and between observed values
        self.assertEqual(p.count(5, 7)["mass"], 4)
        self.assertEqual(p.count(5, 7, closed=(False, True))["mass"], 1)
        self.assertEqual(p.count(5, 7, closed=(True, False))["mass"], 3)
        self.assertEqual(p.count(5, 7, closed=(False, False))["mass"], 0)
        self.assertEqual(p.count(Fraction(11, 2), Fraction(13, 2))["mass"], 0)
        self.assertEqual(p.cdf(Fraction(9, 2)), dict(mass=0, total=4, fraction=Fraction(0)))
        self.assertEqual(p.cdf(5), dict(mass=3, total=4, fraction=Fraction(3, 4)))
        self.assertEqual(p.cdf(5, closed=False)["mass"], 0)
        self.assertEqual(p.tail(7, closed=False)["mass"], 0)
        self.assertEqual(p.count(7, 5)["mass"], 0)                       # an empty interval
        # empty input: no quantile, no fraction, mass 0 (an absent group and a group without selected rows)
        for g in (1, 3):
            self.assertIsNone(q("1/2", "cont", g))
            self.assertIsNone(p.value_at_rank(1, group=g))
            self.assertEqual(p.cdf(0, group=g), dict(mass=0, total=0, fraction=None))
            self.assertEqual(p.moment(1, group=g), 0)
            self.assertEqual(p.trimmed(1, 0, group=g), 0)
            self.assertIsNone(p.trimmed_mean(1, 0, group=g))
        self.assertEqual([q(f, r, 2) for f in ("0", "1/2", "1") for r in ("disc", "cont")], [9] * 6)
        # NULLs are outside the support; a group of NULLs only is present with no mass
        s = mbk.Scope.from_arrays({"x": [1, 2, 3]}, valid={"x": [False, False, True]}, groups=[0, 0, 1], ngroups=2)
        pn = md.histogram(s, ["x"]).prefix()
        self.assertEqual(pn.support(0), dict(values=[], nulls=2, rows=2))
        self.assertIsNone(pn.quantile(Fraction(1, 2), group=0))
        self.assertEqual(pn.support(1), dict(values=[(3, 1)], nulls=0, rows=1))
        # weights replicate values: zero weights leave the support, NULL weights carry no mass
        sw = mbk.Scope.from_arrays({"x": [1, 2, 3, 4], "w": [2, 0, 1, 5]}, valid={"w": [True, True, True, False]})
        pw = md.histogram(sw, ["x"], weight="w").prefix("weight")
        self.assertEqual(pw.support(0)["values"], [(1, 2), (3, 1)])
        self.assertEqual([pw.quantile(Fraction(f), "cont") for f in ("0", "1/2", "1")], [1, 1, 3])
        self.assertEqual(pw.quantile(Fraction(3, 4), "cont"), 2)            # h = 3/2 between y_2 = 1 and y_3 = 3
        with self.assertRaises(TypeError):
            p.quantile(0.5)                                                  # a float is refused: exact levels only
        with self.assertRaises(ValueError):
            p.quantile(Fraction(3, 2))
        # every answer above, on the CUDA path too
        items = [dict(op="quantile", p=Fraction(f), rule=r, group=g) for g in range(4) for r in md.RULES
                 for f in ("0", "1/4", "1/2", "2/3", "3/4", "9/10", "1")]
        items += [dict(op="range", lo=lo, hi=hi, closed=c, group=g) for g in range(4) for c in
                  ((True, True), (False, True), (True, False), (False, False))
                  for lo, hi in ((5, 7), (Fraction(11, 2), Fraction(13, 2)), (None, Fraction(9, 2)), (None, 5), (7, None), (7, 5))]
        items += [dict(op="rank", r=r, group=g) for g in range(4) for r in (1, 3, 4, 5)]
        items += [dict(op=o, r1=1, r2=0, k=1, group=g) for g in (1, 3) for o in ("trimmed", "trimmed_mean")]
        items += [dict(op="moment", k=1, group=g) for g in range(4)]
        self.same_on_cuda(p, items)
        self.same_on_cuda(pn, [dict(op="quantile", p=Fraction(1, 2), group=g) for g in range(2)])
        self.same_on_cuda(pw, [dict(op="quantile", p=Fraction(f), rule="cont") for f in ("0", "1/2", "3/4", "1")])

    def test_dist002_extreme_values_are_exact(self):
        lo, hi = -(1 << 63), (1 << 63) - 1
        s = mbk.Scope.from_arrays({"x": [lo, hi, hi, 0]})
        p = md.histogram(s, ["x"]).prefix("count", 0)
        ys = sorted([lo, hi, hi, 0])
        for lev in (Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(2, 3), Fraction(5, 6), Fraction(1)):
            for rule in md.RULES:
                self.assertEqual(p.quantile(lev, rule), q_oracle(ys, lev, rule), (lev, rule))
                self.assertEqual(p.quantile(lev, rule, backend="cuda"), p.quantile(lev, rule))
        self.assertEqual(p.count(Fraction(lo), Fraction(hi))["mass"], 4)
        self.assertEqual(p.count(Fraction(lo), Fraction(hi), closed=(False, False))["mass"], 1)
        self.assertEqual(p.cdf(Fraction(hi, 2))["mass"], 2)
        self.same_on_cuda(p, [dict(op="range", lo=Fraction(lo), hi=Fraction(hi), closed=c) for c in
                              ((True, True), (False, False))] + [dict(op="range", hi=Fraction(hi, 2))] +
                          [dict(op="rank", r=r) for r in range(1, 6)])
        with self.assertRaises(UnsupportedError):
            p.moment(1)                                                      # the state keeps degree 0 only
        # the moment proof: T * X^k <= 2^127 - 1
        three = mbk.Scope.from_arrays({"x": [1 << 62, -(1 << 62), 1 << 62]})
        h3 = md.histogram(three, ["x"])
        p2 = h3.prefix("count", 2)
        self.assertEqual(p2.moment(2), 3 << 124)
        self.assertEqual(p2.moment(1), 1 << 62)
        self.assertEqual(p2.trimmed(2, 3, k=2), 2 << 124)
        self.same_on_cuda(p2, [dict(op="moment", k=k) for k in range(3)] + [dict(op="trimmed", r1=2, r2=3, k=k)
                                                                            for k in range(3)])
        with self.assertRaises(OverflowError):
            h3.prefix("count", 3)

    # ------------------------------------------------------------------ CUDA batches, budgets, refusals
    def test_dist002_cuda_batches_equal_cpu(self):
        rng = np.random.default_rng(2005)
        n, G = 30000, 50
        x = rng.integers(-10**6, 10**6, n)
        w = rng.integers(0, 1000, n)
        groups = rng.integers(0, G, n)
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, valid={"x": mask(rng, n, 0.9)}, groups=groups, ngroups=G)
        for mass, weight in (("count", None), ("weight", "w")):
            p = md.histogram(scope, ["x"], weight=weight).prefix(mass, 2)
            N = 10000
            r = np.zeros(N, dtype=md.REQUEST_DTYPE)
            r["op"] = rng.integers(1, 7, N)
            r["group"] = rng.integers(0, G + 1, N)                         # some past the groups: per-request status
            r["flags"] = rng.integers(0, 16, N)
            r["a_num"] = rng.integers(-10**6, 10**6, N); r["a_den"] = rng.integers(1, 7, N)
            r["b_num"] = rng.integers(-10**6, 10**6, N); r["b_den"] = rng.integers(1, 7, N)
            r["power"] = rng.integers(0, 3, N)
            r["rule"] = rng.integers(1, 5, N)
            qs = r["op"] == 3
            r["a_den"][qs] = rng.integers(1, 1000, int(qs.sum()))
            r["a_num"][qs] = (rng.random(int(qs.sum())) * r["a_den"][qs]).astype(np.int64)
            rk = (r["op"] == 4) | (r["op"] >= 5)
            r["a_den"][rk] = 1; r["b_den"][rk] = 1
            r["a_num"][rk] = rng.integers(1, 500, int(rk.sum())); r["b_num"][rk] = r["a_num"][rk] + rng.integers(-1, 300, int(rk.sum()))
            cpu = p.query_arrays(r)
            gpu = p.query_arrays(r, backend="cuda")
            self.assertTrue(np.array_equal(cpu, gpu))
            self.assertGreater(int((cpu["status"] == 0).sum()), N // 2)
            self.assertTrue((cpu["status"][r["group"] >= G] == -1).all())
            self.assertGreater(p.device(), 0)

    def test_dist002_budgets_and_refusals(self):
        rng = np.random.default_rng(2006)
        scope, xs, ws, groups, rows = self.make(rng, 2000, 7, True, span=100)
        h = md.histogram(scope, ["x"], weight="w")
        ref = h.prefix("weight", 2)
        raw = md._Need()
        md._check(md._call("tr_dist_prefix_need", h._handle, 2, 2, C.byref(raw)))
        needs = md._needs_dict(raw)
        self.assertEqual(needs["states"], ref.info["support"] * 3)
        self.assertEqual(needs["support"], ref.info["max_support"])
        self.assertEqual(needs["bytes"], ref.info["bytes"])
        for resource, value in needs.items():
            self.assertTrue(h.prefix("weight", 2, budget=Budget(**{resource: value})).equal(ref))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                h.prefix("weight", 2, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        reqs = md.requests([dict(op="quantile", p=Fraction(1, 2), group=g) for g in range(7)] * 3)
        for backend in ("cpu", "cuda"):
            self.assertEqual(len(ref.query_arrays(reqs, backend=backend, budget=Budget(output=21))), 21)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ref.query_arrays(reqs, backend=backend, budget=Budget(output=20))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 21))
        fresh = h.prefix("weight", 2)                                    # the device copy is budgeted (bytes)
        need = fresh.info["device_bytes"]
        with self.assertRaises(AlgebraBudgetError) as ctx:
            fresh.device(budget=Budget(bytes=need - 1))
        self.assertEqual(ctx.exception.need, need)
        self.assertEqual(fresh.device(budget=Budget(bytes=need)), need)
        # refusals: negative weights as a mass, a joint histogram, degree, invalid requests
        neg = mbk.Scope.from_arrays({"x": [1, 2], "w": [3, -1]})
        with self.assertRaises(DomainError):
            md.histogram(neg, ["x"], weight="w").prefix("weight")
        self.assertEqual(md.histogram(neg, ["x"], weight="w").prefix("count").cdf(1)["mass"], 1)
        with self.assertRaises(ValueError):
            md.histogram(neg, ["x"]).prefix("weight")                   # no weight column
        with self.assertRaises(UnsupportedError):
            md.histogram(neg, ["x", "w"]).prefix()
        with self.assertRaises(UnsupportedError):
            md.histogram(neg, ["x"]).prefix("count", 5)
        with self.assertRaises(ValueError):
            ref.trimmed(3, 1)
        with self.assertRaises(ValueError):
            ref.value_at_rank(0)
        with self.assertRaises(OverflowError):
            ref.cdf(Fraction(1, 1 << 64))
        bad = md.requests([dict(op="range", hi=1)])
        bad["b_den"] = 0
        self.assertEqual(int(ref.query_arrays(bad)["status"][0]), -1)
        self.assertEqual(int(ref.query_arrays(bad, backend="cuda")["status"][0]), -1)


if __name__ == "__main__":
    unittest.main()
