# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""MULTI-001: certified refinement maps, hierarchies of exact moment banks rolled up from the finest level, localized
updates and exact Haar-like contrasts (torbles/math_multi.c, math_multi_gpu.cu, math_multi.py).

Oracles: DuckDB SQL over the typed source tables (GROUP BY each level's key with COUNT(*), COUNT(term) and SUM(term) in
HUGEINT on the logical values, LEFT JOIN for dimension attributes, FILTER for the listwise mode), the FND-004 bank
built directly on each level's scope (complete reconstruction), and Python integers and Fractions written here (the
smallest violating row of a refinement, per-level states of array hierarchies, every contrast). Every certification,
finest bank and contrast is also computed on the CUDA path (emulated here, TORBLES_EMU_SCRAMBLE=1 for scrambled thread
order) and compared with the CPU result for equality. Seeds are fixed.
"""
import math
import subprocess
import sys
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_multi as mm
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SCALES = {"x": 0, "d": 2}
DDL = """CREATE TABLE dim (id BIGINT PRIMARY KEY, city VARCHAR, region VARCHAR, country VARCHAR, zone VARCHAR);
CREATE TABLE f (day BIGINT, month BIGINT, year BIGINT, x BIGINT, d DECIMAL(10,2), c BIGINT REFERENCES dim(id));"""
BASIS = mbk.monomials(["x", "d"], 2) + [(3, 0), (1, 2)]
TIME = [["day"], ["month"], ["year"], []]
JOIN = 'LEFT JOIN dim ON f."c" = dim."id"'


def obj(values):
    return pd.Series(values, dtype=object)


def dataset(rng, n):
    """facts over two years with a day -> month -> year hierarchy (NULL days are NULL months and years), signed and
    decimal values with NULLs, one month (`quiet`) whose values are all NULL on every day, absent month codes between
    the years, and foreign keys (unmatched and NULL included) to a dimension id -> city -> region -> country; zone is
    not a function of the region (the NULL region holds two zones)"""
    days = sorted({int(y) * 10000 + int(m) * 100 + int(dd) for y, m, dd in
                   zip(rng.choice([2023, 2024], 70), rng.integers(1, 13, 70), rng.integers(1, 29, 70))})
    day = [int(v) for v in rng.choice(days, n)]
    x = [int(v) for v in rng.integers(-5000, 5001, n)]
    d = [f"{int(v) / 100:.2f}" for v in rng.integers(-99999, 100000, n)]
    c = [int(v) for v in rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99], n)]
    for col, p in ((x, 0.15), (d, 0.15), (c, 0.05), (day, 0.03)):
        for i in np.flatnonzero(rng.random(n) < p):
            col[int(i)] = None
    quiet = days[3]
    for i in range(n):
        if day[i] is not None and day[i] // 100 == quiet // 100:
            x[i] = d[i] = None                                    # the month of `quiet`: present, every value NULL
    month = [None if v is None else v // 100 for v in day]
    year = [None if v is None else v // 10000 for v in day]
    fact = pd.DataFrame({"day": obj(day), "month": obj(month), "year": obj(year), "x": obj(x), "d": obj(d), "c": obj(c)})
    dim = pd.DataFrame({"id": list(range(1, 11)),
                        "city": obj(["Lyon", "Paris", "Nice", "Berlin", "Munich", "Bonn", "Rome", "Milan", "Lyon", "Oslo"]),
                        "region": obj(["ARA", "IDF", "PACA", "BE", "BY", "NRW", "LAZ", "LOM", "ARA", None]),
                        "country": obj(["FR", "FR", "FR", "DE", "DE", "DE", "IT", "IT", "FR", None]),
                        "zone": obj(["W", "W", None, "C", "C", "C", "S", "S", "W", "N"])})
    db = torbles.from_ddl(DDL, {"f": fact, "dim": dim})
    assert db["f"].surr.get("c") is None                      # a plain foreign key (no derived surrogate)
    return db, quiet


def logical(name):
    return f'CAST(f."{name}" * {10 ** SCALES[name]} AS HUGEINT)' if SCALES[name] else f'CAST(f."{name}" AS HUGEINT)'


def term_sql(names, exps):
    factors = [logical(n) for n, e in zip(names, exps) for _ in range(e)]
    return "(" + " * ".join(factors) + ")" if factors else "1"


def duck_states(db, names, terms, keys_sql, mode="sql", rows=None, join=""):
    """{key tuple: (rows, [count per term], [sum per term])} by DuckDB over the source table f"""
    con = db.con
    where = "TRUE"
    if rows is not None:
        con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(np.asarray(rows)).astype(np.int64)}))
        where = "f.rowid IN (SELECT rid FROM sel_rows)"
    used = sorted({n for t in terms for n, e in zip(names, t) if e})
    joint = " AND ".join(f'f."{n}" IS NOT NULL' for n in used) or "TRUE"
    parts = []
    for t in terms:
        expr = term_sql(names, t)
        if mode == "sql":
            parts += [f"COUNT({expr})", f"SUM({expr})"]
        else:
            parts += [f"COUNT(*) FILTER (WHERE {joint})", f"SUM({expr}) FILTER (WHERE {joint})"]
    gk = [f"{e} AS g{i}" for i, e in enumerate(keys_sql)]
    sql = (f"SELECT {', '.join(gk + ['COUNT(*)'] + parts)} FROM f {join} WHERE {where}"
           + (f" GROUP BY {', '.join(f'g{i}' for i in range(len(keys_sql)))}" if keys_sql else ""))
    out = {}
    for r in con.execute(sql).fetchall():
        nk = len(keys_sql)
        if not keys_sql and r[0] == 0:
            continue
        vals = r[nk + 1:]
        out[tuple(r[:nk])] = (int(r[nk]), [int(v) for v in vals[0::2]], [0 if v is None else int(v) for v in vals[1::2]])
    return out


def level_states(h, level):
    return {row["key"]: (row["rows"], row["count"], row["sum"]) for row in h.table(level)}


def smallest_violation(fine_ids, coarse_ids, rows):
    """the witness of a refinement by hand: the first selected row whose coarse id differs from that of the first
    selected row of its fine group"""
    first = {}
    for r in np.flatnonzero(rows):
        f, q = int(fine_ids[r]), int(coarse_ids[r])
        if f not in first:
            first[f] = (int(r), q)
        elif first[f][1] != q:
            return dict(row=int(r), first_row=first[f][0], fine=f, coarse=q, first_coarse=first[f][1])
    return None


def python_states(groups, ngroups, cols, valid, rows, exps, mode="sql"):
    """per group (rows, counts, sums) by Python integers"""
    names = list(cols)
    out = [[0, [0] * len(exps), [0] * len(exps)] for _ in range(ngroups)]
    for i in np.flatnonzero(rows):
        g = int(groups[i])
        out[g][0] += 1
        for j, e in enumerate(exps):
            need = [n for n, k in zip(names, e) if k] if mode == "sql" else names
            if all(valid[n][i] for n in need):
                out[g][1][j] += 1
                out[g][2][j] += math.prod(int(cols[n][i]) ** k for n, k in zip(names, e))
    return out


def array_hierarchy(rng, n, sizes, skew=False):
    """random leaf ids and their ancestors: level k has sizes[k] groups; balanced: parent = id * size_k+1 // size_k,
    skewed: most groups under parent 0"""
    ids = [rng.integers(0, sizes[0], n)]
    parents = []
    for k in range(1, len(sizes)):
        lo, hi = sizes[k - 1], sizes[k]
        if skew:
            p = np.where(np.arange(lo) < lo * 3 // 4, 0, rng.integers(0, hi, lo))
        else:
            p = np.arange(lo) * hi // lo
        parents.append(p.astype(np.int64))
        ids.append(parents[-1][ids[-1]])
    return ids, parents


class Multi(unittest.TestCase):
    def assert_absent_zero(self, bank):
        for g, s in enumerate(bank.states()):
            if not s["present"]:
                self.assertEqual((s["rows"], set(s["count"]), set(s["sum"])), (0, {0}, {0}), g)

    def certify_both(self, fine, coarse):
        """the CPU map, checked equal to the (emulated, scrambled) CUDA map"""
        cpu = mm.certify(fine, coarse)
        gpu = mm.certify(fine, coarse, backend="cuda")
        self.assertTrue(cpu.equal(gpu))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        self.assertTrue(np.array_equal(cpu.ids, gpu.ids))
        return cpu

    # ------------------------------------------------------------------ acceptance 1: exact merges with presence
    def test_multi001_acceptance_child_states_merge_exactly_into_parents(self):
        rng = np.random.default_rng(1001)
        db, quiet = dataset(rng, 4000)
        t, dim = db["f"], db["dim"]
        rows = rng.random(t.n) < 0.85
        hierarchies = [
            ([[k] if k else [] for k in ("day", "month", "year", None)], [['f."day"'], ['f."month"'], ['f."year"'], []], ""),
            ([[("c", dim, a)] for a in ("id", "city", "region", "country")] + [[]],
             [['dim."id"'], ['dim."city"'], ['dim."region"'], ['dim."country"'], []], JOIN),
        ]
        for keys, keys_sql, join in hierarchies:
            levels = [mbk.Scope.from_table(t, ["x", "d"], group_by=k, rows=rows) for k in keys]
            maps = [self.certify_both(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]
            for mode in ("sql", "joint"):
                for backend in ("cpu", "cuda"):
                    h = mm.Hierarchy(levels, BASIS, mode, maps=maps, backend=backend)
                    self.assertEqual(h.status, "proven")
                    info = h.info
                    self.assertEqual(info["scoped"], [True] * len(levels))
                    self.assertEqual(info["strategy"], ["copy"] + ["rollup"] * (len(levels) - 1))
                    for k, scope in enumerate(levels):
                        direct = mbk.build(scope, BASIS, mode)           # complete reconstruction of the level
                        self.assertTrue(h.bank(k).equal(direct), (keys[k], mode, backend))
                        expect = duck_states(db, ["x", "d"], BASIS, keys_sql[k], mode, rows, join)
                        self.assertEqual(level_states(h, k), expect, (keys[k], mode))
                        self.assertEqual(info["present"][k], len(expect))
                        self.assert_absent_zero(h.bank(k))
                        self.assertEqual(h.bank(k).rows, int(rows.sum()))
            # coarse levels without value columns: unscoped banks with the same states
            light = [levels[0]] + [mbk.Scope.from_table(t, [], group_by=k, rows=rows) for k in keys[1:]]
            hl = mm.Hierarchy(light, BASIS, maps=[self.certify_both(light[i], light[i + 1]) for i in range(len(light) - 1)])
            self.assertEqual(hl.status, "proven")
            self.assertEqual(hl.info["scoped"], [True] + [False] * (len(levels) - 1))
            for k in range(len(levels)):
                self.assertEqual(hl.bank(k).states(), mbk.build(levels[k], BASIS).states())
            # the composition of the maps rolls the finest level up to any level in one step
            comp = maps[0]
            for k in range(1, len(maps)):
                comp = comp.compose(maps[k])
                self.assertEqual(comp.status, "proven")
                up = mbk.build(levels[0], BASIS).rollup(map=np.where(comp.ids == mm.UNMAPPED, 0, comp.ids),
                                                        groups=levels[k + 1].groups)
                self.assertTrue(up.equal(mbk.build(levels[k + 1], BASIS)))
        # explicit presence at every level: the all-NULL day (and its month and year) are present; absent codes stay
        # absent (month codes between the years, days never drawn)
        levels = [mbk.Scope.from_table(t, ["x", "d"], group_by=k) for k in TIME]
        h = mm.Hierarchy(levels, BASIS)
        by_day = {levels[0].key_values(g): h.state(0, g) for g in range(levels[0].groups)}
        s = by_day[(quiet,)]
        self.assertTrue(s["present"] and s["rows"] > 0)
        self.assertEqual((s["count"][1:], s["sum"][1:]), ([0] * (len(BASIS) - 1), [0] * (len(BASIS) - 1)))
        months = {levels[1].key_values(g): h.state(1, g) for g in range(levels[1].groups)}
        m = months[(quiet // 100,)]                                   # the all-NULL month, rolled up from its days
        self.assertTrue(m["present"] and m["rows"] >= s["rows"])
        self.assertEqual((m["count"][1:], m["sum"][1:]), ([0] * (len(BASIS) - 1), [0] * (len(BASIS) - 1)))
        self.assertFalse(months[(202313,)]["present"])               # code between 202312 and 202401
        self.assertEqual(months[(202313,)]["rows"], 0)
        self.assertTrue(months[(None,)]["present"])                  # NULL days are NULL months
        self.assertGreater(h.info["groups"][1] - h.info["present"][1], 50)

    def test_multi001_array_hierarchies_balanced_and_skewed_up_to_eight_levels(self):
        rng = np.random.default_rng(1002)
        n = 3000
        x = rng.integers(-(1 << 40), 1 << 40, n)
        y = rng.integers(-1000, 1001, n)
        valid = {"x": rng.random(n) < 0.85, "y": rng.random(n) < 0.9}
        cols = {"x": x, "y": y}
        exps = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (0, 3)]
        rows = rng.random(n) < 0.8
        for sizes, skew in (([1024, 64, 8], False), ([1024, 256, 64, 16, 8, 4, 2, 1], False),
                            ([64, 8], True), ([1024, 512, 128, 64, 32, 8, 4, 1], True)):
            ids, _ = array_hierarchy(rng, n, sizes, skew)
            for g in rng.choice(sizes[0], 5, replace=False):        # some leaves absent
                rows[ids[0] == g] = False
            levels = [mbk.Scope.from_arrays(cols, valid=valid, groups=ids[k], ngroups=sizes[k], rows=rows)
                      for k in range(len(sizes))]
            for mode in ("sql", "joint"):
                h = mm.Hierarchy(levels, exps, mode, backend="cuda")
                self.assertEqual(h.status, "proven")
                self.assertEqual(h.depth, len(sizes))
                for k in range(len(sizes)):
                    expect = python_states(ids[k], sizes[k], cols, valid, rows, exps, mode)
                    got = [[s["rows"], s["count"], s["sum"]] for s in h.states(k)]
                    self.assertEqual(got, expect, (sizes, k, mode))
                    self.assertTrue(h.bank(k).equal(mbk.build(levels[k], exps, mode)))
        with self.assertRaises(UnsupportedError):
            mm.Hierarchy(levels + [levels[-1]], exps)                 # nine levels

    # ------------------------------------------------------------------ acceptance 2: incompatible maps, codecs
    def test_multi001_acceptance_rejects_incompatible_maps_and_preserves_codecs(self):
        rng = np.random.default_rng(1003)
        db, _ = dataset(rng, 3000)
        t, dim = db["f"], db["dim"]
        rows = rng.random(t.n) < 0.8
        day, month, year = (mbk.Scope.from_table(t, ["x", "d"], group_by=k, rows=rows) for k in TIME[:3])
        # a map whose levels disagree with the data: month does not determine day; the witness is the smallest row
        sel = np.asarray(rows)
        mview, dview = month.view(), day.view()
        mid = np.ctypeslib.as_array((mbk.C.c_uint32 * t.n).from_address(mview.groups)).copy()
        did = np.ctypeslib.as_array((mbk.C.c_uint32 * t.n).from_address(dview.groups)).copy()
        expect = smallest_violation(mid, did, sel)
        self.assertIsNotNone(expect)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(mm.RefinementError) as ctx:
                mm.certify(month, day, backend=backend)
            w = ctx.exception.witness
            self.assertEqual({k: w[k] for k in expect}, expect, backend)
            self.assertEqual(w["fine_key"], month.key_values(expect["fine"]))
            self.assertTrue(isinstance(ctx.exception, DomainError))
        # NULL semantics: the NULL region holds the zones N and W (city Oslo, and unmatched references): refused
        region = mbk.Scope.from_table(t, ["x"], group_by=[("c", dim, "region")])
        zone = mbk.Scope.from_table(t, ["x"], group_by=[("c", dim, "zone")])
        with self.assertRaises(mm.RefinementError) as ctx:
            mm.certify(region, zone)
        self.assertEqual(ctx.exception.witness["fine_key"], (None,))
        self.assertEqual({ctx.exception.witness["coarse_key"], ctx.exception.witness["first_coarse_key"]},
                         {("N",), (None,)})
        # a map built for other rows (another predicate), another grouping or another revision is refused
        m_dm, m_my = self.certify_both(day, month), self.certify_both(month, year)
        other = rows.copy(); other[int(np.flatnonzero(rows)[0])] = False
        day2, month2, year2 = (mbk.Scope.from_table(t, ["x", "d"], group_by=k, rows=other) for k in TIME[:3])
        self.assertTrue(m_dm.check(day, month))
        self.assertFalse(m_dm.check(day2, month2))
        with self.assertRaises(DomainError):
            mm.Hierarchy([day2, month2, year2], BASIS, maps=[m_dm, m_my])
        with self.assertRaises(DomainError):
            mm.Hierarchy([day, year], BASIS, maps=[m_dm])             # day -> month used as day -> year
        with self.assertRaises(DomainError):
            m_my.compose(m_dm)                                        # the levels do not chain
        with self.assertRaises(DomainError):
            mm.Hierarchy([day, month, year], BASIS, maps=[m_dm, m_my], bank=mbk.build(day2, BASIS))   # another scope's bank
        h = mm.Hierarchy([day, month, year], BASIS, maps=[m_dm, m_my])
        self.assertEqual(h.status, "proven")
        t.flip(0, 0); t.flip(0, 0)                                    # a new revision (the data flipped back)
        with self.assertRaises(StaleStateError):
            mm.certify(day, month)
        with self.assertRaises(StaleStateError):
            mm.Hierarchy([day, month, year], BASIS, maps=[m_dm, m_my])
        now = [mbk.Scope.from_table(t, ["x", "d"], group_by=k, rows=rows) for k in TIME[:3]]
        self.assertFalse(m_dm.check(now[0], now[1]))                  # proven on the earlier revision
        with self.assertRaises(DomainError):
            mm.Hierarchy(now, BASIS, maps=[m_dm, m_my])
        self.assertTrue(mm.Hierarchy(now, BASIS).bank(2).equal(h.bank(2)))   # re-certified: the same states
        # logical codec semantics: the same declared map under other codecs is refused; declared in logical keys it
        # is translated through each table's codecs
        def city_table(cities, regions):
            frame = pd.DataFrame({"city": cities, "region": regions, "x": list(range(len(cities)))})
            return torbles.from_ddl("CREATE TABLE f (city VARCHAR, region VARCHAR, x BIGINT)", {"f": frame})
        a = city_table(["Berlin", "Paris", "Paris", "Berlin"], ["DE", "FR", "FR", "DE"])
        b = city_table(["Munich", "Lyon", "Lyon", "Munich"], ["DE", "FR", "FR", "DE"])   # Lyon: code 0, region FR
        a_city, a_region = (mbk.Scope.from_table(a["f"], ["x"], group_by=[k]) for k in ("city", "region"))
        b_city, b_region = (mbk.Scope.from_table(b["f"], ["x"], group_by=[k]) for k in ("city", "region"))
        self.assertEqual(a_city.groups, b_city.groups)
        declared = mm.declare(a_city, a_region, {"Berlin": "DE", "Paris": "FR"})
        self.assertEqual(declared.status, "declared")
        self.assertTrue(declared.check(a_city, a_region))
        self.assertFalse(declared.check(b_city, b_region))            # same grouping specs, other dictionaries
        with self.assertRaises(DomainError):
            mm.Hierarchy([b_city, b_region], ["1", "x"], maps=[declared])
        with self.assertRaises(DomainError):
            declared.verify(b_city, b_region)
        on_b = mm.declare(b_city, b_region, {"Munich": "DE", "Lyon": "FR"})
        self.assertFalse(np.array_equal(on_b.ids, declared.ids))     # other codes for the same logical map
        hb = mm.Hierarchy([b_city, b_region], ["1", "x"], maps=[on_b.verify(b_city, b_region)])
        self.assertEqual(hb.status, "proven")
        got = {row["key"]: (row["rows"], row["sum"]) for row in hb.table(1)}
        duck = {(r[0],): (int(r[1]), [int(r[1]), int(r[2])]) for r in
                b.con.execute('SELECT region, COUNT(*), SUM(x) FROM f GROUP BY region').fetchall()}
        self.assertEqual(got, duck)
        twin = city_table(["Berlin", "Paris", "Paris", "Berlin"], ["DE", "FR", "FR", "DE"])   # the same codecs
        t_city, t_region = (mbk.Scope.from_table(twin["f"], ["x"], group_by=[k]) for k in ("city", "region"))
        self.assertTrue(declared.check(t_city, t_region))
        self.assertEqual(mm.Hierarchy([t_city, t_region], ["1", "x"], maps=[declared]).status, "declared")
        with self.assertRaises(ValueError):
            mm.declare(b_city, b_region, {"Munich": "AT"})            # no such coarse key in the coarse codec

    def test_multi001_declared_maps_verified_and_marked(self):
        rng = np.random.default_rng(1004)
        n = 2000
        leaf = rng.integers(0, 30, n)
        parent = (np.arange(30) % 7).astype(np.int64)
        x = rng.integers(-100, 101, n)
        rows = rng.random(n) < 0.9
        fine = mbk.Scope.from_arrays({"x": x}, groups=leaf, ngroups=30, rows=rows)
        coarse = mbk.Scope.from_arrays({"x": x}, groups=parent[leaf], ngroups=7, rows=rows)
        good = mm.declare(fine, coarse, parent)
        self.assertEqual((good.status, good.info["checked"], good.info["rows_bound"]), ("declared", 0, False))
        proven = good.verify(fine, coarse)
        self.assertEqual((proven.status, proven.info["checked"]), ("proven", int(rows.sum())))
        self.assertTrue(proven.equal(mm.certify(fine, coarse)))
        hd = mm.Hierarchy([fine, coarse], ["1", "x"], maps=[good])
        self.assertEqual((hd.status, hd.info["maps"], hd.info["scoped"]), ("declared", ["declared"], [True, False]))
        self.assertTrue(hd.bank(1).equal(mbk.build(coarse, ["1", "x"])))
        # a declaration that disagrees with the data: verify refuses at the smallest such row; a hierarchy along it
        # is the exact rollup of the declaration, marked declared, and not the bank of the coarse level
        wrong = parent.copy(); wrong[int(leaf[np.flatnonzero(rows)[5]])] ^= 1
        bad = mm.declare(fine, coarse, wrong)
        with self.assertRaises(mm.RefinementError) as ctx:
            bad.verify(fine, coarse)
        r = next(int(i) for i in np.flatnonzero(rows) if wrong[leaf[i]] != parent[leaf[i]])
        self.assertEqual((ctx.exception.witness["row"], ctx.exception.witness["first_coarse"]), (r, int(wrong[leaf[r]])))
        hw = mm.Hierarchy([fine, coarse], ["1", "x"], maps=[bad])
        self.assertEqual(hw.status, "declared")
        self.assertFalse(hw.bank(1).equal(mbk.build(coarse, ["1", "x"])))
        alt = mbk.Scope.from_arrays({"x": x}, groups=wrong[leaf], ngroups=7, rows=rows)
        self.assertEqual(hw.bank(1).states(), mbk.build(alt, ["1", "x"]).states())
        # a present group without a parent is refused; an absent one may have none
        holes = parent.astype(np.uint32).copy()
        holes[int(leaf[np.flatnonzero(rows)[0]])] = mm.UNMAPPED
        with self.assertRaises(DomainError) as ctx:
            mm.Hierarchy([fine, coarse], ["1", "x"], maps=[mm.declare(fine, coarse, holes)])
        self.assertNotIsInstance(ctx.exception, mm.RefinementError)
        sparse = mbk.Scope.from_arrays({"x": x}, groups=leaf, ngroups=31, rows=rows)   # group 30: absent
        sparse_c = mbk.Scope.from_arrays({"x": x}, groups=parent[leaf], ngroups=7, rows=rows)
        cert = mm.certify(sparse, sparse_c)
        self.assertEqual((cert.parent(30), cert.info["mapped"]), (None, 30))
        self.assertEqual(mm.Hierarchy([sparse, sparse_c], ["1", "x"], maps=[cert]).status, "proven")
        with self.assertRaises(ValueError):
            mm.declare(fine, coarse, np.full(30, 7))                  # ids must be < coarse groups

    def test_multi001_certification_cuda_equals_cpu_scrambled(self):
        rng = np.random.default_rng(1005)
        n = 20000
        for G, Q, violate in ((1, 1, False), (5000, 300, False), (5000, 300, True), (20000, 7, True), (3, 2, True)):
            fine_ids = rng.integers(0, G, n)
            parent = rng.integers(0, Q, G)
            coarse_ids = parent[fine_ids].copy()
            if violate:
                for i in rng.choice(n, 25, replace=False):
                    coarse_ids[i] = (coarse_ids[i] + 1) % Q
            rows = rng.random(n) < 0.7
            fine = mbk.Scope.from_arrays({}, groups=fine_ids, ngroups=G, rows=rows)
            coarse = mbk.Scope.from_arrays({}, groups=coarse_ids, ngroups=Q, rows=rows)
            expect = smallest_violation(fine_ids, coarse_ids, rows)
            if expect is None:
                cpu = self.certify_both(fine, coarse)
                first = {}
                for r in np.flatnonzero(rows):
                    first.setdefault(int(fine_ids[r]), int(coarse_ids[r]))
                self.assertEqual(cpu.ids.tolist(), [first.get(g, mm.UNMAPPED) for g in range(G)])
                self.assertEqual(cpu.info["checked"], int(rows.sum()))
                continue
            for backend in ("cpu", "cuda"):
                with self.assertRaises(mm.RefinementError) as ctx:
                    mm.certify(fine, coarse, backend=backend)
                w = ctx.exception.witness
                self.assertEqual({k: w[k] for k in expect}, expect, (G, backend))
        empty = mbk.Scope.from_arrays({}, groups=rng.integers(0, 9, 100), ngroups=9, rows=np.zeros(100, bool))
        top = mbk.Scope.from_arrays({}, groups=np.zeros(100, np.int64), ngroups=2, rows=np.zeros(100, bool))
        m = self.certify_both(empty, top)
        self.assertEqual((m.info["mapped"], m.info["checked"]), (0, 0))

    # ------------------------------------------------------------------ contrasts
    def oracle_contrasts(self, child, parent, cmap, kind):
        """per child group: (other, [(status, Fraction or None) per term]) from two banks, by Python Fractions"""
        cs, ps = child.states(), parent.states()
        nxt = {}
        last = {}
        for c in range(len(cs) - 1, -1, -1):
            if cs[c]["present"]:
                p = int(cmap[c])
                nxt[c] = last.get(p)
                last[p] = c
        out = []
        for c, s in enumerate(cs):
            if not s["present"]:
                out.append((None, [("absent", None)] * len(s["count"])))
                continue
            o = nxt[c] if kind.startswith("sibling") else int(cmap[c])
            if o is None:
                out.append((None, [("no sibling", None)] * len(s["count"])))
                continue
            os_ = cs[o] if kind.startswith("sibling") else ps[o]
            cells = []
            for j in range(len(s["count"])):
                nc, sc, no, so = s["count"][j], s["sum"][j], os_["count"][j], os_["sum"][j]
                if kind in ("total", "sibling_total"):
                    cells.append(("defined", Fraction(sc - so)))
                elif kind == "share":
                    cells.append(("undefined", None) if no == 0 else ("defined", sc - Fraction(so * nc, no)))
                else:
                    cells.append(("undefined", None) if nc == 0 or no == 0 else ("defined", Fraction(sc, nc) - Fraction(so, no)))
            out.append((o, cells))
        return out

    def test_multi001_contrasts_exact_against_python_fractions(self):
        rng = np.random.default_rng(1006)
        db, quiet = dataset(rng, 3000)
        t = db["f"]
        levels = [mbk.Scope.from_table(t, ["x", "d"], group_by=k) for k in TIME]
        h = mm.Hierarchy(levels, BASIS)
        for level in range(3):
            direct_child, direct_parent = mbk.build(levels[level], BASIS), mbk.build(levels[level + 1], BASIS)
            cmap = np.where(h.map(level).ids == mm.UNMAPPED, 0, h.map(level).ids)
            for kind in mm.KINDS:
                cpu = h.contrasts(level, kind)
                gpu = h.contrasts(level, kind, backend="cuda")
                self.assertEqual(cpu.raw(), gpu.raw(), (level, kind))
                self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
                expect = self.oracle_contrasts(direct_child, direct_parent, cmap, kind)
                for row in cpu.rows():
                    o, cells = expect[row["group"]]
                    self.assertEqual(row["other"], o, (level, kind, row["group"]))
                    self.assertEqual(list(zip(row["status"], row["value"])), cells, (level, kind, row["group"]))
                for child, other, term, status, num, den in cpu.raw():   # canonical numerators and denominators
                    if status == "defined":
                        self.assertTrue(den >= 1 and math.gcd(num, den) == 1)
                if kind == "share":                                   # the excess totals of siblings sum to zero
                    by_parent = {}
                    for row in cpu.rows():
                        if row["other"] is not None:
                            acc = by_parent.setdefault(row["other"], [Fraction(0)] * len(BASIS))
                            for j, v in enumerate(row["value"]):
                                acc[j] += v or 0
                    self.assertTrue(all(v == 0 for acc in by_parent.values() for v in acc))
        # the all-NULL day: its means are undefined but its totals are defined; selected terms and group ranges
        g = next(g for g in range(levels[0].groups) if levels[0].key_values(g) == (quiet,))
        row = h.contrasts(0, "mean", terms=[0, 1, 4], groups=(g, g + 1)).rows()[0]
        self.assertEqual(row["status"], ["defined", "undefined", "undefined"])
        self.assertEqual(row["value"][0], 0)
        tot = h.contrasts(0, "total", terms=[1], groups=(g, g + 1), backend="cuda").rows()[0]
        self.assertEqual(tot["value"], [-h.state(1, row["other"])["sum"][1]])
        # sums of x^3 near 2^126 (16 rows of |x| < 2^41, inside the FND-004 proof): numerators past 2^128
        top = (1 << 41) - 1
        big = [top - int(v) for v in rng.integers(0, 50, 8)] + [-(top - int(v)) for v in rng.integers(0, 50, 6)] + [top, 7]
        leaf = [0] * 8 + [1] * 6 + [2, 3]
        fine = mbk.Scope.from_arrays({"x": big}, groups=leaf, ngroups=5)        # group 4: absent
        coarse = mbk.Scope.from_arrays({"x": big}, groups=[0] * 14 + [1, 1], ngroups=2)
        hb = mm.Hierarchy([fine, coarse], ["x", "x^3", "1"])
        for kind in mm.KINDS:
            cpu, gpu = hb.contrasts(0, kind), hb.contrasts(0, kind, backend="cuda")
            self.assertEqual(cpu.raw(), gpu.raw())
            expect = self.oracle_contrasts(mbk.build(fine, ["x", "x^3", "1"]), mbk.build(coarse, ["x", "x^3", "1"]),
                                           [0, 0, 1, 1, 0], kind)
            self.assertEqual([list(zip(r["status"], r["value"])) for r in cpu.rows()], [e[1] for e in expect], kind)
        fs, cs = mbk.build(fine, ["x^3"]).states(), mbk.build(coarse, ["x^3"]).states()
        cross = abs(fs[0]["sum"][0] * cs[0]["count"][0] - cs[0]["sum"][0] * fs[0]["count"][0])
        self.assertGreater(cross, 1 << 129)                           # the products need the 256-bit carrier
        with self.assertRaises(ValueError):
            h.contrasts(3, "mean")                                    # the coarsest level has no parent
        with self.assertRaises(ValueError):
            h.contrasts(0, "median")
        with self.assertRaises(ValueError):
            h.contrasts(0, "mean", terms=[0, 0])

    # ------------------------------------------------------------------ localized updates
    def test_multi001_localized_update_equals_complete_reconstruction(self):
        rng = np.random.default_rng(1007)
        n = 6000
        sizes = [512, 64, 8, 1]
        exps = [(0,), (1,), (2,), (3,)]
        for skew in (False, True):
            ids, parents = array_hierarchy(rng, n, sizes, skew)
            x = rng.integers(-10**6, 10**6, n)
            valid = {"x": rng.random(n) < 0.9}
            rows = rng.random(n) < 0.9
            levels = [mbk.Scope.from_arrays({"x": x}, valid=valid, groups=ids[k], ngroups=sizes[k], rows=rows)
                      for k in range(len(sizes))]
            h = mm.Hierarchy(levels, exps)
            for frac in (0.01, 0.1, 1.0):
                leaves = rng.choice(sizes[0], max(1, int(frac * sizes[0])), replace=False)
                x2 = x.copy()
                hit = np.isin(ids[0], leaves)
                x2[hit] = rng.integers(-10**6, 10**6, int(hit.sum()))
                new = [mbk.Scope.from_arrays({"x": x2}, valid=valid, groups=ids[k], ngroups=sizes[k], rows=rows)
                       for k in range(len(sizes))]
                complete = mm.Hierarchy(new, exps)                    # complete reconstruction
                fine = mbk.build(new[0], exps)
                carried = h.update(fine)
                self.assertEqual(carried.status, "carried")
                for k in range(len(sizes)):
                    self.assertEqual(carried.states(k), complete.states(k), (skew, frac, k))
                info = carried.info
                self.assertEqual(info["changed"], len(set(ids[0][hit & rows].tolist())))
                if frac == 0.01:
                    self.assertEqual(info["strategy"][1], "local")
                    self.assertLess(info["work"], complete.info["work"] + sizes[0] * len(exps) + sizes[0])
                if frac == 1.0:
                    self.assertEqual(info["strategy"][1], "rollup")      # adaptive: most children moved
                proven = h.update(fine, levels=new)                   # re-certified on the new data
                self.assertEqual((proven.status, proven.info["scoped"]), ("proven", [True] * len(sizes)))
                for k in range(len(sizes)):
                    self.assertTrue(proven.bank(k).equal(complete.bank(k)))
            # a change of parents: remapped children move their states to their new parents
            moved = parents[0].copy()
            for g in rng.choice(sizes[0], 6, replace=False):
                moved[g] = (moved[g] + 1) % sizes[1]
            ids2 = [ids[0], moved[ids[0]]]
            for k in range(2, len(sizes)):
                ids2.append(parents[k - 1][ids2[-1]])
            new = [mbk.Scope.from_arrays({"x": x}, valid=valid, groups=ids2[k], ngroups=sizes[k], rows=rows)
                   for k in range(len(sizes))]
            remapped = h.update(mbk.build(new[0], exps), levels=new)
            complete = mm.Hierarchy(new, exps)
            self.assertEqual(remapped.info["changed"], 0)
            self.assertEqual(remapped.info["strategy"][1], "local")
            for k in range(len(sizes)):
                self.assertTrue(remapped.bank(k).equal(complete.bank(k)), (skew, k))
        # a leaf without rows at certification has no parent in the carried map: rows arriving there are refused
        # (certify the new data instead); a declared map composed with a proven one is declared
        sizes = [16, 4, 1]
        leaf = rng.integers(0, 15, 500)                               # leaf 15: no rows
        x = rng.integers(-50, 50, 500)
        ids = [leaf, leaf // 4, np.zeros(500, np.int64)]
        old = [mbk.Scope.from_arrays({"x": x}, groups=ids[k], ngroups=sizes[k]) for k in range(3)]
        hs = mm.Hierarchy(old, exps)
        self.assertIsNone(hs.map(0).parent(15))
        leaf2 = leaf.copy(); leaf2[:7] = 15
        new = [mbk.Scope.from_arrays({"x": x}, groups=g, ngroups=s) for g, s in ((leaf2, 16), (leaf2 // 4, 4),
                                                                                 (ids[2], 1))]
        with self.assertRaises(DomainError):
            hs.update(mbk.build(new[0], exps))
        self.assertTrue(hs.update(mbk.build(new[0], exps), levels=new).bank(1).equal(mbk.build(new[1], exps)))
        declared = mm.declare(old[0], old[1], np.arange(16) // 4)
        self.assertEqual(declared.compose(hs.map(1)).status, "declared")
        self.assertEqual(hs.map(0).compose(hs.map(1)).status, "proven")
        with self.assertRaises(DomainError):
            h.update(mbk.build(levels[0], [(0,), (1,)]))              # another basis
        with self.assertRaises(ValueError):
            h.update(mbk.build(levels[0], exps), maps=[h.map(0)])     # maps without their levels

    # ------------------------------------------------------------------ budgets
    def test_multi001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(1008)
        n = 4000
        ids, _ = array_hierarchy(rng, n, [256, 16, 2])
        x = rng.integers(-1000, 1000, n)
        levels = [mbk.Scope.from_arrays({"x": x}, groups=ids[k], ngroups=[256, 16, 2][k]) for k in range(3)]
        for backend in ("cpu", "cuda"):                               # certification: states and bytes
            needs = mm.map_need(levels[0], levels[1], backend)
            self.assertEqual(needs["states"], 256)
            ref = mm.certify(levels[0], levels[1])
            for resource, value in needs.items():
                self.assertTrue(mm.certify(levels[0], levels[1], backend=backend,
                                           budget=Budget(**{resource: value})).equal(ref))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mm.certify(levels[0], levels[1], backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        with self.assertRaises(AlgebraBudgetError):
            mm.declare(levels[0], levels[1], ref.ids, budget=Budget(states=255))
        bank = mbk.build(levels[0], ["1", "x", "x^2"])
        maps = [mm.certify(levels[0], levels[1]), mm.certify(levels[1], levels[2])]
        needs = mm.hierarchy_need(levels, bank)                       # the hierarchy: every resource
        self.assertEqual(needs["states"], (16 + 2) * 3)
        full = mm.Hierarchy(levels, maps=maps, bank=bank)
        for resource, value in needs.items():
            self.assertEqual(mm.Hierarchy(levels, maps=maps, bank=bank, budget=Budget(**{resource: value})).states(2),
                             full.states(2))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mm.Hierarchy(levels, maps=maps, bank=bank, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), resource)
        for backend in ("cpu", "cuda"):                               # contrasts: output and bytes
            needs = full.contrasts_need(0, backend=backend)
            self.assertEqual(needs["output"], 256 * 3)
            ref = full.contrasts(0, "mean").raw()
            for resource in ("output", "bytes"):
                value = needs[resource]
                self.assertEqual(full.contrasts(0, "mean", backend=backend, budget=Budget(**{resource: value})).raw(), ref)
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    full.contrasts(0, "mean", backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        self.assertNotEqual(full.contrasts_need(0)["bytes"], full.contrasts_need(0, backend="cuda")["bytes"])
        with self.assertRaises(AlgebraBudgetError):                   # an update: the staged states and scratch
            full.update(bank, budget=Budget(bytes=needs_bytes(full, levels, bank) - 1))
        self.assertEqual(full.update(bank, budget=Budget(bytes=needs_bytes(full, levels, bank))).status, "carried")

    def test_multi001_invalid_inputs_retained_bytes_and_opt_in_import(self):
        rng = np.random.default_rng(1009)
        n = 500
        ids, _ = array_hierarchy(rng, n, [32, 4])
        x = rng.integers(-10, 10, n)
        a = mbk.Scope.from_arrays({"x": x}, groups=ids[0], ngroups=32)
        b = mbk.Scope.from_arrays({"x": x}, groups=ids[1], ngroups=4)
        with self.assertRaises(ValueError):
            mm.Hierarchy([a], ["1"])                                  # one level
        other_rows = mbk.Scope.from_arrays({"x": x}, groups=ids[1], ngroups=4, rows=np.arange(n) % 2 == 0)
        with self.assertRaises(DomainError):
            mm.certify(a, other_rows)                                 # levels of other rows
        h = mm.Hierarchy([a, b], ["1", "x", "x^2"])
        info = h.info                                                 # retained: the banks, the map, the object
        self.assertGreater(info["bytes"], sum(info["level_bytes"]) + h.map(0).info["bytes"])
        self.assertLess(info["bytes"], sum(info["level_bytes"]) + h.map(0).info["bytes"] + 2048)
        self.assertEqual(info["level_bytes"], [h.bank(k).bytes for k in range(2)])
        c = h.bank(1)
        h.close()
        self.assertEqual(c.rows, n)                                   # a returned bank outlives its hierarchy
        with self.assertRaises(ValueError):
            h.bank(0)
        code = "import sys, torbles; print('torbles.math_multi' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


def needs_bytes(h, levels, bank):
    """the bytes an update of h checks (its native need, found by refusal at zero)"""
    try:
        h.update(bank, budget=Budget(bytes=0))
    except AlgebraBudgetError as e:
        return e.need
    raise AssertionError("an update with no bytes was admitted")


if __name__ == "__main__":
    unittest.main()
