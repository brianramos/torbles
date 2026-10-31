# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-004: data scopes, grouped exact moment banks, refinement rollups and the bounded bank cache
(torbles/math_scope.c, math_bank.c, math_bank_gpu.cu, math_bank.py).

Oracles: DuckDB SQL over the typed source table of the same Database (GROUP BY with COUNT(*), COUNT(expr) and
SUM(expr) in HUGEINT on the logical values, SQL NULL semantics, LEFT JOIN for dimension attributes, FILTER for the
listwise mode), and Python integers written here for values past HUGEINT-safe ranges. Every bank is also built on the
CUDA path (emulated here, with TORBLES_EMU_SCRAMBLE=1 for scrambled thread order) and compared with the CPU bank for
exact equality. Seeds are fixed.
"""
import ctypes as C
import math
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SCALES = {"x": 0, "d": 2, "y": 0}
DDL = """CREATE TABLE dim (id BIGINT PRIMARY KEY, region VARCHAR, tier BIGINT);
CREATE TABLE f (k BIGINT, s VARCHAR, x BIGINT, d DECIMAL(10,2), y BIGINT, c BIGINT REFERENCES dim(id));"""
BASIS = mbk.monomials(["x", "d"], 2) + [(3, 0), (2, 1), (1, 3), (0, 4), (4, 0), (2, 2)]


def obj(values):
    return pd.Series(values, dtype=object)


def dataset(rng, n, nulls=0.15):
    """a fact table with signed and decimal values, NULLs, an all-NULL group (k = 7), an absent key (k = 6) and
    foreign keys (including unmatched ones) to a dimension with a NULL attribute"""
    k = [int(v) for v in rng.integers(0, 6, n)]
    x = [int(v) for v in rng.integers(-5000, 5001, n)]
    d = [f"{int(v) / 100:.2f}" for v in rng.integers(-99999, 100000, n)]
    y = [int(v) for v in rng.integers(-300, 301, n)]
    s = [str(v) for v in rng.choice(["a", "b", "c", "d"], n)]
    c = [int(v) for v in rng.choice([10, 20, 30, 40, 50, 99], n)]          # 99 matches no dimension row
    for col in (k, x, d, s, c):
        for i in np.flatnonzero(rng.random(n) < nulls):
            col[int(i)] = None
    for i in rng.choice(n, size=min(5, n), replace=False):                 # group k = 7: every value NULL
        k[int(i)], x[int(i)], d[int(i)] = 7, None, None
    fact = pd.DataFrame({"k": obj(k), "s": obj(s), "x": obj(x), "d": obj(d), "y": obj(y), "c": obj(c)})
    dim = pd.DataFrame({"id": [10, 20, 30, 40, 50], "region": obj(["EU", "US", None, "EU", "APAC"]),
                        "tier": [1, 2, 1, 3, 2]})
    return torbles.from_ddl(DDL, {"f": fact, "dim": dim})


def logical(name):
    return f'CAST("{name}" * {10 ** SCALES[name]} AS HUGEINT)' if SCALES[name] else f'CAST("{name}" AS HUGEINT)'


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
        expr = term_sql(names, t).replace('CAST("', 'CAST(f."')
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
            continue                                      # no selected row: no group
        vals = r[nk + 1:]
        out[tuple(r[:nk])] = (int(r[nk]), [int(v) for v in vals[0::2]], [0 if v is None else int(v) for v in vals[1::2]])
    return out


def bank_states(bank):
    return {row["key"]: (row["rows"], row["count"], row["sum"]) for row in bank.table()}


def mask(rng, n, p):
    return rng.random(n) < p


class Bank(unittest.TestCase):
    def assert_absent_zero(self, bank):
        for g, s in enumerate(bank.states()):
            if not s["present"]:
                self.assertEqual((s["rows"], set(s["count"]), set(s["sum"])), (0, {0}, {0}), g)

    def both(self, scope, basis, mode="sql", **kw):
        """the CPU bank, checked equal to the (emulated, scrambled) CUDA bank: natively and word for word"""
        cpu = mbk.build(scope, basis, mode, **kw)
        gpu = mbk.build(scope, basis, mode, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu), (mode, basis))
        for a, b in zip(cpu.arrays(), gpu.arrays()):
            self.assertTrue(np.array_equal(a, b))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    # ------------------------------------------------------------------ against DuckDB
    def test_fnd004_grouped_states_match_duckdb_with_sql_nulls(self):
        rng = np.random.default_rng(4001)
        db = dataset(rng, 3000)
        t = db["f"]
        names = ["x", "d"]
        for keys, keys_sql in ((["k"], ['f."k"']), (["k", "s"], ['f."k"', 'f."s"']), ([], [])):
            for rows in (None, mask(rng, t.n, 0.6)):
                scope = mbk.Scope.from_table(t, names, group_by=keys, rows=rows)
                for mode in ("sql", "joint"):
                    bank = self.both(scope, BASIS, mode)
                    expect = duck_states(db, names, BASIS, keys_sql, mode, rows)
                    self.assertEqual(bank_states(bank), expect, (keys, mode))
                    self.assertEqual(bank.present, len(expect))
                    self.assertEqual(bank.rows, scope.selected)
                    self.assert_absent_zero(bank)
                if keys == ["k"] and rows is None:        # k = 7: present, every value NULL; k = 6: absent
                    s7 = bank_states(mbk.build(scope, BASIS))[(7,)]
                    self.assertTrue(s7[0] > 0 and s7[1][0] == s7[0] and set(s7[1][1:]) == {0} and set(s7[2][1:]) == {0})
                    self.assertNotIn((6,), bank_states(bank))
                    g6 = [g for g in range(scope.groups) if scope.key_values(g) == (6,)][0]
                    self.assertFalse(bank.state(g6)["present"])

    def test_fnd004_dimension_lookups_match_left_join_groups(self):
        rng = np.random.default_rng(4002)
        db = dataset(rng, 2500)
        t, dim = db["f"], db["dim"]
        join = 'LEFT JOIN dim ON f."c" = dim."id"'
        for keys, keys_sql in (([("c", dim, "region")], ['dim."region"']),
                               ([("c", dim, "tier"), "k"], ['dim."tier"', 'f."k"']),
                               ([("c", dim, "id")], ['dim."id"'])):
            rows = mask(rng, t.n, 0.8)
            scope = mbk.Scope.from_table(t, ["x", "y"], group_by=keys, rows=rows)
            basis = mbk.monomials(["x", "y"], 2)
            for mode in ("sql", "joint"):
                bank = self.both(scope, basis, mode)
                self.assertEqual(bank_states(bank), duck_states(db, ["x", "y"], basis, keys_sql, mode, rows, join), keys)
        # unmatched references (c = 99 or NULL) and the NULL region are the NULL group of the key
        scope = mbk.Scope.from_table(t, ["x"], group_by=[("c", dim, "region")])
        nulls = bank_states(mbk.build(scope, ["1"]))[(None,)][0]
        expect = db.con.execute('SELECT COUNT(*) FROM f LEFT JOIN dim ON f."c" = dim."id" WHERE dim."region" IS NULL').fetchone()[0]
        self.assertEqual(nulls, expect)

    def test_fnd004_signed_decimal_and_self_extending_columns_exact(self):
        rng = np.random.default_rng(4003)
        n = 400
        big = [int(v) for v in rng.integers(-(1 << 37), 1 << 37, n)]          # span > 2^32: a self-extending column
        cents = [int(v) for v in rng.integers(-10**10, 10**10, n)]
        for i in rng.choice(n, 40, replace=False):
            big[int(i)] = None
        frame = pd.DataFrame({"g": obj([int(v) for v in rng.integers(0, 3, n)]), "b": obj(big),
                              "m": obj([f"{c // 100}.{abs(c) % 100:02d}" if c >= 0 else f"-{abs(c) // 100}.{abs(c) % 100:02d}" for c in cents])})
        db = torbles.from_ddl("CREATE TABLE f (g BIGINT, b BIGINT, m DECIMAL(18,2))", {"f": frame})
        t = db["f"]
        self.assertIn("b", t.wide)
        scope = mbk.Scope.from_table(t, ["b", "m"], group_by=["g"])
        self.assertEqual(scope.column("b")["storage"], "values")
        basis = mbk.monomials(["b", "m"], 2) + [(3, 0), (0, 3)]
        bank = self.both(scope, basis)
        groups = [int(v) for v in frame["g"]]
        for row in bank.table():
            (g,) = row["key"]
            for j, e in enumerate(basis):
                vals = [(b ** e[0] if e[0] else 1) * (m ** e[1]) for b, m, gg in zip(big, cents, groups)
                        if gg == g and (e[0] == 0 or b is not None)]
                self.assertEqual((row["count"][j], row["sum"][j]), (len(vals), sum(vals)), (g, e))
        self.assertEqual(scope.column("m")["scale"], 2)
        self.assertEqual(scope.column("b")["min"], min(v for v in big if v is not None))

    def test_fnd004_arrays_scope_matches_python_integers(self):
        rng = np.random.default_rng(4004)
        n, G = 3000, 37
        x = rng.integers(-(1 << 62), 1 << 62, n)
        y = rng.integers(-(1 << 25), 1 << 25, n)
        vx, vy = rng.random(n) < 0.8, rng.random(n) < 0.9
        groups = rng.integers(0, G, n)
        groups[groups == 5] = 6                                  # group 5 absent
        rows = rng.random(n) < 0.7
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, valid={"x": vx, "y": vy}, groups=groups, ngroups=G, rows=rows)
        basis = ["1", "x", "y", "y^2", "x*y", "y^3", "y^4"]
        for mode in ("sql", "joint"):
            bank = self.both(scope, basis, mode)
            exps = [(0, 0), (1, 0), (0, 1), (0, 2), (1, 1), (0, 3), (0, 4)]
            for g in range(G):
                st = bank.state(g)
                idx = [i for i in range(n) if rows[i] and groups[i] == g]
                self.assertEqual((st["present"], st["rows"]), (bool(idx), len(idx)))
                for j, (ex, ey) in enumerate(exps):
                    if mode == "sql":
                        ok = [i for i in idx if (not ex or vx[i]) and (not ey or vy[i])]
                    else:
                        ok = [i for i in idx if vx[i] and vy[i]]
                    vals = [int(x[i]) ** ex * int(y[i]) ** ey for i in ok]
                    self.assertEqual((st["count"][j], st["sum"][j]), (len(vals), sum(vals)), (mode, g, j))
            self.assertFalse(bank.state(5)["present"])
        # the descriptor is canonical: values at NULL or unselected rows do not matter, nor an all-present mask
        x2 = x.copy(); x2[~(rows & vx)] = 12345
        same = mbk.Scope.from_arrays({"x": x2, "y": y}, valid={"x": vx, "y": vy}, groups=groups, ngroups=G, rows=rows)
        self.assertTrue(scope.same(same))
        a = mbk.Scope.from_arrays({"x": x}, valid={"x": np.ones(n, bool)})
        b = mbk.Scope.from_arrays({"x": x})
        self.assertTrue(a.same(b))
        self.assertFalse(a.same(mbk.Scope.from_arrays({"x": x}, rows=rows)))

    # ------------------------------------------------------------------ acceptance 1: the cache never mixes domains
    def raw_scope(self, t, specs, keys=(), rows=None):
        """a table scope from explicit native column specs (e.g. another NULL policy for the same column)"""
        cols = (mbk._Column * len(specs))(*[mbk._Column(*s) for s in specs])
        ks = (mbk._Key * max(len(keys), 1))(*[mbk._Key(*k) for k in keys])
        bits = t.used_bm if rows is None else mbk._bitmap(rows, t.n)[0]
        token = (C.c_uint8 * 16)(*t.__dict__.setdefault("_math_scope_identity", b"\x01" * 16))
        error, refusal = C.c_int32(), mbk._Refusal()
        h = mbk._call("tr_scope_from_table", t.at, token, bits.ctypes.data, cols, len(specs), ks, len(keys),
                      C.byref(Budget().raw), C.byref(refusal), C.byref(error))
        s = mbk.Scope._new()
        s._handle = mbk._made(h, error, refusal, "raw scope")
        s.source, s.table, s.names, s.keys, s._rows = "table", t, ("x", "d"), (), None if rows is None else bits
        return s

    def test_fnd004_acceptance_predicate_null_policy_revision_or_grain_never_hit(self):
        rng = np.random.default_rng(4005)
        db = dataset(rng, 2000)
        t, dim = db["f"], db["dim"]
        p = mask(rng, t.n, 0.5)
        cache = mbk.Cache(capacity=64, max_bytes=64 << 20)
        base = mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=p)
        original = cache.bank(base, BASIS, "sql")
        self.assertFalse(original.cache_hit)
        again = cache.bank(mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=p.copy()), BASIS, "sql")
        self.assertTrue(again.cache_hit)                          # the same domain from a new scope object: a hit
        self.assertTrue(again.equal(original))
        q = p.copy(); q[int(np.flatnonzero(p)[0])] = False        # a predicate that differs in one row
        x, d = t.idx("x"), t.idx("d")
        vx, vd = t.idx(t.nullable["x"]), t.idx(t.nullable["d"])
        ox, od = t.codecs["x"].affine[0], t.codecs["d"].affine[0]
        variants = {
            "predicate": (mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=q), BASIS, "sql"),
            "other predicate": (mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=~p), BASIS, "sql"),
            "all rows": (mbk.Scope.from_table(t, ["x", "d"], group_by=["k"]), BASIS, "sql"),
            "listwise NULLs": (base, BASIS, "joint"),
            "x never NULL": (self.raw_scope(t, [(x, -1, -1, 0, ox, 0, 0), (d, vd, -1, 0, od, 2, 0)],
                                            keys=[(t.idx("k"), t.idx(t.nullable["k"]), 1, t.maxcode("k") + 1,
                                                   t.maxcode("k") + 2, None, None, 0)], rows=p), BASIS, "sql"),
            "grain k, s": (mbk.Scope.from_table(t, ["x", "d"], group_by=["k", "s"], rows=p), BASIS, "sql"),
            "grain s": (mbk.Scope.from_table(t, ["x", "d"], group_by=["s"], rows=p), BASIS, "sql"),
            "grain none": (mbk.Scope.from_table(t, ["x", "d"], rows=p), BASIS, "sql"),
            "grain region": (mbk.Scope.from_table(t, ["x", "d"], group_by=[("c", dim, "region")], rows=p), BASIS, "sql"),
            "grain tier": (mbk.Scope.from_table(t, ["x", "d"], group_by=[("c", dim, "tier")], rows=p), BASIS, "sql"),
            "basis": (base, BASIS[:-1], "sql"),
            "basis order": (base, BASIS[::-1], "sql"),
        }
        # the "x never NULL" scope must keep the same key spec as `base`: only its NULL policy differs
        raw_k = self.raw_scope(t, [(x, vx, -1, 0, ox, 0, 0), (d, vd, -1, 0, od, 2, 0)],
                               keys=[(t.idx("k"), t.idx(t.nullable["k"]), 1, t.maxcode("k") + 1, t.maxcode("k") + 2,
                                      None, None, 0)], rows=p)
        self.assertTrue(raw_k.same(base))                         # control: the raw path describes `base` exactly
        for i, (name, (scope, basis, mode)) in enumerate(variants.items()):
            self.assertIsNone(cache.get(scope, basis, mode), name)
            built = cache.bank(scope, basis, mode, backend=("cpu", "cuda")[i % 2])   # now live in the cache too
            self.assertFalse(built.cache_hit, name)
        stats = cache.stats
        self.assertEqual(stats["entries"], 1 + len(variants))
        self.assertEqual(stats["evictions"], 0)
        for name, (scope, basis, mode) in variants.items():       # every live entry answers only its own domain
            hit = cache.get(scope, basis, mode)
            self.assertTrue(hit is not None and hit.equal(mbk.build(scope, basis, mode)), name)
        self.assertTrue(cache.get(base, BASIS, "sql").equal(original))
        # revision: a mutation, even one flipped back, makes every earlier scope stale and the new one another domain
        t.flip(0, 0); t.flip(0, 0)
        with self.assertRaises(StaleStateError):
            cache.get(base, BASIS, "sql")
        with self.assertRaises(StaleStateError):
            mbk.build(base, BASIS)
        fresh = mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=p)
        self.assertFalse(fresh.same(base))
        self.assertIsNone(cache.get(fresh, BASIS, "sql"))
        self.assertTrue(cache.bank(fresh, BASIS, "sql").equal(original))   # same data, built again, not hit

    # ------------------------------------------------------------------ acceptance 2: exact rollup with presence
    def test_fnd004_acceptance_refined_states_roll_up_exactly_with_presence(self):
        rng = np.random.default_rng(4006)
        db = dataset(rng, 3000)
        t, dim = db["f"], db["dim"]
        rows = mask(rng, t.n, 0.7)
        cases = [(["k", "s"], ["k"], ['f."k"'], ""), (["k", "s"], ["s"], ['f."s"'], ""), (["k"], [], [], ""),
                 ([("c", dim, "id")], [("c", dim, "region")], ['dim."region"'], 'LEFT JOIN dim ON f."c" = dim."id"'),
                 ([("c", dim, "id"), "k"], [("c", dim, "tier")], ['dim."tier"'], 'LEFT JOIN dim ON f."c" = dim."id"')]
        for fine_keys, coarse_keys, coarse_sql, join in cases:
            fine = mbk.Scope.from_table(t, ["x", "d"], group_by=fine_keys, rows=rows)
            coarse = mbk.Scope.from_table(t, ["x", "d"], group_by=coarse_keys, rows=rows)
            for mode in ("sql", "joint"):
                fb = self.both(fine, BASIS, mode)
                direct = self.both(coarse, BASIS, mode)
                for backend in ("cpu", "cuda"):
                    up = fb.rollup(coarse, backend=backend)
                    self.assertTrue(up.equal(direct), (fine_keys, coarse_keys, mode, backend))
                    self.assertEqual(up.states(), direct.states())
                    self.assertTrue(up.info["scoped"])
                expect = duck_states(db, ["x", "d"], BASIS, coarse_sql, mode, rows, join)
                self.assertEqual(bank_states(up), expect, (coarse_keys, mode))
                self.assert_absent_zero(up)
        # explicit presence: k = 7 (all NULL) stays present with zero counts; k = 6 stays absent
        fine = mbk.Scope.from_table(t, ["x", "d"], group_by=["k", "s"], rows=rows)
        coarse = mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=rows)
        up = mbk.build(fine, BASIS).rollup(coarse)
        by_key = {coarse.key_values(g): up.state(g) for g in range(coarse.groups)}
        s7 = by_key[(7,)]
        self.assertTrue(s7["present"] and s7["rows"] > 0 and s7["count"][1:] == [0] * (len(BASIS) - 1)
                        and s7["sum"][1:] == [0] * (len(BASIS) - 1))
        self.assertEqual((by_key[(6,)]["present"], by_key[(6,)]["rows"]), (False, 0))
        # a refinement is required: k does not determine s; scopes of other rows are another domain
        with self.assertRaises(DomainError):
            mbk.build(coarse, BASIS).rollup(fine)
        with self.assertRaises(DomainError):
            mbk.build(fine, BASIS).rollup(mbk.Scope.from_table(t, ["x", "d"], group_by=["k"]))
        # an explicit map: unscoped result, the same states
        fmap = fine.rollup_map(coarse)
        for backend in ("cpu", "cuda"):
            explicit = mbk.build(fine, BASIS).rollup(map=fmap, groups=coarse.groups, backend=backend)
            self.assertTrue(explicit.equal(mbk.build(coarse, BASIS)))
            self.assertFalse(explicit.info["scoped"])

    # ------------------------------------------------------------------ chunks, merges, empty selections
    def test_fnd004_chunked_and_merged_accumulation_equals_one_shot(self):
        rng = np.random.default_rng(4007)
        db = dataset(rng, 5000)
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["x", "d", "y"], group_by=["k", "s"], rows=mask(rng, t.n, 0.75))
        basis = mbk.monomials(["x", "d", "y"], 2) + [(1, 1, 1), (0, 0, 4)]
        for mode in ("sql", "joint"):
            one = mbk.build(scope, basis, mode)
            for trial in range(4):
                cuts = sorted(set(int(v) for v in rng.integers(0, t.n, int(rng.integers(1, 30))))) + [t.n]
                ranges = list(zip([0] + cuts[:-1], cuts))
                order = rng.permutation(len(ranges))
                parts = [mbk.Builder(scope, basis, mode) for _ in range(3)]
                for i, r in enumerate(order):
                    parts[i % 3].accumulate(*ranges[int(r)])
                parts[2].merge(parts[0]).merge(parts[1])
                self.assertTrue(parts[2].finish().equal(one), (mode, trial))
            seq = mbk.Builder(scope, basis, mode)
            for a in range(0, t.n, 777):
                seq.accumulate(a, min(t.n, a + 777))
            self.assertTrue(seq.finish().equal(one))
        b = mbk.Builder(scope, basis)
        b.accumulate(0, 100)
        with self.assertRaises(ValueError):
            b.accumulate(50, 150)                                 # a row is never counted twice
        with self.assertRaises(ValueError):
            b.finish()                                            # rows 100.. not covered
        other = mbk.Builder(scope, basis)
        other.accumulate(99, 200)
        with self.assertRaises(ValueError):
            b.merge(other)                                        # overlapping coverage
        with self.assertRaises(ValueError):
            b.merge(mbk.Builder(scope, basis[:-1]))               # another basis
        with self.assertRaises(ValueError):
            b.merge(mbk.Builder(scope, basis, "joint"))           # another mode
        twin = mbk.Scope.from_table(t, ["x", "d", "y"], group_by=["k", "s"], rows=scope._rows.copy())
        rest = mbk.Builder(twin, basis).accumulate(100, t.n)      # an equal descriptor from another scope object
        self.assertTrue(b.merge(rest).finish().equal(mbk.build(scope, basis)))

    def test_fnd004_empty_selection_and_absent_groups(self):
        rng = np.random.default_rng(4008)
        db = dataset(rng, 700)
        t = db["f"]
        for keys in (["k"], []):
            scope = mbk.Scope.from_table(t, ["x", "d"], group_by=keys, rows=np.zeros(t.n, bool))
            self.assertEqual(scope.selected, 0)
            self.assertEqual(scope.column("x")["valid"], 0)
            self.assertIsNone(scope.column("x")["min"])
            for mode in ("sql", "joint"):
                bank = self.both(scope, BASIS, mode)
                self.assertEqual((bank.present, bank.rows, bank.table()), (0, 0, []))
                self.assert_absent_zero(bank)
        empty = mbk.Scope.from_arrays({"x": np.zeros(0, np.int64)})
        bank = self.both(empty, ["1", "x", "x^2"])
        self.assertEqual((bank.groups, bank.present), (1, 0))
        scope = mbk.Scope.from_arrays({"x": [1, 2, 3]}, groups=[2, 2, 0], ngroups=4)
        bank = self.both(scope, ["1", "x"])
        self.assertEqual([s["present"] for s in bank.states()], [True, False, True, False])
        self.assertEqual(bank.state(2)["sum"], [2, 3])

    # ------------------------------------------------------------------ exactness and refusals
    def test_fnd004_overflow_is_refused_never_wrapped(self):
        big = 1 << 62
        for n, ok in ((7, True), (8, False)):          # n * (2^62)^2 <= 2^127 - 1 iff n <= 7
            vals = [big if i % 2 else -big for i in range(n)]
            scope = mbk.Scope.from_arrays({"x": vals})
            if ok:
                bank = self.both(scope, ["x", "x^2"])
                self.assertEqual(bank.state(0)["sum"], [sum(vals), sum(v * v for v in vals)])
            else:
                for backend in ("cpu", "cuda"):
                    with self.assertRaises(OverflowError):
                        mbk.build(scope, ["x", "x^2"], backend=backend)
                self.both(scope, ["x"])                     # degree one still fits
        # the bound is per term and uses only the rows a term can be valid on
        x = [3, -(1 << 62), 5, 7]
        scope = mbk.Scope.from_arrays({"x": x, "y": [1, 2, 3, 4]}, valid={"x": [True, True, True, True],
                                                                          "y": [True, False, False, False]})
        bank = self.both(scope, ["x*y", "y^4"])
        self.assertEqual(bank.state(0)["sum"], [3, 1])
        with self.assertRaises(OverflowError):
            mbk.build(scope, ["x^3"])
        # a logical value past int64 (code + offset) is refused when the scope is built
        db = torbles.from_ddl("CREATE TABLE f (x BIGINT)", {"f": pd.DataFrame({"x": [5, 20, 7]})})
        t = db["f"]
        cols = (mbk._Column * 1)(mbk._Column(t.idx("x"), -1, -1, 0, (1 << 63) - 10, 0, 0))
        error, refusal = C.c_int32(), mbk._Refusal()
        h = mbk._call("tr_scope_from_table", t.at, None, None, cols, 1, None, 0, None, C.byref(refusal), C.byref(error))
        self.assertEqual((h, error.value), (None, -3))
        # a rollup whose sums cannot fit is refused, on the CPU and on the device
        m = 1
        states, keep = mbk._alloc_states(2, m)
        rows, count, sums = keep
        rows[:] = 1; count[:] = 1
        sums[0::2] = 0; sums[1::2] = 3 << 61                     # each 3 * 2^125: fits; their total does not
        exps = np.array([[1]], dtype=np.uint8)
        h = mbk._call("tr_bank_from_states", None, 1, exps.ctypes.data, 1, 0, C.byref(states), None, None, C.byref(error))
        self.assertTrue(h and error.value == 0)
        unscoped = mbk.Bank._wrap(h, None, ("x",), "cpu", "native CPU")
        self.assertEqual(unscoped.state(0)["sum"], [3 << 125])
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):
                unscoped.rollup(map=[0, 0], groups=1, backend=backend)
            self.assertEqual(unscoped.rollup(map=[0, 1], groups=2, backend=backend).states(), unscoped.states())

    def test_fnd004_adopted_states_are_checked(self):
        scope = mbk.Scope.from_arrays({"x": [1, -2, 3]}, valid={"x": [True, True, False]}, groups=[0, 1, 1], ngroups=2)
        exps = mbk._basis(("x",), ["1", "x"])
        good = mbk.build(scope, ["1", "x"])
        r, cnt, lo, hi = good.arrays()
        def adopt(rows, count, s):
            st, keep = mbk._alloc_states(2, 2)
            keep[0][:] = rows
            keep[1][:] = np.asarray(count, np.uint64).ravel()
            w = [int(v) for v in np.asarray(s, dtype=object).ravel()]
            keep[2][0::2] = np.array([v & ((1 << 64) - 1) for v in w], dtype=np.uint64).view(np.int64)
            keep[2][1::2] = [v >> 64 for v in w]
            error, refusal = C.c_int32(), mbk._Refusal()
            h = mbk._call("tr_bank_from_states", scope._handle, 1, exps.ctypes.data, 2, 0, C.byref(st), None,
                          C.byref(refusal), C.byref(error))
            if h:
                return mbk.Bank._wrap(h, scope, ("x",), "cpu", "native CPU")
            return error.value
        ok = adopt([1, 2], [[1, 1], [2, 1]], [[1, 1], [2, -2]])               # the states the CPU computes
        self.assertTrue(isinstance(ok, mbk.Bank) and ok.equal(good))
        self.assertEqual(adopt([1, 1], [[1, 1], [2, 1]], [[1, 1], [2, -2]]), -1)    # total rows differ
        self.assertEqual(adopt([1, 2], [[1, 1], [2, 3]], [[1, 1], [2, -2]]), -1)    # count > rows
        self.assertEqual(adopt([1, 2], [[1, 0], [2, 1]], [[1, 1], [2, -2]]), -1)    # a sum without valid rows
        self.assertEqual(adopt([1, 2], [[1, 1], [2, 1]], [[1, 1], [3, -2]]), -1)    # the constant term
        self.assertEqual(adopt([1, 2], [[1, 1], [2, 1]], [[1, 4], [2, -2]]), -1)    # past the proven bound

    def test_fnd004_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(4009)
        db = dataset(rng, 1500)
        t = db["f"]
        rows = mask(rng, t.n, 0.6)
        for resource in ("states", "bytes"):                      # the scope: states = groups, bytes = decoded arrays
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mbk.Scope.from_table(t, ["x", "d"], group_by=["k", "s"], rows=rows, budget=Budget(**{resource: 0}))
            need = ctx.exception.need
            self.assertEqual(ctx.exception.resource, resource)
            scope = mbk.Scope.from_table(t, ["x", "d"], group_by=["k", "s"], rows=rows, budget=Budget(**{resource: need}))
            with self.assertRaises(AlgebraBudgetError):
                mbk.Scope.from_table(t, ["x", "d"], group_by=["k", "s"], rows=rows, budget=Budget(**{resource: need - 1}))
        self.assertEqual(scope.groups, (t.maxcode("k") + 2) * (t.maxcode("s") + 2))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mbk.Scope.from_arrays({"x": [1, 2]}, groups=[0, 1], ngroups=10, budget=Budget(states=9))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", 10))
        for backend in ("cpu", "cuda"):                           # the bank: every resource, at and past its limit
            needs = mbk.need(scope, BASIS, "sql", backend)
            self.assertEqual(needs["states"], scope.groups * len(BASIS))
            self.assertEqual((needs["support"], needs["degree"], needs["variables"]), (len(BASIS), 4, 2))
            ref = mbk.build(scope, BASIS)
            for resource, value in needs.items():
                self.assertTrue(mbk.build(scope, BASIS, backend=backend, budget=Budget(**{resource: value})).equal(ref))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mbk.build(scope, BASIS, backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        self.assertNotEqual(mbk.need(scope, BASIS, "sql", "cpu")["bytes"], mbk.need(scope, BASIS, "sql", "cuda")["bytes"])
        # the device copies alone, at the native entry point
        exps = mbk._basis(scope.names, BASIS)
        dev = mbk._need("tr_bank_need", scope._handle, exps.ctypes.data, len(BASIS), 0, 1)
        dbytes = int(dev.need[8])
        gpu = mbk._gpu()
        for limit, code in ((dbytes, 0), (dbytes - 1, -5)):
            st, keep = mbk._alloc_states(scope.groups, len(BASIS))
            refusal, view = mbk._Refusal(), scope.view()
            rc = gpu.lib().tr_mgpu_bank_build(C.byref(view), exps.ctypes.data, len(BASIS), 0,
                                              C.byref(Budget(bytes=limit).raw), C.byref(st), C.byref(refusal))
            self.assertEqual(rc, code)
            if code:
                self.assertEqual((refusal.resource, refusal.need), (8, dbytes))
        # the rollup: the coarse bank (CPU), the device copies and the adopted bank (CUDA)
        coarse = mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=rows)
        fine = mbk.build(scope, BASIS)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                fine.rollup(coarse, backend=backend, budget=Budget(bytes=0))
            need = ctx.exception.need
            self.assertTrue(fine.rollup(coarse, backend=backend, budget=Budget(bytes=need)).equal(mbk.build(coarse, BASIS)))
            with self.assertRaises(AlgebraBudgetError):
                fine.rollup(coarse, backend=backend, budget=Budget(bytes=need - 1))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                fine.rollup(coarse, backend=backend, budget=Budget(states=coarse.groups * len(BASIS) - 1))
            self.assertEqual(ctx.exception.resource, "states")
        # the cache: an entry fits exactly at max_bytes, not one byte below
        probe = mbk.Cache(capacity=4, max_bytes=1 << 30)
        fixed = probe.stats["bytes"]
        probe.put(fine)
        entry = probe.stats["bytes"] - fixed
        exact = mbk.Cache(capacity=4, max_bytes=fixed + entry)
        exact.put(fine)
        self.assertEqual(exact.stats["entries"], 1)
        tight = mbk.Cache(capacity=4, max_bytes=fixed + entry - 1)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            tight.put(fine)
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit),
                         ("bytes", fixed + entry, fixed + entry - 1))
        self.assertEqual(tight.stats["entries"], 0)
        with self.assertRaises(AlgebraBudgetError):
            mbk.Cache(capacity=128, max_bytes=1000)               # the fixed entry array alone is larger
        with self.assertRaises(ValueError):
            mbk.Cache(capacity=0)
        with self.assertRaises(ValueError):
            mbk.Cache(capacity=129)

    def test_fnd004_staleness_after_a_mutation(self):
        rng = np.random.default_rng(4010)
        db = dataset(rng, 600)
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["x", "d"], group_by=["k"])
        explicit = np.ones((t.n + 63) // 64, dtype=np.uint64) * np.uint64(0xFFFFFFFFFFFFFFFF)
        bitmap_scope = mbk.Scope.from_table(t, ["x"], rows=explicit)
        self.assertTrue(scope.fresh() and bitmap_scope.fresh())
        bank = mbk.build(scope, BASIS)
        cache = mbk.Cache(4)
        cache.put(bank)
        explicit[0] ^= np.uint64(1)                               # the caller's selection bitmap changed
        self.assertFalse(bitmap_scope.fresh())
        before = scope.revision
        t.flip(3, 5)                                              # one stored bit: a new revision
        self.assertFalse(scope.fresh())
        for fn in (lambda: mbk.build(scope, BASIS), lambda: mbk.build(scope, BASIS, backend="cuda"),
                   lambda: cache.get(scope, BASIS), lambda: cache.put(bank), lambda: mbk.Builder(scope, BASIS),
                   lambda: bank.rollup(mbk.Scope.from_table(t, ["x", "d"]))):
            with self.assertRaises(StaleStateError):
                fn()
        now = mbk.Scope.from_table(t, ["x", "d"], group_by=["k"])
        self.assertEqual(now.revision, before + 1)
        self.assertTrue(now.fresh())
        self.assertFalse(now.same(scope))
        self.assertIsNone(cache.get(now, BASIS))
        refreshed = cache.refresh(now, BASIS)                     # rebuild and replace: the old version is dropped
        stats = cache.stats
        self.assertEqual((stats["entries"], stats["refreshes"], stats["invalidations"]), (1, 1, 1))
        self.assertTrue(cache.get(now, BASIS).equal(refreshed))
        cuda_refresh = cache.refresh(now, BASIS, backend="cuda")
        self.assertTrue(cuda_refresh.equal(refreshed))
        self.assertEqual(cache.stats["entries"], 1)
        self.assertTrue(mbk.Scope.from_arrays({"x": [1]}).fresh())
        # a lookup is a snapshot of its dimension: a mutation of the dimension makes the scope stale too
        dim = db["dim"]
        by_region = mbk.Scope.from_table(t, ["x"], group_by=[("c", dim, "region")])
        self.assertTrue(by_region.fresh())
        dim.flip(0, 1)
        self.assertFalse(by_region.fresh())
        with self.assertRaises(StaleStateError):
            mbk.build(by_region, ["1", "x"])

    def test_fnd004_cache_eviction_by_capacity_and_bytes(self):
        rng = np.random.default_rng(4011)
        db = dataset(rng, 800)
        t = db["f"]
        scopes = [mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=mask(rng, t.n, 0.5)) for _ in range(4)]
        banks = [self.both(s, BASIS) for s in scopes]
        cache = mbk.Cache(capacity=2, max_bytes=1 << 30)
        cache.put(banks[0]); cache.put(banks[1])
        self.assertIsNotNone(cache.get(scopes[0], BASIS))         # 0 becomes the most recent
        cache.put(banks[2])                                       # capacity: evicts 1 (least recently used)
        self.assertIsNone(cache.get(scopes[1], BASIS))
        self.assertTrue(cache.get(scopes[0], BASIS).equal(banks[0]))
        self.assertTrue(cache.get(scopes[2], BASIS).equal(banks[2]))
        s = cache.stats
        self.assertEqual((s["entries"], s["evictions"], s["puts"], s["hits"], s["misses"]), (2, 1, 3, 3, 1))
        cache.put(banks[2])                                       # replacing the same identity is not an eviction
        self.assertEqual((cache.stats["entries"], cache.stats["evictions"]), (2, 1))
        probe = mbk.Cache(capacity=8, max_bytes=1 << 30)
        fixed = probe.stats["bytes"]
        sizes = []
        for b in banks[:3]:
            before = probe.stats["bytes"]
            probe.put(b)
            sizes.append(probe.stats["bytes"] - before)
        small = mbk.Cache(capacity=8, max_bytes=fixed + sizes[0] + sizes[1] + sizes[2] - 1)
        for b in banks[:3]:
            small.put(b)
        s = small.stats                                           # bytes: the least recent is evicted to fit
        self.assertEqual((s["entries"], s["evictions"]), (2, 1))
        self.assertLessEqual(s["bytes"], s["max_bytes"])
        self.assertIsNone(small.get(scopes[0], BASIS))
        self.assertIsNotNone(small.get(scopes[2], BASIS))
        for cap in (1, 2, 3):                                     # a cyclic access pattern over 4 domains
            c = mbk.Cache(capacity=cap, max_bytes=1 << 30)
            for _ in range(3):
                for sc in scopes:
                    c.bank(sc, BASIS)
            st = c.stats
            self.assertEqual((st["hits"], st["entries"]), (0, cap))   # LRU on a cycle longer than its capacity
        c = mbk.Cache(capacity=4, max_bytes=1 << 30)
        for _ in range(3):
            for sc in scopes:
                c.bank(sc, BASIS)
        self.assertEqual((c.stats["hits"], c.stats["misses"], c.stats["evictions"]), (8, 4, 0))
        c.clear()
        self.assertEqual((c.stats["entries"], c.stats["hits"], c.stats["bytes"]), (0, 0, mbk.Cache(capacity=4).stats["bytes"]))

    def test_fnd004_returned_banks_are_independently_owned(self):
        rng = np.random.default_rng(4012)
        db = dataset(rng, 500)
        scope = mbk.Scope.from_table(db["f"], ["x", "d"], group_by=["k"])
        original = mbk.build(scope, BASIS, backend="cuda")
        self.assertTrue(original.equal(mbk.build(scope, BASIS)))
        reference = original.states()
        cache = mbk.Cache(capacity=1, max_bytes=1 << 30)
        cache.put(original)
        original.close()                                          # the cache kept its own copy
        a, b = cache.get(scope, BASIS), cache.get(scope, BASIS)
        self.assertNotEqual(a._handle, b._handle)
        self.assertEqual(a.states(), reference)
        a.close()
        self.assertEqual(b.states(), reference)                   # closing one copy leaves the other
        cache.put(mbk.build(mbk.Scope.from_table(db["f"], ["x", "d"]), BASIS))   # evicts the entry b came from
        self.assertEqual(cache.stats["evictions"], 1)
        self.assertEqual(b.states(), reference)
        cache.clear(); cache.close()
        self.assertEqual(b.states(), reference)                   # and so do clearing and freeing the cache
        scope.close()
        self.assertEqual(b.states(), reference)                   # banks do not reference their scope natively

    # ------------------------------------------------------------------ CUDA and sizing
    def test_fnd004_cuda_equals_cpu_on_many_groups_scrambled(self):
        rng = np.random.default_rng(4013)
        n = 20000
        cols = {c: rng.integers(-(1 << 20), 1 << 20, n) for c in ("a", "b", "c")}
        valid = {"a": rng.random(n) < 0.9, "b": rng.random(n) < 0.95}
        basis = mbk.monomials(["a", "b", "c"], 2) + ["a^4", "a*b*c", "a^2*b*c", "c^3"]
        for G, order in ((1, None), (7, "scattered"), (3000, "scattered"), (3000, "clustered"), (20000, "scattered")):
            groups = None if G == 1 else rng.integers(0, G, n)
            if order == "clustered":
                groups = np.sort(groups)
            rows = rng.random(n) < 0.8
            scope = mbk.Scope.from_arrays(cols, valid=valid, groups=groups, ngroups=G if G > 1 else None, rows=rows)
            for mode in ("sql", "joint"):
                bank = self.both(scope, basis, mode)
                self.assertEqual(bank.rows, int(rows.sum()))
            cpu = mbk.build(scope, basis)
            target = rng.integers(0, 50, G).astype(np.uint32)
            for backend in ("cpu", "cuda"):
                up = cpu.rollup(map=target, groups=50, backend=backend)
                if backend == "cpu":
                    ref = up
                self.assertTrue(up.equal(ref))
            direct = mbk.build(mbk.Scope.from_arrays(cols, valid=valid, groups=target[groups] if G > 1 else
                                                     np.repeat(target[0], n), ngroups=50, rows=rows), basis)
            self.assertTrue(ref.equal(direct), G)

    def test_fnd004_thirty_two_columns_and_561_term_basis(self):
        rng = np.random.default_rng(4014)
        n = 300
        names = [f"v{i}" for i in range(32)]
        cols = {nm: rng.integers(-1000, 1001, n) for nm in names}
        valid = {nm: rng.random(n) < 0.95 for nm in names[:8]}
        groups = rng.integers(0, 3, n)
        scope = mbk.Scope.from_arrays(cols, valid=valid, groups=groups, ngroups=3)
        basis = mbk.monomials(names, 2)
        self.assertEqual(len(basis), 561)
        bank = self.both(scope, basis)
        for g in range(3):
            st = bank.state(g)
            for j in rng.choice(len(basis), 25, replace=False):
                e = basis[int(j)]
                idx = [i for i in range(n) if groups[i] == g and all(not e[c] or valid.get(names[c], np.ones(n, bool))[i]
                                                                      for c in range(32))]
                vals = [math.prod(int(cols[names[c]][i]) ** e[c] for c in range(32)) for i in idx]
                self.assertEqual((st["count"][int(j)], st["sum"][int(j)]), (len(vals), sum(vals)))
        joint = self.both(scope, basis, "joint")
        self.assertEqual(joint.state(0)["count"][0],
                         sum(1 for i in range(n) if groups[i] == 0 and all(valid[nm][i] for nm in valid)))
        with self.assertRaises(ValueError):
            mbk.Scope.from_arrays({f"v{i}": [1] for i in range(33)})
        triples = [tuple(1 if c in (a, b, d) else 0 for c in range(32))
                   for a in range(32) for b in range(a + 1, 32) for d in range(b + 1, 32)][:464]
        with self.assertRaises(UnsupportedError):
            mbk.build(scope, basis + triples)                     # 1025 terms
        self.assertEqual(len(mbk._basis(names, basis + triples[:-1])), 1024)
        many = mbk._basis(names, basis + triples[:-1])
        self.assertEqual(mbk.need(scope, basis + triples[:-1])["support"], 1024)   # at the limit: admitted
        over = np.ascontiguousarray(np.vstack([many, np.eye(1, 32, dtype=np.uint8) * 4]))
        with self.assertRaises(UnsupportedError):
            mbk._need("tr_bank_need", scope._handle, over.ctypes.data, len(over), 0, 0)
        with self.assertRaises(UnsupportedError):
            mbk.build(scope, ["v0^5"])
        with self.assertRaises(UnsupportedError):
            mbk.build(scope, ["v0^2*v1^2*v2"])
        with self.assertRaises(ValueError):
            mbk.build(scope, ["v0", "v0^1"])                      # a duplicate term
        self.assertTrue(mbk.build(scope, ["v0^2*v1*v2", "v3^4", "1"]).equal(
            mbk.build(scope, ["v0^2*v1*v2", "v3^4", "1"], backend="cuda")))

    def test_fnd004_invalid_inputs_and_opt_in_import(self):
        with self.assertRaises(ValueError):
            mbk.Scope.from_arrays({"x": [1, 2, 3]}, groups=[0, 5, 1], ngroups=3)          # an id past ngroups
        self.assertEqual(mbk.Scope.from_arrays({"x": [1, 2, 3]}, groups=[0, 5, 1], ngroups=3,
                                               rows=np.array([True, False, True])).groups, 3)   # unselected: ignored
        db = torbles.from_ddl("CREATE TABLE f (k BIGINT, x BIGINT)",
                              {"f": pd.DataFrame({"k": obj([1, None, 2]), "x": obj([1, 2, 3])})})
        t = db["f"]
        k, x = t.idx("k"), t.idx("x")
        def raw(key):
            cols = (mbk._Column * 1)(mbk._Column(x, -1, -1, 0, 0, 0, 0))
            keys = (mbk._Key * 1)(mbk._Key(*key))
            error, refusal = C.c_int32(), mbk._Refusal()
            h = mbk._call("tr_scope_from_table", t.at, None, None, cols, 1, keys, 1, None, C.byref(refusal), C.byref(error))
            if h:
                mbk._call("tr_scope_free", h)
            return error.value
        vk = t.idx(t.nullable["k"])
        self.assertEqual(raw((k, vk, 1, 3, 4, None, None, 0)), 0)
        self.assertEqual(raw((k, vk, 0, 0, 3, None, None, 0)), -1)        # a NULL key in a non-nullable key
        self.assertEqual(raw((k, -1, 0, 0, 1, None, None, 0)), -1)        # a code >= radix
        self.assertEqual(raw((k, vk, 1, 1, 4, None, None, 0)), -1)        # a present code equal to the NULL code
        self.assertEqual(raw((k, vk, 1, 9, 4, None, None, 0)), -1)        # NULL code >= radix
        self.assertEqual(raw((99, -1, 0, 0, 4, None, None, 0)), -1)       # no such column
        g = torbles.from_ddl("CREATE TABLE g (s VARCHAR)", {"g": pd.DataFrame({"s": ["a", "b"]})})["g"]
        with self.assertRaises(UnsupportedError):
            mbk.Scope.from_table(g, ["s"])                                # a string has no exact logical value
        scope = mbk.Scope.from_arrays({"x": [1]})
        for bad in (lambda: mbk.build(scope, ["y"]), lambda: mbk.build(scope, [(1, 2)]),
                    lambda: mbk.build(scope, ["x"], mode="pairwise"), lambda: mbk.build(scope, ["x"], backend="tpu")):
            with self.assertRaises(ValueError):
                bad()
        code = "import sys, torbles; print('torbles.math_bank' in sys.modules, 'torbles.math_scope' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False", "False"])


if __name__ == "__main__":
    unittest.main()
