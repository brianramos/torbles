# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-007: proof certificates and evidence scope.

Oracles (independent of the implementation): every relation recomputed row by row with Python integers from the
source values (the witness is the smallest violating row, the uniqueness violations the rows after a first
occurrence, coverage per group from the source keys), the stored codes decoded again natively after each mutation and
recomputed in Python, the hypothesis samplers recomputed from their documented formulas, and DuckDB SQL for
duplicate, NULL and orphan counts. Every native result is also computed by the CUDA kernel (emulated here, threads in
a scrambled order) and compared for equality. Seeds are fixed.
"""
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import algq
import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 7007
M64 = (1 << 64) - 1


def obj(values):
    return pd.Series(values, dtype=object)


def decode(table, col):
    """the stored codes of a column decoded now, natively (an independent read of the planes)"""
    out = np.empty(max(table.n, 1), dtype=np.uint32)
    algq.lib.at_decode_column(table.at, table.idx(col), out.ctypes.data)
    return out[:table.n]


def flip_column(table, col, row, predicate=None):
    """flips one stored bit so that exactly column `col` changes at `row` (and predicate(new codes) holds); returns
    the new codes. Planes that change something else are flipped back."""
    before = {c: decode(table, c) for c in table.cols}
    for p in range(table.nplanes()):
        table.flip(p, row)
        after = {c: decode(table, c) for c in table.cols}
        changed = [c for c in table.cols if not np.array_equal(before[c], after[c])]
        if changed == [col] and (predicate is None or predicate(after[col])):
            return after[col]
        table.flip(p, row)
    raise AssertionError(f"no single plane changes only {col} at row {row}")


def fact_db(rng, n):
    x = [int(v) for v in rng.integers(-60, 61, n)]
    y = [int(v) for v in rng.integers(-40, 90, n)]
    k = [int(v) for v in rng.integers(0, 6, n)]
    q = [a + b for a, b in zip(x, y)]
    near = int(rng.integers(n // 2, n))
    q[near] += 1                                               # a one-row near miss of a conservation law
    z = [2 * a + 1 for a in x]
    for i in rng.choice(n, size=max(1, n // 10), replace=False):
        z[int(i)] = None                                       # a nullable affine copy
    w = [None if v is None else v + int(rng.integers(-1, 2)) for v in z]   # within 1 of z
    m = [3 * a - 7 + 101 * int(rng.integers(-3, 4)) for a in x]            # 3x - 7 modulo 101 only
    cat = [int(v) for v in rng.integers(0, 4, n)]
    for i in rng.choice(n, size=max(1, n // 25), replace=False):
        cat[int(i)] = None
    b1 = [int(v) for v in rng.integers(0, 2, n)]
    b2 = [int(v) for v in rng.integers(0, 2, n)]
    values = {"x": x, "y": y, "k": k, "a": [3 * v - 7 for v in x], "s": [a + b for a, b in zip(x, y)],
              "p": [a * b for a, b in zip(x, y)], "q": q, "z": z, "w": w, "m": m, "cat": cat,
              "xz": [None if v is None else a for a, v in zip(x, z)],
              "id": list(range(1000, 1000 + n)), "b1": b1, "b2": b2, "b3": [u ^ v for u, v in zip(b1, b2)],
              "sq": [v * v for v in x], "big": [v * 2 ** 40 + 5 for v in x]}           # past 32 bits: int64 values
    frame = pd.DataFrame({c: obj(v) for c, v in values.items()})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in values) + ")"
    return torbles.from_ddl(ddl, {"t": frame}), values, near


def star_db(rng, n, orphans=True):
    ddl = """CREATE TABLE dim (id BIGINT PRIMARY KEY, region VARCHAR, tier BIGINT);
             CREATE TABLE f (x BIGINT, y BIGINT, c BIGINT REFERENCES dim(id));"""
    ids = [10, 20, 30, 40, 50, 60, 70, 80]
    # a NULL region and distinct tiers: no attribute qualifies for a derived surrogate, so the foreign key stores
    # the dimension row (FND-004's convention for reaching a dimension)
    dim = pd.DataFrame({"id": ids, "region": obj(["EU", "US", None, "APAC", "US", "EU", "APAC", "US"]),
                        "tier": [11, 12, 13, 14, 15, 16, 17, 18]})
    x = [int(v) for v in rng.integers(-50, 50, n)]
    c = [int(v) for v in rng.choice(ids, n)]
    if orphans:
        c[5], c[11] = 99, None                                 # an unmatched and a NULL reference
    frame = pd.DataFrame({"x": obj(x), "y": obj([2 * v + 1 for v in x]), "c": obj(c)})
    return torbles.from_ddl(ddl, {"f": frame, "dim": dim}), x, c, ids


# ------------------------------------------------------------------ the oracle
def bit_of(v, kind, bit, base):
    return int(v != 0) if kind == "nonzero" else (((v - base) & M64) >> bit) & 1


def oracle(values, mask, claim, groups=None):
    """the evidence of a claim recomputed from the source values with Python integers"""
    kind, cols, policy = claim["kind"], claim["cols"], claim.get("nulls", "forbid")
    n = len(values[cols[0]])
    checked = excluded = violations = 0
    witness = None

    def violate(at):
        nonlocal violations, witness
        violations += 1
        witness = at if witness is None else min(witness, at)
    if kind == "coverage":
        present, cats = set(), {}
        for r in range(n):
            if not mask[r]:
                continue
            g = groups[r]
            present.add(g)
            v = values[cols[0]][r]
            if v is None:
                excluded += 1
            else:
                cats.setdefault(g, set()).add(v)
        req = set(claim["required"])
        for g in sorted(present):
            checked += 1
            if not req <= cats.get(g, set()):
                violate(g)
        return dict(checked=checked, excluded=excluded, violations=violations, witness=witness)
    seen = set()
    for r in range(n):
        if not mask[r]:
            continue
        vals = [values[c][r] for c in cols]
        if kind == "parity":
            used = sorted({b[0] for b in claim["bits"]})
            nulls = [vals[i] is None for i in used]
            any_null, out_null = any(nulls), False
        else:
            out_null = kind in ("affine", "sum", "product", "refinement") and vals[0] is None
            in_null = any(v is None for i, v in enumerate(vals) if not (i == 0 and kind in ("affine", "sum", "product", "refinement")))
            any_null = out_null or in_null
        if any_null:
            if policy == "skip":
                excluded += 1
                continue
            checked += 1
            if policy == "propagate":
                if out_null != in_null:
                    violate(r)
                continue
            violate(r)
            continue
        checked += 1
        if kind == "affine":
            ok = abs(vals[0] - (claim["scale"] * vals[1] + claim["bias"])) <= claim.get("tolerance", 0)
            if claim.get("modulus"):
                ok = (vals[0] - (claim["scale"] * vals[1] + claim["bias"])) % claim["modulus"] == 0
        elif kind == "sum":
            ok = vals[0] == vals[1] + vals[2]
        elif kind == "product":
            ok = vals[0] == vals[1] * vals[2]
        elif kind == "parity":
            acc = 0
            for col, bk, bit, base in claim["bits"]:
                acc ^= bit_of(vals[col], bk, bit, base)
            ok = acc == claim["constant"]
        elif kind == "refinement":
            ok = vals[1] in claim["mapping"] and claim["mapping"][vals[1]] == vals[0]
        elif kind == "polynomial":
            total = 0
            for exps, coef in claim["terms"].items():
                t = coef
                for v, e in zip(vals, exps):
                    t *= v ** e
                total += t
            ok = total == 0
        elif kind == "unique":
            ok = vals[0] not in seen
            seen.add(vals[0])
        elif kind in ("not_null",):
            ok = True
        elif kind == "reference":
            ok = 0 <= vals[0] < claim["rows"] and claim["used"][vals[0]]
        else:
            raise AssertionError(kind)
        if not ok:
            violate(r)
    return dict(checked=checked, excluded=excluded, violations=violations, witness=witness)


def native_claim(spec):
    k = spec["kind"]
    nulls = spec.get("nulls", "forbid")
    if k == "affine":
        return mc.affine(*spec["cols"], spec["scale"], spec["bias"], nulls=nulls, tolerance=spec.get("tolerance", 0),
                         modulus=spec.get("modulus"))
    if k == "sum":
        return mc.sum_of(*spec["cols"], nulls=nulls)
    if k == "product":
        return mc.product(*spec["cols"], nulls=nulls)
    if k == "parity":
        names = spec["cols"]
        return mc.parity([(names[c], bk, bit, base) if bk != "nonzero" else (names[c], "nonzero")
                          for c, bk, bit, base in spec["bits"]], spec["constant"], nulls=nulls)
    if k == "refinement":
        return mc.refinement(*spec["cols"], spec["mapping"], nulls=nulls)
    if k == "coverage":
        return mc.coverage(spec["cols"][0], spec["required"])
    if k == "polynomial":
        return mc.polynomial(spec["terms"], spec["cols"], nulls=nulls)
    if k == "unique":
        return mc.unique(spec["cols"][0], nulls=nulls)
    if k == "not_null":
        return mc.not_null(spec["cols"][0])
    raise AssertionError(k)


SPECS = [
    dict(kind="affine", cols=["a", "x"], scale=3, bias=-7),
    dict(kind="affine", cols=["a", "x"], scale=3, bias=-6),
    dict(kind="sum", cols=["s", "x", "y"]),
    dict(kind="sum", cols=["q", "x", "y"]),
    dict(kind="product", cols=["p", "x", "y"]),
    dict(kind="product", cols=["sq", "x", "x"]),
    dict(kind="affine", cols=["z", "x"], scale=2, bias=1, nulls="forbid"),
    dict(kind="affine", cols=["z", "x"], scale=2, bias=1, nulls="skip"),
    dict(kind="affine", cols=["z", "x"], scale=2, bias=1, nulls="propagate"),
    dict(kind="affine", cols=["w", "x"], scale=2, bias=1, nulls="skip", tolerance=1),
    dict(kind="affine", cols=["m", "x"], scale=3, bias=-7, modulus=101),
    dict(kind="parity", cols=["b1", "b2", "b3"], bits=[(0, "nonzero", 0, 0), (1, "nonzero", 0, 0), (2, "nonzero", 0, 0)],
         constant=0),
    dict(kind="parity", cols=["b1", "b2"], bits=[(0, "nonzero", 0, 0), (1, "nonzero", 0, 0)], constant=1),
    dict(kind="parity", cols=["x", "a"], bits=[(0, "bit", 0, 0), (1, "bit", 0, 0)], constant=1),   # 3x - 7 flips the low bit
    dict(kind="polynomial", cols=["p", "x", "y"], terms={(1, 1, 0): 1, (0, 2, 1): -1}),          # p x - x^2 y
    dict(kind="polynomial", cols=["q", "x", "y"], terms={(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}),
    dict(kind="unique", cols=["id"]),
    dict(kind="unique", cols=["k"]),
    dict(kind="unique", cols=["z"], nulls="skip"),
    dict(kind="not_null", cols=["x"]),
    dict(kind="not_null", cols=["z"]),
    dict(kind="affine", cols=["big", "x"], scale=2 ** 40, bias=5),
    dict(kind="affine", cols=["big", "x"], scale=2 ** 40, bias=4),
]


def evidence_of(e):
    return {k: e[k] for k in ("checked", "excluded", "violations", "witness")}


class Fnd007Certificates(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)

    def scope(self, db, names, **kw):
        return mbk.Scope.from_table(db["t"], names, **kw)

    def test_fnd007_evidence_matches_python_oracle_on_cpu_and_cuda(self):
        db, values, near = fact_db(self.rng, 700)
        names = list(values)
        mask = self.rng.random(700) < 0.8
        for rows in (None, mask):
            scope = self.scope(db, names, rows=rows)
            used = np.ones(700, bool) if rows is None else mask
            claims = [native_claim(s) for s in SPECS]
            cpu = mc.verify(scope, claims, backend="cpu")
            gpu = mc.verify(scope, claims, backend="cuda")
            for spec, e, g in zip(SPECS, cpu, gpu):
                with self.subTest(spec=spec, masked=rows is not None):
                    self.assertEqual(evidence_of(e), oracle(values, used, spec))
                    self.assertEqual({k: v for k, v in e.items() if k not in ("backend", "device")},
                                     {k: v for k, v in g.items() if k not in ("backend", "device")})
                    self.assertEqual(e["holds"], e["violations"] == 0)
                    self.assertIn(g["backend"], ("cuda", "cuda-emulated"))
        self.assertEqual(oracle(values, np.ones(700, bool), SPECS[3])["witness"], near)
        full = self.scope(db, names)
        self.assertEqual(full.column("big")["storage"], "values")                    # the int64 path was read
        # UNIQUE over a wide domain: the CPU sorts; the device needs the [min, max] domain and refuses it in advance
        wide = dict(kind="unique", cols=["big"])
        self.assertEqual(evidence_of(mc.verify(full, [native_claim(wide)])[0]),
                         oracle(values, np.ones(700, bool), wide))
        need = mc.need(full, [native_claim(wide)], backend="cuda")["bytes"]
        self.assertGreater(need, 2 ** 40)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mc.verify(full, [native_claim(wide)], backend="cuda")
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
        empty = self.scope(db, names, rows=np.zeros(700, bool))                       # nothing selected: vacuous
        claims = [native_claim(s) for s in SPECS]
        for e, g in zip(mc.verify(empty, claims), mc.verify(empty, claims, backend="cuda")):
            self.assertEqual((e["checked"], e["violations"], e["holds"]), (0, 0, True))
            self.assertEqual(evidence_of(e), evidence_of(g))

    def test_fnd007_refinement_and_coverage_claims(self):
        db, values, _ = fact_db(self.rng, 500)
        parity_map = {v: v % 2 for v in range(6)}
        db2 = torbles.from_ddl('CREATE TABLE t ("k" BIGINT, "kk" BIGINT, "cat" BIGINT)',
                               {"t": pd.DataFrame({"k": obj(values["k"]), "kk": obj([v % 2 for v in values["k"]]),
                                                   "cat": obj(values["cat"])})})
        s2 = mbk.Scope.from_table(db2["t"], ["kk", "k", "cat"], group_by=["k"])
        key_of = {s2.key_values(g)[0]: g for g in range(s2.groups)}
        groups = [key_of[v] for v in values["k"]]
        vals2 = {"kk": [v % 2 for v in values["k"]], "k": values["k"], "cat": values["cat"]}
        specs = [dict(kind="refinement", cols=["kk", "k"], mapping=parity_map),
                 dict(kind="refinement", cols=["kk", "k"], mapping={**parity_map, 3: 0}),
                 dict(kind="refinement", cols=["kk", "k"], mapping={v: v % 2 for v in range(5)}),
                 dict(kind="coverage", cols=["cat"], required=[0, 1, 2]),
                 dict(kind="coverage", cols=["cat"], required=[0, 1, 2, 3, 9]),
                 dict(kind="coverage", cols=["cat"], required=[])]
        for spec in specs:
            claims = [native_claim(spec)]
            for backend in ("cpu", "cuda"):
                e = mc.verify(s2, claims, backend=backend)[0]
                with self.subTest(spec=spec, backend=backend):
                    self.assertEqual(evidence_of(e), oracle(vals2, np.ones(500, bool), spec, groups))
                    self.assertEqual(e["unit"], "groups" if spec["kind"] == "coverage" else "rows")
        cert = mc.snapshot(s2, native_claim(specs[3]))
        self.assertTrue(cert.source()["grouping_bound"])
        regrouped = mbk.Scope.from_table(db2["t"], ["kk", "k", "cat"], group_by=["kk"])
        self.assertIn("grouping", cert.fresh(regrouped).reasons)

    def test_fnd007_sampled_or_approximate_claims_never_authorize_exact_rewrites(self):
        db, values, near = fact_db(self.rng, 600)
        names = list(values)
        scope = self.scope(db, names)
        exact = mc.snapshot(scope, mc.affine("a", "x", 3, -7))
        self.assertEqual(exact.authorize_exact(scope), mc.Authorization("snapshot", "affine", 0, "forbid"))
        self.assertTrue(mc.snapshot(scope, mc.affine("a", "x", 3, -7), backend="cuda").decide(scope)["authorized"])
        # a hypothesis that holds on every row is still only a hypothesis
        h = mc.hypothesis(scope, mc.affine("a", "x", 3, -7), size=64, seed=11)
        self.assertEqual(h.kind, "hypothesis")
        self.assertEqual(h.decide(scope)["reason"], "statistical")
        with self.assertRaises(mc.NotAuthorizedError) as ctx:
            h.authorize_exact(scope)
        self.assertEqual(ctx.exception.reason, "statistical")
        # a sample that misses the near miss: the hypothesis exists, the exhaustive certificate does not
        h2 = mc.hypothesis(scope, mc.sum_of("q", "x", "y"), size=near, method="prefix")
        self.assertEqual((h2.sample["distinct"], h2.evidence["checked"], h2.evidence["violations"]), (near, near, 0))
        prefix = dict(kind="sum", cols=["q", "x", "y"])
        self.assertEqual(oracle(values, [r < near for r in range(600)], prefix)["violations"], 0)   # rows 0..near-1
        self.assertEqual(oracle(values, [True] * 600, prefix)["witness"], near)
        self.assertEqual(h2.decide(scope)["reason"], "statistical")
        with self.assertRaises(mc.ClaimViolated) as ctx:
            mc.snapshot(scope, mc.sum_of("q", "x", "y"))
        self.assertEqual(ctx.exception.evidence["witness"], near)
        with self.assertRaises(mc.ClaimViolated):
            mc.snapshot(scope, mc.sum_of("q", "x", "y"), backend="cuda")
        # approximate: |w - (2x + 1)| <= 1 holds, exactly it does not
        approx = mc.snapshot(scope, mc.affine("w", "x", 2, 1, nulls="skip", tolerance=1))
        self.assertFalse(approx.info["exact"])
        self.assertEqual(approx.decide(scope)["reason"], "approximate")
        with self.assertRaises(mc.ClaimViolated):
            mc.snapshot(scope, mc.affine("w", "x", 2, 1, nulls="skip"))
        # a Z claim checked modulo 101 holds there only: approximate for Z, and not exact for GF(101) either
        modp = mc.snapshot(scope, mc.affine("m", "x", 3, -7, modulus=101))
        self.assertEqual(modp.decide(scope)["reason"], "approximate")
        self.assertEqual(modp.decide(scope, field=101)["reason"], "approximate")
        with self.assertRaises(mc.ClaimViolated):
            mc.snapshot(scope, mc.affine("m", "x", 3, -7))
        # a claim stated in GF(101) and checked there: exact in GF(101), not in Z
        gf = mc.snapshot(scope, mc.polynomial({(1, 0): 1, (0, 1): -3, (0, 0): 7}, ["m", "x"], field=101))
        self.assertTrue(gf.decide(scope, field=101)["authorized"])
        self.assertEqual(gf.decide(scope, field=0)["reason"], "carrier")
        # an exact Z snapshot also authorizes a GF(p) target (Z -> GF(p) is a ring homomorphism)
        self.assertTrue(exact.decide(scope, field=101)["authorized"])
        # a derivation from a sampled premise stays a hypothesis
        p_cert = mc.snapshot(scope, mc.product("p", "x", "y"))
        hp = mc.hypothesis(scope, mc.product("p", "x", "y"), size=10, seed=1)
        d = mc.derive(mc.polynomial({(1, 0, 0): 2, (0, 1, 1): -2}, ["p", "x", "y"]), [hp],
                      [mc.polynomial({(0,): 2}, ["p"])], scope)
        self.assertEqual(d.kind, "hypothesis")
        self.assertEqual(d.decide(scope)["reason"], "statistical")
        d2 = mc.derive(mc.polynomial({(1, 0, 0): 2, (0, 1, 1): -2}, ["p", "x", "y"]), [p_cert],
                       [mc.polynomial({(0,): 2}, ["p"])], scope)
        self.assertEqual(d2.kind, "snapshot")
        self.assertTrue(d2.decide(scope)["authorized"])

    def test_fnd007_laws_and_schema_facts_authorize(self):
        sq = mc.polynomial([(1, {"x": 2}), (2, {"x": 1, "y": 1}), (1, {"y": 2}), (-1, {"x": 2}), (-2, {"x": 1, "y": 1}),
                            (-1, {"y": 2})], ["x", "y"])
        law = mc.law(sq)
        self.assertEqual(law.kind, "law")
        self.assertTrue(law.decide()["authorized"])
        self.assertTrue(law.decide(field=2)["authorized"])
        self.assertTrue(law.fresh().fresh)
        with self.assertRaises(DomainError):
            mc.law(mc.polynomial({(2, 0): 1, (0, 2): -1}, ["x", "y"]))
        boolean = mc.law(mc.polynomial({(2,): 1, (1,): -1}, ["b"], field=2), "gfp_function")       # x^2 = x in GF(2)
        self.assertTrue(boolean.decide(field=2)["authorized"])
        self.assertEqual(boolean.decide(field=0)["reason"], "carrier")
        self.assertEqual(boolean.decide(field=3)["reason"], "carrier")
        self.assertTrue(mc.law(mc.polynomial({(3,): 1, (1,): -1}, ["b"], field=3), "gfp_function").decide(field=3)["authorized"])
        with self.assertRaises(DomainError):
            mc.law(mc.polynomial({(2,): 1, (1,): -1}, ["b"], field=3), "gfp_function")
        with self.assertRaises(DomainError):
            mc.law(mc.polynomial({(2,): 1, (1,): -1}, ["b"]))                                  # not a ring identity
        with self.assertRaises(ValueError):
            mc.law(mc.polynomial({(2,): 1, (1,): -1}, ["b"], field=2))                         # a ring identity is a Z claim
        tables = mc.describe()
        self.assertEqual(set(tables["claims"]), set(mc.CLAIMS))
        self.assertEqual(tables["claims"]["coverage"]["nulls"], ["skip"])
        # schema facts: the primary key of a dimension, verified natively; DuckDB counts the same
        db, x, c, ids = star_db(self.rng, 200)
        dim = db["dim"]
        dscope = mbk.Scope.from_table(dim, ["id", "tier"])
        pk = mc.schema(dscope, mc.unique("id"), enforcement="load")
        self.assertEqual(pk.authorize_exact(dscope).kind, "schema")
        self.assertEqual(db.con.execute("SELECT count(*) - count(DISTINCT id) FROM dim").fetchone()[0], 0)
        self.assertEqual(db.con.execute("SELECT count(*) FROM dim WHERE id IS NULL").fetchone()[0], 0)
        sub = mbk.Scope.from_table(dim, ["id", "tier"], rows=np.array([True, False] * 4))
        self.assertTrue(pk.fresh(sub).fresh)                                   # hereditary: any sub-selection
        rscope = mc.code_scope(dim, ["region"])
        with self.assertRaises(mc.ClaimViolated) as ctx:
            mc.schema(rscope, mc.unique("region", nulls="skip"))
        regions = [0, 1, None, 2, 1, 0, 2, 1]                     # region codes: dictionary ranks, NULL apart
        expect = oracle({"region": regions}, [True] * 8, dict(kind="unique", cols=["region"], nulls="skip"))
        self.assertEqual(evidence_of(ctx.exception.evidence), expect)
        self.assertEqual(ctx.exception.evidence["violations"],
                         db.con.execute("SELECT count(region) - count(DISTINCT region) FROM dim").fetchone()[0])
        # a foreign key with an unmatched and a NULL reference is not enforced: the witness is the first of them
        f = db["f"]
        refs = mc.code_scope(f, ["c"])
        with self.assertRaises(mc.ClaimViolated) as ctx:
            mc.schema(refs, mc.reference("c", dim))
        orphans = db.con.execute("SELECT count(*) FROM f WHERE c IS NULL OR c NOT IN (SELECT id FROM dim)").fetchone()[0]
        self.assertEqual(ctx.exception.evidence["violations"], orphans)
        self.assertEqual(ctx.exception.evidence["witness"], 5)
        db_ok, *_ = star_db(np.random.default_rng(SEED + 1), 120, orphans=False)
        fk = mc.schema(mc.code_scope(db_ok["f"], ["c"]), mc.reference("c", db_ok["dim"]))
        self.assertTrue(fk.authorize_exact(mc.code_scope(db_ok["f"], ["c"])))
        self.assertEqual(fk.source()["dimensions"][0]["unique"], True)
        written = mc.schema(mbk.Scope.from_table(f, ["x"]), mc.not_null("x"), enforcement="write")
        loaded = mc.schema(mbk.Scope.from_table(f, ["x"]), mc.not_null("x"), enforcement="load")
        flip_column(f, "x", 3)
        after = mbk.Scope.from_table(f, ["x"])
        self.assertTrue(written.fresh(after).fresh)                            # the storage checks every write
        self.assertEqual(loaded.fresh(after).reasons, ("revision",))           # verified at load: this revision only
        self.assertEqual(written.fresh(mbk.Scope.from_table(db_ok["f"], ["x"])).reasons, ("identity",))

    def test_fnd007_mutation_invalidates_snapshot_certificates(self):
        db, values, _ = fact_db(self.rng, 400)
        names = ["x", "y", "a", "s"]
        t = db["t"]
        scope = self.scope(db, names)
        cert = mc.snapshot(scope, mc.affine("a", "x", 3, -7))
        other = mc.snapshot(scope, mc.sum_of("s", "x", "y"))
        law = mc.law(mc.polynomial([(1, {"x": 1, "y": 1}), (-1, {"x": 1, "y": 1})], ["x", "y"]))
        revision = scope.revision
        t.flip(0, 9); t.repair()                                       # identical data, a new revision
        repaired = self.scope(db, names)
        for c in (cert, other):
            f = c.fresh(repaired)
            self.assertFalse(f.fresh)
            self.assertEqual(f.reasons, ("revision",))
            self.assertEqual(f.recorded, revision)
            with self.assertRaises(StaleStateError) as ctx:
                c.authorize_exact(repaired)
            self.assertEqual(ctx.exception.reasons, ("revision",))
        self.assertTrue(law.fresh().fresh)
        # re-verification after the invalidation: the same claims hold again, on the CPU and the device alike
        again = mc.reverify([cert, other], repaired, backend="cpu")
        again_gpu = mc.reverify([cert, other], repaired, backend="cuda")
        for a, b in zip(again, again_gpu):
            self.assertTrue(a.fresh(repaired).fresh and b.fresh(repaired).fresh)
            self.assertEqual({k: v for k, v in a.evidence.items() if k not in ("backend", "device")},
                             {k: v for k, v in b.evidence.items() if k not in ("backend", "device")})
            self.assertEqual(b.info["verified_by"], "cuda library")
        # a mutation that breaks the law of column a: the witness is the mutated row (codes decoded again)
        codes = flip_column(t, "a", 17)
        offset = t.codecs["a"].affine[0]
        now = [int(v) + offset for v in codes]
        expect = oracle({"a": now, "x": values["x"]}, [True] * 400, dict(kind="affine", cols=["a", "x"], scale=3, bias=-7))
        self.assertEqual(expect["witness"], 17)
        broken = self.scope(db, names)
        self.assertEqual(cert.fresh(broken).reasons, ("revision",))
        for backend in ("cpu", "cuda"):
            with self.assertRaises(mc.ClaimViolated) as ctx:
                cert.reverify(broken, backend=backend)
            self.assertEqual(evidence_of(ctx.exception.evidence), expect)
        self.assertTrue(other.reverify(broken, backend="cuda").fresh(broken).fresh)

    def test_fnd007_null_domain_changes_invalidate_applicable_certificates(self):
        db, values, _ = fact_db(self.rng, 300)
        names = ["x", "z", "a", "xz"]
        t = db["t"]
        scope = self.scope(db, names)
        with self.assertRaises(mc.ClaimViolated) as ctx:
            mc.snapshot(scope, mc.affine("z", "x", 2, 1))                     # FORBID: a NULL violates
        first_null = min(r for r, v in enumerate(values["z"]) if v is None)
        self.assertEqual(ctx.exception.evidence["witness"], first_null)
        skip = mc.snapshot(scope, mc.affine("z", "x", 2, 1, nulls="skip"))
        with self.assertRaises(mc.ClaimViolated) as ctx:                     # z NULL while x is present
            mc.snapshot(scope, mc.affine("z", "x", 2, 1, nulls="propagate"))
        self.assertEqual(ctx.exception.evidence["witness"], first_null)
        prop = mc.snapshot(scope, mc.affine("z", "xz", 2, 1, nulls="propagate"))   # NULL exactly with its input
        plain = mc.snapshot(scope, mc.affine("a", "x", 3, -7))
        self.assertEqual(skip.authorize_exact(scope).nulls, "skip")           # the complete rows only
        self.assertEqual(skip.evidence["excluded"], sum(v is None for v in values["z"]))
        self.assertEqual(skip.source()["columns"][1]["nulls"], sum(v is None for v in values["z"]))
        row = next(r for r, v in enumerate(values["z"]) if v is not None)
        valid = t.nullable["z"]
        flip_column(t, valid, row)                                            # z becomes NULL at row
        now = self.scope(db, names)
        for c in (skip, prop):
            f = c.fresh(now)
            self.assertEqual(set(f.reasons), {"revision", "null_domain"})
            self.assertEqual(f.first, "revision")
        self.assertEqual(plain.fresh(now).reasons, ("revision",))           # NULL domain of z: not applicable
        renewed = skip.reverify(now, backend="cuda")
        self.assertEqual(renewed.evidence["excluded"], skip.evidence["excluded"] + 1)
        with self.assertRaises(mc.ClaimViolated) as ctx:
            prop.reverify(now)                                               # now z is NULL while xz is not
        self.assertEqual(ctx.exception.evidence["witness"], row)
        # array scopes: one more NULL in the data
        x = np.array(values["x"], np.int64)
        z = np.array([0 if v is None else v for v in values["z"]], np.int64)
        present = np.array([v is not None for v in values["z"]])
        a1 = mbk.Scope.from_arrays({"x": x, "z": z}, valid={"z": present})
        ca = mc.snapshot(a1, mc.affine("z", "x", 2, 1, nulls="skip"))
        self.assertTrue(ca.fresh(mbk.Scope.from_arrays({"x": x, "z": z}, valid={"z": present})).fresh)
        less = present.copy(); less[row] = False
        f = ca.fresh(mbk.Scope.from_arrays({"x": x, "z": z}, valid={"z": less}))
        self.assertIn("null_domain", f.reasons)

    def test_fnd007_join_multiplicity_changes_invalidate_applicable_certificates(self):
        db, x, c, ids = star_db(self.rng, 300)
        f, dim = db["f"], db["dim"]
        reach = mbk.Scope.from_table(f, ["x", "y"], group_by=[("c", dim, "region")])
        plain = mbk.Scope.from_table(f, ["x", "y"])
        through = mc.snapshot(reach, mc.affine("y", "x", 2, 1))
        local = mc.snapshot(plain, mc.affine("y", "x", 2, 1))
        recorded = through.source()["dimensions"]
        self.assertEqual(len(recorded), 1)
        self.assertTrue(recorded[0]["unique"])
        self.assertEqual(local.source()["dimensions"], [])
        self.assertTrue(through.fresh(mbk.Scope.from_table(f, ["x", "y"], group_by=[("c", dim, "region")])).fresh)
        flip_column(dim, "tier", 2)                                          # a dimension mutation
        now = mbk.Scope.from_table(f, ["x", "y"], group_by=[("c", dim, "region")])
        fr = through.fresh(now)
        self.assertEqual(fr.reasons, ("join_revision",))
        self.assertEqual(fr.dimension, 0)
        self.assertTrue(local.fresh(mbk.Scope.from_table(f, ["x", "y"])).fresh)   # it does not reach the dimension
        with self.assertRaises(StaleStateError) as ctx:
            through.authorize_exact(now)
        self.assertEqual(ctx.exception.reasons, ("join_revision",))
        through2 = through.reverify(now)
        self.assertTrue(through2.fresh(now).fresh)

        def duplicated(codes):
            used = codes[np.asarray(dim.used, bool)]
            return len(np.unique(used)) < len(used)
        codes = flip_column(dim, "id", 7, duplicated)                         # the key is no longer unique
        dup = len(codes) - len(np.unique(codes))
        d = mc.Dimension(dim, "id")
        self.assertEqual(d.info["duplicates"], dup)
        self.assertFalse(d.info["unique"])
        now2 = mbk.Scope.from_table(f, ["x", "y"], group_by=[("c", dim, "region")])
        fr2 = through2.fresh(now2)
        self.assertEqual(set(fr2.reasons), {"join_revision", "join_key"})
        self.assertTrue(local.fresh(mbk.Scope.from_table(f, ["x", "y"])).fresh)

    def test_fnd007_selection_rows_and_identity_reasons(self):
        db, values, _ = fact_db(self.rng, 256)
        mask = self.rng.random(256) < 0.5
        scope = self.scope(db, ["x", "a"], rows=mask)
        cert = mc.snapshot(scope, mc.affine("a", "x", 3, -7))
        self.assertTrue(cert.source()["exact_selection"])
        self.assertTrue(cert.fresh(self.scope(db, ["x", "a"], rows=mask)).fresh)
        other = mask.copy(); other[int(np.flatnonzero(mask)[0])] = False
        f = cert.fresh(self.scope(db, ["x", "a"], rows=other))
        self.assertEqual(f.reasons, ("selection",))
        self.assertEqual((f.recorded, f.current), (int(mask.sum()), int(mask.sum()) - 1))
        twin, _, _ = fact_db(np.random.default_rng(SEED), 256)                # the same data, another table
        self.assertIn("identity", cert.fresh(mbk.Scope.from_table(twin["t"], ["x", "a"], rows=mask)).reasons)
        moved = self.scope(db, ["a", "x"], rows=mask)                         # the columns in another order
        self.assertIn("columns", cert.fresh(moved).reasons)
        x = np.array(values["x"], np.int64)
        arr = mbk.Scope.from_arrays({"x": x, "a": 3 * x - 7})
        ca = mc.snapshot(arr, mc.affine("a", "x", 3, -7))
        longer = np.append(x, 5)
        f = ca.fresh(mbk.Scope.from_arrays({"x": longer, "a": 3 * longer - 7}))
        self.assertIn("rows", f.reasons)
        self.assertIn("identity", f.reasons)

    def test_fnd007_derivation_is_an_ideal_membership_certificate(self):
        db, values, _ = fact_db(self.rng, 300)
        names = ["a", "x", "y", "p"]
        scope = self.scope(db, names)
        g1 = mc.snapshot(scope, mc.affine("a", "x", 3, -7))                   # a - 3x + 7
        g2 = mc.snapshot(scope, mc.product("p", "x", "y"))                    # p - x y
        claim = mc.polynomial([(1, {"a": 1, "y": 1}), (-3, {"p": 1}), (7, {"y": 1})], names)   # a y - 3 p + 7 y
        d = mc.derive(claim, [g1, g2], [mc.polynomial({(1,): 1}, ["y"]), mc.polynomial({(0,): -3}, ["y"])], scope)
        self.assertEqual(d.kind, "snapshot")
        self.assertEqual(d.info["premises"], 2)
        self.assertTrue(d.authorize_exact(scope))
        for r in range(300):                                                  # the oracle: P = 0 on every row
            a, y, p = values["a"][r], values["y"][r], values["p"][r]
            self.assertEqual(a * y - 3 * p + 7 * y, 0)
        self.assertTrue(mc.snapshot(scope, claim).decide(scope)["authorized"])
        with self.assertRaises(DomainError):
            mc.derive(claim, [g1, g2], [mc.polynomial({(1,): 1}, ["y"]), mc.polynomial({(0,): 3}, ["y"])], scope)
        flip_column(db["t"], "y", 4)
        now = self.scope(db, names)
        self.assertEqual(d.fresh(now).reasons, ("revision",))
        self.assertEqual(d.decide(now)["reason"], "premise")
        back = mc.Certificate.from_bytes(d.to_bytes(), names)
        self.assertEqual(back, d)
        self.assertEqual(len(back.premises), 2)

    def test_fnd007_hypothesis_samplers_match_their_formulas(self):
        db, values, _ = fact_db(self.rng, 500)
        mask = self.rng.random(500) < 0.7
        scope = self.scope(db, ["x", "a"], rows=mask)
        selected = [r for r in range(500) if mask[r]]
        N = len(selected)

        def splitmix(z):
            z = (z + 0x9E3779B97F4A7C15) & M64
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
            return z ^ (z >> 31)
        for method, size, seed in (("uniform", 90, 5), ("stride", 60, 0), ("prefix", 25, 0)):
            if method == "uniform":
                ranks = [(splitmix((seed + i * 0x9E3779B97F4A7C15) & M64) * N) >> 64 for i in range(size)]
            elif method == "stride":
                ranks = [i * N // size for i in range(size)]
            else:
                ranks = list(range(size))
            rows = {selected[r] for r in ranks}
            h = mc.hypothesis(scope, mc.affine("a", "x", 3, -7), size=size, seed=seed, method=method)
            self.assertEqual(h.sample, dict(method=method, size=size, seed=seed, population=N, distinct=len(rows)))
            self.assertEqual(h.evidence["checked"], len(rows))

    def test_fnd007_serialization_round_trip_and_tamper(self):
        db, values, _ = fact_db(self.rng, 200)
        scope = self.scope(db, ["x", "y", "a", "s", "b1", "b2", "b3"])
        certs = [mc.snapshot(scope, mc.affine("a", "x", 3, -7)),
                 mc.snapshot(scope, mc.parity(["b1", "b2", "b3"], 0)),
                 mc.hypothesis(scope, mc.sum_of("s", "x", "y"), size=20, seed=3),
                 mc.law(mc.polynomial([(12345, {"u": 1, "v": 1}), (-12345, {"v": 1, "u": 1})], ["u", "v"]))]
        for c in certs:
            data = c.to_bytes()
            back = mc.Certificate.from_bytes(data, scope.names)
            self.assertEqual(back, c)
            self.assertEqual(back.to_bytes(), data)
            self.assertEqual(back.info["digest"], c.info["digest"])
            with self.assertRaises(ValueError):
                mc.Certificate.from_bytes(data[:-8])
        law_bytes = certs[3].to_bytes()                                       # its coefficients: 12345, -12345
        idx = law_bytes.find((-12345).to_bytes(8, "little", signed=True))
        self.assertGreater(idx, 0)
        bad = law_bytes[:idx] + (-12344).to_bytes(8, "little", signed=True) + law_bytes[idx + 8:]
        with self.assertRaises(ValueError):                                   # the law is decided again on load
            mc.Certificate.from_bytes(bad)
        slot = law_bytes.find((-1).to_bytes(8, "little", signed=True))      # an unused column slot
        with self.assertRaises(ValueError):                                   # only canonical bytes are accepted
            mc.Certificate.from_bytes(law_bytes[:slot] + (-2).to_bytes(8, "little", signed=True) + law_bytes[slot + 8:])

    def test_fnd007_budgets_at_the_limit_and_one_past(self):
        db, values, _ = fact_db(self.rng, 300)
        scope = self.scope(db, ["id", "x", "p", "y", "cat"], group_by=["k"])
        claims = [mc.unique("id"), mc.polynomial({(1, 1, 0): 1, (0, 2, 1): -1}, ["p", "x", "y"]),
                  mc.coverage("cat", [0, 1])]
        for backend in ("cpu", "cuda"):
            need = mc.need(scope, claims, backend=backend)
            for resource in ("bytes", "support", "output", "states"):
                if not need.get(resource):
                    continue
                with self.subTest(backend=backend, resource=resource):
                    ok = mc.verify(scope, claims, backend=backend, budget=Budget(**{resource: need[resource]}))
                    self.assertEqual(len(ok), 3)
                    with self.assertRaises(AlgebraBudgetError) as ctx:
                        mc.verify(scope, claims, backend=backend, budget=Budget(**{resource: need[resource] - 1}))
                    self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit),
                                     (resource, need[resource], need[resource] - 1))
        with self.assertRaises(AlgebraBudgetError):
            mc.snapshot(scope, mc.unique("id"), budget=Budget(bytes=64))
        with self.assertRaises(UnsupportedError):
            mc.coverage("cat", range(65))
        with self.assertRaises(ValueError):
            mc.verify(scope, [mc.affine("nope", "x")])


if __name__ == "__main__":
    unittest.main()
