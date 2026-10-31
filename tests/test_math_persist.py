# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-011: versioned binary persistence of typed mathematical states.

Oracles (independent of the implementation): the checksum recomputed from its documented formula with Python
integers (so tampered records can be re-sealed and reach the later checks), byte-level truncation and corruption of
saved records, DuckDB SQL for the row counts and sums the reloaded states hold, the original Python values of the
carriers, the native equality of the rebuilt state with the original, and the record bytes searched for live
addresses. The device checksum (emulated here, threads in a scrambled order) equals the CPU one. Seeds are fixed.
"""
import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

import algq
import torbles
from torbles import math_bank as mbk
from torbles import math_carriers as mcar
from torbles import math_certificates as mc
from torbles import math_dist as md
from torbles import math_lin as ml
from torbles import math_persist as mp
from torbles.math_budgets import Budget
from torbles.math_errors import StaleStateError
from torbles.mathematics import AlgebraBudgetError

SEED = 1011
M64 = (1 << 64) - 1


def obj(values):
    return pd.Series(values, dtype=object)


def decode(table, col):
    out = np.empty(max(table.n, 1), dtype=np.uint32)
    algq.lib.at_decode_column(table.at, table.idx(col), out.ctypes.data)
    return out[:table.n]


def flip_column(table, col, row):
    """flips one stored bit so that exactly column `col` changes at `row`"""
    before = {c: decode(table, c) for c in table.cols}
    for p in range(table.nplanes()):
        table.flip(p, row)
        after = {c: decode(table, c) for c in table.cols}
        if [c for c in table.cols if not np.array_equal(before[c], after[c])] == [col]:
            return
        table.flip(p, row)
    raise AssertionError(f"no single plane changes only {col}")


def weight(i, lane):
    """tr_persist_weight, from its documented definition"""
    z = (i + (0xD1B54A32D192ED03 if lane else 0x9E3779B97F4A7C15)) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z ^ (z >> 31)) | 1


def words_of(data):
    return [int.from_bytes(data[i:i + 8], "little") for i in range(0, len(data), 8)]


def python_checksum(data, skip=True):
    """the record checksum: every little-endian word but words 12 and 13, weighted by position, mod 2^64"""
    out = [0, 0]
    for i, w in enumerate(words_of(data)):
        if skip and i in (12, 13):
            continue
        for lane in (0, 1):
            out[lane] = (out[lane] + w * weight(i, lane)) & M64
    return tuple(out)


def reseal(data):
    """the record with its checksum recomputed (by the oracle), so a deliberate change reaches the later checks"""
    a, b = python_checksum(data)
    return data[:96] + a.to_bytes(8, "little") + b.to_bytes(8, "little") + data[112:]


def dataset(rng, n):
    k = [int(v) for v in rng.integers(0, 5, n)]
    x = [int(v) for v in rng.integers(-500, 501, n)]
    y = [int(v) for v in rng.integers(-90, 91, n)]
    d = [f"{int(v) / 100:.2f}" for v in rng.integers(-9999, 10000, n)]
    for col in (x, d):
        for i in np.flatnonzero(rng.random(n) < 0.1):
            col[int(i)] = None
    frame = pd.DataFrame({"k": obj(k), "x": obj(x), "y": obj(y), "d": obj(d)})
    return torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT, y BIGINT, d DECIMAL(10,2))", {"t": frame})


class Fnd011Persistence(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self.db = dataset(self.rng, 800)
        self.t = self.db["t"]
        self.scope = mbk.Scope.from_table(self.t, ["x", "y", "d"], group_by=["k"])

    def states(self):
        s = self.scope
        return dict(
            bank_sql=mbk.build(s, mbk.monomials(["x", "y", "d"], 2), "sql"),
            bank_joint=mbk.build(s, ["1", "x", "y^2", "x*d"], "joint", backend="cuda"),
            hist_sparse=md.histogram(s, ["x", "y"], weight="d"),
            hist_dense=md.histogram(s, ["y"], repr="dense"),
            moments_listwise=ml.moments(s, ["x", "d"], "y"),
            moments_pairwise=ml.moments(s, ["x", "d"], "y", policy="pairwise"),
            certificate=mc.snapshot(s, mc.not_null("y")),
            hypothesis=mc.hypothesis(s, mc.not_null("y"), size=40, seed=2),
            law=mc.law(mc.polynomial([(1, {"u": 2}), (-1, {"u": 2})], ["u"])))

    def test_fnd011_round_trip_every_state_and_carrier(self):
        s = self.scope
        for name, state in self.states().items():
            with self.subTest(state=name):
                data = mp.dumps(state, metadata={"case": name})
                rec = mp.loads(data)
                h = rec.header
                self.assertEqual((h["format"], h["header_bytes"], h["type_version"], h["endianness"]), (1, 128, 1, "little"))
                self.assertTrue(h["immutable"])
                self.assertEqual(h["total_bytes"], len(data))
                self.assertEqual(rec.metadata["user"], {"case": name})
                self.assertEqual(mp.validate(data)["reason"], "ok")
                self.assertEqual(mp.validate(data, backend="cuda")["reason"], "ok")
                if name.startswith("bank"):
                    back = rec.bank(s)
                    self.assertTrue(back.equal(state))
                    self.assertTrue(back.info["scoped"])
                    self.assertEqual(back.terms, state.terms)
                    self.assertTrue(rec.bank().equal(state))                    # unscoped: the same states
                    self.assertIsNone(rec.bank().scope)
                elif name.startswith("hist"):
                    back = rec.histogram(s)
                    self.assertEqual(md._call("tr_dist_hist_equal", back._open(), state._open()), 1)
                    self.assertEqual(back.cells(2), state.cells(2))
                elif name.startswith("moments"):
                    back = rec.moments(s)
                    for a, b in zip(back.arrays(), state.arrays()):
                        self.assertTrue(np.array_equal(a, b))
                    self.assertEqual(back.matrix(1), state.matrix(1))
                else:
                    back = rec.certificate()
                    self.assertEqual(back, state)
                    self.assertEqual(back.info["kind"], state.info["kind"])
                self.assertEqual(mp.dumps(back, metadata={"case": name}), data)    # canonical: the same bytes
        # DuckDB: the reloaded bank's rows and sums per group
        bank = mbk.build(s, ["1", "x", "y"], "sql")
        rec = mp.loads(mp.dumps(bank))
        st = rec.bank_states()
        by = {s.key_values(g)[0]: g for g in range(s.groups)}
        for k, rows, cx, sx, sy in self.db.con.execute(
                'SELECT k, count(*), count(x), sum(x), sum(y) FROM t GROUP BY k').fetchall():
            g = by[k]
            self.assertEqual(int(st["rows"][g]), rows)
            self.assertEqual(int(st["count"][g][1]), cx)
            self.assertEqual(st["sum"][g][1:], [int(sx or 0), int(sy)])
        # carriers, compared with the Python values they were built from
        for value in (0, -1, 2 ** 64, -(2 ** 4000) + 12345, 10 ** 60 + 7):
            back = mp.loads(mp.dumps(mcar.BigInt(value))).value()
            self.assertEqual(int(back), value)
        for value in (Fraction(0), Fraction(-7, 3), Fraction(2 ** 200 + 1, 3 ** 90), Fraction(5)):
            back = mp.loads(mp.dumps(mcar.BigRational(value))).value()
            self.assertEqual(back.to_fraction(), value)
        F = mcar.GF(2 ** 61 - 1)
        for v in (0, 1, 2 ** 61 - 2, 123456789):
            e = mp.loads(mp.dumps(F(v))).value()
            self.assertEqual((e.field.p, e.v), (2 ** 61 - 1, v))
        rec = mp.loads(mp.dumps(mcar.GF(2)(1)))
        self.assertEqual(rec.header["carrier"], "gfp")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.tms"
            self.assertEqual(mp.save(bank, path), path.stat().st_size)
            self.assertTrue(mp.load(path, expected="bank").bank(s).equal(bank))

    def test_fnd011_rejects_truncated_corrupted_unknown_version_and_wrong_type(self):
        s = self.scope
        records = {name: mp.dumps(state) for name, state in self.states().items()}
        records["bigint"] = mp.dumps(mcar.BigInt(-(2 ** 300)))
        for name, data in records.items():
            with self.subTest(record=name):
                self.assertEqual(python_checksum(data), mp.loads(data).header["checksum"])
                for cut in sorted({0, 1, 64, 127, 128, len(data) // 2, len(data) - 8, len(data) - 1}):
                    for backend in ("cpu", "cuda"):
                        with self.assertRaises(mp.TruncatedStateError):
                            mp.validate(data[:cut], backend=backend)
                    with self.assertRaises(mp.TruncatedStateError) as ctx:
                        mp.loads(data[:cut])
                    self.assertEqual(ctx.exception.status["bytes"], cut)
                h = mp.loads(data).header
                regions = [(112 - 16, 112)]                                       # the checksum itself
                at = 128
                for part in ("source_bytes", "basis_bytes", "metadata_bytes", "payload_bytes"):
                    if h[part]:
                        regions.append((at, at + h[part]))
                    at += (h[part] + 7) // 8 * 8
                for lo, hi in regions:
                    for pos in (lo, (lo + hi) // 2, hi - 1):
                        bad = bytearray(data)
                        bad[pos] ^= 0x10
                        for backend in ("cpu", "cuda"):
                            with self.assertRaises(mp.CorruptStateError):
                                mp.validate(bytes(bad), backend=backend)
                        with self.assertRaises(mp.CorruptStateError) as ctx:
                            mp.loads(bytes(bad))
                        self.assertNotEqual(ctx.exception.status["stored"], ctx.exception.status["computed"])
                version = bytearray(data); version[8] = 2
                with self.assertRaises(mp.UnknownVersionError) as ctx:
                    mp.loads(bytes(version))
                self.assertEqual(ctx.exception.status["format"], 2)
                tv = bytearray(data); tv[28] = 9                                  # a future layout, sealed
                with self.assertRaises(mp.UnknownVersionError) as ctx:
                    mp.loads(reseal(bytes(tv)))
                self.assertEqual(ctx.exception.status["type_version"], 9)
                with self.assertRaises(mp.NotAStateError):
                    mp.loads(b"NOTMATH!" + data[8:])
                with self.assertRaises(mp.ByteOrderError):
                    mp.loads(data[:16] + data[16:24][::-1] + data[24:])
                with self.assertRaises(mp.MalformedStateError):
                    mp.loads(data + bytes(8))
                wrong = "histogram" if name != "hist_sparse" and name != "hist_dense" else "bank"
                with self.assertRaises(mp.WrongTypeError) as ctx:
                    mp.loads(data, expected=wrong)
        # sealed but inconsistent contents: refused after the checksum
        data = records["bank_sql"]
        bad = bytearray(data)
        basis = 128 + (mp.loads(data).header["source_bytes"] + 7) // 8 * 8
        bad[basis + 24] ^= 1                                                     # the group count
        with self.assertRaises(mp.MalformedStateError):
            mp.loads(reseal(bytes(bad)))
        with self.assertRaises(mp.WrongTypeError):
            mp.loads(data).value()
        with self.assertRaises(ValueError):
            mp.loads(data).histogram(s)

    def test_fnd011_reloaded_states_cannot_be_presented_as_refreshed_views(self):
        s = self.scope
        bank = mbk.build(s, ["1", "x", "y"], "sql")
        hist = md.histogram(s, ["y"])
        mom = ml.moments(s, ["x"], "y")
        cert = mc.snapshot(s, mc.not_null("y"))
        saved = {"bank": mp.dumps(bank), "hist": mp.dumps(hist), "mom": mp.dumps(mom), "cert": mp.dumps(cert)}
        revision = s.revision
        recs = {k: mp.loads(v) for k, v in saved.items()}
        for rec in recs.values():
            self.assertTrue(rec.fresh(s).fresh)
            self.assertEqual(rec.source["revision"], revision)
        self.assertTrue(recs["bank"].bank(s).equal(bank))
        flip_column(self.t, "y", 6)                                              # the source changes
        now = mbk.Scope.from_table(self.t, ["x", "y", "d"], group_by=["k"])
        for name, data in saved.items():
            rec = mp.loads(data)                                                 # reloaded after the change
            f = rec.fresh(now)
            self.assertFalse(f.fresh)
            self.assertEqual(f.reasons, ("revision",))
            self.assertEqual(rec.source["revision"], revision)                  # never re-stamped
            self.assertEqual((f.recorded, f.current), (revision, now.revision))
        with self.assertRaises(mp.StaleSourceError) as ctx:
            recs["bank"].bank(now)
        self.assertEqual(ctx.exception.reasons, ("revision",))
        with self.assertRaises(StaleStateError):
            recs["hist"].histogram(now)
        with self.assertRaises(StaleStateError):
            recs["mom"].moments(now)
        reloaded = recs["cert"].certificate()
        self.assertEqual(reloaded.fresh(now).reasons, ("revision",))
        with self.assertRaises(StaleStateError):
            reloaded.authorize_exact(now)
        unscoped = recs["bank"].bank()                                           # a state, but of no source
        self.assertIsNone(unscoped.scope)
        with self.assertRaises(ValueError):
            mbk.Cache().put(unscoped)
        rebuilt = mbk.build(now, ["1", "x", "y"], "sql")                         # the refreshed view is a rebuild
        self.assertFalse(rebuilt.equal(bank))
        # other selections, groupings and sources are not the recorded source either
        mask = np.ones(self.t.n, bool); mask[3] = False
        cases = {("selection",): mbk.Scope.from_table(self.t, ["x", "y", "d"], group_by=["k"], rows=mask),
                 ("grouping",): mbk.Scope.from_table(self.t, ["x", "y", "d"])}
        for reasons, scope in cases.items():
            f = mp.loads(saved["bank"]).fresh(scope)
            self.assertTrue(set(reasons) <= set(f.reasons), (reasons, f.reasons))
        twin = dataset(np.random.default_rng(SEED), 800)["t"]                   # the same data, another table
        f = mp.loads(saved["bank"]).fresh(mbk.Scope.from_table(twin, ["x", "y", "d"], group_by=["k"]))
        self.assertIn("identity", f.reasons)
        self.t.flip(0, 0); self.t.repair()
        self.assertIn("revision", mp.loads(saved["bank"]).fresh(
            mbk.Scope.from_table(self.t, ["x", "y", "d"], group_by=["k"])).reasons)

    def test_fnd011_certificate_records_keep_the_selection_once(self):
        mask = np.ones(self.t.n, bool); mask[::3] = False
        scope = mbk.Scope.from_table(self.t, ["x", "y", "d"], group_by=["k"], rows=mask)
        cert = mc.snapshot(scope, mc.not_null("y"))
        rec = mp.loads(mp.dumps(cert))
        self.assertFalse(rec.source["exact_selection"])                          # the record: digests only
        self.assertTrue(rec.certificate().source()["exact_selection"])           # the certificate: the exact bits
        self.assertEqual(rec.source["selection_digest"], cert.source()["selection_digest"])
        self.assertTrue(rec.fresh(scope).fresh)
        other = mask.copy(); other[1] = not other[1]
        self.assertIn("selection", rec.fresh(mbk.Scope.from_table(self.t, ["x", "y", "d"], group_by=["k"],
                                                                  rows=other)).reasons)

    def test_fnd011_no_live_address_is_serialized(self):
        s = self.scope
        bank = mbk.build(s, ["1", "x"], "sql")
        cert = mc.snapshot(s, mc.not_null("y"))
        for data in (mp.dumps(bank), mp.dumps(cert), mp.dumps(md.histogram(s, ["x"]))):
            for address in (self.t.at, s._handle, bank._handle, cert._handle):
                self.assertNotIn(int(address).to_bytes(8, "little"), data)
            self.assertEqual(mp.loads(data).header["total_bytes"], len(data))

    def test_fnd011_checksum_cpu_equals_cuda_and_its_formula(self):
        for n in (0, 1, 7, 13, 4096, 20001):
            words = self.rng.integers(0, 2 ** 63, size=n, dtype=np.int64).astype(np.uint64) * np.uint64(2) + \
                self.rng.integers(0, 2, size=n, dtype=np.int64).astype(np.uint64)
            data = words.astype("<u8").tobytes()
            cpu = mp.checksum(words)
            self.assertEqual(cpu, mp.checksum(words, backend="cuda"))
            self.assertEqual(cpu, python_checksum(data, skip=False))
        big = mbk.build(mbk.Scope.from_arrays({"x": np.arange(30000, dtype=np.int64)},
                                              groups=np.arange(30000, dtype=np.uint32) % 3000, ngroups=3000),
                        mbk.monomials(["x"], 2))
        data = mp.dumps(big)
        self.assertGreater(len(data), 100000)
        self.assertEqual(mp.validate(data, backend="cuda")["computed"], python_checksum(data))

    def test_fnd011_budgets_at_the_limit_and_one_past(self):
        bank = mbk.build(self.scope, ["1", "x", "y"], "sql")
        data = mp.dumps(bank)
        for op, call in (("save", lambda b: mp.dumps(bank, budget=b)), ("load", lambda b: mp.loads(data, budget=b))):
            need = mp.need(op, len(data))["bytes"]
            self.assertGreater(need, len(data))
            call(Budget(bytes=need))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                call(Budget(bytes=need - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("bytes", need, need - 1))
        need = mp.need("checksum", len(data))["bytes"]
        self.assertEqual(need, len(data) + 16)
        self.assertEqual(mp.validate(data, backend="cuda", budget=Budget(bytes=need))["reason"], "ok")
        with self.assertRaises(AlgebraBudgetError):
            mp.validate(data, backend="cuda", budget=Budget(bytes=need - 1))
        rec = mp.loads(data)
        with self.assertRaises(AlgebraBudgetError):                              # the rebuilt object's own budget
            rec.bank(self.scope, budget=Budget(bytes=64))


if __name__ == "__main__":
    unittest.main()
