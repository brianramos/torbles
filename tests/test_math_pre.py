# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PRE-001: affine and monotone-piece preimages as exact sets of stored codes, CPU and CUDA.

Oracles (never the native code): every code of small domains decoded and the original predicate evaluated with
Python Fractions written here; DuckDB evaluating the original predicates over the same engine tables (multiplied
through by positive denominators, so the SQL is exact decimal arithmetic); the engine's own bit-plane predicate on
the transformed code intervals. Every scan also runs on the CUDA path (emulated here, threads scrambled) and is
compared bit for bit. Seeds are fixed.
"""
import ctypes
import os
import random
import subprocess
import sys
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import algq
import torbles
from torbles import math_bank as mbk
from torbles import math_pre as mp
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

OPS = ["<", "<=", "=", "<>", ">=", ">"]
CODECS = [mp.Codec(0, 0, 0, 255), mp.Codec(-128, 0, 0, 255), mp.Codec(-150, 1, 0, 300), mp.Codec(37, 2, 0, 200),
          mp.Codec(-1000, 0, 500, 700), mp.Codec(5, 0, 0, 0), mp.Codec(-77, 3, 0, 160)]


def holds(y, op, c):
    if op == "in":
        return y in c
    return {"<": y < c, "<=": y <= c, "=": y == c, "<>": y != c, ">=": y >= c, ">": y > c}[op]


def f_of(m, x):
    if m.kind == "affine":
        return m.a * x + m.b
    if m.kind == "square":
        return m.a * x * x + m.b
    if m.kind == "residue":
        return x % m.modulus
    for p in m.pieces:
        if (p.lo is None or x >= p.lo) and (p.hi is None or x <= p.hi):
            return p.a * x + p.b if p.kind == "affine" else p.a * x * x + p.b
    raise AssertionError("no piece holds x")


def members(s, codec):
    """the codes of a set from its native spans"""
    out = set()
    for sp in s.spans:
        if len(sp) == 2:
            out.update(range(sp[0], sp[1] + 1))
        else:
            lo, hi, M, rlo, rhi = sp
            out.update(k for k in range(lo, hi + 1) if rlo <= k % M <= rhi)
    return out


def sql_frac(f):
    return f"({f.numerator})", f.denominator


class Pre(unittest.TestCase):
    def check_exact(self, codec, m, op, c, what=""):
        s = mp.preimage(codec, m, op, c) if op != "in" else mp.preimage(codec, m, op, values=c)
        want = {k for k in range(codec.lo, codec.hi + 1) if holds(f_of(m, codec.value(k)), op,
                                                                  c if op != "in" else [Fraction(v) for v in c])}
        got = members(s, codec)
        self.assertEqual(got, want, (what, m, op, c, codec))
        info = s.info
        self.assertEqual(info["empty"], not want, (m, op, c))
        self.assertEqual(info["all"], len(want) == codec.hi - codec.lo + 1 and not info["residue"], (m, op, c))
        self.assertEqual(info["codes"], len(want))
        spans = s.spans
        if spans and len(spans[0]) == 2:                 # sorted, disjoint, never adjacent
            for (a, b), (c2, d) in zip(spans, spans[1:]):
                self.assertLess(b + 1, c2)
            for a, b in spans:
                self.assertLessEqual(a, b)
        for k in (codec.lo, codec.hi, (codec.lo + codec.hi) // 2):
            self.assertEqual(s.contains(k), k in want)
        return s

    # ------------------------------------------------------------------ acceptance 1: every operator, every case
    def test_pre001_every_operator_agrees_on_boundary_negative_slope_fractional_threshold_and_empty_cases(self):
        rng = random.Random(1101)
        cases = 0
        slopes = [Fraction(1), Fraction(-1), Fraction(3), Fraction(-3, 2), Fraction(7, 3), Fraction(-1, 100),
                  Fraction(1000, 7)]
        for codec in CODECS:
            xs = [codec.value(k) for k in (codec.lo, codec.hi)]
            for a in slopes:
                for b in (Fraction(0), Fraction(5, 2), Fraction(-11, 3)):
                    ys = sorted(a * x + b for x in xs)
                    mid = codec.value((codec.lo + codec.hi) // 2) * a + b
                    thresholds = {ys[0], ys[1], mid, ys[0] - 1, ys[1] + 1,   # boundaries, empty and whole cases
                                  mid + Fraction(1, 3), mid - Fraction(1, 7)}
                    thresholds.add(Fraction(rng.randint(-500, 500), rng.choice([1, 2, 3, 10, 100])))
                    for c in thresholds:
                        for op in OPS:
                            self.check_exact(codec, mp.affine(a, b), op, c)
                            cases += 1
        self.assertGreater(cases, 4000)
        # the flip of a negative slope and the rounding at a fractional threshold, by hand
        cd = mp.Codec(0, 0, 0, 20)
        self.assertEqual(mp.preimage(cd, mp.affine(-2, 0), "<", Fraction(-9, 2)).spans, [(3, 20)])   # x > 9/4
        self.assertEqual(mp.preimage(cd, mp.affine(-2, 0), "<=", -6).spans, [(3, 20)])
        self.assertEqual(mp.preimage(cd, mp.affine(2, 1), "=", 6).spans, [])                           # x = 5/2
        self.assertEqual(mp.preimage(cd, mp.affine(2, 1), "<>", 6).spans, [(0, 20)])
        self.assertTrue(mp.preimage(cd, mp.affine(2, 1), "<>", 6).all)
        self.assertTrue(mp.preimage(cd, mp.affine(1, 0), ">", 20).empty)

    def test_pre001_squares_give_symmetric_intervals_and_zero_one_or_two_points(self):
        for codec in CODECS[:5] + [mp.Codec(-128, 0, 0, 255)]:
            for a, b in ((Fraction(1), Fraction(0)), (Fraction(-2), Fraction(3)), (Fraction(1, 3), Fraction(-5, 2))):
                for c in (Fraction(0), Fraction(49), Fraction(50), Fraction(-1), Fraction(9, 4), Fraction(2),
                          b, b + a * 49, b + a * Fraction(1, 100), Fraction(10 ** 6)):
                    for op in OPS:
                        self.check_exact(codec, mp.square(a, b), op, c)
        cd = mp.Codec(-128, 0, 0, 255)                       # x in -128 .. 127
        self.assertEqual(mp.preimage(cd, mp.square(), "<=", 50).spans, [(128 - 7, 128 + 7)])   # |x| <= 7: symmetric
        self.assertEqual(mp.preimage(cd, mp.square(), "=", 49).spans, [(121, 121), (135, 135)])   # two points
        self.assertEqual(mp.preimage(cd, mp.square(), "=", 0).spans, [(128, 128)])               # one point
        self.assertEqual(mp.preimage(cd, mp.square(), "=", 50).spans, [])                        # none
        self.assertEqual(mp.preimage(cd, mp.square(), "=", -4).spans, [])

    def test_pre001_residue_maps_give_residue_classes(self):
        for codec in (mp.Codec(0, 0, 0, 255), mp.Codec(-128, 0, 0, 255), mp.Codec(-150, 1, 0, 300),
                      mp.Codec(37, 2, 0, 200), mp.Codec(-1000, 0, 500, 700)):
            for m in (1, 2, 7, 12, 1000):
                for op in OPS:
                    for c in (0, 1, Fraction(5, 2), m - 1, m, -1, Fraction(m, 2)):
                        self.check_exact(codec, mp.residue(m), op, c)
                for vals in ([1, 3], [0], [Fraction(1, 2), 2, 99], []):
                    self.check_exact(codec, mp.residue(m), "in", vals)
        s = mp.preimage(mp.Codec(-128, 0, 0, 255), mp.residue(7), "in", values=[1, 3])
        self.assertTrue(s.info["residue"])
        self.assertEqual({sp[2] for sp in s.spans}, {7})                 # residue classes of codes, modulus 7
        small = mp.preimage(mp.Codec(0, 0, 0, 5), mp.residue(1000), "<", 3)
        self.assertEqual(small.spans, [(0, 2)])                          # a domain below the modulus: intervals

    def test_pre001_monotone_pieces_union_every_branch(self):
        rng = random.Random(1102)
        for trial in range(60):
            codec = rng.choice(CODECS[:5])
            T = 10 ** codec.scale
            L0, L1 = codec.lo + codec.offset, codec.hi + codec.offset
            n = rng.choice([1, 2, 4, 8, 16])
            cuts = sorted(rng.sample(range(L0, L1), min(n - 1, L1 - L0)))   # piece i ends at logical cuts[i]
            offgrid = trial % 2 == 1                     # bounds between codes: closed pieces touch off the grid
            pcs, start = [], None
            for i in range(len(cuts) + 1):
                end = None if i == len(cuts) else (Fraction(2 * cuts[i] + 1, 2 * T) if offgrid else Fraction(cuts[i], T))
                a = Fraction(rng.choice([1, -1, 2, -3, 5]), rng.choice([1, 2, 3]))
                b = Fraction(rng.randint(-20, 20), rng.choice([1, 2]))
                pcs.append(mp.Piece(start, end, rng.choice(["affine", "affine", "square"]), a, b))
                start = None if end is None else (end if offgrid else end + Fraction(1, T))
            m = mp.pieces(pcs)
            for op in OPS:
                c = Fraction(rng.randint(-40, 40), rng.choice([1, 3]))
                s = self.check_exact(codec, m, op, c, trial)
                self.assertLessEqual(s.info["decided"], s.info["branches"])
                self.assertEqual(s.info["branches"], len(pcs))
            self.check_exact(codec, m, "in", [Fraction(rng.randint(-20, 20)) for _ in range(5)], trial)
        # a piece decided by UNC-001 (its image lies entirely on one side) is not solved
        cd = mp.Codec(-100, 1, 0, 255)
        pw = mp.pieces([(None, Fraction(-1, 10), "affine", -1, 0), (0, 10, "square", 1, 0),
                        (Fraction(101, 10), None, "affine", 3, -200)])
        s = self.check_exact(cd, pw, ">", 4)
        self.assertEqual((s.info["branches"], s.info["decided"]), (3, 1))

    # ------------------------------------------------------------------ acceptance 2: divisors and branches
    def test_pre001_requires_nonzero_divisors_and_all_branches(self):
        cd = mp.Codec(-100, 1, 0, 255)                       # x in -10 .. 15.5
        for bad in (mp.affine(0, 3), mp.square(0, 1), mp.residue(0),
                    mp.pieces([(None, 0, "affine", 1, 0), (Fraction(1, 10), None, "affine", 0, 2)])):
            for op in ("<", "=", "in"):
                with self.assertRaises(DomainError, msg=(bad, op)):
                    mp.preimage(cd, bad, op, 1) if op != "in" else mp.preimage(cd, bad, op, values=[1])
        gaps = [[(None, Fraction(-1, 10), "affine", -1, 0), (Fraction(1, 10), None, "affine", 1, 0)],   # x = 0 missing
                [(Fraction(-10), Fraction(15), "affine", 1, 0)],                                       # x > 15 missing
                [(Fraction(-9), None, "square", 1, 0)]]                                                # x < -9 missing
        for g in gaps:
            with self.assertRaises(DomainError, msg=g):
                mp.preimage(cd, mp.pieces(g), ">", 4)
        with self.assertRaises(ValueError):                  # overlapping pieces: not a function
            mp.preimage(cd, mp.pieces([(None, 1, "affine", -1, 0), (1, None, "affine", 1, 0)]), ">", 4)
        with self.assertRaises(ValueError):
            mp.preimage(cd, mp.pieces([(3, 1, "affine", 1, 0)]), ">", 4)
        # every branch given: the same map is exact
        full = [(None, Fraction(-1, 10), "affine", -1, 0), (0, None, "affine", 1, 0)]
        self.check_exact(cd, mp.pieces(full), ">", 4)
        self.check_exact(cd, mp.pieces(full), "=", Fraction(3, 10))
        with self.assertRaises(UnsupportedError):
            mp.preimage(mp.Codec(0, 18, 0, 10), mp.residue(10), "<", 1)   # m 10^18 past int64

    # ------------------------------------------------------------------ engine tables, NULLs, DuckDB, planes, CUDA
    def test_pre001_scans_match_duckdb_engine_planes_and_cuda_with_null_guards(self):
        rng = np.random.default_rng(1103)
        n = 2500
        x = [None if r < 0.12 else f"{v / 100:.2f}" for r, v in zip(rng.random(n), rng.integers(-5000, 5001, n))]
        k = [None if r < 0.05 else int(v) for r, v in zip(rng.random(n), rng.integers(-300, 301, n))]
        frame = pd.DataFrame({"x": pd.Series(x, dtype=object), "k": pd.Series(k, dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (x DECIMAL(10,2), k BIGINT)", {"t": frame})
        t = db["t"]
        cx, ck = mp.Codec.from_table(t, "x"), mp.Codec.from_table(t, "k")
        self.assertEqual((cx.offset, cx.scale, cx.nullable), (t.codecs["x"].affine[0], 2, True))
        X = "CAST(x AS DECIMAL(38,2))"
        cases = []
        for a, b, c in ((Fraction(-3, 7), Fraction(2), Fraction(5, 3)), (Fraction(5), Fraction(-1, 4), Fraction(0)),
                        (Fraction(1, 100), Fraction(0), Fraction(-7, 3))):
            for op in OPS:
                p, q = a.numerator, a.denominator
                r, s_ = b.numerator, b.denominator
                u, v = c.numerator, c.denominator
                sql = f"{X} * {p * s_ * v} + {r * q * v} {op} {u * q * s_}"
                cases.append(("x", cx, mp.affine(a, b), op, c, sql))
        for op in OPS:
            cases.append(("x", cx, mp.square(2, -1), op, Fraction(9, 2), f"{X} * {X} * 4 - 2 {op} 9"))
            cases.append(("k", ck, mp.residue(7), op, 3, f"((k % 7) + 7) % 7 {op} 3"))
        cases.append(("k", ck, mp.residue(12), "in", [0, 5, 11], "((k % 12) + 12) % 12 IN (0, 5, 11)"))
        pw = mp.pieces([(None, -1, "affine", -2, 0), (0, None, "square", 1, -50)])   # the whole code domain
        cases.append(("k", ck, pw, ">", 30, "CASE WHEN k <= -1 THEN -2 * k ELSE k * k - 50 END > 30"))
        for sel in (None, rng.random(n) < 0.5):
            for name in ("x", "k"):
                scope = mbk.Scope.from_table(t, [name], rows=sel)
                where = "TRUE"
                if sel is not None:
                    db.con.register("pre_sel", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
                    where = "rowid IN (SELECT rid FROM pre_sel)"
                for col, codec, m, op, c, sql in cases:
                    if col != name:
                        continue
                    s = mp.preimage(codec, m, op, c) if op != "in" else mp.preimage(codec, m, op, values=c)
                    bits, st = s.scan(scope, name, direct=True)
                    gbits, gst = s.scan(scope, name, direct=True, backend="cuda")
                    self.assertIn(gst["backend"], ("cuda", "cuda-emulated"))
                    self.assertTrue(np.array_equal(bits, gbits), sql)
                    strip = lambda d: {a: b for a, b in d.items() if a not in ("backend", "device")}
                    self.assertEqual(strip(st), strip(gst))
                    self.assertEqual(st["mismatches"], 0, sql)
                    self.assertEqual((st["overflow"], st["outside"]), (0, 0))
                    got = np.flatnonzero(np.unpackbits(bits.view(np.uint8), bitorder="little")[:n]).tolist()
                    want = [r for r, in db.con.execute(f"SELECT rowid FROM t WHERE {where} AND ({sql}) ORDER BY rowid").fetchall()]
                    self.assertEqual(got, want, sql)
                    self.assertEqual(st["in_set"], len(want))
                    self.assertEqual(st["direct_true"], len(want))
                    nulls = int(sum(1 for r in range(n) if (sel is None or sel[r]) and frame[name][r] is None))
                    self.assertEqual(st["nulls"], nulls)
                    plain, pst = s.scan(scope, name)                 # without the direct check: the same bits
                    self.assertTrue(np.array_equal(plain, bits))
                    if not s.info["residue"]:                        # the engine's bit planes on the code intervals
                        prog = s.plane_program(t, name)
                        arr = (algq.PInstr * len(prog))(*prog)
                        bm = np.zeros((t.n + 63) // 64 + 1, dtype=np.uint64)
                        used = t.used_bm if sel is None else np.packbits(np.pad(sel, (0, (64 - n % 64) % 64)),
                                                                         bitorder="little").view(np.uint64)
                        cnt = algq.lib.at_filter_bitmap_masked(t.at, arr, len(prog), bm.ctypes.data, used.ctypes.data)
                        self.assertEqual(int(cnt), len(want))
                        self.assertTrue(np.array_equal(bm[:len(bits)], bits), sql)

    def test_pre001_scope_from_arrays_and_rows_outside_the_domain(self):
        rng = np.random.default_rng(1104)
        n = 700
        vals = rng.integers(-60, 61, n)
        valid = rng.random(n) > 0.1
        scope = mbk.Scope.from_arrays({"x": vals}, valid={"x": valid})
        codec = mp.Codec(-50, 0, 0, 100)                     # x in -50 .. 50: some values lie outside
        s = mp.preimage(codec, mp.affine(Fraction(-1, 3), 2), ">=", 0)    # x <= 6
        bits, st = s.scan(scope, "x", direct=True)
        g, gst = s.scan(scope, "x", direct=True, backend="cuda")
        self.assertTrue(np.array_equal(bits, g))
        outside = int(np.sum(valid & ((vals < -50) | (vals > 50))))
        self.assertEqual(st["outside"], outside)
        self.assertEqual(st["mismatches"], 0)
        got = np.flatnonzero(np.unpackbits(bits.view(np.uint8), bitorder="little")[:n])
        want = np.flatnonzero(valid & (vals >= -50) & (vals <= 6))
        self.assertTrue(np.array_equal(got, want))
        with self.assertRaises(ValueError):                  # another value unit (scale): refused
            s.scan(mbk.Scope.from_arrays({"x": vals}, scales={"x": 1}), "x")

    def test_pre001_certificates_are_fnd007_laws_of_the_transformation(self):
        for codec, m in ((mp.Codec(-128, 0, 0, 255), mp.affine(Fraction(-3, 2), Fraction(7, 5))),
                         (mp.Codec(37, 2, 0, 200), mp.square(Fraction(2, 3), -1)),
                         (mp.Codec(-1000, 1, 0, 999), mp.residue(7)),
                         (mp.Codec(-100, 1, 0, 255), mp.pieces([(None, 0, "affine", -1, 0),
                                                                (Fraction(1, 10), None, "square", 1, 0)]))):
            s = mp.preimage(codec, m, "<", 3)
            certs = s.certificates
            self.assertEqual(len(certs), s.info["branches"])
            self.assertEqual(s.info["certified"], len(certs))
            for cert in certs:
                self.assertEqual(cert.kind, "law")
                self.assertEqual(cert.info["law"], "ring_identity")
                self.assertTrue(cert.fresh().fresh)
                self.assertEqual(cert.authorize_exact(field=0).kind, "law")
                c = cert._native_claim()
                nt = int(c.nterms)
                exps = (ctypes.c_uint8 * nt).from_address(c.exponents)
                co = (ctypes.c_int64 * nt).from_address(c.coefficients)
                total = {}
                for i in range(nt):
                    total[exps[i]] = total.get(exps[i], 0) + co[i]
                self.assertTrue(all(v == 0 for v in total.values()), total)   # canonically zero, independently
                self.assertGreater(nt, 2)
        # coefficients past int64: no certificate, the set is still exact
        big = mp.Codec(-(1 << 40), 9, 0, 300)
        s = self.check_exact(big, mp.square(Fraction(1, 3)), "<", 5)
        self.assertEqual(s.info["certified"], 0)
        self.assertEqual(s.certificates, [None])

    def test_pre001_budgets_at_the_limit_and_one_past(self):
        cd = mp.Codec(0, 0, 0, 255)
        pcs = [mp.Piece(None if i == 0 else 16 * i, None if i == 15 else 16 * i + 15, "affine", 1 + i % 3, -i)
               for i in range(16)]
        mp.preimage(cd, mp.pieces(pcs), "<", 40, budget=Budget(support=16))
        with self.assertRaises(AlgebraBudgetError) as cm:
            mp.preimage(cd, mp.pieces(pcs), "<", 40, budget=Budget(support=15))
        self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), ("support", 16, 15))
        vals = list(range(0, 200, 2))
        s = mp.preimage(cd, mp.affine(1, 0), "in", values=vals)
        nspans = len(s.spans)
        self.assertEqual(nspans, 100)
        mp.preimage(cd, mp.affine(1, 0), "in", values=vals, budget=Budget(output=nspans))
        with self.assertRaises(AlgebraBudgetError) as cm:
            mp.preimage(cd, mp.affine(1, 0), "in", values=vals, budget=Budget(output=nspans - 1))
        self.assertEqual(cm.exception.resource, "output")
        with self.assertRaises(UnsupportedError):
            mp.pieces([(i, i, "affine", 1, 0) for i in range(mp.MAX_PIECES + 1)])
        frame = pd.DataFrame({"x": np.arange(300)})
        t = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})["t"]
        scope = mbk.Scope.from_table(t, ["x"])
        s = mp.preimage(mp.Codec.from_table(t, "x"), mp.affine(2, 0), "<", 100)
        need = (300 + 63) // 64 * 8
        s.scan(scope, "x", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError):
            s.scan(scope, "x", budget=Budget(bytes=need - 1))
        # the CUDA device copies: the refusal reports the exact need; admitted at it, refused one byte below
        with self.assertRaises(AlgebraBudgetError) as cm:
            s.scan(scope, "x", backend="cuda", direct=True, budget=Budget(bytes=64))
        dev = cm.exception.need
        self.assertEqual(cm.exception.resource, "bytes")
        s.scan(scope, "x", backend="cuda", direct=True, budget=Budget(bytes=dev))
        with self.assertRaises(AlgebraBudgetError):
            s.scan(scope, "x", backend="cuda", direct=True, budget=Budget(bytes=dev - 1))

    def test_pre001_malformed_requests_are_refused(self):
        cd = mp.Codec(0, 0, 0, 10)
        with self.assertRaises(ValueError):
            mp.preimage(cd, mp.affine(1, 0), "~", 1)
        with self.assertRaises(TypeError):
            mp.affine(0.5)                                   # floats are not exact input
        with self.assertRaises(ValueError):
            mp.preimage(mp.Codec(0, 0, 10, 0), mp.affine(1, 0), "<", 1)   # lo > hi
        with self.assertRaises(ValueError):
            mp.preimage(mp.Codec(0, 19, 0, 10), mp.affine(1, 0), "<", 1)  # scale past 18
        with self.assertRaises(ValueError):
            mp.preimage(cd, mp.residue(-3), "<", 1)
        with self.assertRaises(UnsupportedError):
            mp.preimage(cd, mp.affine(1, 0), "in", values=list(range(mp.MAX_VALUES + 1)))

    def test_pre001_import_torbles_does_not_load_the_module(self):
        code = "import sys, torbles; print('torbles.math_pre' in sys.modules, 'torbles.math_unc' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ),
                             check=True).stdout.strip()
        self.assertEqual(out, "False False")


if __name__ == "__main__":
    unittest.main()
