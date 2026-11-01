# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PRE-002: finite roots, residues and periodic preimages as exact sets of stored codes, CPU and CUDA.

Oracles (never the native code): every code of small domains decoded and the original predicate evaluated with Python
integers and Fractions written here (polynomials, residues, bit operations); the residue classes of the congruences
recomputed by brute force over one period; DuckDB evaluating the original predicates over the same engine tables
(integer arithmetic, Python's floor modulo spelled out); the engine's own bit-plane predicate on the transformed
intervals and cubes. Every scan and verification also runs on the CUDA path (emulated here, threads scrambled) and is
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
from torbles import math_preimg as mq
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

OPS = ["<", "<=", "=", "<>", ">=", ">"]
SEED = 2202


def holds(y, op, c):
    if op == "in":
        return y in c
    return {"<": y < c, "<=": y <= c, "=": y == c, "<>": y != c, ">=": y >= c, ">": y > c}[op]


def f_of(m, codec, code):
    """the derived value of a code, in Python numbers"""
    L = code + codec.offset
    if m.kind == "poly":
        x = Fraction(L, 10 ** codec.scale)
        return sum(c * x ** i for i, c in enumerate(m.coefficients))
    if m.kind == "polymod":
        return sum(int(c) * L ** i for i, c in enumerate(m.coefficients)) % m.modulus
    if m.kind == "affmod":
        return (m.a * L + m.b) % m.modulus
    base = codec.offset if m.base is None else m.base
    u = (L - base) % (1 << m.width)
    return {"xor": u ^ m.b, "and": u & m.b, "or": u | m.b}[m.kind]


def values_of(m, codec):
    """the derived value of every code of the domain (computed once per map)"""
    return [f_of(m, codec, k) for k in range(codec.lo, codec.hi + 1)]


def truth(m, codec, op, c, ys=None):
    vals = [Fraction(v) for v in c] if op == "in" else c
    ys = ys if ys is not None else values_of(m, codec)
    if op == "in":
        vs = set(vals)
        return {codec.lo + i for i, y in enumerate(ys) if y in vs}
    return {codec.lo + i for i, y in enumerate(ys) if holds(y, op, vals)}


def members(s, codec):
    """the codes of a set from its native representation (branches decoded here)"""
    info = s.info
    if info["mode"] != "branches":
        return {k for k in range(codec.lo, codec.hi + 1) if s.contains(k)}
    out = set()
    branches = s.branches
    if info["repr"] == "intervals":
        for lo, hi in branches:
            out.update(range(lo, hi + 1))
    elif info["repr"] == "residues":
        res = set()
        for lo, hi, M, rlo, rhi in branches:
            res.update(range(rlo, rhi + 1))
        if branches:
            M = branches[0][2]
            out = {k for k in range(codec.lo, codec.hi + 1) if k % M in res}
    else:
        for k in range(codec.lo, codec.hi + 1):
            if any((k & mask) == value for lo, hi, mask, value in branches):
                out.add(k)
    return out


def make(c, scale=0):
    return mp.Codec(*c) if isinstance(c, tuple) else c


class Pre002(unittest.TestCase):
    def check_exact(self, codec, m, op, c, what="", fallback="auto", budget=None, ys=None):
        s = (mq.preimage(codec, m, op, c, fallback=fallback, budget=budget) if op != "in" else
             mq.preimage(codec, m, op, values=c, fallback=fallback, budget=budget))
        want = truth(m, codec, op, c, ys)
        got = members(s, codec)
        self.assertEqual(got, want, (what, m, op, c, codec, s.info))
        info = s.info
        if info["codes"] is not None:
            self.assertEqual(info["codes"], len(want), (m, op, c))
            self.assertEqual(info["empty"], not want)
        for k in (codec.lo, codec.hi, (codec.lo + codec.hi) // 2):
            self.assertEqual(s.contains(k), k in want)
        if info["mode"] == "branches" and info["repr"] == "intervals":
            spans = s.branches
            for (a, b), (c2, d) in zip(spans, spans[1:]):
                self.assertLess(b + 1, c2)                               # sorted, disjoint, never adjacent
        return s

    # ------------------------------------------------------------------ acceptance 1: exhaustive small domains
    def test_pre002_polynomial_maps_match_exhaustive_enumeration_of_small_domains(self):
        rng = random.Random(SEED)
        codecs = [mp.Codec(0, 0, 0, 255), mp.Codec(-128, 0, 0, 255), mp.Codec(-150, 1, 0, 300),
                  mp.Codec(37, 2, 0, 200), mp.Codec(-1000, 0, 500, 700), mp.Codec(5, 0, 0, 0)]
        cases = 0
        for codec in codecs:
            for trial in range(16):
                d = rng.choice([1, 2, 3, 4, 5, 6])
                coef = [Fraction(rng.randint(-30, 30), rng.choice([1, 1, 2, 3, 7])) for _ in range(d + 1)]
                coef[-1] = coef[-1] or Fraction(1)
                m = mq.poly(coef)
                ys = values_of(m, codec)
                for op in OPS:
                    for c in (rng.choice(ys), rng.choice(ys) + Fraction(1, 3), Fraction(rng.randint(-10 ** 4, 10 ** 4), 7),
                              max(ys) + 1, min(ys) - 1):
                        s = self.check_exact(codec, m, op, c, ys=ys)
                        cases += 1
                        if trial < 6:
                            v, g = s.verify(), s.verify(backend="cuda")
                            self.assertEqual(v["mismatches"], 0)
                            self.assertEqual({k: x for k, x in v.items() if k not in ("backend", "device")},
                                             {k: x for k, x in g.items() if k not in ("backend", "device")})
                self.check_exact(codec, m, "in", [rng.choice(ys) for _ in range(4)] + [Fraction(1, 9)], ys=ys)
        self.assertGreater(cases, 2800)
        # the monotone pieces between certified critical points: (x - 1)(x - 2)(x - 3) has two critical points
        cd = mp.Codec(-20, 0, 0, 40)
        s = self.check_exact(cd, mq.poly([-6, 11, -6, 1]), ">", 0)
        self.assertEqual((s.info["critical"], s.info["pieces"] >= 3), (2, True))
        self.assertEqual(s.branches, [(24, 40)])                         # x = code - 20 > 3 (no integer in (1, 2))
        self.assertEqual(sorted(members(s, cd)), [k for k in range(41) if (k - 21) * (k - 22) * (k - 23) > 0])
        # the exact integer roots: = gives the finite set of roots in the domain (single points)
        r = self.check_exact(cd, mq.poly([-6, 11, -6, 1]), "=", 0)
        self.assertEqual(r.branches, [(21, 23)])                         # x = 1, 2, 3: adjacent points merge
        r = self.check_exact(cd, mq.poly([36, 0, -13, 0, 1]), "=", 0)    # (x^2 - 4)(x^2 - 9)
        self.assertEqual(r.branches, [(17, 18), (22, 23)])
        self.assertEqual(r.info["codes"], 4)
        self.check_exact(cd, mq.poly([1, 0, 1]), "=", 0)                 # no real root: empty
        self.assertTrue(mq.preimage(cd, mq.poly([1, 0, 1]), "=", 0).empty)
        self.assertTrue(mq.preimage(cd, mq.poly([5]), ">", 4).all)      # a constant map
        self.assertTrue(mq.preimage(cd, mq.poly([5, 0, 0]), "<", 4).empty)

    def test_pre002_squares_moduli_and_periods_match_exhaustive_domains(self):
        codecs = [mp.Codec(0, 0, 0, 255), mp.Codec(-128, 0, 0, 255), mp.Codec(-1000, 0, 500, 900), mp.Codec(3, 0, 0, 60),
                  mp.Codec(-50, 0, 0, 3000)]
        maps = [mq.poly([0, 0, 1]), mq.poly([-49, 0, 1]), mq.poly([0, 0, Fraction(-2, 3)]),   # squares (both branches)
                mq.powmod(2, 15), mq.powmod(2, 16), mq.powmod(3, 28), mq.powmod(2, 1000), mq.powmod(5, 7),
                mq.powmod(2, 997), mq.polymod([1, -3, 0, 2], 36), mq.polymod([4], 9), mq.powmod(1, 1),
                mq.affmod(6, 1, 20), mq.affmod(-4, 3, 30), mq.affmod(7, 0, 1000), mq.affmod(0, 5, 12), mq.affmod(12, 4, 18)]
        cases = 0
        for codec in codecs:
            for m in maps:
                ys = values_of(m, codec)
                thresholds = [0, 1, 4, 7, Fraction(5, 2), 13, -1, 999, 49]
                for op in OPS:
                    for c in thresholds:
                        self.check_exact(codec, m, op, c, fallback="none", ys=ys)
                        cases += 1
                for vals in ([0, 1, 4, 9, 10, Fraction(1, 2)], [], [3]):
                    self.check_exact(codec, m, "in", vals, fallback="none", ys=ys)
        self.assertGreater(cases, 4000)
        # residue classes by solving congruences, CRT over the prime powers of a composite modulus
        cd = mp.Codec(0, 0, 0, 65535)
        s = self.check_exact(cd, mq.powmod(2, 15), "=", 4)
        self.assertEqual(s.info["repr"], "residues")
        self.assertEqual({b[2] for b in s.branches}, {15})
        roots = sorted(r for r in range(15) if r * r % 15 == 4)            # 2, 7, 8, 13
        self.assertEqual([(b[3], b[4]) for b in s.branches], [(2, 2), (7, 8), (13, 13)])
        self.assertEqual(roots, [2, 7, 8, 13])
        s = self.check_exact(cd, mq.powmod(2, 1000), "=", 1)               # 1000 = 2^3 5^3: 4 x 2 roots
        self.assertEqual(sum(b[4] - b[3] + 1 for b in s.branches), sum(1 for r in range(1000) if r * r % 1000 == 1))
        # an affine periodic map has the period m / gcd(a, m)
        s = self.check_exact(cd, mq.affmod(6, 1, 20), "<", 7)
        self.assertEqual({b[2] for b in s.branches}, {10})
        s = self.check_exact(mp.Codec(-5, 0, 0, 7), mq.affmod(6, 1, 20), "<", 7)   # a domain below the period
        self.assertEqual(s.info["repr"], "intervals")
        # a zero modulus is a zero divisor
        for m in (mq.powmod(2, 0), mq.affmod(3, 1, 0), mq.polymod([1, 1], 0)):
            with self.assertRaises(DomainError):
                mq.preimage(cd, m, "<", 3)

    def test_pre002_bit_maps_give_boolean_cubes_of_the_stored_code(self):
        rng = random.Random(SEED + 1)
        codecs = [mp.Codec(0, 0, 0, 255), mp.Codec(-100, 0, 0, 255), mp.Codec(-3, 0, 5, 200), mp.Codec(40, 0, 0, 1023)]
        for codec in codecs:
            for trial in range(10):
                w = rng.choice([8, 10, 12])
                c = rng.randrange(1 << w)
                base = rng.choice([None, None, 0, -1, 3, 256])
                for m in (mq.xor(c, w, base=base), mq.bitand(c, w, base=base), mq.bitor(c, w, base=base)):
                    ys = values_of(m, codec)
                    for op in OPS:
                        for t in (0, rng.randrange(1 << w), Fraction(rng.randrange(1 << w), 2), (1 << w) - 1, 1 << w):
                            s = self.check_exact(codec, m, op, t, ys=ys)
                            if base is None and s.info["mode"] == "branches":
                                self.assertEqual(s.info["repr"], "cubes" if s.info["branches"] and not s.info["all"]
                                                 else s.info["repr"])
                    self.check_exact(codec, m, "in", [rng.randrange(1 << w) for _ in range(5)], ys=ys)
        # the GF(2) images of the dyadic blocks: x XOR c < k has at most w cubes, x AND mask = v is one cube
        cd = mp.Codec(0, 0, 0, 1023)
        s = self.check_exact(cd, mq.xor(0b1011001110, 10), "<", 613)
        self.assertEqual(s.info["repr"], "cubes")
        self.assertLessEqual(s.info["branches"], 10)
        self.assertEqual(s.info["branches"], bin(613).count("1"))
        a = self.check_exact(cd, mq.bitand(0b1100110000, 10), "=", 0b0100010000)
        self.assertEqual(a.branches, [(0, 1023, 0b1100110000, 0b0100010000)])
        self.assertTrue(mq.preimage(cd, mq.bitand(0b1100, 10), "=", 0b0010).empty)   # outside the mask: no cube
        o = self.check_exact(cd, mq.bitor(0b1001, 10), "=", 0b1111)
        self.assertEqual(o.branches, [(0, 1023, 0b1111111111 & ~0b1001, 0b0110)])
        # a base that is not aligned with the code: residues of a power-of-two period (or intervals)
        s = self.check_exact(mp.Codec(-3, 0, 0, 5000), mq.xor(0b1010, 8, base=0), "<", 40)
        self.assertEqual((s.info["repr"], s.info["modulus"]), ("residues", 256))   # y < 40 fixes bits 6 and 7
        s = self.check_exact(mp.Codec(-3, 0, 0, 5000), mq.xor(0b1010, 8, base=0), "=", 9)
        self.assertEqual((s.info["repr"], s.info["modulus"], len(s.branches)), ("residues", 256, 1))

    # ------------------------------------------------------------------ acceptance 2: branch explosion falls back
    def test_pre002_branch_explosion_triggers_a_budgeted_fallback_rather_than_dropping_solutions(self):
        cd = mp.Codec(0, 0, 0, 65535)
        cases = [(mq.affmod(37, 3, 1000), "<", 500), (mq.powmod(2, 997), "<", 300), (mq.bitand(0x8001, 16, base=7), "=", 0x8001),
                 (mq.poly([0, -40, 0, 1]), "in", [k ** 3 - 40 * k for k in range(-30, 200, 3)])]
        for m, op, c in cases:
            free = mq.preimage(cd, m, op, c) if op != "in" else mq.preimage(cd, m, op, values=c)
            nb = free.info["branches"]
            self.assertEqual(free.info["mode"], "branches")
            self.assertGreater(nb, 4)
            want = truth(m, cd, op, c)
            kw = dict(values=c) if op == "in" else {}
            args = (cd, m, op) + (() if op == "in" else (c,))
            at = mq.preimage(*args, budget=Budget(output=nb), **kw)           # at the limit: the branches
            self.assertEqual((at.info["mode"], at.info["fallback"]), ("branches", None))
            past = mq.preimage(*args, budget=Budget(output=nb - 1), **kw)     # one past: the lookup
            self.assertEqual((past.info["mode"], past.info["fallback"], past.info["why"]), ("lookup", "lookup", "output"))
            direct = mq.preimage(*args, budget=Budget(output=nb - 1), fallback="direct", **kw)
            self.assertEqual(direct.info["mode"], "direct")
            for s in (at, past, direct):                                    # never a dropped solution
                self.assertEqual(members(s, cd), want, (m, s.info["mode"]))
                v = s.verify()
                self.assertEqual((v["mismatches"], v["members"]), (0, len(want)))
                g = s.verify(backend="cuda")
                self.assertEqual((g["mismatches"], g["members"], g["truths"]), (0, len(want), len(want)))
            with self.assertRaises(AlgebraBudgetError) as e:
                mq.preimage(*args, budget=Budget(output=nb - 1), fallback="none", **kw)
            self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("output", nb, nb - 1))
            # the lookup's bytes are budgeted too: at its size the lookup, one byte below the direct evaluation
            lb = past.info["bytes"]                                          # the set and its bitmap
            self.assertEqual(lb - (past.info["nlookup"] + 63) // 64 * 8, at.info["bytes"] - nb * 64)
            self.assertEqual(mq.preimage(*args, budget=Budget(output=nb - 1, bytes=lb), **kw).info["mode"], "lookup")
            tight = mq.preimage(*args, budget=Budget(output=nb - 1, bytes=lb - 1), **kw)
            self.assertEqual(tight.info["mode"], "direct")
            self.assertEqual(members(tight, cd), want)
        # the iterations budget (residues enumerated) triggers the fallback the same way
        m = mq.powmod(3, 1 << 20)
        want = truth(m, cd, "<", 1000)
        s = mq.preimage(cd, m, "<", 1000, budget=Budget(iterations=1 << 19))   # the domain bitmap fits: a lookup
        self.assertEqual((s.info["mode"], s.info["fallback"], s.info["why"]), ("lookup", "lookup", "iterations"))
        d = mq.preimage(cd, m, "<", 1000, budget=Budget(iterations=1 << 19), fallback="direct")
        self.assertEqual((d.info["mode"], d.info["why"]), ("direct", "iterations"))
        for x in (s, d):
            self.assertEqual(members(x, cd), want)
            self.assertEqual((x.verify()["mismatches"], x.verify()["members"]), (0, len(want)))
        self.assertEqual(mq.preimage(cd, m, "<", 1000).info["mode"], "branches")
        with self.assertRaises(AlgebraBudgetError) as e:
            mq.preimage(cd, m, "<", 1000, budget=Budget(iterations=1 << 19), fallback="none")
        self.assertEqual(e.exception.resource, "iterations")

    # ------------------------------------------------------------------ engine tables, NULLs, DuckDB, planes, CUDA
    def test_pre002_scans_match_duckdb_engine_planes_and_cuda_with_null_guards(self):
        rng = np.random.default_rng(SEED)
        n = 3000
        k = [None if r < 0.07 else int(v) for r, v in zip(rng.random(n), rng.integers(-300, 301, n))]
        frame = pd.DataFrame({"k": pd.Series(k, dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT)", {"t": frame})
        t = db["t"]
        ck = mp.Codec.from_table(t, "k")
        off = ck.offset
        w = int(t.width("k"))
        K = "CAST(k AS HUGEINT)"
        cases = [
            (mq.poly([5, -7, 0, 1]), ">", 100, f"{K} * {K} * {K} - 7 * {K} + 5 > 100"),
            (mq.poly([0, -2, Fraction(1, 3)]), "<=", 50, f"{K} * {K} - 6 * {K} <= 150"),
            (mq.poly([-36, 0, 1]), "=", 0, f"{K} * {K} = 36"),
            (mq.powmod(2, 15), "=", 4, "(({K} * {K}) % 15 + 15) % 15 = 4".format(K=K)),
            (mq.polymod([0, 1, 0, 2], 28), "in", [1, 3, 5], f"((2 * {K} * {K} * {K} + {K}) % 28 + 28) % 28 IN (1, 3, 5)"),
            (mq.affmod(6, 1, 20), "<", 7, f"((6 * {K} + 1) % 20 + 20) % 20 < 7"),
            (mq.xor(0b1011001, w), "<=", 300, f"xor(CAST(k - ({off}) AS BIGINT) & {(1 << w) - 1}, {0b1011001}) <= 300"),
            (mq.bitand(0b1100, w), "=", 0b0100, f"(CAST(k - ({off}) AS BIGINT) & 12) = 4"),
            (mq.xor(0b101, 8, base=0), "<", 40, "xor(((k % 256) + 256) % 256, 5) < 40"),
        ]
        for sel in (None, rng.random(n) < 0.5):
            scope = mbk.Scope.from_table(t, ["k"], rows=sel)
            where = "TRUE"
            if sel is not None:
                db.con.register("pre2_sel", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
                where = "rowid IN (SELECT rid FROM pre2_sel)"
            for m, op, c, sql in cases:
                s = mq.preimage(ck, m, op, c) if op != "in" else mq.preimage(ck, m, op, values=c)
                bits, st = s.scan(scope, "k", direct=True)
                gbits, gst = s.scan(scope, "k", direct=True, backend="cuda")
                self.assertIn(gst["backend"], ("cuda", "cuda-emulated"))
                self.assertTrue(np.array_equal(bits, gbits), sql)
                strip = lambda d: {a: b for a, b in d.items() if a not in ("backend", "device")}
                self.assertEqual(strip(st), strip(gst))
                self.assertEqual((st["mismatches"], st["overflow"], st["outside"]), (0, 0, 0), sql)
                got = np.flatnonzero(np.unpackbits(bits.view(np.uint8), bitorder="little")[:n]).tolist()
                want = [r for r, in db.con.execute(f"SELECT rowid FROM t WHERE {where} AND k IS NOT NULL AND ({sql}) "
                                                   "ORDER BY rowid").fetchall()]
                self.assertEqual(got, want, sql)
                self.assertEqual((st["in_set"], st["direct_true"]), (len(want), len(want)))
                nulls = int(sum(1 for r in range(n) if (sel is None or sel[r]) and k[r] is None))
                self.assertEqual(st["nulls"], nulls)
                if s.info["mode"] == "branches" and (s.info["repr"] != "residues" or s.info["modulus"] & (s.info["modulus"] - 1) == 0):
                    prog = s.plane_program(t, "k")                     # the engine's bit planes on the branches
                    arr = (algq.PInstr * len(prog))(*prog)
                    bm = np.zeros((t.n + 63) // 64 + 1, dtype=np.uint64)
                    used = t.used_bm if sel is None else np.packbits(np.pad(sel, (0, (64 - n % 64) % 64)),
                                                                     bitorder="little").view(np.uint64)
                    cnt = algq.lib.at_filter_bitmap_masked(t.at, arr, len(prog), bm.ctypes.data, used.ctypes.data)
                    self.assertEqual(int(cnt), len(want), sql)
                    self.assertTrue(np.array_equal(bm[:len(bits)], bits), sql)
                elif s.info["mode"] == "branches":
                    with self.assertRaises(UnsupportedError):          # residues of another period: a lookup
                        s.plane_program(t, "k")
        # the residues mod 2^k of an unaligned bit map are low-bit cubes in the plane program
        s = mq.preimage(ck, mq.xor(0b101, 8, base=0), "<", 40)
        self.assertEqual(s.info["repr"], "residues")
        self.assertTrue(all(lo == ck.lo and hi == ck.hi and mask < 256 for lo, hi, mask, value in s.cubes(w)))
        # a lookup or a direct set has no plane program
        with self.assertRaises(UnsupportedError):
            mq.preimage(ck, mq.affmod(7, 3, 1000), "<", 500, budget=Budget(output=2)).plane_program(t, "k")

    def test_pre002_scope_from_arrays_outside_rows_and_the_direct_check(self):
        rng = np.random.default_rng(SEED + 3)
        n = 800
        vals = rng.integers(-70, 71, n)
        valid = rng.random(n) > 0.1
        scope = mbk.Scope.from_arrays({"x": vals}, valid={"x": valid})
        codec = mp.Codec(-50, 0, 0, 100)                                 # x in -50 .. 50: some values lie outside
        for m, op, c in ((mq.poly([3, 0, -1, Fraction(1, 5)]), "<", 2), (mq.powmod(2, 12), "=", 4),
                         (mq.xor(3, 7), ">", 50), (mq.affmod(5, 2, 9), "<>", 3)):
            s = mq.preimage(codec, m, op, c)
            for fb in (s, mq.preimage(codec, m, op, c, budget=Budget(output=1)),
                       mq.preimage(codec, m, op, c, budget=Budget(output=1), fallback="direct")):
                bits, st = fb.scan(scope, "x", direct=True)
                g, gst = fb.scan(scope, "x", direct=True, backend="cuda")
                self.assertTrue(np.array_equal(bits, g))
                self.assertEqual(st["outside"], int(np.sum(valid & ((vals < -50) | (vals > 50)))))
                self.assertEqual(st["mismatches"], 0)
                got = np.flatnonzero(np.unpackbits(bits.view(np.uint8), bitorder="little")[:n])
                inside = valid & (vals >= -50) & (vals <= 50)
                want = [r for r in range(n) if inside[r] and holds(f_of(m, codec, int(vals[r]) + 50), op, c)]
                self.assertEqual(got.tolist(), want)
        with self.assertRaises(ValueError):                              # another value unit (scale): refused
            s.scan(mbk.Scope.from_arrays({"x": vals}, scales={"x": 1}), "x")
        # a polynomial whose bound passes 512 bits has no direct evaluation: its direct check is refused
        huge = mq.preimage(mp.Codec(-(1 << 40), 0, 0, 1000), mq.poly([0] * 16 + [1]), ">", 5)
        self.assertEqual(huge.info["words"], 0)
        with self.assertRaises(UnsupportedError):
            huge.scan(mbk.Scope.from_arrays({"x": vals}), "x", direct=True)

    def test_pre002_exhaustive_verification_cpu_equals_cuda_in_every_mode(self):
        cd = mp.Codec(-300, 0, 0, 20000)
        sets = [mq.preimage(cd, mq.poly([7, -3, 0, Fraction(1, 2)]), ">=", 11),
                mq.preimage(cd, mq.powmod(3, 91), "in", values=[1, 8, 27, 64]),
                mq.preimage(cd, mq.affmod(9, 4, 66), "<", 30),
                mq.preimage(cd, mq.bitor(0b1000110, 15), "=", 0b1001110),
                mq.preimage(cd, mq.affmod(9, 4, 66), "<", 30, budget=Budget(output=2)),
                mq.preimage(cd, mq.affmod(9, 4, 66), "<", 30, budget=Budget(output=2), fallback="direct"),
                mq.preimage(cd, mq.poly([7, -3, 0, Fraction(1, 2)]), "in", values=list(range(-50, 50, 7)), budget=Budget(output=3))]
        modes = set()
        for s in sets:
            v, g = s.verify(), s.verify(backend="cuda")
            strip = lambda d: {k: x for k, x in d.items() if k not in ("backend", "device")}
            self.assertEqual(strip(v), strip(g))
            self.assertEqual(v["mismatches"], 0)
            self.assertEqual(v["codes"], cd.hi - cd.lo + 1)
            self.assertEqual(v["members"], len(members(s, cd)))
            modes.add(s.info["mode"])
        self.assertEqual(modes, {"branches", "lookup", "direct"})
        with self.assertRaises(AlgebraBudgetError):
            sets[0].verify(budget=Budget(iterations=20000))
        sets[0].verify(budget=Budget(iterations=20001))
        with self.assertRaises(UnsupportedError):
            mq.preimage(mp.Codec(0, 0, 0, (1 << 30)), mq.poly([0, 1]), ">", 5).verify()

    def test_pre002_certificates_are_fnd007_laws_of_the_transformation(self):
        for codec, m in ((mp.Codec(-128, 0, 0, 255), mq.poly([1, -2, Fraction(1, 3), 4])),
                         (mp.Codec(37, 2, 0, 200), mq.poly([0, 0, Fraction(2, 3)])),
                         (mp.Codec(-1000, 0, 0, 999), mq.powmod(3, 28)),
                         (mp.Codec(-1000, 0, 0, 999), mq.polymod([1, 0, -5, 2], 1000)),
                         (mp.Codec(5, 0, 0, 999), mq.affmod(6, 1, 20))):
            s = mq.preimage(codec, m, "<", 3)
            certs = s.certificates
            self.assertEqual((len(certs), s.info["certified"]), (1, 1))
            cert = certs[0]
            self.assertEqual((cert.kind, cert.info["law"]), ("law", "ring_identity"))
            self.assertEqual(cert.authorize_exact(field=0).kind, "law")
            c = cert._native_claim()
            nt, nv = int(c.nterms), int(c.ncols)
            exps = (ctypes.c_uint8 * (nt * nv)).from_address(c.exponents)
            co = (ctypes.c_int64 * nt).from_address(c.coefficients)
            total = {}
            for i in range(nt):
                key = tuple(exps[i * nv + q] for q in range(nv))
                total[key] = total.get(key, 0) + co[i]
            self.assertTrue(all(v == 0 for v in total.values()), total)   # canonically zero, independently
            self.assertGreater(nt, 2)
        # the periodicity law's content: P(r + m t) - P(r) has the factor m in every term (variables r, t, m)
        s = mq.preimage(mp.Codec(0, 0, 0, 99), mq.polymod([2, 3, 1], 7), "=", 1)
        c = s.certificates[0]._native_claim()
        nt = int(c.nterms)
        exps = (ctypes.c_uint8 * (nt * 3)).from_address(c.exponents)
        self.assertTrue(all(exps[3 * i + 1] == exps[3 * i + 2] for i in range(nt)))   # t and m together
        # coefficients past int64: no certificate, the set is still exact
        big = mp.Codec(-(1 << 40), 9, 0, 300)
        s = self.check_exact(big, mq.poly([0, 0, 0, Fraction(1, 3)]), "<", 5)
        self.assertEqual((s.info["certified"], s.certificates), (0, []))
        self.assertEqual(mq.preimage(mp.Codec(0, 0, 0, 10), mq.xor(1, 4), "<", 3).certificates, [])

    def test_pre002_budgets_at_the_limit_and_one_past_and_malformed_requests(self):
        cd = mp.Codec(0, 0, 0, 255)
        vals = list(range(0, 400, 3))
        mq.preimage(cd, mq.poly([0, 1]), "in", values=vals, budget=Budget(support=len(vals)))
        with self.assertRaises(AlgebraBudgetError) as e:
            mq.preimage(cd, mq.poly([0, 1]), "in", values=vals, budget=Budget(support=len(vals) - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("support", len(vals)))
        mq.preimage(cd, mq.poly([0, 0, 0, 0, 1]), "<", 9, budget=Budget(degree=4))
        with self.assertRaises(AlgebraBudgetError) as e:
            mq.preimage(cd, mq.poly([0, 0, 0, 0, 1]), "<", 9, budget=Budget(degree=3))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("degree", 4, 3))
        frame = pd.DataFrame({"x": np.arange(300)})
        t = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})["t"]
        scope = mbk.Scope.from_table(t, ["x"])
        s = mq.preimage(mp.Codec.from_table(t, "x"), mq.powmod(2, 77), "<", 30)
        need = (300 + 63) // 64 * 8
        s.scan(scope, "x", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError):
            s.scan(scope, "x", budget=Budget(bytes=need - 1))
        with self.assertRaises(AlgebraBudgetError) as cm:                # the device copies: the exact need
            s.scan(scope, "x", backend="cuda", direct=True, budget=Budget(bytes=64))
        dev = cm.exception.need
        s.scan(scope, "x", backend="cuda", direct=True, budget=Budget(bytes=dev))
        with self.assertRaises(AlgebraBudgetError):
            s.scan(scope, "x", backend="cuda", direct=True, budget=Budget(bytes=dev - 1))
        # malformed or unsupported requests
        with self.assertRaises(ValueError):
            mq.preimage(cd, mq.poly([0, 1]), "~", 1)
        with self.assertRaises(TypeError):
            mq.poly([0.5, 1])                                             # floats are not exact input
        with self.assertRaises(TypeError):
            mq.preimage(cd, mp.affine(1, 0), "<", 1)                      # a PRE-001 map
        with self.assertRaises(TypeError):
            mq.polymod([Fraction(1, 2)], 5)                               # polymod coefficients are integers
        with self.assertRaises(UnsupportedError):
            mq.preimage(mp.Codec(0, 1, 0, 10), mq.powmod(2, 5), "<", 1)   # residues of a decimal value
        with self.assertRaises(UnsupportedError):
            mq.preimage(cd, mq.xor(1, 64), "<", 1)
        with self.assertRaises(UnsupportedError):
            mq.preimage(mp.Codec(0, 0, -5, 10), mq.xor(1, 4), "<", 1)      # bit maps of negative codes
        with self.assertRaises(UnsupportedError):
            mq.poly([1] * 18)
        with self.assertRaises(ValueError):
            mq.preimage(cd, mq.affmod(1, 0, -3), "<", 1)
        with self.assertRaises(ValueError):
            mq.preimage(cd, mq.xor(16, 4), "<", 1)                        # the constant past the width
        with self.assertRaises(ValueError):
            mq.preimage(cd, mq.poly([0, 1]), "<", 1, fallback="sometimes")
        with self.assertRaises(UnsupportedError):
            mq.preimage(cd, mq.poly([0, 1]), "in", values=list(range(mq.MAX_VALUES + 1)))

    def test_pre002_import_torbles_does_not_load_the_module(self):
        code = "import sys, torbles; print('torbles.math_preimg' in sys.modules, 'torbles.math_inv' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ),
                             check=True).stdout.strip()
        self.assertEqual(out, "False False")


if __name__ == "__main__":
    unittest.main()
