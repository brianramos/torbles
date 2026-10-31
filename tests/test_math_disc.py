# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-001: bounded affine, sum and product discovery, and its exhaustive verification on the CPU and in CUDA.

Oracle: the relation recomputed row by row with Python integers from the source values (never the native code),
plus the existing relationship suite, which must keep passing unchanged. Seeds are fixed.
"""
import ctypes as C
import unittest

import numpy as np
import pandas as pd

import torbles
from torbles import math_disc
from torbles.math_budgets import Budget
from torbles.math_errors import StaleStateError
from torbles.mathematics import AlgebraBudgetError
from torbles.relationships import discover


def make(values):
    frame = pd.DataFrame({name: pd.Series(v, dtype=object) for name, v in values.items()})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in values) + ")"
    return torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)


def oracle(values, used, module, out, left, right=None, scale=1, bias=0):
    """the evidence recomputed from the source values with Python integers"""
    n = len(values[out])
    checked = excluded = violations = 0
    witness, xs, ys = None, set(), set()
    for row in range(n):
        if not used[row]:
            continue
        names = [out, left] + ([right] if right is not None else [])
        if any(values[c][row] is None for c in names):
            excluded += 1
            continue
        a, x = values[out][row], values[left][row]
        y = values[right][row] if right is not None else 0
        predicted = scale * x + bias if module == "affine" else (x * y if module == "product" else x + y)
        checked += 1
        xs.add(x); ys.add(y)
        if predicted != a:
            violations += 1
            witness = row if witness is None else min(witness, row)
    usable = (all(used) and checked >= 2 and not excluded and not violations and len(xs) > 1
              and (module == "affine" or len(ys) > 1) and (module != "affine" or scale != 0))
    return dict(checked=checked, excluded=excluded, violations=violations, witness=witness, usable=usable)


def planted(rng, n):
    x = [int(v) for v in rng.integers(-60, 61, size=n)]
    y = [int(v) for v in rng.integers(-40, 90, size=n)]
    near = [a + b for a, b in zip(x, y)]
    near[int(rng.integers(0, n))] += 1                        # one-row near miss of a conservation law
    z = [2 * a + 1 for a in x]
    for i in rng.choice(n, size=max(1, n // 20), replace=False):
        z[int(i)] = None                                      # nullable copy of an affine law
    return {"x": x, "y": y, "s": [a + b for a, b in zip(x, y)], "p": [a * b for a, b in zip(x, y)],
            "a": [3 * v - 7 for v in x], "q": near, "z": z, "sq": [v * v for v in x]}


HYPOTHESES = [("affine", "a", "x", None, 3, -7), ("affine", "a", "x", None, 3, -6), ("affine", "a", "x", None, 0, 5),
              ("sum", "s", "x", "y", 1, 0), ("product", "p", "x", "y", 1, 0), ("sum", "q", "x", "y", 1, 0),
              ("affine", "z", "x", None, 2, 1), ("product", "sq", "x", "x", 1, 0), ("sum", "p", "x", "y", 1, 0)]


class Discovery(unittest.TestCase):
    def catalog(self, db, columns=None, **options):
        cat = discover(db, "t", columns, **options)
        self.addCleanup(cat.close)
        return cat

    def compare(self, values, cat, used):
        for module, out, left, right, scale, bias in HYPOTHESES:
            expect = oracle(values, used, module, out, left, right, scale, bias)
            probe = cat.probe(module, out, left, right, scale=scale, bias=bias)
            got_probe = dict(checked=probe["checked"], excluded=probe["excluded"], violations=probe["violations"],
                             witness=probe.get("witness", {}).get("stored_row"), usable=probe["usable"])
            if not probe["violations"]:
                got_probe["witness"] = None
            self.assertEqual(got_probe, expect, (module, out, left, right, scale, bias))
            for backend in ("cpu", "cuda"):
                ev = math_disc.verify(cat, module, out, left, right, scale=scale, bias=bias, backend=backend)
                got = {k: ev[k] for k in expect}
                self.assertEqual(got, expect, (backend, module, out, left, right, scale, bias))

    def test_disc001_existing_relationship_suite_passes(self):
        suite = unittest.TestLoader().loadTestsFromName("test_relationships")
        result = unittest.TestResult()
        suite.run(result)
        self.assertTrue(result.wasSuccessful(), [str(f[1])[-400:] for f in result.failures + result.errors])
        self.assertGreaterEqual(result.testsRun, 15)

    def test_disc001_cpu_cuda_and_probe_match_the_oracle(self):
        rng = np.random.default_rng(101)
        for n in (2, 7, 300, 5000):
            values = planted(rng, n)
            db = make(values)
            cat = self.catalog(db, list(values), modules=(), max_candidates=0)
            self.compare(values, cat, [True] * n)

    def test_disc001_signed_offsets_square_and_one_row_violation(self):
        xs = [-17, -11, -3, 0, 8, 19]
        values = {"x": xs, "y": [2 * x + 3 for x in xs], "sq": [x * x for x in xs],
                  "s": [x + (x * x) for x in xs]}
        values["s"][4] += 1
        db = make(values)
        cat = self.catalog(db, list(values), modules=(), max_candidates=0)
        for backend in ("cpu", "cuda"):
            ev = math_disc.verify(cat, "affine", "y", "x", scale=2, bias=3, backend=backend)
            self.assertEqual((ev["checked"], ev["violations"], ev["usable"]), (6, 0, True))
            ev = math_disc.verify(cat, "product", "sq", "x", "x", backend=backend)
            self.assertTrue(ev["usable"])
            ev = math_disc.verify(cat, "sum", "s", "x", "sq", backend=backend)
            self.assertEqual((ev["violations"], ev["witness"], ev["usable"]), (1, 4, False))

    def test_disc001_partial_nullable_and_stale_proofs_are_refused_as_rewrites(self):
        values = {"x": [1, 2, 3, 4, 5], "y": [5, 7, 9, 11, 14], "z": [3, None, 7, 9, 11]}
        db = make(values)
        t = db["t"]
        t.used_bm[0] = np.uint64(0b01111)                     # the law holds on the first four rows only
        cat = self.catalog(db, list(values), modules=(), max_candidates=0)
        for backend in ("cpu", "cuda"):
            ev = math_disc.verify(cat, "affine", "y", "x", scale=2, bias=3, backend=backend)
            self.assertEqual((ev["checked"], ev["violations"], ev["usable"]), (4, 0, False))   # partial
            ev = math_disc.verify(cat, "affine", "z", "x", scale=2, bias=1, backend=backend)
            self.assertEqual((ev["checked"], ev["excluded"], ev["usable"]), (3, 1, False))     # nullable
        self.assertEqual(cat.rules, [])
        t.used_bm[0] = np.uint64(0b11111)                     # the used rows changed since discovery: stale
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                math_disc.verify(cat, "affine", "y", "x", scale=2, bias=3, backend=backend)
        db2 = make({"x": [1, 2, 3, 4], "y": [5, 7, 9, 11]})
        cat2 = self.catalog(db2, ["x", "y"], modules=(), max_candidates=0)
        self.assertTrue(math_disc.verify(cat2, "affine", "y", "x", scale=2, bias=3)["usable"])
        db2["t"].flip(0, 0)                                   # a mutation: new revision
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                math_disc.verify(cat2, "affine", "y", "x", scale=2, bias=3, backend=backend)
        with self.assertRaisesRegex(ValueError, "stale"):
            cat2.probe("affine", "y", "x", scale=2, bias=3)

    def raw_view(self, codes, offsets, used=None, valid=None, module=1):
        n = len(codes[0])
        arrays = [np.ascontiguousarray(c, dtype=np.uint32) for c in codes]
        valids = [None if v is None else np.ascontiguousarray(v, dtype=np.uint32) for v in (valid or [None] * 3)]
        used_bits = np.zeros(max(1, (n + 63) // 64), dtype=np.uint64)
        mask = np.ones(n, dtype=bool) if used is None else np.asarray(used, dtype=bool)
        for i in np.flatnonzero(mask):
            used_bits[i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        v = math_disc._View()
        v.rows, v.used_count, v.used = n, int(mask.sum()), used_bits.ctypes.data
        v.module, v.arity = module, 1 if module == 1 else 2
        for i in range(v.arity + 1):
            v.codes[i] = arrays[i].ctypes.data
            v.validity[i] = valids[i].ctypes.data if valids[i] is not None else None
            v.offset[i] = offsets[i]
        self._keep = (arrays, valids, used_bits)
        return v

    def run_raw(self, v, scale, bias, backend):
        e = math_disc._Evidence()
        if backend == "cpu":
            rc = math_disc._api()["verify"](C.byref(v), scale, bias, C.byref(e))
        else:
            gpu = math_disc._gpu()
            rc = gpu.lib().tr_mgpu_disc_verify(C.byref(v), scale, bias, C.byref(Budget().raw), C.byref(e),
                                               C.byref(math_disc._Refusal()))
        return rc, e

    def test_disc001_logical_overflow_refused_identically(self):
        big = (1 << 63) - 10
        v = self.raw_view([[5, 20], [1, 2], [0, 0]], [big, 0, 0])          # out = code + offset overflows on row 1
        for backend in ("cpu", "cuda"):
            rc, _ = self.run_raw(v, 1, 0, backend)
            self.assertEqual(rc, -3, backend)
        v = self.raw_view([[5, 20], [1, 2], [0, 0]], [big, 0, 0], valid=[[1, 0], None, None])
        for backend in ("cpu", "cuda"):                                     # the overflowing row is NULL: excluded
            rc, e = self.run_raw(v, 1, 0, backend)
            self.assertEqual((rc, e.checked, e.excluded), (0, 1, 1), backend)

    def test_disc001_affine_proposals_equal_discovery(self):
        rng = np.random.default_rng(9)
        for n in (1, 2, 50, 3000):
            x = [int(v) for v in rng.integers(-30, 31, size=n)]
            values = {"x": x, "a": [5 * v - 11 for v in x], "h": [v // 2 for v in x],
                      "c": [7] * n, "r": [int(v) for v in rng.integers(-99, 99, size=n)]}
            db = make(values)
            cat = self.catalog(db, list(values), modules=("affine",), max_candidates=1000)
            registered = {r["expression"] for r in cat.rules}
            for out in values:
                for left in values:
                    if out == left:
                        continue
                    proposal = math_disc.propose_affine(cat, out, left)
                    if proposal is None:
                        continue
                    ev = math_disc.verify(cat, "affine", out, left, scale=proposal[0], bias=proposal[1])
                    self.assertEqual(ev["usable"], f"{out} = {proposal[0]}*{left} + {proposal[1]}" in registered,
                                     (n, out, left, proposal))
            if n >= 2:
                self.assertEqual(math_disc.propose_affine(cat, "a", "x"), (5, -11))

    def test_disc001_device_copies_are_budgeted(self):
        values = planted(np.random.default_rng(7), 1000)
        cat = self.catalog(make(values), list(values), modules=(), max_candidates=0)
        need = math_disc.device_bytes(cat, "sum", "s", "x", "y")
        self.assertTrue(math_disc.verify(cat, "sum", "s", "x", "y", backend="cuda", budget=Budget(bytes=need))["usable"])
        with self.assertRaises(AlgebraBudgetError) as ctx:
            math_disc.verify(cat, "sum", "s", "x", "y", backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))

    def test_disc001_discovery_budget_truncates_explicitly_and_keeps_proofs_exact(self):
        values = planted(np.random.default_rng(8), 2000)
        cols = ["x", "y", "s", "p", "a"]
        db = make({c: values[c] for c in cols})
        full = self.catalog(db, cols, max_candidates=100_000)
        self.assertFalse(full.stats["truncated"])
        found = {r["expression"] for r in full.rules}
        self.assertTrue({"a = 3*x + -7", "s = x + y", "p = x * y"} <= found, found)
        for r in full.rules:                                    # every registered rule re-verifies exactly
            module = r["module"]
            out, rest = r["expression"].split(" = ")
            if module == "affine":
                scale, rhs = rest.split("*")
                left, bias = rhs.split(" + ")
                ev = math_disc.verify(full, "affine", out, left, scale=int(scale), bias=int(bias), backend="cuda")
            else:
                left, right = rest.split(" * " if module == "product" else " + ")
                ev = math_disc.verify(full, module, out, left, right, backend="cuda")
            self.assertTrue(ev["usable"], r)
        small = discover(db, "t", cols, max_candidates=3, attach=False)
        self.addCleanup(small.close)
        self.assertTrue(small.stats["truncated"])
        self.assertEqual(small.stats["candidates"], 3)


if __name__ == "__main__":
    unittest.main()
