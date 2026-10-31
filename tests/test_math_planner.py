# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-008: complete-plan choice and cost evidence (torbles/math_planner.c, math_planner_gpu.cu).

Oracles, independent of the implementation: sampled and exact selected counts recomputed in Python from the bitmap
at the documented word positions; every executed plan's result recomputed row by row with Python integers and with
DuckDB SQL over the source values; costs recomputed as amounts x the registry's coefficients and the cheapest
admissible candidate found in Python; DISC-001 evidence recomputed from the source values. The sampling kernel and
every forced CUDA execution run on the CUDA path (emulated here, scrambled thread order) and must equal the CPU.
Seeds are fixed.
"""
import os
import unittest
from fractions import Fraction
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_gpu, math_planner as mp, math_registry as reg
from torbles.math_budgets import Budget
from torbles.math_domains import CarrierError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError


def make(columns):
    frame = pd.DataFrame({name: pd.Series(v, dtype=object) for name, v in columns.items()})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in columns) + ")"
    return torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)


def poly_value(p, x):
    return sum(c * x ** k for k, c in enumerate(p))


def oracle(values, mask, polys, aggregate="sum"):
    valid = [v for v, m in zip(values, mask) if m and v is not None]
    out = []
    for p in polys:
        vals = [poly_value(p, x) for x in valid]
        out.append(sum(vals) if aggregate == "sum" else ((min(vals) if aggregate == "min" else max(vals)) if vals else None))
    return out


def sample_oracle(bits, rows, n, exact):
    """the documented sample in Python: positions, counts, bounds, estimate"""
    words = (rows + 63) // 64
    n = words if exact or n >= words else n
    pos = list(range(words)) if n >= words else ([0] if n <= 1 else [i * (words - 1) // (n - 1) for i in range(n)])
    m = k = 0
    for w in pos:
        here = min(64, rows - 64 * w)
        mask = (1 << here) - 1
        word = int(bits[w]) if bits is not None else (1 << 64) - 1
        m += here
        k += bin(word & mask).count("1")
    is_exact = exact or n == words
    if is_exact:
        return dict(sample_words=n, sample_rows=m, sample_selected=k, estimate=k, lower=k, upper=k, exact=True,
                    empty_proven=k == 0, zero_sample=False)
    est = Fraction(rows * (k + 1), m + 2)
    est = int(est) + (1 if est - int(est) >= Fraction(1, 2) else 0)
    lo, hi = k, k + rows - m
    return dict(sample_words=n, sample_rows=m, sample_selected=k, estimate=min(max(est, lo), hi), lower=lo, upper=hi,
                exact=False, empty_proven=False, zero_sample=k == 0)


def packed(mask):
    n = len(mask)
    padded = np.zeros(max((n + 63) // 64, 1) * 64, dtype=np.uint8)
    padded[:n] = mask
    return np.packbits(padded, bitorder="little").view(np.uint64)


class Planner(unittest.TestCase):
    def table(self, seed, n, nulls=0.05):
        rng = np.random.default_rng(seed)
        x = [int(v) for v in rng.integers(-30, 150, size=n)]
        for i in rng.choice(n, size=int(n * nulls), replace=False):
            x[int(i)] = None
        y = [int(v) for v in rng.integers(0, 20, size=n)]
        db = make({"x": x, "y": y})
        self.addCleanup(db.close)
        col = reg.Column(db["t"], "x")
        self.addCleanup(col.close)
        return db, col, x, y

    def setUp(self):
        self._env = os.environ.pop("TORBLES_MATH_PLAN_FORCE", None)

    def tearDown(self):
        os.environ.pop("TORBLES_MATH_PLAN_FORCE", None)
        if self._env is not None:
            os.environ["TORBLES_MATH_PLAN_FORCE"] = self._env

    # -------------------------------------------------------------- selectivity
    def test_fnd008_sampling_kernel_equals_cpu_and_python(self):
        rng = np.random.default_rng(81)
        for rows in (0, 1, 63, 64, 65, 1000, 64 * 100, 100_003):
            for density in (0.0, 0.003, 0.4, 1.0):
                mask = rng.random(rows) < density
                bits = packed(mask)
                for n in (1, 2, 7, 32, 10 ** 6):
                    for exact in (False, True):
                        expect = sample_oracle(bits, rows, n, exact)
                        cpu = mp.sample(bits=mask, rows=rows, sample_words=n, exact=exact)
                        cuda = mp.sample(bits=mask, rows=rows, sample_words=n, exact=exact, backend="cuda")
                        self.assertEqual({k: cpu[k] for k in expect}, expect, (rows, density, n, exact))
                        self.assertEqual({k: v for k, v in cpu.items() if k != "backend"},
                                         {k: v for k, v in cuda.items() if k != "backend"}, (rows, density, n, exact))
                        if exact:
                            self.assertEqual(cpu["sample_selected"], int(mask.sum()))
        everything = mp.sample(rows=5000, sample_words=8)                     # no selection: every row
        self.assertEqual(everything, mp.sample(rows=5000, sample_words=8, backend="cuda") | {"backend": "cpu"})
        self.assertEqual({k: everything[k] for k in ("sample_selected", "sample_rows")},
                         {k: sample_oracle(None, 5000, 8, False)[k] for k in ("sample_selected", "sample_rows")})
        self.assertEqual(everything["sample_selected"], everything["sample_rows"])
        self.assertIn(math_gpu.device()["emulated"], (True, False))
        words = (100_003 + 63) // 64
        need = words * 8 + 16                                                # the device copy is budgeted
        mask = rng.random(100_003) < 0.5
        mp.sample(bits=mask, rows=100_003, backend="cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mp.sample(bits=mask, rows=100_003, backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))

    # -------------------------------------------------------------- acceptance: no empty proof from a sample
    def test_fnd008_zero_sampled_match_is_never_an_empty_proof(self):
        rows = 64 * 100
        rng = np.random.default_rng(82)
        x = [int(v) for v in rng.integers(1, 90, size=rows)]
        sampled = {i * 99 // 31 for i in range(32)}
        self.assertNotIn(1, sampled)
        mask = np.zeros(rows, dtype=bool)
        mask[64:69] = True                                                   # five rows, in an unsampled word
        codes = np.array(x, dtype=np.uint32)
        req = reg.Request(codes, [[3, 0, 1]], rows=mask)
        expect = oracle(x, mask, req.polys)
        self.assertGreater(expect[0], 0)
        for backend in ("cpu", "cuda"):
            s = mp.sample(req, backend=backend)
            self.assertEqual((s["sample_selected"], s["zero_sample"], s["empty_proven"], s["exact"]),
                             (0, True, False, False))
            self.assertGreaterEqual(s["upper"], 5)
            plan = mp.plan(req, sample_backend=backend)
            empty = plan["candidates"][0]
            self.assertEqual((empty["representation"], empty["status"], empty["refusal"]["kind"]),
                             ("empty", "rejected", "proof"))
            self.assertNotEqual(plan["choice"]["representation"], "empty")
            out = mp.run(req, sample_backend=backend)
            self.assertEqual(out["values"], expect)
            self.assertEqual(out["explanation"]["rows_scanned"], 5)
        with patch.object(reg, "execute", side_effect=AssertionError("a refused plan executed")):
            with self.assertRaises(UnsupportedError) as ctx:
                mp.run(req, force="empty")
        self.assertEqual((ctx.exception.plan["chosen"], ctx.exception.plan["reason"]),
                         (None, "forced representation refused"))
        exact = mp.plan(req, exact=True)
        self.assertEqual((exact["selectivity"]["sample_selected"], exact["selectivity"]["empty_proven"]), (5, False))
        self.assertEqual(exact["candidates"][0]["status"], "rejected")
        # a selection that is empty: a sample still proves nothing, so the plan scans (and finds nothing) ...
        nothing = reg.Request(codes, [[3, 0, 1]], rows=np.zeros(rows, dtype=bool))
        for backend in ("cpu", "cuda"):
            sampled_plan = mp.run(nothing, sample_backend=backend)
            self.assertNotEqual(sampled_plan["explanation"]["representation"], "empty")
            self.assertEqual((sampled_plan["values"], sampled_plan["selected"]), ([0], 0))
            self.assertTrue(sampled_plan["plan"]["selectivity"]["zero_sample"])
            # ... and only an exact zero count admits the empty answer, which then needs no scan
            proven = mp.run(nothing, exact=True, sample_backend=backend)
            self.assertTrue(proven["plan"]["selectivity"]["empty_proven"])
            self.assertEqual((proven["explanation"]["representation"], proven["values"],
                              proven["explanation"]["rows_scanned"]), ("empty", [0], 0))
        self.assertEqual(mp.run(nothing.with_polys([[1, 1]], "max"), exact=True)["values"], [None])
        self.assertEqual(mp.run(nothing, exact=True, force="empty")["explanation"]["representation"], "empty")
        # the empty executor recounts natively: a wrong proof cannot produce a wrong answer
        with self.assertRaises(UnsupportedError):
            reg.execute(req, "empty")

    # -------------------------------------------------------------- acceptance: cost evidence
    def test_fnd008_selected_and_rejected_candidates_retain_cost_evidence(self):
        db, col, x, y = self.table(83, 4000)
        ids = db.query(db["t"]).where(db["t"].y < 7).rows()
        mask = [v < 7 for v in y]
        req = reg.Request(col, [[1, -1, 2], [0, 3], [5, 0, 0, 1]], row_ids=ids)
        ncodes = int(max(db["t"].decoded("x"))) + 1
        rates = {(m["id"], dev): m["costs"][dev] for m in reg.modules() for dev in ("cpu", "cuda")}
        for budget in (Budget(), Budget(states=ncodes), Budget(states=ncodes - 1)):
            plan = mp.plan(req, devices=("cpu", "cuda"), budget=budget)
            cands = plan["candidates"]
            self.assertEqual(len(cands), 10)
            for c in cands:
                r = rates[(c["module"], c["device"])]
                for comp in reg.COST_COMPONENTS:
                    if comp != "fusion":
                        self.assertAlmostEqual(c["costs"][comp], c["amounts"][comp] * r[comp], places=6)
                shared = sum(c["costs"][s] for s in c["shared"])
                self.assertAlmostEqual(c["costs"]["fusion"], -r["fusion"] * shared * 2 / 3, places=6)
                self.assertAlmostEqual(c["total"], sum(c["costs"].values()), places=6)
                self.assertGreater(c["total"], 0)
                self.assertIn(c["status"], ("chosen", "admissible", "rejected"))
            admissible = [i for i, c in enumerate(cands) if c["status"] != "rejected"]
            best = min(admissible, key=lambda i: (cands[i]["total"], i))
            self.assertEqual(plan["chosen"], best)
            self.assertEqual([c["status"] for c in cands].count("chosen"), 1)
            self.assertEqual(cands[best]["reason"], "cheapest")
            self.assertTrue(all(cands[i]["reason"] == "costlier" for i in admissible if i != best))
            self.assertEqual((cands[0]["status"], cands[0]["reason"]), ("rejected", "refused by the registry"))
            hist = [c for c in cands if c["representation"] == "histogram"]
            if budget.limits["states"] < ncodes:
                for h in hist:
                    self.assertEqual((h["status"], h["reason"], h["budget"]),
                                     ("rejected", "over budget", dict(resource="states", need=ncodes, limit=ncodes - 1)))
                    self.assertGreater(h["amounts"]["construct"], 0)
            else:
                self.assertTrue(all(h["status"] != "rejected" for h in hist))
            out = mp.run(req, devices=("cpu", "cuda"), budget=budget)
            self.assertEqual(out["values"], oracle(x, mask, req.polys))
            self.assertEqual((out["explanation"]["representation"], out["explanation"]["device"].startswith("cuda")),
                             (plan["choice"]["representation"], plan["choice"]["device"] == "cuda"))
        # rejected by laws and conversions: they keep their amounts and costs too
        mx = mp.plan(req.with_polys([[1, 2]], "max"))
        kinds = {c["representation"]: c for c in mx["candidates"]}
        self.assertEqual((kinds["power"]["refusal"]["law"], kinds["bitplane"]["refusal"]["conversion"]),
                         ("distributive", ("count", "minmax")))
        self.assertTrue(kinds["power"]["total"] > 0 and kinds["bitplane"]["amounts"]["construct"] > 0)
        # CPU and GPU finalization are accounted for separately
        plan = mp.plan(req, devices=("cpu", "cuda"))
        hg, hc = [c for c in plan["candidates"] if c["representation"] == "histogram" and c["device"] == "cuda"]
        self.assertEqual((hg["finalize"], hc["finalize"]), ("gpu", "cpu"))
        self.assertGreater(hg["amounts"]["gpu_final"], 0); self.assertEqual(hg["amounts"]["cpu_final"], 0)
        self.assertGreater(hc["amounts"]["cpu_final"], 0); self.assertEqual(hc["amounts"]["gpu_final"], 0)
        self.assertEqual(hc["amounts"]["transfer"] - hg["amounts"]["transfer"], 3 * 8 * ncodes + 24 - (16 * 3 + 24 + 16))
        # coefficient overrides change the costs and the choice, never the result
        scan_first = mp.coefficients({(r, "cpu"): {"chain": 1e6} for r in ("histogram", "power", "bitplane")})
        p2 = mp.plan(req, coefficients=scan_first)
        self.assertEqual(p2["choice"]["representation"], "scan")
        self.assertEqual(mp.run(req, coefficients=scan_first)["values"], oracle(x, mask, req.polys))

    # -------------------------------------------------------------- acceptance: forced controls
    def test_fnd008_forced_controls_execute_their_declared_representation(self):
        db, col, x, y = self.table(84, 3000)
        mask = [v % 3 == 0 for v in y]
        req = reg.Request(col, [[2, 1, 0, -1]], rows=np.array(mask))
        expect = oracle(x, mask, req.polys)
        emulated = math_gpu.device()["emulated"]
        forces = [("scan@cpu", "scan", "cpu", "cpu"), ("histogram@cpu", "histogram", "cpu", "cpu"),
                  ("power@cpu", "power", "cpu", "cpu"), ("bitplane@cpu", "bitplane", "cpu", "cpu"),
                  ("scan@cuda", "scan", "cuda", "gpu"), ("histogram@cuda:gpu", "histogram", "cuda", "gpu"),
                  ("histogram@cuda:cpu", "histogram", "cuda", "cpu"), ("power@cuda", "power", "cuda", "cpu"),
                  ("bitplane@cuda", "bitplane", "cuda", "cpu")]
        for via_env in (False, True):
            for text, representation, device, finalize in forces:
                if via_env:
                    os.environ["TORBLES_MATH_PLAN_FORCE"] = text
                out = mp.run(req, force=None if via_env else text)
                ex, plan = out["explanation"], out["plan"]
                where = device if device == "cpu" else ("cuda-emulated" if emulated else "cuda")
                self.assertEqual((ex["representation"], ex["device"], ex["finalize"], ex["status"]),
                                 (representation, where, finalize, "ran"), (text, via_env))
                self.assertEqual(out["values"], expect, text)
                self.assertEqual((plan["forced"], plan["reason"], plan["choice"]["representation"]),
                                 (True, "forced", representation))
                others = [c for i, c in enumerate(plan["candidates"]) if i != plan["chosen"] and c["status"] != "rejected"]
                self.assertTrue(others and all(c["reason"] == "not forced" for c in others))
            os.environ.pop("TORBLES_MATH_PLAN_FORCE", None)
        os.environ["TORBLES_MATH_PLAN_FORCE"] = "verify@cuda, bitplane@cpu"     # one entry per request kind
        self.assertEqual(mp.run(req)["explanation"]["representation"], "bitplane")
        os.environ["TORBLES_MATH_PLAN_FORCE"] = "warp@cpu"
        with self.assertRaises(ValueError):
            mp.plan(req)
        os.environ.pop("TORBLES_MATH_PLAN_FORCE")
        ncodes = int(max(db["t"].decoded("x"))) + 1
        huge = req.with_polys([[0] * 8 + [1 << 62]])                         # no bound is provable for it
        refused = [("histogram", dict(budget=Budget(states=ncodes - 1)), req, AlgebraBudgetError),
                   ("histogram@cuda", dict(budget=Budget(states=ncodes - 1)), req, AlgebraBudgetError),
                   ("power", {}, req.with_polys([[1, 1]], "max"), CarrierError),
                   ("bitplane@cuda", {}, req.with_polys([[1, 1]], "min"), CarrierError),
                   ("empty", {}, req, UnsupportedError),
                   ("power", {}, huge, OverflowError),
                   ("scan", {}, huge, OverflowError),
                   ("verify", {}, req, UnsupportedError)]
        for text, kw, r, exc in refused:
            with patch.object(reg, "execute", side_effect=AssertionError("a refused plan executed")):
                with self.assertRaises(exc, msg=text) as ctx:
                    mp.run(r, force=text, **kw)
            plan = ctx.exception.plan
            self.assertEqual(plan["chosen"], None, text)
            self.assertIn(plan["reason"], ("forced representation refused", "no candidate matches the force"))
            if r is huge:                                 # unforced, nothing is admissible either
                with self.assertRaises(OverflowError):
                    mp.plan(r)
                continue
            # the same request unforced still has an admissible plan: the force was refused, not replaced
            self.assertIsNotNone(mp.plan(r, **kw)["choice"], text)
        # a plan admits SCAN / HISTOGRAM only with a proven int64 evaluation; when only the mod 2^128 bound holds the
        # plan computes the exact sum where a direct scan refuses
        wide = reg.Request(np.array([0, 1, 2, 3, 60000], dtype=np.uint32), [[0, 0, 0, 0, 0, 0, 1]],
                           rows=np.array([True, True, True, True, False]), offset=-30000)
        cands = {c["representation"]: c for c in mp.plan(wide)["candidates"]}
        self.assertEqual((cands["scan"]["refusal"]["kind"], cands["scan"]["refusal"]["operator"]), ("bound", "eval"))
        self.assertEqual(cands["histogram"]["status"], "rejected")
        with self.assertRaises(OverflowError):
            reg.execute(wide, "scan")
        out = mp.run(wide)
        self.assertIn(out["explanation"]["representation"], ("power", "bitplane"))
        self.assertEqual(out["values"], [sum((c - 30000) ** 6 for c in range(4))])
        with self.assertRaises(ValueError):
            mp.plan(req, force="histogram@gpu")
        # forced reuse of a cached state, and a stale one refused rather than rebuilt
        state = reg.build_state(req, "histogram")
        self.addCleanup(state.close)
        out = mp.run(req, state=state, force="histogram@reuse")
        self.assertEqual((out["explanation"]["reused"], out["explanation"]["rows_scanned"], out["values"]),
                         (True, 0, expect))
        moved = reg.Request(col, req.polys, rows=~np.array(mask))
        with self.assertRaises(StaleStateError):
            mp.run(moved, state=state, force="histogram@reuse")
        self.assertEqual(mp.run(moved, state=state)["values"], oracle(x, [not m for m in mask], req.polys))

    def test_fnd008_forced_disc_verifier_on_cpu_and_cuda(self):
        from torbles import math_disc
        from torbles.relationships import discover
        rng = np.random.default_rng(85)
        n = 1500
        xs = [int(v) for v in rng.integers(-60, 60, size=n)]
        ys = [int(v) for v in rng.integers(-30, 90, size=n)]
        s = [a + b for a, b in zip(xs, ys)]
        s[123] -= 2
        db = make({"x": xs, "y": ys, "s": s})
        self.addCleanup(db.close)
        cat = discover(db, "t", ["x", "y", "s"], modules=(), max_candidates=0)
        self.addCleanup(cat.close)
        expect = dict(checked=n, excluded=0, violations=1, witness=123, usable=False)
        emulated = math_gpu.device()["emulated"]
        for force, device in ((None, "cpu"), ("verify@cpu", "cpu"), ("verify@cuda", "cuda"), ("verify", "cpu")):
            out = mp.verify(cat, "sum", "s", "x", "y", force=force)
            self.assertEqual({k: out["evidence"][k] for k in expect}, expect, force)
            want = device if device == "cpu" else ("cuda-emulated" if emulated else "cuda")
            self.assertEqual((out["explanation"]["device"], out["explanation"]["module"]), (want, 4), force)
            self.assertEqual(out["plan"]["forced"], force is not None)
        os.environ["TORBLES_MATH_PLAN_FORCE"] = "scan@cpu,verify@cuda"
        out = mp.verify(cat, "sum", "s", "x", "y")
        self.assertTrue(out["explanation"]["device"].startswith("cuda"))
        self.assertEqual({k: out["evidence"][k] for k in expect}, expect)
        os.environ.pop("TORBLES_MATH_PLAN_FORCE")
        need = math_disc.device_bytes(cat, "sum", "s", "x", "y")
        with patch.object(reg, "verify", side_effect=AssertionError("a refused plan executed")):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mp.verify(cat, "sum", "s", "x", "y", force="verify@cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.plan["chosen"]), ("bytes", need, None))
        plan = mp.plan_verify(cat, "sum", "s", "x", "y", devices=("cpu", "cuda"), budget=Budget(bytes=need - 1))
        self.assertEqual(plan["choice"]["device"], "cpu")             # unforced, the CPU verifier is admissible
        self.assertEqual([c["status"] for c in plan["candidates"]], ["chosen", "rejected"])
        with self.assertRaises(UnsupportedError):                     # a polynomial force does not fit a verification
            mp.verify(cat, "sum", "s", "x", "y", force="scan")

    # -------------------------------------------------------------- reuse, refusals, engine predicates
    def test_fnd008_reuse_staleness_and_refused_plans(self):
        db, col, x, y = self.table(86, 2500)
        ids = db.query(db["t"]).where(db["t"].y >= 10).rows()
        mask = [v >= 10 for v in y]
        req = reg.Request(col, [[4, 0, 1]], row_ids=ids)
        for name in ("histogram", "power", "bitplane"):
            state = reg.build_state(req, name)
            self.addCleanup(state.close)
            plan = mp.plan(req, state=state)
            reuse = [c for c in plan["candidates"] if c["reuse"]]
            self.assertEqual((len(reuse), reuse[0]["representation"], reuse[0]["status"]), (1, name, "chosen"))
            self.assertEqual(reuse[0]["amounts"]["scan"], 0)
            out = mp.run(req, state=state)
            self.assertEqual((out["values"], out["explanation"]["reused"]), (oracle(x, mask, req.polys), True))
        db["t"].flip(1, 2)                                            # a mutation: the column and its states are stale
        with self.assertRaises(StaleStateError):
            reg.Request(col, req.polys)
        col2 = reg.Column(db["t"], "x")
        self.addCleanup(col2.close)
        req2 = reg.Request(col2, [[4, 0, 1]], row_ids=ids)
        plan = mp.plan(req2, state=state)
        reuse = [c for c in plan["candidates"] if c["reuse"]][0]
        self.assertEqual((reuse["status"], reuse["refusal"]["kind"]), ("rejected", "stale"))
        decoded = [None if not v else int(c) + col2.offset for c, v in
                   zip(db["t"].decoded("x"), db["t"].decoded(db["t"].nullable["x"]))]
        self.assertEqual(mp.run(req2, state=state)["values"], oracle(decoded, mask, req2.polys))
        # no admissible candidate: refused with the budget's reason, every candidate's evidence kept
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mp.plan(req2, budget=Budget(degree=1))
        plan = ctx.exception.plan
        self.assertEqual((plan["reason"], plan["chosen"], ctx.exception.resource), ("no admissible candidate", None, "degree"))
        self.assertTrue(all(c["status"] == "rejected" for c in plan["candidates"]))
        # the bounded search itself: iterations = candidates searched
        mp.plan(req2, budget=Budget(iterations=5))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mp.plan(req2, budget=Budget(iterations=4))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.plan["reason"]),
                         ("iterations", 5, "request or candidate search refused"))

    def test_fnd008_executed_plans_match_duckdb_on_engine_predicates(self):
        db, col, x, y = self.table(87, 5000)
        con = duckdb.connect()
        con.execute("CREATE TABLE t (x BIGINT, y BIGINT)")
        con.executemany("INSERT INTO t VALUES (?, ?)", list(zip(x, y)))
        t = db["t"]
        rng = np.random.default_rng(88)
        for k in (0, 1, 5, 12, 20):
            polys = [[int(c) for c in rng.integers(-5, 6, size=int(rng.integers(1, 6)))] for _ in range(2)]
            req = reg.Request(col, polys, row_ids=db.query(t).where(t.y < k).rows())
            for aggregate in ("sum", "min", "max"):
                r = req.with_polys(polys, aggregate)
                exprs = [" + ".join(f"CAST({c} AS HUGEINT)" + " * CAST(x AS HUGEINT)" * j for j, c in enumerate(p))
                         for p in polys]
                fn = {"sum": "COALESCE(SUM({}), 0)", "min": "MIN({})", "max": "MAX({})"}[aggregate]
                sql = ", ".join(fn.format(e) for e in exprs)
                want = [None if v is None else int(v) for v in
                        con.execute(f"SELECT {sql} FROM t WHERE y < {k} AND x IS NOT NULL").fetchone()]
                for sample_backend in ("cpu", "cuda"):
                    out = mp.run(r, devices=("cpu", "cuda"), sample_backend=sample_backend)
                    self.assertEqual(out["values"], want, (k, aggregate, sample_backend))
                    self.assertEqual(out["plan"]["selectivity"]["backend"], "cpu" if sample_backend == "cpu" else
                                     ("cuda-emulated" if math_gpu.device()["emulated"] else "cuda"))
        con.close()


if __name__ == "__main__":
    unittest.main()
