# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-013: state ownership, concurrency and the native ABI (torbles/math_handles.c, math_handles_gpu.cu).

Oracles: Python integers for every polynomial value and overflow flag (the canonical terms merged with a dict, each
term and the sum checked against the int128 range), Python counters for the expected handle values, cache counters
and serialized increments, and ctypes.sizeof / ctypes.alignment for the ABI. Every polynomial result is also computed
by the CUDA kernel (emulated here, scrambled thread order) and compared bit for bit with the CPU result. The
concurrency tests use Python threads (native calls release the GIL) and a native pthread harness built plain, with
ASAN+UBSAN and with TSAN (tests/math_handles_native_threads.c). Seeds are fixed.
"""
import ctypes as C
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from torbles import math_handles as mh
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.math_domains import DomainError
from torbles.mathematics import AlgebraBudgetError

ROOT = Path(__file__).resolve().parent.parent
M64 = (1 << 64) - 1
LO, HI = -(1 << 127), (1 << 127) - 1
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


# ------------------------------------------------------------------ the oracle (Python integers)
def canonical(terms):
    merged = {}
    for e, c in (terms.items() if isinstance(terms, dict) else terms):
        merged[tuple(e)] = merged.get(tuple(e), 0) + c
    return {e: c for e, c in sorted(merged.items()) if c}


def value_at(canon, x):
    """the exact value, or None when a term or the sum lies outside the int128 range"""
    total = 0
    for e, c in canon.items():
        term = c
        for xi, ei in zip(x, e):
            term *= xi ** ei
        if not LO <= term <= HI:
            return None
        total += term
    return total if LO <= total <= HI else None


def words(v):
    if v is None:
        return 0, 0
    lo = v & M64
    return (lo - (1 << 64) if lo >> 63 else lo), v >> 64


def oracle_raw(canon, points):
    vals = [value_at(canon, [int(c) for c in row]) for row in points]
    raw = np.array([words(v) for v in vals], dtype=np.int64).reshape(len(vals), 2)
    return vals, raw, np.array([v is None for v in vals], dtype=np.uint8)


def random_terms(rng, nvars, nterms, max_exp, coef_bits):
    out = []
    for _ in range(nterms):
        e = tuple(int(v) for v in rng.integers(0, max_exp + 1, size=nvars))
        if coef_bits >= 63:
            c = int(rng.integers(I64_MIN, I64_MAX, endpoint=True))
        else:
            c = int(rng.integers(-(1 << coef_bits), 1 << coef_bits))
        out.append((e, c))
    return out


def sparse_terms(rng, nvars, nterms, max_degree, coef_bits):
    """rows with a few nonzero exponents (total degree <= max_degree), as sparse polynomials in many variables"""
    out = []
    for _ in range(nterms):
        e = [0] * nvars
        for v in rng.integers(0, nvars, size=int(rng.integers(0, max_degree + 1))):
            e[int(v)] += 1
        out.append((tuple(e), int(rng.integers(-(1 << coef_bits), 1 << coef_bits))))
    return out


def random_points(rng, n, nvars):
    bits = rng.choice([2, 8, 20, 32, 44, 62], size=(n, nvars))
    pts = np.array([[int(rng.integers(-(1 << int(b)), 1 << int(b))) for b in row] for row in bits], dtype=np.int64)
    special = np.array([0, 1, -1, 2, -2, I64_MAX, I64_MIN], dtype=np.int64)
    mask = rng.random((n, nvars)) < 0.1
    pts[mask] = rng.choice(special, size=int(mask.sum()))
    return pts.reshape(n, nvars)


class Handles(unittest.TestCase):
    def table(self, *args, **kwargs):
        t = mh.HandleTable(*args, **kwargs)
        self.addCleanup(t.close)
        return t

    def same_on_both_backends(self, poly, points, expect_raw=None, expect_flags=None, budget=None):
        cpu = poly.evaluate_raw(points, "cpu", budget)
        cuda = poly.evaluate_raw(points, "cuda", budget)
        self.assertEqual(cuda["backend"], "cuda-emulated" if mh._gpu().device()["emulated"] else "cuda")
        self.assertTrue(np.array_equal(cpu["values"], cuda["values"]), "CUDA values differ from the CPU")
        self.assertTrue(np.array_equal(cpu["flags"], cuda["flags"]), "CUDA overflow flags differ from the CPU")
        self.assertEqual(cpu["overflowed"], cuda["overflowed"])
        self.assertEqual(cpu["overflowed"], int(cpu["flags"].sum()))
        if expect_raw is not None:
            self.assertTrue(np.array_equal(cpu["values"], expect_raw), "values differ from Python integers")
            self.assertTrue(np.array_equal(cpu["flags"], expect_flags), "flags differ from Python integers")
        return cpu

    def assert_clean(self, t, live=0):
        s = t.stats()
        self.assertEqual((s["live"], s["open"], s["deferred"]), (live, live, 0), s)
        self.assertEqual(s["retains"], s["releases"], s)
        if not live:
            self.assertEqual((s["payload_bytes"], s["allocations"]), (0, 1), s)
            self.assertEqual(s["destroyed"], s["inserted"], s)
            self.assertEqual(sum(s["live_kind"].values()), 0, s)
        return s

    # ------------------------------------------------------------------ the ABI of every ctypes boundary
    def test_fnd013_abi_registry_matches_every_ctypes_structure(self):
        report = mh.abi_check()
        names = {r["structure"] for r in report["checked"]}
        for structure in ("torbles.mathematics._Rational", "torbles.mathematics._Term",
                          "torbles.math_domains._DomainSpec", "torbles.math_domains._ModuleInfo",
                          "torbles.math_domains._ModuleTrace", "torbles.math_domains._CacheStats",
                          "torbles.math_budgets._Budget", "torbles.math_budgets._Need", "torbles.math_budgets._Refusal",
                          "torbles.math_budgets._Choice", "torbles.math_disc._View", "torbles.math_disc._Evidence",
                          "torbles.math_handles._Stats", "torbles.math_handles._Info", "torbles.math_handles._View",
                          "torbles.math_handles._I128", "torbles.math_handles._AbiEntry"):
            self.assertIn(structure, names)
        self.assertEqual(report["structures"], len(mh.python_structures()))
        self.assertTrue(all(r["match"] for r in report["checked"]))
        cpu = {e["name"]: e for e in mh.abi_entries("cpu")}
        cuda = {e["name"]: e for e in mh.abi_entries("cuda")}
        expected_sizes = dict(tr_math_rational=16, tr_math_term=24, tr_math_domain_spec=48, tr_math_module_info=72,
                              tr_math_module_trace=40, tr_math_state_stats=56, tr_math_budget=72, tr_math_need=72,
                              tr_math_refusal=24, tr_math_choice=72, tr_disc_view=104, tr_disc_evidence=72,
                              tr_math_i128=16, tr_math_interval=16, tr_math_handle_stats=216, tr_math_handle_info=40,
                              tr_math_ipoly_view=32, tr_math_abi_entry=64)
        self.assertEqual({k: v["size"] for k, v in cpu.items()}, expected_sizes)
        self.assertEqual(set(cuda), {"tr_math_budget", "tr_math_refusal", "tr_disc_view", "tr_disc_evidence",
                                     "tr_math_ipoly_view", "tr_math_i128", "tr_math_abi_entry"})
        for name, e in cuda.items():                    # one layout per struct across both libraries
            self.assertEqual((e["size"], e["align"], e["version"]), (cpu[name]["size"], cpu[name]["align"],
                                                                      cpu[name]["version"]), name)
        # every Structure that crosses into the CUDA library is compared there too
        crossing = {(r["native"], r["library"]) for r in report["checked"]}
        for name in cuda:
            self.assertIn((name, "cuda"), crossing)
        self.assertEqual(report["without_python_mirror"], ["tr_math_interval"])
        # the registry refuses a caller whose entry struct differs
        native = mh._api()
        self.assertEqual(native["abi"](None, 0, C.sizeof(mh._AbiEntry) - 1), -1)
        self.assertEqual(native["abi"](None, -1, C.sizeof(mh._AbiEntry)), -1)
        self.assertEqual(native["abi"](None, 0, C.sizeof(mh._AbiEntry)), len(cpu))   # count only
        self.assertEqual(mh._gpu().lib().tr_mgpu_handles_abi(None, 0, C.sizeof(mh._AbiEntry) + 8), -1)

    def test_fnd013_abi_check_fails_loudly_on_a_mismatch(self):
        class Short(C.Structure):              # tr_math_refusal is int32, int32, uint64, uint64 (24 bytes)
            _native_ = "tr_math_refusal"
            _fields_ = [("resource", C.c_int32), ("refused", C.c_int32), ("need", C.c_uint64)]

        class Reordered(C.Structure):          # same size, different alignment
            _native_ = "tr_math_rational"
            _pack_ = 4
            _fields_ = [("a", C.c_int32), ("b", C.c_int64), ("c", C.c_int32)]

        class Versioned(C.Structure):          # right layout, wrong ABI version
            _native_ = "tr_math_refusal"
            _abi_version_ = 2
            _fields_ = [("resource", C.c_int32), ("refused", C.c_int32), ("need", C.c_uint64), ("limit", C.c_uint64)]

        class Unregistered(C.Structure):
            _fields_ = [("x", C.c_int32)]

        class Unknown(C.Structure):
            _native_ = "tr_no_such_struct"
            _fields_ = [("x", C.c_int32)]

        real = mh.python_structures()
        for bad, needle in ((Short, "Python {'size': 16"), (Reordered, "'align': 4"), (Versioned, "'version': 2"),
                            (Unregistered, "without a native registration"), (Unknown, "in no native ABI registry")):
            with mock.patch.object(mh, "python_structures", return_value=real + [bad]):
                with self.assertRaises(mh.AbiError) as ctx:
                    mh.abi_check()
            self.assertIn(needle, str(ctx.exception), bad.__name__)
            self.assertIn(bad.__qualname__, str(ctx.exception))
        # the check at first use (non-strict) flags a mismatch but not an unregistered or unknown Structure
        cpu = {"libalg": mh.abi_entries("cpu")}
        self.assertTrue(mh._compare([Short], cpu, strict=False)[0])
        self.assertFalse(mh._compare([Unregistered, Unknown], cpu, strict=False)[0])
        # registries that disagree, or a struct registered twice with two layouts
        entry = dict(name="tr_x", size=8, align=8, version=1, reserved=0)
        problems, _ = mh._compare([], {"libalg": [entry], "cuda": [dict(entry, size=16)]})
        self.assertTrue(any("disagree" in p for p in problems))
        problems, _ = mh._compare([], {"libalg": [entry, dict(entry, align=4)]})
        self.assertTrue(any("registered twice" in p for p in problems))
        self.assertFalse(mh._compare([], {"libalg": [entry, dict(entry)]})[0])     # an identical duplicate is fine

    # ------------------------------------------------------------------ handle identity and deterministic errors
    def test_fnd013_handles_combine_slot_and_generation(self):
        t = self.table(capacity=4)
        a, b = t.polynomial({(1,): 1}), t.polynomial({(1,): 2})
        self.assertEqual((a.value, b.value), ((1 << 32) | 0, (1 << 32) | 1))
        self.assertEqual((a.slot, a.generation, b.slot, b.generation), (0, 1, 1, 1))
        self.assertEqual(a.close(), "freed")
        c = t.polynomial({(1,): 3})                         # the freed slot is reused with the next generation
        self.assertEqual((c.slot, c.generation), (0, 2))
        d = t.polynomial({(1,): 4})
        self.assertEqual((d.slot, d.generation), (2, 1))
        info = c.info()
        self.assertEqual({k: info[k] for k in ("kind", "mutable", "refs", "slot", "generation")},
                         dict(kind="polynomial", mutable=False, refs=0, slot=0, generation=2))
        self.assertGreater(info["bytes"], 0)
        self.assertEqual((c.close(), b.close()), ("freed", "freed"))
        e, f = t.polynomial({(1,): 5}), t.polynomial({(1,): 6})   # last freed, first reused
        self.assertEqual([(h.slot, h.generation) for h in (e, f)], [(1, 2), (0, 3)])
        self.assertEqual(e.evaluate([[7]])["values"], [35])
        self.assertIsInstance(t.handle(f.value), mh.Polynomial)

    def test_fnd013_closed_and_never_issued_handles_are_deterministic_errors(self):
        t = self.table(capacity=4)
        native = mh._api()
        p = t.polynomial({(1,): 5})
        v = p.value
        self.assertEqual(p.close(), "freed")
        q = t.polynomial({(1,): 9})                         # the same slot, generation 2
        self.assertEqual(q.slot, p.slot)
        for _ in range(3):
            with self.assertRaises(mh.ClosedHandleError):
                t.handle(v)
            for op in (lambda: p.evaluate([[1]]), lambda: p.evaluate([[1]], backend="cuda"), lambda: p.terms,
                       lambda: p.info(), lambda: p.device_bytes(1)):
                with self.assertRaises(mh.ClosedHandleError):
                    op()
            with self.assertRaises(mh.ClosedHandleError):
                with p.retained():
                    pass
            self.assertEqual(p.close(), "closed")
            self.assertTrue(p.closed)
        self.assertEqual(q.evaluate([[2]])["values"], [18])  # the new occupant is unaffected, the old id never hits it
        never = [0, 1, 5, (1 << 32) | 3, (1 << 32) | 4, (3 << 32) | 0, ((1 << 32) - 1) << 32 | 1, M64]
        for _ in range(3):
            for value in never:                         # ValueError, never ClosedHandleError (a RuntimeError)
                with self.assertRaises(ValueError):
                    t.handle(value)
        t_ptr = t._enter()
        try:
            out = C.c_void_p()
            for _ in range(2):
                self.assertEqual(native["close"](t_ptr, v), mh.CLOSED)
                self.assertEqual(native["retain"](t_ptr, v, 0, C.byref(out)), mh.CLOSED)
                self.assertEqual(native["info"](t_ptr, v, C.byref(mh._Info())), mh.CLOSED)
                self.assertEqual(native["release"](t_ptr, v), -1)            # no reference to release
                for value in never:
                    self.assertEqual(native["close"](t_ptr, value), -1)
                    self.assertEqual(native["retain"](t_ptr, value, 0, C.byref(out)), -1)
                    self.assertEqual(native["release"](t_ptr, value), -1)
            self.assertEqual(native["release"](t_ptr, q.value), -1)          # open, but not retained
            # a handle of another kind is refused the same way
            d = t.domain("snapshot", b"rows", nvars=1, ndata=1, universe_rows=3)
            self.assertEqual(native["retain"](t_ptr, d.value, mh.KINDS["polynomial"], C.byref(out)), -1)
            self.assertEqual(native["retain"](t_ptr, d.value, 0, C.byref(out)), 0)
            self.assertEqual(native["release"](t_ptr, d.value), 0)
        finally:
            t._exit()
        with self.assertRaises(ValueError):
            mh.Polynomial(t, d.value, owned=False)
        # the refusals are counted natively, one per refused call
        before = t.stats()
        for _ in range(5):
            with self.assertRaises(mh.ClosedHandleError):
                t.handle(v)
            with self.assertRaises(ValueError):
                t.handle(never[3])
        after = t.stats()
        self.assertEqual((after["refused_closed"] - before["refused_closed"],
                          after["refused_invalid"] - before["refused_invalid"]), (5, 5))
        self.assertEqual((q.close(), d.close()), ("freed", "freed"))
        self.assert_clean(t)

    def test_fnd013_generation_limit_retires_slots(self):
        t = self.table(capacity=2, generation_limit=2)
        values = []
        for k in range(4):
            h = t.polynomial({(0,): k + 1}, nvars=1)
            values.append(h.value)
            self.assertEqual(h.close(), "freed")
        self.assertEqual(values, [(1 << 32) | 0, (2 << 32) | 0, (1 << 32) | 1, (2 << 32) | 1])
        s = t.stats()
        self.assertEqual((s["retired"], s["free_slots"], s["live"]), (2, 0, 0))
        with self.assertRaises(AlgebraBudgetError) as ctx:              # every slot used up: never reused
            t.polynomial({(0,): 1}, nvars=1)
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("states", 3, 2))
        for v in values:
            with self.assertRaises(mh.ClosedHandleError):
                t.handle(v)
        for v in ((3 << 32) | 0, (3 << 32) | 1):
            with self.assertRaises(ValueError):
                t.handle(v)
        with self.assertRaises(ValueError):
            mh.HandleTable(capacity=2, generation_limit=0)
        self.assert_clean(t)

    # ------------------------------------------------------------------ acceptance: lifetimes do not leak or reuse
    def test_fnd013_repeated_create_release_returns_every_allocation(self):
        rng = np.random.default_rng(1306)
        t = self.table(capacity=8)
        shapes = [canonical(random_terms(rng, 2, 12, 5, 30)) for _ in range(5)]
        points = random_points(rng, 64, 2)
        expected = [oracle_raw(canon, points) for canon in shapes]
        issued = set()
        for i in range(1500):
            k = i % len(shapes)
            p = t.polynomial(shapes[k], nvars=2)
            issued.add(p.value)
            backend = "cuda" if i % 5 == 0 else "cpu"
            r = p.evaluate_raw(points, backend)
            self.assertTrue(np.array_equal(r["values"], expected[k][1]) and np.array_equal(r["flags"], expected[k][2]))
            self.assertEqual(p.close(), "freed")
            if i % 3 == 0:                               # the first-slice states through the same table
                d = t.domain("snapshot", f"scope-{i}", nvars=1, ndata=1, universe_rows=100)
                b = t.bank({(0,): 4, (1,): i, (2,): i * i + 3})
                c = t.state_cache(capacity=2)
                self.assertEqual(c.put(d, b"basis", b), "stored")
                g = c.get(d, b"basis", 1)
                self.assertEqual((g.moment((1,)), g.moment((2,))), (i, i * i + 3))
                for h in (g, c, b, d):
                    self.assertEqual(h.close(), "freed")
        self.assertEqual(len(issued), 1500)              # no handle value was ever issued twice
        s = self.assert_clean(t)
        self.assertGreaterEqual(s["inserted"], 1500 + 4 * 500)
        self.assertLessEqual(s["peak_payload_bytes"], 8 * 1024 * 1024 + 4096)   # one cache bound + small states
        # shared immutable evaluations of one handle: the same answer every time, every reference returned
        p = t.polynomial(shapes[0], nvars=2)
        for i in range(300):
            r = p.evaluate_raw(points, "cuda" if i % 7 == 0 else "cpu")
            self.assertTrue(np.array_equal(r["values"], expected[0][1]))
        self.assertEqual(p.close(), "freed")
        self.assert_clean(t)

    def test_fnd013_close_during_a_live_operation_is_deferred(self):
        rng = np.random.default_rng(1307)
        t = self.table(capacity=8)
        canon = canonical(random_terms(rng, 3, 20, 4, 40))
        p = t.polynomial(canon, nvars=3)
        points = random_points(rng, 200, 3)
        _, raw, flags = oracle_raw(canon, points)
        native, gpu = mh._api(), mh._gpu()
        with p.retained() as view:
            self.assertEqual(p.info()["refs"], 1)
            self.assertEqual(p.close(), "deferred")
            s = t.stats()
            self.assertEqual((s["live"], s["open"], s["deferred"], s["live_kind"]["polynomial"]), (1, 0, 1, 1))
            self.assertGreater(s["payload_bytes"], 0)
            for op in (lambda: p.evaluate(points), lambda: p.evaluate(points, backend="cuda"), lambda: p.info(),
                       lambda: p.terms):
                with self.assertRaises(mh.ClosedHandleError):
                    op()
            self.assertEqual(p.close(), "closed")
            # the retained reference still reads the immutable terms: CPU and the CUDA kernel (no use after free)
            values = np.zeros((len(points), 2), dtype=np.int64)
            got_flags = np.zeros(len(points), dtype=np.uint8)
            over = C.c_uint64()
            vptr = values.ctypes.data_as(C.POINTER(mh._I128))
            self.assertEqual(native["view_evaluate"](C.byref(view), points.ctypes.data, len(points), vptr,
                                                     got_flags.ctypes.data, C.byref(over)), 0)
            self.assertTrue(np.array_equal(values, raw) and np.array_equal(got_flags, flags))
            values[:] = 0
            gpu.run("tr_mgpu_handles_evaluate", C.byref(view), points.ctypes.data, len(points), vptr,
                    got_flags.ctypes.data, C.byref(over), C.byref(Budget().raw), C.byref(mh._Refusal()))
            self.assertTrue(np.array_equal(values, raw) and np.array_equal(got_flags, flags))
            self.assertEqual(over.value, int(flags.sum()))
        s = self.assert_clean(t)                          # the block's release freed it
        self.assertEqual((s["deferred_closes"], s["closes"], s["destroyed"]), (1, 1, 1))
        # the same for a mutable state cache: retained, closed (deferred), refused, freed at the release
        c = t.state_cache()
        with c.retained():
            self.assertEqual(c.close(), "deferred")
            with self.assertRaises(mh.ClosedHandleError):
                c.stats
        self.assert_clean(t)

    def test_fnd013_concurrent_reads_while_another_thread_closes(self):
        rng = np.random.default_rng(1308)
        points = random_points(rng, 600, 3)
        shapes = [canonical(random_terms(rng, 3, 25, 4, 36)) for _ in range(6)]
        expected = [oracle_raw(canon, points)[1:] for canon in shapes]
        deferred_seen = 0
        for attempt in range(12):
            t = self.table(capacity=64)
            polys = [t.polynomial(canon, nvars=3) for canon in shapes]
            closed_after = [False] * len(polys)
            failures, outcomes = [], []
            start, stop = threading.Barrier(5), threading.Event()

            def reader(r):
                seen = [False] * len(polys)
                start.wait()
                rounds, final = 0, False
                while not final:
                    final = stop.is_set()               # the last round starts after every close returned
                    for i, p in enumerate(polys):
                        was_closed = closed_after[i]
                        try:
                            got = p.evaluate_raw(points, "cuda" if (rounds + i + r) % 4 == 0 else "cpu")
                        except mh.ClosedHandleError:
                            seen[i] = True
                            continue
                        except Exception as exc:   # noqa: BLE001  (any other answer is a failure)
                            failures.append(f"reader {r}: {exc!r}")
                            continue
                        if seen[i] or was_closed:
                            failures.append(f"reader {r}: polynomial {i} answered after its close")
                        if not (np.array_equal(got["values"], expected[i][0])
                                and np.array_equal(got["flags"], expected[i][1])):
                            failures.append(f"reader {r}: polynomial {i} gave another state's values")
                    rounds += 1
                if not all(seen):
                    failures.append(f"reader {r}: a closed handle was not refused")

            def closer():
                start.wait()
                time.sleep(0.01)
                for i, p in enumerate(polys):
                    outcomes.append(p.close())
                    closed_after[i] = True
                    time.sleep(0.004)

            def churn():                                 # reuses the freed slots with other polynomials
                start.wait()
                for k in range(300):
                    q = t.polynomial({(0, 0, 0): 10_000 + k, (1, 1, 1): 1}, nvars=3)
                    if q.close() != "freed":
                        failures.append("churn: unexpected close")

            threads = [threading.Thread(target=reader, args=(r,)) for r in range(3)]
            threads += [threading.Thread(target=closer), threading.Thread(target=churn)]
            for th in threads:
                th.start()
            threads[3].join()
            threads[4].join()
            stop.set()
            for th in threads[:3]:
                th.join()
            self.assertEqual(failures, [])
            self.assertTrue(set(outcomes) <= {"deferred", "freed"}, outcomes)
            self.assertEqual(len(outcomes), len(polys))
            self.assert_clean(t)
            deferred_seen += outcomes.count("deferred")
            if deferred_seen:
                break
        self.assertGreater(deferred_seen, 0, "no close ever overlapped a live evaluation")

    def test_fnd013_serialized_state_cache_under_threads(self):
        t = self.table(capacity=512)
        cache = t.state_cache(capacity=16)
        domains = [t.domain("snapshot", f"scope-{j}", nvars=1, ndata=1, universe_rows=1000) for j in range(4)]
        banks = [t.bank({(0,): 10 + j, (1,): 3 * j, (2,): 7 * j + 1}) for j in range(4)]
        failures, workers, iterations = [], 4, 150

        def work(w, stop_on_close):
            closed = False
            for i in range(iterations):
                j = (w + i) % 4
                try:
                    if cache.put(domains[j], b"basis", banks[j]) != "stored":
                        failures.append("put declined")
                    got = cache.get(domains[j], b"basis", 1)
                except mh.ClosedHandleError:
                    closed = True
                    continue
                if closed and stop_on_close:
                    failures.append(f"worker {w}: the cache answered after its close")
                if got is None:
                    failures.append(f"worker {w}: lost entry {j}")
                    continue
                if (got.moment((0,)), got.moment((1,)), got.moment((2,))) != (10 + j, 3 * j, 7 * j + 1):
                    failures.append(f"worker {w}: entry {j} holds another state")
                got.close()

        threads = [threading.Thread(target=work, args=(w, False)) for w in range(workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(failures, [])
        stats = cache.stats                               # no update was lost: the native mutex serialized them
        n = workers * iterations
        self.assertEqual((stats["puts"], stats["hits"], stats["misses"]), (n, n, 0))
        self.assertEqual((stats["entries"], stats["evictions"]), (4, 0))
        # the same work while another thread closes the cache
        outcome = []
        threads = [threading.Thread(target=work, args=(w, True)) for w in range(workers)]

        def closer():
            time.sleep(0.005)
            outcome.append(cache.close())
        threads.append(threading.Thread(target=closer))
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(failures, [])
        self.assertIn(outcome[0], ("freed", "deferred"))
        with self.assertRaises(mh.ClosedHandleError):
            cache.stats
        for h in domains + banks:
            self.assertEqual(h.close(), "freed")
        self.assert_clean(t)

    def test_fnd013_native_threads_plain_asan_and_tsan(self):
        cc = shutil.which(os.environ.get("CC", "gcc")) or shutil.which("cc")
        if not cc:
            self.skipTest("no C compiler")
        sources = [str(ROOT / p) for p in ("tests/math_handles_native_threads.c", "torbles/math_handles.c",
                                           "torbles/mathematics.c", "torbles/math_domain.c", "torbles/math_state.c")]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LD_PRELOAD", "ASAN", "UBSAN", "TSAN"))}
        variants = (("plain", ["-O2", "-Wall", "-Wextra"], {}),
                    ("asan", ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined",
                              "-fno-sanitize-recover=undefined"], dict(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                                                                       UBSAN_OPTIONS="halt_on_error=1")),
                    ("tsan", ["-O1", "-g", "-fsanitize=thread"], dict(TSAN_OPTIONS="halt_on_error=1")))
        with tempfile.TemporaryDirectory() as tmp:
            for name, flags, extra in variants:
                with self.subTest(variant=name):
                    exe = str(Path(tmp) / f"threads_{name}")
                    build = subprocess.run([cc, "-std=c11", *flags, "-I", str(ROOT / "torbles"), *sources,
                                            "-lpthread", "-o", exe], env=env, capture_output=True, text=True,
                                           timeout=300)
                    if build.returncode and name == "tsan":
                        self.skipTest("this compiler cannot build ThreadSanitizer programs")
                    self.assertEqual(build.returncode, 0, build.stderr[-3000:])
                    run = subprocess.run([exe], env=dict(env, **extra), capture_output=True, text=True, timeout=300)
                    if name == "tsan" and "unexpected memory mapping" in run.stderr:
                        self.skipTest("ThreadSanitizer cannot map its shadow memory on this kernel")
                    self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-4000:])
                    self.assertIn("native handle threads: OK", run.stdout)
                    self.assertNotIn("WARNING: ThreadSanitizer", run.stderr)
                    self.assertNotIn("ERROR: AddressSanitizer", run.stderr)

    # ------------------------------------------------------------------ the immutable polynomial state
    def test_fnd013_polynomial_values_are_exact_on_cpu_and_cuda(self):
        rng = np.random.default_rng(1311)
        t = self.table(capacity=16)
        flagged = exact = refused = evaluated = 0
        for trial in range(40):
            nvars = int(rng.choice([1, 2, 3, 4, 8, 9, 13, 17]))         # rows of one word, several, and tails
            if nvars > 4 and trial % 2:
                terms = sparse_terms(rng, nvars, int(rng.integers(1, 40)), int(rng.integers(1, 9)),
                                     int(rng.choice([8, 30, 62])))
            else:
                terms = random_terms(rng, nvars, int(rng.integers(1, 24)), int(rng.integers(1, 7)),
                                     int(rng.choice([8, 30, 50, 63])))
            terms += [(e, -(c // 2)) for e, c in terms[:3]]     # repeated exponents are merged exactly
            merged = {}
            for e, c in terms:
                merged[e] = merged.get(e, 0) + c
            if any(not I64_MIN <= c <= I64_MAX for c in merged.values()):
                with self.assertRaises(OverflowError):         # a merged coefficient past int64 is refused
                    t.polynomial(terms, nvars=nvars)
                refused += 1
                continue
            canon = canonical(terms)
            p = t.polynomial(terms, nvars=nvars)
            evaluated += 1
            self.assertEqual(p.terms, canon)
            self.assertEqual(p.term_count, len(canon))
            self.assertEqual(p.degree, max((sum(e) for e in canon), default=0))
            points = random_points(rng, 250, nvars)
            vals, raw, flags = oracle_raw(canon, points)
            self.same_on_both_backends(p, points, raw, flags)
            result = p.evaluate(points, backend="cuda")
            self.assertEqual(result["values"], vals)
            self.assertEqual(result["overflow"], [v is None for v in vals])
            flagged += sum(v is None for v in vals)
            exact += sum(v is not None and not I64_MIN <= v <= I64_MAX for v in vals)
            self.assertEqual(p.close(), "freed")
        self.assertGreater(flagged, 0)                          # the grid reaches both sides of the int128 range
        self.assertGreater(exact, 0)                            # and exact values past 64 bits
        self.assertGreaterEqual(evaluated, 25)
        self.assert_clean(t)

    def test_fnd013_overflow_flags_at_the_int128_boundaries(self):
        t = self.table(capacity=32)
        wide = Budget(degree=400)                               # the cases go up to total degree 328
        high = {(0, 0, 0, 128, 0, 0, 0, 0, 0, 200): 3}          # exponents >= 128 in a row of two words
        cases = [  # (terms, nvars, point)
            ({(1,): 1}, 1, [I64_MAX]), ({(1,): 1}, 1, [I64_MIN]), ({(2,): 1}, 1, [I64_MIN]),
            ({(2,): -2}, 1, [I64_MIN]),                         # -2^127 exactly: fits
            ({(2,): 2}, 1, [I64_MIN]),                          # 2^127: outside
            ({(64,): -(1 << 63)}, 1, [2]), ({(64,): -(1 << 63)}, 1, [-2]),    # -2^127
            ({(65,): 1 << 62}, 1, [2]),                         # 2^127: outside
            ({(64,): I64_MAX}, 1, [2]),                         # 2^127 - 2^64
            ({(64, 0): 1 << 62, (0, 64): 1 << 62}, 2, [2, 2]),            # sum 2^127: outside
            ({(64, 0): 1 << 62, (0, 64): 1 << 62, (0, 0): -1}, 2, [2, 2]),  # 2^127 - 1: exact in any term order
            ({(64, 0): -(1 << 62), (0, 64): -(1 << 62)}, 2, [2, 2]),        # -2^127: exact
            ({(64, 0): -(1 << 62), (0, 64): -(1 << 62), (0, 0): -1}, 2, [2, 2]),  # outside
            ({(65, 0): 1 << 62, (0, 0): -1}, 2, [2, 0]),        # a term outside int128 is flagged even if the sum fits
            ({(1, 200): 5}, 2, [0, 1 << 40]),                   # a zero factor: exactly 0, never an overflow
            ({(1, 200): 5}, 2, [1, 1 << 40]),
            ({(0,): 7}, 1, [0]), ({(1,): 7}, 1, [0]),           # x^0 = 1 also at 0
            ({(255,): 3}, 1, [-1]), ({(255,): 3}, 1, [1]), ({(255,): 3}, 1, [2]),
            ({(1,): I64_MAX}, 1, [I64_MAX]), ({(3,): 1}, 1, [I64_MAX]),
            ({(1, 1): I64_MIN}, 2, [-1, 1]),                    # 2^63: exact past int64
            ({(2, 1): I64_MIN}, 2, [I64_MIN, -1]),
            (high, 10, [5, 5, 5, -1, 5, 5, 5, 5, 5, 1]), (high, 10, [5, 5, 5, -1, 5, 5, 5, 5, 5, -1]),
            (high, 10, [5, 5, 5, 2, 5, 5, 5, 5, 5, 1]),
            (high, 10, [5, 5, 5, 2, 5, 5, 5, 5, 5, 0]),         # past 2^128, then a zero factor in the row's tail
            ({(0,) * 8 + (1,): -7, (1,) + (0,) * 8: 2}, 9, [3, 0, 0, 0, 0, 0, 0, 0, 4]),
        ]
        for terms, nvars, point in cases:
            canon = canonical(terms)
            p = t.polynomial(terms, nvars=nvars, budget=wide)
            vals, raw, flags = oracle_raw(canon, np.array([point], dtype=np.int64))
            got = self.same_on_both_backends(p, np.array([point], dtype=np.int64), raw, flags)
            self.assertEqual(p.evaluate([point])["values"], vals, (terms, point))
            self.assertEqual(got["overflowed"], int(vals[0] is None))
            p.close()
        # the expected flags of the table above, written out (the oracle is checked too)
        outcomes = []
        for terms, nvars, point in cases:
            outcomes.append(value_at(canonical(terms), point) is None)
        self.assertEqual(outcomes, [False, False, False, False, True, False, False, True, False, True, False, False,
                                    True, True, False, True, False, False, False, False, True, False, True, False,
                                    True, False, False, True, False, False])

    def test_fnd013_canonical_terms_and_construction_refusals(self):
        t = self.table(capacity=16)
        p = t.polynomial([((1, 0), 3), ((1, 0), 4), ((0, 1), 5), ((0, 1), -5), ((0, 0), 0)], nvars=2)
        self.assertEqual((p.terms, p.term_count, p.degree), ({(1, 0): 7}, 1, 1))
        q = t.polynomial({(2, 1): 1, (0, 3): -2, (1, 1): 4, (0, 0): 9})
        self.assertEqual(list(q.terms), sorted(q.terms))                    # canonical (lexicographic) order
        r = t.polynomial([((1,), I64_MAX), ((1,), 1), ((1,), -1)], nvars=1)  # merged exactly in 128 bits
        self.assertEqual(r.terms, {(1,): I64_MAX})
        with self.assertRaises(OverflowError):                              # a merged coefficient past int64
            t.polynomial([((1,), I64_MAX), ((1,), 1)], nvars=1)
        arrays = t.polynomial_from_arrays(np.array([[1, 0], [1, 0], [0, 2]]), np.array([3, 4, -1]))
        self.assertEqual(arrays.terms, {(0, 2): -1, (1, 0): 7})
        for e, c, error in (([[256]], [1], ValueError), ([[-1]], [1], ValueError), ([[1]], [1.5], TypeError),
                            ([[1]], np.array([1 << 63], dtype=np.uint64), OverflowError), ([[1], [2]], [1], ValueError),
                            ([1, 2], [1, 2], ValueError)):
            with self.assertRaises(error):
                t.polynomial_from_arrays(np.asarray(e), np.asarray(c))
        arrays.close()
        zero = t.polynomial({}, nvars=2)
        self.assertEqual(zero.evaluate([[3, 4], [5, 6]])["values"], [0, 0])
        constant = t.polynomial({(): -4}, nvars=0)
        self.assertEqual(constant.evaluate(np.zeros((3, 0), dtype=np.int64))["values"], [-4, -4, -4])
        self.same_on_both_backends(constant, np.zeros((5, 0), dtype=np.int64))
        self.assertEqual(constant.evaluate(np.zeros((0, 0), dtype=np.int64))["values"], [])
        for bad, error in ((lambda: t.polynomial({(1,): 1 << 63}), OverflowError),
                           (lambda: t.polynomial({(1,): 1.5}), TypeError),
                           (lambda: t.polynomial({(256,): 1}), ValueError),
                           (lambda: t.polynomial({(-1,): 1}), ValueError),
                           (lambda: t.polynomial([((1, 2), 1), ((1,), 1)]), ValueError),
                           (lambda: q.evaluate([[1.5, 2]]), TypeError),
                           (lambda: q.evaluate([[1, 2, 3]]), ValueError),
                           (lambda: q.evaluate([[1 << 63, 2]]), OverflowError),
                           (lambda: q.evaluate([[1, 2]], backend="tpu"), ValueError)):
            with self.assertRaises(error):
                bad()
        # native refusals: the points' coordinates must match, and the representation limit of the variables
        native = mh._api()
        values, flags, over = np.zeros((1, 2), dtype=np.int64), np.zeros(1, dtype=np.uint8), C.c_uint64()
        point = np.array([[1, 2]], dtype=np.int64)
        with t._use() as ptr:
            self.assertEqual(native["ipoly_evaluate"](ptr, q.value, point.ctypes.data, 1, 3,
                                                      values.ctypes.data_as(C.POINTER(mh._I128)), flags.ctypes.data,
                                                      C.byref(over), None, None), -1)
            out = C.c_uint64()
            wide = Budget(variables=1 << 20)
            self.assertEqual(native["ipoly_create"](ptr, 70_000, 0, None, None, C.byref(wide.raw), C.byref(out),
                                                    None), -12)
        with self.assertRaises(UnsupportedError):
            mh._check(-12)
        for h in (p, q, r, zero, constant):
            h.close()
        self.assert_clean(t)

    def test_fnd013_budgets_at_the_limit_and_one_past(self):
        t = self.table(capacity=8)
        terms = {(i, 1): i + 1 for i in range(20)}                       # 20 terms, degree 20, 2 variables
        for resource, limit in (("support", 20), ("degree", 20), ("variables", 2)):
            p = t.polynomial(terms, budget=Budget(**{resource: limit}))
            p.close()
            with self.assertRaises(AlgebraBudgetError) as ctx:
                t.polynomial(terms, budget=Budget(**{resource: limit - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit),
                             (resource, limit, limit - 1))
        need = int(mh._api()["ipoly_bytes"](2, 20))
        self.assertEqual(need, 32 + 20 * (8 + 2) + 20 * 16)
        t.polynomial(terms, budget=Budget(bytes=need)).close()
        with self.assertRaises(AlgebraBudgetError) as ctx:
            t.polynomial(terms, budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
        # evaluation: output = points, the same on both backends; the device copies are budgeted too
        p = t.polynomial(terms)
        points = np.arange(200, dtype=np.int64).reshape(100, 2)
        self.same_on_both_backends(p, points, budget=Budget(output=100))
        for backend in ("cpu", "cuda"):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                p.evaluate(points, backend=backend, budget=Budget(output=99))
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("output", 100, 99))
        device = p.device_bytes(100)
        self.assertEqual(device, 20 * (2 + 8) + 100 * (2 * 8 + 17) + 8)
        p.evaluate(points, backend="cuda", budget=Budget(bytes=device))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            p.evaluate(points, backend="cuda", budget=Budget(bytes=device - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", device))
        p.evaluate(points, backend="cpu", budget=Budget(bytes=0))       # the CPU path copies nothing
        p.close()
        # a table: states = capacity, bytes = its fixed allocation; a full table refuses the next state
        size = int(mh._api()["table_bytes"](3))
        self.table(3, budget=Budget(states=3, bytes=size))
        for budget, resource, need_ in ((Budget(states=2), "states", 3), (Budget(bytes=size - 1), "bytes", size)):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mh.HandleTable(3, budget=budget)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need_))
        small = self.table(3)
        held = [small.polynomial({(0,): k}, nvars=1) for k in range(3)]
        for make in (lambda: small.polynomial({(0,): 9}, nvars=1), lambda: small.state_cache(),
                     lambda: small.bank({(): 1}, ndata=0), lambda: small.domain("formal", b"ctx", nvars=8, ndata=0,
                                                                                  carrier="rational")):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                make()
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("states", 4, 3))
        held[1].close()
        held[1] = small.polynomial({(0,): 9}, nvars=1)
        self.assertEqual(held[1].evaluate([[5]])["values"], [9])
        self.assert_clean(t)

    # ------------------------------------------------------------------ heterogeneous states and table lifetime
    def test_fnd013_heterogeneous_states_share_one_table(self):
        t = self.table(capacity=32)
        rows = t.domain("snapshot", b"rows\0A", nvars=2, ndata=2, revision=7, universe_rows=100)
        formal = t.domain("formal", b"context", nvars=8, ndata=2, carrier="rational", origin=rows)
        self.assertEqual(rows.spec["revision"], 7)
        self.assertEqual((formal.spec["kind"], formal.spec["ndata"]), ("formal", 2))
        with self.assertRaises(DomainError):                       # first-slice semantics: an origin needs ndata
            t.domain("formal", b"bad", nvars=8, ndata=1, carrier="rational", origin=rows)
        with self.assertRaises(ValueError):
            t.domain("formal", b"bad", nvars=9)
        rows.close()                                               # the formal domain owns a copy of its origin
        self.assertEqual(formal.spec["kind"], "formal")
        source = t.domain("snapshot", b"scope", nvars=1, ndata=1, universe_rows=50)
        bank = t.bank({(0,): 5, (1,): -7, (2,): 29})
        self.assertEqual([bank.moment(e) for e in ((0,), (1,), (2,))], [5, -7, 29])
        with self.assertRaises(ValueError):
            t.bank({(0,): 5, (1,): -7})                            # incomplete degree-two basis
        poly = t.polynomial({(1, 1): 2})
        cache = t.state_cache(capacity=2)
        self.assertEqual(cache.put(source, "basis", bank), "stored")
        hit = cache.get(source, "basis", 1)
        self.assertEqual(hit.moment((2,)), 29)
        self.assertIsNone(cache.get(source, "other", 1))
        self.assertEqual(t.stats()["live_kind"], {"other": 0, "polynomial": 1, "domain": 2, "bank": 2,
                                                  "state cache": 1})
        self.assertIsInstance(t.handle(cache.value), mh.StateCache)
        self.assertTrue(cache.info()["mutable"] and not bank.info()["mutable"])
        with self.assertRaises(TypeError):
            cache.put(bank, "basis", source)
        native = mh._api()
        with t._use() as ptr:                                      # kinds are checked natively as well
            self.assertEqual(native["cache_put"](ptr, cache.value, bank.value, b"basis", 5, source.value), -1)
            self.assertEqual(native["cache_put"](ptr, poly.value, source.value, b"basis", 5, bank.value), -1)
            self.assertEqual(native["bank_moment"](ptr, poly.value, None, C.byref(C.c_int64())), -1)
        # a domain closed while an operation holds it: the operation completes, later ones are refused
        with source.retained():
            self.assertEqual(source.close(), "deferred")
            with self.assertRaises(mh.ClosedHandleError):
                cache.put(source, "basis", bank)
            with self.assertRaises(mh.ClosedHandleError):
                cache.get(source, "basis", 1)
        cache.close()
        self.assertEqual(hit.moment((1,)), -7)                     # a returned bank is independent of the cache
        for h in (hit, bank, poly, formal):
            self.assertEqual(h.close(), "freed")
        self.assert_clean(t)

    def test_fnd013_table_close_waits_for_operations_and_refuses_inside_one(self):
        t = mh.HandleTable(capacity=8)
        p = t.polynomial({(1,): 3})
        inside, order = threading.Event(), []

        def holder():
            with p.retained():
                inside.set()
                time.sleep(0.2)
                order.append("released")
        th = threading.Thread(target=holder)
        th.start()
        inside.wait()
        with self.assertRaises(RuntimeError):                    # this thread is not inside an operation, but...
            with p.retained():
                t.close()                                         # ...here it is
        native = mh._api()
        ptr = t._ptr
        self.assertEqual(native["free"](ptr), -1)                 # the native table refuses while a handle is held
        t.close()                                                 # waits for the holder's operation to end
        order.append("closed")
        th.join()
        self.assertEqual(order, ["released", "closed"])
        self.assertTrue(t.closed)
        for op in (lambda: p.evaluate([[1]]), t.stats, lambda: t.polynomial({(1,): 1}), lambda: t.handle(p.value)):
            with self.assertRaises(mh.ClosedHandleError):
                op()
        self.assertEqual(p.close(), "closed")
        t.close()                                                 # idempotent
        with mh.HandleTable(capacity=2) as u:                     # a context manager frees every live payload
            u.polynomial({(1,): 1})
            u.state_cache()
        self.assertTrue(u.closed)

    def test_fnd013_existing_ownership_suites_still_pass(self):
        for name, minimum in (("test_math_domains", 8), ("test_math_state", 11)):
            suite = unittest.TestLoader().loadTestsFromName(name)
            result = unittest.TestResult()
            suite.run(result)
            self.assertTrue(result.wasSuccessful(), [str(f[1])[-400:] for f in result.failures + result.errors])
            self.assertGreaterEqual(result.testsRun, minimum, name)


if __name__ == "__main__":
    unittest.main()
