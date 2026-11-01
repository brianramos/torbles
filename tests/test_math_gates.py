# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-012: the differential validation and benchmark gates of the mathematical capabilities.

The gates themselves are tested: the inventory renderer refuses a `validated` claim without test, benchmark and
actual-device CUDA receipts or with a stale source hash; the validation driver refuses missing or uncommitted
sources and test modules without the capability's tests; device runs refuse the emulation; emulated runs are labeled
as such; native error codes map to distinct exceptions. The CUDA core kernel (exact 128-bit accumulation, min, max)
is checked against Python integers, and the first slice's independent native validators still pass.
"""
import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import render_algebra_inventory as renderer          # noqa: E402
import validate_math                                   # noqa: E402
from torbles import math_errors, math_gpu              # noqa: E402
from torbles.mathematics import AlgebraBudgetError     # noqa: E402


def inventory():
    return json.loads((ROOT / "docs" / "algebra_inventory.json").read_text())


class Gates(unittest.TestCase):
    def test_fnd012_core_kernel_is_exact_against_python_integers(self):
        rng = np.random.default_rng(12)
        cases = [np.array([], dtype=np.int64), np.array([5], dtype=np.int64),
                 np.array([2**63 - 1] * 1000, dtype=np.int64), np.array([-2**63] * 1000, dtype=np.int64),
                 rng.integers(-2**63, 2**63 - 1, size=100_000, dtype=np.int64, endpoint=True),
                 rng.integers(2**62, 2**63 - 1, size=50_000, dtype=np.int64)]
        lib = math_gpu.lib()
        for x in cases:
            arr, ptr = math_gpu.i64_array(x)
            out = (math_gpu.C.c_int64 * 4)()
            math_gpu.check(lib.tr_mgpu_core_selftest(ptr, len(arr), out))
            lo, hi, mn, mx = (int(v) for v in out)
            total = (hi << 64) + (lo & ((1 << 64) - 1))
            self.assertEqual(total, sum(int(v) for v in x))
            self.assertEqual((mn, mx), (min(map(int, x)), max(map(int, x))) if len(x) else (2**63 - 1, -2**63))

    def test_fnd012_renderer_refuses_unbacked_or_stale_validation(self):
        data = inventory()
        renderer.validate(data)                                     # the committed inventory passes
        target = next(c for c in data["capabilities"] if c["status"] != "validated")   # any capability not yet validated
        for change, needle in (
                (lambda c: c.update(status="validated", validation=None), "test and benchmark receipts"),
                (lambda c: c.update(status="validated", validation=dict(tests=["tests/test_math_gates.py"])),
                 "test and benchmark receipts"),
                (lambda c: c.update(status="validated", validation=dict(
                    tests=["tests/test_math_gates.py"], benchmarks=["results/math/none.json"])), "missing validation receipt")):
            d = copy.deepcopy(data)
            change(next(c for c in d["capabilities"] if c["id"] == target["id"]))
            with self.assertRaisesRegex(ValueError, needle):
                renderer.validate(d)
        validated = [c for c in data["capabilities"] if c["status"] == "validated"]
        self.assertTrue(validated)
        d = copy.deepcopy(data)
        entry = next(c for c in d["capabilities"] if c["id"] == validated[0]["id"])
        path = next(iter(entry["validation"]["source_sha256"]))
        entry["validation"]["source_sha256"][path] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from validated receipt"):
            renderer.validate(d)
        d = copy.deepcopy(data)
        entry = next(c for c in d["capabilities"] if c["id"] == validated[0]["id"])
        entry["validation"].pop("cuda_standard", None); entry["validation"].pop("cuda", None)
        with self.assertRaisesRegex(ValueError, "actual-device CUDA receipt"):
            renderer.validate(d)
        d = copy.deepcopy(data)
        entry = next(c for c in d["capabilities"] if c["id"] == validated[0]["id"])
        entry["validation"]["cuda"] = dict(kernels=["torbles/math_core_gpu.cu"], device_receipts=[])
        with self.assertRaisesRegex(ValueError, "actual-device receipts"):
            renderer.validate(d)

    def test_fnd012_driver_refuses_before_writing_receipts(self):
        fake = dict(id="FND-012", planned_files=["torbles/does_not_exist.c"])
        with self.assertRaisesRegex(SystemExit, "planned files missing"):
            validate_math.preflight("FND-012", fake)
        with self.assertRaisesRegex(SystemExit, "need a test module, a CUDA kernel and a benchmark"):
            validate_math.roles("FND-012", ["tests/test_math_gates.py", "bench/benchmark_math_gates.py"])
        with self.assertRaisesRegex(SystemExit, "need a test module"):
            # a test module without the capability's tests is not its evidence
            validate_math.roles("LIN-001", ["tests/test_math_gates.py", "torbles/math_core_gpu.cu",
                                            "bench/benchmark_math_gates.py"])
        record = dict(returncode=0, stdout_tail="", stderr_tail="\nRan 0 tests in 0.000s\n\nNO TESTS RAN")
        self.assertFalse(validate_math.ran_ok(record))
        record = dict(returncode=0, stdout_tail="", stderr_tail="\nRan 3 tests in 0.1s\n\nOK")
        self.assertTrue(validate_math.ran_ok(record))
        self.assertFalse(validate_math.ran_ok(dict(record, returncode=1)))
        cover = validate_math.covered(["torbles/math_disc.c", "torbles/math_disc.py"])
        for header in ("torbles/math_disc.h", "torbles/math_num.h", "torbles/math_budgets.h", "torbles/math_errors.py"):
            self.assertIn(header, cover)

    def test_fnd012_device_runs_refuse_the_emulation_and_emulated_runs_are_labeled(self):
        env = dict(os.environ, TORBLES_MATH_GPU="emulate", PYTHONPATH=f"{ROOT}:{ROOT / 'torbles'}:{ROOT / 'bench'}")
        code = ("from torbles import math_gpu, math_errors\n"
                "info = math_gpu.device(); assert info['emulated'], info\n"
                "try:\n    math_gpu.require_device()\nexcept math_errors.DeviceError as e:\n"
                "    print('refused:', type(e).__name__)\n")
        run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        self.assertIn("refused: DeviceError", run.stdout, run.stderr)
        bench = subprocess.run([sys.executable, str(ROOT / "bench" / "benchmark_math_gates.py"), "--capability",
                                "FND-012", "--out", os.devnull, "--device"], env=env, capture_output=True, text=True,
                               timeout=300)
        self.assertNotEqual(bench.returncode, 0)
        self.assertIn("DeviceError", bench.stderr)
        required = subprocess.run([sys.executable, "-c", "from torbles import math_budgets as mb\n"
                                   "mb.count_cells([1, 2], 4, backend='cuda')"],
                                  env=dict(env, TORBLES_MATH_REQUIRE_DEVICE="1"), capture_output=True, text=True,
                                  timeout=120)
        self.assertNotEqual(required.returncode, 0)
        self.assertIn("DeviceError", required.stderr)
        if math_gpu.device()["emulated"]:
            from torbles import math_disc  # noqa: F401  (labels are checked in test_math_disc as well)
            self.assertEqual(math_gpu.device()["name"], "CPU emulation of the CUDA kernels")

    def test_fnd012_error_codes_map_to_distinct_exceptions(self):
        seen = {}
        for code in (-1, -2, -3, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17):
            with self.assertRaises(Exception) as ctx:
                math_errors.check(code)
            seen[code] = type(ctx.exception)
        self.assertEqual(len(set(seen.values())), len(seen))
        self.assertIs(seen[-5], AlgebraBudgetError)
        self.assertEqual(math_errors.check(0), 0)
        self.assertEqual(math_errors.check(7), 7)
        with self.assertRaises(ValueError):
            math_errors.check(-99)
        err = math_errors.budget_error(2, 129, 128, "product")
        self.assertEqual((err.resource, err.need, err.limit), ("support", 129, 128))
        self.assertIn("129 support", str(err))

    def test_fnd012_first_slice_native_validators_still_pass(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:   # the committed first-slice receipts are left untouched
            for script in ("validate_mathematics_native.py",):
                receipt = Path(tmp) / (script + ".json")
                run = subprocess.run([sys.executable, str(ROOT / "tests" / script), "--receipt", str(receipt)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=1800)
                self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
                self.assertTrue(json.loads(receipt.read_text())["all_passed"])

    def test_fnd012_inventory_view_is_current(self):
        run = subprocess.run([sys.executable, str(ROOT / "bench" / "render_algebra_inventory.py"), "--check"],
                             cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
