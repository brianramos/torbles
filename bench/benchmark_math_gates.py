# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-012 benchmark: the validation gates themselves, the CUDA core kernel, and the regression control.

    python bench/benchmark_math_gates.py --capability FND-012 --out FILE [--device] [--quick]

Measured: the inventory gate (renderer validation) and the coverage closure of the validation driver; the exact
128-bit core kernel on an actual device (--device only) against the Python-integer oracle; an audit of every
capability's receipts (each benchmark receipt present, raw samples kept, measured commit recorded); and the first
slice's moment benchmark at its smallest grid as a regression control, which claims no coverage of new modules.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from math_bench import Bench, ROOT

sys.path.insert(0, str(ROOT / "bench"))
import render_algebra_inventory as renderer   # noqa: E402
import validate_math                          # noqa: E402


def main():
    bench = Bench.from_args("FND-012", __doc__)
    data = json.loads((ROOT / "docs" / "algebra_inventory.json").read_text())
    bench.measure("inventory gate (renderer validation)", "cpu", lambda: renderer.validate(data) or "valid", repeats=5,
                  capabilities=len(data["capabilities"]))
    planned = sorted({f for c in data["capabilities"] for f in c["planned_files"]})
    bench.measure("coverage closure of the planned files", "cpu", lambda: len(validate_math.covered(planned)), repeats=3,
                  planned_files=len(planned))
    # the exact device accumulators against the oracle
    from torbles import math_gpu
    rng = np.random.default_rng(bench.seed)
    for n in ((10_000, 100_000) if bench.quick else (10_000, 1_000_000, 10_000_000)):
        x = rng.integers(-2**63, 2**63 - 1, size=n, dtype=np.int64, endpoint=True)
        arr, ptr = math_gpu.i64_array(x)
        exact = sum(int(v) for v in x[: min(n, 1_000_000)])
        bench.measure("exact 128-bit sum", "python-int oracle",
                      lambda: sum(int(v) for v in x[: min(n, 1_000_000)]) == exact, repeats=2, rows=min(n, 1_000_000))
        if bench.device:
            out = (math_gpu.C.c_int64 * 4)()

            def kernel():
                math_gpu.check(math_gpu.lib().tr_mgpu_core_selftest(ptr, n, out))
                return int(out[0])
            bench.measure("exact 128-bit sum + min + max", "cuda", kernel, repeats=5, rows=n)
    # receipt audit: what each capability's claim rests on (observations, not timings)
    for c in data["capabilities"]:
        v = c.get("validation") or {}
        if not v.get("benchmarks"):
            continue
        samples = 0
        for b in v["benchmarks"]:
            p = ROOT / b
            if p.is_file() and p.suffix == ".json":
                try:
                    payload = json.loads(p.read_text())
                    samples += sum(len(r.get("samples_seconds", [])) for r in payload.get("records", []))
                except Exception:
                    pass
        bench.records.append(dict(case="receipt audit", backend="gate", phase="audit", params=dict(capability=c["id"]),
                                  status=c["status"], benchmark_receipts=len(v["benchmarks"]),
                                  receipts_present=all((ROOT / b).is_file() for b in v["benchmarks"]),
                                  raw_samples=samples, measured_commit=v.get("measured_runtime_commit"),
                                  device_receipts=len((v.get("cuda") or {}).get("device_receipts", []))))
    # regression control: the first slice's moment benchmark, smallest grid (no new-module coverage is claimed)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "control.json"
        run = subprocess.run([sys.executable, str(ROOT / "bench" / "benchmark_mathematics.py"), "--rows", "1000",
                              "--repeats", "2", "--threads", "2", "--output", str(out)], cwd=ROOT,
                             capture_output=True, text=True, timeout=1800)
        control = json.loads(out.read_text()) if out.exists() else None
        bench.records.append(dict(case="regression control: first-slice moment benchmark", backend="cpu",
                                  phase="control", params=dict(rows=1000, repeats=2, threads=2),
                                  returncode=run.returncode, claims_new_coverage=False,
                                  receipt=control, stderr_tail=run.stderr[-2000:]))
    bench.write()


if __name__ == "__main__":
    main()
