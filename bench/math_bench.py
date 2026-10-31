# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The receipt format shared by the mathematical benchmarks (docs/MATH_CONVENTIONS.md, "Benchmarks").

    from math_bench import Bench
    bench = Bench.from_args("FND-010", __doc__)       # --capability ID --out FILE [--device] [--seed N] [--quick]
    bench.measure("pre-pass", "cpu", fn, repeats=5, rows=n)   # every raw sample kept
    bench.threshold("histogram", admitted=dict(cells=65536), refused=dict(cells=131072))
    bench.write()

Phases are kept apart: "build" (state construction), "first" (first execution) and "warm" (reuse); a measurement
records the first sample separately from the warm samples. Backends: "cpu" (native), "cuda" (an actual device:
only with --device, which refuses the emulation), and named baselines ("numpy", "duckdb", "reconstruction", ...).
An emulated CUDA run is never timed as CUDA.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "torbles")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def machine():
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    libs = {str(p.relative_to(ROOT)): _sha(p) for p in sorted((ROOT / "torbles").glob("lib*.so"))}
    import numpy
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return dict(host=platform.node(), cpu=cpu, cores=os.cpu_count(), python=platform.python_version(),
                numpy=numpy.__version__, libraries=libs, commit=commit, dirty_worktree=dirty,
                when=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))


class Bench:
    def __init__(self, capability, out, device=False, seed=2026, quick=False):
        self.capability, self.out, self.device, self.seed, self.quick = capability, Path(out), device, seed, quick
        self.records, self.thresholds, self.notes = [], [], []
        self.gpu = None
        if device:
            from torbles import math_gpu
            self.gpu = math_gpu.require_device()        # refuses the emulation
        self.started = time.perf_counter()

    @classmethod
    def from_args(cls, capability, description=""):
        ap = argparse.ArgumentParser(description=(description or "").split("\n\n")[0])
        ap.add_argument("--capability", default=capability)
        ap.add_argument("--out", required=True)
        ap.add_argument("--device", action="store_true", help="time the CUDA path on an actual device")
        ap.add_argument("--seed", type=int, default=2026)
        ap.add_argument("--quick", action="store_true", help="the smallest grid (smoke run)")
        a = ap.parse_args()
        return cls(a.capability, a.out, a.device, a.seed, a.quick)

    def backends(self):
        """the backends timed in this run: the CPU always, CUDA only on an actual device"""
        return ("cpu", "cuda") if self.device else ("cpu",)

    def measure(self, case, backend, fn, repeats=5, phase="run", **params):
        """time fn() repeats times; the first sample is the first execution, the others warm reuse"""
        if backend == "cuda" and not self.device:
            raise RuntimeError("the CUDA path is timed only with --device (an emulated run is not a CUDA timing)")
        samples, result = [], None
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            result = fn()
            samples.append(time.perf_counter() - t0)
        warm = samples[1:] or samples
        rec = dict(case=case, backend=backend, phase=phase, params=params, samples_seconds=samples,
                   first_seconds=samples[0], warm_median_seconds=statistics.median(warm),
                   warm_min_seconds=min(warm))
        if isinstance(result, (int, float, str, bool)) or result is None:
            rec["result"] = result
        elif isinstance(result, dict):
            rec["result"] = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool, type(None)))}
        self.records.append(rec)
        return result

    def refusal(self, case, backend, fn, **params):
        """run fn() expecting a budget refusal; record the refusal (resource, need, limit) instead of a timing"""
        from torbles.mathematics import AlgebraBudgetError
        t0 = time.perf_counter()
        try:
            fn()
        except AlgebraBudgetError as e:
            rec = dict(case=case, backend=backend, phase="refusal", params=params, refused=True,
                       refusal_seconds=time.perf_counter() - t0,
                       resource=getattr(e, "resource", None), need=getattr(e, "need", None),
                       limit=getattr(e, "limit", None), message=str(e))
            self.records.append(rec)
            return rec
        rec = dict(case=case, backend=backend, phase="refusal", params=params, refused=False)
        self.records.append(rec)
        return rec

    def threshold(self, grid, admitted, refused, **detail):
        """the largest measured admitted and the smallest measured refused point of a grid (never extrapolated)"""
        self.thresholds.append(dict(grid=grid, largest_admitted=admitted, smallest_refused=refused, **detail))

    def note(self, text):
        self.notes.append(text)

    def write(self):
        payload = dict(capability=self.capability, seed=self.seed, quick=self.quick, device_run=self.device,
                       device=self.gpu, machine=machine(), seconds=time.perf_counter() - self.started,
                       records=self.records, thresholds=self.thresholds, notes=self.notes,
                       argv=sys.argv[1:])
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
        print(f"{self.capability}: {len(self.records)} measurements -> {self.out}")
