# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Validation receipts of mathematical capabilities (docs/MATH_CONVENTIONS.md), and their inventory entries.

    python bench/validate_math.py CAP-ID [CAP-ID ...]            here: tests, sanitizers, nvcc, emulation, CPU benchmark
    python bench/validate_math.py --device CAP-ID [CAP-ID ...]   on a machine with an NVIDIA GPU (after `make gpu`):
                                                                 the tests and the benchmark on the device
    python bench/validate_math.py --device --all-ready           every capability whose CPU receipts are current

A capability's tests are the test methods of its planned test modules whose names contain its ID in lower case
without the dash (LIN-001 -> "lin001"). Receipts go to results/math/<ID>/. The covered sources are the planned files
and, transitively, the local headers they include and the torbles.math* modules they import; they must equal the
committed ones (each receipt names the commit it measured). Status: `in_progress` with CPU, compile and emulation
receipts; `validated` once actual-device receipts exist for the same sources.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "docs" / "algebra_inventory.json"


def sh(cmd, env=None, timeout=7200):
    started = dt.datetime.now(dt.timezone.utc)
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return dict(command=cmd if isinstance(cmd, str) else " ".join(cmd), returncode=run.returncode,
                seconds=(dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
                stdout_tail=run.stdout[-6000:], stderr_tail=run.stderr[-6000:])


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
_IMPORT = re.compile(r"^\s*(?:from\s+(?:\.|torbles\.?)\s*(math\w*)\s+import|from\s+(?:\.|torbles)\s+import\s+([\w ,]+)"
                     r"|import\s+torbles\.(math\w*))", re.M)


def covered(files):
    """the planned files plus the local headers they include and the torbles.math* modules they import"""
    seen, todo = [], list(files)
    while todo:
        f = todo.pop(0)
        if f in seen or not (ROOT / f).is_file():
            continue
        seen.append(f)
        text = (ROOT / f).read_text(errors="replace")
        base = Path(f).parent
        if f.endswith((".c", ".h", ".cu")):
            for inc in _INCLUDE.findall(text):
                cand = (base / inc).as_posix()
                if (ROOT / cand).is_file():
                    todo.append(cand)
        elif f.endswith(".py"):
            for a, b, c in _IMPORT.findall(text):
                names = [a or c] if (a or c) else [n.strip().split(" as ")[0] for n in b.split(",")]
                for n in names:
                    if n.startswith("math") and (ROOT / "torbles" / f"{n}.py").is_file():
                        todo.append(f"torbles/{n}.py")
    return seen


def machine():
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        cpu = platform.processor()
    libs = {p: sha(p) for p in ("torbles/libalg.so", "torbles/libmathgpu_emu.so", "torbles/libmathgpu.so")
            if (ROOT / p).exists()}
    import numpy
    return dict(host=platform.node(), cpu=cpu, cores=os.cpu_count(), python=platform.python_version(),
                numpy=numpy.__version__, libraries=libs,
                commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                when=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))


def env_for(device):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'torbles'}:{ROOT / 'tests'}:{ROOT / 'bench'}",
               TORBLES_EMU_SCRAMBLE="1")
    if device:
        env.pop("TORBLES_MATH_GPU", None); env["TORBLES_MATH_REQUIRE_DEVICE"] = "1"
    else:
        env["TORBLES_MATH_GPU"] = "emulate"; env.pop("TORBLES_MATH_REQUIRE_DEVICE", None)
    return env


def ran_ok(record):
    m = re.search(r"Ran (\d+) tests?", record["stdout_tail"] + record["stderr_tail"])
    record["tests_ran"] = int(m.group(1)) if m else 0
    return record["returncode"] == 0 and record["tests_ran"] > 0


def write(cap, name, payload):
    out = ROOT / "results" / "math" / cap / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return str(out.relative_to(ROOT))


def roles(cap, files):
    """(test modules with the capability's tests, CUDA kernels, benchmarks); other planned files are only covered"""
    key = cap.lower().replace("-", "")
    tests = [f for f in files if f.startswith("tests/test_") and f.endswith(".py")
             and re.search(rf"def test\w*{key}", (ROOT / f).read_text())]
    kernels = [f for f in files if f.endswith("_gpu.cu")]
    def receipt_writer(f):   # a benchmark with the receipt CLI (--capability ID --out FILE), not a first-slice script
        text = (ROOT / f).read_text()
        return "from math_bench import" in text or "--capability" in text
    benches = [f for f in files if f.startswith("bench/benchmark_") and f.endswith(".py") and receipt_writer(f)]
    if not tests or not kernels or not benches:
        raise SystemExit(f"{cap}: planned files need a test module, a CUDA kernel and a benchmark ({files})")
    return tests, kernels, benches


def preflight(cap, entry):
    """refusals before anything runs or is written: missing or uncommitted covered files, missing roles"""
    files = entry["planned_files"]
    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit(f"{cap}: planned files missing: {missing}")
    cover = covered(files)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *cover], cwd=ROOT, capture_output=True, text=True).stdout
    if dirty.strip():
        raise SystemExit(f"{cap}: commit the covered files first (each receipt names the commit it measured):\n{dirty}")
    return (*roles(cap, files), cover)


def validate(cap, entry, device, skip_bench):
    tests, kernels, benches, cover = preflight(cap, entry)
    key = cap.lower().replace("-", "")
    info = machine()
    validation = dict(entry.get("validation") or {})
    hashes = {f: sha(f) for f in cover}
    if device:
        old = validation.get("source_sha256") or {}
        if old != hashes or not (validation.get("cuda") or {}).get("emulated_receipts"):
            raise SystemExit(f"{cap}: the CPU receipts are missing or older than the sources; run "
                             f"`python bench/validate_math.py {cap}` (without --device) and commit first")
    validation["source_sha256"] = hashes
    validation["measured_runtime_commit"] = info["commit"]
    cuda = dict(validation.get("cuda") or {})
    cuda["kernels"] = kernels
    receipts, ok = [], True
    for test in tests:
        module = Path(test).stem
        record = sh([sys.executable, "-m", "unittest", "-k", key, module], env=env_for(device))
        passed = ran_ok(record)
        ok &= passed
        receipts.append(write(cap, f"{'device_' if device else ''}tests_{module}.json",
                              dict(capability=cap, machine=info, passed=passed, device=device,
                                   emulated=not device, scrambled_thread_order=not device, run=record)))
    if not device:
        sanitized = []
        for test in tests:
            san = sh(["bash", "tests/sanitize_math.sh", Path(test).name, key])
            san_ok = san["returncode"] == 0 and "ALL MATCH (sanitized)" in san["stdout_tail"]
            ok &= san_ok
            sanitized.append(write(cap, f"sanitized_{Path(test).stem}.json",
                                   dict(capability=cap, machine=info, passed=san_ok, run=san)))
        receipts += sanitized
        nvcc = shutil.which("nvcc")
        compiles = []
        for k in kernels:
            if nvcc is None:
                compiles.append(dict(kernel=k, compiled=False, reason="nvcc not installed")); ok = False; continue
            out = ROOT / "results" / "math" / cap / (Path(k).stem + ".o")
            out.parent.mkdir(parents=True, exist_ok=True)
            rec = sh([nvcc, "-O3", "-gencode", "arch=compute_80,code=sm_80", "-Xcompiler", "-fPIC", "-c", k, "-o", str(out)])
            if out.exists(): out.unlink()
            version = sh([nvcc, "--version"])["stdout_tail"].strip().splitlines()[-1]
            compiles.append(dict(kernel=k, compiled=rec["returncode"] == 0, nvcc=version,
                                 flags="-O3 -gencode arch=compute_80,code=sm_80",
                                 warnings=len(re.findall(r"\bwarning\b", rec["stderr_tail"])), run=rec))
            ok &= rec["returncode"] == 0
        cuda["compile_receipts"] = [write(cap, "nvcc.json", dict(capability=cap, machine=info, kernels=compiles))]
        # the test runs above executed every CUDA path in the emulation with scrambled thread order
        cuda["emulated_receipts"] = [write(cap, "emulated.json", dict(
            capability=cap, machine=info, passed=all(json.loads((ROOT / r).read_text())["passed"] for r in receipts),
            scrambled_thread_order=True, test_receipts=receipts,
            note="the capability's tests compare every emulated CUDA result with the CPU result"))]
        cuda.pop("device_receipts", None); cuda.pop("device", None)
    else:
        from torbles import math_gpu   # noqa: E402  (the receipts name the actual device)
        dev = math_gpu.device()
        if dev["emulated"]:
            raise SystemExit("--device needs libmathgpu.so on a machine with an NVIDIA GPU (run `make gpu`)")
        cuda["device_receipts"] = receipts
        cuda["device"] = dev["name"]
    benchmarks = [b for b in (validation.get("benchmarks") or []) if (ROOT / b).is_file()]
    if not skip_bench:
        for b in benches:
            host = re.sub(r"[^A-Za-z0-9]+", "_", info["host"])
            name = f"benchmark_{'device' if device else 'cpu'}_{Path(b).stem.replace('benchmark_', '')}_{host}.json"
            out = ROOT / "results" / "math" / cap / name
            rec = sh([sys.executable, b, "--capability", cap, "--out", str(out)] + (["--device"] if device else []),
                     env=env_for(device))
            ok &= rec["returncode"] == 0 and out.exists()
            if out.exists():
                path = str(out.relative_to(ROOT))
                if path not in benchmarks: benchmarks.append(path)
            else:
                write(cap, name.replace(".json", "_failed.json"), dict(capability=cap, machine=info, run=rec))
    if not device:
        validation["tests"] = receipts
    validation["benchmarks"] = benchmarks
    validation["cuda"] = cuda
    entry["validation"] = validation
    if not ok:
        entry["status"] = "in_progress"
        return False
    have_device = bool(cuda.get("device_receipts")) and all((ROOT / p).is_file() for p in cuda["device_receipts"])
    entry["status"] = "validated" if have_device and validation.get("tests") and benchmarks else "in_progress"
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capabilities", nargs="*")
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--all-ready", action="store_true", help="with --device: every capability whose CPU receipts are current")
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args()
    data = json.loads(INVENTORY.read_text())
    by = {c["id"]: c for c in data["capabilities"]}
    caps = list(a.capabilities)
    if a.all_ready:
        caps += [c["id"] for c in data["capabilities"] if c["status"] == "in_progress" and (c.get("validation") or {}).get("cuda")
                 and c["id"] not in caps
                 and all((ROOT / f).is_file() and sha(f) == h for f, h in c["validation"]["source_sha256"].items())]
    if not caps:
        raise SystemExit("name the capabilities to validate (or --device --all-ready)")
    results = {}
    for cap in caps:
        if cap not in by:
            raise SystemExit(f"unknown capability {cap}")
        results[cap] = validate(cap, by[cap], a.device, a.skip_bench)
        INVENTORY.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(f"{cap}: {'receipts written' if results[cap] else 'FAILED (see results/math/' + cap + ')'}; status {by[cap]['status']}", flush=True)
    subprocess.run([sys.executable, "bench/render_algebra_inventory.py"], cwd=ROOT, capture_output=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
