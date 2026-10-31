# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-008 benchmark: the planner's choice against every candidate actually executed, on held-out query shapes and
volume scales, reporting misses as well as gains, with the raw batch distributions, setup, state bytes and backends.

    python bench/benchmark_math_planner.py --capability FND-008 --out FILE [--device] [--quick]

Training shapes (smaller volumes; code domains 16 / 4096, selectivities 2% / 60%, degrees 1 / 4) fit CPU cost
coefficients per representation: non-negative least squares of the measured native times on the candidates' work
amounts. Held-out shapes (domains 256 / 65536, selectivities 10% / 100%, degrees 2 / 6, larger volumes) were used for
nothing: each is planned with the registry's default coefficients and with the fitted ones, then every admissible
candidate is executed (results checked equal), and the receipt records the chosen candidate, the fastest, the regret
(chosen time / fastest time), whether it is a miss, and the gain over the fixed policy "always scan". Batches of
two or three polynomials (fusion) and shapes with a cached state (reuse; its construction is the shape's setup) are
included. Sampling (CPU; CUDA with --device) and planning are timed apart from execution. --device adds the CUDA
candidates to the plans and executes them on the device.
"""
import ctypes as C
import statistics
import time

import numpy as np
from scipy.optimize import nnls

from math_bench import Bench
from torbles import math_planner as mp, math_registry as reg
from torbles.math_budgets import Budget

COMPONENTS = reg.COST_COMPONENTS
FITTED = ("words", "scan", "chain", "combine", "construct", "reuse", "materialize", "cpu_final", "fixed")


def request(rng, rows, ncodes, selectivity, degree, batch):
    codes = rng.integers(0, ncodes, size=rows, dtype=np.int64).astype(np.uint32)
    mask = rng.random(rows) < selectivity
    polys = [[int(c) for c in rng.integers(-3, 4, size=degree + 1)] for _ in range(batch)]
    for p in polys:
        p[-1] = p[-1] or 1
    return reg.Request(codes, polys, rows=mask, offset=-(ncodes // 2))


def timed(fn, slow=0.25, repeats=3):
    """native samples (one when the first exceeds `slow` seconds)"""
    samples, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
        if samples[0] > slow:
            break
    return samples, out


def executor(req, cand, state, device):
    """the candidate's entry point, natively: (run, explanation)"""
    res, ex, b = reg._Result(), reg._Explain(), Budget()
    name = cand["representation"]
    rid = reg.REPRESENTATIONS[name]
    _, fn = reg.entry(name, cand["device"])
    handle = state._open() if cand["reuse"] else None
    final = 1 if cand["finalize"] == "gpu" else 0

    def run():
        rc = (fn(C.byref(req.raw), rid, handle, C.byref(b.raw), C.byref(res), C.byref(ex)) if cand["device"] == "cpu"
              else fn(C.byref(req.raw), rid, final, C.byref(b.raw), C.byref(res), C.byref(ex)))
        if rc:
            return None
        return tuple(reg._i128(res.value[j]) for j in range(len(req.polys)))
    return run, ex


def effective(cand):
    """the work amounts of the fused execution: shared components counted once"""
    k = cand["batch"]
    return [cand["amounts"][c] / (k if c in cand["shared"] else 1) for c in FITTED]


def shape_run(bench, rng, shape, devices, coefficient_sets, setname):
    rows, ncodes, sel, degree, batch, cached = shape
    req = request(rng, rows, ncodes, sel, degree, batch)
    p = dict(set=setname, rows=rows, ncodes=ncodes, selectivity=sel, degree=degree, batch=batch, cached=cached)
    state = None
    if cached:
        kind = "power" if degree >= 4 or ncodes > 4096 else "histogram"
        t0 = time.perf_counter()
        state = reg.build_state(req, kind)
        setup = time.perf_counter() - t0
        info = state.info
        bench.records.append(dict(case="setup: cached state", backend="cpu", phase="build", params=dict(p, state=kind),
                                  samples_seconds=[setup], first_seconds=setup, warm_median_seconds=setup,
                                  warm_min_seconds=setup, result=dict(state_bytes=int(info["bytes"]),
                                                                     elements=int(info["elements"]))))
    sample_samples, _ = timed(lambda: mp.sample(req), repeats=5)
    bench.records.append(dict(case="selectivity sample", backend="cpu", phase="plan", params=p,
                              samples_seconds=sample_samples, first_seconds=sample_samples[0],
                              warm_median_seconds=statistics.median(sample_samples),
                              warm_min_seconds=min(sample_samples)))
    if "cuda" in bench.backends():
        cs, _ = timed(lambda: mp.sample(req, backend="cuda"), repeats=5)
        bench.records.append(dict(case="selectivity sample", backend="cuda", phase="plan", params=p, samples_seconds=cs,
                                  first_seconds=cs[0], warm_median_seconds=statistics.median(cs), warm_min_seconds=min(cs)))
    plans = {}
    for label, coef in coefficient_sets.items():
        ps, plan = timed(lambda: mp.plan(req, devices=devices, state=state, coefficients=coef), repeats=5)
        plans[label] = plan
        bench.records.append(dict(case="planning", backend="cpu", phase="plan", params=dict(p, coefficients=label),
                                  samples_seconds=ps, first_seconds=ps[0], warm_median_seconds=statistics.median(ps),
                                  warm_min_seconds=min(ps)))
    base = plans["default"]
    measured, reference = [], None
    for i, cand in enumerate(base["candidates"]):
        if cand["status"] == "rejected":
            bench.records.append(dict(case="candidate", backend=cand["device"], phase="rejected",
                                      params=dict(p, index=i, representation=cand["representation"], reuse=cand["reuse"],
                                                  finalize=cand["finalize"]),
                                      refused=True, reason=cand["reason"], refusal=cand["refusal"], budget=cand["budget"],
                                      amounts=cand["amounts"], costs=cand["costs"], total=cand["total"]))
            continue
        run, ex = executor(req, cand, state, cand["device"])
        samples, out = timed(run)
        if out is None:                                  # refused at execution (a value outside int64)
            bench.records.append(dict(case="candidate", backend=cand["device"], phase="refused at execution",
                                      params=dict(p, index=i, representation=cand["representation"]),
                                      refused=True, code=int(ex.code), amounts=cand["amounts"], total=cand["total"]))
            continue
        if reference is None:
            reference = out
        elif out != reference:
            raise SystemExit(f"candidates disagree at {p}: {cand['representation']} {out} != {reference}")
        explain = reg._explanation(ex)
        rec = dict(case="candidate", backend=cand["device"], phase="first" if not cand["reuse"] else "warm",
                   params=dict(p, index=i, representation=cand["representation"], reuse=cand["reuse"],
                               finalize=cand["finalize"], path=explain["device"], state_bytes=explain["state_bytes"],
                               rows_scanned=explain["rows_scanned"]),
                   samples_seconds=samples, first_seconds=samples[0], warm_median_seconds=statistics.median(samples),
                   warm_min_seconds=min(samples), amounts=cand["amounts"], costs=cand["costs"], total=cand["total"])
        bench.records.append(rec)
        measured.append((i, statistics.median(samples), cand))
    if not measured:
        return None
    fastest = min(measured, key=lambda m: m[1])
    scan = next((m for m in measured if m[2]["representation"] == "scan" and m[2]["device"] == "cpu"), None)
    verdict = dict(params=p, fastest=fastest[0], fastest_seconds=fastest[1], measured=len(measured))
    for label, plan in plans.items():
        chosen = plan["chosen"]
        t = next((m[1] for m in measured if m[0] == chosen), None)
        verdict[label] = dict(chosen=chosen, representation=plan["choice"]["representation"],
                              device=plan["choice"]["device"], reuse=plan["choice"]["reuse"],
                              seconds=t, miss=chosen != fastest[0],
                              regret=None if t is None else t / fastest[1],
                              gain_over_scan=None if (t is None or scan is None) else scan[1] / t)
    bench.records.append(dict(case="plan verdict", backend="cpu", phase="evaluation", params=p, verdict=verdict))
    if state is not None:
        state.close()
    return dict(verdict=verdict, measured=measured)


def fit(training):
    """per representation: time (ns) ~ coefficients . effective amounts, non-negative least squares"""
    by = {}
    for run in training:
        for _, seconds, cand in run["measured"]:
            if cand["device"] != "cpu":
                continue
            by.setdefault(cand["representation"], []).append((effective(cand), seconds * 1e9))
    overrides, report = {}, {}
    for name, rows in by.items():
        A = np.array([r[0] for r in rows], dtype=np.float64)
        b = np.array([r[1] for r in rows], dtype=np.float64)
        scale = A.max(axis=0)
        scale[scale == 0] = 1
        coef, residual = nnls(A / scale, b)
        coef = coef / scale
        rates = dict(zip(FITTED, (float(c) for c in coef)))
        rates["fusion"] = 1.0
        overrides[(name, "cpu")] = rates
        report[name] = dict(rates=rates, samples=len(rows), residual_ns=float(residual))
    return overrides, report


def summary(bench, runs, label, setname):
    verdicts = [r["verdict"] for r in runs if r]
    misses = [v for v in verdicts if v[label]["miss"]]
    regrets = [v[label]["regret"] for v in verdicts if v[label]["regret"] is not None]
    gains = [v[label]["gain_over_scan"] for v in verdicts if v[label]["gain_over_scan"] is not None]
    out = dict(coefficients=label, shapes=len(verdicts), misses=len(misses),
               median_regret=statistics.median(regrets) if regrets else None, max_regret=max(regrets) if regrets else None,
               median_gain_over_scan=statistics.median(gains) if gains else None,
               min_gain_over_scan=min(gains) if gains else None,
               worst=[dict(v["params"], chosen=v[label]["representation"], regret=v[label]["regret"]) for v in
                      sorted(misses, key=lambda v: -(v[label]["regret"] or 0))[:5]])
    bench.records.append(dict(case=f"{setname} summary", backend="cpu", phase="evaluation",
                              params=dict(coefficients=label, set=setname),
                              result={k: v for k, v in out.items() if not isinstance(v, list)}, worst=out["worst"]))
    bench.note(f"{setname}, {label} coefficients: {out['misses']} misses of {out['shapes']} shapes; median regret "
               f"{out['median_regret']:.3f}, max {out['max_regret']:.3f}; median gain over always-scan "
               f"{out['median_gain_over_scan']:.3f}, min {out['min_gain_over_scan']:.3f}")
    return out


def zero_sample_cases(bench):
    rows = 64 * 400
    codes = (np.arange(rows) % 97).astype(np.uint32)
    mask = np.zeros(rows, dtype=bool)
    mask[64:70] = True                                   # rows in one unsampled word
    req = reg.Request(codes, [[1, 2, 3]], rows=mask)
    plan = mp.plan(req)
    out = mp.run(req)
    bench.records.append(dict(case="zero sampled match", backend="cpu", phase="evaluation", params=dict(rows=rows),
                              result=dict(zero_sample=plan["selectivity"]["zero_sample"],
                                          empty_proven=plan["selectivity"]["empty_proven"],
                                          chosen=plan["choice"]["representation"], value=out["values"][0],
                                          rows_scanned=out["explanation"]["rows_scanned"])))
    empty = reg.Request(codes, [[1, 2, 3]], rows=np.zeros(rows, dtype=bool))
    for exact in (False, True):
        ps, out = timed(lambda: mp.run(empty, exact=exact), repeats=5)
        bench.records.append(dict(case="empty selection", backend="cpu", phase="evaluation",
                                  params=dict(rows=rows, exact=exact), samples_seconds=ps, first_seconds=ps[0],
                                  warm_median_seconds=statistics.median(ps), warm_min_seconds=min(ps),
                                  result=dict(chosen=out["explanation"]["representation"],
                                              empty_proven=out["plan"]["selectivity"]["empty_proven"],
                                              rows_scanned=out["explanation"]["rows_scanned"])))


def main():
    bench = Bench.from_args("FND-008", __doc__)
    rng = np.random.default_rng(bench.seed)
    devices = tuple(bench.backends())
    if bench.quick:
        training = [(20_000, n, s, d, k, c) for n in (16, 4096) for s in (0.02, 0.6) for d in (1, 4) for k in (1,)
                    for c in (False, True)]
        held = [(100_000, n, s, d, k, c) for n in (256, 65536) for s in (0.1, 1.0) for d in (2, 6) for k in (1, 2)
                for c in (False,)]
    else:
        training = [(r, n, s, d, k, c) for r in (20_000, 60_000) for n in (16, 4096) for s in (0.02, 0.6)
                    for d in (1, 4) for k in (1, 3) for c in (False, True)]
        held = [(r, n, s, d, k, c) for r in (200_000, 600_000) for n in (256, 65536) for s in (0.1, 1.0)
                for d in (2, 6) for k in (1, 2) for c in (False, True)]
    train_runs = [shape_run(bench, rng, s, devices, {"default": None}, "training") for s in training]
    overrides, report = fit([r for r in train_runs if r])
    bench.records.append(dict(case="fitted coefficients", backend="cpu", phase="calibration", params={},
                              result={k: v["residual_ns"] for k, v in report.items()}, coefficients=report))
    fitted = mp.coefficients(overrides)
    held_runs = [shape_run(bench, rng, s, devices, {"default": None, "fitted": fitted}, "held-out") for s in held]
    summary(bench, train_runs, "default", "training")
    for label in ("default", "fitted"):
        summary(bench, held_runs, label, "held-out")
    zero_sample_cases(bench)
    bench.note("A miss is a shape where the chosen candidate is not the fastest measured; regret = chosen time / "
               "fastest time (1.0 = optimal); gain over always-scan = scan time / chosen time. The fitted coefficients "
               "come from the training shapes only; the held-out shapes are evaluated, never fitted.")
    bench.note("Times are native calls of the entry points the registry names (no Python formatting); costs are "
               "relative work units. CUDA candidates are planned and executed only with --device.")
    bench.write()


if __name__ == "__main__":
    main()
