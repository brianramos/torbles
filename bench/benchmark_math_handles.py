# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-013 benchmark: handle and state lifetime memory under repeated workloads, and concurrent immutable evaluation
versus serialized cache updates, without changing the process priority or CPU affinity.

    python bench/benchmark_math_handles.py --capability FND-013 --out FILE [--device] [--quick]

Lifetime: repeated create / use / close cycles of every state kind (sparse integer polynomials of 16, 256 and 4096
terms in 16 variables, domains, moment banks, state caches with a put and a get) and deferred closes (retain, close,
release), in batches; after every batch the table's native counters (live handles, declared payload bytes, the
module's own live allocations, peak payload bytes) and the process RSS are recorded, so any growth shows. Baseline:
the first-slice Python-owned objects (torbles.math_domains.Domain, MomentCache) created and closed directly.
Construction, first execution and warm reuse: table and polynomial construction ("build"), then the first and the
warm evaluations of the polynomials at integer points (CPU native; CUDA with --device). Baselines: NumPy int64 (exact
here only because every value of the grid fits int64; it detects no overflow) and exact Python integers (a sample).
Concurrency: T = 1, 2 and 4 Python threads (native calls release the GIL) evaluating one shared immutable polynomial,
versus T threads putting and getting through one shared state cache (serialized by its native mutex); the total work
is fixed per configuration. The nice value and CPU affinity are read before and after, recorded, and never changed.
Refusal thresholds (measured, never extrapolated): polynomial support and evaluation output at a stated budget, and
the table capacity.
"""
import os
import threading
import time

import numpy as np

from math_bench import Bench
from torbles import math_handles as mh
from torbles.math_budgets import Budget
from torbles.math_domains import Domain, MomentCache

NVARS = 16


def rss_bytes():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def scheduling():
    """the process's scheduling state, read only"""
    return dict(nice=os.getpriority(os.PRIO_PROCESS, 0),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)


def random_polynomial(rng, nterms):
    """nterms distinct exponent vectors of total degree <= 4 in NVARS variables, coefficients below 2^20 (so every
    value at points |x| <= 100 fits int64: 4096 * 2^20 * 100^4 < 2^63)"""
    seen, rows = set(), []
    while len(rows) < nterms:
        e = np.zeros(NVARS, dtype=np.uint8)
        for v in rng.integers(0, NVARS, size=int(rng.integers(0, 5))):
            e[v] += 1
        key = e.tobytes()
        if key not in seen:
            seen.add(key)
            rows.append(e)
    exps = np.array(rows, dtype=np.uint8)
    coefs = rng.integers(-(1 << 20), 1 << 20, size=nterms, dtype=np.int64)
    coefs[coefs == 0] = 1
    return exps, coefs


def numpy_values(exps, coefs, points, chunk=32):
    """NumPy int64 baseline (no overflow detection; exact for this grid)"""
    out = np.empty(len(points), dtype=np.int64)
    e = exps.astype(np.int64)
    for s in range(0, len(points), chunk):
        x = points[s:s + chunk]
        out[s:s + chunk] = (x[:, None, :] ** e[None, :, :]).prod(axis=2) @ coefs
    return out


def python_values(exps, coefs, points):
    out = []
    for row in points.tolist():
        total = 0
        for e, c in zip(exps.tolist(), coefs.tolist()):
            term = c
            for xi, ei in zip(row, e):
                if ei:
                    term *= xi ** ei
            total += term
        out.append(total)
    return out


def words(values):
    return [(int(hi) << 64) + (int(lo) & ((1 << 64) - 1)) for lo, hi in values.tolist()]


def evaluation(bench, rng):
    npoints = 2000 if bench.quick else 10_000
    points = rng.integers(-100, 101, size=(npoints, NVARS), dtype=np.int64)
    sizes = (16, 256, 1024) if bench.quick else (16, 256, 4096)
    made = []
    bench.measure("table construction", "cpu", lambda: made.append(mh.HandleTable(1024)) or made[-1].capacity,
                  repeats=5, phase="build", capacity=1024)
    table = made[0]
    for extra in made[1:]:
        extra.close()
    for nterms in sizes:
        exps, coefs = random_polynomial(rng, nterms)
        p = dict(terms=nterms, variables=NVARS, points=npoints)
        built = []
        bench.measure("polynomial construction", "cpu",
                      lambda: built.append(table.polynomial_from_arrays(exps, coefs)) or built[-1].term_count,
                      repeats=5, phase="build", **p)
        poly = built[0]
        for extra in built[1:]:
            extra.close()
        results = {}
        for backend in bench.backends():
            results[backend] = bench.measure("evaluation", backend, lambda: poly.evaluate_raw(points, backend),
                                             repeats=6, phase="run", **p)
        cpu = results["cpu"]
        if cpu["overflowed"]:
            raise SystemExit("the benchmark grid must not overflow (its baselines are int64)")
        if "cuda" in results and not (np.array_equal(results["cuda"]["values"], cpu["values"])
                                      and np.array_equal(results["cuda"]["flags"], cpu["flags"])):
            raise SystemExit(f"CUDA and CPU values differ at {nterms} terms")
        ref = bench.measure("evaluation", "numpy", lambda: numpy_values(exps, coefs, points), repeats=3,
                            phase="run", **p)
        if not np.array_equal(ref, cpu["values"][:, 0]) or np.any(cpu["values"][:, 1] != (cpu["values"][:, 0] >> 63)):
            raise SystemExit(f"NumPy and native values differ at {nterms} terms")
        sample = points[:max(20, 2000 // nterms)]
        exact = bench.measure("evaluation", "python-int", lambda: python_values(exps, coefs, sample), repeats=2,
                              phase="run", **dict(p, points=len(sample)))
        if exact != words(cpu["values"][:len(sample)]):
            raise SystemExit(f"Python integers and native values differ at {nterms} terms")
        poly.close()
    # refusal thresholds at stated budgets (measured)
    exps, coefs = random_polynomial(rng, 257)
    budget = Budget(support=256)
    table.polynomial_from_arrays(exps[:256], coefs[:256], budget=budget).close()
    bench.refusal("polynomial construction", "cpu", lambda: table.polynomial_from_arrays(exps, coefs, budget=budget),
                  terms=257, budget_support=256)
    bench.threshold("polynomial support", dict(terms=256), dict(terms=257), budget_support=256)
    poly = table.polynomial_from_arrays(exps[:16], coefs[:16])
    out = Budget(output=1000)
    poly.evaluate_raw(points[:1000], budget=out)
    for backend in bench.backends():
        bench.refusal("evaluation", backend, lambda: poly.evaluate_raw(points[:1001], backend, budget=out),
                      points=1001, budget_output=1000)
    bench.threshold("evaluation output", dict(points=1000), dict(points=1001), budget_output=1000)
    if bench.device:
        need = poly.device_bytes(1000)
        poly.evaluate_raw(points[:1000], "cuda", budget=Budget(bytes=need))
        bench.refusal("evaluation", "cuda", lambda: poly.evaluate_raw(points[:1000], "cuda", budget=Budget(bytes=need - 1)),
                      points=1000, budget_bytes=need - 1)
        bench.threshold("device copies", dict(bytes=need), dict(bytes=need - 1), points=1000)
    poly.close()
    small = mh.HandleTable(4)
    held = [small.polynomial_from_arrays(exps[:1], coefs[:1]) for _ in range(4)]
    bench.refusal("insert into a full table", "cpu", lambda: small.polynomial_from_arrays(exps[:1], coefs[:1]),
                  capacity=4, live=len(held))
    bench.threshold("table capacity", dict(live=4), dict(live=5), capacity=4)
    small.close()
    table.close()


def lifetime(bench, rng):
    scale = 1 if bench.quick else 8
    batches = 5
    table = mh.HandleTable(64)
    points = rng.integers(-100, 101, size=(64, NVARS), dtype=np.int64)
    shapes = {n: random_polynomial(rng, n) for n in (16, 256, 4096)}
    source = table.domain("snapshot", b"benchmark scope", nvars=1, ndata=1, universe_rows=1 << 20)
    moments = table.bank({(0,): 100, (1,): -5, (2,): 900})

    def polynomial_cycle(n):
        exps, coefs = shapes[n]
        def cycle(i):
            h = table.polynomial_from_arrays(exps, coefs)
            h.evaluate_raw(points)
            h.close()
        return cycle

    def domain_cycle(i):
        h = table.domain("snapshot", b"scope %d" % i, nvars=2, ndata=2, revision=i, universe_rows=1000)
        h.spec
        h.close()

    def bank_cycle(i):
        h = table.bank({(0,): 10, (1,): i, (2,): i * i})
        h.moment((2,))
        h.close()

    def cache_cycle(i):
        h = table.state_cache(capacity=4)
        h.put(source, b"basis", moments)
        h.get(source, b"basis", 1).close()
        h.close()

    def deferred_cycle(i):
        exps, coefs = shapes[16]
        h = table.polynomial_from_arrays(exps, coefs)
        with h.retained():
            if h.close() != "deferred":
                raise SystemExit("a close under a live reference was not deferred")

    def first_slice_domain(i):
        d = Domain("snapshot", b"scope %d" % i, carrier="integer", nvars=2, ndata=2, revision=i, universe_rows=1000)
        d.spec
        d.close()

    def first_slice_cache(i):
        MomentCache(4).close()

    kinds = [("polynomial, 16 terms", polynomial_cycle(16), 400, "table"),
             ("polynomial, 256 terms", polynomial_cycle(256), 200, "table"),
             ("polynomial, 4096 terms", polynomial_cycle(4096), 20, "table"),
             ("domain", domain_cycle, 400, "table"), ("bank", bank_cycle, 400, "table"),
             ("state cache with put and get", cache_cycle, 200, "table"),
             ("deferred close (retain, close, release)", deferred_cycle, 400, "table"),
             ("domain", first_slice_domain, 400, "first-slice"),
             ("state cache", first_slice_cache, 400, "first-slice")]
    for name, cycle, per_batch, owner in kinds:
        per_batch *= scale
        samples, memory, i = [], [], 0
        rss_before = rss_bytes()
        for _ in range(batches):
            t0 = time.perf_counter()
            for _ in range(per_batch):
                cycle(i)
                i += 1
            samples.append((time.perf_counter() - t0) / per_batch)
            snapshot = dict(rss_bytes=rss_bytes())
            if owner == "table":
                s = table.stats()
                snapshot.update({k: s[k] for k in ("live", "payload_bytes", "peak_payload_bytes", "allocations",
                                                   "allocations_total", "destroyed")})
            memory.append(snapshot)
        backend = "cpu" if owner == "table" else "first-slice"
        bench.records.append(dict(case="lifetime", backend=backend, phase="repeated",
                                  params=dict(kind=name, cycles_per_batch=per_batch, batches=batches),
                                  samples_seconds=samples, first_seconds=samples[0],
                                  warm_median_seconds=float(np.median(samples[1:])), warm_min_seconds=min(samples[1:]),
                                  unit="seconds per cycle", memory_after_batches=memory, rss_before=rss_before))
        if owner == "table" and memory[-1]["live"] != 2:      # only the shared domain and bank stay live
            raise SystemExit(f"{name}: {memory[-1]['live'] - 2} states leaked")
    s = table.stats()
    if s["retains"] != s["releases"]:
        raise SystemExit("a native reference was not released")
    source.close()
    moments.close()
    final = table.stats()
    bench.records.append(dict(case="lifetime totals", backend="cpu", phase="final", params={},
                              result={k: final[k] for k in ("live", "payload_bytes", "allocations", "inserted",
                                                            "destroyed", "deferred_closes", "retains", "releases")}))
    table.close()


def concurrency(bench, rng):
    before = scheduling()
    table = mh.HandleTable(256)
    exps, coefs = random_polynomial(rng, 256)
    poly = table.polynomial_from_arrays(exps, coefs)
    points = rng.integers(-100, 101, size=(1000 if bench.quick else 4000, NVARS), dtype=np.int64)
    expected = poly.evaluate_raw(points)["values"]
    cache = table.state_cache(capacity=16)
    domains = [table.domain("snapshot", b"scope %d" % j, nvars=1, ndata=1, universe_rows=1000) for j in range(4)]
    banks = [table.bank({(0,): 10 + j, (1,): j, (2,): j * j}) for j in range(4)]
    calls = 24 if bench.quick else 96
    operations = 1200 if bench.quick else 6000
    errors = []

    def run(threads, target, total):
        share = [total // threads + (k < total % threads) for k in range(threads)]
        workers = [threading.Thread(target=target, args=(share[k], k)) for k in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return total

    def evaluate(n, k):
        for _ in range(n):
            if not np.array_equal(poly.evaluate_raw(points)["values"], expected):
                errors.append("a concurrent evaluation differs")

    def update(n, k):
        for i in range(n):
            j = (k + i) % 4
            cache.put(domains[j], b"basis", banks[j])
            got = cache.get(domains[j], b"basis", 1)
            if got is None or got.moment((1,)) != j:
                errors.append("a serialized cache update was lost")
            if got is not None:
                got.close()

    for threads in (1, 2, 4):
        bench.measure("concurrent immutable evaluation", "cpu", lambda: run(threads, evaluate, calls), repeats=3,
                      phase="run", threads=threads, calls=calls, points=len(points), terms=256)
        bench.measure("serialized cache updates", "cpu", lambda: run(threads, update, operations), repeats=3,
                      phase="run", threads=threads, operations=operations, unit="put + get pairs")
    if errors:
        raise SystemExit(errors[0])
    after = scheduling()
    s = table.stats()
    bench.records.append(dict(case="process scheduling", backend="os", phase="read only", params={},
                              result=dict(nice_before=before["nice"], nice_after=after["nice"],
                                          affinity_unchanged=before["affinity"] == after["affinity"],
                                          cpus=len(before["affinity"] or [])),
                              before=before, after=after))
    if before != after:
        raise SystemExit(f"the process scheduling changed: {before} -> {after}")
    bench.records.append(dict(case="concurrency totals", backend="cpu", phase="final", params={},
                              result=dict(retains=s["retains"], releases=s["releases"],
                                          cache_puts=cache.stats["puts"], cache_hits=cache.stats["hits"])))
    table.close()


def main():
    bench = Bench.from_args("FND-013", __doc__)
    rng = np.random.default_rng(bench.seed)
    evaluation(bench, rng)
    lifetime(bench, rng)
    concurrency(bench, rng)
    bench.note("Lifetime records give seconds per create/use/close cycle per batch and the native counters and RSS "
               "after every batch; first-slice rows are the Python-owned objects without a table.")
    bench.note("Concurrency: Python threads; native evaluations release the GIL and run in parallel (up to the "
               "machine's cores), cache calls are short and serialized by the cache's native mutex (and the GIL). "
               "The process priority and CPU affinity are only read.")
    bench.note("NumPy int64 is a speed baseline only: it is exact on this grid because every value fits int64; it "
               "would wrap silently past it, where the native evaluation returns 128-bit values or flags overflow.")
    bench.write()


if __name__ == "__main__":
    main()
