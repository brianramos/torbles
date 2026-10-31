# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-001 benchmark: bounded affine, sum and product discovery and its exhaustive verification.

    python bench/benchmark_math_disc.py --capability DISC-001 --out FILE [--device] [--quick]

Grid: 2 / 4 / 8 observed variables over 1k / 100k / 1M rows with planted affine, product and conservation (sum)
relations plus one-row near misses. Reported separately: native discovery (proposals and verification of every
candidate), the affine proposal alone, and the exhaustive verification of each planted hypothesis (the catalog's
probe, the native verifier, and the CUDA kernel with --device). Then the candidate budget (truncation) and revision
stability (a mutation makes the catalog stale: verification is refused, discovery reruns). DISC-001 keeps no Gram or
nullspace state (that is DISC-002), so none is measured here.
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_disc
from torbles.math_errors import StaleStateError
from torbles.relationships import discover


def table(rng, k, rows):
    x = rng.integers(-1000, 1001, size=rows)
    y = rng.integers(-500, 2001, size=rows)
    cols = {"x": x, "a": 3 * x - 7}
    hyps = [("affine", "a", "x", None, 3, -7)]
    if k >= 4:
        q = x + y
        q[rows // 2] += 1                                       # one-row near miss of a conservation law
        cols.update(y=y, q=q)
        hyps += [("sum", "q", "x", "y", 1, 0)]
    if k >= 8:
        r = 2 * y + 5
        r[rows // 3] -= 1                                       # one-row near miss of an affine law
        cols.update(s=x + y, p=x * y, r=r, w=rng.integers(-50, 50, size=rows))
        hyps += [("sum", "s", "x", "y", 1, 0), ("product", "p", "x", "y", 1, 0), ("affine", "r", "y", None, 2, 5)]
    names = list(cols)[:k]
    frame = pd.DataFrame({c: cols[c] for c in names})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in names) + ")"
    db = torbles.from_ddl(ddl, {"t": frame}, keep_plain=False)
    return db, names, [h for h in hyps if h[1] in names and h[2] in names and (h[3] is None or h[3] in names)]


def grid(bench):
    rng = np.random.default_rng(bench.seed)
    for rows in ((1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)):
        for k in (2, 4, 8):
            db, names, hyps = table(rng, k, rows)
            p = dict(variables=k, rows=rows)

            def run_discovery():
                cat = discover(db, "t", names, max_candidates=100_000, attach=False)
                stats = cat.stats
                out = dict(candidates=stats["candidates"], rules=stats["rules"], truncated=bool(stats["truncated"]))
                cat.close()
                return out
            bench.measure("discovery (proposals + verification)", "cpu", run_discovery, repeats=3, **p)
            cat = discover(db, "t", names, modules=(), max_candidates=0, attach=False)
            for module, out, left, right, scale, bias in hyps:
                h = dict(p, hypothesis=f"{module}:{out}~{left}{'' if right is None else ',' + right}")
                if module == "affine":
                    bench.measure("affine proposal", "cpu", lambda: math_disc.propose_affine(cat, out, left),
                                  repeats=5, **h)
                bench.measure("verification", "cpu-probe",
                              lambda: cat.probe(module, out, left, right, scale=scale, bias=bias)["violations"],
                              repeats=5, **h)
                for backend in bench.backends():
                    bench.measure("verification", backend,
                                  lambda: math_disc.verify(cat, module, out, left, right, scale=scale, bias=bias,
                                                           backend=backend)["violations"], repeats=5, **h)
            cat.close()


def budgets_and_revisions(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 20_000 if bench.quick else 100_000
    db, names, hyps = table(rng, 8, rows)
    for budget in (1, 4, 16, 64, 256, 1000, 100_000):
        def run():
            cat = discover(db, "t", names, max_candidates=budget, attach=False)
            s = cat.stats
            cat.close()
            return dict(candidates=s["candidates"], rules=s["rules"], truncated=bool(s["truncated"]))
        bench.measure("candidate budget", "cpu", run, repeats=3, rows=rows, variables=8, max_candidates=budget)
    cat = discover(db, "t", names, max_candidates=100_000, attach=False)
    before = cat.stats["rules"]
    db["t"].flip(0, 0)                                          # a mutation: the catalog's proofs are stale

    def stale():
        try:
            math_disc.verify(cat, "sum", "s", "x", "y")
        except StaleStateError:
            return "refused: stale"
        return "not refused"
    bench.measure("stale verification", "cpu", stale, repeats=3, rows=rows, variables=8)
    cat.close()

    def rediscover():
        c = discover(db, "t", names, max_candidates=100_000, attach=False)
        r = c.stats["rules"]
        c.close()
        return dict(rules=r, rules_before=before)
    bench.measure("rediscovery after mutation", "cpu", rediscover, repeats=3, rows=rows, variables=8)


def main():
    bench = Bench.from_args("DISC-001", __doc__)
    grid(bench)
    budgets_and_revisions(bench)
    bench.note("Discovery is native and single-threaded; verification backends: cpu-probe (relationships.c tr_probe, "
               "which also registers usable rules), cpu (math_disc.c), cuda (math_disc_gpu.cu, device runs only).")
    bench.note("No Gram/nullspace state exists in DISC-001; its growth is DISC-002's benchmark.")
    bench.write()


if __name__ == "__main__":
    main()
