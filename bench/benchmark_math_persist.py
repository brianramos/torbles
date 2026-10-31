# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-011 benchmark: versioned binary persistence of typed mathematical states.

    python bench/benchmark_math_persist.py --capability FND-011 --out FILE [--device] [--quick]

Case 1, serialization / load / validation time and bytes separately from mathematical evaluation: for a FND-004
bank, a DIST-001 histogram, a LIN-001 moment state, a FND-007 certificate and FND-009 big numbers, each phase is
timed alone: building the state from the scope (evaluation), dumps (with the record bytes), loads, validate (header
and checksum on the CPU; on the device with --device), rehydration into the scope, and one query family evaluated on
the state. Case 2, persisted versus rebuilt state under fixed data and changing query families (10k / 100k / 1M
rows): for group means (bank), per-group regressions (moments), quantiles (histogram prefix) and certificate
authorization, the persisted path (load + freshness + rehydration + the query) against the rebuilt path (scope +
build + the query). Budget refusals of loading are recorded as thresholds.
"""
from fractions import Fraction

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_carriers as mcar
from torbles import math_certificates as mc
from torbles import math_dist as md
from torbles import math_lin as ml
from torbles import math_persist as mp
from torbles.math_budgets import Budget


def table(rng, rows, groups):
    frame = pd.DataFrame({"k": rng.integers(0, groups, rows), "x": rng.integers(-1000, 1001, rows),
                          "y": rng.integers(-300, 301, rows), "d": rng.integers(0, 50, rows)})
    frame["z"] = 2 * frame["x"] + 1
    return torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT, y BIGINT, d BIGINT, z BIGINT)", {"t": frame},
                            keep_plain=False)


NAMES = ["x", "y", "d", "z"]


def builders(scope):
    return {
        "bank": lambda: mbk.build(scope, ["1", "x", "y", "x^2", "x*y", "y^2"]),
        "histogram": lambda: md.histogram(scope, ["d"], weight="x"),
        "moments": lambda: ml.moments(scope, ["x", "d"], "y"),
        "certificate": lambda: mc.snapshot(scope, mc.affine("z", "x", 2, 1)),
    }


def queries(scope):
    return {
        "bank": lambda b: sum(b.state(g)["count"][1] for g in range(min(b.groups, 64))),
        "histogram": lambda h: h.prefix("count", 1).quantile(Fraction(1, 2), "disc"),
        "moments": lambda m: sum(1 for g in range(min(m.groups, 16)) if m.fit(g).kind == "unique"),
        "certificate": lambda c: c.authorize_exact(scope).kind,
    }


def rehydrate(rec, scope, kind):
    if kind == "bank":
        return rec.bank(scope)
    if kind == "histogram":
        return rec.histogram(scope)
    if kind == "moments":
        return rec.moments(scope)
    return rec.certificate()


def phases(bench, rows, groups):
    rng = np.random.default_rng(bench.seed)
    db = table(rng, rows, groups)
    t = db["t"]
    p = dict(rows=rows, groups=groups)
    scope = bench.measure("scope", "cpu", lambda: mbk.Scope.from_table(t, NAMES, group_by=["k"]), repeats=3,
                          phase="construction", **p)
    build, query = builders(scope), queries(scope)
    for kind in build:
        q = dict(p, state=kind)
        state = bench.measure("evaluation: build the state", "cpu", build[kind], repeats=3, phase="evaluation", **q)
        data = bench.measure("dumps", "cpu", lambda: mp.dumps(state), repeats=5, phase="serialization", **q)
        bench.note(f"{kind} at {rows} rows / {groups} groups: {len(data)} record bytes")
        bench.measure("record bytes", "cpu", lambda: len(data), repeats=1, phase="bytes", bytes=len(data), **q)
        rec = bench.measure("loads", "cpu", lambda: mp.loads(data), repeats=5, phase="load", **q)
        for backend in bench.backends():
            bench.measure("validate (header + checksum)", backend, lambda: mp.validate(data, backend=backend)["reason"],
                          repeats=5, phase="validation", **q)
        back = bench.measure("rehydrate into the scope", "cpu", lambda: rehydrate(rec, scope, kind), repeats=5,
                             phase="rehydration", **q)
        bench.measure("evaluation: a query on the state", "cpu", lambda: query[kind](back), repeats=3,
                      phase="evaluation", **q)
        need = mp.need("load", len(data))["bytes"]
        refused = bench.refusal("load budget", "cpu", lambda: mp.loads(data, budget=Budget(bytes=need - 1)), **q)
        bench.threshold(f"load bytes ({kind})", need, need - 1 if refused["refused"] else None, rows=rows)
    for name, value in (("bigint 2^20000", mcar.BigInt(2 ** 20000 + 1)),
                        ("bigq (2^4000+1)/3^2000", mcar.BigRational(Fraction(2 ** 4000 + 1, 3 ** 2000)))):
        q = dict(state=name)
        data = bench.measure("dumps", "cpu", lambda: mp.dumps(value), repeats=5, phase="serialization", **q)
        bench.measure("record bytes", "cpu", lambda: len(data), repeats=1, phase="bytes", bytes=len(data), **q)
        rec = bench.measure("loads", "cpu", lambda: mp.loads(data), repeats=5, phase="load", **q)
        bench.measure("rehydrate the value", "cpu", lambda: rec.value(), repeats=5, phase="rehydration", **q)


def persisted_vs_rebuilt(bench, rows):
    rng = np.random.default_rng(bench.seed + rows)
    groups = 256
    db = table(rng, rows, groups)
    t = db["t"]
    scope = mbk.Scope.from_table(t, NAMES, group_by=["k"])
    build, query = builders(scope), queries(scope)
    saved = {kind: mp.dumps(build[kind]()) for kind in build}
    for kind in build:
        q = dict(rows=rows, groups=groups, family=kind, bytes=len(saved[kind]))

        def persisted():
            rec = mp.loads(saved[kind])
            if not rec.fresh(scope).fresh:
                raise AssertionError("the source changed")
            return query[kind](rehydrate(rec, scope, kind))

        def rebuilt():
            now = mbk.Scope.from_table(t, NAMES, group_by=["k"])
            return queries(now)[kind](builders(now)[kind]())
        a = bench.measure("persisted (load + fresh + rehydrate + query)", "cpu", persisted, repeats=3,
                          phase="persisted", **q)
        b = bench.measure("rebuilt (scope + build + query)", "cpu", rebuilt, repeats=3, phase="rebuilt", **q)
        if a != b:
            raise AssertionError(f"{kind}: persisted {a!r} != rebuilt {b!r}")


def main():
    bench = Bench.from_args("FND-011", __doc__)
    if bench.quick:
        phases(bench, 20_000, 64)
        grid = (10_000, 100_000)
    else:
        for rows, groups in ((100_000, 64), (100_000, 16_384)):
            phases(bench, rows, groups)
        grid = (10_000, 100_000, 1_000_000)
    for rows in grid:
        persisted_vs_rebuilt(bench, rows)
    bench.note("Serialization, load, validation and rehydration are timed apart from building the state and from "
               "evaluating queries on it; validation on the device (--device) computes the checksum there.")
    bench.note("The persisted path checks the recorded source against the current scope before adopting the state; "
               "a changed source makes it a rebuild (StaleSourceError).")
    bench.write()


if __name__ == "__main__":
    main()
