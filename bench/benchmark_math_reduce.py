# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-006 benchmark: fused multi-output basis reduction against the preserved repeated plan (the independent native
reductions one by one) and DuckDB executing each output as its own query.

    python bench/benchmark_math_reduce.py --capability FND-006 --out results/math/FND-006/bench.json [--quick] [--device]

Grid: 1, 3, 6 and 15 basis outputs (banks of overlapping degree <= 2 bases, counts, sparse histograms; shared and
private predicates; the scope's grouping and a rollup) x selectivity (1 %, 10 %, 100 % of the rows pass the shared
predicate) x group domain (4, 256, 4096 groups). Phases: "build" (plan construction: validation, merging, proofs),
"first" (the first execution, sample 0) and "warm" (reuse of the same plan, samples 1..). Each record keeps the run's
actual scans, kernel launches, decoded bytes, workspace, product evaluations, accumulations and state bytes, and the
plan's shared products and the planner's choice. CUDA is timed only with --device on an actual device.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_bench import Bench  # noqa: E402

import torbles  # noqa: E402
from torbles import math_bank as mbk, math_bool as mb, math_reduce as mr  # noqa: E402
from torbles.math_budgets import Budget  # noqa: E402

POOL = ["1", "x", "y", "d", "w", "x^2", "x*y", "y^2", "x*d", "d^2", "w*x", "w^2", "y*d", "w*y", "w*d"]


def table(rng, n, groups):
    k = rng.integers(0, groups, n)
    u = rng.integers(0, 1000, n)                       # the selectivity column
    x = rng.integers(-5000, 5001, n)
    y = rng.integers(-40, 41, n).astype(object)
    y[rng.random(n) < 0.1] = None
    d = [f"{v / 100:.2f}" for v in rng.integers(-99999, 100000, n)]
    w = rng.integers(0, 1000, n)
    frame = pd.DataFrame({"k": k, "u": u, "x": x, "y": pd.Series(y, dtype=object), "d": pd.Series(d, dtype=object),
                          "w": w})
    ddl = "CREATE TABLE f (k BIGINT, u BIGINT, x BIGINT, y BIGINT, d DECIMAL(10,2), w BIGINT);"
    return torbles.from_ddl(ddl, {"f": frame})


def request(scope, coarse, m, selectivity):
    """m outputs with overlapping bases, shared and private predicates, two groupings"""
    cut = int(round(selectivity * 1000))
    conds = [mb.condition("u", "<", cut), mb.condition("y", "is not null"), mb.condition("x", ">", 0),
             mb.condition("w", "<", 500)]
    req = mr.Request(scope, conditions=conds, groupings={"all": mr.rollup(coarse)})
    sel, ynn, xpos, wlow = mb.variables(4)
    shared = sel & ynn
    preds = [sel, shared, sel & xpos, sel & wlow, shared & xpos]
    for i in range(m):
        where = preds[i % len(preds)]
        group = "all" if i % 4 == 3 else None
        kind = i % 5
        if kind == 4:
            req.histogram(f"o{i}", ["y"], weight="w", where=where, group=group)
        elif kind == 3:
            req.count(f"o{i}", where=where, group=group)
        else:
            lo = (3 * i) % (len(POOL) - 6)
            req.bank(f"o{i}", POOL[lo:lo + 6], where=where, mode="joint" if kind == 2 else "sql", group=group)
    return req


def duck_queries(db, req, selectivity):
    """each bank / count output as its own DuckDB query (WHERE, GROUP BY, COUNT, SUM in HUGEINT)"""
    con = db.con
    cut = int(round(selectivity * 1000))
    preds = ["u < {c}", "u < {c} AND y IS NOT NULL", "u < {c} AND x > 0", "u < {c} AND w < 500",
             "u < {c} AND y IS NOT NULL AND x > 0"]
    sqls = []
    for i, o in enumerate(req.outputs):
        where = preds[i % len(preds)].format(c=cut)
        group = "" if o["group"] == "all" else " GROUP BY k"
        key = "" if o["group"] == "all" else "k, "
        if o["kind"] == mr.HIST:
            sqls.append(f"SELECT {key}y, COUNT(*), COUNT(w), SUM(CAST(w AS HUGEINT)) FROM f WHERE {where}"
                        f"{group.replace('GROUP BY k', 'GROUP BY k, y') if group else ' GROUP BY y'}")
            continue
        parts = ["COUNT(*)"]
        for t in o["exps"]:
            cols = [n for n, e in zip(req.scope.names, t) for _ in range(e)]
            expr = " * ".join(f"CAST({c} * 100 AS HUGEINT)" if c == "d" else f"CAST({c} AS HUGEINT)" for c in cols) or "1"
            parts += [f"COUNT({expr})", f"SUM({expr})"]
        sqls.append(f"SELECT {key}{', '.join(parts)} FROM f WHERE {where}{group}")
    return lambda: [con.execute(q).fetchall() for q in sqls]


def main():
    bench = Bench.from_args("FND-006", __doc__)
    rng = np.random.default_rng(bench.seed)
    n = 20000 if bench.quick else 200000
    counts = (1, 15) if bench.quick else (1, 3, 6, 15)
    sels = (0.1,) if bench.quick else (0.01, 0.1, 1.0)
    domains = (4, 256) if bench.quick else (4, 256, 4096)
    repeats = 3 if bench.quick else 5
    bench.note(f"{n} rows; outputs {counts}; selectivity {sels}; groups {domains}; {repeats} samples per measurement "
               "(sample 0 the first execution, the rest warm reuse of the same plan)")
    for groups in domains:
        db = table(rng, n, groups)
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["x", "y", "d", "w", "u"], group_by=["k"])
        coarse = mbk.Scope.from_table(t, ["x", "y", "d", "w", "u"])
        for sel in sels:
            for m in counts:
                params = dict(outputs=m, selectivity=sel, groups=groups, rows=n)
                req = request(scope, coarse, m, sel)
                t0 = time.perf_counter()
                plan = req.compile()
                build = time.perf_counter() - t0
                info = plan.info
                ev = plan.choose(devices=bench.backends())
                bench.records.append(dict(case="construction", backend="cpu", phase="build", params=params,
                                          samples_seconds=[build], first_seconds=build, warm_median_seconds=build,
                                          warm_min_seconds=build,
                                          result=dict(products=info["nproducts"], term_rows=info["term_rows"],
                                                      shared_products=info["shared_products"],
                                                      merged_nodes=info["merged_nodes"], nodes=info["nsched"],
                                                      states=info["states"],
                                                      chosen=f"{ev['choice']['plan']}@{ev['choice']['device']}",
                                                      fused_total=next(c["total"] for c in ev["candidates"]
                                                                       if c["plan"] == "fused" and c["device"] == "cpu"),
                                                      repeated_total=next(c["total"] for c in ev["candidates"]
                                                                          if c["plan"] == "repeated" and c["device"] == "cpu"))))
                for backend in bench.backends():
                    for which in ("fused", "repeated"):
                        def run(which=which, backend=backend):
                            res = plan.execute(which, backend)
                            st = res.stats
                            res.close()
                            return {k: v for k, v in st.items() if k != "device_name"}
                        bench.measure(f"{which}", backend, run, repeats=repeats, phase="first+warm", **params)
                if sel == 0.1:
                    bench.measure("independent queries", "duckdb", duck_queries(db, req, sel), repeats=repeats,
                                  phase="first+warm", **params)
                plan.close()
    # refusal thresholds: each plan's states and bytes at the limit and one past (measured, never extrapolated)
    db = table(rng, 5000, 64)
    scope = mbk.Scope.from_table(db["f"], ["x", "y", "d", "w", "u"], group_by=["k"])
    coarse = mbk.Scope.from_table(db["f"], ["x", "y", "d", "w", "u"])
    plan = request(scope, coarse, 4, 0.1).compile()      # banks and counts: no data-dependent sparse final state
    bench.note("thresholds: 4 outputs (banks and counts) over 5000 rows, 64 groups; a sparse histogram's final state "
               "is checked once its cells are counted, so its limit is data-dependent and not part of this grid")
    for which in ("fused", "repeated"):
        need = plan.need(which, "cpu")
        for resource in ("states", "bytes"):
            bench.measure(f"{which} at the {resource} limit", "cpu",
                          lambda r=resource, w=which, nd=need: len(plan.execute(w, "cpu", budget=Budget(**{r: nd[r]}))),
                          repeats=1, phase="limit", resource=resource, need=need[resource])
            bench.refusal(f"{which} one {resource[:-1] if resource.endswith('s') else resource} past the limit", "cpu",
                          lambda r=resource, w=which, nd=need: plan.execute(w, "cpu", budget=Budget(**{r: nd[r] - 1})),
                          resource=resource, limit=need[resource] - 1)
            bench.threshold(f"{which} {resource}", admitted={resource: need[resource]},
                            refused={resource: need[resource] - 1})
    bench.write()


if __name__ == "__main__":
    main()
