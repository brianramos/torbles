# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""META-002 benchmark: families of related queries compiled into one program against the same queries executed
independently (one native program per query, and DuckDB per query), and the equivalence checks of query pairs.

    python bench/benchmark_math_program.py --capability META-002 --out results/math/META-002/bench.json [--quick] [--device]

Families: 1, 8, 32 and 128 queries, "shared" (predicates drawn from a few classes, many written differently so
that canonical forms merge them; measures over overlapping polynomials) and "disjoint" (each query its own threshold
and its own measure columns). Recorded: the construction (canonicalization, CSE and, with prove, the bounded proof
search) apart from execution, the actual shared scans and kernel launches, states, classes and merges, and the
first / warm times of the program, the independent native queries and DuckDB. Pairs: equivalent (canonical and by
search), one-row-different, universally different but equal on the snapshot, undecided (BIT) and budget-limited;
snapshot checking and universal proof search are timed separately and every outcome, unknown and budget included,
is kept.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_bench import Bench  # noqa: E402

import torbles  # noqa: E402
from torbles import math_bank as mbk, math_bool as mb, math_program as mp  # noqa: E402
from torbles.math_budgets import Budget  # noqa: E402

POLYS = ["x", "y", "w", {"x": 2, "y": -1}, {"x*y": 1}, {"w": 1, "x": 3}, {"d": 1}, {"x^2": 1, "y": 1}]


def table(rng, n, groups):
    k = rng.integers(0, groups, n)
    u = rng.integers(0, 1000, n)
    x = rng.integers(-5000, 5001, n)
    y = rng.integers(-40, 41, n).astype(object)
    y[rng.random(n) < 0.1] = None
    d = pd.Series([f"{v / 100:.2f}" for v in rng.integers(-99999, 100000, n)], dtype=object)
    w = rng.integers(0, 1000, n)
    x[5] = 777                                          # exactly one row with x = 777 (a one-row-different pair)
    x[x == 777] = 778
    x[5] = 777
    frame = pd.DataFrame({"k": k, "u": u, "x": x, "y": pd.Series(y, dtype=object), "d": d, "w": w})
    return torbles.from_ddl("CREATE TABLE f (k BIGINT, u BIGINT, x BIGINT, y BIGINT, d DECIMAL(10,2), w BIGINT);",
                            {"f": frame})


def family(scope, size, regime):
    """size queries; shared: 5 predicate classes written 3 ways, overlapping measures; disjoint: a threshold each"""
    if regime == "shared":
        conds = [mb.condition("u", "<", 300), mb.condition("u", "<=", 299), mb.condition("y", "is null"),
                 mb.condition("x", ">", 0), mb.condition("x", ">=", 1)]
        v = mb.variables(len(conds))
        forms = [[v[0], v[1], ~~v[0]], [v[0] & ~v[2], ~v[2] & v[1], ~(~v[0] | v[2])],
                 [v[3], v[4], v[3] & v[4]], [v[3] | v[2], v[2] | v[4], ~(~v[3] & ~v[2])],
                 [(v[3] & v[2]) | (v[3] & ~v[2]), v[4] & (v[2] | ~v[2]), (v[2] & v[4]) | (~v[2] & v[3])]]
        fam = mp.Family(scope, conditions=conds)   # the last class merges only by the proof search (distributivity)
        sql = ["u < 300", "u < 300 AND y IS NOT NULL", "x > 0", "x > 0 OR y IS NULL", "x > 0"]
        queries = []
        for i in range(size):
            cls = i % 5
            where = forms[cls][(i // 5) % 3]
            ms = [mp.count(), mp.sum(POLYS[i % 4]), mp.avg(POLYS[(i + 1) % 4])]
            fam.query(f"q{i}", ms, where=where)
            queries.append((sql[cls], [None, POLYS[i % 4], POLYS[(i + 1) % 4]]))
        return fam, queries
    cuts = [int(c) for c in np.linspace(10, 990, max(size, 1))]
    conds = [mb.condition("u", "<", c) for c in cuts[:64]]
    v = mb.variables(len(conds))
    fam = mp.Family(scope, conditions=conds)
    queries = []
    for i in range(size):
        j = i % len(conds)
        ms = [mp.count(), mp.sum(POLYS[i % len(POLYS)])]
        fam.query(f"q{i}", ms, where=v[j] if i < len(conds) else v[j] & ~v[(j + 1) % len(conds)])
        where = f"u < {cuts[j]}" if i < len(conds) else f"u < {cuts[j]} AND NOT (u < {cuts[(j + 1) % len(conds)]})"
        queries.append((where, [None, POLYS[i % len(POLYS)]]))
    return fam, queries


def poly_sql(p):
    items = [(p, 1)] if isinstance(p, str) else list(p.items())
    out = []
    for m, c in items:
        fs = []
        for f in m.split("*"):
            name, _, e = f.partition("^")
            col = f"CAST({name} * 100 AS HUGEINT)" if name == "d" else f"CAST({name} AS HUGEINT)"
            fs += [col] * int(e or 1)
        out.append(f"({int(c)}) * " + " * ".join(fs))
    return "(" + " + ".join(out) + ")"


def duck(db, queries):
    con = db.con
    sqls = []
    for where, polys in queries:
        parts = ["COUNT(*)"] + [f"SUM({poly_sql(p)}), COUNT({poly_sql(p)})" for p in polys if p is not None]
        sqls.append(f"SELECT k, {', '.join(parts)} FROM f WHERE {where} GROUP BY k")
    return lambda: [con.execute(q).fetchall() for q in sqls]


def main():
    bench = Bench.from_args("META-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    n = 20000 if bench.quick else 200000
    sizes = (1, 8, 32) if bench.quick else (1, 8, 32, 128)
    repeats = 3 if bench.quick else 5
    db = table(rng, n, 64)
    t = db["f"]
    scope = mbk.Scope.from_table(t, ["x", "y", "w", "d", "u"], group_by=["k"])
    bench.note(f"{n} rows, 64 groups; families {sizes}; {repeats} samples (sample 0 first, then warm)")
    bench.note("the second benchmark case is shared with META-001: here the program's own comparisons of WHERE "
               "predicates (canonical forms, the bounded 3^k search with per-column feasibility, the snapshot check); "
               "measure equivalence, implication / containment and META-001's decision procedures (BDDs with "
               "difference logic, Nullstellensatz grids) are benchmarked by bench/benchmark_math_equiv.py")
    for regime in ("shared", "disjoint"):
        for size in sizes:
            params = dict(queries=size, regime=regime, rows=n)
            for prove in (False, True):
                fam, queries = family(scope, size, regime)
                t0 = time.perf_counter()
                prog = fam.compile(prove=prove)
                build = time.perf_counter() - t0
                info = prog.info
                bench.records.append(dict(case="construction" + (" + proof search" if prove else ""), backend="cpu",
                                          phase="build", params=params, samples_seconds=[build], first_seconds=build,
                                          warm_median_seconds=build, warm_min_seconds=build,
                                          result={k: info[k] for k in ("nclasses", "outputs", "conds_in", "conds",
                                                                      "nodes_in", "nodes", "merged_canonical",
                                                                      "merged_search", "search_unknown",
                                                                      "search_budget", "search_iterations", "slots")}))
            for backend in bench.backends():
                def run(backend=backend):
                    res = prog.run(plan="fused", backend=backend)
                    st = res.stats
                    return dict(scans=st["scans"], kernels=st["kernels"], decoded_bytes=st["decoded_bytes"],
                                state_bytes=st["state_bytes"], products=st["products"], plan=res.how["plan"])
                bench.measure("program (one shared pass)", backend, run, repeats=repeats, phase="first+warm", **params)

            def independent(fam=fam):
                scans = 0
                for q in fam.queries:                 # each query its own program: compiled and run on its own
                    single = mp.Family(scope, conditions=fam.conditions)
                    single.query(q["name"], q["measures"], where=q["where"])
                    res = single.compile().run(plan="fused")
                    scans += res.stats["scans"]
                return dict(scans=scans, queries=size)
            bench.measure("independent native queries", "cpu", independent, repeats=max(2, repeats - 2),
                          phase="first+warm", **params)
            bench.measure("independent queries", "duckdb", duck(db, queries), repeats=repeats, phase="first+warm",
                          **params)
    # equivalence pairs: universal proof search and snapshot checking, timed apart, every outcome kept
    conds = [mb.condition("u", "<", 300), mb.condition("u", "<=", 299), mb.condition("y", "is null"),
             mb.condition("y", "is not null"), mb.condition("x", ">", 777), mb.condition("x", ">=", 777),
             mb.condition("x", ">", 10000), mb.condition("x", ">", 20000), mb.condition("x", "bit", 0)]
    conds += [mb.condition("w", "<", c) for c in range(50, 1000, 75)]          # many conditions: budget-limited
    v = mb.variables(len(conds))
    wide_a = v[9]
    wide_b = v[9]
    for i in range(10, len(conds)):
        wide_a = wide_a & v[i]
        wide_b = wide_b & (v[i] | v[i])
    wide_b = wide_b | (v[9] & mb.FALSE) | (v[10] & v[2] & v[3])
    pairs = {"equivalent (canonical)": (v[0], v[1]),
             "equivalent (search)": ((v[0] & v[2]) | (v[0] & v[3] & ~v[2]), v[0] & (v[2] | (v[3] & ~v[2]))),
             "one-row-different": (v[4], v[5]), "universally different, snapshot-equal": (v[6], v[7]),
             "undecided (bit)": (v[8], v[8] & v[0]), "budget-limited": (wide_a, wide_b)}
    fam = mp.Family(scope, conditions=conds)
    for name, (a, b) in pairs.items():
        fam.query(name + " a", [mp.count()], where=a)
        fam.query(name + " b", [mp.count()], where=b)
    prog = fam.compile()
    for name in pairs:
        a, b = name + " a", name + " b"
        budget = Budget(iterations=100000)
        bench.measure("universal proof search", "cpu", lambda a=a, b=b: dict(
            (k, v) for k, v in prog.compare(a, b, budget=budget).items() if k in ("outcome", "how", "iterations",
                                                                                 "grouping_equal", "measures_equal")),
            repeats=repeats, phase="first+warm", pair=name, iterations_budget=100000)
        bench.measure("snapshot check", "cpu", lambda a=a, b=b: dict(
            (k, v) for k, v in prog.snapshot_compare(a, b).items() if k in ("equal", "differ", "first", "words")),
            repeats=repeats, phase="first+warm", pair=name, rows=n)
    bench.write()


if __name__ == "__main__":
    main()
