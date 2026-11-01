# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""META-001 benchmark: scoped query equivalence, implication and counterexamples.

    python bench/benchmark_math_equiv.py --capability META-001 --out FILE [--device] [--quick]

Families: 1 / 8 / 32 / 128 related queries over 200k rows (quick: 20k) with shared filters and measures (filters
are conjunctions of two conditions of a pool of eight, measures from a pool of four polynomials: at most 8 distinct
filters and 4 distinct measures) and with disjoint ones (each query its own filter root and its own measure; the
conditions, at most 36, are reused across roots). Reported: the set construction (the common subexpressions found:
distinct conditions, shared DAG nodes, distinct filters and measures), the shared run (one scan: every condition
once, the DAG once per word, each distinct measure once per row) and the separate runs (one scan per query), each
with the scans, condition rows, node words and measure rows it actually evaluated, and DuckDB running the family as
one multi-aggregate query and as one query per member (the baselines).
Equivalence: equivalent, one-row-different and universally-different predicate / measure / query pairs over 200k
rows (quick: 20k), batched; the snapshot check (CPU; CUDA only with --device; DuckDB counting the differing rows as
the baseline) is timed separately from the universal proof search over a declared domain (BDDs with integer
difference logic, Nullstellensatz grids; exhaustive enumeration), and the unknown / budget-limited outcomes are
kept: BIT conditions outside the theory, finite value sets in decide mode, an enumeration past its support budget
and a BDD past its circuit budget.
"""
import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk
from torbles import math_equiv as meq
from torbles import math_geom as mg
from torbles.math_budgets import Budget

NAMES = ["x", "y", "z", "w"]


def data(rng, rows):
    x = rng.integers(-100, 101, size=rows)
    x[x == 7] = 8
    x[rows // 2] = 7                                        # exactly one row with x = 7 (and y > 0 there)
    y = rng.integers(-50, 51, size=rows)
    y[rows // 2] = 5
    z = x * y
    w = rng.integers(-9, 10, size=rows)
    valid = rng.random(rows) > 0.05                         # w is NULL on about 5% of the rows
    return {"x": x, "y": y, "z": z, "w": w}, {"w": valid}


def scope_of(cols, valid):
    return mbk.Scope.from_arrays(cols, valid=valid)


def frame_of(cols, valid):
    f = pd.DataFrame({c: pd.array(v, dtype="Int64") for c, v in cols.items()})
    f.loc[~valid["w"], "w"] = pd.NA
    f["i"] = np.arange(len(f))
    return f


POOL = [("x", ">", -50), ("x", "<", 60), ("y", ">=", -20), ("y", "<=", 30), ("z", ">", 0), ("w", "is not null", None),
        ("x", "<>", 7), ("w", ">", -5)]
MEASURES = [{(1, 0, 0, 0): 1, (0, 1, 0, 0): 1}, {(1, 1, 0, 0): 1}, {(0, 0, 1, 0): 2, (0, 0, 0, 1): 1},
            {(2, 0, 0, 0): 1}]


def cond(c, op, v):
    return meq.cond(c, op) if v is None else meq.cond(c, op, v)


def sql_cond(c, op, v):
    return f"({c} {op.upper()})" if v is None else f"({c} {op} {v})"


def poly_sql(terms):
    parts = []
    for e, k in terms.items():
        fac = [f"CAST({NAMES[i]} AS HUGEINT)" for i, p in enumerate(e) for _ in range(p)]
        parts.append("*".join([str(k)] + fac))
    return "(" + " + ".join(parts) + ")"


def family(n, shared):
    """(set parts, the SQL of each member): filters, measures, queries"""
    preds, psql, polys, msql, queries = [], [], [], [], []
    if shared:
        pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)][:8]
        conds = [cond(*c) for c in POOL]
        for a, b in pairs:
            preds.append(conds[a] & conds[b])
            psql.append(f"({sql_cond(*POOL[a])} AND {sql_cond(*POOL[b])})")
        for t in MEASURES:
            polys.append(mg.poly(t, NAMES))
            msql.append(poly_sql(t))
        queries = [(i % len(preds), i % len(polys)) for i in range(n)]
    else:
        for i in range(n):
            a, b = i % 32, i // 32
            preds.append(meq.cond("x", ">", a - 16) & meq.cond("y", "<", 10 * b))
            psql.append(f"((x > {a - 16}) AND (y < {10 * b}))")
            t = {(1, 0, 0, 0): 1, (0, 1, 0, 0): i + 1}
            polys.append(mg.poly(t, NAMES))
            msql.append(poly_sql(t))
        queries = [(i, i) for i in range(n)]
    return preds, psql, polys, msql, queries


def duck_family(frame, psql, msql, queries, together):
    con = duckdb.connect()
    con.register("t", frame)
    aggs = [f"count(*) FILTER (WHERE coalesce({psql[f]}, false)), sum({msql[m]}) FILTER (WHERE coalesce({psql[f]}, false))"
            for f, m in queries]
    if together:
        out = con.execute("SELECT " + ", ".join(aggs) + " FROM t").fetchone()
    else:
        out = [con.execute(f"SELECT {a} FROM t").fetchone() for a in aggs]
    con.close()
    return len(out)


def families(bench):
    rng = np.random.default_rng(bench.seed)
    rows = 20_000 if bench.quick else 200_000
    cols, valid = data(rng, rows)
    scope = scope_of(cols, valid)
    frame = frame_of(cols, valid)
    for shared in (True, False):
        kind = "shared" if shared else "disjoint"
        for n in (1, 8, 32, 128):
            preds, psql, polys, msql, queries = family(n, shared)
            p = dict(queries=n, family=kind, rows=rows)
            s = bench.measure("set construction (common subexpressions)", "cpu",
                              lambda: meq.Set(NAMES, predicates=preds, measures=polys, queries=queries), repeats=3,
                              phase="build", **p)
            info = s.info()
            bench.note(f"{kind} family of {n}: {info['conditions']} distinct conditions, {info['nodes']} DAG nodes "
                       f"({info['shared_nodes']} shared by several filters), {info['distinct_filters']} distinct "
                       f"filters, {info['distinct_measures']} distinct measures, {info['queries']} distinct queries")
            for mode in ("shared", "separate"):
                def run(mode=mode):
                    answers, st = s.run(scope, mode=mode)
                    return {k: st[k] for k in ("scans", "condition_rows", "node_words", "measure_rows")}
                bench.measure(f"family run ({mode} scans)", "cpu", run, repeats=3 if rows <= 20_000 or n <= 32 else 1,
                              phase="run", **p)
            a1, _ = s.run(scope, mode="shared")
            a2, _ = s.run(scope, mode="separate")
            assert a1 == a2
            bench.measure("family as one multi-aggregate query", "duckdb",
                          lambda: duck_family(frame, psql, msql, queries, True), repeats=1 if n >= 32 else 3,
                          phase="run", **p)
            bench.measure("family as one query per member", "duckdb",
                          lambda: duck_family(frame, psql, msql, queries, False),
                          repeats=1 if n >= 32 else 3, phase="run", **p)
            s.close()


def pairs():
    names = NAMES
    x_lt7, x_le7, x_le6 = meq.cond("x", "<", 7), meq.cond("x", "<=", 7), meq.cond("x", "<=", 6)
    a, b = meq.cond("y", ">", 0), meq.cond("w", "is null")
    preds = [x_lt7 & a, ~(~x_lt7 | ~a), x_le6 & a, x_le7 & a, meq.cond("x", "bit", 0), meq.cond("x", "bit", 0) | b,
             x_lt7 & a & meq.cond("x", ">", -101)]                  # 6: equal on the data (x >= -100), not on the box
    polys = [mg.poly({(1, 1, 0, 0): 1}, names), mg.poly({(0, 0, 1, 0): 1}, names),
             mg.poly({(2, 0, 0, 0): 1, (1, 1, 0, 0): 2, (0, 2, 0, 0): 1}, names),
             mg.poly({(1, 1, 0, 0): 1, (0, 0, 0, 0): 0}, names)]
    sq = mg.poly([(1, {"x": 2}), (2, {"x": 1, "y": 1}), (1, {"y": 2})], names)
    polys.append(sq)
    queries = [(0, 0), (1, 1), (2, 3), (3, 0)]
    qs = {"equivalent": [meq.predicates(0, 1, semantics="kleene"), meq.predicates(0, 2), meq.measures(2, 4),
                         meq.queries(0, 2, nulls="skip")],
          "one-row-different": [meq.predicates(0, 3), meq.queries(0, 3), meq.predicates(3, 0, relation="implies")],
          "universally-different": [meq.measures(0, 1, nulls="skip"), meq.queries(0, 1, nulls="skip"),
                                    meq.predicates(0, 6)]}
    sql = {"equivalent": ["(x < 7 AND y > 0) IS DISTINCT FROM (NOT (NOT (x < 7) OR NOT (y > 0)))",
                          "coalesce(x < 7 AND y > 0, false) <> coalesce(x <= 6 AND y > 0, false)",
                          "(x*y + 0) <> (x*x + 2*x*y + y*y) - (x*x + x*y + y*y)", None],
           "one-row-different": ["coalesce(x < 7 AND y > 0, false) <> coalesce(x <= 7 AND y > 0, false)", None, None],
           "universally-different": ["CAST(x AS HUGEINT) * y <> z", None,
                                     "coalesce(x < 7 AND y > 0, false) <> coalesce(x < 7 AND y > 0 AND x > -101, false)"]}
    return preds, polys, queries, qs, sql


def equivalence(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 20_000 if bench.quick else 200_000
    cols, valid = data(rng, rows)
    scope = scope_of(cols, valid)
    frame = frame_of(cols, valid)
    preds, polys, queries, qs, sql = pairs()
    s = meq.Set(NAMES, predicates=preds, measures=polys, queries=queries)
    dom = meq.domain({"x": (-200, 200), "y": (-50, 50), "z": (-10000, 10000), "w": (-9, 9)}, nullable=["w"])
    small = meq.domain({"x": (-6, 8), "y": (-3, 3), "z": (-20, 20), "w": (-2, 2)}, nullable=["w"], mode="enumerate")
    for kind, batch in qs.items():
        p = dict(pairs=kind, questions=len(batch), rows=rows)
        for backend in bench.backends():
            def snap(backend=backend):
                r = s.check(scope, batch, backend=backend)
                return {f"q{i}": f"{it['verdict']} {it['label']} witness={it['witness']}" for i, it in enumerate(r)}
            bench.measure("snapshot check", backend, snap, repeats=3, phase="run", **p)
        con = duckdb.connect()
        con.register("t", frame)
        texts = [q for q in sql[kind] if q]
        bench.measure("snapshot check (differing rows)", "duckdb",
                      lambda: {f"q{j}": str(con.execute(f"SELECT count(*), min(i) FROM t WHERE {q}").fetchone())
                               for j, q in enumerate(texts)},
                      repeats=3, phase="run", note="the SQL-expressible questions only", **p)
        con.close()

        def universal():
            r = s.decide(batch, dom)
            return {f"q{i}": f"{it['verdict']} {it['label']} {it['method']} {it['why']}" for i, it in enumerate(r)}
        bench.measure("universal proof search (decide)", "cpu", universal, repeats=3, phase="run", **p)

        def enumerate_():
            r = s.decide(batch, small, budget=Budget(support=1 << 21))
            return {f"q{i}": f"{it['verdict']} {it['label']} {it['why']}" for i, it in enumerate(r)}
        bench.measure("universal by exhaustive enumeration (small box)", "cpu", enumerate_, repeats=1, phase="run",
                      points=15 * 7 * 41 * 6, **p)
    snap = s.check(scope, qs["universally-different"])
    uni = s.decide(qs["universally-different"], dom)
    bench.note("snapshot vs universal on the universally-different pairs: " +
               "; ".join(f"{a['verdict']}/{a['label']} vs {b['verdict']}/{b['label']} {b['why'] or ''}"
                         for a, b in zip(snap, uni)))
    # unknown and budget-limited outcomes, kept as results
    vals = meq.domain({"x": list(range(-5, 6)), "y": (-3, 3), "z": (-9, 9), "w": (-2, 2)}, nullable=["w"])
    out = {}
    out["bit (outside the theory)"] = s.decide([meq.predicates(4, 5)], dom)[0]
    out["finite value set in decide mode"] = s.decide([meq.predicates(0, 2)], vals)[0]
    out["enumeration past its support budget"] = s.decide(        # x, y read: 201 x 101 points
        [meq.predicates(0, 2)], meq.domain({"x": (-100, 100), "y": (-50, 50), "z": (-5000, 5000), "w": (-9, 9)},
                                           mode="enumerate"), budget=Budget(support=20_000))[0]
    out["BDD past its circuit budget"] = s.decide([meq.predicates(0, 2)], dom, budget=Budget(circuit=3))[0]
    for case, it in out.items():
        bench.measure(f"unknown outcome: {case}", "cpu", lambda it=it: dict(verdict=it["verdict"], why=it["why"],
                      resource=(it["budget"] or {}).get("resource"), need=(it["budget"] or {}).get("need"),
                      limit=(it["budget"] or {}).get("limit")), repeats=1, phase="result")
        if it["budget"]:
            bench.threshold(f"universal question: {case}", admitted=None, refused=it["budget"])
    # batching: many questions in one check
    for k in (8, 64, 256):
        batch = [qs["equivalent"][i % 4] for i in range(k)]
        for backend in bench.backends():
            bench.measure("batched snapshot check", backend, lambda b=backend: len(s.check(scope, batch, backend=b)),
                          repeats=3, phase="run", questions=k, rows=rows)


def main():
    bench = Bench.from_args("META-001", __doc__)
    families(bench)
    equivalence(bench)
    bench.note("Timings: first = the first execution, warm = the median of the others; the CUDA path is timed only "
               "with --device on an actual device. Shared and separate runs give identical answers (asserted).")
    bench.write()


if __name__ == "__main__":
    main()
