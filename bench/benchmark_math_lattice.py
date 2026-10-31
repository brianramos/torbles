# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LATTICE-001 / LATTICE-002 benchmark: bounded Boolean occupancy histograms and their subset-lattice transforms.

    python bench/benchmark_math_lattice.py --capability LATTICE-001 | LATTICE-002 --out FILE [--device] [--quick]

Case 1 (histograms): k = 3 / 6 / 10 / 12 conditions over 200k rows (40k with --quick) at three predicate overlaps
(independent: bits of uniform columns, every cell occupied; nested: thresholds on one column, only k + 1 occupied
cells; mixed: half of each) and two selections (every row; a sparse 1% selection: occupancy sparsity), under both NULL
conventions. Reported separately: the condition masks (state construction), the native histogram scan (first and
warm), the 2^k (3^k) state bytes and the occupied cells; baselines: NumPy (cell ids from the condition arrays and
np.bincount) and DuckDB (GROUP BY of the condition values). Every baseline's cells are compared with the native ones.
Case 2 (requests against one histogram): for k = 10 / 12 / 16 (10 with --quick), 1 (a set of k / 2 conditions) /
100 / all 2^k intersection requests answered one by one from the histogram (direct superset sums), by one zeta
transform followed by lookups, and (1 and 100 requests) by DuckDB COUNT(*) per request; the same for coalition
(Moebius) coefficients and for inclusion-exclusion union counts. The cases where the full-domain transform loses to
the direct sums are kept and noted. Then the refusal thresholds (histogram states, transform tensor), each measured
at the last admitted and the first refused point. CUDA is timed only with --device. Both capabilities run both cases
(the inventory gives them the same benchmark cases).
"""
import random

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_lattice as ml
from torbles.math_budgets import Budget


def table(rng, rows):
    u = [rng.integers(0, 1 << 16, rows) for _ in range(3)]
    v = rng.integers(0, 1000, rows)
    w = [None if m else int(x) for x, m in zip(rng.integers(-100, 101, rows), rng.random(rows) < 0.1)]
    frame = pd.DataFrame({"u0": u[0], "u1": u[1], "u2": u[2], "v": v, "w": pd.Series(w, dtype=object)})
    ddl = "CREATE TABLE t (u0 BIGINT, u1 BIGINT, u2 BIGINT, v BIGINT, w BIGINT)"
    return torbles.from_ddl(ddl, {"t": frame}), dict(u0=u[0], u1=u[1], u2=u[2], v=v)


def specs(k, overlap):
    """(column, op, value, sql) of k conditions: independent bits, nested thresholds, or a mix"""
    indep = [(f"u{i % 3}", "bit", i // 3) for i in range(k)]
    nested = [("v", ">", (i + 1) * 1000 // (k + 2)) for i in range(k)]
    chosen = indep if overlap == "independent" else nested if overlap == "nested" else \
        [indep[i] if i % 2 else nested[i] for i in range(k)]
    out = []
    for col, op, val in chosen:
        sql = f'((("{col}" >> {val}) & 1) = 1)' if op == "bit" else f'("{col}" {op} {val})'
        out.append((col, op, val, sql))
    return out


def condition(spec):
    col, op, val, _ = spec
    return mb.condition(col, "bit", val) if op == "bit" else mb.condition(col, op, val)


def numpy_cells(src, sp, sel):
    bits = []
    for col, op, val, _ in sp:
        x = src[col]
        bits.append(((x >> val) & 1).astype(bool) if op == "bit" else x > val)
    ids = np.zeros(len(src["v"]), dtype=np.int64)
    for i, b in enumerate(bits):
        ids |= b.astype(np.int64) << i
    if sel is not None:
        ids = ids[sel]
    return np.bincount(ids, minlength=1 << len(sp))


def duck_cells(con, sp, where):
    cols = ", ".join(f"{s[3]} AS a{i}" for i, s in enumerate(sp))
    rows = con.execute(f"SELECT {cols}, COUNT(*) FROM t WHERE {where} GROUP BY ALL").fetchall()
    cells = [0] * (1 << len(sp))
    for r in rows:
        cells[sum(1 << i for i, b in enumerate(r[:-1]) if b)] += int(r[-1])
    return cells


def case_histograms(bench, db, src, rows):
    con = db.con
    nrng = np.random.default_rng(bench.seed)
    sparse = nrng.random(rows) < 0.01
    con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sparse).astype(np.int64)}))
    ks = (3, 6, 10) if bench.quick else (3, 6, 10, 12)
    for label, sel in (("every row", None), ("sparse 1%", sparse)):
        scope = mbk.Scope.from_table(db["t"], ["u0", "u1", "u2", "v", "w"], rows=sel)
        where = "TRUE" if sel is None else "rowid IN (SELECT rid FROM sel_rows)"
        for k in ks:
            for overlap in ("independent", "nested", "mixed"):
                sp = specs(k, overlap)
                conds = [condition(s) for s in sp]
                p = dict(rows=rows, selected=scope.selected, conditions=k, overlap=overlap, selection=label)
                for backend in bench.backends():
                    bench.measure("condition masks (state construction)", backend,
                                  lambda: mb.Masks.from_scope(scope, conds, backend=backend).info["bytes"],
                                  repeats=2, phase="build", **p)
                masks = mb.Masks.from_scope(scope, conds)
                for nulls in ("apart", "three"):
                    if nulls == "three" and k > 10:
                        continue                              # 3^12 cells: measured in the threshold case
                    for backend in bench.backends():
                        h = bench.measure("occupancy histogram scan", backend,
                                          lambda: ml.Histogram.build(masks, nulls=nulls, backend=backend),
                                          repeats=4, phase="build", nulls=nulls, **p)
                    info = h.info
                    bench.note(f"{label}, k={k}, {overlap}, {nulls}: {info['cells']} cells, {info['occupied']} "
                               f"occupied, state bytes {info['bytes']}")
                    bench.records[-1]["result"] = dict(cells=info["cells"], occupied=info["occupied"],
                                                       state_bytes=info["bytes"])
                native = ml.Histogram.build(masks).counts().tolist()
                got = bench.measure("histogram: condition arrays + bincount", "numpy",
                                    lambda: numpy_cells(src, sp, sel), repeats=3, **p)
                if got.tolist() != native:
                    raise SystemExit(f"NumPy disagrees at k={k}, {overlap}, {label}")
                duck = bench.measure("histogram: GROUP BY of the condition values", "duckdb",
                                     lambda: duck_cells(con, sp, where), repeats=2, **p)
                if duck != native:
                    raise SystemExit(f"DuckDB disagrees at k={k}, {overlap}, {label}")


def case_requests(bench, db, rows):
    con = db.con
    rng = random.Random(bench.seed)
    scope = mbk.Scope.from_table(db["t"], ["u0", "u1", "u2", "v", "w"])
    for k in ((10,) if bench.quick else (10, 12, 16)):
        sp = specs(k, "independent")
        h = ml.Histogram.build(mb.Masks.from_scope(scope, [condition(s) for s in sp]))
        vec = h.vector("complete")
        for q in (1, 100, 1 << k):
            sets = [list(range(k // 2))] if q == 1 else (
                [[i for i in range(k) if (S >> i) & 1] for S in range(1 << k)] if q == 1 << k else
                [sorted(rng.sample(range(k), rng.randint(1, k))) for _ in range(q)])
            masks_ = [sum(1 << i for i in S) for S in sets]
            p = dict(conditions=k, requests=q, states=1 << k, rows=rows)
            direct = bench.measure("intersection requests: direct superset sums", "cpu",
                                   lambda: ml.coefficients(vec, sets, "zeta_up"), repeats=5, **p)
            d_med = bench.records[-1]["warm_median_seconds"]
            for backend in bench.backends():
                full = bench.measure("intersection requests: one zeta transform + lookups", backend,
                                     lambda: (lambda z: [int(z[m]) for m in masks_])(
                                         ml.zeta(vec, carrier="i64", backend=backend)["values"]), repeats=5, **p)
                if backend == "cpu":
                    t_med = bench.records[-1]["warm_median_seconds"]
                if direct != full:
                    raise SystemExit(f"direct and transformed intersections differ at k={k}, q={q}")
            bench.note(f"k={k}, {q} intersection requests: direct {d_med * 1e3:.3f} ms, zeta transform + lookups "
                       f"{t_med * 1e3:.3f} ms: the full-domain transform {'loses' if t_med > d_med else 'wins'}")
            if q <= 100:
                where = [" AND ".join(sp[i][3] for i in S) or "TRUE" for S in sets]
                duck = bench.measure("intersection requests: one COUNT(*) per request", "duckdb",
                                     lambda: [int(con.execute(f"SELECT COUNT(*) FROM t WHERE {w}").fetchall()[0][0])
                                              for w in where], repeats=1, **p)
                if duck != direct:
                    raise SystemExit(f"DuckDB intersections differ at k={k}, q={q}")
            game = ml.zeta(vec, carrier="i64")["values"]      # a set function (the intersection counts), int64
            dc = bench.measure("coalition requests: direct Moebius sums", "cpu",
                               lambda: ml.coefficients(game, sets, "mobius_down"), repeats=5, **p)
            dc_med = bench.records[-1]["warm_median_seconds"]
            for backend in bench.backends():
                fc = bench.measure("coalition requests: one Moebius transform + lookups", backend,
                                   lambda: (lambda z: [int(z[m]) for m in masks_])(
                                       ml.mobius(game, "down", carrier="i64", backend=backend)["values"]),
                                   repeats=5, **p)
                if backend == "cpu":
                    fc_med = bench.records[-1]["warm_median_seconds"]
                if dc != fc:
                    raise SystemExit(f"coalition coefficients differ at k={k}, q={q}")
            bench.note(f"k={k}, {q} coalition requests: direct {dc_med * 1e3:.3f} ms, Moebius transform + lookups "
                       f"{fc_med * 1e3:.3f} ms: the full-domain transform {'loses' if fc_med > dc_med else 'wins'}")
            if q <= 100:
                bench.measure("inclusion-exclusion union counts", "cpu", lambda: ml.counts(h, sets), repeats=3, **p)
        bench.measure("zeta / Moebius round trip (full domain)", "cpu",
                      lambda: ml.mobius(ml.zeta(vec)["values"])["values"] == vec.tolist(), repeats=3,
                      conditions=k, states=1 << k)


def thresholds(bench, db):
    scope = mbk.Scope.from_table(db["t"], ["u0", "u1", "u2", "v", "w"])
    budget = Budget(states=1 << 12)
    m12 = mb.Masks.from_scope(scope, [condition(s) for s in specs(12, "independent")])
    m13 = mb.Masks.from_scope(scope, [condition(s) for s in specs(13, "independent")])
    bench.measure("histogram at a 2^12 state budget", "cpu", lambda: ml.Histogram.build(m12, budget=budget).k,
                  repeats=1, conditions=12, states=1 << 12)
    bench.refusal("histogram past a 2^12 state budget", "cpu", lambda: ml.Histogram.build(m13, budget=budget),
                  conditions=13, states=1 << 13)
    bench.threshold("histogram conditions (states 2^12)", admitted=dict(conditions=12), refused=dict(conditions=13))
    m7 = mb.Masks.from_scope(scope, [condition(s) for s in specs(7, "independent")])
    m8 = mb.Masks.from_scope(scope, [condition(s) for s in specs(8, "independent")])
    b3 = Budget(states=3 ** 7)
    bench.measure("three-valued histogram at a 3^7 state budget", "cpu",
                  lambda: ml.Histogram.build(m7, nulls="three", budget=b3).k, repeats=1, conditions=7, states=3 ** 7)
    bench.refusal("three-valued histogram past a 3^7 state budget", "cpu",
                  lambda: ml.Histogram.build(m8, nulls="three", budget=b3), conditions=8, states=3 ** 8)
    bench.threshold("three-valued histogram conditions (states 3^7)", admitted=dict(conditions=7),
                    refused=dict(conditions=8))
    tb = Budget(tensor=1 << 16)
    for backend in bench.backends():
        bench.measure("zeta transform at a 2^16 tensor budget", backend,
                      lambda: len(ml.zeta(np.ones(1 << 16, dtype=np.int64), backend=backend, budget=tb)["values"]),
                      repeats=1, conditions=16)
        bench.refusal("zeta transform past a 2^16 tensor budget", backend,
                      lambda: ml.zeta(np.ones(1 << 17, dtype=np.int64), backend=backend, budget=tb), conditions=17)
    bench.threshold("transform variables (tensor 2^16)", admitted=dict(conditions=16), refused=dict(conditions=17))


def main():
    bench = Bench.from_args("LATTICE-001", __doc__)
    rows = 40_000 if bench.quick else 200_000
    db, src = table(np.random.default_rng(bench.seed), rows)
    case_histograms(bench, db, src, rows)
    case_requests(bench, db, rows)
    thresholds(bench, db)
    bench.note("Native CPU paths are single-threaded. The histogram scan visits every selected row once (cell id from "
               "the condition words); the transforms cost k 2^(k-1) butterflies whatever the row count; a direct "
               "request costs 2^(k-|S|) (superset sums) or 2^|S| (subset sums) cell reads.")
    bench.write()


if __name__ == "__main__":
    main()
