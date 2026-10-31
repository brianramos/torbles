# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DELTA-001 benchmark: maintained FND-004 states under signed updates against complete reconstruction.

    python bench/benchmark_math_delta.py --capability DELTA-001 --out FILE [--device] [--quick]

Batches: insert, delete, weight-change and group-move batches of 1 / 10 / 1,000 rows on states of 1k / 100k / 1M rows
(quick: 1k / 100k), 64 groups, two value columns (5% NULLs) and the 6 monomials of degree <= 2. Phases: "build" = the
initial state (scope, FND-004 bank, maintained state) and each delta's construction, timed apart from "run" = each
maintained update (native CPU; CUDA with --device) and its baseline, complete reconstruction (the scope and the
FND-004 bank of the updated rows; DuckDB GROUP BY with SUM(weight * term) in HUGEINT for the 1,000-row batches).
Every update is checked equal to its reconstruction. Group-move frequency: 1,000-row batches on 100k rows where 0 /
50 / 100% of the changed rows also change group. Refusals kept: deletes of rows that are not there (negative
multiplicity, reported with group and need) and updates the overflow proof refuses (then the rebuild path is
timed). Engine tables: a 100k-row table (quick: 20k): selection changes of 1 / 10 / 1,000 rows and value mutations
(t.flip) as diffs of two scopes (Delta.between) against the new scope's complete rebuild.
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_delta as md

TERMS = mbk.monomials(["x", "y"], 2)
G = 64


def data(rng, n):
    return dict(x=rng.integers(-10**6, 10**6, n), y=rng.integers(-1000, 1000, n), vx=rng.random(n) >= 0.05,
                g=rng.integers(0, G, n))


def scope_of(d, mult=None):
    """the FND-004 scope of rows d, each repeated mult times (a weighted bag materialized)"""
    if mult is not None:
        idx = np.repeat(np.arange(len(d["x"])), mult)
        d = {k: v[idx] for k, v in d.items()}
    return mbk.Scope.from_arrays({"x": d["x"], "y": d["y"]}, valid={"x": d["vx"]}, groups=d["g"], ngroups=G)


def delta_of(rows):
    """a delta from (x, y, vx, g, w) arrays"""
    x, y, vx, g, w = rows
    return md.Delta.from_arrays({"x": x, "y": y}, valid={"x": vx}, groups=g, ngroups=G, weights=w)


def duck_reconstruction(con, d, mult):
    x = pd.array(d["x"], dtype="Int64")                       # exact BIGINT with NULLs (never floats)
    x[~d["vx"]] = pd.NA
    con.register("bag", pd.DataFrame({"g": d["g"], "w": mult, "x": x, "y": d["y"]}))
    return con.execute("SELECT g, SUM(w), SUM(w) FILTER (WHERE x IS NOT NULL), SUM(CAST(w AS HUGEINT) * x), "
                       "SUM(CAST(w AS HUGEINT) * y), SUM(CAST(w AS HUGEINT) * x * x) FROM bag GROUP BY g").fetchall()


def change(rng, d, kind, size, move_fraction=None):
    """(delta rows, the updated data, the updated multiplicities) for one batch on data d (multiplicity 1 each)"""
    n = len(d["x"])
    mult = np.ones(n, dtype=np.int64)
    pick = rng.choice(n, size, replace=False)
    if kind == "insert":
        new = data(rng, size)
        out = {k: np.concatenate([d[k], new[k]]) for k in d}
        rows = (new["x"], new["y"], new["vx"], new["g"], np.ones(size, np.int64))
        return rows, out, np.concatenate([mult, np.ones(size, np.int64)])
    if kind == "delete":
        mult[pick] = 0
        rows = (d["x"][pick], d["y"][pick], d["vx"][pick], d["g"][pick], -np.ones(size, np.int64))
        return rows, d, mult
    if kind == "weight":
        w = rng.integers(0, 4, size)
        w[w == 1] = 2
        mult[pick] = w
        rows = (d["x"][pick], d["y"][pick], d["vx"][pick], d["g"][pick], (w - 1).astype(np.int64))
        return rows, d, mult
    # moves: delete the old version and insert the new one (a new group for move_fraction of them, new values)
    frac = 1.0 if move_fraction is None else move_fraction
    out = {k: v.copy() for k, v in d.items()}
    moving = rng.random(size) < frac
    out["g"][pick[moving]] = (d["g"][pick[moving]] + 1 + rng.integers(0, G - 1, int(moving.sum()))) % G
    out["x"][pick[~moving]] = rng.integers(-10**6, 10**6, int((~moving).sum()))
    rows = tuple(np.concatenate([a, b]) for a, b in zip(
        (d["x"][pick], d["y"][pick], d["vx"][pick], d["g"][pick], -np.ones(size, np.int64)),
        (out["x"][pick], out["y"][pick], out["vx"][pick], out["g"][pick], np.ones(size, np.int64))))
    return rows, out, mult


def batches(bench, rng):
    import duckdb
    con = duckdb.connect()
    for n in ((1_000, 100_000) if bench.quick else (1_000, 100_000, 1_000_000)):
        d = data(rng, n)
        p0 = dict(rows=n, groups=G, terms=len(TERMS))
        scope = bench.measure("initial scope", "cpu", lambda: scope_of(d), repeats=2, phase="build", **p0)
        bank = bench.measure("initial FND-004 bank", "cpu", lambda: mbk.build(scope, TERMS), repeats=2, phase="build", **p0)
        state = bench.measure("initial maintained state", "cpu", lambda: md.State.from_bank(bank), repeats=2,
                              phase="build", **p0)
        bench.records[-1]["state_bytes"] = state.info["bytes"]
        for kind in ("insert", "delete", "weight", "move"):
            for size in (1, 10, 1000):
                rows, after, mult = change(rng, d, kind, size)
                p = dict(p0, kind=kind, batch=size)
                delta = bench.measure("delta construction", "cpu", lambda: delta_of(rows), repeats=3, phase="build", **p)
                for backend in bench.backends():
                    new = bench.measure("maintained update", backend, lambda: state.apply(delta, backend=backend),
                                        repeats=3, phase="run", **p)
                rebuilt = bench.measure("complete reconstruction (scope + bank)", "reconstruction",
                                        lambda: mbk.build(scope_of(after, mult), TERMS), repeats=2, phase="run", **p)
                if not new.equal_bank(rebuilt):
                    raise SystemExit(f"DELTA-001: an update differs from its reconstruction ({p})")
                if size == 1000:
                    got = bench.measure("complete reconstruction (DuckDB GROUP BY)", "duckdb",
                                        lambda: duck_reconstruction(con, after, mult), repeats=2, phase="run", **p)
                    for r in got:
                        s = new.state(int(r[0]))
                        if [s["rows"], s["count"][1], s["sum"][1], s["sum"][2], s["sum"][3]] != [int(v or 0) for v in r[1:]]:
                            raise SystemExit(f"DELTA-001: an update differs from DuckDB ({p})")
        # refusals: a delete of rows that are not there; an update the proof refuses, then the rebuild path
        absent = md.Delta.from_arrays({"x": [7], "y": [7]}, groups=[0], ngroups=G, weights=[-(n + 1)])

        def negative():
            try:
                state.apply(absent)
            except md.NegativeMultiplicityError as e:
                return f"refused: group {e.group}, need {e.need}, have {e.have}"
            raise SystemExit("DELTA-001: a negative multiplicity was accepted")
        bench.measure("refusal: delete of rows that are not there", "cpu", negative, repeats=3, phase="run", **p0)
    big = 1 << 61
    tight = mbk.Scope.from_arrays({"x": np.full(31, big), "y": np.zeros(31, np.int64)}, groups=np.zeros(31, np.int64),
                                  ngroups=G)
    st = md.State.from_bank(mbk.build(tight, TERMS))
    over = delta_of((np.array([big, big]), np.zeros(2, np.int64), np.ones(2, bool), np.zeros(2, np.int64),
                     np.array([1, -1])))

    def refused():
        try:
            st.apply(over)
        except OverflowError:
            return "refused: the overflow proof does not hold"
        raise SystemExit("DELTA-001: the proof accepted an update past its bound")
    bench.measure("refusal: overflow proof (net-zero swap at 31 * 2^122)", "cpu", refused, repeats=3, phase="run",
                  rows=31)
    bench.measure("rebuild after the refusal", "reconstruction",
                  lambda: mbk.build(mbk.Scope.from_arrays({"x": np.full(31, big), "y": np.zeros(31, np.int64)},
                                                          groups=np.zeros(31, np.int64), ngroups=G), TERMS),
                  repeats=3, phase="run", rows=31)


def move_frequency(bench, rng):
    n = 100_000
    d = data(rng, n)
    state = md.State.from_bank(mbk.build(scope_of(d), TERMS))
    for frac in (0.0, 0.5, 1.0):
        rows, after, mult = change(rng, d, "move", 1000, move_fraction=frac)
        p = dict(rows=n, batch=1000, move_fraction=frac)
        delta = delta_of(rows)
        bench.records.append(dict(case="delta shape", backend="cpu", phase="measure", params=p, **{
            k: v for k, v in delta.info.items() if k in ("rows", "touched", "inserted", "deleted")}))
        for backend in bench.backends():
            new = bench.measure("maintained update", backend, lambda: state.apply(delta, backend=backend), repeats=3,
                                phase="run", **p)
        rebuilt = bench.measure("complete reconstruction (scope + bank)", "reconstruction",
                                lambda: mbk.build(scope_of(after, mult), TERMS), repeats=2, phase="run", **p)
        if not new.equal_bank(rebuilt):
            raise SystemExit(f"DELTA-001: a move batch differs from its reconstruction ({p})")


def engine_table(bench, rng):
    n = 20_000 if bench.quick else 100_000
    nulls = rng.random(n) < 0.05
    frame = pd.DataFrame({"k": rng.integers(0, G, n),
                          "x": pd.Series([None if z else int(v) for z, v in zip(nulls, rng.integers(-10**6, 10**6, n))],
                                         dtype=object),
                          "y": rng.integers(-1000, 1000, n)})
    db = torbles.from_ddl("CREATE TABLE f (k BIGINT, x BIGINT, y BIGINT)", {"f": frame})
    t = db["f"]
    sel = rng.random(n) < 0.8
    scope = mbk.Scope.from_table(t, ["x", "y"], group_by=["k"], rows=sel)
    state = md.State.from_bank(mbk.build(scope, TERMS))
    for size in (1, 10, 1000):
        new_sel = sel.copy()
        flip = rng.choice(n, size, replace=False)
        new_sel[flip] = ~new_sel[flip]
        p = dict(rows=n, changed=size, change="selection")
        new_scope = bench.measure("new scope (engine decode)", "cpu",
                                  lambda: mbk.Scope.from_table(t, ["x", "y"], group_by=["k"], rows=new_sel),
                                  repeats=2, phase="build", **p)
        delta = bench.measure("diff of two scopes", "cpu", lambda: md.Delta.between(scope, new_scope), repeats=3,
                              phase="build", **p)
        for backend in bench.backends():
            new = bench.measure("maintained update", backend, lambda: state.apply(delta, backend=backend), repeats=3,
                                phase="run", **p)
        rebuilt = bench.measure("complete reconstruction (bank of the new scope)", "reconstruction",
                                lambda: mbk.build(new_scope, TERMS), repeats=2, phase="run", **p)
        if not (new.equal_bank(rebuilt) and new.bound_to(new_scope)):
            raise SystemExit(f"DELTA-001: an engine-table update differs from its reconstruction ({p})")
    plane = next(q for q in range(t.nplanes()) if _changes(t, q, "x"))
    for size in (1, 10):                                                  # value mutations: flips of a plane of x
        rows = rng.choice(np.flatnonzero(sel), size, replace=False)
        for r in rows:
            t.flip(plane, int(r))
        p = dict(rows=n, changed=size, change="t.flip of x")
        new_scope = mbk.Scope.from_table(t, ["x", "y"], group_by=["k"], rows=sel)
        delta = bench.measure("diff of two scopes", "cpu", lambda: md.Delta.between(scope, new_scope), repeats=3,
                              phase="build", **p)
        new = bench.measure("maintained update", "cpu", lambda: state.apply(delta), repeats=3, phase="run", **p)
        rebuilt = bench.measure("complete reconstruction (bank of the new scope)", "reconstruction",
                                lambda: mbk.build(new_scope, TERMS), repeats=2, phase="run", **p)
        if not new.equal_bank(rebuilt):
            raise SystemExit(f"DELTA-001: a mutation differs from its reconstruction ({p})")
        state, scope = new, new_scope


def _changes(t, plane, column):
    before = mbk._decode(t, column).copy()
    t.flip(plane, 0)
    after = mbk._decode(t, column).copy()
    t.flip(plane, 0)
    return not np.array_equal(before, after)


def main():
    bench = Bench.from_args("DELTA-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    batches(bench, rng)
    move_frequency(bench, rng)
    engine_table(bench, rng)
    bench.note("Backends: cpu = math_delta.c (single-threaded), cuda = math_delta_gpu.cu (device runs only), "
               "reconstruction = the FND-004 scope and bank of the updated rows (math_bank.c), duckdb = GROUP BY with "
               "SUM(weight * term) in HUGEINT over the updated bag (DuckDB's own threads).")
    bench.note("An update costs the delta's rows times the terms plus one copy of the state's cells (groups x terms: "
               "the new state is an independent immutable object); it does not depend on the rows of the state. A "
               "reconstruction rescans every row. The initial build is timed apart from each update.")
    bench.note("Group moves are a delete and an insert of the moved rows: a batch of k moves is 2k delta rows; the "
               "move frequency changes the touched groups, not the work per row.")
    bench.note("Bilinear joins (one versus both changing sides of a join) belong to DELTA-002 (a later wave): DELTA-001 "
               "maintains linear (single-table) states, so the join cases of the inventory's second benchmark case are "
               "not measured here; the group-move frequency and the comparison of every update with complete "
               "reconstruction, refusals and rebuilds included, are.")
    bench.write()


if __name__ == "__main__":
    main()
