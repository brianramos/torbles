# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DELTA-002 benchmark: maintained bilinear join states and nonlinear finalizer states against complete
reconstruction.

    python bench/benchmark_math_bilinear.py --capability DELTA-002 --out FILE [--device] [--quick]

Joins: side A of 1k / 100k / 1M rows (quick: 1k / 100k) on 1,000 keys and 4 output groups, side B of 20k rows on the
same keys; pairs COUNT(*), SUM(a.x), SUM(a.x b.y), SUM(a.x^2 b.y^2). Phases: "build" = the initial state (scopes,
FND-004 banks, DELTA-001 side states, the join) and each delta's construction, apart from "run" = each maintained
update (the side's DELTA-001 update and the join's cross terms; native CPU, CUDA with --device) and its baselines:
complete reconstruction (the side's scope, bank and state of the updated rows and the join rebuilt) and, for the
1,000-row batches, DuckDB re-executing the join aggregate over the updated rows. Insert, delete and weight-change
batches of 1 / 10 / 1,000 rows on side A. Then one versus both changing sides (1,000-row batches on A, on B, on both)
and the key-move frequency (0 / 50 / 100% of the changed A rows move to another key: changing join multiplicities).
Every update is checked equal to its reconstruction. Refusals and rebuilds kept: an update DELTA-001's overflow proof
refuses (then the rebuild is timed), a delete of rows that are not there, and an EXTREMUM state's deleted minimum
(RebuildRequired, then the rebuild of the value state); over-budget build and update of the 100k-row join (states and
iterations one below the exact need: refused with that need; the thresholds recorded, never extrapolated).
Nonlinear finalizers: MULTISET value states of 100k rows under
1 / 10 / 1,000-row batches against their rebuild and DuckDB's COUNT(DISTINCT), MIN, MAX. Dense matrix deltas: G = K = H
= 8 / 32 / 64 against explicit products with Python integers (NumPy object arrays).
"""
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles.math_budgets import Budget
from torbles import math_bank as mbk
from torbles import math_delta as md
from torbles import math_bilinear as mbl

K, GA = 1000, 4
PAIRS = [("1", "1"), ("x", "1"), ("x", "y"), ("x^2", "y^2")]
BASIS_A, BASIS_B = ["1", "x", "x^2"], ["1", "y", "y^2"]


def side_a(rng, n):
    return dict(k=rng.integers(0, K, n), g=rng.integers(0, GA, n), x=rng.integers(-1000, 1000, n),
                vx=rng.random(n) >= 0.05)


def side_b(rng, n):
    return dict(k=rng.integers(0, K, n), y=rng.integers(-100, 100, n))


def scope_a(d, mult=None):
    if mult is not None:
        idx = np.repeat(np.arange(len(d["x"])), mult)
        d = {k: v[idx] for k, v in d.items()}
    return mbk.Scope.from_arrays({"x": d["x"]}, valid={"x": d["vx"]}, groups=d["k"] * GA + d["g"], ngroups=K * GA)


def scope_b(d, mult=None):
    if mult is not None:
        idx = np.repeat(np.arange(len(d["y"])), mult)
        d = {k: v[idx] for k, v in d.items()}
    return mbk.Scope.from_arrays({"y": d["y"]}, groups=d["k"], ngroups=K)


def state_of(scope, basis):
    return md.State.from_bank(mbk.build(scope, basis))


def join_of(sa, sb):
    return mbl.Join.build(sa, sb, keys=K, groups_a=GA, groups_b=1, pairs=PAIRS)


def delta_a(rows):
    k, g, x, vx, w = rows
    return md.Delta.from_arrays({"x": x}, valid={"x": vx}, groups=k * GA + g, ngroups=K * GA, weights=w)


def delta_b(rows):
    k, y, w = rows
    return md.Delta.from_arrays({"y": y}, groups=k, ngroups=K, weights=w)


def change_a(rng, d, kind, size, move=0.0):
    """(delta rows, the updated data, the multiplicities) of one batch on side A (multiplicity 1 each)"""
    n = len(d["x"])
    mult = np.ones(n, dtype=np.int64)
    pick = rng.choice(n, size, replace=False)
    if kind == "insert":
        new = side_a(rng, size)
        out = {k: np.concatenate([d[k], new[k]]) for k in d}
        return (new["k"], new["g"], new["x"], new["vx"], np.ones(size, np.int64)), out, np.ones(n + size, np.int64)
    if kind == "delete":
        mult[pick] = 0
        return (d["k"][pick], d["g"][pick], d["x"][pick], d["vx"][pick], -np.ones(size, np.int64)), d, mult
    if kind == "weight":
        w = rng.integers(0, 4, size)
        w[w == 1] = 2
        mult[pick] = w
        return (d["k"][pick], d["g"][pick], d["x"][pick], d["vx"][pick], (w - 1).astype(np.int64)), d, mult
    out = {k: v.copy() for k, v in d.items()}                    # moves: a new key for a fraction, new x otherwise
    moving = rng.random(size) < move
    out["k"][pick[moving]] = (d["k"][pick[moving]] + 1 + rng.integers(0, K - 1, int(moving.sum()))) % K
    out["x"][pick[~moving]] = rng.integers(-1000, 1000, int((~moving).sum()))
    rows = tuple(np.concatenate([a, b]) for a, b in zip(
        (d["k"][pick], d["g"][pick], d["x"][pick], d["vx"][pick], -np.ones(size, np.int64)),
        (out["k"][pick], out["g"][pick], out["x"][pick], out["vx"][pick], np.ones(size, np.int64))))
    return rows, out, mult


def change_b(rng, d, size):
    n = len(d["y"])
    mult = np.ones(n, dtype=np.int64)
    pick = rng.choice(n, size, replace=False)
    mult[pick] = 0
    new = side_b(rng, size)
    out = {k: np.concatenate([d[k], new[k]]) for k in d}
    rows = (np.concatenate([d["k"][pick], new["k"]]), np.concatenate([d["y"][pick], new["y"]]),
            np.concatenate([-np.ones(size, np.int64), np.ones(size, np.int64)]))
    return rows, out, np.concatenate([mult, np.ones(size, np.int64)])


def duck_join(con, a, ma, b, mb):
    x = pd.array(a["x"], dtype="Int64")
    x[~a["vx"]] = pd.NA
    con.register("A", pd.DataFrame({"k": a["k"], "g": a["g"], "w": ma, "x": x}))
    con.register("B", pd.DataFrame({"k": b["k"], "w": mb, "y": b["y"]}))
    return con.execute(
        "SELECT a.g, SUM(CAST(a.w AS HUGEINT) * b.w), SUM(CAST(a.w AS HUGEINT) * b.w * a.x), "
        "SUM(CAST(a.w AS HUGEINT) * b.w * a.x * b.y) FROM A a JOIN B b ON a.k = b.k GROUP BY a.g ORDER BY a.g").fetchall()


def joins(bench, rng):
    import duckdb
    con = duckdb.connect()
    db_ = side_b(rng, 20_000)
    sb = state_of(scope_b(db_), BASIS_B)
    for n in ((1_000, 100_000) if bench.quick else (1_000, 100_000, 1_000_000)):
        da_ = side_a(rng, n)
        p0 = dict(rows_a=n, rows_b=20_000, keys=K, groups_a=GA, pairs=len(PAIRS))
        sca = bench.measure("initial scope (side A)", "cpu", lambda: scope_a(da_), repeats=2, phase="build", **p0)
        sa = bench.measure("initial side state (bank + DELTA-001 state)", "cpu", lambda: state_of(sca, BASIS_A),
                           repeats=2, phase="build", **p0)
        j = bench.measure("initial join state", "cpu", lambda: join_of(sa, sb), repeats=2, phase="build", **p0)
        if "cuda" in bench.backends():
            bench.measure("initial join state", "cuda", lambda: mbl.Join.build(
                sa, sb, keys=K, groups_a=GA, pairs=PAIRS, backend="cuda"), repeats=2, phase="build", **p0)
        bench.records[-1]["join_bytes"] = j.info["bytes"]
        for kind in ("insert", "delete", "weight"):
            for size in (1, 10, 1000):
                rows, after, mult = change_a(rng, da_, kind, size)
                p = dict(p0, kind=kind, batch=size, side="A")
                delta = bench.measure("delta construction", "cpu", lambda: delta_a(rows), repeats=3, phase="build", **p)
                for backend in bench.backends():
                    new = bench.measure("maintained join update (side A + cross terms)", backend,
                                        lambda: j.apply(a=delta, backend=backend), repeats=3, phase="run", **p)
                rebuilt = bench.measure("complete reconstruction (side A + join)", "reconstruction",
                                        lambda: join_of(state_of(scope_a(after, mult), BASIS_A), sb), repeats=2,
                                        phase="run", **p)
                if not new.equal(rebuilt):
                    raise SystemExit(f"DELTA-002: an update differs from its reconstruction ({p})")
                if size == 1000:
                    got = bench.measure("complete reconstruction (DuckDB join aggregate)", "duckdb",
                                        lambda: duck_join(con, after, mult, db_, np.ones(20_000, np.int64)),
                                        repeats=2, phase="run", **p)
                    for r in got:
                        c = new.cell(int(r[0]), 0)
                        if [c["rows"], c["sum"][1], c["sum"][2]] != [int(v or 0) for v in r[1:]]:
                            raise SystemExit(f"DELTA-002: an update differs from DuckDB ({p})")
        if n == 100_000:
            sides(bench, rng, da_, sa, db_, sb, j, p0)
            budgets(bench, rng, da_, sa, sb, j, p0)
    return db_, sb


def budgets(bench, rng, da_, sa, sb, j, p0):
    """over-budget build and update: admitted at the exact need, refused one below with that need"""
    need = mbl.Join.need(sa, sb, keys=K, groups_a=GA, groups_b=1, pairs=PAIRS)["states"]
    if not mbl.Join.build(sa, sb, keys=K, groups_a=GA, pairs=PAIRS, budget=Budget(states=need)).equal(j):
        raise SystemExit("DELTA-002: a build admitted at its exact need differs")
    rec = bench.refusal("over-budget join build (states = cells x values - 1)", "cpu", lambda: mbl.Join.build(
        sa, sb, keys=K, groups_a=GA, pairs=PAIRS, budget=Budget(states=need - 1)), **p0)
    if rec.get("need") != need:
        raise SystemExit("DELTA-002: an over-budget build did not report its exact need")
    bench.threshold("join build states (cells x values)", admitted=dict(states=need), refused=dict(states=need - 1),
                    **p0)
    delta = delta_a(change_a(rng, da_, "insert", 1000)[0])
    sa2 = sa.apply(delta)                                        # the side's own (DELTA-001) budget apart
    need = j.update_need(a=delta, new_a=sa2)["iterations"]
    if not j.apply(a=delta, new_a=sa2, budget=Budget(iterations=need)).equal(j.apply(a=delta)):
        raise SystemExit("DELTA-002: an update admitted at its exact need differs")
    p = dict(p0, kind="insert", batch=1000)
    rec = bench.refusal("over-budget join update (iterations = touched keys x GA x GB x values - 1)", "cpu",
                        lambda: j.apply(a=delta, new_a=sa2, budget=Budget(iterations=need - 1)), **p)
    if rec.get("need") != need:
        raise SystemExit("DELTA-002: an over-budget update did not report its exact need")
    bench.threshold("join update iterations (touched keys x GA x GB x values)", admitted=dict(iterations=need),
                    refused=dict(iterations=need - 1), **p)


def sides(bench, rng, da_, sa, db_, sb, j, p0):
    """one versus both changing sides, key-move frequency (changing join multiplicities)"""
    for which in ("A", "B", "both"):
        ra = change_a(rng, da_, "move", 1000, 0.5) if which in ("A", "both") else None
        rb = change_b(rng, db_, 500) if which in ("B", "both") else None
        dA = delta_a(ra[0]) if ra else None
        dB = delta_b(rb[0]) if rb else None
        p = dict(p0, batch=1000, changing=which)
        for backend in bench.backends():
            new = bench.measure("maintained join update", backend, lambda: j.apply(a=dA, b=dB, backend=backend),
                                repeats=3, phase="run", **p)
        terms = new.terms
        bench.records[-1]["cross_terms_nonzero"] = {t: sum(1 for c in terms[t] if c["rows"] or any(c["sum"]))
                                                    for t in terms}
        bench.records[-1]["touched_keys"] = new.info["touched"]
        rebuilt = bench.measure("complete reconstruction (changed sides + join)", "reconstruction",
                                lambda: join_of(state_of(scope_a(ra[1], ra[2]), BASIS_A) if ra else sa,
                                                state_of(scope_b(rb[1], rb[2]), BASIS_B) if rb else sb),
                                repeats=2, phase="run", **p)
        if not new.equal(rebuilt):
            raise SystemExit(f"DELTA-002: a two-sided update differs from its reconstruction ({p})")
    for frac in (0.0, 0.5, 1.0):
        rows, after, mult = change_a(rng, da_, "move", 1000, frac)
        p = dict(p0, batch=1000, move_fraction=frac)
        delta = delta_a(rows)
        for backend in bench.backends():
            new = bench.measure("maintained join update (key moves)", backend, lambda: j.apply(a=delta, backend=backend),
                                repeats=3, phase="run", **p)
        bench.records[-1]["touched_keys"] = new.info["touched"]
        rebuilt = bench.measure("complete reconstruction (side A + join)", "reconstruction",
                                lambda: join_of(state_of(scope_a(after, mult), BASIS_A), sb), repeats=2, phase="run", **p)
        if not new.equal(rebuilt):
            raise SystemExit(f"DELTA-002: a move batch differs from its reconstruction ({p})")


def refusals(bench, rng, sb):
    big = 1 << 61
    zeros = mbk.Scope.from_arrays({"x": np.zeros(K * GA, np.int64)}, groups=np.arange(K * GA), ngroups=K * GA)
    tight = md.State.from_bank(mbk.build(zeros, BASIS_A)).apply(
        md.Delta.from_arrays({"x": np.full(31, big)}, groups=np.zeros(31, np.int64), ngroups=K * GA))
    j = join_of(tight, sb)
    over = md.Delta.from_arrays({"x": [big, big]}, groups=[0, 0], ngroups=K * GA, weights=[1, -1])

    def refused():
        try:
            j.apply(a=over)
        except OverflowError:
            return "refused: DELTA-001's proof (net-zero swap at 31 * 2^122)"
        raise SystemExit("DELTA-002: the side proof accepted an update past its bound")
    bench.measure("refusal: side overflow proof", "cpu", refused, repeats=3, phase="run", rows=31)
    bench.measure("rebuild after the refusal (side + join)", "reconstruction", lambda: join_of(
        md.State.from_bank(mbk.build(zeros, BASIS_A)).apply(
            md.Delta.from_arrays({"x": np.full(31, big)}, groups=np.zeros(31, np.int64), ngroups=K * GA)), sb),
        repeats=3, phase="run", rows=31)
    absent = md.Delta.from_arrays({"y": [7]}, groups=[3], ngroups=K, weights=[-(10 ** 6)])

    def negative():
        try:
            j.apply(b=absent)
        except md.NegativeMultiplicityError as e:
            return f"refused: group {e.group}, need {e.need}"
        raise SystemExit("DELTA-002: a negative multiplicity was accepted")
    bench.measure("refusal: delete of B rows that are not there", "cpu", negative, repeats=3, phase="run")


def values(bench, rng):
    import duckdb
    con = duckdb.connect()
    n, G = 100_000, 64
    x, g = rng.integers(-500, 500, n), rng.integers(0, G, n)
    vx = rng.random(n) >= 0.05
    scope = mbk.Scope.from_arrays({"x": x}, valid={"x": vx}, groups=g, ngroups=G)
    p0 = dict(rows=n, groups=G)
    ms = bench.measure("initial MULTISET value state", "cpu", lambda: mbl.Values.from_scope(scope, "x"), repeats=2,
                       phase="build", **p0)
    for size in (1, 10, 1000):
        pick = rng.choice(n, size, replace=False)
        mult = np.ones(n, np.int64)
        mult[pick] = 0
        d = md.Delta.from_arrays({"x": x[pick]}, valid={"x": vx[pick]}, groups=g[pick], ngroups=G,
                                 weights=-np.ones(size, np.int64))
        p = dict(p0, kind="delete", batch=size)
        new = bench.measure("maintained MULTISET update (COUNT DISTINCT, MIN, MAX)", "cpu", lambda: ms.apply(d),
                            repeats=3, phase="run", **p)
        keep = mult.astype(bool)
        rebuilt = bench.measure("complete reconstruction (value state)", "reconstruction", lambda: mbl.Values.from_scope(
            mbk.Scope.from_arrays({"x": x[keep]}, valid={"x": vx[keep]}, groups=g[keep], ngroups=G), "x"),
            repeats=2, phase="run", **p)
        if not new.equal(rebuilt):
            raise SystemExit(f"DELTA-002: a value-state update differs from its reconstruction ({p})")
        if size == 1000:
            xs = pd.array(x[keep], dtype="Int64")
            xs[~vx[keep]] = pd.NA
            con.register("v", pd.DataFrame({"g": g[keep], "x": xs}))
            got = bench.measure("COUNT(DISTINCT), MIN, MAX (DuckDB)", "duckdb", lambda: con.execute(
                "SELECT g, COUNT(DISTINCT x), MIN(x), MAX(x) FROM v GROUP BY g ORDER BY g").fetchall(), repeats=2,
                phase="run", **p)
            for r in got:
                if (new.finalize(int(r[0]), "count_distinct"), new.finalize(int(r[0]), "min"),
                        new.finalize(int(r[0]), "max")) != (int(r[1]), int(r[2]), int(r[3])):
                    raise SystemExit("DELTA-002: a value state differs from DuckDB")
    ex = mbl.Values.from_scope(scope, "x", "extremum")
    gv = ex.group(0)
    kill = md.Delta.from_arrays({"x": [gv["min"]]}, groups=[0], ngroups=G, weights=[-gv["min_count"]])

    def rebuild_needed():
        try:
            ex.apply(kill)
        except mbl.RebuildRequired as e:
            return f"refused: group {e.group}, value {e.value}: rebuild"
        return "accepted (no unknown value left)"
    bench.measure("refusal: EXTREMUM minimum deleted", "cpu", rebuild_needed, repeats=3, phase="run", **p0)
    bench.measure("rebuild of the EXTREMUM state", "reconstruction",
                  lambda: mbl.Values.from_scope(scope, "x", "extremum"), repeats=2, phase="run", **p0)


def matrices(bench, rng):
    for m in (8, 32, 64):
        A, B = rng.integers(-10**9, 10**9, (m, m)), rng.integers(-10**9, 10**9, (m, m))
        dA, dB = rng.integers(-10**9, 10**9, (m, m)), rng.integers(-10**9, 10**9, (m, m))
        p = dict(G=m, K=m, H=m)
        for backend in bench.backends():
            got = bench.measure("matrix-product delta (AB, dA B, A dB, dA dB, updated)", backend,
                                lambda: mbl.matrix_delta(A, B, dA, dB, backend=backend), repeats=3, phase="run", **p)
        Ao, Bo, dAo, dBo = (M.astype(object) for M in (A, B, dA, dB))
        want = bench.measure("explicit products (Python ints, NumPy object arrays)", "numpy",
                             lambda: ((Ao + dAo).dot(Bo + dBo), Ao.dot(Bo), dAo.dot(Bo), Ao.dot(dBo), dAo.dot(dBo)),
                             repeats=2, phase="run", **p)
        if got["updated"] != want[0].tolist() or got["dA_dB"] != want[4].tolist():
            raise SystemExit("DELTA-002: a matrix delta differs from the explicit product")


def main():
    bench = Bench.from_args("DELTA-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    db_, sb = joins(bench, rng)
    refusals(bench, rng, sb)
    values(bench, rng)
    matrices(bench, rng)
    bench.note("Backends: cpu = math_bilinear.c with DELTA-001's math_delta.c for the side (single-threaded), cuda = "
               "math_bilinear_gpu.cu and math_delta_gpu.cu (device runs only), reconstruction = the FND-004 scope, bank "
               "and DELTA-001 state of the updated rows and the join rebuilt from the side states, duckdb = the join "
               "aggregate re-executed over the updated rows (DuckDB's own threads), numpy = explicit products of "
               "Python ints in object arrays.")
    bench.note("A join update costs the side's DELTA-001 update (one copy of the side's cells plus the delta's rows) "
               "and the cross terms of the touched keys (touched keys x GA x GB x values); a reconstruction rescans "
               "every row of the changed side and rebuilds the join over every key.")
    bench.note("Changing join multiplicities: a key move is a delete at the old key and an insert at the new one; the "
               "other side's per-key cells give the changed matches (the T2 / T1 terms); both-side updates add the "
               "dA dB term of keys touched on both sides.")
    bench.write()


if __name__ == "__main__":
    main()
