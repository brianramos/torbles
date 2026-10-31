# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LIN-001 / LIN-002 benchmark: grouped exact moment matrices (assembly, bytes) and their finalization (exact solves,
inertia, eigensystems), with repeated ridge / deletion requests against complete reconstruction.

    python bench/benchmark_math_lin.py --capability LIN-001 --out FILE [--device] [--quick]
    python bench/benchmark_math_lin.py --capability LIN-002 --out FILE [--device] [--quick]

LIN-001: feature dimensions 2 / 4 / 8 / 16, groups 1 / 16 / 256, rows 1k / 100k / 1M (int64 values with 5% NULLs, an
array scope): scope construction ("build"), the fused moment state (native CPU, listwise; pairwise and the FND-004
bank path at selected points; CUDA with --device), its bytes, and the baselines DuckDB (GROUP BY with SUM of HUGEINT
products and FILTER for listwise rows: exact; checked equal to the state) and NumPy (float64 per-group Z^T Z: not
exact, a speed reference). The DuckDB baseline is trimmed at 1M rows to d <= 8 (the 171-aggregate query at d = 16
takes tens of seconds per run); every trimmed point is recorded in the notes.
LIN-002: per (d, groups) at 100k rows the exact fits of every group (CPU fraction-free elimination; with --device the
batched modular elimination on CUDA), the exact inertia, PCA (Jacobi with certified bounds), separately from the
moment assembly; well-conditioned, nearly singular and exactly rank-deficient designs (kind, rank, and NumPy lstsq's
distance from the exact coefficients); repeated requests from one retained state: a ridge sweep, feature deletions
and a row deletion (state subtraction), against complete reconstruction (rebuild the state from the rows each time)
and NumPy (float64 normal equations from the rows each time).
"""
import numpy as np

from math_bench import Bench
from fractions import Fraction as F
import duckdb
import pandas as pd

from torbles import math_bank as mbk
from torbles import math_lin as lin
from torbles.math_budgets import Budget


def data(rng, rows, d, groups, kind="random"):
    cols = {f"x{i}": rng.integers(-10**6, 10**6, rows) for i in range(d)}
    if kind == "nearly":                                    # x_{d-1} = x_0 + rare unit noise: nearly singular
        cols[f"x{d - 1}"] = cols["x0"] + rng.integers(-1, 2, rows) * (rng.random(rows) < 0.01)
    elif kind == "deficient":                               # x_{d-1} = x_0 + x_1: exactly rank deficient
        cols[f"x{d - 1}"] = cols["x0"] + cols["x1"]
    beta = rng.integers(-5, 6, d)
    cols["y"] = sum(beta[i] * cols[f"x{i}"] for i in range(d)) // 7 + rng.integers(-1000, 1000, rows)
    valid = {f"x{i}": rng.random(rows) >= 0.05 for i in range(d)} if kind == "random" else {}
    g = rng.integers(0, groups, rows) if groups > 1 else None
    return cols, valid, g


def scope_of(cols, valid, g, groups):
    return mbk.Scope.from_arrays(cols, valid=valid, groups=g, ngroups=groups)


def duck_sql(d, listwise):
    z = ["CAST(1 AS HUGEINT)"] + [f"CAST(x{i} AS HUGEINT)" for i in range(d)] + ["CAST(y AS HUGEINT)"]
    cond = " AND ".join(f"x{i} IS NOT NULL" for i in range(d))
    parts = []
    for a in range(len(z)):
        for b in range(a, len(z)):
            e = f"{z[a]} * {z[b]}"
            parts.append(f"SUM({e}) FILTER (WHERE {cond})" if listwise else f"SUM({e})")
    return f"SELECT g, COUNT(*), {', '.join(parts)} FROM t GROUP BY g ORDER BY g"


def frame_of(cols, valid, g, rows):
    out = {}
    for name, v in cols.items():
        s = pd.array(v, dtype="Int64")
        if name in valid:
            s[~valid[name]] = pd.NA
        out[name] = s
    out["g"] = np.zeros(rows, np.int64) if g is None else g.astype(np.int64)
    return pd.DataFrame(out)


def check(st, result, d):
    k = d + 2
    for row in result:
        g = int(row[0])
        m = st.matrix(g)
        got = [m[a][b] for a in range(k) for b in range(a, k)]
        want = [0 if v is None else int(v) for v in row[2:]]
        if got != want or st.state(g)["rows"] != int(row[1]):
            raise SystemExit(f"LIN-001 moment state differs from DuckDB (d={d}, group {g})")


def numpy_gram(cols, valid, g, groups, d):
    Z = np.stack([np.ones(len(cols["y"]))] + [cols[f"x{i}"].astype(np.float64) for i in range(d)] +
                 [cols["y"].astype(np.float64)], 1)
    ok = np.ones(len(cols["y"]), bool)
    for v in valid.values():
        ok &= v
    gid = np.zeros(len(ok), np.int64) if g is None else g
    return [Z[ok & (gid == q)].T @ Z[ok & (gid == q)] for q in range(groups)]


def lin001(bench, rng):
    grid_rows = (1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)
    grid_d = (2, 16) if bench.quick else (2, 4, 8, 16)
    grid_g = (1, 256) if bench.quick else (1, 16, 256)
    con = duckdb.connect()
    for rows in grid_rows:
        for d in grid_d:
            for groups in grid_g:
                cols, valid, g = data(rng, rows, d, groups)
                feats = [f"x{i}" for i in range(d)]
                p = dict(rows=rows, features=d, groups=groups)
                scope = bench.measure("scope", "cpu", lambda: scope_of(cols, valid, g, groups), repeats=2, phase="build", **p)
                st = None
                for backend in bench.backends():
                    st = bench.measure("moments listwise", backend, lambda: lin.moments(scope, feats, "y", backend=backend),
                                       repeats=3, phase="run", **p)
                bench.records.append(dict(case="state bytes", backend="cpu", phase="measure", params=p, bytes=st.bytes,
                                          slots=st.nslots, width=st.width, scope_bytes=scope.bytes))
                if groups == grid_g[-1] and rows == grid_rows[-1] or (rows == 100_000 and groups == 16):
                    for backend in bench.backends():
                        sp = bench.measure("moments pairwise", backend,
                                           lambda: lin.moments(scope, feats, "y", policy="pairwise", backend=backend),
                                           repeats=2, phase="run", **p)
                    bench.records.append(dict(case="state bytes pairwise", backend="cpu", phase="measure", params=p,
                                              bytes=sp.bytes, slots=sp.nslots))
                    basis = mbk.monomials(feats + ["y"], 2)
                    bank = bench.measure("bank build (FND-004)", "cpu", lambda: mbk.build(scope, basis, "joint"),
                                         repeats=2, phase="build", **p)
                    fb = bench.measure("moments from bank", "cpu", lambda: lin.moments_from_bank(bank, feats, "y"),
                                       repeats=2, phase="run", **p)
                    if not fb.equal(st):
                        raise SystemExit("LIN-001 bank path differs from the fused state")
                bench.measure("numpy float64 Z^T Z", "numpy", lambda: numpy_gram(cols, valid, g, groups, d),
                              repeats=2, phase="run", **p)
                if rows < 1_000_000 or d <= 8:
                    con.register("t", frame_of(cols, valid, g, rows))
                    sql = duck_sql(d, True)
                    res = bench.measure("duckdb HUGEINT sums", "duckdb", lambda: con.execute(sql).fetchall(),
                                        repeats=1 if rows == 1_000_000 else 2, phase="run", **p)
                    check(st, res, d)
                    con.unregister("t")
                else:
                    bench.note(f"LIN-001: DuckDB baseline skipped at rows={rows}, d={d}, groups={groups} (trimmed)")
                del scope, st, cols, valid, g
    bench.note("LIN-001: every DuckDB-baselined point was checked equal to the native state (exact HUGEINT sums)")
    # measured refusal thresholds (never extrapolated): the matrix-element and state budgets at d = 16, 256 groups
    cols, valid, g = data(rng, 1000, 16, 256)
    scope = scope_of(cols, valid, g, 256)
    feats = [f"x{i}" for i in range(16)]
    need = lin.need(scope, feats, "y")
    for resource in ("tensor", "states", "bytes"):
        lim = need[resource]
        lin.moments(scope, feats, "y", budget=Budget(**{resource: lim}))
        bench.refusal(f"moments {resource} budget", "cpu", lambda: lin.moments(scope, feats, "y",
                                                                             budget=Budget(**{resource: lim - 1})),
                      features=16, groups=256, limit=lim - 1)
        bench.threshold(f"moments {resource} (d = 16, 256 groups)", admitted={resource: lim}, refused={resource: lim - 1})


def exact_beta(st, g):
    return st.fit(g).x


def lin002(bench, rng):
    rows = 20_000 if bench.quick else 100_000
    grid_d = (2, 16) if bench.quick else (2, 4, 8, 16)
    grid_g = (1, 16) if bench.quick else (1, 16, 256)
    for d in grid_d:
        for groups in grid_g:
            cols, valid, g = data(rng, rows, d, groups)
            feats = [f"x{i}" for i in range(d)]
            p = dict(rows=rows, features=d, groups=groups)
            scope = scope_of(cols, valid, g, groups)
            st = bench.measure("moments (assembly)", "cpu", lambda: lin.moments(scope, feats, "y"), repeats=2,
                               phase="build", **p)
            present = [q for q in range(groups) if st.state(q)["present"]]
            reqs = [dict(group=q) for q in present]
            bench.measure("exact fits, every group", "cpu", lambda: st.fits(reqs), repeats=2, phase="run", **p)
            if bench.device:
                bench.measure("exact fits, every group (modular)", "cuda", lambda: st.fits(reqs, backend="cuda"),
                              repeats=2, phase="run", **p)
            q0 = present[0]
            A = st.system(q0)["A"]
            bench.measure("exact inertia (normal matrix)", "cpu", lambda: lin.inertia(A), repeats=3, phase="run", **p)
            bench.measure("pca covariance (Jacobi, certified bounds)", "cpu", lambda: st.pca(q0), repeats=3, phase="run", **p)
            e = st.pca(q0)
            bench.records.append(dict(case="pca bounds", backend="cpu", phase="measure", params=p,
                                      max_residual=float(max(e.residual)), max_bound=float(max(e.bound)),
                                      orthogonality=e.orthogonality, sweeps=e.sweeps))
            del scope, st
    # measured refusal thresholds: the normal system's tensor (k (k + 1) at d = 16) and the eigensolver's sweep cap
    cols, valid, g = data(rng, 2000, 16, 1)
    st = lin.moments(scope_of(cols, valid, g, 1), [f"x{i}" for i in range(16)], "y")
    st.fit(0, budget=Budget(tensor=18 * 19))
    bench.refusal("fit tensor budget", "cpu", lambda: st.fit(0, budget=Budget(tensor=18 * 19 - 1)), features=16)
    bench.threshold("fit tensor (d = 16)", admitted=dict(tensor=18 * 19), refused=dict(tensor=18 * 19 - 1))
    st.pca(0, sweeps=64, budget=Budget(iterations=64))
    bench.refusal("pca sweep budget", "cpu", lambda: st.pca(0, sweeps=64, budget=Budget(iterations=63)), features=16)
    bench.threshold("pca sweeps (cap 64)", admitted=dict(iterations=64), refused=dict(iterations=63))
    # conditioning: well-conditioned, nearly singular, exactly rank deficient
    d = 4 if bench.quick else 8
    for kind in ("random", "nearly", "deficient"):
        cols, valid, g = data(rng, rows, d, 1, kind)
        feats = [f"x{i}" for i in range(d)]
        p = dict(rows=rows, features=d, groups=1, design=kind)
        st = lin.moments(scope_of(cols, valid, g, 1), feats, "y")
        fit = bench.measure("exact fit by design", "cpu", lambda: st.fit(0), repeats=3, phase="run", **p)
        X = np.stack([np.ones(rows)] + [cols[f] for f in feats], 1).astype(np.float64)
        ok = np.ones(rows, bool)
        for v in valid.values():
            ok &= v
        lst = bench.measure("numpy lstsq (float64)", "numpy",
                            lambda: np.linalg.lstsq(X[ok], cols["y"][ok].astype(np.float64), rcond=None), repeats=3,
                            phase="run", **p)
        rec = dict(case="design", backend="cpu", phase="measure", params=p, kind=fit.kind, rank=fit.rank,
                   numpy_rank=int(lst[2]))
        if fit.kind == "unique":
            exact = np.array([float(v) for v in fit.x])
            rec["numpy_max_abs_error"] = float(np.max(np.abs(lst[0] - exact)))
            rec["numpy_max_rel_error"] = float(np.max(np.abs(lst[0] - exact) / np.maximum(np.abs(exact), 1e-300)))
        bench.records.append(rec)
    # repeated requests from one retained state against complete reconstruction
    d = 4 if bench.quick else 8
    for rr in ((20_000,) if bench.quick else (100_000, 1_000_000)):
        cols, valid, g = data(rng, rr, d, 1)
        feats = [f"x{i}" for i in range(d)]
        scope = scope_of(cols, valid, g, 1)
        st = bench.measure("retained state", "cpu", lambda: lin.moments(scope, feats, "y"), repeats=1, phase="build",
                           rows=rr, features=d)
        ridges = [F(k, 4) for k in range(16)]
        p = dict(rows=rr, features=d, requests=len(ridges))
        bench.measure("ridge sweep from the state", "cpu", lambda: [st.fit(0, ridge=lam) for lam in ridges],
                      repeats=2, phase="warm", **p)
        bench.measure("ridge sweep, complete reconstruction", "reconstruction",
                      lambda: [lin.moments(scope_of(cols, valid, g, 1), feats, "y").fit(0, ridge=lam) for lam in ridges],
                      repeats=1, phase="run", **p)
        X = np.stack([np.ones(rr)] + [cols[f] for f in feats], 1).astype(np.float64)
        ok = np.ones(rr, bool)
        for v in valid.values():
            ok &= v
        yv = cols["y"].astype(np.float64)

        def numpy_ridge():
            out = []
            for lam in ridges:
                G = X[ok].T @ X[ok] + float(lam) * np.diag([0.0] + [1.0] * d)
                out.append(np.linalg.solve(G, X[ok].T @ yv[ok]))
            return out
        bench.measure("ridge sweep, numpy float64 from the rows", "numpy", numpy_ridge, repeats=2, phase="run", **p)
        dels = [[f for f in feats if f != drop] for drop in feats]
        p = dict(rows=rr, features=d, requests=len(dels))
        bench.measure("feature deletions from the state", "cpu", lambda: [st.fit(0, features=fs) for fs in dels],
                      repeats=2, phase="warm", **p)
        bench.measure("feature deletions, complete reconstruction", "reconstruction",
                      lambda: [lin.moments(scope_of(cols, valid, g, 1), fs, "y").fit(0) for fs in dels],
                      repeats=1, phase="run", **p)
        drop = rng.random(rr) < 0.1
        part = mbk.Scope.from_arrays(cols, valid=valid, rows=drop)
        sub = lin.moments(part, feats, "y")
        p = dict(rows=rr, features=d, deleted=int(drop.sum()))
        rest = bench.measure("row deletion: state minus deleted rows", "cpu", lambda: (st - sub).fit(0), repeats=3,
                             phase="warm", **p)
        full = bench.measure("row deletion, complete reconstruction", "reconstruction",
                             lambda: lin.moments(mbk.Scope.from_arrays(cols, valid=valid, rows=~drop), feats, "y").fit(0),
                             repeats=2, phase="run", **p)
        if not rest.same(full):
            raise SystemExit("LIN-002: the downdated state's fit differs from the reconstruction")
        del scope, st, cols, valid, X


def main():
    bench = Bench.from_args("LIN-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    if bench.capability == "LIN-001":
        lin001(bench, rng)
    elif bench.capability == "LIN-002":
        lin002(bench, rng)
    else:
        raise SystemExit("--capability LIN-001 or LIN-002")
    if not bench.device:
        bench.note("CUDA paths not timed (no --device): the emulation is never timed as CUDA")
    bench.write()


if __name__ == "__main__":
    main()
