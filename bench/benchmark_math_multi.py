# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""MULTI-001 benchmark: hierarchies of group keys (certified refinement maps, banks built once at the finest level and
rolled up), repeated coarse/fine requests, retained state bytes and rollup work, localized changes against complete
reconstruction, and batched exact contrasts.

    python bench/benchmark_math_multi.py --capability MULTI-001 --out FILE [--device] [--quick]

Hierarchies: balanced (each level splits its parents evenly) and skewed (three quarters of each level under one
parent) with 2 / 4 / 8 levels and 8 / 64 / 1,024 finest groups over 500k rows (quick: 50k) with two value columns and
the 6 monomials of degree <= 2. Phases: "build" = scopes, certification of every map, the finest bank and the
hierarchy (its rollups), each timed apart; "warm" = repeated requests for the finest and the coarsest level's bank (an
independent copy) and for one group's state. Baselines: complete reconstruction (the FND-004 bank built directly on
each level's scope from the rows) and DuckDB GROUP BY per level (COUNT and SUM in HUGEINT); every hierarchy level is
checked equal to the direct bank once. Recorded: retained bytes per level and in all, cells visited by the rollups.
Localized changes: the values of the rows of 1% / 10% / all finest groups change; complete reconstruction (new
scopes, maps, finest bank, rollups) against (a) the finest bank rebuilt and the hierarchy updated locally (maps
carried) and (b) a DELTA-001 update of the finest bank (the changed rows deleted and inserted again) followed by the
local update; each result is checked equal to the reconstruction; the strategy per level (local / rollup) and the
cells visited are recorded, including the cases where the adaptive refinement gains nothing (all leaves changed, a
skewed hierarchy). Contrasts: every kind over all child groups and terms of a level, against Python Fractions over
the same states (the oracle, also checked equal).
"""
from fractions import Fraction

import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk
from torbles import math_delta as md
from torbles import math_multi as mm

TERMS = mbk.monomials(["x", "y"], 2)


def sizes_of(levels, groups):
    """groups per level, finest first, geometric down to one group (strictly decreasing where possible)"""
    out = []
    for k in range(levels):
        s = int(round(groups ** (1 - k / (levels - 1)))) if levels > 1 else groups
        out.append(max(1, min(s, out[-1] - 1 if out and out[-1] > 1 else s)))
    return out


def hierarchy_ids(rng, n, sizes, skew):
    ids, parents = [rng.integers(0, sizes[0], n)], []
    for k in range(1, len(sizes)):
        lo, hi = sizes[k - 1], sizes[k]
        if skew:
            p = np.where(np.arange(lo) < lo * 3 // 4, 0, rng.integers(0, hi, lo))
        else:
            p = np.arange(lo) * hi // lo
        parents.append(p.astype(np.int64))
        ids.append(parents[-1][ids[-1]])
    return ids, parents


def duck_level(con, ids):
    con.register("lv", pd.DataFrame({"g": ids}))
    return con.execute("SELECT g, COUNT(*), SUM(CAST(x AS HUGEINT)), SUM(CAST(y AS HUGEINT)), "
                       "SUM(CAST(x AS HUGEINT) * x), SUM(CAST(x AS HUGEINT) * y), SUM(CAST(y AS HUGEINT) * y) "
                       "FROM vals POSITIONAL JOIN lv GROUP BY g").fetchall()


def grid(bench, rng):
    import duckdb
    n = 50_000 if bench.quick else 500_000
    x = rng.integers(-10**6, 10**6, n)
    y = rng.integers(-1000, 1000, n)
    con = duckdb.connect()
    con.register("vals", pd.DataFrame({"x": x, "y": y}))
    cols = {"x": x, "y": y}
    for levels in (2, 4, 8):
        for groups in (8, 64, 1024):
            sizes = sizes_of(levels, groups)
            for skew in (False, True):
                ids, _ = hierarchy_ids(rng, n, sizes, skew)
                p = dict(rows=n, levels=levels, groups=groups, sizes=sizes, shape="skewed" if skew else "balanced",
                         terms=len(TERMS))
                scopes = bench.measure("scopes", "cpu", lambda: [mbk.Scope.from_arrays(cols, groups=ids[k],
                                                                                       ngroups=sizes[k])
                                                                 for k in range(levels)], repeats=1, phase="build", **p)
                for backend in bench.backends():
                    maps = bench.measure("certification", backend, lambda: [mm.certify(scopes[k], scopes[k + 1],
                                                                                      backend=backend)
                                                                          for k in range(levels - 1)],
                                         repeats=2, phase="build", **p)
                    fine = bench.measure("finest bank", backend, lambda: mbk.build(scopes[0], TERMS, backend=backend),
                                         repeats=2, phase="build", **p)
                h = bench.measure("hierarchy (rollups)", "cpu", lambda: mm.Hierarchy(scopes, maps=maps, bank=fine),
                                  repeats=3, phase="build", **p)
                info = h.info
                bench.records.append(dict(case="retained state", backend="cpu", phase="measure", params=p,
                                          bytes=info["bytes"], level_bytes=info["level_bytes"],
                                          rollup_cells=info["work"], level_cells=info["level_work"],
                                          present=info["present"]))
                bench.measure("request finest bank", "cpu", lambda: h.bank(0).rows, repeats=10, phase="warm", **p)
                bench.measure("request coarsest bank", "cpu", lambda: h.bank(levels - 1).rows, repeats=10,
                              phase="warm", **p)
                bench.measure("request one coarse state", "cpu", lambda: h.state(levels - 1, 0)["rows"], repeats=10,
                              phase="warm", **p)
                direct = bench.measure("complete reconstruction (every level from rows)", "reconstruction",
                                       lambda: [mbk.build(s, TERMS) for s in scopes], repeats=2, phase="run", **p)
                for k in range(levels):
                    if not h.bank(k).equal(direct[k]):
                        raise SystemExit(f"MULTI-001: level {k} differs from its direct bank ({p})")
                if groups == 1024 or bench.quick:
                    got = bench.measure("DuckDB GROUP BY (every level)", "duckdb",
                                        lambda: [duck_level(con, ids[k]) for k in range(levels)], repeats=2,
                                        phase="run", **p)
                    for k, rows in enumerate(got):
                        st = h.states(k)
                        for r in rows:
                            s = st[int(r[0])]
                            if [s["rows"]] + s["sum"][1:] != [int(v) for v in r[1:]]:
                                raise SystemExit(f"MULTI-001: level {k} differs from DuckDB ({p})")


def localized(bench, rng):
    n = 50_000 if bench.quick else 500_000
    sizes = [1024, 128, 16, 1]
    for skew in (False, True):
        ids, _ = hierarchy_ids(rng, n, sizes, skew)
        x = rng.integers(-10**6, 10**6, n)
        y = rng.integers(-1000, 1000, n)
        levels = [mbk.Scope.from_arrays({"x": x, "y": y}, groups=ids[k], ngroups=sizes[k]) for k in range(4)]
        h = mm.Hierarchy(levels, TERMS)
        state = md.State.from_bank(h.bank(0))                         # the finest level's bank is its scope's
        for frac in (0.01, 0.10, 1.0):
            leaves = rng.choice(sizes[0], max(1, int(frac * sizes[0])), replace=False)
            hit = np.flatnonzero(np.isin(ids[0], leaves))
            x2 = x.copy()
            x2[hit] = rng.integers(-10**6, 10**6, len(hit))
            p = dict(rows=n, sizes=sizes, shape="skewed" if skew else "balanced", changed_leaves=len(leaves),
                     changed_rows=int(len(hit)), fraction=frac)

            def complete():
                new = [mbk.Scope.from_arrays({"x": x2, "y": y}, groups=ids[k], ngroups=sizes[k]) for k in range(4)]
                return mm.Hierarchy(new, TERMS)
            ref = bench.measure("complete reconstruction (scopes, maps, bank, rollups)", "reconstruction", complete,
                                repeats=2, phase="run", **p)
            new0 = mbk.Scope.from_arrays({"x": x2, "y": y}, groups=ids[0], ngroups=sizes[0])
            fine = bench.measure("finest bank rebuilt", "cpu", lambda: mbk.build(new0, TERMS), repeats=2,
                                 phase="run", **p)
            up = bench.measure("local update of the coarse levels", "cpu", lambda: h.update(fine), repeats=3,
                               phase="run", **p)
            delta = bench.measure("delta of the changed rows (DELTA-001)", "cpu",
                                  lambda: md.Delta.from_arrays({"x": np.concatenate([x[hit], x2[hit]]),
                                                                "y": np.concatenate([y[hit], y[hit]])},
                                                               groups=np.concatenate([ids[0][hit], ids[0][hit]]),
                                                               ngroups=sizes[0],
                                                               weights=np.concatenate([-np.ones(len(hit), np.int64),
                                                                                       np.ones(len(hit), np.int64)])),
                                  repeats=2, phase="build", **p)
            applied = bench.measure("finest bank by the delta (DELTA-001)", "cpu", lambda: state.apply(delta),
                                    repeats=3, phase="run", **p)
            via = bench.measure("local update from the delta's bank", "cpu", lambda: h.update(applied.bank()),
                                repeats=3, phase="run", **p)
            for k in range(4):
                if up.states(k) != ref.states(k) or via.states(k) != ref.states(k):
                    raise SystemExit(f"MULTI-001: a localized update differs from the reconstruction ({p}, level {k})")
            i = up.info
            bench.records.append(dict(case="localized work", backend="cpu", phase="measure", params=p,
                                      strategy=i["strategy"], level_cells=i["level_work"], changed=i["changed"],
                                      complete_cells=ref.info["work"]))


def contrasts(bench, rng):
    n = 50_000 if bench.quick else 500_000
    sizes = [1024, 64, 8]
    ids, _ = hierarchy_ids(rng, n, sizes, False)
    x = rng.integers(-10**6, 10**6, n)
    y = rng.integers(-1000, 1000, n)
    levels = [mbk.Scope.from_arrays({"x": x, "y": y}, groups=ids[k], ngroups=sizes[k]) for k in range(3)]
    h = mm.Hierarchy(levels, TERMS)
    child, parent = h.states(0), h.states(1)
    pm = h.map(0).ids
    p = dict(rows=n, child_groups=sizes[0], terms=len(TERMS))
    for kind in mm.KINDS:
        for backend in bench.backends():
            res = bench.measure("contrasts (native cells)", backend, lambda: h.contrasts(0, kind, backend=backend),
                                repeats=3, phase="run", kind=kind, **p)
        got = bench.measure("contrasts formatted (exact ints)", "cpu", res.raw, repeats=3, phase="run", kind=kind, **p)
        bench.measure("contrasts formatted (Fractions with decoded keys)", "cpu", res.rows, repeats=2, phase="run",
                      kind=kind, **p)
        if kind == "mean":
            def fractions():
                out = []
                for c, s in enumerate(child):
                    q = parent[int(pm[c])]
                    out.append([Fraction(s["sum"][j], s["count"][j]) - Fraction(q["sum"][j], q["count"][j])
                                if s["count"][j] else None for j in range(len(TERMS))])
                return out
            ref = bench.measure("contrasts (native cells)", "python-fractions", fractions, repeats=1, phase="run",
                                kind=kind, **p)
            mine = [[Fraction(num, den) if st == "defined" else None for (_, _, _, st, num, den) in got[c * 6:(c + 1) * 6]]
                    for c in range(sizes[0])]
            if mine != ref:
                raise SystemExit("MULTI-001: contrasts differ from Python Fractions")


def main():
    bench = Bench.from_args("MULTI-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    grid(bench, rng)
    localized(bench, rng)
    contrasts(bench, rng)
    bench.note("Backends: cpu = math_multi.c (and the FND-004 bank for the finest level, single-threaded), cuda = "
               "math_multi_gpu.cu certification / contrasts and math_bank_gpu.cu for the finest bank (device runs only; "
               "the hierarchy's rollups and localized updates are CPU code), reconstruction = the FND-004 bank built "
               "directly on every level's scope from the rows, duckdb = GROUP BY per level with COUNT and SUM in "
               "HUGEINT (DuckDB's own threads), python-fractions = the oracle of the contrasts.")
    bench.note("Scopes, certification, the finest bank and the rollups are timed apart; a request returns an "
               "independent copy of a level's bank, so its cost grows with groups x terms, not with rows. Rollup work "
               "is counted in cells (groups x (terms + 1)) visited.")
    bench.note("Localized changes: the local update visits the dirty children and the touched parents; with every leaf "
               "changed (and on skewed levels where the dirty children are most of a level) it falls back to a full "
               "rollup (strategy 'rollup'), and the gain over the reconstruction is then only the rows not rescanned. "
               "The DELTA-001 path rescans no rows but deletes and inserts every changed row, which costs more than a "
               "rebuild when all leaves change.")
    bench.write()


if __name__ == "__main__":
    main()
