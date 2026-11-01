# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""MULTI-002 benchmark: adaptive refinement and change localization over MULTI-001 hierarchies.

    python bench/benchmark_math_refine.py --capability MULTI-002 --out FILE [--device] [--quick]

Case 1, adaptive requests over balanced and skew hierarchies with 2 / 4 / 8 levels and 8 / 64 / 1,024 leaf groups
(level sizes geometric from the leaves to one group): the build (the MULTI-001 hierarchy with certified maps, then the
refinement state) with the retained state bytes per level; repeated coarse and fine requests (a heavy-hitter
threshold SUM(x) > 10 x the median group, the top 1 and top 10 by SUM(x), AVG(x) >= a quantile; light values with a
few heavy leaves) answered adaptively (CPU; CUDA
with --device) against the complete fine-grain answer of the same native states (tr_refine_fine) and DuckDB SQL over
the same rows (GROUP BY the level's key with HAVING, ORDER BY ... LIMIT), first and warm executions apart; the
refinement work (cells read, regions evaluated and expanded per level, waves) against the fine-grain work. The
no-gain cases: uniform data and a median threshold, where almost every region is undecided and expanded, and a top-k
whose k exceeds the heavy leaves (the k-th witness is a light region's mean, which prunes little).
Case 2, localized changes in 1% / 10% / all leaves: the DELTA-001 update rolled up along the affected paths (CPU; the
CUDA rollup with --device) against the complete reconstruction from the rows after the update (scopes, certified
maps, the hierarchy, the state); the cells the localized rollup visits against a complete rollup, the changed regions
per level. Updating every leaf is the case with no localization gain.
Case 3, budget thresholds: the expansion limit at an iterations budget and the regions of a device bound evaluation
at a fixed device-copy budget (largest admitted, smallest refused).
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import duckdb
from torbles import math_bank as mbk
from torbles import math_delta as md
from torbles import math_multi as mm
from torbles import math_refine as mr
from torbles.math_budgets import Budget

BASIS = ["1", "x", "x^2"]


def level_sizes(G, L):
    return [max(1, int(round(G ** ((L - 1 - k) / (L - 1))))) for k in range(L)]


def data(rng, n, sizes, skew, heavy):
    if skew:
        p = 1.0 / np.arange(1, sizes[0] + 1) ** 1.1
        leaf = rng.choice(sizes[0], n, p=p / p.sum())
    else:
        leaf = rng.integers(0, sizes[0], n)
    ids, parents = [leaf], []
    for k in range(1, len(sizes)):
        lo, hi = sizes[k - 1], sizes[k]
        if skew:
            par = np.where(np.arange(lo) < lo * 3 // 4, 0, rng.integers(0, hi, lo))
        else:
            par = np.arange(lo) * hi // lo
        parents.append(par.astype(np.int64))
        ids.append(parents[-1][ids[-1]])
    if heavy:                                     # light background, a few heavy leaves
        x = rng.integers(0, 50, n)
        hot = rng.choice(sizes[0], max(1, sizes[0] // 32), replace=False)
        x = np.where(np.isin(leaf, hot), x + 20000, x)
    else:                                         # uniform values: nothing for the bounds to separate
        x = rng.integers(0, 1000, n)
    valid = rng.random(n) > 0.05
    return ids, parents, x, valid


def build(bench, ids, sizes, x, valid, p):
    scopes = [mbk.Scope.from_arrays({"x": x}, valid={"x": valid}, groups=ids[k], ngroups=sizes[k])
              for k in range(len(sizes))]
    h = bench.measure("build: MULTI-001 hierarchy (certified maps)", "cpu", lambda: mm.Hierarchy(scopes, BASIS),
                      repeats=2, phase="build", **p)
    st = bench.measure("build: refinement state", "cpu", lambda: mr.State(h), repeats=3, phase="build", **p)
    info = st.info
    bench.records[-1]["result"] = dict(state_bytes=info["bytes"], hierarchy_bytes=h.info["bytes"],
                                       level_bytes=str(info["level_bytes"]), groups=str(info["groups"]))
    return scopes, h, st


def frame(ids, x, valid):
    cols = {f"l{k}": ids[k] for k in range(len(ids))}
    cols["x"] = pd.array(np.where(valid, x, 0), dtype="Int64")
    cols["x"][~valid] = pd.NA
    return pd.DataFrame(cols)


def requests(bench, st, ids, x, valid, sizes, con, p, no_gain=False):
    L = len(sizes)
    for target in sorted({0, L - 2}):
        key = f"l{target}"
        sums = con.execute(f"SELECT SUM(x) s FROM f GROUP BY {key} ORDER BY s").fetchall()
        vals = [0 if v is None else int(v) for (v,) in sums]
        avgs = con.execute(f"SELECT AVG(x) a FROM f GROUP BY {key} HAVING COUNT(x) > 0 ORDER BY a").fetchall()
        if no_gain:
            reqs = {"threshold at the median (no gain)": ("threshold", dict(op=">", threshold=vals[len(vals) // 2]),
                                                         f"HAVING SUM(x) > {vals[len(vals) // 2]}")}
        else:
            T = 10 * vals[len(vals) // 2]
            A = int(avgs[min(len(avgs) - 1, int(len(avgs) * 0.9))][0])
            k = min(10, max(1, len(vals) // 2))
            reqs = {"heavy hitters SUM(x) > 10 x median": ("threshold", dict(op=">", threshold=T), f"HAVING SUM(x) > {T}"),
                    "top 1 by SUM(x)": ("topk", dict(k=1), f"ORDER BY SUM(x) DESC NULLS LAST, {key} LIMIT 1"),
                    f"top {k} by SUM(x)": ("topk", dict(k=k), f"ORDER BY SUM(x) DESC NULLS LAST, {key} LIMIT {k}"),
                    "AVG(x) >= p90": ("ratio", dict(d="count:x", op=">=", threshold=A), f"HAVING AVG(x) >= {A}")}
        for label, (kind, kw, sql) in reqs.items():
            q = dict(p, request=label, target=target, target_groups=sizes[target])
            for backend in bench.backends():
                r = bench.measure("adaptive request", backend, lambda: st.query(kind, target, "sum:x", backend=backend, **kw),
                                  repeats=3, phase="run", **q)
                s = r.stats
                bench.records[-1]["result"] = dict(work=s["work"], fine_work=s["fine_work"],
                                                   work_ratio=s["work"] / max(1, s["fine_work"]),
                                                   evaluated=str(s["evaluated"]), expanded=str(s["expanded"]),
                                                   waves=s["waves"], entries=s["entries"], answer=len(r.answer),
                                                   complete=r.complete)
            f = bench.measure("baseline: complete fine-grain states", "cpu",
                              lambda: st.fine(kind, target, "sum:x", **kw), repeats=3, phase="baseline", **q)
            bench.records[-1]["result"] = dict(answer=len(f.answer), agrees=bool(f.answer == r.answer),
                                               fine_work=f.stats["fine_work"])
            n = bench.measure("baseline: DuckDB", "duckdb",
                              lambda: len(con.execute(f"SELECT {key} FROM f GROUP BY {key} {sql}").fetchall()),
                              repeats=3, phase="baseline", **q)


def adaptive(bench, quick):
    grid = [(L, G) for L in ((2, 4) if quick else (2, 4, 8)) for G in ((8, 64) if quick else (8, 64, 1024))]
    states = {}
    for L, G in grid:
        for skew in (False, True):
            rng = np.random.default_rng(bench.seed + L * 1000 + G + skew)
            sizes = level_sizes(G, L)
            n = 1500 if quick else max(4000, 16 * G)
            ids, parents, x, valid = data(rng, n, sizes, skew, heavy=True)
            p = dict(levels=L, leaf_groups=G, sizes=str(sizes), shape="skew" if skew else "balanced", rows=n)
            scopes, h, st = build(bench, ids, sizes, x, valid, p)
            con = duckdb.connect()
            con.register("f", frame(ids, x, valid))
            requests(bench, st, ids, x, valid, sizes, con, p)
            con.close()
            states[(L, G, skew)] = (st, ids, parents, x, valid, sizes)
        # the no-gain case: uniform data, a median threshold
        rng = np.random.default_rng(bench.seed + 7)
        sizes = level_sizes(G, L)
        n = 1500 if quick else max(4000, 16 * G)
        ids, parents, x, valid = data(rng, n, sizes, False, heavy=False)
        p = dict(levels=L, leaf_groups=G, sizes=str(sizes), shape="uniform", rows=n)
        _, _, st = build(bench, ids, sizes, x, valid, p)
        con = duckdb.connect()
        con.register("f", frame(ids, x, valid))
        requests(bench, st, ids, x, valid, sizes, con, p, no_gain=True)
        con.close()
    return states


def localized(bench, states):
    for (L, G, skew), (st, ids, parents, x, valid, sizes) in states.items():
        if G < 64:
            continue
        for frac in (0.01, 0.1, 1.0):
            rng = np.random.default_rng(bench.seed + int(frac * 1000) + G + L)
            present = np.unique(ids[0])
            k = max(1, int(round(len(present) * frac)))
            touched = rng.choice(present, k, replace=False)
            m = 2 * k
            leaf = rng.choice(touched, m)
            nx = rng.integers(0, 1000, m)
            delta = md.Delta.from_arrays({"x": nx}, groups=leaf, ngroups=sizes[0])
            p = dict(levels=L, leaf_groups=G, shape="skew" if skew else "balanced", leaves=frac, touched=k,
                     delta_rows=m)
            for backend in bench.backends():
                new = bench.measure("localized update", backend, lambda: st.update(delta, backend=backend), repeats=3,
                                    phase="run", **p)
                info = new.info
                bench.records[-1]["result"] = dict(strategy=info["strategy"], work=info["work"],
                                                   rebuild_work=info["rebuild_work"],
                                                   work_ratio=info["work"] / max(1, info["rebuild_work"]),
                                                   changed=str(info["changed"]), state_bytes=info["bytes"])
            ids2 = [np.concatenate([ids[0], leaf])]
            for par in parents:
                ids2.append(par[ids2[-1]])
            x2, v2 = np.concatenate([x, nx]), np.concatenate([valid, np.ones(m, bool)])

            def rebuild():
                scopes = [mbk.Scope.from_arrays({"x": x2}, valid={"x": v2}, groups=ids2[j], ngroups=sizes[j])
                          for j in range(len(sizes))]
                return mr.State(mm.Hierarchy(scopes, BASIS))
            rb = bench.measure("baseline: complete reconstruction", "reconstruction", rebuild, repeats=2,
                               phase="baseline", **p)
            same = all(np.array_equal(new.cells(l)["rows"], rb.cells(l)["rows"]) and
                       new.cells(l)["sum"] == rb.cells(l)["sum"] for l in range(len(sizes)))
            bench.records[-1]["result"] = dict(agrees=bool(same))


def thresholds(bench, states):
    cands = [(v[0].request_need("threshold", 0, "sum:x", op=">", threshold=0)["iterations"], i)
             for i, v in enumerate(states.values())]
    most, i = max(cands)
    (st, ids, parents, x, valid, sizes) = list(states.values())[i]
    limit = max(1, most // 2)
    admitted, refused = [], []
    for e in (limit - 2, limit - 1, limit, limit + 1, limit + 2):
        if e < 1:
            continue
        rec = bench.refusal("expansion limit at an iterations budget", "cpu",
                            lambda: st.threshold(0, "sum:x", ">", 0, expand_limit=e, budget=Budget(iterations=limit)),
                            expand_limit=e, iterations_limit=limit, regions_above_target=most)
        (refused if rec["refused"] else admitted).append(e)
    bench.threshold(f"expansion limit at iterations budget {limit}", max(admitted, default=None),
                    min(refused, default=None))
    present = np.flatnonzero(st.cells(0)["rows"])
    n0 = min(len(present), 40)
    dev = st.bounds_need("threshold", 0, "sum:x", 0, n0)["bytes"]
    admitted, refused = [], []
    for n in (n0 - 2, n0 - 1, n0, n0 + 1, n0 + 2):
        if n < 1 or n > len(present):
            continue
        rec = bench.refusal("CUDA device bytes of a bound evaluation", "cpu",
                            lambda: st.bounds("threshold", 0, "sum:x", 0, present[:n], backend="cuda",
                                              budget=Budget(bytes=dev)),
                            regions=n, bytes_limit=dev)
        (refused if rec["refused"] else admitted).append(n)
    bench.threshold(f"regions at a CUDA device-copy budget of {dev} bytes", max(admitted, default=None),
                    min(refused, default=None))


def main():
    bench = Bench.from_args("MULTI-002", __doc__)
    states = adaptive(bench, bench.quick)
    localized(bench, states)
    thresholds(bench, states)
    bench.note("CPU = native libalg; CUDA timings only with --device (an emulated run is never timed as CUDA); "
               "work counts the cells a request reads (regions x (terms + 1) and their columns' abstract inputs) and the "
               "cells an update visits; fine_work is the complete fine-grain evaluation's; the reconstruction rebuilds "
               "the scopes, certified maps, hierarchy and state from the rows after the update.")
    bench.write()


if __name__ == "__main__":
    main()
