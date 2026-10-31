# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DIST-003 benchmark: information states (certified entropy, mutual information, chi-square, G, Cramer's V) built
from the retained support of DIST-001 histograms, and batches of information requests from that shared support.

    python bench/benchmark_math_info.py --capability DIST-003 --out FILE [--device] [--quick]

Grid: 1k / 100k / 1M rows (quick: 1k / 100k), support sizes 8 / 64 / 1024 per coordinate, joint dimensions 1 / 2 / 3,
5% NULLs per column (a category of its own). Per point: the histogram (DIST-001, the one scan of the rows) and, from
its states without reading a row, the information support (marginal tables and contingency cells: bytes, cells) and
the certified sums (CPU; CUDA with --device), each measured as construction; then batches of 1 / 100 / 10,000 mixed
requests (entropies of every marginal, conditional entropies, mutual information, chi-square, G, V over the
coordinate pairs) from the one shared state, first and warm. A benchmark budget of 2^28 bytes refuses the largest
supports (recorded, never extrapolated). Baselines: NumPy complete reconstruction per request (np.unique over the rows,
then -sum p log p or the chi-square sum in float64) and DuckDB SQL (GROUP BY counts, then the entropy sum), up to 100
requests; their float answers are checked inside the certified enclosures up to their own rounding. Extra cases: the
exact chi-square (FND-009 big rationals) of 8- and 64-value tables, a near-row-sized support (2 x rows values per
column) and structural zeros (even values only, dense and sparse histograms give the same state).
"""
import math

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_dist as md, math_info as mi
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError

NAMES = ("c0", "c1", "c2")


def data(rng, rows, support, k, nulls=0.05):
    vals = {NAMES[i]: rng.integers(0, support, rows).astype(np.int64) for i in range(k)}
    valid = {n: rng.random(rows) >= nulls for n in vals}
    return vals, valid


def mixed(rng, k, n):
    names = list(NAMES[:k])
    items = []
    for _ in range(n):
        kind = int(rng.integers(0, 6 if k > 1 else 1))
        if kind == 0 or k == 1:
            size = int(rng.integers(1, k + 1))
            items.append(dict(op="entropy", a=list(rng.choice(names, size, replace=False)),
                              units="bits" if rng.random() < 0.5 else "nats"))
            continue
        a, b = (str(v) for v in rng.choice(names, 2, replace=False))
        op = ("conditional_entropy", "mutual_information", "chi_square", "g_test", "cramers_v")[kind - 1]
        items.append(dict(op=op, a=[a], b=[b]))
    return items


def numpy_entropy(stack):
    _, counts = np.unique(stack, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def grid(bench, rng):
    budget = Budget(bytes=1 << 28)
    for rows in ((1_000, 100_000) if bench.quick else (1_000, 100_000, 1_000_000)):
        for support in (8, 64, 1024):
            for k in (1, 2, 3):
                vals, valid = data(rng, rows, support, k)
                p = dict(rows=rows, support=support, dims=k)
                scope = mbk.Scope.from_arrays(vals, valid=valid)
                names = list(vals)
                h = bench.measure("histogram (DIST-001, one scan)", "cpu", lambda: md.histogram(scope, names),
                                  repeats=2, phase="build", **p)
                bench.records[-1].update(scans=h.info["scans"], occupied=h.info["occupied"], state_bytes=h.bytes)
                need = mi.need(h)
                try:
                    inf = bench.measure("information support + certified sums", "cpu",
                                        lambda: mi.information(h, budget=budget), repeats=2 if rows < 1_000_000 else 1,
                                        phase="build", **p)
                except AlgebraBudgetError:
                    bench.refusal("information support + certified sums", "cpu",
                                  lambda: mi.information(h, budget=budget), first_stage=need, **p)
                    continue
                i = inf.info
                bench.records[-1].update(support_bytes=i["support_bytes"], state_bytes=i["state_bytes"],
                                         joint_cells=i["joint_cells"], table_cells=i["table_cells"],
                                         contingency_cells=i["contingency_cells"], rows_scanned=0)
                for backend in bench.backends():
                    if backend == "cuda":
                        bench.measure("information support + certified sums", "cuda",
                                      lambda: mi.information(h, backend="cuda", budget=budget), repeats=2,
                                      phase="build", **p)
                stack = np.stack([np.where(valid[n], vals[n], -1) for n in names], axis=1)
                for nreq in (1, 100, 10_000):
                    items = mixed(rng, k, nreq)
                    reqs = inf.requests(items)
                    out = bench.measure("requests from the shared support", "cpu", lambda: inf.query_arrays(reqs),
                                        repeats=3, phase="first+warm", requests=nreq, **p)
                    if (out["status"] != 0).any():
                        raise SystemExit(f"DIST-003: a request failed at {p}")
                    if nreq <= 100:
                        ent = [q for q in items if q["op"] == "entropy" and q.get("units") == "nats"][:20]

                        def recon(ent=ent):
                            return [numpy_entropy(stack[:, [names.index(c) for c in q["a"]]]) for q in ent]
                        got = bench.measure("entropy requests (reconstruction: np.unique per request)", "numpy",
                                            recon, repeats=1, phase="one-shot", requests=len(ent), **p)
                        for q, v in zip(ent, got):
                            r = inf.entropy(q["a"])
                            tol = 1e-9 * max(1.0, abs(v))
                            if not r["lo"] - tol <= v <= r["hi"] + tol:
                                raise SystemExit(f"DIST-003: NumPy entropy {v} outside {r['lo']}, {r['hi']} at {p}")
                if rows <= 100_000:
                    con = duckdb.connect()
                    con.register("t", pd.DataFrame({n: pd.Series(np.where(valid[n], vals[n], 0), dtype="Int64")
                                                    .where(valid[n]) for n in names}))
                    cols = ", ".join(names)
                    sql = (f"SELECT -SUM(c / s * LN(c / s)) FROM (SELECT CAST(COUNT(*) AS DOUBLE) c, "
                           f"SUM(COUNT(*)) OVER () s FROM t GROUP BY {cols})")
                    v = bench.measure("joint entropy (DuckDB GROUP BY + sum)", "duckdb",
                                      lambda: con.execute(sql).fetchone()[0], repeats=3, **p)
                    r = inf.entropy(names)
                    if not r["lo"] - 1e-9 <= v <= r["hi"] + 1e-9:
                        raise SystemExit(f"DIST-003: DuckDB entropy {v} outside {r['lo']}, {r['hi']}")
                    con.close()
                inf.close()
    admitted = [r for r in bench.records if r["case"] == "information support + certified sums" and
                r.get("phase") == "build" and "support_bytes" in r]
    refused = [r for r in bench.records if r.get("phase") == "refusal" and r.get("refused")]
    if admitted and refused:
        big = max(admitted, key=lambda r: r["support_bytes"])
        small = min(refused, key=lambda r: r["need"])
        bench.threshold("information support under a 2^28-byte budget",
                        dict(bytes=big["support_bytes"], **big["params"]),
                        dict(bytes=small["need"], resource=small["resource"], **small["params"]), budget_bytes=1 << 28)


def extras(bench, rng):
    rows = 100_000 if bench.quick else 1_000_000
    # exact chi-square (big rationals) of small tables
    for support in (8, 64):
        vals, valid = data(rng, 20_000, support, 2, nulls=0)
        inf = mi.information(md.histogram(mbk.Scope.from_arrays(vals), list(vals)))
        r = bench.measure("exact chi-square (FND-009 big rationals)", "cpu",
                          lambda: inf.chi_square("c0", "c1", exact=True)["exact"], repeats=2, rows=20_000,
                          support=support)
        bench.records[-1].update(denominator_bits=r.denominator.bit_length())
        enc = inf.chi_square("c0", "c1")
        if not enc["bounds"][0] <= r <= enc["bounds"][1]:
            raise SystemExit("DIST-003: the exact chi-square lies outside its enclosure")
    bench.refusal("exact chi-square over the default bytes budget", "cpu",
                  lambda: inf.chi_square("c0", "c1", exact=True, budget=Budget(bytes=1 << 10)), support=64)
    # a near-row-sized support: every row its own cell
    near = {"x": rng.integers(0, 2 * rows, rows).astype(np.int64), "y": rng.integers(0, 2 * rows, rows).astype(np.int64)}
    h = md.histogram(mbk.Scope.from_arrays(near), ["x", "y"])
    for backend in bench.backends():
        inf = bench.measure("near-row-sized support (support + sums)", backend,
                            lambda: mi.information(h, backend=backend), repeats=1, phase="build", rows=rows)
        bench.records[-1].update(joint_cells=inf.info["joint_cells"], support_bytes=inf.info["support_bytes"])
    names = {"c0": "x", "c1": "y"}
    near_items = [dict(q, a=[names[c] for c in q["a"]], b=[names[c] for c in q["b"]] if q.get("b") else None)
                  for q in mixed(rng, 2, 10_000)]
    reqs = inf.requests(near_items)
    bench.measure("near-row-sized support: 10,000 requests", "cpu", lambda: inf.query_arrays(reqs), repeats=3,
                  phase="first+warm", rows=rows, requests=10_000)
    # structural zeros: even values only; the dense histogram (zero cells stored) gives the same state as the sparse
    even = {"x": (2 * rng.integers(0, 512, rows)).astype(np.int64), "y": rng.integers(0, 8, rows).astype(np.int64)}
    scope = mbk.Scope.from_arrays(even)
    states = {}
    for repr_ in ("sparse", "dense"):
        h = md.histogram(scope, ["x", "y"], repr=repr_)
        states[repr_] = bench.measure("structural zeros (even values)", "cpu", lambda: mi.information(h), repeats=2,
                                      phase="build", rows=rows, repr=repr_, histogram_cells=h.info["cells"])
    if not states["sparse"].equal(states["dense"]):
        raise SystemExit("DIST-003: dense and sparse histograms give different information states")


def main():
    bench = Bench.from_args("DIST-003", __doc__)
    rng = np.random.default_rng(bench.seed)
    grid(bench, rng)
    extras(bench, rng)
    bench.note("cpu = math_info.c (single-threaded: the support from the histogram's states, the certified sums, the "
               "requests); cuda = math_info_gpu.cu (the per-cell terms and their exact 256-bit sums; device runs "
               "only); numpy = complete reconstruction from the rows per request in float64; duckdb = GROUP BY counts "
               "then the entropy sum in SQL (float64). Float baselines are checked inside the certified enclosures.")
    bench.note("The histogram is the only pass over the rows (scans = 1); the information support and every request "
               "read the histogram's states (rows_scanned = 0). Requests cost O(1) each from the shared sums.")
    bench.write()


if __name__ == "__main__":
    main()
