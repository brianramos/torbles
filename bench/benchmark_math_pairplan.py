# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PAIR-002 benchmark: structured pairwise query planning (support aggregation, KERNEL-001 finite features, PAIR-001
sorted prefixes, bounded direct pairs) against explicit pairs (NumPy, DuckDB) and per-request reconstruction.

    python bench/benchmark_math_pairplan.py --capability PAIR-002 --out FILE [--device] [--quick]

Scalar supports 8 / 64 / 1024 (ties: about rows / support elements per value).
  explicit pairs   2,000 rows (quick: 600) in two classes, imbalance 1:1 and 9:1, counts and positive weights: the
                   statistics |x - y| and Mann-Whitney within class 0 (unordered, self-pairs excluded) and across the
                   classes; the plan (its candidates' costs recorded) and each strategy forced (support, prefix,
                   direct; transient states, built per call) against NumPy broadcasting over every pair and a DuckDB
                   self-join; every answer checked equal.
  retained prefix  200,000 rows (quick: 30,000) in 16 classes (imbalance up to 15:1), counts and weights: support
  and support      sorting apart (the retained support table, the retained PAIR-001 prefix state), then batches of
                   1 / 100 / 10,000 mixed requests (within / across, ordered, diagonal excluded or not, means) through
                   the planner with the retained states (the chosen strategy, and each retained strategy forced while
                   its evaluations stay below a cap); baseline: NumPy reconstruction per request (sort + searchsorted)
                   for 1 and 100 requests. A degree-2 polynomial statistic compares the KERNEL-001 feature state with
                   the support table on the same batches.
CUDA (support and direct, with --device only): the table upload once, then the batches.
Refusals: the declared pair budget (iterations) of a direct plan at its need and one past it.
"""
import time
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_kernel as mk, math_pairplan as pp
from torbles.math_budgets import Budget

BIG = Budget(iterations=10**9, output=10**6)
CAP = 30_000_000                       # evaluations above which a forced strategy is not timed (recorded as a note)


def numpy_pairs(op, a, b, within):
    """explicit pairs by broadcasting (exact: int64 sums of small values)"""
    if op == "abs":
        d = np.abs(a[:, None] - b[None, :])
        return int(np.triu(d, 1).sum()) if within else int(d.sum())
    gt = (a[:, None] > b[None, :]).sum() * 2 + (a[:, None] == b[None, :]).sum()
    if within:
        gt -= len(a)                                       # the diagonal ties (self-pairs excluded), ordered pairs
    return Fraction(int(gt), 2)


def numpy_reconstruction(a, b):
    """|x - y| over i in a, j in b from sorted rows (searchsorted), exact"""
    pts = np.union1d(a, b)
    ca, cb = np.searchsorted(a, pts, "right").astype(object), np.searchsorted(b, pts, "right").astype(object)
    na, nb = len(a), len(b)
    gaps = np.diff(pts).astype(object)
    return int(np.dot(ca[:-1] * (nb - cb[:-1]) + cb[:-1] * (na - ca[:-1]), gaps))


def plan_costs(p):
    return {c["name"]: (round(c["total"], 1) if c["status"] != "rejected" else c["reason"]) for c in p.candidates}


def explicit(bench, rng):
    n = 600 if bench.quick else 2000
    for support in (8, 64, 1024):
        for ratio in (1, 9):
            na = n * ratio // (ratio + 1)
            x = rng.integers(0, support, n).astype(np.int64)
            w = rng.integers(1, 4, n).astype(np.int64)
            g = np.r_[np.zeros(na, np.int64), np.ones(n - na, np.int64)]
            scope = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=2)
            a, b = x[:na], x[na:]
            con = duckdb.connect()
            con.register("t", pd.DataFrame({"rid": np.arange(n), "g": g, "x": x}))
            for op, kern in (("abs", pp.absdiff()), ("mann_whitney", pp.mann_whitney())):
                for within in (True, False):
                    req = [dict(a=0)] if (within and op == "abs") else \
                          [dict(a=0, ordered=True)] if within else [dict(a=0, b=1)]
                    p = dict(support=support, rows=n, imbalance=f"{ratio}:1", statistic=op,
                             pairs="within class 0" if within else "across", weights=False)
                    plan = bench.measure("plan", "cpu", lambda: pp.plan(scope, ["x"], kern, req, budget=BIG),
                                         repeats=3, phase="plan", **p)
                    bench.records[-1]["planned_costs"] = plan_costs(plan)
                    bench.records[-1]["chosen"] = plan.candidate(plan.chosen)["name"]
                    answers = {}
                    for force in ("support", "prefix", "direct") + (("support@cuda", "direct@cuda")
                                                                    if bench.device else ()):
                        backend = "cuda" if force.endswith("@cuda") else "cpu"
                        out = bench.measure(f"{op} via {force.split('@')[0]} (state built per call)", backend,
                                            lambda: pp.query(scope, ["x"], kern, req, force=force, budget=BIG),
                                            repeats=5, phase="first+warm", **p)
                        answers[force] = out["results"][0]["value"]
                        bench.records[-1]["evaluations"] = out["evaluations"]
                    auto = bench.measure(f"{op} via the planner's choice", "cpu",
                                         lambda: pp.query(scope, ["x"], kern, req, budget=BIG), repeats=5,
                                         phase="first+warm", **p)
                    bench.records[-1]["chosen"] = auto["name"]
                    want = bench.measure(f"{op}, explicit pairs", "numpy",
                                         lambda: numpy_pairs(op, a, a if within else b, within), repeats=2, **p)
                    if op == "abs":
                        sql = ("SELECT SUM(ABS(a.x - b.x)) FROM t a JOIN t b ON a.g = 0 AND " +
                               ("b.g = 0 AND a.rid < b.rid" if within else "b.g = 1"))
                    else:
                        sql = ("SELECT SUM(CASE WHEN a.x > b.x THEN 2 WHEN a.x = b.x THEN 1 ELSE 0 END) FROM t a JOIN "
                               "t b ON a.g = 0 AND " + ("b.g = 0 AND a.rid <> b.rid" if within else "b.g = 1"))
                    duck = bench.measure(f"{op}, explicit pairs", "duckdb", lambda: con.execute(sql).fetchone()[0],
                                         repeats=2, **p)
                    duck = Fraction(int(duck), 2) if op == "mann_whitney" else int(duck)
                    for force, v in answers.items():
                        if v != want or v != duck or auto["results"][0]["value"] != want:
                            raise SystemExit(f"PAIR-002: {op} via {force} differs from the explicit pairs at {p}")
            con.close()


def batch(rng, G, nreq):
    items = []
    for _ in range(nreq):
        a, b = int(rng.integers(0, G)), int(rng.integers(0, G))
        if rng.random() < 0.4:
            b = a                                          # within one class
        items.append(dict(a=a, b=b, ordered=bool(rng.integers(0, 2)), self_pairs=bool(rng.integers(0, 2)),
                          mean=bool(rng.integers(0, 2))))
    return items


def retained(bench, rng):
    rows = 30_000 if bench.quick else 200_000
    G = 16
    for support in (8, 64, 1024):
        sizes = rng.integers(1, 16, G).astype(np.float64)                 # class imbalance up to 15:1
        g = rng.choice(G, rows, p=sizes / sizes.sum())
        x = rng.integers(0, support, rows).astype(np.int64)
        w = rng.integers(1, 100, rows).astype(np.int64)
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=G)
        cls = [np.sort(x[g == k]) for k in range(G)]
        for weight in (None, "w"):
            p = dict(support=support, rows=rows, classes=G, weights=weight is not None)
            st = bench.measure("support sorting (retained support table)", "cpu",
                               lambda: pp.retain(scope, ["x"], "support", weight=weight), repeats=3, phase="build", **p)
            bench.records[-1].update(entries=st.info["entries"], state_bytes=st.info["bytes"])
            pre = bench.measure("support sorting (retained PAIR-001 prefix state)", "cpu",
                                lambda: pp.retain(scope, ["x"], "prefix", weight=weight), repeats=3, phase="build", **p)
            bench.records[-1].update(entries=pre.info["entries"], state_bytes=pre.info["bytes"])
            if bench.device:
                bench.measure("support table upload", "cuda", lambda: st.device(), repeats=1, phase="build", **p)
            for nreq in (1, 100, 10_000):
                items = batch(rng, G, nreq)
                plan = bench.measure("plan with the retained states", "cpu",
                                     lambda: pp.plan(scope, ["x"], pp.absdiff(), items, weight=weight,
                                                     devices=bench.backends(), retained=[st, pre], budget=BIG),
                                     repeats=2, phase="plan", requests=nreq, **p)
                bench.records[-1]["planned_costs"] = plan_costs(plan)
                chosen = plan.candidate(plan.chosen)
                bench.records[-1]["chosen"] = chosen["name"]
                ref = None
                for c in plan.candidates:
                    if c["status"] == "rejected" or c["strategy"] not in ("support", "prefix", "direct"):
                        continue
                    if c["evaluations"] > CAP:
                        bench.note(f"{c['name']} not timed at support {support}, {nreq} requests: "
                                   f"{c['evaluations']} planned evaluations (> {CAP})")
                        continue
                    out = bench.measure(f"|x - y| via {c['strategy']}" + (" (retained)" if c["reuse"] is not None
                                                                           else " (built per call)"),
                                        c["device"], lambda: plan.run(c["index"], raw=True, budget=BIG),
                                        repeats=5 if nreq < 10_000 else 2, phase="first+warm", requests=nreq, **p)
                    bench.records[-1].update(planned_total=c["total"], evaluations=out["evaluations"],
                                             chosen=c["index"] == plan.chosen)
                    res = out["results"]
                    if (res["status"] != 0).any():
                        raise SystemExit("PAIR-002: a benchmark request failed")
                    if ref is None:
                        ref = res
                    elif not all(np.array_equal(res[f], ref[f]) for f in ("num", "den", "pairs")):
                        raise SystemExit(f"PAIR-002: {c['name']} differs from the other strategies")
                if weight is None and nreq <= 100:
                    across = [q for q in items if q["a"] != q["b"] and not q["ordered"] and not q["mean"]]
                    got = pp.query(scope, ["x"], pp.absdiff(), across, retained=[pre], budget=BIG)["results"]
                    for q, r in zip(across, got):
                        if r["value"] != numpy_reconstruction(cls[q["a"]], cls[q["b"]]):
                            raise SystemExit("PAIR-002: the prefix answer differs from NumPy")

                    def recon(items=across):
                        return [numpy_reconstruction(np.sort(x[g == q["a"]]), np.sort(x[g == q["b"]])) for q in items]
                    bench.measure("|x - y| across classes (reconstruction: select + sort + searchsorted per request)",
                                  "numpy", recon, repeats=1, phase="one-shot", requests=len(across), **p)
            # a polynomial statistic: the KERNEL-001 feature state against the support table
            if weight is not None and support <= 64:
                kern = pp.features(mk.polynomial(2, c=1))
                feats = bench.measure("feature state (retained KERNEL-001, degree 2)", "cpu",
                                      lambda: pp.retain(scope, ["x"], "features", kernel=kern, weight=weight),
                                      repeats=2, phase="build", **p)
                for nreq in (1, 100, 10_000):
                    items = batch(rng, G, nreq)
                    plan = pp.plan(scope, ["x"], kern, items, weight=weight, retained=[st, feats], budget=BIG)
                    ref = None
                    for c in plan.candidates:
                        if c["status"] == "rejected" or c["reuse"] is None or c["evaluations"] > CAP:
                            continue
                        out = bench.measure(f"(x y + 1)^2 via {c['strategy']} (retained)", "cpu",
                                            lambda: plan.run(c["index"], raw=True, budget=BIG), repeats=2,
                                            phase="first+warm", requests=nreq, **p)
                        bench.records[-1].update(planned_total=c["total"], chosen=c["index"] == plan.chosen,
                                                 evaluations=out["evaluations"])
                        res = out["results"]
                        if ref is not None and not all(np.array_equal(res[f], ref[f]) for f in ("num", "den", "pairs")):
                            raise SystemExit("PAIR-002: the feature and support answers differ")
                        ref = res


def refusals(bench, rng):
    n = 400
    x = rng.integers(0, 1000, n)
    g = rng.integers(0, 2, n)
    scope = mbk.Scope.from_arrays({"x": x}, groups=g, ngroups=2)
    tab = pp.table({(1, 2): 3}, default=1)                               # an arbitrary similarity: no structure
    p = pp.plan(scope, ["x"], tab, [dict(a=0, b=1)], force="direct", budget=BIG)
    need = p.candidate(p.chosen)["evaluations"]
    bench.measure("an arbitrary similarity via direct pairs at the pair budget", "cpu",
                  lambda: pp.query(scope, ["x"], tab, [dict(a=0, b=1)], force="direct",
                                   budget=Budget(iterations=need)), repeats=2, pairs=need)
    bench.refusal("the same one past the pair budget", "cpu",
                  lambda: pp.plan(scope, ["x"], tab, [dict(a=0, b=1)], force="direct",
                                  budget=Budget(iterations=need - 1)), pairs=need)
    bench.threshold("direct pairs against the declared pair budget (iterations)", dict(pairs=need, budget=need),
                    dict(pairs=need, budget=need - 1))


def main():
    bench = Bench.from_args("PAIR-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    t0 = time.perf_counter()
    explicit(bench, rng)
    retained(bench, rng)
    refusals(bench, rng)
    bench.note("cpu = math_pairplan.c (single-threaded); cuda = math_pairplan_gpu.cu, one work item per request tile "
               "(device runs only, --device); numpy = broadcasting over every pair (explicit) or select + sort + "
               "searchsorted per request (reconstruction); duckdb = self-joins.")
    bench.note("Support sorting (the retained support table, the retained PAIR-001 prefix state, the KERNEL-001 feature "
               "state) is measured apart from the request batches; 'built per call' runs include their state build. "
               "planned_costs / planned_total are the FND-008 relative work units of each candidate (not time), "
               "recorded next to the measured times.")
    bench.note(f"total wall time {time.perf_counter() - t0:.1f} s")
    bench.write()


if __name__ == "__main__":
    main()
