# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PAIR-001 benchmark: pairwise summaries (absolute-difference sums, Mann-Whitney U / AUC, KS, 1-Wasserstein,
Cramer-von Mises) from retained prefix states, against explicit pairs and complete reconstruction.

    python bench/benchmark_math_pair.py --capability PAIR-001 --out FILE [--device] [--quick]

Scalar supports 8 / 64 / 1024 (ties: about rows / support per value).
  explicit pairs   2,000 rows in two classes (imbalance 1:1 and 9:1): the native answer (support sorting = the
                   histogram and prefix state, measured apart, then one request) against NumPy broadcasting over every
                   pair (1M pairs) and a DuckDB self-join; the answers are checked equal.
  retained prefix  1M rows (quick: 100k) in 16 classes, counts and weights: support sorting once (build), then batches of
                   1 / 100 / 10,000 mixed requests (every op, ordered / self-pair / mean flags, class pairs with 1:1 to
                   15:1 imbalance) on the CPU (and CUDA with --device: the upload once, then the batches); baseline:
                   NumPy reconstruction from the rows per request (sort both classes, searchsorted), one-shot (with
                   the sorts) and warm (rows pre-sorted). Answers are checked against NumPy on a sample.
Diagonal exclusion (self-pairs) and ordering change only the pair count and the factor 2: they are timed as flags of
the same batches. Refusals: the output budget of a batch at the limit and one past.
"""
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_dist as md, math_pair as mp
from torbles.math_budgets import Budget

OPS = ("abs_within", "abs_cross", "mann_whitney", "ks", "wasserstein", "cramer_von_mises")


def build(scope, weight):
    h = md.histogram(scope, ["x"], weight=weight)
    return h.prefix("weight" if weight else "count", degree=0)


def numpy_answer(op, a, b):
    """complete reconstruction from (sorted) rows: exact Python ints / Fractions"""
    if op == "abs_within":
        n = len(a)
        k = np.arange(n, dtype=np.int64)
        return int(np.dot(a.astype(object), (2 * k - n + 1).astype(object)))
    if op == "mann_whitney":
        lo, hi = np.searchsorted(b, a, "left"), np.searchsorted(b, a, "right")
        return Fraction(int(lo.sum() + hi.sum()), 2)
    pts = np.union1d(a, b)
    ca, cb = np.searchsorted(a, pts, "right").astype(object), np.searchsorted(b, pts, "right").astype(object)
    na, nb = len(a), len(b)
    diff = np.abs(ca * nb - cb * na)
    if op == "ks":
        return Fraction(int(diff.max()), na * nb)
    if op == "wasserstein":
        gaps = np.diff(pts).astype(object)
        return Fraction(int(np.dot(diff[:-1], gaps)), na * nb)
    if op == "cramer_von_mises":
        m = (np.searchsorted(a, pts, "right") - np.searchsorted(a, pts, "left") +
             np.searchsorted(b, pts, "right") - np.searchsorted(b, pts, "left")).astype(object)
        return Fraction(int(np.dot(diff * diff, m)), na * nb * (na + nb) ** 2)
    gaps = np.diff(pts).astype(object)
    return int(np.dot(ca[:-1] * (nb - cb[:-1]) + cb[:-1] * (na - ca[:-1]), gaps))     # abs_cross


def explicit(bench, rng):
    for support in (8, 64, 1024):
        for ratio in (1, 9):
            n = 2000
            na = n * ratio // (ratio + 1)
            x = rng.integers(0, support, n).astype(np.int64)
            g = np.r_[np.zeros(na, np.int64), np.ones(n - na, np.int64)]
            p = dict(support=support, rows=n, imbalance=f"{ratio}:1")
            scope = mbk.Scope.from_arrays({"x": x}, groups=g, ngroups=2)
            pre = bench.measure("support sorting", "cpu", lambda: build(scope, None), repeats=3, phase="build", **p)
            a, b = x[:na], x[na:]
            con = duckdb.connect()
            con.register("t", pd.DataFrame({"rid": np.arange(n), "g": g, "x": x}))
            for op, item in (("abs_within", dict(op="abs_within", a=0)),
                             ("mann_whitney", dict(op="mann_whitney", a=0, b=1)),
                             ("abs_cross", dict(op="abs_cross", a=0, b=1))):
                got = bench.measure(f"{op}, explicit pairs", "cpu", lambda: mp.query(pre, [item])[0]["value"],
                                    repeats=5, phase="first+warm", **p)
                if op == "abs_within":
                    want = bench.measure(f"{op}, explicit pairs", "numpy",
                                         lambda: int(np.abs(a[:, None] - a[None, :]).sum()) // 2, repeats=2, **p)
                    sql = "SELECT SUM(ABS(a.x - b.x)) FROM t a JOIN t b ON a.g = 0 AND b.g = 0 AND a.rid < b.rid"
                elif op == "mann_whitney":
                    want = bench.measure(f"{op}, explicit pairs", "numpy", lambda: Fraction(
                        int(2 * (a[:, None] > b[None, :]).sum() + (a[:, None] == b[None, :]).sum()), 2), repeats=2, **p)
                    sql = ("SELECT SUM(CASE WHEN a.x > b.x THEN 2 WHEN a.x = b.x THEN 1 ELSE 0 END) / 2 "
                           "FROM t a JOIN t b ON a.g = 0 AND b.g = 1")
                else:
                    want = bench.measure(f"{op}, explicit pairs", "numpy",
                                         lambda: int(np.abs(a[:, None] - b[None, :]).sum()), repeats=2, **p)
                    sql = "SELECT SUM(ABS(a.x - b.x)) FROM t a JOIN t b ON a.g = 0 AND b.g = 1"
                duck = bench.measure(f"{op}, explicit pairs", "duckdb", lambda: con.execute(sql).fetchone()[0],
                                     repeats=2, **p)
                if got != want or Fraction(str(duck)) != Fraction(want):
                    raise SystemExit(f"PAIR-001: {op} differs from the explicit pairs at {p}: {got} {want} {duck}")
            con.close()


def retained(bench, rng):
    rows = 100_000 if bench.quick else 1_000_000
    G = 16
    for support in (8, 64, 1024):
        sizes = rng.integers(1, 16, G).astype(np.float64)           # class imbalance up to 15:1
        g = rng.choice(G, rows, p=sizes / sizes.sum())
        x = rng.integers(0, support, rows).astype(np.int64)
        w = rng.integers(0, 100, rows).astype(np.int64)
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=G)
        cls = [np.sort(x[g == k]) for k in range(G)]
        raw = [x[g == k] for k in range(G)]
        for weight in (None, "w"):
            p = dict(support=support, rows=rows, classes=G, mass="weight" if weight else "count")
            pre = bench.measure("support sorting (histogram + prefix)", "cpu", lambda: build(scope, weight),
                                repeats=3, phase="build", **p)
            bench.records[-1].update(support_entries=pre.info["support"], state_bytes=pre.info["bytes"])
            for nreq in (1, 100, 10_000):
                items = []
                for _ in range(nreq):
                    op = OPS[int(rng.integers(0, len(OPS)))]
                    q = dict(op=op, a=int(rng.integers(0, G)), b=int(rng.integers(0, G)))
                    if op == "abs_within":
                        q.update(ordered=bool(rng.integers(0, 2)), self_pairs=bool(rng.integers(0, 2)),
                                 mean=bool(rng.integers(0, 2)))
                    elif op in ("abs_cross", "mann_whitney"):
                        q.update(self_pairs=bool(rng.integers(0, 2)), mean=bool(rng.integers(0, 2)))
                    items.append(q)
                reqs = mp.requests(items)
                for backend in bench.backends():
                    if backend == "cuda":
                        bench.measure("prefix upload", "cuda", lambda: pre.device(), repeats=1, phase="build", **p)
                    out = bench.measure("requests", backend, lambda: mp.query_arrays(pre, reqs, backend=backend),
                                        repeats=3 if nreq < 10_000 else 2, phase="first+warm", requests=nreq, **p)
                    if (out["status"] != 0).any():
                        raise SystemExit("PAIR-001: a benchmark request failed")
                if weight is None:
                    sample = items[: min(nreq, 20)]
                    answers = mp.query(pre, sample)
                    for q, r in zip(sample, answers):
                        want = numpy_answer(q["op"], cls[q["a"]], cls[q["b"]])
                        if q["op"] in ("abs_within", "abs_cross", "mann_whitney"):
                            if q.get("ordered") or q.get("mean") or q.get("self_pairs") or (
                                    q["op"] != "abs_within" and q["a"] == q["b"]):
                                continue                       # the sample checks the raw sums of distinct classes
                        if r["value"] != (want.numerator if isinstance(want, Fraction) and want.denominator == 1
                                          else want):
                            raise SystemExit(f"PAIR-001: {q} differs from NumPy: {r['value']} {want}")
                    if nreq <= 100:
                        def recon(items=items):
                            return [numpy_answer(q["op"], np.sort(raw[q["a"]]), np.sort(raw[q["b"]])) for q in items]
                        bench.measure("requests (reconstruction: sort + searchsorted per request)", "numpy", recon,
                                      repeats=1, phase="one-shot", requests=nreq, **p)

                        def warm(items=items):
                            return [numpy_answer(q["op"], cls[q["a"]], cls[q["b"]]) for q in items]
                        bench.measure("requests (rows pre-sorted)", "numpy", warm, repeats=2, phase="warm",
                                      requests=nreq, **p)
    # the output budget of a batch at the limit and one past
    scope = mbk.Scope.from_arrays({"x": rng.integers(0, 64, 1000)}, groups=rng.integers(0, 2, 1000), ngroups=2)
    pre = build(scope, None)
    reqs = mp.requests([dict(op="ks", a=0, b=1)] * 1000)
    bench.measure("requests at the output limit", "cpu", lambda: mp.query_arrays(pre, reqs, budget=Budget(output=1000)),
                  repeats=2, requests=1000)
    bench.refusal("requests one past the output limit", "cpu",
                  lambda: mp.query_arrays(pre, reqs, budget=Budget(output=999)), requests=1000)
    bench.threshold("requests per batch against the output budget", dict(requests=1000, output_budget=1000),
                    dict(requests=1000, output_budget=999))


def main():
    bench = Bench.from_args("PAIR-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    explicit(bench, rng)
    retained(bench, rng)
    bench.note("cpu = math_pair.c (single-threaded, one pass per class or one merge per class pair); cuda = "
               "math_pair_gpu.cu, one work item per request (device runs only); numpy = broadcasting over every "
               "pair (explicit) or sort + searchsorted per request (reconstruction); duckdb = self-joins.")
    bench.note("Support sorting (the DIST-001 histogram and the DIST-002 prefix state) is measured apart from the "
               "requests; a request never reads the rows. Ties: support 8 puts about rows / 8 elements on a value.")
    bench.write()


if __name__ == "__main__":
    main()
