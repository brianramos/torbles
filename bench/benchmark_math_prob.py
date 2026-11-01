# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROB-001 benchmark: declared finite-support models evaluated from retained statistics (a DIST-001 histogram aligned
to the declared support) for batches of parameter requests, against repeated support enumeration.

    python bench/benchmark_math_prob.py --capability PROB-001 --out FILE [--device] [--quick]

Grid (the inventory's first case): support sizes 8 / 64 / 1024 and 1 / 100 / 10,000 parameter requests (quick: 8 / 64
and 1 / 100), over 100,000 rows in 4 groups (quick: 10,000). Per support size, the state construction: the histogram
(the one scan of the rows) and the statistics aligned to the declared support (no row read). Per (support, requests):
an exponential family with two rational features (x, x^2 / d) evaluated natively with Z, log Z, the log-likelihood, its
gradient and Hessian (certified enclosures; CUDA with --device), first execution and warm reuse; a categorical model
(exact Z, log-likelihood enclosure and exact one-hot gradient) under a benchmark tensor budget of 2^22 request x support
entries, whose refusal is recorded as a threshold (never extrapolated). Baselines: NumPy float64 enumeration of the
support from the same retained counts (uncertified, vectorized over requests: logsumexp, moments), whose answers are
checked against the certified enclosures up to their own rounding; and repeated support enumeration with a rescan of
the rows per request (NumPy bincount, and DuckDB SQL: GROUP BY counts joined with the declared support, then the
log-likelihood in SQL), up to 100 requests. The inventory's second case (factorized inference: graph size and
elimination width) belongs to PROB-002 and is recorded as a note, not measured.
"""
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_dist as md, math_prob as mp
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError

GROUPS = 4


def data(rng, rows, d):
    """skewed values in [0, d): a discretized normal (every declared point may or may not be observed)"""
    x = np.clip(np.round(rng.normal(d / 2, d / 6, rows)), 0, d - 1).astype(np.int64)
    g = rng.integers(0, GROUPS, rows).astype(np.uint32)
    return x, g


def thetas(rng, n):
    num = rng.integers(-64, 65, (n, 2)).astype(np.int64)
    den = rng.integers(16, 65, (n, 2)).astype(np.int64)
    return num, den


def numpy_enumeration(counts, d, num, den, scales=False):
    """float64 enumeration of the support for every request (vectorized), from the retained counts of group 0; with
    scales, also the magnitudes of the terms each answer cancels (to judge the float's own rounding)"""
    xs = np.arange(d, dtype=np.float64)
    T = np.stack([xs, xs * xs / d], axis=1)                         # d x 2
    th = num / den                                                   # n x 2
    s = th @ T.T                                                     # n x d
    mx = s.max(axis=1, keepdims=True)
    e = np.exp(s - mx)
    Zs = e.sum(axis=1)
    logz = mx[:, 0] + np.log(Zs)
    p = e / Zs[:, None]
    N = counts.sum()
    ll = s @ counts - N * logz
    ET = p @ T
    grad = (counts @ T)[None, :] - N * ET
    ETT = np.einsum("nd,dj,dl->njl", p, T, T)
    hess = -N * (ETT - np.einsum("nj,nl->njl", ET, ET))
    if not scales:
        return logz, ll, grad, hess
    sc = dict(logz=np.maximum(1.0, np.abs(mx[:, 0]) + np.log(d)),
              loglik=np.abs(s) @ counts + N * (np.abs(mx[:, 0]) + np.log(d)),
              grad=(counts @ np.abs(T))[None, :] + N * (p @ np.abs(T)),
              hess=N * (ETT + np.einsum("nj,nl->njl", np.abs(ET), np.abs(ET))))
    return logz, ll, grad, hess, sc


def deviation(iv_lo, iv_hi, v, scale):
    """the largest distance of a float answer outside its certified enclosure, relative to the magnitude of the terms
    it cancels (float64 enumeration rounds at about 1e-16 of that magnitude; a certified enclosure is never wrong)"""
    if not v.size:
        return 0.0
    out = np.maximum(0.0, np.maximum(iv_lo - v, v - iv_hi)) / np.maximum(scale, 1e-300)
    return float(out.max())


def rescan_numpy(x, g, d, num, den):
    """repeated support enumeration with a rescan: per request the counts recomputed from the rows, then evaluated"""
    out = []
    for r in range(num.shape[0]):
        counts = np.bincount(x[g == 0], minlength=d).astype(np.float64)
        out.append(numpy_enumeration(counts, d, num[r:r + 1], den[r:r + 1])[1][0])
    return out


def rescan_duckdb(con, d, num, den):
    """per request: GROUP BY counts of the rows joined with the declared support, the log-likelihood in SQL"""
    out = []
    for r in range(num.shape[0]):
        a, b = float(num[r, 0]) / float(den[r, 0]), float(num[r, 1]) / float(den[r, 1])
        q = (f"WITH c AS (SELECT x, COUNT(*) AS n FROM t WHERE g = 0 GROUP BY x), "
             f"s AS (SELECT p.x, COALESCE(c.n, 0) AS n, {a!r} * p.x + {b!r} * p.x * p.x / {d} AS e "
             f"FROM support p LEFT JOIN c ON p.x = c.x), m AS (SELECT MAX(e) AS mx FROM s) "
             f"SELECT SUM(n * e) - SUM(n) * (MAX(mx) + LN(SUM(EXP(e - mx)))) FROM s, m")
        out.append(con.execute(q).fetchone()[0])
    return out


def grid(bench, rng):
    rows = 10_000 if bench.quick else 100_000
    supports = (8, 64) if bench.quick else (8, 64, 1024)
    requests = (1, 100) if bench.quick else (1, 100, 10_000)
    cat_budget = Budget(tensor=1 << 22)
    admitted, refused = None, None
    for d in supports:
        x, g = data(rng, rows, d)
        scope = mbk.Scope.from_arrays({"x": x}, groups=g, ngroups=GROUPS)
        S = mp.support(range(d))
        p = dict(rows=rows, support=d, groups=GROUPS)
        h = bench.measure("histogram (DIST-001, the one scan)", "cpu", lambda: md.histogram(scope, ["x"]), repeats=3,
                          phase="build", **p)
        st = bench.measure("statistics aligned to the declared support (no row read)", "cpu",
                           lambda: mp.stats(h, S), repeats=3, phase="build", **p)
        bench.records[-1].update(state_bytes=st.info["bytes"], observed=int(sum(1 for c in st.counts(0).values() if c)))
        E = mp.exponential(S, features=lambda v: (v, Fraction(v * v, d)))
        M = mp.categorical(S)
        counts0 = st.arrays()["counts"][0].astype(np.float64)
        frame = pd.DataFrame({"x": x, "g": g.astype(np.int64)})
        con = duckdb.connect()
        con.register("t", frame)
        con.register("support", pd.DataFrame({"x": np.arange(d, dtype=np.int64)}))
        for n in requests:
            num, den = thetas(rng, n)
            q = dict(p, requests=n)
            big = n * d >= 1_000_000
            for backend in bench.backends():
                ev = bench.measure("exponential family: Z, log Z, loglik, gradient, Hessian (retained state)", backend,
                                   lambda: E.evaluate(st, (num, den), grad=True, hess=True, backend=backend),
                                   repeats=2 if big else 4, phase="run", **q)
            a = ev.arrays()
            ans, gr, hs = a["answers"], a["grad"], a["hess"]
            bench.measure("numpy float64 enumeration from the retained counts (uncertified)", "numpy",
                          lambda: numpy_enumeration(counts0, d, num, den), repeats=2 if big else 4, phase="run", **q)
            logz_f, ll_f, grad_f, hess_f, sc = numpy_enumeration(counts0, d, num, den, scales=True)
            ok = (ans["status"] == 0) & (ans["group"] == 0)
            bench.records[-1].update(
                logz_deviation=deviation(ans["logz"][ok, 0], ans["logz"][ok, 1], logz_f[ok], sc["logz"][ok]),
                loglik_deviation=deviation(ans["loglik"][ok, 0], ans["loglik"][ok, 1], ll_f[ok], sc["loglik"][ok]),
                grad_deviation=deviation(gr[ok, :, 0], gr[ok, :, 1], grad_f[ok], sc["grad"][ok]),
                hess_deviation=deviation(hs[ok, 0, 0], hs[ok, 0, 1], hess_f[ok, 0, 0], sc["hess"][ok, 0, 0]),
                max_z_rel_width=float(np.max(ans["width"][ok, 0])) if ok.any() else None,
                max_logz_width=float(np.max(ans["width"][ok, 1])) if ok.any() else None,
                max_loglik_width=float(np.max(ans["width"][ok, 2])) if ok.any() else None,
                refused_requests=int((ans["status"] != 0).sum()))
            if n <= 100:
                bench.measure("repeated support enumeration with a rescan per request (numpy bincount)", "numpy",
                              lambda: rescan_numpy(x, g, d, num, den), repeats=2, phase="run", **q)
                bench.measure("repeated support enumeration with a rescan per request (DuckDB SQL)", "duckdb",
                              lambda: rescan_duckdb(con, d, num, den), repeats=2, phase="run", **q)
            # categorical: exact Z, the log-likelihood enclosure and the exact one-hot gradient (natural parameters)
            W = rng.integers(0, 100, (n, d)).astype(np.int64)
            W[:, 0] = np.maximum(W[:, 0], 1)
            try:
                for backend in bench.backends():
                    bench.measure("categorical: exact Z, loglik enclosure, exact one-hot gradient (retained state)",
                                  backend, lambda: M.evaluate(st, W, grad=True, backend=backend, budget=cat_budget),
                                  repeats=2 if big else 4, phase="run", **q)
                if admitted is None or n * d > admitted["tensor"]:
                    admitted = dict(support=d, requests=n, tensor=n * d)
            except AlgebraBudgetError:
                bench.refusal("categorical: exact Z, loglik enclosure, exact one-hot gradient (retained state)", "cpu",
                              lambda: M.evaluate(st, W, grad=True, budget=cat_budget), **q)
                if refused is None or n * d < refused["tensor"]:
                    refused = dict(support=d, requests=n, tensor=n * d)
        con.close()
    bench.threshold("categorical requests x support (benchmark budget tensor = 2^22)", admitted=admitted,
                    refused=refused, budget=dict(tensor=1 << 22))


def main():
    bench = Bench.from_args("PROB-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    grid(bench, rng)
    bench.note("Inventory case 2 (factorized inference: vary graph size and elimination width on brute-force-checked "
               "tiny instances and bounded larger models; report intermediate support and precision) belongs to "
               "PROB-002 (factorized probabilistic inference), which is not implemented here: not measured.")
    bench.note("CUDA timings exist only with --device on an actual GPU; without it the CUDA path is not timed (the "
               "tests check its emulated results bit for bit against the CPU).")
    bench.note("NumPy and DuckDB baselines are uncertified float64. The *_deviation fields are the largest distance of "
               "a NumPy float answer outside its certified enclosure, relative to the magnitude of the terms it "
               "cancels: float64's own error, which grows with |theta . T| (exponents up to about 10^4 at 1024 "
               "points give relative errors up to |theta . T| 2^-53).")
    bench.write()


if __name__ == "__main__":
    main()
