# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""UNC-002 benchmark: explicit approximations with certified error records.

    python bench/benchmark_math_approx.py --capability UNC-002 --out FILE [--device] [--quick]

Case 1, how approximate bounds change UNC-001's decided fraction, with 1% / 50% / 99% ambiguous groups: grouped
rows whose groups are decided by an interval threshold (exp(x) > e^(1/2) through a certified series of degree 2 / 4 /
8 / 16; its binary64 form in the row-evaluation baseline), by a correlated cancellation ((exp + sin - sin)(x) > e^(1/2): the polynomials cancel
exactly, the errors add and are never cancelled) and by a modular constraint (x on the lattice 7 k + 3, the threshold
between lattice points 7 k + 1: a group interval straddles it while no row does). The proof and pruning work (groups decided
ALL / NONE by the binary64 UNC-001 proof of p(t) op T -+ e, rows proven) and the concrete fallback work (rows
evaluated by their certified enclosure, rows left undecided) are recorded separately with the timing of the whole
approximate filter, against the baselines: the exact UNC-001 filter of the equivalent exact predicate (x > 1/2: what
an exact relationship would decide), the certified row evaluation of every row (no pruning) and DuckDB's binary64
EXP (fast, uncertified).
Case 2, coefficient precision, approximation rank and error budget: series quantized to 8 / 16 / 24 / 32 / 48 bits
and in binary64, compared with a 90-digit decimal oracle at 33 points; rank-k cross and seeded sketch approximations
of small integer matrices compared with the exact residual (Fractions) and numpy's singular values; error budgets
from 1e-3 to 1e-15 at degrees 2 .. 24, recording the unproved cases (the measured error is within the budget but the
certified bound is not: refused, never reported as certified).
Case 3, budget thresholds: the series degree at a fixed degree budget and the rows of a row evaluation at a fixed
CUDA device-copy budget (largest admitted, smallest refused).
"""
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd

from math_bench import Bench
import duckdb
from torbles import math_approx as ma
from torbles import math_bank as mbk
from torbles import math_unc as mu
from torbles.math_budgets import Budget
from torbles.math_errors import NotConvergedError

PREC = 90


def oracle(function, q):
    with localcontext() as ctx:
        ctx.prec = PREC
        x = Decimal(q.numerator) / Decimal(q.denominator)
        if function == "exp":
            return Fraction(x.exp())
        if function == "log":
            return Fraction(x.ln())
        if function == "sqrt":
            return Fraction(x.sqrt())
        if function in ("sin", "cos"):
            start = 1 if function == "sin" else 0
            term = Decimal(1)
            for i in range(1, start + 1):
                term = term * x / i
            total, k = Decimal(0), 0
            while abs(term) > Decimal(10) ** -(PREC + 5):
                total += term
                k += 1
                n = 2 * k + start
                term = -term * x * x / ((n - 1) * n)
            return Fraction(total)
    raise ValueError(function)


def grid(a, b, n=33):
    a, b = Fraction(a), Fraction(b)
    pts = {a, b}
    for k in range(n):
        t = Fraction(math.cos(math.pi * (2 * k + 1) / (2 * n))).limit_denominator(10 ** 9)
        pts.add(min(max((a + b) / 2 + (b - a) / 2 * t, a), b))
    return sorted(pts)


# ------------------------------------------------------------------ case 1
def table(rng, groups, per, ambiguous, kind):
    """grouped rows of x (logical, scale 3: x / 1000 in [0, 1]); a fraction `ambiguous` of the groups straddles 1/2,
    the others lie far from it or within 7/1000 of it (decided exactly, not by a coarse approximation)"""
    amb = np.zeros(groups, bool)
    amb[rng.choice(groups, size=max(1, int(round(ambiguous * groups))), replace=False)] = True
    x = np.empty(groups * per, np.int64)
    for g in range(groups):
        rows = slice(g * per, (g + 1) * per)
        if kind == "modular":                                 # the lattice 7 k + 1: 498 and 505, never 500
            if amb[g]:
                v = 7 * rng.integers(70, 73, per) + 1        # 491, 498, 505: straddles 1/2, no row at it
            else:
                v = 7 * rng.integers(0, 60, per) + 1 if g % 2 else 7 * rng.integers(80, 140, per) + 1
        else:                                                 # far below / far above / just below / just above
            if amb[g]:
                v = 500 + rng.integers(-6, 7, per)
            else:
                lo, hi = ((0, 480), (520, 1000), (493, 500), (501, 508))[g % 4]
                v = rng.integers(lo, hi, per)
        x[rows] = v
    groups_id = np.repeat(np.arange(groups), per)
    return x, groups_id, int(amb.sum())


def decided(bench, groups, per):
    T = oracle("exp", Fraction(1, 2))
    Tq = Fraction(T).limit_denominator(10 ** 15)              # a rational threshold just around e^(1/2)
    for kind in ("interval threshold", "correlated cancellation", "modular constraint"):
        for frac in (0.01, 0.5, 0.99):
            rng = np.random.default_rng(bench.seed + int(frac * 100))
            x, gid, namb = table(rng, groups, per, frac, "modular" if kind == "modular constraint" else "interval")
            scope = bench.measure("state construction: scope", "cpu",
                                  lambda: mbk.Scope.from_arrays({"x": x}, scales={"x": 3}, groups=gid, ngroups=groups),
                                  repeats=3, phase="build", scenario=kind, ambiguous=frac, groups=groups,
                                  rows=groups * per)
            state = bench.measure("state construction: abstract inputs", "cpu", lambda: mu.State.from_scope(scope),
                                  repeats=3, phase="build", scenario=kind, ambiguous=frac, groups=groups)
            p0 = dict(scenario=kind, ambiguous=frac, groups=groups, rows=groups * per, ambiguous_groups=namb)
            exact = mu.Program(mu.column("x") > Fraction(1, 2), ["x"])
            bits, est = bench.measure("baseline: exact UNC-001 filter of x > 1/2", "cpu",
                                      lambda: state.filter(scope, exact), repeats=5, phase="baseline", **p0)
            bench.records[-1]["result"] = {k: est[k] for k in ("rows_true", "rows_proven", "rows_evaluated",
                                                                "groups_all", "groups_none", "groups_unknown")}
            for degree in (2, 4, 8, 16, "binary64"):
                f = ma.series("exp", 0, 1, 16 if degree == "binary64" else degree)
                if kind == "correlated cancellation":
                    s = ma.series("sin", 0, 1, 8)
                    f = (f + s) - s
                if degree == "binary64":
                    f = f.to_float()
                p = dict(p0, degree=degree, error=float(f.error))
                for backend in bench.backends():
                    bench.measure("baseline: certified row evaluation of every row", backend,
                                  lambda: f.evaluate(scope, "x", backend=backend), repeats=3, phase="baseline", **p)
                if degree == "binary64":                      # the filter proves series; the float form is per row
                    continue
                b, u, st = bench.measure("approximate filter: proof + fallback", "cpu",
                                         lambda: ma.filter(state, scope, "x", f, ">", Tq), repeats=5, phase="filter", **p)
                bench.records[-1]["result"] = dict(
                    proof_groups_all=st["groups_all"], proof_groups_none=st["groups_none"],
                    proof_groups_unknown=st["groups_unknown"], rows_proven=st["rows_proven"],
                    fallback_rows_evaluated=st["rows_evaluated"], fallback_rows_undecided=st["rows_undecided"],
                    rows_true=st["rows_true"], decided_fraction=(st["rows_proven"] + st["rows_evaluated"] -
                                                                 st["rows_undecided"]) / max(1, groups * per),
                    exact_groups_decided=est["groups_all"] + est["groups_none"],
                    approx_groups_decided=st["groups_all"] + st["groups_none"], error_used=st["error_used"])
            con = duckdb.connect()
            con.register("t", pd.DataFrame({"x": x}))
            bench.measure("baseline: DuckDB binary64 EXP (uncertified)", "duckdb",
                          lambda: int(con.execute("SELECT count(*) FROM t WHERE exp(x / 1000.0) > exp(0.5)").fetchone()[0]),
                          repeats=5, phase="baseline", **p0)
            con.close()


# ------------------------------------------------------------------ case 2
def precision(bench, quick):
    unproved = []
    for function, a, b, n in (("exp", 0, 1, 12), ("log", 1, 2, 12), ("sin", -1, 2, 14)):
        f = ma.series(function, a, b, n)
        pts = grid(a, b, 17 if quick else 33)
        ref = {x: oracle(function, x) for x in pts}
        for bits in (8, 16, 24, 32, 48, "exact", "binary64"):
            if bits == "exact":
                g = f
            elif bits == "binary64":
                g = f.to_float()
            else:
                g = bench.measure("quantization", "cpu", lambda: f.quantize(bits), repeats=3, phase="build",
                                  function=function, degree=n, bits=bits)
            m = max(abs(g.value(x) - ref[x]) for x in pts)
            bench.records.append(dict(case="certified vs measured (coefficient precision)", backend="cpu", phase="proof",
                                      params=dict(function=function, degree=n, bits=bits),
                                      result=dict(certified=float(g.error), measured=float(m),
                                                  ratio=float(g.error / m) if m else None, holds=bool(m <= g.error))))
    rng = np.random.default_rng(bench.seed)
    sizes = (6, 12) if quick else (6, 12, 24)
    for size in sizes:
        A = np.ascontiguousarray(rng.integers(-20, 21, (size, size // 2)) @ rng.integers(-3, 4, (size // 2, size))
                                 + np.diag(rng.integers(0, 3, size)), dtype=np.int64)
        sv = np.linalg.svd(A.astype(float), compute_uv=False)
        for k in sorted({1, 2, size // 4, size // 2, size - 1}):
            for method in ("cross", "sketch"):
                kw = dict(method=method, seed=7, probes=16) if method == "sketch" else dict(method="cross")
                f = bench.measure("low-rank approximation", "cpu", lambda: ma.lowrank(A, k, **kw), repeats=3,
                                  phase="build", size=size, rank=k, method=method)
                Af = [[Fraction(int(v)) for v in row] for row in A]
                E = [[Af[i][j] - e for j, e in enumerate(row)] for i, row in enumerate(f.entries())]
                F2 = sum(v * v for row in E for v in row)
                bnd = f.bounds()
                opt = float(np.sqrt(np.sum(sv[k:] ** 2)))         # the best rank-k Frobenius error (Eckart-Young)
                bench.records[-1]["result"] = dict(
                    label=f.label, frobenius_bound=float(bnd["frobenius"]), frobenius_exact=math.sqrt(float(F2)),
                    spectral_bound=float(bnd["spectral"]), sigma_k_plus_1=float(sv[k]) if k < len(sv) else 0.0,
                    best_rank_k_frobenius=opt, holds=bool(bnd["frobenius"] ** 2 >= F2),
                    failure=float(f.provenance.get("failure") or 0))
    for function, a, b in (("exp", 0, 1), ("log", 1, 2), ("cos", 0, 3)):
        pts = grid(a, b, 9 if quick else 17)
        ref = {x: oracle(function, x) for x in pts}
        for n in (2, 4, 8, 12, 16, 24):
            f = ma.series(function, a, b, n)
            m = max(abs(f.value(x) - ref[x]) for x in pts)
            for e in (3, 6, 9, 12, 15):
                budget = Fraction(1, 10 ** e)
                rec = dict(case="error budget", backend="cpu", phase="proof",
                           params=dict(function=function, degree=n, budget=f"1e-{e}"))
                try:
                    ma.series(function, a, b, n, error_budget=budget)
                    rec["result"] = dict(certified=True, bound=float(f.error), measured=float(m))
                except NotConvergedError:
                    rec["result"] = dict(certified=False, bound=float(f.error), measured=float(m),
                                         unproved=bool(m <= budget))
                    if m <= budget:
                        unproved.append(dict(function=function, degree=n, budget=f"1e-{e}", bound=float(f.error),
                                             measured=float(m)))
                bench.records.append(rec)
    bench.records.append(dict(case="unproved cases", backend="cpu", phase="summary", params={}, unproved=unproved))
    bench.note(f"case 2: {len(unproved)} (function, degree, budget) cases whose measured error is within the budget "
               "but whose certified bound is not: refused (NotConvergedError), never reported as certified")


# ------------------------------------------------------------------ case 3
def thresholds(bench):
    limit = 12
    admitted, refused = [], []
    for n in (10, 11, 12, 13, 14):
        rec = bench.refusal("series degree", "cpu", lambda: ma.series("exp", 0, 1, n, budget=Budget(degree=limit)),
                            degree=n, degree_limit=limit)
        (refused if rec["refused"] else admitted).append(n)
    bench.threshold(f"series degree at degree budget {limit}", max(admitted, default=None), min(refused, default=None))
    f = ma.series("exp", 0, 1, 8)
    base = mbk.Scope.from_arrays({"x": np.arange(4096, dtype=np.int64) % 1000}, scales={"x": 3})
    limit = f.need(base, "x", "cuda")["bytes"]
    admitted, refused = [], []
    for rows in (4032, 4064, 4096, 4128, 4160):
        sc = mbk.Scope.from_arrays({"x": np.arange(rows, dtype=np.int64) % 1000}, scales={"x": 3})
        rec = bench.refusal("CUDA device bytes of a row evaluation", "cpu",
                            lambda: f.evaluate(sc, "x", backend="cuda", budget=Budget(bytes=limit)),
                            rows=rows, bytes_limit=limit)
        (refused if rec["refused"] else admitted).append(rows)
    bench.threshold(f"rows at a CUDA device-copy budget of {limit} bytes", max(admitted, default=None),
                    min(refused, default=None))


def main():
    bench = Bench.from_args("UNC-002", __doc__)
    groups, per = (40, 25) if bench.quick else (400, 100)
    decided(bench, groups, per)
    precision(bench, bench.quick)
    thresholds(bench)
    bench.note("CPU = native libalg; CUDA timings only with --device (an emulated run is never timed as CUDA); the "
               "approximate filter decides groups by the binary64 UNC-001 proof of the approximating polynomial with "
               "its certified error and the other rows by their certified enclosure; DuckDB's EXP is binary64 and "
               "uncertified (a speed baseline, not an oracle).")
    bench.write()


if __name__ == "__main__":
    main()
