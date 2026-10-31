# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEN-002 benchmark: generating states with explicit independence (factors, keyed families, joint states),
their coefficient / moment / evaluation requests, convolutions (declared products, powers, the conditional per-key
form, batched 192-bit convolutions) and bounded truncated series (powers, walk tables).

    python bench/benchmark_math_conv.py --capability GEN-002 --out FILE [--device] [--quick]

States: 100k rows (quick: 20k) over 64 keys, supports 8 / 64 / 1024 values centred on 0 and a two-dimensional
support (x by 16 values), sparse and dense: the histogram (DIST-001), the factor (the marginal over the keys), the
two-variable joint state and the keyed family are built and measured apart. Requests, measured separately on a factor
and on a derived state (the conditional product): coefficients (1 / 100 / 1,000), raw moments 1..4, exact evaluation
(1 / 10 / 100 rational points). Baselines: DuckDB GROUP BY counts (coefficients), and the complete reconstruction of
the key-conditioned join by DuckDB (a self-join on the key, GROUP BY a.x + b.y; over a 20k-row slice, quick: 5k), the
per-row sum (GROUP BY x + y) for the joint state; every native answer is checked equal.
Convolutions: declared powers G^k, k = 4 / 16 / 64 (support growth, coefficient pairs and the intermediate bytes
checked before the first product; refusals past the benchmark budget of 2^24 pairs per product are recorded, never
extrapolated), truncated powers G^8 mod z^N and bounded-series powers and walk tables for N = 4 / 16 / 64, against
explicit tiny products (dict polynomials and enumerated walks, support 8); batched 192-bit convolutions of
1 / 100 / 1,000 pairs against Python-int products. CUDA (--device): the batch, the conditional product and the walks.
"""
import itertools
import time
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_conv as mc, math_dist as md
from torbles.math_budgets import Budget


def pmul(a, b):
    out = {}
    for u, x in a.items():
        for v, y in b.items():
            out[u + v] = out.get(u + v, 0) + x * y
    return out


def bound_refusal(bench, case, fn, **params):
    """run fn() expecting a declared-bound refusal (OverflowError) and record it"""
    t0 = time.perf_counter()
    try:
        fn()
    except OverflowError as e:
        bench.records.append(dict(case=case, backend="cpu", phase="refusal", params=params, refused=True,
                                  resource="declared bound", refusal_seconds=time.perf_counter() - t0, message=str(e)))
        return
    bench.records.append(dict(case=case, backend="cpu", phase="refusal", params=params, refused=False))


def requests(bench, f, rng, label, **p):
    terms = f.terms()
    values = list(terms)
    for n in (1, 100, 1000):
        vs = [int(values[int(i)]) for i in rng.integers(0, len(values), n)]
        got = bench.measure(f"coefficient requests ({label})", "cpu", lambda: [f.coefficient(v) for v in vs],
                            repeats=3, phase="first+warm", requests=n, **p)
        if got != [terms[v] for v in vs]:
            raise SystemExit(f"GEN-002: coefficients of the {label} differ from its terms")
    for k in (1, 2, 3, 4):
        m = bench.measure(f"raw moment ({label})", "cpu", lambda: f.moment(k), repeats=3, order=k, **p)
        if m != sum(c * v ** k for v, c in terms.items()):
            raise SystemExit(f"GEN-002: moment {k} of the {label} differs")
    for n in (1, 10, 100):
        pts = [Fraction(int(a) * int(s), int(b)) for a, s, b in
               zip(rng.integers(1, 10, n), rng.choice([-1, 1], n), rng.integers(1, 6, n))]
        bench.measure(f"exact evaluation ({label})", "cpu", lambda: [f(z) for z in pts], repeats=2,
                      phase="first+warm", points=n, **p)


def states(bench, rng):
    rows = 20_000 if bench.quick else 100_000
    keys = 64
    for support in (8, 64, 1024):
        x = rng.integers(-(support // 2), support - support // 2, rows).astype(np.int64)
        y = rng.integers(0, 16, rows).astype(np.int64)
        k = rng.integers(0, keys, rows).astype(np.int64)
        valid = rng.random(rows) > 0.02
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, valid={"x": valid}, groups=k, ngroups=keys)
        p = dict(rows=rows, support=support, keys=keys)
        hx = bench.measure("histogram (DIST-001)", "cpu", lambda: md.histogram(scope, ["x"]), repeats=2,
                           phase="build", **p)
        hy = md.histogram(scope, ["y"])
        hxy = md.histogram(scope, ["x", "y"])
        for repr_ in ("sparse", "dense"):
            f = bench.measure("factor (marginal over the keys)", "cpu", lambda: mc.factor(hx, repr=repr_), repeats=3,
                              phase="build", repr=repr_, **p)
            bench.records[-1].update(terms=f.gen.info["terms"], bytes=f.gen.info["bytes"])
            j = bench.measure("joint state (2-D support, x by 16)", "cpu", lambda: mc.factor(hxy, repr=repr_),
                              repeats=2, phase="build", repr=repr_, **p)
            bench.records[-1].update(terms=j.gen.info["terms"], bytes=j.gen.info["bytes"])
        fam = bench.measure("keyed family (per-key dense states)", "cpu", lambda: mc.family(hx), repeats=3,
                            phase="build", **p)
        bench.records[-1].update(coefficients=fam.info["coefficients"], bytes=fam.info["bytes"])
        fy = mc.family(hy)
        X = mc.factor(hx)
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"x": pd.Series(np.where(valid, x, 0), dtype="Int64").where(valid), "y": y,
                                        "k": k}))
        counts = dict(bench.measure("coefficients (GROUP BY)", "duckdb", lambda: con.execute(
            "SELECT x, COUNT(*) FROM t WHERE x IS NOT NULL GROUP BY x").fetchall(), repeats=2, **p))
        if X.terms() != {int(a): int(b) for a, b in counts.items()}:
            raise SystemExit(f"GEN-002: factor coefficients differ from DuckDB at {p}")
        requests(bench, X, rng, "factor", **p)
        # the per-row sum from the joint state, against DuckDB's GROUP BY x + y
        js = bench.measure("joint sum G(z, z) (per-row x + y)", "cpu", lambda: mc.joint_sum(mc.factor(hxy)),
                           repeats=2, **p)
        want = dict(bench.measure("per-row x + y (GROUP BY)", "duckdb", lambda: con.execute(
            "SELECT x + y, COUNT(*) FROM t WHERE x IS NOT NULL GROUP BY 1").fetchall(), repeats=2, **p))
        if js.terms() != {int(a): int(b) for a, b in want.items()}:
            raise SystemExit("GEN-002: the joint sum differs from DuckDB")
        # the conditional form (per key, summed) against the complete reconstruction of the key-equijoin
        for backend in bench.backends():
            c = bench.measure("conditional product (key-equijoin)", backend,
                              lambda: mc.conditional(fam, fy, backend=backend), repeats=3, phase="first+warm", **p)
        need = mc.conditional_need(fam, fy)
        bench.records[-1].update(pairs=need["tensor"], per_key_coefficients=need["support"], bytes_checked=need["bytes"],
                                 terms=len(c.terms()), join_tuples=c.info["tuples"])
        requests(bench, c, rng, "conditional product", **p)
        cut = 5_000 if bench.quick else 20_000
        sx, sk, sy, sv = x[:cut], k[:cut], y[:cut], valid[:cut]
        sl = mbk.Scope.from_arrays({"x": sx, "y": sy}, valid={"x": sv}, groups=sk, ngroups=keys)
        sc = mc.conditional(mc.family(md.histogram(sl, ["x"])), mc.family(md.histogram(sl, ["y"])))
        con.register("s", pd.DataFrame({"x": pd.Series(np.where(sv, sx, 0), dtype="Int64").where(sv), "y": sy,
                                        "k": sk}))
        q = "SELECT a.x + b.y, COUNT(*) FROM s a JOIN s b ON a.k = b.k WHERE a.x IS NOT NULL GROUP BY 1"
        rec = dict(bench.measure("key-equijoin GROUP BY a.x + b.y (complete reconstruction)", "duckdb",
                                 lambda: con.execute(q).fetchall(), repeats=2, slice_rows=cut, **p))
        bench.measure("conditional product (same slice)", "cpu",
                      lambda: mc.conditional(mc.family(md.histogram(sl, ["x"])), mc.family(md.histogram(sl, ["y"]))),
                      repeats=2, slice_rows=cut, **p)
        if sc.terms() != {int(a): int(b) for a, b in rec.items()}:
            raise SystemExit("GEN-002: the conditional product differs from the reconstructed join")
        con.close()


def convolutions(bench, rng):
    budget = Budget(tensor=1 << 24)                    # the benchmark's limit: 2^24 coefficient pairs per product
    for support in (8, 64, 1024):
        x = rng.integers(-(support // 2), support - support // 2, 5000).astype(np.int64)
        scope = mbk.Scope.from_arrays({"x": x})
        X = mc.factor(md.histogram(scope, ["x"]))
        d = mc.independent(scope)
        for k in (4, 16, 64):
            need = mc.need([X], mult=[k], budget=budget)
            p = dict(support=support, power=k, rows=5000)
            if any(need[r] > budget.limits[r] for r in ("tensor", "support", "bytes")):
                bench.refusal("declared power G^k", "cpu", lambda: mc.power(X, k, declaration=d, budget=budget),
                              predicted=need, **p)
                continue
            r = bench.measure("declared power G^k", "cpu", lambda: mc.power(X, k, declaration=d, budget=budget),
                              repeats=1 if k == 64 or (support == 1024 and k >= 16) else 2, phase="build", **p)
            bench.records[-1].update(terms=r.gen.info["terms"], max_bits=r.gen.info["max_bits"],
                                     support_growth=need["terms"], pairs_checked=need["tensor"],
                                     intermediate_bytes_checked=need["bytes"])
            if r.total() != 5000 ** k:
                raise SystemExit("GEN-002: G^k(1) != rows^k")
            if support == 8:
                A = X.terms()

                def explicit(A=A, k=k):
                    out = {0: 1}
                    for _ in range(k):
                        out = pmul(out, A)
                    return out
                want = bench.measure("power (explicit dict products)", "python", explicit, repeats=1, **p)
                if r.terms() != {v: c for v, c in want.items() if c}:
                    raise SystemExit("GEN-002: G^k differs from the explicit products")
        ind = mc.from_terms({v: 1 for v in range(support)})            # one of each value: small coefficients
        for order in (4, 16, 64):
            p = dict(support=support, series_order=order, power=8)
            bench.measure("truncated declared power G^8 mod z^N", "cpu",
                          lambda: mc.power(X, 8, order=order, declaration=d), repeats=2, **p)
            bench.records[-1].update(support_growth=mc.need([X], mult=[8], order=order)["terms"],
                                     intermediate_bytes_checked=mc.need([X], mult=[8], order=order)["bytes"])
            s = mc.series(ind, order, bound=(1 << 62))
            bs = bench.measure("bounded series power s^8 (192-bit sums, declared bound)", "cpu", lambda: s.pow(8),
                               repeats=3, **p)
            want = mc.power(ind, 8, order=order)
            if {e: c for e, c in enumerate(bs.coefficients) if c} != want.terms():
                raise SystemExit("GEN-002: the bounded series power differs from the truncated power")
            for backend in bench.backends():
                w = bench.measure("walk table, lengths <= 8", backend, lambda: s.walks(8, backend=backend), repeats=3,
                                  **p)
            bench.records[-1].update(maxabs=w["maxabs"], lengths=w["lengths"])
            tight = mc.series(ind, order, bound=max(1, w["maxabs"] - 1))
            bound_refusal(bench, "walk table one past its declared bound", lambda: tight.walks(8),
                          bound=max(1, w["maxabs"] - 1), **p)
            if support == 8 and order == 4:                    # 4^8 step sequences (larger steps exceed N)
                def walks_enum(order=order):
                    tot = [0] * order
                    for length in range(9):
                        for seq in itertools.product(range(min(support, order)), repeat=length):
                            if sum(seq) < order:
                                tot[sum(seq)] += 1
                    return tot
                want = bench.measure("walk table (enumerated walks)", "python", walks_enum, repeats=1, **p)
                if want != w["total"].coefficients:
                    raise SystemExit("GEN-002: the walk totals differ from the enumerated walks")
        # batched exact convolutions of 1 / 100 / 1,000 pairs against Python ints
        for n in (1, 100, 1000):
            A = [[int(v) for v in rng.integers(-1 << 40, 1 << 40, min(support, 64))] for _ in range(n)]
            B = [[int(v) for v in rng.integers(-1 << 40, 1 << 40, min(support, 64))] for _ in range(n)]
            p = dict(support=support, pairs=n, length=min(support, 64))
            for backend in bench.backends():
                got = bench.measure("batched 192-bit convolutions", backend, lambda: mc.convolve(A, B, backend=backend),
                                    repeats=3, phase="first+warm", **p)
            if n <= 100:
                def py(A=A, B=B):
                    return [[sum(a[i] * b[e - i] for i in range(len(a)) if 0 <= e - i < len(b))
                             for e in range(len(a) + len(b) - 1)] for a, b in zip(A, B)]
                want = bench.measure("convolutions (Python ints)", "python", py, repeats=1, **p)
                if got != want:
                    raise SystemExit("GEN-002: the batched convolutions differ from Python")
    # the refusal threshold of the declared power chain (support 1024)
    x = rng.integers(-512, 512, 5000).astype(np.int64)
    scope = mbk.Scope.from_arrays({"x": x})
    X, d = mc.factor(md.histogram(scope, ["x"])), mc.independent(scope)
    admitted = refused = None
    for k in range(2, 65):
        need = mc.need([X], mult=[k], budget=budget)
        if any(need[r] > budget.limits[r] for r in ("tensor", "support", "bytes")):
            refused = dict(power=k, **{r: need[r] for r in ("tensor", "support", "bytes")})
            bench.refusal("declared power past the benchmark budget (support 1024)", "cpu",
                          lambda: mc.power(X, k, declaration=d, budget=budget), power=k)
            break
        admitted = dict(power=k, **{r: need[r] for r in ("tensor", "support", "bytes")})
    bench.threshold("declared power of a 1024-value support (2^24 pairs per product)", admitted,
                    refused or "none up to G^64", limits={r: budget.limits[r] for r in ("tensor", "support", "bytes")})
    # the independence check itself: the refusal of an undeclared product of one source (no allocation)
    Y = mc.factor(md.histogram(scope, ["x"]))
    bench.measure("independence check (refused: same source, no declaration)", "cpu",
                  lambda: mc.check([X, Y]) is not None, repeats=5)


def main():
    bench = Bench.from_args("GEN-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    states(bench, rng)
    convolutions(bench, rng)
    bench.note("cpu = math_conv.c on GEN-001 (math_gen.c: products, powers and requests with exact big integers); "
               "per-key products and batches accumulate exactly in 192 bits (tr_conv_coef192); cuda = math_conv_gpu.cu "
               "(device runs only); duckdb = GROUP BY counts, the self-join on the key (the complete reconstruction "
               "of the key-conditioned join) and the per-row sums; python = dict polynomials, enumerated walks, "
               "Python-int convolutions.")
    bench.note("A declared power checks its largest step (support growth, pairs, intermediate bytes) before the first "
               "product; refused powers record that need. A bounded series refuses a coefficient past its declared "
               "bound (OverflowError), recorded with the walk table's largest magnitude.")
    bench.write()


if __name__ == "__main__":
    main()
