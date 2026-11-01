# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SPEC-002 benchmark: exact Walsh spectra and parity correlations of bounded Boolean histograms.

    python bench/benchmark_math_spec.py --capability SPEC-002 --out FILE [--device] [--quick]

Case 1 (the full 2^k domain): k = 3 / 6 / 10 / 12 / 16 / 20 conditions (3 / 6 / 10 / 12 with --quick), histograms of
independent bit conditions over 20k and 200k rows (the row count changes the scan, never the transform): the
histogram (state construction), the forward transform and the inverse (first and warm), each with its reported cost
(2^k states, k passes, k 2^(k-1) butterflies, bytes) and normalization, Parseval's identity, and the baselines NumPy
(an int64 fast Walsh-Hadamard transform in vectorized passes, and the dense Sylvester-Hadamard matrix product for
k <= 10). Every baseline's spectrum is compared with the native one.
Case 2 (fixed frequency sets): on a k = 12 histogram, frequency sets of 1 / 8 / 32 subsets: their construction as new
frequencies (direct sums over the 2^k cells, no full transform) against one full transform and lookups; then 1 / 100
/ 10,000 evaluation requests of the fixed expansion (the fixed-basis action: m terms per point, exact dyadic values)
against NumPy (parities by XOR folding, float64 sums of integer terms below 2^53, then compared exactly); and the
truncation to the same sizes with its exact residual energy (an explicit approximation). Then the refusal threshold of
the transform tensor, measured at the last admitted and the first refused point. CUDA is timed only with --device.
"""
import random

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_lattice as ml
from torbles import math_spec as ms
from torbles.math_budgets import Budget


def table(rng, rows):
    cols = {f"u{i}": rng.integers(0, 1 << 20, rows) for i in range(2)}
    return torbles.from_ddl("CREATE TABLE t (u0 BIGINT, u1 BIGINT)", {"t": pd.DataFrame(cols)})


def conditions(k):
    return [mb.condition(f"u{i % 2}", "bit", i // 2) for i in range(k)]


def sylvester_hadamard(n):
    """the Sylvester Hadamard matrix of order n = 2^k, H_2n = [[H, H], [H, -H]] (what scipy.linalg.hadamard builds)"""
    H = np.ones((1, 1), dtype=np.int64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def numpy_fwht(v):
    a = np.array(v, dtype=np.int64)
    n, h = len(a), 1
    while h < n:
        b = a.reshape(-1, 2, h)
        x, y = b[:, 0, :].copy(), b[:, 1, :].copy()
        b[:, 0, :] = x + y
        b[:, 1, :] = x - y
        a = b.reshape(n)
        h *= 2
    return a


def parity(x):
    x = x.copy()
    for s in (32, 16, 8, 4, 2, 1):
        x ^= x >> np.uint64(s)
    return (x & np.uint64(1)).astype(np.int64)


def case_domain(bench, rng):
    ks = (3, 6, 10, 12) if bench.quick else (3, 6, 10, 12, 16, 20)
    for rows in ((20_000,) if bench.quick else (20_000, 200_000)):
        db = table(rng, rows)
        scope = mbk.Scope.from_table(db["t"], ["u0", "u1"])
        for k in ks:
            masks = mb.Masks.from_scope(scope, conditions(k))
            h = bench.measure("histogram (state construction)", "cpu", lambda: ml.Histogram.build(masks), repeats=2,
                              phase="build", rows=rows, conditions=k)
            vec = h.vector("complete")
            p = dict(rows=rows, conditions=k, states=1 << k)
            for backend in bench.backends():
                f = bench.measure("forward Walsh transform", backend, lambda: ms.walsh(vec, backend=backend),
                                  repeats=4, **p)
                bench.records[-1]["result"] = dict(f["cost"], normalization="forward, unnormalized")
                spectrum = np.array(f["spectrum"], dtype=np.int64)
                inv = bench.measure("inverse Walsh transform", backend,
                                    lambda: ms.inverse(spectrum, backend=backend, arrays=True), repeats=3, **p)
                bench.records[-1]["result"] = dict(inv["cost"], normalization="inverse, divided by 2^k")
                if inv["den_log2"].any() or inv["numerators"][:, 0].tolist() != vec.tolist():
                    raise SystemExit(f"round trip failed at k={k}")
            par = bench.measure("Parseval energies", "cpu", lambda: ms.parseval(vec, f["spectrum"])["holds"],
                                repeats=3, **p)
            if not par:
                raise SystemExit(f"Parseval failed at k={k}")
            got = bench.measure("forward Walsh transform", "numpy", lambda: numpy_fwht(vec), repeats=4, **p)
            if got.tolist() != f["spectrum"]:
                raise SystemExit(f"NumPy disagrees at k={k}")
            if k <= 10:
                H = sylvester_hadamard(1 << k)
                dense = bench.measure("forward Walsh transform (dense Hadamard product)", "numpy",
                                      lambda: H @ vec.astype(np.int64), repeats=3, **p)
                if dense.tolist() != f["spectrum"]:
                    raise SystemExit(f"the dense Hadamard product disagrees at k={k}")
            bench.note(f"rows={rows}, k={k}: cost {f['cost']['states']} states, {f['cost']['passes']} passes, "
                       f"{f['cost']['butterflies']} butterflies, {f['cost']['bytes']} bytes (independent of the rows)")


def case_frequencies(bench, rng):
    prng = random.Random(bench.seed)
    k = 12
    db = table(rng, 50_000)
    h = ml.Histogram.build(mb.Masks.from_scope(mbk.Scope.from_table(db["t"], ["u0", "u1"]), conditions(k)))
    vec = h.vector("complete")
    full = ms.walsh(vec)["spectrum"]
    for m in (1, 8, 32):
        freqs = sorted(prng.sample(range(1 << k), m))
        sets = [[i for i in range(k) if (S >> i) & 1] for S in freqs]
        p = dict(conditions=k, frequencies=m)
        direct = bench.measure("new frequencies: direct sums over the cells", "cpu",
                               lambda: [r["coefficient"] for r in ms.coefficients(vec, sets)], repeats=3, **p)
        for backend in bench.backends():
            bench.measure("new frequencies: one full transform + lookups", backend,
                          lambda: [ms.walsh(vec, backend=backend)["spectrum"][S] for S in freqs], repeats=3, **p)
        if direct != [full[S] for S in freqs]:
            raise SystemExit(f"direct coefficients differ at m={m}")
        basis = dict(k=k, kept_sets=freqs, kept_coefficients=direct)
        S_arr = np.array(freqs, dtype=np.uint64)
        C_arr = np.array(direct, dtype=np.int64)
        for q in (1, 100, 10_000):
            points = [prng.randrange(1 << k) for _ in range(q)]
            got = bench.measure("fixed-basis evaluations (exact dyadic)", "cpu", lambda: ms.evaluate(basis, points),
                                repeats=3, requests=q, **p)
            P_arr = np.array(points, dtype=np.uint64)

            def numpy_eval():
                signs = 1 - 2 * parity(P_arr[:, None] & S_arr[None, :])
                return (signs * C_arr[None, :]).sum(axis=1)       # integer sums (exact below 2^63)
            ref = bench.measure("fixed-basis evaluations (integer sums)", "numpy", numpy_eval, repeats=3, requests=q, **p)
            if [x * (1 << k) for x in got] != ref.tolist():
                raise SystemExit(f"NumPy evaluations differ at m={m}, q={q}")
        t = bench.measure("truncation to the largest coefficients (explicit approximation)", "cpu",
                          lambda: ms.truncate(full, m), repeats=3, **p)
        bench.note(f"k={k}, m={m}: truncation approximation={t['approximation']}, residual energy "
                   f"{t['residual_energy']} of {t['total_energy']}, l2 error^2 = {t['l2_error_squared']}")


def thresholds(bench):
    budget = Budget(tensor=1 << 16)
    for backend in bench.backends():
        bench.measure("forward transform at a 2^16 tensor budget", backend,
                      lambda: ms.walsh(np.ones(1 << 16, dtype=np.int64), backend=backend, budget=budget)["k"],
                      repeats=1, conditions=16)
        bench.refusal("forward transform past a 2^16 tensor budget", backend,
                      lambda: ms.walsh(np.ones(1 << 17, dtype=np.int64), backend=backend, budget=budget),
                      conditions=17)
    bench.threshold("transform conditions (tensor 2^16)", admitted=dict(conditions=16), refused=dict(conditions=17))


def main():
    bench = Bench.from_args("SPEC-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    case_domain(bench, rng)
    case_frequencies(bench, rng)
    thresholds(bench)
    bench.note("Native CPU paths are single-threaded; values are exact 128-bit integers (the inverse exact dyadic "
               "rationals), the NumPy baseline int64. The transform cost is the full domain: 2^k states, k passes.")
    bench.write()


if __name__ == "__main__":
    main()
