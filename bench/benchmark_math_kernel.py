# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""KERNEL-001 benchmark: exact finite-feature kernel sums, means and MMDs from retained feature states, against direct
kernel matrices.

    python bench/benchmark_math_kernel.py --capability KERNEL-001 --out FILE [--device] [--quick]

  exact features   polynomial kernels (x.y + 1)^d at dimensions 2 / 4 / 8 and degrees 1 / 2 / 4 over two groups of
                   600 rows each (quick 200): the feature-state construction ("build": the moment tensor over (1, x)
                   and the self-pair terms, dense and sparse, CPU and with --device CUDA) apart from its reuse ("run":
                   one request, then batches of 1 / 100 / 10,000 mixed requests: sums with and without the diagonal,
                   means, biased and unbiased MMD, witness values), against the direct kernel matrices (NumPy int64
                   when the products provably fit, else object integers: exact), which are checked equal to the native
                   answers; the feature count ((n + 1)^d cells) and the state bytes are recorded.
  approximate      feature counts 16 / 64 / 256 with fixed seeds: approximate kernels (random features of RBF and other
                   kernels) are KERNEL-002's scope and are not implemented here (an RBF kernel is refused natively);
                   this case measures only the exact baseline: exact polynomial features with exactly 16 / 64 / 256
                   cells (n = 3, d = 2; n = 7, d = 2; n = 3, d = 4) against the direct kernel matrix (observed error 0),
                   and the float64 RBF kernel matrix of the same rows, the small numerical oracle KERNEL-002 will be
                   compared against (its time and memory).
"""
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_kernel as mk
from torbles.math_errors import UnsupportedError


def data(rng, rows, n):
    x = rng.integers(-9, 10, (2 * rows, n)).astype(np.int64)
    g = np.r_[np.zeros(rows, np.int64), np.ones(rows, np.int64)]
    names = [f"x{j}" for j in range(n)]
    scope = mbk.Scope.from_arrays({names[j]: x[:, j] for j in range(n)}, groups=g, ngroups=2)
    return scope, names, x[:rows], x[rows:]


def direct(A, B, d, c):
    """the exact kernel matrix (x.y + c)^d: int64 when it provably fits, else Python ints"""
    bound = (A.shape[1] * 81 + abs(c)) ** d
    if bound * A.shape[0] * B.shape[0] < 2 ** 62:
        return (A @ B.T + c) ** d
    return (A.astype(object) @ B.T.astype(object) + c) ** d


def exact_features(bench, rng):
    rows = 200 if bench.quick else 600
    for n in (2, 4, 8):
        for d in (1, 2, 4):
            scope, names, A, B = data(rng, rows, n)
            p = dict(dims=n, degree=d, rows_per_group=rows, features=(n + 1) ** d)
            kern = mk.polynomial(d, 1)
            for repr_ in ("dense", "sparse"):
                for backend in bench.backends():
                    st = bench.measure("feature state", backend,
                                       lambda: mk.features(scope, names, kern, repr=repr_, backend=backend),
                                       repeats=3, phase="build", repr=repr_, **p)
                    bench.records[-1]["result"] = dict(bytes=st.info["bytes"], cells=st.info["cells"],
                                                       nnz=st.info["nnz"])
            # reuse: a fresh state; the first sample of "one request" includes making the evaluation handle (CPU) or
            # uploading the feature state (CUDA), the warm samples reuse it
            st = mk.features(scope, names, kern)
            mixed = [dict(op="sum", a=0, b=1), dict(op="sum", a=0), dict(op="sum", a=1, self_pairs=True, ordered=False),
                     dict(op="mean", a=0, b=1), dict(op="mmd2", a=0, b=1, unbiased=True),
                     dict(op="mmd2", a=0, b=1, unbiased=False), dict(op="witness", a=0, point=0),
                     dict(op="witness", a=1, point=1)]
            points = [tuple(int(v) for v in A[0]), tuple(int(v) for v in B[0])]
            for backend in bench.backends():
                bench.measure("one request", backend, lambda: st.mmd2(0, 1, backend=backend), repeats=5, phase="run",
                              op="mmd2", **p)
                for batch in ((1, 100) if bench.quick else (1, 100, 10000)):
                    reqs = [mixed[i % len(mixed)] for i in range(batch)]
                    bench.measure("request batch", backend, lambda: st.query(reqs, points=points, backend=backend),
                                  repeats=3, phase="run", batch=batch, **p)
            # the direct kernel matrices (baseline): checked equal to the native answers, then timed
            kab, kaa, kbb = direct(A, B, d, 1), direct(A, A, d, 1), direct(B, B, d, 1)
            sab, saa, sbb = int(kab.sum()), int(kaa.sum()), int(kbb.sum())
            daa, dbb = int(np.trace(kaa)), int(np.trace(kbb))
            na = nb = rows
            want = Fraction(saa - daa, na * na - na) + Fraction(sbb - dbb, nb * nb - nb) - 2 * Fraction(sab, na * nb)
            assert st.mmd2(0, 1, unbiased=True) == want, (n, d)
            assert st.sum(0, 1)["value"] == sab and st.sum(0)["value"] == saa - daa, (n, d)
            assert st.sum(1, self_pairs=True)["value"] == sbb, (n, d)
            assert st.witness(0, points[1:]) == [Fraction(int(direct(A, B[:1], d, 1).sum()), na)], (n, d)
            m = dict(matrix_bytes=int(kab.nbytes), dtype=str(kab.dtype), **p)
            bench.measure("direct kernel matrix", "numpy", lambda: direct(A, B, d, 1), repeats=3, phase="run",
                          pair="ab", **m)
            bench.measure("direct mmd2", "numpy", lambda: (direct(A, A, d, 1).sum(), direct(B, B, d, 1).sum(),
                                                           direct(A, B, d, 1).sum()), repeats=3, phase="run", **m)


def approximate(bench, rng):
    rows = 100 if bench.quick else 300
    bench.note("approximate kernels (random features of RBF / Laplace and other kernels with 16 / 64 / 256 features and "
               "fixed seeds) are KERNEL-002's scope and are not implemented by KERNEL-001; mk.kernel('rbf') is refused "
               "natively (UnsupportedError). This case measures only the exact baseline.")
    for count, (n, d) in ((16, (3, 2)), (64, (7, 2)), (256, (3, 4))):
        seed = bench.seed + count
        r = np.random.default_rng(seed)
        scope, names, A, B = data(r, rows, n)
        p = dict(features=count, dims=n, degree=d, seed=seed, rows_per_group=rows)
        st = bench.measure("exact features", "cpu", lambda: mk.features(scope, names, mk.polynomial(d, 1)), repeats=3,
                           phase="build", **p)
        bench.records[-1]["result"] = dict(bytes=st.info["bytes"], cells=st.info["cells"])
        got = bench.measure("exact kernel sum", "cpu", lambda: st.sum(0, 1)["value"], repeats=3, phase="run", **p)
        want = int(direct(A, B, d, 1).sum())
        bench.records[-1]["result"] = dict(observed_error=abs(got - want), exact=got == want)
        bench.measure("direct kernel matrix", "numpy", lambda: direct(A, B, d, 1).sum(), repeats=3, phase="run", **p)
        Af, Bf = A.astype(np.float64), B.astype(np.float64)

        def rbf():
            sq = (Af * Af).sum(1)[:, None] + (Bf * Bf).sum(1)[None, :] - 2 * Af @ Bf.T
            return np.exp(-sq / (2.0 * n)).sum()
        bench.measure("rbf oracle (float64)", "numpy", rbf, repeats=3, phase="run", memory_bytes=8 * rows * rows, **p)
        try:
            mk.features(scope, names, mk.kernel("rbf", gamma=1.0))
            bench.note("unexpected: an RBF kernel was accepted")
        except UnsupportedError as e:
            bench.note(f"features={count}: rbf refused natively ({e})")


def main():
    bench = Bench.from_args("KERNEL-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    exact_features(bench, rng)
    approximate(bench, rng)
    bench.note("answers are exact rationals; the direct baselines are checked equal before they are timed")
    bench.write()


if __name__ == "__main__":
    main()
