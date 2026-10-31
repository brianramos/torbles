# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SPEC-001 benchmark: fixed spectral basis states.

    python bench/benchmark_math_fourier.py --capability SPEC-001 --out FILE [--device] [--quick]

Case 1 (fixed frequency sets of 1 / 8 / 32 elements, 1 / 100 / 10,000 phase or evaluation requests; new-frequency
construction distinguished from fixed-basis actions): a basis of m two-dimensional frequencies is constructed from a
scope of 10,000 rows (2,000 with --quick): "build" is the new-frequency construction (a scan of every row: certified
per-row enclosures summed exactly; CPU, and CUDA with --device), against the NumPy float64 computation of the same
features (uncertified baseline); a request for a subset of a retained basis is a hit, a new frequency a new
construction (math_fourier.Retained). Then 1 / 100 / 10,000 fixed-basis actions (translation and phase: certified
rotations of the retained coefficients) and characteristic-function evaluations (F / W), against direct NumPy
evaluation that rescans every row for every request (run for 1 and 100 requests; for 10,000 only with one frequency,
the other points are not run and not extrapolated).
Case 2 (Boolean transforms, k = 3 / 6 / 10 / 12 bits, all 2^k states): the basis of all 2^k Walsh frequencies over
2,000 rows (500 with --quick, k <= 10): its construction (the forward transform from the rows: rows x 2^k exact
characters) and its exact inverse (SPEC-002's inverse transform from the 2^k exact coefficients, with the declared
normalization and the full-domain cost), against SPEC-002 from the LATTICE-001 histogram (histogram, forward and
inverse butterflies) and a NumPy bincount plus fast Walsh-Hadamard transform.
"""
import time
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_fourier as mf


def scope_of(rng, rows):
    cols = {"x": rng.integers(-10_000, 10_001, rows), "y": rng.integers(-500, 501, rows), "w": rng.integers(-50, 51, rows)}
    return mbk.Scope.from_arrays(cols, scales={"x": 2}), cols


def frequencies(m, offset=0):
    return [(Fraction(i + 1 + offset, 7), Fraction(i - 3, 5)) for i in range(m)]


def numpy_features(cols, freqs):
    x, y, w = cols["x"] / 100.0, cols["y"].astype(float), cols["w"].astype(float)
    return [complex(np.sum(w * np.exp(1j * (float(a) * x + float(b) * y)))) for a, b in freqs]


def case_actions(bench):
    rng = np.random.default_rng(bench.seed)
    rows = 2_000 if bench.quick else 10_000
    scope, cols = scope_of(rng, rows)
    x, y, w = cols["x"] / 100.0, cols["y"].astype(float), cols["w"].astype(float)
    for m in (1, 8, 32):
        freqs = frequencies(m)
        p = dict(rows=rows, frequencies=m, dimensions=2)
        for backend in bench.backends():
            B = bench.measure("new-frequency construction (basis build: scan, certified sums)", backend,
                              lambda: mf.basis(scope, ["x", "y"], freqs, weight="w", backend=backend), repeats=3,
                              phase="build", **p)
        B = mf.basis(scope, ["x", "y"], freqs, weight="w")
        widths = [max(c[0][1] - c[0][0], c[1][1] - c[1][0]) for c in B.coefficients()]
        bench.records[-1]["result"] = dict(max_width=max(widths), scale=B.info["scale"])
        bench.measure("features (numpy float64, uncertified)", "numpy", lambda: len(numpy_features(cols, freqs)),
                      repeats=3, phase="build", **p)
        cache = mf.Retained()
        cache.basis(scope, ["x", "y"], freqs, weight="w")
        bench.measure("retained basis: a request for a subset (hit, no scan)", "cpu",
                      lambda: cache.basis(scope, ["x", "y"], freqs[: max(1, m // 2)], weight="w"), repeats=5, **p)
        bench.measure("retained basis: a new frequency (a new construction)", "cpu",
                      lambda: mf.Retained().basis(scope, ["x", "y"], frequencies(1, 100), weight="w"), repeats=3,
                      phase="build", **p)
        for n in (1, 100, 10_000):
            taus = [(float(t), float(-t / 3)) for t in np.linspace(-5, 5, n)]
            phis = np.linspace(-1, 1, n)
            reqs = [dict(shift=t, phase=float(ph)) for t, ph in zip(taus, phis)]
            q = dict(requests=n, **p)
            out = bench.measure("fixed-basis actions (translation and phase)", "cpu",
                                lambda: B.act(reqs, arrays=True), repeats=3 if n < 10_000 else 2, **q)
            bench.records[-1]["result"] = dict(max_width=float(np.max(np.maximum(out[..., 1] - out[..., 0],
                                                                                  out[..., 3] - out[..., 2]))))
            bench.measure("fixed-basis characteristic function F / W (no shift)", "cpu",
                          lambda: B.act([dict()] * n, normalize=True, arrays=True), repeats=3, **q)
            if n <= 100 or m == 1:
                def direct():
                    res = []
                    for (t1, t2), ph in zip(taus, phis):
                        for a, b in freqs:
                            res.append(complex(np.sum(w * np.exp(1j * (float(a) * (x + t1) + float(b) * (y + t2)
                                                                       + ph)))))
                    return len(res)
                bench.measure("direct evaluation (numpy, rescans the rows per request)", "numpy", direct,
                              repeats=2 if n < 10_000 else 1, **q)
            else:
                bench.note(f"direct numpy evaluation not run for {n} requests x {m} frequencies "
                           f"({n * m * rows:.2e} complex exponentials); not extrapolated")


def case_walsh(bench):
    from torbles import math_bool as mb, math_lattice as ml, math_spec as ms
    rng = np.random.default_rng(bench.seed + 1)
    rows = 500 if bench.quick else 2_000
    for k in ((3, 6, 10) if bench.quick else (3, 6, 10, 12)):
        bits = rng.integers(0, 2, (rows, k))
        cols = {f"b{j}": bits[:, j] for j in range(k)}
        cols["w"] = rng.integers(0, 9, rows)
        scope = mbk.Scope.from_arrays(cols)
        names = [f"b{j}" for j in range(k)]
        p = dict(k=k, states=1 << k, rows=rows)
        for backend in bench.backends():
            W = bench.measure("Walsh basis: forward from the rows (rows x 2^k exact characters)", backend,
                              lambda: mf.walsh_basis(scope, names, weight="w", backend=backend), repeats=2,
                              phase="build", **p)
        W = mf.walsh_basis(scope, names, weight="w")
        inv = bench.measure("Walsh basis: exact inverse (SPEC-002 inverse of the 2^k coefficients)", "cpu",
                            lambda: W.walsh_inverse(), repeats=3, **p)
        bench.records[-1]["result"] = dict(states=inv["cost"]["states"], passes=inv["cost"]["passes"],
                                           butterflies=inv["cost"]["butterflies"], exact=W.info["exact"],
                                           normalization=inv["normalization"])
        masks = mb.Masks.from_scope(scope, [mb.condition(c, "bit", 0) for c in names])
        h = bench.measure("SPEC-002 baseline: LATTICE-001 histogram of the bits (unweighted)", "spec002",
                          lambda: ml.Histogram.build(masks), repeats=2, phase="build", **p)
        f = bench.measure("SPEC-002 baseline: forward transform (k 2^(k-1) butterflies)", "spec002",
                          lambda: ms.walsh(h), repeats=3, **p)
        bench.records[-1]["result"] = dict(states=f["cost"]["states"], normalization=f["normalization"])
        bench.measure("SPEC-002 baseline: inverse transform", "spec002", lambda: ms.inverse(f["spectrum"]),
                      repeats=3, **p)
        idx = (bits * (1 << np.arange(k))).sum(1)

        def numpy_fwht():
            v = np.bincount(idx, weights=cols["w"], minlength=1 << k).astype(np.int64)
            hlen = 1
            while hlen < len(v):
                v = v.reshape(-1, 2 * hlen)
                a, b = v[:, :hlen].copy(), v[:, hlen:].copy()
                v[:, :hlen], v[:, hlen:] = a + b, a - b
                v = v.reshape(-1)
                hlen *= 2
            return v
        spec = bench.measure("numpy baseline: bincount + fast Walsh-Hadamard (int64, unchecked)", "numpy",
                             numpy_fwht, repeats=3, **p)
        bench.records[-1]["result"] = dict(equal=[int(v) for v in spec] == [int(b[0][0]) for b in W.bounds()])


def main():
    bench = Bench.from_args("SPEC-001", __doc__)
    t0 = time.perf_counter()
    case_actions(bench)
    case_walsh(bench)
    bench.note("CUDA is timed only with --device; actions, derivatives and the Walsh inverse run on the CPU. Total "
               "wall time %.1f s." % (time.perf_counter() - t0))
    bench.write()


if __name__ == "__main__":
    main()
