# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""KERNEL-002 benchmark: approximate spectral feature kernels against direct small kernel matrices.

    python bench/benchmark_math_kapprox.py --capability KERNEL-002 --out FILE [--device] [--quick]

  exact polynomial  polynomial kernels (x.y + 1)^d at dimensions 2 / 4 / 8 and degrees 1 / 2 / 4 through ka.features,
                    which answers them exactly with KERNEL-001's feature states: the feature-state construction
                    ("build") apart from its reuse ("run": one MMD, a batch of 100 mixed requests), against the direct
                    kernel matrices (NumPy int64 when the products provably fit, else Python integers), checked equal
                    before they are timed; the state bytes are recorded.
  taylor gaussian   the deterministic certified features of exp(-gamma |x - y|^2) (gamma = 1 / (2 d), values in
                    [-0.9, 0.9]) at dimensions 2 / 4 / 8 and degrees 2 / 4 / 8: build (CPU; CUDA with --device) apart
                    from reuse (one MMD, a request batch), the feature count, the state bytes, the certified per-pair
                    bound eps and the per-answer bounds, and the observed error of the kernel sum, the mean and the
                    biased MMD against a small exact numerical oracle (Python decimal, 40 digits) on the same rows;
                    the float64 NumPy kernel matrix as the baseline (time and memory).
  random features   gaussian / laplace / cauchy kernels with D = 16 / 64 / 256 features and fixed seeds (1, 2, 3; quick:
                    1): build (the native seeded draws and SPEC-001's certified sums) apart from reuse, the state
                    bytes, the probabilistic per-answer bound (failure 1e-6) and the observed error of the same answers
                    against the decimal oracle; the float64 NumPy kernel matrix baseline.
  thresholds        the representation and budget limits measured, never extrapolated: the largest admitted and the
                    smallest refused Taylor feature count under the default budget at d = 16.
"""
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_kapprox as ka, math_kernel as mk


def scope_of(rng, rows, n, lo=-9, hi=10, scale=1):
    x = rng.integers(lo, hi, (2 * rows, n)).astype(np.int64)
    w = rng.integers(1, 4, 2 * rows).astype(np.int64)
    g = np.r_[np.zeros(rows, np.int64), np.ones(rows, np.int64)]
    names = [f"x{j}" for j in range(n)]
    cols = {names[j]: x[:, j] for j in range(n)}
    cols["w"] = w
    scope = mbk.Scope.from_arrays(cols, scales={c: scale for c in names}, groups=g, ngroups=2)
    return scope, names, x, w


def direct_poly(A, B, d, c):
    bound = (A.shape[1] * 81 + abs(c)) ** d
    if bound * A.shape[0] * B.shape[0] < 2 ** 62:
        return (A @ B.T + c) ** d
    return (A.astype(object) @ B.T.astype(object) + c) ** d


def exact_polynomial(bench, rng):
    rows = 60 if bench.quick else 200
    for n in (2, 4, 8):
        for d in (1, 2, 4):
            scope, names, x, w = scope_of(rng, rows, n)
            A, B = x[:rows], x[rows:]
            p = dict(dims=n, degree=d, rows_per_group=rows, features=(n + 1) ** d)
            kern = mk.polynomial(d, 1)
            for backend in bench.backends():
                st = bench.measure("exact feature state (KERNEL-001)", backend,
                                   lambda: ka.features(scope, names, kern, backend=backend), repeats=3, phase="build", **p)
                bench.records[-1]["result"] = dict(bytes=st.info["bytes"], exact=True)
            st = ka.features(scope, names, kern)
            mixed = [dict(op="sum", a=0, b=1), dict(op="mean", a=0, b=1), dict(op="mmd2", a=0, b=1, unbiased=True),
                     dict(op="mmd2", a=0, b=1, unbiased=False), dict(op="witness", a=0, point=0)]
            points = [tuple(int(v) for v in B[0])]
            for backend in bench.backends():
                bench.measure("one request", backend, lambda: st.mmd2(0, 1, backend=backend), repeats=5, phase="run",
                              op="mmd2", **p)
                reqs = [mixed[i % len(mixed)] for i in range(100)]
                bench.measure("request batch", backend, lambda: st.query(reqs, points=points, backend=backend),
                              repeats=3, phase="run", batch=100, **p)
            kab = direct_poly(A, B, d, 1)
            assert st.sum(0, 1)["value"] == int(kab.sum()), (n, d)          # exact: checked before timing
            bench.records[-1]["result"] = dict(observed_error=0, exact=True)
            bench.measure("direct kernel matrix", "numpy", lambda: direct_poly(A, B, d, 1).sum(), repeats=3,
                          phase="run", matrix_bytes=int(kab.nbytes), **p)


# ------------------------------------------------------------------ the decimal oracle (40 digits)
def _d(q):
    q = Fraction(q)
    return Decimal(q.numerator) / Decimal(q.denominator)


def kernel_value(kind, gamma, dx):
    if kind == "gaussian":
        return (-_d(gamma * sum(t * t for t in dx))).exp()
    if kind == "laplace":
        return (-_d(gamma * sum(abs(t) for t in dx))).exp()
    return _d(math.prod(Fraction(1) / (1 + gamma * gamma * t * t) for t in dx))


def oracle(kind, gamma, x, w, rows, scale=1):
    """exact K_ab, K_aa, K_bb (decimal, 40 digits) of the two groups' rows, value units"""
    with localcontext() as ctx:
        ctx.prec = 40
        u = Fraction(10) ** scale
        pts = [tuple(Fraction(int(v)) / u for v in r) for r in x]
        K = {}
        for key, (ia, ib) in dict(ab=(range(rows), range(rows, 2 * rows)), aa=(range(rows), range(rows)),
                                  bb=(range(rows, 2 * rows), range(rows, 2 * rows))).items():
            s = Decimal(0)
            for i in ia:
                for j in ib:
                    if key != "ab" and j < i:
                        continue
                    kv = kernel_value(kind, gamma, [a - b for a, b in zip(pts[i], pts[j])])
                    s += Decimal(int(w[i]) * int(w[j])) * kv * (2 if key != "ab" and j > i else 1)
            K[key] = s
        Wa, Wb = int(w[:rows].sum()), int(w[rows:].sum())
        return dict(sum_ab=K["ab"], mean_ab=K["ab"] / (Wa * Wb),
                    mmd2_biased=K["aa"] / (Wa * Wa) + K["bb"] / (Wb * Wb) - 2 * K["ab"] / (Wa * Wb))


ANSWERS = [("sum_ab", dict(op="sum", a=0, b=1)), ("mean_ab", dict(op="mean", a=0, b=1)),
           ("mmd2_biased", dict(op="mmd2", a=0, b=1, unbiased=False))]


def observed(st, exact):
    """per answer: the observed error |midpoint - exact| and the stated bound (error)"""
    out = {}
    for (tag, q), a in zip(ANSWERS, st.query([q for _, q in ANSWERS])):
        mid = (a.value[0] + a.value[1]) / 2
        out[tag] = dict(observed=float(abs(_d(mid) - exact[tag])), stated=float(a.error),
                        inside=bool(_d(a.bound[0]) <= exact[tag] <= _d(a.bound[1])))
    return out


def numpy_rbf(Af, Bf, gamma):
    sq = (Af * Af).sum(1)[:, None] + (Bf * Bf).sum(1)[None, :] - 2 * Af @ Bf.T
    return np.exp(-gamma * sq).sum()


def taylor_gaussian(bench, rng):
    rows_t = 60 if bench.quick else 200
    rows_e = 12 if bench.quick else 25
    for n in (2, 4, 8):
        gamma = Fraction(1, 2 * n)
        kern = ka.gaussian(gamma)
        scope_t, names, xt, wt = scope_of(rng, rows_t, n)
        scope_e, _, xe, we = scope_of(rng, rows_e, n)
        exact = oracle("gaussian", gamma, xe, we, rows_e)
        for N in (2, 4, 8):
            p = dict(dims=n, degree=N, gamma=str(gamma), rows_per_group=rows_t, items=math.comb(n + N, N))
            for backend in bench.backends():
                st = bench.measure("taylor feature state", backend,
                                   lambda: ka.features(scope_t, names, kern, method="taylor", degree=N, weight="w",
                                                       backend=backend), repeats=3, phase="build", **p)
                bench.records[-1]["result"] = dict(bytes=st.info["bytes"], items=st.info["items"],
                                                   per_pair=float(st.q("per_pair")))
            st = ka.features(scope_t, names, kern, method="taylor", degree=N, weight="w")
            bench.measure("one request", "cpu", lambda: st.mmd2(0, 1), repeats=5, phase="run", op="mmd2", **p)
            reqs = [q for _, q in ANSWERS] * 10
            bench.measure("request batch", "cpu", lambda: st.query(reqs), repeats=3, phase="run", batch=len(reqs), **p)
            se = ka.features(scope_e, names, kern, method="taylor", degree=N, weight="w")
            res = observed(se, exact)
            bench.records.append(dict(case="taylor observed error", backend="cpu", phase="accuracy",
                                      params=dict(p, rows_per_group=rows_e), result=dict(
                                          per_pair=float(se.q("per_pair")), U=float(se.q("U")),
                                          **{f"{k}_{m}": v[m] for k, v in res.items() for m in v})))
            assert all(v["inside"] for v in res.values()), (n, N, res)
        Af, Bf = xt[:rows_t] / 10.0, xt[rows_t:] / 10.0
        bench.measure("rbf kernel matrix (float64)", "numpy", lambda: numpy_rbf(Af, Bf, float(gamma)), repeats=3,
                      phase="run", dims=n, rows_per_group=rows_t, memory_bytes=8 * rows_t * rows_t)


def random_features(bench, rng):
    rows_t = 60 if bench.quick else 200
    rows_e = 12 if bench.quick else 25
    n = 3
    scope_t, names, xt, wt = scope_of(rng, rows_t, n)
    scope_e, _, xe, we = scope_of(rng, rows_e, n)
    seeds = (1,) if bench.quick else (1, 2, 3)
    for kind, gamma in (("gaussian", Fraction(1, 2)), ("laplace", Fraction(1, 2)), ("cauchy", Fraction(1, 1))):
        kern = getattr(ka, kind)(gamma)
        exact = oracle(kind, gamma, xe, we, rows_e)
        for count in (16, 64, 256):
            for seed in seeds:
                p = dict(kernel=kind, gamma=str(gamma), features=count, seed=seed, dims=n, rows_per_group=rows_t)
                for backend in bench.backends():
                    st = bench.measure("random feature state", backend,
                                       lambda: ka.features(scope_t, names, kern, method="rff", features=count,
                                                           seed=seed, weight="w", failure=Fraction(1, 10 ** 6),
                                                           backend=backend), repeats=3, phase="build", **p)
                    bench.records[-1]["result"] = dict(bytes=st.info["bytes"], distinct=st.info["frequencies"],
                                                       per_pair=float(st.q("per_pair")))
                st = ka.features(scope_t, names, kern, method="rff", features=count, seed=seed, weight="w",
                                 failure=Fraction(1, 10 ** 6))
                bench.measure("one request", "cpu", lambda: st.mmd2(0, 1), repeats=5, phase="run", op="mmd2", **p)
                se = ka.features(scope_e, names, kern, method="rff", features=count, seed=seed, weight="w",
                                 failure=Fraction(1, 10 ** 6))
                res = observed(se, exact)
                bench.records.append(dict(case="random feature observed error", backend="cpu", phase="accuracy",
                                          params=dict(p, rows_per_group=rows_e), result=dict(
                                              per_pair=float(se.q("per_pair")), hoeffding=float(se.q("hoeffding")),
                                              bias=float(se.q("bias")), failure=1e-6, bytes=se.info["bytes"],
                                              **{f"{k}_{m}": v[m] for k, v in res.items() for m in v})))
    Af, Bf = xt[:rows_t] / 10.0, xt[rows_t:] / 10.0
    bench.measure("rbf kernel matrix (float64)", "numpy", lambda: numpy_rbf(Af, Bf, 0.5), repeats=3, phase="run",
                  dims=n, rows_per_group=rows_t, memory_bytes=8 * rows_t * rows_t)


def thresholds(bench, rng):
    scope, names, x, w = scope_of(rng, 4, 16)
    kern = ka.gaussian(Fraction(1, 32))
    admitted = refused = None
    for N in range(4, 13):
        try:
            ka.features(scope, names, kern, method="taylor", degree=N)
            admitted = dict(dims=16, degree=N, items=math.comb(16 + N, N))
        except Exception:
            rec = bench.refusal("taylor items, default budget", "cpu",
                                lambda: ka.features(scope, names, kern, method="taylor", degree=N), dims=16, degree=N,
                                items=math.comb(16 + N, N))
            refused = dict(dims=16, degree=N, items=math.comb(16 + N, N), resource=rec.get("resource"))
            break
    bench.threshold("taylor feature count at d = 16 (default budget: support 2^20)", admitted=admitted, refused=refused)


def main():
    bench = Bench.from_args("KERNEL-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    exact_polynomial(bench, rng)
    taylor_gaussian(bench, rng)
    random_features(bench, rng)
    thresholds(bench, rng)
    bench.note("exact polynomial kernels are KERNEL-001's exact states (observed error 0, checked); approximate "
               "answers are exact rational enclosures of the declared approximant with their error terms: 'stated' "
               "is the certified (taylor) or probabilistic (random features, failure 1e-6 per answer) bound, "
               "'observed' the distance of the enclosure's midpoint to the decimal oracle's exact value")
    bench.write()


if __name__ == "__main__":
    main()
