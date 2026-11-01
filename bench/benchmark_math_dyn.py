# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DYN-001 / DYN-002 benchmark: declared dynamics as exact transfer operators on retained moment states, their powers,
against repeated explicit row evolution.

    python bench/benchmark_math_dyn.py --capability DYN-001 --out FILE [--device] [--quick]

  dynamics     affine dynamics (an exact rotation (3/5, 4/5) of the first plane, x_i' = x_i / 2 + 1 elsewhere) and
               closed-observable nonlinear dynamics (u' = u / 2, v' = v / 3 + u^2 per coordinate pair, on the closed
               basis {1, u, u^2, v} of every pair) on 2 / 4 / 8 coordinates (quick: 2 / 4), degree-2 bases of
               6 / 15 / 45 moments (closed: 4 / 7 / 13), 4 groups, 4,000 rows (quick 500): the retained state (an
               FND-004 bank) and the operator K ("build", apart), then for 0 / 1 / 10 / 1,000 steps (quick: 0 / 1 /
               10 / 100) the powers K^t ("power": squaring, modular) and the evolved moments of every group
               ("evolve": stepping, squaring, modular; CUDA with --device), each checked equal first.
  row baseline the explicit row evolution: the native exact row evolution (every row evolved in big integers, then
               its moments) on every row up to 10 steps, on a 64-row subset at 1,000 steps (compared there with the
               operator evolution of the same subset's state), and NumPy float64 rows (inexact: the largest relative
               deviation from the exact moments recorded).
  nonclosed    a nonlinear map on the full degree-2 basis: the closure refusal (NotClosedError with its witness), and
               the truncation under a declared F-invariant box (certified approximate: the error bound relative to the
               moments recorded) for 1 / 10 steps; the refusal of a non-invariant box past one step.
  growth       the proven coefficient bound of K^t (bits) against the largest actual numerator, and the bytes needed.
  stability    floating interval powers K^1000 of a contracting and an expanding operator at tolerance 1e-9: the
               certified error, or the instability diagnosis (refused), against NumPy's matrix_power.
  thresholds   the largest measured admitted and the smallest refused exponent within a 1 MiB budget, and step-count
               and output refusals at the limit + 1.
"""
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_dyn as dyn, math_sym as ms
from torbles.math_budgets import Budget

F = Fraction


def frame(rng, n, rows, G=4, lo=-50, hi=51):
    names = [f"x{i}" for i in range(n)]
    X = rng.integers(lo, hi, (rows, n)).astype(np.int64)
    g = rng.integers(0, G, rows)
    scope = mbk.Scope.from_arrays({names[i]: X[:, i] for i in range(n)}, groups=g, ngroups=G)
    geo = ms.geometry(scope, numeric={k: "m" for k in names})
    return names, X, g, scope, geo


def affine_dynamics(geo, names):
    parts = [ms.rotate(names[0], names[1], cos=F(3, 5), sin=F(4, 5), center=(1, 0))]
    if len(names) > 2:
        k = len(names) - 2
        parts.append(ms.affine(names[2:], [[F(1, 2) if i == j else 0 for j in range(k)] for i in range(k)], [1] * k))
    return dyn.affine(geo, ms.compose(*parts)), parts


def closed_dynamics(geo, names, coupling=1):
    comps, terms = {}, ["1"]
    for i in range(0, len(names) - 1, 2):
        u, v = names[i], names[i + 1]
        comps[u] = {u: F(1, 2)}
        comps[v] = {v: F(1, 3), f"{u}^2": F(coupling)}
        terms += [u, f"{u}^2", v]
    return dyn.polynomial(geo, comps), terms


def float_rows(X, n, kind, t, coupling=1):
    """the rows evolved in float64 (NumPy), value units"""
    Y = X.astype(np.float64).copy()
    for _ in range(t):
        if kind == "affine":
            x, y = Y[:, 0] - 1, Y[:, 1].copy()
            Y[:, 0], Y[:, 1] = 1 + 0.6 * x - 0.8 * y, 0.8 * x + 0.6 * y
            if n > 2:
                Y[:, 2:] = Y[:, 2:] / 2 + 1
        else:
            for i in range(0, n - 1, 2):
                u = Y[:, i].copy()
                Y[:, i], Y[:, i + 1] = u / 2, Y[:, i + 1] / 3 + coupling * u * u
    return Y


def float_moments(Y, g, terms, G):
    out = np.zeros((G, len(terms)))
    for j, t in enumerate(terms):
        v = np.ones(len(Y))
        for i in np.nonzero(t)[0]:
            v = v * Y[:, i] ** int(t[i])
        for gg in range(G):
            out[gg, j] = v[g == gg].sum()
    return out


def dynamics(bench, rng):
    dims = (2, 4) if bench.quick else (2, 4, 8)
    rows = 500 if bench.quick else 4000
    steps = (0, 1, 10, 100) if bench.quick else (0, 1, 10, 1000)
    G = 4
    for n in dims:
        names, X, g, scope, geo = frame(rng, n, rows, G)
        for kind in ("affine", "closed"):
            if kind == "affine":
                D, _ = affine_dynamics(geo, names)
                B = ms.basis(geo, degree=2)
            else:
                D, terms = closed_dynamics(geo, names)
                B = ms.basis(geo, terms=terms)
            p = dict(kind=kind, coordinates=n, moments=len(B), rows=rows, groups=G)
            st = bench.measure("state (FND-004 bank)", "cpu", lambda: ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo),
                               repeats=3, phase="build", **p)
            K = bench.measure("operator K (closure and transfer matrix)", "cpu", lambda: dyn.operator(B, D), repeats=3,
                              phase="build", **p)
            bench.records[-1]["result"] = dict(den=str(K.info["den"]), nonzeros=K.info["nonzeros"],
                                               sym_action=K.info["sym_action"], norm_bits=K.info["norm_bits"])
            for t in steps:
                q = dict(steps=t, **p)
                ref = dyn.evolve(st, K, t, method="stepping")
                want = [ref.moments(gg) for gg in range(G)]
                for method in ("squaring", "modular"):
                    pw = K.power(t, method=method)
                    assert pw.equal(K.power(t, method="squaring"))
                    bench.measure("power K^t", "cpu", lambda: K.power(t, method=method), repeats=3 if t < 1000 else 1,
                                  phase="run", method=method, **q)
                    bench.records[-1]["result"] = dict(bound_bits=pw.info["bound_bits"], max_bits=pw.info["max_bits"],
                                                       primes=pw.info["primes"], products=pw.info["products"])
                for method in ("stepping", "squaring", "modular"):
                    ev = dyn.evolve(st, K, t, method=method)
                    assert [ev.moments(gg) for gg in range(G)] == want, (n, kind, t, method)
                    bench.measure("evolve moments", "cpu", lambda: dyn.evolve(st, K, t, method=method),
                                  repeats=3 if t < 1000 else 1, phase="run", method=method, **q)
                if "cuda" in bench.backends():
                    ev = dyn.evolve(st, K, t, backend="cuda")
                    assert [ev.moments(gg) for gg in range(G)] == want
                    bench.measure("evolve moments", "cuda", lambda: dyn.evolve(st, K, t, backend="cuda"),
                                  repeats=3 if t < 1000 else 1, phase="run", method="modular", **q)
                auto = dyn.evolve(st, K, t)
                bench.records[-1]["result"] = dict(auto=auto.info["method"], cost_stepping=auto.info["cost_stepping"],
                                                   cost_squaring=auto.info["cost_squaring"],
                                                   cost_modular=auto.info["cost_modular"])
                if t == 1:
                    for backend in bench.backends():
                        assert K.apply(st, backend=backend) == want
                        bench.measure("one step K m (512-bit)", backend, lambda: K.apply(st, backend=backend),
                                      repeats=5, phase="run", **q)
                # the explicit row evolution: exact (native), and NumPy float64
                if t <= 10:
                    rv = dyn.evolve_rows(scope, D, B, t)
                    assert [rv.moments(gg) for gg in range(G)] == want
                    bench.measure("explicit row evolution (exact, native)", "reconstruction",
                                  lambda: dyn.evolve_rows(scope, D, B, t), repeats=3 if t < 10 else 1, phase="run", **q)
                else:
                    sub = 64
                    sscope = mbk.Scope.from_arrays({names[i]: X[:sub, i] for i in range(n)}, groups=g[:sub], ngroups=G)
                    sgeo = ms.geometry(sscope, numeric={k: "m" for k in names})
                    SD = affine_dynamics(sgeo, names)[0] if kind == "affine" else closed_dynamics(sgeo, names)[0]
                    SB = ms.basis(sgeo, degree=2) if kind == "affine" else ms.basis(sgeo, terms=closed_dynamics(sgeo, names)[1])
                    sst = ms.state(mbk.build(sscope, SB.terms_for(sscope), "joint"), sgeo)
                    SK = dyn.operator(SB, SD)
                    if kind == "affine":
                        rv = bench.measure("explicit row evolution (exact, native)", "reconstruction",
                                           lambda: dyn.evolve_rows(sscope, SD, SB, t), repeats=1, phase="run",
                                           subset_rows=sub, **q)
                        ev = dyn.evolve(sst, SK, t)
                        assert [rv.moments(gg) for gg in range(G)] == [ev.moments(gg) for gg in range(G)]
                    else:                           # one common denominator Q_(t+1) = L Q_t^2: doubly exponential
                        rec = bench.refusal("explicit row evolution (exact, native)", "reconstruction",
                                            lambda: dyn.evolve_rows(sscope, SD, SB, t), subset_rows=sub, **q)
                        assert rec["refused"]
                    bench.measure("evolve moments (the subset's state)", "cpu", lambda: dyn.evolve(sst, SK, t),
                                  repeats=1, phase="run", subset_rows=sub, **q)
                terms = np.array(B.terms, dtype=np.int64)
                fm = float_moments(float_rows(X, n, kind, t), g, terms, G)
                ex = np.array([[float(v) for v in want[gg]] for gg in range(G)])
                with np.errstate(all="ignore"):
                    dev = float(np.nanmax(np.abs(fm - ex) / np.maximum(np.abs(ex), 1.0)))
                bench.measure("explicit row evolution (float64)", "numpy",
                              lambda: float_moments(float_rows(X, n, kind, t), g, terms, G), repeats=3, phase="run",
                              exact=False, **q)
                bench.records[-1]["result"] = dict(max_relative_deviation=dev)


def nonclosed(bench, rng):
    rows = 500 if bench.quick else 4000
    names, X, g, scope, geo = frame(rng, 4, rows)
    B = ms.basis(geo, degree=2)
    st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
    D, _ = closed_dynamics(geo, names, coupling=F(1, 100))

    def refusal():
        try:
            dyn.operator(B, D)
        except dyn.NotClosedError as e:
            return dict(refused=True, term=str(e.term), missing=str(e.missing))
        return dict(refused=False)
    r = bench.measure("closure check (not closed)", "cpu", refusal, repeats=5, phase="run", moments=len(B))
    assert r["refused"]
    bx = dyn.box(geo, {k: (-50, 50) for k in names})
    Kt = bench.measure("truncated operator (box contract)", "cpu", lambda: dyn.operator(B, D, truncate=bx), repeats=3,
                       phase="build", moments=len(B))
    assert Kt.info["approximate"] and Kt.info["invariant"]
    bench.records[-1]["result"] = dict(dropped=Kt.info["dropped"], invariant=Kt.info["invariant"])
    for t in (1, 10):
        ev = bench.measure("truncated evolution (certified bound)", "cpu", lambda: dyn.evolve(st, Kt, t, scope=scope),
                           repeats=3, phase="run", steps=t, moments=len(B), rows=rows)
        rel = max(float(F(ev.errors(gg)[a])) / max(1.0, abs(float(F(ev.moments(gg)[a]))))
                  for gg in range(4) for a in range(len(B)))
        bench.records[-1]["result"] = dict(label=ev.info["label"], max_relative_bound=rel)
    D2, _ = closed_dynamics(geo, names)
    Kw = dyn.operator(B, D2, truncate=bx)
    try:
        dyn.evolve(st, Kw, 2, scope=scope)
        outcome = "allowed"
    except Exception as e:                                  # noqa: BLE001 - the refusal is the measurement
        outcome = f"refused ({type(e).__name__})"
    assert outcome.startswith("refused") and not Kw.info["invariant"]
    bench.note(f"nonclosed: a box that is not F-invariant (v' = v/3 + u^2 on [-50, 50]) certifies one step; two steps: "
               f"{outcome}")


def growth_and_stability(bench, rng):
    names, X, g, scope, geo = frame(rng, 4, 200)
    B = ms.basis(geo, degree=2)
    K = dyn.operator(B, affine_dynamics(geo, names)[0])
    for t in ((10, 100) if bench.quick else (10, 100, 1000)):
        nd = dyn.need("power", K, steps=t, method="squaring")
        pw = K.power(t, method="squaring")
        bench.measure("coefficient growth of K^t", "cpu", lambda: K.power(t, method="squaring"), repeats=1, phase="run",
                      steps=t, moments=len(B))
        bench.records[-1]["result"] = dict(bound_bits=pw.info["bound_bits"], max_bits=pw.info["max_bits"],
                                           den_bits=pw.info["den_bits"], need_bytes=nd["bytes"])
    contracting = dyn.operator(B, dyn.affine(geo, ms.affine(names[:2], [[F(1, 2), F(1, 4)], [F(-1, 4), F(1, 3)]], [1, 0])))
    expanding = dyn.operator(B, dyn.affine(geo, ms.affine(names[:2], [[F(3, 2), F(1, 3)], [0, F(5, 4)]], [1, 1])))
    n = 1000
    for label, op in (("contracting", contracting), ("expanding", expanding)):
        def attempt():
            try:
                fp = op.float_power(n, tolerance=1e-9)
                return dict(stable=True, error=fp.info["error"], reached=fp.info["reached"])
            except dyn.UnstableError as e:
                return dict(stable=False, error=e.info["error"], reached=e.info["reached"])
        r = bench.measure("floating power (rigorous bound)", "cpu", attempt, repeats=3, phase="run", operator=label,
                          exponent=n, tolerance=1e-9)
        assert r["stable"] == (label == "contracting")
        M = np.array([[float(x) for x in row] for row in op.matrix()])

        def numpy_power():                                  # no bound and no diagnosis: an overflow is silently inf
            with np.errstate(over="ignore", invalid="ignore"):
                P = np.linalg.matrix_power(M, n)
            return dict(finite=bool(np.isfinite(P).all()))
        bench.measure("floating power", "numpy", numpy_power, repeats=3, phase="run", operator=label, exponent=n,
                      certified=False)


def thresholds(bench, rng):
    names, X, g, scope, geo = frame(rng, 4, 200)
    B = ms.basis(geo, degree=2)
    st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
    K = dyn.operator(B, affine_dynamics(geo, names)[0])
    limit = 1 << 20
    lo, hi = 1, 1 << 16                                      # the largest exponent whose need fits (needs only)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if dyn.need("power", K, steps=mid, method="squaring")["bytes"] <= limit:
            lo = mid
        else:
            hi = mid
    small = Budget(bytes=limit)
    bench.measure("power within bytes=1 MiB", "cpu", lambda: K.power(lo, method="squaring", budget=small), repeats=1,
                  phase="build", steps=lo)
    rec = bench.refusal("power within bytes=1 MiB", "cpu", lambda: K.power(lo + 1, method="squaring", budget=small),
                        steps=lo + 1)
    bench.threshold("power bytes budget 1 MiB (coefficient growth)", admitted=dict(steps=lo),
                    refused=dict(steps=lo + 1, need=rec.get("need")), resource=rec.get("resource"))
    rec = bench.refusal("evolve within iterations=1000", "cpu",
                        lambda: dyn.evolve(st, K, 1001, budget=Budget(iterations=1000)), steps=1001)
    bench.threshold("evolution step count 1000", admitted=dict(steps=1000), refused=dict(steps=1001),
                    resource=rec.get("resource"))
    dyn.evolve(st, K, 1000, budget=Budget(iterations=1000))
    out = 4 * len(B)
    rec = bench.refusal("evolve within output", "cpu", lambda: dyn.evolve(st, K, 5, budget=Budget(output=out - 1)),
                        output=out)
    bench.threshold("evolution output (groups x moments)", admitted=dict(output=out), refused=dict(output=out - 1),
                    resource=rec.get("resource"))


def main():
    bench = Bench.from_args("DYN-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    dynamics(bench, rng)
    nonclosed(bench, rng)
    growth_and_stability(bench, rng)
    thresholds(bench, rng)
    bench.note("exact results are integer numerators over D^t (every method gives the same integers, checked before "
               "timing); the float64 rows and NumPy powers are inexact baselines; the floating powers are certified "
               "interval enclosures or refused as unstable; the 1,000-step exact row baseline runs on a 64-row subset "
               "(affine), and is refused for the nonlinear map: its proven bound (one common denominator, "
               "Q_(t+1) = L Q_t^2) grows doubly exponentially, while the closed operator's grows linearly")
    bench.write()


if __name__ == "__main__":
    main()
