# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROB-002 benchmark: factorized probabilistic inference over factor graphs with declared finite supports.

    python bench/benchmark_math_factor.py --capability PROB-002 --out FILE [--device] [--quick]

Case 1 (the inventory's first case, with factor graphs as the retained states): joint supports of 8 / 64 / 1024 points
(rings of 3 / 6 / 10 binary variables whose pairwise factors are TENSOR-001 count tensors of one data scope, admitted by
a GEN-002 independence declaration) and 1 / 100 / 10,000 likelihood requests (each observes half of the variables;
quick: 8 / 64 and 1 / 100). State construction (the scope, the count tensors, the graph, the plan and program, the
normalizer Z) is timed apart from the batches (first execution and warm reuse of the retained query, CPU; CUDA with
--device). Baseline: repeated support enumeration per request (NumPy over the joint support, float64 for the timing;
the exact answers of up to 100 requests are checked against Python-integer enumeration of the same tables).

Case 2: graph size and elimination width: chains, random trees, grids and cliques of binary variables, from tiny
brute-force-checked instances (COUNT-002's native brute force over the factor universe, and Python Fraction
enumeration for the smallest) to bounded larger models; per instance the plan (the chosen heuristic, the induced width,
the largest joint table, the largest intermediate support, products, the limbs L of the exact arithmetic), the
partition function (CPU; CUDA with --device), a marginal of two distant variables and the maximum with its ties, the
exact precision (bit lengths of Z's numerator and denominator), and COUNT-002's variable elimination as a reference
timing. Refusal thresholds: grids under a benchmark tensor budget (2^12 joint cells): the largest admitted and the
smallest refused grid, with whether the refusal is proven for every order (never extrapolated).
"""
import itertools
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_conv as mc, math_factor as mf, math_tensor as mt
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError


# ---------------------------------------------------------------------- case 1: retained likelihood states
def ring_data(rng, k, rows):
    """k correlated binary columns (a noisy ring: each column copies its predecessor with probability 0.7)"""
    cols = {}
    prev = rng.integers(0, 2, rows)
    for i in range(k):
        cur = np.where(rng.random(rows) < 0.7, prev, rng.integers(0, 2, rows))
        cols[f"x{i}"] = cur.astype(np.int64)
        prev = cur
    return cols


def ring_graph(scope, k):
    """pairwise count-tensor factors around the ring (one data source: declared independent)"""
    F = [mf.tensor(mt.counts(scope, [f"x{i}", f"x{(i + 1) % k}"]), [f"x{i}", f"x{(i + 1) % k}"]) for i in range(k)]
    return mf.graph({f"x{i}": 2 for i in range(k)}, F, declaration=mc.independent(scope))


def joint_table(G, k):
    """the joint weights of every point (Python ints: exact), from the graph's own tables"""
    tabs = [G.table(f) for f in range(len(G.factors))]
    scopes = [G.factors[f].scope for f in range(len(G.factors))]
    pts = list(itertools.product(range(2), repeat=k))
    W = []
    for p in pts:
        a = {f"x{i}": p[i] for i in range(k)}
        w = 1
        for sc, tab in zip(scopes, tabs):
            w *= int(tab[tuple(a[v] for v in sc)])
        W.append(w)
    return pts, W


def enumerate_requests(W_float, pts, obs, values):
    """repeated support enumeration: per request, the sum of the joint weights of the consistent points (float64)"""
    P = np.array(pts, dtype=np.int64)
    out = []
    for r in range(values.shape[0]):
        mask = np.ones(len(pts), dtype=bool)
        for j, v in enumerate(obs):
            mask &= P[:, v] == values[r, j]
        out.append(W_float[mask].sum())
    return out


def case1(bench, rng):
    rows = 2_000 if bench.quick else 20_000
    sizes = (3, 6) if bench.quick else (3, 6, 10)
    requests = (1, 100) if bench.quick else (1, 100, 10_000)
    for k in sizes:
        d = 1 << k
        cols = ring_data(rng, k, rows)
        p = dict(support=d, variables=k, rows=rows)
        scope = bench.measure("scope (FND-004, the data)", "cpu", lambda: mbk.Scope.from_arrays(cols), repeats=2,
                              phase="build", **p)
        G = bench.measure("factor graph: count tensors (one scan each) and the graph", "cpu",
                          lambda: ring_graph(scope, k), repeats=2, phase="build", **p)
        bench.records[-1].update(graph=G.info)
        Zr = bench.measure("normalizer Z (planned elimination, retained)", "cpu", lambda: G.normalizer(), repeats=1,
                           phase="build", **p)
        Z = Zr[0].Z_e
        obs = list(range(0, k, 2))
        pts, W = joint_table(G, k)
        Wf = np.array([float(w) for w in W])
        for R in requests:
            values = rng.integers(0, 2, (R, len(obs))).astype(np.uint32)
            q = dict(p, requests=R, observed=len(obs))
            query = bench.measure("retained query: plan and program for the evidence shape", "cpu",
                                  lambda: G.query(evidence=[f"x{i}" for i in obs], requests=R), repeats=2,
                                  phase="build", **q)
            bench.records[-1].update(plan=query.plan.summary())
            for backend in bench.backends():
                res = bench.measure("likelihood batch: exact Z_e and P(e) per request (retained state)", backend,
                                    lambda: query.run(values, backend=backend), repeats=2 if R >= 10_000 else 4,
                                    phase="run", **q)
            col = bench.measure("exact readout of the batch's P(e) column as Fractions (one native call)", "cpu",
                                lambda: res.column("probability"), repeats=2, phase="run", **q)
            raw = bench.measure("the raw elimination alone (native arena, no exact readout)", "cpu",
                                lambda: query.raw(values, backend="cpu"), repeats=2 if R >= 10_000 else 4,
                                phase="run", **q)
            bench.measure("numpy: repeated support enumeration per request (float64, uncertified)", "numpy",
                          lambda: enumerate_requests(Wf, pts, obs, values), repeats=2 if R >= 10_000 else 3,
                          phase="run", **q)
            # exactness: the first requests against Python-integer enumeration of the same tables
            m = min(R, 100)
            P = np.array(pts)
            ok = True
            for r in range(m):
                mask = np.all(P[:, obs] == values[r], axis=1)
                ze = sum(w for w, mk in zip(W, mask) if mk)
                ok = ok and res[r].Z_e == ze and res[r].probability == Fraction(ze, int(Z)) == col[r]
            bench.records[-1].update(exact_checked=m, exact_equal=bool(ok),
                                     z_bits=[int(Z).bit_length()], arena_entries=int(raw["vals"].size))


# ---------------------------------------------------------------------- case 2: graph size and elimination width
def tables(rng, names, scopes):
    V = {n: 2 for n in names}
    F = []
    for sc in scopes:
        t = rng.integers(1, 7, [2] * len(sc)).astype(object)
        F.append((sc, t))
    return V, F


def family(kind, size, rng):
    if kind == "chain":
        names = [f"v{i}" for i in range(size)]
        return names, [(names[i], names[i + 1]) for i in range(size - 1)]
    if kind == "tree":
        names = [f"v{i}" for i in range(size)]
        return names, [(names[int(rng.integers(0, i))], names[i]) for i in range(1, size)]
    if kind == "grid":
        r = c = size
        names = [f"g{i}_{j}" for i in range(r) for j in range(c)]
        sc = [(f"g{i}_{j}", f"g{i}_{j + 1}") for i in range(r) for j in range(c - 1)]
        sc += [(f"g{i}_{j}", f"g{i + 1}_{j}") for i in range(r - 1) for j in range(c)]
        return names, sc
    names = [f"k{i}" for i in range(size)]
    return names, list(itertools.combinations(names, 2))


def fraction_enumeration(names, F):
    Z = Fraction(0)
    for p in itertools.product(range(2), repeat=len(names)):
        a = dict(zip(names, p))
        w = 1
        for sc, t in F:
            w *= int(t[tuple(a[v] for v in sc)])
        Z += w
    return Z


BIG = Budget(variables=4096)        # FND-010's default admits 64 variables; the larger models declare more


def case2(bench, rng):
    grid = {"chain": (4, 8, 64) if bench.quick else (4, 8, 16, 64, 256, 1024),
            "tree": (8, 32) if bench.quick else (8, 32, 128, 512),
            "grid": (2, 3, 4) if bench.quick else (2, 3, 4, 5, 6, 8, 10, 12),
            "clique": (3, 4, 6) if bench.quick else (3, 4, 5, 6, 8, 10, 12, 14)}
    for kind, sizes in grid.items():
        for size in sizes:
            names, scopes = family(kind, size, rng)
            V, F = tables(rng, names, scopes)
            n = len(names)
            p = dict(family=kind, size=size, variables=n, factors=len(F))
            G = bench.measure("graph construction", "cpu",
                              lambda: mf.graph(V, [mf.table(list(sc), t.tolist()) for sc, t in F], budget=BIG),
                              repeats=2, phase="build", **p)
            plan = bench.measure("plan: costed candidate orders (FND-008)", "cpu", lambda: G.plan(budget=BIG),
                                 repeats=2, phase="build", **p)
            s = plan.summary()
            bench.records[-1].update(plan=s, complete=plan.info["complete"], dp_states=plan.info["dp_states"])
            for backend in bench.backends():
                r = bench.measure("partition function (exact)", backend,
                                  lambda: G.partition(backend=backend, budget=BIG), repeats=3, phase="run", **p)
            Z = r.Z_e
            rec = dict(width=s["width"], joint=s["joint"], intermediate_support=s["support"], products=s["products"],
                       limbs=s["limbs"], heuristic=s["heuristic"], precision="exact rational",
                       z_numerator_bits=Z.numerator.bit_length(), z_denominator_bits=Z.denominator.bit_length())
            bench.records[-1].update(rec)
            a, b = names[0], names[-1]
            bench.measure("marginal of two distant variables (exact)", "cpu", lambda: G.marginal([a, b], budget=BIG),
                          repeats=2, phase="run", **p)
            try:
                m = bench.measure("maximum with every tie (exact)", "cpu", lambda: G.map(budget=BIG), repeats=2,
                                  phase="run", **p)
                bench.records[-1].update(ties=m.count)
            except AlgebraBudgetError:                    # more ties than the output budget: the exact count refused
                bench.records.pop() if bench.records and bench.records[-1].get("case") == "maximum with every tie (exact)" else None
                rec = bench.refusal("maximum with every tie (exact)", "cpu", lambda: G.map(budget=BIG), **p)
                rec.update(note="every tied maximizer is enumerated: the exact number of ties is the output need")
            if n <= 24:
                c = bench.measure("COUNT-002 variable elimination (reference)", "count002",
                                  lambda: G.count(method="symcount", budget=BIG), repeats=2, phase="run", **p)
                bench.records[-1].update(equal=c == Z)
            if n <= 16:
                c = bench.measure("COUNT-002 brute force over the factor universe (check)", "count002-brute",
                                  lambda: G.count(method="brute", budget=BIG), repeats=1, phase="run", **p)
                bench.records[-1].update(equal=c == Z, brute_force_checked=True)
            if n <= 10:
                e = bench.measure("Python Fraction enumeration (check)", "python", lambda: fraction_enumeration(names, F),
                                  repeats=1, phase="run", **p)
                bench.records[-1].update(equal=e == Z)
    # refusal thresholds: grids under a benchmark tensor budget (never extrapolated)
    budget = Budget(variables=4096, tensor=1 << 12)
    admitted = refused = None
    for size in ((2, 3, 4, 6, 8, 12, 14) if not bench.quick else (2, 3, 12)):
        names, scopes = family("grid", size, rng)
        V, F = tables(rng, names, scopes)
        G = mf.graph(V, [mf.table(list(sc), t.tolist()) for sc, t in F], budget=BIG)
        try:
            plan = G.plan(budget=budget)
            admitted = dict(grid=f"{size}x{size}", joint=plan.chosen["joint"], width=plan.chosen["width"])
        except AlgebraBudgetError as err:
            rec = bench.refusal("grid partition under tensor = 2^12", "cpu", lambda: G.plan(budget=budget),
                                grid=f"{size}x{size}")
            best = min(c["joint"] for c in err.plan.candidates)       # the smallest joint table among the candidates
            rec.update(every_order=bool(err.every_order), minima=err.minima, best_candidate_joint=best)
            if refused is None:
                refused = dict(grid=f"{size}x{size}", first_candidate_need=err.need, best_candidate_joint=best,
                               every_order=bool(err.every_order), minima=err.minima)
    bench.threshold("grid partition under a tensor budget of 2^12 joint cells", admitted=admitted, refused=refused,
                    budget=dict(tensor=1 << 12))


def main():
    bench = Bench.from_args("PROB-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    case1(bench, rng)
    case2(bench, rng)
    bench.note("CUDA timings exist only with --device on an actual GPU; without it the CUDA path is not timed (the "
               "tests check its emulated arena bit for bit against the CPU).")
    bench.note("Every answer is an exact rational (precision: exact); the limbs L bound every intermediate integer "
               "(64 L bits); z_*_bits are the bit lengths of the reduced Z. The NumPy enumeration baseline is float64 "
               "and uncertified; its requests are checked exactly (exact_equal) against Python-integer enumeration.")
    bench.note("Plans: 'complete' means the exact subset DP considered every order (at most 14 eliminated variables "
               "by default); larger models use the heuristic candidates (min-fill, min-degree, min-weight, "
               "weighted min-fill, natural, reverse) chosen by FND-008's cost (relative work units, not time).")
    bench.write()


if __name__ == "__main__":
    main()
