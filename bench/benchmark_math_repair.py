# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ATTR-002 benchmark: bounded explanation and repair search (redundant predicates, minimal distinguishing sets,
least-cost repairs; exhaustive and branch and bound, exact or incomplete with proven bounds), against exhaustive
coalition calculations.

    python bench/benchmark_math_repair.py --capability ATTR-002 --out FILE [--device] [--quick]

Conditions (case 1): 3 / 6 / 10 / 12 conditions c_i < 50 (quick: 3 / 6 / 10) over 5,000 rows (quick: 2,000), sparse
interactions (every row fails at most two conditions) and dense ones (independent failures, some NULL). The
LATTICE-002 conjunction game v(S) = COUNT(*) WHERE the conditions of S; searches: the cheapest explanation v(S) <= t
(which conditions cut the rows to at most t) and the cheapest repair v(P - S) >= t (which conditions to drop to keep
at least t rows), unit and weighted costs, exhaustive and by branch and bound in index order and in the decreasing
order of ATTR-001's Shapley values (the supported contribution indices, from the sparse Moebius support, checked equal
to the exhaustive enumeration's); the conjunction's keep sets (the jointly redundant predicates), every condition's
redundancy, and the minimal sets explaining a program's membership. Baselines: NumPy (the rows' TRUE masks, the
coalition values by the zeta transform, every coalition's feasibility and cost: the exhaustive minimum with all ties)
and Python enumeration; every answer checked equal, the states each order explores recorded.
Budgets and multiplicities (case 2): planted hitting sets with w disjoint pairs (2^w minimum solutions) or triples
(3^w), w = 2 .. 5 (quick: 2 .. 4), and deletion repairs SUM(v) <= t over 14 deletable rows (quick: 12); each searched
with state budgets of 1/8, 1/2, all but one and all of the states its complete search explores, exhaustively and by
branch and bound: incomplete cases are retained with their proven lower / upper bounds, multiplicity bounds, explored
states and open nodes, every bound checked against the Python enumeration (timed as the baseline). Refusal
thresholds: max_states at the iterations limit (2^20) admitted and one past it refused with that need; a weighted
exhaustive search over 20 allowed elements (quick: 16) measured and over 21 refused as unsupported (the same 21
elements then measured by branch and bound).
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_attr as ma
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_lattice as ml
from torbles import math_repair as mr
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError

METHODS = ("exhaustive", "branch_bound")


def frac(x):
    return None if x is None else (f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator))


# ---------------------------------------------------------------------- case 1: conditions, sparse versus dense
def condition_table(rng, k, rows, mode):
    cols = {}
    if mode == "sparse":                                   # every row fails at most two conditions
        fail = np.zeros((rows, k), dtype=bool)
        for r in range(rows):
            for i in rng.choice(k, size=min(int(rng.integers(0, 3)), k), replace=False):
                fail[r, int(i)] = True
        for i in range(k):
            cols[f"c{i}"] = pd.Series(np.where(fail[:, i], 90, 10).tolist(), dtype=object)
    else:                                                   # independent failures, some NULL (UNKNOWN)
        for i in range(k):
            v = rng.integers(0, 100, rows).tolist()
            for j in np.flatnonzero(rng.random(rows) < 0.03):
                v[int(j)] = None
            cols[f"c{i}"] = pd.Series(v, dtype=object)
    return pd.DataFrame(cols)


def true_masks(frame, k):
    t = np.zeros(len(frame), dtype=np.int64)
    for i in range(k):
        col = frame[f"c{i}"].to_numpy(dtype=object)
        t |= np.array([x is not None and x < 50 for x in col], dtype=np.int64) << i
    return t


def numpy_values(t, k):
    """v(S) = rows TRUE on every condition of S: the TRUE-set histogram, zeta-up"""
    v = np.bincount(t, minlength=1 << k).astype(np.int64)
    idx = np.arange(1 << k)
    for j in range(k):
        lo = idx[(idx >> j) & 1 == 0]
        v[lo] += v[lo | (1 << j)]
    return v


def numpy_minima(feasible, cost):
    """the exhaustive minimum over every coalition (feasible: a boolean array over the 2^k masks) and its ties"""
    if not feasible.any():
        return None, []
    c = np.where(feasible, cost, np.iinfo(np.int64).max)
    m = int(c.min())
    return m, np.flatnonzero(c == m).tolist()


def popcounts(k):
    idx = np.arange(1 << k)
    return np.array([bin(int(s)).count("1") for s in idx], dtype=np.int64)


def conditions(bench, rng):
    rows = 2000 if bench.quick else 5000
    ks = (3, 6, 10) if bench.quick else (3, 6, 10, 12)
    for k in ks:
        full = (1 << k) - 1
        pc = popcounts(k)
        for mode in ("sparse", "dense"):
            p = dict(conditions=k, rows=rows, interactions=mode)
            frame = condition_table(rng, k, rows, mode)
            names = [f"c{i}" for i in range(k)]
            db = torbles.from_ddl("CREATE TABLE t (" + ", ".join(f"{n} BIGINT" for n in names) + ")", {"t": frame})

            def build():
                scope = mbk.Scope.from_table(db["t"], names)
                masks = mb.Masks.from_scope(scope, [mb.condition(n, "<", 50) for n in names])
                hist = ml.Histogram.build(masks, nulls="three")
                return masks, ma.Game.conjunction(hist, players=names)
            masks, g = bench.measure("masks and the conjunction game (state construction)", "cpu", build, repeats=1,
                                     phase="build", **p)
            sup = g.support()
            sh_sup = bench.measure("supported contribution indices (Shapley from the sparse support)", "cpu",
                                   lambda: ma.shapley(sup)["values"], repeats=3, **p)
            bench.records[-1].update(support_terms=sup.info["terms"])
            sh_ex = bench.measure("Shapley by exhaustive coalition enumeration (ATTR-001)", "cpu",
                                  lambda: ma.exhaustive(g)["shapley"], repeats=3, **p)
            if sh_sup != sh_ex:
                raise AssertionError("supported and exhaustive indices differ")
            t = true_masks(frame, k)
            v = bench.measure("coalition values (NumPy TRUE masks, zeta transform)", "numpy",
                              lambda: numpy_values(t, k), repeats=3, **p)
            if v.tolist() != g.values(range(1 << k)):
                raise AssertionError("coalition values differ from NumPy")
            w = [Fraction(int(a), int(b)) for a, b in zip(rng.integers(1, 9, k), rng.integers(1, 4, k))]
            L = math.lcm(*[x.denominator for x in w])
            wcost = np.zeros(1 << k, dtype=np.int64)
            for i in range(k):
                wcost += ((np.arange(1 << k) >> i) & 1) * int(w[i] * L)
            order_by = tuple(sorted(range(k), key=lambda i: (-sh_ex[i], i)))
            vfull = int(v[full])
            for kind, op, th in (("explanation", "<=", vfull + (int(v[0]) - vfull) // 4),
                                 ("repair", ">=", vfull + (int(v[0]) - vfull) // 2)):
                feas = v <= th if kind == "explanation" else v[full & ~np.arange(1 << k)] >= th
                P = mr.Problem.game(g, kind, op, th)
                for cost, cvec, cname in ((None, pc, "unit"), (w, wcost, "weighted")):
                    q = dict(p, kind=kind, op=op, threshold=th, cost=cname)
                    best, ties = bench.measure("every coalition (NumPy feasibility and cost)", "numpy",
                                               lambda: numpy_minima(feas, cvec), repeats=3, **q)
                    want = None if best is None else Fraction(best, 1 if cost is None else L)
                    for method, order in (("exhaustive", "index"), ("branch_bound", "index"),
                                          ("branch_bound", "shapley")):
                        for backend in bench.backends():
                            r = bench.measure(f"search ({method}, {order} order)", backend,
                                              lambda b=backend, me=method, o=order: mr.search(
                                                  P, cost=cost, method=me, order=o, max_solutions=1 << k, backend=b),
                                              repeats=3, method=method, order=order, **q)
                            bench.records[-1].update(status=r["status"], minimum=frac(r["minimum"]),
                                                     multiplicity=r["multiplicity"], explored=r["explored"],
                                                     pruned=r["pruned"], total=r["total"])
                            if (r["minimum"], sorted(r["masks"])) != (want, ties):
                                raise AssertionError(f"search differs from the enumeration ({k}, {mode}, {kind})")
                            if order == "shapley" and r["order_elements"] != tuple(names[i] for i in order_by):
                                raise AssertionError("the Shapley order differs from ATTR-001's indices")
            # keep sets (jointly redundant predicates), redundancy, membership explanations
            K = mr.Problem.keep(masks, range(k))
            sel_full = (t & full) == full
            keeps = np.array([np.array_equal((t & S) == S, sel_full) for S in range(1 << k)])
            best, ties = numpy_minima(keeps, pc)
            for method in METHODS:
                for backend in bench.backends():
                    r = bench.measure(f"keep sets ({method})", backend,
                                      lambda b=backend, me=method: mr.search(K, method=me, max_solutions=1 << k,
                                                                             backend=b), repeats=3, **p)
                    bench.records[-1].update(minimum=frac(r["minimum"]), multiplicity=r["multiplicity"],
                                             explored=r["explored"], masks=K.info["masks"])
                    if (r["minimum"], sorted(r["masks"])) != (best, ties):
                        raise AssertionError("keep sets differ from the enumeration")
            red = bench.measure("redundancy of every condition", "cpu", lambda: mr.redundant(K, witnesses=0),
                                repeats=3, **p)
            if [x["redundant"] for x in red] != [bool(keeps[full & ~(1 << i)]) for i in range(k)]:
                raise AssertionError("redundancy differs from the enumeration")
            half = max(1, k // 2)
            xs = mb.variables(k)
            expr = xs[0]
            for i in range(1, half):
                expr = expr & xs[i]
            alt = xs[half] if half < k else xs[0]
            for i in range(half + 1, k):
                alt = alt & xs[i]
            prog = mb.Program(expr | alt, k)
            E = mr.Problem.explain(masks, prog)
            r = bench.measure("membership explanation (minimal distinguishing sets)", "cpu",
                              lambda: mr.search(E, max_solutions=1 << k), repeats=3, **p)
            bench.records[-1].update(minimum=frac(r["minimum"]), multiplicity=r["multiplicity"],
                                     masks=E.info["masks"], pairs=E.info["pairs"])
            sel = ((t & ((1 << half) - 1)) == (1 << half) - 1) | ((t >> half) == (full >> half))
            ta, tb = np.unique(t[sel]), np.unique(t[~sel])

            def separating():
                return np.array([not np.intersect1d(np.unique(ta & S), np.unique(tb & S)).size
                                 for S in range(1 << k)])
            sep = bench.measure("membership explanations: every condition set (NumPy patterns)", "numpy", separating,
                                repeats=1, **p)
            eb, et = numpy_minima(sep, pc)
            if (r["minimum"], sorted(r["masks"])) != (eb, et):
                raise AssertionError("membership explanations differ from the enumeration")
            sup.close(); g.close(); masks.close()
    bench.note("conditions: every search (exhaustive; branch and bound in index and Shapley order; CUDA with "
               "--device) gives the NumPy enumeration's minimum with all its ties; the supported indices equal the "
               "exhaustive ones; the explored states show the work each order needs")


# ---------------------------------------------------------------------- case 2: budgets, multiplicities
def python_minimum(n, feasible):
    best, mult = None, 0
    for size in range(n + 1):
        for C in itertools.combinations(range(n), size):
            if feasible(sum(1 << c for c in C)):
                mult += 1
        if mult:
            return size, mult
    return best, mult


def budgets(bench, rng):
    ws = (2, 3, 4) if bench.quick else (2, 3, 4, 5)
    problems = []
    for arity in (2, 3):
        for w in ws:
            msk = [((1 << arity) - 1) << (arity * i) for i in range(w)]
            n = arity * w
            problems.append((f"planted {arity}-sets", dict(arity=arity, witnesses=w, elements=n),
                             mr.Problem.masks(n, msk), n, lambda S, msk=msk: all(S & m for m in msk)))
    nd = 12 if bench.quick else 14
    vals = [int(x) for x in rng.integers(1, 60, nd)]
    frame = pd.DataFrame({"v": pd.Series(vals + [int(x) for x in rng.integers(0, 60, 40)], dtype=object)})
    db = torbles.from_ddl("CREATE TABLE t (v BIGINT)", {"t": frame})
    scope = mbk.Scope.from_table(db["t"], ["v"])
    masks = mb.Masks.from_scope(scope, [mb.condition("v", ">=", 0)])
    total = sum(int(x) for x in frame["v"])
    target = total - sum(vals) // 2
    problems.append(("deletion SUM(v) <= t", dict(elements=nd, target=target),
                     mr.Problem.delete(masks, list(range(nd)), aggregate="sum", value="v", op="<=", target=target),
                     nd, lambda S: total - sum(vals[i] for i in range(nd) if (S >> i) & 1) <= target))
    for name, p, P, n, feasible in problems:
        best, mult = bench.measure(f"{name}: every subset (Python)", "python", lambda: python_minimum(n, feasible),
                                   repeats=1, **p)
        for method in METHODS:
            full = mr.search(P, method=method, max_solutions=8)
            if (full["minimum"], full["multiplicity"]) != (best, mult):
                raise AssertionError(f"{name}: the complete search differs from the enumeration")
            needed = full["explored"]
            for label, states in (("1/8", max(1, needed // 8)), ("1/2", max(1, needed // 2)),
                                  ("all but one", max(1, needed - 1)), ("all", needed)):
                for backend in bench.backends():
                    r = bench.measure(f"{name}: search under a state budget", backend,
                                      lambda b=backend, m=method, s=states: mr.search(P, method=m, max_states=s,
                                                                                      max_solutions=8, backend=b),
                                      repeats=2, method=method, budget=label, max_states=states, **p)
                    lo, hi = r["multiplicity_bounds"]
                    bench.records[-1].update(status=r["status"], lower=frac(r["lower"]), upper=frac(r["upper"]),
                                             multiplicity_lower=lo, multiplicity_upper=hi, explored=r["explored"],
                                             open=r["open"], pruned=r["pruned"], expected_minimum=best,
                                             expected_multiplicity=mult)
                    if r["explored"] > states:
                        raise AssertionError("a search explored past its state budget")
                    if r["status"] == "found":
                        if (r["minimum"], r["multiplicity"]) != (best, mult):
                            raise AssertionError(f"{name}: a complete answer differs from the enumeration")
                    else:
                        if not (r["lower"] <= best and (r["upper"] is None or best <= r["upper"])):
                            raise AssertionError(f"{name}: the proven bounds miss the minimum")
                        if r["lower"] == r["upper"] and not (lo <= mult and (hi is None or mult <= hi)):
                            raise AssertionError(f"{name}: the multiplicity bounds miss the count")
        P.close()
    bench.note("budgets: incomplete searches are retained with their proven cost and multiplicity bounds, explored "
               "states and open nodes; every bound contains the enumerated answer, every complete answer equals it")
    # refusal thresholds: the declared state budget against the iterations limit, the weighted exhaustive order
    P = mr.Problem.masks(10, [0b11 << (2 * i) for i in range(5)])
    lim = Budget(iterations=1 << 20)
    bench.measure("search with max_states at the iterations limit", "cpu",
                  lambda: mr.search(P, max_states=1 << 20, budget=lim), repeats=1, max_states=1 << 20,
                  iterations=1 << 20)
    rec = bench.refusal("search with max_states one past the iterations limit", "cpu",
                        lambda: mr.search(P, max_states=(1 << 20) + 1, budget=lim), max_states=(1 << 20) + 1,
                        iterations=1 << 20)
    if not rec["refused"] or rec["need"] != (1 << 20) + 1:
        raise AssertionError("the state budget is not refused at the iterations limit + 1")
    bench.threshold("search state budget (iterations = max_states)", admitted=dict(max_states=1 << 20),
                    refused=dict(max_states=(1 << 20) + 1), need_reported=(1 << 20) + 1)
    m = 16 if bench.quick else 20
    Wm = mr.Problem.masks(m, [[i, i + 1] for i in range(m - 1)])         # a path's vertex covers
    r = bench.measure("weighted exhaustive search (the sorted order of 2^m subsets)", "cpu",
                      lambda: mr.search(Wm, cost=[1] * (m - 1) + [2], method="exhaustive", max_states=1 << m),
                      repeats=1, allowed=m)
    bench.records[-1].update(status=r["status"], minimum=frac(r["minimum"]), explored=r["explored"])
    W21 = mr.Problem.masks(21, [[i, i + 1] for i in range(20)])
    try:
        mr.search(W21, cost=[1] * 20 + [2], method="exhaustive")
        raise AssertionError("a weighted exhaustive order above 20 allowed elements was not refused")
    except UnsupportedError as e:
        bench.note(f"refused as unsupported: a weighted exhaustive search over 21 allowed elements ({e})")
    r = bench.measure("the same 21 elements by branch and bound (weighted)", "cpu",
                      lambda: mr.search(W21, cost=[1] * 20 + [2], method="branch_bound", max_states=200_000),
                      repeats=1, allowed=21)
    bench.records[-1].update(status=r["status"], minimum=frac(r["minimum"]), explored=r["explored"])
    bench.threshold("weighted exhaustive search (allowed elements; unsupported above 20)", admitted=dict(allowed=m),
                    refused=dict(allowed=21))


def main():
    bench = Bench.from_args("ATTR-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    conditions(bench, rng)
    budgets(bench, rng)
    bench.write()


if __name__ == "__main__":
    main()
