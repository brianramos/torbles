# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ATTR-001 benchmark: exact attribution of declared cooperative games (sparse Moebius supports, exhaustive weighted
marginal contributions, bounded exploration, minimum explanation / repair search), against exhaustive coalition
calculations.

    python bench/benchmark_math_attr.py --capability ATTR-001 --out FILE [--device] [--quick]

Conditions (case 1): conjunction games v(S) = COUNT(*) WHERE the conditions of S, over 3 / 6 / 10 / 12 conditions
(quick: 3 / 6 / 10) of 20,000 rows (quick: 5,000), with sparse interactions (every row fails at most two conditions:
the Moebius support has sets of at most 2 conditions) and dense interactions (every condition fails independently
with probability 1/2, some NULL: support near 2^k). Recorded: the histogram and game construction (build), the
sparse support (terms, degree), the Shapley and Banzhaf values from the support and by the native exhaustive
enumeration (CUDA with --device), and the baselines: NumPy (the TRUE-set histogram, its zeta transform and the exact
int64 subset-formula numerators), Python Fractions (the subset formula over every coalition) and DuckDB (COUNT(*) re-run
per coalition, up to 10 conditions; quick: 6), every index checked equal. Provenance: the COUNT(*) of a join with one
player per source tuple, sparse support against the exhaustive enumeration as the players grow (16 / 20 tuples), and
the exhaustive iteration budget's largest admitted / smallest refused size.
Search (case 2): minimum repairs (remove a smallest set of tuples so that the join's output disappears) and minimum
explanations (a smallest sufficient set) of planted witness structures: w disjoint pairs (the minimum repair has
size w and multiplicity 2^w) and w disjoint triples (multiplicity 3^w), w = 2 .. 5 (quick: 2 .. 4), under coalition
budgets 1/8, 1/2, all but one and all of the evaluations the exact answer needs; incomplete results are retained with
their certified size and multiplicity bounds and the explored states; baseline: Python enumeration of every coalition.
Bounded attribution: a join game's Shapley bounds (Moebius regime) and an existence game's (range regime) under
growing exploration budgets, the bound widths and the explored states recorded.
"""
import itertools
import math
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_attr as ma
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_lattice as ml
from torbles import math_prov as mp
from torbles.math_budgets import Budget


def frac(x):
    return f"{x.numerator}/{x.denominator}" if isinstance(x, Fraction) and x.denominator != 1 else str(int(x))


# ---------------------------------------------------------------------- case 1: conditions, sparse versus dense
def condition_table(rng, k, rows, mode):
    """columns c0 .. c(k-1); condition i is c_i < 50"""
    cols = {}
    if mode == "sparse":                                   # every row fails at most two conditions
        fail = np.zeros((rows, k), dtype=bool)
        for r in range(rows):
            m = int(rng.integers(0, 3))
            for i in rng.choice(k, size=min(m, k), replace=False):
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


def numpy_indices(frame, k):
    """the independent NumPy baseline: TRUE-set histogram, zeta-up, exact int64 subset-formula numerators"""
    true = np.zeros(len(frame), dtype=np.int64)
    for i in range(k):
        col = frame[f"c{i}"].to_numpy(dtype=object)
        ok = np.array([x is not None and x < 50 for x in col])
        true |= ok.astype(np.int64) << i
    v = np.bincount(true, minlength=1 << k).astype(np.int64)
    for j in range(k):                                      # zeta-up: points TRUE on at least S
        half = 1 << j
        idx = np.arange(1 << k)
        lo = idx[(idx & half) == 0]
        v[lo] += v[lo | half]
    idx = np.arange(1 << k)
    pc = np.array([bin(int(s)).count("1") for s in idx])
    fact = [math.factorial(t) for t in range(k + 1)]
    sh, bz = [], []
    for i in range(k):
        S = idx[(idx >> i) & 1 == 0]
        d = v[S | (1 << i)] - v[S]
        w = np.array([fact[s] * fact[k - s - 1] for s in pc[S]], dtype=np.int64)
        sh.append(Fraction(int(np.dot(w, d)), fact[k]))     # |numerator| <= k! rows < 2^63
        bz.append(Fraction(int(d.sum()), 1 << (k - 1)))
    return v.tolist(), sh, bz


def fraction_indices(v, k):
    sh, bz = [], []
    for i in range(k):
        a, b = Fraction(0), Fraction(0)
        for S in range(1 << k):
            if (S >> i) & 1:
                continue
            s = bin(S).count("1")
            d = v[S | (1 << i)] - v[S]
            a += Fraction(math.factorial(s) * math.factorial(k - s - 1), math.factorial(k)) * d
            b += d
        sh.append(a)
        bz.append(b / (1 << (k - 1)))
    return sh, bz


def conditions(bench, rng):
    rows = 5000 if bench.quick else 20000
    ks = (3, 6, 10) if bench.quick else (3, 6, 10, 12)
    duck_max = 6 if bench.quick else 10
    for k in ks:
        for mode in ("sparse", "dense"):
            p = dict(conditions=k, rows=rows, interactions=mode)
            frame = condition_table(rng, k, rows, mode)
            names = [f"c{i}" for i in range(k)]
            db = torbles.from_ddl("CREATE TABLE t (" + ", ".join(f"{n} BIGINT" for n in names) + ")", {"t": frame})

            def build():
                scope = mbk.Scope.from_table(db["t"], names)
                masks = mb.Masks.from_scope(scope, [mb.condition(n, "<", 50) for n in names])
                hist = ml.Histogram.build(masks, nulls="three")
                return masks, hist, ma.Game.conjunction(hist, players=names)
            masks, hist, g = bench.measure("conjunction game (masks, histogram, coalition values)", "cpu", build,
                                           repeats=2, phase="build", **p)
            sup = bench.measure("sparse Moebius support", "cpu", g.support, repeats=3, phase="build", **p)
            info = sup.info
            bench.records[-1].update(support_terms=info["terms"], support_degree=info["degree"],
                                     coalitions=1 << k)
            native = bench.measure("Shapley and Banzhaf from the support", "cpu",
                                   lambda: (ma.shapley(sup)["values"], ma.banzhaf(sup)["values"]), repeats=5, **p)
            for backend in bench.backends():
                ex = bench.measure("Shapley and Banzhaf by exhaustive enumeration", backend,
                                   lambda b=backend: ma.exhaustive(g, backend=b), repeats=3, **p)
                if (ex["shapley"], ex["banzhaf"]) != native:
                    raise AssertionError(f"exhaustive {backend} differs from the support ({k}, {mode})")
            v_np, sh_np, bz_np = bench.measure("exhaustive indices (NumPy: histogram, zeta, int64 numerators)",
                                               "numpy", lambda: numpy_indices(frame, k), repeats=2, **p)
            if (sh_np, bz_np) != native or v_np != g.values(range(1 << k)):
                raise AssertionError(f"NumPy baseline differs ({k}, {mode})")
            sh_f, bz_f = bench.measure("exhaustive indices (Python Fractions, subset formula)", "python",
                                       lambda: fraction_indices(v_np, k), repeats=1, **p)
            if (sh_f, bz_f) != native:
                raise AssertionError(f"Fraction baseline differs ({k}, {mode})")
            if k <= duck_max:
                def duck():
                    return [int(db.con.execute("SELECT COUNT(*) FROM t WHERE " + " AND ".join(
                        ["TRUE"] + [f"c{i} < 50" for i in range(k) if (S >> i) & 1])).fetchone()[0])
                        for S in range(1 << k)]
                vd = bench.measure("coalition values (DuckDB COUNT(*) re-run per coalition)", "duckdb", duck,
                                   repeats=1, **p)
                if vd != v_np:
                    raise AssertionError(f"DuckDB coalition values differ ({k}, {mode})")
            it = bench.measure("Shapley interactions up to order 2 (support)", "cpu",
                               lambda: ma.interactions(sup, 2, index="shapley"), repeats=3, **p)
            bench.records[-1].update(nonzero_sets=len(it["sets"]))
            sup.close(); g.close(); hist.close(); masks.close()
    bench.note("conditions: the support, exhaustive (CPU and, with --device, CUDA), NumPy and Fraction indices are "
               "equal for every grid point; DuckDB coalition values equal the native ones where run")


def provenance(bench, rng):
    sizes = (12, 16) if bench.quick else (16, 20)
    for n in sizes:
        half = n // 2
        st = mp.Store()
        R = st.source("R", {"k": [int(x) for x in rng.integers(0, 4, half)], "a": list(range(half))})
        S = st.source("S", {"k": [int(x) for x in rng.integers(0, 4, n - half)], "b": list(range(n - half))})
        g = ma.Game.provenance(R.join(S, on=[("k", "k")]).group([]))
        p = dict(players=g.n, coalitions=1 << g.n)
        sup = bench.measure("join COUNT(*): sparse support", "cpu", g.support, repeats=3, phase="build", **p)
        bench.records[-1].update(support_terms=sup.info["terms"])
        a = bench.measure("join COUNT(*): Shapley from the support", "cpu", lambda: ma.shapley(sup)["values"],
                          repeats=5, **p)
        for backend in bench.backends():
            ex = bench.measure("join COUNT(*): exhaustive enumeration", backend,
                               lambda b=backend: ma.exhaustive(g, backend=b)["shapley"], repeats=2, **p)
            if ex != a:
                raise AssertionError("provenance exhaustive differs from the support")
        e = bench.measure("join COUNT(*): bounded exploration (degree 2: 2n evaluations)", "cpu",
                          lambda: ma.explore(g, 2 * g.n), repeats=3, **p)
        if not e["exact"] or e["values"]["shapley"] != a:
            raise AssertionError("bounded exploration did not reach the exact values")
        bench.records[-1].update(explored=e["explored"], exact=e["exact"])
    # the exhaustive enumeration's iteration budget: the largest admitted and the smallest refused player count
    st = mp.Store()
    R = st.source("R", {"k": list(range(21)), "a": [0] * 21})
    g20 = ma.Game.provenance(R.group([]), tuples=None, players=[("R", i) for i in range(20)])
    g21 = ma.Game.provenance(R.group([]))
    b = Budget(iterations=1 << 20)
    bench.measure("exhaustive at the iteration budget", "cpu", lambda: ma.exhaustive(g20, budget=b), repeats=1,
                  players=20, iterations=1 << 20)
    bench.refusal("exhaustive past the iteration budget", "cpu", lambda: ma.exhaustive(g21, budget=b),
                  players=21, iterations=1 << 21)
    bench.threshold("exhaustive enumeration (iterations = 2^n, default limit 2^20)", admitted=dict(players=20),
                    refused=dict(players=21))


# ---------------------------------------------------------------------- case 2: search budgets, multiplicities
def planted(arity, w):
    """w disjoint witnesses of `arity` tuples (a join of `arity` sources on k): the minimum repair has size w and
    multiplicity arity^w; the minimum explanation is one witness (size arity, multiplicity w)"""
    st = mp.Store()
    rels = [st.source(f"R{j}", {"k": list(range(w)), f"a{j}": [j] * w}) for j in range(arity)]
    J = rels[0]
    for r in rels[1:]:
        J = J.join(r, on=[("k", "k")])
    return st, ma.Game.provenance(J.group([]), semiring="bool")


def brute_search(g, arity, w, kind):
    n = g.n
    pos = {lab: i for i, lab in enumerate(g.players)}
    wit = [sum(1 << pos[(f"R{j}", i)] for j in range(arity)) for i in range(w)]
    full = (1 << n) - 1

    def exists(S):
        return any(S & m == m for m in wit)
    best, mult = None, 0
    for d in range(n + 1):
        for C in itertools.combinations(range(n), d):
            m = sum(1 << c for c in C)
            ok = exists(m) if kind == "explanation" else not exists(full & ~m)
            if ok:
                mult += 1
        if mult:
            best = d
            break
    return best, mult


def search(bench, rng):
    ws = (2, 3, 4) if bench.quick else (2, 3, 4, 5)
    for arity in (2, 3):
        for w in ws:
            st, g = planted(arity, w)
            n = g.n
            for kind, op, th in (("repair", "<=", 0), ("explanation", ">=", 1)):
                best, mult = bench.measure(f"{kind}: every coalition (Python)", "python",
                                           lambda: brute_search(g, arity, w, kind), repeats=1, players=n,
                                           witnesses=w, arity=arity)
                needed = sum(math.comb(n, d) for d in range(1, best + 1))
                for frac_ in ("1/8", "1/2", "all but one", "all"):
                    budget = {"1/8": needed // 8, "1/2": needed // 2, "all but one": needed - 1, "all": needed}[frac_]
                    for backend in bench.backends():
                        r = bench.measure(f"minimum {kind} search", backend,
                                          lambda b=backend, m=budget: ma.search(g, kind, op, th, max_coalitions=m,
                                                                                 max_solutions=8, backend=b),
                                          repeats=2, players=n, witnesses=w, arity=arity, budget=frac_,
                                          max_coalitions=budget)
                        bench.records[-1].update(
                            status=r["status"], min_size=list(r["min_size"]) if r["min_size"] else None,
                            multiplicity=list(r["multiplicity"]) if r["multiplicity"] else None,
                            explored=r["explored"], level=r["level"], level_explored=r["level_explored"],
                            level_total=r["level_total"], expected_minimum=best, expected_multiplicity=mult)
                        if r["status"] == "found" and (r["minimum"], r["count"]) != (best, mult):
                            raise AssertionError("search result differs from the enumeration")
                        if r["status"] == "incomplete":
                            lo, hi = r["min_size"]
                            if not (lo <= best and (hi is None or best <= hi)):
                                raise AssertionError("certified size bounds do not contain the minimum")
                            if r["multiplicity"] and not (r["multiplicity"][0] <= mult and
                                                          (r["multiplicity"][1] is None or mult <= r["multiplicity"][1])):
                                raise AssertionError("certified multiplicity bounds do not contain the count")
            g.close()
    bench.note("search: incomplete cases are retained with their certified size / multiplicity bounds and explored "
               "states; every bound contains the enumerated answer, every complete answer equals it")


def bounded(bench, rng):
    st = mp.Store()
    for semiring, nr in (("count", 14 if bench.quick else 18), ("bool", 6 if bench.quick else 7)):
        R = st.source(f"R{semiring}", {"k": [int(x) for x in rng.integers(0, 4, nr)], "a": list(range(nr))})
        S = st.source(f"S{semiring}", {"k": [int(x) for x in rng.integers(0, 4, nr)], "b": list(range(nr))})
        J = R.join(S, on=[("k", "k")]).group([])
        g = ma.Game.provenance(J, semiring=semiring)
        n = g.n
        # the exact reference: the sparse support in N (one term per join pair); every coalition in B (the
        # existence game's inclusion-exclusion over many witnesses is refused by design)
        exact = (ma.shapley(g) if semiring == "count" else ma.shapley(g, method="exhaustive"))["values"]
        for budget in (n - 1, n, 2 * n, n + math.comb(n, 2), 2 * n + math.comb(n, 2), 3 * n + math.comb(n, 2)):
            for backend in bench.backends():
                e = bench.measure("bounded Shapley / Banzhaf", backend,
                                  lambda b=backend, m=budget: ma.explore(g, m, backend=b), repeats=2, players=n,
                                  semiring=semiring, max_coalitions=budget)
                width = max(hi - lo for lo, hi in e["bounds"]["shapley"])
                bench.records[-1].update(status=e["status"], regime=e["regime"], explored=e["explored"],
                                         levels=e["levels"], max_shapley_width=frac(width))
                if any(not (lo <= x <= hi) for (lo, hi), x in zip(e["bounds"]["shapley"], exact)):
                    raise AssertionError("a certified bound misses the exact value")
        g.close()


def main():
    bench = Bench.from_args("ATTR-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    conditions(bench, rng)
    provenance(bench, rng)
    search(bench, rng)
    bounded(bench, rng)
    bench.write()


if __name__ == "__main__":
    main()
