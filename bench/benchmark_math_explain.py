# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROV-002 benchmark: deletion propagation, explanations and probabilities over retained provenance, against
complete query reconstruction.

    python bench/benchmark_math_explain.py --capability PROV-002 --out FILE [--device] [--quick]

Case 1 (shared versus independent derivations, duplicate multiplicities): R(k, a) and T(k, c) with 200 / 1,000 /
4,000 rows (quick: 200 / 1,000), every distinct row repeated 1 / 4 / 16 times; the query grouped then joined (each
per-key sum shared by every output: shared derivations) against joined then grouped (one product per pair:
independent derivations). Recorded: PROV-001's DAG nodes, child entries and annotation bytes, the explanation state
(build time, bytes), the propagation of 100 single-source deletions (time, mean nodes recomputed against the cone), the
exact answer probability of the largest group (decision-diagram nodes and time) and its certified bounds.
Case 2 (requests against reconstruction): 1 / 100 / 1,000 source-deletion requests (single tuples, and sets of 4)
propagated in one batch (CPU; CUDA with --device) against PROV-001's evaluation of the whole DAG per request and the
complete reconstruction of the query per request (a new native store without the rows; DuckDB re-executing the query,
for up to 100 requests), checked equal; 1 / 100 / 1,000 explanation requests (minimal sufficient sets, counterfactual
sources, why-not restorations under a base with absent tuples) against PROV-001's polynomial expansion (minimal
witnesses). Correlated derivations: a self-join's exact answer probabilities against the enumeration of every world
(tiny) and the naive product of per-derivation probabilities (which differs). Annotation-budget overflow: PROV-001's
store limit, and the lineage diagram of sum_i r_i s_i under the natural order past a 2^14-node budget (the refusal,
and "auto" returning certified bounds instead), with the largest admitted and smallest refused sizes.
"""
from fractions import Fraction as Fr

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_explain as mx, math_prov as mp
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError


def tables(rng, n, dup):
    base = max(1, n // dup)
    rk, ra = rng.integers(0, 16, base), rng.integers(0, 50, base)
    tk, tc = rng.integers(0, 16, base), rng.integers(0, 8, base)
    rep = lambda x: [int(v) for v in np.repeat(x, dup)][:n]   # noqa: E731
    return dict(k=rep(rk), a=rep(ra)), dict(k=rep(tk), c=rep(tc))


def query_shared(R, T):
    return R.group(["k"]).join(T.group(["k", "c"]), on=[("k", "k")]).group(["k_r", "c"])


def query_independent(R, T):
    return R.join(T, on=[("k", "k")]).group(["k_r", "c"])


def bag(con, sql):
    return {tuple(int(v) for v in row[:-1]): int(row[-1]) for row in con.execute(sql).fetchall()}


def factorization(bench, rng):
    sizes = (200, 1000) if bench.quick else (200, 1000, 4000)
    for n in sizes:
        for dup in (1, 4, 16):
            rcols, tcols = tables(rng, n, dup)
            for label, q in (("shared derivations (group, then join)", query_shared),
                             ("independent derivations (join, then group)", query_independent)):
                p = dict(rows=n, duplicates=dup, derivations=label.split(" (")[0])
                st = mp.Store(budget=Budget(circuit=1 << 22, support=1 << 23))
                R, T = st.source("R", rcols), st.source("T", tcols)
                G = q(R, T)
                info = st.info
                ex = bench.measure(f"explanation state: {label}", "cpu", lambda: mx.Explainer(st, [G]), repeats=2,
                                   phase="build", **p)
                bench.records[-1].update(dag_nodes=info["nodes"], child_entries=info["children"],
                                         shared_nodes=info["shared"], annotation_bytes=info["bytes"],
                                         cone_nodes=ex.info["nodes"], state_bytes=ex.info["bytes"], answers=len(G))
                nr, nt = len(rcols["k"]), len(tcols["k"])
                labels = [("R", int(i)) for i in rng.integers(0, nr, 50)] + [("T", int(i)) for i in rng.integers(0, nt, 50)]
                d = bench.measure("100 single-source deletions propagated", "cpu",
                                  lambda: ex.delete([[lab] for lab in labels]), repeats=3, phase="first+warm", **p)
                bench.records[-1].update(mean_affected=float(np.mean(d.affected)), cone_nodes=ex.info["nodes"])
                j = int(np.argmax(ex.baseline[0]))
                probs = {lab: Fr(1, 2) for lab in ex.labels}
                try:
                    pr = bench.measure("exact probability of the largest answer", "cpu",
                                       lambda: ex.probability(j, probs, budget=Budget(circuit=1 << 18)), repeats=2, **p)
                    bench.records[-1].update(diagram_nodes=pr.diagram["nodes"], levels=pr.diagram["levels"],
                                             peak_nodes=pr.diagram["peak_nodes"])
                except AlgebraBudgetError:
                    bench.refusal("exact probability of the largest answer past 2^18 diagram nodes", "cpu",
                                  lambda: ex.probability(j, probs, budget=Budget(circuit=1 << 18)), **p)
                bd = bench.measure("certified probability bounds of the largest answer", "cpu",
                                   lambda: ex.probability(j, probs, method="bounds"), repeats=2, **p)
                bench.records[-1].update(lo=bd.bounds[0], hi=bd.bounds[1], read_once=bd.read_once)


def requests(bench, rng):
    n = 400 if bench.quick else 1000
    rcols, tcols = tables(rng, n, 2)
    st = mp.Store()
    R, T = st.source("R", rcols), st.source("T", tcols)
    G = query_shared(R, T)
    ex = bench.measure("explanation state (cone, reverse edges, baseline)", "cpu", lambda: mx.Explainer(st, [G]),
                       repeats=3, phase="build")
    bench.records[-1].update(cone_nodes=ex.info["nodes"], answers=ex.info["outputs"])
    plan = st.plan(G)
    labels = [("R", i) for i in range(n)] + [("T", i) for i in range(n)]
    con = duckdb.connect()
    con.register("R_all", pd.DataFrame(dict(rid=range(n), **rcols)))
    con.register("T_all", pd.DataFrame(dict(rid=range(n), **tcols)))
    tuples = G.tuples()
    for m in (1, 100, 1000):
        for size in (1, 4):
            dels = [[labels[int(i)] for i in rng.choice(len(labels), size, replace=False)] for _ in range(m)]
            p = dict(requests=m, tuples_per_request=size, rows=n)
            for backend in bench.backends():
                prop = bench.measure("deletion requests propagated in one batch (N and B)", backend,
                                     lambda: ex.delete(dels, backend=backend), repeats=3, phase="first+warm", **p)
            bench.records[-1].update(mean_affected=float(np.mean(prop.affected)), cone_nodes=ex.info["nodes"])
            full = bench.measure("PROV-001: the whole DAG evaluated per request", "cpu",
                                 lambda: st.evaluate(G, "count", deletions=dels, plan=plan), repeats=3,
                                 phase="first+warm", **p)
            if full.values != prop.counts:
                raise SystemExit("PROV-002 benchmark: propagation differs from the whole-DAG evaluation")

            def native_rebuild(dels=dels):
                res = []
                for d in dels:
                    gone = set(d)
                    rc = {c: [v for i, v in enumerate(vs) if ("R", i) not in gone] for c, vs in rcols.items()}
                    tc = {c: [v for i, v in enumerate(vs) if ("T", i) not in gone] for c, vs in tcols.items()}
                    s2 = mp.Store()
                    g2 = query_shared(s2.source("R", rc), s2.source("T", tc))
                    res.append(dict(zip(g2.tuples(), g2.counts())))
                return res
            k = min(m, 100)
            rebuilt = bench.measure("complete reconstruction per request (native)", "reconstruction",
                                    lambda: native_rebuild(dels[:k]), repeats=1, reconstructed=k, **p)
            for w in range(k):
                if {tp: c for tp, c in zip(tuples, prop.counts[w]) if c} != rebuilt[w]:
                    raise SystemExit("PROV-002 benchmark: a propagation differs from the reconstructed query")
            if m <= 100:
                def duck(dels=dels):
                    res = []
                    for d in dels:
                        rr = ", ".join(str(i) for s, i in d if s == "R") or "-1"
                        tr = ", ".join(str(i) for s, i in d if s == "T") or "-1"
                        res.append(bag(con, f"SELECT t.k, t.c, COUNT(*) FROM R_all r JOIN T_all t ON r.k = t.k "
                                            f"WHERE r.rid NOT IN ({rr}) AND t.rid NOT IN ({tr}) GROUP BY ALL"))
                    return res
                dk = bench.measure("query re-executed per request", "duckdb", duck, repeats=1, **p)
                if dk != rebuilt[:m]:
                    raise SystemExit("PROV-002 benchmark: DuckDB's re-execution differs")
    # explanation requests: minimal sufficient sets, counterfactual sources, why-not under a base
    base = {lab: 0 for lab in labels[:n:7]}                    # every 7th R tuple and every T tuple with c = 0 absent
    base.update({("T", i): 0 for i, c in enumerate(tcols["c"]) if c == 0})
    eb = mx.Explainer(st, [G], base=base)
    absent = [j for j, e in enumerate(eb.baseline[1]) if not e]
    for m in (1, 100, 1000):
        idx = [int(i) for i in rng.integers(0, len(G), min(m, len(G)))]
        p = dict(requests=len(idx), rows=n)
        bench.measure("explanations: minimal sufficient sets", "cpu", lambda: [ex.sufficient(j) for j in idx],
                      repeats=1, **p)
        bench.measure("explanations: counterfactual sources", "cpu", lambda: [ex.necessary(j) for j in idx], repeats=1,
                      **p)
        bench.measure("PROV-001: minimal witnesses by expansion", "cpu", lambda: [G.witnesses(j) for j in idx],
                      repeats=1, **p)
        if absent:
            wn = [absent[i % len(absent)] for i in range(len(idx))]
            bench.measure("explanations: why-not restorations (base with absent tuples)", "cpu",
                          lambda: [eb.why_not(j) for j in wn], repeats=1, absent_answers=len(absent), **p)
    for j in idx[:20]:
        if set(ex.sufficient(j).sets) != set(G.witnesses(j).sets):
            raise SystemExit("PROV-002 benchmark: the minimal sufficient sets differ from PROV-001's witnesses")
    con.close()


def correlated(bench, rng):
    rrows = [(int(k), int(a)) for k, a in zip(rng.integers(0, 3, 6), rng.integers(0, 3, 6))]
    srows = [(int(k), int(b)) for k, b in zip(rng.integers(0, 3, 6), rng.integers(0, 3, 6))]
    st = mp.Store()
    R = st.source("R", dict(k=[x[0] for x in rrows], a=[x[1] for x in rrows]))
    S = st.source("S", dict(k=[x[0] for x in srows], b=[x[1] for x in srows]))
    G = R.join(S, on=[("k", "k")]).group(["b"])          # an R tuple joins several S tuples: shared variables
    ex = mx.Explainer(st, [G])
    labels = [("R", i) for i in range(6)] + [("S", i) for i in range(6)]
    probs = {lab: Fr(1 + i % 5, 7) for i, lab in enumerate(labels)}
    for j, (b,) in enumerate(G.tuples()):
        exact = bench.measure("correlated lineage: exact probability", "cpu", lambda: ex.probability(j, probs),
                              repeats=2, answer=j)
        bench.records[-1].update(diagram_nodes=exact.diagram["nodes"], levels=exact.diagram["levels"])

        def worlds(b=b):
            total = Fr(0)
            for bits in range(1 << 12):
                rr = [r for i, r in enumerate(rrows) if (bits >> i) & 1]
                ss = [x for i, x in enumerate(srows) if (bits >> (6 + i)) & 1]
                if any(r[0] == x[0] and x[1] == b for r in rr for x in ss):
                    pw = Fr(1)
                    for i, lab in enumerate(labels):
                        pw *= probs[lab] if (bits >> i) & 1 else 1 - probs[lab]
                    total += pw
            return total
        want = bench.measure("correlated lineage: enumeration of the 2^12 worlds", "python", worlds, repeats=1, answer=j)
        if want != exact.value:
            raise SystemExit("PROV-002 benchmark: the exact probability differs from the world enumeration")
        mono = G.polynomial(j).terms
        naive = 1 - np.prod([1 - np.prod([float(probs[lab]) for lab in set(m)]) for m in mono])
        bench.records[-1].update(exact=float(exact.value), naive_independent_derivations=float(naive),
                                 derivations=len(mono))


def overflow(bench, rng):
    limit = 1 << 14
    admitted = refused = None
    for n in (64, 128, 256, 512, 1024, 2048):
        st = mp.Store(budget=Budget(circuit=limit))
        R = st.source("R", dict(k=[0] * n, a=list(range(n))))
        T = st.source("T", dict(k=[0] * 16, c=list(range(16))))
        try:
            J = R.join(T, on=[("k", "k")])
            admitted = dict(rows=n, pairs=len(J), nodes=st.info["nodes"])
        except AlgebraBudgetError:
            bench.refusal("PROV-001: join past the annotation limit (2^14 nodes)", "cpu",
                          lambda: R.join(T, on=[("k", "k")]), rows=n, pairs=16 * n)
            refused = dict(rows=n, pairs=16 * n)
            break
    bench.threshold("annotation store: n x 16 matching tuples under 2^14 DAG nodes", admitted, refused, limit=limit)
    limit = 1 << 14
    admitted = refused = None
    for n in (4, 6, 8, 10, 12, 14):
        st = mp.Store()
        R = st.source("R", {"k": list(range(n)), "g": [0] * n})
        S = st.source("S", {"k": list(range(n))})
        P = R.join(S, on=[("k", "k")]).group(["g"])
        ex = mx.Explainer(st, [P])
        probs = {lab: Fr(1, 3) for lab in ex.labels}
        good = bench.measure("lineage sum r_i s_i, dfs order: exact probability", "cpu",
                             lambda: ex.probability(0, probs, budget=Budget(circuit=limit)), repeats=2, n=n)
        bench.records[-1].update(diagram_nodes=good.diagram["nodes"])
        try:
            bad = bench.measure("lineage sum r_i s_i, natural order: exact probability", "cpu",
                                lambda: ex.probability(0, probs, order="natural", budget=Budget(circuit=limit)),
                                repeats=1, n=n)
            bench.records[-1].update(diagram_nodes=bad.diagram["nodes"], peak_nodes=bad.diagram["peak_nodes"])
            admitted = dict(n=n, nodes=bad.diagram["nodes"], peak_nodes=bad.diagram["peak_nodes"])
        except AlgebraBudgetError:
            bench.refusal("lineage sum r_i s_i, natural order past 2^14 diagram nodes", "cpu",
                          lambda: ex.probability(0, probs, order="natural", budget=Budget(circuit=limit)), n=n)
            refused = refused or dict(n=n)
            auto = bench.measure("method auto: certified bounds after the refusal", "cpu",
                                 lambda: ex.probability(0, probs, order="natural", method="auto",
                                                        budget=Budget(circuit=limit)), repeats=1, n=n)
            if not auto.bounds[0] <= good.value <= auto.bounds[1]:
                raise SystemExit("PROV-002 benchmark: the certified bounds miss the exact probability")
            bench.records[-1].update(lo=auto.bounds[0], hi=auto.bounds[1], exact=float(good.value),
                                     refused_resource=auto.refusal["resource"])
    bench.threshold("lineage diagram of sum r_i s_i under the natural order, 2^14 nodes", admitted, refused, limit=limit)


def main():
    bench = Bench.from_args("PROV-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    factorization(bench, rng)
    requests(bench, rng)
    correlated(bench, rng)
    overflow(bench, rng)
    bench.note("cpu = math_explain.c (incremental propagation over the retained DAG; explanations and probabilities on "
               "COUNT-002 decision diagrams); cuda = math_explain_gpu.cu (one work item per change set; device runs "
               "only); reconstruction = a new native store per request; duckdb = the query re-executed; PROV-001 = "
               "the whole DAG evaluated per request.")
    bench.note("The naive product of per-derivation probabilities is recorded beside the exact value of correlated "
               "lineages to show the difference; it is never returned by the capability.")
    bench.write()


if __name__ == "__main__":
    main()
