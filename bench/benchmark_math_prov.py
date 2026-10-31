# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROV-001 benchmark: factorized provenance of small joins and groups (the shared DAG, its size and bytes),
evaluations under many valuations (source deletions, multiplicities) in N and B, explanations (N[X] expansions,
minimal witnesses), against complete query reconstruction.

    python bench/benchmark_math_prov.py --capability PROV-001 --out FILE [--device] [--quick]

Factorization: a group-then-join query whose per-key sums are shared by every output (shared derivations) against
the same multiplicities computed by joining first and grouping after (independent derivations: one product per
pair), for 200 / 1,000 / 4,000 source rows (quick: 200 / 1,000) and duplicate multiplicities 1 / 4 / 16 (each
distinct row repeated: its tuple's annotation sums the copies); recorded: DAG nodes, child entries, shared nodes,
store bytes, the expanded size where it is small, build and evaluation times; baseline: DuckDB's bag multiplicities
of the same query (COUNT(*) GROUP BY ALL), checked equal.
Requests: 1 / 100 / 1,000 single-source deletions evaluated in one batch (N and B; CUDA with --device) against the
complete reconstruction of the query per deletion (natively: a new store without that row; DuckDB: the query
re-executed with the row filtered out, for up to 100 deletions), checked equal; explanation requests (N[X]
expansions and minimal witnesses of 1 / 100 / 1,000 output tuples, as many as exist). Correlated derivations: a
self-join (squared variables) under multiplicity valuations against DuckDB on duplicated rows. Annotation-budget
overflow: the largest admitted and the smallest refused join under a node limit (refusals recorded, never
extrapolated).
"""
import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_prov as mp
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError


def tables(rng, n, dup):
    """R(k, a) and T(k, c): n rows each, every distinct row repeated dup times"""
    base = max(1, n // dup)
    rk, ra = rng.integers(0, 16, base), rng.integers(0, 50, base)
    tk, tc = rng.integers(0, 16, base), rng.integers(0, 8, base)
    rep = lambda x: [int(v) for v in np.repeat(x, dup)][:n]   # noqa: E731
    return dict(k=rep(rk), a=rep(ra)), dict(k=rep(tk), c=rep(tc))


def query_shared(st, R, T):
    return R.group(["k"]).join(T.group(["k", "c"]), on=[("k", "k")]).group(["k_r", "c"])


def query_independent(st, R, T):
    return R.join(T, on=[("k", "k")]).group(["k_r", "c"])


def bag(con, sql):
    return {tuple(int(v) for v in row[:-1]): int(row[-1]) for row in con.execute(sql).fetchall()}


def factorization(bench, rng):
    sizes = (200, 1000) if bench.quick else (200, 1000, 4000)
    for n in sizes:
        for dup in (1, 4, 16):
            rcols, tcols = tables(rng, n, dup)
            con = duckdb.connect()
            con.register("R", pd.DataFrame(rcols))
            con.register("T", pd.DataFrame(tcols))
            want = dict(bench.measure("bag multiplicities (GROUP BY)", "duckdb", lambda: list(bag(
                con, "SELECT t.k, t.c, COUNT(*) FROM R r JOIN T t ON r.k = t.k GROUP BY ALL").items()), repeats=2,
                rows=n, duplicates=dup))
            for label, q in (("shared derivations (group, then join)", query_shared),
                             ("independent derivations (join, then group)", query_independent)):
                p = dict(rows=n, duplicates=dup, derivations=label.split(" (")[0])

                def build(q=q):
                    st = mp.Store(budget=Budget(circuit=1 << 22, support=1 << 23))
                    R = st.source("R", rcols)
                    T = st.source("T", tcols)
                    return st, q(st, R, T)
                st, G = bench.measure(f"annotated query: {label}", "cpu", build, repeats=2, phase="build", **p)
                info = st.info
                bench.records[-1].update(dag_nodes=info["nodes"], child_entries=info["children"],
                                         shared_nodes=info["shared"], hash_hits=info["hits"],
                                         annotation_bytes=info["bytes"], output_tuples=len(G))
                counts = bench.measure("multiplicities (N at 1)", "cpu", lambda: G.counts(), repeats=3,
                                       phase="first+warm", **p)
                if dict(zip(G.tuples(), counts)) != want:
                    raise SystemExit(f"PROV-001: multiplicities differ from DuckDB at {p}")
                bench.measure("existence (B at 1)", "cpu", lambda: G.exists(), repeats=3, phase="first+warm", **p)
                if n <= 200:
                    exp = bench.measure("expanded size (N[X] of every output)", "cpu",
                                        lambda: sum(len(G.polynomial(i).terms) for i in range(len(G))), repeats=1, **p)
                    bench.records[-1].update(expanded_monomials=exp)
            con.close()


def requests(bench, rng):
    n = 400 if bench.quick else 1000
    rcols, tcols = tables(rng, n, 2)
    st = mp.Store()
    R, T = st.source("R", rcols), st.source("T", tcols)
    G = query_shared(st, R, T)
    plan = bench.measure("evaluation plan (the cone)", "cpu", lambda: st.plan(G), repeats=3, phase="build")
    bench.records[-1].update(cone_nodes=plan.info["nodes"], outputs=plan.info["outputs"])
    labels = [("R", i) for i in range(n)] + [("T", i) for i in range(n)]
    con = duckdb.connect()
    con.register("R_all", pd.DataFrame(dict(rid=range(n), **rcols)))
    con.register("T_all", pd.DataFrame(dict(rid=range(n), **tcols)))
    for m in (1, 100, 1000):
        dels = [[labels[int(i)]] for i in rng.integers(0, len(labels), m)]
        p = dict(requests=m, rows=n)
        for backend in bench.backends():
            for semiring in ("count", "bool"):
                out = bench.measure(f"source deletions in one batch ({semiring})", backend,
                                    lambda: st.evaluate(G, semiring, deletions=dels, plan=plan, backend=backend),
                                    repeats=3, phase="first+warm", **p)
                if semiring == "count":
                    batch = out

        def native_rebuild(dels=dels):
            res = []
            for (src, row), in dels:
                rc = {c: [v for i, v in enumerate(vs) if not (src == "R" and i == row)] for c, vs in rcols.items()}
                tc = {c: [v for i, v in enumerate(vs) if not (src == "T" and i == row)] for c, vs in tcols.items()}
                s2 = mp.Store()
                g2 = query_shared(s2, s2.source("R", rc), s2.source("T", tc))
                res.append(dict(zip(g2.tuples(), g2.counts())))
            return res
        k = m if m <= 100 or not bench.quick else 100
        rebuilt = bench.measure("complete reconstruction per deletion (native)", "reconstruction",
                                lambda: native_rebuild(dels[:k]), repeats=1, reconstructed=k, **p)
        tuples = G.tuples()
        for w in range(k):
            if {tp: c for tp, c in zip(tuples, batch.values[w]) if c} != rebuilt[w]:
                raise SystemExit("PROV-001: a deletion differs from the reconstructed query")
        if m <= 100:
            def duck(dels=dels):
                res = []
                for (src, row), in dels:
                    rr, tr = (row if src == "R" else -1), (row if src == "T" else -1)
                    res.append(bag(con, f"SELECT t.k, t.c, COUNT(*) FROM R_all r JOIN T_all t ON r.k = t.k WHERE "
                                        f"r.rid <> {rr} AND t.rid <> {tr} GROUP BY ALL"))
                return res
            dk = bench.measure("query re-executed per deletion", "duckdb", duck, repeats=1, **p)
            if dk != rebuilt[:m]:
                raise SystemExit("PROV-001: DuckDB's re-execution differs")
    # explanations: N[X] expansions and minimal witnesses of output tuples
    for m in (1, 100, 1000):
        idx = [int(i) for i in rng.integers(0, len(G), min(m, len(G)))]
        p = dict(requests=len(idx), rows=n)
        bench.measure("explanations: N[X] expansions", "cpu", lambda: [G.polynomial(i) for i in idx], repeats=1, **p)
        bench.measure("explanations: minimal witnesses (B)", "cpu", lambda: [G.witnesses(i) for i in idx], repeats=1,
                      **p)
    con.close()


def correlated(bench, rng):
    rows = [(int(k), int(a)) for k, a in zip(rng.integers(0, 10, 300), rng.integers(0, 30, 300))]
    st = mp.Store()
    R = st.source("R", dict(k=[x[0] for x in rows], a=[x[1] for x in rows]))
    SJ = bench.measure("self-join (correlated derivations: squared variables)", "cpu",
                       lambda: R.join(R, on=[("k", "k")]).group(["k"]), repeats=2, phase="build", rows=300)
    vals = [{("R", int(i)): 2 for i in rng.integers(0, 300, 5)} for _ in range(100)]
    for backend in bench.backends():
        got = bench.measure("multiplicity valuations (N)", backend,
                            lambda: st.evaluate(SJ, "count", valuations=vals, backend=backend), repeats=3,
                            phase="first+warm", valuations=100)
    con = duckdb.connect()
    w = vals[0]
    dup = rows + [rows[i] for (_, i) in w]
    con.register("D", pd.DataFrame(dict(k=[x[0] for x in dup], a=[x[1] for x in dup])))
    want = dict(bench.measure("self-join on duplicated rows", "duckdb", lambda: list(bag(
        con, "SELECT a.k, COUNT(*) FROM D a JOIN D b ON a.k = b.k GROUP BY 1").items()), repeats=2))
    if dict(zip(SJ.tuples(), got.values[0])) != want:
        raise SystemExit("PROV-001: the correlated multiplicities differ from DuckDB")
    con.close()


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
            bench.refusal("join past the annotation limit (2^14 nodes)", "cpu",
                          lambda: R.join(T, on=[("k", "k")]), rows=n, pairs=16 * n)
            refused = dict(rows=n, pairs=16 * n)
            break
    bench.threshold("join of n x 16 matching tuples under 2^14 DAG nodes", admitted, refused, limit=limit)
    st = mp.Store()
    R = st.source("R", dict(k=[0] * 40, a=list(range(40))))
    deep = R.group(["k"])
    for _ in range(6):
        deep = deep.join(R.group(["k"]), on=[("k", "k")]).group(["k"])
    bench.refusal("expansion past the support budget (40^7 derivations, a factored DAG of a few nodes)", "cpu",
                  lambda: deep.polynomial(0), dag_nodes=st.info["nodes"])
    bench.measure("the same annotation evaluated factored (N at 1)", "cpu", lambda: deep.counts(), repeats=3,
                  dag_nodes=st.info["nodes"])


def main():
    bench = Bench.from_args("PROV-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    factorization(bench, rng)
    requests(bench, rng)
    correlated(bench, rng)
    overflow(bench, rng)
    bench.note("cpu = math_prov.c (hash-consed DAG, K-relation operators, one valuation at a time); cuda = "
               "math_prov_gpu.cu (one work item per valuation; device runs only); duckdb = the query re-executed "
               "(bag multiplicities by GROUP BY ALL / COUNT(*)); reconstruction = a new native store per deletion.")
    bench.note("Annotation limits are the store's (nodes, child entries, bytes); an operator refuses before adding a "
               "node when its bound does not fit. An expansion is budgeted by its number of derivations before it "
               "starts; the factored DAG is evaluated without expanding.")
    bench.write()


if __name__ == "__main__":
    main()
