# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""COUNT-002 benchmark: symbolic counting over the declared assignment universe, against the observed snapshot.

    python bench/benchmark_math_symcount.py --capability COUNT-002 --out FILE [--device] [--quick]

Case 1 (observed snapshot versus universe): 1k / 100k / 1M rows (1k / 100k with --quick) of a table whose k = 8
conditions are structured (bits 0..7 of an integer column) or irregular (random thresholds on two columns with NULLs),
with a structured predicate (an AND / OR / XOR circuit of the bits, also written in SQL) and an irregular one (a
random predicate DAG). Measured: COUNT-001's observed counts and 16 smallest violating rows (the domain built once,
counts first and warm), DuckDB's COUNT(*) WHERE on the same rows for the structured predicate, COUNT-001's enumeration
of the 2^8 universe, and COUNT-002's symbolic count (compilation, count, 16 witnesses). Recorded per point: the
observed TRUE count against the universe's TRUE count and the coverage (distinct observed assignments of 2^8): two
different quantities, never substituted for one another.
Case 2 (symbolic counts): tiny formulas (k = 4 / 8 / 12) checked against the native brute force and, for k <= 8, a
Python enumeration; bounded larger circuits with variable-order controls: OR_i (x_i AND x_(i+n)) for n = 4 .. 20
(k <= 40) under the interleaved order (2n + 2 nodes) and the natural one (2^(n+1)), compile time, nodes, peak nodes,
apply steps, and the refusals past a 2^16-node circuit budget (largest admitted and smallest refused n, never
extrapolated); "dfs" against "natural" orders on random circuits; exact weighted counts with rational literal weights
(CPU; the level kernels with --device); variable elimination on w x w grids of binary variables with rational pairwise
factors (min-fill against natural widths, the refusal past a 2^12 tensor budget, the factor brute force on the 3 x 3
grid); the bounded Shannon expansion's certified bounds as its evaluation budget grows.
"""
import random
from fractions import Fraction as Fr

import numpy as np
import pandas as pd
import duckdb

from math_bench import Bench
import torbles
from torbles import math_bank as mbk, math_bool as mb, math_count as mc, math_symcount as ms
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError


def random_nodes(rng, k, n):
    """a random circuit that reads its variables deeply: operands mostly among the latest nodes, every variable used"""
    nodes = [("var", i, -1) for i in range(k)]

    def pick():
        return len(nodes) - 1 - min(int(rng.expovariate(0.25)), len(nodes) - 1)
    for j in range(n):
        if rng.random() < 0.15:
            nodes.append(("not", pick(), -1))
        else:
            a = j % k if j < k else pick()                       # the first k gates read each variable once
            nodes.append((rng.choice(["and", "or", "xor", "and", "or"]), a, pick()))
    return nodes


def random_tree(rng, k, leaves):
    """a random formula: literals covering every variable combined pairwise by random connectives into one root,
    a combined operand sometimes kept for reuse (shared subformulas)"""
    x = mb.variables(k)
    items = [x[v] if rng.random() < 0.6 else ~x[v] for v in list(range(k)) + [rng.randrange(k) for _ in range(leaves - k)]]
    rng.shuffle(items)
    while len(items) > 1:
        a = items.pop(rng.randrange(len(items)))
        b = items.pop(rng.randrange(len(items)))
        e = {0: a & b, 1: a | b, 2: a ^ b}[rng.randrange(3)]
        items.append(e)
        if rng.random() < 0.1 and len(items) > 1:
            items.append(a)
    return items[0]


def random_cnf(rng, k, m):
    """a random 3-CNF over k variables with m clauses (an expression)"""
    x = mb.variables(k)
    f = mb.TRUE
    for _ in range(m):
        c = mb.FALSE
        for v in rng.sample(range(k), 3):
            c = c | (x[v] if rng.random() < 0.5 else ~x[v])
        f = f & c
    return f


def python_count(nodes, k):
    true = 0
    for a in range(1 << k):
        val = []
        for op, x, y in nodes:
            if op == "var":
                val.append((a >> x) & 1)
            elif op == "not":
                val.append(1 - val[x])
            elif op == "and":
                val.append(val[x] & val[y])
            elif op == "or":
                val.append(val[x] | val[y])
            else:
                val.append(val[x] ^ val[y])
        true += val[-1]
    return true


def observed_universe(bench, rng):
    sizes = (1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)
    b = mb.variables(8)
    structured = ((b[0] & b[1]) | (b[2] ^ b[3]) | (b[4] & ~b[5])) & (b[6] | b[7])
    sql_structured = ("((((x >> 0) & 1) = 1 AND ((x >> 1) & 1) = 1) OR (((x >> 2) & 1) <> ((x >> 3) & 1)) OR "
                      "(((x >> 4) & 1) = 1 AND ((x >> 5) & 1) = 0)) AND (((x >> 6) & 1) = 1 OR ((x >> 7) & 1) = 1)")
    pick = random.Random(5)
    irregular = random_tree(pick, 8, 24)
    while ms.compile(irregular, k=8).count().true in (0, 256):          # a predicate that is not constant
        irregular = random_tree(pick, 8, 24)
    for n in sizes:
        x = rng.integers(0, 1 << 8, n)
        x[rng.random(n) < 0.5] &= 0x3F                                  # bits 6 and 7 mostly 0: a skewed snapshot
        y = rng.integers(-50, 51, n).astype(object)
        y[rng.random(n) < 0.1] = None
        frame = pd.DataFrame({"x": [int(v) for v in x], "y": pd.Series(list(y), dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x", "y"])
        thresholds = [int(v) for v in rng.integers(-40, 41, 4)]
        for label, conds, f in (
                ("structured (bit planes, AND / OR / XOR circuit)", [mb.condition("x", "bit", i) for i in range(8)],
                 structured),
                ("irregular (thresholds with NULLs, random DAG)",
                 [mb.condition("x", ">", int(t) + 128) for t in thresholds] + [mb.condition("y", "<", t) for t in thresholds],
                 irregular)):
            p = dict(rows=n, predicate=label.split(" (")[0])
            dom = bench.measure("observed domain (COUNT-001: condition bitmaps, FND-007 record)", "cpu",
                                lambda: mc.Domain.observe(scope, conds), repeats=2, phase="build", **p)
            obs = bench.measure("observed counts + 16 smallest violating rows (COUNT-001)", "cpu",
                                lambda: mc.predicate(dom, f, witnesses=16), repeats=3, phase="first+warm", **p)
            res, cert = obs.results[0], obs.certificate
            if label.startswith("structured"):
                con = duckdb.connect()
                con.register("t", frame)
                got = bench.measure("observed TRUE count (COUNT(*) WHERE)", "duckdb",
                                    lambda: con.execute(f"SELECT COUNT(*) FROM t WHERE {sql_structured}").fetchone()[0],
                                    repeats=3, **p)
                con.close()
                if int(got) != res["satisfied"]:
                    raise SystemExit("COUNT-002 benchmark: DuckDB's observed count differs from COUNT-001's")
            uni = bench.measure("universe by enumeration of the 2^8 assignments (COUNT-001)", "cpu",
                                lambda: mc.predicate(mc.Domain.universe(8), f).results[0], repeats=3, **p)
            d = bench.measure("symbolic: compile the predicate (COUNT-002)", "cpu", lambda: ms.compile(f, k=8),
                              repeats=3, phase="build", **p)
            for backend in bench.backends():
                sym = bench.measure("symbolic: universe count (COUNT-002)", backend,
                                    lambda: d.count(backend=backend), repeats=3, phase="first+warm", **p)
            wit = bench.measure("symbolic: 16 witnesses of the universe (COUNT-002)", "cpu", lambda: d.witnesses(16),
                                repeats=3, **p)
            if (sym.true, sym.false, sym.unknown) != (uni["satisfied"], uni["violated"], uni["unknown"]):
                raise SystemExit("COUNT-002 benchmark: the symbolic universe count differs from the enumeration")
            bench.records[-1].update(observed_rows=res["points"], observed_true=res["satisfied"],
                                     observed_false=res["violated"], observed_unknown=res["unknown"],
                                     universe_assignments=sym.assignments, universe_true=sym.true,
                                     universe_false=sym.false, distinct_observed_assignments=cert["observed"],
                                     unobserved_assignments=cert["unobserved"], diagram_nodes=d.info["nodes"],
                                     witnesses=len(wit))
        bench.note(f"{n} rows: the observed snapshot counts rows, the universe counts the 2^8 assignments; with bits 6"
                   " and 7 skewed the observed TRUE fraction differs from the universe's.")


def symbolic(bench, rng):
    prng = random.Random(bench.seed)
    for k in (4, 8, 12):
        nodes = random_nodes(prng, k, 4 * k)
        while len({python_count(nodes, k)} - {0, 1 << k}) == 0:   # a non-constant formula
            nodes = random_nodes(prng, k, 4 * k)
        prog = mb.Program.from_nodes(k, nodes)
        p = dict(k=k, nodes=len(nodes))
        d = bench.measure("tiny formula: compile", "cpu", lambda: ms.compile(prog, order="dfs"), repeats=3,
                          phase="build", **p)
        c = bench.measure("tiny formula: symbolic count", "cpu", lambda: d.count(), repeats=3, **p)
        bf = bench.measure("tiny formula: native brute force", "cpu", lambda: ms.brute(prog), repeats=3, **p)
        if (bf.true, bf.false) != (c.true, c.false):
            raise SystemExit("COUNT-002 benchmark: brute force and diagram differ")
        if k <= 8:
            want = bench.measure("tiny formula: Python enumeration", "python", lambda: python_count(nodes, k),
                                 repeats=1, **p)
            if want != c.true:
                raise SystemExit("COUNT-002 benchmark: the Python enumeration differs")
        bench.records[-1].update(true=c.true, diagram_nodes=d.info["nodes"])
    # order controls: the pairs family
    limit = 1 << 16
    admitted = refused = None
    for n in ((4, 8, 12, 14, 16) if bench.quick else (4, 8, 12, 14, 16, 20)):
        x = mb.variables(2 * n)
        f = mb.FALSE
        for i in range(n):
            f = f | (x[i] & x[i + n])
        prog = mb.Program(f, 2 * n)
        inter = [v for i in range(n) for v in (i, i + n)]
        p = dict(n=n, k=2 * n)
        di = bench.measure("pairs family, interleaved order: compile", "cpu",
                           lambda: ms.compile(prog, order=inter, budget=Budget(circuit=limit)), repeats=2,
                           phase="build", order="interleaved", **p)
        bench.records[-1].update(diagram_nodes=di.info["nodes"], peak_nodes=di.info["peak_nodes"],
                                 apply_steps=di.info["steps"], width=di.info["width"])
        try:
            dn = bench.measure("pairs family, natural order: compile", "cpu",
                               lambda: ms.compile(prog, budget=Budget(circuit=limit)), repeats=1, phase="build",
                               order="natural", **p)
            bench.records[-1].update(diagram_nodes=dn.info["nodes"], peak_nodes=dn.info["peak_nodes"],
                                     apply_steps=dn.info["steps"], width=dn.info["width"])
            if dn.count().true != di.count().true:
                raise SystemExit("COUNT-002 benchmark: the orders count differently")
            admitted = dict(n=n, nodes=dn.info["nodes"], peak_nodes=dn.info["peak_nodes"])
        except AlgebraBudgetError:
            bench.refusal("pairs family, natural order past the node budget", "cpu",
                          lambda: ms.compile(prog, budget=Budget(circuit=limit)), order="natural", **p)
            if refused is None:
                refused = dict(n=n, nodes_needed_at_least=limit + 1)
    bench.threshold("pairs family under the natural order, 2^16 diagram nodes", admitted, refused, limit=limit)
    # exact weighted counts of a 40-variable circuit (rational literal weights)
    n = 20
    x = mb.variables(2 * n)
    f = mb.FALSE
    for i in range(n):
        f = f | (x[i] & x[i + n])
    d = ms.compile(f, k=2 * n, order=[v for i in range(n) for v in (i, i + n)])
    w = {v: (Fr(prng.randint(1, 9), 10), Fr(prng.randint(1, 9), 10)) for v in range(2 * n)}
    for backend in bench.backends():
        c = bench.measure("weighted count, 40 variables, rational weights", backend,
                          lambda: d.count(weights=w, backend=backend), repeats=3, phase="first+warm", k=2 * n)
    bench.records[-1].update(limbs=c.info["limbs"], diagram_nodes=d.info["nodes"])
    # dfs against natural on random circuits
    for k in ((16, 24) if bench.quick else (16, 24, 32)):
        prog = mb.Program(random_tree(prng, k, 3 * k), k)
        sizes = {}
        shuffled = list(range(k))
        prng.shuffle(shuffled)
        for order in ("natural", "dfs", shuffled):
            name = order if isinstance(order, str) else "random"
            try:
                dd = bench.measure(f"random formula, {name} order: compile", "cpu",
                                   lambda: ms.compile(prog, order=order, budget=Budget(circuit=1 << 18)), repeats=1,
                                   phase="build", k=k, order=name)
                sizes[name] = dd.count().true
                bench.records[-1].update(diagram_nodes=dd.info["nodes"], peak_nodes=dd.info["peak_nodes"],
                                         apply_steps=dd.info["steps"], true=sizes[name])
            except AlgebraBudgetError:
                bench.refusal(f"random formula, {name} order past 2^18 nodes", "cpu",
                              lambda: ms.compile(prog, order=order, budget=Budget(circuit=1 << 18)), k=k, order=name)
        if len(set(sizes.values())) > 1:
            raise SystemExit("COUNT-002 benchmark: orders disagree on a random formula")
    # variable elimination on grids
    for wdt in ((3, 4, 6, 12) if bench.quick else (3, 4, 6, 8, 12)):
        nv = wdt * wdt
        factors = []
        for r in range(wdt):
            for c in range(wdt):
                v = r * wdt + c
                for u in ([v + 1] if c + 1 < wdt else []) + ([v + wdt] if r + 1 < wdt else []):
                    factors.append(((v, u), [[Fr(prng.randint(1, 5), 3), Fr(prng.randint(0, 5), 4)],
                                             [Fr(prng.randint(0, 5), 2), Fr(prng.randint(1, 5), 5)]]))
        fs = ms.Factors([2] * nv, factors, budget=Budget(variables=1 << 10))
        p = dict(grid=f"{wdt}x{wdt}", variables=nv)
        shuffled = list(range(nv))
        prng.shuffle(shuffled)
        widths = {o: fs.width(o) for o in ("min-fill", "min-degree", "natural")}
        widths["random"] = fs.width(shuffled)
        try:
            r = bench.measure("grid elimination (min-fill)", "cpu",
                              lambda: fs.count(order="min-fill", budget=Budget(variables=1 << 10, tensor=1 << 12)),
                              repeats=2, **p)
            bench.records[-1].update(width_min_fill=widths["min-fill"], width_min_degree=widths["min-degree"],
                                     width_natural=widths["natural"], width_random_order=widths["random"],
                                     limbs=r.info["limbs"])
            if wdt == 3:
                bfr = bench.measure("grid brute force over 2^9 assignments", "cpu",
                                    lambda: fs.brute(budget=Budget(variables=1 << 10)), repeats=1, **p)
                if bfr.value != r.value:
                    raise SystemExit("COUNT-002 benchmark: elimination and brute force differ")
        except AlgebraBudgetError:
            bench.refusal("grid elimination past the 2^12 tensor budget", "cpu",
                          lambda: fs.count(order="min-fill", budget=Budget(variables=1 << 10, tensor=1 << 12)),
                          width_min_fill=widths["min-fill"], **p)
    # bounded search: certified bounds as the budget grows (a random 3-CNF: falsified clauses decide branches early)
    prog = mb.Program(random_cnf(prng, 16, 40), 16)
    exact = ms.compile(prog).count()
    for evals in (16, 256, 4096, 1 << 17):
        s = bench.measure("bounded Shannon expansion", "cpu", lambda: ms.search(prog, max_evaluations=evals),
                          repeats=1, max_evaluations=evals, k=16)
        if not (s.true[0] <= exact.true <= s.true[1] and s.false[0] <= exact.false <= s.false[1]):
            raise SystemExit("COUNT-002 benchmark: the bounds do not contain the exact count")
        bench.records[-1].update(status=s.status, true_lo=s.true[0], true_hi=s.true[1], exact_true=exact.true,
                                 false_lo=s.false[0], false_hi=s.false[1], exact_false=exact.false,
                                 open_branches=s.info["open"], evaluations=s.info["evaluations"])


def main():
    bench = Bench.from_args("COUNT-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    observed_universe(bench, rng)
    symbolic(bench, rng)
    bench.note("cpu = math_symcount.c (reduced ordered BDDs, exact L-limb counts, variable elimination, bounded "
               "Shannon expansion); cuda = math_symcount_gpu.cu (one launch per diagram level; brute-force slices; "
               "device runs only); COUNT-001 / duckdb = the observed-row baselines; python = an enumeration oracle.")
    bench.note("Refusals are recorded with the refused resource and need; nothing past a budget is extrapolated.")
    bench.write()


if __name__ == "__main__":
    main()
