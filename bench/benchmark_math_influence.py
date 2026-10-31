# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""BOOL-002 benchmark: influences and interactions of predicates under the uniform and the empirical measure.

    python bench/benchmark_math_influence.py --capability BOOL-002 --out FILE [--device] [--quick]

Case 1: predicate DAGs over 4 / 8 / 16 row conditions at three levels of shared subexpressions (a tree-like DAG and
DAGs whose new nodes reuse recent nodes with probability 0.6 / 0.9), over 100k rows (20k with --quick). Every
condition's influence (empirical measure) by shared-mask execution, every condition at once (one pass: the program
once per word, then each condition's cone), against independent evaluation of the same cofactors (one BOOL-001
shared-mask pass per condition: the program re-evaluated in every pass) and against DuckDB (one SQL query per
condition: `f|x_i=TRUE <> f|x_i=FALSE` counted with three-valued WHERE). Every baseline's counts are compared.
Case 2: XOR and signed first and mixed differences (orders 1 / 2 / 3) under the uniform measure of the 2^k assignments
(k = 8 / 12 / 16, from the truth table; against BOOL-001's cone re-evaluation over the generated assignment words) and
under the empirical measure of the rows: every row, a sparse 1% selection, and NULL-conditioned conditions (columns
NULL on half the rows). Then the refusal threshold of the uniform measure (assignment support), measured at the last
admitted and the first refused point. CUDA is timed only with --device.
"""
import random

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_influence as mi

COLS = ["x0", "x1", "x2", "x3", "x4", "x5", "n0", "n1"]


def table(rng, rows):
    data = {}
    for c in COLS:
        v = rng.integers(-100, 101, rows)
        mask = rng.random(rows) < (0.5 if c.startswith("n") else 0.1)
        data[c] = pd.Series([None if m else int(x) for x, m in zip(v, mask)], dtype=object)
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in COLS) + ")"
    return torbles.from_ddl(ddl, {"t": pd.DataFrame(data)})


def condition_specs(k, null_heavy=False):
    pool = ["n0", "n1"] if null_heavy else COLS[:6]
    ops = [">", "<=", "<>", ">=", "<", "="]
    out = []
    for i in range(k):
        col, op = pool[i % len(pool)], ops[i % len(ops)]
        out.append((col, op, (i * 37) % 120 - 60 if op != "=" else (i * 11) % 7))
    return out


def random_expr(rng, k, size, share):
    pool = [mb.var(i) for i in range(k)]
    built = []
    for _ in range(size):
        cand = built[-6:] if built and rng.random() < share else pool + built[-3:]
        op = rng.choice("&|^~&|")
        a = rng.choice(cand)
        if op == "~":
            built.append(~a)
            continue
        b = rng.choice(pool + built)
        built.append(a & b if op == "&" else a | b if op == "|" else a ^ b)
    top = built[-1]
    for i in range(k):
        top = top ^ pool[i] if i % 2 else top | (pool[i] & built[len(built) // 2])
    return top


def chain_sql(prog, conds_sql, substitute):
    parts = []
    for i, (op, a, b) in enumerate(prog.nodes):
        if op == mb.CONST:
            s = ("FALSE", "TRUE", "CAST(NULL AS BOOLEAN)")[a]
        elif op == mb.VAR:
            s = substitute.get(a, conds_sql[a])
        elif op == mb.NOT:
            s = f"NOT _b{a}"
        else:
            s = f"(_b{a} {dict([(mb.AND, 'AND'), (mb.OR, 'OR'), (mb.XOR, '<>')])[op]} _b{b})"
        parts.append(f"{s} AS _b{i}")
    return ", ".join(parts), f"_b{prog.root}"


def duck_influence(con, prog, conds_sql, i):
    c1, r1 = chain_sql(prog, conds_sql, {i: "TRUE"})
    c0, r0 = chain_sql(prog, conds_sql, {i: "FALSE"})
    sql = (f"SELECT COUNT(*) FILTER (WHERE a.r <> b.r), COUNT(*) FILTER (WHERE (a.r <> b.r) IS NULL) FROM "
           f"(SELECT rowid AS rid, {r1} AS r FROM (SELECT rowid, {c1} FROM t)) a JOIN "
           f"(SELECT rowid AS rid, {r0} AS r FROM (SELECT rowid, {c0} FROM t)) b ON a.rid = b.rid")
    s, u = con.execute(sql).fetchall()[0]
    return int(s), int(u)


def case_dags(bench, db, rows):
    rng = random.Random(bench.seed)
    scope = mbk.Scope.from_table(db["t"], COLS)
    for k in (4, 8, 16):
        specs = condition_specs(k)
        masks = mb.Masks.from_scope(scope, [mb.condition(c, op, v) for c, op, v in specs])
        conds_sql = [f'("{c}" {op} {v})' for c, op, v in specs]
        for share in (0.0, 0.6, 0.9):
            prog = mb.Program(random_expr(rng, k, 3 * k, share), k)
            info = prog.info
            p = dict(rows=rows, conditions=k, sharing=share, nodes=info["nodes"], evaluated=info["evaluated"])
            for backend in bench.backends():
                res = bench.measure("every influence at once, shared masks (empirical)", backend,
                                    lambda: mi.empirical(prog, masks, backend=backend), repeats=4, **p)
            native = [(r["sensitive"], r["unknown"]) for r in res["influences"]]
            indep = bench.measure("influences, one shared-mask pass per condition (BOOL-001)", "cpu-independent",
                                  lambda: [(r["xor"]["true"], r["unknown"]) for i in range(k)
                                           for r in mb.differences(masks, prog, [[i]])], repeats=3, **p)
            if indep != native:
                raise SystemExit(f"per-condition passes disagree at k={k}, sharing={share}")
            if k <= 8 or not bench.quick:
                duck = bench.measure("influences, one SQL query per condition", "duckdb",
                                     lambda: [duck_influence(db.con, prog, conds_sql, i) for i in range(k)],
                                     repeats=1, **p)
                if duck != native:
                    raise SystemExit(f"DuckDB disagrees at k={k}, sharing={share}")
            bench.note(f"k={k}, sharing={share}: total influence {res['total_influence']} "
                       f"(max sensitivity {res['max_sensitivity']})")


def case_measures(bench, db, rows):
    rng = random.Random(bench.seed + 1)
    nrng = np.random.default_rng(bench.seed + 1)
    for k in ((8, 12) if bench.quick else (8, 12, 16)):
        prog = mb.Program(random_expr(rng, k, 3 * k, 0.6), k)
        assignments = mb.Masks.assignments(k)
        for order in (1, 2, 3):
            terms = [sorted(rng.sample(range(k), order)) for _ in range(min(10, k))] if order > 1 else []
            p = dict(points=1 << k, conditions=k, order=order, terms=len(terms))
            for backend in bench.backends():
                res = bench.measure("uniform measure: truth table, every influence + terms", backend,
                                    lambda: mi.uniform(prog, terms=terms, first=order == 1, backend=backend),
                                    repeats=4, **p)
            queries = [[i] for i in range(k)] if order == 1 else terms
            base = bench.measure("uniform measure: BOOL-001 cone re-evaluation on generated words", "cpu-bool001",
                                 lambda: mb.differences(assignments, prog, queries), repeats=3, **p)
            got = [r["sensitive"] for r in (res["influences"] if order == 1 else res["interactions"])]
            if got != [r["xor"]["true"] for r in base]:
                raise SystemExit(f"BOOL-001 disagrees on the uniform measure at k={k}, order={order}")
    k = 8
    prog = mb.Program(random_expr(rng, k, 3 * k, 0.6), k)
    sparse = nrng.random(rows) < 0.01
    for label, sel, null_heavy in (("every row", None, False), ("sparse 1% of rows", sparse, False),
                                   ("NULL-conditioned (half NULL)", None, True)):
        scope = mbk.Scope.from_table(db["t"], COLS, rows=sel)
        masks = mb.Masks.from_scope(scope, [mb.condition(c, op, v) for c, op, v in condition_specs(k, null_heavy)])
        for order in (1, 2, 3):
            terms = [sorted(rng.sample(range(k), order)) for _ in range(10)] if order > 1 else []
            for backend in bench.backends():
                res = bench.measure("empirical measure: every influence + terms", backend,
                                    lambda: mi.empirical(prog, masks, terms=terms, first=order == 1,
                                                         backend=backend), repeats=4, rows=rows,
                                    selected=scope.selected, conditions=k, order=order, scope=label)
            first = res["influences"][0] if order == 1 else res["interactions"][0]
            bench.note(f"{label}, order {order}: first term sensitive {first['sensitive']}, unknown "
                       f"{first['unknown']} of {res['points']} points")


def thresholds(bench):
    prog = mb.Program(mb.var(0) ^ mb.var(19), 20)
    bench.measure("uniform measure at the default support limit", "cpu",
                  lambda: mi.uniform(prog)["total_influence"].numerator, repeats=1, conditions=20, points=1 << 20)
    bench.refusal("uniform measure past the default support limit", "cpu",
                  lambda: mi.uniform(mb.Program(mb.var(0) ^ mb.var(20), 21)), conditions=21, points=1 << 21)
    bench.threshold("uniform measure conditions (default budget)", admitted=dict(conditions=20, points=1 << 20),
                    refused=dict(conditions=21, points=1 << 21))


def main():
    bench = Bench.from_args("BOOL-002", __doc__)
    rows = 20_000 if bench.quick else 100_000
    db = table(np.random.default_rng(bench.seed), rows)
    case_dags(bench, db, rows)
    case_measures(bench, db, rows)
    thresholds(bench)
    bench.note("Native CPU paths are single-threaded. Influence = coordinate sensitivity of the predicate (the measure "
               "of the points where its XOR derivative is TRUE), under the declared measure; not a causal effect.")
    bench.write()


if __name__ == "__main__":
    main()
