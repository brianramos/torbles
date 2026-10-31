# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""BOOL-001 benchmark: predicate cofactors and finite differences.

    python bench/benchmark_math_bool.py --capability BOOL-001 --out FILE [--device] [--quick]

Case 1: predicate DAGs over 4 / 8 / 16 row conditions at three levels of shared subexpressions (a tree-like DAG, and
DAGs whose new nodes reuse recent nodes with probability 0.6 / 0.9), over 100k rows (20k with --quick). Reported
separately: the condition masks (state construction: the conditions evaluated once into value and valid bitmaps), the
program (DAG) construction, all first differences with their cofactors by shared-mask execution (first execution and
warm reuse), the same cofactors by independent evaluation (tr_bool_direct: one cofactor at a time, row by row from the
scope's values), and the DuckDB baseline (one SQL query per cofactor, SQL three-valued logic, the DAG as a chain of
lateral column aliases). Every baseline's counts are compared with the native ones.
Case 2: signed and XOR first and mixed differences (orders 1 / 2 / 3 / 4 / 8) under the uniform measure of the 2^k
assignments (k = 8 / 12 / 16) and under the empirical measure of the rows: every row selected, a sparse 1% selection,
and NULL-conditioned conditions (columns that are NULL on half the rows); DuckDB computes one order-2 signed
difference histogram (GROUP BY of the signed sum) as the baseline. Then the refusal thresholds (assignment support,
cofactor tensor), measured at the last admitted and the first refused point. CUDA is timed only with --device.
"""
import random

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles.math_budgets import Budget

COLS = ["x0", "x1", "x2", "x3", "x4", "x5", "n0", "n1"]


def table(rng, rows):
    data = {}
    for c in COLS:
        v = rng.integers(-100, 101, rows)
        mask = rng.random(rows) < (0.5 if c.startswith("n") else 0.1)   # n*: NULL on half the rows
        data[c] = pd.Series([None if m else int(x) for x, m in zip(v, mask)], dtype=object)
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in COLS) + ")"
    return torbles.from_ddl(ddl, {"t": pd.DataFrame(data)})


def condition_specs(k, null_heavy=False):
    """k conditions over the columns (thresholds spread so that each condition is neither rare nor universal)"""
    pool = ["n0", "n1"] if null_heavy else COLS[:6]
    ops = [">", "<=", "<>", ">=", "<", "="]
    out = []
    for i in range(k):
        col, op = pool[i % len(pool)], ops[i % len(ops)]
        value = (i * 37) % 120 - 60 if op != "=" else (i * 11) % 7
        out.append((col, op, value))
    return out


def sql_of(spec):
    col, op, value = spec
    return f'("{col}" {op} {value})'


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
    for i in range(k):                                   # every condition is read
        top = top ^ pool[i] if i % 2 else top | (pool[i] & built[len(built) // 2])
    return top


def chain_sql(prog, conds_sql, substitute):
    """the program as lateral column aliases _b0, _b1, ... (names no table column uses), the root last
    (substitute: {condition: 'TRUE' / 'FALSE'})"""
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


def duck_counts(con, prog, conds_sql, substitute, where):
    chain, root = chain_sql(prog, conds_sql, substitute)
    row = con.execute(f"SELECT COUNT(*) FILTER (WHERE r), COUNT(*) FILTER (WHERE NOT r), COUNT(*) FILTER (WHERE r IS NULL) "
                      f"FROM (SELECT {chain}, {root} AS r FROM t WHERE {where})").fetchall()[0]
    return dict(true=int(row[0]), false=int(row[1]), unknown=int(row[2]))


def case_dags(bench, db, rows):
    rng = random.Random(bench.seed)
    t = db["t"]
    scope = mbk.Scope.from_table(t, COLS)
    con = db.con
    where = "TRUE"
    for k in (4, 8, 16):
        specs = condition_specs(k)
        conds = [mb.condition(c, op, v) for c, op, v in specs]
        conds_sql = [sql_of(s) for s in specs]
        p = dict(rows=rows, conditions=k)
        for backend in bench.backends():
            bench.measure("condition masks (state construction)", backend,
                          lambda: mb.Masks.from_scope(scope, conds, backend=backend).info["bytes"], repeats=3,
                          phase="build", **p)
        masks = mb.Masks.from_scope(scope, conds)
        for share in (0.0, 0.6, 0.9):
            expr = random_expr(rng, k, 3 * k, share)
            prog = bench.measure("program construction", "cpu", lambda: mb.Program(expr, k), repeats=3, phase="build",
                                 **p, sharing=share)
            info = prog.info
            q = dict(p, sharing=share, nodes=info["nodes"], evaluated=info["evaluated"])
            queries = [[i] for i in range(k)]
            for backend in bench.backends():
                res = bench.measure("first differences + cofactors, shared masks", backend,
                                    lambda: mb.differences(masks, prog, queries, cofactors=True, backend=backend),
                                    repeats=5, **q)
            native = [[{x: c[x] for x in ("true", "false", "unknown")} for c in r["cofactors"]] for r in res]

            def independent():
                return [[mb.direct(prog, {i: tv}, scope=scope, conditions=conds) for tv in (0, 1)] for i in range(k)]
            got = bench.measure("first-difference cofactors, independent evaluation", "cpu-independent", independent,
                                repeats=2, cofactors=2 * k, **q)
            duck = bench.measure("first-difference cofactors, one SQL query each", "duckdb",
                                 lambda: [[duck_counts(con, prog, conds_sql, {i: tv}, where) for tv in ("FALSE", "TRUE")]
                                          for i in range(k)], repeats=1, cofactors=2 * k, **q)
            bench.note(f"k={k} sharing={share}: independent == shared-mask: {got == native}; duckdb == native: "
                       f"{duck == native}")
            if got != native or duck != native:
                raise SystemExit(f"baseline disagreement at k={k}, sharing={share}")


def case_measures(bench, db, rows):
    rng = random.Random(bench.seed + 1)
    nrng = np.random.default_rng(bench.seed + 1)
    for k in (8, 12, 16):
        masks = mb.Masks.assignments(k)
        expr = random_expr(rng, k, 3 * k, 0.6)
        prog = mb.Program(expr, k)
        for order, count in ((1, k), (2, 10), (3, 5), (4, 3), (8, 1)):
            queries = [sorted(rng.sample(range(k), order)) for _ in range(count)]
            for backend in bench.backends():
                bench.measure("signed + XOR differences, uniform measure", backend,
                              lambda: mb.differences(masks, prog, queries, backend=backend), repeats=5,
                              points=1 << k, conditions=k, order=order, queries=count)
    t = db["t"]
    k = 8
    expr = random_expr(rng, k, 3 * k, 0.6)
    prog = mb.Program(expr, k)
    sparse = nrng.random(rows) < 0.01
    for label, sel, null_heavy in (("every row", None, False), ("sparse 1% of rows", sparse, False),
                                   ("NULL-conditioned (half NULL)", None, True)):
        scope = mbk.Scope.from_table(t, COLS, rows=sel)
        specs = condition_specs(k, null_heavy)
        conds = [mb.condition(c, op, v) for c, op, v in specs]
        masks = mb.Masks.from_scope(scope, conds)
        for order, count in ((1, k), (2, 10), (3, 5)):
            queries = [sorted(rng.sample(range(k), order)) for _ in range(count)]
            for backend in bench.backends():
                res = bench.measure("signed + XOR differences, empirical measure", backend,
                                    lambda: mb.differences(masks, prog, queries, backend=backend), repeats=5,
                                    rows=rows, selected=scope.selected, conditions=k, order=order, queries=count,
                                    scope=label)
            bench.note(f"{label}, order {order}: unknown points of the first query {res[0]['unknown']} of "
                       f"{res[0]['points']}")
        # the DuckDB baseline of one order-2 signed difference histogram
        S = [0, 1]
        con = db.con
        conds_sql = [sql_of(s) for s in specs]
        if sel is not None:
            con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
        where = "rowid IN (SELECT rid FROM sel_rows)" if sel is not None else "TRUE"

        def duck_hist():
            terms, chains = [], []
            for tt in range(4):
                sub = {c: ("TRUE" if (tt >> j) & 1 else "FALSE") for j, c in enumerate(S)}
                chain, root = chain_sql(prog, conds_sql, sub)
                chains.append(f"(SELECT rowid AS rid, {root} AS r FROM (SELECT rowid, {chain} FROM t WHERE {where}))")
                terms.append(("-" if (2 - bin(tt).count("1")) % 2 else "+") + f" CAST(c{tt}.r AS INTEGER)")
            joins = " ".join(f"JOIN {c} c{i} ON c{i}.rid = c0.rid" for i, c in enumerate(chains) if i)
            sql = f"SELECT ({' '.join(terms)}) AS v, COUNT(*) FROM {chains[0]} c0 {joins} GROUP BY v"
            return {("unknown" if v is None else str(int(v))): int(c) for v, c in con.execute(sql).fetchall()}
        hist = bench.measure("order-2 signed difference histogram", "duckdb", duck_hist, repeats=2, rows=rows,
                             selected=scope.selected, scope=label)
        native = mb.differences(masks, prog, [S])[0]
        want = {str(v): c for v, c in native["signed"].items() if c}
        if native["unknown"]:
            want["unknown"] = native["unknown"]
        bench.note(f"{label}: DuckDB order-2 histogram == native: {hist == want}")
        if hist != want:
            raise SystemExit(f"DuckDB disagrees on {label}")


def thresholds(bench):
    bench.measure("assignment domain at the default support limit", "cpu",
                  lambda: mb.differences(mb.Masks.assignments(20), mb.Program(mb.var(0) ^ mb.var(19), 20), [[0]])[0][
                      "xor"]["true"], repeats=1, conditions=20, points=1 << 20)
    bench.refusal("assignment domain past the default support limit", "cpu", lambda: mb.Masks.assignments(21),
                  conditions=21, points=1 << 21)
    bench.threshold("assignment points (default budget)", admitted=dict(conditions=20, points=1 << 20),
                    refused=dict(conditions=21, points=1 << 21))
    k = 16
    masks = mb.Masks.assignments(k)
    prog = mb.Program(random_expr(random.Random(bench.seed + 2), k, 48, 0.6), k)
    budget = Budget(tensor=1 << 12)
    bench.measure("order-12 difference within a 2^12 cofactor tensor", "cpu",
                  lambda: mb.differences(masks, prog, [list(range(12))], budget=budget)[0]["nonzero"], repeats=1,
                  order=12, tensor_limit=1 << 12)
    bench.refusal("order-13 difference past a 2^12 cofactor tensor", "cpu",
                  lambda: mb.differences(masks, prog, [list(range(13))], budget=budget), order=13,
                  tensor_limit=1 << 12)
    bench.threshold("difference order (tensor 2^12)", admitted=dict(order=12), refused=dict(order=13))


def main():
    bench = Bench.from_args("BOOL-001", __doc__)
    rows = 20_000 if bench.quick else 100_000
    db = table(np.random.default_rng(bench.seed), rows)
    case_dags(bench, db, rows)
    case_measures(bench, db, rows)
    thresholds(bench)
    bench.note("Native CPU paths are single-threaded. cpu-independent = tr_bool_direct (one cofactor per pass, row by "
               "row, conditions re-read from the scope); shared masks = tr_bool_differences (conditions evaluated once, "
               "the base schedule once per word, only the cone of S per assignment).")
    bench.write()


if __name__ == "__main__":
    main()
