# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""COUNT-001 benchmark: observed-domain constraint counts and witnesses, and the assignment universe.

    python bench/benchmark_math_count.py --capability COUNT-001 --out FILE [--device] [--quick]

Case 1: counts and witnesses over 1k / 100k / 1M observed rows (1k / 100k with --quick) for structured predicates
(ranges and bit planes of regular columns: equivalences and implications that hold by construction, one that does
not) and irregular ones (random columns with NULLs, random thresholds, random predicate DAGs). Measured separately: the
observed domain (state construction: the condition bitmaps and the FND-007 record), the counts with the 16 smallest
violating rows (first execution and warm reuse; CPU, CUDA with --device), the counts with every satisfying row, and
the baselines: DuckDB (COUNT(*) FILTER per constraint with SQL three-valued WHERE semantics, MIN(rowid) of the
violations, the satisfying rowids) and NumPy (Kleene logic on boolean value / known arrays). Every baseline result is
compared with the native one. Per grid point the certificate's coverage is recorded (rows covered, the observed
assignments of the support against its 2^s universe) and the same constraints are counted over the assignment universe
(a different result, labelled as such).
Case 2: the full-universe count of formulas by exhaustive assignment enumeration: tiny formulas (k <= 10) checked against
a Python brute force, bounded larger circuits (k = 12, 16, 20; 16 .. 1024 nodes) with variable-order controls (the same
formula under the reversed and a shuffled variable order: equal counts), their compilation size (nodes, evaluated nodes,
cone total), and the budget refusals (k = 21 past the default support of 2^20 assignments; a circuit one node past its
budget). COUNT-002 adds symbolic (decision-diagram) counting; this is enumeration only.
"""
import random

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_count as mc
from torbles.math_budgets import Budget

CONST, VAR, NOT, AND, OR, XOR = mb.CONST, mb.VAR, mb.NOT, mb.AND, mb.OR, mb.XOR


def obj(values):
    return pd.Series(values, dtype=object)


# ---------------------------------------------------------------------- data
def structured(n):
    i = np.arange(n, dtype=np.int64)
    x = i % 1000
    y = 4 * x + (i % 4)                                      # bit 2 of y = bit 0 of x; y >= 2000 iff x >= 500
    frame = pd.DataFrame({"x": x, "y": y})
    db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT)", {"t": frame})
    conds = [("x", ">=", 500), ("y", ">=", 2000), ("x", "bit", 0, 0), ("y", "bit", 2, 0), ("x", "<", 100),
             ("y", "bit", 0, 0)]
    v = mb.variables(len(conds))
    cs = [mc.equiv(v[0], v[1]), mc.equiv(v[2], v[3]), mc.implies(v[4], ~v[0]), mc.excludes(v[4], v[0]),
          mc.holds((v[2] ^ v[5]) | v[4]), mc.implies(v[5], v[2])]
    return db, ["x", "y"], conds, cs, dict(x=x, y=y)


def irregular(rng, n, prng):
    cols = {}
    for name in ("a", "b", "d"):
        vals = rng.integers(-1000, 1001, n).astype(np.int64)
        null = rng.random(n) < 0.1
        cols[name] = (vals, ~null)
    frame = pd.DataFrame({k: obj([None if not ok else int(v) for v, ok in zip(*cols[k])]) for k in cols})
    db = torbles.from_ddl("CREATE TABLE t (a BIGINT, b BIGINT, d BIGINT)", {"t": frame})
    conds = [("a", ">", int(rng.integers(-500, 500))), ("b", "<=", int(rng.integers(-500, 500))),
             ("d", ">", int(rng.integers(-500, 500))), ("a", "<", "b"), ("d", "bit", 3, 0), ("b", "is null"),
             ("a", "bit", 0, 0), ("d", "<=", int(rng.integers(-500, 500)))]
    k = len(conds)
    cs = []
    for j in range(8):
        kind = ("implies", "equiv", "excludes", "holds")[j % 4]
        a = random_expr(prng, k, 6)
        cs.append(mc.holds(a) if kind == "holds" else mc.Constraint(kind, a, random_expr(prng, k, 6)))
    return db, ["a", "b", "d"], conds, cs, {k: cols[k] for k in cols}


def random_expr(rng, k, size, unknown=False):
    pool = [mb.var(i) for i in range(k)] + [mb.TRUE, mb.FALSE] + ([mb.UNKNOWN] if unknown else [])
    built = []
    for _ in range(size):
        op = rng.choice([AND, OR, XOR, NOT, AND, OR])
        a = rng.choice(pool + built[-4:])
        if op == NOT:
            built.append(~a)
        else:
            b = rng.choice(pool + built)
            built.append(a & b if op == AND else a | b if op == OR else a ^ b)
    return built[-1]


def native_condition(spec):
    col, op = spec[0], spec[1]
    if op in ("is null", "is not null"):
        return mb.condition(col, op)
    if op == "bit":
        return mb.condition(col, "bit", spec[2], base=spec[3])
    if isinstance(spec[2], str):
        return mb.condition(col, op, mb.column(spec[2]))
    return mb.condition(col, op, spec[2])


def cond_sql(spec):
    col, op = spec[0], spec[1]
    if op in ("is null", "is not null"):
        return f'("{col}" {op.upper()})'
    if op == "bit":
        return f'((((CAST("{col}" AS HUGEINT) - ({spec[3]})) >> {spec[2]}) & 1) = 1)'
    rhs = f'"{spec[2]}"' if isinstance(spec[2], str) else str(spec[2])
    return f'("{col}" {op} {rhs})'


def expr_sql(e, conds):
    if e.op == CONST:
        return ("FALSE", "TRUE", "CAST(NULL AS BOOLEAN)")[e.a]
    if e.op == VAR:
        return conds[e.a]
    if e.op == NOT:
        return f"(NOT {expr_sql(e.a, conds)})"
    return f"({expr_sql(e.a, conds)} {dict([(AND, 'AND'), (OR, 'OR'), (XOR, '<>')])[e.op]} {expr_sql(e.b, conds)})"


def constraint_sql(c, conds):
    a = expr_sql(c.a, conds)
    if c.kind == "holds":
        return a
    b = expr_sql(c.b, conds)
    return {"implies": f"((NOT {a}) OR {b})", "equiv": f"(NOT ({a} <> {b}))", "excludes": f"(NOT ({a} AND {b}))"}[c.kind]


# ---------------------------------------------------------------------- the NumPy baseline (Kleene logic on arrays)
def numpy_conditions(arrays, conds, n):
    out = []
    for spec in conds:
        col, op = spec[0], spec[1]
        vals, ok = arrays[col] if isinstance(arrays[col], tuple) else (arrays[col], np.ones(n, bool))
        if op == "is null":
            out.append((~ok, np.ones(n, bool)))
            continue
        if op == "bit":
            v = (((vals - spec[3]) >> spec[2]) & 1) == 1
        elif isinstance(spec[2], str):
            other, ook = arrays[spec[2]] if isinstance(arrays[spec[2]], tuple) else (arrays[spec[2]], np.ones(n, bool))
            v = {"<": vals < other, ">": vals > other}[op]
            ok = ok & ook
        else:
            v = {">": vals > spec[2], ">=": vals >= spec[2], "<": vals < spec[2], "<=": vals <= spec[2]}[op]
        out.append((v & ok, ok))
    return out


def numpy_eval(e, leaves, n, memo):
    if id(e) in memo:
        return memo[id(e)]
    if e.op == CONST:
        r = (np.full(n, e.a == 1), np.full(n, e.a != 2))
    elif e.op == VAR:
        r = leaves[e.a]
    elif e.op == NOT:
        v, k = numpy_eval(e.a, leaves, n, memo)
        r = (k & ~v, k)
    else:
        (va, ka), (vb, kb) = numpy_eval(e.a, leaves, n, memo), numpy_eval(e.b, leaves, n, memo)
        if e.op == AND:
            v = va & vb
            r = (v, v | (ka & ~va) | (kb & ~vb))
        elif e.op == OR:
            v = va | vb
            r = (v, v | (ka & ~va & kb & ~vb))
        else:
            k = ka & kb
            r = ((va ^ vb) & k, k)
    memo[id(e)] = r
    return r


def numpy_counts(cs, leaves, n):
    out, memo = [], {}
    for c in cs:
        va, ka = numpy_eval(c.a, leaves, n, memo)
        if c.kind == "holds":
            v, k = va, ka
        else:
            vb, kb = numpy_eval(c.b, leaves, n, memo)
            if c.kind == "implies":
                v = (ka & ~va) | vb
                k = v | (ka & va & kb & ~vb)
            elif c.kind == "equiv":
                k = ka & kb
                v = k & ~(va ^ vb)
            else:
                k = (va & vb) | (ka & ~va) | (kb & ~vb)
                v = k & ~(va & vb)
        bad = k & ~v
        out.append((int(v.sum()), int(bad.sum()), int((~k).sum()), int(np.argmax(bad)) if bad.any() else None))
    return out


# ---------------------------------------------------------------------- case 1
def case_observed(bench):
    rng = np.random.default_rng(bench.seed)
    prng = random.Random(bench.seed)
    sizes = (1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)
    for n in sizes:
        for shape in ("structured", "irregular"):
            if shape == "structured":
                db, names, specs, constraints, arrays = structured(n)
            else:
                db, names, specs, constraints, arrays = irregular(rng, n, prng)
            scope = mbk.Scope.from_table(db["t"], names)
            conds = [native_condition(s) for s in specs]
            k = len(conds)
            cs = mc.Constraints(k, constraints)
            p = dict(rows=n, shape=shape, conditions=k, constraints=len(constraints), nodes=cs.info["nodes"])
            for backend in bench.backends():
                dom = bench.measure("observed domain (condition bitmaps and FND-007 record)", backend,
                                    lambda: mc.Domain.observe(scope, conds, backend=backend), repeats=3, phase="build",
                                    **p)
            dom = mc.Domain.observe(scope, conds)
            for backend in bench.backends():
                ans = bench.measure("counts and the 16 smallest violating rows", backend,
                                    lambda: mc.count(dom, cs, witnesses=16, backend=backend), repeats=5, **p)
                bench.measure("counts and every satisfying row", backend,
                              lambda: mc.count(dom, cs, rows=True, witnesses=16, backend=backend), repeats=3, **p)
            res = mc.count(dom, cs, rows=True, witnesses=16).results
            leaves = numpy_conditions(arrays, specs, n)
            got_np = bench.measure("counts and the first violating row", "numpy-kleene",
                                   lambda: numpy_counts(constraints, leaves, n), repeats=3, **p)
            if got_np != [(r["satisfied"], r["violated"], r["unknown"], r["witness"]) for r in res]:
                raise SystemExit(f"NumPy disagrees at {p}")
            csql = [cond_sql(s) for s in specs]
            qs = [constraint_sql(c, csql) for c in constraints]
            select = ", ".join(f"COUNT(*) FILTER (WHERE {q}), COUNT(*) FILTER (WHERE NOT {q}), "
                               f"COUNT(*) FILTER (WHERE ({q}) IS NULL), MIN(rowid) FILTER (WHERE NOT {q})" for q in qs)
            con = db.con
            row = bench.measure("counts and the first violating row", "duckdb",
                                lambda: con.execute(f"SELECT {select} FROM t").fetchall()[0], repeats=3, **p)
            want = [tuple(row[4 * j:4 * j + 4]) for j in range(len(qs))]
            if [(int(a), int(b), int(c), None if d is None else int(d)) for a, b, c, d in want] != \
                    [(r["satisfied"], r["violated"], r["unknown"], r["witness"]) for r in res]:
                raise SystemExit(f"DuckDB disagrees at {p}")
            if n <= 100_000:
                q0 = qs[0]
                rows = bench.measure("the satisfying rowids of one constraint", "duckdb",
                                     lambda: con.execute(f"SELECT rowid FROM t WHERE {q0} ORDER BY rowid").fetchall(),
                                     repeats=2, **p)
                if [int(r) for r, in rows] != [int(x) for x in res[0]["rows"]]:
                    raise SystemExit(f"DuckDB rowids disagree at {p}")
            cert = mc.count(dom, cs).certificate
            uni = mc.count(mc.Domain.universe(k), cs, witnesses=4)
            bench.measure("the same constraints over the 2^k assignment universe", "cpu",
                          lambda: mc.count(mc.Domain.universe(k), cs, witnesses=4), repeats=3, **p)
            bench.note(f"{shape} {n} rows: covered {cert['points']} rows; support {len(cert['support'])} conditions, "
                       f"observed {cert['observed']} of {cert['universe']} assignments (unobserved, never checked: "
                       f"{cert['unobserved']}); violations on the snapshot "
                       f"{[r['violated'] for r in res]}, over the universe {[r['violated'] for r in uni.results]}")


# ---------------------------------------------------------------------- case 2
def scalar(e, a):
    if e.op == CONST:
        return (False, True, None)[e.a]
    if e.op == VAR:
        return bool((a >> e.a) & 1)
    if e.op == NOT:
        v = scalar(e.a, a)
        return None if v is None else not v
    u, v = scalar(e.a, a), scalar(e.b, a)
    if e.op == AND:
        return False if u is False or v is False else (None if u is None or v is None else True)
    if e.op == OR:
        return True if u is True or v is True else (None if u is None or v is None else False)
    return None if u is None or v is None else u != v


def brute(expr, k):
    vals = [scalar(expr, a) for a in range(1 << k)]
    return vals.count(True), vals.count(False), vals.count(None)


def wide_formula(rng, k, nodes):
    """a DAG of exactly `nodes` operator nodes over k variables (leaves drawn from the variables, operands from
    everything built so far: a wide, shared circuit)"""
    pool = [mb.var(i) for i in range(k)]
    built = []
    while len(built) < nodes:
        op = rng.choice([AND, OR, XOR, XOR])
        a, b = rng.choice(pool + built[-64:]), rng.choice(pool + built)
        built.append(a & b if op == AND else a | b if op == OR else a ^ b)
    top = built[-1]
    for e in built[-8:-1]:
        top = top ^ e
    return top


def relabel(e, perm, memo):
    """the same formula with variable i renamed perm[i]"""
    if id(e) in memo:
        return memo[id(e)]
    if e.op == CONST:
        r = e
    elif e.op == VAR:
        r = mb.var(perm[e.a])
    elif e.op == NOT:
        r = ~relabel(e.a, perm, memo)
    else:
        a, b = relabel(e.a, perm, memo), relabel(e.b, perm, memo)
        r = a & b if e.op == AND else a | b if e.op == OR else a ^ b
    memo[id(e)] = r
    return r


def case_universe(bench):
    prng = random.Random(bench.seed + 7)
    for k in (2, 4, 6, 8, 10):                                # tiny formulas, brute-force checked
        for trial in range(3):
            e = random_expr(prng, k, 2 * k, unknown=True)
            cs = mc.Constraints(k, [mc.holds(e)])
            uni = mc.Domain.universe(k)
            p = dict(k=k, trial=trial, nodes=cs.info["nodes"])
            r = bench.measure("universe count by enumeration (tiny formula)", "cpu", lambda: mc.count(uni, cs),
                              repeats=3, **p).results[0]
            want = bench.measure("universe count by enumeration (tiny formula)", "python-brute-force",
                                 lambda: brute(e, k), repeats=1, **p)
            if (r["satisfied"], r["violated"], r["unknown"]) != want:
                raise SystemExit(f"brute force disagrees at {p}")
    ks = (12, 16) if bench.quick else (12, 16, 20)
    widths = (16, 64, 256) if bench.quick else (16, 64, 256, 1024)
    for k in ks:
        budget = Budget(support=1 << k)
        uni = mc.Domain.universe(k, budget=budget)
        for width in widths:
            e = wide_formula(prng, k, width)
            orders = {"natural": list(range(k)), "reversed": list(range(k))[::-1],
                      "shuffled": prng.sample(range(k), k)}
            counts = {}
            for name, perm in orders.items():
                f = relabel(e, perm, {})
                cs = mc.Constraints(k, [mc.holds(f), mc.implies(f, mb.var(perm[0])), mc.excludes(f, ~f)])
                info = cs.info
                p = dict(k=k, width=width, order=name, nodes=info["nodes"], evaluated=info["evaluated"],
                         cones=info["cones"], assignments=1 << k)
                for backend in bench.backends():
                    ans = bench.measure("universe count by enumeration (bounded circuit)", backend,
                                        lambda: mc.count(uni, cs, witnesses=4, backend=backend, budget=budget),
                                        repeats=3, **p)
                counts[name] = [(r["satisfied"], r["violated"], r["unknown"]) for r in ans.results]
            if len({tuple(v) for v in counts.values()}) != 1:
                raise SystemExit(f"the variable order changed the counts at k={k}, width={width}: {counts}")
    bench.refusal("universe of 21 conditions (default support 2^20)", "cpu", lambda: mc.Domain.universe(21), k=21)
    mc.Domain.universe(20)
    bench.threshold("assignment universe (support 2^20)", admitted=dict(k=20), refused=dict(k=21))
    e = wide_formula(prng, 12, 64)
    nn = mc.Constraints(12, [mc.holds(e)]).info["nodes"]
    mc.Constraints(12, [mc.holds(e)], budget=Budget(circuit=nn))
    bench.refusal("constraint set one node past its circuit budget", "cpu",
                  lambda: mc.Constraints(12, [mc.holds(e)], budget=Budget(circuit=nn - 1)), nodes=nn)
    bench.threshold("constraint set nodes (circuit budget)", admitted=dict(nodes=nn, budget=nn),
                    refused=dict(nodes=nn, budget=nn - 1))


def thresholds(bench):
    db, names, specs, constraints, arrays = structured(5000)
    scope = mbk.Scope.from_table(db["t"], names)
    dom = mc.Domain.observe(scope, [native_condition(s) for s in specs])
    cs = mc.Constraints(len(specs), constraints)
    res = mc.count(dom, cs, rows=True, witnesses=16).results
    out = len(constraints) + sum(r["satisfied"] for r in res) + sum(min(16, r["violated"]) for r in res)
    bench.measure("every satisfying row within the output budget", "cpu",
                  lambda: mc.count(dom, cs, rows=True, witnesses=16, budget=Budget(output=out)), repeats=1, output=out)
    bench.refusal("every satisfying row one past the output budget", "cpu",
                  lambda: mc.count(dom, cs, rows=True, witnesses=16, budget=Budget(output=out - 1)), output=out - 1)
    bench.threshold("satisfying rows returned (output budget)", admitted=dict(output=out), refused=dict(output=out - 1))


def main():
    bench = Bench.from_args("COUNT-001", __doc__)
    case_observed(bench)
    case_universe(bench)
    thresholds(bench)
    bench.note("The observed snapshot covers exactly the selected rows (the FND-007 record keeps their bits); the "
               "assignment universe is a separate enumeration over 2^k points. Native CPU paths are single-threaded; "
               "the CUDA path (counts with atomics, chunked extraction) is timed only with --device.")
    bench.write()


if __name__ == "__main__":
    main()
