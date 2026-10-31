# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""QUANT-001 / QUANT-002 benchmark: existential and universal group predicates; relational division and
required-category coverage.

    python bench/benchmark_math_quant.py --capability QUANT-001 | QUANT-002 --out FILE [--device] [--quick]

The data are engine tables: parents(id) and lines(k REFERENCES parents(id), x, c); the child scope groups the lines by
their parent (the FND-004 dimension lookup), the parent scope is the parents table (TR_QUANT_PARENTS).
QUANT-001: 1 / 100 / 10,000 groups with 1 / 8 / 64 children each (up to 640k child rows) and 1% / 50% / 100% qualifying
rows (the predicate x < q, x uniform in 0..99), plus the special child sets of the inventory: empty groups (a fifth of
the parents without children), duplicated children (every row twice) and all-NULL children (x NULL in a tenth of the
groups). Timed separately: the child scope (the input snapshot), the quantifier state construction (first execution
and warm repeats), all ten answers from the warm state, and the DuckDB baselines: one GROUP BY query over parents LEFT
JOIN lines (COUNT FILTER, bool_or, bool_and) and the correlated quantified subqueries (TRUE = ANY, TRUE = ALL,
EXISTS, NOT EXISTS) per parent; both are compared with the native counts and answers.
QUANT-002: required category sets of 4 / 16 / 64 elements over 256 categories, 100 / 10,000 groups with 8 / 64
children (a third of the groups hold the categories 0..children-1, the others random ones): the coverage state as
masks and as sorted sets (state construction), the coverage of another required set of the same size from the
presence states (require: no rescan, warm reuse), and the DuckDB baselines: the classic double NOT EXISTS division and
the materialized matching children (lines JOIN required, GROUP BY HAVING COUNT(DISTINCT)) with their counts; the
dividing groups and the matched counts are compared with the native ones. Refusal thresholds (measured at the last
admitted and the first refused point): the group states of QUANT-001, the masks bytes of QUANT-002 (AUTO then runs the
sets; MASKS forced is refused). CUDA is timed only with --device; native CPU paths are single-threaded.
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_quant as mq
from torbles.math_budgets import Budget

DDL = ("CREATE TABLE parents (id BIGINT PRIMARY KEY); "
       "CREATE TABLE lines (k BIGINT REFERENCES parents(id), x BIGINT, c BIGINT);")
QUANTIFIERS = list(mq.QUANTIFIERS)


def tables(rng, groups, children, variant="plain", categories=256):
    """the engine tables of one grid point (variant: plain, empty, duplicates, all-null)"""
    k = np.repeat(np.arange(groups), children)
    if variant == "empty":
        k = k[(k % 5) != 1]                                  # parents 1, 6, 11, ... have no children
    x = rng.integers(0, 100, len(k)).astype(object)
    if variant == "all-null":
        x[(k % 10) == 3] = None                              # every child of these groups is NULL
    c = rng.integers(0, categories, len(k))
    c[0] = categories - 1                                    # the whole domain is in the codec
    cover = (k % 3) == 0                                     # a third of the groups: categories 0 .. children-1
    c[cover] = np.tile(np.arange(children), groups)[: len(k)][cover] % categories
    lines = pd.DataFrame({"k": k, "x": pd.Series(x, dtype=object), "c": c})
    if variant == "duplicates":
        lines = pd.concat([lines, lines], ignore_index=True)
    db = torbles.from_ddl(DDL, {"parents": pd.DataFrame({"id": np.arange(groups)}), "lines": lines})
    return db


def scopes(db, columns=("x",)):
    child = mbk.Scope.from_table(db["lines"], list(columns), group_by=[("k", db["parents"], "id")])
    parents = mbk.Scope.from_table(db["parents"], [], group_by=["id"])
    return child, parents


# ---------------------------------------------------------------------- QUANT-001
def duck_counts(con, q):
    sql = (f"SELECT p.id, count(l.k), count(*) FILTER (WHERE l.x < {q}), count(*) FILTER (WHERE NOT (l.x < {q})), "
           f"count(*) FILTER (WHERE l.k IS NOT NULL AND l.x IS NULL), bool_or(l.x < {q}), bool_and(l.x < {q}) "
           f"FROM parents p LEFT JOIN lines l ON l.k = p.id GROUP BY p.id ORDER BY p.id")
    return con.execute(sql).fetchall()


def duck_subqueries(con, q):
    m = "FROM lines l WHERE l.k = p.id"
    sql = (f"SELECT p.id, TRUE = ANY (SELECT l.x < {q} {m}), TRUE = ALL (SELECT l.x < {q} {m}), "
           f"EXISTS (SELECT 1 {m} AND l.x < {q}), NOT EXISTS (SELECT 1 {m} AND NOT (l.x < {q})) "
           f"FROM parents p ORDER BY p.id")
    return con.execute(sql).fetchall()


def compare_quant(state, counts, subq):
    s = state.states()
    G = len(counts)
    want = np.array([r[1:5] for r in counts], dtype=np.int64)
    got = np.stack([s[k][:G].astype(np.int64) for k in ("rows", "true", "false", "unknown")], axis=1)
    ok = bool(np.array_equal(got, want))
    code = {True: 1, False: 0, None: 2}
    for col, name in ((5, "bool_or"), (6, "bool_and")):
        ok &= bool(np.array_equal(state.answers(name)[:G], np.array([code[r[col]] for r in counts])))
    for col, name in ((1, "some"), (2, "all"), (3, "exists"), (4, "no_counterexample")):
        ok &= bool(np.array_equal(state.answers(name)[:G], np.array([code[r[col]] for r in subq])))
    return ok


def case_quantifiers(bench, rng):
    grid_groups = (1, 100) if bench.quick else (1, 100, 10_000)
    grid_children = (1, 8) if bench.quick else (1, 8, 64)
    grid_q = (1, 100) if bench.quick else (1, 50, 100)
    conds = lambda q: [mb.condition("x", "<", q)]
    for G in grid_groups:
        for ch in grid_children:
            db = tables(rng, G, ch)
            p = dict(groups=G, children=ch, rows=G * ch)
            child, parents = bench.measure("child scope (FND-004 input snapshot)", "cpu", lambda: scopes(db), repeats=3,
                                           phase="build", **p)
            for q in grid_q:
                pq = dict(p, qualifying=f"{q}%")
                for backend in bench.backends():
                    state = bench.measure("quantifier state construction", backend,
                                          lambda: mq.Quantifier.build(child, conds(q), mb.var(0), parents=parents,
                                                                      backend=backend), repeats=4, phase="build", **pq)
                bench.measure("ten answers from the warm state", "cpu",
                              lambda: [state.answers(n) for n in QUANTIFIERS], repeats=4, phase="warm", **pq)
                counts = bench.measure("GROUP BY parents LEFT JOIN lines (COUNT FILTER, bool_or, bool_and)", "duckdb",
                                       lambda: duck_counts(db.con, q), repeats=2, **pq)
                subq = bench.measure("correlated TRUE = ANY / ALL, EXISTS, NOT EXISTS per parent", "duckdb",
                                     lambda: duck_subqueries(db.con, q), repeats=1, **pq)
                ok = compare_quant(state, counts, subq)
                bench.note(f"QUANT-001 groups={G} children={ch} q={q}%: DuckDB == native: {ok}")
                if not ok:
                    raise SystemExit(f"DuckDB disagrees at groups={G}, children={ch}, q={q}")
    for G in ((100,) if bench.quick else (100, 10_000)):
        for variant in ("empty", "duplicates", "all-null"):
            db = tables(rng, G, 8, variant)
            child, parents = scopes(db)
            p = dict(groups=G, children=8, variant=variant, rows=db["lines"].n)
            for backend in bench.backends():
                state = bench.measure("quantifier state construction (special child sets)", backend,
                                      lambda: mq.Quantifier.build(child, conds(50), mb.var(0), parents=parents,
                                                                  backend=backend), repeats=4, phase="build", **p)
            counts = bench.measure("GROUP BY parents LEFT JOIN lines (COUNT FILTER, bool_or, bool_and)", "duckdb",
                                   lambda: duck_counts(db.con, 50), repeats=2, **p)
            subq = duck_subqueries(db.con, 50)
            ok = compare_quant(state, counts, subq)
            info = state.info
            bench.note(f"QUANT-001 {variant} groups={G}: empty={info['empty']} rows={info['rows']} "
                       f"unknown={info['unknown']}; DuckDB == native: {ok}")
            if not ok:
                raise SystemExit(f"DuckDB disagrees on {variant}")
    # threshold: the group states budget (states = groups)
    db = tables(rng, 1000, 4)
    child, parents = scopes(db)
    bench.measure("quantifier state at a states budget of 1000", "cpu",
                  lambda: mq.Quantifier.build(child, conds(50), mb.var(0), parents=parents,
                                              budget=Budget(states=child.groups)).info["present"], repeats=1,
                  groups=child.groups, limit=child.groups)
    bench.refusal("quantifier state past the states budget", "cpu",
                  lambda: mq.Quantifier.build(child, conds(50), mb.var(0), parents=parents,
                                              budget=Budget(states=child.groups - 1)), groups=child.groups,
                  limit=child.groups - 1)
    bench.threshold("QUANT-001 group states (states = groups)", admitted=dict(groups=child.groups, limit=child.groups),
                    refused=dict(groups=child.groups, limit=child.groups - 1))


# ---------------------------------------------------------------------- QUANT-002
def duck_division(con, required):
    con.execute("CREATE OR REPLACE TEMP TABLE req AS SELECT * FROM (VALUES " +
                ", ".join(f"({int(r)})" for r in required) + ") r(c)")
    return {int(r[0]) for r in con.execute(
        "SELECT p.id FROM parents p WHERE NOT EXISTS (SELECT 1 FROM req r WHERE NOT EXISTS "
        "(SELECT 1 FROM lines l WHERE l.k = p.id AND l.c = r.c))").fetchall()}


def duck_materialized(con, required):
    con.execute("CREATE OR REPLACE TEMP TABLE req AS SELECT * FROM (VALUES " +
                ", ".join(f"({int(r)})" for r in required) + ") r(c)")
    rows = con.execute("SELECT l.k, count(DISTINCT l.c) FROM lines l JOIN req r ON l.c = r.c GROUP BY l.k").fetchall()
    return {int(k): int(n) for k, n in rows}


def case_coverage(bench, rng):
    grid_groups = (100,) if bench.quick else (100, 10_000)
    grid_children = (8,) if bench.quick else (8, 64)
    grid_r = (4, 16) if bench.quick else (4, 16, 64)
    for G in grid_groups:
        for ch in grid_children:
            db = tables(rng, G, ch)
            p = dict(groups=G, children=ch, rows=G * ch, categories=256)
            child, parents = scopes(db, columns=())
            cats = bench.measure("category scope (FND-004 input snapshot)", "cpu",
                                 lambda: mbk.Scope.from_table(db["lines"], [], group_by=["c"]), repeats=3,
                                 phase="build", **p)
            for nr in grid_r:
                req, other = list(range(nr)), list(range(nr, 2 * nr))
                pr = dict(p, required=nr)
                for representation in ("masks", "sets"):
                    cov = bench.measure(f"coverage state construction ({representation})", "cpu",
                                        lambda: mq.Coverage.build(child, cats, req, parents=parents,
                                                                  representation=representation), repeats=4,
                                        phase="build", **pr)
                    if representation == "masks":
                        masks = cov
                if bench.device:
                    bench.measure("coverage state construction (masks)", "cuda",
                                  lambda: mq.Coverage.build(child, cats, req, parents=parents, representation="masks",
                                                            backend="cuda"), repeats=4, phase="build", **pr)
                for backend in bench.backends():
                    bench.measure("coverage of another required set (require: presence reused)", backend,
                                  lambda: masks.require(other, backend=backend), repeats=4, phase="warm", **pr)
                divides = bench.measure("division: double NOT EXISTS", "duckdb", lambda: duck_division(db.con, req),
                                        repeats=2, **pr)
                matched = bench.measure("matched counts: materialized matching children (JOIN, COUNT DISTINCT)",
                                        "duckdb", lambda: duck_materialized(db.con, req), repeats=2, **pr)
                s = masks.states()
                native_div = {int(g) for g in np.flatnonzero(masks.answers("divides") == 1)}
                native_matched = {int(g): int(s["matched"][g]) for g in range(G) if s["matched"][g]}
                having = {g for g, n in matched.items() if n == nr}
                ok = native_div == divides == having and native_matched == matched
                bench.note(f"QUANT-002 groups={G} children={ch} |R|={nr}: dividing groups {len(divides)}; DuckDB "
                           f"(double NOT EXISTS, HAVING COUNT DISTINCT, matched counts) == native: {ok}")
                if not ok:
                    raise SystemExit(f"DuckDB disagrees at groups={G}, children={ch}, |R|={nr}")
    # threshold: the masks bytes of a sparse wide domain (AUTO falls back to the sets, MASKS forced is refused)
    db = tables(rng, 2000, 8, categories=1 << 16)
    child, parents = scopes(db, columns=())
    cats = mbk.Scope.from_table(db["lines"], [], group_by=["c"])
    need = mq.cover_need(child, cats, list(range(16)), parents=parents, representation="masks")
    sets = mq.cover_need(child, cats, list(range(16)), parents=parents, representation="sets")["bytes"]
    big = 1 << 40                                             # states not limiting: the bytes decide
    at, past = Budget(states=big, bytes=need["bytes"]), Budget(states=big, bytes=need["bytes"] - 1)
    p = dict(groups=child.groups, categories=cats.groups, cells=need["states"], masks_bytes=need["bytes"],
             sets_bytes=sets)
    bench.measure("coverage MASKS forced at exactly the masks bytes", "cpu",
                  lambda: mq.Coverage.build(child, cats, list(range(16)), parents=parents, representation="masks",
                                            budget=at).info["representation"], repeats=1, limit=need["bytes"], **p)
    bench.refusal("coverage MASKS forced one byte past the budget", "cpu",
                  lambda: mq.Coverage.build(child, cats, list(range(16)), parents=parents, representation="masks",
                                            budget=past), limit=need["bytes"] - 1, **p)
    got = bench.measure("coverage AUTO one byte past the masks bytes", "cpu",
                        lambda: mq.Coverage.build(child, cats, list(range(16)), parents=parents,
                                                  budget=past).info["representation"], repeats=1,
                        limit=need["bytes"] - 1, **p)
    bench.threshold("QUANT-002 masks bytes (2000 groups x 65,536 categories, states not limiting)",
                    admitted=dict(bytes_limit=need["bytes"], representation="masks"),
                    refused=dict(bytes_limit=need["bytes"] - 1), auto_fallback=got, sets_bytes=sets)
    with_default = mq.Coverage.build(child, cats, list(range(16)), parents=parents).info
    bench.note(f"QUANT-002 sparse wide domain with the default budget: {with_default['representation']} "
               f"({with_default['choice']}): masks need {need['states']} cells and {need['bytes']} bytes, sets "
               f"{sets} bytes")


def main():
    bench = Bench.from_args("QUANT-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    if bench.capability == "QUANT-001":
        case_quantifiers(bench, rng)
    elif bench.capability == "QUANT-002":
        case_coverage(bench, rng)
    else:
        raise SystemExit("--capability is QUANT-001 or QUANT-002")
    bench.note("Native CPU paths are single-threaded; the scopes (input snapshots) are timed apart from the states. "
               "DuckDB runs on its own threads over the same source tables.")
    bench.write()


if __name__ == "__main__":
    main()
