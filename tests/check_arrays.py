# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Window functions, GROUP BY, scalar operators, joins and set operations as array algebra (sqlwin) against the
row-by-row definitions and against DuckDB.

Random window queries (every window function, PARTITION BY on zero to two columns, ORDER BY with directions and NULL
placement, ROWS / RANGE / GROUPS frames with offsets, EXCLUDE, FILTER, IGNORE NULLS, FROM LAST) and random GROUP BY
queries (grouping sets, collations, DISTINCT, FILTER, every aggregate the arrays compute) on a table with ties, NULLs,
NaN and infinities, each run twice on torbles: as arrays, and row by row (on each partition / group). The two answers
must be identical value for value (types included); where DuckDB has the same semantics its answer must agree too.
    python tests/check_arrays.py [queries per part, default 500] [seed, default 5]"""
import datetime as dt, math, os, random, sys
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import duckdb, pandas as pd
import torbles
import sqlexec, sqlfuzz

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ENGINE = os.environ.get("TORBLES_SQL_ENGINE", "alg")
r = random.Random(SEED)
N = 160
rows = []
for i in range(N):
    rows.append((i, r.choice([1, 1, 2, 3, None]), r.choice(["a", "b", "b", None]), r.choice([None, 0, 1, 1, 2, 3, 5, 8, 8, 13, -4]),
                 r.choice([None, -2.5, 0.0, 0.5, 0.5, 1.25, 7.0, 1e15]), r.choice([None, Decimal("1.50"), Decimal("-0.25"), Decimal("1.50"), Decimal("10.00")]),
                 r.choice([None, dt.date(2020, 1, 1), dt.date(2020, 1, 3), dt.date(2020, 2, 1), dt.date(2021, 1, 1)]),
                 r.choice([None, -7, 0, 1, 2, 2, 3, 9, 100, 2 ** 40]), r.choice([None, -1.5, 0.1, 0.2, 2.0, 3.75, 1e-300, 1e300]),
                 r.choice([None, Decimal("0.01"), Decimal("2.50"), Decimal("-3.33"), Decimal("1000.00")]),
                 r.choice([None, True, False, True]), r.choice([None, "x", "y", "Z", "", "yy"]),
                 r.choice([None, 1.0, float("nan"), float("inf"), float("-inf"), -2.0])))
cols = ["id", "p", "q", "o", "of", "od", "dd", "x", "y", "z", "b", "s", "yn"]
ddl = ("CREATE TABLE w (id INTEGER, p INTEGER, q VARCHAR, o INTEGER, \"of\" DOUBLE, od DECIMAL(9,2), dd DATE, x BIGINT, y DOUBLE, "
       "z DECIMAL(9,2), b BOOLEAN, s VARCHAR, yn DOUBLE)")
df = pd.DataFrame(rows, columns=cols)
con = duckdb.connect(); con.execute(ddl); con.register("df", df); con.execute("INSERT INTO w SELECT * FROM df")
nan_ids = [row[0] for row in rows if isinstance(row[-1], float) and math.isnan(row[-1])]      # a frame's NaN arrives as NULL
con.execute(f"UPDATE w SET yn = 'NaN'::DOUBLE WHERE id IN ({', '.join(map(str, nan_ids))})")
con.unregister("df")
db = torbles.from_duckdb(con, keep_plain=ENGINE == "plain")

ARGS = {"num": ["x", "y", "z", "o"], "any": ["x", "y", "z", "s", "dd", "b", "o"], "bool": ["b"], "nan": ["yn"]}


def frame(order):
    if not r.random() < 0.7: return ""
    mode = r.choice(["ROWS", "ROWS", "RANGE", "GROUPS"])
    if mode == "RANGE" and len(order) == 1 and order[0][0] in ("o", '"of"', "od", "dd") and r.random() < 0.7:
        off = {"o": ["0", "1", "3"], '"of"': ["0.5", "2", "0"], "od": ["0.25", "1", "0"], "dd": ["INTERVAL 2 DAY", "INTERVAL 1 MONTH", "INTERVAL 0 DAY"]}[order[0][0]]
    else:
        off = ["0", "1", "2", "5"] if mode != "RANGE" else None
    o = lambda: r.choice(off)
    starts = ["UNBOUNDED PRECEDING", "CURRENT ROW"] + (["P", "F", "P"] if off else [])
    a = r.choice(starts)
    ends = {"UNBOUNDED PRECEDING": ["P", "CURRENT ROW", "F", "UNBOUNDED FOLLOWING"], "P": ["P", "CURRENT ROW", "F", "UNBOUNDED FOLLOWING"],
            "CURRENT ROW": ["CURRENT ROW", "F", "UNBOUNDED FOLLOWING"], "F": ["F", "UNBOUNDED FOLLOWING"]}[a]
    b = r.choice([e for e in ends if off or e not in ("P", "F")])
    txt = lambda k: f"{o()} PRECEDING" if k == "P" else f"{o()} FOLLOWING" if k == "F" else k
    a, b = txt(a), txt(b)
    ex = r.choice(["", "", "", " EXCLUDE CURRENT ROW", " EXCLUDE GROUP", " EXCLUDE TIES", " EXCLUDE NO OTHERS"])
    return f" {mode} BETWEEN {a} AND {b}{ex}"


def window():
    part = r.choice([[], [], ["p"], ["q"], ["p", "q"]])
    order = []
    for k in r.sample(["o", '"of"', "od", "dd", "s", "id"], r.choice([0, 1, 1, 1, 2])):
        order.append((k, r.choice(["", " DESC"]), r.choice(["", " NULLS FIRST", " NULLS LAST"])))
    if r.random() < 0.6 and not any(k == "id" for k, _, _ in order): order.append(("id", r.choice(["", " DESC"]), ""))
    w = ""
    if part: w += "PARTITION BY " + ", ".join(part)
    if order: w += (" " if w else "") + "ORDER BY " + ", ".join(k + d + nl for k, d, nl in order)
    return w, order


def func(order):
    """(SQL text, DuckDB has the same semantics, needs ORDER BY)"""
    k = r.random()
    if k < 0.15:
        f = r.choice(["row_number()", "rank()", "dense_rank()", "percent_rank()", "cume_dist()", f"ntile({r.choice([1, 2, 3, 7, 500])})"])
        return f, True, False
    if k < 0.3:
        a = r.choice(ARGS["any"]); f = r.choice(["lag", "lead"])
        args = [a] + ([str(r.choice([0, 1, 2, 5, -1]))] if r.random() < 0.6 else [])
        if len(args) == 2 and r.random() < 0.4: args.append({"x": "-1", "y": "0.5", "z": "1.00", "s": "'d'", "dd": "DATE '1999-01-01'", "b": "TRUE", "o": "0"}[a])
        ign = r.random() < 0.3
        return f"{f}({', '.join(args)}){' IGNORE NULLS' if ign else ''}", True, False
    if k < 0.42:
        a = r.choice(ARGS["any"]); f = r.choice(["first_value", "last_value", "nth_value"])
        ign = r.random() < 0.3
        if f == "nth_value":
            last = r.random() < 0.3
            return f"nth_value({a}, {r.choice([1, 2, 3, 9])}){' FROM LAST' if last else ''}{' IGNORE NULLS' if ign else ''}", not last, False
        return f"{f}({a}){' IGNORE NULLS' if ign else ''}", True, False
    g = r.choice(["count(*)", "count({A})", "sum({N})", "avg({N})", "min({A})", "max({A})", "var_pop({N})", "var_samp({N})",
                  "stddev_pop({N})", "stddev_samp({N})", "covar_pop({N}, {N})", "covar_samp({N}, {N})", "corr({N}, {N})",
                  "regr_slope({N}, {N})", "regr_intercept({N}, {N})", "regr_count({N}, {N})", "regr_r2({N}, {N})",
                  "regr_avgx({N}, {N})", "regr_sxy({N}, {N})", "bool_and(b)", "bool_or(b)", "arg_min({A}, {N})",
                  "arg_max({A}, {N})", "sum(yn)", "avg(yn)", "min(yn)", "max(yn)", "count(DISTINCT {A})", "sum(DISTINCT {N})",
                  "string_agg(s, ',')", "median({N})", "count_if(b)", "any_value({A})", "bit_and(x)", "bit_or(o)", "bit_xor(x)"])
    duck = not ("yn" in g or "arg_" in g or "string_agg" in g or "any_value" in g)   # NaN, ties, order: not DuckDB's
    while "{N}" in g: g = g.replace("{N}", r.choice(ARGS["num"]), 1)
    if g.startswith(("var", "stddev", "covar", "corr", "regr")) and any(c in g for c in ("x", "y")):
        duck = False                     # 2^40 and 1e300 in double arithmetic: DuckDB loses the digits torbles keeps
    while "{A}" in g: g = g.replace("{A}", r.choice(ARGS["any"]), 1)
    if r.random() < 0.2 and "DISTINCT" not in g:
        g += f" FILTER (WHERE {r.choice(['x > 1', 'b', 'o IS NULL', chr(39).join(['s <> ', 'x', ''])])})"
    return g, duck, False


def ident(a, b):
    """the same values and types (NaN equal to NaN)"""
    if len(a) != len(b): return False
    for ra, rb in zip(a, b):
        for u, v in zip(ra, rb):
            if type(u) is not type(v): return False
            if isinstance(u, float) and math.isnan(u) and math.isnan(v): continue
            if u != v: return False
    return True


def run(sql, vec, flag):
    setattr(sqlexec, flag, vec)
    try: return db.sql(sql, ENGINE).rows, None
    except Exception as x: return None, f"{type(x).__name__}: {x}"
    finally: setattr(sqlexec, flag, True)


fails = same_err = row_only = duck_ok = duck_skip = 0
errs = {}


def compare(qi, sql, duck, flag):
    """the query as arrays and row by row (flag off), then against DuckDB when duck"""
    global fails, same_err, row_only, duck_ok, duck_skip
    a, ea = run(sql, True, flag)
    b, eb = run(sql, False, flag)
    if ea or eb:
        if ea and eb and ea.split(":")[0] == eb.split(":")[0]:
            same_err += 1; errs[ea[:70]] = errs.get(ea[:70], 0) + 1
            if os.environ.get("VERBOSE") and "syntax" in ea: print(f"syntax {qi}: {sql}")
            return
        if eb and ea is None and "SqlError" not in eb:          # the row path fails (e.g. Fraction(inf)); arrays answer
            row_only += 1; errs["rows only: " + eb[:60]] = errs.get("rows only: " + eb[:60], 0) + 1
            if os.environ.get("VERBOSE"): print(f"rows only {qi}: {sql}\n   {eb}")
            return
        fails += 1; print(f"FAIL {qi}: {sql}\n   arrays: {ea or 'ok'}\n   rows:   {eb or 'ok'}"); return
    if not ident(a, b):
        fails += 1
        bad = next(i for i, (x, y) in enumerate(zip(a, b)) if not ident([x], [y])) if len(a) == len(b) else -1
        print(f"FAIL {qi}: {sql}\n   arrays {a[bad] if bad >= 0 else len(a)}\n   rows   {b[bad] if bad >= 0 else len(b)}")
        return
    if not duck: return
    try: want = con.execute(sql.replace(") IGNORE NULLS OVER", " IGNORE NULLS) OVER")).fetchall()   # DuckDB's place for it
    except Exception as x:
        duck_skip += 1
        if os.environ.get("VERBOSE"): print(f"duckdb rejected {qi}: {sql}\n   {str(x)[:150]}")
        return
    if sqlfuzz.same(a, want, " ORDER BY " in sql.split(" FROM ")[-1], 1e-9): duck_ok += 1
    else:
        fails += 1
        bad = next((i for i, (x, y) in enumerate(zip(a, want)) if not sqlfuzz.same([x], [y], True, 1e-9)), -1)
        print(f"FAIL {qi} (vs DuckDB): {sql}\n   torbles {a[bad] if bad >= 0 else len(a)}\n   duckdb  {want[bad] if bad >= 0 else len(want)}")


print("== window functions")
for qi in range(NQ):
    w, order = window()
    nf = r.choice([1, 1, 2])
    fs = [func(order) for _ in range(nf)]
    if r.random() < 0.12:                                  # windows over a grouped query
        sql = r.choice([
            "SELECT p, q, sum(x), rank() OVER (ORDER BY sum(x) NULLS FIRST, p NULLS FIRST, q NULLS FIRST) FROM w GROUP BY p, q",
            "SELECT p, q, sum(sum(x)) OVER (PARTITION BY p ORDER BY q NULLS FIRST ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM w GROUP BY p, q",
            "SELECT p, avg(y), max(avg(y)) OVER (ORDER BY p NULLS LAST RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM w GROUP BY p",
            "SELECT q, count(*), lag(count(*), 1, 0) OVER (ORDER BY q NULLS FIRST) FROM w GROUP BY q"]) + " ORDER BY 1 NULLS FIRST, 2 NULLS FIRST"
        duck = True
    else:
        fr = frame(order)
        sql = "SELECT id, " + ", ".join(f"{g} OVER ({w}{fr})" for g, _, _ in fs) + " FROM w ORDER BY id"
        # DuckDB orders ties its own way: compare only answers that do not depend on the order of peers
        positional = ("row_number", "ntile", "lag", "lead", "first_value", "last_value", "nth_value", "any_value", "arg_m", "string_agg")
        det = (order and order[-1][0] == "id") or (not any(g.startswith(positional) for g, _, _ in fs) and " ROWS " not in fr)
        duck = all(d for _, d, _ in fs) and det
    compare(qi, sql, duck, "VEC_WINDOWS")
wins = (fails, duck_ok)

print("== GROUP BY")
GKEYS = ["p", "q", "o", "dd", "s", "b", "od", '"of"']
GAGG = ["count(*)", "count({A})", "sum({N})", "avg({N})", "min({A})", "max({A})", "var_pop({N})", "var_samp({N})", "stddev_samp({N})",
        "covar_pop({N}, {N})", "corr({N}, {N})", "regr_slope({N}, {N})", "regr_count({N}, {N})", "bool_and(b)", "bool_or(b)", "count_if(b)",
        "count(DISTINCT {A})", "sum(DISTINCT {N})", "avg(DISTINCT {N})", "min(DISTINCT {A})", "arg_min({A}, {N})", "arg_max({A}, {N})",
        "any_value({A})", "sum(yn)", "avg(yn)", "max(yn)", "var_pop(yn)", "count(DISTINCT yn)", "string_agg(s, ',' ORDER BY s)", "median({N})",
        "count(DISTINCT s COLLATE NOCASE)", "quantile_cont({N}, 0.25)", "quantile_disc({N}, 0.75)", "median(dd)", "median(s)",
        "percentile_cont(0.3) WITHIN GROUP (ORDER BY {N})", "percentile_disc(0.6) WITHIN GROUP (ORDER BY {A} DESC)", "mode({A})",
        "quantile_cont({N}, [0.1, 0.5, 0.9])", "median(DISTINCT {N})", "bit_and(x)", "bit_or(o)", "bit_xor(x)", "bit_xor(DISTINCT o)"]
for qi in range(NQ):
    keys = r.sample(GKEYS, r.choice([0, 1, 1, 2, 2, 3]))
    coll = keys and "s" in keys and r.random() < 0.3
    kx = [k + (" COLLATE NOCASE" if k == "s" and coll else "") for k in keys]
    aggs = []
    for _ in range(r.choice([1, 2, 3])):
        g = r.choice(GAGG)
        while "{N}" in g: g = g.replace("{N}", r.choice(ARGS["num"]), 1)
        while "{A}" in g: g = g.replace("{A}", r.choice(ARGS["any"]), 1)
        if r.random() < 0.2 and "DISTINCT" not in g and "ORDER BY" not in g:
            g += f" FILTER (WHERE {r.choice(['x > 1', 'b', 'o IS NULL', 'y < 1'])})"
        aggs.append(g)
    where = r.choice(["", "", " WHERE x > 0", " WHERE p IS NOT NULL", " WHERE s <> 'x'"])
    if keys:
        form = r.choice(["plain", "plain", "rollup", "cube", "sets"])
        gb = {"plain": ", ".join(kx), "rollup": f"ROLLUP ({', '.join(kx)})", "cube": f"CUBE ({', '.join(kx)})",
              "sets": f"GROUPING SETS (({', '.join(kx)}), ({kx[0]}), ())"}[form]
        sql = f"SELECT {', '.join(kx)}, {', '.join(aggs)} FROM w{where} GROUP BY {gb}"
    else:
        sql = f"SELECT {', '.join(aggs)} FROM w{where}"
    stat = ("var", "stddev", "covar", "corr", "regr")
    # PERCENTILE_CONT of a DECIMAL: torbles interpolates exactly, DuckDB in double (docs/SQL.md)
    cont_dec = any(g.startswith(("quantile_cont(z", "quantile_cont(od", "percentile_cont")) and ("z" in g or "od" in g) for g in aggs)
    duck = not coll and not cont_dec and "yn" not in sql and "any_value" not in sql and "arg_m" not in sql and "mode(" not in sql and \
        not any(g.startswith(stat) and any(c in g for c in ("x", "y")) for g in aggs)
    compare(qi, sql, duck, "VEC_GROUPS")

print("== scalar operators")
def sexpr(d=0):
    if d > 2 or r.random() < 0.3:
        return r.choice(["x", "y", "o", "p", "id", "1", "2.5", "-3", "CAST(o AS DOUBLE)", "CAST(z AS DOUBLE)", "CAST(o AS SMALLINT)",
                         "CAST(p AS BIGINT)", "NULL", "(x // 3)", "CAST(y > 1 AS INTEGER)"])
    op = r.choice(["+", "-", "*", "/", "+", "*"])
    return f"({sexpr(d + 1)} {op} {sexpr(d + 1)})" if r.random() < 0.8 else f"(-({sexpr(d + 1)}))"
def spred(d=0):
    if d > 1 or r.random() < 0.4:
        return f"({sexpr(1)} {r.choice(['=', '<>', '<', '<=', '>', '>='])} {sexpr(1)})" if r.random() < 0.8 else r.choice(["b", "(b IS NULL)", "(x IS NOT NULL)"])
    k = r.random()
    if k < 0.2: return f"(NOT {spred(d + 1)})"
    return f"({spred(d + 1)} {r.choice(['AND', 'OR'])} {spred(d + 1)})"
for qi in range(NQ // 2):
    e1, e2 = sexpr(), spred()
    sql = f"SELECT id, {e1}, {e2} FROM w ORDER BY id"
    compare(qi, sql, False, "VEC_SCALARS")
    sql = f"SELECT {e1}, s, dd FROM w WHERE {spred()} ORDER BY 1 NULLS FIRST, 2 NULLS FIRST, 3 NULLS FIRST"
    compare(qi, sql, True, "ROWS_BLOCK")                    # rows read from the planes, or grouped first
    if os.environ.get("VERBOSE") and qi < 3: print(sql)
    se = sexpr(1)
    sql = f"SELECT p, count(*), sum({se}) FROM w WHERE {spred()} GROUP BY p"
    compare(qi, sql, "y" not in se, "VEC_SCALARS")       # y holds 1e300 of both signs: DuckDB's double sum cancels

print("== joins, set operations, DISTINCT (against DuckDB)")
JK = ["p", "o", "q", "s", "dd", "b"]
for qi in range(NQ // 2):
    k = r.random()
    if k < 0.5:
        c1, c2 = r.choice(JK), r.choice(JK)
        if (c1 in ("q", "s")) != (c2 in ("q", "s")) or (c1 == "dd") != (c2 == "dd") or (c1 == "b") != (c2 == "b"): c2 = c1
        kind = r.choice(["JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"])
        cond = f"a.{c1} {r.choice(['=', '=', 'IS NOT DISTINCT FROM'])} b.{c2}"
        if r.random() < 0.3: cond += f" AND a.id {r.choice(['<', '>', '<>'])} b.id"
        lhs = r.choice(["w", "(SELECT * FROM w WHERE id % 3 = 0)", "(SELECT p, q, o, s, dd, b, id, x FROM w WHERE x > 0)"])
        sql = f"SELECT a.id, b.id, a.{c1}, b.x FROM {lhs} a {kind} w b ON {cond}"
        if r.random() < 0.4:
            sql = f"SELECT a.{c1}, count(*), count(b.id), sum(b.x) FROM {lhs} a {kind} w b ON {cond} GROUP BY a.{c1}"
    elif k < 0.85:
        c = r.sample(JK, r.choice([1, 2]))
        op = r.choice(["UNION", "UNION ALL", "INTERSECT", "INTERSECT ALL", "EXCEPT", "EXCEPT ALL"])
        w1 = r.choice(["", " WHERE x > 1", " WHERE id % 2 = 0"]); w2 = r.choice(["", " WHERE o < 5", " WHERE id % 3 = 0"])
        sql = f"SELECT {', '.join(c)} FROM w{w1} {op} SELECT {', '.join(c)} FROM w{w2}"
    else:
        c = r.sample(JK + ["o % 3", "x % 5"], r.choice([1, 2, 3]))
        sql = f"SELECT DISTINCT {', '.join(c)} FROM w" + r.choice(["", " WHERE b", " WHERE x IS NULL"])
    a, ea = run(sql, True, "VEC_SCALARS")
    try: want = con.execute(sql).fetchall()
    except Exception as x: duck_skip += 1; continue
    if ea is not None or not sqlfuzz.same(a, want, False, 1e-9):
        fails += 1; print(f"FAIL {qi} (vs DuckDB): {sql}\n   torbles {ea or (len(a), sorted(a, key=str)[:3])}\n   duckdb  {len(want), sorted(want, key=str)[:3]}")
    else: duck_ok += 1

print(f"{NQ} window, {NQ} GROUP BY, {NQ} scalar and {NQ // 2} join / set operation queries: arrays = rows on all{'' if not fails else ' but ' + str(fails)}, "
      f"{same_err} raised in both, {row_only} failed only row by row, {duck_ok} also equal to DuckDB, {duck_skip} DuckDB rejected")
for k, v in sorted(errs.items(), key=lambda x: -x[1]): print(f"  {v:4d}  {k}")
print("ALL MATCH" if not fails else f"{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
