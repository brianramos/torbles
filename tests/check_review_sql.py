# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Regression tests for the SQL findings of the 0.2.0 release review (2026-11-01):
  R02  null-safe joins (IS NOT DISTINCT FROM) keep their NULL matches: inner, LEFT, RIGHT, FULL, semi (EXISTS) and
       anti (NOT EXISTS) joins, composite keys mixing = and IS NOT DISTINCT FROM; the IN / = forms as controls;
  R03  casts and integer / DECIMAL arithmetic keep their SQL types on every path: a narrowing cast or an operation
       whose value leaves its type is DuckDB's conversion / overflow error in projections, aggregates (the engine's
       moment sums), predicates (plane predicates, fact code sets, dimension row masks), GROUP BY keys, joins
       (pre-aggregated ones too) and constant folding; widening casts, casts whose values fit, BIGINT arithmetic and
       an error in a row that no output needs are the passing controls, and so are the values DuckDB decides without
       computing the arithmetic (comparisons with constants, a filter term that drops the row, x * 0);
  R06  DISTINCT ON a collated key keeps one row per collation key, picked by the ORDER BY (under its collations);
and of the two independent reviews of the work after it (2026-11-02), on a table just past the default threshold of
the algebraic paths (2^16 rows), with no setting changed:
  F01  a window in the ORDER BY after its first key, not read from its series, keeps rank selection (which reads only
       the rows up to the k-th) off, with or without windows in the SELECT list that are read from their series;
  F02  frame offsets up to 2^63 - 1 on the series path; a window with two PARTITION BY columns over no rows.
Every case has a fixed expectation (rows, or ERROR where DuckDB raises), checked against DuckDB and against db.sql on
the engines alg, plain and gpu (TORBLES_SQL_ENGINE=<engine>: that engine only).
    python3 tests/check_review_sql.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decimal import Decimal
import numpy as np, pandas as pd
import torbles
from torbles import F

ERROR = "ERROR"
ENGINES = [os.environ["TORBLES_SQL_ENGINE"]] if os.environ.get("TORBLES_SQL_ENGINE") else ["alg", "plain", "gpu"]
fails = 0


def check(ok, what, detail=""):
    global fails
    if not ok: fails += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"\n        {detail}" if not ok and detail else ""))


def outcome(run):
    try: return [tuple(r) for r in run()]
    except Exception as ex: return (ERROR, f"{type(ex).__name__}: {str(ex).splitlines()[0][:140]}")


def same(got, want):
    if want is ERROR: return isinstance(got, tuple) and got[0] == ERROR
    return got == want


def cases(db, title, items):
    """items: (name, SQL, expected rows or ERROR): DuckDB first (it must agree with the expectation), then torbles"""
    print(f"== {title}")
    for name, sql, want in items:
        duck = outcome(lambda: db.con.execute(sql).fetchall())
        if not same(duck, want):
            check(False, f"{name}: DuckDB disagrees with the expectation", f"{sql}\n        DuckDB {duck}  expected {want}")
            continue
        bad = {}
        for e in ENGINES:
            got = outcome(lambda: db.sql(sql, engine=e).rows)
            if not same(got, want): bad[e] = got
        check(not bad, f"{name} [{', '.join(ENGINES)}]", f"{sql}\n        expected {want}\n        " +
              "\n        ".join(f"{e}: {g}" for e, g in bad.items()))


ddl = """CREATE TABLE t (i INTEGER);
CREATE TABLE v (i INTEGER, g INTEGER);
CREATE TABLE u (i INTEGER, j INTEGER, s SMALLINT, b BIGINT, ti TINYINT, ui UINTEGER, d DECIMAL(10,2));
CREATE TABLE w (i INTEGER, b BIGINT);
CREATE TABLE nn (i INTEGER, j INTEGER);
CREATE TABLE na (x INTEGER, y INTEGER);
CREATE TABLE nb (x INTEGER, y INTEGER);
CREATE TABLE p (k INTEGER, x INTEGER, y INTEGER);
CREATE TABLE dm (id INTEGER PRIMARY KEY, z INTEGER, w INTEGER, name VARCHAR);
CREATE TABLE f (id INTEGER, k INTEGER REFERENCES dm(id), x INTEGER, y INTEGER);
CREATE TABLE big (g INTEGER, x INTEGER, y INTEGER, m BIGINT);
CREATE TABLE nz (i INTEGER, n INTEGER);"""
rng = np.random.default_rng(2026)
nbig = 4000
bx = rng.integers(-30000, 30000, nbig); bx[17] = 50000                  # one value whose square leaves INTEGER
big = pd.DataFrame({"g": rng.integers(0, 40, nbig), "x": bx, "y": rng.integers(-40000, 40000, nbig), "m": rng.integers(-10**9, 10**9, nbig)})
big.loc[17, "g"] = 99                                                   # the only row of its group
frames = {
    "t": pd.DataFrame({"i": [100, 200, 200]}),                         # the review's table
    "v": pd.DataFrame({"i": pd.array([3, 50000, None], dtype="Int64"), "g": [1, 2, 1]}),
    "u": pd.DataFrame({"i": [2147483647, 16777217], "j": [1, 50000], "s": [32767, -32768], "b": [9223372036854775807, -9223372036854775808],
                       "ti": [127, -128], "ui": [1, 0], "d": [Decimal("99999999.99"), Decimal("-0.50")]}),
    "w": pd.DataFrame({"i": [-2147483648, 5], "b": [-9223372036854775808, 7]}),
    "nn": pd.DataFrame({"i": [2147483647, 5], "j": [2, 3]}),
    "na": pd.DataFrame({"x": pd.array([None, 1, 2, None], dtype="Int64"), "y": pd.array([None, 1, 2, 7], dtype="Int64")}),
    "nb": pd.DataFrame({"x": pd.array([None, 1, 3], dtype="Int64"), "y": pd.array([None, 1, 3], dtype="Int64")}),
    "p": pd.DataFrame({"k": [1, 1, 2, 2, 3], "x": [46341, 3, 70000, 2, 5], "y": [46341, 7, 1, 40000, 6]}),
    "dm": pd.DataFrame({"id": [1, 2, 3, 4], "z": [100, 200, 5, 300], "w": [5, 5, 6, 6], "name": ["a", "b", "c", "d"]}),
    "f": pd.DataFrame({"id": list(range(8)), "k": [1, 1, 2, 3, 3, 3, 1, 2], "x": [1, 2, 3, 70000, 5, 6, 7, 8], "y": [9, 8, 7, 40000, 5, 4, 3, 2]}),
    "big": big,
    "nz": pd.DataFrame({"i": [50000, 3, 7], "n": pd.array([None, 1, 2], dtype="Int64")}),     # i * i fails on the NULL row
}
for k, cols in {"u": {"i": "int32", "j": "int32", "s": "int16", "b": "int64", "ti": "int8", "ui": "uint32"}}.items():
    frames[k] = frames[k].astype(cols)
db = torbles.from_ddl(ddl, frames, keep_plain=True)

# ---------------------------------------------------------------- R02: null-safe joins
A = "(VALUES (NULL), (1), (2)) a(x)"
B = "(VALUES (NULL), (1), (3)) b(x)"
A2 = "(VALUES (1, NULL), (1, 2), (NULL, 3), (2, NULL)) a(p, q)"
B2 = "(VALUES (1, NULL), (1, 2), (NULL, 3), (2, 5)) b(p, q)"
cases(db, "R02 null-safe joins keep NULL matches", [
    ("the review's inner join", "SELECT a.x,b.x FROM (VALUES (NULL),(1)) a(x) JOIN (VALUES (NULL),(1)) b(x) ON a.x IS NOT DISTINCT FROM b.x ORDER BY 1 NULLS FIRST",
     [(None, None), (1, 1)]),
    ("the review's FULL join", "SELECT a.x,b.x FROM (VALUES (NULL),(1)) a(x) FULL JOIN (VALUES (NULL),(1)) b(x) ON a.x IS NOT DISTINCT FROM b.x ORDER BY 1 NULLS FIRST",
     [(None, None), (1, 1)]),
    ("control: ordinary equality never matches NULL", "SELECT a.x,b.x FROM (VALUES (NULL),(1)) a(x) JOIN (VALUES (NULL),(1)) b(x) ON a.x=b.x ORDER BY 1 NULLS FIRST",
     [(1, 1)]),
    ("inner", f"SELECT a.x, b.x FROM {A} JOIN {B} ON a.x IS NOT DISTINCT FROM b.x ORDER BY 1 NULLS FIRST", [(None, None), (1, 1)]),
    ("LEFT", f"SELECT a.x, b.x FROM {A} LEFT JOIN {B} ON a.x IS NOT DISTINCT FROM b.x ORDER BY 1 NULLS FIRST",
     [(None, None), (1, 1), (2, None)]),
    ("RIGHT", f"SELECT a.x, b.x FROM {A} RIGHT JOIN {B} ON a.x IS NOT DISTINCT FROM b.x ORDER BY 2 NULLS FIRST",
     [(None, None), (1, 1), (None, 3)]),
    ("FULL", f"SELECT a.x, b.x FROM {A} FULL JOIN {B} ON a.x IS NOT DISTINCT FROM b.x ORDER BY 1 NULLS FIRST, 2 NULLS FIRST",
     [(None, None), (None, 3), (1, 1), (2, None)]),
    ("control: FULL with =", f"SELECT a.x, b.x FROM {A} FULL JOIN {B} ON a.x = b.x ORDER BY 1 NULLS FIRST, 2 NULLS FIRST",
     [(None, None), (None, None), (None, 3), (1, 1), (2, None)]),
    ("semi join (EXISTS)", f"SELECT a.x FROM {A} WHERE EXISTS (SELECT 1 FROM {B} WHERE a.x IS NOT DISTINCT FROM b.x) ORDER BY 1 NULLS FIRST",
     [(None,), (1,)]),
    ("anti join (NOT EXISTS)", f"SELECT a.x FROM {A} WHERE NOT EXISTS (SELECT 1 FROM {B} WHERE a.x IS NOT DISTINCT FROM b.x) ORDER BY 1 NULLS FIRST",
     [(2,)]),
    ("control: semi join with =", f"SELECT a.x FROM {A} WHERE EXISTS (SELECT 1 FROM {B} WHERE a.x = b.x) ORDER BY 1 NULLS FIRST", [(1,)]),
    ("control: anti join with =", f"SELECT a.x FROM {A} WHERE NOT EXISTS (SELECT 1 FROM {B} WHERE a.x = b.x) ORDER BY 1 NULLS FIRST",
     [(None,), (2,)]),
    ("control: IN (subquery), NULL never in", f"SELECT a.x FROM {A} WHERE a.x IN (SELECT b.x FROM {B}) ORDER BY 1", [(1,)]),
    ("control: NOT IN with a NULL in the list is never TRUE", f"SELECT a.x FROM {A} WHERE a.x NOT IN (SELECT b.x FROM {B})", []),
    ("composite = and IS NOT DISTINCT FROM", f"SELECT a.p, a.q, b.p, b.q FROM {A2} JOIN {B2} ON a.p = b.p AND a.q IS NOT DISTINCT FROM b.q ORDER BY 1, 2 NULLS FIRST",
     [(1, None, 1, None), (1, 2, 1, 2)]),
    ("composite IS NOT DISTINCT FROM and =", f"SELECT a.p, a.q, b.p, b.q FROM {A2} JOIN {B2} ON a.p IS NOT DISTINCT FROM b.p AND a.q = b.q ORDER BY 1 NULLS FIRST, 2",
     [(None, 3, None, 3), (1, 2, 1, 2)]),
    ("composite, LEFT", f"SELECT a.p, a.q, b.p, b.q FROM {A2} LEFT JOIN {B2} ON a.p = b.p AND a.q IS NOT DISTINCT FROM b.q ORDER BY 1 NULLS FIRST, 2 NULLS FIRST",
     [(None, 3, None, None), (1, None, 1, None), (1, 2, 1, 2), (2, None, None, None)]),
    ("composite, FULL", f"SELECT a.p, a.q, b.p, b.q FROM {A2} FULL JOIN {B2} ON a.p IS NOT DISTINCT FROM b.p AND a.q IS NOT DISTINCT FROM b.q "
     "ORDER BY 1 NULLS FIRST, 2 NULLS FIRST, 3 NULLS FIRST, 4 NULLS FIRST",
     [(None, None, 2, 5), (None, 3, None, 3), (1, None, 1, None), (1, 2, 1, 2), (2, None, None, None)]),
    ("composite, semi join", f"SELECT a.p, a.q FROM {A2} WHERE EXISTS (SELECT 1 FROM {B2} WHERE a.p = b.p AND a.q IS NOT DISTINCT FROM b.q) ORDER BY 1, 2 NULLS FIRST",
     [(1, None), (1, 2)]),
    ("composite, anti join", f"SELECT a.p, a.q FROM {A2} WHERE NOT EXISTS (SELECT 1 FROM {B2} WHERE a.p IS NOT DISTINCT FROM b.p AND a.q = b.q) "
     "ORDER BY 1 NULLS FIRST, 2 NULLS FIRST", [(1, None), (2, None)]),
    ("stored tables, inner", "SELECT na.x, nb.x FROM na JOIN nb ON na.x IS NOT DISTINCT FROM nb.x ORDER BY 1 NULLS FIRST",
     [(None, None), (None, None), (1, 1)]),
    ("stored tables, FULL", "SELECT na.x, nb.x FROM na FULL JOIN nb ON na.x IS NOT DISTINCT FROM nb.x ORDER BY 1 NULLS FIRST, 2 NULLS FIRST",
     [(None, None), (None, None), (None, 3), (1, 1), (2, None)]),
    ("stored tables, composite", "SELECT na.x, na.y, nb.y FROM na JOIN nb ON na.x IS NOT DISTINCT FROM nb.x AND na.y = nb.y ORDER BY 1 NULLS FIRST",
     [(1, 1, 1)]),
    ("stored tables, aggregate over the join", "SELECT count(*), count(na.x) FROM na JOIN nb ON na.x IS NOT DISTINCT FROM nb.x", [(3, 1)]),
])

# ---------------------------------------------------------------- R03: casts and bound integer / DECIMAL arithmetic
cases(db, "R03 the review's cases (t.i = 100, 200, 200)", [
    ("SUM of a narrowing cast", "SELECT sum(CAST(i AS TINYINT)) FROM t", ERROR),
    ("SUM of a narrowing DECIMAL cast", "SELECT sum(CAST(i AS DECIMAL(2,0))) FROM t", ERROR),
    ("predicate on a narrowing cast", "SELECT count(*) FROM t WHERE CAST(i AS TINYINT)>0", ERROR),
    ("control: the scalar cast", "SELECT CAST(i AS TINYINT) FROM t", ERROR),
    ("control: SUM of a widening cast", "SELECT sum(CAST(i AS BIGINT)) FROM t", [(500,)]),
    ("INTEGER constant arithmetic", "SELECT CAST(2147483647 AS INTEGER)+1", ERROR),
    ("control: BIGINT constant arithmetic", "SELECT CAST(2147483647 AS BIGINT)+1", [(2147483648,)]),
    ("INTEGER multiplication inside SUM", "SELECT sum(i*j) FROM u", ERROR),
    ("INTEGER square inside SUM", "SELECT sum(i*i) FROM v", ERROR),
])
cases(db, "R03 casts: proved, or evaluated with their errors", [
    ("COUNT of a narrowing cast", "SELECT count(CAST(i AS TINYINT)) FROM t", ERROR),
    ("MIN of a narrowing cast", "SELECT min(CAST(i AS TINYINT)) FROM t", ERROR),
    ("AVG of a narrowing cast", "SELECT avg(CAST(i AS TINYINT)) FROM t", ERROR),
    ("HAVING on a SUM of a narrowing cast", "SELECT 1 FROM t HAVING sum(CAST(i AS TINYINT)) > 0", ERROR),
    ("GROUP BY a narrowing cast", "SELECT count(*) FROM t GROUP BY CAST(i AS TINYINT)", ERROR),
    ("GROUP BY a narrowing cast, shown", "SELECT CAST(i AS TINYINT) k, count(*) FROM t GROUP BY k", ERROR),
    ("SUM per group of a narrowing cast", "SELECT sum(CAST(i AS TINYINT)) FROM t GROUP BY i ORDER BY 1", ERROR),
    ("DISTINCT of a narrowing cast", "SELECT DISTINCT CAST(i AS TINYINT) FROM t", ERROR),
    ("ORDER BY a narrowing cast", "SELECT i FROM t ORDER BY CAST(i AS TINYINT) LIMIT 1", ERROR),
    ("a narrowing cast in a scalar subquery's WHERE", "SELECT (SELECT count(*) FROM t WHERE CAST(i AS TINYINT) > 0)", ERROR),
    ("a narrowing cast in a derived table", "SELECT sum(i) FROM (SELECT CAST(i AS TINYINT) i FROM t)", ERROR),
    ("a narrowing cast as a join key", "SELECT count(*) FROM t a JOIN t b ON CAST(a.i AS TINYINT) = b.i", ERROR),
    ("... of a LEFT JOIN", "SELECT count(*) FROM t a LEFT JOIN t b ON CAST(a.i AS TINYINT) = b.i", ERROR),
    ("BETWEEN on a narrowing cast", "SELECT count(*) FROM t WHERE CAST(i AS TINYINT) BETWEEN 0 AND 120", ERROR),
    ("IN on a narrowing cast", "SELECT count(*) FROM t WHERE CAST(i AS TINYINT) IN (100, 101)", ERROR),
    ("CAST to a DECIMAL without the integer digits", "SELECT sum(CAST(i AS DECIMAL(3,1))) FROM t", ERROR),
    ("predicate on a narrowing DECIMAL cast", "SELECT count(*) FROM t WHERE CAST(i AS DECIMAL(2,0)) > 0", ERROR),
    ("IS NULL of a narrowing cast", "SELECT count(*) FROM t WHERE CAST(i AS TINYINT) IS NULL", ERROR),
    ("IS NOT NULL of a narrowing cast", "SELECT count(*) FROM t WHERE CAST(i AS TINYINT) IS NOT NULL", ERROR),
    ("IS NULL of arithmetic on a narrowing cast", "SELECT count(*) FROM nn WHERE CAST(i AS SMALLINT) + 1 IS NULL", ERROR),
    ("IS NULL of a COALESCE of a narrowing cast", "SELECT count(*) FROM nn WHERE coalesce(CAST(i AS TINYINT), 0) IS NULL", ERROR),
    ("control: the rows that fail are excluded first", "SELECT sum(CAST(i AS TINYINT)) FROM t WHERE i < 150", [(100,)]),
    ("control: another term excludes the failing rows", "SELECT count(*) FROM t WHERE i < 150 AND CAST(i AS TINYINT) > 0", [(1,)]),
    ("control: ... in either order", "SELECT count(*) FROM t WHERE CAST(i AS TINYINT) > 0 AND i < 150", [(1,)]),
    ("control: a narrowing cast every value fits", "SELECT sum(CAST(i AS SMALLINT)), count(*) FILTER (WHERE CAST(i AS SMALLINT) > 150) FROM t", [(500, 2)]),
    ("control: predicate on a narrowing cast every value fits", "SELECT count(*) FROM t WHERE CAST(i AS SMALLINT) > 150", [(2,)]),
    ("control: predicate on a widening cast", "SELECT count(*) FROM t WHERE CAST(i AS BIGINT) BETWEEN 150 AND 250", [(2,)]),
    ("control: DECIMAL casts with the digits", "SELECT sum(CAST(i AS DECIMAL(3,0))), sum(CAST(i AS DECIMAL(4,1))) FROM t",
     [(Decimal("500"), Decimal("500.0"))]),
    ("control: TRY_CAST gives NULL", "SELECT sum(TRY_CAST(i AS TINYINT)), count(TRY_CAST(i AS TINYINT)) FROM t", [(100, 1)]),
    ("control: CASE evaluates the cast only where taken", "SELECT sum(CASE WHEN i > 150 THEN 0 ELSE CAST(i AS TINYINT) END) FROM t", [(100,)]),
    ("control: IS NULL, the failing rows excluded first", "SELECT count(*) FROM t WHERE i < 150 AND CAST(i AS TINYINT) IS NULL", [(0,)]),
    ("control: IS NULL of arithmetic on columns without NULLs (DuckDB decides it from its statistics)",
     "SELECT count(*) FROM nn WHERE (i + 1) IS NOT NULL", [(2,)]),
    ("control: IS NULL of a CASE whose narrowing cast is in a WHEN condition only",
     "SELECT count(*) FROM nn WHERE CASE WHEN CAST(i AS TINYINT) > 0 THEN 1 ELSE 0 END IS NULL", [(0,)]),
])
cases(db, "R03 integer and DECIMAL operations keep their types", [
    ("INTEGER + literal", "SELECT i + 1 FROM u", ERROR),
    ("INTEGER + INTEGER", "SELECT i + j FROM u", ERROR),
    ("INTEGER * INTEGER", "SELECT i * j FROM u", ERROR),
    ("INTEGER - (-1)", "SELECT i - (-1) FROM u", ERROR),
    ("negation of the least INTEGER", "SELECT -i FROM w", ERROR),
    ("0 - the least INTEGER", "SELECT 0 - i FROM w", ERROR),
    ("ABS of the least INTEGER", "SELECT abs(i) FROM w", ERROR),
    ("the least INTEGER // -1", "SELECT i // -1 FROM w", ERROR),
    ("the least INTEGER % -1", "SELECT i % -1 FROM w", ERROR),
    ("SMALLINT + SMALLINT", "SELECT s + s FROM u", ERROR),
    ("SMALLINT * literal", "SELECT s * 2 FROM u", ERROR),
    ("TINYINT + literal", "SELECT ti + 1 FROM u", ERROR),
    ("BIGINT + literal", "SELECT b + 1 FROM u", ERROR),
    ("BIGINT negation", "SELECT b * -1 FROM w", ERROR),
    ("UINTEGER below zero", "SELECT ui - 2 FROM u", ERROR),
    ("SMALLINT casts multiplied", "SELECT CAST(i AS SMALLINT) * CAST(i AS SMALLINT) FROM t", ERROR),
    ("DECIMAL(18) product", "SELECT d * d FROM u", ERROR),
    ("INTEGER literals", "SELECT 2147483647 + 1", ERROR),
    ("INTEGER constants multiplied", "SELECT 100000 * 100000", ERROR),
    ("DECIMAL constants multiplied", "SELECT CAST(99999999.99 AS DECIMAL(10,2)) * CAST(99999999.99 AS DECIMAL(10,2))", ERROR),
    ("INTEGER left shift", "SELECT i << 1 FROM u", ERROR),
    ("INTEGER ROUND to tens", "SELECT round(i, -1) FROM u", ERROR),
    ("TINYINT ROUND to tens", "SELECT round(ti, -1) FROM u", ERROR),
    ("BIGINT LCM", "SELECT lcm(CAST(2147483647 AS BIGINT) * 4294967296, 3)", ERROR),
    ("HUGEINT FACTORIAL", "SELECT factorial(40)", ERROR),
    ("control: shifts, ROUND, LCM, FACTORIAL in range", "SELECT CAST(1 AS INTEGER) << 30, round(2147483641, -1), lcm(4, 6), factorial(20)",
     [(1073741824, 2147483640, 12, 2432902008176640000)]),
    ("control: the least INTEGER constant negated is a BIGINT", "SELECT -(CAST(-2147483648 AS INTEGER))", [(2147483648,)]),
    ("control: BIGINT arithmetic", "SELECT CAST(i AS BIGINT) * j, CAST(i AS BIGINT) + 1 FROM u ORDER BY 1", [(2147483647, 2147483648), (838860850000, 16777218)]),
    ("control: HUGEINT arithmetic", "SELECT CAST(b AS HUGEINT) * 2 FROM u ORDER BY 1", [(-18446744073709551616,), (18446744073709551614,)]),
    ("control: a folded constant keeps its type", "SELECT ti + (1 + 1), ti + CAST(1 AS INTEGER) FROM u ORDER BY 1", [(-126, -127), (129, 128)]),
    ("control: DECIMAL product with the digits", "SELECT CAST(d AS DECIMAL(20,2)) * d FROM u ORDER BY 1", [(Decimal("0.2500"),), (Decimal("9999999998000000.0001"),)]),
    ("negation in the CASE branch the row takes", "SELECT CASE WHEN i > 0 THEN i ELSE -i END FROM w ORDER BY 1", ERROR),
    ("control: CASE takes the branch that fits", "SELECT CASE WHEN i < 0 THEN 0 ELSE i + 1 END FROM w ORDER BY 1", [(0,), (6,)]),
    ("control: the overflowing row filtered out", "SELECT -i, i * 2, abs(i) FROM w WHERE i > 0", [(-5, 10, 5)]),
])
cases(db, "R03 comparisons with constants (DuckDB decides some without computing their arithmetic)", [
    ("x * c1 = c2, c1 not dividing c2", "SELECT count(*) FROM nn WHERE i * 7 = -3", [(0,)]),
    ("x * c1 <> c2, c1 not dividing c2", "SELECT count(*) FROM nn WHERE i * 7 <> -3", [(2,)]),
    ("x + c1 > c2", "SELECT count(*) FROM nn WHERE i + 1 > 5", [(2,)]),
    ("c1 - x < c2", "SELECT count(*) FROM nn WHERE 1 - i < 2", [(2,)]),
    ("x * -c1 > c2", "SELECT count(*) FROM nn WHERE i * -7 > 14", [(0,)]),
    ("-x < c", "SELECT count(*) FROM w WHERE -i < 0", [(1,)]),
    ("BETWEEN of x + c", "SELECT count(*) FROM nn WHERE i + 1 BETWEEN 5 AND 10", [(1,)]),
    ("a widening cast of x + c", "SELECT count(*) FROM nn WHERE CAST(i + 1 AS BIGINT) > 5", [(2,)]),
    ("projected", "SELECT i + 1 > 5, i * 7 = -3 FROM nn ORDER BY 1", [(True, False), (True, False)]),
    ("a NULL constant", "SELECT (i * i) + NULL, (i * i) > NULL FROM nn", [(None, None), (None, None)]),
    ("x * c1 > c2, c1 not dividing c2: computed", "SELECT count(*) FROM nn WHERE i * 7 > 3", ERROR),
    ("x + c1 >= c2, c2 - c1 outside the type: computed", "SELECT count(*) FROM nn WHERE i + 1 >= -2147483648", ERROR),
    ("IN: computed", "SELECT count(*) FROM nn WHERE (i + 1) IN (6)", ERROR),
    ("x + x: computed", "SELECT count(*) FROM nn WHERE i + i > 5", ERROR),
    ("an integer literal wider than x: computed", "SELECT count(*) FROM nn WHERE i * 2 = 2147483648", ERROR),
    ("x * c1 <> c2 of arithmetic that overflows: computed (constant_or_null)", "SELECT count(*) FROM nn WHERE (i + 1) * 7 <> -3", ERROR),
    ("x * c1 = c2 of arithmetic that overflows, projected: computed", "SELECT (i + 1) * 7 = -3 FROM nn", ERROR),
    ("... under IS NULL: computed", "SELECT count(*) FROM nn WHERE ((i + 1) * 7 = -3) IS NULL", ERROR),
    ("control: ... as a filter (no row can pass)", "SELECT count(*) FROM nn WHERE (i + 1) * 7 = -3", [(0,)]),
    ("NOT of an empty BETWEEN of x + c (its two comparisons drop the arithmetic)", "SELECT count(*) FROM nn WHERE NOT ((i + 1) BETWEEN 10 AND 5)",
     [(2,)]),
    ("an empty BETWEEN of x * x, projected: computed", "SELECT (i * i) BETWEEN 10 AND 5 FROM nn", ERROR),
    ("NOT of a comparison is the opposite one: NOT (x * c1 <> c2) is FALSE or NULL", "SELECT count(*) FROM nn WHERE NOT ((i + 1) * 7 <> -3)",
     [(0,)]),
    ("... NOT (x * c1 = c2): computed", "SELECT count(*) FROM nn WHERE NOT ((i + 1) * 7 = -3)", ERROR),
    ("... NOT NOT (x * c1 = c2)", "SELECT count(*) FROM nn WHERE NOT (NOT ((i + 1) * 7 = -3))", [(0,)]),
    ("a product's constants fold into one, placed last: (c * b) * x = k is (b * x) * c = k", "SELECT count(*) FROM u WHERE (32767 * b) * ui = 46341",
     [(0,)]),
    ("... computed in that order", "SELECT (32767 * b) * ui FROM u WHERE ui = 0", [(0,)]),
    ("... x * 0 is NULL or 0 without computing x", "SELECT (b * 10) * 0 FROM u ORDER BY 1", [(0,), (0,)]),
    ("... (x * y) * 0 computes x * y", "SELECT (b * b) * 0 FROM u", ERROR),
    ("control: a rewritten comparison stays its own grouping column", "SELECT i + 1 > 5, i > 4, count(*) FROM nn GROUP BY ROLLUP (i + 1 > 5, i > 4) "
     "ORDER BY 1 NULLS FIRST, 2 NULLS FIRST", [(None, None, 2), (True, None, 2), (True, True, 2)]),
    ("control: x and x * 1 are two grouping columns", "SELECT j, j * 1, count(*) FROM nn GROUP BY ROLLUP (j, j * 1) ORDER BY 1 NULLS FIRST, 2 NULLS FIRST",
     [(None, None, 2), (2, None, 1), (2, 2, 1), (3, None, 1), (3, 3, 1)]),
    ("control: ... as a filter (no row can pass)", "SELECT count(*) FROM nn WHERE (i * i) BETWEEN 10 AND 5", [(0,)]),
])
cases(db, "R03 inside aggregates, predicates and GROUP BY (the engine's polynomials)", [
    ("SUM(i * i)", "SELECT sum(i * i) FROM u", ERROR),
    ("SUM(i + i)", "SELECT sum(i + i) FROM u", ERROR),
    ("AVG(i * i)", "SELECT avg(i * i) FROM u", ERROR),
    ("MIN(i * i)", "SELECT min(i * i) FROM v", ERROR),
    ("MAX(i * i)", "SELECT max(i * i) FROM v", ERROR),
    ("VAR_POP(i * i)", "SELECT var_pop(i * i) FROM v", ERROR),
    ("SUM(i * 100000000)", "SELECT sum(i * 100000000) FROM t", ERROR),
    ("SUM with a FILTER (the argument is evaluated)", "SELECT sum(i * i) FILTER (WHERE i < 100) FROM v", ERROR),
    ("COUNT(DISTINCT i * i)", "SELECT count(DISTINCT i * i) FROM v", ERROR),
    ("COUNT of a CASE", "SELECT count(CASE WHEN i > 0 THEN i * i END) FROM v", ERROR),
    ("window SUM(i * i)", "SELECT sum(i * i) OVER () FROM v", ERROR),
    ("SUM per group", "SELECT g, sum(i * i) FROM v GROUP BY g ORDER BY g", ERROR),
    ("GROUP BY i * i", "SELECT count(*) FROM v GROUP BY i * i", ERROR),
    ("GROUP BY i * i, shown", "SELECT i * i k, count(*) FROM v GROUP BY k", ERROR),
    ("WHERE i * i > 0", "SELECT count(*) FROM v WHERE i * i > 0", ERROR),
    ("ORDER BY i * i", "SELECT i FROM v ORDER BY i * i", ERROR),
    ("a derived table's product", "SELECT sum(x) FROM (SELECT i * i x FROM v)", ERROR),
    ("SUM of a product, 4000 rows", "SELECT sum(x * x) FROM big", ERROR),
    ("SUM of a product per group, 4000 rows", "SELECT g, sum(x * x) FROM big GROUP BY g ORDER BY g", ERROR),
    ("WHERE on a product, 4000 rows", "SELECT count(*) FROM big WHERE x * x > 100", ERROR),
    ("control: SUM(i * i) where it fits", "SELECT sum(i * i) FROM v WHERE i < 100", [(9,)]),
    ("control: SUM(i * i) of the group a HAVING keeps", "SELECT g, sum(i * i) FROM v GROUP BY g HAVING g = 1", [(1, 9)]),
    ("control: SUM of a CASE that skips the overflow", "SELECT sum(CASE WHEN i < 100 THEN i * i ELSE 0 END) FROM v", [(9,)]),
    ("control: another term excludes the overflow", "SELECT count(*) FROM v WHERE i * i > 0 AND i < 100", [(1,)]),
    ("control: BIGINT products", "SELECT sum(CAST(i AS BIGINT) * i), sum(b) FROM u", [(4611967489142685698, -1)]),
    ("control: COUNT of arithmetic counts its columns' rows (no NULLs)", "SELECT count(i * i), count(i + j) FROM nn", [(2, 2)]),
    ("control: 4000 rows without the overflowing one", "SELECT sum(x * x), sum(x * 2 + y), count(*) FROM big WHERE g <> 99",
     [(int((bx[np.arange(nbig) != 17].astype(object) ** 2).sum()),
       int((bx * 2 + big["y"].to_numpy())[np.arange(nbig) != 17].astype(object).sum()), nbig - 1)]),
    ("4000 rows, the overflowing group kept by HAVING", "SELECT g, sum(x * x) FROM big GROUP BY g HAVING g = 99 OR g = 3 ORDER BY g", ERROR),
    ("control: 4000 rows, the overflowing group dropped by HAVING", "SELECT g, sum(x * x) FROM big GROUP BY g HAVING g = 3",
     [(3, int((bx[big["g"].to_numpy() == 3].astype(object) ** 2).sum()))]),
    ("control: 4000 rows, its group dropped by HAVING", "SELECT count(*) FROM (SELECT g, sum(x * x) s FROM big GROUP BY g HAVING g < 40)", [(40,)]),
    ("control: 4000 rows, a HAVING on the keys drops the rows of a failing WHERE term",
     "SELECT g, count(*) FROM big WHERE x * x > 0 GROUP BY g HAVING g = 3", [(3, int(((big["g"] == 3) & (big["x"] != 0)).sum()))]),
    ("4000 rows, a HAVING term on an aggregate does not", "SELECT g, count(*) FROM big WHERE x * x > 0 GROUP BY g HAVING g = 3 OR count(*) > 0 ORDER BY g",
     ERROR),
    ("control: SSB-like products that fit", "SELECT sum(m * 4), sum(x * y), sum(x - y) FROM big WHERE g < 40",
     [(int((big["m"].to_numpy()[big["g"].to_numpy() < 40].astype(object) * 4).sum()),
       int((bx.astype(object) * big["y"].to_numpy().astype(object))[big["g"].to_numpy() < 40].sum()),
       int((bx - big["y"].to_numpy())[big["g"].to_numpy() < 40].astype(object).sum()))]),
])
cases(db, "R03 a filter drops a row on a FALSE or NULL term before computing the others", [
    ("an AND inside an OR, NULL on the failing row", "SELECT count(*) FROM nz WHERE (n > 0 AND i * i > 0) OR i = 7", [(2,)]),
    ("the same condition as a value (computed in full)", "SELECT (n > 0 AND i * i > 0) OR i = 7 FROM nz", ERROR),
    ("NOT of the AND (computed in full)", "SELECT count(*) FROM nz WHERE NOT (n > 0 AND i * i > 0)", ERROR),
    ("the failing row passes the other terms", "SELECT count(*) FROM nz WHERE n IS NULL AND i * i > 0", ERROR),
    ("a CASE's WHEN is a filter too", "SELECT sum(CASE WHEN n > 0 AND i * i > 0 THEN 1 ELSE 0 END) FROM nz", [(2,)]),
    ("a CASE's WHEN, an AND inside an OR", "SELECT CASE WHEN (n > 0 AND i * i > 0) OR i = 7 THEN 1 ELSE 0 END x FROM nz ORDER BY x",
     [(0,), (1,), (1,)]),
    ("control: a THEN the failing row reaches", "SELECT sum(CASE WHEN n IS NULL THEN i * i ELSE 0 END) FROM nz", ERROR),
])
cases(db, "R03 a star join: fact and dimension columns", [
    ("SUM over a fact product", "SELECT sum(f.x * f.y) FROM f JOIN dm ON f.k = dm.id", ERROR),
    ("SUM over a fact and dimension product", "SELECT sum(f.x * dm.z * 10000) FROM f JOIN dm ON f.k = dm.id", ERROR),
    ("GROUP BY a dimension, SUM of a product", "SELECT dm.name, sum(f.x * f.y) FROM f JOIN dm ON f.k = dm.id GROUP BY 1 ORDER BY 1", ERROR),
    ("a dimension's narrowing cast in WHERE", "SELECT count(*) FROM f JOIN dm ON f.k = dm.id WHERE CAST(dm.z AS TINYINT) > 0", ERROR),
    ("IS NULL of a dimension's narrowing cast", "SELECT count(*) FROM f JOIN dm ON f.k = dm.id WHERE CAST(dm.z AS TINYINT) IS NULL", ERROR),
    ("a LEFT JOIN's SUM of a product", "SELECT sum(f.x * f.y) FROM f LEFT JOIN dm ON f.k = dm.id AND dm.w = 5", ERROR),
    ("control: the overflowing row excluded", "SELECT sum(f.x * f.y), sum(f.x * dm.z) FROM f JOIN dm ON f.k = dm.id WHERE f.id <> 3", [(132, 3255)]),
    ("control: the dimension rows that fail are excluded first", "SELECT count(*) FROM f JOIN dm ON f.k = dm.id WHERE dm.id IN (1, 3) AND CAST(dm.z AS TINYINT) > 0",
     [(6,)]),
    ("control: ... under IS NOT NULL", "SELECT count(*) FROM f JOIN dm ON f.k = dm.id WHERE dm.id IN (1, 3) AND CAST(dm.z AS TINYINT) IS NOT NULL",
     [(6,)]),
    ("control: GROUP BY a dimension, the group dropped", "SELECT dm.name, sum(f.x * f.y) FROM f JOIN dm ON f.k = dm.id GROUP BY 1 HAVING dm.name <> 'c' ORDER BY 1",
     [("a", 46), ("b", 37)]),
])
cases(db, "R03 a join without a key (pre-aggregated sides)", [
    ("an overflowing join key", "SELECT count(*) FROM u a JOIN u b ON a.i + 1 = b.j", ERROR),
    ("an overflowing LEFT JOIN key", "SELECT count(*) FROM u a LEFT JOIN u b ON a.i + 1 = b.j", ERROR),
    ("an overflowing FULL JOIN key", "SELECT count(*) FROM u a FULL JOIN u b ON a.i + 1 = b.j", ERROR),
    ("control: a join term decided without its arithmetic", "SELECT count(*) FROM u a JOIN u b ON a.j = b.j AND a.i * 2 > 0", [(2,)]),
    ("SUM of a product of the two sides", "SELECT sum(a.x * b.y) FROM p a JOIN p b ON a.k = b.k", ERROR),
    ("SUM of a product of one side", "SELECT sum(a.x * a.y) FROM p a JOIN p b ON a.k = b.k", ERROR),
    ("per group", "SELECT a.k, sum(a.x * b.y) FROM p a JOIN p b ON a.k = b.k GROUP BY a.k ORDER BY 1", ERROR),
    ("CTE sides", "WITH c AS (SELECT k, max(x) AS x, max(y) AS y FROM p GROUP BY k) SELECT sum(c1.x * c2.y) FROM c c1 JOIN c c2 ON c1.k = c2.k", ERROR),
    ("control: where the products fit", "SELECT sum(a.x * b.y), sum(a.x * a.y * b.y) FROM p a JOIN p b ON a.k = b.k WHERE a.k = 3", [(30, 180)]),
    ("control: the group a HAVING keeps", "SELECT a.k, sum(a.x * b.y) FROM p a JOIN p b ON a.k = b.k GROUP BY a.k HAVING a.k = 3", [(3, 30)]),
    ("control: a widened product", "SELECT sum(CAST(a.x AS BIGINT) * b.y) FROM p a JOIN p b ON a.k = b.k", [(4948101744,)]),
    ("control: CTE sides where the products fit", "WITH c AS (SELECT k, max(x) AS x, max(y) AS y FROM p GROUP BY k) "
     "SELECT sum(c1.x * c2.y) FROM c c1 JOIN c c2 ON c1.k = c2.k WHERE c1.k = 3", [(30,)]),
])
print("== R03 the Python API (db.select)")
tt, uu = db.table("t"), db.table("u")
api = [("SUM of a narrowing cast", db.select(F.sum(tt.i.cast("TINYINT"))), ERROR),
       ("predicate on a narrowing cast", db.select(F.count()).where(tt.i.cast("TINYINT") > 0), ERROR),
       ("SUM(i * j)", db.select(F.sum(uu.i * uu.j)), ERROR),
       ("COUNT(i * j) counts its columns' rows", db.select(F.count(uu.i * uu.j)), [(2,)]),
       ("control: SUM of a widening cast", db.select(F.sum(tt.i.cast("BIGINT"))), [(500,)])]
for name, q, want in api:
    duck = outcome(lambda: db.con.execute(q.to_sql()).fetchall())
    bad = {"duckdb": duck} if not same(duck, want) else {}
    for e in ENGINES:
        got = outcome(lambda: q.run(e).rows)
        if not same(got, want): bad[e] = got
    check(not bad, f"{name} [duckdb, {', '.join(ENGINES)}]", "\n        ".join(f"{e}: {g}" for e, g in bad.items()))

# ---------------------------------------------------------------- R06: DISTINCT ON a collated key
cases(db, "R06 DISTINCT ON under a collation", [
    ("the review's case", "SELECT DISTINCT ON (x COLLATE NOCASE) x FROM (VALUES('a'),('A')) t(x) ORDER BY x COLLATE NOCASE,x", [("A",)]),
    ("control: DISTINCT under a collation", "SELECT DISTINCT x COLLATE NOCASE FROM (VALUES('a'),('A')) t(x)", [("a",)]),
    ("the ORDER BY picks the row, ties by the next key", "SELECT DISTINCT ON (x COLLATE NOCASE) x, y FROM (VALUES ('a', 1), ('A', 2)) t(x, y) "
     "ORDER BY x COLLATE NOCASE, y", [("a", 1)]),
    ("... descending", "SELECT DISTINCT ON (x COLLATE NOCASE) x, y FROM (VALUES ('a', 1), ('A', 2)) t(x, y) ORDER BY x COLLATE NOCASE, y DESC",
     [("A", 2)]),
    ("two keys of two rows each", "SELECT DISTINCT ON (x COLLATE NOCASE) x, y FROM (VALUES ('a', 1), ('A', 2), ('b', 3), ('B', 0)) t(x, y) "
     "ORDER BY x COLLATE NOCASE, y", [("a", 1), ("B", 0)]),
    ("by position", "SELECT DISTINCT ON (1) x COLLATE NOCASE, y FROM (VALUES ('a', 1), ('A', 2), ('b', 3), ('B', 0)) t(x, y) ORDER BY 1, y DESC",
     [("A", 2), ("b", 3)]),
    ("NOACCENT", "SELECT DISTINCT ON (x COLLATE NOACCENT) x FROM (VALUES ('é'), ('e'), ('f')) t(x) ORDER BY x COLLATE NOACCENT, x",
     [("e",), ("f",)]),
    ("control: without a collation", "SELECT DISTINCT ON (x) x, y FROM (VALUES ('a', 1), ('A', 2)) t(x, y) ORDER BY x, y", [("A", 2), ("a", 1)]),
    ("control: DISTINCT ON an expression", "SELECT DISTINCT ON (lower(x)) x FROM (VALUES('a'),('A')) t(x) ORDER BY lower(x), x", [("A",)]),
    ("a stored column", "SELECT DISTINCT ON (name COLLATE NOCASE) name FROM (SELECT upper(name) AS name FROM dm UNION ALL SELECT name FROM dm) "
     "ORDER BY name COLLATE NOCASE, name", [("A",), ("B",), ("C",), ("D",)]),
])
db.close()

# ---------------------------------------------------------------- F01, F02: the reviews of 2026-11-02
nk = 65538                                     # just past sqlexec.ALG_MIN_ROWS: rank selection and series apply
ids = np.arange(nk)
db = torbles.from_ddl("CREATE TABLE tk (id INTEGER, k INTEGER, x INTEGER)",
                      {"tk": pd.DataFrame({"id": ids, "k": ids % 2, "x": np.where(ids % 2 == 1, 100000 - ids, ids)}).astype("int32")},
                      keep_plain=True)
cases(db, "F01 windows in the ORDER BY under ORDER BY ... LIMIT", [
    ("the follow-up review's case: LAG only in the ORDER BY", "SELECT id FROM tk ORDER BY k, lag(x) OVER (ORDER BY id) NULLS LAST, id LIMIT 3",
     [(65536,), (65534,), (65532,)]),
    ("... beside a SELECT window read from its series", "SELECT id, count(*) OVER (PARTITION BY k) AS c FROM tk "
     "ORDER BY k, lag(x) OVER (ORDER BY id) NULLS LAST, id LIMIT 3", [(65536, 32769), (65534, 32769), (65532, 32769)]),
    ("LEAD, descending, beside a SELECT window", "SELECT id, rank() OVER (ORDER BY x) AS r FROM tk "
     "ORDER BY k DESC, lead(id, 3) OVER (ORDER BY x DESC) DESC NULLS FIRST, id LIMIT 4",
     [(34459, 48309), (34463, 48307), (34465, 48305), (34467, 48303)]),
    ("control: a running count in the ORDER BY", "SELECT id FROM tk ORDER BY k, count(*) OVER (PARTITION BY k ORDER BY x) DESC, id LIMIT 3",
     [(65536,), (65534,), (65532,)]),
])
cases(db, "F02 frame offsets near 2^63, windows over no rows", [
    ("GROUPS ... 9223372036854775807 FOLLOWING", "SELECT id, count(*) OVER (ORDER BY k GROUPS BETWEEN CURRENT ROW AND 9223372036854775807 FOLLOWING) "
     "AS c FROM tk ORDER BY id LIMIT 3", [(0, 65538), (1, 32769), (2, 65538)]),
    ("a SUM to the partition's end", "SELECT id, sum(x) OVER (PARTITION BY k ORDER BY x GROUPS BETWEEN 1 PRECEDING AND 9223372036854775807 "
     "FOLLOWING) AS s FROM tk ORDER BY id LIMIT 3", [(0, 1073774592), (1, 199996), (2, 1073774592)]),
    ("a frame starting past every group", "SELECT id, count(*) OVER (ORDER BY k GROUPS BETWEEN 9223372036854775807 FOLLOWING AND "
     "UNBOUNDED FOLLOWING) AS c FROM tk ORDER BY id LIMIT 3", [(0, 0), (1, 0), (2, 0)]),
    ("two PARTITION BY columns over no rows", "SELECT id, count(*) OVER (PARTITION BY k, x) AS c FROM tk WHERE id < 0", []),
    ("CUME_DIST over no rows", "SELECT id, cume_dist() OVER () AS c FROM tk WHERE id < 0", []),
])
db.close()
print("\nALL MATCH" if fails == 0 else f"\n{fails} FAILURES")
sys.exit(1 if fails else 0)
