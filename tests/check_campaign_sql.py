# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Regression tests for the faults the release fuzzing campaign of 2026-11-02 found (an hour on 24 threads: about
1.3 million SQL statements and 1.3 million engine queries against DuckDB, on the engines alg, plain and gpu):
  C01  an aggregate's FILTER is computed on the rows the WHERE keeps: an error on a row the WHERE drops (abs of the
       least INTEGER, a DECIMAL(18) product that leaves its type) is not the query's;
  C02  a LEFT JOIN's ON term on the dimension alone that fails on a dimension row: an error only where the join's
       equality pairs the row (the block then runs on the general path, ON terms on the matched pairs; DuckDB
       pushes such a term into the dimension's scan and raises it for every row, in an order of its own);
  C03  a LEFT JOIN whose ON term is not TRUE pads its rows with NULLs: values cached while the ON terms ran (a cast,
       a comparison) are not read as NULL for those rows by the WHERE after the join (it lost rows: 0 for 36);
  C04  SUM and AVG of values holding the least BIGINT, -2^63 (whose np.abs is itself): exact, not wrapped mod 2^64;
  C05  FROM a, b, c WHERE a.k = b.k AND ... on the general path matches rows on those equalities instead of forming
       every pair (a 65-row fact asked for 120 GiB);
  C06  the SQL fuzzer's own checks: a statistic NULL in torbles and NaN in DuckDB that reorders ORDER BY ... LIMIT is
       the documented difference; an overflow DuckDB's statistics let it skip is DuckDB's (it raises it without its
       optimizer); -0.0 and 0.0 group and sort as one value.
Every case has a fixed expectation (rows, or ERROR where DuckDB raises), checked against DuckDB (C02: against the same
statement with the term's operand widened, which cannot fail) and against db.sql on the engines alg, plain and gpu
(TORBLES_SQL_ENGINE=<engine>: that engine only).
    python3 tests/check_campaign_sql.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decimal import Decimal
import numpy as np, pandas as pd
import torbles

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
    if not isinstance(got, list) or len(got) != len(want): return False
    for r, s in zip(got, want):
        for a, b in zip(r, s):
            if isinstance(a, float) and isinstance(b, float):
                if not (a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))): return False
            elif a != b: return False
    return True


def cases(db, title, items, duck_too=True):
    """items: (name, SQL, expected rows or ERROR): DuckDB first (it must agree with the expectation), then torbles"""
    print(f"== {title}")
    for name, sql, want in items:
        duck = outcome(lambda: db.con.execute(sql).fetchall()) if duck_too else None
        if duck_too and not same(duck, want):
            check(False, f"{name}: DuckDB disagrees with the expectation", f"{sql}\n        DuckDB {duck}  expected {want}")
            continue
        bad = {}
        for e in ENGINES:
            got = outcome(lambda: db.sql(sql, engine=e).rows)
            if not same(got, want): bad[e] = got
        check(not bad, f"{name} [{', '.join(ENGINES)}]", f"{sql}\n        expected {want}\n        " +
              "\n        ".join(f"{e}: {g}" for e, g in bad.items()))


rng = np.random.default_rng(1102)
ddl = """CREATE TABLE c1 (a BIGINT, x INTEGER, y DOUBLE, p DECIMAL(9,2), q INTEGER);
CREATE TABLE c2d (id INTEGER PRIMARY KEY, a DECIMAL(18,0), h VARCHAR);
CREATE TABLE c2f (k INTEGER REFERENCES c2d(id), x INTEGER, v INTEGER);
CREATE TABLE c3d (id VARCHAR PRIMARY KEY, h VARCHAR);
CREATE TABLE c3f (k VARCHAR REFERENCES c3d(id), m0 INTEGER, m1 DECIMAL(12,6));
CREATE TABLE c4 (g INTEGER, b BIGINT);"""
n1 = 3000
x1 = rng.integers(-1000, 1000, n1).astype("int64"); y1 = rng.integers(-1000, 1000, n1).astype(float)
x1[7], y1[7] = -2147483648, 0.0                     # the least INTEGER, on a row WHERE NOT x < y drops
p1 = [Decimal(int(v)) / 100 for v in rng.integers(-10**6, 10**6, n1)]; q1 = rng.integers(-50, 50, n1)
p1[9], q1[9] = Decimal("5802840.15"), -2147483645    # p * q leaves DECIMAL(18,2), on a row the WHERE drops
x1[9], y1[9] = 5, 9.0
nd2 = 64
a2 = [Decimal(int(v)) for v in rng.integers(-10**6, 10**6, nd2)]
a2[nd2 - 1] = Decimal(-891393061400433257)          # a * 10 leaves DECIMAL(18,0); no fact row references this one
n2 = 2000
k2 = rng.integers(0, nd2 - 1, n2)
nd3 = 40
k3 = [f"K{i}" for i in rng.integers(0, nd3, 3000)]
m0 = rng.integers(-500, 500, 3000).astype(object); m1 = [Decimal(int(v)) / 1000 for v in rng.integers(-10**6, 10**6, 3000)]
for i in range(36): m0[i] = None; m1[i] = Decimal("12.345")      # ON f.m0 <= f.m1 is NULL there: padded, kept by the OR
n4 = 400
b4 = rng.integers(-10**12, 10**12, n4).astype(object); g4 = rng.integers(0, 6, n4)
for i in range(0, n4, 3): b4[i] = None
frames = {
    "c1": pd.DataFrame({"a": rng.integers(0, 5, n1), "x": x1, "y": y1, "p": p1, "q": q1}).astype({"x": "int32", "q": "int32"}),
    "c2d": pd.DataFrame({"id": np.arange(nd2, dtype="int32"), "a": a2, "h": [f"h{i % 5}" for i in range(nd2)]}),
    "c2f": pd.DataFrame({"k": k2, "x": rng.integers(0, 10, n2), "v": rng.integers(-100, 100, n2)}).astype("int32"),
    "c3d": pd.DataFrame({"id": [f"K{i}" for i in range(nd3)], "h": ["" if i % 7 == 0 else f"h{i}" for i in range(nd3)]}),
    "c3f": pd.DataFrame({"k": k3, "m0": pd.array(m0, dtype="Int32"), "m1": m1}),
    "c4": pd.DataFrame({"g": g4.astype("int32"), "b": pd.array(b4, dtype="Int64")}),
}
db = torbles.from_ddl(ddl, frames, keep_plain=True)
q = db.con.execute

# ---------------------------------------------------------------- C01
w01 = q("SELECT a, count(*) FROM c1 WHERE NOT x < y GROUP BY 1 ORDER BY 1").fetchall()
cases(db, "C01 an aggregate's FILTER on the rows the WHERE keeps", [
    ("abs(the least INTEGER) in a FILTER, its row dropped by the WHERE",
     "SELECT a, count(*) FILTER (WHERE abs(x) <> -2147483648) FROM c1 WHERE NOT x < y GROUP BY 1 ORDER BY 1", w01),
    ("a DECIMAL(18) product in a FILTER, its row dropped by the WHERE",
     "SELECT a, count(*) FILTER (WHERE p * q >= 2147483647) FROM c1 WHERE NOT x < y GROUP BY 1 ORDER BY 1",
     q("SELECT a, count(*) FILTER (WHERE x >= y AND CAST(p AS DECIMAL(38,2)) * q >= 2147483647) FROM c1 WHERE NOT x < y GROUP BY 1 ORDER BY 1").fetchall()),
    ("control: the same FILTER where the WHERE keeps the row fails",
     "SELECT a, count(*) FILTER (WHERE abs(x) <> -2147483648) FROM c1 WHERE NOT x > y GROUP BY 1 ORDER BY 1", ERROR),
])

# ---------------------------------------------------------------- C02
w02 = q("SELECT d.h, count(*), sum(f.v) FROM c2f f LEFT JOIN c2d d ON f.k = d.id AND CAST(d.a AS DECIMAL(38,0)) * 10 < 11 "
        "GROUP BY 1 ORDER BY 1 NULLS FIRST").fetchall()
cases(db, "C02 a LEFT JOIN's dimension ON term that fails on a dimension row no fact row matches", [
    ("d.a * 10 leaves DECIMAL(18,0) on the unreferenced row",
     "SELECT d.h, count(*), sum(f.v) FROM c2f f LEFT JOIN c2d d ON f.k = d.id AND (d.a * 10) < 11 GROUP BY 1 ORDER BY 1 NULLS FIRST", w02),
    ("... with a WHERE on the fact",
     "SELECT d.h, count(*), sum(f.v) FROM c2f f LEFT JOIN c2d d ON f.k = d.id AND (d.a * 10) < 11 WHERE f.x <= 5 GROUP BY 1 ORDER BY 1 NULLS FIRST",
     q("SELECT d.h, count(*), sum(f.v) FROM c2f f LEFT JOIN c2d d ON f.k = d.id AND CAST(d.a AS DECIMAL(38,0)) * 10 < 11 WHERE f.x <= 5 "
       "GROUP BY 1 ORDER BY 1 NULLS FIRST").fetchall()),
    ("control: the term in the WHERE, over every dimension row, still fails",
     "SELECT count(*) FROM c2d d WHERE (d.a * 10) < 11", ERROR),
    ("control: ... and where the equality pairs the row", "SELECT count(*) FROM c2f f LEFT JOIN c2d d ON f.k + 1 = d.id "
     "AND (d.a * 10) < 11", ERROR),
], duck_too=False)

# ---------------------------------------------------------------- C03
L3 = "f.m1 IN (12.345, -327047.867839, 859636.709157, 7)"
cnt3 = q(f"SELECT count(*) FROM c3f f WHERE {L3}").fetchall()
cases(db, "C03 a LEFT JOIN's padded rows read their own values after the join", [
    ("the fuzzer's statement: ON f.m0 <= f.m1 NULL, the OR kept by the fact term",
     f"SELECT count(*) FROM c3f f LEFT JOIN c3d d ON f.k = d.id AND f.m0 <= f.m1 WHERE (d.h IN ('') OR {L3})",
     q(f"SELECT count(*) FROM c3f f LEFT JOIN c3d d ON f.k = d.id AND f.m0 <= f.m1 WHERE (d.h IN ('') OR {L3})").fetchall()),
    ("... a dimension term that is never TRUE", f"SELECT count(*) FROM c3f f LEFT JOIN c3d d ON f.k = d.id AND f.m0 <= f.m1 WHERE d.h = 'zz' OR {L3}", cnt3),
    ("... the padded rows' cast column read as values",
     f"SELECT CAST(f.m1 AS DECIMAL(16,6)), count(*) FROM c3f f LEFT JOIN c3d d ON f.k = d.id AND f.m0 <= f.m1 WHERE d.h IS NULL AND {L3} GROUP BY 1",
     q(f"SELECT CAST(f.m1 AS DECIMAL(16,6)), count(*) FROM c3f f WHERE {L3} GROUP BY 1").fetchall()),
    ("control: the fact term alone", f"SELECT count(*) FROM c3f f LEFT JOIN c3d d ON f.k = d.id AND f.m0 <= f.m1 WHERE {L3}", cnt3),
])

# ---------------------------------------------------------------- C04
M = "-9223372036854775808"
exact = {}
for g, b in zip(g4.tolist(), b4.tolist()):
    v = -(1 << 63) if b is None else b
    s, c = exact.get(g, (0, 0)); exact[g] = (s + v, c + 1)
w04s = [(g, s) for g, (s, c) in sorted(exact.items())]
w04a = [(g, s / c) for g, (s, c) in sorted(exact.items())]
cases(db, "C04 sums of values holding the least BIGINT", [
    ("SUM(coalesce(b, -2^63)) per group", f"SELECT g, sum(coalesce(b, {M})) FROM c4 GROUP BY 1 ORDER BY 1", w04s),
    ("AVG(coalesce(b, -2^63)) per group", f"SELECT g, avg(coalesce(b, {M})) FROM c4 GROUP BY 1 ORDER BY 1", w04a),
    ("AVG(greatest(b, -2^63)): DuckDB's GREATEST skips NULLs", f"SELECT g, avg(greatest(b, {M})) FROM c4 GROUP BY 1 ORDER BY 1", w04a),
    ("AVG of a CASE yielding -2^63", f"SELECT g, avg(CASE WHEN b > {M} THEN b ELSE {M} END) FROM c4 GROUP BY 1 ORDER BY 1", w04a),
    ("only -2^63 in a group", f"SELECT sum(coalesce(b, {M})), avg(coalesce(b, {M})) FROM c4 WHERE b IS NULL AND g = 0",
     [(-(1 << 63) * sum(1 for g, b in zip(g4, b4) if g == 0 and b is None), float(-(1 << 63)))]),
    ("a window SUM", f"SELECT DISTINCT g, sum(coalesce(b, {M})) OVER (PARTITION BY g) FROM c4 ORDER BY 1", w04s),
])
db.close()

# ---------------------------------------------------------------- C05
# no keys declared: the general path joins the FROM items itself
na, nb5, nc = 300, 3000, 3000
db = torbles.from_ddl("CREATE TABLE e1 (a INTEGER, v INTEGER); CREATE TABLE e2 (a INTEGER, b INTEGER); CREATE TABLE e3 (b INTEGER, w INTEGER)",
                      {"e1": pd.DataFrame({"a": rng.integers(0, 50, na), "v": rng.integers(0, 100, na)}).astype("int32"),
                       "e2": pd.DataFrame({"a": np.arange(nb5) % 50, "b": np.arange(nb5)}).astype("int32"),
                       "e3": pd.DataFrame({"b": np.arange(nc), "w": rng.integers(0, 9, nc)}).astype("int32")}, keep_plain=True)
q = db.con.execute
J = "FROM e1, e2, e3 WHERE e1.a = e2.a AND e2.b = e3.b"
cases(db, "C05 comma joins on the general path match on their WHERE equalities", [
    ("three tables, two equalities (every pair: 2.7 billion)", f"SELECT count(*), sum(e1.v), sum(e3.w) {J}",
     q(f"SELECT count(*), sum(e1.v), sum(e3.w) {J}").fetchall()),
    ("... with a term on two of them and a GROUP BY", f"SELECT e3.w, count(*) {J} AND e1.v < e3.w * 10 GROUP BY 1 ORDER BY 1",
     q(f"SELECT e3.w, count(*) {J} AND e1.v < e3.w * 10 GROUP BY 1 ORDER BY 1").fetchall()),
    ("control: a cross join without an equality", "SELECT count(*) FROM e1, e3 WHERE e3.b < 10", [(na * 10,)]),
])
db.close()

# ---------------------------------------------------------------- C06
from torbles import sqlfuzz, fuzz
import duckdb
print("== C06 the SQL fuzzer's checks")
con = duckdb.connect()
con.execute("CREATE TABLE s (g INTEGER, x INTEGER, y DOUBLE)")
con.execute("INSERT INTO s SELECT i % 5, CASE WHEN i % 5 < 2 THEN 7 ELSE i END, i * 0.5 FROM range(40) r(i)")
con.execute("INSERT INTO s VALUES (9, -2147483648, 0.0), (9, 3, -0.0), (8, 1, 0.0)")
dbs = torbles.from_duckdb(con, keep_plain=True)
sql = "SELECT g, corr(x, x) FROM s WHERE g < 5 GROUP BY 1 ORDER BY 2 NULLS FIRST, 1 LIMIT 2"
st, detail = sqlfuzz.run_one(dbs, con, sql)
check(st == "ok" and "documented" in detail, "CORR NULL in torbles, NaN in DuckDB, under ORDER BY ... LIMIT: the documented difference",
      f"{st}: {detail[:200]}")
sql = "(SELECT x FROM s WHERE g BETWEEN 3 AND 2) EXCEPT (SELECT x FROM s WHERE abs(x) >= 5)"
st, detail = sqlfuzz.run_one(dbs, con, sql)
check(st == "duckdb-error" and "without it DuckDB raises it too" in detail,
      "an overflow DuckDB's optimizer skips (EXCEPT after an empty side): DuckDB's, it raises it without its optimizer", f"{st}: {detail[:200]}")
from sqltypes import SqlError


class Fake:
    """an engine that answers DuckDB's rows, in reverse after ORDER BY ... LIMIT, or raises: what a fault would look like"""
    def __init__(self, raise_=None): self.raise_ = raise_
    def sql(self, sql, engine=None):
        if self.raise_: raise SqlError(self.raise_)
        rows = con.execute(sql).fetchall()
        return type("R", (), {"rows": rows[::-1] if " LIMIT " in sql else rows})()


st, detail = sqlfuzz.run_one(Fake("Out of Range Error: Overflow on abs(-2147483648)"), con, "SELECT count(*) FROM s WHERE g = 1")
check(st == "ERROR", "control: an overflow DuckDB raises in no setting stays an ERROR", f"{st}: {detail[:200]}")
st, detail = sqlfuzz.run_one(Fake(), con, "SELECT g, x FROM s WHERE g < 5 ORDER BY 2, 1 LIMIT 4")
check(st == "MISMATCH", "control: rows in a wrong order under ORDER BY ... LIMIT stay a MISMATCH", f"{st}: {detail[:200]}")
class Signed(Fake):
    """an engine whose group of zeros shows -0.0 where DuckDB's shows 0.0 (either is the group's value)"""
    def sql(self, sql, engine=None):
        rows = [tuple(-0.0 if isinstance(v, float) and v == 0 else v for v in r) for r in con.execute(sql).fetchall()]
        return type("R", (), {"rows": rows})()


con.execute("CREATE TABLE z AS SELECT * FROM (VALUES (-0.000002), (-0.000001), (0.0)) t(y)")
st, detail = sqlfuzz.run_one(Signed(), con, "SELECT CAST(y AS DOUBLE), count(*) FROM z GROUP BY 1")
check(st == "ok", "-0.0 and 0.0 group and sort as one value (beside values within the tolerance of both)", f"{st}: {detail[:200]}")
dbs.close()
print("\nALL MATCH" if fails == 0 else f"\n{fails} FAILURES")
sys.exit(1 if fails else 0)
