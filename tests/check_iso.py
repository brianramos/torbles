# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ISO/IEC 9075 forms of the SQL front end beyond the core of check_sql.py, each checked against DuckDB where DuckDB has
the same semantics, else against values worked out by hand:
  lexical forms (0x / 0o / 0b and 1_000 numbers, U&'...' UESCAPE, N'...', continued literals), CURRENT_* values,
  TIMESTAMP / TIME WITH TIME ZONE (literals, casts, AT TIME ZONE / AT LOCAL, SET TIME ZONE, stored zoned columns),
  COLLATE (comparisons, ORDER BY, GROUP BY, DISTINCT, MIN / MAX, windows), TABLESAMPLE (BERNOULLI, SYSTEM,
  REPEATABLE), FETCH FIRST / OFFSET with computed counts, renamed table columns (AS t(a, b))."""
import datetime as dt, os, sys
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import duckdb, numpy as np, pandas as pd
import torbles
import sqlfuzz

fails = 0
ENGINE = os.environ.get("TORBLES_SQL_ENGINE", "alg")             # alg | plain | gpu
UTC = dt.timezone.utc


def check(ok, what, detail=""):
    global fails
    if not ok: fails += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"\n        {detail}" if not ok and detail else ""))


def tz(h, m=0): return dt.timezone(dt.timedelta(hours=h, minutes=m))


r = np.random.default_rng(3)
n = 3000
gdf = pd.DataFrame(dict(gid=np.arange(40), gname=[f"G{i % 7}" if i % 3 else f"g{i % 7}" for i in range(40)]))
tdf = pd.DataFrame(dict(id=np.arange(n), k=r.integers(0, 40, n), v=r.integers(0, 100, n),
                        x=r.choice(["b", "A", "a", "B", "é", "E", "e", None], n),
                        z=[dt.datetime(2020, 1, 1, tzinfo=UTC) + dt.timedelta(minutes=int(m)) if m % 11 else None for m in r.integers(0, 10**6, n)],
                        ts=[dt.datetime(2020, 1, 1) + dt.timedelta(minutes=int(m)) for m in r.integers(0, 10**6, n)]))
ddl = """CREATE TABLE g (gid INTEGER PRIMARY KEY, gname VARCHAR);
CREATE TABLE t (id INTEGER, k INTEGER REFERENCES g(gid), v INTEGER, x VARCHAR, z TIMESTAMP WITH TIME ZONE, ts TIMESTAMP);"""
db = torbles.from_ddl(ddl, {"g": gdf, "t": tdf}, keep_plain=ENGINE == "plain")
if ENGINE != "alg":
    _sql = db.sql; db.sql = lambda text, engine=ENGINE: _sql(text, engine)
con = duckdb.connect(); con.execute("SET TimeZone = 'UTC'")
con.execute(ddl.replace(" REFERENCES g(gid)", ""))
con.register("gdf", gdf); con.register("tdf", tdf)
con.execute("INSERT INTO g SELECT * FROM gdf"); con.execute("INSERT INTO t SELECT * FROM tdf")
db.sql("SET TIME ZONE 'UTC'")


def vs_duckdb(sql, what=None, ordered=None, tol=1e-9):
    want = con.execute(sql).fetchall()
    try: got = db.sql(sql).rows
    except Exception as x: check(False, what or sql[:110], f"torbles: {type(x).__name__}: {x}"); return
    check(sqlfuzz.same(got, want, "ORDER BY" in sql if ordered is None else ordered, tol), what or sql[:110],
          f"torbles {got[:5]}\n        duckdb  {want[:5]}")


def expect(sql, want, what=None):
    try: got = db.sql(sql).rows
    except Exception as x: check(False, what or sql[:110], f"{type(x).__name__}: {x}"); return
    check(sqlfuzz.same(got, want, True, 1e-12), what or sql[:110], f"torbles {got}\n        expected {want}")


def refuses(sql, words, what=None):
    try: db.sql(sql); check(False, what or sql[:110], "no error")
    except Exception as x: check(all(w.lower() in str(x).lower() for w in words), what or sql[:110], f"{type(x).__name__}: {x}")


print("== lexical forms")
expect("SELECT 0x1F, 0X_ff, 0o17, 0b1011, 1_000_000, 1_0.2_5, 2e1_0", [(31, 255, 15, 11, 1000000, Decimal("10.25"), 2e10)])
expect("SELECT U&'d\\0061t\\+000061', U&'!00e9t!!' UESCAPE '!', N'national', 'con' 'tin'\n'ued'", [("data", "ét!", "national", "continued")])
expect('SELECT 1 AS U&"\\0041b"', [(1,)])
check(db.sql('SELECT 1 AS U&"\\0041b"').columns == ["Ab"], "U&\"...\" identifiers", str(db.sql('SELECT 1 AS U&"\\0041b"').columns))

print("== CURRENT_* values")
expect("SELECT CURRENT_DATE = CAST(CURRENT_TIMESTAMP AS DATE), CURRENT_SCHEMA, CURRENT_CATALOG, CURRENT_ROLE IS NULL, "
       "CURRENT_USER = SESSION_USER, LOCALTIMESTAMP <= CURRENT_TIMESTAMP", [(True, "main", "memory", True, True, True)])
expect("SELECT typeof(CURRENT_TIMESTAMP), typeof(LOCALTIMESTAMP), typeof(CURRENT_TIME), typeof(LOCALTIME), "
       "EXTRACT(MICROSECOND FROM CURRENT_TIMESTAMP(0)) % 1000000",
       [("TIMESTAMP WITH TIME ZONE", "TIMESTAMP", "TIME WITH TIME ZONE", "TIME", 0)])

print("== time zones")
expect("SELECT TIMESTAMP '2020-01-01 10:00:00+02', TIMESTAMP WITH TIME ZONE '2020-01-01 10:00:00', TIME '10:00:00-05:30', "
       "typeof(TIMESTAMP '2020-01-01 10:00:00+02'), typeof(TIMESTAMP '2020-01-01 10:00:00')",
       [(dt.datetime(2020, 1, 1, 10, tzinfo=tz(2)), dt.datetime(2020, 1, 1, 10, tzinfo=UTC), dt.time(10, tzinfo=tz(-5, -30)),
         "TIMESTAMP WITH TIME ZONE", "TIMESTAMP")])
expect("SELECT EXTRACT(HOUR FROM TIMESTAMP '2020-01-01 10:00:00+02'), EXTRACT(TIMEZONE_HOUR FROM TIMESTAMP '2020-01-01 10:00:00+02'), "
       "EXTRACT(TIMEZONE_MINUTE FROM TIME '10:00:00-05:30'), CAST(TIMESTAMP '2020-01-01 10:00:00+02' AS VARCHAR)",
       [(10, 2, -30, "2020-01-01 10:00:00+02")], "EXTRACT: the value's own displacement (ISO)")
expect("SELECT TIMESTAMP '2020-01-01 10:00:00+02' = TIMESTAMP '2020-01-01 08:00:00Z', "
       "TIMESTAMP '2020-01-01 10:00:00+02' - TIMESTAMP '2020-01-01 00:00:00+00', TIME '10:00:00+02' < TIME '09:00:00+00'",
       [(True, dt.timedelta(hours=8), True)])
expect("SET TIME ZONE INTERVAL '-05:00' HOUR TO MINUTE; SELECT TIMESTAMP '2020-01-01 10:00:00+02' AT LOCAL, "
       "TIMESTAMP '2020-01-01 10:00:00' AT TIME ZONE INTERVAL '01:00' HOUR TO MINUTE, CAST(TIMESTAMP '2020-01-01 10:00:00+02' AS TIMESTAMP), "
       "CAST(TIMESTAMP '2020-01-01 10:00:00' AS TIMESTAMP WITH TIME ZONE), TIMESTAMP '2020-01-01 10:00:00+02' AT TIME ZONE 'Asia/Kolkata'",
       [(dt.datetime(2020, 1, 1, 3, tzinfo=tz(-5)), dt.datetime(2020, 1, 1, 16, tzinfo=tz(1)), dt.datetime(2020, 1, 1, 3),
         dt.datetime(2020, 1, 1, 10, tzinfo=tz(-5)), dt.datetime(2020, 1, 1, 13, 30, tzinfo=tz(5, 30)))], "SET TIME ZONE, AT LOCAL, AT TIME ZONE")
db.sql("SET TIME ZONE 'UTC'")
for q in ["SELECT id, z, ts FROM t WHERE id < 50 ORDER BY id",
          "SELECT count(*) FROM t WHERE z > TIMESTAMPTZ '2020-06-01 11:00:00+02'",
          "SELECT count(*) FROM t WHERE z > ts", "SELECT min(z), max(z), count(z) FROM t",
          "SELECT date_trunc('month', z), count(*), min(v) FROM t GROUP BY 1 ORDER BY 1 NULLS FIRST",
          "SELECT id, z - ts, CAST(z AS DATE), CAST(z AS TIMESTAMP), extract(hour from z), z + INTERVAL '1' MONTH FROM t WHERE id < 40 ORDER BY id",
          "SELECT g.gname, max(z) FROM t JOIN g ON t.k = g.gid GROUP BY g.gname ORDER BY 1"]:
    vs_duckdb(q)

print("== COLLATE")
for q in ["SELECT 'Abc' COLLATE NOCASE = 'abc', 'é' COLLATE NOACCENT = 'e', 'b' COLLATE NOCASE > 'A', 'B' < 'a'",
          "SELECT count(*) FROM t WHERE x COLLATE NOCASE = 'a'",
          "SELECT count(*) FROM t WHERE x COLLATE NOCASE IN ('a', 'E')",
          "SELECT count(*), count(DISTINCT x COLLATE NOCASE), count(DISTINCT x COLLATE NOCASE.NOACCENT), count(DISTINCT x) FROM t",
          "SELECT max(x COLLATE NOCASE) = 'é', min(x COLLATE NOCASE) IN ('a', 'A') FROM t",
          "SELECT lower(x COLLATE NOCASE), count(*) FROM t GROUP BY x COLLATE NOCASE ORDER BY 1 NULLS FIRST",
          "SELECT id, x FROM t WHERE id < 60 ORDER BY x COLLATE NOCASE NULLS FIRST, id",
          "SELECT id, rank() OVER (ORDER BY x COLLATE NOCASE) FROM t WHERE id < 60 ORDER BY id",
          "SELECT count(*) FROM t JOIN g ON t.k = g.gid WHERE g.gname COLLATE NOCASE = 'g3'"]:
    vs_duckdb(q)
refuses("SELECT 'a' COLLATE NOCASE = 'A' COLLATE NOACCENT", ["collation conflict"])
refuses("SELECT 'a' COLLATE klingon", ["collation"])

print("== TABLESAMPLE")
a = db.sql("SELECT count(*), sum(v) FROM t TABLESAMPLE BERNOULLI (20) REPEATABLE (7)").rows
b = db.sql("SELECT id, v FROM t TABLESAMPLE BERNOULLI (20) REPEATABLE (7)").rows
c = db.sql("SELECT k, count(*) FROM t TABLESAMPLE BERNOULLI (20) REPEATABLE (7) WHERE v < 50 GROUP BY k").rows
want_c = {}
for i, v in b:
    if v < 50: want_c[int(tdf.k[i])] = want_c.get(int(tdf.k[i]), 0) + 1
check(a == [(len(b), sum(v for _, v in b))] and dict(c) == want_c and 400 < len(b) < 800,
      f"BERNOULLI (20) REPEATABLE: the same rows in every query ({len(b)} of {n})", f"{a} {len(b)}")
s1 = db.sql("SELECT count(*) FROM t TABLESAMPLE SYSTEM (50) REPEATABLE (1)").rows[0][0]
check(db.sql("SELECT count(*) FROM t TABLESAMPLE BERNOULLI (100)").rows == [(n,)] and db.sql("SELECT count(*) FROM t TABLESAMPLE BERNOULLI (0)").rows == [(0,)]
      and 900 < s1 < 2100, f"SYSTEM (50) samples 64-row blocks ({s1}); 0 % and 100 %")
d1 = db.sql("SELECT count(*) FROM t JOIN g TABLESAMPLE BERNOULLI (50) REPEATABLE (3) ON t.k = g.gid").rows[0][0]
kept = [x[0] for x in db.sql("SELECT gid FROM g TABLESAMPLE BERNOULLI (50) REPEATABLE (3)").rows]
check(d1 == int(tdf.k.isin(kept).sum()), f"a sampled dimension joins as a row mask ({len(kept)} of 40 kept)")
check(db.sql("SELECT count(*) FROM t USING SAMPLE 100 ROWS").rows == [(100,)], "USING SAMPLE n ROWS")

print("== FETCH / OFFSET counts, renamed columns")
expect("SELECT id FROM t ORDER BY id OFFSET (SELECT count(*) FROM g) - 38 ROWS FETCH FIRST (SELECT 3) ROWS ONLY", [(2,), (3,), (4,)])
expect("SELECT a, b, x.v FROM t AS x(a, b) WHERE a = 2", [(2, int(tdf.k[2]), int(tdf.v[2]))])
check(db.sql("SELECT * FROM g AS h(a) LIMIT 1").columns == ["a", "gname"], "SELECT * shows the new names")
refuses("SELECT gid FROM g AS h(a)", ["gid"], "a renamed column is not visible by its old name")

print("== predicates: NORMALIZED, MATCH, PERIOD, row IS NULL, extended CASE")
expect("SELECT 'é' IS NORMALIZED, 'é' IS NFD NORMALIZED, 'e' IS NOT NFKC NORMALIZED, NORMALIZE('é', NFD) IS NFD NORMALIZED, "
       "CAST(NULL AS VARCHAR) IS NORMALIZED", [(True, False, False, True, None)])
expect("SELECT (1, 2) MATCH (SELECT 1, 2), (1, NULL) MATCH (SELECT 3, 4), (1, NULL) MATCH FULL (SELECT 1, 2), "
       "(1, NULL) MATCH PARTIAL (SELECT 1, 2), (1, NULL) MATCH UNIQUE PARTIAL (SELECT * FROM (VALUES (1, 2), (1, 3)) v(a, b)), "
       "(NULL, NULL) MATCH FULL (SELECT 1, 2), 5 MATCH UNIQUE (SELECT v FROM t WHERE id < 0)", [(True, True, False, True, False, True, False)])
expect("SELECT count(*) FROM g WHERE gid MATCH (SELECT k FROM t WHERE v > 98)",
       [(int(gdf.gid.isin(tdf.k[tdf.v > 98]).sum()),)], "MATCH correlated to a table, as a filter")
expect("SELECT PERIOD (DATE '2020-01-01', DATE '2020-02-01') OVERLAPS PERIOD (DATE '2020-01-31', DATE '2020-03-01'), "
       "PERIOD (1, 5) CONTAINS 3, PERIOD (1, 5) CONTAINS 5, PERIOD (1, 5) CONTAINS PERIOD (2, 5), PERIOD (1, 3) IMMEDIATELY PRECEDES PERIOD (3, 4), "
       "PERIOD (1, 3) PRECEDES PERIOD (2, 4), PERIOD (5, 6) SUCCEEDS PERIOD (1, 5), PERIOD (5, 6) IMMEDIATELY SUCCEEDS PERIOD (1, 4), PERIOD (1, 2) EQUALS PERIOD (1, 2)",
       [(True, True, False, True, True, False, True, False, True)])
expect("SELECT count(*) FROM t WHERE PERIOD (ts, ts + INTERVAL '1' DAY) OVERLAPS PERIOD (TIMESTAMP '2020-06-01 00:00:00', TIMESTAMP '2020-06-02 00:00:00')",
       [(int(((tdf.ts < pd.Timestamp("2020-06-02")) & (tdf.ts + pd.Timedelta(days=1) > pd.Timestamp("2020-06-01"))).sum()),)], "PERIOD predicates on columns")
expect("SELECT (1, NULL) IS NULL, (NULL, NULL) IS NULL, (1, 2) IS NOT NULL, (1, NULL) IS NOT NULL", [(False, True, True, False)])
expect("SELECT count(*) FILTER (WHERE c = 'small'), count(*) FILTER (WHERE c = 'mid'), count(*) FILTER (WHERE c = 'big') FROM "
       "(SELECT CASE v WHEN 0, 1, 2 THEN 'small' WHEN < 50 THEN 'mid' WHEN BETWEEN 50 AND 89 THEN 'big' WHEN IS NULL THEN 'none' ELSE 'top' END AS c FROM t) s",
       [(int((tdf.v <= 2).sum()), int(((tdf.v > 2) & (tdf.v < 50)).sum()), int(((tdf.v >= 50) & (tdf.v <= 89)).sum()))], "extended CASE (F262, F263)")

print("== GROUP BY DISTINCT / ALL, functional dependencies")
expect("SELECT k, count(*) FROM (VALUES (1), (1), (2)) v(k) GROUP BY DISTINCT ROLLUP (k), ROLLUP (k) ORDER BY 1 NULLS FIRST", [(None, 3), (1, 2), (2, 1)])
expect("SELECT count(*) FROM (SELECT k, count(*) FROM (VALUES (1), (1), (2)) v(k) GROUP BY ALL ROLLUP (k), ROLLUP (k)) x", [(7,)])
vs_duckdb("SELECT k, count(*) FROM t GROUP BY ALL ORDER BY 1", "GROUP BY ALL without a list (DuckDB)")
gname = dict(zip(gdf.gid, gdf.gname))
expect("SELECT g.gid, g.gname, count(*) FROM t JOIN g ON t.k = g.gid GROUP BY g.gid ORDER BY 1 LIMIT 3",
       [(i, gname[i], int((tdf.k == i).sum())) for i in range(3)], "columns of a table grouped by its primary key")
expect("SELECT t.k, g.gname, count(*) FROM t JOIN g ON t.k = g.gid GROUP BY t.k ORDER BY 1 LIMIT 3",
       [(i, gname[i], int((tdf.k == i).sum())) for i in range(3)], "... and through the join equality")
refuses("SELECT k, v FROM t GROUP BY k", ["must appear in the GROUP BY"])

print("== intervals")
expect("SELECT CAST('1-2' AS INTERVAL YEAR TO MONTH) = INTERVAL '14' MONTH, CAST('3 04:05:06.5' AS INTERVAL DAY TO SECOND), "
       "CAST(90 AS INTERVAL MINUTE), CAST(1.5 AS INTERVAL DAY), INTERVAL '100' HOUR(3), INTERVAL -'1:30' HOUR TO MINUTE, ABS(INTERVAL '-3' DAY)",
       [(True, dt.timedelta(days=3, hours=4, minutes=5, seconds=6.5), dt.timedelta(minutes=90), dt.timedelta(days=1.5), dt.timedelta(hours=100),
         -dt.timedelta(minutes=90), dt.timedelta(days=3))])
expect("SELECT (DATE '2020-03-15' - DATE '2020-01-20') DAY, (TIMESTAMP '2020-03-15 10:30:00' - TIMESTAMP '2020-01-20 12:00:00') DAY TO SECOND, "
       "(DATE '2021-03-15' - DATE '2020-04-20') YEAR TO MONTH = INTERVAL '10' MONTH, (TIMESTAMP '2020-01-01 10:59:00' - TIMESTAMP '2020-01-01 09:00:00') HOUR, "
       "(DATE '2020-01-01' - DATE '2020-03-01') MONTH = INTERVAL '-2' MONTH",
       [(dt.timedelta(days=55), dt.timedelta(days=54, hours=22, minutes=30), True, dt.timedelta(hours=1), True)], "(a - b) <qualifier>")
refuses("SELECT CAST('13-1' AS INTERVAL HOUR TO MINUTE) + CAST('1-13' AS INTERVAL YEAR TO MONTH)", ["invalid interval"])

print("== character and binary strings")
expect("SELECT CHAR_LENGTH('héllo' USING OCTETS), CHARACTER_LENGTH('héllo' USING CHARACTERS), POSITION('l' IN 'héllo' USING OCTETS), "
       "SUBSTRING('héllo' FROM 4 FOR 2 USING OCTETS), OVERLAY('héllo' PLACING 'XY' FROM 2 FOR 2 USING OCTETS), TRANSLATE('héllo' USING LATIN1)",
       [(6, 5, 4, "ll", "hXYllo", "héllo")])
refuses("SELECT TRANSLATE('日本' USING ASCII)", ["untranslatable"])
refuses("SELECT SUBSTRING('héllo' FROM 3 FOR 1 USING OCTETS)", ["splits a character"])
expect("SELECT X'ab00', X'01' || X'02ff', OCTET_LENGTH(X'0102'), SUBSTRING(X'010203' FROM 2 FOR 1), POSITION(X'03' IN X'010203'), "
       "TRIM(X'000100'), TRIM(LEADING X'01' FROM X'010102'), X'01' < X'ff', CAST('a\\xAAb' AS BLOB), CAST(X'610062ff' AS VARCHAR), typeof(X'00')",
       [(b"\xab\x00", b"\x01\x02\xff", 2, b"\x02", 3, b"\x01", b"\x02", True, b"a\xaab", "a\\x00b\\xFF", "BLOB")])
expect("SELECT X'6162' LIKE X'6125', X'6162' LIKE X'5F5F5F', hex(X'0aff'), unhex('0aff'), encode('é'), decode(X'c3a9'), "
       "OVERLAY(X'010203' PLACING X'ff' FROM 2), CAST(X'00' AS BINARY VARYING(3)), CAST('x' AS CHARACTER LARGE OBJECT), CAST(X'01' AS BINARY LARGE OBJECT)",
       [(True, False, "0AFF", b"\n\xff", b"\xc3\xa9", "é", b"\x01\xff\x03", b"\x00", "x", b"\x01")])

print("ALL MATCH" if not fails else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
