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

print("== arrays, multisets, UNNEST, LATERAL")
expect("SELECT ARRAY[1, 2], ARRAY[1] || ARRAY[2, 3], CARDINALITY(ARRAY[1, 2, 3]), (ARRAY[1, 2])[2], TRIM_ARRAY(ARRAY[1, 2, 3], 2), "
       "ARRAY_MAX_CARDINALITY(CAST(ARRAY[1] AS INTEGER ARRAY[10])), ARRAY(SELECT gid FROM g WHERE gid < 3 ORDER BY gid DESC)",
       [([1, 2], [1, 2, 3], 3, 2, [1], 10, [2, 1, 0])])
refuses("SELECT (ARRAY[1, 2])[3]", ["array element error"])
refuses("SELECT CAST(ARRAY[1, 2, 3] AS INTEGER ARRAY[2])", ["right truncation"])
expect("SELECT MULTISET[1, 2, 2], CARDINALITY(MULTISET[1, 2, 2]), SET(MULTISET[1, 2, 2]), ELEMENT(MULTISET[7]), ELEMENT(MULTISET(SELECT gid FROM g WHERE gid = 5)), "
       "MULTISET[1, 2] = MULTISET[2, 1], MULTISET[1, 2] <> MULTISET[2, 2], CAST(ARRAY[1, 2] AS INTEGER MULTISET) = MULTISET[2, 1]",
       [([1, 2, 2], 3, [1, 2], 7, 5, True, True, True)])
refuses("SELECT ELEMENT(MULTISET[1, 2])", ["cardinality violation"])
expect("SELECT 2 MEMBER OF MULTISET[1, 2], 3 NOT MEMBER MULTISET[1, 2], MULTISET[1, 1] SUBMULTISET OF MULTISET[1, 2, 1], "
       "MULTISET[1, 1] SUBMULTISET MULTISET[1, 2], MULTISET[1, 2] IS A SET, MULTISET[1, 1] IS NOT A SET, NULL MEMBER OF MULTISET[1], 1 MEMBER OF MULTISET[NULL, 2]",
       [(True, True, True, False, True, True, None, None)])
expect("SELECT MULTISET[1, 1, 2] MULTISET UNION MULTISET[2, 3], MULTISET[1, 1, 2] MULTISET UNION DISTINCT MULTISET[2, 3], "
       "MULTISET[1, 1, 2] MULTISET EXCEPT MULTISET[1], MULTISET[1, 1, 2] MULTISET EXCEPT DISTINCT MULTISET[1], "
       "MULTISET[1, 1, 2] MULTISET INTERSECT ALL MULTISET[1, 1, 1], MULTISET[1] MULTISET UNION MULTISET[1, 1, 2] MULTISET INTERSECT MULTISET[1]",
       [([1, 1, 2, 2, 3], [1, 2, 2, 3], [1, 2], [2], [1, 1], [1, 1])], "MULTISET UNION / EXCEPT / INTERSECT (INTERSECT binds tighter)")
cnt = tdf[tdf.v < 3].groupby("k").size()
got = db.sql("SELECT k, CARDINALITY(COLLECT(v)), CARDINALITY(SET(COLLECT(v))), CARDINALITY(FUSION(MULTISET[v, v])), INTERSECTION(MULTISET[1, v]) "
             "FROM t WHERE v < 3 GROUP BY k ORDER BY k").rows
from collections import Counter
def inter(vs):
    acc = None
    for v in vs: acc = Counter([1, v]) if acc is None else acc & Counter([1, v])
    return acc
check([(k, c) for k, c, *_ in got] == [(int(k), int(c)) for k, c in cnt.items()] and all(f == 2 * c for _, c, _, f, _ in got)
      and all(Counter(x) == inter(tdf[(tdf.v < 3) & (tdf.k == k)].v.tolist()) for k, _, _, _, x in got)
      and all(d == len(set(tdf[(tdf.v < 3) & (tdf.k == k)].v)) for k, _, d, _, _ in got),
      "COLLECT, FUSION, INTERSECTION per group", str(got[:3]))
for q in ["SELECT * FROM unnest([10, 20]) WITH ORDINALITY AS u(x, n)",
          "SELECT g.gid, u.x FROM g, unnest([g.gid, g.gid * 10]) AS u(x) WHERE g.gid < 4 ORDER BY 1, 2",
          "SELECT g.gid, s.c FROM g, LATERAL (SELECT count(*) AS c, sum(v) AS sv FROM t WHERE t.k = g.gid AND t.v > g.gid) s ORDER BY 1",
          "SELECT g.gid, s.b FROM g LEFT JOIN LATERAL (SELECT g.gid * 2 AS b WHERE g.gid % 3 = 0) s ON TRUE ORDER BY 1",
          "SELECT g.gid, s.v FROM g CROSS JOIN LATERAL (SELECT v FROM t WHERE t.k = g.gid ORDER BY v DESC, id LIMIT 2) s ORDER BY 1, 2",
          "SELECT k, x FROM (SELECT k, list(v ORDER BY v) AS a FROM t WHERE v < 2 GROUP BY k) q, unnest(q.a) AS u(x) ORDER BY 1, 2"]:
    vs_duckdb(q)
expect("SELECT * FROM UNNEST(ARRAY[1, 2, 3], ARRAY['a']) AS u(x, y)", [(1, "a"), (2, None), (3, None)], "UNNEST of several collections (ISO: padded)")

print("== SQL/JSON")
J = '{"a": 1, "b": [1, 2, {"c": "x"}], "d": {"e": null, "f": 2.5}, "s": "str", "t": "2020-03-04"}'
jq = lambda q: q.replace("$J", "'" + J + "'")
expect(jq("SELECT JSON_VALUE($J, '$.a'), JSON_VALUE($J, '$.a' RETURNING INTEGER), JSON_VALUE($J, '$.d.f' RETURNING DECIMAL(5,2)), "
          "JSON_VALUE($J, 'lax $.b.c'), JSON_VALUE($J, '$.zz' DEFAULT 'none' ON EMPTY), JSON_VALUE($J, '$.b'), "
          "JSON_VALUE($J, 'strict $.zz' DEFAULT -1 ON ERROR), JSON_VALUE($J, '$.d.e'), JSON_VALUE($J, '$.t' RETURNING DATE)"),
       [("1", 1, Decimal("2.50"), "x", "none", None, "-1", None, dt.date(2020, 3, 4))], "JSON_VALUE: RETURNING, ON EMPTY, ON ERROR, lax unwrapping")
refuses(jq("SELECT JSON_VALUE($J, '$.b' ERROR ON ERROR)"), ["not a scalar"])
refuses(jq("SELECT JSON_EXISTS($J, 'strict $.b[9]' ERROR ON ERROR)"), ["strict"])
refuses("SELECT JSON_VALUE('{}', '$.a ? (')", ["path"], "a path syntax error fails when the query is bound")
expect(jq("SELECT JSON_QUERY($J, '$.b'), JSON_QUERY($J, '$.b[*]' WITH WRAPPER), JSON_QUERY($J, '$.b[0 to 1]' WITH CONDITIONAL WRAPPER), "
          "JSON_QUERY($J, '$.a' WITH CONDITIONAL WRAPPER), JSON_QUERY($J, '$.a' WITH UNCONDITIONAL ARRAY WRAPPER), JSON_QUERY($J, '$.s' OMIT QUOTES), "
          "JSON_QUERY($J, '$.zz' EMPTY ARRAY ON EMPTY), JSON_QUERY($J, '$.zz' EMPTY OBJECT ON EMPTY), JSON_QUERY($J, '$.b[*]' NULL ON ERROR)"),
       [('[1,2,{"c":"x"}]', '[1,2,{"c":"x"}]', "[1,2]", "1", "[1]", "str", "[]", "{}", None)], "JSON_QUERY: wrappers, quotes, ON EMPTY")
expect(jq("SELECT JSON_EXISTS($J, '$.b[1]'), JSON_EXISTS($J, '$.b[9]'), JSON_EXISTS($J, 'strict $.b[9]'), JSON_EXISTS($J, 'strict $.b[9]' UNKNOWN ON ERROR), "
          "JSON_EXISTS($J, '$.b[*] ? (@ > 1)'), JSON_EXISTS($J, '$ ? (exists (@.d.e))'), JSON_EXISTS(CAST(NULL AS VARCHAR), '$')"),
       [(True, False, False, None, True, True, None)])
expect(jq("SELECT JSON_QUERY($J, '$.b[*] ? (@.type() == \"number\" && @ >= 1)' WITH WRAPPER), JSON_VALUE($J, '$.b.size()'), JSON_VALUE($J, '$.d.f.ceiling()'), "
          "JSON_QUERY($J, '$.d.keyvalue().name' WITH WRAPPER), JSON_VALUE($J, '$.a + $x * 2' PASSING 10 AS x RETURNING INTEGER), "
          "JSON_EXISTS($J, '$.s ? (@ starts with \"st\")'), JSON_EXISTS($J, '$.s ? (@ like_regex \"^S\" flag \"i\")'), "
          "JSON_VALUE($J, '$.t.datetime().type()'), JSON_VALUE($J, '$.t.datetime(\"YYYY-MM-DD\") ? (@ > \"2020-01-01\".datetime())' RETURNING DATE), "
          "JSON_QUERY($J, '$.b[last]'), JSON_QUERY($J, '$.b[last - 1 to last]' WITH WRAPPER), JSON_QUERY($J, '$.* ? (@.type() == \"string\")' WITH WRAPPER), "
          "JSON_QUERY($J, '$.b[*] ? (@ == \"1\")' WITH WRAPPER), JSON_QUERY($J, '$.b[*] ? ((@ == \"1\") is unknown)' WITH WRAPPER)"),
       [("[1,2]", "3", "3", '["e","f"]', 21, True, True, "date", dt.date(2020, 3, 4), '{"c":"x"}', '[2,{"c":"x"}]', '["str","2020-03-04"]', None, '[1,2,{"c":"x"}]')],
       "the SQL/JSON path language: filters, methods, variables, last, comparisons that are unknown")
expect("SELECT JSON_OBJECT('a' VALUE 1, 'b' : 'x', KEY 'c' VALUE NULL), JSON_OBJECT('a' VALUE NULL ABSENT ON NULL), JSON_OBJECT('j' VALUE '[1,2]' FORMAT JSON), "
       "JSON_ARRAY(1, 'a', NULL, TRUE), JSON_ARRAY(1, NULL NULL ON NULL), JSON_ARRAY(SELECT gid FROM g WHERE gid < 3 ORDER BY gid), "
       "JSON_OBJECT('d' VALUE DATE '2020-01-02', 'n' VALUE 1.50, 'o' VALUE JSON_OBJECT('x' VALUE 1))",
       [('{"a":1,"b":"x","c":null}', "{}", '{"j":[1,2]}', '[1,"a",true]', "[1,null]", "[0,1,2]", '{"d":"2020-01-02","n":1.50,"o":{"x":1}}')])
refuses("SELECT JSON_OBJECT('a' VALUE 1, 'a' VALUE 2 WITH UNIQUE KEYS)", ["duplicate"])
expect("SELECT JSON_OBJECTAGG(KEY CAST(gid AS VARCHAR) VALUE gname), JSON_ARRAYAGG(gid ORDER BY gid DESC), JSON_ARRAYAGG(CASE WHEN gid < 2 THEN gid END ORDER BY gid NULL ON NULL) FROM g WHERE gid < 3",
       [('{"0":"g0","1":"G1","2":"G2"}', "[2,1,0]", "[0,1,null]")])
expect("SELECT '{\"a\":1}' IS JSON, '[1' IS JSON, '[1]' IS JSON OBJECT, '{\"a\":1,\"a\":2}' IS JSON WITH UNIQUE KEYS, '{\"a\":1,\"a\":2}' IS JSON, "
       "'1' IS JSON SCALAR, 'x' IS NOT JSON, CAST(NULL AS VARCHAR) IS JSON",
       [(True, False, False, False, True, True, True, None)])
expect("SELECT JSON('{\"a\" : 1}'), JSON_SCALAR(1.5), JSON_SCALAR('x'), JSON_SCALAR(DATE '2020-01-02'), JSON_SERIALIZE(JSON('[1, 2]')), typeof(JSON('1')), "
       "JSON('[1]') IS JSON ARRAY", [('{"a":1}', "1.5", '"x"', '"2020-01-02"', "[1,2]", "JSON", True)])
refuses("SELECT JSON('{\"a\":1,\"a\":2}' WITH UNIQUE KEYS)", ["duplicate"])
expect(jq("SELECT * FROM JSON_TABLE($J, '$.b[*]' COLUMNS (n FOR ORDINALITY, v INTEGER PATH '$', c VARCHAR PATH '$.c', x BOOLEAN EXISTS PATH '$.c', "
          "j JSON FORMAT JSON PATH '$', w VARCHAR PATH '$.c' DEFAULT 'none' ON EMPTY))"),
       [(1, 1, None, False, "1", "none"), (2, 2, None, False, "2", "none"), (3, None, "x", True, '{"c":"x"}', "x")], "JSON_TABLE columns")
expect("SELECT jt.* FROM JSON_TABLE('[{\"a\":1,\"xs\":[10,20],\"ys\":[\"p\"]},{\"a\":2,\"xs\":[]}]', '$[*]' COLUMNS (a INTEGER PATH '$.a', "
       "NESTED PATH '$.xs[*]' COLUMNS (x INTEGER PATH '$'), NESTED PATH '$.ys[*]' COLUMNS (y VARCHAR PATH '$'))) AS jt",
       [(1, 10, None), (1, 20, None), (1, None, "p"), (2, None, None)], "NESTED PATH: outer join to the parent, union of siblings")
expect("SELECT g.gid, jt.v FROM g, JSON_TABLE(JSON_ARRAY(g.gid, g.gid * 10), '$[*]' COLUMNS (v INTEGER PATH '$')) jt WHERE g.gid < 3 ORDER BY 1, 2",
       [(0, 0), (0, 0), (1, 1), (1, 10), (2, 2), (2, 20)], "JSON_TABLE reading an earlier FROM item (lateral)")
docs = pd.DataFrame(dict(id=range(5), doc=['{"name":"a","tags":["x","y"],"n":1,"o":{"p":10}}', '{"name":"b","tags":[],"n":2.5}', '{"name":"c","n":"3"}', None, "[1,2]"]))
dbj = torbles.from_ddl("CREATE TABLE d (id INTEGER, doc JSON)", {"d": docs})
r = dbj.sql("SELECT id, doc.name, doc.tags, doc.o.p, doc.tags[0], doc[1], JSON_VALUE(doc, '$.n' RETURNING DOUBLE), typeof(doc) FROM d ORDER BY id").rows
check(r == [(0, '"a"', '["x","y"]', "10", '"x"', None, 1.0, "JSON"), (1, '"b"', "[]", None, None, None, 2.5, "JSON"), (2, '"c"', None, None, None, None, 3.0, "JSON"),
            (3, None, None, None, None, None, None, "JSON"), (4, None, None, None, None, "2", None, "JSON")], "a stored JSON column; the simplified accessor j.a.b, j[i]", str(r))

print("== WITH RECURSIVE ... SEARCH / CYCLE, MATCH_RECOGNIZE, partitioned outer joins")
edges = pd.DataFrame(dict(src=[1, 1, 2, 3, 4, 5], dst=[2, 3, 4, 4, 1, 6]))
prices = [12, 17, 19, 21, 25, 12, 15, 20, 24, 25, 19, 15, 25, 25, 14, 12, 14, 24, 23, 22]
tick = pd.DataFrame(dict(symbol=["ACME"] * 20 + ["XYZ"] * 5, tstamp=list(range(1, 21)) + list(range(1, 6)), price=prices + [5, 4, 6, 3, 9]))
sales = pd.DataFrame(dict(product=["a", "a", "b"], day=[1, 3, 2], qty=[10, 30, 20]))
dbg = torbles.from_ddl("CREATE TABLE e (src INTEGER, dst INTEGER); CREATE TABLE ticker (symbol VARCHAR, tstamp INTEGER, price INTEGER); "
                       "CREATE TABLE days (day INTEGER PRIMARY KEY); CREATE TABLE sales (product VARCHAR, day INTEGER, qty INTEGER)",
                       {"e": edges, "ticker": tick, "days": pd.DataFrame(dict(day=[1, 2, 3])), "sales": sales})
def expect_g(sql, want, what):
    try: got = dbg.sql(sql).rows
    except Exception as x: check(False, what, f"{type(x).__name__}: {x}"); return
    check(got == want, what, f"torbles {got}\n        expected {want}")
walk = "WITH RECURSIVE r(n, d) AS (SELECT 1, 0 UNION ALL SELECT e.dst, r.d + 1 FROM r JOIN e ON e.src = r.n WHERE r.d < 3) "
expect_g(walk + "SEARCH DEPTH FIRST BY n SET ord SELECT n, d FROM r ORDER BY ord",
         [(1, 0), (2, 1), (4, 2), (1, 3), (3, 1), (4, 2), (1, 3)], "SEARCH DEPTH FIRST")
expect_g(walk + "SEARCH BREADTH FIRST BY n SET ord SELECT n, d FROM r ORDER BY ord, n",
         [(1, 0), (2, 1), (3, 1), (4, 2), (4, 2), (1, 3), (1, 3)], "SEARCH BREADTH FIRST")
cyc = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT e.dst FROM r JOIN e ON e.src = r.n) "
expect_g(cyc + "CYCLE n SET is_cycle USING path SELECT n, is_cycle, CARDINALITY(path) FROM r ORDER BY path",
         [(1, False, 1), (2, False, 2), (4, False, 3), (1, True, 4), (3, False, 2), (4, False, 3), (1, True, 4)], "CYCLE: rows on a cycle marked, not expanded")
expect_g(cyc + "CYCLE n SET c TO 'Y' DEFAULT 'N' USING p SELECT count(*), count(*) FILTER (WHERE c = 'Y') FROM r", [(7, 2)], "CYCLE ... TO 'Y' DEFAULT 'N'")
mr = """SELECT * FROM ticker MATCH_RECOGNIZE (PARTITION BY symbol ORDER BY tstamp
        MEASURES STRT.tstamp AS start_tstamp, LAST(DOWN.tstamp) AS bottom_tstamp, LAST(UP.tstamp) AS end_tstamp {extra}
        {rows} AFTER MATCH SKIP TO LAST UP PATTERN (STRT DOWN+ UP+)
        DEFINE DOWN AS DOWN.price < PREV(DOWN.price), UP AS UP.price > PREV(UP.price)) MR ORDER BY MR.symbol, MR.start_tstamp{order}"""
expect_g(mr.format(extra="", rows="ONE ROW PER MATCH", order=""),
         [("ACME", 5, 6, 10), ("ACME", 10, 12, 13), ("ACME", 14, 16, 18), ("XYZ", 1, 2, 3), ("XYZ", 3, 4, 5)], "MATCH_RECOGNIZE: ONE ROW PER MATCH (the V shapes)")
got = dbg.sql(mr.format(extra=", MATCH_NUMBER() AS m, CLASSIFIER() AS v, FINAL COUNT(UP.*) AS ups, RUNNING COUNT(*) AS so_far",
                        rows="ALL ROWS PER MATCH", order=", MR.tstamp")).rows
check([r[:2] + r[5:9] for r in got[:6]] == [("ACME", 5, 1, "STRT", 4, 1), ("ACME", 6, 1, "DOWN", 4, 2), ("ACME", 7, 1, "UP", 4, 3),
                                            ("ACME", 8, 1, "UP", 4, 4), ("ACME", 9, 1, "UP", 4, 5), ("ACME", 10, 1, "UP", 4, 6)] and len(got) == 21,
      "ALL ROWS PER MATCH: CLASSIFIER, MATCH_NUMBER, FINAL and RUNNING aggregates", str(got[:6]))
expect_g("""SELECT * FROM ticker MATCH_RECOGNIZE (PARTITION BY symbol ORDER BY tstamp MEASURES FIRST(A.tstamp) AS s, LAST(A.tstamp) AS e,
        COUNT(A.*) AS n, AVG(A.price) AS av, SUM(price) AS tot PATTERN (A{3,}) DEFINE A AS A.price >= 15) WHERE symbol = 'ACME' ORDER BY s""",
         [("ACME", 2, 5, 4, 20.5, 82), ("ACME", 7, 14, 8, 21.0, 168), ("ACME", 18, 20, 3, 23.0, 69)], "quantifier {3,}, FIRST / LAST, aggregates")
expect_g("""SELECT s, n FROM ticker MATCH_RECOGNIZE (PARTITION BY symbol ORDER BY tstamp MEASURES FIRST(A.tstamp) AS s, COUNT(*) AS n
        AFTER MATCH SKIP TO NEXT ROW PATTERN (A B*? C) DEFINE A AS price < 15, C AS price > 20) WHERE symbol = 'ACME' ORDER BY s""",
         [(1, 4), (6, 4), (15, 4), (16, 3), (17, 2)], "a reluctant quantifier, SKIP TO NEXT ROW")
expect_g("""SELECT s, e FROM ticker MATCH_RECOGNIZE (PARTITION BY symbol ORDER BY tstamp MEASURES FIRST(tstamp) AS s, LAST(tstamp) AS e
        PATTERN (PERMUTE(A, B) | ^ C) DEFINE A AS price = 5, B AS price = 4, C AS price > 100) WHERE symbol = 'XYZ'""",
         [(1, 2)], "PERMUTE and anchors")
expect_g("SELECT s.product, d.day, s.qty FROM sales s PARTITION BY (s.product) RIGHT JOIN days d ON s.day = d.day ORDER BY 1, 2",
         [("a", 1, 10), ("a", 2, None), ("a", 3, 30), ("b", 1, None), ("b", 2, 20), ("b", 3, None)], "a partitioned outer join (F403) densifies each partition")

print("== more ISO forms")
expect("SELECT 1 AS a, 2 AS b UNION CORRESPONDING BY (b) SELECT 3 AS b, 4 AS a ORDER BY 1", [(2,), (3,)], "CORRESPONDING BY (columns)")
expect("SELECT POSITION_REGEX('(a)(n)' IN 'banana' GROUP 2), SUBSTRING_REGEX('(a)(n)' IN 'banana' GROUP 2), OCCURRENCES_REGEX('an' IN 'banana' FROM 3), "
       "POSITION_REGEX(AFTER 'an' IN 'banana'), POSITION_REGEX('l' IN 'héllo' USING OCTETS), POSITION_REGEX('n' IN 'banana' OCCURRENCE 2)",
       [(3, "n", 1, 4, 4, 5)], "the *_REGEX functions: GROUP, AFTER, FROM, USING OCTETS, OCCURRENCE")
expect("SELECT CAST('2020/03/04' AS DATE FORMAT 'YYYY/MM/DD'), CAST('04.03.2020 13:05' AS TIMESTAMP FORMAT 'DD.MM.YYYY HH24:MI'), "
       "CAST(DATE '2020-03-04' AS VARCHAR(20) FORMAT 'DD/MM/YYYY'), CAST(TIMESTAMP '2020-03-04 15:06:07.123' AS VARCHAR FORMAT 'YYYY-MM-DD HH12:MI:SS.FF3 A.M.'), "
       "CAST('10:20 +02:00' AS TIME WITH TIME ZONE FORMAT 'HH24:MI TZH:TZM')",
       [(dt.date(2020, 3, 4), dt.datetime(2020, 3, 4, 13, 5), "04/03/2020", "2020-03-04 03:06:07.123 P.M.", dt.time(10, 20, tzinfo=tz(2)))], "CAST ... FORMAT (T839)")
expect("SELECT NORMALIZE('é', NFD, 10) = NORMALIZE('é', NFD), ANY_VALUE(gid) IS NOT NULL FROM g", [(True, True)])
vs_duckdb("SELECT k, sum(v) OVER (w ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t WHERE id < 30 WINDOW w AS (PARTITION BY k) ORDER BY id",
          "a window defined from a named window")

print("== DECFLOAT (T076)")
expect("SELECT CAST(1 AS DECFLOAT(16)) / 3, CAST(1 AS DECFLOAT) / 3, CAST(id AS DECFLOAT(7)) * 1234567, CAST('1.20' AS DECFLOAT) = 1.2 FROM t WHERE id = 3",
       [(Decimal("0.3333333333333333"), Decimal("0.3333333333333333333333333333333333"), Decimal("3703701"), True)], "rounded to p digits")
expect("SELECT typeof(CAST(1 AS DECFLOAT(16)) / 3), typeof(CAST(1 AS DECFLOAT) + 1.5), CAST(CAST(1 AS DECFLOAT(16)) / 3 AS VARCHAR)",
       [("DECFLOAT(16)", "DECFLOAT(34)", "0.3333333333333333")], "result types and text")
expect("SELECT CAST('NaN' AS DECFLOAT) = CAST('NaN' AS DECFLOAT), CAST('-Infinity' AS DECFLOAT) < -1e300, CAST('Infinity' AS DECFLOAT) > 1, "
       "CAST('NaN' AS DECFLOAT) > CAST('Infinity' AS DECFLOAT)", [(True, True, True, True)], "special values in a total order")
expect("SELECT sum(CAST(v AS DECFLOAT(16))) = sum(v), max(CAST(v AS DECFLOAT)) = max(v) FROM t", [(True, True)], "aggregates")
refuses("SELECT CAST('1E+400' AS DECFLOAT(16))", ["out of range", "DECFLOAT"], "exponent overflow is a data exception")
refuses("SELECT CAST('1E+300' AS DECFLOAT(16)) * CAST('1E+300' AS DECFLOAT(16))", ["out of range"], "overflow in arithmetic")

print("ALL MATCH" if not fails else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
