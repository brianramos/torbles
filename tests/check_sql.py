# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The SQL front end (db.sql and the Python API) on a typed star schema with NULLs and unmatched references:
  1. scalar functions and operators over table columns, every result compared with DuckDB's;
  2. aggregates, grouping sets, window functions, joins of every kind, subqueries, CTEs, set operations: DuckDB;
  3. ISO/IEC 9075 forms DuckDB does not have (BETWEEN SYMMETRIC, LIKE_REGEX, the *_REGEX functions, SIMILAR TO,
     SUBSTRING ... SIMILAR, WIDTH_BUCKET, hypothetical set functions, LISTAGG ... WITHIN GROUP, UNIQUE, OVERLAPS,
     FETCH ... WITH TIES / PERCENT, NTH_VALUE ... FROM LAST, IGNORE NULLS after the call, INTERVAL ... TO ...,
     CORRESPONDING, EVERY / ANY / SOME), checked against their values worked out by hand;
  4. the Python API (db.select, F), each query also rendered as SQL text and run by DuckDB;
  5. a short SQL fuzzing run (python -m torbles.sqlfuzz)."""
import datetime as dt, os, random, subprocess, sys
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import duckdb, pandas as pd
import torbles
from torbles import F
import sqlfuzz

fails = 0


def check(ok, what, detail=""):
    global fails
    if not ok: fails += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"\n        {detail}" if not ok and detail else ""))


ddl = """CREATE TABLE g (gid INTEGER PRIMARY KEY, gname VARCHAR, gw DECIMAL(5,2), gday DATE);
CREATE TABLE t (id INTEGER, k INTEGER REFERENCES g(gid), i INTEGER, j INTEGER, b BIGINT, d DECIMAL(9,2), e DECIMAL(18,4), f DOUBLE,
                s VARCHAR, u VARCHAR, dt DATE, ts TIMESTAMP, flag BOOLEAN);"""
r = random.Random(11)
gdf = pd.DataFrame([(1, "one", Decimal("1.5"), dt.date(2020, 1, 1)), (2, "two", None, dt.date(2021, 7, 4)),
                    (3, "three", Decimal("-2.25"), None), (4, None, Decimal("0"), dt.date(1999, 12, 31))],
                   columns=["gid", "gname", "gw", "gday"])
rows = []
for n in range(80):
    rows.append((n, r.choice([1, 2, 3, 4, None, 9]), r.choice([None, -7, -3, 0, 1, 2, 2, 5, 9, 13, 100]), r.choice([None, -4, 1, 2, 3, 5]),
                 r.choice([None, 0, -5, 10**12, -(2**62), 3]), r.choice([None, Decimal("1.25"), Decimal("-0.50"), Decimal("3.00"), Decimal("10.10"), Decimal("2.55")]),
                 r.choice([None, Decimal("-3.1416"), Decimal("123456.7891"), Decimal("0.0001"), Decimal("2.5000")]),
                 r.choice([None, 0.5, -1.25, 2.0, 1e10, 0.1, -0.0]), r.choice([None, "a", "b", "bb", "Hello", "straße", "", "ababab", "日本"]),
                 r.choice([None, "a_b%c", "x", "  pad  ", "AbC"]),
                 r.choice([None, dt.date(2020, 1, 31), dt.date(2021, 6, 30), dt.date(1999, 12, 31), dt.date(2024, 2, 29)]),
                 r.choice([None, dt.datetime(2020, 2, 29, 13, 45, 7, 120000), dt.datetime(1970, 1, 1), dt.datetime(2024, 12, 31, 23, 59, 59)]),
                 r.choice([None, True, False])))
df = pd.DataFrame(rows, columns=["id", "k", "i", "j", "b", "d", "e", "f", "s", "u", "dt", "ts", "flag"])
ENGINE = os.environ.get("TORBLES_SQL_ENGINE", "alg")             # alg | plain | gpu
db = torbles.from_ddl(ddl, {"g": gdf, "t": df}, keep_plain=ENGINE == "plain")
if ENGINE != "alg":                                               # every db.sql / Python API run on that engine
    _sql = db.sql; db.sql = lambda text, engine=ENGINE: _sql(text, engine)
    import sqlapi; _run = sqlapi.Query.run; sqlapi.Query.run = lambda self, engine=ENGINE: _run(self, engine)
con = duckdb.connect(); con.execute(ddl.replace(" REFERENCES g(gid)", ""))
con.register("gdf", gdf); con.register("df", df)
con.execute("INSERT INTO g SELECT * FROM gdf"); con.execute("INSERT INTO t SELECT * FROM df")


def vs_duckdb(sql, what=None, tol=1e-9):
    try: want = con.execute(sql).fetchall()
    except Exception as x: check(False, what or sql, f"DuckDB: {x}"); return
    try: got = db.sql(sql).rows
    except Exception as x: check(False, what or sql, f"torbles: {type(x).__name__}: {x}"); return
    check(sqlfuzz.same(got, want, True, tol), what or sql[:110], f"torbles {got[:4]}\n        duckdb  {want[:4]}")


print("== scalar functions and operators (vs DuckDB)")
exprs = """i + j | i - j | i * j | i / j | i // nullif(j, 0) | i % nullif(j, 0) | -i | abs(i) | sign(i) | d + e | d * e | d - 1 | d / 3 | f + i | f * 2
round(d, 1) | round(e, 2) | round(f) | floor(d) | ceil(e) | trunc(e, 1) | sqrt(abs(f)) | power(i, 2) | ln(abs(f) + 1) | exp(j) | cbrt(i) | mod(i, 3)
greatest(i, j, 3) | least(i, j) | coalesce(i, j, 0) | nullif(i, 2) | i between j and 10 | i in (2, 13) | i not in (0, 1) | i is null | i is distinct from j
length(s) | upper(s) | lower(s) | substring(s, 2, 3) | substr(s, -2) | position('a' in s) | left(s, 2) | right(s, 3) | reverse(s) | trim(u) | ltrim(u)
trim(both 'a' from s) | lpad(s, 8, '*') | rpad(s, 8, '-') | repeat(s, 2) | replace(s, 'b', 'XY') | translate(s, 'ab', 'xy') | s || '-' || u | concat(s, NULL, u)
s like 'H%' | s like '%a_' | u like 'a\\_b%' escape '\\' | s ilike '%L%' | starts_with(s, 'ab') | split_part(s, 'a', 2) | regexp_matches(s, '[a-z]+') | regexp_replace(s, 'a', 'Z', 'g')
dt + 1 | dt + interval '1' month | ts + interval '90' minute | dt - date '2000-01-01' | ts - timestamp '2000-01-01 00:00:00' | extract(year from dt) | extract(month from ts)
extract(dow from dt) | extract(doy from dt) | extract(hour from ts) | extract(epoch from ts) | date_trunc('month', dt) | last_day(dt) | dayname(dt) | date_diff('day', dt, date '2020-06-01')
cast(i as varchar) | cast(d as varchar) | cast(f as varchar) | cast(dt as varchar) | cast(ts as varchar) | cast(d as integer) | cast(f as integer) | try_cast(s as integer)
case when i > 0 then 'pos' when i < 0 then 'neg' else 'zero' end | case i when 2 then 'two' end | case when flag then d else e end | not flag | flag and i > 0 | flag is not false
f > 1 | d > 1.5 | s < 'b' | dt < date '2020-01-01' | dt = '2020-01-31' | i = '2' | d = 1.5 | sin(f) | atan2(i, j) | isnan(f)""".replace("\n", " | ").split(" | ")
bad = []
for e in exprs:
    sql = f"SELECT id, {e.strip()} FROM t ORDER BY id"
    try: want = con.execute(sql).fetchall()
    except Exception: continue                                    # DuckDB rejects (an overflow): not a comparison
    try: got = db.sql(sql).rows
    except Exception as x: bad.append(f"{e}: {x}"); continue
    if not sqlfuzz.same(got, want, True, 1e-12): bad.append(f"{e}: {got[:3]} vs {want[:3]}")
check(not bad, f"{len(exprs)} expressions on every row", "\n        ".join(bad[:8]))

print("== aggregates, grouping sets, windows, joins, subqueries (vs DuckDB)")
for q in """SELECT k, count(*), count(i), count(DISTINCT i), sum(i), avg(i), min(i), max(i), sum(DISTINCT i), avg(DISTINCT d) FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT k, sum(d), avg(d), min(d), max(d), sum(f), min(s), max(s), min(dt), max(ts), bool_and(flag), bool_or(flag) FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT k, stddev_pop(i), stddev_samp(i), var_pop(d), var_samp(d), covar_pop(i, d), covar_samp(i, d), regr_count(i, d), regr_avgx(i, d), regr_sxx(i, d), regr_sxy(i, d) FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT k, median(i), median(d), quantile_cont(i, 0.25), quantile_disc(i, 0.75), percentile_cont(0.4) WITHIN GROUP (ORDER BY d), percentile_disc(0.5) WITHIN GROUP (ORDER BY i DESC), mode() WITHIN GROUP (ORDER BY s) FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT k, string_agg(s, ',' ORDER BY s, id), string_agg(DISTINCT s, '|' ORDER BY s), array_agg(i ORDER BY id), count(*) FILTER (WHERE flag), sum(i) FILTER (WHERE s = 'a') FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT k, sum(CASE WHEN flag THEN i ELSE 0 END), sum(CASE WHEN s = 'a' THEN d END), count(CASE WHEN i > 1 THEN 1 END), avg(CASE WHEN dt > DATE '2020-06-01' THEN e WHEN flag THEN 1 END) FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT gname, count(*), sum(i * gw), avg(i + 1), max(gday) FROM t JOIN g ON t.k = g.gid GROUP BY gname ORDER BY gname NULLS FIRST
SELECT gname, count(*), sum(i), max(gw) FROM t LEFT JOIN g ON t.k = g.gid GROUP BY gname ORDER BY gname NULLS FIRST
SELECT gname, count(*), count(gw) FROM t LEFT JOIN g ON t.k = g.gid AND g.gw > 0 AND t.i > 1 GROUP BY gname ORDER BY gname NULLS FIRST
SELECT gname, count(i), sum(i) FROM t RIGHT JOIN g ON t.k = g.gid GROUP BY gname ORDER BY gname NULLS FIRST
SELECT gname, count(i), sum(i) FROM t FULL JOIN g ON t.k = g.gid GROUP BY gname ORDER BY gname NULLS FIRST
SELECT k, count(*), sum(i) FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT k, s, count(*), grouping(k, s), sum(i) FROM t GROUP BY ROLLUP (k, s) ORDER BY k NULLS FIRST, s NULLS FIRST, 4
SELECT k, flag, count(*) FROM t GROUP BY CUBE (k, flag) ORDER BY 1 NULLS FIRST, 2 NULLS FIRST, 3
SELECT k, flag, count(*) FROM t GROUP BY GROUPING SETS ((k), (flag), ()) ORDER BY 1 NULLS FIRST, 2 NULLS FIRST, 3
SELECT id, i, row_number() OVER (ORDER BY id), rank() OVER (ORDER BY i NULLS FIRST), dense_rank() OVER (ORDER BY i), percent_rank() OVER (ORDER BY i), cume_dist() OVER (ORDER BY i), ntile(4) OVER (ORDER BY id) FROM t ORDER BY id
SELECT id, k, i, sum(i) OVER (PARTITION BY k ORDER BY id), avg(i) OVER (PARTITION BY k), count(*) OVER (PARTITION BY k ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t ORDER BY id
SELECT id, i, lag(i) OVER (ORDER BY id), lead(i, 2) OVER (ORDER BY id), lag(i, 1, -1) OVER (ORDER BY id), first_value(i) OVER (ORDER BY id), last_value(i) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), nth_value(i, 3) OVER (ORDER BY id) FROM t ORDER BY id
SELECT id, i, sum(i) OVER (ORDER BY i RANGE BETWEEN 2 PRECEDING AND 2 FOLLOWING), count(*) OVER (ORDER BY i GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING), sum(i) OVER (ORDER BY id ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) FROM t ORDER BY id
SELECT id, i, sum(i) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING EXCLUDE CURRENT ROW), count(*) OVER (ORDER BY i RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING EXCLUDE TIES) FROM t ORDER BY id
SELECT id, dt, count(*) OVER (ORDER BY dt RANGE BETWEEN INTERVAL 400 DAY PRECEDING AND CURRENT ROW) FROM t ORDER BY id
SELECT id, i, lag(i IGNORE NULLS) OVER (ORDER BY id), first_value(i IGNORE NULLS) OVER (ORDER BY id), last_value(i IGNORE NULLS) OVER (ORDER BY id) FROM t ORDER BY id
SELECT k, sum(i), rank() OVER (ORDER BY sum(i) DESC NULLS LAST, k), sum(sum(i)) OVER () FROM t GROUP BY k ORDER BY k NULLS FIRST
SELECT id, i FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY id) <= 2 ORDER BY id
SELECT id, i FROM t t1 WHERE i > (SELECT avg(i) FROM t t2 WHERE t2.k = t1.k) ORDER BY id
SELECT id FROM t t1 WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.k = t1.k AND t2.i > t1.i + 50) ORDER BY id
SELECT id FROM t WHERE k IN (SELECT gid FROM g WHERE gw > 0) ORDER BY id
SELECT id FROM t WHERE k NOT IN (SELECT gid FROM g WHERE gw IS NOT NULL) ORDER BY id
SELECT id FROM t WHERE i > ALL (SELECT i FROM t WHERE s = 'b' AND i IS NOT NULL) ORDER BY id
SELECT gname, (SELECT count(*) FROM t WHERE t.k = g.gid), (SELECT max(i) FROM t WHERE t.k = g.gid) FROM g ORDER BY gid
WITH a AS (SELECT k, sum(i) s FROM t GROUP BY k), b AS (SELECT gid, gname FROM g) SELECT gname, s FROM a JOIN b ON a.k = b.gid ORDER BY gname
WITH RECURSIVE r(n, f) AS (SELECT 1, 1 UNION ALL SELECT n + 1, f * (n + 1) FROM r WHERE n < 10) SELECT n, f FROM r ORDER BY n
SELECT s FROM t UNION SELECT gname FROM g ORDER BY 1 NULLS FIRST
SELECT s FROM t EXCEPT ALL SELECT s FROM t WHERE i > 1 ORDER BY 1 NULLS FIRST
SELECT s FROM t INTERSECT SELECT u FROM t ORDER BY 1
SELECT DISTINCT ON (k) k, id, i FROM t ORDER BY k NULLS FIRST, id DESC
SELECT * FROM t ORDER BY id LIMIT 5 OFFSET 10
SELECT t.*, g.gname FROM t NATURAL JOIN (SELECT gid AS k, gname FROM g) g ORDER BY id
SELECT a.id, b.id FROM t a JOIN t b ON a.i = b.i AND a.id < b.id WHERE a.k = 1 ORDER BY 1, 2
SELECT gid, count(t.id) FROM g LEFT JOIN t ON g.gid = t.k AND t.flag GROUP BY gid ORDER BY gid
SELECT k, i, count(*) FROM t GROUP BY k, i ORDER BY 1 NULLS FIRST, 2 NULLS FIRST
SELECT k, count(*), count(k) FROM t GROUP BY k ORDER BY 1 NULLS FIRST
SELECT count(*), sum(i) FROM t WHERE false
SELECT x, count(*) FROM (VALUES (1), (2), (NULL), (1)) v(x) GROUP BY x ORDER BY x NULLS FIRST""".split("\n"):
    vs_duckdb(q)

print("== ISO forms (values worked out by hand)")
iso = [
    ("SELECT 5 BETWEEN SYMMETRIC 10 AND 1, 5 NOT BETWEEN SYMMETRIC 10 AND 6", [(True, True)]),
    ("SELECT 'abc' LIKE_REGEX 'b', 'abc' LIKE_REGEX '^b', 'ABC' LIKE_REGEX 'b' FLAG 'i'", [(True, False, True)]),
    ("SELECT OCCURRENCES_REGEX('a' IN 'banana'), POSITION_REGEX('n' IN 'banana'), POSITION_REGEX('n' IN 'banana' OCCURRENCE 2)", [(3, 3, 5)]),
    ("SELECT SUBSTRING_REGEX('a+n' IN 'banana'), TRANSLATE_REGEX('a' IN 'banana' WITH 'o')", [("an", "bonono")]),
    ("SELECT SUBSTRING('abcdef' SIMILAR 'a#\"bcd#\"ef' ESCAPE '#')", [("bcd",)]),
    ("SELECT 'abc' SIMILAR TO 'a%', 'abc' SIMILAR TO '(a|b)c', 'bc' SIMILAR TO '(a|b)c', 'a.c' SIMILAR TO 'a.c', 'abc' SIMILAR TO 'a.c'", [(True, False, True, True, False)]),
    ("SELECT OVERLAY('abcdef' PLACING 'XY' FROM 2 FOR 3), OVERLAY('abcdef' PLACING 'XY' FROM 2)", [("aXYef", "aXYdef")]),
    ("SELECT WIDTH_BUCKET(5.5, 0, 10, 4), WIDTH_BUCKET(-1, 0, 10, 4), WIDTH_BUCKET(10, 0, 10, 4), WIDTH_BUCKET(2, 10, 0, 5)", [(3, 0, 5, 5)]),
    ("SELECT RANK(2) WITHIN GROUP (ORDER BY x), DENSE_RANK(3) WITHIN GROUP (ORDER BY x), PERCENT_RANK(2) WITHIN GROUP (ORDER BY x), "
     "CUME_DIST(2) WITHIN GROUP (ORDER BY x) FROM (VALUES (1), (2), (2), (4)) v(x)", [(2, 3, 0.25, 0.8)]),
    ("SELECT RANK(2, 'b') WITHIN GROUP (ORDER BY x, y) FROM (VALUES (1, 'z'), (2, 'a'), (2, 'c'), (4, 'a')) v(x, y)", [(3,)]),
    ("SELECT LISTAGG(x, '-') WITHIN GROUP (ORDER BY x DESC) FROM (VALUES ('a'), ('c'), ('b')) v(x)", [("c-b-a",)]),
    ("SELECT UNIQUE (SELECT x FROM (VALUES (1), (2), (NULL), (NULL)) v(x)), UNIQUE (SELECT x FROM (VALUES (1), (1)) v(x))", [(True, False)]),
    ("SELECT (DATE '2020-01-01', DATE '2020-02-01') OVERLAPS (DATE '2020-01-15', DATE '2020-03-01'), "
     "(DATE '2020-01-01', DATE '2020-02-01') OVERLAPS (DATE '2020-02-01', DATE '2020-03-01')", [(True, False)]),
    ("SELECT x FROM (VALUES (1), (2), (2), (3)) v(x) ORDER BY x FETCH FIRST 2 ROWS WITH TIES", [(1,), (2,), (2,)]),
    ("SELECT x FROM (VALUES (1), (2), (3)) v(x) ORDER BY x FETCH FIRST 50 PERCENT ROWS ONLY", [(1,), (2,)]),
    ("SELECT x, NTH_VALUE(x, 2) FROM LAST OVER (ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM (VALUES (1), (2), (3), (4)) v(x) ORDER BY x",
     [(1, 3), (2, 3), (3, 3), (4, 3)]),
    ("SELECT i, LAG(x) IGNORE NULLS OVER (ORDER BY i) FROM (VALUES (1, 10), (2, NULL), (3, 30)) v(i, x) ORDER BY i", [(1, None), (2, 10), (3, 10)]),
    ("SELECT INTERVAL '1-2' YEAR TO MONTH, INTERVAL '3 04:05:06' DAY TO SECOND, INTERVAL '1:30' HOUR TO MINUTE",
     [(dt.timedelta(days=420), dt.timedelta(days=3, hours=4, minutes=5, seconds=6), dt.timedelta(minutes=90))]),
    ("SELECT x IS UNKNOWN, x IS NOT UNKNOWN FROM (VALUES (CAST(NULL AS BOOLEAN)), (TRUE)) v(x)", [(True, False), (False, True)]),
    ("SELECT EVERY(x > 0), ANY(x > 3), SOME(x > 9) FROM (VALUES (1), (2), (4)) v(x)", [(True, True, False)]),
    ("SELECT CORR(x, y), REGR_SLOPE(x, y), REGR_R2(x, y) FROM (VALUES (1, 2), (3, 2)) v(x, y)", [(None, None, None)]),
    ("SELECT 1 AS a, 2 AS b UNION ALL CORRESPONDING SELECT 3 AS b, 4 AS a, 5 AS c", [(1, 2), (4, 3)]),
    ("SELECT PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY x DESC) FROM (VALUES (1), (2), (3), (4)) v(x)", [(3,)]),
    ("SELECT GREATEST(1, 5, 3), LEAST('b', 'a'), BTRIM('xxaxx', 'x'), LPAD('7', 3, '0'), RPAD('ab', 5, 'xy')", [(5, "a", "a", "007", "abxyx")]),
    ("SELECT CHAR_LENGTH('héllo'), CHARACTER_LENGTH('ab'), OCTET_LENGTH('héllo'), POSITION('l' IN 'hello')", [(5, 2, 6, 3)]),
    ("SELECT MOD(-7, 3), ABS(-2.5), CEILING(1.2), LN(1), EXP(0), POWER(2, 10), SQRT(16), FLOOR(-1.5), LOG10(1000), LOG(2, 8)", [(-1, Decimal("2.5"), Decimal("2"), 0.0, 1.0, 1024.0, 4.0, Decimal("-2"), 3.0, 3.0)]),
    ("SELECT NORMALIZE('é', NFD) = 'é', NULLIF(1, 1), COALESCE(NULL, 2), 1 IS DISTINCT FROM NULL, NULL IS NOT DISTINCT FROM NULL", [(False, None, 2, True, True)]),
    ("SELECT (1, NULL) = (1, 2), (1, 2) < (1, 3), (2, 0) > (1, 9)", [(None, True, True)]),
    ("SELECT EXTRACT(YEAR FROM DATE '2020-05-06'), EXTRACT(TIMEZONE_HOUR FROM TIMESTAMP '2020-01-01 10:00:00'), CAST('2020-02-29' AS DATE) + INTERVAL '1' YEAR",
     [(2020, 0, dt.datetime(2021, 2, 28))]),
]
for sql, want in iso:
    try: got = db.sql(sql).rows
    except Exception as x: check(False, sql[:100], f"{type(x).__name__}: {x}"); continue
    check(sqlfuzz.same(got, want, True, 1e-12), sql[:100], f"torbles {got}\n        expected {want}")

print("== the Python API (each query also run by DuckDB as SQL text)")
tt, gg = db.table("t"), db.table("g")
api = {
    "aggregates by a dimension (implicit foreign-key join)": db.select(gg.gname, F.count(), F.sum(tt.i), F.avg(tt.d), F.max(tt.s)).group_by(gg.gname).order_by(gg.gname.asc(nulls_first=True)),
    "LEFT JOIN with ON terms": db.select(gg.gname, F.count(), F.count(gg.gw)).from_(tt).join(gg, (tt.k == gg.gid) & (gg.gw > 0), "left").group_by(gg.gname).order_by(gg.gname.asc(nulls_first=True)),
    "CASE, FILTER, percentile": db.select(tt.k, F.sum(F.case((tt.flag, tt.i), else_=0)), F.count(filter=tt.i > 1), F.percentile_cont(0.5).within_group(tt.i)).from_(tt).group_by(tt.k).order_by(tt.k.asc(nulls_first=True)),
    "window over aggregates": db.select(tt.k, F.sum(tt.i).alias("s"), F.rank().over(order_by=[F.desc(F.sum(tt.i), nulls_first=False), tt.k])).from_(tt).group_by(tt.k).order_by(tt.k.asc(nulls_first=True)),
    "frames": db.select(tt.id, F.sum(tt.i).over(order_by=[tt.id], rows=(-2, 0)), F.lag(tt.i, 1, 0).over(order_by=[tt.id])).from_(tt).order_by(tt.id),
    "ROLLUP and GROUPING": db.select(tt.k, tt.flag, F.count(), F.grouping(tt.k, tt.flag)).from_(tt).rollup(tt.k, tt.flag).order_by(tt.k.asc(nulls_first=True), tt.flag.asc(nulls_first=True), F.grouping(tt.k, tt.flag)),
    "subquery IN": db.select(F.count()).from_(tt).where(tt.k.isin(db.select(gg.gid).from_(gg).where(gg.gw > 0))),
    "string and date functions": db.select(tt.id, F.upper(F.substring(tt.s, 1, 3)), F.extract("year", tt.dt), tt.dt + F.interval(1, "month")).from_(tt).order_by(tt.id).limit(10),
}
for what, q in api.items():
    try: ok, got, want = q.verify()
    except Exception as x: check(False, what, f"{type(x).__name__}: {x}\n        {q.to_sql()}"); continue
    check(ok, what, f"torbles {got[:3]}\n        duckdb  {want[:3]}\n        {q.to_sql()}")

if ENGINE != "alg":
    print("ALL MATCH" if not fails else f"{fails} FAILURES"); sys.exit(1 if fails else 0)
print("== SQL fuzzing against DuckDB (seed 3, 12 cases)")
here = os.path.dirname(os.path.abspath(__file__))
p = subprocess.run([sys.executable, "-m", "torbles.sqlfuzz", "--seed", "3", "--cases", "12", "--queries", "25"], cwd=os.path.join(here, ".."),
                   capture_output=True, text=True)
check(p.returncode == 0 and "NO MISMATCHES" in p.stdout, "sqlfuzz", p.stdout[-1500:] + p.stderr[-1500:])
print("ALL MATCH" if not fails else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
