# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The installed package at its dependency floor (release review R01: on the floor declared then, Python 3.9 / NumPy
1.22 / DuckDB 1.1, even Database().sql("SELECT 1") failed). Run outside the source tree against the installed wheel,
with only the declared dependencies (no pandas, no pytz):
  1. the torbles imported is the installed distribution; the Python, NumPy and DuckDB found satisfy the floors it
     declares (with --floor: are those floors), and its SQL front end enforces the DuckDB floor it declares;
  2. SQL on a typed load (from_duckdb, keys from PRIMARY KEY / REFERENCES; INTEGER, BIGINT, DECIMAL, VARCHAR, DATE,
     TIMESTAMP WITH TIME ZONE, BOOLEAN, NULLs): SELECT 1, SUM, GROUP BY, joins, a window, a CTE, ... compared with
     DuckDB (TIMESTAMP WITH TIME ZONE with exact values: DuckDB returns it through pytz) on the CPU engine, the plain
     columnar baseline and the GPU executor (or its CPU emulation), and the Python API;
  3. a DuckDB below the floor is refused by name at the first SQL use (with --below-floor: the one installed is).
    mkdir /tmp/t && cp -r tests /tmp/t && cd /tmp/t && python tests/check_min_deps.py [--floor | --below-floor]"""
import datetime as dt, importlib.metadata as md, os, platform, re, sys, types

mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode not in ("", "--floor", "--below-floor"): sys.exit(__doc__)
import numpy as np
import duckdb
import torbles
from torbles import F
import sqlfront                                             # the installed package's (torbles puts its directory on sys.path)

fails = 0


def check(ok, what, detail=""):
    global fails
    if not ok: fails += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"\n        {detail}" if not ok and detail else ""))


def ver(text, n=3):
    """the leading numbers of a version, padded to n: '2.0' -> (2, 0, 0)"""
    m = re.match(r"\d+(?:\.\d+)*", text)
    v = [int(x) for x in m.group().split(".")][:n] if m else []
    return tuple(v + [0] * (n - len(v)))


have = dict(python=platform.python_version(), numpy=np.__version__, duckdb=duckdb.__version__)
print(f"== torbles {torbles.__version__} at {os.path.dirname(torbles.__file__)}")
print(f"   Python {have['python']}, NumPy {have['numpy']}, DuckDB {have['duckdb']}")
try: dist = md.distribution("torbles")
except md.PackageNotFoundError: dist = None
where = os.path.realpath(str(dist.locate_file("torbles/__init__.py"))) if dist else None
installer = ((dist.read_text("INSTALLER") or "").strip() if dist else "") or None    # a build's egg-info in a checkout has none
check(where == os.path.realpath(torbles.__file__) and installer is not None,
      f"the torbles imported is the installed distribution (installed by {installer})",
      f"imported {torbles.__file__}, installed {where} by {installer}: run it outside the source tree, without PYTHONPATH")
floor = {}
for r in (dist.requires or []) if dist else []:
    m = re.fullmatch(r"\s*(numpy|duckdb)\s*>=\s*([0-9.]+)\s*", r)        # the extras carry markers: they do not match
    if m: floor[m.group(1)] = m.group(2)
m = re.fullmatch(r"\s*>=\s*([0-9.]+)\s*", (dist.metadata.get("Requires-Python") or "") if dist else "")
if m: floor["python"] = m.group(1)
check(sorted(floor) == ["duckdb", "numpy", "python"], f"declared floors {floor}", f"Requires-Dist {dist.requires if dist else None}")
check(ver(floor.get("duckdb", "0"), 2) == sqlfront.DUCKDB_FLOOR,
      f"the SQL front end enforces the declared DuckDB floor (DUCKDB_FLOOR {sqlfront.DUCKDB_FLOOR})")
for k in ("python", "numpy", "duckdb"):
    if k not in floor: continue
    if k == "duckdb" and mode == "--below-floor":
        check(ver(have[k]) < ver(floor[k]), f"DuckDB {have[k]} is below the floor {floor[k]} (the case under test)"); continue
    check(ver(have[k]) >= ver(floor[k]), f"{k} {have[k]} >= {floor[k]}")
    if mode == "--floor":                                   # the lowest versions allowed are the ones under test
        n = 2 if k == "python" else 3
        check(ver(have[k], n) == ver(floor[k], n), f"{k} {have[k]} is the declared floor {floor[k]}")

print("== a DuckDB below the floor is refused by name at the first SQL use")
# measured: the SQL suite fails on every 1.x release (1.1.0 to 1.5.6: their syntax tree) and passes on 2.0.0
for v, ok in (("1.1.0", False), ("1.4.5", False), ("1.5.6", False), ("2.0.0", True), ("2.1.0.dev42", True), ("3.0.0", True)):
    check(sqlfront.duckdb_supported(v) == ok, f"DuckDB {v} {'accepted' if ok else 'refused'}")
needs = f"needs DuckDB {'.'.join(map(str, sqlfront.DUCKDB_FLOOR))} or later"
try: sqlfront.check_duckdb(types.SimpleNamespace(__version__="1.4.5", __file__="/x/duckdb/__init__.py")); err = None
except ImportError as x: err = str(x)
check(err is not None and needs in err and "DuckDB 1.4.5" in err,
      "an old duckdb module gets an ImportError naming the floor and its version", repr(err))
if mode == "--below-floor":
    for what, call in (('Database().sql("SELECT 1")', lambda: torbles.Database().sql("SELECT 1")),
                       ('F.expr("1 + 1") (the Python API parses through the same call)', lambda: F.expr("1 + 1"))):
        try: call(); err = None
        except Exception as x: err = x
        check(isinstance(err, ImportError) and f"DuckDB {duckdb.__version__} " in str(err) and needs in str(err),
              f"{what} raises that ImportError", f"{type(err).__name__}: {err}")
    t = torbles.Table("n", {"a": np.arange(10, dtype=np.uint32)})
    check(torbles.Query(t).sum(t.a).run() == 45, "the engine API, which needs no parser, still runs")
    print("ALL MATCH" if not fails else f"{fails} FAILURES"); sys.exit(1 if fails else 0)

print("== SQL on a typed load, against DuckDB, on every engine")
con = duckdb.connect()
con.execute("""CREATE TABLE d (id INTEGER PRIMARY KEY, name VARCHAR, region VARCHAR);
INSERT INTO d VALUES (1, 'a', 'east'), (2, 'b', 'west'), (3, 'c', 'east'), (4, NULL, 'west');
CREATE TABLE t (id INTEGER, fk INTEGER REFERENCES d (id), x BIGINT, amt DECIMAL(12, 2), day DATE,
                ts TIMESTAMP WITH TIME ZONE, ok BOOLEAN, s VARCHAR);
INSERT INTO t SELECT i,
    CASE WHEN i % 5 = 4 THEN NULL ELSE i % 4 + 1 END,
    CASE WHEN i % 7 = 3 THEN NULL ELSE (i * 37) % 23 - 11 END,
    CASE WHEN i % 6 = 5 THEN NULL ELSE ((i * 13) % 50 - 20) * 0.25 END,
    CASE WHEN i % 8 = 7 THEN NULL ELSE DATE '2024-01-01' + CAST((i * 11) % 60 AS INTEGER) END,
    CASE WHEN i % 9 = 8 THEN NULL ELSE TIMESTAMPTZ '2024-03-10 12:00:00+01' + to_minutes(i * 90) END,
    CASE WHEN i % 4 = 0 THEN NULL ELSE i % 3 = 0 END,
    CASE WHEN i % 10 = 9 THEN NULL ELSE ['x', 'y', 'it''s', 'Zürich', ''][i % 5 + 1] END
FROM range(40) r(i)""")
try: db = torbles.from_duckdb(con, keep_plain=True); load = None
except Exception as x: db, load = None, f"{type(x).__name__}: {x}"
check(db is not None and sorted(db.tables) == ["d", "t"], "typed load: from_duckdb with the declared keys", load or "")
if db is None: print(f"{fails} FAILURES"); sys.exit(1)


def same(got, want):
    """equal rows, values of the same type; floats to 1e-12"""
    if len(got) != len(want): return False
    for g, w in zip(got, want):
        if len(g) != len(w): return False
        for a, b in zip(g, w):
            if a is None or b is None:
                if a is not b: return False
            elif isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > 1e-12 * max(1.0, abs(b)): return False
            elif type(a) is not type(b) or a != b: return False
    return True


queries = """SELECT 1
SELECT 'it''s', 2.5, NULL, TRUE, DATE '2024-02-29', -7
SELECT SUM(x), COUNT(*), COUNT(x), AVG(x), MIN(x), MAX(x) FROM t
SELECT SUM(amt), MIN(amt), MAX(day), COUNT(ok), COUNT(ts) FROM t
SELECT fk, SUM(x), COUNT(*), SUM(amt) FROM t GROUP BY fk ORDER BY fk NULLS FIRST
SELECT name, SUM(amt), COUNT(x) FROM t JOIN d ON fk = d.id GROUP BY name ORDER BY name NULLS FIRST
SELECT region, SUM(x), COUNT(*) FROM t JOIN d ON t.fk = d.id WHERE day >= DATE '2024-01-20' GROUP BY region ORDER BY region
SELECT name, COUNT(*) FROM t LEFT JOIN d ON fk = d.id GROUP BY name ORDER BY name NULLS FIRST
SELECT id, SUM(x) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t ORDER BY id
SELECT id FROM t WHERE x IS NULL ORDER BY id
SELECT id, COALESCE(x, 7), CAST(amt AS VARCHAR), s, ok FROM t ORDER BY id
WITH q AS (SELECT fk, x FROM t WHERE ok) SELECT fk, SUM(x) FROM q GROUP BY fk ORDER BY fk NULLS FIRST
SELECT s, COUNT(*) FROM t WHERE s LIKE '%''%' OR s = 'Zürich' GROUP BY s ORDER BY s
SELECT COUNT(*) FROM t WHERE ts >= TIMESTAMPTZ '2024-03-11 00:00:00+00'""".split("\n")
utc = dt.timezone.utc
ts_want = [(i, None if i % 9 == 8 else dt.datetime(2024, 3, 10, 11, tzinfo=utc) + dt.timedelta(minutes=90 * i)) for i in range(40)]
tt, dd = db.table("t"), db.table("d")
api = {"Python API: SUM by a dimension (implicit foreign-key join)":
           db.select(dd.name, F.sum(tt.x), F.count()).group_by(dd.name).order_by(dd.name.asc(nulls_first=True)),
       "Python API: a join filtered on the dimension":
           db.select(F.sum(tt.amt), F.count()).from_(tt).join(dd, tt.fk == dd.id).where(dd.region == "east")}
for engine in ("alg", "plain", "gpu"):
    bad = []
    for q in queries:
        want = con.execute(q).fetchall()
        try: got = db.sql(q, engine).rows
        except Exception as x: bad.append(f"{q}\n          {type(x).__name__}: {x}"); continue
        if not same(got, want): bad.append(f"{q}\n          torbles {got[:4]}\n          duckdb  {want[:4]}")
    check(not bad, f"{engine}: {len(queries)} statements equal DuckDB's", "\n        ".join(bad[:6]))
    try: got = db.sql("SELECT id, ts FROM t ORDER BY id", engine).rows
    except Exception as x: got = f"{type(x).__name__}: {x}"
    check(got == ts_want, f"{engine}: TIMESTAMP WITH TIME ZONE values", f"torbles {got if isinstance(got, str) else got[:3]}")
    for what, q in api.items():
        try: ok, got, want = q.verify(engine)
        except Exception as x: check(False, f"{engine}: {what}", f"{type(x).__name__}: {x}\n        {q.to_sql()}"); continue
        check(ok, f"{engine}: {what}", f"torbles {got[:3]}\n        duckdb  {want[:3]}\n        {q.to_sql()}")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
