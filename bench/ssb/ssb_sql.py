# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SSB written in SQL: the official DuckDB database loaded with torbles.from_duckdb (keys from schema.sql, layout by the
automatic key design), the 13 official queries and K1-K3 / M1-M3 run as SQL text by db.sql() and by DuckDB (an in-memory
copy, same threads), every answer compared. Then a few queries that leave the engine (window functions over rows,
GROUP BY on values, string functions, row output) on a generated table: torbles runs that part as array algebra on what
the engine returns. Times: best of 3, end to end.
    python3 ssb_sql.py <SF> [threads, default all] [engines, default alg,gpu] [rows for the second part, default 200000]
SSB_SQL_JSON=<file>: the scaling sweep's form (wsl/run_scaling.sh): only the 13 official queries, each engine's first
run and best of 3 written to <file> as JSON; DuckDB answers each query once from the database file (memory limit
SSB_SQL_DUCK_MEMORY, default 4GB) to check torbles' answers, and is not timed here (ssb_duck.py times it in memory)."""
import sys, os, time, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", ".."), os.path.join(here, "..", "..", "torbles")]
import duckdb, numpy as np, pandas as pd, torbles, sqlfuzz
from ssb_common import QUERIES, EXTRA, FACT_COLS, dbfile

sf = int(sys.argv[1]); threads = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else (os.cpu_count() or 1)
engines = (sys.argv[3] if len(sys.argv) > 3 else "alg,gpu").split(","); nrows = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
src = dbfile(sf); JSON_OUT = os.environ.get("SSB_SQL_JSON")
t0 = time.perf_counter()
db = torbles.from_duckdb(src, schema=os.path.join(here, "schema.sql"), columns={"lineorder": FACT_COLS})
load_s = time.perf_counter() - t0
print(f"SF{sf}: {db['lineorder'].n:,} fact rows, loaded in {load_s:.1f} s; {threads} threads", flush=True)
torbles.set_threads(threads)
if JSON_OUT:                                       # DuckDB checks the answers from its file (ssb_duck.py times it)
    con = duckdb.connect(src, read_only=True); con.execute(f"SET threads = {threads}")
    con.execute(f"SET memory_limit = '{os.environ.get('SSB_SQL_DUCK_MEMORY', '4GB')}'")
else:
    con = duckdb.connect(); con.execute(f"SET threads = {threads}"); con.execute(f"ATTACH '{src}' AS s (READ_ONLY)")
    for t in ("lineorder", "date", "customer", "supplier", "part"): con.execute(f"CREATE TABLE {t} AS SELECT * FROM s.{t}")
    con.execute("DETACH s")
gpu = None
if "gpu" in engines:
    try: gpu = torbles.gpu_name()
    except Exception as e: print("no GPU:", e); engines = [e_ for e_ in engines if e_ != "gpu"]

if JSON_OUT:
    out = dict(sf=sf, rows=int(db["lineorder"].n), threads=threads, load_s=load_s, gpu=gpu, version=torbles.__version__,
               duckdb_version=duckdb.__version__, times={}, all_match=True)
    for name, q in QUERIES.items():
        want = con.execute(q).fetchall()
        ordered = "order by" in q.lower()
        for e in engines:
            tag = f"sql-{e}"
            try:
                t1 = time.perf_counter(); got = db.sql(q, e).rows; first = time.perf_counter() - t1
                ok = sqlfuzz.same(got, want, ordered, 1e-9)
                b = None
                for _ in range(3):
                    t1 = time.perf_counter(); got = db.sql(q, e).rows; dt = time.perf_counter() - t1
                    b = dt if b is None else min(b, dt)
                    ok = ok and sqlfuzz.same(got, want, ordered, 1e-9)
                out["times"][f"{name}/{tag}"] = dict(first=first, best=b, match=bool(ok))
                print(f"SF{sf} {name} {tag}: first {first * 1e3:8.1f} ms  best {b * 1e3:8.1f} ms" + ("" if ok else "  DIFFERENT"), flush=True)
            except Exception as x:
                ok = False; out["times"][f"{name}/{tag}"] = dict(first=None, best=None, match=False, error=f"{type(x).__name__}: {x}")
                print(f"SF{sf} {name} {tag}: failed: {type(x).__name__}: {x}", flush=True)
            out["all_match"] = out["all_match"] and bool(ok)
    for e in engines:
        tot = [out["times"][f"{n}/sql-{e}"]["best"] for n in QUERIES]
        if all(t is not None for t in tot): print(f"SF{sf} 13 SSB queries in SQL, {e}: {sum(tot) * 1e3:.1f} ms", flush=True)
    print("all results equal DuckDB's" if out["all_match"] else "MISMATCH: some results differ from DuckDB's", flush=True)
    json.dump(out, open(JSON_OUT, "w"), indent=1)
    sys.exit(0 if out["all_match"] else 1)

def best(f, k=3):
    b, out = None, None
    for _ in range(k):
        t1 = time.perf_counter(); out = f(); dt = time.perf_counter() - t1
        b = dt if b is None else min(b, dt)
    return b, out

tot = {e: 0.0 for e in engines + ["duckdb"]}
print(f"{'':6s}" + "".join(f"{e:>14s}" for e in ["DuckDB"] + [f"torbles {e}" for e in engines]))
for name, q in (QUERIES | EXTRA).items():
    td, want = best(lambda: con.execute(q).fetchall())
    line = f"{name:6s}{td * 1e3:11.1f} ms"
    if name.startswith("Q"): tot["duckdb"] += td
    for e in engines:
        try:
            tt, got = best(lambda: db.sql(q, e).rows)
            ok = sqlfuzz.same(got, want, "order by" in q.lower(), 1e-9)
            line += f"{tt * 1e3:11.1f} ms" + ("" if ok else " DIFFERENT")
            if name.startswith("Q"): tot[e] += tt
        except Exception as x: line += f"   failed: {type(x).__name__}"
    print(line)
print("13 SSB queries in SQL:" + "".join(f"  {k} {v * 1e3:.1f} ms" for k, v in tot.items()))

print(f"\nqueries that leave the engine, {nrows:,} rows (torbles: array algebra on what the engine returns):")
rng = np.random.default_rng(1)
df = pd.DataFrame(dict(id=np.arange(nrows), k=rng.integers(0, 1000, nrows), v=rng.integers(0, 10000, nrows),
                       s=rng.choice([f"name{i}" for i in range(5000)], nrows)))
gdb = torbles.from_ddl("CREATE TABLE t (id INTEGER, k INTEGER, v INTEGER, s VARCHAR)", {"t": df})
con.register("df", df); con.execute("CREATE TABLE t AS SELECT * FROM df")
qs = {"filter + GROUP BY (on the engine)": "SELECT k, sum(v), count(*) FROM t WHERE v < 5000 AND s LIKE 'name1%' GROUP BY k",
      "a string function in WHERE (distinct values)": "SELECT count(*) FROM t WHERE upper(s) = 'NAME42'",
      "GROUP BY a string expression": "SELECT substring(s FROM 1 FOR 5), count(*) FROM t GROUP BY 1",
      "window: running SUM per partition": "SELECT id, sum(v) OVER (PARTITION BY k ORDER BY id) FROM t",
      "window: rank over all rows": "SELECT id, rank() OVER (ORDER BY v) FROM t",
      "window: sliding AVG, 10 rows": "SELECT id, avg(v) OVER (ORDER BY id ROWS BETWEEN 5 PRECEDING AND 4 FOLLOWING) FROM t",
      "GROUP BY: median and DISTINCT count": "SELECT k, median(v), count(DISTINCT s) FROM t GROUP BY k",
      "100 rows out": "SELECT id, v * 2, s || '!' FROM t WHERE v < 10 ORDER BY id LIMIT 100"}
for name, q in qs.items():
    td, want = best(lambda: con.execute(q).fetchall(), 1)
    tt, got = best(lambda: gdb.sql(q).rows, 1)
    ok = sqlfuzz.same(got, want, "LIMIT" in q, 1e-9)
    print(f"  {name:46s} DuckDB {td:8.3f} s   torbles {tt:8.3f} s   {'same answer' if ok else 'DIFFERENT'}")
