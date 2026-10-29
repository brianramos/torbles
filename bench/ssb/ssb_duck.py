# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DuckDB on the official SSB: the 13 official queries verbatim, 1 and 2 threads, first run and best of 3.
Saves the results (for cross-checking torbles) and timings. Runs alone in its process.
Usage: python3 ssb_duck.py <SF> [memory limit, default: DuckDB's own (80% of RAM)]
SSB_DUCK_INMEM=1: copy the tables into an in-memory database first (DuckDB's native in-memory storage; for the fact,
the 9 columns the queries read, the same columns torbles loads), so DuckDB,
like torbles, answers from memory from the first query on: no buffer-pool loads from the database file in the timings.
SSB_DUCK_MEMLIMIT: the memory limit (overridden by the argument)."""
import sys, time, json, os, duckdb
from ssb_common import QUERIES, EXTRA, GROUPS, SCHEMA, FACT_COLS, dbfile, datadir

sf = int(sys.argv[1]); mem = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SSB_DUCK_MEMLIMIT")
inmem = bool(os.environ.get("SSB_DUCK_INMEM"))
if inmem:
    con = duckdb.connect(":memory:")
    if mem: con.execute(f"SET memory_limit = '{mem}'")
    t0 = time.perf_counter(); con.execute(f"ATTACH '{dbfile(sf)}' AS f (READ_ONLY)")
    for t in SCHEMA:                                  # the fact: the columns the queries read (as torbles loads)
        cols = ", ".join(FACT_COLS) if t == "lineorder" else "*"
        con.execute(f"CREATE TABLE {t} AS SELECT {cols} FROM f.{t}")
    con.execute("DETACH f"); load_s = time.perf_counter() - t0
    print(f"SF{sf}: tables copied into an in-memory DuckDB database in {load_s:.1f} s", flush=True)
else:
    con = duckdb.connect(dbfile(sf), read_only=True); load_s = None
    if mem: con.execute(f"SET memory_limit = '{mem}'")
def result(name, cur):
    names = [d[0] for d in cur.description]; rows = cur.fetchall(); g = GROUPS[name]
    agg = [n for n in names if n not in g][0]
    if not g: return None if rows[0][0] is None else int(rows[0][0])
    return [[[r[names.index(c)] for c in g], int(r[names.index(agg)])] for r in rows]   # in the query's ORDER BY order
out = {"sf": sf, "engine": f"DuckDB {duckdb.__version__}", "results": {}, "times": {}, "in_memory": inmem, "load_s": load_s,
       "memory_limit": con.execute("SELECT current_setting('memory_limit')").fetchone()[0]}
THREADS = [int(x) for x in os.environ.get('SSB_THREADS', '1,2').split(',')]
for th in THREADS:
    con.execute(f"SET threads = {th}")
    sel = os.environ.get("SSB_QUERIES")                 # optional subset, comma-separated
    for name, sql in {**QUERIES, **EXTRA}.items():
        if sel and name not in sel.split(","): continue
        t = time.perf_counter(); res = result(name, con.execute(sql)); first = time.perf_counter() - t
        best = 1e9
        for _ in range(3):
            t = time.perf_counter(); r2 = result(name, con.execute(sql)); best = min(best, time.perf_counter() - t)
            assert r2 == res
        out["results"][name] = res
        out["times"][f"{name}/{th}T"] = dict(first=first, best=best)
        print(f"SF{sf} {name} {th}T: first {first * 1e3:8.1f} ms  best {best * 1e3:8.1f} ms", flush=True)
json.dump(out, open(os.path.join(datadir(sf), "duckdb_results.json"), "w"), default=str)
