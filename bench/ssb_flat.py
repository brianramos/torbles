# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The Star Schema Benchmark flattened into one wide table (0.4.0): lineorder joined with its four dimensions, every
column of each (54 columns: the dimensions' keys are the foreign keys), loaded as one table. torbles finds the columns
that one column determines and stores them once per value, in dimensions of its own, and keeps the rest in
row-aligned parts; the 13 queries, rewritten for the flat table (no joins), run through SQL on the CPU and GPU engines
and on DuckDB over the same flat table. Every answer is compared with DuckDB's.
    python3 bench/ssb_flat.py [SF, default 1] [threads, default all] [out.json]
The SSB DuckDB database comes from bench/ssb (SSB_DATA, as ssb_common.py finds it) and is only read. The flat table is
made in memory, or with SSB_FLAT_DB=<file> in that DuckDB file (created, and removed at the end) for scales whose flat
table does not fit in memory beside torbles; SSB_FLAT_MEMORY caps DuckDB's memory (DuckDB's default otherwise)."""
import json, os, re, sys, time
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(here), os.path.join(os.path.dirname(here), "torbles"), os.path.join(here, "ssb")]
import duckdb
import torbles
from ssb_common import QUERIES, dbfile

sf = int(sys.argv[1]) if len(sys.argv) > 1 else 1
threads = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else (os.cpu_count() or 1)
out_path = sys.argv[3] if len(sys.argv) > 3 else None
torbles.set_threads(threads)

def flat_query(q):
    """an SSB query on the flat table: the join equalities dropped, the FROM clause one table"""
    q = re.sub(r"lo_(custkey|suppkey|partkey|orderdate) = [cspd]_(custkey|suppkey|partkey|datekey) and ", "", q)
    return re.sub(r"\bfrom\s+[a-z, \n]+?\s+where\b", "from flat where", q)

flat_db = os.environ.get("SSB_FLAT_DB")
if flat_db and os.path.exists(flat_db): sys.exit(f"{flat_db} exists: give a new file (it is removed at the end)")
con = duckdb.connect(flat_db or ":memory:")
con.execute(f"SET threads = {threads}")
if os.environ.get("SSB_FLAT_MEMORY"): con.execute(f"SET memory_limit = '{os.environ['SSB_FLAT_MEMORY']}'")
con.execute(f"ATTACH '{dbfile(sf)}' AS ssb (READ_ONLY)")
t0 = time.perf_counter()
con.execute("""CREATE TABLE flat AS SELECT lo.*, c.* EXCLUDE (c_custkey), s.* EXCLUDE (s_suppkey), p.* EXCLUDE (p_partkey),
               d.* EXCLUDE (d_datekey)
               FROM ssb.lineorder lo JOIN ssb.customer c ON lo_custkey = c_custkey JOIN ssb.supplier s ON lo_suppkey = s_suppkey
               JOIN ssb.part p ON lo_partkey = p_partkey JOIN ssb.date d ON lo_orderdate = d_datekey
               ORDER BY lo_orderkey, lo_linenumber""")
con.execute("DETACH ssb")
ncols = len(con.execute("SELECT * FROM flat LIMIT 0").description)
nrows = con.execute("SELECT count(*) FROM flat").fetchone()[0]
flat_s = time.perf_counter() - t0
print(f"SF{sf}: flat table of {nrows:,} rows and {ncols} columns made in DuckDB in {flat_s:.1f} s", flush=True)

t0 = time.perf_counter()
db = torbles.Database(con); db.add_table("flat"); db.build()
load_s = time.perf_counter() - t0
t = db["flat"]
print(f"loaded into torbles in {load_s:.1f} s: {len(t.cols)} physical columns"
      + (f" in {len(t.parts)} parts" if t.parts else " in one torble")
      + f", {len(t.moved)} columns moved to {len(t.groups)} dimensions of their own", flush=True)
for x in db.notes: print("  ", x)

def best(fn, k=3):
    fn(); times = []
    for _ in range(k):
        t1 = time.perf_counter(); r = fn(); times.append((time.perf_counter() - t1) * 1e3)
    return min(times), r

def same(a, b):
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            if isinstance(u, float) or isinstance(v, float):
                if abs(float(u) - float(v)) > 1e-9 * max(1.0, abs(float(v))): return False
            elif u != v: return False
    return True

res = dict(sf=sf, threads=threads, rows=nrows, columns=ncols, flat_s=flat_s, load_s=load_s, gpu=torbles.gpu_name(),
           version=torbles.__version__, physical=len(t.cols), parts=[len(p.cols) for p in t.parts] if t.parts else None,
           groups={d: dict(table=g, columns=[c for c, x in t.moved.items() if x == d], rows=db[g].n) for d, g in t.groups.items()},
           describe=db.describe(), queries={})
tot = {"duckdb": 0.0, "alg": 0.0, "gpu": 0.0}; bad = 0
print(f"{'query':6s} {'DuckDB':>9s} {'SQL CPU':>9s} {'SQL GPU':>9s}  (ms, best of 3; answers equal DuckDB's)")
for name, q in QUERIES.items():
    fq = flat_query(q)
    dms, want = best(lambda: con.execute(fq).fetchall())
    row = dict(sql=fq, duckdb_ms=dms)
    for e in ("alg", "gpu"):
        ms, got = best(lambda: db.sql(fq, e).rows)
        ok = same([tuple(r) for r in got], [tuple(r) for r in want])
        row[f"{e}_ms"], row[f"{e}_ok"] = ms, ok
        bad += not ok; tot[e] += ms
    tot["duckdb"] += dms
    res["queries"][name] = row
    print(f"{name:6s} {dms:9.1f} {row['alg_ms']:9.1f} {row['gpu_ms']:9.1f}" + ("" if row["alg_ok"] and row["gpu_ok"] else "  DIFFERS"), flush=True)
print(f"{'13 SSB':6s} {tot['duckdb']:9.1f} {tot['alg']:9.1f} {tot['gpu']:9.1f}")
res["totals_ms"] = tot; res["mismatches"] = bad
print("ALL MATCH" if not bad else f"{bad} ANSWERS DIFFER")
if out_path: json.dump(res, open(out_path, "w"), indent=1, default=str)
db.close(); con.close()
if flat_db:
    for f in (flat_db, flat_db + ".wal"):
        if os.path.exists(f): os.remove(f)
