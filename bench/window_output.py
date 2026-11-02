# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The two window queries of the 2026-11-02 cross-engine benchmark that output every row (its mixed tables, built as
its protocol/build_data.py builds them, loaded by torbles.from_ddl as its adapter loads them): torbles on the CPU and
the GPU, and DuckDB on the same tables, every answer compared in order; one warm-up, then the median of 5 runs, end to
end (torbles: db.sql(...).rows; DuckDB: fetchall()), in ms.
    python3 bench/window_output.py [rows, default 1000000] [threads, default all] [output JSON]"""
import json, os, statistics, sys, time
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, ".."), os.path.join(here, "..", "torbles")]
import duckdb, torbles, sqlfuzz

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
threads = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else (os.cpu_count() or 1)
out_path = sys.argv[3] if len(sys.argv) > 3 else None
con = duckdb.connect(); con.execute(f"SET threads = {threads}")
con.execute("CREATE TABLE d AS SELECT i::INTEGER id, (i % 8)::INTEGER region FROM range(256) r(i)")
con.execute(f"CREATE TABLE t AS SELECT i::BIGINT id, (i % 256)::INTEGER k, CASE WHEN i % 7 = 0 THEN NULL ELSE i % 1000 - 500 END::BIGINT v, "
            f"(i % 31)::INTEGER w, 'name' || (i % 2000)::VARCHAR s FROM range({n}) r(i)")
torbles.set_threads(threads)
t0 = time.perf_counter()
db = torbles.from_ddl("CREATE TABLE d (id INTEGER PRIMARY KEY, region INTEGER); "
                      "CREATE TABLE t (id BIGINT, k INTEGER REFERENCES d(id), v BIGINT, w INTEGER, s VARCHAR);",
                      {"d": con.execute("SELECT * FROM d").df(), "t": con.execute("SELECT * FROM t").df()})
load_s = time.perf_counter() - t0
engines = ["alg"]
try:
    if "emulat" not in torbles.gpu_name().lower(): engines.append("gpu")
except Exception: pass
Q = {"running_window": "SELECT id,sum(w) OVER (PARTITION BY k ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t ORDER BY id",
     "rank_window": "SELECT id,rank() OVER (ORDER BY w) FROM t ORDER BY id"}

def median_ms(f, k=5):
    f()                                                   # warm-up
    ts = []
    for _ in range(k):
        t1 = time.perf_counter(); f(); ts.append(time.perf_counter() - t1)
    return statistics.median(ts) * 1e3

res = dict(rows=n, threads=threads, load_s=load_s, version=torbles.__version__, duckdb_version=duckdb.__version__,
           gpu=torbles.gpu_name() if "gpu" in engines else None, queries={}, all_match=True)
for name, q in Q.items():
    want = con.execute(q).fetchall()
    r = {"duckdb": median_ms(lambda: con.execute(q).fetchall())}
    for e in engines:
        ok = sqlfuzz.same(db.sql(q, e).rows, want, True, 0)
        res["all_match"] = res["all_match"] and ok
        r[f"torbles_{'cpu' if e == 'alg' else 'gpu'}"] = median_ms(lambda: db.sql(q, e).rows)
        r[f"match_{e}"] = ok
    res["queries"][name] = r
    print(f"{name:15s} " + "  ".join(f"{k} {v:8.1f} ms" for k, v in r.items() if not k.startswith("match")), flush=True)
tot = {k: sum(r[k] for r in res["queries"].values()) for k in next(iter(res["queries"].values())) if not k.startswith("match")}
res["total_ms"] = tot
print("both queries: " + "  ".join(f"{k} {v:8.1f} ms" for k, v in tot.items()))
print("all answers equal DuckDB's" if res["all_match"] else "MISMATCH: some answers differ from DuckDB's")
if out_path: json.dump(res, open(out_path, "w"), indent=1)
sys.exit(0 if res["all_match"] else 1)
