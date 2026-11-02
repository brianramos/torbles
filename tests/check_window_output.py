# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Window functions whose values every row outputs, on a table just past the algebraic paths' threshold (2^16 rows),
with the default settings: the two full-output window queries of the 2026-11-02 cross-engine benchmark and the forms
around them. Window values stay arrays (sqlexec.ColArr) from the window step to the output, rows are reordered by
array positions, an ORDER BY that keeps the rows' order is not applied, and a series window's values come from an
array table; each answer must equal DuckDB's, in order, on the engines alg, plain and gpu
(TORBLES_SQL_ENGINE=<engine>: that engine only).
    python3 tests/check_window_output.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import duckdb, numpy as np, pandas as pd
import torbles, sqlfuzz

ENGINES = [os.environ["TORBLES_SQL_ENGINE"]] if os.environ.get("TORBLES_SQL_ENGINE") else ["alg", "plain", "gpu"]
N = 70_000
con = duckdb.connect()
con.execute("CREATE TABLE d AS SELECT i::INTEGER id, (i % 8)::INTEGER region FROM range(256) r(i)")
con.execute(f"CREATE TABLE t AS SELECT i::BIGINT id, (i % 256)::INTEGER k, CASE WHEN i % 7 = 0 THEN NULL ELSE i % 1000 - 500 END::BIGINT v, "
            f"(i % 31)::INTEGER w, 'name' || (i % 2000)::VARCHAR s, ((i * 7919) % 10007)::INTEGER u FROM range({N}) r(i)")
frames = {"d": con.execute("SELECT * FROM d").df(), "t": con.execute("SELECT * FROM t").df()}
db = torbles.from_ddl("CREATE TABLE d (id INTEGER PRIMARY KEY, region INTEGER); "
                      "CREATE TABLE t (id BIGINT, k INTEGER REFERENCES d(id), v BIGINT, w INTEGER, s VARCHAR, u INTEGER);",
                      frames, keep_plain=True)

RUN = "sum(w) OVER (PARTITION BY k ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
QUERIES = [
    ("the benchmark's running window", f"SELECT id, {RUN} FROM t ORDER BY id"),
    ("the benchmark's rank window", "SELECT id, rank() OVER (ORDER BY w) FROM t ORDER BY id"),
    ("a running sum over a column with NULLs", "SELECT id, sum(v) OVER (PARTITION BY k ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t ORDER BY id"),
    ("rows in another order than the table's", "SELECT id, rank() OVER (ORDER BY w), dense_rank() OVER (ORDER BY u DESC) FROM t ORDER BY u, id DESC"),
    ("a window in an expression", "SELECT id, rank() OVER (ORDER BY w) * 2 + 1 AS r, row_number() OVER (PARTITION BY w ORDER BY id) - 1 FROM t ORDER BY id"),
    ("QUALIFY on a window", "SELECT id, w FROM t QUALIFY rank() OVER (ORDER BY w) <= 3 ORDER BY id"),
    ("ORDER BY a window, OFFSET", "SELECT id, rank() OVER (ORDER BY w) AS r FROM t ORDER BY r DESC, id LIMIT 20 OFFSET 5"),
    ("doubles: PERCENT_RANK and CUME_DIST", "SELECT id, percent_rank() OVER (PARTITION BY k ORDER BY w), cume_dist() OVER (ORDER BY u) FROM t ORDER BY id"),
    ("counts per partition", "SELECT id, count(*) OVER (PARTITION BY k), count(v) OVER (PARTITION BY w) FROM t ORDER BY id"),
    ("DISTINCT over window values", "SELECT DISTINCT w, rank() OVER (ORDER BY w) FROM t ORDER BY 1"),
    ("a window in a CTE, aggregated", f"WITH r AS (SELECT id, {RUN} AS c FROM t) SELECT sum(c), max(c), count(*) FROM r"),
    ("windows on both sides of UNION ALL", "SELECT id, row_number() OVER (ORDER BY id) FROM t WHERE id < 10 UNION ALL "
     "SELECT id, rank() OVER (ORDER BY w) FROM t WHERE id >= 69990 ORDER BY 1, 2"),
    ("a filter, then the order of the rows", f"SELECT id, {RUN}, rank() OVER (ORDER BY w DESC) FROM t WHERE w < 9 AND id % 3 = 1 ORDER BY id"),
    ("a join to the dimension", "SELECT t.id, d.region, rank() OVER (PARTITION BY d.region ORDER BY t.w) FROM t JOIN d ON t.k = d.id ORDER BY t.id"),
    ("LIMIT over rows already in order", "SELECT id, rank() OVER (ORDER BY w) FROM t ORDER BY id LIMIT 7 OFFSET 69990"),
]
fails = 0
for name, sql in QUERIES:
    want = con.execute(sql).fetchall()
    bad = {}
    for e in ENGINES:
        try: got = db.sql(sql, e).rows
        except Exception as x: got = f"{type(x).__name__}: {x}"
        if not (isinstance(got, list) and sqlfuzz.same(got, want, True, 1e-12)):
            bad[e] = str(got[:3] if isinstance(got, list) else got)[:200]
    if bad: fails += 1
    print(f"  {'ok  ' if not bad else 'FAIL'} {name} [{', '.join(ENGINES)}]"
          + ("".join(f"\n        {e}: {g}" for e, g in bad.items()) + f"\n        DuckDB: {str(want[:3])[:200]}\n        {sql}" if bad else ""))
db.close()
print("\nALL MATCH" if fails == 0 else f"\n{fails} FAILURES")
sys.exit(1 if fails else 0)
