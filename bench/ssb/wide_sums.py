# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Sums past 64 bits on SSB data: 64-bit plans, 128-bit plans (the 64 x 64 plane pairs) and the histogram bridge (past
128 bits), each compared with DuckDB's answer (HUGEINT arithmetic in the SQL) and timed against it. Only answers equal
to DuckDB's are timed.
    python3 wide_sums.py <SF> [threads]"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import duckdb, torbles
from ssb_common import dbfile, FACT_COLS

sf = int(sys.argv[1]) if len(sys.argv) > 1 else 1
th = int(sys.argv[2]) if len(sys.argv) > 2 else 1
here = os.path.dirname(os.path.abspath(__file__))
db = torbles.from_duckdb(dbfile(sf), schema=os.path.join(here, "schema.sql"), columns={"lineorder": FACT_COLS})
lo, d = db["lineorder"], db["date"]
Q = lambda: db.query(lo)
H = "CAST({} AS HUGEINT)"
cases = [   # name, torbles query, DuckDB SQL (HUGEINT products: DuckDB's BIGINT products would overflow), what it needs
    ("SUM(revenue)", Q().sum(lo.lo_revenue), "SELECT SUM(lo_revenue) FROM lineorder", "64-bit"),
    ("SUM(extendedprice * discount) BY year", Q().join(d).group_by(d.d_year).sum(lo.lo_extendedprice * lo.lo_discount),
     "SELECT d_year, SUM(lo_extendedprice * lo_discount) FROM lineorder JOIN date ON lo_orderdate = d_datekey GROUP BY d_year", "64-bit"),
    ("SUM(extendedprice * revenue)", Q().sum(lo.lo_extendedprice * lo.lo_revenue),
     f"SELECT SUM({H.format('lo_extendedprice')} * lo_revenue) FROM lineorder", "128-bit"),
    ("SUM(extendedprice * revenue) BY year", Q().join(d).group_by(d.d_year).sum(lo.lo_extendedprice * lo.lo_revenue),
     f"SELECT d_year, SUM({H.format('lo_extendedprice')} * lo_revenue) FROM lineorder JOIN date ON lo_orderdate = d_datekey GROUP BY d_year", "128-bit"),
    ("SUM(extendedprice * revenue * quantity) BY year", Q().join(d).group_by(d.d_year).sum(lo.lo_extendedprice * lo.lo_revenue * lo.lo_quantity),
     f"SELECT d_year, SUM({H.format('lo_extendedprice')} * lo_revenue * lo_quantity) FROM lineorder JOIN date ON lo_orderdate = d_datekey GROUP BY d_year",
     "degree 3: bridge"),
]
con = duckdb.connect(dbfile(sf), read_only=True)
con.execute(f"SET threads = {th}")
torbles.set_threads(th)
def best(f, n=5):
    b = None
    for _ in range(n):
        t = time.perf_counter(); r = f(); dt = time.perf_counter() - t
        b = dt if b is None else min(b, dt)
    return b, r
print(f"SF{sf}, {th} thread(s); ms, best of 5")
for name, q, sql, kind in cases:
    td, rows = best(lambda: con.execute(sql).fetchall())
    want = rows[0][0] if not q.keys else {(r[0],): r[1] for r in rows}
    try:
        tt, got = best(lambda: q.run("alg"))
    except Exception as e:
        print(f"  {name:48s} {kind:16s} torbles: {type(e).__name__}: {e}"); continue
    ok = got == want
    print(f"  {name:48s} {kind:16s} DuckDB {td * 1e3:8.1f}   torbles " + (f"{tt * 1e3:8.1f}" if ok else "  (not timed: MISMATCH)"))
