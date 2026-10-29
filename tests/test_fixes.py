# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Regression tests for (1) full-width uint32 values and (2) orphaned foreign keys.
Every result is checked against numpy and DuckDB, on both engines, all join modes, 1 and 2 threads."""
import ctypes, random, numpy as np, duckdb, pandas as pd
from algq import Table, Query, set_threads, lib
from star import generate, build, load_duckdb, queries

fails = 0
def check(cond, what):
    global fails
    if not cond: fails += 1; print("  FAIL:", what)

# ---------------------------------------------------------------- 1. full-width uint32
print("1. full-width uint32 values")
rng = np.random.default_rng(1); n = 200_000
cols = dict(a=rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32),          # uniform over all 32 bits
            b=(np.uint32(2**31) + rng.integers(0, 1000, n)).astype(np.uint32),         # bit 31 always set
            c=rng.integers(0, 16, n, dtype=np.uint32))
cols["a"][:4] = [0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0]
t = Table("t", cols)
print(f"   widths: a={t.width('a')} b={t.width('b')} c={t.width('c')} bits")
for k in cols:
    out = np.empty(n, np.uint32); lib.at_decode_column(t.at, t.idx(k), out.ctypes.data)
    check(np.array_equal(out, cols[k]), f"decode of {k} lost information")
check(t.validate()[0] == 0, "clean table reports corruption")
try:
    Table("bad", dict(x=np.array([1, 2**32])))
    check(False, "values above uint32 accepted silently")
except ValueError: pass
con = duckdb.connect(); con.register("df", pd.DataFrame({k: v.astype(np.int64) for k, v in cols.items()}))
con.execute("CREATE TABLE t AS SELECT * FROM df")
A, B, C = (Query(t).sum(t.a), Query(t).where(t.a >= 2**31).count(), Query(t).where(t.a.between(2**31 - 5, 2**32 - 1) & ~(t.b == 2**31 + 7)).sum(t.a),
           )
tests = [(Query(t).sum(t.a), "SELECT SUM(a) FROM t"),
         (Query(t).where(t.a >= 2**31).count(), "SELECT COUNT(*) FROM t WHERE a >= 2147483648"),
         (Query(t).where(t.a.between(2**31 - 5, 2**32 - 1) & ~(t.b == 2**31 + 7)).sum(t.a),
          "SELECT SUM(a) FROM t WHERE a BETWEEN 2147483643 AND 4294967295 AND NOT (b = 2147483655)"),
         (Query(t).where(t.a == 0xFFFFFFFF).count(), "SELECT COUNT(*) FROM t WHERE a = 4294967295"),
         (Query(t).sum(t.a * t.c), "SELECT SUM(a * c) FROM t"),
         (Query(t).sum(t.b - t.a + 3 * t.c), "SELECT SUM(b - a + 3 * c) FROM t"),
         (Query(t).where(t.b > 2**31 + 500).group_by(t.c).sum(t.a), "SELECT c, SUM(a) FROM t WHERE b > 2147484148 GROUP BY c")]
for q, sql in tests:
    rows = con.execute(sql).fetchall()
    want = {(int(r[0]),): int(r[1]) for r in rows} if q.keys else (None if rows[0][0] is None else int(rows[0][0]))
    for th in (1, 2):
        set_threads(th)
        for eng in ("alg", "plain"):
            check(q.run(eng) == want, f"{sql} [{eng}, {th}T] = {q.run(eng)} != {want}")
set_threads(1)
print(f"   decode, predicates, linear/bilinear sums and GROUP BY on full-range values: {'all match DuckDB' if not fails else 'FAILED'}")
# faults in the top bit planes
f0 = fails
top = [p for p in range(t.nplanes())]
r = random.Random(4); rows = r.sample(range(n), 500)
for row in rows: t.flip(r.randrange(t.nplanes()), row)
nb = t.validate()[0]; fixed = t.repair()
for k in cols:
    out = np.empty(n, np.uint32); lib.at_decode_column(t.at, t.idx(k), out.ctypes.data)
    check(np.array_equal(out, cols[k]), f"repair did not restore {k}")
check(nb == 500 and fixed == 500, f"faults: detected {nb}, repaired {fixed} of 500")
print(f"   500 random bit faults (any plane, incl. bit 31): detected {nb}, repaired {fixed}, values restored: {fails == f0}")

# ---------------------------------------------------------------- 2. orphaned foreign keys
print("2. orphaned foreign keys")
raw, used = generate(300_000)
lo = raw["lineorder"]; r = random.Random(7)
injected = {}
for fk, dim in (("cust", "customer"), ("supp", "supplier"), ("part", "part"), ("date", "dates")):
    unused_ids = np.nonzero(~used[dim])[0]
    rows = r.sample(range(len(lo[fk])), 10)
    for k, row in enumerate(rows):          # half inside the id space (unused ids), half beyond its end
        lo[fk][row] = int(unused_ids[r.randrange(len(unused_ids))]) if k < 5 else len(used[dim]) + r.randrange(1000)
    injected[fk] = rows
T, _ = build((raw, used)); con = duckdb.connect(); load_duckdb((raw, used), con)
L = T["lineorder"]
print(f"   orphans recorded at load: {L.orphans}")
for fk in injected: check(L.orphans[fk] == 10, f"{fk}: {L.orphans[fk]} orphans recorded, expected 10")
c, d, s, p = T["customer"], T["dates"], T["supplier"], T["part"]
extra = {"join without dimension filter, COUNT": Query(L).join(c, L.cust).count(),
         "4 joins, no filters, SUM": Query(L).join(c, L.cust).join(s, L.supp).join(p, L.part).join(d, L.date).sum(L.price),
         "join, group by dimension attribute": Query(L).join(c, L.cust).group_by(c.region).count()}
qs = dict(queries(T)); qs.update(extra)
nq = 0
for name, q in qs.items():
    rows = con.execute(q.sql()).fetchall()
    want = {tuple(int(v) for v in r_[:-1]): int(r_[-1]) for r_ in rows} if q.keys else (None if rows[0][0] is None else int(rows[0][0]))
    for th in (1, 2):
        set_threads(th)
        for mode in ("auto", "factored", "expanded", "lookup"):
            q.join_mode = mode
            for eng in ("alg", "plain"):
                got = q.run(eng); nq += 1
                check(got == want, f"{name} [{eng}, {mode}, {th}T]")
    q.join_mode = "auto"
set_threads(1)
total = int(con.execute("SELECT COUNT(*) FROM lineorder").fetchone()[0])
joined = Query(L).join(c, L.cust).count().run("alg")
print(f"   {len(qs)} queries x 4 join forms x 2 engines x 1/2 threads = {nq} runs vs DuckDB inner joins: {'all match' if fails == 0 else 'FAILURES'}")
print(f"   e.g. lineorder has {total:,} rows; joined to customer: {joined:,} (the 10 orphaned customer references are excluded)")
# ---------------------------------------------------------------- 3. filters that match nothing
print("3. filters matching zero rows")
T, _ = build(generate(300_000)); con = duckdb.connect(); load_duckdb(generate(300_000), con)
L, c, d, s, p = T["lineorder"], T["customer"], T["dates"], T["supplier"], T["part"]
rev = L.price * (100 - L.discount)
empty = {
    "date matches nothing (sorted-order path), SUM": Query(L).join(d, L.date).where(d.year == 99).sum(rev),
    "date matches nothing, COUNT": Query(L).join(d, L.date).where(d.yearmonth == 5000).count(),
    "date matches nothing, GROUP BY date (mask split)": Query(L).join(d, L.date).where(d.weeknum == 60).group_by(L.date).sum(rev),
    "date matches nothing, bilinear SUM": Query(L).join(d, L.date).where(d.year > 6).sum(L.price * L.discount),
    "customer matches nothing, GROUP BY": Query(L).join(c, L.cust).join(d, L.date).where(c.region == 7).group_by(d.year).sum(rev),
    "two dims, one empty": Query(L).join(p, L.part).join(s, L.supp).where((p.category == 12) & (s.city == 9999)).group_by(p.brand).sum(rev),
    "fact filter matches nothing": Query(L).where(L.quantity > 50).sum(L.price),
    "contradictory fact filter, COUNT": Query(L).where((L.discount < 3) & (L.discount > 5)).count(),
}
f0 = fails; runs = 0
for name, q in empty.items():
    rows = con.execute(q.sql()).fetchall()
    want = {tuple(int(v) for v in r_[:-1]): int(r_[-1]) for r_ in rows} if q.keys else (None if rows[0][0] is None else int(rows[0][0]))
    for th in (1, 2):
        set_threads(th)
        for mode in ("auto", "factored", "expanded", "lookup"):
            q.join_mode = mode
            for eng in ("alg", "plain"):
                try: got = q.run(eng)
                except Exception as e: got = f"{type(e).__name__}: {e}"
                runs += 1; check(got == want, f"{name} [{eng}, {mode}, {th}T]: {got!r} != {want!r}")
    q.join_mode = "auto"
set_threads(1)
print(f"   {len(empty)} empty-result queries x 4 join forms x 2 engines x 1/2 threads = {runs} runs vs DuckDB: {'all match' if fails == f0 else 'FAILURES'}")
print("\nALL FIX TESTS PASSED" if fails == 0 else f"\n{fails} FAILURES")
