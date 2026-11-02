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

# ---------------------------------------------------------------- 4. GROUP BY a dimension attribute, fk stored lowest bit first
# The default join mode read a dimension attribute's coordinate or prefix field from the fk's planes as if the fk were
# stored most significant bit first: on a Table built with refs= and no msb=, the codes were the fk's bits reversed
# (missing groups, or an IndexError past the dimension). The same queries with msb= are the control.
print("4. GROUP BY dimension attributes through a foreign key stored lowest bit first")
f0 = fails; runs = 0
for nd, n, seed in ((64, 3000, 1), (300, 20000, 2), (5000, 200_000, 3)):
    rng = np.random.default_rng(seed)
    dc = dict(a=rng.integers(0, 50, nd).astype(np.uint32), b=rng.integers(0, 7, nd).astype(np.uint32),
              h=(np.arange(nd) // 8).astype(np.uint32))                     # h: a function of the fk's high bits
    e_n = 40; ec = dict(g=rng.integers(0, 5, e_n).astype(np.uint32))
    fc = dict(x=rng.integers(0, 1000, n).astype(np.uint32), y=rng.integers(0, 3, n).astype(np.uint32),
              fk=rng.integers(0, nd, n).astype(np.uint32), ek=rng.integers(0, e_n, n).astype(np.uint32))
    fk, ek, x = fc["fk"].astype(np.int64), fc["ek"].astype(np.int64), fc["x"].astype(np.int64)
    def oracle(keys, sel, agg):
        out = {}
        for i in np.nonzero(sel)[0].tolist():
            k = tuple(int(v[i]) for v in keys)
            if agg == "count": out[k] = out.get(k, 0) + 1
            elif agg == "sum": out[k] = out.get(k, 0) + int(x[i])
            else: lo, hi = out.get(k, (x[i], x[i])); out[k] = (int(min(lo, x[i])), int(max(hi, x[i])))
        return out
    A, B, H, G, Y = dc["a"][fk], dc["b"][fk], dc["h"][fk], ec["g"][ek], fc["y"]
    every = np.ones(n, bool)
    for msb in ((), ("fk", "ek")):
        D = Table("d", dict(dc)); E = Table("e", dict(ec))
        F = Table("f", dict(fc), msb=msb, refs={"fk": D, "ek": E})
        cases = [
            ("GROUP BY d.a, COUNT", lambda q: q.group_by(D.a).count(), [A], every, "count"),
            ("GROUP BY d.h, SUM", lambda q: q.group_by(D.h).sum(F.x), [H], every, "sum"),
            ("d.b = 3, GROUP BY d.a, SUM", lambda q: q.where(D.b == 3).group_by(D.a).sum(F.x), [A], B == 3, "sum"),
            ("d.a < 10, GROUP BY d.b, d.h", lambda q: q.where(D.a < 10).group_by(D.b, D.h).count(), [B, H], A < 10, "count"),
            ("GROUP BY d.a, f.y, MIN / MAX", lambda q: q.group_by(D.a, F.y).minmax(F.x), [A, Y], every, "minmax"),
            ("two dimensions, GROUP BY d.h, e.g", lambda q: q.join(E, F.ek).where(E.g != 2).group_by(D.h, E.g).sum(F.x),
             [H, G], G != 2, "sum"),
        ]
        for name, build_q, keys, sel, agg in cases:
            want = oracle(keys, sel, agg)
            for mode in ("auto", "factored", "expanded", "lookup"):
                for eng in ("alg", "plain", "gpu"):
                    q = build_q(Query(F, join_mode=mode).join(D, F.fk))
                    try: got = q.run(eng)
                    except Exception as ex: got = f"{type(ex).__name__}: {ex}"
                    runs += 1
                    check(got == want, f"{name} [nd {nd}, msb {bool(msb)}, {mode}, {eng}]: {str(got)[:120]} != {str(want)[:120]}")
print(f"   6 queries x 3 dimension sizes x stored lowest / highest bit first x 4 join forms x 3 engines = {runs} runs vs exact counts: {'all match' if fails == f0 else 'FAILURES'}")

# ---------------------------------------------------------------- 5. more looked-up dimension columns than a plan holds
# An engine query holds eight looked-up dimension columns (torble.h); a ninth raised an IndexError from the plan's
# arrays (the SQL fuzzer found it). It is now a ValueError that says so, and SQL runs such a block on its general path
# (tests/check_sql.py replays the fuzzer's statement).
print("5. more looked-up dimension columns than one engine query holds")
f0 = fails
rng = np.random.default_rng(11)
D9 = Table("d9", {f"a{i}": rng.integers(0, 3, 40).astype(np.uint32) for i in range(10)})
F9 = Table("f9", dict(x=rng.integers(0, 9, 900).astype(np.uint32), fk=rng.integers(0, 40, 900).astype(np.uint32)), refs={"fk": D9})
for k in (8, 9):
    q = Query(F9, join_mode="lookup").join(D9, F9.fk).group_by(*[getattr(D9, f"a{i}") for i in range(k)]).count()
    for eng in ("alg", "plain", "gpu"):
        try: got = q.run(eng); err = None
        except Exception as ex: got, err = None, ex
        if k == 8: check(err is None and sum(got.values()) == 900, f"8 looked-up columns on {eng}: {err!r}")
        else: check(isinstance(err, ValueError) and "dimension columns through lookups" in str(err), f"9 looked-up columns on {eng}: {err!r}")
print(f"   eight looked-up columns run, nine are refused by name, on 3 engines: {'all match' if fails == f0 else 'FAILURES'}")

# ---------------------------------------------------------------- 6. row ranges that share a 64-row word
# On a fact sorted by a foreign key, a dimension filter becomes row ranges, rounded out to 64-row words for the
# executors; two ranges can share a word. The CPU executors scanned it once (alg.c, make_chunks); the GPU executor and
# its emulation scanned it once per range and counted its rows again (an independent review found it; the code dated
# from 0.1.0). The ranges are now merged before they are planned, and the GPU executors clamp them as the CPU's do.
print("6. row ranges sharing a 64-row word, on a fact sorted by its foreign key")
f0 = fails; runs = 0
for nd, n, seed in ((1000, 30000, 4), (14000, 70000, 5), (300, 4000, 6)):
    rng = np.random.default_rng(seed)
    dc = dict(a=rng.integers(0, 50, nd).astype(np.uint32), b=rng.integers(0, 7, nd).astype(np.uint32))
    fk = np.sort(rng.integers(0, nd, n)); x = rng.integers(0, 1000, n)
    D = Table("d", dict(dc))
    F = Table("f", dict(x=x.astype(np.uint32), fk=fk.astype(np.uint32)), msb=("fk",), refs={"fk": D}, cluster="fk")
    for name, pred, sel in (("d.b = 2", lambda: D.b == 2, dc["b"][fk] == 2),
                            ("d.a < 5", lambda: D.a < 5, dc["a"][fk] < 5),
                            ("d.a in 7 values", lambda: D.a.isin([1, 3, 5, 7, 9, 11, 13]), np.isin(dc["a"][fk], [1, 3, 5, 7, 9, 11, 13]))):
        want = (int(sel.sum()), int(x[sel].sum()))
        for mode in ("auto", "factored", "expanded", "lookup"):
            for eng in ("alg", "plain", "gpu"):
                got = (Query(F, join_mode=mode).join(D, F.fk).where(pred()).count().run(eng),
                       Query(F, join_mode=mode).join(D, F.fk).where(pred()).sum(F.x).run(eng))
                runs += 1
                check(got == want, f"{name} [nd {nd}, {mode}, {eng}]: {got} != {want}")
# the executors themselves: word ranges that overlap are scanned once (a range restricts the scan, at the executor's
# granularity, to words where the predicate can hold; the predicate selects the rows)
v6 = np.zeros(64 * 8, np.uint32)
for w in (0, 1, 2, 3, 4, 7): v6[64 * w:64 * (w + 1)] = 1
T6 = Table("t6", dict(v=v6))
for eng in ("alg", "plain", "gpu"):
    q = Query(T6).where(T6.v == 1).count()
    q._inpass = True; q._word_ranges = np.array([0, 2, 1, 4, 3, 5, 7, 8], np.uint64)    # as a pass of a partitioned run
    runs += 1; got = q.run(eng)
    check(got == 64 * 6, f"overlapping word ranges on {eng}: {got} rows, want {64 * 6}")
# through SQL: a dimension with only its key and a fact loaded in foreign key order
con = duckdb.connect()
con.execute("CREATE TABLE dim (k INTEGER PRIMARY KEY)")
con.execute("CREATE TABLE fact (id INTEGER PRIMARY KEY, k INTEGER REFERENCES dim(k), v INTEGER)")
nd, per = 14_000, 5
fkk = np.repeat(np.arange(nd), per)
con.register("di", pd.DataFrame(dict(k=np.arange(nd))))
con.register("fa", pd.DataFrame(dict(id=np.arange(len(fkk)), k=fkk, v=np.arange(len(fkk)) % 7)))
con.execute("INSERT INTO dim SELECT * FROM di"); con.execute("INSERT INTO fact SELECT * FROM fa")
import torbles
db6 = torbles.from_duckdb(con, keep_plain=True)
for sql in ("SELECT count(*), sum(v) FROM fact JOIN dim ON fact.k = dim.k WHERE dim.k IN (1, 3, 5, 7, 9, 11, 13)",
            "SELECT count(*), sum(v) FROM fact JOIN dim ON fact.k = dim.k WHERE dim.k % 3 = 0 AND dim.k < 2000",
            "SELECT dim.k % 4 AS m, count(*), sum(v) FROM fact JOIN dim ON fact.k = dim.k WHERE dim.k BETWEEN 10 AND 90 GROUP BY m ORDER BY m"):
    want = con.execute(sql).fetchall()
    for eng in ("alg", "plain", "gpu"):
        runs += 1
        got = db6.sql(sql, eng).rows
        check(got == want, f"{sql} [{eng}]: {got} != {want}")
print(f"   3 filters x 3 dimension sizes x 4 join forms x 3 engines, overlapping word ranges, 3 SQL statements = {runs} runs vs exact answers: {'all match' if fails == f0 else 'FAILURES'}")
print("\nALL FIX TESTS PASSED" if fails == 0 else f"\n{fails} FAILURES")
