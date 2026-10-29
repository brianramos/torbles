# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Sums and GROUP BY codes past 64 bits. 128-bit sums (the 64 x 64 plane pairs of two 64-bit values, folded with
128-bit coefficients: matrix product, mask splitting, row kernel, general path, plain baseline), 128-bit group codes
(GROUP BY two 64-bit columns), and the histogram bridge (past 128 bits; dimension columns wider than 32 bits). Every
answer is compared with DuckDB's, computed in HUGEINT (DuckDB's own BIGINT products would overflow), on every engine,
join form and thread count; then the same on plain uint32 tables against Python's integers."""
import sys, numpy as np, pandas as pd
import torbles, algq
from algq import Table, Query

rng = np.random.default_rng(11)
nd, nf = 400, int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
dim = pd.DataFrame(dict(d_key=np.arange(nd), d_grp=rng.integers(0, 7, nd),
                        d_wide=rng.integers(-2**62, 2**62, nd),                  # spans 63 bits: rank-coded in a dimension
                        d_price=np.round(rng.uniform(-1e6, 1e6, nd), 2)))
fact = pd.DataFrame(dict(f_d=rng.integers(0, nd, nf), f_g=rng.integers(0, 5, nf), f_qty=rng.integers(-50, 50, nf),
                         f_big=rng.integers(-2**62, 2**62, nf), f_big2=rng.integers(0, 2**44, nf, dtype=np.int64),
                         f_few=rng.choice(np.array([-2**62, -5, 0, 7, 2**61 + 3, 2**62], np.int64), nf),
                         f_amt=np.round(rng.uniform(-1e9, 1e9, nf), 2),
                         f_u=rng.choice(np.array([0, 1, 2**63, 2**64 - 1], np.uint64), nf)))
ddl = """
CREATE TABLE dim (d_key INTEGER PRIMARY KEY, d_grp INTEGER, d_wide BIGINT, d_price DECIMAL(10, 2));
CREATE TABLE fact (f_d INTEGER REFERENCES dim(d_key), f_g INTEGER, f_qty INTEGER, f_big BIGINT, f_big2 BIGINT, f_few BIGINT,
                   f_amt DECIMAL(12, 2), f_u UBIGINT);
"""
db = torbles.from_ddl(ddl, {"dim": dim, "fact": fact}, keep_plain=True)
f, d = db["fact"], db["dim"]
con = db.con
H = lambda c: f"CAST({c} AS HUGEINT)"
J = "FROM fact JOIN dim ON f_d = d_key"
cases = {   # name: (query, DuckDB SQL)
    "SUM(big)":                       (db.query(f).sum(f.f_big), "SELECT SUM(f_big) FROM fact"),
    "SUM(big) BY g":                  (db.query(f).group_by(f.f_g).sum(f.f_big), "SELECT f_g, SUM(f_big) FROM fact GROUP BY f_g"),
    "SUM(big * qty) BY g":            (db.query(f).group_by(f.f_g).sum(f.f_big * f.f_qty), f"SELECT f_g, SUM({H('f_big')} * f_qty) FROM fact GROUP BY f_g"),
    "SUM(big * big2) BY g (64 x 64)": (db.query(f).group_by(f.f_g).sum(f.f_big * f.f_big2 - f.f_big2 * 3),
                                       f"SELECT f_g, SUM({H('f_big')} * f_big2 - {H('f_big2')} * 3) FROM fact GROUP BY f_g"),
    "SUM(big * amt) BY dim group":    (db.query(f).join(d).group_by(d.d_grp).sum(f.f_big * f.f_amt),
                                       f"SELECT d_grp, SUM(CAST(f_big AS DECIMAL(38, 0)) * f_amt) {J} GROUP BY d_grp"),
    "SUM(big - 2 big2), filter":      (db.query(f).where(f.f_qty > 10).sum(f.f_big - 2 * f.f_big2),
                                       f"SELECT SUM({H('f_big')} - 2 * {H('f_big2')}) FROM fact WHERE f_qty > 10"),
    "GROUP BY big, few (128-bit codes)": (db.query(f).where(f.f_qty < -40).group_by(f.f_few, f.f_big).sum(f.f_qty),
                                          "SELECT f_few, f_big, SUM(f_qty) FROM fact WHERE f_qty < -40 GROUP BY f_few, f_big"),
    "GROUP BY ubigint: 2^64 codes":   (db.query(f).group_by(f.f_u).count(), "SELECT f_u, COUNT(*) FROM fact GROUP BY f_u"),
    "GROUP BY big2, big, g COUNT":    (db.query(f).where(f.f_qty == 7).group_by(f.f_big2, f.f_big, f.f_g).count(),
                                       "SELECT f_big2, f_big, f_g, COUNT(*) FROM fact WHERE f_qty = 7 GROUP BY f_big2, f_big, f_g"),
    "bridge: SUM(dim wide * qty)":    (db.query(f).join(d).group_by(f.f_g).sum(d.d_wide * f.f_qty),
                                       f"SELECT f_g, SUM({H('d_wide')} * f_qty) {J} GROUP BY f_g"),
    "bridge: SUM(big * big * few)":   (db.query(f).where(f.f_qty < 0).sum(f.f_big * f.f_big * f.f_few),
                                       f"SELECT SUM(CAST(f_big AS DOUBLE)) FROM fact WHERE false"),      # exact: Python below
}
# the past-128-bit sum, exactly
want_big3 = sum(int(b) * int(b) * int(w) for b, w, q in zip(fact.f_big, fact.f_few, fact.f_qty) if q < 0)
fails = 0
for name, (q, sql) in cases.items():
    if name.startswith("bridge: SUM(big * big * few)"): want = want_big3
    else:
        cur = con.execute(sql); rows = cur.fetchall(); nk = len(q.keys)
        want = (rows[0][0] if rows else None) if not nk else {tuple(r[:nk]): r[nk] for r in rows}
    bad = []
    for eng in ("alg", "gpu", "plain"):
        for mode in ("auto", "lookup"):
            q.join_mode = mode
            for th in (1, 2):
                algq.set_threads(th)
                try: got = q.run(eng)
                except Exception as e: got = f"{type(e).__name__}: {e}"
                if got != want: bad.append(f"{eng}/{mode}/{th}T")
    q.join_mode = "auto"; algq.set_threads(1)
    fails += bool(bad)
    print(f"  {name:36s} {'ok' if not bad else 'FAIL ' + ' '.join(bad[:6])}")

# plain uint32 tables: products of two 32-bit columns over many rows pass 2^64
n = 300_000
a = rng.integers(2**31, 2**32, n, dtype=np.uint64).astype(np.uint32); b = rng.integers(2**31, 2**32, n, dtype=np.uint64).astype(np.uint32)
g = rng.integers(0, 9, n).astype(np.uint32)
T = Table("t", dict(a=a, b=b, g=g))
want_all = sum(int(x) * int(y) for x, y in zip(a.tolist(), b.tolist()))
want_g = {}
for x, y, k in zip(a.tolist(), b.tolist(), g.tolist()): want_g[(k,)] = want_g.get((k,), 0) + x * y - x
bad = []
for eng in ("alg", "gpu", "plain"):
    for th in (1, 2):
        algq.set_threads(th)
        if Query(T).sum(T.a * T.b).run(eng) != want_all: bad.append(f"sum {eng}/{th}T")
        if Query(T).group_by(T.g).sum(T.a * T.b - T.a).run(eng) != want_g:
            bad.append(f"group {eng}/{th}T")
algq.set_threads(1)
fails += bool(bad)
print(f"  {'uint32 tables: SUM(a * b) past 2^64':36s} {'ok' if not bad else 'FAIL ' + ' '.join(bad)}")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
