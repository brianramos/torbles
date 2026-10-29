# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Joins through a derived surrogate: the fact stores s = (coordinates << w) | official key, the dimension stays indexed
by its official key = s mod 2^w. DuckDB sees only the official schema (fact fk = official key). Checks every query on
both engines, all join forms, 1/2 threads; plus inconsistent surrogates and missing keys (never join)."""
import sys, numpy as np, pandas as pd, duckdb
from algq import Table, Query, Surrogate, set_threads

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
rng = np.random.default_rng(21)
def dimension(n, name):
    k = np.arange(1, n + 1)                                    # official keys: dense, attributes random per key
    region = rng.integers(0, 5, n); nation = region * 5 + rng.integers(0, 5, n); city = nation * 10 + rng.integers(0, 10, n)
    flag = rng.integers(0, 10, n)
    cols = {c: np.zeros(n + 1, np.uint32) for c in ("region", "nation", "city", "flag", "key")}
    for c, v in (("region", region), ("nation", nation), ("city", city), ("flag", flag), ("key", k)): cols[c][k] = v
    used = np.zeros(n + 1, bool); used[k] = True
    # derived surrogate: region | nation-in-region | city-in-nation, then the official key verbatim
    w = int(n).bit_length()
    coord = (region << 7) | ((nation % 5) << 4) | (city % 10)
    s_of_row = np.zeros(n + 1, np.int64); s_of_row[k] = (coord.astype(np.int64) << w) | k
    return Table(name, cols, used=used), Surrogate(w, s_of_row), k
dA, sgA, kA = dimension(50_000, "dima")
dB, sgB, kB = dimension(2_000, "dimb")
pick_a = kA[rng.integers(0, len(kA), N)]; pick_b = kB[rng.integers(0, len(kB), N)]
fa = sgA.s_of_row[pick_a].astype(np.uint32); fb = pick_b.astype(np.uint32)          # dimb joined on its plain official key
m = rng.integers(0, 1000, N).astype(np.uint32); q_ = rng.integers(1, 50, N).astype(np.uint32)
# 20 bad references: wrong coordinates for an existing key (inconsistent surrogate), or a key that does not exist
bad = rng.choice(N, 20, replace=False)
fa[bad[:10]] ^= np.uint32(1 << (sgA.w + 2))
fa[bad[10:]] = (fa[bad[10:]] & ~np.uint32(sgA.mask)) | np.uint32(len(kA) + 5)
fact = Table("fact", dict(fka=fa, fkb=fb, m=m, q=q_), msb=("fka", "fkb"), refs=dict(fka=(dA, sgA), fkb=dB))
print(f"surrogate width {fact.width('fka')} bits (official key {sgA.w} bits); bad references recorded: {fact.orphans['fka']}")

con = duckdb.connect()
for t, k in ((dA, kA), (dB, kB)):
    con.register("t", pd.DataFrame({"id": k, **{c: t.data[c][k].astype(np.int64) for c in ("region", "nation", "city", "flag", "key")}}))
    con.execute(f"CREATE TABLE {t.name} AS SELECT * FROM t"); con.unregister("t")
off_a = (fa & np.uint32(sgA.mask)).astype(np.int64)
ok_a = (off_a <= len(kA)) & (sgA.s_of_row[np.minimum(off_a, len(kA))] == fa)        # DuckDB sees the bad rows as missing keys
off_a = np.where(ok_a, off_a, 10**9)
con.register("t", pd.DataFrame(dict(fka=off_a, fkb=fb.astype(np.int64), m=m.astype(np.int64), q=q_.astype(np.int64))))
con.execute("CREATE TABLE fact AS SELECT * FROM t")
F, A, B = fact, dA, dB
Q = {
 "coordinate filter, count": Query(F).join(A, F.fka).where(A.region == 2).count(),
 "nation IN, by region": Query(F).join(A, F.fka).where(A.nation.isin([3, 7, 11, 19])).group_by(A.region).sum(F.m),
 "non-coordinate filter, by city": Query(F).join(A, F.fka).where(A.flag < 3).group_by(A.city).sum(F.m * F.q),
 "official key range, by region": Query(F).join(A, F.fka).where(A.key.between(1000, 2500)).group_by(A.region).sum(F.m),
 "key range OR region": Query(F).join(A, F.fka).where((A.key < 500) | (A.region == 4)).count(),
 "two dims, surrogate + plain": Query(F).join(A, F.fka).join(B, F.fkb).where((A.region < 2) & (B.flag == 7)).group_by(A.nation, B.region).sum(F.m),
 "GROUP BY non-coordinate (lookup)": Query(F).join(A, F.fka).where(A.city < 100).group_by(A.flag).sum(F.m),
 "dimension attribute in measure": Query(F).join(A, F.fka).where(A.region == 1).group_by(A.nation).sum(F.m * A.flag),
 "no filter (integrity only)": Query(F).join(A, F.fka).count(),
}
fails = 0
for name, q in Q.items():
    rows = con.execute(q.sql()).fetchall()
    want = {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in rows} if q.keys else (None if rows[0][0] is None else int(rows[0][0]))
    bad_ = []
    for th in (1, 2):
        set_threads(th)
        for mode in ("auto", "factored", "expanded", "lookup"):
            q.join_mode = mode
            for eng in ("alg", "plain"):
                if q.run(eng) != want: bad_.append(f"{eng}/{mode}/{th}T")
    q.join_mode = "auto"; set_threads(1); q.run("alg")
    fails += len(bad_)
    how = ", ".join(f"{d}:{k}x{n} ({sp} vars)" for d, k, n, sp in q.range_used) or ("lookup" if q.last_plan.nsemi else "-")
    print(f"{name:36s} {'ok' if not bad_ else 'FAIL ' + ' '.join(bad_)}   joins as {how}")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
