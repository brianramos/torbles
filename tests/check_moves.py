# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Dimension changes as additive deltas. Customers move (region / nation / city change); DuckDB simply UPDATEs the
dimension. torbles rewrites nothing in the fact: the surrogates written before the move become stale points, and every
query that filters or groups on the moved dimension is split additively (a delta plane marks the fact rows of moved keys;
pass A = all rows but the delta plane on stored surrogates, pass B = the delta rows with current values).
Some fact rows are written after the move (they carry the new surrogate), so both versions coexist.
Compared with DuckDB on every engine, join form and thread count, for several numbers of moved keys, and against
a compacted fact (surrogates rewritten)."""
import sys, numpy as np, pandas as pd, duckdb
import algq
from algq import Table, Query, Surrogate, set_threads

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
ENGINES = ["alg", "plain"] + (["gpu"] if "nogpu" not in sys.argv else [])
fails = 0
for MOVES in (10, 200, 3000):
    rng = np.random.default_rng(MOVES)
    n = 50_000; k = np.arange(1, n + 1); w = int(n).bit_length()
    def attrs(region, natr, cityr): return region, region * 5 + natr, (region * 5 + natr) * 10 + cityr
    reg, nat, cit = attrs(rng.integers(0, 5, n), rng.integers(0, 5, n), rng.integers(0, 10, n))
    flag = rng.integers(0, 10, n)
    def surrogate_of(reg, nat, cit, keys): return ((((reg << 3) | (nat % 5)) << 4 | (cit % 10)).astype(np.int64) << w) | keys
    s_of_row = np.zeros(n + 1, np.int64); s_of_row[k] = surrogate_of(reg, nat, cit, k)
    sg = Surrogate(w, s_of_row)
    # the move: MOVES customers get a new region/nation/city
    mv = rng.choice(n, MOVES, replace=False)
    reg2, nat2, cit2 = reg.copy(), nat.copy(), cit.copy()
    reg2[mv], nat2[mv], cit2[mv] = attrs(rng.integers(0, 5, MOVES), rng.integers(0, 5, MOVES), rng.integers(0, 10, MOVES))
    sg.move(k[mv], surrogate_of(reg2[mv], nat2[mv], cit2[mv], k[mv]))
    # facts: rows of moved customers carry the old surrogate (written before) or the new one (written after)
    pick = rng.integers(0, n, N); fk = sg.s_of_row[k[pick]].copy()
    old_of = {r: v[0] for r, v in sg.stale.items()}
    before = rng.random(N) < 0.7
    for i in np.nonzero(before & np.isin(k[pick], list(old_of)))[0]: fk[i] = old_of[int(k[pick[i]])]
    m = rng.integers(0, 1000, N).astype(np.uint32)
    def dimension(rg, nt, ct):
        cols = {c: np.zeros(n + 1, np.uint32) for c in ("region", "nation", "city", "flag", "key")}
        for c, v in (("region", rg), ("nation", nt), ("city", ct), ("flag", flag), ("key", k)): cols[c][k] = v
        u = np.zeros(n + 1, bool); u[k] = True; return Table("dima", cols, used=u)
    dA = dimension(reg2, nat2, cit2)                                   # the dimension after the update
    F = Table("fact", dict(fka=fk.astype(np.uint32), m=m), msb=("fka",), refs=dict(fka=(dA, sg)))
    stale_rows = int(np.isin(fk, sg.stale_arrays()[0]).sum())
    # compacted: the same facts with every surrogate rewritten to the current derivation
    sgc = Surrogate(w, sg.s_of_row)
    Fc = Table("fact", dict(fka=sg.s_of_row[(fk & sg.mask)].astype(np.uint32), m=m), msb=("fka",), refs=dict(fka=(dA, sgc)))
    con = duckdb.connect()
    con.register("t", pd.DataFrame({"id": k, "region": reg, "nation": nat, "city": cit, "flag": flag, "key": k}))
    con.execute("CREATE TABLE dima AS SELECT * FROM t"); con.unregister("t")
    con.register("t", pd.DataFrame({"id": k[mv], "region": reg2[mv], "nation": nat2[mv], "city": cit2[mv]}))
    con.execute("UPDATE dima SET region = t.region, nation = t.nation, city = t.city FROM t WHERE dima.id = t.id"); con.unregister("t")
    con.register("t", pd.DataFrame(dict(fka=(fk & sg.mask).astype(np.int64), m=m.astype(np.int64))))
    con.execute("CREATE TABLE fact AS SELECT * FROM t"); con.unregister("t")
    print(f"{MOVES} moved customers: {len(sg.stale_arrays()[0])} stale points, {stale_rows:,} fact rows with a stale surrogate")
    Q = lambda T: {
        "region filter, count": Query(T).join(dA, T.fka).where(dA.region == 2).count(),
        "nation IN, sum": Query(T).join(dA, T.fka).where(dA.nation.isin([3, 7, 11, 19])).sum(T.m),
        "city range, by region": Query(T).join(dA, T.fka).where(dA.city.between(40, 120)).group_by(dA.region).sum(T.m),
        "no filter, by nation": Query(T).join(dA, T.fka).group_by(dA.nation).sum(T.m),
        "region filter, by city": Query(T).join(dA, T.fka).where(dA.region < 2).group_by(dA.city).count(),
        "key range, by region": Query(T).join(dA, T.fka).where(dA.key.between(1000, 9000)).group_by(dA.region).sum(T.m),
        "non-coordinate filter, by region": Query(T).join(dA, T.fka).where(dA.flag == 3).group_by(dA.region).sum(T.m),
    }
    for (name, q), (_, qc) in zip(Q(F).items(), Q(Fc).items()):
        rows = con.execute(q.sql()).fetchall()
        want = {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in rows} if q.keys else (None if rows[0][0] is None else int(rows[0][0]))
        bad = []
        for th in (1, 2):
            set_threads(th)
            for mode in ("auto", "factored", "expanded", "lookup"):
                q.join_mode = qc.join_mode = mode
                for eng in ENGINES:
                    if q.run(eng) != want: bad.append(f"{eng}/{mode}/{th}T")
                if qc.run("alg") != want: bad.append(f"compacted/{mode}/{th}T")
        q.join_mode = "auto"; set_threads(1); q.flips_used = 0; q.run("alg")
        how = f"A + B (delta rows {q.delta_rows:,})" if hasattr(q, "delta_rows") else "-"
        fails += len(bad)
        print(f"   {name:34s} {'ok' if not bad else 'FAIL ' + ' '.join(bad[:6])}   {how}")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
