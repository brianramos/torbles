# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Range / band / inequality joins: correctness vs DuckDB across both evaluation paths (fact grouped by x, and
dominance slices on x's planes), both engines, all equi-join forms and 1/2 threads, incl. empty and unbounded cases."""
import sys, numpy as np, pandas as pd, duckdb
from star import generate, build, load_duckdb
from algq import Table, Query, set_threads

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
gen = generate(N); raw, used = gen; T, _ = build(gen); con = duckdb.connect(); load_duckdb(gen, con)
lo, d, c, s = T["lineorder"], T["dates"], T["customer"], T["supplier"]
days = np.nonzero(used["dates"])[0]; rng = np.random.default_rng(9)
def dimtable(name, cols):
    t = Table(name, {k: np.asarray(v, np.uint32) for k, v in cols.items()})
    con.register("t", pd.DataFrame({"id": np.arange(t.n), **{k: np.asarray(v, np.int64) for k, v in cols.items()}}))
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM t"); con.unregister("t"); return t
i0 = rng.integers(0, len(days), 300)
pr = dimtable("pr", dict(a=days[i0], b=days[np.minimum(i0 + rng.integers(0, 90, 300), len(days) - 1)], k=rng.integers(0, 5, 300)))
pmax = int(raw["lineorder"]["price"].max())
qb = dimtable("qb", dict(a=rng.integers(0, pmax, 25), w=rng.integers(0, 20000, 25), k=rng.integers(0, 3, 25)))
qb_hi = qb.a + qb.w
rev = lo.price * (100 - lo.discount)
Q = {
 "interval, by kind": Query(lo).join_range(pr, lo.date, pr.a, pr.b).group_by(pr.k).sum(rev),
 "interval, fact key + dim key": Query(lo).join(s, lo.supp).join_range(pr, lo.date, pr.a, pr.b).group_by(s.region, pr.k).sum(lo.quantity),
 "interval, dim filter, count": Query(lo).join_range(pr, lo.date, pr.a, pr.b).where(pr.k >= 3).count(),
 "band, fact filter": Query(lo).join_range(pr, lo.date, pr.a - 10, pr.a + 10).where(lo.discount < 4).group_by(pr.k).sum(lo.price),
 "x < y with equi-join filter": Query(lo).join(c, lo.cust).where(c.region == 1).join_ineq(pr, lo.date, "<", pr.a).group_by(pr.k).sum(rev),
 "x >= y": Query(lo).join_ineq(pr, lo.date, ">=", pr.b).group_by(pr.k).count(),
 "x <= y, total": Query(lo).join_ineq(pr, lo.date, "<=", pr.a).sum(lo.price * lo.discount),
 "x > y with group by date year": Query(lo).join(d, lo.date).join_ineq(pr, lo.date, ">", pr.b).group_by(d.year, pr.k).count(),
 "price overlapping ranges (slices)": Query(lo).join_range(qb, lo.price, qb.a, qb_hi).group_by(qb.k).sum(rev),
 "price ranges, filters (slices)": Query(lo).join(c, lo.cust).where((c.region == 2) & (lo.quantity > 10)).join_range(qb, lo.price, qb.a, qb_hi).group_by(qb.k).count(),
 "price < y (slices)": Query(lo).join_ineq(qb, lo.price, "<", qb.a).sum(lo.quantity),
 "price ranges, fact key (groups)": Query(lo).join(c, lo.cust).join_range(qb, lo.price, qb.a, qb_hi).group_by(c.region, qb.k).sum(lo.quantity),
 "empty: no dim rows": Query(lo).join_range(pr, lo.date, pr.a, pr.b).where(pr.k > 10).group_by(pr.k).sum(rev),
 "empty: no fact rows, total": Query(lo).join_range(pr, lo.date, pr.a, pr.b).where(lo.quantity > 99).sum(rev),
 "empty: bounds outside data": Query(lo).join_range(qb, lo.price, qb.a + 10_000_000, qb.a + 20_000_000).count(),
}
fails = 0
for name, q in Q.items():
    rows = con.execute(q.sql()).fetchall()
    want = {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in rows} if q.keys else (None if rows[0][0] is None else int(rows[0][0]))
    pre = con.execute(q.sql_preagg()).fetchall()
    want_pre = {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in pre} if q.keys else (None if pre[0][0] is None else int(pre[0][0]))
    bad = [] if want_pre == want or (not q.keys and q.measure is None and want_pre in (None, 0) and want == 0) else ["preagg-sql"]
    for th in (1, 2):
        set_threads(th)
        for mode in ("auto", "factored", "expanded", "lookup"):
            q.join_mode = mode
            for path in ("groups", "slices"):
                q.range_mode = path
                for eng in ("alg", "plain"):
                    got = q.run(eng)
                    if got != want: bad.append(f"{eng}/{mode}/{path}/{th}T")
    q.join_mode, q.range_mode = "auto", "auto"; set_threads(1); q.run("alg")
    fails += len(bad)
    print(f"{name:36s} {'ok' if not bad else 'FAIL ' + ' '.join(bad)}  ({q.range_path})")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
