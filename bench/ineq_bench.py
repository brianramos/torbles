# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Range, band and inequality joins with aggregation: torbles (comparator polynomial over the bit-plane engine's
per-coordinate aggregates) vs DuckDB on the SQL as written and on the same SQL with the aggregation pushed below the
join (the algebra handed to DuckDB). All results are cross-checked. Usage: python3 ineq_bench.py <fact rows> [reps]"""
import sys, time, json, copy, numpy as np, pandas as pd, duckdb
from star import generate, build, load_duckdb
from algq import Table, Query, set_threads

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
gen = generate(N); raw, used = gen; T, _ = build(gen)
duck = duckdb.connect(); load_duckdb(gen, duck)
lo, d, c = T["lineorder"], T["dates"], T["customer"]
days = np.nonzero(used["dates"])[0]                     # date ids in chronological order (year * 512 + day of year)
rng = np.random.default_rng(3)
def dimtable(name, cols):
    t = Table(name, {k: np.asarray(v, np.uint32) for k, v in cols.items()})
    df = pd.DataFrame({"id": np.arange(t.n), **{k: np.asarray(v, np.int64) for k, v in cols.items()}})
    duck.register("t", df); duck.execute(f"CREATE TABLE {name} AS SELECT * FROM t"); duck.unregister("t")
    return t
i0 = rng.integers(0, len(days), 2000); dur = rng.integers(0, 60, 2000)
promo = dimtable("promo", dict(dfrom=days[i0], dto=days[np.minimum(i0 + dur, len(days) - 1)], kind=rng.integers(0, 8, 2000)))
ev = dimtable("events", dict(eday=days[rng.integers(3, len(days) - 3, 500)], kind=rng.integers(0, 5, 500)))
ms = dimtable("milestones", dict(mday=days[rng.integers(0, len(days), 40)], kind=rng.integers(0, 4, 40)))
pmax = int(raw["lineorder"]["price"].max())
cuts = np.sort(rng.choice(np.arange(2, pmax), 9, replace=False))
tiers = dimtable("tiers", dict(lo=np.r_[0, cuts], hi=np.r_[cuts - 1, pmax], tier=np.arange(10)))
rev = lo.price * (100 - lo.discount)
Q = {
 "I1 promo interval join, by kind":            Query(lo).join_range(promo, lo.date, promo.dfrom, promo.dto).group_by(promo.kind).sum(rev),
 "I2 + customer join, by region x kind":       Query(lo).join(c, lo.cust).join_range(promo, lo.date, promo.dfrom, promo.dto).group_by(c.region, promo.kind).sum(rev),
 "I3 band join date in [day-3, day+3], count": Query(lo).join_range(ev, lo.date, ev.eday - 3, ev.eday + 3).group_by(ev.kind).count(),
 "I4 inequality date < start, region 3":       Query(lo).join(c, lo.cust).where(c.region == 3).join_ineq(ms, lo.date, "<", ms.mday).group_by(ms.kind).sum(rev),
 "I5 price tiers (range lookup), by tier":     Query(lo).join_range(tiers, lo.price, tiers.lo, tiers.hi).group_by(tiers.tier).sum(lo.quantity),
 "I6 filtered promos + fact filter, total":    Query(lo).join_range(promo, lo.date, promo.dfrom, promo.dto).where((promo.kind < 3) & (lo.discount >= 5)).sum(rev),
}
def best(fn, reps):
    b, out = 1e9, None
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); b = min(b, time.perf_counter() - t)
    return b, out
def conv(q, rows):
    if q.keys: return {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in rows}
    return None if rows[0][0] is None else int(rows[0][0])
recs, allok = [], True
print(f"{N:,} fact rows; promo 2000 intervals, events 500, milestones 40, tiers 10")
print(f"{'query':44s}{'torbles 1T':>11s}{'torbles 2T':>11s}{'DuckDB 2T':>11s}{'Duck pre 2T':>12s}  join pairs")
for name, q in Q.items():
    tm, res = {}, {}
    for th in (1, 2):
        set_threads(th); tm[f"torble{th}"], res[f"torble{th}"] = best(lambda: q.run("alg"), REPS)
    set_threads(1); res["plain"] = q.run("plain")
    duck.execute("SET threads = 2")
    sql, pre = q.sql(), q.sql_preagg()
    tm["duckdb2"], res["duckdb2"] = best(lambda: conv(q, duck.execute(sql).fetchall()), REPS if N <= 3_000_000 else 1)
    tm["duckpre2"], res["duckpre2"] = best(lambda: conv(q, duck.execute(pre).fetchall()), REPS)
    qc = copy.copy(q); qc.keys, qc.measure = [], None; pairs = qc.run("alg")      # join cardinality, by the same algebra
    if N <= 1_000_000: res["pairs"] = pairs; res["pairs_duck"] = int(duck.execute("SELECT COUNT(*) FROM " + sql.split(" FROM ", 1)[1].split(" GROUP BY")[0]).fetchone()[0])
    pc = (res.pop("pairs", None), res.pop("pairs_duck", None))
    ok = len({repr(sorted(v.items())) if isinstance(v, dict) else repr(v) for v in res.values()}) == 1 and pc[0] == pc[1]
    allok &= ok
    recs.append(dict(query=name, match=ok, pairs=pairs, **tm))
    print(f"{name:44s}" + "".join(f"{tm[k] * 1e3:9.1f}ms" for k in ("torble1", "torble2", "duckdb2")) + f"{tm['duckpre2'] * 1e3:10.1f}ms"
          + f"  {pairs:,}" + ("" if ok else "  MISMATCH"), flush=True)
json.dump(dict(rows=N, results=recs), open(f"ineq_{N}.json", "w"))
print("all engines agree" if allok else "MISMATCHES")
