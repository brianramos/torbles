# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Automatic key design: the hierarchy is found from the data (functional dependencies), laid out in the derived
surrogate, and the planner uses it (filters as cubes / comparators on the fact's planes, GROUP BY read from them).
Cases: a three-level geography with string values, a calendar, a hierarchy too wide for 32 bits next to the row (the
cell is dense-ranked), one that cannot fit at all (plain row numbers), and attributes that are not part of a chain.
Each design's queries are compared with DuckDB."""
import numpy as np, pandas as pd
import torbles, keydesign
rng = np.random.default_rng(5)
fails = 0
def check(cond, what):
    global fails
    fails += not cond; print(f"  {what:64s} {'ok' if cond else 'FAIL'}")

# 1. geography: region > nation > city, as strings; plus an independent attribute
n = 5000
nat = rng.integers(0, 25, n); city = nat * 10 + rng.integers(0, 10, n)
geo = pd.DataFrame(dict(id=np.arange(n), region=[f"R{x // 5}" for x in nat], nation=[f"N{x:02d}" for x in nat],
                        city=[f"C{x:03d}" for x in city], segment=rng.choice(["a", "b", "c", "d"], n)))
days = pd.date_range("2001-01-01", "2006-12-31")
cal = pd.DataFrame(dict(dk=np.arange(len(days)) * 3 + 17, year=days.year, month=[d.strftime("%Y-%m") for d in days],
                        quarter=[f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in days], dow=days.dayofweek))
m = 200_000
fact = pd.DataFrame(dict(g=rng.integers(0, n, m), dk=rng.choice(cal.dk, m), v=rng.integers(0, 1000, m)))
db = torbles.from_ddl("""CREATE TABLE geo (id INTEGER PRIMARY KEY, region VARCHAR, nation VARCHAR, city VARCHAR, segment VARCHAR);
                         CREATE TABLE cal (dk INTEGER PRIMARY KEY, year INTEGER, month VARCHAR, quarter VARCHAR, dow INTEGER);
                         CREATE TABLE f (g INTEGER REFERENCES geo(id), dk INTEGER REFERENCES cal(dk), v INTEGER);""",
                      {"geo": geo, "cal": cal, "f": fact})
dg, dc = db.meta["geo"]["design"], db.meta["cal"]["design"]
print(db.describe().split("f:")[0])
check(dg["kind"] == "derived" and dg["chain"] == ["region", "nation", "city"], f"geography chain found: {dg['chain']}")
check(set(dg["covered"]) == {"region", "nation", "city"}, "segment (independent) is not in the chain")
check(dc["kind"] == "derived" and dc["chain"][0] == "year" and "month" in dc["chain"], f"calendar chain found: {dc['chain']}")
f, g, c = db["f"], db["geo"], db["cal"]
Q = {"region filter": db.query(f).join(g).where(g.region == "R2").group_by(g.nation).sum(f.v),
     "city IN, by region": db.query(f).join(g).where(g.city.isin(["C101", "C102", "C230"])).group_by(g.region).count(),
     "segment (lookup)": db.query(f).join(g).where(g.segment == "b").group_by(g.region).sum(f.v),
     "year range, by quarter": db.query(f).join(c).where(c.year.between(2002, 2004)).group_by(c.quarter).sum(f.v),
     "month, by region": db.query(f).join(c).join(g).where((c.month == "2005-02") & (g.region == "R4")).group_by(g.city).count()}
for name, q in Q.items():
    ok, a, b = db.verify(q)
    how = ", ".join(f"{x[0]}:{x[1]}" for x in q.range_used) or "lookup"
    check(ok, f"{name} ({how})")
check(Q["region filter"].range_used[0][1] == "cubes", "region filter is a cube on the key's planes")
lp = Q["year range, by quarter"].last_plan
check(all(lp.key_kind[i] in (2, 3) for i in range(lp.nkey)), "GROUP BY quarter is read from the key's planes")

# 2. the design function on its own: dense cell ranks when the fields are too wide, plain when nothing fits
n2 = 1 << 18                                     # row: 19 bits; fields 2 + 9 + 3 = 14 bits (33 > 32); 4500 cells: 13 bits
a = rng.integers(0, 3, n2); b = a * 1000 + rng.integers(0, 300, n2); cc = b * 8 + rng.integers(0, 5, n2)
d2 = keydesign.design(n2, dict(a=a, b=b, c=cc))
check(d2["kind"] == "derived" and d2["dense"] and d2["chain"] == ["a", "b", "c"], f"fields wider than 32 bits with the row: dense-ranked cell ({d2['fields']})")
check(int(d2["s_of_row"].max()) < 1 << 32, "the surrogate fits 32 bits")
d3 = keydesign.design(n2, dict(x=rng.integers(0, 1 << 19, n2)))
check(d3["kind"] in ("plain", "derived") and (d3["kind"] == "plain" or int(d3["s_of_row"].max()) < 1 << 32), f"too wide or no chain: {d3['kind']} ({d3.get('reason') or d3['fields']})")
d4 = keydesign.design(10, {})
check(d4["kind"] == "plain", "no attributes: plain row numbers")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
