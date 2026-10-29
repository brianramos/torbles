# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Typed columns through the loader: a star schema declared in standard SQL DDL with strings (quotes, Unicode),
decimals, signed and 64-bit integers, dates, timestamps and booleans; orphaned references. Every query on every engine
(algebraic CPU, GPU executor or its emulation, plain columnar baseline) and join form is compared with DuckDB."""
import sys, datetime as dt, numpy as np, pandas as pd
import torbles, algq
rng = np.random.default_rng(7)
nc, nf = 3000, int(sys.argv[1]) if len(sys.argv) > 1 else 120_000
regions = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
nation = rng.integers(0, 25, nc)
cust = pd.DataFrame(dict(c_key=np.arange(10, 10 + 2 * nc, 2), c_name=[f"Customer#{i:06d}" for i in range(nc)],
                         c_region=[regions[x // 5] for x in nation],
                         c_nation=[f"N{x:02d}" + ("'s" if x == 7 else "é" if x == 9 else "") for x in nation],
                         c_city=[f"N{x:02d}-C{c}" for x, c in zip(nation, rng.integers(0, 10, nc))], c_segment=rng.choice(["A", "B", "C"], nc),
                         c_balance=np.round(rng.normal(0, 5000, nc), 2)))
days = pd.date_range("1992-01-01", "1998-12-31")
date = pd.DataFrame(dict(d_key=[int(d.strftime("%Y%m%d")) for d in days], d_date=days.date, d_year=days.year, d_month=days.month,
                         d_yearmonth=[d.strftime("%b%Y") for d in days], d_week=days.isocalendar().week.values.astype(int)))
fact = pd.DataFrame(dict(f_cust=rng.choice(cust.c_key, nf), f_date=rng.choice(date.d_key, nf), f_qty=rng.integers(-5, 50, nf),
                         f_price=np.round(rng.uniform(0, 1e5, nf), 2), f_disc=rng.integers(0, 11, nf), f_big=rng.integers(-2**62, 2**62, nf),
                         f_ts=pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 10**12, nf), unit="us"), f_flag=rng.random(nf) < 0.3))
fact.loc[:50, "f_cust"] = 999999                    # orphans: never join
ddl = """
CREATE TABLE customer (c_key INTEGER PRIMARY KEY, c_name VARCHAR, c_region VARCHAR, c_nation VARCHAR, c_city VARCHAR,
                       c_segment VARCHAR, c_balance DECIMAL(12,2));
CREATE TABLE date (d_key INTEGER PRIMARY KEY, d_date DATE, d_year INTEGER, d_month INTEGER, d_yearmonth VARCHAR, d_week INTEGER);
CREATE TABLE sales (f_cust INTEGER REFERENCES customer(c_key), f_date INTEGER REFERENCES date(d_key), f_qty INTEGER,
                    f_price DECIMAL(15,2), f_disc INTEGER, f_big BIGINT, f_ts TIMESTAMP, f_flag BOOLEAN);
"""
db = torbles.from_ddl(ddl, {"customer": cust, "date": date, "sales": fact}, keep_plain=True)
s, c, d = db["sales"], db["customer"], db["date"]
Q = {
 "region filter, nation groups": db.query(s).join(c).where(c.c_region == "ASIA").group_by(c.c_nation).sum(s.f_price * (100 - s.f_disc)),
 "string prefix + date range": db.query(s).join(c).join(d).where(c.c_city.startswith("N07") & d.d_date.between(dt.date(1994, 3, 1), "1995-02-28")).group_by(d.d_year, c.c_segment).count(),
 "quote and accent in strings": db.query(s).join(c).where(c.c_nation.isin(["N07's", "N09é"])).group_by(c.c_nation).count(),
 "signed values, decimal sum": db.query(s).where((s.f_qty < 0) | (s.f_qty >= 45)).sum(s.f_price),
 "decimal constants in the measure": db.query(s).join(d).where(d.d_year == 1996).group_by(d.d_month).sum(s.f_price * 0.075 - 1),
 "64-bit filter, boolean groups": db.query(s).where(s.f_big > 2**61).group_by(s.f_flag).count(),
 "64-bit groups": db.query(s).where(s.f_qty == 3).group_by(s.f_big).count(),
 "timestamp filter": db.query(s).where(s.f_ts < dt.datetime(2020, 1, 5, 12)).group_by(s.f_flag).sum(s.f_qty),
 "value not in the data": db.query(s).join(c).where(c.c_region == "ATLANTIS").count(),
 "not-equal, IN": db.query(s).join(c).where((c.c_segment != "B") & c.c_nation.isin(["N01", "N02", "N24"])).group_by(c.c_region).sum(s.f_qty),
 "dimension decimal filter": db.query(s).join(c).where(c.c_balance > 1000.5).group_by(c.c_segment).count(),
 "primary key range": db.query(s).join(c).where(c.c_key.between(100, 2000)).group_by(c.c_region).count(),
 "week of year": db.query(s).join(d).where((d.d_week == 6) & (d.d_year == 1994)).sum(s.f_price * s.f_disc),
 "group by dates": db.query(s).join(d).where(d.d_yearmonth == "Feb1995").group_by(d.d_date).count(),
 "no row passes": db.query(s).where(s.f_qty > 1000).sum(s.f_price),
}
refused = {"sum that can overflow 64 bits": lambda: db.query(s).sum(s.f_big).run(),
           "sum over a string column": lambda: db.query(s).join(c).sum(c.c_name).run(),
           "filter on a foreign key itself": lambda: db.query(s).where(s.f_cust == 10).count().run()}
fails = 0
for name, q in Q.items():
    want = db.sql_result(q); bad = []
    for eng in ("alg", "gpu", "plain"):
        for mode in ("auto", "factored", "expanded", "lookup"):
            q.join_mode = mode
            for th in (1, 2):
                algq.set_threads(th)
                if q.run(eng) != want: bad.append(f"{eng}/{mode}/{th}T")
    q.join_mode = "auto"; q.run("alg")
    fails += bool(bad)
    print(f"  {name:36s} {'ok' if not bad else 'FAIL ' + ' '.join(bad[:6])}   joins as {', '.join(f'{x[0]}:{x[1]}' for x in q.range_used) or '-'}")
for name, f in refused.items():
    try: f(); print(f"  refused: {name:28s} NOT REFUSED"); fails += 1
    except (OverflowError, TypeError) as e: print(f"  refused: {name:28s} ok ({type(e).__name__}: {str(e)[:70]})")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
