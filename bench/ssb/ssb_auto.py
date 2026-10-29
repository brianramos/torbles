# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SSB with nothing designed by hand: the official DuckDB database is loaded by torbles.from_duckdb, the keys come from
the standard DDL (schema.sql), and the key layout is found by the automatic key design. The 13 official queries are
written with the values as they appear in the data (strings, years), and every result is compared with DuckDB's
answer to the official SQL. Times: best of 3, end to end.
    python3 ssb_auto.py <SF> [engines, default cpu,gpu; cpu@N for N threads]   (writes torbles_auto.json next to the data)"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import duckdb, torbles
from ssb_common import QUERIES, EXTRA, GROUPS, FACT_COLS, dbfile, datadir

sf = int(sys.argv[1]); engines = (sys.argv[2] if len(sys.argv) > 2 else "cpu,gpu").split(",")
here = os.path.dirname(os.path.abspath(__file__))
t0 = time.perf_counter()
db = torbles.from_duckdb(dbfile(sf), schema=os.path.join(here, "schema.sql"), columns={"lineorder": FACT_COLS})
t_load = time.perf_counter() - t0
print(f"SF{sf}: loaded in {t_load:.1f} s (DuckDB read, types, key design, torbles build), {torbles.mm_kernel()}")
print(db.describe())
lo, d, c, s, p = (db[t] for t in ("lineorder", "date", "customer", "supplier", "part"))
Q = lambda: db.query(lo)
rev = lo.lo_revenue
queries = {
 "Q1.1": Q().join(d).where((d.d_year == 1993) & lo.lo_discount.between(1, 3) & (lo.lo_quantity < 25)).sum(lo.lo_extendedprice * lo.lo_discount),
 "Q1.2": Q().join(d).where((d.d_yearmonthnum == 199401) & lo.lo_discount.between(4, 6) & lo.lo_quantity.between(26, 35)).sum(lo.lo_extendedprice * lo.lo_discount),
 "Q1.3": Q().join(d).where((d.d_weeknuminyear == 6) & (d.d_year == 1994) & lo.lo_discount.between(5, 7) & lo.lo_quantity.between(26, 35)).sum(lo.lo_extendedprice * lo.lo_discount),
 "Q2.1": Q().join(d).join(p).join(s).where((p.p_category == "MFGR#12") & (s.s_region == "AMERICA")).group_by(d.d_year, p.p_brand1).sum(rev),
 "Q2.2": Q().join(d).join(p).join(s).where(p.p_brand1.between("MFGR#2221", "MFGR#2228") & (s.s_region == "ASIA")).group_by(d.d_year, p.p_brand1).sum(rev),
 "Q2.3": Q().join(d).join(p).join(s).where((p.p_brand1 == "MFGR#2221") & (s.s_region == "EUROPE")).group_by(d.d_year, p.p_brand1).sum(rev),
 "Q3.1": Q().join(c).join(s).join(d).where((c.c_region == "ASIA") & (s.s_region == "ASIA") & d.d_year.between(1992, 1997)).group_by(c.c_nation, s.s_nation, d.d_year).sum(rev),
 "Q3.2": Q().join(c).join(s).join(d).where((c.c_nation == "UNITED STATES") & (s.s_nation == "UNITED STATES") & d.d_year.between(1992, 1997)).group_by(c.c_city, s.s_city, d.d_year).sum(rev),
 "Q3.3": Q().join(c).join(s).join(d).where(c.c_city.isin(["UNITED KI1", "UNITED KI5"]) & s.s_city.isin(["UNITED KI1", "UNITED KI5"]) & d.d_year.between(1992, 1997)).group_by(c.c_city, s.s_city, d.d_year).sum(rev),
 "Q3.4": Q().join(c).join(s).join(d).where(c.c_city.isin(["UNITED KI1", "UNITED KI5"]) & s.s_city.isin(["UNITED KI1", "UNITED KI5"]) & (d.d_yearmonth == "Dec1997")).group_by(c.c_city, s.s_city, d.d_year).sum(rev),
 "Q4.1": Q().join(d).join(c).join(s).join(p).where((c.c_region == "AMERICA") & (s.s_region == "AMERICA") & p.p_mfgr.isin(["MFGR#1", "MFGR#2"])).group_by(d.d_year, c.c_nation).sum(rev - lo.lo_supplycost),
 "Q4.2": Q().join(d).join(c).join(s).join(p).where((c.c_region == "AMERICA") & (s.s_region == "AMERICA") & d.d_year.isin([1997, 1998]) & p.p_mfgr.isin(["MFGR#1", "MFGR#2"])).group_by(d.d_year, s.s_nation, p.p_category).sum(rev - lo.lo_supplycost),
 "Q4.3": Q().join(d).join(c).join(s).join(p).where((c.c_region == "AMERICA") & (s.s_nation == "UNITED STATES") & d.d_year.isin([1997, 1998]) & (p.p_category == "MFGR#14")).group_by(d.d_year, s.s_city, p.p_brand1).sum(rev - lo.lo_supplycost),
 "K1": Q().join(c).where(c.c_custkey.between(10000, 15000)).group_by(c.c_region).sum(rev),
 "K3": Q().join(s).join(d).where(s.s_suppkey == 77).group_by(d.d_year).sum(rev),
 "M1": Q().join(d).join(c).group_by(d.d_year, c.c_region).sum(rev),
 "M2": Q().join(c).join(s).group_by(c.c_nation, s.s_region).sum(rev - lo.lo_supplycost),
 "M3": Q().join(d).group_by(d.d_year, lo.lo_discount).sum(lo.lo_extendedprice * lo.lo_discount),
}
con = duckdb.connect(dbfile(sf), read_only=True)
def official(name):
    sql = (QUERIES | EXTRA)[name]; cur = con.execute(sql); cols = [x[0] for x in cur.description]; rows = cur.fetchall()
    g = GROUPS[name]
    if not g: return rows[0][0]
    agg = [i for i, x in enumerate(cols) if x not in g][0]
    return {tuple(r[cols.index(k)] for k in g): r[agg] for r in rows}
wants = {name: official(name) for name in queries}          # DuckDB first, then closed: nothing runs beside the timings
con.close()
out = dict(sf=sf, rows=lo.n, load_s=t_load, kernel=torbles.mm_kernel(), times={}, match={}, plans={},
           design={t: db.meta[t].get("design") for t in db.tables})
for eng in engines:
    torbles.set_threads(int(eng.split("@")[1]) if "@" in eng else (os.cpu_count() or 1))
    if eng == "gpu":
        try: torbles.gpu_name()
        except Exception as e: print("no GPU:", e); continue
    tot, ok_all = 0.0, True
    for name, q in queries.items():
        want = wants[name]
        best = None
        for i in range(3):
            t1 = time.perf_counter(); got = q.run("alg" if eng.startswith("cpu") else eng); dt_ = time.perf_counter() - t1
            best = dt_ if best is None else min(best, dt_)
        ok = got == want; ok_all &= ok
        out["times"][f"{name}/{eng}"] = best; out["match"][f"{name}/{eng}"] = ok
        out["plans"][name] = [list(x) for x in q.range_used]
        if name.startswith("Q"): tot += best
        how = ", ".join(f"{x[0]}:{x[1]}" for x in q.range_used) or "-"
        print(f"  {eng:4s} {name:5s} {best * 1e3:9.2f} ms  {'ok' if ok else 'MISMATCH'}  joins as {how}  ({getattr(q, 'device', '')})")
    print(f"  {eng}: 13 SSB queries {tot * 1e3:.1f} ms in all; {'all results match DuckDB' if ok_all else 'MISMATCHES'}")
    out[f"total_ms/{eng}"] = tot * 1e3
json.dump(out, open(os.path.join(datadir(sf), "torbles_auto.json"), "w"), indent=1, default=str)
