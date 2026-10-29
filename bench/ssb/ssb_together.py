# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Derived surrogates + additive deltas + self-extending records, together, on the official SSB.
  - keys: official keys kept, derived surrogates in the fact (customer/supplier/part: hierarchy << w | key; date:
    year|month|day by arithmetic)
  - deltas: M customers move to another city (and so nation / region). DuckDB UPDATEs the customer table; torbles
    rewrites nothing in the fact: the old surrogates become stale points corrected per query
  - wide values: lo_ordtotalprice * 100 (order totals in 1/100 cent), a BIGINT measure that exceeds 32 bits only for
    the largest orders, stored as a 32-bit base + flag plane + sparse sidecar
Every result is compared with DuckDB after the update; timings compare deltas with a compacted fact (surrogates
rewritten). Usage: python3 ssb_together.py <SF> [moves ...]"""
import sys, os, time, shutil, numpy as np, duckdb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "torbles"))
import algq
from algq import Table, Query, Surrogate, set_threads
from ssb_common import dbfile

sf = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MOVES = [int(x) for x in sys.argv[2:]] or [100, 2000]
ENGINES = [e for e in ("alg", "gpu") if e != "gpu" or os.environ.get("TORBLES_GPU", "1") == "1"]
THREADS = int(os.environ.get("SSB_THREADS", "1,2").split(",")[-1])
src = duckdb.connect(dbfile(sf), read_only=True); src.execute("SET memory_limit = '2GB'")

def strings(v): return np.asarray(v).astype(str)
def cells(levels):
    parent = np.zeros(len(levels[0]), np.int64)
    for lv in levels:
        pairs, inv = np.unique(np.stack([parent, lv], 1), axis=0, return_inverse=True)
        first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]; pos = np.arange(len(pairs))
        rank = pos - np.maximum.accumulate(np.where(first, pos, 0)); wdt = int(rank.max()).bit_length()
        parent = (parent << wdt) | rank[np.asarray(inv).ravel()]
    return parent

# ---------------------------------------------------------------- dimensions (official keys) and their surrogates
HIER = {"customer": ("c_custkey", ["c_region", "c_nation", "c_city"]), "supplier": ("s_suppkey", ["s_region", "s_nation", "s_city"]),
        "part": ("p_partkey", ["p_mfgr", "p_category", "p_brand1"])}
raw, dicts, sg = {}, {}, {}
for name, (key, levels) in HIER.items():
    a = src.execute(f"SELECT {key}, {', '.join(levels)} FROM {name}").fetchnumpy()
    keys = np.asarray(a[key]).astype(np.int64); dicts[name] = {l: np.unique(strings(a[l])) for l in levels}
    codes = {l: np.searchsorted(dicts[name][l], strings(a[l])).astype(np.int64) for l in levels}
    raw[name] = (keys, codes)
def dim_table(name, codes_):
    keys, _ = raw[name]; key = HIER[name][0]; size = int(keys.max()) + 1
    cols = {l: np.zeros(size, np.uint32) for l in list(codes_) + [key]}
    for l, v in codes_.items(): cols[l][keys] = v
    cols[key][keys] = keys
    u = np.zeros(size, bool); u[keys] = True
    t = Table(name, cols, used=u); t.dicts = dicts[name]; return t
def surrogate_of(name, codes_):
    keys, _ = raw[name]; w = int(keys.max()).bit_length()
    s = np.zeros(int(keys.max()) + 1, np.int64); s[keys] = (cells([codes_[l] for l in HIER[name][1]]) << w) | keys
    return w, s
T = {}
for name in HIER:
    T[name] = dim_table(name, raw[name][1]); w, s = surrogate_of(name, raw[name][1]); sg[name] = Surrogate(w, s)
da = src.execute("SELECT d_datekey, d_year FROM date").fetchnumpy(); dk = np.asarray(da["d_datekey"]).astype(np.int64)
def date_s(k): k = np.asarray(k, np.int64); return ((k // 10000 - 1992) << 9) | ((k // 100 % 100) << 5) | (k % 100)
dcols = {"d_year": np.zeros(4096, np.uint32), "d_datekey": np.zeros(4096, np.uint32)}
dcols["d_year"][date_s(dk)] = np.asarray(da["d_year"]); dcols["d_datekey"][date_s(dk)] = dk
du = np.zeros(4096, bool); du[date_s(dk)] = True; T["date"] = Table("date", dcols, used=du)

# ---------------------------------------------------------------- fact
FK = {"customer": "lo_custkey", "supplier": "lo_suppkey", "part": "lo_partkey", "date": "lo_orderdate"}
fact = {}
for c in ["lo_custkey", "lo_suppkey", "lo_partkey", "lo_orderdate", "lo_discount", "lo_revenue", "lo_supplycost", "lo_ordtotalprice"]:
    v = np.asarray(src.execute(f"SELECT {c} FROM lineorder").fetchnumpy()[c]).astype(np.int64)
    if c == "lo_orderdate": v = date_s(v)
    elif c in ("lo_custkey", "lo_suppkey", "lo_partkey"): v = sg[[d for d, f in FK.items() if f == c][0]].s_of_row[v]
    if c == "lo_ordtotalprice": fact["ordtotal_x100"] = (v * 100).astype(np.uint64)
    else: fact[c] = v.astype(np.uint32)
src.close()
official_cust = fact["lo_custkey"] & np.uint32(sg["customer"].mask)

# ---------------------------------------------------------------- the update: customers move to another city
city_info = {}                                                       # city -> (nation, region) codes, from the data
ck = raw["customer"][1]
for ci, ni, ri in zip(ck["c_city"].tolist(), ck["c_nation"].tolist(), ck["c_region"].tolist()): city_info[ci] = (ni, ri)
dcust = dicts["customer"]
def S(name, col, v): return int(np.searchsorted(dicts[name][col], v))

def queries(F, C):
    s, d, p = T["supplier"], T["date"], T["part"]
    J = lambda *ds: (lambda q: [q.join(x, getattr(F, FK[x.name])) for x in ds] and q)(Query(F))
    rev, profit, big = F.lo_revenue, F.lo_revenue - F.lo_supplycost, F.ordtotal_x100
    return {
     "Q3.2 US cities (moved hierarchy as keys)": (
        J(C, s, d).where((C.c_nation == S("customer", "c_nation", "UNITED STATES")) & (s.s_nation == S("supplier", "s_nation", "UNITED STATES"))
                         & d.d_year.between(1992, 1997)).group_by(C.c_city, s.s_city, d.d_year).sum(rev),
        """select c_city, s_city, d_year, sum(lo_revenue) from customer, lineorder, supplier, date where lo_custkey = c_custkey and lo_suppkey = s_suppkey
           and lo_orderdate = d_datekey and c_nation = 'UNITED STATES' and s_nation = 'UNITED STATES' and d_year >= 1992 and d_year <= 1997 group by 1, 2, 3""",
        ["c_city", "s_city", None]),
     "Q4.1 profit by year, customer nation": (
        J(d, C, s, p).where((C.c_region == S("customer", "c_region", "AMERICA")) & (s.s_region == S("supplier", "s_region", "AMERICA"))
                            & p.p_mfgr.isin([S("part", "p_mfgr", "MFGR#1"), S("part", "p_mfgr", "MFGR#2")])).group_by(d.d_year, C.c_nation).sum(profit),
        """select d_year, c_nation, sum(lo_revenue - lo_supplycost) from date, customer, supplier, part, lineorder where lo_custkey = c_custkey
           and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_region = 'AMERICA'
           and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by 1, 2""",
        [None, "c_nation"]),
     "wide measure by nation, region filter": (
        J(C).where(C.c_region == S("customer", "c_region", "ASIA")).group_by(C.c_nation).sum(big),
        """select c_nation, sum(lo_ordtotalprice::BIGINT * 100) from lineorder, customer where lo_custkey = c_custkey and c_region = 'ASIA' group by 1""",
        ["c_nation"]),
     "wide predicate > 2^32, nation filter, by year": (
        J(C, d).where((C.c_nation == S("customer", "c_nation", "CHINA")) & (big > (1 << 32))).group_by(d.d_year).count(),
        """select d_year, count(*) from lineorder, customer, date where lo_custkey = c_custkey and lo_orderdate = d_datekey and c_nation = 'CHINA'
           and lo_ordtotalprice::BIGINT * 100 > 4294967296 group by 1""",
        [None]),
     "region filter, total (filter correction only)": (
        J(C).where(C.c_region == S("customer", "c_region", "EUROPE")).sum(rev),
        """select sum(lo_revenue) from lineorder, customer where lo_custkey = c_custkey and c_region = 'EUROPE'""", []),
     "official key range by region": (
        J(C).where(C.c_custkey.between(1000, 9000)).group_by(C.c_region).sum(big),
        """select c_region, sum(lo_ordtotalprice::BIGINT * 100) from lineorder, customer where lo_custkey = c_custkey and c_custkey between 1000 and 9000 group by 1""",
        ["c_region"]),
    }
def decode(res, spec):
    if not spec: return res
    D = {"c": dicts["customer"], "s": dicts["supplier"], "p": dicts["part"]}
    return {tuple(str(D[c[0]][c][k]) if c else int(k) for c, k in zip(spec, key)): v for key, v in res.items()}
def duck_res(con, sql, spec):
    rows = con.execute(sql).fetchall()
    if not spec: return None if rows[0][0] is None else int(rows[0][0])
    return {tuple(r[:-1]): int(r[-1]) for r in rows}
def best(f, n=3):
    b = 1e9
    for _ in range(n): t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b, r

tmpdb = dbfile(sf) + ".together"
fails = 0
for M in MOVES:
    shutil.copy(dbfile(sf), tmpdb)
    con = duckdb.connect(tmpdb); rng = np.random.default_rng(M)
    keys, codes = raw["customer"]
    mv = rng.choice(len(keys), M, replace=False); new_city = rng.choice(np.array(sorted(city_info)), M)
    codes2 = {l: v.copy() for l, v in codes.items()}
    codes2["c_city"][mv] = new_city
    codes2["c_nation"][mv] = [city_info[c][0] for c in new_city.tolist()]; codes2["c_region"][mv] = [city_info[c][1] for c in new_city.tolist()]
    C = dim_table("customer", codes2)                                     # the dimension after the update
    sgM = Surrogate(sg["customer"].w, sg["customer"].s_of_row)            # surrogates as the fact was written
    _, s_new = surrogate_of("customer", codes2)
    t0 = time.perf_counter(); sgM.move(keys[mv], s_new[keys[mv]]); t_move = time.perf_counter() - t0
    refs = {fk: ((C if d == "customer" else T[d]), (sgM if d == "customer" else sg[d])) if d in sg else T[d] for d, fk in FK.items()}
    F = Table("lineorder", fact, msb=tuple(FK.values()), refs=refs, extend={"ordtotal_x100": 32})
    # compacted: the customer surrogates rewritten (what deltas avoid)
    t0 = time.perf_counter()
    fact_c = dict(fact); fact_c["lo_custkey"] = s_new[official_cust].astype(np.uint32)
    sgC = Surrogate(sg["customer"].w, s_new)
    refs_c = dict(refs); refs_c["lo_custkey"] = (C, sgC)
    Fc = Table("lineorder", fact_c, msb=tuple(FK.values()), refs=refs_c, extend={"ordtotal_x100": 32}); t_compact = time.perf_counter() - t0
    upd = [(int(keys[i]), dcust["c_city"][codes2["c_city"][i]], dcust["c_nation"][codes2["c_nation"][i]], dcust["c_region"][codes2["c_region"][i]]) for i in mv]
    t0 = time.perf_counter()
    con.executemany("UPDATE customer SET c_city = ?, c_nation = ?, c_region = ? WHERE c_custkey = ?", [(c, n, r, k) for k, c, n, r in upd])
    t_duck_update = time.perf_counter() - t0
    npts = len(sgM.stale_arrays()[0]); stale_rows = int(np.isin(F.data["lo_custkey"].astype(np.int64), sgM.stale_arrays()[0]).sum()) if hasattr(F, "data") and "lo_custkey" in F.data else -1
    wide = F.sidecars["ordtotal_x100#hi"]["rows"]
    print(f"\nSF{sf}: {M} customers move: {npts} stale points; recorded in {t_move * 1e3:.1f} ms (fact untouched), "
          f"compaction rebuild {t_compact:.1f} s, DuckDB UPDATE {t_duck_update * 1e3:.0f} ms; wide measure: {wide:,} of {F.n:,} rows extend")
    Qd, Qc = queries(F, C), queries(Fc, C)
    for (name, (q, sql, spec)), (_, (qc, _, _)) in zip(Qd.items(), Qc.items()):
        want = duck_res(con, sql, spec)
        tq, tc, ok, dev = {}, {}, True, {}
        for eng in ENGINES:
            set_threads(THREADS)
            tq[eng], r = best(lambda: q.run(eng)); ok &= decode(r, spec) == want; dev[eng] = q.device
            tc[eng], rc = best(lambda: qc.run(eng)); ok &= decode(rc, spec) == want
        td, _ = best(lambda: duck_res(con, sql, spec))
        how = f"A + B (delta rows {q.delta_rows:,})" if hasattr(q, "delta_rows") else "-"
        fails += 0 if ok else 1
        print(f"  {name:46s} {'match' if ok else 'MISMATCH'}  deltas {tq['alg'] * 1e3:7.1f} ms  compacted {tc['alg'] * 1e3:7.1f} ms  "
              + (f"gpu-path {tq['gpu'] * 1e3:7.1f} ms ({dev['gpu']})  " if "gpu" in tq else "") + f"DuckDB {td * 1e3:7.1f} ms  {how}")
    con.close(); os.remove(tmpdb)
print("\nALL MATCH" if not fails else f"\n{fails} MISMATCHES")
