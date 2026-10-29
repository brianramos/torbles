# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""torbles on the official SSB. Data are read from the DuckDB database built from the official dbgen files (same rows,
same row order), strings are dictionary-encoded order-preserving (so =, IN and BETWEEN on strings keep their meaning),
and the 13 official queries are expressed in algq. Every result is compared with DuckDB's, row by row and in ORDER BY
order. Two key designs:
  official     the keys as generated: dense surrogate keys, dates as yyyymmdd integers
  coordinate   keys re-encoded at load as coordinates of the dimension hierarchy (an order-preserving bijection):
               customer/supplier region|nation|city|seq, part mfgr|category|brand|seq, date year|month|day;
               the fact's foreign keys are rewritten through the same map
  derived      the official keys kept, with an algebraically derived surrogate stored in the fact:
               customer/supplier/part: s = (region|nation|city or mfgr|category|brand) << w | official key, so the
               official key is the low w bits (a projection) and the dimensions stay indexed by it;
               date: s = year|month|day computed from yyyymmdd by arithmetic, official key = 10000 y + 100 m + d
Also runs K1-K3 (not SSB): predicates on the official keys themselves.
Usage: python3 ssb_torbles.py <SF> official|coordinate|derived"""
import sys, os, time, json, gc, resource, numpy as np, duckdb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "torbles"))
import algq
from algq import Table, Query, Surrogate, set_threads, gpu_name
from ssb_common import QUERIES, EXTRA, GROUPS, FACT_COLS, dbfile, datadir

sf, design = int(sys.argv[1]), sys.argv[2]
con = duckdb.connect(dbfile(sf), read_only=True)
DIMS = {"customer": ("c_custkey", ["c_region", "c_nation", "c_city"]),
        "supplier": ("s_suppkey", ["s_region", "s_nation", "s_city"]),
        "part": ("p_partkey", ["p_mfgr", "p_category", "p_brand1"]),
        "date": ("d_datekey", ["d_year", "d_yearmonthnum", "d_yearmonth", "d_weeknuminyear", "d_monthnuminyear", "d_daynuminmonth"])}
HIER = {"customer": ["c_region", "c_nation", "c_city"], "supplier": ["s_region", "s_nation", "s_city"],
        "part": ["p_mfgr", "p_category", "p_brand1"], "date": ["d_year", "d_monthnuminyear", "d_daynuminmonth"]}
FK = {"customer": "lo_custkey", "supplier": "lo_suppkey", "part": "lo_partkey", "date": "lo_orderdate"}

def rss(tag):
    if os.environ.get("SSB_MEM"):
        cur = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        print(f"  [mem] {tag}: now {cur:.2f} GiB, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB", flush=True)
def _level_rank(parent, lv):
    """rank of each row's value within its parent cell (order-preserving), vectorized: (rank per row, width)"""
    key = (parent << 32) | lv.astype(np.int64)                   # parent fields stay below 31 bits
    u, inv = np.unique(key, return_inverse=True)
    first = np.r_[True, (u[1:] >> 32) != (u[:-1] >> 32)]; pos = np.arange(len(u))
    rank = pos - np.maximum.accumulate(np.where(first, pos, 0))
    return rank[inv.ravel()], int(rank.max()).bit_length()
def coordinate_cells(levels):
    """the hierarchy as concatenated codes, each level ranked within its parent (order-preserving): (cell, widths)"""
    n = len(levels[0]); parent = np.zeros(n, np.int64); widths = []
    for lv in levels:
        r_row, w = _level_rank(parent, lv); widths.append(w); parent = (parent << w) | r_row
    return parent, widths
def date_surrogate(k):
    """yyyymmdd -> year|month|day (3|4|5 bits), by arithmetic; inverse: 10000 (1992 + y) + 100 m + d"""
    k = np.asarray(k, np.int64); return ((k // 10000 - 1992) << 9) | ((k // 100 % 100) << 5) | (k % 100)
def date_official(s):
    s = np.asarray(s, np.int64); return 10000 * (1992 + (s >> 9)) + 100 * ((s >> 5) & 15) + (s & 31)
def coordinate_ids(levels):
    """ids = concatenated codes: each level ranked within its parent (order-preserving), then a sequence number
    inside the finest cell. Returns (ids, field widths)."""
    parent, widths = coordinate_cells(levels); n = len(parent)
    order = np.argsort(parent, kind="stable"); ps = parent[order]
    first = np.r_[True, ps[1:] != ps[:-1]]; pos = np.arange(n)
    seq = np.empty(n, np.int64); seq[order] = pos - np.maximum.accumulate(np.where(first, pos, 0))
    ws = int(seq.max()).bit_length(); widths.append(ws)
    return (parent << ws) | seq, widths

t_load = time.perf_counter()
T, remap, info, surr = {}, {}, {}, {}
for name, (key, attrs) in DIMS.items():
    if design != "coordinate": attrs = [a for a in attrs if a not in ("d_monthnuminyear", "d_daynuminmonth")]
    # strings are dictionary-encoded inside DuckDB (native, parallel): code = dense rank in byte order, the
    # order-preserving dictionary is the sorted distinct values
    types = dict(con.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{name}'").fetchall())
    strs = [a for a in attrs if types.get(a) == "VARCHAR"]
    sel = [key] + [f"dense_rank() OVER (ORDER BY {a}) - 1 AS {a}" if a in strs else a for a in attrs]
    arr = con.execute(f"SELECT {', '.join(sel)} FROM {name}").fetchnumpy()
    keys = np.asarray(arr[key]).astype(np.int64)
    dicts, codes = {}, {key: keys}; attrs = attrs + [key]           # the official key is also an attribute
    for a in attrs:
        if a == key: continue
        codes[a] = np.asarray(arr[a]).astype(np.int64)
        if a in strs: dicts[a] = np.array([r[0] for r in con.execute(f"SELECT DISTINCT {a} FROM {name} ORDER BY {a}").fetchall()])
    if design == "coordinate":
        ids, widths = coordinate_ids([codes[a] for a in HIER[name]])
        m = np.full(int(keys.max()) + 1, -1, np.int64); m[keys] = ids; remap[name] = m
        info[name] = dict(fields=dict(zip(HIER[name] + ["seq"], widths)), bits=int(ids.max()).bit_length(), official_bits=int(keys.max()).bit_length())
    elif design == "derived" and name == "date":
        ids = date_surrogate(keys); assert np.array_equal(date_official(ids), keys)
        info[name] = dict(derivation="year|month|day by arithmetic", bits=int(ids.max()).bit_length(), official_bits=int(keys.max()).bit_length())
    elif design == "derived":
        cell, widths = coordinate_cells([codes[a] for a in HIER[name]]); w = int(keys.max()).bit_length()
        fields = dict(zip(HIER[name] + ["official key"], widths + [w]))
        if sum(widths) + w > 32 or os.environ.get("SSB_DENSE_CELLS"):   # (the variable forces it, for testing)
            # one bit field per level no longer fits next to the official key in 32 bits (part from SF50 on): the
            # hierarchy cell is stored as its dense rank instead. Order-preserving, so every hierarchy prefix is still
            # one interval of surrogates (comparators on the high planes) and the attributes are prefixes of the id.
            u, cell = np.unique(cell, return_inverse=True); cell = cell.astype(np.int64).ravel()
            fields = {"hierarchy cell (dense rank)": max(1, int(len(u) - 1).bit_length()), "official key": w}
        s_of_row = np.zeros(int(keys.max()) + 1, np.int64); s_of_row[keys] = (cell << w) | keys
        if int(s_of_row.max()) >> 32: raise SystemExit(f"{name}: the derived surrogate needs {int(s_of_row.max()).bit_length()} bits")
        surr[name] = Surrogate(w, s_of_row); ids = keys
        info[name] = dict(fields=fields, bits=int(s_of_row.max()).bit_length(), official_bits=w)
    else: ids = keys
    size = int(ids.max()) + 1
    cols = {}
    for a in attrs:
        col = np.zeros(size, np.uint32); col[ids] = codes[a]; cols[a] = col
    u = np.zeros(size, bool); u[ids] = True
    t = Table(name, cols, used=u); t.dicts = dicts; T[name] = t
    del arr, cols
t_dims = time.perf_counter() - t_load
rss("dimensions loaded")
fact = {}
con.execute("SET memory_limit = '1GB'")                # bounded cache while the columns are read
n = con.execute("SELECT count(*) FROM lineorder").fetchone()[0]
t_read = t_enc = 0.0
for cname in FACT_COLS:
    t1 = time.perf_counter()
    col = np.asarray(con.execute(f"SELECT {cname} FROM lineorder").fetchnumpy()[cname])
    col = col.astype(np.uint32, copy=False) if col.dtype != np.uint32 else col
    t2 = time.perf_counter(); t_read += t2 - t1
    for dname, fk in FK.items():                       # foreign keys rewritten natively, on all cores
        if fk == cname and design == "coordinate": col = algq.par_map(col, remap[dname])
        if fk == cname and design == "derived":
            col = algq.par_date_surrogate(col) if dname == "date" else algq.par_map(col, surr[dname].s_of_row)
    t_enc += time.perf_counter() - t2
    fact[cname] = col; rss(f"fetched {cname}")
con.close(); del remap; gc.collect()
t_fetch = time.perf_counter() - t_load
rss("before build"); t0 = time.perf_counter()
lo = T["lineorder"] = Table("lineorder", fact, msb=tuple(FK.values()),
                            refs={fk: ((T[d], surr[d]) if d in surr else T[d]) for d, fk in FK.items()})
del fact; lo.drop_plain(); gc.collect()                 # the bit planes are the data from here on
t_build = time.perf_counter() - t0; rss("after build")
print(f"ingestion ({algq.lib.at_get_build_threads()} build threads): dimensions {t_dims:.2f} s, fact columns read from DuckDB {t_read:.2f} s, "
      f"keys re-encoded {t_enc:.2f} s, torble build {t_build:.2f} s", flush=True)
d, c, s, p = T["date"], T["customer"], T["supplier"], T["part"]
if design == "derived":
    # the official keys are still there: recover every row's official key from the stored planes and compare
    con = duckdb.connect(dbfile(sf), read_only=True); con.execute("SET memory_limit = '1GB'"); same = {}
    for dname, fk in FK.items():
        got = lo.decoded(fk).astype(np.int64)
        got = date_official(got) if dname == "date" else got & surr[dname].mask
        want = np.asarray(con.execute(f"SELECT {fk} FROM lineorder").fetchnumpy()[fk]).astype(np.int64)
        same[fk] = bool(np.array_equal(got, want)); del got, want; lo._decoded.pop(fk, None); gc.collect()
    con.close()
    print("official keys recovered from the stored surrogates (projection / arithmetic), all rows:", same)
    info["official_keys_recovered"] = same

def S(t, col, v): return int(np.searchsorted(t.dicts[col], v))            # string constant -> dictionary code
def SB(t, col, a, b): return int(np.searchsorted(t.dicts[col], a, "left")), int(np.searchsorted(t.dicts[col], b, "right")) - 1
rev, profit, disc_rev = lo.lo_revenue, lo.lo_revenue - lo.lo_supplycost, lo.lo_extendedprice * lo.lo_discount
J = lambda *dims: (lambda q: [q.join(x, getattr(lo, FK[x.name])) for x in dims] and q)(Query(lo))
Q = {
 "Q1.1": J(d).where((d.d_year == 1993) & lo.lo_discount.between(1, 3) & (lo.lo_quantity < 25)).sum(disc_rev),
 "Q1.2": J(d).where((d.d_yearmonthnum == 199401) & lo.lo_discount.between(4, 6) & lo.lo_quantity.between(26, 35)).sum(disc_rev),
 "Q1.3": J(d).where((d.d_weeknuminyear == 6) & (d.d_year == 1994) & lo.lo_discount.between(5, 7) & lo.lo_quantity.between(26, 35)).sum(disc_rev),
 "Q2.1": J(d, p, s).where((p.p_category == S(p, "p_category", "MFGR#12")) & (s.s_region == S(s, "s_region", "AMERICA"))).group_by(d.d_year, p.p_brand1).sum(rev),
 "Q2.2": J(d, p, s).where(p.p_brand1.between(*SB(p, "p_brand1", "MFGR#2221", "MFGR#2228")) & (s.s_region == S(s, "s_region", "ASIA"))).group_by(d.d_year, p.p_brand1).sum(rev),
 "Q2.3": J(d, p, s).where((p.p_brand1 == S(p, "p_brand1", "MFGR#2221")) & (s.s_region == S(s, "s_region", "EUROPE"))).group_by(d.d_year, p.p_brand1).sum(rev),
 "Q3.1": J(c, s, d).where((c.c_region == S(c, "c_region", "ASIA")) & (s.s_region == S(s, "s_region", "ASIA")) & d.d_year.between(1992, 1997)).group_by(c.c_nation, s.s_nation, d.d_year).sum(rev),
 "Q3.2": J(c, s, d).where((c.c_nation == S(c, "c_nation", "UNITED STATES")) & (s.s_nation == S(s, "s_nation", "UNITED STATES")) & d.d_year.between(1992, 1997)).group_by(c.c_city, s.s_city, d.d_year).sum(rev),
 "Q3.3": J(c, s, d).where(c.c_city.isin([S(c, "c_city", "UNITED KI1"), S(c, "c_city", "UNITED KI5")]) & s.s_city.isin([S(s, "s_city", "UNITED KI1"), S(s, "s_city", "UNITED KI5")]) & d.d_year.between(1992, 1997)).group_by(c.c_city, s.s_city, d.d_year).sum(rev),
 "Q3.4": J(c, s, d).where(c.c_city.isin([S(c, "c_city", "UNITED KI1"), S(c, "c_city", "UNITED KI5")]) & s.s_city.isin([S(s, "s_city", "UNITED KI1"), S(s, "s_city", "UNITED KI5")]) & (d.d_yearmonth == S(d, "d_yearmonth", "Dec1997"))).group_by(c.c_city, s.s_city, d.d_year).sum(rev),
 "Q4.1": J(d, c, s, p).where((c.c_region == S(c, "c_region", "AMERICA")) & (s.s_region == S(s, "s_region", "AMERICA")) & p.p_mfgr.isin([S(p, "p_mfgr", "MFGR#1"), S(p, "p_mfgr", "MFGR#2")])).group_by(d.d_year, c.c_nation).sum(profit),
 "Q4.2": J(d, c, s, p).where((c.c_region == S(c, "c_region", "AMERICA")) & (s.s_region == S(s, "s_region", "AMERICA")) & d.d_year.isin([1997, 1998]) & p.p_mfgr.isin([S(p, "p_mfgr", "MFGR#1"), S(p, "p_mfgr", "MFGR#2")])).group_by(d.d_year, s.s_nation, p.p_category).sum(profit),
 "K1": J(c).where(c.c_custkey.between(10000, 15000)).group_by(c.c_region).sum(rev),
 "K2": J(d).where(d.d_datekey.between(19950101, 19950331)).sum(rev),
 "K3": J(s, d).where(s.s_suppkey == 77).group_by(d.d_year).sum(rev),
 "M1": J(d, c).group_by(d.d_year, c.c_region).sum(rev),
 "M2": J(c, s).group_by(c.c_nation, s.s_region).sum(profit),
 "M3": J(d).group_by(d.d_year, lo.lo_discount).sum(disc_rev),
 "Q4.3": J(d, c, s, p).where((c.c_region == S(c, "c_region", "AMERICA")) & (s.s_nation == S(s, "s_nation", "UNITED STATES")) & d.d_year.isin([1997, 1998]) & (p.p_category == S(p, "p_category", "MFGR#14"))).group_by(d.d_year, s.s_city, p.p_brand1).sum(profit),
}
Q = {k: Q[k] for k in list(QUERIES) + list(EXTRA)}
if os.environ.get("TORBLES_QUERIES"): Q = {k: Q[k] for k in os.environ["TORBLES_QUERIES"].split(",")}
def decode(name, res):
    """codes -> values, in the query's GROUP BY column order, then the query's ORDER BY"""
    if not GROUPS[name]: return res
    q = Q[name]; keys = list(res); n = len(keys)
    cols = list(zip(*keys)) if n else [()] * len(q.keys)
    vals = {}
    for k, col in zip(q.keys, cols):                   # one lookup per column, not per value
        d = getattr(k.table, "dicts", {}).get(k.name)
        vals[k.name] = d[np.asarray(col, np.int64)].tolist() if d is not None and n else list(col)
    out = [[[vals[g][i] for g in GROUPS[name]], v] for i, v in enumerate(res.values())]
    if name.startswith("Q3"): out.sort(key=lambda r: (r[0][2], -r[1]))     # order by d_year asc, revenue desc
    else: out.sort(key=lambda r: r[0])                                     # order by the group columns
    return out
ref = json.load(open(os.path.join(datadir(sf), "duckdb_results.json")))
rec = {"sf": sf, "design": design, "rows": int(lo.n), "load_fetch_s": t_fetch, "build_s": t_build,
       "ingest": dict(threads=algq.lib.at_get_build_threads(), dims_s=t_dims, read_s=t_read, encode_s=t_enc, build_s=t_build),
       "torble_bytes": {k: int(t.torus_bytes()) for k, t in T.items()},
       "fk_bits": {fk: lo.width(fk) for fk in FK.values()}, "keys": info, "times": {}, "plans": {}}
print(f"SF{sf} {design} keys: {lo.n:,} rows, fetch+re-encode {t_fetch:.1f}s, torble build {t_build:.1f}s, "
      f"fk bits {rec['fk_bits']} = {sum(rec['fk_bits'].values())}, lineorder torble {rec['torble_bytes']['lineorder'] / 2**20:.0f} MiB")
if info: print("  key fields:", {k: v.get("fields", v.get("derivation")) for k, v in info.items() if isinstance(v, dict) and k in DIMS})
allok = True
THREADS = [int(x) for x in os.environ.get("SSB_THREADS", "1,2").split(",")]
# engines: (tag, engine, threads, GROUP BY mode, GPU matrix kernel). Default: the CPU at each thread count (+ the GPU).
# TORBLES_ENGINES, e.g. "cpu-split@1,cpu-mm@24,gpu-split,gpu-mma,gpu-popc": split = per-word mask splitting / per-row
# path with dense group arrays; mm = GROUP BY as one AND-popcount matrix product; gpu-mma = the product on the tensor
# cores (b1 MMA), gpu-popc = the same product with __popc.
def parse_engines(spec):
    out = []
    for it in spec.split(","):
        it = it.strip(); th = THREADS[-1]
        if "@" in it: it, th = it.split("@"); th = int(th)
        if it.startswith("cpu"): out.append((f"{it}@{th}T", "alg", th, it.split("-")[1], None))
        elif it == "gpu-split": out.append(("gpu-split", "gpu", th, "split", None))
        elif it in ("gpu-word", "gpu-mma", "gpu-popc"): out.append((it, "gpu", th, "mm", it.split("-")[1]))
        else: raise SystemExit(f"unknown engine {it}")
    return out
if os.environ.get("TORBLES_ENGINES"): ENGINES = parse_engines(os.environ["TORBLES_ENGINES"])
else:
    ENGINES = [(f"{th}T", "alg", th, algq.GROUPBY, None) for th in THREADS]
    if os.environ.get("TORBLES_GPU"): ENGINES.append(("GPU", "gpu", THREADS[-1], algq.GROUPBY, None))
rec["cpu_mm_kernel"] = algq.mm_kernel()
if any(e[1] == "gpu" for e in ENGINES):
    rec["gpu"] = gpu_name(); t0 = time.perf_counter(); lo.gpu_handle()
    rec["gpu_upload_s"] = time.perf_counter() - t0
    rec["gpu_mm_kernel"] = algq.gpu_mm()
    print(f"GPU: {rec['gpu']}; value planes uploaded and decoded in {rec['gpu_upload_s']:.1f} s; matrix kernel: {rec['gpu_mm_kernel']}")
print("CPU matrix kernel:", rec["cpu_mm_kernel"])
for tag, eng, th, gb, gmode in ENGINES:
    set_threads(th); algq.set_groupby(gb)
    if gmode: rec.setdefault("gpu_mm_kernels", {})[tag] = algq.gpu_mm(gmode)
    if eng == "gpu":                                   # warm the GPU up (clocks) before timing: ~1 s of queries
        tw = time.perf_counter()
        while time.perf_counter() - tw < 1.0:
            for q in Q.values(): q.run(eng)
    for name, q in Q.items():
        t = time.perf_counter(); res = decode(name, q.run(eng)); first = time.perf_counter() - t
        best, kbest, ebest = 1e9, None, None
        for _ in range(3):
            t = time.perf_counter(); r2 = decode(name, q.run(eng)); el = time.perf_counter() - t
            if el < best: best, kbest, ebest = el, q.kernel_ms, getattr(q, "exec_ms", None)
        want = ref["results"][name]
        ok = (res == want) and (r2 == want); allok &= ok
        dt = ref["times"].get(f"{name}/{th}T") or ref["times"][f"{name}/{THREADS[-1]}T"]
        lp = q.last_plan; st = list(lp.mm_stats) if lp.agg == algq.A_MM else None
        rec["times"][f"{name}/{tag}"] = dict(first=first, best=best, match=ok, device=q.device, kernel_ms=kbest, exec_ms=ebest,
                                             agg=q.plan_kind(), mm_stats=st, groups=len(res) if isinstance(res, list) else None)
        joins = [f"{dn}:{k}" for dn, k, *_ in q.range_used] + ["lookup"] * q.last_plan.nsemi
        rec["plans"][name] = dict(joins=joins, agg=q.plan_kind())
        kern = (f" engine {ebest:6.1f} ms" if ebest is not None else "") + (f" kernel {kbest:6.2f} ms" if kbest is not None else "")
        print(f"SF{sf} {design:10s} {name} {tag}: first {first * 1e3:8.1f} ms  best {best * 1e3:8.1f} ms{kern}   DuckDB {th}T best {dt['best'] * 1e3:8.1f} ms  "
              f"{'match' if ok else 'MISMATCH'}  on {q.device}  joins {joins}", flush=True)
rec["all_match"] = allok
rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"peak memory {rec['peak_rss_gib']:.2f} GiB")
json.dump(rec, open(os.path.join(datadir(sf), f"torbles_{design}{os.environ.get('TORBLES_OUT_TAG', '')}.json"), "w"))
print("all results match DuckDB" if allok else "MISMATCHES")
