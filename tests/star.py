# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Star schema modeled on the Star Schema Benchmark (SSB): lineorder fact table + dates, customer, supplier, part.
Builds the torbles (algebraic engine + plain columnar baseline) and the same tables in DuckDB."""
import numpy as np, time
from algq import Table, Query, Col

def _cells(rng, keys, n):
    """sequence number of each row within its hierarchy cell, and the bits needed for it"""
    order = np.lexsort(keys[::-1]); seq = np.empty(n, np.int64)
    cell = np.zeros(n, np.int64)
    for k in keys: cell = cell * 1024 + k
    cs = cell[order]; first = np.concatenate([[True], cs[1:] != cs[:-1]])
    grp_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    seq[order] = np.arange(n) - grp_start
    return seq, int(seq.max()).bit_length() or 1

def _dense(ids, cols, size):
    out = {}
    for k, v in cols.items():
        a = np.zeros(size, np.uint32); a[ids] = v; out[k] = a
    used = np.zeros(size, bool); used[ids] = True
    return out, used

def generate(n_fact, seed=2026):
    """Dimension ids are coordinates in the hierarchy: power-of-two-aligned bit fields, most significant first.
    customer/supplier id = region | nation-in-region | city-in-nation | seq;  part id = mfgr | category | brand | seq;
    date id = year | day-of-year. A hierarchy filter therefore selects an aligned block of ids."""
    rng = np.random.default_rng(seed)
    nc, ns, npart = max(1000, n_fact // 200), max(100, n_fact // 3000), max(2000, n_fact // 50)
    # dates
    day = np.arange(2557); year = np.minimum(day // 365, 6); doy = day - year * 365
    did = year * 512 + doy; dsize = int(did.max()) + 1
    ay, adoy = np.arange(dsize) >> 9, np.arange(dsize) & 511
    dates = dict(year=ay, month=np.minimum(adoy // 31 + 1, 12), yearmonth=ay * 12 + np.minimum(adoy // 31, 11),
                 weeknum=np.minimum(adoy // 7 + 1, 53), weekday=np.zeros(dsize, int))
    dates["weekday"][did] = day % 7
    dates = {k: np.asarray(v, np.uint32) for k, v in dates.items()}; dused = np.zeros(dsize, bool); dused[did] = True
    # customer and supplier: region | nation-in-region | city-in-nation | seq
    def geo(n, extra):
        nation = rng.integers(0, 25, n); region, nir, ci = nation // 5, nation % 5, rng.integers(0, 10, n)
        seq, S = _cells(rng, [region, nir, ci], n)
        ids = ((((region * 8 + nir) * 16 + ci) << S) | seq).astype(np.int64)
        size = int(ids.max()) + 1; a = np.arange(size)
        reg, nr, cc = a >> (S + 7), (a >> (S + 4)) & 7, (a >> S) & 15        # unused ids follow their prefix
        cols = dict(nation=(reg * 5 + nr), region=reg, city=(reg * 5 + nr) * 10 + cc)
        cols = {k: v.astype(np.uint32) for k, v in cols.items()}
        for k, v in extra.items(): cols[k] = np.zeros(size, np.uint32); cols[k][ids] = v
        used = np.zeros(size, bool); used[ids] = True
        return cols, used, ids
    customer, cused, cids = geo(nc, dict(segment=rng.integers(0, 5, nc)))
    supplier, sused, sids = geo(ns, {})
    # part: mfgr | category-in-mfgr | brand-in-category | seq
    mf, cim, bic = rng.integers(0, 5, npart), rng.integers(0, 5, npart), rng.integers(0, 40, npart)
    seq, S = _cells(rng, [mf, cim, bic], npart)
    pids = ((((mf * 8 + cim) * 64 + bic) << S) | seq).astype(np.int64)
    psize = int(pids.max()) + 1; a = np.arange(psize)
    pm, pc, pb = a >> (S + 9), (a >> (S + 6)) & 7, (a >> S) & 63
    part = dict(mfgr=pm, category=pm * 5 + pc, brand=(pm * 5 + pc) * 40 + pb)
    part = {k: v.astype(np.uint32) for k, v in part.items()}
    part["size"] = np.zeros(psize, np.uint32); part["size"][pids] = rng.integers(1, 51, npart)
    pprice = np.zeros(psize, np.uint32); pprice[pids] = rng.integers(1, 2001, npart)
    pused = np.zeros(psize, bool); pused[pids] = True
    # fact
    lp = pids[rng.integers(0, npart, n_fact)].astype(np.uint32); qty = rng.integers(1, 51, n_fact, dtype=np.uint32)
    price = (qty * pprice[lp]).astype(np.uint32)
    lineorder = dict(date=did[rng.integers(0, 2557, n_fact)].astype(np.uint32), cust=cids[rng.integers(0, nc, n_fact)].astype(np.uint32),
                     part=lp, supp=sids[rng.integers(0, ns, n_fact)].astype(np.uint32), quantity=qty, price=price,
                     discount=rng.integers(0, 11, n_fact, dtype=np.uint32), tax=rng.integers(0, 9, n_fact, dtype=np.uint32),
                     supplycost=(price.astype(np.uint64) * rng.integers(40, 71, n_fact) // 100).astype(np.uint32))
    order = np.argsort(lineorder["date"], kind="stable")          # facts arrive in time order
    lineorder = {k: v[order] for k, v in lineorder.items()}
    raw = dict(dates=dates, customer=customer, supplier=supplier, part=part, lineorder=lineorder)
    used = dict(dates=dused, customer=cused, supplier=sused, part=pused)
    return raw, used

FKS = ("date", "cust", "part", "supp")

def build(gen):
    raw, used = gen
    t0 = time.perf_counter()
    T = {name: Table(name, cols, used=used.get(name)) for name, cols in raw.items() if name != "lineorder"}
    T["lineorder"] = Table("lineorder", raw["lineorder"], msb=FKS, cluster="date",
                           refs=dict(date=T["dates"], cust=T["customer"], part=T["part"], supp=T["supplier"]))
    return T, time.perf_counter() - t0

def load_duckdb(gen, con):
    import pandas as pd
    raw, used = gen
    t0 = time.perf_counter()
    for name, cols in raw.items():
        if name == "lineorder":                      # stream the fact table in chunks: bounded temporary memory
            n = len(next(iter(cols.values())))
            for i in range(0, n, 2_000_000):
                df = pd.DataFrame({k: np.asarray(v)[i:i + 2_000_000].view(np.int32) for k, v in cols.items()})
                con.register("tmp_df", df)
                con.execute("CREATE TABLE lineorder AS SELECT * FROM tmp_df" if i == 0 else "INSERT INTO lineorder SELECT * FROM tmp_df")
                con.unregister("tmp_df"); del df
            continue
        else:
            ids = np.nonzero(used[name])[0]
            df = pd.DataFrame({"id": ids.astype(np.int32), **{k: np.asarray(v)[ids].astype(np.int32) for k, v in cols.items()}})
        con.register("tmp_df", df); con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df"); con.unregister("tmp_df")
        del df
    return time.perf_counter() - t0

def queries(T):
    lo, d, c, s, p = T["lineorder"], T["dates"], T["customer"], T["supplier"], T["part"]
    revenue = lo.price * (100 - lo.discount)
    Q = {}
    Q["Q1.1 discount revenue, 1 join"] = (Query(lo).join(d, lo.date)
        .where((d.year == 1) & lo.discount.between(1, 3) & (lo.quantity < 25)).sum(lo.price * lo.discount))
    Q["Q1.2 same, one month"] = (Query(lo).join(d, lo.date)
        .where((d.yearmonth == 25) & lo.discount.between(4, 6) & lo.quantity.between(26, 35)).sum(lo.price * lo.discount))
    Q["Q1.3 same, one week"] = (Query(lo).join(d, lo.date)
        .where((d.weeknum == 6) & (d.year == 2) & lo.discount.between(5, 7) & lo.quantity.between(26, 35)).sum(lo.price * lo.discount))
    Q["Q2.1 brand x year, 3 joins"] = (Query(lo).join(d, lo.date).join(p, lo.part).join(s, lo.supp)
        .where((p.category == 12) & (s.region == 1)).group_by(d.year, p.brand).sum(revenue))
    Q["Q2.3 one brand, 3 joins"] = (Query(lo).join(d, lo.date).join(p, lo.part).join(s, lo.supp)
        .where((p.brand == 482) & (s.region == 3)).group_by(d.year, p.brand).sum(revenue))
    Q["Q3.1 nation x nation x year"] = (Query(lo).join(c, lo.cust).join(s, lo.supp).join(d, lo.date)
        .where((c.region == 2) & (s.region == 2) & d.year.between(1, 5)).group_by(c.nation, s.nation, d.year).sum(revenue))
    Q["Q3.3 city x city x year"] = (Query(lo).join(c, lo.cust).join(s, lo.supp).join(d, lo.date)
        .where(c.city.isin([221, 225]) & s.city.isin([221, 225]) & d.year.between(1, 5)).group_by(c.city, s.city, d.year).sum(revenue))
    Q["Q4.1 profit, 4 joins"] = (Query(lo).join(d, lo.date).join(c, lo.cust).join(s, lo.supp).join(p, lo.part)
        .where((c.region == 1) & (s.region == 1) & p.mfgr.isin([0, 1])).group_by(d.year, c.nation).sum(revenue - lo.supplycost))
    Q["Q4.3 profit by city, 4 joins"] = (Query(lo).join(d, lo.date).join(c, lo.cust).join(s, lo.supp).join(p, lo.part)
        .where((s.nation == 7) & (c.region == 1) & d.year.between(5, 6) & (p.category == 7))
        .group_by(d.year, s.city, p.brand).sum(revenue - lo.supplycost))
    Q["X1 gross margin, linear form"] = Query(lo).where(lo.quantity > 40).sum(lo.price - lo.supplycost + 3 * lo.tax)
    Q["X2 not/or filters + count"] = (Query(lo).where(~(lo.discount.between(2, 8)) & ((lo.quantity < 5) | (lo.tax == 8))).count())
    return Q
