# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Save and load: a database with every column kind (dictionary strings with Unicode, decimals, 64-bit columns stored
in self-extending sidecars, rank-coded wide dimension columns, derived surrogates with coordinate hints) is saved,
memory-mapped back and queried; the answers must equal those before saving and DuckDB's. Integrity: one flipped bit
in a stored plane is found and repaired on load; two flipped bits in one row are refused; a damaged header and a
truncated file are refused; saving over an open database leaves it intact (files are replaced, not rewritten)."""
import os, shutil, tempfile, datetime as dt, numpy as np, pandas as pd
import torbles, database
rng = np.random.default_rng(11)
n_dim, n_fact = 900, 50_000
cat = rng.integers(0, 6, n_dim)
dim = pd.DataFrame(dict(k=np.arange(n_dim) * 7 - 1000, grp=[f"G{x}-ü" for x in cat], sub=[f"G{x}-S{y}" for x, y in zip(cat, rng.integers(0, 4, n_dim))],
                        ts=pd.Timestamp("2000-01-01") + pd.to_timedelta(rng.integers(0, 10**14, n_dim), unit="us"),
                        big=rng.integers(-2**62, 2**62, n_dim), w=np.round(rng.uniform(-50, 50, n_dim), 3)))
fact = pd.DataFrame(dict(fk=rng.choice(dim.k, n_fact), amount=np.round(rng.uniform(-1e4, 1e4, n_fact), 2),
                         qty=rng.integers(0, 100, n_fact), huge=rng.integers(0, 2**63 - 1, n_fact, dtype=np.int64),
                         day=pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 4000, n_fact), unit="D")))
ddl = """CREATE TABLE dim (k INTEGER PRIMARY KEY, grp VARCHAR, sub VARCHAR, ts TIMESTAMP, big BIGINT, w DECIMAL(9,3));
CREATE TABLE fact (fk INTEGER REFERENCES dim(k), amount DECIMAL(12,2), qty INTEGER, huge BIGINT, day DATE);"""
db = torbles.from_ddl(ddl, {"dim": dim, "fact": fact})
def queries(db):
    f, d = db["fact"], db["dim"]
    return {
        "hierarchy filter, groups": db.query(f).join(d).where(d.grp == "G3-ü").group_by(d.sub).sum(f.amount * f.qty),
        "wide dimension columns": db.query(f).join(d).where((d.ts > dt.datetime(2001, 6, 1)) & (d.big < 0)).group_by(d.grp).count(),
        "64-bit fact column": db.query(f).where(f.huge > 2**62).group_by(f.qty).count(),
        "dates and decimals": db.query(f).where(f.day.between("2012-01-01", "2013-12-31") & (f.amount < 0)).sum(f.amount),
        "decimal dimension column": db.query(f).join(d).where(d.w >= 12.5).group_by(d.grp).sum(f.qty),
    }
before = {k: q.run() for k, q in queries(db).items()}
want = {k: db.sql_result(q) for k, q in queries(db).items()}
fails = sum(before[k] != want[k] for k in want)
tmp = tempfile.mkdtemp(prefix="torbles-persist-")
try:
    p = os.path.join(tmp, "db.torbles"); db.save(p)
    print("files:", sorted(os.listdir(p)))
    loaded = torbles.load(p, con=db.con)
    after = {k: q.run() for k, q in queries(loaded).items()}
    for k in want:
        ok = after[k] == want[k]; fails += not ok
        print(f"  {k:28s} {'ok' if ok else 'FAIL'}")
    print("  integrity on load:", loaded.integrity)
    # saving over an open database: the open one keeps its (old) files
    db.save(p); again = {k: q.run() for k, q in queries(loaded).items()}
    ok = again == want; fails += not ok; print(f"  re-saved under an open database: {'ok' if ok else 'FAIL'}")
    # integrity: one flipped bit in the fact's planes is repaired, two in one row are refused
    def damaged(flips, name="fact.tbl"):
        d2 = p + f"-{len(flips)}"; shutil.rmtree(d2, ignore_errors=True); shutil.copytree(p, d2)
        f = os.path.join(d2, name); b = bytearray(open(f, "rb").read())
        for pos, bit in flips: b[pos] ^= 1 << bit
        open(f, "wb").write(b); return d2
    hdr = 4096; plane = ((n_fact + 63) // 64 * 8 + 63) // 64 * 64
    one = torbles.load(damaged([(hdr + 3 * plane + 100, 2)]), con=db.con)
    ok = one.integrity["fact"]["rows_with_faults"] == 1 and one.integrity["fact"]["repaired"] == 1
    ok &= {k: q.run() for k, q in queries(one).items()} == want
    fails += not ok; print(f"  one flipped bit: found and repaired, answers unchanged: {'ok' if ok else 'FAIL'} {one.integrity['fact']}")
    try:
        torbles.load(damaged([(hdr + 3 * plane + 100, 2), (hdr + 7 * plane + 100, 2)]), con=db.con); print("  two flipped bits in one row: NOT REFUSED"); fails += 1
    except torbles.LoadError as e: print(f"  two flipped bits in one row: refused ok ({str(e)[:80]})")
    for what, flips in (("damaged header", [(40, 0)]),):
        try: torbles.load(damaged(flips), con=db.con); print(f"  {what}: NOT REFUSED"); fails += 1
        except torbles.LoadError as e: print(f"  {what}: refused ok ({e})")
    d3 = p + "-cut"; shutil.copytree(p, d3); f = os.path.join(d3, "fact.tbl"); data = open(f, "rb").read(); open(f, "wb").write(data[:-1000])
    try: torbles.load(d3); print("  truncated file: NOT REFUSED"); fails += 1
    except torbles.LoadError as e: print(f"  truncated file: refused ok ({e})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("ALL MATCH" if not fails else f"{fails} FAILURES")
