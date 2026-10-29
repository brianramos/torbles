# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Self-extending records: a 64-bit-capable column stored as a narrow base for every row plus a flag plane, with the
high part only for flagged rows in a sidecar torble addressed by rank (popcount over the flag plane) and scattered
back per word with PDEP. Checked against DuckDB (BIGINT) on every engine and thread count; storage and integrity."""
import sys, numpy as np, pandas as pd, duckdb
from algq import Table, Query, set_threads, lib

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
ENGINES = ["alg", "plain"] + (["gpu"] if "nogpu" not in sys.argv else [])
rng = np.random.default_rng(8)
amt = rng.integers(0, 1 << 16, N).astype(np.uint64)                     # skewed: almost all values fit 16 bits
tail = rng.random(N) < 0.01                                              # 1% need up to 40 bits
amt[tail] = rng.integers(1 << 16, 1 << 40, int(tail.sum()), dtype=np.uint64)
big = np.uint64(1 << 40) - np.uint64(1); amt[:3] = [big, np.uint64(1 << 16), np.uint64((1 << 16) - 1)]
small = rng.integers(0, 1 << 12, N).astype(np.uint64)                  # declared extensible, never extends
q = rng.integers(1, 50, N).astype(np.uint32); g = rng.integers(0, 8, N).astype(np.uint32)
dim = Table("dim", dict(r=(np.arange(64) % 5).astype(np.uint32)))
fk = rng.integers(0, 64, N).astype(np.uint32)
F = Table("fact", dict(amt=amt, small=small, q=q, g=g, fk=fk), msb=("fk",), refs=dict(fk=dim), extend=dict(amt=16, small=16))
sc = F.sidecars["amt#hi"]
print(f"{N:,} rows; amt: {int(tail.sum()) + 1:,} rows extend ({100 * sc['rows'] / N:.2f}%), widths base {F.width('amt')} + flag 1 "
      f"+ sidecar {lib.at_width(sc['side'], 0)} bits for {sc['rows']:,} rows; small: {F.sidecars['small#hi']['rows']} rows extend")
dense_planes = 40 * N / 8; ext_planes = (16 + 1) * N / 8 + lib.at_bytes(sc["side"]) / 2      # V planes only, roughly
print(f"value storage for amt: dense 40-bit planes {dense_planes / 2**20:.1f} MiB vs base+flag+sidecar {ext_planes / 2**20:.1f} MiB")

con = duckdb.connect()
con.register("t", pd.DataFrame(dict(amt=amt.astype(np.int64), small=small.astype(np.int64), q=q.astype(np.int64), g=g.astype(np.int64), fk=fk.astype(np.int64))))
con.execute("CREATE TABLE fact AS SELECT * FROM t"); con.unregister("t")
con.register("t", pd.DataFrame(dict(id=np.arange(64), r=np.arange(64) % 5))); con.execute("CREATE TABLE dim AS SELECT * FROM t")
A = F.amt
cases = {
    "SUM(amt)": Query(F).sum(A),
    "SUM(amt) WHERE amt > 2^35": Query(F).where(A > (1 << 35)).sum(A),
    "COUNT amt BETWEEN 2^20 AND 2^38": Query(F).where(A.between(1 << 20, 1 << 38)).count(),
    "COUNT amt = 2^40 - 1": Query(F).where(A == (1 << 40) - 1).count(),
    "COUNT amt <> 65536": Query(F).where(A != 1 << 16).count(),
    "SUM(amt) WHERE amt < 1000": Query(F).where(A < 1000).sum(A),
    "SUM(amt * q)": Query(F).sum(A * F.q),
    "SUM(amt + 3 small)": Query(F).sum(A + 3 * F.small),
    "SUM(amt) GROUP BY g": Query(F).group_by(F.g).sum(A),
    "COUNT GROUP BY g, q WHERE amt >= 2^39": Query(F).where(A >= 1 << 39).group_by(F.g, F.q).count(),
    "GROUP BY small (extensible, 2^16 x 1 domain)": Query(F).where(F.small < 100).group_by(F.small).sum(A),
    "OR / NOT on wide values": Query(F).where((A < 100) | ~(A <= (1 << 30))).sum(F.q),
    "join + wide filter, by dim": Query(F).join(dim, F.fk).where((dim.r == 2) & (A > 5000)).group_by(dim.r).sum(A),
    "empty: amt > 2^40": Query(F).where(A > (1 << 40)).sum(A),
}
fails = 0
for name, x in cases.items():
    rows = con.execute(x.sql()).fetchall()
    want = {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in rows} if x.keys else (None if rows[0][0] is None else int(rows[0][0]))
    bad = []
    for th in (1, 2):
        set_threads(th)
        for eng in ENGINES:
            if x.run(eng) != want: bad.append(f"{eng}/{th}T")
    set_threads(1); x.run("gpu" if "gpu" in ENGINES else "alg")
    fails += len(bad)
    print(f"   {name:40s} {'ok' if not bad else 'FAIL ' + ' '.join(bad)}" + (f"   (on {x.device})" if "gpu" in ENGINES else ""))
# integrity: the sidecar is a torble with its own faces; a flipped bit is found and repaired
side = sc["side"]; nb = lib.at_nplanes(side)
bad_rows = np.zeros(4, np.uint64)
for p, r in ((0, 3), (nb - 1, min(17, sc["rows"] - 1))): lib.at_flip(side, p, int(r))
det = lib.at_validate(side, bad_rows.ctypes.data, 4); fixed = lib.at_repair(side)
after = Query(F).sum(A).run("alg")
ok = det == 2 and fixed == 2 and after == int(amt.astype(object).sum())
fails += 0 if ok else 1
print(f"   sidecar integrity: 2 bit faults injected, {det} detected, {fixed} repaired, SUM(amt) restored: {after == int(amt.astype(object).sum())}")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
