# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Table-count scaling: one fact table with 8 dimensions; queries joining k = 1..8 dimension tables.
Dimension ids are coordinates: id = grp | sub | tag | seq (aligned bit fields).
Encoding "plain":  seq is just a row counter inside each (grp, sub, tag) cell; rnd is an ordinary attribute.
Encoding "bounds": seq is ordered by rnd with global offsets, seq = O[rnd] + rank within (cell, rnd), where
  O[v+1] = O[v] + max over cells of count(cell, v). Then rnd is a function of the seq bits alone (the same in every
  cell) and rnd < c  <=>  seq < O[c]: one bound on the fk's low planes, i.e. a few aligned cubes.
rnd2 is a second non-coordinate attribute in both encodings (only one attribute can own the order of seq).
  H family: every joined dimension (beyond the first) filtered on grp < 6   (leading coordinate, keeps 3/4)
  N family: filtered on tag < 7   (a coordinate in the middle of the id, keeps 7/10)
  X family: filtered on rnd < 7   (plain: not a coordinate, bitmaps; bounds: one bound on seq, keeps 7/10)
  Y family: filtered on rnd2 < 7  (not a coordinate in either encoding: bitmaps, keeps 7/10)
Both group by dim0.grp (dim0 is time; facts are in time order) and sum price * (100 - disc).
Engines: torbles 1T/2T, DuckDB 1T/2T, SQLite. Results cross-checked. Usage: python3 scale_tables.py <fact rows> [plain|bounds]"""
import sys, time, json, gc, sqlite3, duckdb, numpy as np, pandas as pd
from algq import Table, Query, set_threads

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000; REPS = 3
MODE = sys.argv[2] if len(sys.argv) > 2 else "plain"
FAMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["H", "N", "X", "Y"]
from star8 import make_data, SIZES, D
dims, used, fact, seqbits = make_data(N, MODE)

T = {name: Table(name, cols, used=used[name]) for name, cols in dims.items()}
T["fact"] = Table("fact", fact, msb=tuple(f"fk{i}" for i in range(D)), cluster="fk0",
                  refs={f"fk{i}": T[f"dim{i}"] for i in range(D)})

duck = duckdb.connect()
lite = sqlite3.connect(":memory:")
for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-2000000", "temp_store=MEMORY"): lite.execute("PRAGMA " + pragma)
for name, cols in dims.items():
    ids = np.nonzero(used[name])[0]
    df = pd.DataFrame({"id": ids.astype(np.int32), **{k: v[ids].astype(np.int32) for k, v in cols.items()}})
    duck.register("tmp", df); duck.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp"); duck.unregister("tmp")
    lite.execute(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY, {', '.join(c + ' INTEGER' for c in cols)})")
    lite.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * (len(cols) + 1))})", df.to_numpy().tolist())
names = list(fact)
lite.execute(f"CREATE TABLE fact ({', '.join(c + ' INTEGER' for c in names)})")
for i in range(0, N, 1_000_000):
    df = pd.DataFrame({k: v[i:i + 1_000_000].view(np.int32) for k, v in fact.items()})
    duck.register("tmp", df); duck.execute("CREATE TABLE fact AS SELECT * FROM tmp" if i == 0 else "INSERT INTO fact SELECT * FROM tmp"); duck.unregister("tmp")
    lite.executemany(f"INSERT INTO fact VALUES ({', '.join('?' * len(names))})", df.to_numpy().tolist())
lite.commit(); del df; gc.collect()

def best(fn, reps):
    b, out = 1e9, None
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); b = min(b, time.perf_counter() - t)
    return b, out
F = T["fact"]
recs = []
print(f"{N:,} fact rows, {D} dimensions {SIZES}, id encoding: {MODE}")
print("fk widths:", [F.width(f"fk{i}") for i in range(D)], "=", sum(F.width(f"fk{i}") for i in range(D)), "bits per fact row")
print(f"{'family':6s}{'k':>3s}{'torbles 1T':>11s}{'torbles 2T':>11s}{'DuckDB 1T':>11s}{'DuckDB 2T':>11s}{'SQLite':>11s}  joins as")
for fam, attr, cut in [f for f in (("H", "grp", 6), ("N", "tag", 7), ("X", "rnd", 7), ("Y", "rnd2", 7)) if f[0] in FAMS]:
    for k in range(1, D + 1):
        q = Query(F)
        for i in range(k): q.join(T[f"dim{i}"], getattr(F, f"fk{i}"))
        pred = None
        for i in range(1, k):
            p = getattr(T[f"dim{i}"], attr) < cut
            pred = p if pred is None else (pred & p)
        if pred is not None: q.where(pred)
        q.group_by(T["dim0"].grp).sum(F.price * (100 - F.disc))
        sql = q.sql(); tm, res = {}, {}
        conv = lambda rows: {(int(r[0]),): int(r[1]) for r in rows}
        for th in (1, 2):
            set_threads(th); tm[f"torble{th}"], res[f"torble{th}"] = best(lambda: q.run("alg"), REPS)
            duck.execute(f"SET threads = {th}"); tm[f"duckdb{th}"], res[f"duckdb{th}"] = best(lambda: conv(duck.execute(sql).fetchall()), REPS)
        set_threads(1)
        tm["sqlite"], res["sqlite"] = best(lambda: conv(lite.execute(sql).fetchall()), 1)
        ok = len({repr(sorted(v.items())) for v in res.values()}) == 1
        kinds = [u[1] for u in q.range_used] + ["bitmaps"] * q.last_plan.nsemi
        how = "-" if not kinds else (kinds[0] if len(set(kinds)) == 1 else "mixed")
        recs.append(dict(family=fam, k=k, match=ok, joins=how, **tm))
        print(f"{fam:6s}{k:>3d}" + "".join(f"{tm[e] * 1e3:9.1f}ms" for e in ("torble1", "torble2", "duckdb1", "duckdb2", "sqlite"))
              + f"  {how}" + ("" if ok else "  MISMATCH"), flush=True)
json.dump(dict(rows=N, sizes=SIZES, mode=MODE, fk_bits=[F.width(f"fk{i}") for i in range(D)], results=recs),
          open(f"scale_tables_{MODE}.json", "w"))
print("all engines agree" if all(r["match"] for r in recs) else "MISMATCHES")
