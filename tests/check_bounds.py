# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Order-encoded attributes as bounds: correctness across predicate shapes, engines, join modes and threads.
In the "bounds" encoding rnd orders the seq field, so any predicate on rnd is a union of seq intervals and the
planner discovers it as a coordinate (cubes on the fk's low planes). Every result is checked against DuckDB."""
import sys, numpy as np, pandas as pd, duckdb
from algq import Table, Query, set_threads
from star8 import make_data, D

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
fails = 0
for MODE in ("bounds", "plain"):
    dims, used, fact, S = make_data(N, MODE)
    T = {k: Table(k, v, used=used[k]) for k, v in dims.items()}
    F = T["fact"] = Table("fact", fact, msb=tuple(f"fk{i}" for i in range(D)), cluster="fk0",
                          refs={f"fk{i}": T[f"dim{i}"] for i in range(D)})
    con = duckdb.connect()
    for name, cols in dims.items():
        ids = np.nonzero(used[name])[0]
        con.register("t", pd.DataFrame({"id": ids, **{k: v[ids].astype(np.int64) for k, v in cols.items()}}))
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM t"); con.unregister("t")
    con.register("t", pd.DataFrame({k: v.astype(np.int64) for k, v in fact.items()})); con.execute("CREATE TABLE fact AS SELECT * FROM t")
    d1, d3, d5 = T["dim1"], T["dim3"], T["dim5"]
    rev = F.price * (100 - F.disc)
    def q(*ds):
        x = Query(F)
        for d in ds: x.join(d, getattr(F, "fk" + d.name[3:]))
        return x
    cases = {
        "rnd < 7":                     q(d5).where(d5.rnd < 7).sum(rev),
        "rnd = 4":                     q(d5).where(d5.rnd == 4).count(),
        "rnd BETWEEN 2 AND 8":         q(d5).where(d5.rnd.between(2, 8)).sum(F.price),
        "rnd <> 3":                    q(d5).where(d5.rnd != 3).sum(rev),
        "rnd IN (1, 5, 9)":            q(d5).where(d5.rnd.isin([1, 5, 9])).count(),
        "rnd >= 7 AND tag < 5":        q(d5).where((d5.rnd >= 7) & (d5.tag < 5)).sum(rev),
        "rnd < 2 OR grp = 0":          q(d5).where((d5.rnd < 2) | (d5.grp == 0)).count(),
        "rnd > 20 (empty)":            q(d5).where(d5.rnd > 20).sum(rev),
        "GROUP BY rnd":                q(d3).group_by(d3.rnd).sum(rev),
        "GROUP BY rnd, tag WHERE rnd<5": q(d3).where(d3.rnd < 5).group_by(d3.rnd, d3.tag).sum(F.price * F.disc),
        "3 dims, rnd bounds":          q(d1, d3, d5).where((d1.rnd < 7) & (d3.rnd.between(3, 6)) & (d5.rnd != 0)).group_by(d1.grp).sum(rev),
        "rnd2 < 7 (not ordered)":      q(d5).where(d5.rnd2 < 7).sum(rev),
    }
    print(f"encoding {MODE}: seq bits {sum(S.values())}, fk bits per fact row {sum(F.width(f'fk{i}') for i in range(D))}")
    for label, x in cases.items():
        rows = con.execute(x.sql()).fetchall()
        want = {tuple(int(v) for v in r[:-1]): int(r[-1]) for r in rows} if x.keys else (None if rows[0][0] is None else int(rows[0][0]))
        bad = []
        for th in (1, 2):
            set_threads(th)
            for mode in ("auto", "factored", "expanded", "lookup"):
                x.join_mode = mode
                for eng in ("alg", "plain"):
                    if x.run(eng) != want: bad.append(f"{eng}/{mode}/{th}T")
        x.join_mode = "auto"; set_threads(1); x.run("alg")
        how = ", ".join(f"{k} x{n} ({sp} vars)" for _, k, n, sp in x.range_used) or ("bitmaps" if x.last_plan.nsemi else "-")
        fails += len(bad)
        print(f"   {label:32s} {'ok' if not bad else 'FAIL ' + ' '.join(bad):6s}  joins as: {how}")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
