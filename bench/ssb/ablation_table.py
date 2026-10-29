# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Table of an ablation run (ablation.sh): best time per query over the rounds, counting only runs whose answer equals
DuckDB's; DuckDB's best over its own rounds (duckdb_r*.json from ssb_duck.py). Writes summary.json in the out dir.
    python3 ablation_table.py <out dir>"""
import json, glob, os, sys
from ssb_common import datadir
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_out")
D = datadir(1) + "/"
Q = [f"Q{x}" for x in ("1.1","1.2","1.3","2.1","2.2","2.3","3.1","3.2","3.3","3.4","4.1","4.2","4.3")]; M = ["M1", "M2", "M3"]
names = {"a_committed": "0.2.0 as committed", "b_transpose": "+ vectorized transpose (bridges off)", "c_factor": "+ factor through the key",
         "d_tree": "+ splitting tree, direct slots", "e_direct": "+ direct slots only", "f_rows": "+ row kernel (cost-chosen)",
         "g_mobius": "+ Möbius (cost-chosen)", "h_factor_rows": "+ factor + row kernel", "i_factor_mobius": "+ factor + Möbius",
         "j_factor_rows_mobius": "+ factor + row kernel + Möbius", "k_rows_forced": "factor + row kernel on every tile",
         "l_mobius_forced": "factor + Möbius on every tile", "m_nomargin": "factor + row kernel, no margin (model as is)", "n_defaults": "defaults: factor + row kernel at margin 0.5"}
ds = [json.load(open(f)) for f in sorted(glob.glob(D + "duckdb_r*.json"))] or [json.load(open(D + "duckdb_results.json"))]
duck = {th: {q: min(d["times"][f"{q}/{th}T"]["best"] for d in ds) * 1e3 for q in Q + M} for th in (1, 2)}
rows = []
for k in sorted(names):
    js = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/{k}.r*.json"))]
    if not js: continue
    ok = all(all(j["match"].values()) for j in js)
    # only correct answers are timed: a run's time counts only when that run matched DuckDB
    best = {th: {q: min([j["times"][f"{q}/cpu@{th}"] for j in js if j["match"].get(f"{q}/cpu@{th}")] or [None], key=lambda x: (x is None, x))
                 for q in Q + M} for th in (1, 2)}
    best = {th: {q: (v * 1e3 if v is not None else None) for q, v in b.items()} for th, b in best.items()}
    rows.append((k, names[k], ok, best))
def fmt(x): return "-" if x is None else f"{x:.1f}"
def tot(b, qs): v = [b[q] for q in qs if b[q] is not None]; return sum(v) if len(v) == len(qs) else None   # None: not every query answered
print("| configuration | 13 SSB 1T | 13 SSB 2T | M1 1T | M2 1T | M3 1T | M1-M3 1T | M1-M3 2T | answers |")
print("|---|---|---|---|---|---|---|---|---|")
print(f"| DuckDB (in memory) | {fmt(sum(duck[1][q] for q in Q))} | {fmt(sum(duck[2][q] for q in Q))} | {fmt(duck[1]['M1'])} | {fmt(duck[1]['M2'])} | {fmt(duck[1]['M3'])} | {fmt(sum(duck[1][q] for q in M))} | {fmt(sum(duck[2][q] for q in M))} | reference |")
for k, n, ok, b in rows:
    nans = sum(b[1][q] is not None for q in Q + M)
    print(f"| {n} | {fmt(tot(b[1], Q))} | {fmt(tot(b[2], Q))} | {fmt(b[1]['M1'])} | {fmt(b[1]['M2'])} | {fmt(b[1]['M3'])} | {fmt(tot(b[1], M))} | {fmt(tot(b[2], M))} | {'all equal DuckDB' if ok else f'{nans} of {len(Q + M)} correct'} |")
json.dump({k: dict(name=n, ok=ok, best=b) for k, n, ok, b in rows} | {"duckdb": duck}, open(f"{OUT}/summary.json", "w"), indent=1)
# per-query detail for the main configurations
sel = [r for r in rows if r[0] in ("a_committed", "b_transpose", "c_factor", "m_nomargin", "n_defaults")]
print("\n| query (1 thread) | DuckDB | " + " | ".join(r[1] for r in sel) + " |")
print("|---" * (len(sel) + 2) + "|")
for q in Q + M:
    print(f"| {q} | {fmt(duck[1][q])} | " + " | ".join(fmt(r[3][1][q]) for r in sel) + " |")
