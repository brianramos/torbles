# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Summary of a matrix-product run (ssb_torbles.py with TORBLES_ENGINES): end to end, engine call and kernel times.
    python ssb_mm_report.py torbles_derived_mm.json duckdb_results.json"""
import json, sys
R, D = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
T = R["times"]
names = []
for k in T:
    n = k.split("/")[0]
    if n not in names: names.append(n)
tags = []
for k in T:
    t = k.split("/", 1)[1]
    if t not in tags: tags.append(t)
dth = sorted({k.split("/")[1] for k in D["times"]}, key=lambda x: int(x[:-1]))
ssb = [n for n in names if n.startswith("Q")]; grp = [n for n in ssb if not n.startswith("Q1")]; extra = [n for n in names if n.startswith("K")]; dense = [n for n in names if n.startswith("M")]
print(f"SF{R['sf']}, {R['rows']:,} rows. GPU: {R.get('gpu', '-')}")
print(f"CPU matrix kernel: {R.get('cpu_mm_kernel')}; GPU matrix kernels: {R.get('gpu_mm_kernels')}")
if R.get("ingest"):
    g = R["ingest"]; print(f"ingestion ({g['threads']} threads): dimensions {g['dims_s']:.2f} s, fact columns read from DuckDB {g['read_s']:.2f} s, "
                          f"keys re-encoded {g['encode_s']:.2f} s, torble build {g['build_s']:.2f} s; GPU upload {R.get('gpu_upload_s', 0):.2f} s")
print("all results equal DuckDB's:", R["all_match"])
print()
print("End to end (Python planning + engine), best of 3, ms")
hdr = ["query"] + [f"DuckDB {t}" for t in dth] + tags
print(" | ".join(f"{h:>13s}" for h in hdr))
def row(n):
    vals = [D["times"][f"{n}/{t}"]["best"] * 1e3 for t in dth] + [T[f"{n}/{t}"]["best"] * 1e3 for t in tags]
    return vals
for n in names:
    ok = all(T[f"{n}/{t}"]["match"] for t in tags)
    print(" | ".join([f"{n:>13s}"] + [f"{v:13.1f}" for v in row(n)]) + ("" if ok else "  MISMATCH"))
for label, sel in (("13 SSB queries", ssb), ("GROUP BY (Q2-Q4)", grp), ("K1-K3", extra), ("M1-M3", dense)):
    tot = [sum(row(n)[i] for n in sel) for i in range(len(hdr) - 1)]
    print(" | ".join([f"{label:>13s}"[:13].rjust(13)] + [f"{v:13.1f}" for v in tot]))
print(" | ".join([f"{'SSB, 1st run':>13s}"] + [f"{sum(D['times'][f'{n}/{t}']['first'] for n in ssb) * 1e3:13.1f}" for t in dth]
                + [f"{sum(T[f'{n}/{t}']['first'] for n in ssb) * 1e3:13.1f}" for t in tags]))
print("(1st run: the query planned from scratch; best of 3: torbles reuses the compiled plan, DuckDB plans every run)")
print()
print("Engine call only (no planning) and GPU kernel time, ms; groups = result rows")
gt = [t for t in tags if t.startswith("gpu")]; ct = [t for t in tags if t.startswith("cpu")]
print(" | ".join(f"{h:>13s}" for h in ["query", "groups"] + [f"{t} eng" for t in ct + gt] + [f"{t} kern" for t in gt]))
for n in names:
    e = [T[f"{n}/{t}"].get("exec_ms") or 0 for t in ct + gt]; k = [T[f"{n}/{t}"].get("kernel_ms") or 0 for t in gt]
    g = T[f"{n}/{tags[0]}"].get("groups")
    print(" | ".join([f"{n:>13s}", f"{g if g is not None else '-':>13}"] + [f"{v:13.2f}" for v in e + k]))
for label, sel in (("13 SSB", ssb), ("GROUP BY", grp), ("K1-K3", extra), ("M1-M3", dense)):
    e = [sum(T[f"{n}/{t}"].get("exec_ms") or 0 for n in sel) for t in ct + gt]; k = [sum(T[f"{n}/{t}"].get("kernel_ms") or 0 for n in sel) for t in gt]
    print(" | ".join([f"{label:>13s}", f"{'':>13s}"] + [f"{v:13.2f}" for v in e + k]))
print()
print("matrix-product tiles: [group-slots dense, group-slots sparse, pairs evaluated per word (tile table full), tiles with groups]")
for n in grp + dense + [x for x in extra if T[f"{x}/{tags[0]}"].get("groups")]:
    st = {t: T[f"{n}/{t}"].get("mm_stats") for t in tags if T[f"{n}/{t}"].get("mm_stats") and t.startswith("cpu")}
    if st: t0 = list(st)[-1]; print(f"  {n}: {t0} {st[t0]}")
