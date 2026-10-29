# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Collect a scaling run (wsl/run_scaling.sh) into scaling.json and print a summary.
    python ssb_scaling.py <results/scaling directory>
Totals are over the 13 official SSB queries, best of 3, end to end (planning, execution, result). Only queries that
torbles answered correctly count: a query that failed or differed from DuckDB is left out of both engines' totals."""
import json, os, sys, glob

out = sys.argv[1]
SSB13 = ["Q1.1", "Q1.2", "Q1.3", "Q2.1", "Q2.2", "Q2.3", "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q4.1", "Q4.2", "Q4.3"]
def answered(T, tag):
    """the SSB queries torbles answered correctly under this engine tag"""
    return [q for q in SSB13 if T["times"].get(f"{q}/{tag}", {}).get("match")]
def total(times, tag, key="best", qs=SSB13):
    v = [times.get(f"{q}/{tag}", {}).get(key) for q in qs]
    return None if not qs or any(x is None for x in v) else sum(v) * 1e3
summary = {"queries": SSB13, "scale": [], "threads": None}
for d in sorted(glob.glob(os.path.join(out, "sf*")), key=lambda p: int(os.path.basename(p)[2:])):
    try:
        D = json.load(open(os.path.join(d, "duckdb_results.json"))); T = json.load(open(os.path.join(d, "torbles_derived_scaling.json")))
    except FileNotFoundError: continue
    th = sorted({int(k.split("/")[1][:-1]) for k in D["times"]})[-1]
    cpu = [t for t in {k.split("/", 1)[1] for k in T["times"]} if t.startswith("cpu")][0]
    qc, qg = answered(T, cpu), answered(T, "gpu-word")                # both engines are timed on the same queries
    qs = [q for q in qc if q in qg] if qg else qc
    row = dict(sf=T["sf"], rows=T["rows"], threads=th, all_match=T["all_match"], answered=len(qs),
               duckdb_ms=total(D["times"], f"{th}T", qs=qs), torbles_cpu_ms=total(T["times"], cpu, qs=qs),
               torbles_gpu_ms=total(T["times"], "gpu-word", qs=qs),
               gpu_kernel_ms=sum(T["times"][f"{q}/gpu-word"]["kernel_ms"] or 0 for q in qs if f"{q}/gpu-word" in T["times"]),
               duckdb_first_ms=total(D["times"], f"{th}T", "first", qs), torbles_gpu_first_ms=total(T["times"], "gpu-word", "first", qs),
               torbles_cpu_first_ms=total(T["times"], cpu, "first", qs), duckdb_in_memory=D.get("in_memory"),
               duckdb_memory_limit=D.get("memory_limit"), duckdb_load_s=D.get("load_s"),
               ingest=T.get("ingest"), gpu_upload_s=T.get("gpu_upload_s"), gpu=T.get("gpu"))
    summary["duckdb_version"] = D.get("engine", "DuckDB").replace("DuckDB ", "")
    summary["scale"].append(row)
td = os.path.join(out, "threads_sf10")
if os.path.isdir(td):
    D = json.load(open(os.path.join(td, "duckdb_results.json"))); T = json.load(open(os.path.join(td, "torbles_derived_threads.json")))
    ths = sorted({int(k.split("/")[1][:-1]) for k in D["times"]})
    summary["threads"] = dict(sf=10, rows=T["rows"], all_match=T["all_match"], points=[
        dict(threads=t, answered=len(qs), duckdb_ms=total(D["times"], f"{t}T", qs=qs), torbles_cpu_ms=total(T["times"], f"cpu-mm@{t}T", qs=qs))
        for t in ths for qs in [answered(T, f"cpu-mm@{t}T")]])
json.dump(summary, open(os.path.join(out, "scaling.json"), "w"), indent=1)
print("13 official SSB queries, total ms, best of 3, end to end (queries answered correctly only)")
print(f"{'SF':>4} {'rows':>12} {'DuckDB':>10} {'torbles CPU':>11} {'torbles GPU':>11} {'GPU kernels':>12} {'GPU vs DuckDB':>14}  all match   "
      f"first runs: DuckDB / torbles GPU   DuckDB in memory")
for r in summary["scale"]:
    sp = r["duckdb_ms"] / r["torbles_gpu_ms"] if r["torbles_gpu_ms"] else float("nan")
    if r["answered"] < len(SSB13): print(f"  SF{r['sf']}: only {r['answered']} of {len(SSB13)} queries answered correctly; totals are over those")
    print(f"{r['sf']:>4} {r['rows']:>12,} {r['duckdb_ms']:>10.1f} {r['torbles_cpu_ms']:>11.1f} {r['torbles_gpu_ms']:>11.1f} {r['gpu_kernel_ms']:>12.2f} {sp:>13.1f}x  "
          f"{str(r['all_match']):9s} {r['duckdb_first_ms']:10.1f} / {r['torbles_gpu_first_ms']:6.1f}      {r['duckdb_in_memory']} ({r['duckdb_memory_limit']})")
if summary["threads"]:
    print("\nthreads at SF10:")
    for p in summary["threads"]["points"]: print(f"  {p['threads']:>3} threads: DuckDB {p['duckdb_ms']:8.1f} ms   torbles CPU {p['torbles_cpu_ms']:8.1f} ms")
