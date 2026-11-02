# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Where each engine query of an SSB query in SQL runs: at SF30 and SF50, the 13 official queries through db.sql() on
the CPU and the GPU engines, and for every engine query they issue, the device it ran on, why a GPU query fell back
(if it did), its plan's aggregation and joins, and its time. The release run of 2026-11-02 found Q3.1 through SQL on
the GPU at 3.5 ms at SF30 and 43.8 ms at SF50 (through the Python API: 2.3 and 4.0 ms).
    python3 torbles_sql_devices.py <repo> <out.json> [scales, default "30 50"] [threads, default all]"""
import json, os, sys, time
repo = os.path.abspath(sys.argv[1]); out_path = sys.argv[2]
sfs = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "30 50").split()]
threads = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 1)
sys.path[:0] = [repo, os.path.join(repo, "torbles"), os.path.join(repo, "bench", "ssb")]
import torbles, algq
from ssb_common import QUERIES, FACT_COLS, dbfile

trace = []
_loop = algq.Query._exec_loop
def traced(self, dev, table, plan):
    """every engine call: the device asked for, the one it ran on, why a GPU call did not run there"""
    t0 = time.perf_counter(); self.__dict__.pop("_gpu_fail", None)
    ok = _loop(self, dev, table, plan)
    trace.append(dict(asked=dev, ran=bool(ok), device=getattr(self, "device", None), gpu_fail=getattr(self, "_gpu_fail", None),
                      ms=(time.perf_counter() - t0) * 1e3, kernel_ms=getattr(self, "kernel_ms", None),
                      agg=int(getattr(plan, "agg", -1)), nkey=int(getattr(plan, "nkey", -1)), ngroups=int(getattr(plan, "ngroups", 0) or 0),
                      nattr=int(getattr(plan, "nattr", 0) or 0), keys=[getattr(k, "name", type(k).__name__) for k in getattr(self, "keys", [])],
                      join_mode=getattr(self, "join_mode", None)))
    return ok
algq.Query._exec_loop = traced
torbles.set_threads(threads)
res = dict(threads=threads, version=torbles.__version__, gpu=torbles.gpu_name(), scales={})
for sf in sfs:
    t0 = time.perf_counter()
    db = torbles.from_duckdb(dbfile(sf), schema=os.path.join(repo, "bench", "ssb", "schema.sql"), columns={"lineorder": FACT_COLS})
    r = dict(load_s=time.perf_counter() - t0, describe=db.describe(), queries={})
    print(f"SF{sf}: loaded in {r['load_s']:.1f} s", flush=True)
    for name, q in QUERIES.items():
        for e in ("alg", "gpu"):
            runs = []
            for k in range(4):                         # the first run uploads to the GPU: the last three are kept
                trace.clear(); t1 = time.perf_counter(); db.sql(q, e).rows; ms = (time.perf_counter() - t1) * 1e3
                runs.append(dict(ms=ms, engine_queries=list(trace)))
            best = min(runs[1:], key=lambda x: x["ms"])
            r["queries"][f"{name}/{e}"] = best
            devs = sorted({f"{t['asked']}->{t['device'] if t['ran'] else 'fallback'}" for t in best["engine_queries"]})
            fails = sorted({str(t["gpu_fail"]) for t in best["engine_queries"] if t["gpu_fail"]})
            print(f"SF{sf} {name} {e}: {best['ms']:7.1f} ms, {len(best['engine_queries'])} engine calls {devs}"
                  + (f", GPU fallback: {fails}" if fails else ""), flush=True)
    res["scales"][f"sf{sf}"] = r
    db.close()
json.dump(res, open(out_path, "w"), indent=1, default=str)
print("written", out_path)
