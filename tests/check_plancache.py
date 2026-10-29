# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Compiled-plan reuse: a query run again reuses its plan; the results must equal a fresh plan's, including after the
query is changed (new parts), the join form or GROUP BY mode is switched, or the GPU executor runs it."""
import sys, algq
from algq import Query, set_threads
from star import generate, build, queries
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
T, _ = build(generate(N)); lo, d, c = T["lineorder"], T["dates"], T["customer"]
fails = 0
def fresh(q, eng):
    algq.PLAN_CACHE = False
    try: return q.run(eng)
    finally: algq.PLAN_CACHE = True
for name, q in queries(T).items():
    for eng in ("alg", "plain", "gpu"):
        for mode in ("auto", "factored", "lookup"):
            q.join_mode = mode
            for gb in ("mm", "split"):
                algq.set_groupby(gb)
                a = q.run(eng); b = q.run(eng); hit = q.plan_cached; want = fresh(q, eng)
                if not (a == b == want) or not hit: fails += 1; print("  FAIL", name, eng, mode, gb, "cached" if hit else "not cached")
    q.join_mode = "auto"; algq.set_groupby("mm")
# a query changed after it ran must be planned again
q = Query(lo).join(c, lo.cust).where(c.region == 1).group_by(c.nation).sum(lo.price)
r1 = q.run("alg"); q.where(lo.quantity < 10); r2 = q.run("alg")
if q.plan_cached or r2 != fresh(q, "alg") or r1 == r2: fails += 1; print("  FAIL changed query")
q.group_by(c.region); r3 = q.run("alg")
if q.plan_cached or r3 != fresh(q, "alg"): fails += 1; print("  FAIL changed keys")
for th in (1, 2):
    set_threads(th); x = q.run("alg")
    if x != r3: fails += 1; print("  FAIL threads", th)
print("ALL MATCH" if not fails else f"{fails} FAILURES")
