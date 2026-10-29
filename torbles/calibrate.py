# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Measure the join-polynomial cost model (algq.COST) on this machine, one thread, ns per 64-row word:
  lit         one plane literal of a cube or comparator
  instr       one instruction of the filter program
  lookup_row  one live row checked against a dimension bitmap
The selection is a union of K aligned id blocks of size 2^k, evaluated as K cubes of (b - k) literals (expanded),
and the same selection by lookup. Fitted by least squares over several (K, k)."""
import time, numpy as np
from algq import Table, Query, set_threads, COST

N, b = 8_000_000, 17
rng = np.random.default_rng(5)
set_threads(1)
def best(fn, reps=5):
    t = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t = min(t, time.perf_counter() - t0)
    return t
words = (N + 63) // 64
rows = []
fk = rng.integers(0, 1 << b, N).astype(np.uint32)
for K, k in ((4, 12), (16, 10), (16, 4), (64, 8), (64, 2), (128, 6), (256, 3)):
    starts = rng.choice((1 << b) >> k, K, replace=False) << k
    g = np.zeros(1 << b, np.uint32)
    for s in starts: g[s:s + (1 << k)] = 1
    dim = Table("dim", dict(g=g))
    fact = Table("fact", dict(fk=fk), msb=("fk",), refs=dict(fk=dim))
    base = Query(fact).join(dim, fact.fk).count()
    t0 = best(lambda: base.run("alg"))
    qe = Query(fact, "expanded").join(dim, fact.fk).where(dim.g == 1).count()
    ql = Query(fact, "lookup").join(dim, fact.fk).where(dim.g == 1).count()
    te, tl = best(lambda: qe.run("alg")), best(lambda: ql.run("alg"))
    assert qe.run("alg") == ql.run("alg") == int(g[fk].sum())
    nterm = qe.range_used[0][2]
    rows.append((K, k, nterm, (te - t0) / words * 1e9, (tl - t0) / words * 1e9))
    print(f"K={K:4d} blocks of 2^{k:<2d}: cubes {nterm:4d} x {b - k:2d} literals  expanded {rows[-1][3]:7.1f} ns/word   lookup {rows[-1][4]:6.1f} ns/word")
# expanded time = nterm * (L * lit + 2 * instr)  (one cube instruction + one OR each)
A = np.array([[r[2] * (b - r[1]), 2 * r[2]] for r in rows]); y = np.array([r[3] for r in rows])
lit, instr = np.linalg.lstsq(A, y, rcond=None)[0]
lookup_row = float(np.median([r[4] for r in rows])) / 64
print(f"\nfitted: lit={lit:.2f} instr={instr:.2f} lookup_row={lookup_row:.2f}   (algq.COST now {COST})")
