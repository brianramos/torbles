# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GPU executor vs CPU engine: every query shape, every join form, results must be identical. Runs on a real GPU
(libgpu.so) or on the CPU emulation of the same kernels (libgpu_emu.so). Prints which plans ran on the GPU and which
fell back to the CPU (lookups and row ranges stay on the CPU)."""
import sys, numpy as np
from algq import Table, Query, Surrogate, set_threads, gpu_name
from star import generate, build, queries
from star8 import make_data, D

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
print("GPU executor:", gpu_name())
fails, ran = 0, {}
def check(name, q, modes=("auto", "factored", "expanded")):
    global fails
    for mode in modes:
        q.join_mode = mode
        want = q.run("alg"); got = q.run("gpu")
        ran.setdefault(q.device, 0); ran[q.device] += 1
        if got != want: fails += 1; print(f"  MISMATCH {name} [{mode}] on {q.device}: {str(got)[:120]} != {str(want)[:120]}")
    q.join_mode = "auto"; q.run("gpu")
    print(f"  {name:42s} {q.device}")

print("star schema (hierarchical ids, time-ordered facts)")
T, _ = build(generate(N)); lo, d, c, s, p = (T[k] for k in ("lineorder", "dates", "customer", "supplier", "part"))
for name, q in queries(T).items(): check(name, q)
rev = lo.price * (100 - lo.discount)
check("count, fact filter only", Query(lo).where(lo.quantity.between(10, 20)).count())
check("linear measure", Query(lo).where(lo.discount < 3).sum(lo.price + 2 * lo.quantity))
check("degree-2 measure, total", Query(lo).where(lo.tax == 2).sum(rev))
check("group by fact column", Query(lo).where(lo.quantity < 10).group_by(lo.discount).sum(rev))
check("group by clustering key", Query(lo).join(d, lo.date).where(d.year == 3).group_by(lo.date).sum(rev))
check("group by coordinate + fact column", Query(lo).join(c, lo.cust).where(c.region == 1).group_by(c.nation, lo.discount).sum(lo.price))
check("count with keys", Query(lo).join(s, lo.supp).group_by(s.region).count())
check("empty result", Query(lo).where(lo.quantity > 99).sum(rev))

print("order-encoded attribute (bounds) and unordered attribute")
dims, used, fact, S = make_data(N, "bounds")
T8 = {k: Table(k, v, used=used[k]) for k, v in dims.items()}
F = Table("fact", fact, msb=tuple(f"fk{i}" for i in range(D)), cluster="fk0", refs={f"fk{i}": T8[f"dim{i}"] for i in range(D)})
q = Query(F)
for i in range(4): q.join(T8[f"dim{i}"], getattr(F, f"fk{i}"))
check("4 dims, rnd < 7 bounds", q.where((T8["dim1"].rnd < 7) & (T8["dim2"].rnd.between(2, 8)) & (T8["dim3"].tag < 5)).group_by(T8["dim0"].grp).sum(F.price * (100 - F.disc)))
q2 = Query(F).join(T8["dim1"], F.fk1).where(T8["dim1"].rnd2 < 7).sum(F.price)
check("unordered attribute (lookup)", q2)

print("derived surrogates")
rng = np.random.default_rng(3); n = 40_000; k = np.arange(1, n + 1)
region = rng.integers(0, 5, n); city = region * 10 + rng.integers(0, 10, n)
cols = {"region": np.zeros(n + 1, np.uint32), "city": np.zeros(n + 1, np.uint32), "key": np.zeros(n + 1, np.uint32)}
cols["region"][k], cols["city"][k], cols["key"][k] = region, city, k
u = np.zeros(n + 1, bool); u[k] = True; dA = Table("dima", cols, used=u)
w = int(n).bit_length(); s_of_row = np.zeros(n + 1, np.int64); s_of_row[k] = (((region << 4) | (city % 10)).astype(np.int64) << w) | k
sg = Surrogate(w, s_of_row)
FA = Table("fa", dict(fk=sg.s_of_row[k[rng.integers(0, n, N)]].astype(np.uint32), m=rng.integers(0, 1000, N).astype(np.uint32)), msb=("fk",), refs=dict(fk=(dA, sg)))
check("surrogate: region filter, group by city", Query(FA).join(dA, FA.fk).where(dA.region == 2).group_by(dA.city).sum(FA.m))
check("surrogate: official key range", Query(FA).join(dA, FA.fk).where(dA.key.between(1000, 9000)).group_by(dA.region).sum(FA.m))

print("range joins (dominance slices on x's planes, and groups)")
tiers = Table("tiers", dict(lo=np.array([0, 20000, 50000, 80000], np.uint32), hi=np.array([19999, 49999, 79999, 100000], np.uint32), t=np.arange(4, dtype=np.uint32)))
check("price tiers (slices)", Query(lo).join_range(tiers, lo.price, tiers.lo, tiers.hi).group_by(tiers.t).sum(rev))
check("date intervals (groups)", Query(lo).join_range(tiers, lo.date, tiers.lo, tiers.hi).group_by(tiers.t).count())

print(f"\nplans run: {ran}")
print("ALL MATCH" if not fails else f"{fails} MISMATCHES")
