# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-003 benchmark: conservation and stability of certified identities across revisions.

    python bench/benchmark_math_conserve.py --capability DISC-003 --out FILE [--device] [--quick]

Grid: 2 / 4 / 8 observed variables over 1k / 100k / 1M rows (quick: 1k / 20k) with planted affine (a = 3x - 7),
conservation (s = x + y) and product (z = x y) relations plus one-row near misses (q = x + y and r = 2y + 5 off by
one at one row). Reported separately: the proposal (DISC-002 discovery: plan, Gram, nullspace, labels), the tracker
build (its Gram and the exhaustive verification and FND-007 certificates of every identity), and per revision the
DELTA-001 diff, the incremental update (CPU; CUDA only with --device), the complete recomputation (rebind: Gram and
exhaustive verification from scratch; the baseline) and DuckDB re-counting the violating rows of the new revision.
Revisions move rows along the conserved direction (x += t, y -= t, the dependent columns restated): the identities
stay certified; a perturbation (one row of a dependent column off by one) fails its identity with that row as the
witness. Revision stability: a lineage of conserving and perturbing revisions, the per-identity records. Degree
budgets: the Gram / nullspace state per basis degree, a bytes refusal and the 63-monomial representation refusal
(recorded, never extrapolated). Conservation laws: symbolic certification (universal) and the Nullstellensatz
counterexample, then the data check (snapshot certificate or witness rows).
"""
import math

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk
from torbles import math_conserve as mcons
from torbles import math_delta as md
from torbles import math_ident as mi
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError


def data(rng, k, rows):
    x = rng.integers(-1000, 1001, size=rows)
    y = rng.integers(-500, 2001, size=rows)
    cols = {"x": x, "a": 3 * x - 7}
    if k >= 4:
        q = x + y
        q[rows // 2] += 1                                   # a one-row near miss of a conservation law
        cols.update(y=y, q=q)
    if k >= 8:
        r = 2 * y + 5
        r[rows // 3] -= 1                                   # a one-row near miss of an affine law
        cols.update(s=x + y, z=x * y, r=r, w=rng.integers(-50, 51, size=rows))
    names = list(cols)[:k]
    return {c: np.array(cols[c], dtype=np.int64) for c in names}, names


def restate(cols, rows):
    """the dependent columns recomputed on the given rows (the near misses elsewhere untouched)"""
    x, y = cols["x"], cols.get("y")
    cols["a"][rows] = 3 * x[rows] - 7
    if "s" in cols:
        cols["s"][rows] = x[rows] + y[rows]
        cols["z"][rows] = x[rows] * y[rows]


def conserving(rng, cols, m):
    """a revision moving m rows along x += t, y -= t (s = x + y and q = x + y keep their values)"""
    new = {c: v.copy() for c, v in cols.items()}
    rows = rng.choice(len(new["x"]), size=m, replace=False)
    t = rng.integers(-5, 6, size=m)
    new["x"][rows] += t
    if "y" in new:
        new["y"][rows] -= t
    restate(new, rows)
    return new


def perturbing(rng, cols):
    new = {c: v.copy() for c, v in cols.items()}
    row = int(rng.integers(0, len(new["x"])))
    new["s" if "s" in new else "a"][row] += 1
    return new, row


def scope_of(cols):
    return mbk.Scope.from_arrays(cols)


def duck_violations(cols, identities):
    """DuckDB: the rows of a revision violating the tracked identities (a full rescan)"""
    frame = pd.DataFrame(cols)
    con = duckdb.connect()
    con.register("t", frame)
    total = 0
    for it in identities:
        expr = " + ".join(f"({c})" + "".join(f"*{n}" * k for n, k in zip(it["columns"], e) if k)
                          for e, c in it["terms"].items())
        total += con.execute(f"SELECT count(*) FROM t WHERE {expr} <> 0").fetchone()[0]
    con.close()
    return total


def grid(bench):
    rng = np.random.default_rng(bench.seed)
    for rows in ((1000, 20_000) if bench.quick else (1000, 100_000, 1_000_000)):
        for k in (2, 4, 8):
            cols, names = data(rng, k, rows)
            s0 = scope_of(cols)
            p = dict(variables=k, rows=rows, degree=1)
            disc = bench.measure("proposal (DISC-002 discovery)", "cpu",
                                 lambda: mi.discover(s0, names, degree=1, certify=False), repeats=1 if rows > 100_000 else 3,
                                 phase="first", **p)
            ids = disc.identities
            tr = bench.measure("tracker build (Gram + exhaustive verification + certificates)", "cpu",
                               lambda: mcons.track(s0, names, degree=1, identities=ids), repeats=1 if rows > 100_000 else 3,
                               phase="build", identities=len(ids), **p)
            info = tr.info
            bench.note(f"k={k} rows={rows}: {info['identities']} identities tracked "
                       f"({'; '.join(tr.identity(i)['text'] for i in range(len(tr)))}); certified {info['certified']}")
            m = max(1, rows // 1000)
            new = conserving(rng, cols, m)
            s1 = scope_of(new)
            d = bench.measure("delta (DELTA-001 diff of two revisions)", "cpu", lambda: md.Delta.between(s0, s1),
                              repeats=3, phase="build", changed_rows=m, **p)
            for backend in bench.backends():
                up = bench.measure("incremental update (conserving revision)", backend,
                                   lambda: tr.update(d, scope=s1, backend=backend), repeats=3, phase="run",
                                   delta_rows=int(d.info["rows"]), **p)
            up = tr.update(d, scope=s1)
            bench.measure("complete recomputation (rebind: Gram + exhaustive verification)", "cpu",
                          lambda: tr.rebind(s1), repeats=1 if rows > 100_000 else 3, phase="run", **p)
            tracked = [tr.identity(i) for i in range(len(tr))]
            bench.measure("full rescan of the identities", "duckdb", lambda: duck_violations(new, tracked),
                          repeats=1 if rows > 100_000 else 3, phase="run", **p)
            self_check = up.info
            bench.note(f"k={k} rows={rows}: after the conserving revision certified {self_check['certified']}, "
                       f"failed {self_check['failed']}")
            bad, row = perturbing(rng, new)
            s2 = scope_of(bad)
            d2 = md.Delta.between(s1, s2)

            def perturbed():
                t2 = up.update(d2, scope=s2)
                fails = [t2.record(i) for i in range(len(t2)) if t2.record(i)["status"] == "failed"]
                where = fails[0]["first_failure"] if fails else None
                return dict(failed=len(fails), witness_source=where["source"] if where else None,
                            located=t2.locate(next(i for i in range(len(t2)) if t2.record(i)["status"] == "failed"),
                                              s2)["witness"] if fails else None, planted_row=row)
            bench.measure("perturbation (one-row near miss: witness)", "cpu", perturbed, repeats=1, phase="run", **p)
            for o in (disc.plan, disc.gram, disc.kernel, ids):
                o.close()


def stability(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 5000 if bench.quick else 50_000
    cols, names = data(rng, 8, rows)
    s = scope_of(cols)
    ids = mi.discover(s, names, degree=1, certify=False).identities
    tr = mcons.track(s, names, degree=1, identities=ids)
    plan = ["conserve", "conserve", "perturb", "conserve", "conserve", "conserve", "perturb", "conserve"]
    for step, kind in enumerate(plan[:5] if bench.quick else plan, 1):
        new = conserving(rng, cols, 25) if kind == "conserve" else perturbing(rng, cols)[0]
        s2 = scope_of(new)
        tr = bench.measure("revision lineage step", "cpu", lambda: tr.update(md.Delta.between(s, s2), scope=s2),
                           repeats=1, phase="run", step=step, kind=kind, rows=rows)
        cols, s = new, s2
    for i in range(len(tr)):
        r = tr.record(i)
        bench.note(f"stability of {tr.identity(i)['text']}: status {r['status']}, certified at {r['certified']} steps, "
                   f"runs {r['runs']}, streak {r['streak']}, first failure "
                   f"{None if r['first_failure'] is None else (r['first_failure']['step'], r['first_failure']['source'])}")


def degrees(bench):
    rng = np.random.default_rng(bench.seed + 2)
    rows = 5000 if bench.quick else 20_000
    cols, names = data(rng, 4, rows)
    s = scope_of(cols)
    admitted = None
    for d in (1, 2, 3, 4):
        basis = math.comb(4 + d, 4)
        try:
            tr = bench.measure("tracker state per basis degree", "cpu", lambda: mcons.track(s, names, degree=d),
                               repeats=1, phase="build", degree=d, basis=basis, rows=rows)
        except UnsupportedError as e:
            bench.threshold("basis degree (representation: 63 monomials)", admitted=admitted,
                            refused=dict(degree=d, basis=basis), reason=str(e))
            break
        i = tr.info
        admitted = dict(degree=d, basis=i["basis"], moments=i["moments"], bytes=i["bytes"])
        bench.note(f"degree {d}: basis {i['basis']}, Gram moments {i['moments']}, tracker bytes {i['bytes']}")
    tr = mcons.track(s, names, degree=2)
    need = tr.info["bytes"]
    r = bench.refusal("tracker build under a bytes budget", "cpu",
                      lambda: mcons.track(s, names, degree=2, budget=Budget(bytes=need // 4)), degree=2,
                      limit=need // 4, rows=rows)
    bench.threshold("tracker bytes (degree 2, 4 variables)", admitted=dict(bytes_used=need),
                    refused=dict(limit=need // 4, need=r.get("need"), resource=r.get("resource")))
    pcols = {"x": cols["x"], "y": cols["y"], "z": cols["x"] * cols["y"]}
    ps = scope_of(pcols)
    prod = mi.candidates(ps, ["x", "y", "z"], [{(1, 1, 0): 1, (0, 0, 1): -1}])
    t = bench.measure("product relation tracked at degree 2", "cpu",
                      lambda: mcons.track(ps, ["x", "y", "z"], degree=2, identities=prod), repeats=1, phase="build",
                      rows=rows)
    bench.note(f"z = x y at degree 2: {t.record(0)['status']}")


def laws(bench):
    rng = np.random.default_rng(bench.seed + 3)
    q = {(1, 0, 0): 1, (0, 1, 0): 1}
    T = {"x": {(1, 0, 0): 1, (0, 0, 1): 1}, "y": {(0, 1, 0): 1, (0, 0, 1): -1}}
    law = bench.measure("conservation law (symbolic: q(T(x)) - q(x) = 0)", "cpu",
                        lambda: mcons.conserve(q, T, ["x", "y", "t"]), repeats=5, phase="first")
    bench.note(f"x' = x + t, y' = y - t conserves x + y: {law.conserved} ({law.certificate.kind} certificate)")
    q2 = {(2, 0, 0): 1, (0, 2, 0): 1}
    T2 = {"x": {(1, 0, 0): 1, (0, 0, 1): 1}, "y": {(0, 1, 0): 1}}
    bad = bench.measure("non-conserved transformation (Nullstellensatz counterexample)", "cpu",
                        lambda: mcons.conserve(q2, T2, ["x", "y", "t"]), repeats=5, phase="first")
    bench.note(f"x' = x + t does not conserve x^2 + y^2: counterexample {bad.counterexample}")
    rows = 20_000 if bench.quick else 200_000
    x = rng.integers(-1000, 1001, size=rows)
    y = rng.integers(-1000, 1001, size=rows)
    t = rng.integers(-9, 10, size=rows)
    s = scope_of({"x": x, "y": y, "t": t})
    for backend in bench.backends():
        bench.measure("conservation law on the data (snapshot certificate)", backend,
                      lambda: law.check(s, {"x": "x", "y": "y", "t": "t"}, backend=backend), repeats=3, phase="run",
                      rows=rows)
    bench.measure("non-conserved law on the data (witness rows)", "cpu",
                  lambda: bad.check(s, {"x": "x", "y": "y", "t": "t"}), repeats=3, phase="run", rows=rows)


def main():
    bench = Bench.from_args("DISC-003", __doc__)
    grid(bench)
    stability(bench)
    degrees(bench)
    laws(bench)
    bench.note("Timings: first = the first execution, warm = the median of the others; the CUDA path is timed only "
               "with --device on an actual device. DuckDB and rebind are full rescans of the new revision.")
    bench.write()


if __name__ == "__main__":
    main()
