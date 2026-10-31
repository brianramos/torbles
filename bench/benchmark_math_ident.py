# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-002 benchmark: Moment-Gram polynomial identity discovery, its exhaustive verification and certificates.

    python bench/benchmark_math_ident.py --capability DISC-002 --out FILE [--device] [--quick]

Grid: 2 / 4 / 8 observed variables over 1k / 100k / 1M rows (quick: 1k / 20k) with planted affine (a = 3x - 7),
conservation (s = x + y) and product (z = x y) relations plus one-row near misses (q = x + y and r = 2y + 5 off by
one at one row). Reported separately: the plan, the exact Gram state (fused CPU; CUDA only with --device; the NumPy
binary64 X^T X as a baseline, which is approximate), the exact nullspace (LIN-002), the proposal (identities and
their exact labels), the exhaustive verification (CPU; CUDA with --device; DuckDB counting the violating rows as a
baseline) and the certificates. The near misses: never in the exact nullspace, and rejected by the exhaustive
verification with the smallest violating row. Then the basis degree budget (state growth of the Gram and the
nullspace per degree, the refusal of a states budget and of the 63-monomial representation, never extrapolated),
revision stability (a mutation makes the Gram stale; rediscovery) and the number-theory relation miner (n = 1..10^5:
omega, Omega, I = 2^omega, mu^2, d by a sieve; the gated identities found; on n <= 29 the interpolations through
three points labelled underdetermined, never discoveries).
"""
import itertools
import math

import numpy as np
import pandas as pd

from math_bench import Bench
import duckdb
import torbles
from torbles import math_bank as mbk
from torbles import math_ident as mi
from torbles.math_budgets import Budget
from torbles.math_errors import StaleStateError, UnsupportedError


def table(rng, k, rows):
    x = rng.integers(-1000, 1001, size=rows)
    y = rng.integers(-500, 2001, size=rows)
    cols = {"x": x, "a": 3 * x - 7}
    if k >= 4:
        q = x + y
        q[rows // 2] += 1                                       # a one-row near miss of a conservation law
        cols.update(y=y, q=q)
    if k >= 8:
        r = 2 * y + 5
        r[rows // 3] -= 1                                       # a one-row near miss of an affine law
        cols.update(s=x + y, z=x * y, r=r, w=rng.integers(-50, 51, size=rows))
    names = list(cols)[:k]
    frame = pd.DataFrame({c: cols[c] for c in names})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in names) + ")"
    db = torbles.from_ddl(ddl, {"t": frame}, keep_plain=False)
    return db, names, frame


NEAR = {"q": ([("q", 1), ("x", 1), ("y", 1)], {(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}),
        "r": ([("r", 1), ("y", 1)], {(1, 0): 1, (0, 1): -2, (0, 0): -5})}
DUCK = {2: "a <> 3 * x - 7", 4: "a <> 3 * x - 7", 8: "a <> 3 * x - 7 OR s <> x + y OR z <> x * y"}


def duck(frame, where):
    """DuckDB: the rows violating the planted identities"""
    con = duckdb.connect()
    con.register("frame", frame)
    return con.execute(f"SELECT count(*) FROM frame WHERE {where}").fetchone()[0]


def numpy_gram(frame, names, degree, chunk=100_000):
    """the binary64 X^T X of the monomials (a baseline: approximate past 2^53)"""
    basis = [e for e in itertools.product(range(degree + 1), repeat=len(names)) if sum(e) <= degree]
    G = np.zeros((len(basis), len(basis)))
    data = frame[names].to_numpy(dtype=np.float64)
    for s in range(0, len(data), chunk):
        block = data[s:s + chunk]
        X = np.stack([np.prod(block ** np.array(e, dtype=np.float64), axis=1) for e in basis], axis=1)
        G += X.T @ X
    return dict(basis=len(basis), trace=float(np.trace(G)))


def grid(bench):
    rng = np.random.default_rng(bench.seed)
    for rows in ((1000, 20_000) if bench.quick else (1000, 100_000, 1_000_000)):
        for k in (2, 4, 8):
            db, names, frame = table(rng, k, rows)
            scope = mbk.Scope.from_table(db["t"], names)
            p = dict(variables=k, rows=rows, degree=2)
            plan = bench.measure("plan", "cpu", lambda: mi.plan(scope, names, degree=2), repeats=3, phase="build", **p)
            info = plan.info
            for backend in bench.backends():
                gram = bench.measure("gram", backend, lambda: mi.gram(scope, plan, backend=backend), repeats=3,
                                     phase="build", basis=info["basis"], moments=info["moments"], repr=info["repr"], **p)
            gram = mi.gram(scope, plan)
            bench.measure("gram", "numpy", lambda: numpy_gram(frame, names, 2), repeats=1 if rows > 100_000 else 3,
                          phase="build", note="binary64, approximate", **p)
            kern = bench.measure("nullspace", "cpu", lambda: gram.kernel(), repeats=3, phase="first",
                                 **dict(p, basis=info["basis"]))
            ki = kern.info
            ids = bench.measure("proposal", "cpu", lambda: mi.identities(scope, gram, kern), repeats=3, phase="first",
                                nullity=ki["nullity"], **p)
            for backend in bench.backends():
                bench.measure("exhaustive verification", backend,
                              lambda: sum(not e["holds"] for e in ids.verify(backend=backend)), repeats=3,
                              phase="warm", candidates=len(ids), **p)
            bench.measure("exhaustive verification", "duckdb", lambda: duck(frame, DUCK[k]), repeats=3, phase="warm",
                          note="the planted identities only", **p)

            def certify():
                ids.certify()
                found = [i for i in ids if i["discovery"]]
                return dict(identities=len(ids), discoveries=len(found), texts="; ".join(i["text"] for i in found),
                            consequences=sum(i["consequence"] for i in ids),
                            gram_bytes=gram.info["bytes"], nullspace_bytes=ki["bytes"])
            bench.measure("certify (verification + FND-007 snapshots)", "cpu", certify, repeats=1, phase="first", **p)
            for col, (vars_, poly) in NEAR.items():
                if col not in names or any(v not in names for v, _ in vars_):
                    continue
                cv = [v for v, _ in vars_]

                def near():
                    g1 = mi.gram(scope, mi.plan(scope, cv, degree=1))       # sos "nonzero": c^T G c != 0 exactly
                    it = mi.candidates(scope, cv, [poly], gram=g1).certify()[0]
                    return dict(status=it["status"], violations=it["evidence"]["violations"],
                                witness=it["evidence"]["witness"], sos=it["sos"])
                bench.measure("one-row near miss", "cpu", near, repeats=3, phase="warm", relation=col, **p)
            for c in (plan, gram, kern, ids):
                c.close()


def degrees(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 5000 if bench.quick else 20_000
    db, names, frame = table(rng, 4, rows)
    scope = mbk.Scope.from_table(db["t"], names)
    admitted = None
    for d in (1, 2, 3, 4):
        try:
            plan = mi.plan(scope, names, degree=d)
        except UnsupportedError as e:
            bench.threshold("basis degree (representation: 63 monomials)", admitted=admitted,
                            refused=dict(degree=d, basis=math.comb(4 + d, 4)),
                            reason=str(e))
            break
        need = plan.need()
        g = bench.measure("gram state growth", "cpu", lambda: mi.gram(scope, plan), repeats=3, phase="build",
                          rows=rows, variables=4, degree=d, basis=plan.info["basis"], moments=plan.info["moments"],
                          need_bytes=need.get("bytes"), need_states=need.get("states"))
        k = bench.measure("nullspace state growth", "cpu", lambda: g.kernel(), repeats=3, phase="first", rows=rows,
                          variables=4, degree=d, gram_bytes=g.info["bytes"])
        bench.note(f"degree {d}: basis {plan.info['basis']}, moments {plan.info['moments']}, Gram {g.info['bytes']} "
                   f"bytes, nullspace {k.info['bytes']} bytes, rank {k.rank}, nullity {k.nullity}")
        admitted = dict(degree=d, basis=plan.info["basis"])
    # a states budget of the degree-2 moments admits degree 2 and refuses degree 3 (no extrapolation)
    lim = mi.plan(scope, names, degree=2).info["moments"]
    b = Budget(states=lim)
    bench.measure("gram under a states budget", "cpu", lambda: mi.gram(scope, mi.plan(scope, names, degree=2), budget=b),
                  repeats=1, phase="build", states=lim, degree=2)
    rec = bench.refusal("gram under a states budget", "cpu", lambda: mi.gram(scope, mi.plan(scope, names, degree=3), budget=b),
                        states=lim, degree=3)
    bench.threshold("gram states budget", admitted=dict(degree=2, states=lim),
                    refused=dict(degree=3, need=rec.get("need"), resource=rec.get("resource")))


def revisions(bench):
    rng = np.random.default_rng(bench.seed + 2)
    rows = 20_000 if bench.quick else 100_000
    db, names, frame = table(rng, 8, rows)
    t = db["t"]
    scope = mbk.Scope.from_table(t, names)
    found = mi.discover(scope, names, degree=2)
    before = sorted(i["text"] for i in found.discoveries())
    t.flip(0, 0)                                                # a mutation: every state of the scope is stale

    def stale():
        try:
            mi.gram(scope, found.plan)
        except StaleStateError:
            return "refused: stale"
        return "not refused"
    bench.measure("stale Gram", "cpu", stale, repeats=3, phase="warm", rows=rows, variables=8)

    def rediscover():
        now = mbk.Scope.from_table(t, names)
        after = sorted(i["text"] for i in mi.discover(now, names, degree=2).discoveries())
        return dict(before="; ".join(before), after="; ".join(after), stable=before == after)
    bench.measure("rediscovery after a mutation", "cpu", rediscover, repeats=1, phase="first", rows=rows, variables=8)


def sieve(N):
    spf = list(range(N + 1))
    for p in range(2, int(N ** 0.5) + 1):
        if spf[p] == p:
            for q in range(p * p, N + 1, p):
                if spf[q] == q:
                    spf[q] = p
    om, Om, d, sq = [0] * (N + 1), [0] * (N + 1), [1] * (N + 1), [1] * (N + 1)
    for n in range(2, N + 1):
        p, m, e = spf[n], n // spf[n], 1
        while m % p == 0:
            m //= p
            e += 1
        om[n], Om[n], d[n], sq[n] = om[m] + 1, Om[m] + e, d[m] * (e + 1), sq[m] * (1 if e == 1 else 0)
    return om, Om, d, sq


def number_theory(bench):
    N = 10_000 if bench.quick else 100_000
    om, Om, d, sq = sieve(N)
    frame = pd.DataFrame({"n": np.arange(1, N + 1), "om": om[1:], "big": Om[1:],
                          "i": [2 ** t for t in om[1:]], "mu2": sq[1:], "d": d[1:]})
    db = torbles.from_ddl("CREATE TABLE t (n BIGINT, om BIGINT, big BIGINT, i BIGINT, mu2 BIGINT, d BIGINT)",
                          {"t": frame}, keep_plain=False)
    names = list(frame)
    scope = mbk.Scope.from_table(db["t"], names)

    def mine():
        found = mi.discover(scope, ["om", "big", "i", "mu2", "d"], degree=2)
        disc = sorted(i["text"] for i in found.discoveries())
        return dict(discoveries="; ".join(disc), identities=len(found.identities),
                    consequences=sum(i["consequence"] for i in found.identities), points=found.report["points"])
    bench.measure("relation miner (omega, Omega, 2^omega, mu^2, d)", "cpu", mine, repeats=1, phase="first", n=N, degree=2)
    small = mbk.Scope.from_table(db["t"], names, rows=np.arange(1, N + 1) <= 29)

    def artifacts():
        art = mi.discover(small, ["om", "i"], degree=2)
        return dict(points=art.report["points"], identities=len(art.identities),
                    underdetermined=sum(i["underdetermined"] for i in art.identities),
                    discoveries=len(art.discoveries()), texts="; ".join(i["text"] for i in art.identities))
    bench.measure("three-point interpolation artifacts (n <= 29)", "cpu", artifacts, repeats=3, phase="first", n=29)


def main():
    bench = Bench.from_args("DISC-002", __doc__)
    grid(bench)
    degrees(bench)
    revisions(bench)
    number_theory(bench)
    bench.note("Gram: exact 128-bit moments when the plan proves N W prod M^e <= 2^127 - 1, else residues modulo the "
               "recorded LIN-002 primes (repr 'modular'); the nullspace is LIN-002's verified rational one (exact) or "
               "the CRT + rational reconstruction of agreeing primes (modular). numpy is binary64 (approximate).")
    bench.note("CUDA (tr_mgpu_ident_gram, tr_mgpu_ident_verify) is timed only with --device; emulated runs are "
               "checked for equality in tests/test_math_ident.py, never timed as CUDA.")
    bench.write()


if __name__ == "__main__":
    main()
