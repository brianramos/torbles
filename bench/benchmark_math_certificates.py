# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-007 benchmark: proof certificates and their evidence scope.

    python bench/benchmark_math_certificates.py --capability FND-007 --out FILE [--device] [--quick]

Case 1, certificate checking separately from candidate discovery: over a table with planted affine, sum and product
relations, the native candidate discovery of DISC-001 (proposals and verification of every candidate), the
exhaustive certification of the discovered relations (snapshot certificates), and the checks of a certificate
against the current scope (fresh, authorize_exact), each timed alone; the scope construction is its own phase.
Case 2, certificate reuse, invalidation and full re-verification over growing row counts (10k / 100k / 1M rows):
reuse (the current scope rebuilt and the certificate found fresh), invalidation (after a mutation the same check
refuses: revision), full re-verification of 1 / 4 / 16 claims (CPU; CUDA with --device), and the baselines: the
complete rediscovery of DISC-001 after the mutation and a NumPy recomputation of the relations from the decoded
columns. Budget refusals of the verification are recorded as thresholds.
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import algq
import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles.math_budgets import Budget
from torbles.relationships import discover


def table(rng, rows):
    x = rng.integers(-1000, 1001, size=rows)
    y = rng.integers(-500, 2001, size=rows)
    k = rng.integers(0, 64, size=rows)
    frame = pd.DataFrame({"x": x, "y": y, "a": 3 * x - 7, "s": x + y, "p": x * y, "k": k,
                          "b1": x & 1, "b2": y & 1, "b3": (x ^ y) & 1, "id": np.arange(rows)})
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in frame.columns) + ")"
    return torbles.from_ddl(ddl, {"t": frame}, keep_plain=False)


NAMES = ["x", "y", "a", "s", "p", "k", "b1", "b2", "b3", "id"]


def claims():
    base = [mc.affine("a", "x", 3, -7), mc.sum_of("s", "x", "y"), mc.product("p", "x", "y"),
            mc.parity(["b1", "b2", "b3"], 0)]
    more = [mc.polynomial({(1, 1, 0): 1, (0, 2, 1): -1}, ["p", "x", "y"]), mc.not_null("x"), mc.unique("id"),
            mc.refinement("b1", "x", {v: v & 1 for v in range(-1000, 1001)})]
    out = base + more
    while len(out) < 16:
        out += [mc.affine("a", "x", 3, -7), mc.sum_of("s", "x", "y")]
    return out[:16]


def decoded(t, name):
    out = np.empty(t.n, dtype=np.uint32)
    algq.lib.at_decode_column(t.at, t.idx(name), out.ctypes.data)
    return out.astype(np.int64) + int(t.codecs[name].affine[0])


def checking_vs_discovery(bench, rows):
    rng = np.random.default_rng(bench.seed)
    db = table(rng, rows)
    t = db["t"]
    p = dict(rows=rows)
    bench.measure("discovery (DISC-001 proposals + verification)", "cpu",
                  lambda: _discover(db), repeats=3, phase="discovery", **p)
    scope = bench.measure("scope", "cpu", lambda: mbk.Scope.from_table(t, NAMES), repeats=3, phase="construction", **p)
    found = [mc.affine("a", "x", 3, -7), mc.sum_of("s", "x", "y"), mc.product("p", "x", "y")]
    for c in found:
        q = dict(p, claim=c.kind)
        cert = bench.measure("certification (snapshot)", "cpu", lambda: mc.snapshot(scope, c), repeats=3,
                             phase="certification", **q)
        for backend in bench.backends():
            if backend == "cuda":
                bench.measure("certification (snapshot)", "cuda", lambda: mc.snapshot(scope, c, backend="cuda"),
                              repeats=3, phase="certification", **q)
        bench.measure("check: fresh", "cpu", lambda: cert.fresh(scope).fresh, repeats=7, phase="check", **q)
        bench.measure("check: authorize_exact", "cpu", lambda: cert.authorize_exact(scope).kind, repeats=7,
                      phase="check", **q)
    h = mc.hypothesis(scope, found[0], size=1000, seed=1)
    bench.measure("check: a hypothesis is refused", "cpu", lambda: h.decide(scope)["reason"], repeats=7, phase="check", **p)


def _discover(db):
    cat = discover(db, "t", ["x", "y", "a", "s", "p"], max_candidates=100_000, attach=False)
    s = cat.stats
    cat.close()
    return dict(candidates=s["candidates"], rules=s["rules"])


def reuse_invalidate_reverify(bench, rows):
    rng = np.random.default_rng(bench.seed + rows)
    db = table(rng, rows)
    t = db["t"]
    p = dict(rows=rows)
    scope = mbk.Scope.from_table(t, NAMES)
    all_claims = claims()
    certs = [mc.snapshot(scope, c) for c in all_claims]

    def reuse():
        now = mbk.Scope.from_table(t, NAMES)
        return all(c.fresh(now).fresh for c in certs[:4])
    bench.measure("reuse (current scope + 4 checks)", "cpu", reuse, repeats=5, phase="reuse", **p)
    now = mbk.Scope.from_table(t, NAMES)
    bench.measure("reuse (4 checks, scope given)", "cpu", lambda: all(c.fresh(now).fresh for c in certs[:4]),
                  repeats=7, phase="reuse", **p)
    t.flip(0, 0); t.repair()                                       # the same data, a new revision

    def invalidation():
        current = mbk.Scope.from_table(t, NAMES)
        return ",".join(sorted({r for c in certs[:4] for r in c.fresh(current).reasons}))
    bench.measure("invalidation (current scope + 4 checks)", "cpu", invalidation, repeats=5, phase="invalidation", **p)
    current = mbk.Scope.from_table(t, NAMES)
    for n in (1, 4, 16):
        q = dict(p, claims=n)
        for backend in bench.backends():
            bench.measure("full re-verification", backend,
                          lambda: len(mc.reverify(certs[:n], current, backend=backend)), repeats=3,
                          phase="re-verification", **q)
            bench.measure("exhaustive evidence (verify)", backend,
                          lambda: sum(e["violations"] for e in mc.verify(current, all_claims[:n], backend=backend)),
                          repeats=3, phase="re-verification", **q)
    bench.measure("baseline: complete rediscovery (DISC-001)", "cpu", lambda: _discover(db), repeats=3,
                  phase="baseline", **p)

    def numpy_check():
        x, y, a, s, pr = (decoded(t, c) for c in ("x", "y", "a", "s", "p"))
        return bool(np.all(a == 3 * x - 7) and np.all(s == x + y) and np.all(pr == x * y))
    bench.measure("baseline: NumPy recomputation (decode + 3 relations)", "numpy", numpy_check, repeats=3,
                  phase="baseline", **p)
    budgeted = [mc.unique("id")] + all_claims[:3]                    # the UNIQUE claim's sort scratch is the need
    need = mc.need(current, budgeted, backend="cpu")["bytes"]
    admitted = bench.refusal("verification budget", "cpu",
                             lambda: mc.verify(current, budgeted, budget=Budget(bytes=need - 1)), **p)
    ok = mc.verify(current, budgeted, budget=Budget(bytes=need))
    bench.threshold("verification bytes (4 claims incl. UNIQUE scratch)", need, need - 1 if admitted["refused"] else None,
                    rows=rows, admitted_claims=len(ok))


def main():
    bench = Bench.from_args("FND-007", __doc__)
    grid = (10_000, 100_000) if bench.quick else (10_000, 100_000, 1_000_000)
    checking_vs_discovery(bench, 20_000 if bench.quick else 200_000)
    for rows in grid:
        reuse_invalidate_reverify(bench, rows)
    bench.note("fresh() rebuilds the address-free record of the current scope (selection words, NULL domains of the "
               "claim columns): its cost grows with rows / 64 and the NULL scan, not with the claim; a mutation is "
               "detected from the revision without reading the data again.")
    bench.note("Re-verification is exhaustive (every selected row); CUDA timings exist only with --device "
               "(the emulation is never timed as CUDA).")
    bench.write()


if __name__ == "__main__":
    main()
