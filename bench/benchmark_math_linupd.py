# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LIN-003 benchmark: leave-group-out and low-rank model updates.

    python bench/benchmark_math_linupd.py --capability LIN-003 --out FILE [--device] [--quick]

Case 1 (feature dimensions 2 / 4 / 8 / 16, groups 1 / 16 / 256, 1k / 100k / 1M rows; assembly, bytes and solve
reported separately): the scope and the LIN-001 moment state are built ("build"), then the leave-one-group-out plan
(the copied group states, their total and the exact integer complement systems: "assembly", with its bytes and the
device bytes), then every group's model solved exactly on the CPU (LIN-002's Bareiss), and by the modular path: the
residues of every (group, prime) (the kernel's work: its CPU reference here, the device with --device) and the
native finish (CRT, reconstruction, exact verification) timed apart. Baseline "reconstruction": the complete
reconstruction of a group's model from the remaining rows (a scope without the group, LIN-001 again, LIN-002's fit),
measured per group for the first groups only (the time is not extrapolated to all groups).
Case 2 (well-conditioned, nearly singular and exactly rank-deficient matrices; repeated model / ridge / deletion
requests against complete reconstruction): a 4-feature model with its inverse; 20 ridge requests (the Woodbury
rank-4 update from the previous model, the rebuild from A0, LIN-002's fit of the moments); 20 sequential single-row
deletions (Woodbury chain against the complete reconstruction from the remaining rows at each step, and their final
equality); the floating variant (certified bound or refusal, with its condition estimate) against the exact solve.
Case 3 (the cost of an exact update against rebuilds, feature dimensions 2 / 4 / 8 / 16 (2 / 4 with --quick), one
group, 20,000 rows (5,000 with --quick)): one row deleted from a model with its inverse by the Woodbury downdate,
against the exact rebuild of the updated normal equations A - z z^T, b - z y with its inverse (tr_linupd_system) and
against the complete reconstruction from the remaining rows (scope, LIN-001, fit with its inverse); all three equal.
"""
import time
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_lin as lin, math_linupd as lu
from torbles.math_errors import NotConvergedError, SingularError


def data(rng, rows, d, G, kind="random"):
    cols = {f"x{j}": rng.integers(-1000, 1001, rows) for j in range(d)}
    if kind == "nearly" and d >= 3:
        cols["x2"] = 1000 * (cols["x0"] + cols["x1"]) + rng.integers(-1, 2, rows)
    elif kind == "deficient" and d >= 3:
        cols["x2"] = cols["x0"] + cols["x1"]
    cols["y"] = sum((j + 1) * cols[f"x{j}"] for j in range(d)) + rng.integers(-500, 501, rows)
    g = rng.integers(0, G, rows) if G > 1 else None
    return cols, g


def case_grid(bench):
    rng = np.random.default_rng(bench.seed)
    dims = (2, 4) if bench.quick else (2, 4, 8, 16)
    groups = (1, 16) if bench.quick else (1, 16, 256)
    sizes = (1_000, 10_000) if bench.quick else (1_000, 100_000, 1_000_000)
    for d in dims:
        names = [f"x{j}" for j in range(d)]
        for rows in sizes:
            for G in groups:
                cols, g = data(rng, rows, d, G)
                p = dict(features=d, groups=G, rows=rows)
                scope = bench.measure("scope (FND-004 arrays)", "cpu", lambda: mbk.Scope.from_arrays(
                    cols, groups=g, ngroups=G if G > 1 else None), repeats=1, phase="build", **p)
                st = bench.measure("moment state (LIN-001)", "cpu", lambda: lin.moments(scope, names, "y"),
                                   repeats=2, phase="build", **p)
                bench.records[-1]["result"] = dict(state_bytes=st.bytes, slots=st.nslots)
                heavy = d >= 16 and G >= 256
                plan = bench.measure("leave-one-group-out plan (assembly: total, complements, integer systems)",
                                     "cpu", lambda: lu.logo_plan(st), repeats=2, phase="build", **p)
                k = plan.primes_needed
                bench.records[-1]["result"] = dict(systems=plan.systems, unknowns=plan.n, primes=k,
                                                   device_bytes=plan.device_bytes(k))
                reps = 1 if heavy else 2
                models = bench.measure("LOGO solve (CPU exact, every group)", "cpu", lambda: plan.solve("cpu"),
                                       repeats=reps, **p)
                bench.records[-1]["result"] = dict(kinds=",".join(sorted({m.kind for m in models})))
                res, fields = bench.measure("LOGO residues (the kernel's work, CPU reference)", "cpu",
                                            lambda: plan.residues(k, "cpu"), repeats=reps, primes=k, **p)
                out = bench.measure("LOGO finish (CRT, reconstruction, exact verification)", "cpu",
                                    lambda: plan.finish(k, res, fields)[0], repeats=reps, primes=k, **p)
                bench.records[-1]["result"] = dict(equal_cpu=all(a.same(b) for a, b in zip(models, out)))
                if "cuda" in bench.backends():
                    bench.measure("LOGO residues on the device", "cuda", lambda: plan.residues(k, "cuda"),
                                  repeats=reps, primes=k, **p)
                    bench.measure("LOGO solve on the device (residues + finish)", "cuda",
                                  lambda: plan.solve("cuda"), repeats=reps, **p)
                for grp in range(min(G, 2 if heavy or rows >= 1_000_000 else 3)):
                    def rebuild():
                        keep = (g != grp) if g is not None else np.zeros(rows, bool)
                        sc = mbk.Scope.from_arrays(cols, rows=keep)
                        return lin.moments(sc, names, "y").fit(0)
                    F = bench.measure("complete reconstruction of one group's model (scope, LIN-001, LIN-002)",
                                      "reconstruction", rebuild, repeats=1, group=grp, **p)
                    bench.records[-1]["result"] = dict(equal=F.x == models[grp].x and F.rss == models[grp].rss)
                plan.close()
                del scope, st


def case_conditioning(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 5_000 if bench.quick else 20_000
    nreq = 8 if bench.quick else 20
    names = ["x0", "x1", "x2", "x3"]
    for kind in ("random", "nearly", "deficient"):
        cols, _ = data(rng, rows, 4, 1, kind)
        scope = mbk.Scope.from_arrays(cols)
        st = lin.moments(scope, names, "y")
        p = dict(matrix=kind, rows=rows, features=4)
        M = bench.measure("model with its exact inverse (fit)", "cpu", lambda: lu.fit(st, 0), repeats=3,
                          phase="build", **p)
        bench.records[-1]["result"] = dict(kind=M.kind, inverse=M.has_inverse,
                                           det_bits=(M.det.numerator.bit_length() if M.det else None))
        bench.measure("LIN-002 fit (no inverse)", "reconstruction", lambda: st.fit(0), repeats=3, phase="build", **p)
        # repeated ridge requests
        lams = [Fraction(i + 1, 10) for i in range(nreq)]

        def woodbury_chain():
            m = M
            for lam in lams:
                m = m.ridge(lam)
            return m

        W = bench.measure("ridge requests (Woodbury chain, rank 4)", "cpu", woodbury_chain, repeats=1,
                          requests=nreq, **p)
        bench.measure("ridge requests (rebuild from A0)", "cpu",
                      lambda: [M.ridge(lam, method="rebuild") for lam in lams][-1], repeats=1, requests=nreq, **p)
        F = bench.measure("ridge requests (LIN-002 fit of the moments)", "reconstruction",
                          lambda: [st.fit(0, ridge=lam) for lam in lams][-1], repeats=1, requests=nreq, **p)
        bench.records[-1]["result"] = dict(equal=W.x == F.x, method=W.method)
        # repeated deletions of single rows
        drop = list(range(nreq))
        rowsd = [dict(x0=int(cols["x0"][i]), x1=int(cols["x1"][i]), x2=int(cols["x2"][i]), x3=int(cols["x3"][i]),
                      y=int(cols["y"][i])) for i in drop]

        def delete_chain():
            m = M
            for r in rowsd:
                m = m.delete_rows([r])
            return m

        D = bench.measure("single-row deletions (Woodbury chain)", "cpu", delete_chain, repeats=1, requests=nreq,
                          **p)
        bench.records[-1]["result"] = dict(method=D.method, fallback=D.fallback)

        def reconstruct_chain():
            keep = np.ones(rows, bool)
            out = None
            for i in drop:
                keep[i] = False
                out = lu.fit(lin.moments(mbk.Scope.from_arrays(cols, rows=keep.copy()), names, "y"), 0)
            return out

        R = bench.measure("single-row deletions (complete reconstruction each time)", "reconstruction",
                          reconstruct_chain, repeats=1, requests=nreq, **p)
        bench.records[-1]["result"] = dict(equal=D.same(R))
        # the floating variant
        def float_solve():
            try:
                return dict(certified=True, **{k: v for k, v in M.float_solve(tol=1e-9).items()
                                               if k in ("bound", "alpha", "kappa")})
            except NotConvergedError as e:
                return dict(certified=False, reason=e.info["reason"], alpha=e.info["alpha"], kappa=e.info["kappa"])
            except SingularError:
                return dict(certified=False, reason="exactly singular")
        bench.measure("floating solve with a certified bound (tol 1e-9)", "cpu", float_solve, repeats=3, **p)
        bench.measure("exact solve of the same system (LIN-002)", "reconstruction",
                      lambda: lin.solve(M.A, M.b).kind, repeats=3, **p)


def case_update_cost(bench):
    rng = np.random.default_rng(bench.seed + 2)
    rows = 5_000 if bench.quick else 20_000
    for d in ((2, 4) if bench.quick else (2, 4, 8, 16)):
        names = [f"x{j}" for j in range(d)]
        cols, _ = data(rng, rows, d, 1)
        st = lin.moments(mbk.Scope.from_arrays(cols), names, "y")
        M = lu.fit(st, 0)
        p = dict(features=d, unknowns=M.n, rows=rows)
        row = {f: int(cols[f][0]) for f in names + ["y"]}
        D = bench.measure("one-row deletion (Woodbury downdate from the kept inverse)", "cpu",
                          lambda: M.delete_rows([row]), repeats=3, **p)
        bench.records[-1]["result"] = dict(method=D.method, fallback=D.fallback)
        z = [1 if f == "1" else row[f] for f in M.design]
        A = [[a - z[i] * z[j] for j, a in enumerate(r)] for i, r in enumerate(M.A)]
        b = [v - z[i] * row["y"] for i, v in enumerate(M.b)]
        S = bench.measure("one-row deletion (exact rebuild of the updated normal equations, with its inverse)",
                          "reconstruction", lambda: lu.system(A, b), repeats=3, **p)
        bench.records[-1]["result"] = dict(equal=S.x == D.x and S.inverse == D.inverse)
        keep = np.ones(rows, bool)
        keep[0] = False
        R = bench.measure("one-row deletion (complete reconstruction: scope, LIN-001, fit with its inverse)",
                          "reconstruction", lambda: lu.fit(lin.moments(mbk.Scope.from_arrays(cols, rows=keep),
                                                                       names, "y"), 0), repeats=3, **p)
        bench.records[-1]["result"] = dict(equal=D.same(R))


def main():
    bench = Bench.from_args("LIN-003", __doc__)
    t0 = time.perf_counter()
    case_grid(bench)
    case_conditioning(bench)
    case_update_cost(bench)
    bench.note("LOGO cost does not depend on the rows once the moment state exists: the plan copies groups x slots "
               "and the solves are per system; reconstruction is measured per group for the first groups only.")
    bench.note("The CUDA path is timed only with --device; the residues' CPU reference is the same inline code as "
               "the kernel. Total wall time %.1f s." % (time.perf_counter() - t0))
    bench.write()


if __name__ == "__main__":
    main()
