# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""UNC-001 benchmark: certified abstract states (intervals, affine forms, congruences) for pruning.

    python bench/benchmark_math_unc.py --capability UNC-001 --out FILE [--device] [--quick]

Case 1, pruning with 1% / 50% / 99% ambiguous groups: engine tables whose groups are decided by an interval
threshold (x > T), by a correlated cancellation ((x + y) - y > T: affine forms decide it, intervals alone cannot) and
by a modular constraint (x mod 7 = 3: congruences decide it), with the fraction of ambiguous groups set to 1%, 50% and
99%. Proof work (the state construction and the abstract evaluation of every group) and concrete fallback work (the
exact evaluation of the rows of the ambiguous groups) are timed separately, then the whole filter, against the
baselines: the exact evaluation of every row (no pruning) and DuckDB SQL of the same predicate. Counts (decided
groups, rows proven, rows evaluated) are recorded with each result.
Case 2, precision, rank and error budget: correlated expressions over small integer domains evaluated with the exact
carrier and binary64 coefficients of 53 / 24 / 12 / 6 bits, ranks 1 / 2 / 4 / 8 / 16 and error budgets none / 4 /
1/2 / 0; the certified bounds are compared with the exact range from exhaustive enumeration (Python Fractions) and the
unproved cases are recorded (the predicates E <= max and E >= min, true on every row, not decided ALL).
"""
import itertools
from fractions import Fraction

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_unc as mu


def table(rng, groups, per, ambiguous, kind):
    """a table of `groups` groups of `per` rows; a fraction `ambiguous` of the groups straddles the predicate"""
    n = groups * per
    k = np.repeat(np.arange(groups), per)
    amb = np.zeros(groups, bool)
    amb[rng.choice(groups, size=max(1, int(round(ambiguous * groups))), replace=False)] = True
    T = 1000
    x = np.empty(n, np.int64)
    y = rng.integers(-10 ** 6, 10 ** 6, n)
    for g in range(groups):
        rows = slice(g * per, (g + 1) * per)
        if kind == "modular":
            if amb[g]:
                v = rng.integers(0, 10 ** 4, per)                     # mixed residues: undecidable by the class
            else:
                v = 7 * rng.integers(0, 1500, per) + (3 if g % 2 else int(rng.choice([0, 1, 2, 4, 5, 6])))
            x[rows] = v
        else:
            base = (T - int(rng.integers(0, 9))) if amb[g] else (T + 1 + int(rng.integers(0, 50)) if g % 2 else T - 60 + int(rng.integers(0, 50)))
            v = base + rng.integers(0, 10, per)
            v[0], v[-1] = base, base + 9                          # the group's extremes are attained
            x[rows] = v
    frame = pd.DataFrame({"k": k, "x": x, "y": y})
    db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT, y BIGINT)", {"t": frame})
    return db, int(amb.sum())


def pruning(bench, groups, per):
    X, Y = mu.column("x"), mu.column("y")
    preds = {"interval threshold": (X > 1000, "x > 1000"),
             "correlated cancellation": ((X + Y) - Y > 1000, "(x + y) - y > 1000"),
             "modular constraint": ((X % 7).eq(3), "x % 7 = 3")}
    for kind, (pred, sql) in preds.items():
        for frac in (0.01, 0.5, 0.99):
            rng = np.random.default_rng(bench.seed + int(frac * 100))
            db, namb = table(rng, groups, per, frac, "modular" if kind == "modular constraint" else "interval")
            t = db["t"]
            p = dict(scenario=kind, ambiguous=frac, groups=groups, rows=groups * per, ambiguous_groups=namb)
            scope = bench.measure("state construction: scope", "cpu", lambda: mbk.Scope.from_table(t, ["x", "y"], group_by=["k"]),
                                  repeats=3, phase="build", **p)
            state = bench.measure("state construction: abstract inputs", "cpu", lambda: mu.State.from_scope(scope, ["x", "y"]),
                                  repeats=3, phase="build", **p)
            prog = mu.Program(pred, ["x", "y"])
            for backend in bench.backends():
                ev = bench.measure("proof: abstract evaluation of every group", backend,
                                   lambda: state.evaluate(prog, backend=backend), repeats=5, phase="proof", **p)
                c = ev.counts()
                bench.records[-1]["result"] = dict(c, decided=c["all"] + c["none"])
                dec = ev.decisions
                bits, st = bench.measure("fallback: exact rows of the unknown groups", backend,
                                         lambda: mu.rows(scope, prog, decisions=dec, backend=backend), repeats=5,
                                         phase="fallback", **p)
                bench.records[-1]["result"] = dict(rows_evaluated=st["rows_evaluated"], rows_proven=st["rows_proven"],
                                                   rows_true=st["rows_true"])
                fbits, fst = bench.measure("filter: proof + fallback", backend, lambda: state.filter(scope, prog, backend=backend),
                                           repeats=5, phase="filter", **p)
                bench.records[-1]["result"] = {k2: fst[k2] for k2 in ("rows_true", "rows_proven", "rows_evaluated",
                                                                      "groups_all", "groups_none", "groups_unknown")}
                allbits, ast = bench.measure("baseline: exact evaluation of every row", backend,
                                             lambda: mu.rows(scope, prog, backend=backend), repeats=5, phase="baseline", **p)
                bench.records[-1]["result"] = dict(rows_true=ast["rows_true"], rows_evaluated=ast["rows_evaluated"],
                                                   agrees=bool(np.array_equal(allbits, fbits)))
            if kind == "correlated cancellation":                  # the same proof with intervals alone
                ivp = mu.Program(pred, ["x", "y"], domains=("congruence",))
                ev = bench.measure("proof without affine forms (intervals and congruences)", "cpu",
                                   lambda: state.evaluate(ivp), repeats=3, phase="proof", **p)
                bench.records[-1]["result"] = ev.counts()
            n = bench.measure("baseline: DuckDB", "duckdb",
                              lambda: int(db.con.execute(f"SELECT count(*) FROM t WHERE {sql}").fetchone()[0]),
                              repeats=5, phase="baseline", **p)


def precision(bench, ngroups):
    x, y, z = mu.column("x"), mu.column("y"), mu.column("z")
    s = x + y
    d = x - y
    xz = x * z
    exprs = {"(x+y)^2 - (x-y)^2": s.sqr() - d.sqr(),
             "|x-y| - |y-x| + x": abs(d) - abs(y - x) + x,
             "xz + yz - (x+y)z": xz + y * z - s * z,
             "min + max - x - y": mu.min_(x, y) + mu.max_(x, y) - x - y,
             "(x/3 + y/7) 21 - 7x - 3y + z": (x / 3 + y / 7) * 21 - x * 7 - y * 3 + z,
             "x(yz) - (xy)z": x * (y * z) - (x * y) * z}
    spec = [dict(rows=8, present=8, min=-8, max=8, modulus=1), dict(rows=8, present=8, min=-5, max=7, modulus=1),
            dict(rows=8, present=8, min=1, max=6, modulus=1)]
    names = ["x", "y", "z"]
    state = mu.State.from_inputs([spec] * ngroups, names)
    values = [list(range(s_["min"], s_["max"] + 1)) for s_ in spec]
    configs = [("exact", None)] + [("binary64", b) for b in (53, 24, 12, 6)]
    unproved = []
    for label, e in exprs.items():
        exact = []
        for xv, yv, zv in itertools.product(*values):
            exact.append(_concrete(e, dict(x=Fraction(xv), y=Fraction(yv), z=Fraction(zv))))
        elo, ehi = min(exact), max(exact)
        for carrier, bits in configs:
            for rank in (1, 2, 4, 8, 16):
                for budget in (None, 4, Fraction(1, 2), 0):
                    prog = mu.Program(e, names, carrier=carrier, precision=bits, rank=rank, budget=budget)
                    p = dict(scenario="precision, rank and budget", expression=label, carrier=carrier, precision=bits or "exact",
                             rank=rank, budget=str(budget), groups=ngroups)
                    ev = bench.measure("certified bounds", "cpu", lambda: state.evaluate(prog), repeats=3, phase="proof", **p)
                    lo, hi = (Fraction(v) for v in ev[0]["interval"])
                    width = hi - lo
                    ewidth = ehi - elo
                    opts = dict(carrier=carrier, precision=bits, rank=rank, budget=budget)
                    le = state.evaluate(mu.Program(e <= ehi, names, **opts))[0]
                    ge = state.evaluate(mu.Program(e >= elo, names, **opts))[0]
                    slack = max(ehi - elo, 1)
                    le2 = state.evaluate(mu.Program(e <= ehi + slack, names, **opts))[0]
                    ge2 = state.evaluate(mu.Program(e >= elo - slack, names, **opts))[0]
                    proven = dict(upper=le["decision"] == "all", lower=ge["decision"] == "all")
                    loose = le2["decision"] == "all" and ge2["decision"] == "all"
                    bench.records[-1]["result"] = dict(certified_lo=str(lo), certified_hi=str(hi), exact_lo=str(elo),
                                                       exact_hi=str(ehi), width_ratio=float(width / ewidth) if ewidth else
                                                       (0.0 if width == 0 else float("inf")),
                                                       encloses=bool(lo <= elo and hi >= ehi), proven_upper=proven["upper"],
                                                       proven_lower=proven["lower"], proven_within_width=loose,
                                                       condensed="condensed" in ev[0]["flags"],
                                                       budget_hit="budget" in ev[0]["flags"])
                    if not (proven["upper"] and proven["lower"]):
                        unproved.append(dict(p, certified=(str(lo), str(hi)), exact=(str(elo), str(ehi)),
                                             within_width=loose))
    loose_fail = [u for u in unproved if not u["within_width"]]
    bench.note(f"case 2: {len(unproved)} configurations do not prove the exact extremes (E <= max, E >= min) and "
               f"{len(loose_fail)} do not prove them within one exact width of slack: "
               + "; ".join(f"{u['expression']} {u['carrier']}/{u['precision']} rank {u['rank']} budget {u['budget']}"
                           for u in loose_fail[:40]) + (" ..." if len(loose_fail) > 40 else ""))
    bench.records.append(dict(case="unproved cases", backend="cpu", phase="summary", params={}, unproved=unproved))


def _concrete(e, env):
    op = e.op
    if op == "col":
        return env[e.name]
    if op == "const":
        return e.k
    a = [_concrete(v, env) for v in e.args]
    return {"add": lambda: a[0] + a[1], "sub": lambda: a[0] - a[1], "mul": lambda: a[0] * a[1], "neg": lambda: -a[0],
            "div": lambda: a[0] / e.k, "min": lambda: min(a), "max": lambda: max(a), "abs": lambda: abs(a[0]),
            "sqr": lambda: a[0] * a[0], "mod": lambda: a[0] % e.k}[op]()


def budget_thresholds(bench):
    """size grids at a fixed budget: the largest admitted and the smallest refused measured point"""
    from torbles.math_budgets import Budget
    x = mu.column("x")
    one = [dict(rows=4, present=4, min=0, max=9, modulus=1)]
    progs, e = [], x
    for i in range(14):                                   # 4 i + 5 nodes
        e = e * 2 + 1
        if i >= 9:
            progs.append(mu.Program(e, ["x"]))            # 41, 45, 49, 53, 57 nodes
    state = mu.State.from_inputs([one] * 256, ["x"])
    limit = 49
    admitted, refused = [], []
    for prog in progs:
        rec = bench.refusal("program nodes (circuit)", "cpu", lambda: state.evaluate(prog, budget=Budget(circuit=limit)),
                            nodes=len(prog), circuit_limit=limit, groups=256)
        (refused if rec["refused"] else admitted).append(len(prog))
    bench.threshold(f"program nodes at circuit budget {limit}", max(admitted, default=None), min(refused, default=None),
                    groups=256)
    prog = progs[2]
    limit = mu.need(prog, 256, backend="cuda")["bytes"]
    admitted, refused = [], []
    for g in (254, 255, 256, 257, 258):
        st = mu.State.from_inputs([one] * g, ["x"])
        rec = bench.refusal("CUDA device bytes", "cpu", lambda: st.evaluate(prog, backend="cuda", budget=Budget(bytes=limit)),
                            groups=g, bytes_limit=limit, nodes=len(prog))
        (refused if rec["refused"] else admitted).append(g)
    bench.threshold(f"groups at a CUDA device-copy budget of {limit} bytes ({len(prog)} nodes)",
                    max(admitted, default=None), min(refused, default=None))


def main():
    bench = Bench.from_args("UNC-001", __doc__)
    groups, per = (60, 40) if bench.quick else (400, 100)
    pruning(bench, groups, per)
    precision(bench, 64 if bench.quick else 512)
    budget_thresholds(bench)
    bench.note("CPU = native libalg; CUDA timings only with --device (an emulated run is never timed as CUDA); "
               "DuckDB evaluates the same predicate over the same rows; the fallback evaluates exactly (int64 rationals).")
    bench.write()


if __name__ == "__main__":
    main()
