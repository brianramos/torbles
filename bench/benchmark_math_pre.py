# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PRE-001 benchmark: predicates on derived values as exact sets of stored codes.

    python bench/benchmark_math_pre.py --capability PRE-001 --out FILE [--device] [--quick]

Case 1, every predicate boundary against direct evaluation: an integer column over a bounded domain (16-bit codes),
maps y = 3/2 x + 7, y = -5/3 x + 11 (a negative slope), y = x^2, y = x mod 97 and 4 monotone pieces, the six
comparison operators at thresholds on the image boundaries, at fractional thresholds and outside the image (the empty
and whole-domain cases). Timed apart: the transform (the exact code set), the scan of the codes (CPU; CUDA with
--device), the native direct check of the original predicate on every row (mismatches recorded, all zero), and the
baselines DuckDB SQL of the original predicate, a NumPy exact-integer evaluation of the decoded values, and the
engine's bit-plane predicate on the transformed code intervals.
Case 2, branch counts 1 / 4 / 16 / 64 and selectivity 1% / 50% / 99%: piecewise maps with that many pieces; the
transformed plane predicate (the engine's bit-plane program of the code intervals) against the direct candidate (NumPy
evaluation of the piecewise map on the decoded values) and the lookup candidate (the predicate tabulated over the code
domain, then gathered by code). Every case records the winner; the cases the plane predicate loses are retained in the
receipt notes.
"""
from fractions import Fraction

import numpy as np
import pandas as pd

from math_bench import Bench
import algq
import torbles
from torbles import math_bank as mbk
from torbles import math_pre as mp

OPS = ["<", "<=", "=", "<>", ">=", ">"]
SQL = {"<": "<", "<=": "<=", "=": "=", "<>": "<>", ">=": ">=", ">": ">"}
NP = {"<": np.less, "<=": np.less_equal, "=": np.equal, "<>": np.not_equal, ">=": np.greater_equal, ">": np.greater}


def make(rng, n):
    x = rng.integers(-20000, 45536, n)
    x[:2] = (-20000, 45535)                                  # 16-bit codes, offset -20000
    frame = pd.DataFrame({"x": x})
    db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
    return db, x


def decoded(t):
    out = np.empty(t.n, dtype=np.uint32)
    algq.lib.at_decode_column(t.at, t.idx("x"), out.ctypes.data)
    return out


def plane(t, s):
    prog = s.plane_program(t, "x")
    arr = (algq.PInstr * len(prog))(*prog)
    bm = np.zeros((t.n + 63) // 64 + 1, dtype=np.uint64)

    def run():
        return int(algq.lib.at_filter_bitmap_masked(t.at, arr, len(prog), bm.ctypes.data, t.used_bm.ctypes.data))
    return run


def exact_int(m, op, c, x):
    """the original predicate on int64 values, multiplied through by positive denominators (exact in int64 here)"""
    if m.kind == "residue":
        return NP[op]((x % m.modulus) * c.denominator, c.numerator)
    if m.kind == "pieces":
        out = np.zeros(len(x), bool)
        for p in m.pieces:
            sel = np.ones(len(x), bool)
            if p.lo is not None:
                sel &= x * p.lo.denominator >= p.lo.numerator
            if p.hi is not None:
                sel &= x * p.hi.denominator <= p.hi.numerator
            out |= sel & _affine_or_square(p.kind, p.a, p.b, op, c, x)
        return out
    return _affine_or_square(m.kind, m.a, m.b, op, c, x)


def _affine_or_square(kind, a, b, op, c, x):
    D = a.denominator * b.denominator * c.denominator
    xx = x * x if kind == "square" else x
    lhs = xx * (a.numerator * b.denominator * c.denominator) + b.numerator * a.denominator * c.denominator
    return NP[op](lhs, c.numerator * a.denominator * b.denominator)


def sql_of(m, op, c):
    """the original predicate in exact SQL integer arithmetic (affine, square and residue maps)"""
    if m.kind == "residue":
        return f"((x % {m.modulus}) + {m.modulus}) % {m.modulus} * {c.denominator} {SQL[op]} {c.numerator}"
    a, b = m.a, m.b
    xx = "x * x" if m.kind == "square" else "x"
    return (f"CAST({xx} AS HUGEINT) * {a.numerator * b.denominator * c.denominator} + {b.numerator * a.denominator * c.denominator}"
            f" {SQL[op]} {c.numerator * a.denominator * b.denominator}")


def boundaries(bench, n):
    rng = np.random.default_rng(bench.seed)
    db, x = make(rng, n)
    t = db["t"]
    codec = mp.Codec.from_table(t, "x")
    scope = mbk.Scope.from_table(t, ["x"])
    lo_x, hi_x = -20000, 45535
    maps = {"affine +3/2 x + 7": mp.affine(Fraction(3, 2), 7), "affine -5/3 x + 11": mp.affine(Fraction(-5, 3), 11),
            "square x^2": mp.square(), "residue x mod 97": mp.residue(97),
            "pieces (4)": mp.pieces([(None, -10001, "affine", -1, 0), (-10000, 0, "square", 1, -5),
                                     (1, 20000, "affine", 2, 3), (20001, None, "affine", Fraction(-1, 2), 50000)])}
    for label, m in maps.items():
        sample = [lo_x, hi_x, 0, 12345]
        ys = [_y(m, Fraction(v)) for v in sample]
        thresholds = {"image boundary": max(ys), "fractional": ys[3] + Fraction(1, 3), "outside": max(ys) + 10 ** 12,
                      "at a value": ys[3]}
        for tlabel, c in thresholds.items():
            for op in OPS:
                p = dict(scenario="boundaries", map=label, threshold=tlabel, op=op, rows=n)
                timed = tlabel == "fractional"
                s = (bench.measure("transform (exact code set)", "cpu", lambda: mp.preimage(codec, m, op, c), repeats=5,
                                   phase="transform", **p) if timed else mp.preimage(codec, m, op, c))
                bits, st = s.scan(scope, "x", direct=True)
                bench.records.append(dict(case="direct check", backend="cpu", phase="check", params=p,
                                          result=dict(mismatches=st["mismatches"], in_set=st["in_set"],
                                                      direct_true=st["direct_true"], spans=s.info["spans"],
                                                      empty=s.info["empty"], all=s.info["all"],
                                                      decided=s.info["decided"])))
                if st["mismatches"]:
                    bench.note(f"MISMATCH {label} {op} {c}: {st}")
                if not timed:
                    continue
                for backend in bench.backends():
                    bench.measure("scan (codes in the set)", backend, lambda: s.scan(scope, "x", backend=backend)[1]["in_set"],
                                  repeats=5, phase="scan", **p)
                    bench.measure("direct check of every row", backend,
                                  lambda: s.scan(scope, "x", direct=True, backend=backend)[1]["mismatches"], repeats=3,
                                  phase="check", **p)
                xs = x
                bench.measure("baseline: NumPy exact evaluation of the values", "numpy",
                              lambda: int(exact_int(m, op, c, xs).sum()), repeats=5, phase="baseline", **p)
                if m.kind != "pieces":
                    bench.measure("baseline: DuckDB SQL of the original predicate", "duckdb",
                                  lambda: int(db.con.execute(f"SELECT count(*) FROM t WHERE {sql_of(m, op, c)}").fetchone()[0]),
                                  repeats=5, phase="baseline", **p)
                if not s.info["residue"]:
                    bench.measure("engine bit-plane predicate on the code intervals", "engine", plane(t, s), repeats=5,
                                  phase="baseline", **p)


def _y(m, x):
    if m.kind == "affine":
        return m.a * x + m.b
    if m.kind == "square":
        return m.a * x * x + m.b
    if m.kind == "residue":
        return x % m.modulus
    for p in m.pieces:
        if (p.lo is None or x >= p.lo) and (p.hi is None or x <= p.hi):
            return p.a * x + p.b if p.kind == "affine" else p.a * x * x + p.b


def branches(bench, n):
    rng = np.random.default_rng(bench.seed + 1)
    db, x = make(rng, n)
    t = db["t"]
    codec = mp.Codec.from_table(t, "x")
    scope = mbk.Scope.from_table(t, ["x"])
    codes = decoded(t)
    losing = []
    for k in (1, 4, 16, 64):
        cuts = np.linspace(-20000, 45536, k + 1).astype(int)
        pcs = []
        for i in range(k):
            lo = None if i == 0 else int(cuts[i])
            hi = None if i == k - 1 else int(cuts[i + 1]) - 1
            a = Fraction((-1) ** i * (1 + i % 3), 1 + i % 2)
            pcs.append(mp.Piece(lo, hi, "affine", a, (i * 7919) % 1000))
        m = mp.pieces(pcs)
        sample = np.sort(np.array([float(_y(m, Fraction(int(v)))) for v in x[:2000]]))
        for sel in (0.01, 0.5, 0.99):
            c = Fraction(int(np.quantile(sample, sel)))
            p = dict(scenario="branches and selectivity", branches=k, selectivity=sel, rows=n)
            s = bench.measure("transform (exact code set)", "cpu", lambda: mp.preimage(codec, m, "<", c), repeats=3,
                              phase="transform", **p)
            bench.records[-1]["result"] = dict(spans=s.info["spans"], decided=s.info["decided"], branches=s.info["branches"])
            timings = {}
            r = bench.measure("transformed plane predicate (engine)", "engine", plane(t, s), repeats=5, phase="candidate", **p)
            timings["plane"] = bench.records[-1]["warm_median_seconds"]
            bench.measure("native scan of the code set", "cpu", lambda: s.scan(scope, "x")[1]["in_set"], repeats=5,
                          phase="candidate", **p)
            timings["scan"] = bench.records[-1]["warm_median_seconds"]
            xs = x
            d = bench.measure("direct candidate: NumPy piecewise evaluation", "numpy",
                              lambda: int(exact_int(m, "<", c, xs).sum()), repeats=5, phase="candidate", **p)
            timings["direct"] = bench.records[-1]["warm_median_seconds"]
            dom = np.arange(0, 65536, dtype=np.int64) - 20000

            def lookup():
                table = exact_int(m, "<", c, dom)                    # the predicate over the code domain
                return int(table[codes].sum())
            lk = bench.measure("lookup candidate: tabulate the code domain, gather", "numpy", lookup, repeats=5,
                               phase="candidate", **p)
            timings["lookup"] = bench.records[-1]["warm_median_seconds"]
            winner = min(timings, key=timings.get)
            agree = r == d == lk
            bench.records.append(dict(case="winner", backend="cpu", phase="summary", params=p,
                                      result=dict(winner=winner, agree=bool(agree), **{f"{a}_s": b for a, b in timings.items()})))
            if winner != "plane":
                losing.append(f"{k} branches, selectivity {sel}: {winner} {timings[winner]:.2e}s vs plane {timings['plane']:.2e}s")
            for backend in bench.backends():
                if backend == "cuda":
                    bench.measure("scan of the code set", "cuda", lambda: s.scan(scope, "x", backend="cuda")[1]["in_set"],
                                  repeats=5, phase="candidate", **p)
    bench.note("case 2, cases the transformed plane predicate loses (retained): " + ("; ".join(losing) or "none"))


def thresholds(bench):
    """a size grid at a fixed output budget: the largest admitted and the smallest refused measured point"""
    from torbles.math_budgets import Budget
    cd = mp.Codec(0, 0, 0, 65535)
    limit = 2000
    admitted, refused = [], []
    for k in (1998, 1999, 2000, 2001, 2002):
        vals = list(range(0, 2 * k, 2))                   # k separate spans
        rec = bench.refusal("output spans", "cpu", lambda: mp.preimage(cd, mp.affine(1, 0), "in", values=vals,
                                                                         budget=Budget(output=limit)),
                            values=k, output_limit=limit)
        (refused if rec["refused"] else admitted).append(k)
    bench.threshold(f"IN values (one span each) at an output budget of {limit} spans", max(admitted, default=None),
                    min(refused, default=None))


def main():
    bench = Bench.from_args("PRE-001", __doc__)
    n = 20_000 if bench.quick else 400_000
    boundaries(bench, n)
    branches(bench, n)
    thresholds(bench)
    bench.note("CPU = native libalg; CUDA timings only with --device; DuckDB and NumPy evaluate the original predicate "
               "exactly (integers multiplied through by positive denominators); the engine plane predicate is "
               "at_filter_bitmap_masked on the OR of BETWEENs of the transformed code intervals.")
    bench.write()


if __name__ == "__main__":
    main()
