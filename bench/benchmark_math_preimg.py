# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PRE-002 benchmark: finite roots, residues and periodic preimages as exact sets of stored codes.

    python bench/benchmark_math_preimg.py --capability PRE-002 --out FILE [--device] [--quick]

Case 1, every predicate boundary against direct evaluation over a bounded integer domain (16-bit codes, x in
[-20000, 45535]): affine maps of positive and negative (fractional) slope, a cubic with two critical points (three
monotone pieces), a square, the power residue x^2 mod 97, the affine residue (37 x + 3) mod 1000, the cubic residue
(x + 2 x^3) mod 28 and the bit maps x XOR c, x AND mask; the six comparison operators and IN at the image's
boundaries, at attained values, at fractional thresholds and outside the image. Every set is checked against the
original predicate on every row (the native direct check: mismatches recorded, all zero) and on every code of the
domain (the exhaustive verification). Timed apart at the fractional threshold: the transform (the exact set), the
native scan of the set (CPU; CUDA with --device), the direct check of every row, the exhaustive verification and the
baselines: NumPy exact integer evaluation of the values, DuckDB SQL of the original predicate and the engine's
bit-plane program of the set (intervals, Boolean cubes, residues mod 2^k).
Case 2, branch counts 1 / 4 / 16 / 64 and selectivity 1% / 50% / 99%: (a x) mod 65537 < c with a = s^-1 mod 65537,
s = 65536 / k and c = k L, exactly k runs of residues, so k intervals of the 16-bit domain (k + 1 when a run wraps).
Candidates: the transformed plane predicate (the engine's program of the intervals), the native scan of the branches,
the native lookup and direct fallbacks (the same predicate past an output budget), NumPy direct evaluation, the NumPy
lookup (the predicate tabulated over the code domain, gathered by code) and DuckDB SQL. Each case records the winner;
the cases the plane predicate loses are retained in the notes. Thresholds (measured, never extrapolated): the output
budget at which the union falls back (and its refusal with fallback none), the iterations of the verification.
"""
from fractions import Fraction
from math import lcm

import numpy as np
import pandas as pd

from math_bench import Bench
import algq
import torbles
from torbles import math_bank as mbk
from torbles import math_pre as mp
from torbles import math_preimg as mq
from torbles.math_budgets import Budget

OPS = ["<", "<=", "=", "<>", ">=", ">"]
NP = {"<": np.less, "<=": np.less_equal, "=": np.equal, "<>": np.not_equal, ">=": np.greater_equal, ">": np.greater}
LO, HI = -20000, 45535                                       # the logical domain: 16-bit codes, offset -20000


def make(rng, n):
    x = rng.integers(LO, HI + 1, n)
    x[:2] = (LO, HI)
    frame = pd.DataFrame({"x": x})
    db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
    return db, x.astype(np.int64)


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
    return run, len(prog)


# ------------------------------------------------------------------ the original predicate, exactly, outside the module
def value_of(m, x):
    """y = f(x) for one Python integer (the benchmark's own reference, used to place thresholds)"""
    if m.kind == "poly":
        return sum(c * x ** i for i, c in enumerate(m.coefficients))
    if m.kind == "polymod":
        return int(sum(int(c) * x ** i for i, c in enumerate(m.coefficients))) % m.modulus
    if m.kind == "affmod":
        return (m.a * x + m.b) % m.modulus
    u = (x - (LO if m.base is None else m.base)) & ((1 << m.width) - 1)
    return {"xor": u ^ m.b, "and": u & m.b, "or": u | m.b}[m.kind]


def np_values(m, x):
    """(integer array, positive denominator L) with f(x) = array / L exactly (every value here fits int64)"""
    if m.kind == "poly":
        L = lcm(*[c.denominator for c in m.coefficients])
        v = np.zeros_like(x)
        for c in reversed(m.coefficients):
            v = v * x + int(c * L)
        return v, L
    if m.kind == "polymod":
        r = np.zeros_like(x)
        for c in reversed(m.coefficients):
            r = (r * x + int(c)) % m.modulus
        return r, 1
    if m.kind == "affmod":
        return (m.a * x + m.b) % m.modulus, 1
    u = (x - (LO if m.base is None else m.base)) & ((1 << m.width) - 1)
    return {"xor": u ^ m.b, "and": u & m.b, "or": u | m.b}[m.kind], 1


def np_predicate(m, op, c, x):
    v, L = np_values(m, x)
    if op == "in":
        return np.isin(v, [int(Fraction(val) * L) for val in c if (Fraction(val) * L).denominator == 1])
    c = Fraction(c)
    return NP[op](v * c.denominator, c.numerator * L)


def sql_of(m, op, c):
    """the original predicate in exact SQL integer arithmetic (HUGEINT), multiplied through by positive denominators"""
    X = "CAST(x AS HUGEINT)"
    if m.kind == "poly":
        L = lcm(*[k.denominator for k in m.coefficients])
        expr = " + ".join(f"({int(k * L)}) * " + " * ".join([X] * i) if i else f"({int(k * L)})"
                          for i, k in enumerate(m.coefficients) if k) or "0"
    elif m.kind == "polymod":
        L = 1
        body = " + ".join(f"({int(k)}) * " + " * ".join([X] * i) if i else f"({int(k)})"
                          for i, k in enumerate(m.coefficients) if k) or "0"
        expr = f"((({body}) % {m.modulus}) + {m.modulus}) % {m.modulus}"
    elif m.kind == "affmod":
        L = 1
        expr = f"((({m.a} * {X} + {m.b}) % {m.modulus}) + {m.modulus}) % {m.modulus}"
    else:
        L = 1
        base = LO if m.base is None else m.base
        u = f"(CAST(x - ({base}) AS BIGINT) & {(1 << m.width) - 1})"
        expr = {"xor": f"xor({u}, {m.b})", "and": f"({u} & {m.b})", "or": f"({u} | {m.b})"}[m.kind]
    if op == "in":
        vals = [int(Fraction(v) * L) for v in c if (Fraction(v) * L).denominator == 1] or [None]
        return f"({expr}) IN ({', '.join('NULL' if v is None else str(v) for v in vals)})"
    c = Fraction(c)
    return f"({expr}) * {c.denominator} {op} {c.numerator * L}"


# ------------------------------------------------------------------ case 1: every boundary against direct evaluation
MAPS = {
    "affine +3 x + 7": mq.poly([7, 3]),
    "affine -5/3 x + 11": mq.poly([11, Fraction(-5, 3)]),
    "cubic (x - 1000)(x + 2000)(x - 30000)": mq.poly([60_000_000_000, -32_000_000, -29_000, 1]),
    "square x^2 - 36": mq.poly([-36, 0, 1]),
    "power residue x^2 mod 97": mq.powmod(2, 97),
    "affine residue (37 x + 3) mod 1000": mq.affmod(37, 3, 1000),
    "cubic residue (x + 2 x^3) mod 28": mq.polymod([0, 1, 0, 2], 28),
    "bit map x XOR 0x5A5A (16 bits)": mq.xor(0x5A5A, 16),
    "bit map x AND 0x0FF0 (16 bits)": mq.bitand(0x0FF0, 16),
}


def thresholds_of(m):
    sample = [LO, HI, 0, 12345, -7, 1000, 30000, -2000]
    ys = [value_of(m, v) for v in sample]
    at = value_of(m, 12345)
    if m.kind == "poly":
        return {"image max (sampled)": max(ys), "image min (sampled)": min(ys), "at a value": at,
                "fractional": at + Fraction(1, 3), "outside": max(ys) + 10 ** 12}
    top = m.modulus - 1 if m.kind in ("polymod", "affmod") else (1 << m.width) - 1
    return {"zero": 0, "at a value": at, "top": top, "fractional": at + Fraction(1, 3), "outside": top + 5}


def boundaries(bench, n):
    rng = np.random.default_rng(bench.seed)
    db, x = make(rng, n)
    t = db["t"]
    codec = mp.Codec.from_table(t, "x")
    scope = mbk.Scope.from_table(t, ["x"])
    checked = mismatched = 0
    for label, m in MAPS.items():
        preds = [(tl, op, c) for tl, c in thresholds_of(m).items() for op in OPS]
        preds.append(("in (8 attained values and one absent)", "in",
                      sorted({value_of(m, v) for v in (LO, HI, 0, 12345, -7, 1000, 30000, -2000)}) + [-(10 ** 15)]))
        for tlabel, op, c in preds:
            p = dict(scenario="boundaries", map=label, threshold=tlabel, op=op, rows=n)
            timed = tlabel == "fractional"
            make_set = (lambda: mq.preimage(codec, m, op, values=c)) if op == "in" else (lambda: mq.preimage(codec, m, op, c))
            s = (bench.measure("transform (exact set)", "cpu", make_set, repeats=3, phase="transform", **p)
                 if timed else make_set())
            info = s.info
            bits, st = s.scan(scope, "x", direct=True)
            v = s.verify()
            checked += 1
            mismatched += int(bool(st["mismatches"] or v["mismatches"]))
            bench.records.append(dict(case="direct check (every row) and verification (every code)", backend="cpu",
                                      phase="check", params=p,
                                      result=dict(mode=info["mode"], repr=info["repr"], branches=info["branches"],
                                                  codes=info["codes"], empty=info["empty"], all=info["all"],
                                                  row_mismatches=st["mismatches"], in_set=st["in_set"],
                                                  direct_true=st["direct_true"], code_mismatches=v["mismatches"],
                                                  members=v["members"])))
            if st["mismatches"] or v["mismatches"]:
                bench.note(f"MISMATCH {label} {op} {c}: rows {st['mismatches']}, codes {v['mismatches']}")
            if not timed:
                continue
            for backend in bench.backends():
                bench.measure("scan (codes in the set)", backend, lambda: s.scan(scope, "x", backend=backend)[1]["in_set"],
                              repeats=3, phase="scan", **p)
                bench.measure("direct check of every row", backend,
                              lambda: s.scan(scope, "x", direct=True, backend=backend)[1]["mismatches"], repeats=3,
                              phase="check", **p)
                bench.measure("exhaustive verification (every code of the domain)", backend,
                              lambda: s.verify(backend=backend)["mismatches"], repeats=3, phase="check", **p)
            bench.measure("baseline: NumPy exact evaluation of the values", "numpy",
                          lambda: int(np_predicate(m, op, c, x).sum()), repeats=3, phase="baseline", **p)
            sql = sql_of(m, op, c)
            bench.measure("baseline: DuckDB SQL of the original predicate", "duckdb",
                          lambda: int(db.con.execute(f"SELECT count(*) FROM t WHERE {sql}").fetchone()[0]), repeats=3,
                          phase="baseline", **p)
            if info["mode"] == "branches":
                try:
                    run, ninstr = plane(t, s)
                except mq.UnsupportedError as e:
                    bench.note(f"{label} {op} {c}: no plane program ({e})")
                else:
                    bench.measure("engine bit-plane program of the set", "engine", run, repeats=3, phase="baseline",
                                  instructions=ninstr, **p)
    bench.note(f"case 1: {checked} predicates (maps x thresholds x operators), {mismatched} with a mismatch on a row "
               "or a code of the domain")


# ------------------------------------------------------------------ case 2: branch counts and selectivity
def branches(bench, n):
    rng = np.random.default_rng(bench.seed + 1)
    db, x = make(rng, n)
    t = db["t"]
    codec = mp.Codec.from_table(t, "x")
    scope = mbk.Scope.from_table(t, ["x"])
    codes = decoded(t)
    dom = np.arange(LO, HI + 1, dtype=np.int64)
    M = 65537
    losing = []
    for k in (1, 4, 16, 64):
        s_ = 65536 // k
        m = mq.affmod(pow(s_, -1, M), 0, M)
        for sel in (0.01, 0.5, 0.99):
            c = k * max(1, int(sel * s_))
            p = dict(scenario="branches and selectivity", runs=k, selectivity=sel, threshold=c, rows=n)
            s = bench.measure("transform (exact set)", "cpu", lambda: mq.preimage(codec, m, "<", c), repeats=3,
                              phase="transform", **p)
            nb = s.info["branches"]
            p["branches"] = nb
            bench.records[-1]["params"]["branches"] = nb
            timings, counts = {}, {}
            run, ninstr = plane(t, s)
            counts["plane"] = bench.measure("transformed plane predicate (engine)", "engine", run, repeats=5,
                                            phase="candidate", instructions=ninstr, **p)
            timings["plane"] = bench.records[-1]["warm_median_seconds"]
            counts["scan"] = bench.measure("native scan of the branches", "cpu",
                                           lambda: s.scan(scope, "x")[1]["in_set"], repeats=5, phase="candidate", **p)
            timings["scan"] = bench.records[-1]["warm_median_seconds"]
            tight = Budget(output=nb - 1)
            lk = mq.preimage(codec, m, "<", c, budget=tight)
            dr = mq.preimage(codec, m, "<", c, budget=tight, fallback="direct")
            assert (lk.info["mode"], dr.info["mode"]) == ("lookup", "direct"), (lk.info, dr.info)
            counts["native lookup"] = bench.measure("native lookup fallback (output budget branches - 1)", "cpu",
                                                    lambda: lk.scan(scope, "x")[1]["in_set"], repeats=5,
                                                    phase="candidate", nlookup=lk.info["nlookup"], **p)
            timings["native lookup"] = bench.records[-1]["warm_median_seconds"]
            counts["native direct"] = bench.measure("native direct fallback (output budget branches - 1)", "cpu",
                                                    lambda: dr.scan(scope, "x")[1]["in_set"], repeats=5,
                                                    phase="candidate", **p)
            timings["native direct"] = bench.records[-1]["warm_median_seconds"]
            counts["numpy direct"] = bench.measure("direct candidate: NumPy evaluation", "numpy",
                                                   lambda: int(np_predicate(m, "<", c, x).sum()), repeats=5,
                                                   phase="candidate", **p)
            timings["numpy direct"] = bench.records[-1]["warm_median_seconds"]

            def lookup():
                table = np_predicate(m, "<", c, dom)                  # the predicate over the code domain
                return int(table[codes].sum())
            counts["numpy lookup"] = bench.measure("lookup candidate: tabulate the code domain, gather", "numpy",
                                                   lookup, repeats=5, phase="candidate", **p)
            timings["numpy lookup"] = bench.records[-1]["warm_median_seconds"]
            sql = sql_of(m, "<", c)
            counts["duckdb"] = bench.measure("direct candidate: DuckDB SQL", "duckdb",
                                             lambda: int(db.con.execute(f"SELECT count(*) FROM t WHERE {sql}").fetchone()[0]),
                                             repeats=5, phase="candidate", **p)
            timings["duckdb"] = bench.records[-1]["warm_median_seconds"]
            for backend in bench.backends():
                if backend == "cuda":
                    bench.measure("scan of the branches", "cuda", lambda: s.scan(scope, "x", backend="cuda")[1]["in_set"],
                                  repeats=5, phase="candidate", **p)
            winner = min(timings, key=timings.get)
            agree = len(set(counts.values())) == 1
            bench.records.append(dict(case="winner", backend="cpu", phase="summary", params=p,
                                      result=dict(winner=winner, agree=bool(agree), rows_selected=counts["scan"],
                                                  **{f"{a}_s": b for a, b in timings.items()})))
            if not agree:
                bench.note(f"DISAGREEMENT {k} runs, selectivity {sel}: {counts}")
            if winner != "plane":
                losing.append(f"{nb} branches, selectivity {sel}: {winner} {timings[winner]:.2e}s vs plane "
                              f"{timings['plane']:.2e}s ({ninstr} instructions)")
    bench.note("case 2, cases the transformed plane predicate loses (retained): " + ("; ".join(losing) or "none"))


def thresholds(bench):
    cd = mp.Codec(LO, 0, 0, 65535)
    M = 65537
    m = mq.affmod(pow(1024, -1, M), 0, M)                     # 64 runs at 50%
    s = mq.preimage(cd, m, "<", 32768)
    nb = s.info["branches"]
    at = mq.preimage(cd, m, "<", 32768, budget=Budget(output=nb))
    past = mq.preimage(cd, m, "<", 32768, budget=Budget(output=nb - 1))
    rec = bench.refusal("branch union one past the output budget (fallback none)", "cpu",
                        lambda: mq.preimage(cd, m, "<", 32768, budget=Budget(output=nb - 1), fallback="none"),
                        output=nb - 1)
    bench.threshold("branches under an output budget", admitted=dict(output=nb, mode=at.info["mode"]),
                    refused=dict(output=nb - 1, mode=past.info["mode"], fallback=past.info["fallback"],
                                 why=past.info["why"], need=rec.get("need"), resource=rec.get("resource")))
    size = 65536
    bench.measure("verification at its iteration need", "cpu",
                  lambda: s.verify(budget=Budget(iterations=size))["mismatches"], repeats=1, phase="check",
                  iterations=size)
    rec = bench.refusal("verification one code short", "cpu", lambda: s.verify(budget=Budget(iterations=size - 1)),
                        iterations=size - 1)
    bench.threshold("exhaustive verification (one iteration per code)", admitted=dict(iterations=size),
                    refused=dict(iterations=size - 1, need=rec.get("need"), resource=rec.get("resource")))


def main():
    bench = Bench.from_args("PRE-002", __doc__)
    n = 20_000 if bench.quick else 400_000
    boundaries(bench, n)
    branches(bench, n)
    thresholds(bench)
    bench.note("CPU = native libalg; CUDA timings only with --device (emulated runs are compared with the CPU in "
               "tests/test_math_preimg.py, never timed as CUDA); NumPy and DuckDB evaluate the original predicate "
               "exactly (integers multiplied through by positive denominators); the engine plane predicate is "
               "at_filter_bitmap_masked on the set's program (an OR of BETWEENs of the intervals, or of ANDs of bit "
               "literals for cubes and residues mod 2^k).")
    bench.write()


if __name__ == "__main__":
    main()
