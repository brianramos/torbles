# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-009 benchmark: coefficient bit width and carrier-conversion cost independently of row count, and native
fixed-width, wide and arbitrary-precision paths on matched expressions.

    python bench/benchmark_math_carriers.py --capability FND-009 --out FILE [--device] [--quick]

Bit width (64 .. 8192 bits): add, mul, divmod (a 2w-bit by a w-bit integer), gcd and the conversions (Python int <->
limbs, decimal text both ways, big rational -> tight enclosure, reduction into GF(p)) per operation, each measured
at two batch sizes so that the per-operation cost is shown to depend on the width, not on the number of rows; the
native batch call against Python's int (the baseline). Construction of the operands is a separate "build" phase.
Matched expressions: the dot product of n int64 pairs of w-bit values in checked int64, checked 128-bit, exact
192-bit, big integers, four GF(p) residues + CRT, Python ints, NumPy int64 (wraps: whether it matched is recorded)
and NumPy float64 (rounds: its relative error is recorded); Horner evaluation in int64 / 128-bit / big integers /
Python. Overflow thresholds of the checked paths and the limb-budget threshold are measured, never extrapolated.
GF(p): the batched elementwise, dot and matrix-vector kernels on the CPU (and CUDA with --device) against NumPy
(p < 2^31, exact in int64) and Python baselines; int64 rationals -> enclosures in batch against NumPy division.
"""
import ctypes as C
import math
import random
import sys
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_carriers as mc
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError

sys.set_int_max_str_digits(0)
P61 = (1 << 61) - 1
P31 = (1 << 31) - 1


def _rand(rng, bits):
    v = rng.getrandbits(bits) | (1 << (bits - 1))
    return -v if rng.random() < 0.5 else v


class NativeBatch:
    """prebuilt handle arrays of big integers (built once: the "build" phase), and a native batch operation whose
    results are released natively, so the timing is the native operation itself"""
    def __init__(self, xs, ys):
        self.a = [mc.BigInt(x) for x in xs]
        self.b = [mc.BigInt(y) for y in ys]
        n = len(xs)
        self.pa = (C.c_void_p * n)(*[o._h for o in self.a])
        self.pb = (C.c_void_p * n)(*[o._h for o in self.b])
        self.out = (C.c_void_p * n)()
        self.n = n
        self.api = mc._api()

    def run(self, op):
        refusal = mc._Refusal()
        rc = self.api["bigint_batch"](mc.OPS[op], self.pa, self.pb, self.n, self.out, None, C.byref(refusal))
        if rc:
            raise RuntimeError(f"batch {op}: native code {rc}")
        self.api["bigint_free_array"](self.out, self.n)
        return self.n


def bit_width(bench):
    rng = random.Random(bench.seed)
    widths = (64, 256, 1024, 4096, 8192) if bench.quick else (64, 128, 256, 512, 1024, 2048, 4096, 8192)
    sizes = (100, 400) if bench.quick else (200, 2000)
    for w in widths:
        for n in sizes:
            p = dict(bits=w, rows=n)
            xs = [_rand(rng, w) for _ in range(n)]
            ys = [_rand(rng, w) for _ in range(n)]
            big = [_rand(rng, 2 * w) for _ in range(n)]
            batch = bench.measure("operands", "cpu", lambda: NativeBatch(xs, ys), repeats=1, phase="build", **p)
            div = NativeBatch(big, ys)
            for op, py in (("add", lambda: [x + y for x, y in zip(xs, ys)]),
                           ("mul", lambda: [x * y for x, y in zip(xs, ys)]),
                           ("gcd", lambda: [math.gcd(x, y) for x, y in zip(xs, ys)])):
                rec = bench.measure(f"bigint {op}", "cpu", lambda: batch.run(op), repeats=5, **p)
                bench.measure(f"bigint {op}", "python", py, repeats=5, **p)
                del rec
            bench.measure("bigint divmod (2w by w bits)", "cpu", lambda: div.run("div"), repeats=5, **p)
            bench.measure("bigint divmod (2w by w bits)", "python", lambda: [divmod(x, y) for x, y in zip(big, ys)],
                          repeats=5, **p)
            # conversions at this width
            bench.measure("python int -> bigint (limbs)", "cpu", lambda: len([mc.BigInt(x) for x in xs]), repeats=3, **p)
            bench.measure("bigint -> python int (limbs)", "cpu", lambda: len([int(b) for b in batch.a]), repeats=3, **p)
            texts = [str(x) for x in xs]
            bench.measure("decimal -> bigint", "cpu", lambda: len([mc.BigInt(t) for t in texts]), repeats=3, **p)
            bench.measure("decimal -> bigint", "python", lambda: len([int(t) for t in texts]), repeats=3, **p)
            bench.measure("bigint -> decimal", "cpu", lambda: len([b.to_decimal() for b in batch.a]), repeats=3, **p)
            bench.measure("bigint -> decimal", "python", lambda: len([str(x) for x in xs]), repeats=3, **p)
            qs = [mc.BigRational(x, y) for x, y in zip(xs, ys)]
            fr = [Fraction(x, y) for x, y in zip(xs, ys)]
            bench.measure("bigq -> tight enclosure", "cpu", lambda: len([q.enclose() for q in qs]), repeats=3, **p)
            bench.measure("bigq -> nearest float (not an enclosure)", "python", lambda: len([float(f) for f in fr]),
                          repeats=3, **p)
            F = mc.GF(P61)
            bench.measure("bigint -> GF(p) reduction", "cpu", lambda: len([F(b) for b in batch.a]), repeats=3, **p)
            bench.measure("bigint -> GF(p) reduction", "python", lambda: len([x % P61 for x in xs]), repeats=3, **p)
    bench.note("Per-operation cost = warm seconds / rows; each width is measured at two row counts. Conversions cross "
               "the Python boundary once per value (marshalling included); the arithmetic rows are one native batch "
               "call per measurement (operands prebuilt in the 'build' phase, results freed natively).")


def crt_widths(bench):
    rng = random.Random(bench.seed + 1)
    pool = [p for p in range((1 << 62) - 57, (1 << 62) - 400000, -2) if mc.GF.is_prime(p)][:256]
    for k in ((2, 8, 32) if bench.quick else (2, 8, 32, 128, 256)):
        fields = [mc.GF(p) for p in pool[:k]]
        M = math.prod(pool[:k])
        x = rng.randrange(M)
        res = [x % p for p in pool[:k]]

        def python_crt():
            acc, m = 0, 1
            for r, p in zip(res, pool[:k]):
                acc += m * ((r - acc) * pow(m, -1, p) % p)
                m *= p
            return acc
        prm = dict(residues=k, bits=M.bit_length())
        got = bench.measure("CRT reconstruction", "cpu", lambda: mc.crt(res, fields)[0], repeats=5, **prm)
        bench.measure("CRT reconstruction", "python", python_crt, repeats=5, **prm)
        if got != x:
            raise SystemExit("CRT mismatch")
        f = Fraction(rng.getrandbits(M.bit_length() // 2 - 2), rng.getrandbits(M.bit_length() // 2 - 2) | 1)
        if math.gcd(f.denominator, M) == 1:
            u = f.numerator * pow(f.denominator, -1, M) % M
            if bench.measure("rational reconstruction", "cpu", lambda: mc.rational_reconstruct(u, M), repeats=5,
                             **prm) != f:
                raise SystemExit("reconstruction mismatch")


def matched(bench):
    rng = np.random.default_rng(bench.seed)
    n = 20_000 if bench.quick else 200_000
    nbig = 2_000 if bench.quick else 20_000
    primes = [(1 << 62) - 57, (1 << 62) - 87, (1 << 62) - 117, (1 << 62) - 143]
    fields = [mc.GF(p) for p in primes]
    last_ok = {"i64": None, "i128": None}
    first_bad = {"i64": None, "i128": None}
    for w in (8, 16, 31, 32, 48, 62, 63):
        hi = (1 << w) - 1 if w < 63 else (1 << 63) - 1
        a = rng.integers(-hi, hi, size=n, dtype=np.int64, endpoint=True)
        b = rng.integers(-hi, hi, size=n, dtype=np.int64, endpoint=True)
        la, lb = [int(v) for v in a], [int(v) for v in b]
        exact = sum(x * y for x, y in zip(la, lb))
        p = dict(bits=w, rows=n)
        for carrier in ("i64", "i128"):
            try:
                got = bench.measure("dot product", f"cpu-{carrier}", lambda: mc.dot(a, b, carrier=carrier), repeats=5, **p)
                assert got == exact
                last_ok[carrier] = w
            except OverflowError:
                bench.records.append(dict(case="dot product", backend=f"cpu-{carrier}", phase="refusal", params=p,
                                          refused=True, reason="overflow (checked fixed width)"))
                if first_bad[carrier] is None:
                    first_bad[carrier] = w
        for backend in bench.backends():
            got = bench.measure("dot product", f"{backend}-wide192", lambda: mc.dot(a, b, carrier="wide", backend=backend),
                                repeats=5, **p)
            assert got == exact

        def multimodular():
            parts = []
            for f in fields:
                ra, rb = f.reduce(a).values, f.reduce(b).values
                parts.append(f.dot(ra, rb).values.v)
            return mc.crt(parts, fields, symmetric=True)[0]
        got = bench.measure("dot product", "cpu-gfp4+crt", multimodular, repeats=3, **p)
        assert got == exact
        bench.measure("dot product", "python", lambda: sum(x * y for x, y in zip(la, lb)), repeats=3, **p)
        wrapped = bench.measure("dot product", "numpy-int64", lambda: int((a * b).sum()), repeats=5, **p)
        approx = bench.measure("dot product", "numpy-float64", lambda: float(a.astype(np.float64) @ b.astype(np.float64)),
                               repeats=5, **p)
        bench.records[-2]["matches_exact"] = wrapped == exact
        bench.records[-1]["relative_error"] = (abs(Fraction(approx) - exact) / abs(exact)) if exact else 0.0
        bench.records[-1]["relative_error"] = float(bench.records[-1]["relative_error"])
        # the arbitrary-precision path on fewer rows (one native object per value)
        sub = dict(bits=w, rows=nbig)
        objs_a = [mc.BigInt(v) for v in la[:nbig]]
        objs_b = [mc.BigInt(v) for v in lb[:nbig]]
        pa = (C.c_void_p * nbig)(*[o._h for o in objs_a])
        pb = (C.c_void_p * nbig)(*[o._h for o in objs_b])
        api = mc._api()

        def bigdot():
            refusal, err = mc._Refusal(), C.c_int32()
            h = api["bigint_dot"](pa, pb, nbig, None, C.byref(refusal), C.byref(err))
            v = mc.BigInt._wrap(h).to_int()
            return v
        got = bench.measure("dot product", "cpu-bigint", bigdot, repeats=5, **sub)
        assert got == sum(x * y for x, y in zip(la[:nbig], lb[:nbig]))
        sa, sb = a[:nbig].copy(), b[:nbig].copy()
        bench.measure("dot product", "cpu-wide192", lambda: mc.dot(sa, sb, carrier="wide"), repeats=5, **sub)
        bench.measure("dot product", "python", lambda: sum(x * y for x, y in zip(la[:nbig], lb[:nbig])), repeats=3, **sub)
    for carrier in ("i64", "i128"):
        bench.threshold(f"dot product {carrier} (bits per value, {n} rows)",
                        dict(bits=last_ok[carrier]) if last_ok[carrier] else None,
                        dict(bits=first_bad[carrier]) if first_bad[carrier] else None, rows=n)
    # Horner: degree d at many points
    pts = 2_000 if bench.quick else 20_000
    for d in (4, 8, 16):
        for w in (4, 8, 12):
            coef = [int(v) for v in rng.integers(-(1 << w), 1 << w, size=d + 1)]
            xs = [int(v) for v in rng.integers(-(1 << w), 1 << w, size=pts)]
            ref = None
            p = dict(degree=d, bits=w, points=pts)
            for carrier in ("i64", "i128", "bigint"):
                try:
                    got = bench.measure("horner", f"cpu-{carrier}", lambda: mc.horner(coef, xs, carrier=carrier),
                                        repeats=3, **p)
                    ref = ref or [sum(c * t ** k for k, c in enumerate(coef)) for t in xs[:20]]
                    assert got[:20] == ref
                except OverflowError:
                    bench.records.append(dict(case="horner", backend=f"cpu-{carrier}", phase="refusal", params=p,
                                              refused=True, reason="overflow (checked fixed width)"))
            bench.measure("horner", "python", lambda: [sum(c * t ** k for k, c in enumerate(coef)) for t in xs],
                          repeats=3, **p)


def limb_budget(bench):
    rng = random.Random(bench.seed + 2)
    budget = Budget(bytes=8 * 128)                      # 128 limbs per number
    last_ok = None
    for w in (1024, 2048, 4096, 8192, 16384):
        a, b = mc.BigInt(_rand(rng, w)), mc.BigInt(_rand(rng, w))
        p = dict(bits=w, limb_budget=128)
        try:
            a.mul(b, budget=budget)
            bench.measure("mul under a 128-limb budget", "cpu", lambda: a.mul(b, budget=budget), repeats=5, **p)
            last_ok = p
        except AlgebraBudgetError:
            bench.refusal("mul under a 128-limb budget", "cpu", lambda: a.mul(b, budget=budget), **p)
            bench.threshold("mul under a 128-limb budget (operand bits)", last_ok, p, limit_bytes=8 * 128)
            break


def gfp_kernels(bench):
    rng = np.random.default_rng(bench.seed + 3)
    sizes = (10_000, 100_000) if bench.quick else (10_000, 100_000, 1_000_000)
    for p in (P31, P61):
        F = mc.GF(p)
        for n in sizes:
            a = rng.integers(1, p, size=n, dtype=np.uint64)
            b = rng.integers(1, p, size=n, dtype=np.uint64)
            prm = dict(p=p, rows=n)
            for op in ("add", "mul", "inv"):
                for backend in bench.backends():
                    bench.measure(f"GF(p) {op}", backend, lambda: F.vec(op, a, b, backend=backend), repeats=5, **prm)
            for backend in bench.backends():
                bench.measure("GF(p) dot", backend, lambda: F.dot(a, b, backend=backend).values.v, repeats=5, **prm)
            if p < 1 << 31:
                ai, bi = a.astype(np.int64), b.astype(np.int64)
                bench.measure("GF(p) mul", "numpy", lambda: (ai * bi) % p, repeats=5, **prm)
                bench.measure("GF(p) dot", "numpy", lambda: int(((ai * bi) % p).sum() % p), repeats=5, **prm)
            elif n <= 100_000:
                la, lb = [int(v) for v in a], [int(v) for v in b]
                bench.measure("GF(p) mul", "python", lambda: [x * y % p for x, y in zip(la, lb)], repeats=3, **prm)
                bench.measure("GF(p) inv", "python", lambda: [pow(x, -1, p) for x in la[:10_000]], repeats=3,
                              **dict(prm, rows=10_000))
        for dim in ((128, 512) if bench.quick else (128, 512, 2048)):
            A = rng.integers(0, p, size=(dim, dim), dtype=np.uint64)
            x = rng.integers(0, p, size=dim, dtype=np.uint64)
            prm = dict(p=p, rows=dim, cols=dim)
            for backend in bench.backends():
                bench.measure("GF(p) matvec", backend, lambda: F.matvec(A, x, backend=backend), repeats=5, **prm)
            if p < 1 << 31:
                Ai, xi = A.astype(np.int64), x.astype(np.int64)
                bench.measure("GF(p) matvec", "numpy", lambda: ((Ai * xi[None, :]) % p).sum(axis=1) % p, repeats=5, **prm)
    # int64 rationals to tight enclosures in batch
    for n in sizes:
        num = rng.integers(-(1 << 62), 1 << 62, size=n, dtype=np.int64)
        den = rng.integers(1, 1 << 62, size=n, dtype=np.int64)
        prm = dict(rows=n)
        for backend in bench.backends():
            bench.measure("q64 -> tight enclosure", backend, lambda: mc.enclose_q64(num, den, backend=backend),
                          repeats=5, **prm)
        bench.measure("q64 -> nearest float (not an enclosure)", "numpy", lambda: num / den, repeats=5, **prm)
        for backend in bench.backends():
            bench.measure("exact 128-bit sum", backend, lambda: mc.sum_i64(num, backend=backend).values[0], repeats=5,
                          **prm)
        bench.measure("int64 sum (wraps)", "numpy", lambda: int(num.sum()), repeats=5, **prm)


def main():
    bench = Bench.from_args("FND-009", __doc__)
    bit_width(bench)
    crt_widths(bench)
    matched(bench)
    limb_budget(bench)
    gfp_kernels(bench)
    bench.note("Backends: cpu (native C), cpu-i64 / cpu-i128 (checked fixed width: refuse on overflow), "
               "cpu-wide192 (exact 192-bit accumulation), cpu-bigint (arbitrary precision), cpu-gfp4+crt (four "
               "62-bit residues and Chinese remaindering), cuda (an actual device, --device only), and the baselines "
               "python (Python int / Fraction), numpy-int64 (wraps silently), numpy-float64 (rounds).")
    bench.note("Python's float(Fraction) and NumPy's division round to nearest: they are timing baselines, not "
               "enclosures; the native enclosures are tight and certified.")
    bench.write()


if __name__ == "__main__":
    main()
