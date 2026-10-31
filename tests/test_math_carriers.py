# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-009: numeric carriers and exactness conversions (torbles/math_carriers.c, math_carriers_gpu.cu).

Oracles, all independent of the implementation: Python integers and fractions.Fraction arithmetic written here,
pow(a, -1, p), a Miller-Rabin test and trial division written here, math.nextafter for the tightness of enclosures,
and exact comparisons of interval endpoints with Fraction (Fraction(lo) <= value <= Fraction(hi)). Every batched
native result is also computed on the CUDA path (emulated here with scrambled thread order, TORBLES_EMU_SCRAMBLE=1)
and compared for equality with the CPU result. Budgets are tested at the limit and one past it. Seeds are fixed.
"""
import ctypes as C
import math
import os
import random
import unittest
from fractions import Fraction

import numpy as np

os.environ.setdefault("TORBLES_EMU_SCRAMBLE", "1")

from torbles import math_carriers as mc, math_gpu
from torbles.math_budgets import Budget
from torbles.math_domains import CarrierError
from torbles.math_errors import NoSolutionError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

INF = float("inf")
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
P61 = (1 << 61) - 1                 # a Mersenne prime
P62 = (1 << 62) - 57                # the largest prime below 2^62
PRIMES = [2, 3, 5, 7, 65537, (1 << 31) - 1, 998244353, 1000000007, P61, P62]


def limbs(x):
    return (abs(x).bit_length() + 63) // 64


def miller_rabin(n):
    """the oracle: trial division by the first primes, then strong probable-prime tests to bases 2..37 (exact below
    3.3e24, so for every 64-bit n), written independently of the native test"""
    if n < 2:
        return False
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for q in bases:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def enclosed(iv, value):
    """Fraction(lo) <= value <= Fraction(hi), infinite endpoints meaning unbounded"""
    lo_ok = iv.lo == -INF or Fraction(iv.lo) <= value
    hi_ok = iv.hi == INF or value <= Fraction(iv.hi)
    return lo_ok and hi_ok


def tight(iv, value):
    """[lo, hi] = [RD(value), RU(value)]: a point exactly when value is a double, else one ulp apart"""
    if value > Fraction(1.7976931348623157e308):
        return iv.lo == 1.7976931348623157e308 and iv.hi == INF
    if value < -Fraction(1.7976931348623157e308):
        return iv.hi == -1.7976931348623157e308 and iv.lo == -INF
    if not enclosed(iv, value):
        return False
    if Fraction(iv.lo) == value:
        return iv.hi == iv.lo
    return math.nextafter(iv.lo, INF) == iv.hi


def rand_int(rng, max_bits=4000):
    bits = rng.choice([0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 200, 1000, max_bits, rng.randrange(1, max_bits)])
    v = rng.getrandbits(bits) if bits else 0
    if rng.random() < 0.15 and bits:
        v |= 1 << (bits - 1)                   # a full top limb
    return -v if rng.random() < 0.45 else v


def rand_frac(rng, max_bits=600):
    d = 0
    while d == 0:
        d = rand_int(rng, max_bits)
    return Fraction(rand_int(rng, max_bits), d)


class Carriers(unittest.TestCase):
    # ------------------------------------------------------------------------------------------ big integers
    def test_fnd009_bigint_arithmetic_matches_python_integers(self):
        rng = random.Random(9001)
        for trial in range(1500):
            x, y = rand_int(rng), rand_int(rng)
            a, b = mc.BigInt(x), mc.BigInt(y)
            self.assertEqual(int(a + b), x + y)
            self.assertEqual(int(a - b), x - y)
            self.assertEqual(int(a * b), x * y)
            self.assertEqual(a.cmp(b), (x > y) - (x < y))
            self.assertEqual((a.sign, a.bit_length(), a.limbs), ((x > 0) - (x < 0), abs(x).bit_length(), limbs(x)))
            self.assertEqual(int(a.gcd(b)), math.gcd(x, y))
            if y:
                q, r = divmod(a, b)
                self.assertEqual((int(q), int(r)), divmod(x, y), (x, y))
                tq = abs(x) // abs(y) * (1 if (x >= 0) == (y > 0) else -1)
                q, r = a.divmod(b, "trunc")
                self.assertEqual((int(q), int(r)), (tq, x - tq * y))
                if x % y == 0:
                    self.assertEqual(int(a.divexact(b)), x // y)
                else:
                    with self.assertRaises(NoSolutionError):
                        a.divexact(b)
            else:
                with self.assertRaises(NoSolutionError):
                    divmod(a, b)
            k = rng.randrange(0, 700)
            self.assertEqual((int(a << k), int(a >> k)), (x << k, x >> k))
            self.assertEqual(a.mod_u64(P61), x % P61)
        for x in (0, 1, 2, 3, 10**40, 2**4000 + 17, (1 << 128) - 1):
            self.assertEqual(int(mc.BigInt(x).isqrt()), math.isqrt(x))
        with self.assertRaises(NoSolutionError):
            mc.BigInt(-4).isqrt()
        for base in (0, 1, -1, 2, -3, 7, 2**64 + 1, -(3**50)):
            for e in (0, 1, 2, 3, 17, 64):
                self.assertEqual(int(mc.BigInt(base) ** e), base ** e, (base, e))
        # Knuth's algorithm D: the rare add-back step (the 3-by-2 trial digit exceeds the true one by one)
        for Q in (1, 12345, (1 << 63) - 1, 1 << 63, (1 << 64) - 1):
            d = (1 << 191) + (1 << 64) - 1
            for x in (Q << 191, (Q << 191) + 7, (Q << 255) + 3):
                q, r = divmod(mc.BigInt(x), mc.BigInt(d))
                self.assertEqual((int(q), int(r)), divmod(x, d))
        # Lehmer's gcd on its worst case (consecutive Fibonacci numbers) and on large common factors
        fib = [0, 1]
        while len(fib) < 2500:
            fib.append(fib[-1] + fib[-2])
        for i in (90, 300, 1000, 2499):
            self.assertEqual(int(mc.BigInt(fib[i]).gcd(mc.BigInt(fib[i - 1]))), 1)
            self.assertEqual(int(mc.BigInt(fib[i]).gcd(mc.BigInt(fib[i // 2]))), math.gcd(fib[i], fib[i // 2]))
        for _ in range(100):
            g = rng.getrandbits(rng.randrange(1, 3000)) + 1
            x, y = g * rng.getrandbits(rng.randrange(1, 3000)), -g * rng.getrandbits(rng.randrange(1, 3000))
            self.assertEqual(int(mc.BigInt(x).gcd(mc.BigInt(y))), math.gcd(x, y))
        # batched elementwise operations, exact dot products and Horner evaluation
        xs = [rand_int(rng, 2000) for _ in range(40)]
        ys = [rand_int(rng, 2000) or 1 for _ in range(40)]
        for op, ref in (("add", lambda u, v: u + v), ("sub", lambda u, v: u - v), ("mul", lambda u, v: u * v),
                        ("div", lambda u, v: u // v), ("mod", lambda u, v: u % v), ("gcd", math.gcd)):
            self.assertEqual([int(z) for z in mc.bigint_batch(op, xs, ys)], [ref(u, v) for u, v in zip(xs, ys)], op)
        self.assertEqual(mc.dot(xs, ys, carrier="bigint"), sum(u * v for u, v in zip(xs, ys)))
        coef = [rand_int(rng, 300) for _ in range(9)]
        pts = [rand_int(rng, 300) for _ in range(5)]
        self.assertEqual(mc.horner(coef, pts, carrier="bigint"),
                         [sum(c * t ** k for k, c in enumerate(coef)) for t in pts])

    def test_fnd009_lossless_round_trips_preserve_values(self):
        rng = random.Random(9002)
        for x in [0, 1, -1, I64_MIN, I64_MAX] + [rng.randrange(I64_MIN, I64_MAX + 1) for _ in range(200)]:
            self.assertEqual(mc.BigInt.from_i64(x).to_i64(), x)                         # i64 -> bigint -> i64
            self.assertEqual(mc.convert(mc.I64(x), "i128").value, x)                     # i64 -> i128
            self.assertEqual(mc.convert(mc.convert(mc.I64(x), "bigint"), "i64").value, x)
            self.assertEqual(mc.convert(mc.I64(x), "q64").as_fraction(), x)
        for x in [-(1 << 127), (1 << 127) - 1, 0] + [rng.randrange(-(1 << 127), 1 << 127) for _ in range(200)]:
            self.assertEqual(mc.BigInt.from_i128(x).to_i128(), x)                       # i128 -> bigint -> i128
            self.assertEqual(mc.convert(mc.convert(mc.I128(x), "bigint"), "i128").value, x)
        for _ in range(300):
            x = rand_int(rng, 6000)
            b = mc.BigInt(x)
            self.assertEqual(int(b), x)                                                  # limbs both ways
            self.assertEqual(b.to_decimal(), str(x))                                     # decimal both ways
            self.assertEqual(int(mc.BigInt.from_decimal(str(x))), x)
            words = [((x >> (64 * i)) & ((1 << 64) - 1)) for i in range(limbs(x) + 1)]
            self.assertEqual(int(mc.BigInt.from_twos(words)), x)                          # two's complement words
            self.assertEqual(int(mc.BigRational.from_bigint(b).to_bigint()), x)          # bigint -> bigq -> bigint
            self.assertEqual(int(mc.convert(mc.convert(b, "bigq"), "bigint")), x)
        for text in ("+000123", "-0", "0", "-00000000000000000000000000000001", "9" * 400):
            self.assertEqual(int(mc.BigInt(text)), int(text))
        for bad in ("", "-", "+", "12a", " 1", "1 ", "0x10", "1_000", "--1"):
            with self.assertRaises(ValueError):
                mc.BigInt(bad)
        for _ in range(300):                                                             # Fraction <-> bigq
            f = rand_frac(rng, 3000)
            q = mc.BigRational(f)
            self.assertEqual(q.to_fraction(), f)
            self.assertEqual((q.numerator, q.denominator), (f.numerator, f.denominator))   # canonical
        for _ in range(300):                                                             # q64 -> bigq -> q64
            n, d = rng.randrange(I64_MIN, I64_MAX + 1), rng.choice([1, -1, 7, I64_MAX, I64_MIN + 1, rng.randrange(1, I64_MAX)])
            f = Fraction(n, d)
            q = mc.BigRational.from_q64((n, d))
            self.assertEqual(q.to_fraction(), f)
            if -(1 << 63) <= f.numerator < 1 << 63 and f.denominator < 1 << 63:
                back = q.to_q64()
                self.assertEqual((back.numerator, back.denominator), (f.numerator, f.denominator))
        for x in [0.0, -0.0, 1.0, 0.1, -2.5e-310, 5e-324, 1.7976931348623157e308, 3.141592653589793] + \
                 [rng.uniform(-1e6, 1e6) for _ in range(100)]:
            self.assertEqual(mc.BigRational.from_double(x).to_fraction(), Fraction(x))     # binary64 -> bigq exact
            self.assertEqual(mc.convert(x, "bigq").to_fraction(), Fraction(x))
            self.assertEqual(mc.convert(x, "interval").as_pair(), (x, x))                 # binary64 -> interval
        iv = mc.Interval(-1.5, 2.25)
        self.assertEqual(mc.convert(mc.convert(iv, "cinterval"), "interval"), iv)          # interval <-> cinterval
        s = mc.convert(Fraction(-7, 3), "series", order=4)
        self.assertEqual(s.coefficients, [Fraction(-7, 3), 0, 0, 0])
        self.assertEqual(s.truncate(4), s)
        F = mc.GF(P61)
        for x in (0, 1, P61 - 1, 12345):                                                # gfp -> lift -> gfp
            e = F(x)
            self.assertEqual(e.lift(), x)
            self.assertEqual(F(e.lift()), e)
            self.assertEqual(F(e.lift(symmetric=True)), e)
            self.assertEqual(int(mc.convert(e, "bigint")), x)
        # the first slice's int64 rationals (tr_math_rational) through the big carrier and back
        from torbles.mathematics import Context
        ctx = Context()
        x0 = ctx.symbol("x0")
        poly = Fraction(-9, 7) * x0 * x0 + Fraction(5, 3) * x0 + Fraction(I64_MAX, 2)
        for exps, coef in poly.terms:
            q = mc.BigRational.from_q64((coef.numerator, coef.denominator))
            back = q.to_q64()
            self.assertEqual(Fraction(back.numerator, back.denominator), coef)

    def test_fnd009_narrowing_fails_explicitly(self):
        for x in (I64_MAX, I64_MIN):
            self.assertEqual(mc.BigInt(x).to_i64(), x)
            self.assertEqual(mc.convert(mc.BigInt(x), "i64").value, x)
        for x in (I64_MAX + 1, I64_MIN - 1, 1 << 200, -(1 << 64)):
            with self.assertRaises(OverflowError):
                mc.BigInt(x).to_i64()
            with self.assertRaises(OverflowError):
                mc.convert(mc.BigInt(x), "i64")
            with self.assertRaises(OverflowError):
                mc.convert(mc.BigInt(x), "q64")
        for x in ((1 << 127) - 1, -(1 << 127)):
            self.assertEqual(mc.BigInt(x).to_i128(), x)
        for x in (1 << 127, -(1 << 127) - 1, 1 << 300):
            with self.assertRaises(OverflowError):
                mc.BigInt(x).to_i128()
            with self.assertRaises(OverflowError):
                mc.convert(mc.BigInt(x), "i128")
        with self.assertRaises(OverflowError):
            mc.convert(mc.I128(I64_MAX + 1), "i64")
        with self.assertRaises(OverflowError):
            mc.BigRational(Fraction(1, 1 << 63)).to_q64()
        with self.assertRaises(OverflowError):
            mc.BigRational(Fraction(1 << 63, 3)).to_q64()
        self.assertEqual(mc.BigRational(Fraction(I64_MIN, I64_MAX)).to_q64().as_fraction(), Fraction(I64_MIN, I64_MAX))
        for f in (Fraction(1, 2), Fraction(-7, 3), Fraction(1, 1 << 200)):
            with self.assertRaises(NoSolutionError):                   # a non-integral rational to an integer
                mc.BigRational(f).to_bigint()
            with self.assertRaises(NoSolutionError):
                mc.convert(mc.BigRational(f), "bigint")
        with self.assertRaises(NoSolutionError):
            mc.convert(mc.Q64(1, 3), "i64")
        self.assertEqual(mc.convert(mc.Q64(-12, 4), "i64").value, -3)
        with self.assertRaises(NoSolutionError):                       # a nonzero imaginary part
            mc.ComplexInterval(mc.Interval(1.0), mc.Interval(0.0, 1e-300)).real()
        with self.assertRaises(NoSolutionError):                       # unknown coefficients past the order
            mc.Series([1, 2], order=3).truncate(4)
        # checked fixed-width arithmetic refuses overflow instead of wrapping
        self.assertEqual((mc.I64(I64_MAX - 1) + mc.I64(1)).value, I64_MAX)
        for op in (lambda: mc.I64(I64_MAX) + mc.I64(1), lambda: mc.I64(I64_MIN) - mc.I64(1),
                   lambda: mc.I64(1 << 32) * mc.I64(1 << 31), lambda: -mc.I64(I64_MIN),
                   lambda: mc.I64(I64_MIN) // mc.I64(-1), lambda: mc.I128((1 << 127) - 1) + mc.I128(1),
                   lambda: mc.I128(1 << 64) * mc.I128(1 << 63), lambda: mc.Q64(I64_MAX, 1) + mc.Q64(1, 1),
                   lambda: mc.Q64(1, I64_MAX) * mc.Q64(1, 2)):
            with self.assertRaises(OverflowError):
                op()
        self.assertEqual((mc.I64(-7) // mc.I64(2)).value, -4)          # floor division, as documented
        self.assertEqual((mc.I128(1 << 64) * mc.I128(1 << 62)).value, 1 << 126)
        self.assertEqual((mc.Q64(1, 3) + mc.Q64(1, 6)).as_fraction(), Fraction(1, 2))
        with self.assertRaises(NoSolutionError):
            mc.Q64(1, 3) / mc.Q64(0, 1)
        with self.assertRaises(OverflowError):                         # marshalling never wraps
            mc.I64(1 << 63)
        with self.assertRaises(OverflowError):
            mc.I128(1 << 127)

    def test_fnd009_conversion_matrix_kinds_are_enforced(self):
        names = list(mc.CARRIERS)
        F = mc.GF(101)
        samples = {
            "i64": [mc.I64(5), mc.I64(-I64_MAX)], "i128": [mc.I128(-9), mc.I128(1 << 100)],
            "bigint": [mc.BigInt(7), mc.BigInt(-(1 << 300))], "q64": [mc.Q64(-3, 4), mc.Q64(8, 1)],
            "bigq": [mc.BigRational(Fraction(2, 7)), mc.BigRational(Fraction(-(1 << 200), 3))],
            "gfp": [F(3), F(100)], "binary64": [0.375, -1e300],
            "interval": [mc.Interval(1.0, 2.0)], "cinterval": [mc.ComplexInterval(mc.Interval(1.0), mc.Interval(0.0))],
            "series": [mc.Series([1, 2, 3], order=3)],
        }

        def exact(v):
            if isinstance(v, (mc.I64, mc.I128)):
                return Fraction(v.value)
            if isinstance(v, mc.Q64):
                return v.as_fraction()
            if isinstance(v, mc.BigInt):
                return Fraction(int(v))
            if isinstance(v, mc.BigRational):
                return v.to_fraction()
            if isinstance(v, float):
                return Fraction(v)
            raise TypeError
        exact_names = ("i64", "i128", "bigint", "q64", "bigq")
        seen = set()
        for src in names:
            for dst in names:
                kind = mc.conversion(src, dst)
                seen.add(kind)
                for v in samples[src]:
                    if kind == "forbidden":
                        with self.assertRaises(CarrierError, msg=(src, dst)):
                            mc.convert(v, dst, field=F, order=3)
                        continue
                    if kind == "identity":
                        self.assertIs(mc.convert(v, dst), v)
                        continue
                    try:
                        out = mc.convert(v, dst, field=F, order=3)
                    except (OverflowError, NoSolutionError):
                        self.assertEqual(kind, "narrowing", (src, dst))     # only narrowing may refuse a value
                        if src in exact_names:
                            x = exact(v)
                            fits = x.denominator == 1 if dst in ("i64", "i128", "bigint") else True
                            bound = {"i64": 63, "q64": 63, "i128": 127}.get(dst)
                            if fits and bound:
                                fits = (-(1 << bound) <= x.numerator < 1 << bound) and x.denominator < 1 << 63
                            self.assertFalse(fits, (src, dst, x))            # refused exactly when it does not fit
                        continue
                    self.assertEqual(mc.carrier_of(out), dst, (src, dst))
                    if kind in ("lossless", "narrowing") and src in exact_names + ("binary64",) and dst in exact_names:
                        self.assertEqual(exact(out), exact(v), (src, dst))
                    if src in exact_names + ("binary64",) and dst in ("interval", "cinterval"):
                        iv = out if dst == "interval" else out.re
                        self.assertTrue(tight(iv, exact(v)), (src, dst))
                        self.assertTrue(out.approximate)
                    if kind == "reduction":
                        x = exact(v)
                        self.assertEqual(out.v, x.numerator * pow(x.denominator, -1, 101) % 101)
                    if kind == "lift":
                        self.assertEqual(exact(out), v.v)
                    if dst == "series" and src != "series":
                        self.assertEqual(out.coefficients, [exact(v), 0, 0])
        self.assertEqual(seen, set(mc.KINDS))
        # the documented forbidden directions, one by one
        for src, dst in (("interval", "bigq"), ("interval", "i64"), ("cinterval", "bigq"), ("gfp", "bigq"),
                         ("gfp", "q64"), ("gfp", "interval"), ("bigq", "binary64"),
                         ("bigint", "binary64"), ("series", "bigq"), ("binary64", "bigint")):
            self.assertEqual(mc.conversion(src, dst), "forbidden", (src, dst))
        self.assertEqual(mc.conversion("bigq", "gfp"), "reduction")
        self.assertEqual(mc.conversion("gfp", "bigint"), "lift")
        with self.assertRaises(CarrierError):
            mc.Interval(1.0, 2.0).to_exact()
        with self.assertRaises(CarrierError):                          # GF(p) -> GF(q)
            mc.GF(103)(F(5))
        with self.assertRaises(CarrierError):
            mc.convert(F(5), "gfp", field=mc.GF(103))
        with self.assertRaises(TypeError):                             # no implicit lift of a residue
            int(F(5))
        with self.assertRaises(TypeError):                             # no implicit rounding of an exact rational
            float(mc.BigRational(Fraction(1, 3)))

    def test_fnd009_incompatible_carrier_operations_fail(self):
        F, G = mc.GF(101), mc.GF(103)
        a, q = mc.BigInt(5), mc.BigRational(Fraction(1, 2))
        mixes = [lambda: a + q, lambda: q + a, lambda: a + Fraction(1, 2), lambda: a * 1.5, lambda: q + 0.5,
                 lambda: F(1) + G(1), lambda: F(1) * G(2), lambda: F(1) == G(1), lambda: F(1) + 1,
                 lambda: F(1) + Fraction(1, 2), lambda: F(1) + a, lambda: q * F(2),
                 lambda: mc.Interval(1.0) + 1, lambda: mc.Interval(1.0) + 1.0, lambda: mc.Interval(1.0) * q,
                 lambda: mc.I64(1) + mc.I128(1), lambda: mc.Q64(1, 2) + Fraction(1, 2), lambda: mc.I64(3) * 2,
                 lambda: mc.ComplexInterval(mc.Interval(1.0)) + mc.Interval(1.0),
                 lambda: mc.Series([1], 2) + mc.Series([1], 3), lambda: mc.Series([1], 2) * mc.Series([1], 3),
                 lambda: mc.Series([1], 2)(mc.Series([0, 1], 3)), lambda: mc.Series([1], 2) == mc.Series([1], 3),
                 lambda: mc.Series([1], 2) + q, lambda: mc.BigInt(F(3)), lambda: mc.BigRational(F(3))]
        for i, fn in enumerate(mixes):
            with self.assertRaises(CarrierError, msg=i):
                fn()
        with self.assertRaises(NoSolutionError):                       # 3/101 has no image in GF(101)
            F(Fraction(3, 101))
        # Python's own exact numbers are members of the operand's carrier, and only there
        self.assertEqual(int(a + 3), 8)
        self.assertEqual((q + Fraction(1, 3)).to_fraction(), Fraction(5, 6))
        self.assertEqual((q + 1).to_fraction(), Fraction(3, 2))
        self.assertEqual(mc.Series([1, 1], 3).truncate(2) + mc.Series([1], 2), mc.Series([2, 1], 2))
        # natively, a handle of another carrier is refused by type (TR_MATH_CARRIER_ERROR), not misread
        api = mc._api()
        refusal, err = mc._Refusal(), C.c_int32()
        h = api["bigint_add"](q._h, a._h, None, C.byref(refusal), C.byref(err))
        self.assertEqual((h, err.value), (None, -7))
        h = api["bigq_add"](a._h, q._h, None, C.byref(refusal), C.byref(err))
        self.assertEqual((h, err.value), (None, -7))
        s = mc.Series([1], 2)
        h = api["series_add"](s._h, q._h, None, C.byref(refusal), C.byref(err))
        self.assertEqual((h, err.value), (None, -7))
        self.assertEqual(api["bigint_sign"](s._h), -7)
        self.assertEqual(api["carrier_combine"](mc.CARRIERS["gfp"], mc.CARRIERS["gfp"]), mc.CARRIERS["gfp"])
        self.assertEqual(api["carrier_combine"](mc.CARRIERS["gfp"], mc.CARRIERS["bigq"]), -7)

    # ------------------------------------------------------------------------------------------ rationals
    def test_fnd009_bigq_matches_fraction(self):
        rng = random.Random(9003)
        for _ in range(500):
            f, g = rand_frac(rng), rand_frac(rng)
            a, b = mc.BigRational(f), mc.BigRational(g)
            self.assertEqual((a + b).to_fraction(), f + g)
            self.assertEqual((a - b).to_fraction(), f - g)
            self.assertEqual((a * b).to_fraction(), f * g)
            self.assertEqual(a.cmp(b), (f > g) - (f < g))
            self.assertEqual((-a).to_fraction(), -f)
            r = a + b
            self.assertEqual((r.numerator, r.denominator), ((f + g).numerator, (f + g).denominator))
            if g:
                self.assertEqual((a / b).to_fraction(), f / g)
                self.assertEqual(b.inverse().to_fraction(), 1 / g)
            else:
                with self.assertRaises(NoSolutionError):
                    a / b
                with self.assertRaises(NoSolutionError):
                    b.inverse()
            e = rng.randrange(-6, 7)
            if f or e >= 0:
                self.assertEqual((a ** e).to_fraction(), f ** e)
            else:
                with self.assertRaises(NoSolutionError):
                    a ** e
        for n, d in ((6, -4), (-6, -4), (0, -5), (1 << 300, 1 << 299), (-(3 ** 100) * 7, 3 ** 99 * 14)):
            q = mc.BigRational(n, d)
            self.assertEqual((q.numerator, q.denominator), (Fraction(n, d).numerator, Fraction(n, d).denominator))
        with self.assertRaises(NoSolutionError):
            mc.BigRational(1, 0)
        for x in (0.5, -1e300, 5e-324, 0.1):
            q = mc.BigRational(Fraction(1, 3))
            self.assertEqual(q.cmp_double(x), (Fraction(1, 3) > Fraction(x)) - (Fraction(1, 3) < Fraction(x)))
        self.assertEqual(mc.BigRational(Fraction(1, 3)).cmp_double(INF), -1)

    # ------------------------------------------------------------------------------------------ numerical carriers
    def test_fnd009_enclosures_are_tight_and_certified(self):
        rng = random.Random(9004)
        values = [Fraction(0), Fraction(1, 3), Fraction(-2, 3), Fraction(1, 10), Fraction(2 ** 53 + 1),
                  Fraction(-(2 ** 53) - 1), Fraction(1, 2 ** 1074), Fraction(1, 2 ** 1075), Fraction(3, 2 ** 1076),
                  Fraction(1, 2 ** 1200), Fraction(2 ** 1024), Fraction(-(2 ** 1024) + 1), Fraction(2 ** 1024 - 2 ** 970),
                  Fraction(2 ** 1023 * 3, 2), Fraction(1, 3 * 2 ** 1022), Fraction(10 ** 400, 3 ** 700)]
        values += [rand_frac(rng, 1500) for _ in range(400)]
        values += [Fraction(rng.getrandbits(80), rng.getrandbits(70) + 1) for _ in range(200)]
        for v in values:
            iv = mc.BigRational(v).enclose()
            self.assertTrue(tight(iv, v), (v, iv))
            self.assertTrue(mc.BigRational(v).in_interval(iv))
            if v.denominator == 1:
                self.assertEqual(mc.BigInt(v.numerator).enclose(), iv)
        for _ in range(400):                                          # q64 inline enclosure == bigq enclosure
            n = rng.choice([I64_MIN, I64_MAX, 0, 1, -1, rng.randrange(I64_MIN, I64_MAX + 1), rng.getrandbits(20)])
            d = rng.choice([1, -1, 3, I64_MAX, I64_MIN, rng.randrange(1, I64_MAX), -rng.randrange(1, 1 << 30)])
            v = Fraction(n, d)
            a = mc.Interval.enclose(mc.Q64(n, d)) if -(1 << 63) <= v.numerator < 1 << 63 and v.denominator < 1 << 63 \
                else None
            b = mc.BigRational(v).enclose()
            self.assertTrue(tight(b, v))
            batch = mc.enclose_q64([n], [d]).values
            self.assertEqual((batch[0][0], batch[1][0]), b.as_pair(), (n, d))
            if a is not None:
                self.assertEqual(a, b)
        for x in (I64_MIN, I64_MAX, 2 ** 53 + 1, -3, 0):
            self.assertEqual(mc.Interval.enclose(mc.I64(x)), mc.BigRational(x).enclose())
        for x in (-(1 << 127), (1 << 127) - 1, (1 << 100) + 1):
            self.assertEqual(mc.Interval.enclose(mc.I128(x)), mc.BigRational(x).enclose())
        self.assertEqual(mc.Interval.enclose(Fraction(1, 4)).as_pair(), (0.25, 0.25))     # a double: a point
        self.assertTrue(mc.Interval.enclose(Fraction(1, 3)).contains(Fraction(1, 3)))
        self.assertTrue(mc.Interval.enclose(Fraction(1, 3)).contains(Fraction(1, 3) + Fraction(1, 2 ** 60)))
        self.assertFalse(mc.Interval.enclose(Fraction(1, 3)).contains(Fraction(1, 3) + Fraction(1, 2 ** 50)))
        with self.assertRaises(ValueError):
            mc.enclose_q64([1], [0])

    def test_fnd009_numerical_results_disclose_precision_and_state(self):
        for name in mc.CARRIERS:
            d = mc.describe(name)
            self.assertEqual(d["carrier"], name)
            self.assertTrue(d["certified"])
            self.assertEqual(d["approximate"], name in ("interval", "cinterval"))
        self.assertEqual(mc.describe("interval")["precision_bits"], 53)
        self.assertEqual(mc.describe("i128")["precision_bits"], 128)
        self.assertIsNone(mc.describe("bigq")["precision_bits"])            # unbounded, budgeted
        # an irrational result is an enclosure flagged approximate, never a rational
        r = mc.Interval.enclose(2).sqrt()
        self.assertIsInstance(r, mc.Interval)
        self.assertEqual((r.approximate, r.certified, r.precision_bits), (True, True, 53))
        self.assertTrue(Fraction(r.lo) ** 2 < 2 < Fraction(r.hi) ** 2)
        self.assertEqual(math.nextafter(r.lo, INF), r.hi)                     # one ulp: the tightest enclosure
        self.assertEqual(r.state, dict(lo=r.lo, hi=r.hi, approximate=True, certified=True, precision_bits=53))
        third = mc.BigRational(Fraction(1, 3)).enclose()
        self.assertTrue(third.approximate and third.lo < third.hi)            # 1/3 is not presented as a double
        z = mc.ComplexInterval(mc.Interval.enclose(Fraction(1, 3)), mc.Interval.enclose(Fraction(2, 7)))
        for w in (z * z, z / z, abs(z), z.inverse(), z.sqr()):
            self.assertTrue(w.approximate and w.certified)
        self.assertTrue(enclosed((z / z).re, Fraction(1)) and enclosed((z / z).im, Fraction(0)))
        self.assertGreater(third.width(), 0)
        self.assertEqual(mc.Interval.enclose(Fraction(1, 4)).width(), 0.0)
        # exact carriers stay exact: the same operations on rationals return exact rationals
        self.assertEqual((mc.BigRational(Fraction(1, 3)) * 3).to_fraction(), 1)
        s = mc.Series([0, 1], order=5).exp()
        self.assertEqual(s.order, 5)
        self.assertIn("O(x^5)", repr(s))                                      # the truncation order is disclosed
        self.assertEqual(mc.Interval.enclose(Fraction(1, 3)).cmp(mc.Interval.enclose(Fraction(1, 3))), "unknown")
        self.assertEqual(mc.Interval(1.0).cmp(mc.Interval(2.0)), "less")
        self.assertEqual(mc.Interval(1.0).cmp(mc.Interval(1.0)), "equal")

    def test_fnd009_interval_arithmetic_encloses_exact_results(self):
        rng = random.Random(9005)

        def rand_iv():
            kind = rng.random()
            if kind < 0.1:
                x = rng.uniform(-10, 10)
                return mc.Interval(x, x)
            a, b = sorted(rng.uniform(-1e3, 1e3) * 10 ** rng.randrange(-5, 5) for _ in range(2))
            if kind < 0.2:
                return mc.Interval(abs(a), abs(a) + abs(b))
            return mc.Interval(a, b)

        def points(iv):
            lo, hi = Fraction(iv.lo), Fraction(iv.hi)
            return [lo, hi, (lo + hi) / 2, lo + (hi - lo) / 3]
        ops = {"add": (lambda x, y: x + y, lambda a, b: a + b), "sub": (lambda x, y: x - y, lambda a, b: a - b),
               "mul": (lambda x, y: x * y, lambda a, b: a * b)}
        for _ in range(600):
            a, b = rand_iv(), rand_iv()
            for name, (exact, op) in ops.items():
                out = op(a, b)
                for x in points(a):
                    for y in points(b):
                        self.assertTrue(enclosed(out, exact(x, y)), (name, a, b, x, y))
            if not (b.lo <= 0 <= b.hi):
                out = a / b
                for x in points(a):
                    for y in points(b):
                        self.assertTrue(enclosed(out, x / y), ("div", a, b))
            else:
                with self.assertRaises(NoSolutionError):
                    a / b
            for out, exact in ((a.sqr(), lambda x: x * x), (abs(a), abs), (a ** 3, lambda x: x ** 3),
                               (a ** 4, lambda x: x ** 4), (-a, lambda x: -x)):
                for x in points(a):
                    self.assertTrue(enclosed(out, exact(x)))
            if a.lo >= 0:
                s = a.sqrt()
                for x in points(a):
                    self.assertTrue(Fraction(s.lo) ** 2 <= x <= Fraction(s.hi) ** 2)
            else:
                with self.assertRaises(NoSolutionError):
                    a.sqrt()
        # tightness: operations on points give the tight enclosure of the exact result
        for _ in range(500):
            x, y = Fraction(rng.uniform(-1e6, 1e6)), Fraction(rng.uniform(-1e6, 1e6)) or Fraction(1)
            px, py = mc.Interval(float(x)), mc.Interval(float(y))
            for out, v in ((px + py, x + y), (px - py, x - y), (px * py, x * y), (px / py, x / y)):
                self.assertEqual(out, mc.BigRational(v).enclose(), v)
        self.assertEqual((mc.Interval(3.0) + mc.Interval(4.0)).as_pair(), (7.0, 7.0))
        self.assertEqual((mc.Interval(1.0) / mc.Interval(3.0)), mc.BigRational(Fraction(1, 3)).enclose())
        # overflow, unbounded endpoints, 0 * inf, the subnormal range
        big = mc.Interval(1e308, 1.5e308)
        self.assertEqual((big + big).as_pair(), (1.7976931348623157e308, INF))
        self.assertEqual((big * mc.Interval(-2.0)).as_pair(), (-INF, -1.7976931348623157e308))
        self.assertEqual((mc.Interval(0.0) * mc.Interval(1.0, INF)).as_pair(), (0.0, 0.0))
        self.assertEqual((mc.Interval(0.0, 1.0) * mc.Interval(2.0, INF)).as_pair(), (0.0, INF))
        self.assertEqual((mc.Interval(1.0, 2.0) / mc.Interval(4.0, INF)).as_pair(), (0.0, 0.5))
        tiny = mc.Interval(5e-324, 1e-320)
        t = tiny * tiny
        self.assertTrue(t.lo <= 0.0 and t.hi > 0.0 and enclosed(t, Fraction(5e-324) ** 2))
        q = mc.Interval(1e-310) / mc.Interval(3.0)
        self.assertTrue(enclosed(q, Fraction(1e-310) / 3))
        self.assertEqual(mc.Interval(4.0).sqrt().as_pair(), (2.0, 2.0))
        self.assertEqual(mc.Interval(-2.0, 3.0).hull(mc.Interval(5.0, 6.0)).as_pair(), (-2.0, 6.0))
        self.assertEqual(mc.Interval(-2.0, 3.0).intersect(mc.Interval(1.0, 6.0)).as_pair(), (1.0, 3.0))
        with self.assertRaises(NoSolutionError):
            mc.Interval(-2.0, 0.0).intersect(mc.Interval(1.0, 6.0))
        with self.assertRaises(NoSolutionError):
            mc.Interval(-1.0, 1.0).inverse()
        with self.assertRaises(NoSolutionError):
            mc.Interval(-1.0, 1.0) ** -2
        inv2 = mc.Interval(2.0, 4.0) ** -2
        self.assertTrue(enclosed(inv2, Fraction(1, 4)) and enclosed(inv2, Fraction(1, 16)))
        for lo, hi in ((2.0, 1.0), (float("nan"), 1.0), (INF, INF), (-INF, -INF), (2 ** 60 + 1, 2 ** 61)):
            with self.assertRaises(ValueError):
                mc.Interval(lo, hi)
        with self.assertRaises(TypeError):                         # exact values are enclosed, never rounded
            mc.Interval(Fraction(1, 3))
        self.assertEqual(mc.Interval(2 ** 60, 2 ** 61).as_pair(), (2.0 ** 60, 2.0 ** 61))
        # complex rectangles: corner and interior points of the operands map inside the result
        for _ in range(300):
            z = mc.ComplexInterval(rand_iv(), rand_iv())
            w = mc.ComplexInterval(rand_iv(), rand_iv())
            zs = [(x, y) for x in points(z.re)[:3] for y in points(z.im)[:3]]
            ws = [(x, y) for x in points(w.re)[:3] for y in points(w.im)[:3]]
            prod, s = z * w, z + w
            for (a1, b1) in zs[::2]:
                for (a2, b2) in ws[::2]:
                    self.assertTrue(enclosed(prod.re, a1 * a2 - b1 * b2) and enclosed(prod.im, a1 * b2 + b1 * a2))
                    self.assertTrue(enclosed(s.re, a1 + a2) and enclosed(s.im, b1 + b2))
            m = abs(z)
            for (a1, b1) in zs:
                self.assertTrue(Fraction(m.lo) ** 2 <= a1 * a1 + b1 * b1 <= Fraction(m.hi) ** 2)
            contains0 = w.re.lo <= 0 <= w.re.hi and w.im.lo <= 0 <= w.im.hi
            if contains0:
                with self.assertRaises(NoSolutionError):
                    z / w
                continue
            try:
                quo = z / w
            except NoSolutionError:
                continue                                 # |w|^2 underflows: refused conservatively
            for (a1, b1) in zs[::3]:
                for (a2, b2) in ws[::3]:
                    den = a2 * a2 + b2 * b2
                    self.assertTrue(enclosed(quo.re, (a1 * a2 + b1 * b2) / den))
                    self.assertTrue(enclosed(quo.im, (b1 * a2 - a1 * b2) / den))
        self.assertEqual(mc.ComplexInterval(mc.Interval(3.0), mc.Interval(4.0)).__abs__().as_pair(), (5.0, 5.0))

    # ------------------------------------------------------------------------------------------ GF(p)
    def test_fnd009_gfp_arithmetic_matches_python(self):
        rng = random.Random(9006)
        for n in [0, 1, 2, 3, 4, 561, 41041, 825265, 3215031751, 3825123056546413051, P61, P62, (1 << 62) + 135,
                  (1 << 63) - 25, (1 << 64) - 59, (1 << 64) - 1] + [rng.getrandbits(64) for _ in range(300)] + \
                 [rng.getrandbits(20) for _ in range(300)]:
            self.assertEqual(mc.GF.is_prime(n), miller_rabin(n), n)
        for n in range(0, 3000):
            self.assertEqual(mc.GF.is_prime(n), n > 1 and all(n % k for k in range(2, math.isqrt(n) + 1)), n)
        for bad in (0, 1, 4, 561, 3825123056546413051, (1 << 62) - 1):
            with self.assertRaises(ValueError):
                mc.GF(bad)
        for big in ((1 << 62) + 135, (1 << 63) - 25, (1 << 64) - 59):
            with self.assertRaises(UnsupportedError):
                mc.GF(big)
        for p in PRIMES:
            F = mc.GF(p)
            for _ in range(150):
                x, y = rng.randrange(p), rng.randrange(p)
                a, b = F(x), F(y)
                self.assertEqual(((a + b).v, (a - b).v, (a * b).v, (-a).v), ((x + y) % p, (x - y) % p, x * y % p, -x % p))
                if y:
                    self.assertEqual((a / b).v, x * pow(y, -1, p) % p)
                    self.assertEqual(b.inverse().v, pow(y, -1, p))
                else:
                    with self.assertRaises(NoSolutionError):
                        a / b
                    with self.assertRaises(NoSolutionError):
                        b.inverse()
                e = rng.choice([0, 1, 2, p - 1, p - 2, rng.getrandbits(64) - (1 << 63)])
                if x or e >= 0:
                    self.assertEqual((a ** e).v, pow(x, e, p) if e >= 0 else pow(pow(x, -1, p), -e, p))
                else:
                    with self.assertRaises(NoSolutionError):
                        a ** e
                self.assertEqual(a.lift(), x)
                self.assertEqual(a.lift(symmetric=True), x - p if x > p // 2 else x)
                n = rand_int(rng, 500)
                self.assertEqual(F(n).v, n % p)
                f = rand_frac(rng, 300)
                if f.denominator % p:
                    self.assertEqual(F(f).v, f.numerator * pow(f.denominator, -1, p) % p)
                else:
                    with self.assertRaises(NoSolutionError):
                        F(f)
                if -(1 << 63) <= f.numerator < 1 << 63 and f.denominator < 1 << 63 and f.denominator % p:
                    self.assertEqual(F(mc.Q64(f.numerator, f.denominator)).v, F(f).v)
                m = rng.randrange(-(1 << 127), 1 << 127)
                self.assertEqual(F(mc.I128(m)).v, m % p)
        F = mc.GF(7)
        with self.assertRaises(NoSolutionError):
            F(Fraction(1, 14))
        self.assertEqual(F(Fraction(14, 3)).v, 0)

    def test_fnd009_gfp_batched_kernels_cuda_equals_cpu(self):
        rng = np.random.default_rng(9007)
        info = math_gpu.device()
        for p in (2, 3, 65537, 998244353, P61, P62):
            F = mc.GF(p)
            for n in (0, 1, 7, 300, 5000):
                a = rng.integers(0, p, size=n, dtype=np.uint64)
                b = rng.integers(0, p, size=n, dtype=np.uint64)
                b[b == 0] = 1 if p > 1 else 0
                for op in ("add", "sub", "mul", "div", "neg", "inv"):
                    x = a.copy()
                    if op == "inv":
                        x[x == 0] = 1
                    cpu = F.vec(op, x, b).values
                    gpu = F.vec(op, x, b, backend="cuda")
                    self.assertEqual(gpu.backend, "cuda-emulated" if info["emulated"] else "cuda")
                    self.assertTrue(np.array_equal(cpu, gpu.values), (p, n, op))
                    xs, ys = [int(v) for v in x[:50]], [int(v) for v in b[:50]]
                    ref = {"add": lambda: [(u + v) % p for u, v in zip(xs, ys)],
                           "sub": lambda: [(u - v) % p for u, v in zip(xs, ys)],
                           "mul": lambda: [u * v % p for u, v in zip(xs, ys)],
                           "div": lambda: [u * pow(v, -1, p) % p for u, v in zip(xs, ys)],
                           "neg": lambda: [-u % p for u in xs], "inv": lambda: [pow(u, -1, p) for u in xs]}[op]()
                    self.assertEqual([int(v) for v in cpu[:50]], ref, (p, op))
                dot = sum(int(u) * int(v) for u, v in zip(a, b)) % p
                self.assertEqual(F.dot(a, b).values.v, dot)
                self.assertEqual(F.dot(a, b, backend="cuda").values.v, dot)
                xi = rng.integers(I64_MIN, I64_MAX, size=n, dtype=np.int64, endpoint=True)
                if n:
                    xi[0] = I64_MIN
                self.assertTrue(np.array_equal(F.reduce(xi).values, F.reduce(xi, backend="cuda").values))
                self.assertEqual([int(v) for v in F.reduce(xi).values[:40]], [int(v) % p for v in xi[:40]])
            for rows, cols in ((0, 5), (3, 0), (1, 1), (5, 7), (40, 300), (3, 1000), (300, 3)):
                A = rng.integers(0, p, size=(rows, cols), dtype=np.uint64)
                x = rng.integers(0, p, size=cols, dtype=np.uint64)
                cpu, gpu = F.matvec(A, x).values, F.matvec(A, x, backend="cuda").values
                self.assertTrue(np.array_equal(cpu, gpu), (p, rows, cols))
                ref = [sum(int(A[i, j]) * int(x[j]) for j in range(cols)) % p for i in range(min(rows, 8))]
                self.assertEqual([int(v) for v in cpu[:8]], ref)
        # invalid residues and zero divisors, refused identically (INVALID before NO_SOLUTION), nothing written
        F = mc.GF(101)
        bad = np.array([1, 101, 3], dtype=np.uint64)
        zero = np.array([1, 0, 3], dtype=np.uint64)
        for backend in ("cpu", "cuda"):
            for op, a, b in (("add", bad, zero), ("div", zero, bad), ("inv", bad, None)):
                with self.assertRaises(ValueError):
                    F.vec(op, a, b, backend=backend)
            with self.assertRaises(NoSolutionError):
                F.vec("div", zero, zero, backend=backend)
            with self.assertRaises(NoSolutionError):
                F.vec("inv", zero, backend=backend)
            with self.assertRaises(ValueError):
                F.dot(bad, zero, backend=backend)
            with self.assertRaises(ValueError):
                F.matvec(np.array([[1, 2], [3, 101]], dtype=np.uint64), np.array([1, 1], dtype=np.uint64), backend=backend)
            with self.assertRaises(ValueError):
                F.matvec(np.array([[1, 2]], dtype=np.uint64), np.array([1, 500], dtype=np.uint64), backend=backend)
        # the device copies are budgeted: at the limit and one past
        a = rng.integers(0, 101, size=1000, dtype=np.uint64)
        A = rng.integers(0, 101, size=(20, 50), dtype=np.uint64)
        for call, need in ((lambda bud: F.vec("mul", a, a, backend="cuda", budget=bud), mc.device_bytes("gfp_vec", 1000)),
                           (lambda bud: F.vec("neg", a, backend="cuda", budget=bud), mc.device_bytes("gfp_vec", 1000, flag=True)),
                           (lambda bud: F.dot(a, a, backend="cuda", budget=bud), mc.device_bytes("gfp_dot", 1000)),
                           (lambda bud: F.matvec(A, a[:50], backend="cuda", budget=bud), mc.device_bytes("gfp_matvec", 20, 50)),
                           (lambda bud: F.reduce(a.astype(np.int64), backend="cuda", budget=bud), mc.device_bytes("gfp_reduce", 1000))):
            call(Budget(bytes=need))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                call(Budget(bytes=need - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
        self.assertEqual(mc.device_bytes("gfp_vec", 1000), 24008)
        self.assertEqual(mc.device_bytes("gfp_matvec", 20, 50), 8 * 1000 + 8 * 50 + 24 * 20 + 8)
        F.matvec(A, a[:50], backend="cuda", budget=Budget(tensor=1000))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            F.matvec(A, a[:50], backend="cuda", budget=Budget(tensor=999))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("tensor", 1000))

    def test_fnd009_crt_and_rational_reconstruction(self):
        rng = random.Random(9008)
        pool = [p for p in (P62 - 2 * k for k in range(0, 4000)) if miller_rabin(p)][:60]
        for k in (1, 2, 3, 10, 60):
            fields = [mc.GF(p) for p in pool[:k]]
            M = math.prod(pool[:k])
            for _ in range(20):
                x = rng.randrange(M)
                res = [x % p for p in pool[:k]]
                got, mod = mc.crt(res, fields)
                self.assertEqual((got, mod), (x, M))
                sym, _ = mc.crt(res, fields, symmetric=True)
                self.assertEqual(sym, x - M if 2 * x > M else x)
                self.assertTrue(-M < 2 * sym <= M)
                y, m = res[0], pool[0]                              # the incremental steps reach the same value
                for p, r in zip(pool[1:k], res[1:]):
                    y, m = mc.crt_combine(y, m, r, mc.GF(p))
                self.assertEqual((y, m), (x, M))
        self.assertEqual(mc.crt([2, 3], [mc.GF(5), mc.GF(7)]), (17, 35))
        with self.assertRaises(ValueError):                         # a repeated prime
            mc.crt([1, 2], [mc.GF(7), mc.GF(7)])
        with self.assertRaises(ValueError):                         # a residue >= p
            mc.crt([7], [mc.GF(7)])
        with self.assertRaises(ValueError):                         # combine needs x < m and m prime to p
            mc.crt_combine(9, 7, 1, mc.GF(5))
        with self.assertRaises(ValueError):
            mc.crt_combine(3, 14, 1, mc.GF(7))
        # rational reconstruction recovers every fraction with 2 N D < m
        for k in (1, 2, 4, 8):
            m = math.prod(pool[:k])
            bound = math.isqrt((m - 1) // 2)
            for _ in range(40):
                d = rng.randrange(1, min(bound, 1 << (30 * k)) + 1)
                n = rng.randrange(-min(bound, 1 << (30 * k)), min(bound, 1 << (30 * k)) + 1)
                f = Fraction(n, d)
                if math.gcd(f.denominator, m) != 1:
                    continue
                u = f.numerator * pow(f.denominator, -1, m) % m
                self.assertEqual(mc.rational_reconstruct(u, m), f)
                # the multi-modular pipeline: images mod each prime, CRT, reconstruction
                images = [mc.GF(p)(f).v for p in pool[:k]]
                self.assertEqual(mc.rational_reconstruct(mc.crt(images, [mc.GF(p) for p in pool[:k]])[0], m), f)
        # a residue with no fraction inside the bounds: the oracle searches every candidate
        m = 1009
        for u in range(0, m, 37):
            N = D = 5
            cands = [Fraction(n, d) for d in range(1, D + 1) for n in range(-N, N + 1)
                     if (n - u * d) % m == 0 and math.gcd(n, d) == 1 and math.gcd(d, m) == 1]
            if cands:
                self.assertIn(mc.rational_reconstruct(u, m, N, D), cands)
            else:
                with self.assertRaises(NoSolutionError):
                    mc.rational_reconstruct(u, m, N, D)
        self.assertEqual(mc.rational_reconstruct(0, 97), 0)
        with self.assertRaises(ValueError):
            mc.rational_reconstruct(1, 1)

    # ------------------------------------------------------------------------------------------ series
    def test_fnd009_series_match_fraction_oracles(self):
        rng = random.Random(9009)

        def conv(a, b, n):
            return [sum(a[k] * b[i - k] for k in range(i + 1)) for i in range(n)]

        def rand_coefs(n, c0=None):
            c = [Fraction(rng.randrange(-9, 10), rng.randrange(1, 6)) for _ in range(n)]
            if c0 is not None:
                c[0] = Fraction(c0)
            return c
        for n in (1, 2, 3, 5, 8):
            for _ in range(6):
                a, b = rand_coefs(n), rand_coefs(n)
                A, B = mc.Series(a, n), mc.Series(b, n)
                self.assertEqual((A + B).coefficients, [x + y for x, y in zip(a, b)])
                self.assertEqual((A - B).coefficients, [x - y for x, y in zip(a, b)])
                self.assertEqual((A * B).coefficients, conv(a, b, n))
                self.assertEqual(A.scale(Fraction(-2, 3)).coefficients, [x * Fraction(-2, 3) for x in a])
                self.assertEqual(A.integral().coefficients, [0] + [a[k] / (k + 1) for k in range(n)])
                self.assertEqual(A.integral().order, n + 1)
                if n >= 2:
                    self.assertEqual(A.derivative().coefficients, [(k + 1) * a[k + 1] for k in range(n - 1)])
                    self.assertEqual(A.derivative().order, n - 1)
                    self.assertEqual(A.integral().derivative(), A)
                else:
                    with self.assertRaises(ValueError):
                        A.derivative()
                if a[0]:
                    inv = A.reciprocal()
                    self.assertEqual(conv(inv.coefficients, a, n), [1] + [0] * (n - 1))   # f (1/f) = 1 mod x^n
                else:
                    with self.assertRaises(NoSolutionError):
                        A.reciprocal()
                g = rand_coefs(n, 0)
                G = mc.Series(g, n)
                powg, comp = [Fraction(1)] + [Fraction(0)] * (n - 1), [Fraction(0)] * n
                for k in range(n):                                  # f(g) = sum f_k g^k (naive)
                    comp = [c + a[k] * t for c, t in zip(comp, powg)]
                    powg = conv(powg, g, n)
                self.assertEqual(A(G).coefficients, comp)
                ex, term = [Fraction(0)] * n, [Fraction(1)] + [Fraction(0)] * (n - 1)
                for k in range(n):                                  # exp(g) = sum g^k / k! (naive)
                    ex = [e + t / math.factorial(k) for e, t in zip(ex, term)]
                    term = conv(term, g, n)
                self.assertEqual(G.exp().coefficients, ex)
                one_plus = [Fraction(1)] + g[1:]
                lg, term = [Fraction(0)] * n, list(g)               # log(1 + u) = sum (-1)^(k+1) u^k / k
                for k in range(1, n):
                    lg = [l + (-1) ** (k + 1) * t / k for l, t in zip(lg, term)]
                    term = conv(term, g, n)
                L = mc.Series(one_plus, n).log()
                self.assertEqual(L.coefficients, lg)
                self.assertEqual(L.exp(), mc.Series(one_plus, n))   # exp(log f) = f
                self.assertEqual(G.exp().log(), G)                  # log(exp g) = g
                with self.assertRaises(NoSolutionError):
                    mc.Series(rand_coefs(n, 1), n).exp()            # exp(1) is not rational
                with self.assertRaises(NoSolutionError):
                    mc.Series(rand_coefs(n, 2), n).log()
                with self.assertRaises(NoSolutionError):
                    A(mc.Series(rand_coefs(n, 3), n))               # the inner series needs a zero constant term
        x = mc.Series([0, 1], order=10)
        self.assertEqual(x.exp().coefficients, [Fraction(1, math.factorial(k)) for k in range(10)])
        self.assertEqual(mc.Series([1, -1], 6).reciprocal().coefficients, [1] * 6)            # 1/(1-x)
        self.assertEqual(mc.Series([1, 1], 6).log().coefficients, [0] + [Fraction((-1) ** (k + 1), k) for k in range(1, 6)])
        big = mc.Series([Fraction(1, 3 ** 200), 1 << 500], 3)
        self.assertEqual(big.reciprocal().coefficients[1], -(1 << 500) * 3 ** 400)
        with self.assertRaises(CarrierError):
            mc.Series([1], 3) + mc.Series([1], 4)
        self.assertEqual(mc.Series([1, 2, 3, 4], 4).truncate(2) * mc.Series([1, 1], 2), mc.Series([1, 3], 2))
        with self.assertRaises(ValueError):
            mc.Series([1, 2, 3], order=2)                           # more coefficients than the order: refused
        with self.assertRaises(ValueError):
            mc.Series([1], order=0)

    # ------------------------------------------------------------------------------------------ budgets
    def test_fnd009_budgets_at_the_limit_and_one_past(self):
        rng = random.Random(9010)
        for _ in range(30):
            x, y = rand_int(rng, 3000) or 1, rand_int(rng, 3000) or 1
            la, lb = limbs(x), limbs(y)
            a, b = mc.BigInt(x), mc.BigInt(y)
            for fn, need in ((lambda bud: a.mul(b, budget=bud), 8 * (la + lb)),
                             (lambda bud: a.add(b, budget=bud), 8 * (max(la, lb) + 1)),
                             (lambda bud: a.sub(b, budget=bud), 8 * (max(la, lb) + 1)),
                             (lambda bud: a.divmod(b, budget=bud), 8 * (max(la, lb) + 2)),
                             (lambda bud: a.gcd(b, budget=bud), 8 * (max(la, lb) + 1)),
                             (lambda bud: a.shl(300, budget=bud), 8 * (la + 300 // 64 + 1)),
                             (lambda bud: mc.BigInt(x, budget=bud), 8 * la)):
                fn(Budget(bytes=need))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    fn(Budget(bytes=need - 1))
                self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("bytes", need, need - 1))
            self.assertEqual(int(a), x)                             # a refused operation leaves its inputs unchanged
        text = "7" * 1000
        need = 8 * ((3322 * 1000 + 63999) // 64000 + 1)
        self.assertEqual(int(mc.BigInt(text, budget=Budget(bytes=need))), int(text))
        with self.assertRaises(AlgebraBudgetError):
            mc.BigInt(text, budget=Budget(bytes=need - 1))
        for base, e in ((3, 1000), (-(2 ** 64 + 1), 9), (2, 4096)):
            need = 8 * (-(-e * abs(base).bit_length() // 64) + 1)
            self.assertEqual(int(mc.BigInt(base).pow(e, budget=Budget(bytes=need))), base ** e)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.BigInt(base).pow(e, budget=Budget(bytes=need - 1))
            self.assertEqual(ctx.exception.need, need)
        with self.assertRaises(AlgebraBudgetError) as ctx:          # decided before anything is computed
            mc.BigInt(3).pow(1 << 62)
        self.assertEqual(ctx.exception.resource, "bytes")
        # rationals budget every number they create; series their order (degree N - 1, support N)
        q = mc.BigRational(Fraction(1 << 1000, 3))
        with self.assertRaises(AlgebraBudgetError):
            q.mul(q, budget=Budget(bytes=8 * 16))
        self.assertEqual(q.mul(q, budget=Budget(bytes=8 * 40)).to_fraction(), Fraction(1 << 2000, 9))
        mc.Series([1, 2], order=10, budget=Budget(degree=9, support=10))
        for bud, resource in ((Budget(degree=8), "degree"), (Budget(support=9), "support")):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.Series([1, 2], order=10, budget=bud)
            self.assertEqual(ctx.exception.resource, resource)
        s = mc.Series([1, 2], order=10, budget=Budget(degree=9))
        with self.assertRaises(AlgebraBudgetError):                 # the integral has order 11
            s.integral()
        self.assertEqual(s.integral(budget=Budget(degree=10)).order, 11)
        pool = [p for p in range(P62, P62 - 3000, -2) if miller_rabin(p)][:8]
        mc.crt([1] * 8, [mc.GF(p) for p in pool], budget=Budget(bytes=8 * 9))
        with self.assertRaises(AlgebraBudgetError):
            mc.crt([1] * 8, [mc.GF(p) for p in pool], budget=Budget(bytes=8 * 9 - 1))
        # the device paths budget their copies (sum: states = groups as well)
        x = np.arange(1000, dtype=np.int64)
        need = mc.device_bytes("enclose_q64", 1000)
        mc.enclose_q64(x, x + 1, backend="cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError):
            mc.enclose_q64(x, x + 1, backend="cuda", budget=Budget(bytes=need - 1))
        g = (x % 10).astype(np.uint32)
        need = mc.device_bytes("sum_i64", 1000, 10, flag=True)
        mc.sum_i64(x, g, 10, backend="cuda", budget=Budget(bytes=need, states=10))
        for bud, resource in ((Budget(bytes=need - 1), "bytes"), (Budget(states=9), "states")):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.sum_i64(x, g, 10, backend="cuda", budget=bud)
            self.assertEqual(ctx.exception.resource, resource)
        need = mc.device_bytes("dot_wide", 1000)
        mc.dot(x, x, carrier="wide", backend="cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError):
            mc.dot(x, x, carrier="wide", backend="cuda", budget=Budget(bytes=need - 1))

    # ------------------------------------------------------------------------------------------ matched paths
    def test_fnd009_matched_expressions_agree_across_widths(self):
        rng = np.random.default_rng(9011)
        for bits in (4, 16, 30, 31, 32, 40, 62, 63):
            for n in (0, 1, 50, 3000):
                hi = (1 << bits) - 1 if bits < 63 else I64_MAX
                a = rng.integers(-hi, hi, size=n, dtype=np.int64, endpoint=True)
                b = rng.integers(-hi, hi, size=n, dtype=np.int64, endpoint=True)
                if n and bits == 63:
                    a[0], b[0] = I64_MIN, I64_MIN
                exact = sum(int(u) * int(v) for u, v in zip(a, b))
                partial, ok64, ok128 = 0, True, True
                for u, v in zip(a, b):                              # the oracle of the checked paths
                    p = int(u) * int(v)
                    ok64 = ok64 and I64_MIN <= p <= I64_MAX and I64_MIN <= partial + p <= I64_MAX
                    ok128 = ok128 and -(1 << 127) <= partial + p < 1 << 127
                    partial += p
                if ok64:
                    self.assertEqual(mc.dot(a, b, carrier="i64"), exact)
                else:
                    with self.assertRaises(OverflowError):
                        mc.dot(a, b, carrier="i64")
                if ok128:
                    self.assertEqual(mc.dot(a, b, carrier="i128"), exact)
                else:
                    with self.assertRaises(OverflowError):
                        mc.dot(a, b, carrier="i128")
                self.assertEqual(mc.dot(a, b, carrier="wide"), exact)
                self.assertEqual(mc.dot(a, b, carrier="wide", backend="cuda"), exact)
                if n <= 50:
                    self.assertEqual(mc.dot([int(v) for v in a], [int(v) for v in b], carrier="bigint"), exact)
        big = np.full(4000, I64_MIN, dtype=np.int64)               # 4000 * 2^126 needs the 192-bit path
        self.assertEqual(mc.dot(big, big, carrier="wide"), 4000 * (1 << 126))
        self.assertEqual(mc.dot(big, big, carrier="wide", backend="cuda"), 4000 * (1 << 126))
        with self.assertRaises(OverflowError):
            mc.dot(big, big, carrier="i128")
        for degree in (0, 1, 3, 7):
            coef = [int(v) for v in rng.integers(-50, 50, size=degree + 1)]
            pts = [int(v) for v in rng.integers(-1000, 1000, size=20)]
            ref = [sum(c * t ** k for k, c in enumerate(coef)) for t in pts]
            self.assertEqual(mc.horner(coef, pts, carrier="bigint"), ref)
            fits64 = all(I64_MIN <= v <= I64_MAX for v in ref) and degree <= 5
            if fits64:
                self.assertEqual(mc.horner(coef, pts, carrier="i64"), ref)
            self.assertEqual(mc.horner(coef, pts, carrier="i128"), ref)
        with self.assertRaises(OverflowError):                     # sum 2^(10 k), k < 12: past int64, inside 128 bits
            mc.horner([1] * 12, [1 << 10], carrier="i64")
        self.assertEqual(mc.horner([1] * 12, [1 << 10], carrier="i128")[0], sum(1 << (10 * k) for k in range(12)))
        with self.assertRaises(OverflowError):
            mc.horner([1] * 12, [1 << 20], carrier="i128")
        self.assertEqual(mc.horner([1] * 12, [1 << 20], carrier="bigint")[0], sum(1 << (20 * k) for k in range(12)))
        # exact 128-bit sums, one group or many, CPU and CUDA
        for n, k in ((0, 1), (1, 1), (5000, 1), (5000, 7), (20000, 300)):
            x = rng.integers(I64_MIN, I64_MAX, size=n, dtype=np.int64, endpoint=True)
            groups = None if k == 1 and n % 2 == 0 else rng.integers(0, k, size=n, dtype=np.uint32)
            ref = [0] * k
            for i, v in enumerate(x):
                ref[0 if groups is None else int(groups[i])] += int(v)
            cpu = mc.sum_i64(x, groups, k)
            gpu = mc.sum_i64(x, groups, k, backend="cuda")
            self.assertEqual(cpu.values, ref)
            self.assertEqual(gpu.values, ref)
        with self.assertRaises(ValueError):
            mc.sum_i64([1, 2], [0, 5], 3)
        with self.assertRaises(ValueError):
            mc.sum_i64([1, 2], [0, 5], 3, backend="cuda")

    def test_fnd009_cuda_paths_equal_cpu_on_scrambled_threads(self):
        rng = np.random.default_rng(9012)
        info = math_gpu.device()
        self.assertIn("emulated", info)
        for trial in range(12):
            n = int(rng.integers(1, 20000))
            num = rng.integers(I64_MIN, I64_MAX, size=n, dtype=np.int64, endpoint=True)
            den = rng.integers(I64_MIN, I64_MAX, size=n, dtype=np.int64, endpoint=True)
            den[den == 0] = 1
            num[: n // 10] = rng.integers(-5, 5, size=n // 10)
            cpu = mc.enclose_q64(num, den)
            gpu = mc.enclose_q64(num, den, backend="cuda")
            self.assertTrue(np.array_equal(cpu.values[0].view(np.uint64), gpu.values[0].view(np.uint64)), trial)
            self.assertTrue(np.array_equal(cpu.values[1].view(np.uint64), gpu.values[1].view(np.uint64)), trial)
            for i in range(0, n, max(1, n // 40)):
                v = Fraction(int(num[i]), int(den[i]))
                self.assertTrue(tight(mc.Interval(cpu.values[0][i], cpu.values[1][i]), v))
            p = int(rng.choice(PRIMES[3:]))
            F = mc.GF(p)
            a = rng.integers(0, p, size=n, dtype=np.uint64)
            b = rng.integers(0, p, size=n, dtype=np.uint64)
            self.assertEqual(F.dot(a, b).values, F.dot(a, b, backend="cuda").values)
            w = rng.integers(I64_MIN, I64_MAX, size=n, dtype=np.int64, endpoint=True)
            self.assertEqual(mc.dot(w, w[::-1].copy(), carrier="wide"), mc.dot(w, w[::-1].copy(), carrier="wide", backend="cuda"))
        with self.assertRaises(ValueError):
            mc.enclose_q64([1, 2], [3, 0], backend="cuda")


if __name__ == "__main__":
    unittest.main()
