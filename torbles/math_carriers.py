# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Numeric carriers and exactness conversions (FND-009), natively (torbles/math_carriers.c, math_carriers_gpu.cu).

    from fractions import Fraction
    from torbles import math_carriers as mc
    a = mc.BigInt(2**5000 + 1); b = mc.BigInt(-3)
    int(a * b), divmod(a, b), a.gcd(b), a ** 3, a >> 7, a.to_decimal()
    q = mc.BigRational(Fraction(1, 3)); q.enclose()     # Interval [RD(1/3), RU(1/3)], approximate
    F = mc.GF(2**61 - 1); x = F(5); x.inverse(); F(Fraction(1, 3)); x.lift()
    mc.crt([2, 3], [mc.GF(5), mc.GF(7)]); mc.rational_reconstruct(u, m)
    mc.Interval(1.0, 2.0) / mc.Interval(3.0, 4.0); mc.Interval.enclose(Fraction(1, 3)).sqrt()
    s = mc.Series([0, 1], order=8); s.exp(); (1 + s).log()...
    mc.conversion("interval", "bigq")                   # "forbidden"
    mc.convert(mc.BigInt(7), "i64")                     # explicit, checked

Every result is computed by the native library; this module marshals values (Python int <-> limbs, Fraction <->
canonical numerator and denominator, GF(p) elements as (p, v), intervals as (lo, hi)) and maps native codes to
exceptions (torbles.math_errors): narrowing OverflowError, mixed carriers CarrierError, no exact answer
NoSolutionError, budgets AlgebraBudgetError. Nothing converts implicitly: operands are of one carrier (Python int and
Fraction are accepted where they are exact members of the operand's carrier: int for BigInt and BigRational, Fraction
for BigRational). Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check

__all__ = ["CARRIERS", "KINDS", "BigInt", "BigRational", "I64", "I128", "Q64", "GF", "GFElement", "Interval",
           "ComplexInterval", "Series", "Batch", "describe", "conversion", "convert", "carrier_of", "crt",
           "crt_combine", "rational_reconstruct", "enclose_q64", "sum_i64", "dot", "horner", "device_bytes", "Budget"]

CARRIERS = {"i64": 1, "i128": 2, "bigint": 3, "q64": 4, "bigq": 5, "gfp": 6, "binary64": 7, "interval": 8,
            "cinterval": 9, "series": 10}
KINDS = ("forbidden", "identity", "lossless", "narrowing", "enclosing", "reduction", "lift", "truncation")
LAWS = {1: "integer", 2: "rational", 3: "prime field", 4: "dyadic", 5: "real enclosure", 6: "complex enclosure",
        7: "truncated series"}
OPS = {"add": 1, "sub": 2, "mul": 3, "div": 4, "neg": 5, "inv": 6, "sqr": 7, "sqrt": 8, "abs": 9, "pow": 10,
       "hull": 11, "intersect": 12, "conj": 13, "gcd": 14, "mod": 15}
KERNELS = {"gfp_vec": 1, "gfp_dot": 2, "gfp_matvec": 3, "gfp_reduce": 4, "enclose_q64": 5, "sum_i64": 6,
           "dot_wide": 7}
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
I128_MIN, I128_MAX = -(1 << 127), (1 << 127) - 1


class _I128(C.Structure):
    _fields_ = [("lo", C.c_int64), ("hi", C.c_int64)]


class _Q(C.Structure):
    _fields_ = [("numerator", C.c_int64), ("denominator", C.c_int64)]


class _Interval(C.Structure):
    _fields_ = [("lo", C.c_double), ("hi", C.c_double)]


class _CInterval(C.Structure):
    _fields_ = [("re", _Interval), ("im", _Interval)]


class _GFP(C.Structure):
    _fields_ = [("p", C.c_uint64), ("mu", C.c_uint64), ("r64", C.c_uint64), ("k", C.c_uint32), ("reserved", C.c_uint32)]


class _Elem(C.Structure):
    _fields_ = [("p", C.c_uint64), ("v", C.c_uint64)]


class _Desc(C.Structure):
    _fields_ = [("carrier", C.c_int32), ("law", C.c_int32), ("exact", C.c_int32), ("certified", C.c_int32),
                ("precision_bits", C.c_int32), ("reserved", C.c_int32), ("name", C.c_char * 24)]


_N = None


def _api():
    """the native functions, bound once (ctypes signatures)"""
    global _N
    if _N is not None:
        return _N
    P, vp, i32, u32, i64, u64, dbl = C.POINTER, C.c_void_p, C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_double
    B, R, E = P(_Budget), P(_Refusal), P(C.c_int32)
    obj1, obj2 = (vp, [vp, B, R, E]), (vp, [vp, vp, B, R, E])
    sig = {
        "tr_carrier_abi": (i32, [P(u64), i32]), "tr_carrier_describe": (i32, [i32, P(_Desc)]),
        "tr_carrier_conversion": (i32, [i32, i32]), "tr_carrier_check_conversion": (i32, [i32, i32]),
        "tr_carrier_combine": (i32, [i32, i32]), "tr_carrier_device_bytes": (u64, [i32, u64, u64, i32]),
        "tr_carrier_i64_op": (i32, [i32, i64, i64, P(i64)]), "tr_carrier_i128_op": (i32, [i32, _I128, _I128, P(_I128)]),
        "tr_carrier_i128_cmp": (i32, [_I128, _I128]), "tr_carrier_q64_op": (i32, [i32, _Q, _Q, P(_Q)]),
        "tr_carrier_q64_cmp": (i32, [_Q, _Q, P(i32)]), "tr_carrier_i128_to_i64": (i32, [_I128, P(i64)]),
        "tr_carrier_q64_to_i64": (i32, [_Q, P(i64)]),
        "tr_bigint_from_i64": (vp, [i64, B, R, E]), "tr_bigint_from_i128": (vp, [_I128, B, R, E]),
        "tr_bigint_from_limbs": (vp, [i32, vp, u64, B, R, E]), "tr_bigint_from_twos": (vp, [vp, u64, B, R, E]),
        "tr_bigint_from_decimal": (vp, [C.c_char_p, u64, B, R, E]), "tr_bigint_to_i64": (i32, [vp, P(i64)]),
        "tr_bigint_to_i128": (i32, [vp, P(_I128)]), "tr_bigint_export": (i32, [vp, P(i32), vp, u64, P(u64)]),
        "tr_bigint_decimal_size": (u64, [vp]), "tr_bigint_to_decimal": (i32, [vp, vp, u64, P(u64), B, R]),
        "tr_bigint_sign": (i32, [vp]), "tr_bigint_limb_count": (u64, [vp]), "tr_bigint_bit_length": (u64, [vp]),
        "tr_bigint_cmp": (i32, [vp, vp]), "tr_bigint_is_handle": (i32, [vp]),
        "tr_bigint_copy": obj1, "tr_bigint_neg": obj1, "tr_bigint_abs": obj1, "tr_bigint_isqrt": obj1,
        "tr_bigint_add": obj2, "tr_bigint_sub": obj2, "tr_bigint_mul": obj2, "tr_bigint_gcd": obj2,
        "tr_bigint_divexact": obj2, "tr_bigint_divmod": (i32, [vp, vp, i32, P(vp), P(vp), B, R]),
        "tr_bigint_shl": (vp, [vp, u64, B, R, E]), "tr_bigint_shr": (vp, [vp, u64, B, R, E]),
        "tr_bigint_pow": (vp, [vp, u64, B, R, E]), "tr_bigint_mod_u64": (i32, [vp, u64, P(u64)]),
        "tr_bigint_enclose": (i32, [vp, P(_Interval)]), "tr_bigint_dot": (vp, [vp, vp, u64, B, R, E]),
        "tr_bigint_horner": (vp, [vp, u32, vp, B, R, E]), "tr_bigint_batch": (i32, [i32, vp, vp, u64, vp, B, R]),
        "tr_bigint_free": (None, [vp]), "tr_bigint_free_array": (None, [vp, u64]),
        "tr_bigq_from_bigints": obj2, "tr_bigq_from_bigint": obj1, "tr_bigq_from_q64": (vp, [_Q, B, R, E]),
        "tr_bigq_from_double": (vp, [dbl, B, R, E]), "tr_bigq_to_q64": (i32, [vp, P(_Q)]), "tr_bigq_to_bigint": obj1,
        "tr_bigq_numerator": (vp, [vp]), "tr_bigq_denominator": (vp, [vp]), "tr_bigq_sign": (i32, [vp]),
        "tr_bigq_cmp": (i32, [vp, vp]), "tr_bigq_cmp_double": (i32, [vp, dbl, P(i32)]), "tr_bigq_is_handle": (i32, [vp]),
        "tr_bigq_copy": obj1, "tr_bigq_neg": obj1, "tr_bigq_inv": obj1, "tr_bigq_add": obj2, "tr_bigq_sub": obj2,
        "tr_bigq_mul": obj2, "tr_bigq_div": obj2, "tr_bigq_pow": (vp, [vp, i64, B, R, E]),
        "tr_bigq_enclose": (i32, [vp, P(_Interval)]), "tr_bigq_in_interval": (i32, [vp, _Interval]),
        "tr_bigq_free": (None, [vp]),
        "tr_gfp_is_prime": (i32, [u64]), "tr_gfp_field": (i32, [u64, P(_GFP)]),
        "tr_gfp_from_u64": (i32, [P(_GFP), u64, P(_Elem)]), "tr_gfp_from_i64": (i32, [P(_GFP), i64, P(_Elem)]),
        "tr_gfp_from_i128": (i32, [P(_GFP), _I128, P(_Elem)]), "tr_gfp_from_q64": (i32, [P(_GFP), _Q, P(_Elem)]),
        "tr_gfp_from_bigint": (i32, [P(_GFP), vp, P(_Elem)]), "tr_gfp_from_bigq": (i32, [P(_GFP), vp, P(_Elem)]),
        "tr_gfp_add": (i32, [_Elem, _Elem, P(_Elem)]), "tr_gfp_sub": (i32, [_Elem, _Elem, P(_Elem)]),
        "tr_gfp_mul": (i32, [_Elem, _Elem, P(_Elem)]), "tr_gfp_div": (i32, [_Elem, _Elem, P(_Elem)]),
        "tr_gfp_neg": (i32, [_Elem, P(_Elem)]), "tr_gfp_inv": (i32, [_Elem, P(_Elem)]),
        "tr_gfp_pow": (i32, [_Elem, i64, P(_Elem)]), "tr_gfp_equal": (i32, [_Elem, _Elem]),
        "tr_gfp_lift_i64": (i32, [_Elem, i32, P(i64)]), "tr_gfp_lift": (vp, [_Elem, i32, B, R, E]),
        "tr_gfp_vec": (i32, [P(_GFP), i32, vp, vp, u64, vp]), "tr_gfp_dot": (i32, [P(_GFP), vp, vp, u64, P(u64)]),
        "tr_gfp_matvec": (i32, [P(_GFP), vp, u64, u64, vp, vp]), "tr_gfp_reduce_i64": (i32, [P(_GFP), vp, u64, vp]),
        "tr_carrier_crt": (i32, [vp, vp, u32, i32, P(vp), P(vp), B, R]),
        "tr_carrier_crt_combine": (i32, [vp, vp, u64, P(_GFP), P(vp), P(vp), B, R]),
        "tr_carrier_ratrecon": (vp, [vp, vp, vp, vp, B, R, E]),
        "tr_carrier_dot_i64": (i32, [vp, vp, u64, P(i64)]), "tr_carrier_dot_i128": (i32, [vp, vp, u64, P(_I128)]),
        "tr_carrier_dot_wide": (i32, [vp, vp, u64, P(u64 * 3)]),
        "tr_carrier_horner_i64": (i32, [vp, u32, vp, u64, vp]), "tr_carrier_horner_i128": (i32, [vp, u32, vp, u64, vp]),
        "tr_carrier_sum_i64": (i32, [vp, u64, vp, u64, vp]),
        "tr_carrier_interval_op": (i32, [i32, P(_Interval), P(_Interval), i64, P(_Interval)]),
        "tr_carrier_interval_cmp": (i32, [P(_Interval), P(_Interval)]),
        "tr_carrier_cinterval_op": (i32, [i32, P(_CInterval), P(_CInterval), P(_CInterval)]),
        "tr_carrier_cinterval_abs": (i32, [P(_CInterval), P(_Interval)]),
        "tr_carrier_cinterval_real": (i32, [P(_CInterval), P(_Interval)]),
        "tr_carrier_enclose_i64": (i32, [i64, P(_Interval)]), "tr_carrier_enclose_i128": (i32, [_I128, P(_Interval)]),
        "tr_carrier_enclose_q64": (i32, [_Q, P(_Interval)]), "tr_carrier_enclose_q64_vec": (i32, [vp, u64, vp]),
        "tr_series_create": (vp, [vp, u32, u32, B, R, E]), "tr_series_constant": (vp, [vp, u32, B, R, E]),
        "tr_series_order": (u32, [vp]), "tr_series_coeff": (vp, [vp, u32]), "tr_series_is_handle": (i32, [vp]),
        "tr_series_equal": (i32, [vp, vp]), "tr_series_truncate": (vp, [vp, u32, B, R, E]),
        "tr_series_add": obj2, "tr_series_sub": obj2, "tr_series_mul": obj2, "tr_series_compose": obj2,
        "tr_series_scale": obj2, "tr_series_derivative": obj1, "tr_series_integral": obj1,
        "tr_series_reciprocal": obj1, "tr_series_exp": obj1, "tr_series_log": obj1, "tr_series_free": (None, [vp]),
    }
    native = {}
    for name, (restype, argtypes) in sig.items():
        fn = getattr(algq.lib, name)
        fn.restype, fn.argtypes = restype, argtypes
        native[name[3:]] = fn
    sizes = (C.c_uint64 * 7)()
    native["carrier_abi"](sizes, 7)
    want = [C.sizeof(t) for t in (_GFP, _Elem, _CInterval, _Desc, _Interval, _I128, _Q)]
    if list(sizes) != want:
        raise ImportError(f"math_carriers ABI mismatch: native {list(sizes)}, bindings {want} (rebuild: make)")
    _N = native
    return _N


def _b(budget):
    return C.byref(budget.raw) if budget is not None else None


def _obj(fn, *args, budget=None, what=""):
    """call a native constructor / operation returning an owned handle; raise its refusal"""
    refusal, err = _Refusal(), C.c_int32(0)
    handle = fn(*args, _b(budget), C.byref(refusal), C.byref(err))
    if not handle:
        _check(err.value if err.value else -2, refusal, what)
    return handle


def _code(code, refusal=None, what=""):
    return _check(code, refusal, what)


def _mixed(a, b, what="operands"):
    """raise CarrierError for operands of different carriers (the native combination rule)"""
    _code(_api()["carrier_combine"](CARRIERS[a], CARRIERS[b]), what=f"{what}: {a} with {b}")
    raise AssertionError("unreachable: different carriers combined")


def _gpu(name, restype, argtypes):
    from . import math_gpu
    math_gpu.bind(name, restype, argtypes)
    return math_gpu


# ---------------------------------------------------------------------------------------------------- carriers
def describe(carrier):
    """a carrier's law and exactness contract, from the native table"""
    d = _Desc()
    _code(_api()["carrier_describe"](CARRIERS[carrier], C.byref(d)), what="carrier")
    return dict(carrier=d.name.decode(), law=LAWS[d.law], exact=bool(d.exact), certified=bool(d.certified),
                approximate=not d.exact, precision_bits=d.precision_bits or None)


def conversion(src, dst):
    """the kind of the conversion src -> dst in the native matrix: forbidden, identity, lossless, narrowing,
    enclosing, reduction, lift or truncation"""
    k = _api()["carrier_conversion"](CARRIERS[src], CARRIERS[dst])
    _code(k if k < 0 else 0, what="carrier")
    return KINDS[k]


def carrier_of(value):
    """the carrier a value belongs to (Python int: bigint; Fraction: bigq; float: binary64)"""
    if isinstance(value, _Carrier):
        return value.carrier
    if isinstance(value, bool):
        raise TypeError("bool is not a numeric carrier")
    if isinstance(value, Integral):
        return "bigint"
    if isinstance(value, Fraction):
        return "bigq"
    if isinstance(value, float):
        return "binary64"
    raise TypeError(f"{type(value).__name__} is not a numeric carrier")


class _Carrier:
    __slots__ = ()
    carrier = None


class Batch:
    """a batched result with the backend that computed it ("cpu", "cuda" or "cuda-emulated") and its device"""
    __slots__ = ("values", "backend", "device")

    def __init__(self, values, backend, device):
        self.values, self.backend, self.device = values, backend, device

    def __repr__(self):
        return f"Batch({self.values!r}, backend={self.backend!r})"


def _run_gpu(name, restype, argtypes, *args, refusal, what):
    gpu = _gpu(name, restype, argtypes)
    gpu.run(name, *args, refusal=refusal, what=what)
    info = gpu.device()
    return "cuda-emulated" if info["emulated"] else "cuda", info["name"]


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def device_bytes(kernel, n, m=0, flag=False):
    """bytes a CUDA kernel copies to the device (tr_carrier_kernel_bytes): gfp_vec (flag = unary), gfp_dot,
    gfp_matvec (n rows, m cols), gfp_reduce, enclose_q64, sum_i64 (m groups, flag = with group ids), dot_wide"""
    return int(_api()["carrier_device_bytes"](KERNELS[kernel], int(n), int(m), int(bool(flag))))


# ---------------------------------------------------------------------------------------------------- fixed width
def _i128(v):
    v = int(v)
    if not I128_MIN <= v <= I128_MAX:
        raise OverflowError("the value does not fit 128 bits")
    raw = (v & ((1 << 128) - 1)).to_bytes(16, "little")
    return _I128.from_buffer_copy(raw)


def _from_i128(x):
    return int.from_bytes(bytes(x), "little", signed=True)


def _q(n, d):
    n, d = int(n), int(d)
    if not (I64_MIN <= n <= I64_MAX and I64_MIN <= d <= I64_MAX):
        raise OverflowError("an int64 rational needs a numerator and a denominator that fit int64")
    return _Q(n, d)


class I64(_Carrier):
    """a checked int64 (the first slice's integers): arithmetic refuses overflow (OverflowError)"""
    __slots__ = ("value",)
    carrier = "i64"

    def __init__(self, value):
        value = int(value)
        if not I64_MIN <= value <= I64_MAX:
            raise OverflowError("the value does not fit int64")
        self.value = value

    def _op(self, op, other):
        if not isinstance(other, I64):
            return _mixed("i64", carrier_of(other))
        out = C.c_int64()
        _code(_api()["carrier_i64_op"](OPS[op], self.value, other.value, C.byref(out)), what=f"i64 {op}")
        return I64(out.value)

    __add__ = lambda s, o: s._op("add", o)
    __sub__ = lambda s, o: s._op("sub", o)
    __mul__ = lambda s, o: s._op("mul", o)
    __floordiv__ = lambda s, o: s._op("div", o)

    def __neg__(self):
        return self._op("neg", self)

    def __eq__(self, other):
        return isinstance(other, I64) and self.value == other.value

    def __hash__(self):
        return hash(("i64", self.value))

    def __repr__(self):
        return f"I64({self.value})"


class I128(_Carrier):
    """an explicit 128-bit integer (tr_math_i128): arithmetic refuses overflow (OverflowError)"""
    __slots__ = ("value",)
    carrier = "i128"

    def __init__(self, value):
        self.value = _from_i128(_i128(value))

    def _op(self, op, other):
        if not isinstance(other, I128):
            return _mixed("i128", carrier_of(other))
        out = _I128()
        _code(_api()["carrier_i128_op"](OPS[op], _i128(self.value), _i128(other.value), C.byref(out)), what=f"i128 {op}")
        return I128(_from_i128(out))

    __add__ = lambda s, o: s._op("add", o)
    __sub__ = lambda s, o: s._op("sub", o)
    __mul__ = lambda s, o: s._op("mul", o)
    __floordiv__ = lambda s, o: s._op("div", o)

    def __neg__(self):
        return self._op("neg", self)

    def cmp(self, other):
        return int(_api()["carrier_i128_cmp"](_i128(self.value), _i128(other.value)))

    def __eq__(self, other):
        return isinstance(other, I128) and self.value == other.value

    def __hash__(self):
        return hash(("i128", self.value))

    def __repr__(self):
        return f"I128({self.value})"


class Q64(_Carrier):
    """a canonical int64 rational (the first slice's tr_math_rational): arithmetic refuses overflow"""
    __slots__ = ("numerator", "denominator")
    carrier = "q64"

    def __init__(self, numerator, denominator=1):
        out = _Q()
        _code(_api()["carrier_q64_op"](OPS["mul"], _q(numerator, denominator), _Q(1, 1), C.byref(out)), what="q64")
        self.numerator, self.denominator = out.numerator, out.denominator

    @property
    def raw(self):
        return _Q(self.numerator, self.denominator)

    def _op(self, op, other):
        if not isinstance(other, Q64):
            return _mixed("q64", carrier_of(other))
        out = _Q()
        _code(_api()["carrier_q64_op"](OPS[op], self.raw, other.raw, C.byref(out)), what=f"q64 {op}")
        return Q64(out.numerator, out.denominator)

    __add__ = lambda s, o: s._op("add", o)
    __sub__ = lambda s, o: s._op("sub", o)
    __mul__ = lambda s, o: s._op("mul", o)
    __truediv__ = lambda s, o: s._op("div", o)

    def __neg__(self):
        return self._op("neg", self)

    def inverse(self):
        return self._op("inv", self)

    def cmp(self, other):
        out = C.c_int32()
        _code(_api()["carrier_q64_cmp"](self.raw, other.raw, C.byref(out)), what="q64 comparison")
        return int(out.value)

    def as_fraction(self):
        return Fraction(self.numerator, self.denominator)

    def __eq__(self, other):
        return isinstance(other, Q64) and (self.numerator, self.denominator) == (other.numerator, other.denominator)

    def __hash__(self):
        return hash(("q64", self.numerator, self.denominator))

    def __repr__(self):
        return f"Q64({self.numerator}, {self.denominator})"


# ---------------------------------------------------------------------------------------------------- big integers
def _limbs(v):
    """sign, uint64 array and limb count of a Python int (two's complement free sign-magnitude marshalling)"""
    sign = -1 if v < 0 else (1 if v else 0)
    m = -v if v < 0 else v
    n = (m.bit_length() + 63) // 64
    arr = (C.c_uint64 * max(n, 1)).from_buffer_copy(m.to_bytes(max(n, 1) * 8, "little"))
    return sign, arr, n


def _bigint_value(handle):
    api = _api()
    n = int(api["bigint_limb_count"](handle))
    arr, sign, count = (C.c_uint64 * max(n, 1))(), C.c_int32(), C.c_uint64()
    _code(api["bigint_export"](handle, C.byref(sign), arr, n, C.byref(count)), what="bigint export")
    m = int.from_bytes(memoryview(arr).cast("B")[: 8 * n], "little")
    return -m if sign.value < 0 else m


class BigInt(_Carrier):
    """an arbitrary-precision integer (sign + 64-bit limbs) with a limb budget (Budget(bytes=8 * limbs))"""
    __slots__ = ("_h", "budget")
    carrier = "bigint"

    def __init__(self, value=0, budget=None):
        self.budget = budget
        if isinstance(value, BigInt):
            self._h = _obj(_api()["bigint_copy"], value._h, budget=budget, what="bigint")
        elif isinstance(value, str):
            data = value.encode()
            self._h = _obj(_api()["bigint_from_decimal"], data, len(data), budget=budget, what="decimal")
        elif isinstance(value, Integral) and not isinstance(value, bool):
            sign, arr, n = _limbs(int(value))
            self._h = _obj(_api()["bigint_from_limbs"], sign, arr, n, budget=budget, what="bigint")
        else:
            self._h = None
            _mixed("bigint", carrier_of(value), "BigInt from")

    @classmethod
    def _wrap(cls, handle, budget=None):
        x = cls.__new__(cls)
        x._h, x.budget = handle, budget
        return x

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _N is not None:
            _N["bigint_free"](h)
            self._h = None

    @classmethod
    def from_i64(cls, v, budget=None):
        v = int(v)
        if not I64_MIN <= v <= I64_MAX:
            raise OverflowError("from_i64 takes an int64")
        return cls._wrap(_obj(_api()["bigint_from_i64"], v, budget=budget, what="bigint"), budget)

    @classmethod
    def from_i128(cls, v, budget=None):
        return cls._wrap(_obj(_api()["bigint_from_i128"], _i128(v), budget=budget, what="bigint"), budget)

    @classmethod
    def from_decimal(cls, text, budget=None):
        return cls(str(text), budget)

    @classmethod
    def from_twos(cls, words, budget=None):
        arr = (C.c_uint64 * len(words))(*[int(w) for w in words])
        return cls._wrap(_obj(_api()["bigint_from_twos"], arr, len(words), budget=budget, what="bigint"), budget)

    def to_int(self):
        return _bigint_value(self._h)

    __int__ = to_int
    __index__ = to_int

    def to_i64(self):
        out = C.c_int64()
        _code(_api()["bigint_to_i64"](self._h, C.byref(out)), what="bigint -> i64")
        return int(out.value)

    def to_i128(self):
        out = _I128()
        _code(_api()["bigint_to_i128"](self._h, C.byref(out)), what="bigint -> i128")
        return _from_i128(out)

    def to_decimal(self, budget=None):
        api = _api()
        size = int(api["bigint_decimal_size"](self._h))
        buf, length, refusal = C.create_string_buffer(size), C.c_uint64(), _Refusal()
        _code(api["bigint_to_decimal"](self._h, buf, size, C.byref(length), _b(budget or self.budget), C.byref(refusal)),
              refusal, "bigint -> decimal")
        return buf.raw[: length.value].decode()

    @property
    def sign(self):
        return int(_api()["bigint_sign"](self._h))

    def bit_length(self):
        return int(_api()["bigint_bit_length"](self._h))

    @property
    def limbs(self):
        return int(_api()["bigint_limb_count"](self._h))

    def _other(self, other):
        if isinstance(other, BigInt):
            return other
        if isinstance(other, Integral) and not isinstance(other, bool):
            return BigInt(other, self.budget)
        return _mixed("bigint", carrier_of(other))

    def _bin(self, name, other, budget=None):
        o = self._other(other)
        b = budget or self.budget
        return BigInt._wrap(_obj(_api()[name], self._h, o._h, budget=b, what=name[10:]), b)

    def add(self, other, budget=None):
        return self._bin("bigint_add", other, budget)

    def sub(self, other, budget=None):
        return self._bin("bigint_sub", other, budget)

    def mul(self, other, budget=None):
        return self._bin("bigint_mul", other, budget)

    def gcd(self, other, budget=None):
        return self._bin("bigint_gcd", other, budget)

    def divexact(self, other, budget=None):
        return self._bin("bigint_divexact", other, budget)

    def divmod(self, other, rounding="floor", budget=None):
        """(q, r) with a = q b + r: rounding "floor" (r has the sign of b) or "trunc" (r has the sign of a)"""
        o = self._other(other)
        b = budget or self.budget
        q, r, refusal = C.c_void_p(), C.c_void_p(), _Refusal()
        _code(_api()["bigint_divmod"](self._h, o._h, {"floor": 0, "trunc": 1}[rounding], C.byref(q), C.byref(r), _b(b),
                                      C.byref(refusal)), refusal, "divmod")
        return BigInt._wrap(q.value, b), BigInt._wrap(r.value, b)

    def shl(self, bits, budget=None):
        b = budget or self.budget
        return BigInt._wrap(_obj(_api()["bigint_shl"], self._h, _shift(bits), budget=b, what="shift"), b)

    def shr(self, bits, budget=None):
        b = budget or self.budget
        return BigInt._wrap(_obj(_api()["bigint_shr"], self._h, _shift(bits), budget=b, what="shift"), b)

    def pow(self, e, budget=None):
        e = int(e)
        if e < 0:
            raise ValueError("an integer power needs e >= 0 (use BigRational for negative powers)")
        b = budget or self.budget
        return BigInt._wrap(_obj(_api()["bigint_pow"], self._h, e, budget=b, what="pow"), b)

    def isqrt(self, budget=None):
        b = budget or self.budget
        return BigInt._wrap(_obj(_api()["bigint_isqrt"], self._h, budget=b, what="isqrt"), b)

    def mod_u64(self, m):
        out = C.c_uint64()
        _code(_api()["bigint_mod_u64"](self._h, int(m), C.byref(out)), what="mod")
        return int(out.value)

    def enclose(self):
        return Interval._enclosure(_api()["bigint_enclose"], self._h)

    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __radd__ = lambda s, o: s._other(o).add(s)
    __rsub__ = lambda s, o: s._other(o).sub(s)
    __rmul__ = lambda s, o: s._other(o).mul(s)
    __lshift__ = shl
    __rshift__ = shr
    __pow__ = pow

    def __floordiv__(self, other):
        return self.divmod(other)[0]

    def __mod__(self, other):
        return self.divmod(other)[1]

    def __divmod__(self, other):
        return self.divmod(other)

    def __neg__(self):
        return BigInt._wrap(_obj(_api()["bigint_neg"], self._h, budget=self.budget, what="neg"), self.budget)

    def __abs__(self):
        return BigInt._wrap(_obj(_api()["bigint_abs"], self._h, budget=self.budget, what="abs"), self.budget)

    def cmp(self, other):
        o = self._other(other)
        return _code(_api()["bigint_cmp"](self._h, o._h) + 1, what="compare") - 1

    def __eq__(self, other):
        if not isinstance(other, (BigInt, Integral)) or isinstance(other, bool):
            return NotImplemented
        return self.cmp(other) == 0

    __lt__ = lambda s, o: s.cmp(o) < 0
    __le__ = lambda s, o: s.cmp(o) <= 0
    __gt__ = lambda s, o: s.cmp(o) > 0
    __ge__ = lambda s, o: s.cmp(o) >= 0

    def __hash__(self):
        return hash(self.to_int())

    def __repr__(self):
        return f"BigInt({self.to_decimal()})"


def _shift(bits):
    bits = int(bits)
    if not 0 <= bits < 1 << 64:
        raise ValueError("a shift count is between 0 and 2^64-1")
    return bits


def _batch_handles(values, budget):
    objs = [v if isinstance(v, BigInt) else BigInt(v, budget) for v in values]
    arr = (C.c_void_p * max(len(objs), 1))(*[o._h for o in objs])
    return objs, arr


def bigint_batch(op, a, b, budget=None):
    """elementwise a[i] op b[i] for op add, sub, mul, div (floor), mod (floor), gcd; all or nothing"""
    if len(a) != len(b):
        raise ValueError("operands of different lengths")
    ka, pa = _batch_handles(a, budget)
    kb, pb = _batch_handles(b, budget)
    out, refusal = (C.c_void_p * max(len(a), 1))(), _Refusal()
    _code(_api()["bigint_batch"](OPS[op], pa, pb, len(a), out, _b(budget), C.byref(refusal)), refusal, f"batch {op}")
    del ka, kb
    return [BigInt._wrap(out[i], budget) for i in range(len(a))]


# ---------------------------------------------------------------------------------------------------- big rationals
class BigRational(_Carrier):
    """an arbitrary-precision canonical rational (numerator, denominator > 0, gcd 1)"""
    __slots__ = ("_h", "budget")
    carrier = "bigq"

    def __init__(self, value=0, denominator=None, budget=None):
        self.budget = budget
        api = _api()
        if denominator is not None:
            n, d = BigInt(value, budget) if not isinstance(value, BigInt) else value, \
                BigInt(denominator, budget) if not isinstance(denominator, BigInt) else denominator
            self._h = _obj(api["bigq_from_bigints"], n._h, d._h, budget=budget, what="rational")
        elif isinstance(value, BigRational):
            self._h = _obj(api["bigq_copy"], value._h, budget=budget, what="rational")
        elif isinstance(value, Fraction):
            n, d = BigInt(value.numerator, budget), BigInt(value.denominator, budget)
            self._h = _obj(api["bigq_from_bigints"], n._h, d._h, budget=budget, what="rational")
        elif isinstance(value, Integral) and not isinstance(value, bool):
            keep = BigInt(value, budget)
            self._h = _obj(api["bigq_from_bigint"], keep._h, budget=budget, what="rational")
        else:
            self._h = None
            _mixed("bigq", carrier_of(value), "BigRational from")

    @classmethod
    def _wrap(cls, handle, budget=None):
        x = cls.__new__(cls)
        x._h, x.budget = handle, budget
        return x

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _N is not None:
            _N["bigq_free"](h)
            self._h = None

    @classmethod
    def from_bigint(cls, x, budget=None):
        """the LOSSLESS conversion bigint -> bigq"""
        if not isinstance(x, BigInt):
            _mixed("bigint", carrier_of(x), "from_bigint")
        return cls._wrap(_obj(_api()["bigq_from_bigint"], x._h, budget=budget, what="rational"), budget)

    @classmethod
    def from_q64(cls, q, budget=None):
        """the LOSSLESS conversion of a first-slice int64 rational (Q64 or a (numerator, denominator) pair)"""
        raw = q.raw if isinstance(q, Q64) else _q(*q)
        return cls._wrap(_obj(_api()["bigq_from_q64"], raw, budget=budget, what="rational"), budget)

    @classmethod
    def from_double(cls, x, budget=None):
        """the exact dyadic value of a finite double (LOSSLESS binary64 -> bigq)"""
        return cls._wrap(_obj(_api()["bigq_from_double"], float(x), budget=budget, what="rational"), budget)

    def _part(self, name):
        return _bigint_value(_api()[name](self._h))

    @property
    def numerator(self):
        return self._part("bigq_numerator")

    @property
    def denominator(self):
        return self._part("bigq_denominator")

    def to_fraction(self):
        n, d = self.numerator, self.denominator
        make = getattr(Fraction, "_from_coprime_ints", None)
        return make(n, d) if make else Fraction(n, d)

    def to_q64(self):
        """NARROWING to an int64 rational: OverflowError unless the numerator and the denominator fit int64"""
        out = _Q()
        _code(_api()["bigq_to_q64"](self._h, C.byref(out)), what="bigq -> q64")
        return Q64(out.numerator, out.denominator)

    def to_bigint(self, budget=None):
        """NARROWING to an integer: NoSolutionError unless the denominator is 1"""
        b = budget or self.budget
        return BigInt._wrap(_obj(_api()["bigq_to_bigint"], self._h, budget=b, what="bigq -> bigint"), b)

    @property
    def sign(self):
        return int(_api()["bigq_sign"](self._h))

    def enclose(self):
        """the tight enclosure [RD(q), RU(q)] (APPROXIMATE carrier)"""
        return Interval._enclosure(_api()["bigq_enclose"], self._h)

    def in_interval(self, iv):
        """exact lo <= q <= hi"""
        return bool(_code(_api()["bigq_in_interval"](self._h, iv._raw()), what="containment"))

    def cmp_double(self, x):
        out = C.c_int32()
        _code(_api()["bigq_cmp_double"](self._h, float(x), C.byref(out)), what="compare")
        return int(out.value)

    def _other(self, other):
        if isinstance(other, BigRational):
            return other
        if isinstance(other, (Fraction, Integral)) and not isinstance(other, bool):
            return BigRational(other, budget=self.budget)
        return _mixed("bigq", carrier_of(other))

    def _bin(self, name, other, budget=None):
        o = self._other(other)
        b = budget or self.budget
        return BigRational._wrap(_obj(_api()[name], self._h, o._h, budget=b, what=name[8:]), b)

    def add(self, other, budget=None):
        return self._bin("bigq_add", other, budget)

    def sub(self, other, budget=None):
        return self._bin("bigq_sub", other, budget)

    def mul(self, other, budget=None):
        return self._bin("bigq_mul", other, budget)

    def div(self, other, budget=None):
        return self._bin("bigq_div", other, budget)

    def pow(self, e, budget=None):
        b = budget or self.budget
        e = int(e)
        if not I64_MIN <= e <= I64_MAX:
            raise OverflowError("the exponent is an int64")
        return BigRational._wrap(_obj(_api()["bigq_pow"], self._h, e, budget=b, what="pow"), b)

    def inverse(self, budget=None):
        b = budget or self.budget
        return BigRational._wrap(_obj(_api()["bigq_inv"], self._h, budget=b, what="inverse"), b)

    __add__, __sub__, __mul__, __truediv__, __pow__ = add, sub, mul, div, pow
    __radd__ = lambda s, o: s._other(o).add(s)
    __rsub__ = lambda s, o: s._other(o).sub(s)
    __rmul__ = lambda s, o: s._other(o).mul(s)
    __rtruediv__ = lambda s, o: s._other(o).div(s)

    def __neg__(self):
        return BigRational._wrap(_obj(_api()["bigq_neg"], self._h, budget=self.budget, what="neg"), self.budget)

    def cmp(self, other):
        o = self._other(other)
        return _code(_api()["bigq_cmp"](self._h, o._h) + 1, what="compare") - 1

    def __eq__(self, other):
        if not isinstance(other, (BigRational, Fraction, Integral)) or isinstance(other, bool):
            return NotImplemented
        return self.cmp(other) == 0

    __lt__ = lambda s, o: s.cmp(o) < 0
    __le__ = lambda s, o: s.cmp(o) <= 0
    __gt__ = lambda s, o: s.cmp(o) > 0
    __ge__ = lambda s, o: s.cmp(o) >= 0

    def __hash__(self):
        return hash(self.to_fraction())

    def __repr__(self):
        return f"BigRational({self.to_fraction()})"


def _as_bigq(x, budget=None):
    if isinstance(x, BigRational):
        return x
    if isinstance(x, BigInt):
        return BigRational.from_bigint(x, budget)
    if isinstance(x, Q64):
        return BigRational.from_q64(x, budget)
    if isinstance(x, I64):
        return BigRational(x.value, budget=budget)
    if isinstance(x, I128):
        return BigRational.from_bigint(BigInt.from_i128(x.value, budget), budget)
    if isinstance(x, (Fraction, Integral)) and not isinstance(x, bool):
        return BigRational(x, budget=budget)
    return _mixed("bigq", carrier_of(x), "a rational coefficient")


# ---------------------------------------------------------------------------------------------------- GF(p)
class GF:
    """the prime field GF(p), p < 2^62, verified prime natively (deterministic Miller-Rabin)"""
    __slots__ = ("raw",)

    def __init__(self, p):
        p = int(p)
        if not 0 <= p < 1 << 64:
            raise ValueError("p is a 64-bit unsigned integer")
        self.raw = _GFP()
        _code(_api()["gfp_field"](p, C.byref(self.raw)), what=f"GF({p})")

    @property
    def p(self):
        return int(self.raw.p)

    @staticmethod
    def is_prime(n):
        return bool(_api()["gfp_is_prime"](int(n)))

    def __eq__(self, other):
        return isinstance(other, GF) and other.p == self.p

    def __hash__(self):
        return hash(("GF", self.p))

    def __repr__(self):
        return f"GF({self.p})"

    def _elem(self, fn, arg, what):
        out = _Elem()
        _code(_api()[fn](C.byref(self.raw), arg, C.byref(out)), what=what)
        return GFElement(self, out.v)

    def __call__(self, x):
        """the REDUCTION of an exact value into GF(p): an integer mod p, a rational n * d^-1 (NoSolutionError when
        d = 0 mod p); an element of this field is returned as is, one of another field is a CarrierError"""
        if isinstance(x, GFElement):
            if x.field.p != self.p:
                _code(_api()["gfp_equal"](x._raw(), _Elem(self.p, 0)), what=f"GF({x.field.p}) into GF({self.p})")
            return x
        if isinstance(x, I64):
            return self._elem("gfp_from_i64", x.value, "reduction")
        if isinstance(x, I128):
            return self._elem("gfp_from_i128", _i128(x.value), "reduction")
        if isinstance(x, Q64):
            return self._elem("gfp_from_q64", x.raw, "reduction")
        if isinstance(x, BigInt):
            return self._elem("gfp_from_bigint", x._h, "reduction")
        if isinstance(x, BigRational):
            return self._elem("gfp_from_bigq", x._h, "reduction")
        if isinstance(x, Integral) and not isinstance(x, bool):
            x = int(x)
            if I64_MIN <= x <= I64_MAX:
                return self._elem("gfp_from_i64", x, "reduction")
            keep = BigInt(x)
            return self._elem("gfp_from_bigint", keep._h, "reduction")
        if isinstance(x, Fraction):
            keep = BigRational(x)
            return self._elem("gfp_from_bigq", keep._h, "reduction")
        return _mixed("gfp", carrier_of(x), "reduction into GF(p)")

    # batched kernels -------------------------------------------------------------------------------------------
    def _arr(self, a):
        return np.ascontiguousarray(a, dtype=np.uint64)

    def vec(self, op, a, b=None, backend="cpu", budget=None):
        """elementwise a op b (add, sub, mul, div) or op a (neg, inv) over residue arrays"""
        unary = op in ("neg", "inv")
        x = self._arr(a)
        y = None if unary else self._arr(b)
        if y is not None and len(y) != len(x):
            raise ValueError("operands of different lengths")
        out = np.empty(len(x), dtype=np.uint64)
        args = (C.byref(self.raw), OPS[op], x.ctypes.data if len(x) else None,
                y.ctypes.data if y is not None and len(y) else None, len(x), out.ctypes.data if len(x) else None)
        if _backend(backend) == "cpu":
            _code(_api()["gfp_vec"](*args), what=f"GF(p) {op}")
            return Batch(out, "cpu", "native CPU")
        refusal = _Refusal()
        used, name = _run_gpu("tr_mgpu_carriers_gfp_vec", C.c_int,
                              [C.POINTER(_GFP), C.c_int32, C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p,
                               C.POINTER(_Budget), C.POINTER(_Refusal)],
                              *args, C.byref((budget or Budget()).raw), C.byref(refusal), refusal=refusal,
                              what=f"GF(p) {op} on the device")
        return Batch(out, used, name)

    def dot(self, a, b, backend="cpu", budget=None):
        x, y = self._arr(a), self._arr(b)
        if len(x) != len(y):
            raise ValueError("operands of different lengths")
        out = C.c_uint64()
        args = (C.byref(self.raw), x.ctypes.data if len(x) else None, y.ctypes.data if len(y) else None, len(x),
                C.byref(out))
        if _backend(backend) == "cpu":
            _code(_api()["gfp_dot"](*args), what="GF(p) dot")
            return Batch(GFElement(self, out.value), "cpu", "native CPU")
        refusal = _Refusal()
        used, name = _run_gpu("tr_mgpu_carriers_gfp_dot", C.c_int,
                              [C.POINTER(_GFP), C.c_void_p, C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64),
                               C.POINTER(_Budget), C.POINTER(_Refusal)],
                              *args, C.byref((budget or Budget()).raw), C.byref(refusal), refusal=refusal,
                              what="GF(p) dot on the device")
        return Batch(GFElement(self, out.value), used, name)

    def matvec(self, A, x, backend="cpu", budget=None):
        M, v = self._arr(A), self._arr(x)
        if M.ndim != 2 or v.ndim != 1 or M.shape[1] != len(v):
            raise ValueError("A is rows x cols and x has cols entries")
        rows, cols = M.shape
        y = np.empty(rows, dtype=np.uint64)
        args = (C.byref(self.raw), M.ctypes.data if M.size else None, rows, cols, v.ctypes.data if len(v) else None,
                y.ctypes.data if rows else None)
        if _backend(backend) == "cpu":
            _code(_api()["gfp_matvec"](*args), what="GF(p) matvec")
            return Batch(y, "cpu", "native CPU")
        refusal = _Refusal()
        used, name = _run_gpu("tr_mgpu_carriers_gfp_matvec", C.c_int,
                              [C.POINTER(_GFP), C.c_void_p, C.c_uint64, C.c_uint64, C.c_void_p, C.c_void_p,
                               C.POINTER(_Budget), C.POINTER(_Refusal)],
                              *args, C.byref((budget or Budget()).raw), C.byref(refusal), refusal=refusal,
                              what="GF(p) matvec on the device")
        return Batch(y, used, name)

    def reduce(self, x, backend="cpu", budget=None):
        """int64 values to their residues in [0, p)"""
        v = np.ascontiguousarray(x, dtype=np.int64)
        out = np.empty(len(v), dtype=np.uint64)
        args = (C.byref(self.raw), v.ctypes.data if len(v) else None, len(v), out.ctypes.data if len(v) else None)
        if _backend(backend) == "cpu":
            _code(_api()["gfp_reduce_i64"](*args), what="GF(p) reduce")
            return Batch(out, "cpu", "native CPU")
        refusal = _Refusal()
        used, name = _run_gpu("tr_mgpu_carriers_gfp_reduce_i64", C.c_int,
                              [C.POINTER(_GFP), C.c_void_p, C.c_uint64, C.c_void_p, C.POINTER(_Budget),
                               C.POINTER(_Refusal)],
                              *args, C.byref((budget or Budget()).raw), C.byref(refusal), refusal=refusal,
                              what="GF(p) reduce on the device")
        return Batch(out, used, name)


class GFElement(_Carrier):
    """a residue v in [0, p) of one field; elements of different fields never mix (CarrierError)"""
    __slots__ = ("field", "v")
    carrier = "gfp"

    def __init__(self, field, v):
        self.field, self.v = field, int(v)

    def _raw(self):
        return _Elem(self.field.p, self.v)

    def _bin(self, name, other):
        if not isinstance(other, GFElement):
            return _mixed("gfp", carrier_of(other))
        out = _Elem()
        _code(_api()[name](self._raw(), other._raw(), C.byref(out)), what=f"GF(p) {name[4:]}")
        return GFElement(self.field, out.v)

    __add__ = lambda s, o: s._bin("gfp_add", o)
    __sub__ = lambda s, o: s._bin("gfp_sub", o)
    __mul__ = lambda s, o: s._bin("gfp_mul", o)
    __truediv__ = lambda s, o: s._bin("gfp_div", o)
    __radd__ = lambda s, o: s._bin("gfp_add", o)
    __rmul__ = lambda s, o: s._bin("gfp_mul", o)

    def _un(self, name):
        out = _Elem()
        _code(_api()[name](self._raw(), C.byref(out)), what=f"GF(p) {name[4:]}")
        return GFElement(self.field, out.v)

    def __neg__(self):
        return self._un("gfp_neg")

    def inverse(self):
        return self._un("gfp_inv")

    def __pow__(self, e):
        e = int(e)
        if not I64_MIN <= e <= I64_MAX:
            raise OverflowError("the exponent is an int64")
        out = _Elem()
        _code(_api()["gfp_pow"](self._raw(), e, C.byref(out)), what="GF(p) pow")
        return GFElement(self.field, out.v)

    def lift(self, symmetric=False):
        """the explicit LIFT to an integer: v in [0, p), or the symmetric representative in (-p/2, p/2]"""
        out = C.c_int64()
        _code(_api()["gfp_lift_i64"](self._raw(), int(bool(symmetric)), C.byref(out)), what="lift")
        return int(out.value)

    def lift_bigint(self, symmetric=False, budget=None):
        return BigInt._wrap(_obj(_api()["gfp_lift"], self._raw(), int(bool(symmetric)), budget=budget, what="lift"),
                            budget)

    def __eq__(self, other):
        if not isinstance(other, GFElement):
            return NotImplemented
        return bool(_code(_api()["gfp_equal"](self._raw(), other._raw()), what="GF(p) equality"))

    def __hash__(self):
        return hash((self.field.p, self.v))

    def __repr__(self):
        return f"GF({self.field.p})({self.v})"


def _fields(fields):
    out = (_GFP * max(len(fields), 1))()
    for i, f in enumerate(fields):
        out[i] = (f if isinstance(f, GF) else GF(f)).raw
    return out


def crt(residues, fields, symmetric=False, budget=None):
    """(x, M): the unique x = residues[i] mod p_i in [0, M) (symmetric: (-M/2, M/2]), M = prod p_i"""
    if len(residues) != len(fields):
        raise ValueError("one residue per field")
    res = (C.c_uint64 * max(len(residues), 1))(*[int(r) for r in residues])
    x, m, refusal = C.c_void_p(), C.c_void_p(), _Refusal()
    _code(_api()["carrier_crt"](res, _fields(fields), len(residues), int(bool(symmetric)), C.byref(x), C.byref(m),
                                _b(budget), C.byref(refusal)), refusal, "CRT")
    return BigInt._wrap(x.value).to_int(), BigInt._wrap(m.value).to_int()


def crt_combine(x, m, residue, field, budget=None):
    """one Garner step: from x mod m and residue mod p to (x', m p) with x' in [0, m p)"""
    bx, bm = BigInt(x, budget), BigInt(m, budget)
    f = field if isinstance(field, GF) else GF(field)
    xo, mo, refusal = C.c_void_p(), C.c_void_p(), _Refusal()
    _code(_api()["carrier_crt_combine"](bx._h, bm._h, int(residue), C.byref(f.raw), C.byref(xo), C.byref(mo),
                                        _b(budget), C.byref(refusal)), refusal, "CRT step")
    return BigInt._wrap(xo.value).to_int(), BigInt._wrap(mo.value).to_int()


def rational_reconstruct(u, m, N=None, D=None, budget=None):
    """the fraction n/d with |n| <= N, 0 < d <= D, n = u d (mod m) (default N = D = isqrt((m-1)/2)); NoSolutionError
    when none exists"""
    bu, bm = BigInt(u, budget), BigInt(m, budget)
    bn = BigInt(N, budget) if N is not None else None
    bd = BigInt(D, budget) if D is not None else None
    h = _obj(_api()["carrier_ratrecon"], bu._h, bm._h, bn._h if bn else None, bd._h if bd else None, budget=budget,
             what="rational reconstruction")
    return BigRational._wrap(h, budget).to_fraction()


# ---------------------------------------------------------------------------------------------------- intervals
_PRECISION = 53


def _endpoint(x):
    """a binary64 endpoint given as is: a float, or an integer that is exactly a double (never rounded silently;
    other exact values go through Interval.enclose)"""
    if isinstance(x, float):
        return x
    if isinstance(x, Integral) and not isinstance(x, bool):
        f = float(x) if abs(int(x)) < 1 << 1024 else None
        if f is not None and f == x:
            return f
        raise ValueError(f"{x} is not exactly a double: use Interval.enclose for its enclosure")
    raise TypeError(f"an interval endpoint is a float, not {type(x).__name__}: use Interval.enclose for exact values")


class Interval(_Carrier):
    """a rigorous real enclosure [lo, hi] (binary64 endpoints): an APPROXIMATE carrier, never an exact answer"""
    __slots__ = ("lo", "hi")
    carrier = "interval"
    approximate = True
    certified = True
    precision_bits = _PRECISION

    def __init__(self, lo, hi=None):
        lo = _endpoint(lo)
        hi = lo if hi is None else _endpoint(hi)
        out = _Interval()
        raw = _Interval(lo, hi)
        _code(_api()["carrier_interval_op"](OPS["hull"], C.byref(raw), C.byref(raw), 0, C.byref(out)),
              what="interval (lo <= hi, no NaN)")
        self.lo, self.hi = lo, hi

    @classmethod
    def _from_raw(cls, raw):
        x = cls.__new__(cls)
        x.lo, x.hi = raw.lo, raw.hi
        return x

    def _raw(self):
        return _Interval(self.lo, self.hi)

    @classmethod
    def _enclosure(cls, fn, *args):
        out = _Interval()
        _code(fn(*args, C.byref(out)), what="enclosure")
        return cls._from_raw(out)

    @classmethod
    def point(cls, x):
        """the LOSSLESS binary64 -> interval conversion [x, x]"""
        return cls(_endpoint(x), _endpoint(x))

    @classmethod
    def enclose(cls, x):
        """the tight ENCLOSING conversion [RD(x), RU(x)] of an exact value"""
        api = _api()
        if isinstance(x, I64):
            return cls._enclosure(api["carrier_enclose_i64"], x.value)
        if isinstance(x, I128):
            return cls._enclosure(api["carrier_enclose_i128"], _i128(x.value))
        if isinstance(x, Q64):
            return cls._enclosure(api["carrier_enclose_q64"], x.raw)
        if isinstance(x, BigInt):
            return x.enclose()
        if isinstance(x, BigRational):
            return x.enclose()
        if isinstance(x, Integral) and not isinstance(x, bool):
            return BigInt(x).enclose()
        if isinstance(x, Fraction):
            return BigRational(x).enclose()
        return _mixed("interval", carrier_of(x), "enclosure of")

    def _op(self, op, other=None, n=0):
        if other is not None and not isinstance(other, Interval):
            return _mixed("interval", carrier_of(other))
        out = _Interval()
        a = self._raw()
        b = other._raw() if other is not None else None
        _code(_api()["carrier_interval_op"](OPS[op], C.byref(a), C.byref(b) if b is not None else None, int(n),
                                            C.byref(out)), what=f"interval {op}")
        return Interval._from_raw(out)

    __add__ = lambda s, o: s._op("add", o)
    __sub__ = lambda s, o: s._op("sub", o)
    __mul__ = lambda s, o: s._op("mul", o)
    __truediv__ = lambda s, o: s._op("div", o)

    def __neg__(self):
        return self._op("neg")

    def __abs__(self):
        return self._op("abs")

    def __pow__(self, n):
        n = int(n)
        if not I64_MIN <= n <= I64_MAX:
            raise OverflowError("the exponent is an int64")
        return self._op("pow", n=n)

    def sqr(self):
        return self._op("sqr")

    def sqrt(self):
        return self._op("sqrt")

    def inverse(self):
        return self._op("inv")

    def hull(self, other):
        return self._op("hull", other)

    def intersect(self, other):
        return self._op("intersect", other)

    def cmp(self, other):
        """three-valued: "less", "greater", "equal" (the same point) or "unknown" (overlapping enclosures)"""
        a, b = self._raw(), other._raw()
        c = _code(_api()["carrier_interval_cmp"](C.byref(a), C.byref(b)) + 1, what="interval comparison") - 1
        return {-1: "less", 0: "equal", 1: "greater", 2: "unknown"}[c]

    def width(self):
        """an upper bound on hi - lo (rounded up)"""
        if self.hi == float("inf") or self.lo == float("-inf"):
            return float("inf")
        return (Interval.point(self.hi) - Interval.point(self.lo)).hi

    def contains(self, x):
        """exact lo <= x <= hi for an exact value x"""
        return _as_bigq(x).in_interval(self)

    def to_exact(self):
        """refused: an enclosure has no exact value (interval -> bigq is FORBIDDEN in the conversion matrix)"""
        _code(_api()["carrier_check_conversion"](CARRIERS["interval"], CARRIERS["bigq"]), what="interval -> exact")

    def as_pair(self):
        return (self.lo, self.hi)

    @property
    def state(self):
        return dict(lo=self.lo, hi=self.hi, approximate=True, certified=True, precision_bits=_PRECISION)

    def __eq__(self, other):
        """the same enclosure (endpoints), not an equality of values"""
        return isinstance(other, Interval) and (self.lo, self.hi) == (other.lo, other.hi)

    def __hash__(self):
        return hash(("interval", self.lo, self.hi))

    def __repr__(self):
        return f"Interval([{self.lo!r}, {self.hi!r}], approximate)"


class ComplexInterval(_Carrier):
    """a rigorous complex enclosure, the rectangle re x im (APPROXIMATE)"""
    __slots__ = ("re", "im")
    carrier = "cinterval"
    approximate = True
    certified = True
    precision_bits = _PRECISION

    def __init__(self, re, im=None):
        self.re = re if isinstance(re, Interval) else Interval(*re) if isinstance(re, tuple) else Interval.point(re)
        self.im = (Interval(0.0) if im is None else im if isinstance(im, Interval)
                   else Interval(*im) if isinstance(im, tuple) else Interval.point(im))

    def _raw(self):
        return _CInterval(self.re._raw(), self.im._raw())

    @classmethod
    def _from_raw(cls, raw):
        return cls(Interval._from_raw(raw.re), Interval._from_raw(raw.im))

    def _op(self, op, other=None):
        if other is not None and not isinstance(other, ComplexInterval):
            return _mixed("cinterval", carrier_of(other))
        out = _CInterval()
        a = self._raw()
        b = other._raw() if other is not None else None
        _code(_api()["carrier_cinterval_op"](OPS[op], C.byref(a), C.byref(b) if b is not None else None, C.byref(out)),
              what=f"complex interval {op}")
        return ComplexInterval._from_raw(out)

    __add__ = lambda s, o: s._op("add", o)
    __sub__ = lambda s, o: s._op("sub", o)
    __mul__ = lambda s, o: s._op("mul", o)
    __truediv__ = lambda s, o: s._op("div", o)

    def __neg__(self):
        return self._op("neg")

    def conj(self):
        return self._op("conj")

    def sqr(self):
        return self._op("sqr")

    def inverse(self):
        return self._op("inv")

    def __abs__(self):
        out, a = _Interval(), self._raw()
        _code(_api()["carrier_cinterval_abs"](C.byref(a), C.byref(out)), what="|z|")
        return Interval._from_raw(out)

    def real(self):
        """NARROWING to a real interval: NoSolutionError unless the imaginary part is exactly [0, 0]"""
        out, a = _Interval(), self._raw()
        _code(_api()["carrier_cinterval_real"](C.byref(a), C.byref(out)), what="complex -> real")
        return Interval._from_raw(out)

    def __eq__(self, other):
        return isinstance(other, ComplexInterval) and self.re == other.re and self.im == other.im

    def __hash__(self):
        return hash(("cinterval", self.re, self.im))

    def __repr__(self):
        return f"ComplexInterval({self.re.as_pair()} + i {self.im.as_pair()}, approximate)"


def enclose_q64(numerators, denominators, backend="cpu", budget=None):
    """tight enclosures of int64 rationals n[i] / d[i] (d != 0): Batch of (lo array, hi array)"""
    n = np.ascontiguousarray(numerators, dtype=np.int64)
    d = np.ascontiguousarray(denominators, dtype=np.int64)
    if n.shape != d.shape or n.ndim != 1:
        raise ValueError("numerators and denominators are 1-d arrays of one length")
    q = np.empty((len(n), 2), dtype=np.int64)
    q[:, 0], q[:, 1] = n, d
    out = np.empty((len(n), 2), dtype=np.float64)
    args = (q.ctypes.data if len(n) else None, len(n), out.ctypes.data if len(n) else None)
    if _backend(backend) == "cpu":
        _code(_api()["carrier_enclose_q64_vec"](*args), what="enclose q64")
        return Batch((out[:, 0].copy(), out[:, 1].copy()), "cpu", "native CPU")
    refusal = _Refusal()
    used, name = _run_gpu("tr_mgpu_carriers_enclose_q64", C.c_int,
                          [C.c_void_p, C.c_uint64, C.c_void_p, C.POINTER(_Budget), C.POINTER(_Refusal)],
                          *args, C.byref((budget or Budget()).raw), C.byref(refusal), refusal=refusal,
                          what="enclose q64 on the device")
    return Batch((out[:, 0].copy(), out[:, 1].copy()), used, name)


def sum_i64(values, groups=None, ngroups=None, backend="cpu", budget=None):
    """exact per-group sums of int64 values in 128 bits: Batch of Python ints (one per group)"""
    x = np.ascontiguousarray(values, dtype=np.int64)
    g = None if groups is None else np.ascontiguousarray(groups, dtype=np.uint32)
    if g is not None and len(g) != len(x):
        raise ValueError("one group id per value")
    k = 1 if ngroups is None else int(ngroups)
    out = (_I128 * max(k, 1))()
    empty = (C.c_uint32 * 1)()                     # a non-NULL group array for zero values
    gp = None if g is None else (g.ctypes.data if len(g) else C.addressof(empty))
    args = (x.ctypes.data if len(x) else None, len(x), gp, k, out)
    if _backend(backend) == "cpu":
        _code(_api()["carrier_sum_i64"](*args), what="exact sums")
        used, name = "cpu", "native CPU"
    else:
        refusal = _Refusal()
        used, name = _run_gpu("tr_mgpu_carriers_sum_i64", C.c_int,
                              [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64, C.c_void_p, C.POINTER(_Budget),
                               C.POINTER(_Refusal)],
                              *args, C.byref((budget or Budget()).raw), C.byref(refusal), refusal=refusal,
                              what="exact sums on the device")
    return Batch([_from_i128(out[i]) for i in range(k)], used, name)


def dot(a, b, carrier="bigint", backend="cpu", budget=None):
    """the matched dot product sum(a[i] b[i]) in one carrier: "i64" (checked; OverflowError), "i128" (checked),
    "wide" (exact 192-bit; backend "cuda" runs the device kernel) or "bigint" (exact, a and b any Python ints)"""
    if carrier == "bigint":
        if len(a) != len(b):
            raise ValueError("operands of different lengths")
        ka, pa = _batch_handles(a, budget)
        kb, pb = _batch_handles(b, budget)
        h = _obj(_api()["bigint_dot"], pa, pb, len(a), budget=budget, what="dot")
        del ka, kb
        return BigInt._wrap(h, budget).to_int()
    x, y = _i64_array(a), _i64_array(b)
    if len(x) != len(y):
        raise ValueError("operands of different lengths")
    px, py = (x.ctypes.data if len(x) else None), (y.ctypes.data if len(y) else None)
    if carrier == "i64":
        out = C.c_int64()
        _code(_api()["carrier_dot_i64"](px, py, len(x), C.byref(out)), what="dot in int64")
        return int(out.value)
    if carrier == "i128":
        out = _I128()
        _code(_api()["carrier_dot_i128"](px, py, len(x), C.byref(out)), what="dot in 128 bits")
        return _from_i128(out)
    if carrier == "wide":
        out = (C.c_uint64 * 3)()
        if _backend(backend) == "cpu":
            _code(_api()["carrier_dot_wide"](px, py, len(x), C.byref(out)), what="exact dot")
        else:
            refusal = _Refusal()
            _run_gpu("tr_mgpu_carriers_dot_wide", C.c_int,
                     [C.c_void_p, C.c_void_p, C.c_uint64, C.POINTER(C.c_uint64 * 3), C.POINTER(_Budget),
                      C.POINTER(_Refusal)],
                     px, py, len(x), C.byref(out), C.byref((budget or Budget()).raw), C.byref(refusal),
                     refusal=refusal, what="exact dot on the device")
        return int.from_bytes(bytes(out), "little", signed=True)
    raise ValueError("carrier is 'i64', 'i128', 'wide' or 'bigint'")


def _i64_array(v):
    if isinstance(v, np.ndarray) and v.dtype == np.int64:
        return np.ascontiguousarray(v)
    vals = [int(t) for t in v]
    for t in vals:
        if not I64_MIN <= t <= I64_MAX:
            raise OverflowError("an int64 array entry does not fit int64")
    return np.array(vals, dtype=np.int64)


def horner(coef, xs, carrier="bigint", budget=None):
    """c[0] + x (c[1] + x (...)) at each point x in one carrier: "i64" (checked), "i128" (checked), "bigint" (exact)"""
    if not len(coef):
        raise ValueError("a polynomial needs a coefficient")
    if carrier == "bigint":
        kc, pc = _batch_handles(coef, budget)
        out = []
        for x in xs:
            bx = x if isinstance(x, BigInt) else BigInt(x, budget)
            out.append(BigInt._wrap(_obj(_api()["bigint_horner"], pc, len(coef), bx._h, budget=budget,
                                         what="horner"), budget).to_int())
        del kc
        return out
    c, x = _i64_array(coef), _i64_array(xs)
    if carrier == "i64":
        out = np.empty(len(x), dtype=np.int64)
        _code(_api()["carrier_horner_i64"](c.ctypes.data, len(c), x.ctypes.data if len(x) else None, len(x),
                                            out.ctypes.data if len(x) else None), what="horner in int64")
        return [int(v) for v in out]
    if carrier == "i128":
        out = (_I128 * max(len(x), 1))()
        _code(_api()["carrier_horner_i128"](c.ctypes.data, len(c), x.ctypes.data if len(x) else None, len(x), out),
              what="horner in 128 bits")
        return [_from_i128(out[i]) for i in range(len(x))]
    raise ValueError("carrier is 'i64', 'i128' or 'bigint'")


# ---------------------------------------------------------------------------------------------------- series
class Series(_Carrier):
    """c[0] + c[1] x + ... + c[N-1] x^(N-1) + O(x^N) over Q, exact up to its order N; series of different orders never
    combine (CarrierError): truncate one explicitly"""
    __slots__ = ("_h", "budget")
    carrier = "series"

    def __init__(self, coefficients, order, budget=None):
        coefs = [_as_bigq(c, budget) for c in coefficients]
        arr = (C.c_void_p * max(len(coefs), 1))(*[c._h for c in coefs])
        order = int(order)
        if not 0 <= order < 1 << 32:
            raise ValueError("the order is a 32-bit unsigned integer")
        self.budget = budget
        self._h = _obj(_api()["series_create"], arr, len(coefs), order, budget=budget, what="series")

    @classmethod
    def _wrap(cls, handle, budget=None):
        s = cls.__new__(cls)
        s._h, s.budget = handle, budget
        return s

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _N is not None:
            _N["series_free"](h)
            self._h = None

    @classmethod
    def constant(cls, c, order, budget=None):
        """the LOSSLESS conversion of an exact rational c to c + O(x^order)"""
        keep = _as_bigq(c, budget)
        return cls._wrap(_obj(_api()["series_constant"], keep._h, int(order), budget=budget, what="series"), budget)

    @property
    def order(self):
        return int(_api()["series_order"](self._h))

    def coefficient(self, k):
        api = _api()
        h = api["series_coeff"](self._h, int(k))       # borrowed: valid while this series lives
        if not h:
            raise IndexError(f"coefficient {k} is beyond O(x^{self.order})")
        n, d = _bigint_value(api["bigq_numerator"](h)), _bigint_value(api["bigq_denominator"](h))
        make = getattr(Fraction, "_from_coprime_ints", None)
        return make(n, d) if make else Fraction(n, d)

    @property
    def coefficients(self):
        return [self.coefficient(k) for k in range(self.order)]

    def _un(self, name, budget=None):
        b = budget or self.budget
        return Series._wrap(_obj(_api()[name], self._h, budget=b, what=name[7:]), b)

    def _bin(self, name, other, budget=None):
        if not isinstance(other, Series):
            return _mixed("series", carrier_of(other))
        b = budget or self.budget
        return Series._wrap(_obj(_api()[name], self._h, other._h, budget=b, what=name[7:]), b)

    def add(self, other, budget=None):
        return self._bin("series_add", other, budget)

    def sub(self, other, budget=None):
        return self._bin("series_sub", other, budget)

    def mul(self, other, budget=None):
        return self._bin("series_mul", other, budget)

    def compose(self, inner, budget=None):
        """self(inner) for inner with a zero constant term"""
        return self._bin("series_compose", inner, budget)

    __add__, __sub__, __mul__, __call__ = add, sub, mul, compose

    def scale(self, c, budget=None):
        b = budget or self.budget
        keep = _as_bigq(c, b)
        return Series._wrap(_obj(_api()["series_scale"], self._h, keep._h, budget=b, what="scale"), b)

    def derivative(self, budget=None):
        return self._un("series_derivative", budget)

    def integral(self, budget=None):
        return self._un("series_integral", budget)

    def reciprocal(self, budget=None):
        return self._un("series_reciprocal", budget)

    def exp(self, budget=None):
        return self._un("series_exp", budget)

    def log(self, budget=None):
        return self._un("series_log", budget)

    def truncate(self, order, budget=None):
        b = budget or self.budget
        return Series._wrap(_obj(_api()["series_truncate"], self._h, int(order), budget=b, what="truncate"), b)

    def __eq__(self, other):
        if not isinstance(other, Series):
            return NotImplemented
        return bool(_code(_api()["series_equal"](self._h, other._h), what="series equality"))

    def __hash__(self):
        return hash((self.order, tuple(self.coefficients)))

    def __repr__(self):
        terms = [f"{c}" + ("" if k == 0 else f"*x^{k}") for k, c in enumerate(self.coefficients) if c]
        return "Series(" + " + ".join(terms + [f"O(x^{self.order})"]) + ")"


# ---------------------------------------------------------------------------------------------------- conversions
def convert(value, to, *, field=None, order=None, symmetric=False, budget=None):
    """the explicit conversion of value to the carrier `to`, with the kind the native matrix gives it: forbidden
    conversions raise CarrierError, narrowing ones OverflowError / NoSolutionError when the value does not fit"""
    src = carrier_of(value)
    kind = conversion(src, to)
    if kind == "forbidden":
        _code(_api()["carrier_check_conversion"](CARRIERS[src], CARRIERS[to]), what=f"{src} -> {to}")
    if kind == "identity":
        if src == "gfp" and field is not None:
            return field(value)
        return value
    api = _api()
    if src == "bigint" and not isinstance(value, BigInt):
        value = BigInt(value, budget)
    if src == "bigq" and not isinstance(value, BigRational):
        value = BigRational(value, budget=budget)
    if to == "gfp":
        if field is None:
            raise ValueError("a reduction into GF(p) needs field=")
        return field(value)
    if kind == "lift":
        return I64(value.lift(symmetric)) if to == "i64" else value.lift_bigint(symmetric, budget)
    if to in ("interval", "cinterval"):
        if src == "binary64":
            iv = Interval.point(value)
        elif src == "interval":
            iv = value
        elif src == "cinterval":
            return value.real()
        else:
            iv = Interval.enclose(value)
        return iv if to == "interval" else ComplexInterval(iv, Interval(0.0))
    if to == "series":
        if src == "series":
            return value.truncate(order if order is not None else value.order, budget)
        if order is None:
            raise ValueError("a constant series needs order=")
        return Series.constant(_exact_bigq(value, src, budget), order, budget)
    # the exact carriers: i64, i128, bigint, q64, bigq
    if src == "binary64":
        return BigRational.from_double(value, budget)
    exact = _exact_bigq(value, src, budget)
    if to == "bigq":
        return exact
    if to == "q64":
        return exact.to_q64()
    integer = exact.to_bigint(budget)
    if to == "bigint":
        return integer
    if to == "i128":
        return I128(integer.to_i128())
    if to == "i64":
        return I64(integer.to_i64())
    raise ValueError(f"unknown carrier {to!r}")


def _exact_bigq(value, src, budget):
    if src == "i64":
        return BigRational.from_bigint(BigInt.from_i64(value.value, budget), budget)
    if src == "i128":
        return BigRational.from_bigint(BigInt.from_i128(value.value, budget), budget)
    if src == "bigint":
        return BigRational.from_bigint(value, budget)
    if src == "q64":
        return BigRational.from_q64(value, budget)
    return value
