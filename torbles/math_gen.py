# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEN-001: generating polynomials of finite-support distributions (torbles/math_gen.c; the 128-bit convolution and the
batched evaluation also in CUDA, math_gen_gpu.cu), with exact big-integer coefficients (FND-009 carriers).

    from torbles import math_dist as md, math_gen as mg
    g = mg.generating(md.histogram(scope, ["x"]), group=0)       # G(z) = sum count(v) z^v; shift: the smallest value
    g.terms(); g.coefficient(-3); g.total()                       # {v: count}; c(-3); G(1) = rows - NULL rows
    g(Fraction(1, 2)); g.moment(2); g.moment(3, falling=True)     # exact G(1/2), sum c v^2, sum c v(v-1)(v-2)
    g.euler(2); g.derivative(1)                                    # (z d/dz)^2 G, G' (Laurent: the shift moves)
    (g * g).terms(); g.pow(4, order=16); g.series(16)             # sums of draws; truncated; an FND-009 Series
    g.mul(other, backend="cuda"); g.evaluate_many(points, backend="cuda")   # the 128-bit kernels
    mg.from_terms({-2: 1, 0: 3, 5: 2}); mg.from_terms({(0, 1): 2, (1, 0): 1})   # explicit (2 variables: tuples)

Values are logical integers (a DECIMAL(p, s) column's value is logical / 10^s); negative values use the declared shift
(G = z^shift P(z), a Laurent polynomial). Results are exact ints / Fractions computed natively; Python only marshals
and formats. Refusals: AlgebraBudgetError (before allocation: support growth, coefficient pairs, bytes), DomainError (a
value below a declared shift, products of different scales), OverflowError, NoSolutionError (a pole at z = 0),
UnsupportedError (3 coordinates, a series of 2 variables). Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_dist import Histogram, _i128, _rational
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["GeneratingPolynomial", "generating", "from_terms", "REPRS", "POINT_DTYPE", "VALUE_DTYPE"]

REPRS = {"dense": 1, "sparse": 2}
MASSES = {"count": 1, "weight": 2}
P = C.POINTER
i32, u32, i64, u64, vp = C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_void_p


class _Info(C.Structure):
    _native_ = "tr_gen_info"
    _fields_ = [("nterms", u64), ("nonzero", u64), ("box", u64), ("span", u64 * 2), ("shift", i64 * 2),
                ("vmin", i64 * 2), ("vmax", i64 * 2), ("rows", u64), ("nulls", u64), ("maxbits", u64), ("bytes", u64),
                ("nvars", i32), ("repr", i32), ("mass", i32), ("nonneg", i32), ("scale", i32 * 2), ("wscale", i32),
                ("reserved", i32)]


class _Point(C.Structure):
    _native_ = "tr_gen_point"
    _fields_ = [("num", i64), ("den", i64)]


class _Value(C.Structure):
    _native_ = "tr_gen_value"
    _fields_ = [("status", i32), ("reserved", i32), ("num", i64 * 2), ("den", i64 * 2)]


assert C.sizeof(_Info) == 152 and C.sizeof(_Point) == 16 and C.sizeof(_Value) == 40
POINT_DTYPE = np.dtype([("num", "<i8"), ("den", "<i8")])
VALUE_DTYPE = np.dtype([("status", "<i4"), ("reserved", "<i4"), ("num", "<i8", (2,)), ("den", "<i8", (2,))])

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_gen_from_hist": (vp, [vp, u64, i32, i32, vp, B, R, e]),
            "tr_gen_from_terms": (vp, [i32, u64, vp, vp, i32, vp, B, R, e]),
            "tr_gen_from_dense_i128": (vp, [i64, u64, vp, i32, B, R, e]),
            "tr_gen_get_info": (i32, [vp, P(_Info)]),
            "tr_gen_export_size": (i32, [vp, P(u64), P(u64)]),
            "tr_gen_export": (i32, [vp, vp, vp, vp, vp, u64, u64]),
            "tr_gen_coeff": (i32, [vp, vp, B, R, P(vp)]),
            "tr_gen_eval": (i32, [vp, vp, vp, B, R, P(vp), P(vp)]),
            "tr_gen_moment": (i32, [vp, vp, i32, B, R, P(vp)]),
            "tr_gen_euler": (vp, [vp, i32, u32, B, R, e]),
            "tr_gen_derivative": (vp, [vp, i32, u32, B, R, e]),
            "tr_gen_mul": (vp, [vp, vp, u64, B, R, e]),
            "tr_gen_pow": (vp, [vp, u32, u64, B, R, e]),
            "tr_gen_mul_need": (i32, [vp, vp, u64, B, P(_Need), P(u64)]),
            "tr_gen_pow_need": (i32, [vp, u32, u64, B, P(_Need), P(u64)]),
            "tr_gen_truncate": (vp, [vp, u64, B, R, e]),
            "tr_gen_to_series": (vp, [vp, u32, B, R, e]),
            "tr_gen_convert": (vp, [vp, i32, B, R, e]),
            "tr_gen_marginal": (vp, [vp, i32, B, R, e]),
            "tr_gen_equal": (i32, [vp, vp]),
            "tr_gen_dense_i64": (i32, [vp, vp, u64]),
            "tr_gen_free": (None, [vp]),
            "tr_gen_conv_i128": (i32, [vp, u64, vp, u64, vp, B, R]),
            "tr_gen_eval_i128": (i32, [vp, u64, i64, vp, u64, vp, B, R]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, args
            native[name] = fn
        _native = native
    return _native


def _call(name, *args):
    return _api()[name](*args)


def _gpu():
    from . import math_gpu
    math_gpu.bind("tr_mgpu_gen_conv", C.c_int, [vp, u64, vp, u64, vp, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_gen_eval", C.c_int, [vp, u64, i64, vp, u64, vp, P(_Budget), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _big(handle):
    """a native FND-009 big integer as a Python int (the handle is freed)"""
    from . import math_carriers as mc
    try:
        return mc._bigint_value(handle)
    finally:
        mc._api()["bigint_free"](handle)


def _int64(v, what):
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError(f"{what} must be an exact integer")
    v = int(v)
    if not -(1 << 63) <= v < (1 << 63):
        raise OverflowError(f"{what} does not fit int64")
    return v


def _shift(shift, nvars):
    if shift is None:
        return None
    s = [shift] if isinstance(shift, Integral) else list(shift)
    if len(s) != nvars:
        raise ValueError(f"one shift per variable ({nvars})")
    return (C.c_int64 * 2)(*([_int64(x, "shift") for x in s] + [0] * (2 - nvars)))


def generating(hist, group=0, *, mass="count", repr="sparse", shift=None, budget=None):
    """the generating polynomial of group g of a 1-D or 2-D histogram: coefficients = rows (count) or weight sums;
    shift None: the smallest present value per variable, else the declared shift (at most every value)"""
    if not isinstance(hist, Histogram):
        raise TypeError("a math_dist.Histogram")
    if hist.k > 2:
        raise UnsupportedError("generating polynomials have one or two variables")
    if mass not in MASSES or repr not in REPRS:
        raise ValueError("mass is 'count' or 'weight'; repr is 'sparse' or 'dense'")
    s = _shift(shift, hist.k)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_gen_from_hist", hist._open(), int(group), MASSES[mass], REPRS[repr], s, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return GeneratingPolynomial._wrap(_made(h, error, refusal, "generating polynomial"), hist.names)


def from_terms(terms, *, repr="sparse", shift=None, budget=None):
    """a generating polynomial from {value: coefficient} (one variable) or {(v1, v2): coefficient}; exact integer
    coefficients of any sign below 2^127; duplicates add, zeros are dropped"""
    items = list(terms.items()) if isinstance(terms, dict) else list(terms)
    if repr not in REPRS:
        raise ValueError("repr is 'sparse' or 'dense'")
    nv = 2 if items and isinstance(items[0][0], tuple) else 1
    vals = np.zeros((max(len(items), 1), nv), dtype=np.int64)
    coef = np.zeros((max(len(items), 1), 2), dtype=np.int64)
    for i, (v, c) in enumerate(items):
        v = v if isinstance(v, tuple) else (v,)
        if len(v) != nv:
            raise ValueError("every value has the same number of variables")
        for j, x in enumerate(v):
            vals[i, j] = _int64(x, "a value")
        if isinstance(c, bool) or not isinstance(c, Integral) or not -(1 << 127) <= int(c) < (1 << 127):
            raise TypeError("coefficients are integers in [-2^127, 2^127)")
        c = int(c) & ((1 << 128) - 1)
        lo, hi = c & ((1 << 64) - 1), c >> 64
        coef[i, 0], coef[i, 1] = lo - (1 << 64) if lo >> 63 else lo, hi - (1 << 64) if hi >> 63 else hi
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_gen_from_terms", nv, len(items), vals.ctypes.data, coef.ctypes.data, REPRS[repr], _shift(shift, nv),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return GeneratingPolynomial._wrap(_made(h, error, refusal, "generating polynomial"), None)


def _rat_point(z, nvars, what="point"):
    zs = list(z) if isinstance(z, (tuple, list)) else [z]
    if len(zs) != nvars:
        raise ValueError(f"{what}: one coordinate per variable ({nvars})")
    nums, dens = (C.c_int64 * 2)(), (C.c_int64 * 2)()
    for i, x in enumerate(zs):
        nums[i], dens[i] = _rational(x, what)
    return nums, dens


class GeneratingPolynomial:
    """An immutable native generating polynomial G(z) = z^shift P(z) with exact coefficients."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_gen.generating(...), math_gen.from_terms(...) or an operation")

    @classmethod
    def _wrap(cls, handle, names=None):
        self = object.__new__(cls)
        self._handle, self.names = handle, tuple(names) if names else None
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the generating polynomial is closed")
        return self._handle

    def _raw_info(self):
        out = _Info()
        _check(_call("tr_gen_get_info", self._open(), C.byref(out)), what="generating polynomial")
        return out

    @property
    def info(self):
        i = self._raw_info()
        nv = int(i.nvars)
        return dict(variables=nv, repr={v: k for k, v in REPRS.items()}[int(i.repr)], terms=int(i.nterms),
                    nonzero=int(i.nonzero), box=int(i.box), span=tuple(int(i.span[c]) for c in range(nv)),
                    shift=tuple(int(i.shift[c]) for c in range(nv)),
                    min=tuple(int(i.vmin[c]) for c in range(nv)) if i.nonzero else None,
                    max=tuple(int(i.vmax[c]) for c in range(nv)) if i.nonzero else None,
                    rows=int(i.rows), nulls=int(i.nulls), max_bits=int(i.maxbits), bytes=int(i.bytes),
                    mass={0: None, 1: "count", 2: "weight"}[int(i.mass)], nonnegative=bool(i.nonneg),
                    scales=tuple(int(i.scale[c]) for c in range(nv)), wscale=int(i.wscale))

    @property
    def nvars(self):
        return int(self._raw_info().nvars)

    @property
    def shift(self):
        i = self._raw_info()
        return tuple(int(i.shift[c]) for c in range(int(i.nvars)))

    def terms(self):
        """{value: coefficient} (two variables: {(v1, v2): coefficient}) of every nonzero term, exact ints"""
        nz, nl = C.c_uint64(), C.c_uint64()
        _check(_call("tr_gen_export_size", self._open(), C.byref(nz), C.byref(nl)), what="export")
        nv, n, L = self.nvars, int(nz.value), int(nl.value)
        vals = np.zeros(max(n * nv, 1), dtype=np.int64)
        signs = np.zeros(max(n, 1), dtype=np.int32)
        loff = np.zeros(n + 1, dtype=np.uint64)
        limbs = np.zeros(max(L, 1), dtype=np.uint64)
        _check(_call("tr_gen_export", self._open(), vals.ctypes.data, signs.ctypes.data, loff.ctypes.data,
                     limbs.ctypes.data, n, L), what="export")
        raw = limbs.tobytes()
        out = {}
        for i in range(n):
            a, b = int(loff[i]), int(loff[i + 1])
            m = int.from_bytes(raw[8 * a:8 * b], "little")
            key = int(vals[i]) if nv == 1 else (int(vals[2 * i]), int(vals[2 * i + 1]))
            out[key] = -m if signs[i] < 0 else m
        return out

    def coefficient(self, value):
        """c(v), 0 off the support"""
        v = list(value) if isinstance(value, tuple) else [value]
        arr = (C.c_int64 * 2)(*([_int64(x, "value") for x in v] + [0] * (2 - len(v))))
        if len(v) != self.nvars:
            raise ValueError("one value per variable")
        out, refusal = vp(), _Refusal()
        rc = _call("tr_gen_coeff", self._open(), arr, None, C.byref(refusal), C.byref(out))
        _check(rc, refusal, "coefficient")
        return _big(out.value)

    def evaluate(self, z, *, budget=None):
        """the exact value G(z) at a rational point (int, Fraction or Decimal per variable)"""
        nums, dens = _rat_point(z, self.nvars)
        num, den, refusal = vp(), vp(), _Refusal()
        rc = _call("tr_gen_eval", self._open(), nums, dens, C.byref(_budget(budget)), C.byref(refusal), C.byref(num),
                   C.byref(den))
        _check(rc, refusal, "evaluation")
        v = Fraction(_big(num.value), _big(den.value))
        return v.numerator if v.denominator == 1 else v

    __call__ = evaluate

    def total(self):
        """G(1): the total mass (rows without a NULL coordinate, or the weight sum)"""
        return self.evaluate((1,) * self.nvars)

    def moment(self, k, *, falling=False, budget=None):
        """sum c(v) v^k (raw: (z d/dz)^k G at 1) or sum c(v) (v)_k (falling: the k-th derivative at 1); two variables:
        k = (k1, k2) for the mixed moment"""
        ks = [k] if isinstance(k, Integral) else list(k)
        if len(ks) != self.nvars or any(int(x) < 0 for x in ks):
            raise ValueError("one nonnegative order per variable")
        arr = (C.c_uint32 * 2)(*([int(x) for x in ks] + [0] * (2 - len(ks))))
        out, refusal = vp(), _Refusal()
        rc = _call("tr_gen_moment", self._open(), arr, int(bool(falling)), C.byref(_budget(budget)), C.byref(refusal),
                   C.byref(out))
        _check(rc, refusal, "moment")
        return _big(out.value)

    def _derived(self, handle, error, refusal, what):
        return GeneratingPolynomial._wrap(_made(handle, error, refusal, what), self.names)

    def euler(self, k=1, *, var=0, budget=None):
        """(z_var d/dz_var)^k G: coefficients c(v) v^k"""
        error, refusal = C.c_int32(), _Refusal()
        return self._derived(_call("tr_gen_euler", self._open(), int(var), int(k), C.byref(_budget(budget)),
                                   C.byref(refusal), C.byref(error)), error, refusal, "Euler operator")

    def derivative(self, k=1, *, var=0, budget=None):
        """d^k/dz_var^k G: coefficients c(v) (v)_k, the shift lowered by k (a Laurent polynomial)"""
        error, refusal = C.c_int32(), _Refusal()
        return self._derived(_call("tr_gen_derivative", self._open(), int(var), int(k), C.byref(_budget(budget)),
                                   C.byref(refusal), C.byref(error)), error, refusal, "derivative")

    def dense_i64(self):
        """the coefficients over the box as an int64 array (one variable; OverflowError past int64)"""
        span = self.info["span"][0]
        out = np.zeros(max(span, 1), dtype=np.int64)
        _check(_call("tr_gen_dense_i64", self._open(), out.ctypes.data, len(out)), what="dense coefficients")
        return out[:span]

    def mul(self, other, *, order=None, backend="cpu", budget=None):
        """the product (convolution): the distribution of the sum of independent draws; order: keep the exponents
        e < order of P (one variable). backend "cuda": the 128-bit kernel (one variable, int64 coefficients)."""
        if not isinstance(other, GeneratingPolynomial):
            raise TypeError("a GeneratingPolynomial")
        o = 0 if order is None else int(order)
        error, refusal = C.c_int32(), _Refusal()
        if backend == "cpu":
            return self._derived(_call("tr_gen_mul", self._open(), other._open(), o, C.byref(_budget(budget)),
                                       C.byref(refusal), C.byref(error)), error, refusal, "product")
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        if self.nvars != 1 or other.nvars != 1:
            raise UnsupportedError("the CUDA convolution takes one variable")
        if self.info["scales"] != other.info["scales"]:
            from .math_domains import DomainError
            raise DomainError("products of values of different decimal scales")
        a, b = self.dense_i64(), other.dense_i64()
        if not len(a) or not len(b):
            return self.mul(other, order=order, budget=budget)          # an empty factor: the empty product (CPU)
        out = np.zeros((len(a) + len(b) - 1, 2), dtype=np.int64)
        gpu = _gpu()
        gpu.run("tr_mgpu_gen_conv", a.ctypes.data, len(a), b.ctypes.data, len(b), out.ctypes.data,
                C.byref(_budget(budget)), C.byref(refusal), refusal=refusal, what="convolution on the device")
        repr_ = "dense" if self.info["repr"] == "dense" and other.info["repr"] == "dense" else "sparse"
        h = _call("tr_gen_from_dense_i128", _int64(self.shift[0] + other.shift[0], "the product's shift"), len(out),
                  out.ctypes.data, REPRS[repr_],
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        r = self._derived(h, error, refusal, "product")
        return r.truncate(o, budget=budget) if o else r

    __mul__ = lambda self, other: self.mul(other)                # noqa: E731

    def pow(self, k, *, order=None, budget=None):
        """G^k (k-fold convolution: sums of k i.i.d. draws), sequential products, optionally truncated"""
        error, refusal = C.c_int32(), _Refusal()
        return self._derived(_call("tr_gen_pow", self._open(), int(k), 0 if order is None else int(order),
                                   C.byref(_budget(budget)), C.byref(refusal), C.byref(error)), error, refusal, "power")

    __pow__ = lambda self, k: self.pow(k)                         # noqa: E731

    def need_mul(self, other, *, order=None, budget=None):
        """what the product checks before allocating, and the bound on its terms ("terms")"""
        need, terms = _Need(), C.c_uint64()
        _check(_call("tr_gen_mul_need", self._open(), other._open(), 0 if order is None else int(order),
                     C.byref(_budget(budget)), C.byref(need), C.byref(terms)), what="product needs")
        return dict({name: int(need.need[i]) for i, name in enumerate(RESOURCES) if need.need[i]}, terms=int(terms.value))

    def need_pow(self, k, *, order=None, budget=None):
        """what the power checks before its first product (its largest step), and the bound on its terms"""
        need, terms = _Need(), C.c_uint64()
        _check(_call("tr_gen_pow_need", self._open(), int(k), 0 if order is None else int(order),
                     C.byref(_budget(budget)), C.byref(need), C.byref(terms)), what="power needs")
        return dict({name: int(need.need[i]) for i, name in enumerate(RESOURCES) if need.need[i]}, terms=int(terms.value))

    def truncate(self, order, *, budget=None):
        """the terms with exponent e < order of P (one variable)"""
        error, refusal = C.c_int32(), _Refusal()
        return self._derived(_call("tr_gen_truncate", self._open(), int(order), C.byref(_budget(budget)),
                                   C.byref(refusal), C.byref(error)), error, refusal, "truncation")

    def series(self, order, *, budget=None):
        """P(z) mod z^order as an FND-009 truncated series (math_carriers.Series; G = z^shift P)"""
        from . import math_carriers as mc
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_gen_to_series", self._open(), int(order), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return mc.Series._wrap(_made(h, error, refusal, "series"), budget)

    def convert(self, repr, *, budget=None):
        if repr not in REPRS:
            raise ValueError("repr is 'sparse' or 'dense'")
        error, refusal = C.c_int32(), _Refusal()
        return self._derived(_call("tr_gen_convert", self._open(), REPRS[repr], C.byref(_budget(budget)),
                                   C.byref(refusal), C.byref(error)), error, refusal, "conversion")

    def to_dense(self, *, budget=None):
        return self.convert("dense", budget=budget)

    def to_sparse(self, *, budget=None):
        return self.convert("sparse", budget=budget)

    def marginal(self, var, *, budget=None):
        """G(z_var, 1) of a two-variable state"""
        error, refusal = C.c_int32(), _Refusal()
        return self._derived(_call("tr_gen_marginal", self._open(), int(var), C.byref(_budget(budget)),
                                   C.byref(refusal), C.byref(error)), error, refusal, "marginal")

    def evaluate_many(self, points, *, backend="cpu", budget=None):
        """G at many rational points by the checked 128-bit Horner (one variable, int64 coefficients): a Fraction per
        point, or None where a value leaves int128 (use evaluate for the exact big value); CPU and CUDA give the same
        bits. A pole (z = 0 with negative values) raises NoSolutionError."""
        if self.nvars != 1:
            raise UnsupportedError("the batched evaluation takes one variable")
        pts = np.zeros(max(len(points), 1), dtype=POINT_DTYPE)
        for i, z in enumerate(points):
            pts["num"][i], pts["den"][i] = _rational(z, "point")
        c = self.dense_i64()
        if not len(c):
            c = np.zeros(1, dtype=np.int64)
        out = np.zeros(max(len(points), 1), dtype=VALUE_DTYPE)
        refusal = _Refusal()
        args = (c.ctypes.data, len(c), self.shift[0], pts.ctypes.data, len(points), out.ctypes.data,
                C.byref(_budget(budget)), C.byref(refusal))
        if backend == "cpu":
            _check(_call("tr_gen_eval_i128", *args), refusal, "batched evaluation")
        elif backend == "cuda":
            _gpu().run("tr_mgpu_gen_eval", *args, refusal=refusal, what="batched evaluation on the device")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        self._last_values = out[:len(points)]
        res = []
        for r in out[:len(points)]:
            st = int(r["status"])
            if st == -3:
                res.append(None)
                continue
            _check(st, what="batched evaluation")
            v = Fraction(_i128(r["num"]), _i128(r["den"]))
            res.append(v.numerator if v.denominator == 1 else v)
        return res

    def equal(self, other):
        return bool(_check(_call("tr_gen_equal", self._open(), other._open()), what="comparison"))

    def __eq__(self, other):
        return isinstance(other, GeneratingPolynomial) and self.equal(other)

    __hash__ = None

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_gen_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "GeneratingPolynomial(closed)"
        i = self.info
        return f"GeneratingPolynomial({i['variables']} var, {i['repr']}, {i['nonzero']} terms, shift={i['shift']})"
