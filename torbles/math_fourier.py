# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SPEC-001: fixed spectral basis states over the rows of a scope, natively on the CPU or with the CUDA kernels
(torbles/math_fourier.c, math_fourier_gpu.cu).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_fourier as mf
    scope = mbk.Scope.from_arrays({"x": xs, "y": ys, "w": ws}, scales={"x": 2})
    B = mf.basis(scope, ["x", "y"], [(1, 0), (Fraction(1, 2), 3)], weight="w")   # F(omega) = sum w exp(i omega.x)
    B.coefficients()            # [((re_lo, re_hi), (im_lo, im_hi))]: certified enclosures, one per frequency
    B.bounds()                  # the exact rational bounds [(re_lo, re_hi), (im_lo, im_hi)] (Fractions)
    B.act([dict(shift=(0.5, -1.0), phase=0.25, order=(1, 0))])        # phase actions from the fixed basis
    B.characteristic(frequencies=[(1, 0)])                             # F / W, the empirical characteristic function
    B.derivative((1, 0)).exact(0, 0)                                   # exact transform: times i omega_1
    B.feature((2, 0))                                                  # DomainError: missing spectral support
    W = mf.walsh_basis(scope2, ["b0", "b1", "b2"])                     # the Boolean case (pi units): exact W(S)
    W.walsh_inverse()                                                  # the histogram, exactly (SPEC-002's inverse)
    cache = mf.Retained(); cache.basis(scope, ["x"], [1, 2]); cache.basis(scope, ["x"], [2])   # a hit (subset)

Frequencies are exact canonical int64 rationals (int, Fraction, or a float's exact value), one tuple of d per
frequency (a scalar per frequency when d = 1). Units: "radian" (theta = omega.x), "pi" (theta = pi omega.x) or
"turn" (theta = 2 pi omega.x). Every coefficient and action value is a complex rectangle that contains the exact
result (math_fourier.h: the precision contract); the derivative transforms are exact. A frequency outside the basis
is refused (DomainError), never treated as zero. Python marshals and formats only. Opt-in: `import torbles` does not
load this module.
"""
import ctypes as C
import math
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from . import math_carriers as _mc
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check, StaleStateError

__all__ = ["basis", "walsh_basis", "walsh_frequencies", "Basis", "Retained", "device_bytes", "UNITS", "MAX_DIMS",
           "CONTRACT"]

UNITS = {"radian": 0, "pi": 1, "turn": 2}
_UNIT_NAMES = {v: k for k, v in UNITS.items()}
MAX_DIMS, MAX_ORDER, WALSH_MAX_K = 16, 64, 16
CONTRACT = ("approximate, certified: every coefficient is the exact rational enclosure [lo, hi] / (2^scale "
            "10^wscale) of sums of outward-rounded per-row enclosures (exact angle reduction, CALC-003's 120-bit "
            "certified sin / cos, radius <= 2^-112 per row); actions are outward-rounded interval arithmetic; the "
            "Boolean and quarter-turn cases are exact points")
_I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER


class _Spec(C.Structure):
    _native_ = "tr_fourier_spec"
    _fields_ = [("ndims", C.c_int32), ("weight", C.c_int32), ("unit", C.c_int32), ("reserved", C.c_int32),
                ("column", C.c_int32 * MAX_DIMS)]


class _PlanView(C.Structure):
    _native_ = "tr_fourier_plan_view"
    _fields_ = [("ndims", C.c_int32), ("weight", C.c_int32), ("scale", C.c_int32), ("unit", C.c_int32),
                ("nfreq", C.c_uint64), ("ngroups", C.c_uint64), ("column", C.c_int32 * MAX_DIMS),
                ("k", C.c_void_p), ("ksign", C.c_void_p), ("kexact", C.c_void_p)]


class _Info(C.Structure):
    _native_ = "tr_fourier_info"
    _fields_ = [(n, C.c_int32) for n in ("ndims", "unit", "scale", "wscale", "weight", "exact", "phase_order",
                                         "unit_power")] + [
        (n, C.c_uint64) for n in ("ngroups", "nfreq", "rows", "selected")] + [("order", C.c_int32 * MAX_DIMS)]


for _s, _n in ((_Spec, 80), (_PlanView, 120), (_Info, 128)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, B, R, E = C.c_void_p, C.c_int32, C.c_uint64, P(_Budget), P(_Refusal), P(C.c_int32)
        sig = {
            "tr_fourier_plan_create": (vp, [vp, P(_Spec), u64, vp, B, R, E]),
            "tr_fourier_plan_get_view": (i32, [vp, P(_PlanView)]),
            "tr_fourier_sums": (i32, [P(_PlanView), vp, vp, vp, vp, vp]),
            "tr_fourier_from_sums": (vp, [vp, vp, vp, vp, vp, vp, B, R, E]),
            "tr_fourier_build": (vp, [vp, P(_Spec), u64, vp, B, R, E]),
            "tr_fourier_plan_free": (None, [vp]),
            "tr_fourier_device_need": (u64, [P(_PlanView), vp]),
            "tr_fourier_get_info": (i32, [vp, P(_Info)]),
            "tr_fourier_group": (i32, [vp, u64, P(u64), P(u64), P(_mc._I128)]),
            "tr_fourier_frequency": (i32, [vp, u64, vp]),
            "tr_fourier_coefficient": (i32, [vp, u64, u64, P(_mc._CInterval)]),
            "tr_fourier_bounds": (i32, [vp, u64, u64, vp]),
            "tr_fourier_exact": (i32, [vp, u64, u64, vp, P(i32), B, R]),
            "tr_fourier_find": (i32, [vp, u64, vp, vp, P(u64)]),
            "tr_fourier_require": (i32, [vp, vp, P(_Spec), u64, vp]),
            "tr_fourier_act": (i32, [vp, u64, u64, vp, vp, vp, vp, u64, vp, i32, vp, B, R]),
            "tr_fourier_derive": (vp, [vp, vp, i32, B, R, E]),
            "tr_fourier_walsh_inverse": (i32, [vp, u64, vp, vp, B, R]),
            "tr_fourier_walsh_support": (i32, [vp, P(i32), P(u64)]),
            "tr_fourier_free": (None, [vp]),
        }
        native = {}
        for name, (restype, argtypes) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            native[name[len("tr_fourier_"):]] = fn
        _native = native
    return _native


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_fourier_sums", C.c_int, [P(_PlanView), C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                                        C.c_void_p, P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _raw(budget):
    return C.byref((budget or Budget()).raw)


def _i128(x):
    return int.from_bytes(bytes(x), "little", signed=True)


def _iv(raw):
    return (raw.lo, raw.hi)


def _civ(raw):
    return ((raw.re.lo, raw.re.hi), (raw.im.lo, raw.im.hi))


# ---------------------------------------------------------------------- marshalling
def _pair(value, what):
    """(numerator, denominator) of an exact rational: int, Fraction, or a float's exact (dyadic) value, both in int64"""
    if isinstance(value, bool):
        raise TypeError(f"{what}: bool is not a number")
    if isinstance(value, Integral):
        n, d = int(value), 1
    elif isinstance(value, Rational):
        n, d = int(value.numerator), int(value.denominator)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{what}: a finite value is required")
        n, d = value.as_integer_ratio()
    else:
        raise TypeError(f"{what}: an int, a Fraction or a float is required")
    if not (_I64[0] <= n <= _I64[1] and 0 < d <= _I64[1]):
        raise OverflowError(f"{what}: {Fraction(n, d)} is not an int64 rational (numerator and denominator < 2^63)")
    return n, d


def _qarray(frequencies, d, what="frequency"):
    """an (n * d, 2) int64 array of (numerator, denominator): the tr_num_rational[] of n frequencies of d components"""
    flat = []
    for item in frequencies:
        if d == 1 and not isinstance(item, (tuple, list, np.ndarray)):
            item = (item,)
        if len(item) != d:
            raise ValueError(f"{what}: {d} components per frequency (got {len(item)})")
        flat.extend(_pair(v, what) for v in item)
    a = np.array(flat or [(0, 1)], dtype=np.int64).reshape(-1, 2)
    return a, len(flat) // d if d else 0


def _vectors(arr, n, d):
    return tuple(tuple(Fraction(int(arr[f * d + j, 0]), int(arr[f * d + j, 1])) for j in range(d)) for f in range(n))


def _spec(scope, columns, weight, unit):
    columns = list(columns)
    names = list(scope.names)
    if unit not in UNITS:
        raise ValueError(f"unit is one of {', '.join(UNITS)}")
    spec = _Spec()
    spec.ndims, spec.unit = len(columns), UNITS[unit]
    if len(columns) > MAX_DIMS:                         # the native refusal decides: UnsupportedError
        return spec
    for j, c in enumerate(columns):
        if c not in names:
            raise ValueError(f"{c!r} is not a column of the scope ({names})")
        spec.column[j] = names.index(c)
    if weight is None:
        spec.weight = -1
    else:
        if weight not in names:
            raise ValueError(f"{weight!r} is not a column of the scope ({names})")
        spec.weight = names.index(weight)
    return spec


def walsh_frequencies(k, unit="pi"):
    """the 2^k Walsh frequencies S in {0, 1}^k (pi units; S / 2 in turns), S = 0 .. 2^k - 1 (bit j = component j)"""
    if unit not in ("pi", "turn"):
        raise ValueError("the Walsh characters are exp(i pi S.x): unit 'pi' or 'turn'")
    one = 1 if unit == "pi" else Fraction(1, 2)
    return [tuple(one if (S >> j) & 1 else 0 for j in range(k)) for S in range(1 << k)]


def _build(scope, columns, frequencies, weight, unit, backend, budget):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    scope.require_fresh("spectral basis")
    spec = _spec(scope, columns, weight, unit)
    d = len(list(columns))
    arr, m = _qarray(frequencies, d) if d <= MAX_DIMS else (np.zeros((1, 2), np.int64), 1)
    api = _api()
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = api["build"](scope._handle, C.byref(spec), m, arr.ctypes.data, _raw(budget), C.byref(refusal),
                         C.byref(error))
        _check(error.value, refusal, "spectral basis")
        if not h:
            raise MemoryError("spectral basis: native allocation failed")
        label, device = "cpu", "native CPU"
    else:
        plan = api["plan_create"](scope._handle, C.byref(spec), m, arr.ctypes.data, _raw(budget),
                                  C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "spectral basis")
        if not plan:
            raise MemoryError("spectral basis: native allocation failed")
        try:
            pv = _PlanView()
            _check(api["plan_get_view"](plan, C.byref(pv)))
            view = scope.view()
            G, m = int(pv.ngroups), int(pv.nfreq)
            sums = np.zeros((max(G * m * 4, 1), 2), dtype=np.int64)
            rows = np.zeros(max(G, 1), dtype=np.uint64)
            valid = np.zeros(max(G, 1), dtype=np.uint64)
            wsum = np.zeros((max(G, 1), 2), dtype=np.int64)
            gpu = _gpu()
            refusal = _Refusal()
            gpu.run("tr_mgpu_fourier_sums", C.byref(pv), C.byref(view), sums.ctypes.data, rows.ctypes.data,
                    valid.ctypes.data, wsum.ctypes.data, _raw(budget), C.byref(refusal), refusal=refusal,
                    what="spectral basis on the device")
            info = gpu.device()
            label, device = ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]
            error, refusal = C.c_int32(), _Refusal()
            h = api["from_sums"](plan, scope._handle, sums.ctypes.data, rows.ctypes.data, valid.ctypes.data,
                                 wsum.ctypes.data, _raw(budget), C.byref(refusal), C.byref(error))
            _check(error.value, refusal, "spectral basis")
            if not h:
                raise MemoryError("spectral basis: native allocation failed")
        finally:
            api["plan_free"](plan)
    identity = dict(columns=tuple(columns), weight=weight, unit=unit, scope=scope)
    return Basis(h, scope, identity, label, device)


def basis(scope, columns, frequencies, weight=None, unit="radian", backend="cpu", budget=None):
    """The fixed spectral basis of a scope: per group F(omega) = sum over the counted rows of w exp(i c omega.x) for
    each declared frequency (columns: the d scope columns of x; weight: a column or None (1); unit: radian / pi /
    turn); backend "cuda" accumulates the sums on the device (Basis.backend: "cuda" or "cuda-emulated")."""
    return _build(scope, columns, frequencies, weight, unit, backend, budget)


def walsh_basis(scope, columns, weight=None, backend="cpu", budget=None):
    """the Boolean special case: all 2^k Walsh frequencies over k columns of 0 / 1 values (pi units), exact"""
    return _build(scope, columns, walsh_frequencies(len(list(columns))), weight, "pi", backend, budget)


def device_bytes(scope, columns, frequencies, weight=None, unit="radian"):
    """the bytes the CUDA build copies to the device (its budget need)"""
    spec = _spec(scope, columns, weight, unit)
    arr, m = _qarray(frequencies, len(list(columns)))
    api = _api()
    error, refusal = C.c_int32(), _Refusal()
    plan = api["plan_create"](scope._handle, C.byref(spec), m, arr.ctypes.data, None, C.byref(refusal),
                              C.byref(error))
    _check(error.value, refusal, "spectral basis")
    try:
        pv = _PlanView()
        _check(api["plan_get_view"](plan, C.byref(pv)))
        view = scope.view()
        return int(api["device_need"](C.byref(pv), C.byref(view)))
    finally:
        api["plan_free"](plan)


def _request(r, d):
    if not isinstance(r, dict):
        raise TypeError("a request is a dict with shift, phase, order, phase_order")
    unknown = set(r) - {"shift", "phase", "order", "phase_order"}
    if unknown:
        raise ValueError(f"unknown request keys {sorted(unknown)}")
    shift = r.get("shift")
    if shift is None:
        shift = (0.0,) * d
    elif d == 1 and not isinstance(shift, (tuple, list, np.ndarray)):
        shift = (shift,)
    shift = tuple(float(v) for v in shift)
    order = r.get("order")
    if order is None:
        order = (0,) * d
    elif d == 1 and isinstance(order, Integral):
        order = (int(order),)
    order = tuple(int(v) for v in order)
    if len(shift) != d or len(order) != d:
        raise ValueError(f"shift and order have {d} components")
    return shift, float(r.get("phase", 0.0)), order, int(r.get("phase_order", 0))


class Basis:
    """A fixed spectral basis state (native, immutable): certified Fourier features per group and frequency, exact
    bounds, phase actions and exact derivative transforms answered from it without rescanning."""

    def __init__(self, handle, scope, identity, backend, device):
        self._h, self.scope, self.identity, self.backend, self.device = handle, scope, dict(identity), backend, device
        i = self._info()
        self.ndims, self.unit = int(i.ndims), _UNIT_NAMES[int(i.unit)]
        self.columns = tuple(identity["columns"])
        self.nfreq = int(i.nfreq)
        self._frequencies = None

    @property
    def frequencies(self):
        """the exact canonical frequencies of the basis (tuples of d Fractions), in the coefficient order"""
        if self._frequencies is None:
            n, d = self.nfreq, self.ndims
            raw = np.zeros((max(n * d, 1), 2), dtype=np.int64)
            for f in range(n):
                _check(_api()["frequency"](self._open(), f, raw[f * d:].ctypes.data), what="frequency")
            self._frequencies = _vectors(raw, n, d)
        return self._frequencies

    def _open(self):
        if not self._h:
            raise ValueError("the basis is closed")
        return self._h

    def _info(self):
        out = _Info()
        _check(_api()["get_info"](self._open(), C.byref(out)), what="spectral basis")
        return out

    @property
    def info(self):
        i = self._info()
        return dict(dimensions=int(i.ndims), unit=_UNIT_NAMES[int(i.unit)], scale=int(i.scale),
                    weight_scale=int(i.wscale), exact=bool(i.exact), frequencies=int(i.nfreq), groups=int(i.ngroups),
                    rows=int(i.rows), selected=int(i.selected), order=[int(i.order[j]) for j in range(i.ndims)],
                    phase_order=int(i.phase_order), unit_power=int(i.unit_power), backend=self.backend,
                    device=self.device, contract=CONTRACT)

    def counts(self, group=0):
        """(selected rows, counted rows: every column read present) of a group"""
        r, v, w = C.c_uint64(), C.c_uint64(), _mc._I128()
        _check(_api()["group"](self._open(), int(group), C.byref(r), C.byref(v), C.byref(w)), what="group")
        return int(r.value), int(v.value)

    def weight(self, group=0):
        """the exact weight sum W of the counted rows (value units; the count without a weight column)"""
        r, v, w = C.c_uint64(), C.c_uint64(), _mc._I128()
        _check(_api()["group"](self._open(), int(group), C.byref(r), C.byref(v), C.byref(w)), what="group")
        return Fraction(_i128(w), 10 ** int(self._info().wscale))

    def coefficients(self, group=0):
        """the certified enclosures ((re_lo, re_hi), (im_lo, im_hi)) of every coefficient, in the basis order"""
        out = []
        for f in range(self.nfreq):
            z = _mc._CInterval()
            _check(_api()["coefficient"](self._open(), int(group), f, C.byref(z)), what="coefficient")
            out.append(_civ(z))
        return out

    def bounds(self, group=0):
        """the exact rational bounds [((re_lo, re_hi), (im_lo, im_hi))] of a basis built from data (no derivative)"""
        i = self._info()
        if any(i.order[j] for j in range(i.ndims)) or i.phase_order:
            raise ValueError("a derivative state: use exact()")
        den = Fraction(2) ** int(i.scale) * 10 ** int(i.wscale)
        out = []
        for f in range(int(i.nfreq)):
            raw = (_mc._I128 * 4)()
            _check(_api()["bounds"](self._open(), int(group), f, raw), what="bounds")
            v = [Fraction(_i128(raw[c])) / den for c in range(4)]
            out.append(((v[0], v[1]), (v[2], v[3])))
        return out

    def exact(self, group=0, frequency=0, budget=None):
        """the exact bounds of one coefficient as Fractions {"re": (lo, hi), "im": (lo, hi), "unit_power": p,
        "factor"}: the value is factor^p times them (factor pi or 2 pi; 1 in radians)"""
        handles = (C.c_void_p * 4)()
        p, refusal = C.c_int32(), _Refusal()
        _check(_api()["exact"](self._open(), int(group), int(frequency), handles, C.byref(p), _raw(budget),
                               C.byref(refusal)), refusal, "exact bounds")
        vals = [_mc.BigRational._wrap(h).to_fraction() for h in handles]     # owned: freed by the wrapper
        factor = {"radian": "1", "pi": "pi", "turn": "2 pi"}[self.unit]
        return dict(re=(vals[0], vals[1]), im=(vals[2], vals[3]), unit_power=int(p.value), factor=factor)

    def index(self, frequencies):
        """the positions of frequencies in the basis; DomainError (missing spectral support) for one absent"""
        arr, n = _qarray(list(frequencies), self.ndims)
        idx = np.zeros(max(n, 1), dtype=np.uint64)
        missing = C.c_uint64()
        rc = _api()["find"](self._open(), n, arr.ctypes.data, idx.ctypes.data, C.byref(missing))
        if rc == -6:
            from .math_domains import DomainError
            bad = _vectors(arr[int(missing.value) * self.ndims:], 1, self.ndims)[0]
            raise DomainError(f"missing spectral support: {bad} is not in the basis (a new frequency requests a new "
                              f"basis state; it is never treated as zero)")
        _check(rc, what="frequency lookup")
        return [int(i) for i in idx[:n]]

    def feature(self, frequency, group=0):
        """the enclosure of the feature at one frequency of the basis (DomainError when it is not in the basis)"""
        return self.coefficients(group)[self.index([frequency])[0]]

    def act(self, requests, group=0, frequencies=None, normalize=False, budget=None, arrays=False):
        """phase actions on the fixed basis: per request (dict: shift tau (d doubles), phase phi, order alpha (d
        ints), phase_order beta) the enclosures of i^(|alpha| + beta) (c omega)^alpha exp(i (c omega.tau + phi))
        F(omega) [/ W when normalize] for the selected frequencies (None: all; a frequency not in the basis:
        DomainError); arrays=True returns them as a float64 array (requests, frequencies, 4): re_lo, re_hi, im_lo,
        im_hi"""
        single = isinstance(requests, dict)
        reqs = [_request(r, self.ndims) for r in ([requests] if single else requests)]
        d, n = self.ndims, len(reqs)
        sel = None if frequencies is None else np.ascontiguousarray(self.index(frequencies), dtype=np.uint64)
        nsel = self.nfreq if sel is None else len(sel)
        tau = np.ascontiguousarray([v for r in reqs for v in r[0]] or [0.0], dtype=np.float64)
        phase = np.ascontiguousarray([r[1] for r in reqs] or [0.0], dtype=np.float64)
        order = np.ascontiguousarray([v for r in reqs for v in r[2]] or [0], dtype=np.int32)
        porder = np.ascontiguousarray([r[3] for r in reqs] or [0], dtype=np.int32)
        out = np.zeros((max(n * nsel, 1), 4), dtype=np.float64)          # tr_carrier_cinterval[]: re, im
        refusal = _Refusal()
        rc = _api()["act"](self._open(), int(group), n, tau.ctypes.data, phase.ctypes.data, order.ctypes.data,
                           porder.ctypes.data, nsel, sel.ctypes.data if sel is not None else None, int(bool(normalize)),
                           out.ctypes.data, _raw(budget), C.byref(refusal))
        if rc == -6 and normalize:
            from .math_domains import DomainError
            raise DomainError("characteristic function: the group's weight sum is 0 (no rows or zero weights)")
        _check(rc, refusal, "phase actions")
        out = out[:n * nsel].reshape(n, nsel, 4)
        if arrays:
            return out[0] if single else out
        res = [[((a, b), (c, d)) for a, b, c, d in row] for row in out.tolist()]
        return res[0] if single else res

    def shift(self, tau, group=0, frequencies=None, phase=0.0, budget=None):
        """the features of the translated data x + tau (rotated by phase): one action"""
        return self.act(dict(shift=tau, phase=phase), group, frequencies, False, budget)

    def characteristic(self, group=0, frequencies=None, shift=None, budget=None):
        """the empirical characteristic function F(omega) / W at the basis frequencies (of x + shift)"""
        return self.act(dict(shift=shift), group, frequencies, True, budget)

    def derivative(self, order, phase_order=0, budget=None):
        """the exact derivative state: translation order alpha (d ints; an int when d = 1) and phase order beta"""
        if self.ndims == 1 and isinstance(order, Integral):
            order = (int(order),)
        order = tuple(int(v) for v in order)
        if len(order) != self.ndims:
            raise ValueError(f"order has {self.ndims} components")
        alpha = (C.c_int32 * MAX_DIMS)(*order)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["derive"](self._open(), alpha, int(phase_order), _raw(budget), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "derivative")
        if not h:
            raise MemoryError("derivative: native allocation failed")
        return Basis(h, self.scope, self.identity, self.backend, self.device)

    def require(self, scope, columns, frequencies, weight=None, unit=None):
        """True when this state answers the request (every frequency in the basis, the same columns, weight, unit and
        scope); DomainError when a new basis state is needed"""
        unit = unit or self.unit
        spec = _spec(scope, columns, weight, unit)
        arr, n = _qarray(list(frequencies), len(list(columns)))
        _check(_api()["require"](self._open(), scope._handle, C.byref(spec), n, arr.ctypes.data),
               what="spectral basis (a new basis state is needed)")
        return True

    def walsh_support(self):
        """{"k", "present", "states": 2^k}: how many of the 2^k Walsh frequencies the basis holds"""
        k, n = C.c_int32(), C.c_uint64()
        _check(_api()["walsh_support"](self._open(), C.byref(k), C.byref(n)), what="Walsh support")
        return dict(k=int(k.value), present=int(n.value), states=1 << int(k.value))

    def walsh_inverse(self, group=0, budget=None):
        """the Boolean case: the weighted histogram v(x) = 2^-k sum_S W(S) (-1)^{|S and x|} of the k 0 / 1 columns,
        exactly (SPEC-002's inverse transform from the 2^k exact coefficients): {"values" (2^k Fractions, value
        units), "k", "normalization", "cost" (the full 2^k domain)}; DomainError when a Walsh frequency is missing
        (missing spectral support is not treated as zero), the data are not 0 / 1 or a coefficient is not exact"""
        from . import math_spec as ms
        sup = self.walsh_support()
        k = sup["k"]
        out = (ms._Dyadic * (1 << k))()
        cost, refusal = ms._Cost(), _Refusal()
        rc = _api()["walsh_inverse"](self._open(), int(group), out, C.byref(cost), _raw(budget), C.byref(refusal))
        if rc == -6:
            from .math_domains import DomainError
            if sup["present"] < sup["states"]:
                raise DomainError(f"missing spectral support: the basis holds {sup['present']} of the {sup['states']} "
                                  f"Walsh frequencies; the inverse needs all of them (missing coefficients are never "
                                  f"treated as zero)")
            raise DomainError("the Walsh inverse needs 0 / 1 data, a basis built from data and exact coefficients")
        _check(rc, refusal, "Walsh inverse")
        den = 10 ** int(self._info().wscale)
        vals = [Fraction(ms._i128(d.num_lo, d.num_hi), (1 << int(d.den_log2)) * den) for d in out]
        return dict(values=vals, k=k, normalization=ms.INVERSE, cost=ms._cost(cost))

    def close(self):
        if getattr(self, "_h", None):
            _api()["free"](self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._h:
            return "Basis(closed)"
        i = self.info
        return (f"Basis({i['dimensions']} dims, {i['frequencies']} frequencies, unit={i['unit']}, groups={i['groups']}, "
                f"order={i['order']}, backend={self.backend!r})")


class Retained:
    """Retained basis states keyed by their exact identity (decided natively by tr_fourier_require): a request whose
    frequencies are all in a retained state of the same columns, weight, unit and scope is answered by it (a hit);
    any new frequency builds a new basis state for the requested set (a construction)."""

    def __init__(self, capacity=16):
        if not 1 <= int(capacity) <= 1024:
            raise ValueError("capacity is 1..1024")
        self.capacity, self.entries = int(capacity), []
        self.stats = dict(hits=0, constructions=0, evictions=0)

    def basis(self, scope, columns, frequencies, weight=None, unit="radian", backend="cpu", budget=None):
        from .math_domains import DomainError
        frequencies = list(frequencies)
        if not scope.fresh():
            self.entries = [e for e in self.entries if e.scope is not scope]
            raise StaleStateError("spectral basis: the scope changed since it was built")
        for i, entry in enumerate(self.entries):
            try:
                entry.require(scope, columns, frequencies, weight, unit)
            except DomainError:
                continue
            self.stats["hits"] += 1
            self.entries.insert(0, self.entries.pop(i))
            return entry
        out = _build(scope, columns, frequencies, weight, unit, backend, budget)
        self.stats["constructions"] += 1
        self.entries.insert(0, out)
        if len(self.entries) > self.capacity:
            self.entries.pop().close()
            self.stats["evictions"] += 1
        return out
