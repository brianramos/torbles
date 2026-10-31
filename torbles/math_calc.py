# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""CALC-003: non-polynomial finite-separation calculus over the rows of a scope, natively on the CPU or with the
CUDA kernels (torbles/math_calc.c, math_calc_gpu.cu).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_calc as mc
    scope = mbk.Scope.from_arrays({"x": xs, "w": ws})                    # or Scope.from_table(db["t"], ["x", "w"])
    F = mc.phase(scope, [("x", Fraction(1, 2))], weight="w")            # F(phi) = sum w sin(x/2 + phi)
    F.coefficients()          # [(lo, hi)]: C_0 = B = sum w sin(x/2), C_1 = A = sum w cos(x/2), outward doubles
    F.exact_bounds()          # the same as exact Fractions [lo, hi]
    F(0.25); F.evaluate([0.1, 0.2, 0.3])                                # enclosures of F at the points
    F.derivative(0, 2); F.antiderivative(0); F.integral(0, 0.0, 1.0); F.definite(0, [0.0], [1.0])
    G = mc.exponential(scope, [("x", Fraction(-1, 4))], weight="w")     # G(theta) = E exp(theta)
    cache = mc.Retained()
    cache.phase(scope, [("x", Fraction(1, 2))], weight="w")             # built once, then retained
    cache.phase(scope, [("x", Fraction(1, 3))], weight="w")             # another frequency: a new construction
    cache.phase(scope, [("x", Fraction(1, 2))], weight="w", domega=[1]) # d/domega: UnsupportedError (not retained)

Every coefficient and value is an interval that contains the exact result (math_calc.h: the precision contract);
the parameter calculus is exact. A state's identity (family, columns, exact frequencies or rates, weight, the scope's
exact descriptor) decides natively whether it answers a request. Python marshals and formats only. Opt-in: `import
torbles` does not load this module.
"""
import ctypes as C
import math
import os
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from . import math_carriers as _mc
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check, StaleStateError

__all__ = ["phase", "exponential", "Family", "Retained", "sincos", "exp", "sincos_scaled", "exp_scaled",
           "FACTORS_MAX", "FRACTION_BITS", "FAMILIES"]

FACTORS_MAX, FRACTION_BITS = 16, 120
FAMILIES = {"phase": 1, "exp": 2}
P = C.POINTER


class _Spec(C.Structure):
    _native_ = "tr_calc_spec"
    _fields_ = [("family", C.c_int32), ("nfactors", C.c_int32), ("weight", C.c_int32), ("reserved", C.c_int32),
                ("column", C.c_int32 * FACTORS_MAX), ("domega", C.c_int32 * FACTORS_MAX)]


class _PlanView(C.Structure):
    _native_ = "tr_calc_plan_view"
    _fields_ = [("family", C.c_int32), ("nfactors", C.c_int32), ("weight", C.c_int32), ("scale", C.c_int32),
                ("nsubsets", C.c_uint64), ("ngroups", C.c_uint64), ("nmax", C.c_int64),
                ("ksign", C.c_int32 * FACTORS_MAX), ("column", C.c_int32 * FACTORS_MAX),
                ("k", (C.c_uint64 * 4) * FACTORS_MAX)]


class _Info(C.Structure):
    _native_ = "tr_calc_info"
    _fields_ = [(n, C.c_int32) for n in ("family", "nparams", "nfactors", "exact", "scale", "wscale", "weight",
                                         "reserved")] + [("ngroups", C.c_uint64), ("ncoef", C.c_uint64),
                                                         ("order", C.c_int32 * FACTORS_MAX),
                                                         ("param", C.c_int32 * FACTORS_MAX)]


class _Interval(C.Structure):
    _native_ = "tr_math_interval"
    _fields_ = [("lo", C.c_double), ("hi", C.c_double)]


class _I128(C.Structure):
    _native_ = "tr_math_i128"
    _fields_ = [("lo", C.c_int64), ("hi", C.c_int64)]


for _s, _n in ((_Spec, 144), (_PlanView, 680), (_Info, 176), (_Interval, 16), (_I128, 16)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, B, R, E = C.c_void_p, C.c_int32, C.c_uint64, P(_Budget), P(_Refusal), P(C.c_int32)
        dp = P(C.c_double)
        sig = {
            "tr_calc_plan_create": (vp, [vp, P(_Spec), vp, B, R, E]),
            "tr_calc_plan_get_view": (i32, [vp, P(_PlanView)]),
            "tr_calc_sums": (i32, [P(_PlanView), vp, vp, vp, vp]),
            "tr_calc_from_sums": (vp, [vp, vp, vp, vp, vp, B, R, E]),
            "tr_calc_build": (vp, [vp, P(_Spec), vp, B, R, E]),
            "tr_calc_plan_free": (None, [vp]),
            "tr_calc_get_info": (i32, [vp, P(_Info)]),
            "tr_calc_group": (i32, [vp, u64, P(u64), P(u64)]),
            "tr_calc_coefficient": (i32, [vp, u64, u64, P(_Interval)]),
            "tr_calc_coefficient_exact": (i32, [vp, u64, u64, P(_I128), P(_I128)]),
            "tr_calc_derive": (vp, [vp, i32, i32, B, R, E]),
            "tr_calc_integrate": (vp, [vp, i32, C.c_double, C.c_double, B, R, E]),
            "tr_calc_evaluate": (i32, [vp, u64, u64, dp, vp, B, R]),
            "tr_calc_definite": (i32, [vp, u64, i32, u64, dp, dp, dp, vp, B, R]),
            "tr_calc_require": (i32, [vp, vp, P(_Spec), vp]),
            "tr_calc_free": (None, [vp]),
            "tr_calc_sincos": (i32, [C.c_double, P(_Interval), P(_Interval)]),
            "tr_calc_exp": (i32, [C.c_double, P(_Interval)]),
            "tr_calc_sincos_scaled": (i32, [vp, C.c_int64, i32, P(_Interval), P(_Interval)]),
            "tr_calc_exp_scaled": (i32, [vp, C.c_int64, i32, P(_Interval)]),
            "tr_calc_device_need": (u64, [P(_PlanView), vp]),
        }
        native = {}
        for name, (restype, argtypes) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            native[name[8:]] = fn
        _native = native
    return _native


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_calc_sums", C.c_int, [P(_PlanView), C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                                     P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _raw(budget):
    return C.byref((budget or Budget()).raw)


def _exact(value, what):
    """an exact rational: int, Fraction, or a float's exact (dyadic) value"""
    if isinstance(value, bool):
        raise TypeError(f"{what}: bool is not a number")
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{what}: a finite value is required")
        return Fraction(value)                                    # exact: the dyadic value of the double
    raise TypeError(f"{what}: an int, a Fraction or a float is required")


def _iv(raw):
    return (raw.lo, raw.hi)


def _i128(x):
    return int.from_bytes(bytes(x), "little", signed=True)


def _spec(scope, family, factors, weight, domega):
    factors = list(factors)
    if not factors:
        raise ValueError("give at least one factor: (column, frequency) pairs")
    spec = _Spec()
    spec.family = FAMILIES[family] if family in FAMILIES else 99
    spec.nfactors = len(factors)
    names = list(scope.names)
    keep, handles = [], (C.c_void_p * max(len(factors), 1))()
    if len(factors) > FACTORS_MAX:                               # the native refusal decides: UnsupportedError
        spec.nfactors = len(factors)
        return spec, keep, handles, names
    for i, item in enumerate(factors):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("a factor is a (column, frequency or rate) pair")
        col, w = item
        if col not in names:
            raise ValueError(f"{col!r} is not a column of the scope ({names})")
        spec.column[i] = names.index(col)
        q = _mc.BigRational(_exact(w, f"the frequency of {col!r}"))
        keep.append(q)
        handles[i] = q._h
        if domega is not None:
            spec.domega[i] = int(domega[i]) if i < len(domega) else 0
    if weight is None:
        spec.weight = -1
    else:
        if weight not in names:
            raise ValueError(f"{weight!r} is not a column of the scope ({names})")
        spec.weight = names.index(weight)
    return spec, keep, handles, names


def _build(scope, family, factors, weight, backend, budget, domega=None):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    scope.require_fresh("separable family")
    spec, keep, handles, names = _spec(scope, family, factors, weight, domega)
    api = _api()
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = api["build"](scope._handle, C.byref(spec), handles, _raw(budget), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "separable family")
        if not h:
            raise MemoryError("separable family: native allocation failed")
        label, device = "cpu", "native CPU"
    else:
        plan = api["plan_create"](scope._handle, C.byref(spec), handles, _raw(budget), C.byref(refusal),
                                  C.byref(error))
        _check(error.value, refusal, "separable family")
        if not plan:
            raise MemoryError("separable family: native allocation failed")
        try:
            pv = _PlanView()
            _check(api["plan_get_view"](plan, C.byref(pv)))
            view = scope.view()
            G, m = int(pv.ngroups), int(pv.nsubsets)
            sums = (_I128 * max(2 * G * m, 1))()
            rows = (C.c_uint64 * max(G, 1))()
            valid = (C.c_uint64 * max(G, 1))()
            gpu = _gpu()
            refusal = _Refusal()
            gpu.run("tr_mgpu_calc_sums", C.byref(pv), C.byref(view), sums, rows, valid, _raw(budget),
                    C.byref(refusal), refusal=refusal, what="separable family on the device")
            info = gpu.device()
            label, device = ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]
            error, refusal = C.c_int32(), _Refusal()
            h = api["from_sums"](plan, scope._handle, sums, rows, valid, _raw(budget), C.byref(refusal),
                                 C.byref(error))
            _check(error.value, refusal, "separable family")
            if not h:
                raise MemoryError("separable family: native allocation failed")
        finally:
            api["plan_free"](plan)
    identity = dict(family=family, factors=tuple((c, _exact(w, "frequency")) for c, w in factors),
                    weight=weight, scope=repr(scope))
    return Family(h, scope, [c for c, _ in factors], identity, label, device)


def phase(scope, factors, weight=None, backend="cpu", budget=None, domega=None):
    """F(phi_1..phi_k) = sum over the counted rows of w prod_i sin(omega_i x_i + phi_i): factors = [(column,
    omega_i)] (omega exact: int, Fraction or a float's exact value), weight a column or None (1)"""
    return _build(scope, "phase", factors, weight, backend, budget, domega)


def exponential(scope, factors, weight=None, backend="cpu", budget=None, domega=None):
    """G(theta_1..theta_k) = sum over the counted rows of w exp(sum_i theta_i + a_i x_i) = E exp(sum theta):
    factors = [(column, a_i)]"""
    return _build(scope, "exp", factors, weight, backend, budget, domega)


class Family:
    """A retained separable family (native, immutable): its coefficients, exact transforms and enclosed values."""

    def __init__(self, handle, scope, params, identity, backend, device):
        self._h, self.scope, self.params = handle, scope, tuple(params)
        self.identity, self.backend, self.device = dict(identity), backend, device

    def _open(self):
        if not self._h:
            raise ValueError("the family is closed")
        return self._h

    def _info(self):
        out = _Info()
        _check(_api()["get_info"](self._open(), C.byref(out)), what="separable family")
        return out

    @property
    def info(self):
        i = self._info()
        return dict(family={1: "phase", 2: "exp"}[i.family], parameters=int(i.nparams), factors=int(i.nfactors),
                    exact=bool(i.exact), scale=int(i.scale), weight_scale=int(i.wscale), groups=int(i.ngroups),
                    coefficients=int(i.ncoef), order=[int(i.order[j]) for j in range(i.nparams)],
                    param_factor=[int(i.param[j]) for j in range(i.nparams)], backend=self.backend,
                    device=self.device)

    @property
    def nparams(self):
        return int(self._info().nparams)

    def counts(self, group=0):
        """(selected rows, counted rows: every column the family reads present) of a group"""
        r, v = C.c_uint64(), C.c_uint64()
        _check(_api()["group"](self._open(), int(group), C.byref(r), C.byref(v)), what="group")
        return int(r.value), int(v.value)

    def coefficients(self, group=0):
        """the outward-rounded double enclosures of the coefficients (PHASE: C_S, bit i of S = sin of phi_i)"""
        n = int(self._info().ncoef)
        out = []
        for S in range(n):
            iv = _Interval()
            _check(_api()["coefficient"](self._open(), int(group), S, C.byref(iv)), what="coefficient")
            out.append(_iv(iv))
        return out

    def exact_bounds(self, group=0):
        """the exact rational enclosures [lo, hi] of the coefficients (states that are exact transforms of the
        built sums; a definite integral's state has only interval coefficients: UnsupportedError)"""
        i = self._info()
        den = Fraction(2) ** int(i.scale) * 10 ** int(i.wscale)
        out = []
        for S in range(int(i.ncoef)):
            lo, hi = _I128(), _I128()
            _check(_api()["coefficient_exact"](self._open(), int(group), S, C.byref(lo), C.byref(hi)),
                   what="exact coefficient")
            out.append((Fraction(_i128(lo)) / den, Fraction(_i128(hi)) / den))
        return out

    def _points(self, points, k):
        if k == 0:
            n = len(points) if points is not None else 1
            return None, max(n, 1)
        a = np.asarray(points, dtype=np.float64)
        if a.ndim == 1 and k == 1:
            a = a.reshape(-1, 1)
        if a.ndim != 2 or a.shape[1] != k:
            raise ValueError(f"points: one value per parameter ({k})")
        a = np.ascontiguousarray(a)
        return a, a.shape[0]

    def evaluate(self, points, group=0, budget=None):
        """enclosures [lo, hi] of the family at the points (n x k doubles; a flat list when k == 1)"""
        k = self.nparams
        a, n = self._points(points, k)
        out = (_Interval * max(n, 1))()
        refusal = _Refusal()
        ptr = a.ctypes.data_as(P(C.c_double)) if a is not None else None
        _check(_api()["evaluate"](self._open(), int(group), n, ptr, out, _raw(budget), C.byref(refusal)), refusal,
               "evaluation")
        return [_iv(out[j]) for j in range(n)]

    def __call__(self, *point, group=0):
        return self.evaluate([list(point)] if point else None, group)[0]

    def derivative(self, param=0, order=1, budget=None):
        """the exact derivative of the given order in parameter param (order < 0: antiderivatives); param "omega"
        (a derivative in a frequency or a rate) is outside the retained basis: UnsupportedError"""
        if isinstance(param, str):
            return self._outside(param, order)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["derive"](self._open(), int(param), int(order), _raw(budget), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "derivative")
        return Family(h, self.scope, self.params, self.identity, self.backend, self.device)

    def antiderivative(self, param=0, order=1, budget=None):
        """the antiderivative in the retained span (no constant): an exact transform"""
        return self.derivative(param, -int(order), budget)

    def integral(self, param, a, b, budget=None):
        """the definite integral over parameter param from a to b: a family of the other parameters"""
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["integrate"](self._open(), int(param), float(a), float(b), _raw(budget), C.byref(refusal),
                                C.byref(error))
        _check(error.value, refusal, "integral")
        params = [p for j, p in enumerate(self.params) if j != int(param)]
        return Family(h, self.scope, params, self.identity, self.backend, self.device)

    def definite(self, param, a, b, rest=None, group=0, budget=None):
        """enclosures of the definite integrals over param from a[j] to b[j] (the other parameters at rest[j])"""
        k = self.nparams
        av = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
        bv = np.ascontiguousarray(np.asarray(b, dtype=np.float64).ravel())
        if av.shape != bv.shape:
            raise ValueError("one lower and one upper bound per integral")
        n = av.size
        rv = None
        if k > 1:
            rv = np.ascontiguousarray(np.asarray(rest, dtype=np.float64).reshape(n, k - 1))
        out = (_Interval * max(n, 1))()
        refusal = _Refusal()
        _check(_api()["definite"](self._open(), int(group), int(param), n, av.ctypes.data_as(P(C.c_double)),
                                  bv.ctypes.data_as(P(C.c_double)),
                                  rv.ctypes.data_as(P(C.c_double)) if rv is not None else None, out, _raw(budget),
                                  C.byref(refusal)), refusal, "definite integrals")
        return [_iv(out[j]) for j in range(n)]

    def require(self, scope, factors, weight=None, family=None, domega=None):
        """True when this state answers the request; DomainError when another construction is needed (another
        frequency, rate, column, weight, family or scope), UnsupportedError outside the retained basis"""
        family = family or self.identity["family"]
        spec, keep, handles, _ = _spec(scope, family, factors, weight, domega)
        _check(_api()["require"](self._open(), scope._handle, C.byref(spec), handles), what="retained family")
        return True

    def _outside(self, param, order):
        factors = list(self.identity["factors"])
        domega = [int(order)] + [0] * (len(factors) - 1)
        self.require(self.scope, factors, self.identity["weight"], domega=domega)
        raise AssertionError("unreachable: a frequency derivative is never retained")

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
            return "Family(closed)"
        i = self.info
        return f"Family({i['family']}, parameters={i['parameters']}, coefficients={i['coefficients']}, " \
               f"order={i['order']}, backend={self.backend!r})"


class Retained:
    """Retained families keyed by their exact identity (decided natively by tr_calc_require): a request with the
    same identity returns the retained state; another frequency, rate, column, weight or scope builds a new one (a
    new construction, counted); a request outside the retained basis raises UnsupportedError."""

    def __init__(self, capacity=16):
        if not 1 <= int(capacity) <= 1024:
            raise ValueError("capacity is 1..1024")
        self.capacity, self.entries = int(capacity), []
        self.stats = dict(hits=0, constructions=0, refusals=0, evictions=0)

    def _get(self, family, scope, factors, weight, backend, budget, domega):
        from .math_domains import DomainError
        from .math_errors import UnsupportedError
        if not scope.fresh():
            self.entries = [e for e in self.entries if e.scope is not scope]
            raise StaleStateError("separable family: the scope changed since it was built")
        for i, entry in enumerate(self.entries):
            try:
                entry.require(scope, factors, weight, family, domega)
            except DomainError:
                continue
            except UnsupportedError:
                self.stats["refusals"] += 1
                raise
            self.stats["hits"] += 1
            self.entries.insert(0, self.entries.pop(i))
            return entry
        try:
            out = _build(scope, family, factors, weight, backend, budget, domega)
        except UnsupportedError:
            self.stats["refusals"] += 1
            raise
        self.stats["constructions"] += 1
        self.entries.insert(0, out)
        if len(self.entries) > self.capacity:
            self.entries.pop().close()
            self.stats["evictions"] += 1
        return out

    def phase(self, scope, factors, weight=None, backend="cpu", budget=None, domega=None):
        return self._get("phase", scope, factors, weight, backend, budget, domega)

    def exponential(self, scope, factors, weight=None, backend="cpu", budget=None, domega=None):
        return self._get("exp", scope, factors, weight, backend, budget, domega)


# ---------------------------------------------------------------------------------------------------- certified
def sincos(x):
    """certified enclosures (sin x, cos x) of a double x (|x| < 2^62)"""
    s, c = _Interval(), _Interval()
    _check(_api()["sincos"](float(x), C.byref(s), C.byref(c)), what="sin / cos")
    return _iv(s), _iv(c)


def exp(x):
    """a certified enclosure of exp(x) for a double x"""
    e = _Interval()
    _check(_api()["exp"](float(x), C.byref(e)), what="exp")
    return _iv(e)


def sincos_scaled(omega, X, scale=0):
    """the per-row factor enclosures of a PHASE build: (sin, cos) of omega X / 10^scale (omega exact, X int64)"""
    q = _mc.BigRational(_exact(omega, "frequency"))
    s, c = _Interval(), _Interval()
    _check(_api()["sincos_scaled"](q._h, int(X), int(scale), C.byref(s), C.byref(c)), what="sin / cos")
    return _iv(s), _iv(c)


def exp_scaled(a, X, scale=0):
    """the per-row factor enclosure of an EXP build: exp(a X / 10^scale) (a exact, X int64)"""
    q = _mc.BigRational(_exact(a, "rate"))
    e = _Interval()
    _check(_api()["exp_scaled"](q._h, int(X), int(scale), C.byref(e)), what="exp")
    return _iv(e)


def device_bytes(scope, family, factors, weight=None):
    """the bytes the CUDA build copies to the device (its budget need)"""
    spec, keep, handles, _ = _spec(scope, family, factors, weight, None)
    api = _api()
    error, refusal = C.c_int32(), _Refusal()
    plan = api["plan_create"](scope._handle, C.byref(spec), handles, None, C.byref(refusal), C.byref(error))
    _check(error.value, refusal, "separable family")
    try:
        pv = _PlanView()
        _check(api["plan_get_view"](plan, C.byref(pv)))
        view = scope.view()
        return int(api["device_need"](C.byref(pv), C.byref(view)))
    finally:
        api["plan_free"](plan)
