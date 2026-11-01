# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROB-001: small-support partition and likelihood functions: declared finite-support probability models evaluated
from retained statistics (the masses of a DIST-001 histogram aligned to a declared support), exact rationals for
categorical models and certified enclosures for exponential families, for many parameter requests at once, natively on
the CPU or with the CUDA kernels (torbles/math_prob.c, math_prob_gpu.cu).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_dist as md, math_prob as mp, math_conv as mc
    S = mp.support(range(8))                                  # declared: points without rows keep mass 0
    S2 = mp.support([(x, y) for x in range(4) for y in (0, 1, None)])   # joint points; None = the NULL category
    st = mp.stats(md.histogram(scope, ["x"]), S)              # DomainError if an observed value is outside S
    M = mp.categorical(S)                                     # weights per request (one-hot: natural parameters)
    E = mp.exponential(S, features=lambda x: (x, x * x), base=None)     # p ~ h(x) exp(theta . T(x))
    ev = E.evaluate(st, [(Fraction(1, 3), -1), (0, Fraction(-1, 8))], grad=True, hess=True, backend="cuda")
    ev[0]   # {"Z", "Z_exact", "logZ", "logZ_bounds", "loglik", "neg_inf", "grad", "hess", "width", ...}
    ev.exact(0)                                               # exact rational bounds of every entry (FND-009)
    M.evaluate(st, [[1, 1, 2, 0, 0, 0, 0, 4]], mode="weights", grad=True)    # exact Z, gradient, Hessian
    joint = mp.product(st_x, st_y, declaration=mc.independent(scope))      # the CROSS JOIN within groups

Every transcendental value is an enclosure [lo, hi] of the exact real value (outward-rounded doubles) with the exact
bounds it comes from; categorical normalizers, gradients and Hessians are exact. A log-likelihood is -inf (never a
large finite number) when a point of zero model probability has mass. Python only marshals and formats; the values are
computed natively. Refusals: AlgebraBudgetError (before allocation), DomainError (an observed value outside the
declared support, another support, Z = 0, negative weights, an undeclared independence), OverflowError,
UnsupportedError, StaleStateError (a table scope whose table changed). Opt-in: `import torbles` does not load this
module.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_dist import Histogram, _i128, _i256
from .math_domains import DomainError
from .math_errors import RESOURCES, UNKNOWN, budget_error, check as _check

__all__ = ["Support", "Stats", "Model", "Evaluation", "support", "stats", "product", "categorical", "exponential",
           "need", "ln_bounds", "KINDS", "MODES", "MASSES", "ANSWER_DTYPE", "F"]

KINDS = {"categorical": 1, "exponential": 2}
MODES = {"weights": 0, "probabilities": 1}
MASSES = {"count": 1, "weight": 2}
WANT_GRAD, WANT_HESS = 1, 2
NEG_INF, EMPTY, EXP_OVERFLOW, INCONSISTENT = 1, 2, 4, 8
NEED_CPU, NEED_DEVICE = 0, 1
F = 120                              # the fixed point of the log bounds: value = num / 2^120
MAX_DIMS, MAX_FEATURES = 3, 16
I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER
i32, u32, i64, u64, vp, dbl = C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_void_p, C.c_double


class _SupportView(C.Structure):
    _native_ = "tr_prob_support_view"
    _fields_ = [("k", i32), ("reserved", i32), ("d", u64), ("scale", i32 * 3), ("reserved2", i32),
                ("value", vp), ("nulls", vp), ("digest", u64 * 2)]


class _ModelView(C.Structure):
    _native_ = "tr_prob_model_view"
    _fields_ = [("kind", i32), ("m", i32), ("d", u64), ("scale", i32), ("reserved", i32), ("bits", i32 * 16),
                ("feat", vp), ("fden", vp), ("base", vp), ("hbits", vp), ("lnh", vp), ("hden", i64)]


class _StatsView(C.Structure):
    _native_ = "tr_prob_stats_view"
    _fields_ = [("ngroups", u64), ("d", u64), ("k", i32), ("mass", i32), ("exclude", u32), ("declared", i32),
                ("counts", vp), ("total", vp), ("rows", vp), ("dropped", vp)]


class _StatsInfo(C.Structure):
    _native_ = "tr_prob_stats_info"
    _fields_ = [(n, u64) for n in ("ngroups", "present", "d", "total", "bytes")] + [
        ("digest", u64 * 2), ("support_digest", u64 * 2)] + [
        (n, i32) for n in ("k", "mass", "exclude", "scoped", "declared", "product", "source_kind", "keyed")]


class _Outside(C.Structure):
    _native_ = "tr_prob_outside"
    _fields_ = [("group", u64), ("mass", u64), ("value", i64 * 3), ("nulls", C.c_uint8), ("reserved", C.c_uint8 * 7)]


class _PlanView(C.Structure):
    _native_ = "tr_prob_plan_view"
    _fields_ = [("kind", i32), ("what", i32), ("m", i32), ("mm", i32), ("n", u64), ("d", u64), ("ngroups", u64),
                ("nsums", u64), ("scale", i32), ("reserved", i32), ("bits", i32 * 16)] + [
        (n, vp) for n in ("feat", "base", "hbits", "counts", "group", "status", "exp", "weight", "lscale")] + [
        ("reserved2", u64 * 2)]


class _Answer(C.Structure):
    _native_ = "tr_prob_answer"
    _fields_ = [("status", i32), ("flags", i32), ("group", u64), ("mass", u64), ("rows", u64), ("zexp", i64),
                ("zden", i64), ("zlo", i64 * 2), ("zhi", i64 * 2), ("logz_lo", u64 * 4), ("logz_hi", u64 * 4),
                ("ll_lo", u64 * 4), ("ll_hi", u64 * 4), ("z", dbl * 2), ("logz", dbl * 2), ("loglik", dbl * 2),
                ("width", dbl * 5)]


for _s, _n in ((_SupportView, 64), (_ModelView, 136), (_StatsView, 64), (_StatsInfo, 104), (_Outside, 48),
               (_PlanView, 208), (_Answer, 296)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))
ANSWER_DTYPE = np.dtype([("status", "<i4"), ("flags", "<i4"), ("group", "<u8"), ("mass", "<u8"), ("rows", "<u8"),
                         ("zexp", "<i8"), ("zden", "<i8"), ("zlo", "<i8", (2,)), ("zhi", "<i8", (2,)),
                         ("logz_lo", "<u8", (4,)), ("logz_hi", "<u8", (4,)), ("ll_lo", "<u8", (4,)),
                         ("ll_hi", "<u8", (4,)), ("z", "<f8", (2,)), ("logz", "<f8", (2,)), ("loglik", "<f8", (2,)),
                         ("width", "<f8", (5,))])
assert ANSWER_DTYPE.itemsize == 296

_native = None


def _api():
    global _native
    if _native is None:
        from .math_conv import _Check
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_prob_support_create": (vp, [i32, u64, vp, vp, vp, vp, B, R, e]),
            "tr_prob_support_product": (vp, [vp, vp, B, R, e]),
            "tr_prob_support_get_view": (i32, [vp, P(_SupportView)]),
            "tr_prob_support_find": (i32, [vp, vp, C.c_uint8, P(u64)]),
            "tr_prob_support_equal": (i32, [vp, vp]),
            "tr_prob_support_free": (None, [vp]),
            "tr_prob_model_create": (vp, [vp, i32, i32, vp, vp, vp, vp, B, R, e]),
            "tr_prob_model_get_view": (i32, [vp, P(_ModelView)]),
            "tr_prob_model_free": (None, [vp]),
            "tr_prob_stats_build": (vp, [vp, vp, vp, i32, u32, B, R, P(_Outside), e]),
            "tr_prob_stats_need": (i32, [vp, vp, vp, i32, u32, P(_Need)]),
            "tr_prob_stats_product": (vp, [vp, vp, vp, vp, vp, B, R, P(_Check), e]),
            "tr_prob_stats_get_view": (i32, [vp, P(_StatsView)]),
            "tr_prob_stats_get_info": (i32, [vp, P(_StatsInfo)]),
            "tr_prob_stats_require": (i32, [vp, vp]),
            "tr_prob_stats_equal": (i32, [vp, vp]),
            "tr_prob_stats_free": (None, [vp]),
            "tr_prob_plan_create": (vp, [vp, vp, u64, vp, vp, vp, i32, i32, B, R, e]),
            "tr_prob_plan_need": (i32, [vp, vp, u64, i32, i32, P(_Need)]),
            "tr_prob_plan_get_view": (i32, [vp, P(_PlanView)]),
            "tr_prob_plan_status": (i32, [vp, u64]),
            "tr_prob_sums": (i32, [P(_PlanView), vp, vp, vp]),
            "tr_prob_finish": (i32, [vp, vp, vp, vp, vp, vp, vp, B, R]),
            "tr_prob_exact": (i32, [vp, u64, vp, vp, vp, u64, B, R, P(vp), P(vp), P(vp), P(vp), P(i32), P(i32)]),
            "tr_prob_plan_free": (None, [vp]),
            "tr_prob_device_need": (u64, [P(_PlanView)]),
            "tr_prob_ln_bounds": (i32, [u64, u64, P(i64)]),
            "tr_math_budget_check": (i32, [B, P(_Need), R]),
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
    math_gpu.bind("tr_mgpu_prob_sums", C.c_int, [P(_PlanView), P(_Budget), vp, vp, vp, P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _precheck(budget, raw_need, what):
    """the native budget check; a size past 64 bits (UNKNOWN) is refused even by an unlimited budget"""
    refusal = _Refusal()
    raw = _budget(budget)
    _check(_call("tr_math_budget_check", C.byref(raw), C.byref(raw_need), C.byref(refusal)), refusal, what)
    for i in range(len(RESOURCES)):
        if raw_need.need[i] == UNKNOWN:
            raise budget_error(i, UNKNOWN, int(raw.limit[i]), what)


def _rational(value, what):
    """an exact rational (num, den) with int64 parts; a float is its exact dyadic value"""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be a number, not bool")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{what} must be finite")
        value = Fraction(float(value))
    elif isinstance(value, Decimal):
        value = Fraction(value)
    elif isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, Integral):
        num, den = int(value), 1
    elif isinstance(value, Rational):
        num, den = int(value.numerator), int(value.denominator)
    else:
        raise TypeError(f"{what} must be exact (int, Fraction, Decimal) or a float (its exact value)")
    if not (I64[0] <= num <= I64[1] and 1 <= den <= I64[1]):
        raise OverflowError(f"{what}: numerator and denominator must fit int64")
    return num, den


def _logical(value, scale, what):
    """a support coordinate: a logical int, or an exact decimal value times 10^scale (must be integral); None = NULL"""
    if value is None:
        return 0, 1
    if isinstance(value, bool) or isinstance(value, (float, np.floating)):
        raise TypeError(f"{what} must be an exact integer, Decimal or Fraction (logical units unless scaled)")
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, Integral):
        v = int(value)                                   # a logical integer, as DIST-001 reports cell keys
    else:
        q = Fraction(value) * (10 ** scale)
        if q.denominator != 1:
            raise ValueError(f"{what} = {value} is not a multiple of 10^-{scale}")
        v = int(q)
    if not I64[0] <= v <= I64[1]:
        raise OverflowError(f"{what} does not fit int64")
    return v, 0


# ---------------------------------------------------------------------- supports
class Support:
    """A declared finite support: points of k coordinates (logical int64 values or None for the NULL category) in
    DIST-001 cell order, with a decimal scale per coordinate. Immutable, native."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_prob.support(points, scales=...)")

    @classmethod
    def _wrap(cls, handle):
        self = object.__new__(cls)
        self._handle = handle
        v = _SupportView()
        _check(_call("tr_prob_support_get_view", handle, C.byref(v)), what="support")
        self.k, self.size = int(v.k), int(v.d)
        self.scales = tuple(int(v.scale[c]) for c in range(self.k))
        vals = np.ctypeslib.as_array((C.c_int64 * (self.size * self.k)).from_address(v.value)).reshape(self.size, self.k)
        nul = np.ctypeslib.as_array((C.c_uint8 * self.size).from_address(v.nulls))
        pts = []
        for i in range(self.size):
            t = tuple(None if (int(nul[i]) >> c) & 1 else int(vals[i, c]) for c in range(self.k))
            pts.append(t[0] if self.k == 1 else t)
        self.points = tuple(pts)
        self.digest = f"{int(v.digest[0]):016x}{int(v.digest[1]):016x}"
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the support is closed")
        return self._handle

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.points)

    def _coords(self, point):
        t = (point,) if self.k == 1 and not isinstance(point, tuple) else tuple(point)
        if len(t) != self.k:
            raise ValueError(f"a point has {self.k} coordinates")
        vals, nulls = (C.c_int64 * 3)(), 0
        for c, x in enumerate(t):
            v, null = _logical(x, self.scales[c], "a point coordinate")
            vals[c] = v
            nulls |= null << c
        return vals, nulls

    def index(self, point):
        """the position of a point in the support's order (ValueError when it is not declared)"""
        vals, nulls = self._coords(point)
        at = C.c_uint64()
        rc = _check(_call("tr_prob_support_find", self._open(), vals, nulls, C.byref(at)), what="support point")
        if rc != 1:
            raise ValueError(f"{point!r} is not a point of the declared support")
        return int(at.value)

    def __contains__(self, point):
        try:
            self.index(point)
            return True
        except (ValueError, TypeError):
            return False

    def equal(self, other):
        return bool(_check(_call("tr_prob_support_equal", self._open(), other._open()), what="support comparison"))

    def __eq__(self, other):
        return isinstance(other, Support) and self.equal(other)

    def __hash__(self):
        return hash(self.digest)

    def product(self, other, *, budget=None):
        """the product support self x other (points (a, b), self's coordinates first; k <= 3)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prob_support_product", self._open(), other._open(), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Support._wrap(_made(h, error, refusal, "product support"))

    def _aligned(self, values, what, default=None):
        """values given per point: a sequence in the support's order, a dict {point: value}, or a callable"""
        if callable(values):
            return [values(p) for p in self.points]
        if isinstance(values, dict):
            out = [default] * self.size
            for p, v in values.items():
                out[self.index(p)] = v
            if any(v is None for v in out):
                missing = [self.points[i] for i, v in enumerate(out) if v is None][:3]
                raise ValueError(f"{what}: no value for the support points {missing}")
            return out
        out = list(values)
        if len(out) != self.size:
            raise ValueError(f"{what}: one value per support point ({self.size}), in the support's order")
        return out

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_prob_support_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        shown = ", ".join(map(repr, self.points[:6])) + (", ..." if self.size > 6 else "")
        return f"Support(k={self.k}, {self.size} points: {shown})"


def support(points, *, scales=None, budget=None):
    """the declared support of a list of points: scalars (one coordinate) or tuples of k <= 3 coordinates; None is the
    NULL category; values are logical integers (Decimal / Fraction values are scaled by 10^scale and must be
    integral). Any order; duplicates are refused."""
    pts = list(points)
    if not pts:
        raise ValueError("a support has at least one point")
    k = len(pts[0]) if isinstance(pts[0], tuple) else 1
    if not 1 <= k <= MAX_DIMS:
        raise ValueError(f"a support point has 1..{MAX_DIMS} coordinates")
    scales = tuple(scales) if scales is not None else (0,) * k
    if len(scales) != k:
        raise ValueError("one scale per coordinate")
    d = len(pts)
    vals = np.zeros((d, k), dtype=np.int64)
    nulls = np.zeros(d, dtype=np.uint8)
    for i, p in enumerate(pts):
        t = p if isinstance(p, tuple) else (p,)
        if len(t) != k:
            raise ValueError("every point has the same number of coordinates")
        for c, x in enumerate(t):
            if x is None:
                nulls[i] |= 1 << c
                continue
            if isinstance(x, (Integral, np.integer)) and not isinstance(x, bool):
                v = int(x)
                if not I64[0] <= v <= I64[1]:
                    raise OverflowError("a support value does not fit int64")
                vals[i, c] = v
            else:
                vals[i, c] = _logical(x, scales[c], "a support value")[0]
    sc = (i32 * 3)(*(list(scales) + [0] * (3 - k)))
    perm = np.zeros(d, dtype=np.uint64)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_prob_support_create", k, d, vals.ctypes.data, nulls.ctypes.data, sc, perm.ctypes.data,
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    if error.value == -1 and h is None:
        raise ValueError("support: invalid points (a duplicate point, a scale outside 0..18)")
    return Support._wrap(_made(h, error, refusal, "support"))


# ---------------------------------------------------------------------- retained statistics
def _exclude_mask(names, exclude):
    if exclude in (None, (), []):
        return 0
    if exclude == "all":
        return (1 << len(names)) - 1
    exclude = [exclude] if isinstance(exclude, str) else list(exclude)
    mask = 0
    for n in exclude:
        if n not in names:
            raise ValueError(f"{n!r} is not a coordinate of the histogram ({', '.join(names)})")
        mask |= 1 << names.index(n)
    return mask


class Stats:
    """The retained statistics of a histogram over a declared support: per group the mass of every support point (0
    for points without rows), N, the selected rows and the mass dropped by a declared NULL exclusion. Immutable."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_prob.stats(histogram, support) or math_prob.product(a, b)")

    @classmethod
    def _wrap(cls, handle, sup, scope, names, mass):
        self = object.__new__(cls)
        self._handle, self.support, self.scope, self.names, self.mass = handle, sup, scope, tuple(names), mass
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the statistics are closed")
        return self._handle

    def require_fresh(self, what="statistics"):
        for s in (self.scope if isinstance(self.scope, tuple) else (self.scope,)):
            if s is not None and s.source == "table":
                s.require_fresh(what)

    def _view(self):
        v = _StatsView()
        _check(_call("tr_prob_stats_get_view", self._open(), C.byref(v)), what="statistics")
        return v

    @property
    def info(self):
        i = _StatsInfo()
        _check(_call("tr_prob_stats_get_info", self._open(), C.byref(i)), what="statistics")
        return dict(groups=int(i.ngroups), present=int(i.present), points=int(i.d), total=int(i.total),
                    bytes=int(i.bytes), k=int(i.k), mass={1: "count", 2: "weight"}[int(i.mass)],
                    excluded=[n for c, n in enumerate(self.names) if (int(i.exclude) >> c) & 1],
                    scoped=bool(i.scoped), declared=bool(i.declared), product=bool(i.product),
                    source={0: None, 1: "table", 2: "arrays"}.get(int(i.source_kind)), keyed=bool(i.keyed),
                    digest=f"{int(i.digest[0]):016x}{int(i.digest[1]):016x}",
                    support_digest=f"{int(i.support_digest[0]):016x}{int(i.support_digest[1]):016x}")

    @property
    def groups(self):
        return int(self._view().ngroups)

    def arrays(self):
        """copies of the native state: counts [G, d], total [G], rows [G], dropped [G]"""
        v = self._view()
        G, d = int(v.ngroups), int(v.d)
        def arr(ptr, n):
            return np.ctypeslib.as_array((C.c_uint64 * n).from_address(ptr)).copy()
        return dict(counts=arr(v.counts, G * d).reshape(G, d), total=arr(v.total, G), rows=arr(v.rows, G),
                    dropped=arr(v.dropped, G))

    def counts(self, g=0):
        """{point: mass} of group g over the declared support (zero masses included)"""
        a = self.arrays()
        return {p: int(c) for p, c in zip(self.support.points, a["counts"][int(g)])}

    def group(self, g=0):
        a = self.arrays()
        g = int(g)
        return dict(rows=int(a["rows"][g]), total=int(a["total"][g]), dropped=int(a["dropped"][g]),
                    present=bool(a["rows"][g]), counts=self.counts(g))

    def equal(self, other):
        return bool(_check(_call("tr_prob_stats_equal", self._open(), other._open()), what="statistics comparison"))

    def require(self, model):
        """None when the model's support is this state's; DomainError otherwise (a new state is needed)"""
        rc = _call("tr_prob_stats_require", self._open(), model._open())
        if rc == -6:
            raise DomainError("the model's declared support differs from the statistics' support: a support change is "
                              "a different state (build the statistics over the model's support)")
        _check(rc, what="statistics")

    def product(self, other, *, declaration=None, budget=None):
        return product(self, other, declaration=declaration, budget=budget)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_prob_stats_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Stats(closed)"
        i = self.info
        return f"Stats({', '.join(self.names)}: {i['points']} points, {i['groups']} groups, total {i['total']})"


def stats(hist, sup, *, mass="count", exclude=(), budget=None):
    """the retained statistics of a DIST-001 histogram over the declared support (no row is read): mass "count" (rows)
    or "weight" (nonnegative weight sums); exclude: coordinates whose NULL cells are dropped (listwise; "all"),
    the others keep NULL as a category. An observed value outside the support raises DomainError (.group, .point)."""
    if not isinstance(hist, Histogram):
        raise TypeError("stats needs a math_dist.Histogram (the retained support state)")
    if not isinstance(sup, Support):
        raise TypeError("stats needs a declared math_prob.Support")
    if mass not in MASSES:
        raise ValueError("mass is 'count' or 'weight'")
    scope = hist.scope
    if scope is not None and scope.source == "table":
        scope.require_fresh("statistics")
    mask = _exclude_mask(list(hist.names), exclude)
    error, refusal, out = C.c_int32(), _Refusal(), _Outside()
    h = _call("tr_prob_stats_build", hist._open(), scope._handle if scope is not None else None, sup._open(),
              MASSES[mass], mask, C.byref(_budget(budget)), C.byref(refusal), C.byref(out), C.byref(error))
    if error.value == -6 and out.mass:
        k = sup.k
        pt = tuple(None if (out.nulls >> c) & 1 else int(out.value[c]) for c in range(k))
        pt = pt[0] if k == 1 else pt
        err = DomainError(f"statistics: group {int(out.group)} observed {pt!r} ({int(out.mass)} rows), which is not a "
                          f"point of the declared support (declare it, or exclude its NULL coordinates)")
        err.group, err.point, err.rows = int(out.group), pt, int(out.mass)
        raise err
    if error.value == -6:
        raise DomainError("statistics: the histogram's coordinates or decimal scales differ from the support's, or a "
                          "weight is negative")
    return Stats._wrap(_made(h, error, refusal, "statistics"), sup, scope, hist.names, mass)


def product(a, b, *, declaration=None, budget=None):
    """the product of two statistics states (per group n(x, y) = n_a(x) n_b(y): the CROSS JOIN within each group) over
    the product support: the joint of x and y under an independence assumption, so states of one data source need an
    independence declaration covering it (math_conv.independent(scope)); otherwise DomainError with GEN-002's
    .reason ("needs_declaration", "not_covered", "grouping_differs", "keys_differ")"""
    from . import math_conv as mcv
    if not isinstance(a, Stats) or not isinstance(b, Stats):
        raise TypeError("product of two math_prob.Stats")
    if a.scope is None or b.scope is None or isinstance(a.scope, tuple) or isinstance(b.scope, tuple):
        raise ValueError("product: both states need a data source (statistics of a scoped histogram)")
    a.require_fresh("product"); b.require_fresh("product")
    error, refusal, check = C.c_int32(), _Refusal(), mcv._Check()
    h = _call("tr_prob_stats_product", a._open(), a.scope._handle, b._open(), b.scope._handle, mcv._decl(declaration),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(check), C.byref(error))
    if error.value == -6 and check.reason:
        raise mcv._domain_error(check, "product of statistics")
    if error.value == -6:
        raise DomainError("product of statistics: the states have different mass kinds")
    h = _made(h, error, refusal, "product of statistics")
    st = Stats._wrap(h, None, (a.scope, b.scope), a.names + b.names, a.mass)
    st.support = a.support.product(b.support, budget=budget)     # equal to the state's own product support
    return st


# ---------------------------------------------------------------------- models
def _matrix(rows, d, what):
    """rows of rationals -> (num, den) int64 arrays [d, m]"""
    rows = [tuple(r) if isinstance(r, (tuple, list, np.ndarray)) else (r,) for r in rows]
    m = len(rows[0]) if rows else 0
    if any(len(r) != m for r in rows):
        raise ValueError(f"{what}: every point has the same number of features")
    num = np.zeros((d, max(m, 1)), dtype=np.int64)
    den = np.ones((d, max(m, 1)), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, x in enumerate(r):
            num[i, j], den[i, j] = _rational(x, what)
    return num, den, m


class Model:
    """A declared model over a support: categorical (weights per request; features T for the exponential tilt, none =
    the one-hot natural parameters) or exponential (p ~ h(x) exp(theta . T(x)), theta per request). Immutable."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_prob.categorical(...) or math_prob.exponential(...)")

    @classmethod
    def _wrap(cls, handle, sup, kind):
        self = object.__new__(cls)
        self._handle, self.support, self.kind = handle, sup, kind
        v = self._view()
        self.m = int(v.m)
        self.size = sup.size
        self.gradient_size = self.size if kind == "categorical" and self.m == 0 else self.m
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the model is closed")
        return self._handle

    def _view(self):
        v = _ModelView()
        _check(_call("tr_prob_model_get_view", self._open(), C.byref(v)), what="model")
        return v

    @property
    def info(self):
        v = self._view()
        d, m = int(v.d), int(v.m)
        out = dict(kind=self.kind, points=d, features=m, gradient_size=self.gradient_size,
                   feature_bits=[int(v.bits[j]) for j in range(m)])
        if m:
            out["feature_denominators"] = [int(x) for x in np.ctypeslib.as_array((C.c_int64 * m).from_address(v.fden))]
        if self.kind == "exponential":
            out.update(scale=int(v.scale), base_denominator=int(v.hden))
        return out

    def features(self):
        """the declared features as exact Fractions [d][m] (the native integers over their denominators)"""
        v = self._view()
        d, m = int(v.d), int(v.m)
        if not m:
            return None
        num = np.ctypeslib.as_array((C.c_int64 * (d * m)).from_address(v.feat)).reshape(d, m)
        den = np.ctypeslib.as_array((C.c_int64 * m).from_address(v.fden))
        return [[Fraction(int(num[i, j]), int(den[j])) for j in range(m)] for i in range(d)]

    def base(self):
        """exponential models: the base measure h as exact Fractions [d]"""
        if self.kind != "exponential":
            return None
        v = self._view()
        d = int(v.d)
        H = np.ctypeslib.as_array((C.c_int64 * d).from_address(v.base))
        return [Fraction(int(H[i]), int(v.hden)) for i in range(d)]

    # ------------------------------------------------------------------ requests
    def _params(self, params):
        """(num, den) int64 arrays [n, width] of the requests' parameters"""
        width = self.m if self.kind == "exponential" else self.size
        if isinstance(params, tuple) and len(params) == 2 and all(isinstance(x, np.ndarray) for x in params):
            num = np.ascontiguousarray(params[0], dtype=np.int64)
            den = np.ascontiguousarray(params[1], dtype=np.int64)
            if num.ndim == 1:
                num, den = num.reshape(-1, width), den.reshape(-1, width)
            if num.shape != den.shape or num.shape[1:] != (width,):
                raise ValueError(f"parameters: (num, den) arrays of shape (n, {width})")
            return num, den
        if isinstance(params, np.ndarray) and params.dtype.kind in "iu":
            num = np.ascontiguousarray(params, dtype=np.int64).reshape(-1, width)
            return num, np.ones_like(num)
        items = list(params)
        n = len(items)
        num = np.zeros((n, max(width, 1)), dtype=np.int64)
        den = np.ones((n, max(width, 1)), dtype=np.int64)
        for r, it in enumerate(items):
            if self.kind == "categorical":
                vals = self.support._aligned(it, "weights", default=0) if isinstance(it, dict) else list(it)
            else:
                vals = list(it) if isinstance(it, (list, tuple, np.ndarray)) else [it]
            if len(vals) != width:
                raise ValueError(f"a request has {width} parameters ({'weights per point' if self.kind == 'categorical' else 'theta'})")
            for j, x in enumerate(vals):
                num[r, j], den[r, j] = _rational(x, "a parameter")
        return np.ascontiguousarray(num[:, :width]), np.ascontiguousarray(den[:, :width])

    def need(self, st, n=1, *, grad=False, hess=False, backend="cpu"):
        return need(st, self, n, grad=grad, hess=hess, backend=backend)

    def evaluate(self, st, params, *, groups=0, grad=False, hess=False, mode="weights", backend="cpu", budget=None):
        """answers of a batch of parameter requests on the statistics st: exponential models take theta vectors (m
        numbers each), categorical models weight vectors (one weight per support point, a dict {point: weight} or
        exact probabilities with mode="probabilities"); groups: one group for all or one per request. grad / hess: the
        gradient and Hessian of the log-likelihood. backend "cuda" computes the sums with the CUDA kernels."""
        if not isinstance(st, Stats):
            raise TypeError("evaluate needs math_prob.Stats (the retained statistics)")
        if mode not in MODES:
            raise ValueError("mode is 'weights' or 'probabilities'")
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend is 'cpu' or 'cuda'")
        st.require_fresh("evaluation")
        st.require(self)
        num, den = self._params(params)
        n = num.shape[0]
        if isinstance(groups, (Integral, np.integer)):
            gs = np.full(n, int(groups), dtype=np.uint64)
        else:
            gs = np.ascontiguousarray([int(g) for g in groups], dtype=np.uint64)
            if len(gs) != n:
                raise ValueError("one group per request")
        what = (WANT_GRAD if grad or hess else 0) | (WANT_HESS if hess else 0)
        raw = _Need()
        _check(_call("tr_prob_plan_need", st._open(), self._open(), n, what, NEED_CPU, C.byref(raw)), what="evaluation")
        _precheck(budget, raw, "evaluation")
        error, refusal = C.c_int32(), _Refusal()
        plan = _call("tr_prob_plan_create", st._open(), self._open(), n, gs.ctypes.data, num.ctypes.data,
                     den.ctypes.data, MODES[mode], what, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        plan = _made(plan, error, refusal, "evaluation plan")
        try:
            view = _PlanView()
            _check(_call("tr_prob_plan_get_view", plan, C.byref(view)), what="evaluation plan")
            ns = int(view.nsums)
            sums = np.zeros(max(n * ns * 2, 2), dtype=np.int64)
            nref = np.zeros(max(n, 1), dtype=np.int64)
            flags = np.zeros(max(n, 1), dtype=np.uint64)
            device = "native CPU"
            if backend == "cpu":
                _check(_call("tr_prob_sums", C.byref(view), sums.ctypes.data, nref.ctypes.data, flags.ctypes.data),
                       what="evaluation sums")
            else:
                gpu = _gpu()
                gpu.run("tr_mgpu_prob_sums", C.byref(view), C.byref(_budget(budget)), sums.ctypes.data,
                        nref.ctypes.data, flags.ctypes.data, C.byref(refusal), refusal=refusal,
                        what="evaluation sums on the device")
                dinfo = gpu.device()
                backend, device = ("cuda-emulated" if dinfo["emulated"] else "cuda"), dinfo["name"]
            mm = self.gradient_size
            answers = np.zeros(max(n, 1), dtype=ANSWER_DTYPE)
            g = np.zeros((max(n, 1), mm if what else 0, 2)) if what else None
            hs = np.zeros((max(n, 1), mm * (mm + 1) // 2, 2)) if what & WANT_HESS else None
            _check(_call("tr_prob_finish", plan, sums.ctypes.data, nref.ctypes.data, flags.ctypes.data,
                         answers.ctypes.data, g.ctypes.data if g is not None and g.size else None,
                         hs.ctypes.data if hs is not None and hs.size else None, C.byref(_budget(budget)),
                         C.byref(refusal)), refusal, "evaluation")
        except BaseException:
            _call("tr_prob_plan_free", plan)
            raise
        return Evaluation._wrap(plan, self, st, n, what, answers[:n], g[:n] if g is not None else None,
                                hs[:n] if hs is not None else None, sums, nref, flags, ns, backend, device, mode)

    def loglik(self, st, params, *, group=0, backend="cpu", budget=None, **kw):
        """the answer of one request (a dict, see Evaluation.__getitem__)"""
        return self.evaluate(st, [params], groups=group, backend=backend, budget=budget, **kw)[0]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_prob_model_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Model({self.kind}, {self.size} points, {self.m} features)"


def categorical(sup, *, features=None, budget=None):
    """a categorical model over the declared support: each request gives weights (or exact probabilities); features
    (one row of rationals per point, a dict or a callable) define the statistic T of the gradient and Hessian (the
    exponential tilt at theta = 0); None: the one-hot statistic (natural parameters eta = ln w)"""
    if not isinstance(sup, Support):
        raise TypeError("categorical needs a declared math_prob.Support")
    if features is None:
        fn = fd = None
        m = 0
    else:
        fnum, fden, m = _matrix(sup._aligned(features, "features"), sup.size, "a feature")
        fn, fd = fnum.ctypes.data, fden.ctypes.data
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_prob_model_create", sup._open(), KINDS["categorical"], m, fn, fd, None, None,
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Model._wrap(_made(h, error, refusal, "categorical model"), sup, "categorical")


def exponential(sup, features, *, base=None, budget=None):
    """an exponential-family model p_theta(x) = h(x) exp(theta . T(x)) / Z(theta) over the declared support:
    features T (1..16 rationals per point: rows in the support's order, a dict or a callable), base measure h >= 0
    (None: 1 everywhere; rows, a dict or a callable); theta is each request's parameter"""
    if not isinstance(sup, Support):
        raise TypeError("exponential needs a declared math_prob.Support")
    fnum, fden, m = _matrix(sup._aligned(features, "features"), sup.size, "a feature")
    if base is None:
        bn = bd = None
    else:
        vals = sup._aligned(base, "base measure")
        bnum = np.zeros(sup.size, dtype=np.int64)
        bden = np.ones(sup.size, dtype=np.int64)
        for i, x in enumerate(vals):
            bnum[i], bden[i] = _rational(x, "a base measure value")
        bn, bd = bnum.ctypes.data, bden.ctypes.data
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_prob_model_create", sup._open(), KINDS["exponential"], m, fnum.ctypes.data, fden.ctypes.data, bn, bd,
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    if error.value == -6:
        raise DomainError("exponential model: the base measure is negative somewhere or zero everywhere (Z = 0)")
    return Model._wrap(_made(h, error, refusal, "exponential model"), sup, "exponential")


def need(st, model, n=1, *, grad=False, hess=False, backend="cpu"):
    """the needs an evaluation of n requests checks before allocating ({resource: need}; "cuda": the device copies)"""
    what = (WANT_GRAD if grad or hess else 0) | (WANT_HESS if hess else 0)
    out = {}
    for target in ((NEED_CPU,) if backend == "cpu" else (NEED_CPU, NEED_DEVICE)):
        raw = _Need()
        rc = _call("tr_prob_plan_need", st._open(), model._open(), int(n), what, target, C.byref(raw))
        if rc == -6:
            raise DomainError("the model's support differs from the statistics' support")
        _check(rc, what="evaluation needs")
        for k, v in _needs_dict(raw).items():
            out[k] = max(out.get(k, 0), v)
    return out


def ln_bounds(m):
    """(lo, hi) with lo <= ln(m) 2^120 <= hi for 1 <= m < 2^128 (DIST-003's certified logarithm, extended)"""
    m = int(m)
    if not 1 <= m < (1 << 128):
        raise ValueError("1 <= m < 2^128")
    out = (i64 * 4)()
    _check(_call("tr_prob_ln_bounds", m & ((1 << 64) - 1), m >> 64, out), what="ln bounds")
    return _i128(out[0:2]), _i128(out[2:4])


# ---------------------------------------------------------------------- answers
def _flt(pair):
    return float(pair[0]), float(pair[1])


class Evaluation:
    """The answers of a batch of requests (natively computed), with the plan and the exact sums they come from
    (exact() derives exact rational bounds from these very sums, CPU or device)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Model.evaluate(...)")

    @classmethod
    def _wrap(cls, plan, model, st, n, what, answers, grad, hess, sums, nref, flags, nsums, backend, device, mode):
        self = object.__new__(cls)
        self._plan, self.model, self.stats, self.n, self.what = plan, model, st, n, what
        self.answers, self._grad, self._hess = answers, grad, hess
        self.sums, self.nref, self.flags, self.nsums = sums, nref, flags, nsums
        self.backend, self.device, self.mode = backend, device, mode
        return self

    def __len__(self):
        return self.n

    def arrays(self):
        """the raw native results: answers (ANSWER_DTYPE), gradient [n, mm, 2] and Hessian [n, mm (mm + 1) / 2, 2]
        enclosures (packed upper triangle), the exact sums [n, nsums] (int128 words), reference exponents, flags"""
        return dict(answers=self.answers, grad=self._grad, hess=self._hess,
                    sums=self.sums[:self.n * self.nsums * 2].reshape(self.n, self.nsums, 2), nref=self.nref[:self.n],
                    flags=self.flags[:self.n])

    def status(self, i):
        return int(self.answers["status"][i])

    def __getitem__(self, i):
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError("request index out of range")
        r = self.answers[i]
        if int(r["status"]):
            _check(int(r["status"]), what=f"request {i}")
        fl = int(r["flags"])
        neg, empty = bool(fl & NEG_INF), bool(fl & EMPTY)
        cat = self.model.kind == "categorical"
        out = dict(group=int(r["group"]), mass=int(r["mass"]), rows=int(r["rows"]), empty=empty, neg_inf=neg,
                   Z=_flt(r["z"]), Z_exact=(_i128(r["zlo"]), _i128(r["zhi"]), int(r["zexp"]), int(r["zden"])),
                   logZ=_flt(r["logz"]),
                   logZ_bounds=(Fraction(_i256(r["logz_lo"]), 1 << F), Fraction(_i256(r["logz_hi"]), 1 << F)),
                   loglik=_flt(r["loglik"]),
                   width=dict(zip(("Z", "logZ", "loglik", "grad", "hess"), (float(x) for x in r["width"]))),
                   exact=cat, backend=self.backend)
        if cat:
            out["Z_value"] = Fraction(_i128(r["zlo"]), int(r["zden"]))
        if not neg:
            # the transcendental part's exact 2^-120 bounds (exponential families add theta . Tobs: exact())
            out["loglik_log_bounds"] = (Fraction(_i256(r["ll_lo"]), 1 << F), Fraction(_i256(r["ll_hi"]), 1 << F))
        mm = self.model.gradient_size
        if self._grad is not None:
            out["grad"] = None if neg else [_flt(self._grad[i, j]) for j in range(mm)]
        if self._hess is not None:
            if neg:
                out["hess"] = None
            else:
                H = [[None] * mm for _ in range(mm)]
                for j in range(mm):
                    for l in range(j, mm):
                        v = _flt(self._hess[i, j * mm - j * (j + 1) // 2 + l])
                        H[j][l] = H[l][j] = v
                out["hess"] = H
        return out

    def __iter__(self):
        return (self[i] for i in range(self.n))

    def _exact_entry(self, i, entry, budget):
        from . import math_carriers as mc
        h = [vp(), vp(), vp(), vp()]
        ex, fl, refusal = C.c_int32(), C.c_int32(), _Refusal()
        rc = _call("tr_prob_exact", self._plan, int(i), self.sums.ctypes.data, self.nref.ctypes.data,
                   self.flags.ctypes.data, int(entry), C.byref(_budget(budget)), C.byref(refusal), *[C.byref(x) for x in h],
                   C.byref(ex), C.byref(fl))
        try:
            _check(rc, refusal, f"exact bounds of request {i}")
            if not h[0].value:
                return None, bool(ex.value)
            v = [mc._bigint_value(x.value) for x in h]
            return (Fraction(v[0], v[1]), Fraction(v[2], v[3])), bool(ex.value)
        finally:
            free = mc._api()["bigint_free"]
            for x in h:
                if x.value:
                    free(x.value)

    def exact(self, i, *, budget=None):
        """the exact rational bounds (lo, hi) of every entry of request i, from the batch's own sums: Z, logZ, loglik
        (None when -inf), grad, hess (None when undefined); "exact" says which are exact values (lo == hi)"""
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError("request index out of range")
        if self.status(i):
            _check(self.status(i), what=f"request {i}")
        mm = self.model.gradient_size
        out, exact = {}, {}
        for name, e in (("Z", 0), ("logZ", 1), ("loglik", 2)):
            out[name], exact[name] = self._exact_entry(i, e, budget)
        if self.what:
            g = [self._exact_entry(i, 3 + j, budget) for j in range(mm)]
            out["grad"] = None if g and g[0][0] is None else [x[0] for x in g]
            exact["grad"] = bool(g and g[0][1])
        if self.what & WANT_HESS:
            H = [[None] * mm for _ in range(mm)]
            ex = True
            for j in range(mm):
                for l in range(j, mm):
                    v, e = self._exact_entry(i, 3 + mm + j * mm - j * (j + 1) // 2 + l, budget)
                    H[j][l] = H[l][j] = v
                    ex = ex and e
            out["hess"] = None if mm and H[0][0] is None else H
            exact["hess"] = ex
        out["exact"] = exact
        return out

    def equal(self, other):
        """the same answers bit for bit (sums, reference exponents, flags, answers, enclosures)"""
        a, b = self.arrays(), other.arrays()
        same = (np.array_equal(a["sums"], b["sums"]) and np.array_equal(a["nref"], b["nref"]) and
                np.array_equal(a["flags"], b["flags"]) and a["answers"].tobytes() == b["answers"].tobytes())
        for k in ("grad", "hess"):
            if (a[k] is None) != (b[k] is None):
                return False
            if a[k] is not None and a[k].tobytes() != b[k].tobytes():
                return False
        return same

    def close(self):
        if getattr(self, "_plan", None):
            _call("tr_prob_plan_free", self._plan)
            self._plan = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Evaluation({self.model.kind}, {self.n} requests, {self.backend})"
