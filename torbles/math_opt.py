# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""OPT-001: exact minimizers of quadratic objectives q(t) = t^T A t / 2 - b^T t + c, natively (torbles/math_opt.c),
with the face filter of box-constrained problems on the CPU or the CUDA kernel (math_opt_gpu.cu).

    from torbles import math_opt as opt
    q = opt.Quadratic.from_matrices(A, b, c)                  # exact entries (int / Fraction), A symmetric
    q = opt.Quadratic.from_moments(state, group, ridge=Fraction(1, 2))   # a LIN-001 least-squares loss
    q = opt.Quadratic.from_polynomial(loss, ["theta"])        # a first-slice polynomial of degree <= 2
    r = opt.minimize(q)                                       # unique / nonunique / unbounded (with a certificate)
    r = opt.minimize(q, bounds=[(0, 1), None, (-2, Fraction(5, 2))], backend="cuda")    # a box (n <= 8)

Results are exact (Fraction): the optimal point and value, the directions of constant optimal value, the active
bounds, the gradient; an unbounded objective returns a feasible ray (anchor, direction) and its rate (curvature
d^T A d < 0, or slope g(anchor)^T d < 0 along a flat direction), verified natively. Bounds per parameter are (lo, hi)
with both finite, or None (free); a one-sided bound is UnsupportedError, lo > hi InfeasibleError. Opt-in.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check
from .math_lin import _QBuf, _decode, _pack, _FitSpec, _fitspec, _exact

__all__ = ["Quadratic", "Result", "minimize", "faces", "MAX_BOX", "MAX_PARAMS", "STATUSES", "FACE_STATES"]

MAX_BOX, MAX_PARAMS = 8, 63
STATUSES = {1: "unique", 2: "nonunique", 3: "unbounded"}
CERTIFICATES = {0: None, 1: "curvature", 2: "slope"}
FACE_STATES = ("candidate", "undecided", "infeasible", "kkt", "not psd", "duplicate")
PART = dict(A=0, b=1, c=2, point=3, value=4, family=5, direction=6, anchor=7, rate=8, gradient=9, active=10, optima=11)
P = C.POINTER


class _Box(C.Structure):
    _native_ = "tr_opt_box"
    _fields_ = [("n", C.c_int32), ("reserved", C.c_int32), ("has_lo", C.c_void_p), ("has_hi", C.c_void_p),
                ("lo", _QBuf), ("hi", _QBuf)]


class _Info(C.Structure):
    _native_ = "tr_opt_info"
    _fields_ = [(n, C.c_int32) for n in ("status", "n", "bounded", "free", "certificate", "family", "optima", "kkt")] + [
        ("faces", C.c_uint64), ("exact", C.c_uint64), ("filtered", C.c_uint64 * 6), ("method", C.c_int32),
        ("convex", C.c_int32)]


class _FaceView(C.Structure):
    _native_ = "tr_opt_face_view"
    _fields_ = [("n", C.c_int32), ("nk", C.c_int32), ("kidx", C.c_int32 * MAX_BOX), ("faces", C.c_uint64),
                ("degenerate", C.c_uint32), ("reserved", C.c_uint32)] + [
        (n, C.c_double * (MAX_BOX * MAX_BOX)) for n in ("alo", "ahi")] + [
        (n, C.c_double * MAX_BOX) for n in ("blo", "bhi", "llo", "lhi", "ulo", "uhi")]


for _s, _n in ((_Box, 120), (_Info, 104), (_FaceView, 1464)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e, B, R = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32), P(_Budget), P(_Refusal)
        sig = {
            "tr_opt_quad_create": (vp, [i32, P(_QBuf), P(_QBuf), P(_QBuf), B, R, e]),
            "tr_opt_quad_from_poly": (vp, [vp, i32, vp, B, R, e]),
            "tr_opt_quad_from_moments": (vp, [vp, P(_FitSpec), B, R, e]),
            "tr_opt_quad_dim": (i32, [vp]),
            "tr_opt_quad_count": (u64, [vp, i32]),
            "tr_opt_quad_limbs": (u64, [vp, i32]),
            "tr_opt_quad_export": (i32, [vp, i32, P(_QBuf)]),
            "tr_opt_quad_value": (i32, [vp, P(_QBuf), P(_QBuf), B, R]),
            "tr_opt_quad_free": (None, [vp]),
            "tr_opt_face_view_get": (i32, [vp, P(_Box), P(_FaceView)]),
            "tr_opt_faces": (i32, [P(_FaceView), vp]),
            "tr_opt_minimize": (vp, [vp, P(_Box), vp, u64, B, R, e]),
            "tr_opt_result_get_info": (i32, [vp, P(_Info)]),
            "tr_opt_result_count": (u64, [vp, i32]),
            "tr_opt_result_limbs": (u64, [vp, i32]),
            "tr_opt_result_export": (i32, [vp, i32, P(_QBuf)]),
            "tr_opt_result_free": (None, [vp]),
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


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _export(count_fn, limbs_fn, export_fn, handle, part, what):
    n = int(_call(count_fn, handle, part))
    if not n:
        return None
    nl = int(_call(limbs_fn, handle, part))
    sign, nlen, dlen = np.zeros(n, np.int8), np.zeros(n, np.uint32), np.zeros(n, np.uint32)
    limbs = np.zeros(max(nl, 1), np.uint64)
    q = _QBuf(n, nl, sign.ctypes.data, nlen.ctypes.data, dlen.ctypes.data, limbs.ctypes.data)
    _check(_call(export_fn, handle, part, C.byref(q)), what=what)
    return _decode(sign, nlen, dlen, limbs, n)


def _rows(flat, n):
    return [flat[i * n:(i + 1) * n] for i in range(len(flat) // n)] if flat else []


class Quadratic:
    """An immutable exact quadratic objective q(t) = t^T A t / 2 - b^T t + c (native); names: its parameters."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Quadratic.from_matrices / from_moments / from_polynomial")

    @classmethod
    def _wrap(cls, handle, names):
        self = object.__new__(cls)
        self._h = handle
        self.n = int(_call("tr_opt_quad_dim", handle))
        self.names = tuple(names) if names is not None else tuple(f"t{i}" for i in range(self.n))
        return self

    @classmethod
    def from_matrices(cls, A, b, c=0, names=None, budget=None):
        rows = [list(r) for r in A]
        n = len(rows)
        if not n or any(len(r) != n for r in rows) or len(list(b)) != n:
            raise ValueError("A is n x n and b has n values")
        qa, ka = _pack([v for r in rows for v in r], "A")
        qb, kb = _pack(list(b), "b")
        qc, kc = _pack([c], "c")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_opt_quad_create", n, C.byref(qa), C.byref(qb), C.byref(qc), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "quadratic"), names)

    @classmethod
    def from_moments(cls, state, group, *, intercept=None, ridge=0, features=None, budget=None):
        """the least-squares loss sum (y - z^T t)^2 + ridge * |t_features|^2 of a LIN-001 fit (value units)"""
        fs = _fitspec(list(state.names), group, intercept, ridge, features, len(state.features))
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_opt_quad_from_moments", state._handle, C.byref(fs), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        q = cls._wrap(_made(h, error, refusal, "least-squares objective"), None)
        q.names = tuple(state.names[int(d)] for d in _export("tr_opt_quad_count", "tr_opt_quad_limbs",
                                                             "tr_opt_quad_export", q._h, PART["active"], "design"))
        return q

    @classmethod
    def from_polynomial(cls, expr, parameters, budget=None):
        """a first-slice polynomial (torbles.mathematics Expression / Function) of degree <= 2 in `parameters`"""
        parameters = list(parameters)
        idx = (C.c_int32 * max(len(parameters), 1))(*[expr.context._index(p) for p in parameters])
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_opt_quad_from_poly", expr._open(), len(parameters), idx, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "polynomial objective"), parameters)

    def _part(self, name):
        return _export("tr_opt_quad_count", "tr_opt_quad_limbs", "tr_opt_quad_export", self._h, PART[name], name)

    @property
    def A(self):
        return _rows(self._part("A"), self.n)

    @property
    def b(self):
        return self._part("b")

    @property
    def c(self):
        return self._part("c")[0]

    def value(self, t, budget=None):
        """q(t) exactly (native)"""
        t = list(t)
        if len(t) != self.n:
            raise ValueError(f"t has {self.n} values")
        qt, kt = _pack(t, "t")
        sign, nlen, dlen = np.zeros(1, np.int8), np.zeros(1, np.uint32), np.zeros(1, np.uint32)
        limbs = np.zeros(4096, np.uint64)
        out = _QBuf(1, len(limbs), sign.ctypes.data, nlen.ctypes.data, dlen.ctypes.data, limbs.ctypes.data)
        refusal = _Refusal()
        _check(_call("tr_opt_quad_value", self._h, C.byref(qt), C.byref(out), C.byref(_budget(budget)), C.byref(refusal)),
               refusal, "objective value")
        return _decode(sign, nlen, dlen, limbs, 1)[0]

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _native is not None:
            _native["tr_opt_quad_free"](h)
            self._h = None

    def __repr__(self):
        return f"Quadratic(n={self.n}, parameters={list(self.names)})"


class Result:
    """The exact minimum: status "unique" / "nonunique" / "unbounded"; point, value, family (directions of
    constant optimal value), active (-1 lower bound, 1 upper bound, 0 free), gradient, optima (every distinct
    optimal point found); unbounded: anchor, direction, rate and certificate ("curvature" or "slope")."""

    def __init__(self, handle, names, backend, device):
        self._h = handle
        i = _Info()
        _check(_call("tr_opt_result_get_info", handle, C.byref(i)), what="minimum")
        self.status, self.certificate = STATUSES[i.status], CERTIFICATES[i.certificate]
        self.n, self.names, self.backend, self.device = int(i.n), tuple(names), backend, device
        self.kkt, self.convex = bool(i.kkt), bool(i.convex)
        self.method = ("", "unconstrained", "faces")[i.method]
        self.faces, self.exact_checks = int(i.faces), int(i.exact)
        self.filtered = {name: int(i.filtered[k]) for k, name in enumerate(FACE_STATES)}
        part = lambda name: _export("tr_opt_result_count", "tr_opt_result_limbs", "tr_opt_result_export", handle,
                                    PART[name], name)
        self.point, self.gradient = part("point"), part("gradient")
        v = part("value")
        self.value = v[0] if v else None
        self.family = _rows(part("family"), self.n)
        a = part("active")
        self.active = [int(x) for x in a] if a else None
        self.optima = _rows(part("optima"), self.n) if i.method == 2 else ([self.point] if self.point else [])
        self.direction, self.anchor = part("direction"), part("anchor")
        r = part("rate")
        self.rate = r[0] if r else None

    @property
    def unique(self):
        return self.status == "unique"

    def same(self, other):
        keys = ("status", "certificate", "point", "value", "family", "active", "gradient", "optima", "direction",
                "anchor", "rate", "kkt")
        return all(getattr(self, k) == getattr(other, k) for k in keys)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _native is not None:
            _native["tr_opt_result_free"](h)
            self._h = None

    def __repr__(self):
        if self.status == "unbounded":
            return f"Result(unbounded, {self.certificate}, direction={self.direction}, rate={self.rate})"
        return f"Result({self.status}, value={self.value}, point={self.point}, active={self.active})"


def _box(q, bounds):
    """the native box and its keepalive arrays; bounds: one (lo, hi) or None per parameter, or {name: (lo, hi)}"""
    if isinstance(bounds, dict):
        unknown = set(bounds) - set(q.names)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}")
        bounds = [bounds.get(name) for name in q.names]
    bounds = list(bounds)
    if len(bounds) != q.n:
        raise ValueError(f"one bound pair (or None) per parameter ({q.n})")
    has_lo, has_hi = np.zeros(max(q.n, 1), np.uint8), np.zeros(max(q.n, 1), np.uint8)
    lo, hi = [], []
    for i, bd in enumerate(bounds):
        l, h = (None, None) if bd is None else bd
        has_lo[i], has_hi[i] = l is not None, h is not None
        lo.append(0 if l is None else _exact(l, "lower bound"))
        hi.append(0 if h is None else _exact(h, "upper bound"))
    ql, kl = _pack(lo, "lower bounds")
    qh, kh = _pack(hi, "upper bounds")
    box = _Box(q.n, 0, has_lo.ctypes.data, has_hi.ctypes.data, ql, qh)
    return box, (has_lo, has_hi, kl, kh)


def faces(q, bounds, backend="cpu", budget=None):
    """the face filter's statuses of a box (FACE_STATES names), on the CPU or the CUDA kernel"""
    box, keep = _box(q, bounds)
    view = _FaceView()
    _check(_call("tr_opt_face_view_get", q._h, C.byref(box), C.byref(view)), what="face view")
    status = np.zeros(max(int(view.faces), 1), np.uint8)
    if backend == "cpu":
        _check(_call("tr_opt_faces", C.byref(view), status.ctypes.data), what="faces")
    elif backend == "cuda":
        from . import math_gpu
        math_gpu.bind("tr_mgpu_opt_faces", C.c_int, [P(_FaceView), P(_Budget), C.c_void_p, P(_Refusal)])
        refusal = _Refusal()
        math_gpu.run("tr_mgpu_opt_faces", C.byref(view), C.byref(_budget(budget)), status.ctypes.data, C.byref(refusal),
                     refusal=refusal, what="faces on the device")
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    return status[:int(view.faces)]


def minimize(q, bounds=None, *, backend="cpu", budget=None):
    """the exact minimum of q over R^n (bounds None) or a box (n <= 8); backend "cuda" runs the face filter of a
    box on the device (the exact enumeration of the surviving faces is native CPU in both cases)"""
    error, refusal = C.c_int32(), _Refusal()
    if bounds is None:
        h = _call("tr_opt_minimize", q._h, None, None, 0, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Result(_made(h, error, refusal, "minimize"), q.names, "cpu", "native CPU")
    box, keep = _box(q, bounds)
    status, name, device = None, "cpu", "native CPU"
    if backend == "cuda":
        view = _FaceView()
        _check(_call("tr_opt_face_view_get", q._h, C.byref(box), C.byref(view)), what="face view")
        status = faces(q, bounds, "cuda", budget)
        from . import math_gpu
        d = math_gpu.device()
        name, device = ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]
    elif backend != "cpu":
        raise ValueError("backend is 'cpu' or 'cuda'")
    h = _call("tr_opt_minimize", q._h, C.byref(box), None if status is None else status.ctypes.data,
              0 if status is None else len(status), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Result(_made(h, error, refusal, "minimize"), q.names, name, device)
