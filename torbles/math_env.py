# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ENV-001: univariate affine / quadratic group envelopes: the winning group on every piece of a parameter interval
and the exact breakpoints with a declared tie policy, natively on the CPU, lookups also with the CUDA kernel
(torbles/math_env.c, math_env_gpu.cu).

    from fractions import Fraction
    from torbles import math_env as me
    env = me.envelope([(3, -1), (0, 1), None, (1, 0, Fraction(1, 2))], lo=-5, hi=None, upper=False, tie="smallest")
    env.regions        # [{"kind": "point" | "piece", "winner", "groups", "classes", "at" / "lo", "hi", ...}]
    env.points         # Fraction, or Algebraic(polynomial, root, interval) for an irrational breakpoint
    env.lookup([0, Fraction(7, 3)], backend="cuda")      # region, winner, tied per parameter value
    env.evaluate_all(ts)                                 # every group function at every t (verification path)
    env.value(Fraction(1, 2))                            # the exact envelope value at a rational t
    me.from_bank(bank, constant="y^2", linear=(-2, "x*y"), quadratic="x^2")     # response functions from moments
    me.from_expressions([loss_g for g in groups], "theta")                    # first-slice polynomials

A group given as None is empty (absent: it never wins). Identical functions form one class, coincident on every
piece. Breakpoints are exact rationals, or the smaller / larger root of a primitive integer quadratic with a
certified isolating interval from INV-001 (refine(width) narrows it); an irrational breakpoint is never a float.
Python marshals values (Fraction <-> native big rationals) and formats results; nothing is computed here. Opt-in:
`import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_carriers as _mc
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check
from .math_inv import _Values, _exact, _fraction as _borrowed

__all__ = ["envelope", "from_bank", "from_expressions", "Envelope", "Algebraic", "TIES", "PRIMES_MAX"]

NONE = (1 << 64) - 1
PRIMES_MAX = 64
TIES = {"smallest": 0, "largest": 1}
MODES = {"search": 1, "all": 2}
STATUS = {0: "ok", 1: "outside", 2: "empty"}
P = C.POINTER


class _Info(C.Structure):
    _native_ = "tr_env_info"
    _fields_ = [(n, C.c_uint64) for n in ("groups", "present", "classes", "regions", "pieces", "points", "iterations",
                                          "bytes")] + [
        (n, C.c_int32) for n in ("kind", "upper", "tie", "degree", "lo_bounded", "hi_bounded", "algebraic",
                                 "reserved")]


class _Region(C.Structure):
    _native_ = "tr_env_region"
    _fields_ = [("kind", C.c_int32), ("boundary", C.c_int32)] + [
        (n, C.c_uint64) for n in ("point", "right", "classes", "cls", "winner", "members")]


class _PointInfo(C.Structure):
    _native_ = "tr_env_point_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "root", "boundary", "sign_lo")] + [("region", C.c_uint64)]


class _ClassInfo(C.Structure):
    _native_ = "tr_env_class_info"
    _fields_ = [("members", C.c_uint64), ("winner", C.c_uint64), ("degree", C.c_int32), ("reserved", C.c_int32)]


class _Answer(C.Structure):
    _native_ = "tr_env_answer"
    _fields_ = [("status", C.c_int32), ("kind", C.c_int32), ("region", C.c_uint64), ("winner", C.c_uint64),
                ("tied", C.c_uint64)]


class _View(C.Structure):
    _native_ = "tr_env_view"
    _fields_ = [(n, C.c_uint64) for n in ("npoints", "nbreaks", "nregions", "nclasses")] + [
        ("nprimes", C.c_uint32), ("limbs", C.c_uint32)] + [
        (n, C.c_int32) for n in ("mode", "kind", "lo_bounded", "hi_bounded", "tie", "reserved")] + [
        (n, C.c_void_p) for n in ("fields", "garner", "half", "bkind", "bsign", "bres", "point_region",
                                  "piece_region", "rwinner", "rclasses", "cres", "cwinner", "qsign", "qnum", "qden")]


for _s, _n in ((_Info, 96), (_Region, 56), (_PointInfo, 24), (_ClassInfo, 24), (_Answer, 32), (_View, 184)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R, N = P(_Budget), P(_Refusal), P(_Need)
        sig = {
            "tr_env_need": (i32, [u64, u64, i32, N]),
            "tr_env_build": (vp, [u64, i32, vp, vp, vp, vp, i32, i32, B, R, e]),
            "tr_env_from_polys": (vp, [u64, vp, i32, vp, vp, i32, i32, B, R, e]),
            "tr_env_from_bank": (vp, [vp, vp, vp, i32, vp, vp, i32, i32, B, R, e]),
            "tr_env_get_info": (i32, [vp, P(_Info)]),
            "tr_env_region_get": (i32, [vp, u64, P(_Region)]),
            "tr_env_region_classes": (i32, [vp, u64, vp, u64]),
            "tr_env_class_get": (i32, [vp, u64, P(_ClassInfo)]),
            "tr_env_class_members": (i32, [vp, u64, vp, u64]),
            "tr_env_class_coef": (vp, [vp, u64, i32]),
            "tr_env_point_get": (i32, [vp, u64, P(_PointInfo)]),
            "tr_env_point_value": (vp, [vp, u64]),
            "tr_env_point_poly": (vp, [vp, u64, i32]),
            "tr_env_point_interval": (i32, [vp, u64, P(vp), P(vp)]),
            "tr_env_point_refine": (i32, [vp, u64, vp, P(vp), P(vp), B, R]),
            "tr_env_bound": (vp, [vp, i32]),
            "tr_env_value": (i32, [vp, vp, P(vp), P(u64), P(i32), B, R]),
            "tr_env_free": (None, [vp]),
            "tr_env_lookup_need": (i32, [vp, u64, vp, i32, i32, N]),
            "tr_env_lookup_create": (vp, [vp, u64, vp, i32, B, R, e]),
            "tr_env_lookup_need_q64": (i32, [vp, u64, vp, vp, i32, i32, N]),
            "tr_env_lookup_create_q64": (vp, [vp, u64, vp, vp, i32, B, R, e]),
            "tr_env_lookup_get_view": (i32, [vp, P(_View)]),
            "tr_env_lookup_bound_bits": (u64, [vp]),
            "tr_env_lookup_eval": (i32, [P(_View), vp, B, R]),
            "tr_env_lookup_free": (None, [vp]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, args
            native[name[7:]] = fn
        _native = native
    return _native


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_env_lookup", C.c_int, [P(_View), C.c_void_p, P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _raw(budget):
    return C.byref((budget or Budget()).raw)


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _owned(handle):
    """a Fraction from an owned tr_bigq handle (freed here)"""
    try:
        return _borrowed(handle)
    finally:
        _mc._api()["bigq_free"](handle)


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _bound(value, what):
    """(native handle or None, keep-alive object) of an interval end (None = unbounded)"""
    if value is None:
        return None, None
    q = _mc.BigRational(_exact(value, what))
    return q._h, q


def _options(upper, tie):
    if tie not in TIES:
        raise ValueError("tie is 'smallest' or 'largest'")
    return int(bool(upper)), TIES[tie]


class Algebraic:
    """An irrational breakpoint: root `root` (0 the smaller, 1 the larger) of the primitive integer polynomial
    q0 + q1 t + q2 t^2 (q2 > 0, irreducible), inside the certified isolating interval (lo, hi) from INV-001 (q changes
    sign there: sign_lo at lo, -sign_lo at hi; exactly one root of q inside). Never a number: refine() narrows it."""
    __slots__ = ("polynomial", "root", "interval", "sign_lo", "_env", "_index")

    def __init__(self, polynomial, root, interval, sign_lo, env=None, index=None):
        self.polynomial, self.root, self.interval, self.sign_lo = tuple(polynomial), int(root), tuple(interval), sign_lo
        self._env, self._index = env, index

    exact = False

    def refine(self, width, budget=None):
        """an isolating interval of width <= width (INV-001's exact refinement)"""
        if self._env is None:
            raise ValueError("refine needs the envelope the point belongs to")
        w, keep = _bound(width, "width")
        if w is None or keep.to_fraction() <= 0:
            raise ValueError("the width is a positive exact number")
        lo, hi, refusal = C.c_void_p(), C.c_void_p(), _Refusal()
        _check(_api()["point_refine"](self._env._open(), self._index, w, C.byref(lo), C.byref(hi), _raw(budget),
                                      C.byref(refusal)), refusal, "refine")
        return _owned(lo.value), _owned(hi.value)

    def __eq__(self, other):
        return isinstance(other, Algebraic) and (self.polynomial, self.root) == (other.polynomial, other.root)

    def __hash__(self):
        return hash((self.polynomial, self.root))

    def __repr__(self):
        q0, q1, q2 = self.polynomial
        return f"Algebraic({'smaller' if self.root == 0 else 'larger'} root of {q2} t^2 + {q1} t + {q0} in {self.interval})"


class Envelope:
    """The lower (or upper) envelope of per-group affine / quadratic functions on a parameter interval (native,
    immutable): regions in increasing order, points (boundaries and breakpoints), classes of identical functions."""

    def __init__(self, handle):
        self._h = handle
        api = _api()
        info = _Info()
        _check(api["get_info"](handle, C.byref(info)), what="envelope")
        self.kind = "empty" if info.kind == 0 else "finite"
        self.upper, self.tie = bool(info.upper), {v: k for k, v in TIES.items()}[info.tie]
        self.info = dict(groups=int(info.groups), present=int(info.present), classes=int(info.classes),
                         regions=int(info.regions), pieces=int(info.pieces), points=int(info.points),
                         iterations=int(info.iterations), bytes=int(info.bytes), degree=int(info.degree),
                         algebraic=int(info.algebraic), lo_bounded=bool(info.lo_bounded),
                         hi_bounded=bool(info.hi_bounded))
        lo, hi = api["bound"](handle, 0), api["bound"](handle, 1)
        self.lo = _borrowed(lo) if lo else None
        self.hi = _borrowed(hi) if hi else None
        self.classes = []
        for c in range(int(info.classes)):
            ci = _ClassInfo()
            _check(api["class_get"](handle, c, C.byref(ci)), what="envelope class")
            mem = np.zeros(max(int(ci.members), 1), dtype=np.uint64)
            _check(api["class_members"](handle, c, mem.ctypes.data, len(mem)), what="envelope class")
            coef = tuple(_borrowed(api["class_coef"](handle, c, k)) for k in range(3))
            self.classes.append(dict(coefficients=coef, groups=[int(g) for g in mem[:int(ci.members)]],
                                     winner=int(ci.winner), degree=int(ci.degree)))
        self.points = []
        for p in range(int(info.points)):
            pi = _PointInfo()
            _check(api["point_get"](handle, p, C.byref(pi)), what="envelope point")
            if pi.kind == 1:
                self.points.append(_borrowed(api["point_value"](handle, p)))
            else:
                lo_, hi_ = C.c_void_p(), C.c_void_p()
                _check(api["point_interval"](handle, p, C.byref(lo_), C.byref(hi_)), what="envelope point")
                poly = tuple(_mc._bigint_value(api["point_poly"](handle, p, k)) for k in range(3))
                self.points.append(Algebraic(poly, pi.root, (_borrowed(lo_.value), _borrowed(hi_.value)),
                                             int(pi.sign_lo), self, p))
        self.regions = []
        for r in range(int(info.regions)):
            ri = _Region()
            _check(api["region_get"](handle, r, C.byref(ri)), what="envelope region")
            cls = np.zeros(max(int(ri.classes), 1), dtype=np.uint64)
            _check(api["region_classes"](handle, r, cls.ctypes.data, len(cls)), what="envelope region")
            classes = [int(c) for c in cls[:int(ri.classes)]]
            groups = sorted(g for c in classes for g in self.classes[c]["groups"])
            if ri.kind == 1:
                region = dict(kind="piece", lo=None if ri.point == NONE else self.points[ri.point],
                              hi=None if ri.right == NONE else self.points[ri.right], cls=classes[0])
            else:
                region = dict(kind="point", at=self.points[ri.point], point=int(ri.point),
                              boundary={0: None, 1: "lo", 2: "hi", 3: "lo and hi"}[ri.boundary])
            region.update(classes=classes, groups=groups, winner=int(ri.winner))
            self.regions.append(region)

    def _open(self):
        if not self._h:
            raise ValueError("the envelope is closed")
        return self._h

    @property
    def breakpoints(self):
        """the interior points (boundaries excluded)"""
        return [r["at"] for r in self.regions if r["kind"] == "point" and r["boundary"] is None]

    @staticmethod
    def _points(ts):
        """(kind, data): int64 numerator / denominator arrays when every value fits them, else native big
        rationals (marshaling only: the exact values are converted, never rounded)"""
        qs = [_exact(t, "parameter value") for t in ts]
        lim = 1 << 63
        if all(-lim <= q.numerator < lim and q.denominator < lim for q in qs):
            num = np.array([q.numerator for q in qs] or [0], dtype=np.int64)
            den = np.array([q.denominator for q in qs] or [1], dtype=np.int64)
            return "q64", (num, den, len(qs))
        return "big", _Values(qs, "parameter value")

    def _job(self, ts, mode, budget):
        kind, data = self._points(ts)
        error, refusal = C.c_int32(), _Refusal()
        if kind == "q64":
            num, den, n = data
            h = _api()["lookup_create_q64"](self._open(), n, num.ctypes.data, den.ctypes.data, MODES[mode],
                                            _raw(budget), C.byref(refusal), C.byref(error))
        else:
            h = _api()["lookup_create"](self._open(), data.n, data.array, MODES[mode], _raw(budget), C.byref(refusal),
                                        C.byref(error))
        return _made(h, error, refusal, "envelope lookup"), data

    def lookup_need(self, ts, mode="search", backend="cpu"):
        kind, data = self._points(ts)
        out = _Need()
        target = 0 if backend == "cpu" else 1
        if kind == "q64":
            num, den, n = data
            rc = _api()["lookup_need_q64"](self._open(), n, num.ctypes.data, den.ctypes.data, MODES[mode], target,
                                           C.byref(out))
        else:
            rc = _api()["lookup_need"](self._open(), data.n, data.array, MODES[mode], target, C.byref(out))
        _check(rc, what="lookup needs")
        return _needs(out)

    def lookup(self, ts, *, backend="cpu", mode="search", budget=None):
        """per parameter value t: {"status" ("ok" / "outside" / "empty"), "kind" ("piece" / "point"), "region"
        (search), "winner", "tied" (classes attaining the envelope at t)}; mode "all" evaluates every class at t"""
        if mode not in MODES:
            raise ValueError("mode is 'search' or 'all'")
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend is 'cpu' or 'cuda'")
        ts = list(ts)
        if backend == "cuda":                          # the host job and its device copies, checked together
            from . import math_budgets
            math_budgets.check(budget or Budget(), "envelope lookup on the device",
                               **self.lookup_need(ts, mode=mode, backend="cuda"))
        job, keep = self._job(ts, mode, budget)
        try:
            view = _View()
            _check(_api()["lookup_get_view"](job, C.byref(view)), what="envelope lookup")
            out = (_Answer * max(len(ts), 1))()
            refusal = _Refusal()
            if backend == "cpu":
                _check(_api()["lookup_eval"](C.byref(view), out, _raw(budget), C.byref(refusal)), refusal,
                       "envelope lookup")
            else:
                gpu = _gpu()
                gpu.run("tr_mgpu_env_lookup", C.byref(view), out, _raw(budget), C.byref(refusal), refusal=refusal,
                        what="envelope lookup on the device")
            self.last_lookup = dict(backend=backend if backend == "cpu" else
                                    ("cuda-emulated" if _gpu().device()["emulated"] else "cuda"),
                                    primes=int(view.nprimes), bound_bits=int(_api()["lookup_bound_bits"](job)))
        finally:
            _api()["lookup_free"](job)
        res = []
        for a in out[:len(ts)]:
            if a.status < 0:
                _check(int(a.status), what="envelope lookup")
            res.append(dict(status=STATUS[a.status], kind={0: None, 1: "piece", 2: "point"}[a.kind],
                            region=None if a.region == NONE else int(a.region),
                            winner=None if a.winner == NONE else int(a.winner), tied=int(a.tied)))
        return res

    def evaluate_all(self, ts, *, backend="cpu", budget=None):
        """every class evaluated exactly at every t (the verification path): winner and classes tied"""
        return self.lookup(ts, backend=backend, mode="all", budget=budget)

    def value(self, t, budget=None):
        """(the exact envelope value at a rational t, its region), or (None, None) outside the interval / empty"""
        h, keep = _bound(t, "parameter value")
        out, region, status, refusal = C.c_void_p(), C.c_uint64(), C.c_int32(), _Refusal()
        _check(_api()["value"](self._open(), h, C.byref(out), C.byref(region), C.byref(status), _raw(budget),
                               C.byref(refusal)), refusal, "envelope value")
        if status.value:
            return None, None
        return _owned(out.value), int(region.value)

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
            return "Envelope(closed)"
        return (f"Envelope({'upper' if self.upper else 'lower'}, {self.kind}, classes={self.info['classes']}, "
                f"regions={self.info['regions']}, algebraic={self.info['algebraic']})")


def need(groups, present=None, degree=2):
    """the a-priori needs of a build ({resource: need})"""
    out = _Need()
    _check(_api()["need"](int(groups), int(groups if present is None else present), int(degree), C.byref(out)),
           what="envelope needs")
    return _needs(out)


def envelope(functions, lo=None, hi=None, *, upper=False, tie="smallest", budget=None):
    """the envelope of functions[g] = coefficients (a0, a1, a2) of a0 + a1 t + a2 t^2 (exact; shorter = affine or
    constant; longer: a nonzero coefficient of degree >= 3 is UnsupportedError), None for an empty group"""
    functions = list(functions)
    ncoef = max([len(f) for f in functions if f is not None] + [1])
    if ncoef > 8:
        raise ValueError("at most 8 coefficients (degree <= 2 is supported)")
    flat, present = [], np.zeros(max(len(functions), 1), dtype=np.uint8)
    for g, f in enumerate(functions):
        if f is None:
            flat += [None] * ncoef
            continue
        present[g] = 1
        f = list(f)
        flat += f + [None] * (ncoef - len(f))
    vals = _Values(flat, "coefficient", allow_none=True)
    up, ti = _options(upper, tie)
    hl, kl = _bound(lo, "lo")
    hh, kh = _bound(hi, "hi")
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["build"](len(functions), ncoef, vals.array, present.ctypes.data, hl, hh, up, ti, _raw(budget),
                        C.byref(refusal), C.byref(error))
    return Envelope(_made(h, error, refusal, "envelope"))


def from_expressions(expressions, variable, lo=None, hi=None, *, upper=False, tie="smallest", budget=None):
    """the envelope of first-slice polynomials (torbles.mathematics Expressions, e.g. RESP-001 response functions) in
    one variable; None for an empty group; a term in another variable raises UnsupportedError (dimension 1 only)"""
    exprs = list(expressions)
    context = next((e.context for e in exprs if e is not None), None)
    if context is None:
        raise ValueError("give at least one expression")
    if variable not in context.names:
        raise ValueError(f"{variable!r} is not a variable of the expressions' context")
    arr = (C.c_void_p * max(len(exprs), 1))(*[(e._open() if e is not None else None) for e in exprs])
    up, ti = _options(upper, tie)
    hl, kl = _bound(lo, "lo")
    hh, kh = _bound(hi, "hi")
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["from_polys"](len(exprs), arr, context.names.index(variable), hl, hh, up, ti, _raw(budget),
                             C.byref(refusal), C.byref(error))
    return Envelope(_made(h, error, refusal, "envelope of expressions"))


def from_bank(bank, constant=None, linear=None, quadratic=None, *, per=None, lo=None, hi=None, upper=False,
              tie="smallest", budget=None):
    """response functions from a FND-004 bank's retained moments: f_g(t) = sum_k m_k sum_g[term_k] t^k / den_g, each
    coefficient a term or (multiplier, term) (None: 0); per None (den 1), "rows" or a term (its count): a mean.
    Groups without rows (or a zero divisor) are empty."""
    from . import math_bank as mbk
    terms, mults = [], []
    for spec in (constant, linear, quadratic):
        if spec is None:
            terms.append(-1); mults.append(0); continue
        m, t = (spec if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], int)
                and not isinstance(spec[1], int) else (1, spec))
        row = tuple(int(k) for k in mbk._basis(bank.names, [t])[0])
        if row not in bank.terms:
            raise ValueError(f"{t!r} is not a term of the bank")
        terms.append(bank.terms.index(row)); mults.append(int(m))
    if per is None:
        dv = -1
    elif per == "rows":
        dv = -2
    else:
        row = tuple(int(k) for k in mbk._basis(bank.names, [per])[0])
        if row not in bank.terms:
            raise ValueError(f"{per!r} is not a term of the bank")
        dv = bank.terms.index(row)
    up, ti = _options(upper, tie)
    hl, kl = _bound(lo, "lo")
    hh, kh = _bound(hi, "hi")
    t_arr, m_arr = (C.c_int32 * 3)(*terms), (C.c_int64 * 3)(*mults)
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["from_bank"](bank._handle, t_arr, m_arr, dv, hl, hh, up, ti, _raw(budget), C.byref(refusal),
                            C.byref(error))
    return Envelope(_made(h, error, refusal, "envelope of response functions"))
