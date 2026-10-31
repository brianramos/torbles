# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""RESP-002: complete piecewise absolute, hinge and squared-hinge loss functions of a parameter theta from the
sorted support and prefix moments of a DIST-002 prefix state, natively on the CPU or with the CUDA kernel
(torbles/math_resp.c, math_resp_gpu.cu).

    from torbles import math_dist as md, math_resp as mr
    p = md.histogram(scope, ["x"], weight="w").prefix("weight", degree=2)    # or prefix("count", ...)
    loss = mr.Loss(p, "abs")          # "hinge" (x - theta)+, "hinge_down" (theta - x)+, "sqhinge", "sqhinge_down",
                                      # "squared", or mr.Loss(p, "pinball", tau=Fraction(1, 4))
    loss.evaluate([0, Fraction(7, 2), 10], group=g)   # [{"value", "left", "right", "at_breakpoint", "index"}]
    loss.evaluate(thetas, group=g, backend="cuda")    # one CUDA work item per theta, same answers
    loss.pieces(group=g)              # the complete function: [{"lo", "hi", "coefficients": (c0, c1, c2)}]
    loss.minimizers(group=g)          # {"kind", "lo", "hi", "point", "value"}: the closed minimizer set

theta and the answers are exact rationals in the logical units of the column (a DECIMAL(p, s) value is logical /
10^s; the loss carries the mass's units: rows, or the weight's logical integers). Slopes are the one-sided
derivatives. The minimizer set is reported whole as [lo, hi] (None for an unbounded end); its canonical point is lo
when bounded below, else hi; an empty mass (absent group, only NULL values) has the loss 0 everywhere ("all"). Every
value is computed natively; Python marshals and formats. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from .math_budgets import _Budget, _Refusal, Budget
from .math_dist import Prefix, _i128, _i256, _rational
from .math_errors import check as _check

__all__ = ["Loss", "LOSSES", "KINDS", "REQUEST_DTYPE", "RESULT_DTYPE", "evaluate_arrays", "requests"]

LOSSES = {"abs": 1, "hinge": 2, "hinge_down": 3, "sqhinge": 4, "sqhinge_down": 5, "squared": 6, "pinball": 7}
KINDS = {1: "all", 2: "point", 3: "interval", 4: "up", 5: "down"}
P = C.POINTER
i32, i64, u64, vp = C.c_int32, C.c_int64, C.c_uint64, C.c_void_p


class _Request(C.Structure):
    _native_ = "tr_resp_request"
    _fields_ = [("loss", i32), ("reserved", i32), ("group", u64), ("num", i64), ("den", i64), ("tau_num", i64),
                ("tau_den", i64)]


class _Result(C.Structure):
    _native_ = "tr_resp_result"
    _fields_ = [("status", i32), ("at_break", i32), ("index", u64), ("value", u64 * 4), ("left", u64 * 4),
                ("right", u64 * 4), ("value_den", i64 * 2), ("slope_den", i64 * 2)]


class _Piece(C.Structure):
    _native_ = "tr_resp_piece"
    _fields_ = [("lo_inf", i32), ("hi_inf", i32), ("lo", i64), ("hi", i64), ("c", u64 * 12), ("den", i64 * 2)]


class _Minimum(C.Structure):
    _native_ = "tr_resp_minimum"
    _fields_ = [("status", i32), ("kind", i32), ("lo_inf", i32), ("hi_inf", i32), ("lo", u64 * 4), ("hi", u64 * 4),
                ("lo_den", i64 * 2), ("hi_den", i64 * 2), ("value", u64 * 4), ("value_den", i64 * 2),
                ("lo_index", u64), ("hi_index", u64)]


for _s, _n in ((_Request, 48), (_Result, 144), (_Piece, 136), (_Minimum, 176)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

REQUEST_DTYPE = np.dtype([("loss", "<i4"), ("reserved", "<i4"), ("group", "<u8"), ("num", "<i8"), ("den", "<i8"),
                          ("tau_num", "<i8"), ("tau_den", "<i8")])
RESULT_DTYPE = np.dtype([("status", "<i4"), ("at_break", "<i4"), ("index", "<u8"), ("value", "<u8", (4,)),
                         ("left", "<u8", (4,)), ("right", "<u8", (4,)), ("value_den", "<i8", (2,)),
                         ("slope_den", "<i8", (2,))])
assert REQUEST_DTYPE.itemsize == 48 and RESULT_DTYPE.itemsize == 144

_native = None


def _api():
    global _native
    if _native is None:
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_resp_eval": (i32, [vp, vp, u64, vp, B, R]),
            "tr_resp_pieces": (i32, [vp, u64, i32, i64, i64, vp, u64, P(u64), B, R]),
            "tr_resp_minimize": (i32, [vp, u64, i32, i64, i64, P(_Minimum)]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, args
            native[name] = fn
        _native = native
    return _native


def _gpu():
    from . import math_dist
    gpu = math_dist._gpu()
    gpu.bind("tr_mgpu_resp_eval", C.c_int, [vp, vp, u64, vp, P(_Budget), P(_Refusal)])
    return gpu


def _budget(budget):
    return (budget or Budget()).raw


def _tau(loss, tau):
    if loss == "pinball":
        if tau is None:
            raise ValueError("the pinball loss needs tau in [0, 1]")
        num, den = _rational(tau, "tau")
        if not 0 <= num <= den:
            raise ValueError("tau lies in [0, 1]")
        return num, den
    if tau is not None:
        raise ValueError("tau applies to the pinball loss only")
    return 0, 1


def requests(loss, thetas, group=0, tau=None):
    """a REQUEST_DTYPE array: one evaluation of loss per theta (exact rationals in logical units)"""
    if loss not in LOSSES:
        raise ValueError(f"loss is one of {', '.join(LOSSES)}")
    tn, td = _tau(loss, tau)
    thetas = list(thetas)
    a = np.zeros(len(thetas), dtype=REQUEST_DTYPE)
    a["loss"], a["group"], a["tau_num"], a["tau_den"] = LOSSES[loss], int(group), tn, td
    for i, t in enumerate(thetas):
        a["num"][i], a["den"][i] = _rational(t, "theta")
    return a


def evaluate_arrays(prefix, reqs, *, backend="cpu", budget=None):
    """the raw RESULT_DTYPE answers of a REQUEST_DTYPE array (every answer with its own status)"""
    if not isinstance(prefix, Prefix):
        raise TypeError("a math_dist.Prefix state")
    reqs = np.ascontiguousarray(reqs, dtype=REQUEST_DTYPE)
    out = np.zeros(len(reqs), dtype=RESULT_DTYPE)
    refusal = _Refusal()
    rp = reqs.ctypes.data if len(reqs) else None
    op = out.ctypes.data if len(out) else None
    if backend == "cpu":
        _check(_api()["tr_resp_eval"](prefix._open(), rp, len(reqs), op, C.byref(_budget(budget)), C.byref(refusal)),
               refusal, "loss evaluation")
    elif backend == "cuda":
        gpu = _gpu()
        prefix.device(budget)
        gpu.run("tr_mgpu_resp_eval", prefix._device, rp, len(reqs), op, C.byref(_budget(budget)), C.byref(refusal),
                refusal=refusal, what="loss evaluation on the device")
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    return out


def _q(words, den):
    return Fraction(_i256(words), _i128(den))


class Loss:
    """One loss family over a prefix state (the state is shared, never copied)."""

    def __init__(self, prefix, loss, *, tau=None):
        if not isinstance(prefix, Prefix):
            raise TypeError("a math_dist.Prefix state (Histogram.prefix(...))")
        if loss not in LOSSES:
            raise ValueError(f"loss is one of {', '.join(LOSSES)}")
        self.prefix, self.loss, self._tau = prefix, loss, _tau(loss, tau)
        self.tau = Fraction(*self._tau) if loss == "pinball" else None

    def evaluate(self, thetas, *, group=0, backend="cpu", budget=None):
        """[{"value", "left", "right" (exact Fractions), "at_breakpoint", "index"}] per theta"""
        res = evaluate_arrays(self.prefix, requests(self.loss, thetas, group, self.tau), backend=backend,
                              budget=budget)
        out = []
        for r in res:
            if int(r["status"]):
                _check(int(r["status"]), what=f"{self.loss} loss")
            out.append(dict(value=_q(r["value"], r["value_den"]), left=_q(r["left"], r["slope_den"]),
                            right=_q(r["right"], r["slope_den"]), at_breakpoint=bool(r["at_break"]),
                            index=int(r["index"])))
        return out

    def value(self, theta, *, group=0, backend="cpu"):
        return self.evaluate([theta], group=group, backend=backend)[0]["value"]

    def slopes(self, theta, *, group=0, backend="cpu"):
        r = self.evaluate([theta], group=group, backend=backend)[0]
        return r["left"], r["right"]

    def pieces(self, *, group=0, budget=None):
        """the complete function: [{"lo", "hi" (None when unbounded), "coefficients": (c0, c1, c2)}] with
        loss(theta) = c0 + c1 theta + c2 theta^2 on each closed piece"""
        fn = _api()["tr_resp_pieces"]
        count, refusal = C.c_uint64(), _Refusal()
        tn, td = self._tau
        _check(fn(self.prefix._open(), int(group), LOSSES[self.loss], tn, td, None, 0, C.byref(count),
                  C.byref(_budget(budget)), C.byref(refusal)), refusal, "loss pieces")
        n = int(count.value)
        arr = (_Piece * max(n, 1))()
        _check(fn(self.prefix._open(), int(group), LOSSES[self.loss], tn, td, arr, n, C.byref(count),
                  C.byref(_budget(budget)), C.byref(refusal)), refusal, "loss pieces")
        out = []
        for pc in arr[:n]:
            den = _i128(pc.den)
            coeffs = tuple(Fraction(_i256(pc.c[4 * j:4 * j + 4]), den) for j in range(3))
            out.append(dict(lo=None if pc.lo_inf else int(pc.lo), hi=None if pc.hi_inf else int(pc.hi),
                            coefficients=coeffs))
        return out

    def minimizers(self, *, group=0):
        """{"kind": "all" | "point" | "interval" | "up" | "down", "lo", "hi" (None when unbounded), "point" (the
        canonical minimizer: lo when bounded below, else hi; None for "all"), "value" (the minimum)}"""
        m = _Minimum()
        tn, td = self._tau
        _check(_api()["tr_resp_minimize"](self.prefix._open(), int(group), LOSSES[self.loss], tn, td, C.byref(m)),
               what="loss minimizers")
        lo = None if m.lo_inf else _q(m.lo, m.lo_den)
        hi = None if m.hi_inf else _q(m.hi, m.hi_den)
        return dict(kind=KINDS[int(m.kind)], lo=lo, hi=hi, point=lo if lo is not None else hi,
                    value=_q(m.value, m.value_den),
                    lo_index=None if m.lo_inf else int(m.lo_index), hi_index=None if m.hi_inf else int(m.hi_index))

    def __repr__(self):
        return f"Loss({self.loss}{'' if self.tau is None else f', tau={self.tau}'}, {self.prefix!r})"
