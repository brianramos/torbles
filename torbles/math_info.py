# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DIST-003: entropy, mutual information and contingency statistics of the retained count support of a DIST-001
joint histogram, with certified logarithms (torbles/math_info.c; the per-cell terms and their exact accumulation also
in CUDA, math_info_gpu.cu).

    from torbles import math_dist as md, math_info as mi
    h = md.histogram(scope, ["x", "y", "z"])                 # the joint support (NULL: a category of its own)
    inf = mi.information(h, exclude=["z"], backend="cpu")     # NULLs of z dropped (declared), or exclude=() / "all"
    inf.entropy("x"); inf.entropy(["x", "y"], units="bits")   # {"lo", "hi", "bounds": exact Fractions (nats), ...}
    inf.conditional_entropy("x", given="y"); inf.mutual_information("x", "y", given="z")
    inf.chi_square("x", "y", exact=True)                      # + "exact": the exact rational X^2, "df", "rows", "cols"
    inf.g_test("x", "y"); inf.cramers_v("x", "y", exact=True)
    inf.query([dict(op="entropy", a=["x"], group=g), ...])    # a batch from the shared support

Every value is the enclosure [lo, hi] of the exact real value (lo <= value <= hi, outward-rounded doubles) with the
exact rational bounds it comes from; chi-square and V^2 are also available as exact rationals (exact=True). The
statistics read the joint support state only: asking for a statistic of coordinates the state does not hold jointly
raises UnsupportedError (nothing is ever inferred from marginals or moments). Python only marshals and formats; the
values are computed natively. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_dist import Histogram, _i256
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Information", "information", "need", "ln_bounds", "OPS", "UNITS", "REQUEST_DTYPE", "RESULT_DTYPE"]

OPS = {"entropy": 1, "conditional_entropy": 2, "mutual_information": 3, "conditional_mi": 4, "chi_square": 5,
       "g_test": 6, "cramers_v": 7}
UNITS = {"nats": 0, "bits": 1}
MASSES = {"count": 1, "weight": 2}
SCALE, CHI_SCALE = 112, 128
P = C.POINTER
i32, u32, i64, u64, vp = C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_void_p


class _View(C.Structure):
    _native_ = "tr_info_view"
    _fields_ = [("ngroups", u64), ("nslots", u64), ("nlog", u64), ("nchi", u64), ("lkey", vp), ("lmass", vp),
                ("ckey", vp), ("co", vp), ("cma", vp), ("cmb", vp)]


class _SupportInfo(C.Structure):
    _native_ = "tr_info_support_info"
    _fields_ = [(n, u64) for n in ("ngroups", "present", "joint_cells", "nlog", "nchi", "nslots", "max_cells", "bytes",
                                   "total")] + [(n, i32) for n in ("k", "mass", "exclude", "npairs")] + [
        ("scale", i32 * 3), ("wscale", i32)]


class _Request(C.Structure):
    _native_ = "tr_info_request"
    _fields_ = [("op", i32), ("units", i32), ("group", u64), ("a", u32), ("b", u32), ("c", u32), ("reserved", u32)]


class _Result(C.Structure):
    _native_ = "tr_info_result"
    _fields_ = [("status", i32), ("empty", i32), ("lo", C.c_double), ("hi", C.c_double), ("num_lo", u64 * 4),
                ("num_hi", u64 * 4), ("den", u64), ("shift", i32), ("squared", i32), ("total", u64), ("rows", u64),
                ("cols", u64), ("df", u64)]


for _s, _n in ((_View, 80), (_SupportInfo, 104), (_Request, 32), (_Result, 136)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))
REQUEST_DTYPE = np.dtype([("op", "<i4"), ("units", "<i4"), ("group", "<u8"), ("a", "<u4"), ("b", "<u4"),
                          ("c", "<u4"), ("reserved", "<u4")])
RESULT_DTYPE = np.dtype([("status", "<i4"), ("empty", "<i4"), ("lo", "<f8"), ("hi", "<f8"), ("num_lo", "<u8", (4,)),
                         ("num_hi", "<u8", (4,)), ("den", "<u8"), ("shift", "<i4"), ("squared", "<i4"),
                         ("total", "<u8"), ("rows", "<u8"), ("cols", "<u8"), ("df", "<u8")])
assert REQUEST_DTYPE.itemsize == 32 and RESULT_DTYPE.itemsize == 136

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_info_support_build": (vp, [vp, i32, u32, B, R, e]),
            "tr_info_support_need": (i32, [vp, i32, u32, P(_Need)]),
            "tr_info_support_get_view": (i32, [vp, P(_View)]),
            "tr_info_support_get_info": (i32, [vp, P(_SupportInfo)]),
            "tr_info_support_group": (i32, [vp, u64, P(u64), P(u64), P(u64)]),
            "tr_info_support_free": (None, [vp]),
            "tr_info_sums": (i32, [P(_View), vp]),
            "tr_info_state_build": (vp, [vp, vp, B, R, e]),
            "tr_info_state_equal": (i32, [vp, vp]),
            "tr_info_state_bytes": (u64, [vp]),
            "tr_info_state_free": (None, [vp]),
            "tr_info_query": (i32, [vp, vp, u64, vp, B, R]),
            "tr_info_exact": (i32, [vp, u64, i32, u32, u32, B, R, P(vp), P(vp)]),
            "tr_info_device_need": (u64, [P(_View)]),
            "tr_info_ln_bounds": (i32, [u64, P(u64)]),
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
    math_gpu.bind("tr_mgpu_info_sums", C.c_int, [P(_View), P(_Budget), vp, P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def ln_bounds(m):
    """(lo, hi) with lo <= ln(m) 2^120 <= hi: the certified fixed-point logarithm of every cell, 1 <= m < 2^64"""
    out = (u64 * 4)()
    _check(_call("tr_info_ln_bounds", int(m), out), what="ln bounds")
    return (int(out[1]) << 64) | int(out[0]), (int(out[3]) << 64) | int(out[2])


def _exclude_mask(hist, exclude):
    names = hist.names
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


def need(hist, *, mass="count", exclude=()):
    """the needs the support build checks first (variables, states = kept joint cells, bytes = its scratch)"""
    if not isinstance(hist, Histogram):
        raise TypeError("a math_dist.Histogram (the joint support state)")
    if mass not in MASSES:
        raise ValueError("mass is 'count' or 'weight'")
    out = _Need()
    _check(_call("tr_info_support_need", hist._open(), MASSES[mass], _exclude_mask(hist, exclude), C.byref(out)),
           what="information needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}


def information(hist, *, mass="count", exclude=(), backend="cpu", budget=None):
    """the information state of a joint histogram: mass "count" (rows) or "weight" (nonnegative weight sums);
    exclude: coordinates whose NULL cells are dropped (listwise; "all" for every coordinate), the others keep NULL as
    a category. backend "cuda" computes the per-cell terms and their sums with the CUDA kernels."""
    if not isinstance(hist, Histogram):
        raise TypeError("information needs a math_dist.Histogram: the joint support state (marginals or moment states "
                        "do not determine a joint distribution)")
    if mass not in MASSES:
        raise ValueError("mass is 'count' or 'weight'")
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    mask = _exclude_mask(hist, exclude)
    error, refusal = C.c_int32(), _Refusal()
    sp = _call("tr_info_support_build", hist._open(), MASSES[mass], mask, C.byref(_budget(budget)), C.byref(refusal),
               C.byref(error))
    sp = _made(sp, error, refusal, "information support")
    try:
        sums, device = None, "native CPU"
        if backend == "cuda":
            gpu = _gpu()
            view = _View()
            _check(_call("tr_info_support_get_view", sp, C.byref(view)), what="support view")
            sums = np.zeros(max(int(view.ngroups) * int(view.nslots) * 8, 1), dtype=np.int64)
            gpu.run("tr_mgpu_info_sums", C.byref(view), C.byref(_budget(budget)), sums.ctypes.data, C.byref(refusal),
                    refusal=refusal, what="information sums on the device")
            d = gpu.device()
            backend, device = ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]
        st = _call("tr_info_state_build", sp, sums.ctypes.data if sums is not None else None, C.byref(_budget(budget)),
                   C.byref(refusal), C.byref(error))
        st = _made(st, error, refusal, "information state")
    except BaseException:
        _call("tr_info_support_free", sp)
        raise
    return Information._wrap(sp, st, hist, mass, mask, backend, device)


class Information:
    """The immutable information state of a joint histogram: its support (marginal tables and contingency cells,
    native) and the certified sums of every table; requests are answered natively from them."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_info.information(histogram, ...)")

    @classmethod
    def _wrap(cls, support, state, hist, mass, mask, backend, device):
        self = object.__new__(cls)
        self._support, self._state, self.names = support, state, tuple(hist.names)
        self.mass, self.exclude_mask, self.backend, self.device = mass, mask, backend, device
        self.scope = hist.scope
        return self

    def _open(self):
        if not self._state:
            raise ValueError("the information state is closed")
        return self._state

    @property
    def info(self):
        out = _SupportInfo()
        _check(_call("tr_info_support_get_info", self._support, C.byref(out)), what="information support")
        k = int(out.k)
        return dict(groups=int(out.ngroups), present=int(out.present), joint_cells=int(out.joint_cells),
                    table_cells=int(out.nlog), contingency_cells=int(out.nchi), slots=int(out.nslots),
                    max_cells=int(out.max_cells), support_bytes=int(out.bytes),
                    state_bytes=int(_call("tr_info_state_bytes", self._open())), total=int(out.total), k=k,
                    mass=self.mass, excluded=[n for i, n in enumerate(self.names) if (self.exclude_mask >> i) & 1],
                    scales=tuple(int(out.scale[c]) for c in range(k)), wscale=int(out.wscale),
                    backend=self.backend, device=self.device)

    def view(self):
        v = _View()
        _check(_call("tr_info_support_get_view", self._support, C.byref(v)), what="support view")
        return v

    def group(self, g=0):
        """{"rows" (selected rows), "total" (kept mass N), "cells": {coordinates: categories of that marginal}}"""
        k = len(self.names)
        rows, total, cells = C.c_uint64(), C.c_uint64(), (u64 * (1 << k))()
        _check(_call("tr_info_support_group", self._support, int(g), C.byref(rows), C.byref(total), cells),
               what="group")
        named = {tuple(n for i, n in enumerate(self.names) if (S >> i) & 1): int(cells[S]) for S in range(1, 1 << k)}
        return dict(rows=int(rows.value), total=int(total.value), cells=named)

    def equal(self, other):
        return bool(_check(_call("tr_info_state_equal", self._open(), other._open()), what="state comparison"))

    # ------------------------------------------------------------------ requests
    def _mask(self, cols, what):
        if cols is None:
            return 0
        cols = [cols] if isinstance(cols, str) else list(cols)
        mask = 0
        for n in cols:
            if n not in self.names:
                raise UnsupportedError(f"{what}: {n!r} is not a coordinate of this joint support ({', '.join(self.names)})"
                                       "; build the histogram over every coordinate the statistic needs (a joint "
                                       "distribution is never inferred from marginals)")
            mask |= 1 << self.names.index(n)
        return mask

    def requests(self, items):
        """a REQUEST_DTYPE array from dicts {"op", "a", "b", "c" (coordinate names), "group", "units"}"""
        items = list(items)
        out = np.zeros(len(items), dtype=REQUEST_DTYPE)
        for i, q in enumerate(items):
            if q["op"] not in OPS:
                raise ValueError(f"op is one of {', '.join(OPS)}")
            units = q.get("units", "nats")
            if units not in UNITS:
                raise ValueError("units are 'nats' or 'bits'")
            out["op"][i], out["units"][i], out["group"][i] = OPS[q["op"]], UNITS[units], int(q.get("group", 0))
            out["a"][i], out["b"][i], out["c"][i] = (self._mask(q.get("a"), q["op"]), self._mask(q.get("b"), q["op"]),
                                                     self._mask(q.get("c"), q["op"]))
        return out

    def query_arrays(self, reqs, *, budget=None):
        """the raw RESULT_DTYPE answers of a REQUEST_DTYPE array (natively; every answer has its own status)"""
        reqs = np.ascontiguousarray(reqs, dtype=REQUEST_DTYPE)
        out = np.zeros(len(reqs), dtype=RESULT_DTYPE)
        refusal = _Refusal()
        _check(_call("tr_info_query", self._open(), reqs.ctypes.data if len(reqs) else None, len(reqs),
                     out.ctypes.data if len(out) else None, C.byref(_budget(budget)), C.byref(refusal)), refusal,
               "information requests")
        return out

    @staticmethod
    def _format(op, r, units):
        if int(r["status"]):
            _check(int(r["status"]), what=f"{op} request")
        out = dict(empty=bool(r["empty"]), total=int(r["total"]), units=units if op in (
            "entropy", "conditional_entropy", "mutual_information", "conditional_mi") else None)
        if op in ("chi_square", "g_test", "cramers_v"):
            out.update(rows=int(r["rows"]), cols=int(r["cols"]), df=int(r["df"]))
        else:
            out["cells"] = int(r["rows"])
        if out["empty"]:
            out.update(lo=None, hi=None, bounds=None)
            return out
        scale = int(r["den"]) << int(r["shift"])
        out.update(lo=float(r["lo"]), hi=float(r["hi"]),
                   bounds=(Fraction(_i256(r["num_lo"]), scale), Fraction(_i256(r["num_hi"]), scale)))
        if r["squared"]:
            out["bounds_of"] = "V^2"
        elif units == "bits":
            out["bounds_of"] = "nats"
        return out

    def query(self, items, *, budget=None):
        """answers of request dicts: [{"lo", "hi", "bounds" (exact Fractions; nats; V: of V^2), "empty", "total",
        and "cells" (categories of a) or "rows", "cols", "df"}]; a request the native code refuses raises"""
        items = list(items)
        res = self.query_arrays(self.requests(items), budget=budget)
        return [self._format(q["op"], r, q.get("units", "nats")) for q, r in zip(items, res)]

    def _one(self, **q):
        return self.query([q])[0]

    def entropy(self, columns, *, group=0, units="nats"):
        """H(columns) (several columns: the joint entropy)"""
        return self._one(op="entropy", a=columns, group=group, units=units)

    def conditional_entropy(self, columns, given, *, group=0, units="nats"):
        """H(columns | given)"""
        return self._one(op="conditional_entropy", a=columns, b=given, group=group, units=units)

    def mutual_information(self, a, b, *, given=None, group=0, units="nats"):
        """I(a; b), or I(a; b | given)"""
        if given is None:
            return self._one(op="mutual_information", a=a, b=b, group=group, units=units)
        return self._one(op="conditional_mi", a=a, b=b, c=given, group=group, units=units)

    def g_test(self, a, b, *, group=0):
        """G = 2 sum O ln(O / E) = 2 N I(a; b), with the degrees of freedom"""
        return self._one(op="g_test", a=a, b=b, group=group)

    def _exact(self, op, a, b, group, budget):
        from . import math_carriers as mc
        num, den, refusal = vp(), vp(), _Refusal()
        rc = _call("tr_info_exact", self._support, int(group), OPS[op], self._mask(a, op), self._mask(b, op),
                   C.byref(_budget(budget)), C.byref(refusal), C.byref(num), C.byref(den))
        try:
            _check(rc, refusal, f"exact {op}")
            if not num.value:
                return None
            return Fraction(mc._bigint_value(num.value), mc._bigint_value(den.value))
        finally:
            free = mc._api()["bigint_free"]
            for h in (num, den):
                if h.value:
                    free(h.value)

    def chi_square(self, a, b, *, group=0, exact=False, budget=None):
        """Pearson's X^2 over T(a) x T(b) (structural zeros included), its enclosure and df; exact=True adds the exact
        rational ("exact")"""
        r = self._one(op="chi_square", a=a, b=b, group=group)
        if exact:
            r["exact"] = self._exact("chi_square", a, b, group, budget)
        return r

    def cramers_v(self, a, b, *, group=0, exact=False, budget=None):
        """Cramer's V = sqrt(X^2 / (N (min(rows, cols) - 1))); exact=True adds the exact rational V^2 ("exact_v2")"""
        r = self._one(op="cramers_v", a=a, b=b, group=group)
        if exact:
            r["exact_v2"] = self._exact("cramers_v", a, b, group, budget)
        return r

    def close(self):
        if getattr(self, "_state", None):
            _call("tr_info_state_free", self._state)
            self._state = None
        if getattr(self, "_support", None):
            _call("tr_info_support_free", self._support)
            self._support = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._state:
            return "Information(closed)"
        return f"Information({', '.join(self.names)}, mass={self.mass}, {self.backend})"
