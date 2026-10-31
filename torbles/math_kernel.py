# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""KERNEL-001: exact finite-feature kernel embeddings, natively on the CPU or with the CUDA kernels
(torbles/math_kernel.c, math_kernel_gpu.cu; every contract in math_kernel.h).

    from torbles import math_bank as mbk, math_kernel as mk
    scope = mbk.Scope.from_table(db["f"], ["x", "y", "w", "c"], group_by=["k"], rows=mask)
    st = mk.features(scope, ["x", "y"], mk.polynomial(2, c=1), weight="w")     # backend="cuda", repr="sparse"
    st.sum(0, 1); st.sum(0, self_pairs=False, ordered=False)    # {"value", "pairs"}: exact ints
    st.mean(0, 1); st.mmd2(0, 1, unbiased=True); st.witness(0, [(1, 2), (0, -3)])   # Fractions (None: empty)
    st.query([dict(op="mmd2", a=0, b=1, unbiased=True), ...], backend="cuda")
    st.sum(0, 1, kernel=mk.linear(["x"]))        # another kernel the state retains (degree, columns), or refused
    mk.kernel("rbf", gamma=1)                     # no exact finite feature map: refused natively (UnsupportedError)

Kernels: polynomial (x.y + c)^d (linear, homogeneous), monomial sum_t w_t x^a_t y^a_t, delta (categorical match, NULL a
category). The feature state is a TENSOR-001 moment tensor over (1, x...) (or a count tensor) with the per-group
weight sums and self-pair terms. Answers are exact (int / Fraction). Nothing is computed in Python. Opt-in: `import
torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_tensor as mt
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Kernel", "FeatureState", "polynomial", "linear", "homogeneous", "monomial", "delta", "kernel", "features",
           "need", "MAX_DEGREE", "MAX_COLUMNS", "MAX_TERMS"]

WORDS, MAX_DEGREE, MAX_COLUMNS, MAX_TERMS = 16, 8, 32, 64
KINDS = {"polynomial": 1, "monomial": 2, "delta": 3}
_REFUSED = {"rbf": 16, "gaussian": 16, "laplace": 17, "laplacian": 17}
OPS = {"sum": 1, "mean": 2, "mmd2": 3, "witness": 4}
SELF, UNORDERED, UNBIASED = 1, 2, 4
P = C.POINTER


class _Spec(C.Structure):
    _native_ = "tr_kernel_spec"
    _fields_ = [(n, C.c_int32) for n in ("kind", "degree", "ncols", "nterms")] + [
        ("c", C.c_int64), ("cols", C.c_int32 * MAX_COLUMNS), ("weight", C.c_int64 * MAX_TERMS),
        ("exps", (C.c_uint8 * MAX_COLUMNS) * MAX_TERMS), ("reserved", C.c_int32 * 4)]


class _Wide(C.Structure):
    _native_ = "tr_kernel_wide"
    _fields_ = [("w", C.c_uint64 * WORDS)]


class _Request(C.Structure):
    _native_ = "tr_kernel_request"
    _fields_ = [("op", C.c_int32), ("flags", C.c_int32), ("a", C.c_uint64), ("b", C.c_uint64), ("point", C.c_uint64),
                ("reserved", C.c_int64)]


class _Result(C.Structure):
    _native_ = "tr_kernel_result"
    _fields_ = [("status", C.c_int32), ("empty", C.c_int32), ("num", _Wide), ("den", _Wide), ("pairs", _Wide)]


class _View(C.Structure):
    _native_ = "tr_kernel_view"
    _fields_ = [(n, C.c_int32) for n in ("kind", "repr", "order", "nclass", "has_self", "npos")] + [
        ("reserved", C.c_int32 * 2)] + [(n, C.c_uint64) for n in ("ngroups", "cells", "nitems", "npoints")] + [
        ("dims", C.c_uint64 * mt.MAX_ORDER), ("stride", C.c_uint64 * mt.MAX_ORDER)] + [
        (n, C.c_void_p) for n in ("item_index", "item_class", "item_mult", "class_coef", "pos_of_axis", "value",
                                  "offset", "coord", "W", "Q", "Dg")] + [
        ("lo", C.c_int64 * mt.MAX_ORDER), ("span", C.c_uint64 * mt.MAX_ORDER)]


class _Info(C.Structure):
    _native_ = "tr_kernel_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "present", "rows", "used", "bytes", "cells", "nnz")] + [
        (n, C.c_int32) for n in ("kind", "degree", "ncols", "order", "repr", "weighted")] + [
        ("reserved", C.c_int32 * 2), ("digest", C.c_uint64 * 2)]


for _s, _n in ((_Spec, 2728), (_Wide, 128), (_Request, 40), (_Result, 392), (_View, 408), (_Info, 104)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e, R, B = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32), P(_Refusal), P(_Budget)
        sig = {
            "tr_kernel_check": (i32, [vp, P(_Spec)]),
            "tr_kernel_build": (vp, [vp, P(_Spec), i32, i32, i32, B, R, e]),
            "tr_kernel_need": (i32, [vp, P(_Spec), i32, i32, i32, i32, P(_Need), P(u64)]),
            "tr_kernel_from_states": (vp, [vp, P(_Spec), i32, i32, i32, P(mt._States), vp, vp, vp, B, R, e]),
            "tr_kernel_get_info": (i32, [vp, P(_Info)]),
            "tr_kernel_get_group": (i32, [vp, u64, P(mt._I128), P(_Wide), P(_Wide)]),
            "tr_kernel_get_tensor": (vp, [vp]),
            "tr_kernel_get_spec": (i32, [vp, P(_Spec)]),
            "tr_kernel_free": (None, [vp]),
            "tr_kernel_eval_make": (vp, [vp, P(_Spec), B, R, e]),
            "tr_kernel_eval_get_view": (i32, [vp, P(_View)]),
            "tr_kernel_eval_free": (None, [vp]),
            "tr_kernel_query": (i32, [vp, vp, u64, vp, vp, u64, vp, B, R]),
            "tr_kernel_view_device_bytes": (u64, [P(_View)]),
            "tr_tensor_clone": (vp, [vp, e]),
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
    math_gpu.bind("tr_mgpu_kernel_build", C.c_int, [P(mbk._View), P(_Spec), C.c_int32, C.c_int32, C.c_int32,
                                                   P(_Budget), P(mt._States), C.c_void_p, C.c_void_p, C.c_void_p,
                                                   P(_Refusal)])
    lib = math_gpu.lib()
    fn = lib.tr_mgpu_kernel_upload
    fn.restype, fn.argtypes = C.c_void_p, [P(_View), C.c_uint64, P(_Budget), P(_Refusal), P(C.c_int)]
    math_gpu.bind("tr_mgpu_kernel_query", C.c_int, [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p,
                                                   C.c_uint64, C.c_void_p, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_kernel_free", None, [C.c_void_p])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _wide(w):
    """the exact int of a 1024-bit two's complement word array"""
    v = 0
    for i in reversed(range(WORDS)):
        v = (v << 64) | int(w.w[i])
    return v - (1 << (64 * WORDS)) if v >> (64 * WORDS - 1) else v


def _i64(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise TypeError(f"{what} must be an exact integer")
    x = int(x)
    if not -(1 << 63) <= x < 1 << 63:
        raise OverflowError(f"{what} does not fit int64")
    return x


class Kernel:
    """A kernel description (columns optional: the feature state's own by default). kind: polynomial, monomial,
    delta, or a name without an exact finite feature map (refused natively when used)."""

    def __init__(self, kind, *, degree=0, c=0, terms=None, columns=None, **params):
        self.kind, self.degree, self.c, self.terms, self.params = kind, int(degree), c, terms, params
        self.columns = None if columns is None else list(columns)

    def spec(self, scope, columns=None):
        """the native spec over a scope's columns (the kernel's own columns, else `columns`)"""
        cols = self.columns if self.columns is not None else (list(columns) if columns is not None else None)
        if cols is None:
            raise ValueError("the kernel needs columns")
        s = _Spec()
        if self.kind in KINDS:
            s.kind = KINDS[self.kind]
        elif self.kind in _REFUSED:
            s.kind = _REFUSED[self.kind]
        else:
            s.kind = 18
        if len(cols) > MAX_COLUMNS:
            raise UnsupportedError(f"at most {MAX_COLUMNS} kernel columns")
        s.ncols = len(cols)
        for j, name in enumerate(cols):
            if name not in scope.names:
                raise ValueError(f"{name!r} is not a scope column")
            s.cols[j] = scope.names.index(name)
        if self.kind == "polynomial":
            s.degree, s.c = self.degree, _i64(self.c, "c")
        elif self.kind == "monomial":
            terms = list(self.terms or [])
            if len(terms) > MAX_TERMS:
                raise UnsupportedError(f"at most {MAX_TERMS} monomials")
            s.nterms = len(terms)
            for t, (mono, w) in enumerate(terms):
                s.weight[t] = _i64(w, "a monomial weight")
                for name, e in dict(mono).items():
                    if name not in cols:
                        raise ValueError(f"monomial column {name!r} is not a kernel column")
                    if not 0 <= int(e) <= 255:
                        raise ValueError("exponents are 0..255")
                    s.exps[t][cols.index(name)] = int(e)
        return s

    def __repr__(self):
        return f"Kernel({self.kind}, degree={self.degree}, c={self.c}, columns={self.columns})"


def polynomial(degree, c=0, columns=None):
    """(x.y + c)^degree, its exact monomial feature expansion"""
    return Kernel("polynomial", degree=degree, c=c, columns=columns)


def linear(columns=None):
    return polynomial(1, 0, columns)


def homogeneous(degree, columns=None):
    return polynomial(degree, 0, columns)


def monomial(terms, columns=None):
    """sum_t w_t x^a_t y^a_t; terms: [({column: exponent}, weight)]"""
    terms = list(terms)
    cols = columns
    if cols is None:
        seen = []
        for mono, _ in terms:
            for name in dict(mono):
                if name not in seen:
                    seen.append(name)
        cols = seen
    return Kernel("monomial", terms=terms, columns=cols)


def delta(columns=None):
    """prod_k [x_k = y_k] over categorical columns, NULL a category of its own"""
    return Kernel("delta", columns=columns)


def kernel(name, **params):
    """a kernel by name: polynomial / linear / homogeneous / monomial / delta, or any other name (rbf, laplace, ...):
    refused natively when used, since it has no exact finite feature map (KERNEL-002 approximates explicitly)"""
    if name == "polynomial":
        return polynomial(params.pop("degree"), params.pop("c", 0), params.pop("columns", None))
    if name == "linear":
        return linear(params.pop("columns", None))
    if name == "homogeneous":
        return homogeneous(params.pop("degree"), params.pop("columns", None))
    if name == "monomial":
        return monomial(params.pop("terms"), params.pop("columns", None))
    if name == "delta":
        return delta(params.pop("columns", None))
    return Kernel(name, columns=params.pop("columns", None), **params)


def _repr(repr):
    if repr not in ("dense", "sparse"):
        raise ValueError("repr is 'dense' or 'sparse'")
    return mt.REPRS[repr]


def _weight(scope, weight):
    if weight is None:
        return -1
    if weight not in scope.names:
        raise ValueError(f"{weight!r} is not a scope column")
    return scope.names.index(weight)


def need(scope, columns, kernel, *, weight=None, degree=None, repr="dense", backend="cpu"):
    """the needs a build checks before allocating ({resource: need}); backend "cuda": its device copies"""
    s = kernel.spec(scope, columns)
    _check(_call("tr_kernel_check", scope._handle, C.byref(s)), what="kernel")
    out = _Need()
    target = 1 if backend == "cuda" else 0
    _check(_call("tr_kernel_need", scope._handle, C.byref(s), _weight(scope, weight), int(degree or 0), _repr(repr),
                 target, C.byref(out), None), what="kernel needs")
    return {n: int(out.need[i]) for i, n in enumerate(RESOURCES) if out.need[i]}


def features(scope, columns, kernel, *, weight=None, degree=None, repr="dense", backend="cpu", budget=None):
    """the feature state of `kernel` over the scope's columns: a TENSOR-001 moment tensor of order `degree` (default
    the kernel's) over (1, columns...) (delta: the count tensor of the columns) with the per-group weight sums W,
    squared weights Q and self-pair terms of the kernel; backend "cuda": the CUDA kernels"""
    if scope.source == "table":
        scope.require_fresh("kernel features")
    columns = list(columns)
    s = kernel.spec(scope, columns)
    _check(_call("tr_kernel_check", scope._handle, C.byref(s)), what="kernel")
    w, D, rp = _weight(scope, weight), int(degree or 0), _repr(repr)
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = _call("tr_kernel_build", scope._handle, C.byref(s), w, D, rp, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return FeatureState(_made(h, error, refusal, "kernel features"), scope, columns, kernel, "cpu", "native CPU")
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    gpu = _gpu()
    dev, adopt, cap = _Need(), _Need(), C.c_uint64()
    _check(_call("tr_kernel_need", scope._handle, C.byref(s), w, D, rp, 1, C.byref(dev), C.byref(cap)), what="kernel")
    _check(_call("tr_kernel_need", scope._handle, C.byref(s), w, D, rp, 2, C.byref(adopt), None), what="kernel")
    both = _Need()
    for i in range(len(RESOURCES)):
        both.need[i] = max(dev.need[i], adopt.need[i])
    mt._precheck(budget, both, "kernel features on the device")
    if kernel.kind == "delta":
        tspec = mt._count_spec(scope, columns, weight, repr, None, "auto")
    else:
        order = max(D or int(s.degree), 1)            # as tr_kernel_build: order >= 1 reads the kernel's columns
        tspec = mt._moment_spec(scope, ["1"] + columns, order, weight, "listwise", repr, "auto")
    st, keep = mt._alloc_states(scope, tspec, int(cap.value))
    G = scope.groups
    W = np.zeros(2 * G, np.int64)
    Q = np.zeros(WORDS * G, np.uint64)
    Dg = np.zeros(WORDS * G, np.uint64)
    view = scope.view()
    gpu.run("tr_mgpu_kernel_build", C.byref(view), C.byref(s), w, D, rp, C.byref(_budget(budget)), C.byref(st),
            W.ctypes.data, Q.ctypes.data, Dg.ctypes.data, C.byref(refusal), refusal=refusal,
            what="kernel features on the device")
    h = _call("tr_kernel_from_states", scope._handle, C.byref(s), w, D, rp, C.byref(st), W.ctypes.data, Q.ctypes.data,
              Dg.ctypes.data, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    d = gpu.device()
    return FeatureState(_made(h, error, refusal, "kernel feature states"), scope, columns, kernel,
                        "cuda-emulated" if d["emulated"] else "cuda", d["name"])


class FeatureState:
    """An immutable feature state (native): the TENSOR-001 tensor and the per-group self-pair terms"""

    def __init__(self, handle, scope, columns, kernel, backend, device):
        self._handle, self.scope, self.columns, self.kernel = handle, scope, list(columns), kernel
        self.backend, self.device = backend, device
        self._evals, self._uploads = {}, {}

    def _open(self):
        if not self._handle:
            raise ValueError("the feature state is closed")
        return self._handle

    @property
    def info(self):
        i = _Info()
        _check(_call("tr_kernel_get_info", self._open(), C.byref(i)), what="feature state")
        kind = {v: k for k, v in KINDS.items()}[i.kind]
        return dict(kind=kind, degree=int(i.degree), columns=list(self.columns), order=int(i.order),
                    repr=mt._REPR_NAMES[i.repr], groups=int(i.ngroups), present=int(i.present), rows=int(i.rows),
                    used=int(i.used), cells=int(i.cells), nnz=int(i.nnz), bytes=int(i.bytes),
                    weighted=bool(i.weighted), backend=self.backend, device=self.device)

    @property
    def tensor(self):
        """an independent copy of the feature tensor (TENSOR-001)"""
        error = C.c_int32()
        h = _call("tr_tensor_clone", _call("tr_kernel_get_tensor", self._open()), C.byref(error))
        _check(error.value, what="feature tensor")
        return mt.Tensor._wrap(h, self.scope, self.backend, self.device)

    def group(self, g):
        """{"present", "rows", "used", "weight" (sum w), "squares" (sum w^2), "self" (sum w^2 k(x, x))}"""
        W, Q, Dg = mt._I128(), _Wide(), _Wide()
        _check(_call("tr_kernel_get_group", self._open(), int(g), C.byref(W), C.byref(Q), C.byref(Dg)),
               what="group (id out of range)")
        t = self.tensor.group(g)
        return dict(t, weight=mt._int(W.lo, W.hi), squares=_wide(Q), self=_wide(Dg))

    def _eval(self, kernel, budget=None):
        key = None if kernel is None else bytes(kernel.spec(self.scope, self.columns))
        if key not in self._evals:
            spec = None if kernel is None else kernel.spec(self.scope, self.columns)
            error, refusal = C.c_int32(), _Refusal()
            h = _call("tr_kernel_eval_make", self._open(), None if spec is None else C.byref(spec),
                      C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            self._evals[key] = (_made(h, error, refusal, "kernel evaluation (a kernel the state retains)"), spec)
        return key, self._evals[key][0]

    def _upload(self, key, ev, budget=None):
        if key not in self._uploads:
            gpu = _gpu()
            view = _View()
            _check(_call("tr_kernel_eval_get_view", ev, C.byref(view)), what="kernel view")
            nnz = self.info["nnz"] if view.repr == mt.REPRS["sparse"] else 0
            refusal, error = _Refusal(), C.c_int()
            h = gpu.lib().tr_mgpu_kernel_upload(C.byref(view), nnz, C.byref(_budget(budget)), C.byref(refusal),
                                                C.byref(error))
            gpu.check(error.value, refusal, "feature state upload")
            if not h:
                raise MemoryError("feature state upload failed")
            self._uploads[key] = h
        return self._uploads[key]

    def _points(self, points, n):
        if points is None:
            return None, None, 0
        pts = [tuple(p) if isinstance(p, (list, tuple)) else (p,) for p in points]
        for p in pts:
            if len(p) != n:
                raise ValueError(f"a point has one value per kernel column ({n})")
        vals = np.zeros(max(len(pts) * n, 1), np.int64)
        valid = np.ones(max(len(pts) * n, 1), np.uint8)
        nulls = False
        for i, p in enumerate(pts):
            for j, x in enumerate(p):
                if x is None:
                    valid[i * n + j], nulls = 0, True
                else:
                    vals[i * n + j] = _i64(x, "a point coordinate")
        return vals, (valid if nulls else None), len(pts)

    def query(self, requests, *, kernel=None, points=None, backend="cpu", budget=None, raw=False):
        """answers of request dicts (op: sum, mean, mmd2, witness; a, b: groups; self_pairs, ordered, unbiased; point:
        an index into points): {"status", "value", "pairs", "num", "den", "empty"}; raw=False raises on a request's
        error (ValueError, UnsupportedError, OverflowError). budget: the batch's (output; CUDA: bytes), and on a
        kernel's first use its evaluation's (degree, support = items, bytes; CUDA: the upload's bytes)"""
        requests = list(requests)
        key, ev = self._eval(kernel, budget)
        ncols = len(kernel.columns) if kernel is not None and kernel.columns is not None else len(self.columns)
        arr = (_Request * max(len(requests), 1))()
        for i, q in enumerate(requests):
            r = arr[i]
            if q["op"] not in OPS:
                raise ValueError(f"op is one of {', '.join(OPS)}")
            b = q.get("b")
            r.op, r.a, r.b, r.point = OPS[q["op"]], int(q["a"]), int(q["a"] if b is None else b), int(q.get("point", 0))
            r.flags = ((SELF if q.get("self_pairs") else 0) | (UNORDERED if q.get("ordered", True) is False else 0) |
                       (UNBIASED if q.get("unbiased") else 0))
        vals, valid, npts = self._points(points, ncols)
        out = (_Result * max(len(requests), 1))()
        refusal = _Refusal()
        args = (arr, len(requests), None if vals is None else vals.ctypes.data,
                None if valid is None else valid.ctypes.data, npts, out, C.byref(_budget(budget)), C.byref(refusal))
        if backend == "cpu":
            _check(_call("tr_kernel_query", ev, *args), refusal, "kernel requests")
        elif backend == "cuda":
            gpu = _gpu()
            gpu.run("tr_mgpu_kernel_query", self._upload(key, ev, budget), *args, refusal=refusal,
                    what="kernel requests")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        answers = []
        for i in range(len(requests)):
            r = out[i]
            num, den, pairs = _wide(r.num), _wide(r.den), _wide(r.pairs)
            if r.status:
                if not raw:
                    _check(r.status, what=f"request {i}")
                answers.append(dict(status=int(r.status), value=None, pairs=None, num=None, den=None, empty=False))
                continue
            op = requests[i]["op"]
            if r.empty:
                value = None
            elif op == "sum":
                value = num
            else:
                value = Fraction(num, den)
            answers.append(dict(status=0, value=value, pairs=pairs if op in ("sum", "mean") else None, num=num, den=den,
                                empty=bool(r.empty)))
        return answers

    def sum(self, a, b=None, *, self_pairs=False, ordered=True, kernel=None, backend="cpu"):
        """the sum over the selected pairs of w_i v_l k(x_i, y_l) and the weighted pair count: {"value", "pairs"}"""
        r = self.query([dict(op="sum", a=a, b=b, self_pairs=self_pairs, ordered=ordered)], kernel=kernel,
                       backend=backend)[0]
        return dict(value=r["value"], pairs=r["pairs"])

    def mean(self, a, b=None, *, self_pairs=False, ordered=True, kernel=None, backend="cpu"):
        """the mean kernel value over the selected pairs, <mu_a, mu_b> (None when there is no pair weight)"""
        return self.query([dict(op="mean", a=a, b=b, self_pairs=self_pairs, ordered=ordered)], kernel=kernel,
                          backend=backend)[0]["value"]

    def mmd2(self, a, b, *, unbiased=True, kernel=None, backend="cpu"):
        """the squared maximum mean discrepancy of groups a and b (unbiased: self-pairs excluded), exact"""
        return self.query([dict(op="mmd2", a=a, b=b, unbiased=unbiased)], kernel=kernel, backend=backend)[0]["value"]

    def witness(self, a, points, *, kernel=None, backend="cpu"):
        """the kernel mean embedding of group a at each point: sum_i w_i k(x_i, z) / W_a (None when W_a = 0)"""
        pts = list(points)
        reqs = [dict(op="witness", a=a, point=i) for i in range(len(pts))]
        return [r["value"] for r in self.query(reqs, kernel=kernel, points=pts, backend=backend)]

    def close(self):
        if getattr(self, "_uploads", None):
            gpu = _gpu()
            for h in self._uploads.values():
                gpu.lib().tr_mgpu_kernel_free(h)
            self._uploads = {}
        for h, _ in getattr(self, "_evals", {}).values():
            _call("tr_kernel_eval_free", h)
        self._evals = {}
        if getattr(self, "_handle", None):
            _call("tr_kernel_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "FeatureState(closed)"
        i = self.info
        return (f"FeatureState({i['kind']}, degree {i['degree']}, columns={i['columns']}, groups={i['groups']}, "
                f"{self.backend})")
