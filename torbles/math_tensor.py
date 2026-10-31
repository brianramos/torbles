# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""TENSOR-001 / TENSOR-002: exact moment and count tensors (dense, sparse, factored; per group) and costed
contraction planning of bounded tensor networks, natively on the CPU or with the CUDA kernels (torbles/math_tensor.c,
math_tensor_gpu.cu; every contract in math_tensor.h).

    from torbles import math_bank as mbk, math_tensor as mt
    scope = mbk.Scope.from_table(db["f"], ["x", "y", "z", "c"], group_by=["k"], rows=mask)
    T = mt.moments(scope, ["x", "y", "z"], order=3, policy="listwise", repr="dense")    # backend="cuda" too
    T = mt.moments(scope, [["1", "x"], ["x", "y"]])          # one column list per axis ("1": the constant)
    C = mt.counts(scope, ["c", "x"], repr="sparse", weight=None, domain=None)           # NULL is a category
    T.entry(g, (0, 1, 2)); T.array(g); T.entries(g); T.terms(g); T.group(g); T.info; T.axes
    T.to_sparse(); T.to_dense(); T.to_factored(); T.marginal([0, 2]); T.slice([1, (0, 2), None]); T.permute([2, 0, 1])
    mt.dense(values); mt.sparse(dims, entries); mt.factors(dims, terms); mt.outer(A, B, repr="factored")
    net = mt.network([(A, "ij"), (B, "jk"), (C, "kl")], "il")
    plan = net.plan(devices=("cpu", "cuda"), max_orders=64, budget=Budget(tensor=1 << 20))   # evidence of every candidate
    out, run = net.contract(plan, candidate=None)            # the chosen candidate, or any admissible one by index

Entries are exact Python ints (|v| <= 2^127 - 1). Budgets are checked natively before anything is allocated
(AlgebraBudgetError with .resource, .need, .limit); a refused plan raises with .plan (the whole evidence) attached.
Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_registry as reg
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["Tensor", "Network", "Plan", "moments", "counts", "need", "dense", "sparse", "factors", "outer", "network",
           "default_rates", "header_bytes", "MAX_ORDER", "MAX_FEATURES", "NET_MAX", "MAX_LABELS", "MAX_CANDIDATES"]

MAX_ORDER, MAX_FEATURES, NET_MAX, MAX_LABELS, MAX_CANDIDATES = 8, 64, 8, 16, 256
PAIR_BYTES, KEY_BYTES = 48, 16
KINDS = {"generic": 0, "moment": 1, "count": 2}
REPRS = {"dense": 1, "sparse": 2, "factored": 3}
POLICIES = {"listwise": 1, "sql": 2}
STRATEGIES = {"auto": 0, "count": 1, "sort": 2}
_STRATEGY_NAMES = {0: "auto", 1: "count", 2: "sort", 3: "direct", 4: "derived", 5: "device", 6: "arrays", 7: "contracted"}
_AXIS_NAMES = {0: "index", 1: "feature", 2: "category", 3: "group"}
KERNELS = {"dense": 1, "sparse": 2}
_KIND_NAMES = {v: k for k, v in KINDS.items()}
_REPR_NAMES = {v: k for k, v in REPRS.items()}
_POLICY_NAMES = {0: None, 1: "listwise", 2: "sql"}
_U64 = (1 << 64) - 1
_I128_MAX = (1 << 127) - 1
P = C.POINTER
COMPONENTS = len(reg.COST_COMPONENTS)


class _Spec(C.Structure):
    _native_ = "tr_tensor_spec"
    _fields_ = [(n, C.c_int32) for n in ("kind", "order", "repr", "policy", "weight", "strategy", "domain",
                                         "reserved")] + [
        ("ncols", C.c_int32 * MAX_ORDER), ("cols", (C.c_int8 * MAX_FEATURES) * MAX_ORDER),
        ("nullslot", C.c_int32 * MAX_ORDER), ("lo", C.c_int64 * MAX_ORDER), ("hi", C.c_int64 * MAX_ORDER)]


class _Axis(C.Structure):
    _native_ = "tr_tensor_axis"
    _fields_ = [("kind", C.c_int32), ("ncols", C.c_int32), ("col", C.c_int32), ("nullslot", C.c_int32),
                ("lo", C.c_int64), ("span", C.c_uint64), ("dim", C.c_uint64), ("scale", C.c_int32),
                ("reserved", C.c_int32), ("cols", C.c_int8 * MAX_FEATURES)]


class _I128(C.Structure):
    _native_ = "tr_math_i128"
    _fields_ = [("lo", C.c_int64), ("hi", C.c_int64)]


class _Info(C.Structure):
    _native_ = "tr_tensor_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "present", "rows", "used", "cells", "nnz", "nterms", "bytes")] + [
        ("dims", C.c_uint64 * MAX_ORDER), ("fdim", C.c_uint64), ("bound", _I128), ("digest", C.c_uint64 * 2)] + [
        (n, C.c_int32) for n in ("kind", "repr", "order", "policy", "has_counts", "weighted", "scoped", "strategy",
                                 "symmetric", "weight", "wscale", "reserved")]


class _States(C.Structure):
    _native_ = "tr_tensor_states"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "cells", "capacity", "nnz")] + [
        (n, C.c_int32) for n in ("order", "repr", "has_counts", "reserved")] + [
        ("dims", C.c_uint64 * MAX_ORDER)] + [(n, C.c_void_p) for n in ("rows", "used", "offset", "coord", "value",
                                                                      "count")] + [
        ("nterms", C.c_uint64), ("fdim", C.c_uint64), ("weight", C.c_void_p), ("factor", C.c_void_p)]


class _NetSpec(C.Structure):
    _native_ = "tr_tensor_net_spec"
    _fields_ = [("ntensors", C.c_int32), ("noutput", C.c_int32), ("nlabels", C.c_int32 * NET_MAX),
                ("labels", (C.c_int32 * (MAX_ORDER + 1)) * NET_MAX), ("output", C.c_int32 * MAX_LABELS)]


class _Options(C.Structure):
    _native_ = "tr_tensor_plan_options"
    _fields_ = [(n, C.c_int32) for n in ("max_orders", "devices", "kernels", "force", "custom_rates", "reserved")] + [
        ("rates", (C.c_double * COMPONENTS) * 2)]


class _Step(C.Structure):
    _native_ = "tr_tensor_step_info"
    _fields_ = [(n, C.c_int32) for n in ("a", "b", "nout", "nsum")] + [
        ("out", C.c_int8 * MAX_LABELS), ("sum", C.c_int8 * MAX_LABELS)] + [
        (n, C.c_uint64) for n in ("cells", "terms", "nnz", "pairs", "bytes", "scratch", "live")] + [
        ("free_a", C.c_int32), ("free_b", C.c_int32), ("bound", _I128)]


class _Candidate(C.Structure):
    _native_ = "tr_tensor_candidate"
    _fields_ = [("reg", reg._Candidate)] + [(n, C.c_int32) for n in ("kernel", "device", "nsteps", "order",
                                                                     "converted", "reserved")] + [
        (n, C.c_uint64) for n in ("peak_bytes", "output_bytes", "output_cells", "output_nnz", "input_bytes")] + [
        ("bound", _I128), ("step", _Step * NET_MAX)]


class _PlanInfo(C.Structure):
    _native_ = "tr_tensor_plan_info"
    _fields_ = [(n, C.c_int32) for n in ("ncandidates", "chosen", "code", "reason", "complete", "norders", "forced",
                                         "devices")] + [
        ("total_orders", C.c_uint64), ("bytes", C.c_uint64), ("refusal", _Refusal), ("ntensors", C.c_int32),
        ("nlabels", C.c_int32)]


class _Exec(C.Structure):
    _native_ = "tr_tensor_exec"
    _fields_ = [(n, C.c_int32) for n in ("candidate", "kernel", "device", "nsteps")] + [
        ("nnz", C.c_uint64 * NET_MAX), ("bytes", C.c_uint64 * NET_MAX), ("pairs", C.c_uint64 * NET_MAX)] + [
        (n, C.c_uint64) for n in ("peak_bytes", "output_nnz", "output_bytes", "converted_bytes")]


class _DenseStep(C.Structure):
    _native_ = "tr_tensor_dense_step"
    _fields_ = [(n, C.c_int32) for n in ("a", "b", "nout", "nsum")] + [("cells", C.c_uint64), ("terms", C.c_uint64)] + [
        (n, C.c_uint64 * MAX_LABELS) for n in ("dim_out", "dim_sum", "sa_out", "sa_sum", "sb_out", "sb_sum")] + [
        ("bound", _I128), ("free_a", C.c_int32), ("free_b", C.c_int32), ("a_cells", C.c_uint64), ("b_cells", C.c_uint64)]


class _Program(C.Structure):
    _native_ = "tr_tensor_program"
    _fields_ = [(n, C.c_int32) for n in ("ninputs", "nsteps", "noutput", "reserved")] + [
        ("input_cells", C.c_uint64 * NET_MAX), ("input_bound", _I128 * NET_MAX), ("output_cells", C.c_uint64),
        ("output_dims", C.c_uint64 * MAX_LABELS), ("step", _DenseStep * NET_MAX)]


class _DenseView(C.Structure):
    _native_ = "tr_tensor_dense_view"
    _fields_ = [("value", C.c_void_p), ("cells", C.c_uint64), ("order", C.c_int32), ("reserved", C.c_int32),
                ("dims", C.c_uint64 * MAX_ORDER)]


for _s, _n in ((_Spec, 736), (_Axis, 112), (_Info, 216), (_States, 192), (_NetSpec, 392), (_Options, 216),
               (_Step, 128), (_Candidate, 1480), (_PlanInfo, 80), (_Exec, 240), (_DenseStep, 840), (_Program, 7064),
               (_DenseView, 88)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u32, u64, e, R, B = C.c_void_p, C.c_int32, C.c_uint32, C.c_uint64, P(C.c_int32), P(_Refusal), P(_Budget)
        sig = {
            "tr_tensor_default_rates": (None, [vp]),
            "tr_tensor_header_bytes": (u64, []),
            "tr_tensor_init": (vp, [vp, P(_Spec), B, R, e]),
            "tr_tensor_accumulate": (i32, [vp, u64, u64]),
            "tr_tensor_merge_builder": (i32, [vp, vp]),
            "tr_tensor_finish": (vp, [vp, B, R, e]),
            "tr_tensor_builder_free": (None, [vp]),
            "tr_tensor_build": (vp, [vp, P(_Spec), B, R, e]),
            "tr_tensor_need": (i32, [vp, P(_Spec), i32, P(_Need), P(u64)]),
            "tr_tensor_from_states": (vp, [vp, P(_Spec), P(_States), B, R, e]),
            "tr_tensor_from_dense": (vp, [i32, vp, u64, vp, vp, B, R, e]),
            "tr_tensor_from_sparse": (vp, [i32, vp, u64, u64, vp, vp, vp, vp, B, R, e]),
            "tr_tensor_from_factors": (vp, [i32, vp, u64, u64, vp, vp, vp, B, R, e]),
            "tr_tensor_convert": (vp, [vp, i32, B, R, e]),
            "tr_tensor_marginal": (vp, [vp, u32, B, R, e]),
            "tr_tensor_slice": (vp, [vp, u32, vp, vp, B, R, e]),
            "tr_tensor_permute": (vp, [vp, vp, B, R, e]),
            "tr_tensor_outer": (vp, [vp, vp, i32, B, R, e]),
            "tr_tensor_ungroup": (vp, [vp, B, R, e]),
            "tr_tensor_values": (vp, [vp, B, R, e]),
            "tr_tensor_clone": (vp, [vp, e]),
            "tr_tensor_get": (i32, [vp, u64, vp, P(_I128), P(u64)]),
            "tr_tensor_equal": (i32, [vp, vp]),
            "tr_tensor_same": (i32, [vp, vp]),
            "tr_tensor_get_info": (i32, [vp, P(_Info)]),
            "tr_tensor_get_axis": (i32, [vp, i32, P(_Axis)]),
            "tr_tensor_get_states": (i32, [vp, P(_States)]),
            "tr_tensor_free": (None, [vp]),
            "tr_tensor_net_create": (vp, [vp, P(_NetSpec), B, R, e]),
            "tr_tensor_net_free": (None, [vp]),
            "tr_tensor_net_plan": (i32, [vp, P(_Options), B, P(vp), R]),
            "tr_tensor_plan_get_info": (i32, [vp, P(_PlanInfo)]),
            "tr_tensor_plan_get": (i32, [vp, i32, P(_Candidate)]),
            "tr_tensor_plan_free": (None, [vp]),
            "tr_tensor_net_execute": (vp, [vp, vp, i32, B, P(_Exec), R, e]),
            "tr_tensor_net_program": (i32, [vp, vp, i32, P(_Program)]),
            "tr_tensor_net_input_dense": (vp, [vp, i32, B, R, e]),
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
    math_gpu.bind("tr_mgpu_tensor_build", C.c_int, [P(mbk._View), P(_Spec), P(_Budget), P(_States), P(_Refusal)])
    math_gpu.bind("tr_mgpu_tensor_contract", C.c_int, [P(_Program), C.c_void_p, P(_Budget), C.c_void_p, P(_Exec),
                                                      P(_Refusal)])
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
    refusal = _Refusal()
    _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw_need), C.byref(refusal)), refusal, what)


def _int(lo, hi):
    return (int(hi) << 64) | (int(lo) & _U64)


def _ints(words):
    """exact Python ints from an int64 array of (lo, hi) pairs"""
    w = np.asarray(words, dtype=np.int64).reshape(-1, 2)
    return [_int(lo, hi) for lo, hi in w.tolist()]


def _exact(v, what="a value"):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{what} must be an exact integer (got {type(v).__name__})")
    v = int(v)
    if not -_I128_MAX <= v <= _I128_MAX:
        raise OverflowError(f"{what} does not fit 128 bits (|v| <= 2^127 - 1)")
    return v


def _words(values, what="values"):
    """an int64 array of (lo, hi) words of exact integers"""
    vals = [_exact(v, what) for v in values]
    out = np.zeros(max(2 * len(vals), 2), dtype=np.int64)
    for i, v in enumerate(vals):
        u = v & ((1 << 128) - 1)
        lo, hi = u & _U64, u >> 64
        out[2 * i] = lo - (1 << 64) if lo >= 1 << 63 else lo
        out[2 * i + 1] = hi - (1 << 64) if hi >= 1 << 63 else hi
    return out


def _ptr(a):
    return None if a is None else a.ctypes.data


def _u64s(values):
    a = np.ascontiguousarray([int(v) for v in values] or [0], dtype=np.uint64)
    return a


def header_bytes():
    """the fixed bytes of a tensor object (included in every tensor's bytes)"""
    return int(_call("tr_tensor_header_bytes"))


def default_rates():
    """the plan's default rates {"cpu": {component: rate}, "cuda": {...}} (relative work units, NOT time)"""
    r = ((C.c_double * COMPONENTS) * 2)()
    _call("tr_tensor_default_rates", C.byref(r))
    return {dev: dict(zip(reg.COST_COMPONENTS, (float(x) for x in r[d]))) for d, dev in enumerate(("cpu", "cuda"))}


# ---------------------------------------------------------------------- tensors
class Tensor:
    """An immutable exact tensor state (native): read it with entry, array, entries, terms, group, info."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_tensor.moments / counts / dense / sparse / factors, or an operation of a Tensor")

    @classmethod
    def _wrap(cls, handle, scope=None, backend="cpu", device="native CPU"):
        self = object.__new__(cls)
        self._handle, self.scope, self.backend, self.device = handle, scope, backend, device
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the tensor is closed")
        return self._handle

    def _raw_info(self):
        out = _Info()
        _check(_call("tr_tensor_get_info", self._open(), C.byref(out)), what="tensor")
        return out

    @property
    def info(self):
        i = self._raw_info()
        m = int(i.order)
        return dict(kind=_KIND_NAMES[i.kind], repr=_REPR_NAMES[i.repr], order=m, dims=tuple(int(d) for d in i.dims[:m]),
                    groups=int(i.ngroups), present=int(i.present), rows=int(i.rows), used=int(i.used),
                    cells=int(i.cells), nnz=int(i.nnz), nterms=int(i.nterms), fdim=int(i.fdim), bytes=int(i.bytes),
                    bound=_int(i.bound.lo, i.bound.hi), policy=_POLICY_NAMES[i.policy], counts=bool(i.has_counts),
                    weighted=bool(i.weighted), scoped=bool(i.scoped), symmetric=bool(i.symmetric),
                    strategy=_STRATEGY_NAMES.get(i.strategy), backend=self.backend, device=self.device)

    kind = property(lambda self: self.info["kind"])
    repr = property(lambda self: self.info["repr"])
    order = property(lambda self: int(self._raw_info().order))
    dims = property(lambda self: self.info["dims"])
    groups = property(lambda self: int(self._raw_info().ngroups))
    bytes = property(lambda self: int(self._raw_info().bytes))
    has_counts = property(lambda self: bool(self._raw_info().has_counts))

    @property
    def axes(self):
        """per axis: kind (index, feature, category, group), dim; feature: columns (names; "1" the constant);
        category: column, lo, span, null (the index of the NULL slot or None), scale"""
        out = []
        names = self.scope.names if self.scope is not None else None
        for k in range(self.order):
            a = _Axis()
            _check(_call("tr_tensor_get_axis", self._open(), k, C.byref(a)), what="tensor axis")
            d = dict(kind=_AXIS_NAMES[a.kind], dim=int(a.dim))
            if a.kind == 1:
                d["columns"] = [("1" if c < 0 else (names[c] if names else int(c))) for c in a.cols[:a.ncols]]
            elif a.kind == 2:
                d.update(column=names[a.col] if names else int(a.col), lo=int(a.lo), span=int(a.span),
                         null=int(a.span) if a.nullslot else None, scale=int(a.scale))
            out.append(d)
        return out

    def _states(self):
        st = _States()
        _check(_call("tr_tensor_get_states", self._open(), C.byref(st)), what="tensor states")
        return st

    def group(self, g):
        """{"present", "rows", "used"} of group g (bag multiplicity and contributing rows; unscoped: zeros)"""
        st = self._states()
        g = self._group_id(g, st)
        rows = int(C.c_uint64.from_address(st.rows + 8 * g).value)
        used = int(C.c_uint64.from_address(st.used + 8 * g).value)
        return dict(present=rows > 0, rows=rows, used=used)

    def _group_id(self, g, st=None):
        st = st or self._states()
        g = int(g)
        if not 0 <= g < int(st.ngroups):
            raise ValueError("group id out of range")
        return g

    def entry(self, g, coords):
        """the exact entry T_g[coords] in any representation (with counts: (value, count))"""
        m = self.order
        coords = tuple(int(c) for c in coords)
        if len(coords) != m:
            raise ValueError(f"{m} coordinates")
        c = (C.c_uint32 * max(m, 1))(*coords)
        v, cnt = _I128(), C.c_uint64()
        _check(_call("tr_tensor_get", self._open(), self._group_id(g), c, C.byref(v), C.byref(cnt)),
               what="entry (coordinates within the dims)")
        value = _int(v.lo, v.hi)
        return (value, int(cnt.value)) if self.has_counts else value

    def array(self, g=0, *, budget=None):
        """the dense image of group g as a numpy object array of Python ints (a non-dense tensor is converted natively
        first: asked for explicitly, and budgeted); with counts: (values, counts)"""
        t = self if self.repr == "dense" else self.to_dense(budget=budget)
        st = t._states()
        g = t._group_id(g, st)
        cells, dims = int(st.cells), tuple(int(d) for d in st.dims[:st.order])
        if cells:
            words = np.ctypeslib.as_array((C.c_int64 * (2 * cells)).from_address(st.value + 16 * g * cells))
            vals = np.array(_ints(words), dtype=object).reshape(dims)
        else:
            vals = np.zeros(dims, dtype=object)
        if not st.has_counts:
            return vals
        cnt = np.ctypeslib.as_array((C.c_uint64 * max(cells, 1)).from_address(st.count + 8 * g * cells))[:cells]
        return vals, np.array([int(x) for x in cnt], dtype=object).reshape(dims)

    def entries(self, g=0, *, budget=None):
        """the stored entries of group g in increasing multi-index order: [(coords, value)] or with counts
        [(coords, value, count)] (a non-sparse tensor is converted natively first)"""
        t = self if self.repr == "sparse" else self.to_sparse(budget=budget)
        st = t._states()
        g = t._group_id(g, st)
        m = int(st.order)
        off = np.ctypeslib.as_array((C.c_uint64 * (int(st.ngroups) + 1)).from_address(st.offset))
        b, e = int(off[g]), int(off[g + 1])
        if e == b:
            return []
        n = e - b
        coords = np.ctypeslib.as_array((C.c_uint32 * max(n * m, 1)).from_address(st.coord + 4 * b * m))[:n * m]
        coords = coords.reshape(n, m) if m else np.zeros((n, 0), np.uint32)
        vals = _ints(np.ctypeslib.as_array((C.c_int64 * (2 * n)).from_address(st.value + 16 * b)))
        if st.has_counts:
            cnt = np.ctypeslib.as_array((C.c_uint64 * n).from_address(st.count + 8 * b)).tolist()
            return [(tuple(int(x) for x in coords[i]), vals[i], int(cnt[i])) for i in range(n)]
        return [(tuple(int(x) for x in coords[i]), vals[i]) for i in range(n)]

    def terms(self, g=0):
        """the declared rank-one terms of group g of a factored tensor: [(weight, [vector per axis])]"""
        if self.repr != "factored":
            raise ValueError("terms() reads a factored tensor (to_factored() converts)")
        st = self._states()
        g = self._group_id(g, st)
        off = np.ctypeslib.as_array((C.c_uint64 * (int(st.ngroups) + 1)).from_address(st.offset))
        b, e, fdim = int(off[g]), int(off[g + 1]), int(st.fdim)
        dims = [int(d) for d in st.dims[:st.order]]
        out = []
        if e == b:
            return out
        w = _ints(np.ctypeslib.as_array((C.c_int64 * (2 * (e - b))).from_address(st.weight + 16 * b)))
        f = np.ctypeslib.as_array((C.c_int64 * max((e - b) * fdim, 1)).from_address(st.factor + 8 * b * fdim))
        for j in range(e - b):
            row, at, vecs = f[j * fdim:(j + 1) * fdim].tolist(), 0, []
            for d in dims:
                vecs.append([int(x) for x in row[at:at + d]])
                at += d
            out.append((w[j], vecs))
        return out

    # ------------------------------------------------------------------ operations (each a new native tensor)
    def _derive(self, name, *args, what=None, budget=None):
        error, refusal = C.c_int32(), _Refusal()
        h = _call(name, self._open(), *args, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Tensor._wrap(_made(h, error, refusal, what or name), self.scope, self.backend, self.device)

    def convert(self, repr, *, budget=None):
        if repr not in REPRS:
            raise ValueError(f"repr is one of {', '.join(REPRS)}")
        return self._derive("tr_tensor_convert", REPRS[repr], what=f"conversion to {repr}", budget=budget)

    def to_dense(self, *, budget=None):
        return self.convert("dense", budget=budget)

    def to_sparse(self, *, budget=None):
        return self.convert("sparse", budget=budget)

    def to_factored(self, *, budget=None):
        return self.convert("factored", budget=budget)

    def marginal(self, keep, *, budget=None):
        """sum out every axis not in keep (axis indices, kept in their order)"""
        mask = 0
        for k in keep:
            if not 0 <= int(k) < self.order:
                raise ValueError("axis index out of range")
            mask |= 1 << int(k)
        return self._derive("tr_tensor_marginal", mask, what="marginal", budget=budget)

    def slice(self, spec, *, budget=None):
        """one item per axis: an int (fix that index; the axis is removed), (lo, hi) (keep the range) or None (all)"""
        m, dims = self.order, self.dims
        spec = list(spec)
        if len(spec) != m:
            raise ValueError(f"one slice item per axis ({m})")
        lo, hi, fix = np.zeros(max(m, 1), np.uint64), np.zeros(max(m, 1), np.uint64), 0
        for k, s in enumerate(spec):
            if s is None:
                lo[k], hi[k] = 0, dims[k]
            elif isinstance(s, (int, np.integer)) and not isinstance(s, bool):
                if not 0 <= int(s) < dims[k]:
                    raise ValueError("fixed index out of range")
                fix |= 1 << k
                lo[k] = int(s)
            else:
                a, b = (int(x) for x in s)
                if not 0 <= a <= b <= dims[k]:
                    raise ValueError("slice range out of the axis")
                lo[k], hi[k] = a, b
        return self._derive("tr_tensor_slice", fix, lo.ctypes.data, hi.ctypes.data, what="slice", budget=budget)

    def permute(self, perm, *, budget=None):
        p = np.ascontiguousarray([int(x) for x in perm] or [0], dtype=np.int32)
        if len(perm) != self.order:
            raise ValueError("a permutation of the axes")
        return self._derive("tr_tensor_permute", p.ctypes.data, what="permutation", budget=budget)

    def ungroup(self, *, budget=None):
        """the G groups as a leading group axis of a one-group tensor"""
        return self._derive("tr_tensor_ungroup", what="ungroup", budget=budget)

    def values(self, *, budget=None):
        """the same entries without per-entry counts (GENERIC): what a contraction accepts"""
        return self._derive("tr_tensor_values", what="values", budget=budget)

    def equal(self, other):
        """the same state in the same representation (native)"""
        return bool(_check(_call("tr_tensor_equal", self._open(), other._open()), what="tensor comparison"))

    def same(self, other):
        """the same tensor in any representation: kind, order, dims, axes, policy, groups, rows, used, entries and
        counts (native)"""
        return bool(_check(_call("tr_tensor_same", self._open(), other._open()), what="tensor comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_tensor_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Tensor(closed)"
        i = self.info
        return f"Tensor({i['kind']}, {i['repr']}, dims={i['dims']}, groups={i['groups']}, nnz={i['nnz']}, {self.backend})"


def _column(scope, name):
    if name in (None, "1", 1):
        return -1
    if name not in scope.names:
        raise ValueError(f"{name!r} is not a scope column")
    return scope.names.index(name)


def _moment_spec(scope, columns, order, weight, policy, repr, strategy):
    columns = list(columns)
    if columns and all(isinstance(c, (list, tuple)) for c in columns):
        axes = [list(c) for c in columns]
        if order is not None and int(order) != len(axes):
            raise ValueError("order differs from the number of axis column lists")
    else:
        if order is None:
            raise ValueError("give order= with one column list (or one list per axis)")
        axes = [columns] * int(order)
    if len(axes) > MAX_ORDER:
        from .math_errors import UnsupportedError
        raise UnsupportedError(f"order {len(axes)}: at most {MAX_ORDER}")
    spec = _Spec()
    spec.kind, spec.order, spec.repr = KINDS["moment"], len(axes), _repr(repr)
    if policy not in POLICIES:
        raise ValueError(f"policy is one of {', '.join(POLICIES)}")
    spec.policy, spec.weight = POLICIES[policy], -1 if weight is None else _column(scope, weight)
    if weight is not None and spec.weight < 0:
        raise ValueError("the weight is a scope column")
    spec.strategy = _strategy(strategy, repr)
    for k, cols in enumerate(axes):
        if not 1 <= len(cols) <= MAX_FEATURES:
            from .math_errors import UnsupportedError
            raise UnsupportedError(f"axis {k}: 1..{MAX_FEATURES} columns")
        spec.ncols[k] = len(cols)
        for i, c in enumerate(cols):
            spec.cols[k][i] = _column(scope, c)
    return spec


def _repr(repr):
    if repr not in REPRS:
        raise ValueError(f"repr is one of {', '.join(REPRS)}")
    return REPRS[repr]


def _strategy(strategy, repr):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy is one of {', '.join(STRATEGIES)}")
    return STRATEGIES[strategy] if repr == "sparse" else 0


def _count_spec(scope, columns, weight, repr, domain, strategy):
    columns = list(columns)
    if len(columns) > MAX_ORDER:
        from .math_errors import UnsupportedError
        raise UnsupportedError(f"order {len(columns)}: at most {MAX_ORDER}")
    spec = _Spec()
    spec.kind, spec.order, spec.repr, spec.policy = KINDS["count"], len(columns), _repr(repr), 0
    spec.weight = -1 if weight is None else _column(scope, weight)
    spec.strategy = _strategy(strategy, repr)
    for k, c in enumerate(columns):
        col = _column(scope, c)
        if col < 0:
            raise ValueError("a count axis reads a scope column")
        spec.ncols[k], spec.cols[k][0] = 1, col
    if domain is not None:
        domain = list(domain)
        if len(domain) != len(columns):
            raise ValueError("one (lo, hi, null) domain per axis")
        spec.domain = 1
        for k, (lo, hi, null) in enumerate(domain):
            spec.lo[k], spec.hi[k], spec.nullslot[k] = int(lo), int(hi), int(bool(null))
    return spec


def _build(scope, spec, backend, budget, what):
    if scope.source == "table":
        scope.require_fresh(what)
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = _call("tr_tensor_build", scope._handle, C.byref(spec), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Tensor._wrap(_made(h, error, refusal, what), scope)
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    gpu = _gpu()
    dev, adopt, cap = _Need(), _Need(), C.c_uint64()
    _check(_call("tr_tensor_need", scope._handle, C.byref(spec), 1, C.byref(dev), C.byref(cap)), what=what)
    _check(_call("tr_tensor_need", scope._handle, C.byref(spec), 2, C.byref(adopt), None), what=what)
    both = _Need()
    for i in range(len(RESOURCES)):
        both.need[i] = max(dev.need[i], adopt.need[i])
    _precheck(budget, both, what + " on the device")
    st, keep = _alloc_states(scope, spec, int(cap.value))
    view = scope.view()
    gpu.run("tr_mgpu_tensor_build", C.byref(view), C.byref(spec), C.byref(_budget(budget)), C.byref(st),
            C.byref(refusal), refusal=refusal, what=what + " on the device")
    h = _call("tr_tensor_from_states", scope._handle, C.byref(spec), C.byref(st), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    d = gpu.device()
    return Tensor._wrap(_made(h, error, refusal, what + " states"), scope, "cuda-emulated" if d["emulated"] else "cuda",
                        d["name"])


def _resolved_dims(scope, spec):
    """the dims the native plan resolves (MOMENT: the column counts; COUNT: the domain), read back from a tiny
    native dry run is not possible, so they are computed from the scope's column extremes exactly as math_tensor.h"""
    dims = []
    for k in range(spec.order):
        if spec.kind == KINDS["moment"]:
            dims.append(int(spec.ncols[k]))
            continue
        col = scope.names[spec.cols[k][0]]
        info = scope.column(col)
        if spec.domain:
            span = max(int(spec.hi[k]) - int(spec.lo[k]) + 1, 0)
            dims.append(span + int(spec.nullslot[k]))
        else:
            span = 0 if info["min"] is None else info["max"] - info["min"] + 1
            dims.append(span + int(_has_null(scope, spec.cols[k][0])))
    return dims


def _has_null(scope, col):
    v = scope.view()
    return bool(v.col[col].valid)


def _alloc_states(scope, spec, cap):
    G, m = scope.groups, int(spec.order)
    dims = _resolved_dims(scope, spec)
    cells = 1
    for d in dims:
        cells *= d
    counts = spec.kind == KINDS["moment"] and spec.policy == POLICIES["sql"]
    st = _States()
    st.ngroups, st.order, st.repr, st.has_counts = G, m, spec.repr, int(counts)
    for k, d in enumerate(dims):
        st.dims[k] = d
    rows, used = np.zeros(G, np.uint64), np.zeros(G, np.uint64)
    n = G * cells if spec.repr == REPRS["dense"] else cap
    value = np.zeros(max(2 * n, 2), np.int64)
    count = np.zeros(max(n, 1), np.uint64) if counts else None
    keep = [rows, used, value, count]
    st.capacity, st.cells = n, cells
    st.rows, st.used, st.value, st.count = rows.ctypes.data, used.ctypes.data, value.ctypes.data, _ptr(count)
    if spec.repr == REPRS["sparse"]:
        offset = np.zeros(G + 1, np.uint64)
        coord = np.zeros(max(n * m, 1), np.uint32)
        keep += [offset, coord]
        st.offset, st.coord = offset.ctypes.data, coord.ctypes.data
    return st, keep


def moments(scope, columns, order=None, *, weight=None, policy="listwise", repr="dense", strategy="auto",
            backend="cpu", budget=None):
    """T_g[i_0..i_{m-1}] = sum over the contributing rows of group g of w prod_k x[col_k[i_k]] (exact). columns: one
    list for every axis (with order=) or one list per axis; "1" is the constant 1. policy "listwise" (complete rows)
    or "sql" (per entry, with per-entry counts). repr "dense", "sparse" or "factored" (listwise: one rank-one term
    per row). backend "cuda": the CUDA kernels (Tensor.backend "cuda" or "cuda-emulated")."""
    spec = _moment_spec(scope, columns, order, weight, policy, repr, strategy)
    return _build(scope, spec, backend, budget, "moment tensor")


def counts(scope, columns, *, weight=None, repr="sparse", domain=None, strategy="auto", backend="cpu", budget=None):
    """joint counts (or weight sums) over categorical columns, NULL a category of its own; domain: None (the scope's
    [min, max] and a NULL slot iff a selected value is NULL) or one (lo, hi, null) per axis"""
    spec = _count_spec(scope, columns, weight, repr, domain, strategy)
    return _build(scope, spec, backend, budget, "count tensor")


def need(scope, columns, order=None, *, kind="moment", weight=None, policy="listwise", repr="dense", domain=None,
         strategy="auto", backend="cpu"):
    """the needs a build checks before allocating ({resource: need}); backend "cuda": the device copies and the
    adopted state, resource by resource the larger"""
    spec = (_moment_spec(scope, columns, order, weight, policy, repr, strategy) if kind == "moment"
            else _count_spec(scope, columns, weight, repr, domain, strategy))
    if backend == "cpu":
        out = _Need()
        _check(_call("tr_tensor_need", scope._handle, C.byref(spec), 0, C.byref(out), None), what="tensor needs")
        return _needs_dict(out)
    dev, adopt = _Need(), _Need()
    _check(_call("tr_tensor_need", scope._handle, C.byref(spec), 1, C.byref(dev), None), what="tensor needs")
    _check(_call("tr_tensor_need", scope._handle, C.byref(spec), 2, C.byref(adopt), None), what="tensor needs")
    return {n: max(int(dev.need[i]), int(adopt.need[i])) for i, n in enumerate(RESOURCES)
            if dev.need[i] or adopt.need[i]}


class Builder:
    """A chunked build: accumulate(begin, end) over disjoint physical row ranges, merge(other builder of the same
    scope and spec), finish() once every row is covered (the same state as a one-shot build, bit for bit)"""

    def __init__(self, scope, columns, order=None, *, kind="moment", weight=None, policy="listwise", repr="dense",
                 domain=None, strategy="auto", budget=None):
        if scope.source == "table":
            scope.require_fresh("tensor builder")
        self.spec = (_moment_spec(scope, columns, order, weight, policy, repr, strategy) if kind == "moment"
                     else _count_spec(scope, columns, weight, repr, domain, strategy))
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_tensor_init", scope._handle, C.byref(self.spec), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        self._handle, self.scope, self.budget = _made(h, error, refusal, "tensor builder"), scope, budget

    def accumulate(self, begin, end):
        _check(_call("tr_tensor_accumulate", self._handle, int(begin), int(end)),
               what="accumulate (rows within the scope, each range once)")
        return self

    def merge(self, other):
        _check(_call("tr_tensor_merge_builder", self._handle, other._handle),
               what="merge (same scope and spec; disjoint rows)")
        return self

    def finish(self):
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_tensor_finish", self._handle, C.byref(_budget(self.budget)), C.byref(refusal), C.byref(error))
        t = Tensor._wrap(_made(h, error, refusal, "finish (every row covered once)"), self.scope)
        self.close()
        return t

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_tensor_builder_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _dims_arg(dims):
    dims = [int(d) for d in dims]
    if len(dims) > MAX_ORDER:
        from .math_errors import UnsupportedError
        raise UnsupportedError(f"order {len(dims)}: at most {MAX_ORDER}")
    return dims, _u64s(dims)


def dense(values, *, groups=False, counts=None, budget=None):
    """a GENERIC dense tensor from nested lists / an array of exact ints (groups=True: the first axis indexes the
    groups); counts: the same shape of per-entry counts, or None"""
    try:
        a = np.asarray(values, dtype=object)
    except ValueError as exc:
        raise ValueError(f"dense values must be a rectangular array ({exc})") from None
    if any(isinstance(v, (list, tuple, np.ndarray)) for v in a.ravel().tolist()):
        raise ValueError("dense values must be a rectangular array of integers")
    shape = a.shape
    G, dims = (shape[0], shape[1:]) if groups else (1, shape)
    if groups and G < 1:
        raise ValueError("at least one group")
    dims, d = _dims_arg(dims)
    words = _words(a.ravel().tolist())
    cnt = None
    if counts is not None:
        c = np.asarray(counts, dtype=object)
        if c.shape != shape:
            raise ValueError("counts have the values' shape")
        cnt = np.ascontiguousarray([int(x) for x in c.ravel().tolist()] or [0], dtype=np.uint64)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_tensor_from_dense", len(dims), d.ctypes.data, int(G), words.ctypes.data, _ptr(cnt),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Tensor._wrap(_made(h, error, refusal, "dense tensor"))


def sparse(dims, entries, *, groups=1, counts=False, budget=None):
    """a GENERIC sparse tensor from entries in any order: (coords, value) or (group, coords, value), with counts=True
    one more item (the count); duplicates are summed, zeros dropped, entries sorted natively"""
    dims, d = _dims_arg(dims)
    m, entries = len(dims), list(entries)
    gids, coords, vals, cnts = [], [], [], []
    for e in entries:
        e = tuple(e)
        has_group = len(e) == 3 + int(counts)
        if len(e) not in (2 + int(counts), 3 + int(counts)):
            raise ValueError("an entry is (coords, value) or (group, coords, value) (+ count)")
        g, rest = (int(e[0]), e[1:]) if has_group else (0, e)
        c = tuple(int(x) for x in rest[0])
        if len(c) != m:
            raise ValueError(f"{m} coordinates per entry")
        gids.append(g); coords.extend(c); vals.append(rest[1])
        if counts:
            cnts.append(int(rest[2]))
    if any(not 0 <= g <= 0xFFFFFFFF for g in gids) or any(not 0 <= x <= 0xFFFFFFFF for x in coords):
        raise ValueError("group ids and coordinates are 32-bit")
    ga = np.ascontiguousarray(gids or [0], dtype=np.uint32)
    ca = np.ascontiguousarray(coords or [0], dtype=np.uint32)
    words = _words(vals)
    cn = np.ascontiguousarray(cnts or [0], dtype=np.uint64) if counts else None
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_tensor_from_sparse", m, d.ctypes.data, int(groups), len(entries), ga.ctypes.data, ca.ctypes.data,
              words.ctypes.data, _ptr(cn), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Tensor._wrap(_made(h, error, refusal, "sparse tensor"))


def factors(dims, terms, *, groups=1, budget=None):
    """a GENERIC factored tensor from declared rank-one terms: (weight, [vector per axis]) or (group, weight,
    vectors); T_g = sum of weight x outer product of the vectors"""
    dims, d = _dims_arg(dims)
    gids, ws, flat = [], [], []
    for t in terms:
        t = tuple(t)
        g, w, vecs = (int(t[0]), t[1], t[2]) if len(t) == 3 else (0, t[0], t[1])
        vecs = list(vecs)
        if len(vecs) != len(dims) or any(len(v) != dk for v, dk in zip(vecs, dims)):
            raise ValueError("one vector of each axis's length per term")
        for v in vecs:
            for x in v:
                if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not -(1 << 63) <= int(x) < 1 << 63:
                    raise TypeError("factor entries are int64 integers")
                flat.append(int(x))
        gids.append(g); ws.append(w)
    ga = np.ascontiguousarray(gids or [0], dtype=np.uint32)
    fa = np.ascontiguousarray(flat or [0], dtype=np.int64)
    words = _words(ws, "a weight")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_tensor_from_factors", len(dims), d.ctypes.data, int(groups), len(ws), ga.ctypes.data, words.ctypes.data,
              fa.ctypes.data, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Tensor._wrap(_made(h, error, refusal, "factored tensor"))


def outer(a, b, repr="factored", *, budget=None):
    """the outer (Kronecker) product of two tensors of the same groups, in the given representation"""
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_tensor_outer", a._open(), b._open(), _repr(repr), C.byref(_budget(budget)), C.byref(refusal),
              C.byref(error))
    return Tensor._wrap(_made(h, error, refusal, "outer product"))


# ---------------------------------------------------------------------- networks and plans
_STATUS = {1: "chosen", 2: "admissible", 3: "rejected"}
_REASON = {0: None, 1: "cheapest", 2: "forced", 3: "costlier", 4: "not forced", 5: "refused", 6: "over budget",
           7: "forced candidate refused", 8: "no candidate matches the force", 9: "no admissible candidate",
           10: "search refused"}


def _labels(spec, names):
    """label ids (0..15) of a label string or list; names: the shared name -> id map (filled in first-use order)"""
    out = []
    for x in (list(spec) if isinstance(spec, str) else list(spec)):
        if x not in names:
            if len(names) >= MAX_LABELS:
                from .math_errors import UnsupportedError
                raise UnsupportedError(f"at most {MAX_LABELS} labels in a network")
            names[x] = len(names)
        out.append(names[x])
    return out


class Network:
    """A bounded tensor network (native): operands with one label per axis (a grouped tensor: its group label
    first) and the output labels. The tensors must stay alive as long as the network (it keeps them)."""

    def __init__(self, operands, output, *, budget=None):
        operands = list(operands)
        if not 1 <= len(operands) <= NET_MAX:
            from .math_errors import UnsupportedError
            raise UnsupportedError(f"1..{NET_MAX} tensors")
        names, spec = {}, _NetSpec()
        spec.ntensors = len(operands)
        self.tensors = []
        for i, (t, labs) in enumerate(operands):
            ids = _labels(labs, names)
            if len(ids) > MAX_ORDER + 1:
                raise ValueError("too many labels for one tensor")
            spec.nlabels[i] = len(ids)
            for k, l in enumerate(ids):
                spec.labels[i][k] = l
            self.tensors.append(t)
        out = []
        for x in (list(output) if isinstance(output, str) else list(output)):
            if x not in names:
                raise ValueError(f"output label {x!r} is in no operand")
            out.append(names[x])
        spec.noutput = len(out)
        for k, l in enumerate(out):
            spec.output[k] = l
        self.labels = names
        self.output = list(output)
        arr = (C.c_void_p * len(operands))(*[t._open() for t in self.tensors])
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_tensor_net_create", arr, C.byref(spec), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        self._handle = _made(h, error, refusal, "network")

    def _open(self):
        if not self._handle:
            raise ValueError("the network is closed")
        return self._handle

    def plan(self, devices=("cpu",), kernels=("dense", "sparse"), max_orders=None, force=None, rates=None,
             budget=None):
        """every candidate (order x kernel x device) with its exact shapes, bytes, needs, amounts and FND-008 costs,
        status and reason; the cheapest admissible one chosen. A refused plan raises with .plan (the evidence)."""
        o = _Options()
        o.max_orders = int(max_orders or 0)
        o.devices = reg._devices(devices)
        o.kernels = 0
        for k in kernels:
            o.kernels |= KERNELS[k]
        o.force = -1 if force is None else int(force)
        if rates is not None:
            o.custom_rates = 1
            base = default_rates()
            for d, dev in enumerate(("cpu", "cuda")):
                row = dict(base[dev], **rates.get(dev, {}))
                for c, comp in enumerate(reg.COST_COMPONENTS):
                    o.rates[d][c] = float(row[comp])
        handle, refusal = C.c_void_p(), _Refusal()
        rc = _call("tr_tensor_net_plan", self._open(), C.byref(o), C.byref(_budget(budget)), C.byref(handle),
                   C.byref(refusal))
        if not handle.value:
            _check(rc, refusal, "contraction plan")
            raise MemoryError("contraction plan: native allocation failed")
        plan = Plan(handle.value, self)
        if rc:
            try:
                _check(rc, refusal if refusal.resource >= 0 else None, "contraction plan: " + (plan.reason or "refused"))
            except Exception as exc:
                exc.plan = plan
                raise
        return plan

    def contract(self, plan=None, candidate=None, *, backend=None, budget=None):
        """(output tensor, execution: actual entries, bytes and products per step, the actual peak) of a candidate
        (None: the chosen one) of plan (None: a fresh CPU plan); a CUDA candidate runs on the device"""
        plan = plan or self.plan(budget=budget)
        index = plan.chosen if candidate is None else int(candidate)
        if index is None:
            raise ValueError("the plan has no chosen candidate")
        cand = plan.candidate(index)
        if backend is not None and backend != cand["device"]:
            raise ValueError(f"candidate {index} runs on {cand['device']}")
        if cand["device"] == "cuda":
            return self._contract_cuda(plan, index, budget)
        rec, error, refusal = _Exec(), C.c_int32(), _Refusal()
        h = _call("tr_tensor_net_execute", self._open(), plan._open(), index, C.byref(_budget(budget)), C.byref(rec),
                  C.byref(refusal), C.byref(error))
        return Tensor._wrap(_made(h, error, refusal, f"candidate {index}")), _exec_dict(rec)

    def _contract_cuda(self, plan, index, budget):
        gpu = _gpu()
        prog = _Program()
        _check(_call("tr_tensor_net_program", self._open(), plan._open(), index, C.byref(prog)), what="program")
        inputs, views = [], (_DenseView * int(prog.ninputs))()
        for i in range(int(prog.ninputs)):
            error, refusal = C.c_int32(), _Refusal()
            h = _call("tr_tensor_net_input_dense", self._open(), i, C.byref(_budget(budget)), C.byref(refusal),
                      C.byref(error))
            t = Tensor._wrap(_made(h, error, refusal, f"dense input {i}"))
            inputs.append(t)
            st = t._states()
            views[i].value, views[i].cells, views[i].order = st.value, st.cells, st.order
            for k in range(int(st.order)):
                views[i].dims[k] = st.dims[k]
        cells = int(prog.output_cells)
        out = np.zeros(max(2 * cells, 2), np.int64)
        rec, refusal = _Exec(), _Refusal()
        gpu.run("tr_mgpu_tensor_contract", C.byref(prog), C.cast(views, C.c_void_p), C.byref(_budget(budget)),
                out.ctypes.data, C.byref(rec), C.byref(refusal), refusal=refusal, what="contraction on the device")
        rec.candidate = index
        dims = _u64s([int(prog.output_dims[k]) for k in range(int(prog.noutput))])
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_tensor_from_dense", int(prog.noutput), dims.ctypes.data, 1, out.ctypes.data, None,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        d = gpu.device()
        t = Tensor._wrap(_made(h, error, refusal, "contraction output"), None,
                         "cuda-emulated" if d["emulated"] else "cuda", d["name"])
        del inputs
        return t, _exec_dict(rec)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_tensor_net_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def network(operands, output, *, budget=None):
    """Network([(tensor, labels), ...], output labels): labels as strings ("ij") or lists of hashable names"""
    return Network(operands, output, budget=budget)


def _exec_dict(rec):
    n = int(rec.nsteps)
    return dict(candidate=int(rec.candidate), kernel={1: "dense", 2: "sparse"}.get(rec.kernel),
                device={1: "cpu", 2: "cuda"}.get(rec.device), steps=[
                    dict(nnz=int(rec.nnz[s]), bytes=int(rec.bytes[s]), pairs=int(rec.pairs[s])) for s in range(n)],
                peak_bytes=int(rec.peak_bytes), output_nnz=int(rec.output_nnz), output_bytes=int(rec.output_bytes),
                converted_bytes=int(rec.converted_bytes))


class Plan:
    """The evidence of a contraction plan (native): every candidate, the choice and why."""

    def __init__(self, handle, net):
        self._handle, self.network = handle, net
        i = _PlanInfo()
        _check(_call("tr_tensor_plan_get_info", handle, C.byref(i)), what="plan")
        self._info = i

    def _open(self):
        if not self._handle:
            raise ValueError("the plan is closed")
        return self._handle

    @property
    def info(self):
        i = self._info
        return dict(candidates=int(i.ncandidates), chosen=None if i.chosen < 0 else int(i.chosen), code=int(i.code),
                    reason=_REASON.get(i.reason), complete=bool(i.complete), orders=int(i.norders),
                    total_orders=int(i.total_orders), forced=bool(i.forced), bytes=int(i.bytes),
                    devices=[d for b, d in ((1, "cpu"), (2, "cuda")) if i.devices & b],
                    refusal=None if not i.refusal.refused else dict(resource=RESOURCES[i.refusal.resource],
                                                                   need=int(i.refusal.need), limit=int(i.refusal.limit)),
                    cost_unit="relative work units; not time")

    chosen = property(lambda self: self.info["chosen"])
    reason = property(lambda self: self.info["reason"])

    def candidate(self, i):
        c = _Candidate()
        _check(_call("tr_tensor_plan_get", self._open(), int(i), C.byref(c)), what="plan candidate")
        inv = {v: k for k, v in self.network.labels.items()}
        g = c.reg
        steps = []
        for s in range(int(c.nsteps)):
            st = c.step[s]
            steps.append(dict(a=int(st.a), b=None if st.b < 0 else int(st.b),
                              out=[inv[int(x)] for x in st.out[:st.nout]], summed=[inv[int(x)] for x in st.sum[:st.nsum]],
                              cells=int(st.cells), terms=int(st.terms), nnz=int(st.nnz), pairs=int(st.pairs),
                              bytes=int(st.bytes), scratch=int(st.scratch), live=int(st.live),
                              free_a=bool(st.free_a), free_b=bool(st.free_b), bound=_int(st.bound.lo, st.bound.hi)))
        refusal = reg._refusal_dict(g.refusal) if g.refusal.kind else None
        return dict(index=int(i), order=int(c.order), kernel={1: "dense", 2: "sparse"}[c.kernel],
                    device={1: "cpu", 2: "cuda"}[c.device], status=_STATUS.get(g.status), reason=_REASON.get(g.reason),
                    valid=bool(g.valid), code=int(g.code), refusal=refusal,
                    budget=None if not g.budget.refused else dict(resource=RESOURCES[g.budget.resource],
                                                                 need=int(g.budget.need), limit=int(g.budget.limit)),
                    need={n: int(g.need.need[k]) for k, n in enumerate(RESOURCES) if g.need.need[k]},
                    amounts=dict(zip(reg.COST_COMPONENTS, (int(v) for v in g.amount))),
                    costs=dict(zip(reg.COST_COMPONENTS, (float(v) for v in g.cost))), total=float(g.total),
                    steps=steps, peak_bytes=int(c.peak_bytes), output_bytes=int(c.output_bytes),
                    output_cells=int(c.output_cells), output_nnz=int(c.output_nnz), input_bytes=int(c.input_bytes),
                    converted=int(c.converted), bound=_int(c.bound.lo, c.bound.hi))

    @property
    def candidates(self):
        return [self.candidate(i) for i in range(int(self._info.ncandidates))]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_tensor_plan_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
