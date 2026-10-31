# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DIST-001 / DIST-002: reusable scalar and joint histograms over a FND-004 scope, and the exact prefix calculus of a
sorted support (CDF, tails, ranges, quantiles, trimmed sums, range integrals), natively on the CPU or with the CUDA
kernels (torbles/math_dist.c, math_dist_gpu.cu).

    from torbles import math_bank as mbk, math_dist as md
    scope = mbk.Scope.from_table(db["f"], ["x", "y", "w"], group_by=["k"], rows=mask)
    h = md.histogram(scope, ["x", "y"], weight="w", repr="sparse", backend="cpu")   # or "dense", backend="cuda"
    h.cells(g)            # [(key, count, wcount, wsum)]: key = (x, y) logical ints, None for the NULL category
    h.table()             # the present groups with their decoded group keys and cells
    h.to_dense(); h.to_sparse(); h.merge(other); h.rollup(coarse_scope); h.marginal(["x"])
    p = md.histogram(scope, ["x"], weight="w").prefix(mass="weight", degree=2)
    p.cdf(10, group=g)                       # {"mass", "total", "fraction"} for x <= 10 (closed=False: x < 10)
    p.count(lo=3, hi=Fraction(15, 2), closed=(False, True))      # 3 < x <= 7.5
    p.quantile(Fraction(1, 4), rule="cont")  # exact Fraction (rules: disc, cont, lower, higher); None when empty
    p.moment(2, lo=0)                        # sum of mass * x^2 over x >= 0 (a range integral)
    p.trimmed(r1, r2, k=1); p.value_at_rank(r)
    p.query([...], backend="cuda")          # a batch: one CUDA work item per request

Values are the scope's logical integers (a DECIMAL(p, s) column's value is logical / 10^s); thresholds and levels are
exact rationals in those units (int, Fraction or Decimal; a float is refused: its binary value is rarely the intended
one). Results are exact ints / Fractions computed natively; Python only marshals and formats. Refusals:
AlgebraBudgetError (before allocation), DomainError (a dense domain that would drop a cell; negative weights as a
mass), OverflowError, UnsupportedError, StaleStateError (a table scope whose table changed). Opt-in: `import torbles`
does not load this module.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, UNKNOWN, budget_error, check as _check

__all__ = ["Histogram", "Builder", "Prefix", "histogram", "need", "prefix", "requests", "REPRS", "STRATEGIES",
           "MASSES", "RULES", "MAX_DIMS", "MAX_DEGREE", "REQUEST_DTYPE", "RESULT_DTYPE"]

MAX_DIMS, MAX_DEGREE = 3, 4
REPRS = {"dense": 1, "sparse": 2}
STRATEGIES = {"auto": 0, "count": 1, "sort": 2, "direct": 3, "derived": 4, "device": 5}
MASSES = {"count": 1, "weight": 2}
RULES = {"disc": 1, "cont": 2, "lower": 3, "higher": 4}
OPS = {"range": 1, "moment": 2, "quantile": 3, "rank": 4, "trimmed": 5, "trimmed_mean": 6}
LO_CLOSED, HI_CLOSED, LO_NONE, HI_NONE = 1, 2, 4, 8
_NEED_CPU, _NEED_DEVICE, _NEED_ADOPT = 0, 1, 2
I64 = (-(1 << 63), (1 << 63) - 1)
_U64 = (1 << 64) - 1
P = C.POINTER
i32, i64, u64, vp = C.c_int32, C.c_int64, C.c_uint64, C.c_void_p


class _Spec(C.Structure):
    _native_ = "tr_dist_spec"
    _fields_ = [("k", i32), ("col", i32 * 3), ("weight", i32), ("repr", i32), ("domain", i32), ("nullslot", i32 * 3),
                ("lo", i64 * 3), ("hi", i64 * 3), ("strategy", i32), ("reserved", i32)]


class _States(C.Structure):
    _native_ = "tr_dist_states"
    _fields_ = [("ngroups", u64), ("ncells", u64), ("capacity", u64), ("cells", u64), ("k", i32), ("repr", i32),
                ("weighted", i32), ("reserved", i32), ("lo", i64 * 3), ("span", u64 * 3), ("dim", u64 * 3),
                ("rows", vp), ("offset", vp), ("value", vp * 3), ("nulls", vp), ("count", vp), ("wcount", vp),
                ("wsum", vp)]


class _HistInfo(C.Structure):
    _native_ = "tr_dist_hist_info"
    _fields_ = [(n, u64) for n in ("ngroups", "present", "rows", "ncells", "occupied", "max_cells", "bytes", "scans",
                                   "passes")] + [
        ("digest", u64 * 2), ("lo", i64 * 3), ("hi", i64 * 3), ("dim", u64 * 3)] + [
        (n, i32) for n in ("k", "repr", "weighted", "scoped")] + [("col", i32 * 3), ("weight", i32),
                                                                   ("scale", i32 * 3), ("wscale", i32),
                                                                   ("nullslot", i32 * 3), ("strategy", i32),
                                                                   ("wnonneg", i32), ("reserved", i32)]


class _PrefixView(C.Structure):
    _native_ = "tr_dist_prefix_view"
    _fields_ = [("ngroups", u64), ("nsupport", u64), ("degree", i32), ("mass", i32), ("scale", i32), ("wscale", i32),
                ("rows", vp), ("nulls", vp), ("offset", vp), ("value", vp), ("prefix", vp)]


class _PrefixInfo(C.Structure):
    _native_ = "tr_dist_prefix_info"
    _fields_ = [(n, u64) for n in ("ngroups", "present", "nsupport", "max_support", "rows", "nulls", "bytes")] + [
        ("digest", u64 * 2)] + [(n, i32) for n in ("degree", "mass", "scale", "wscale", "scoped", "reserved")]


class _Request(C.Structure):
    _native_ = "tr_dist_request"
    _fields_ = [("op", i32), ("flags", i32), ("group", u64), ("a_num", i64), ("a_den", i64), ("b_num", i64),
                ("b_den", i64), ("power", i32), ("rule", i32), ("reserved", i64)]


class _Result(C.Structure):
    _native_ = "tr_dist_result"
    _fields_ = [("status", i32), ("empty", i32), ("lo", u64), ("hi", u64), ("mass", i64 * 2), ("total", i64 * 2),
                ("num", u64 * 4), ("den", i64 * 2)]


for _s, _n in ((_Spec, 96), (_States, 192), (_HistInfo, 232), (_PrefixView, 72), (_PrefixInfo, 96), (_Request, 64),
               (_Result, 104)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

REQUEST_DTYPE = np.dtype([("op", "<i4"), ("flags", "<i4"), ("group", "<u8"), ("a_num", "<i8"), ("a_den", "<i8"),
                          ("b_num", "<i8"), ("b_den", "<i8"), ("power", "<i4"), ("rule", "<i4"), ("reserved", "<i8")])
RESULT_DTYPE = np.dtype([("status", "<i4"), ("empty", "<i4"), ("lo", "<u8"), ("hi", "<u8"), ("mass", "<i8", (2,)),
                         ("total", "<i8", (2,)), ("num", "<u8", (4,)), ("den", "<i8", (2,))])
assert REQUEST_DTYPE.itemsize == 64 and RESULT_DTYPE.itemsize == 104

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_dist_hist_init": (vp, [vp, P(_Spec), B, R, e]),
            "tr_dist_hist_accumulate": (i32, [vp, u64, u64]),
            "tr_dist_hist_merge_builder": (i32, [vp, vp]),
            "tr_dist_hist_finish": (vp, [vp, B, R, e]),
            "tr_dist_hist_builder_free": (None, [vp]),
            "tr_dist_hist_build": (vp, [vp, P(_Spec), B, R, e]),
            "tr_dist_hist_need": (i32, [vp, P(_Spec), i32, P(_Need), P(u64)]),
            "tr_dist_hist_from_states": (vp, [vp, P(_Spec), P(_States), B, R, e]),
            "tr_dist_hist_to_dense": (vp, [vp, vp, vp, vp, B, R, e]),
            "tr_dist_hist_to_sparse": (vp, [vp, B, R, e]),
            "tr_dist_hist_merge": (vp, [vp, vp, B, R, e]),
            "tr_dist_hist_rollup": (vp, [vp, vp, u64, vp, B, R, e]),
            "tr_dist_hist_marginal": (vp, [vp, C.c_uint32, B, R, e]),
            "tr_dist_hist_clone": (vp, [vp, e]),
            "tr_dist_hist_equal": (i32, [vp, vp]),
            "tr_dist_hist_get_info": (i32, [vp, P(_HistInfo)]),
            "tr_dist_hist_get_states": (i32, [vp, P(_States)]),
            "tr_dist_hist_free": (None, [vp]),
            "tr_dist_prefix_build": (vp, [vp, i32, i32, B, R, e]),
            "tr_dist_prefix_need": (i32, [vp, i32, i32, P(_Need)]),
            "tr_dist_prefix_get_view": (i32, [vp, P(_PrefixView)]),
            "tr_dist_prefix_get_info": (i32, [vp, P(_PrefixInfo)]),
            "tr_dist_prefix_equal": (i32, [vp, vp]),
            "tr_dist_prefix_free": (None, [vp]),
            "tr_dist_query": (i32, [vp, vp, u64, vp, B, R]),
            "tr_dist_prefix_device_bytes": (u64, [P(_PrefixView)]),
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
    from .math_bank import _View
    math_gpu.bind("tr_mgpu_dist_hist_build", C.c_int, [P(_View), P(_Spec), P(_Budget), P(_States), P(_Refusal)])
    math_gpu.bind("tr_mgpu_dist_prefix_upload", vp, [P(_PrefixView), P(_Budget), P(_Refusal), P(C.c_int)])
    math_gpu.bind("tr_mgpu_dist_prefix_free", None, [vp])
    math_gpu.bind("tr_mgpu_dist_prefix_bytes", u64, [vp])
    math_gpu.bind("tr_mgpu_dist_query", C.c_int, [vp, vp, u64, vp, P(_Budget), P(_Refusal)])
    return math_gpu


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _budget(budget):
    return (budget or Budget()).raw


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


def _device_label():
    from . import math_gpu
    d = math_gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


def _i128(words):
    lo, hi = int(words[0]), int(words[1])
    return (hi << 64) | (lo & _U64)


def _i256(words):
    v = 0
    for i in range(3, -1, -1):
        v = (v << 64) | (int(words[i]) & _U64)
    return v - (1 << 256) if v >> 255 else v


def _rational(value, what):
    """an exact rational (num, den) with int64 parts; a float is refused"""
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{what} must be exact (int, Fraction or Decimal), not {type(value).__name__}")
    if isinstance(value, Decimal):
        value = Fraction(value)
    if isinstance(value, Integral):
        num, den = int(value), 1
    elif isinstance(value, Rational):
        num, den = int(value.numerator), int(value.denominator)
    else:
        raise TypeError(f"{what} must be exact (int, Fraction or Decimal)")
    if not (I64[0] <= num <= I64[1] and 1 <= den <= I64[1]):
        raise OverflowError(f"{what}: numerator and denominator must fit int64")
    return num, den


# ---------------------------------------------------------------------- the spec
def _spec(scope, columns, weight, repr, domain, strategy):
    columns = [columns] if isinstance(columns, str) else list(columns)
    if not 1 <= len(columns) <= MAX_DIMS:
        raise ValueError(f"a histogram has 1..{MAX_DIMS} coordinates")
    for name in columns + ([weight] if weight is not None else []):
        if name not in scope.names:
            raise ValueError(f"{name!r} is not a column of the scope ({', '.join(scope.names)})")
    if repr not in ("dense", "sparse"):
        raise ValueError("repr is 'dense' or 'sparse'")
    if strategy not in ("auto", "count", "sort"):
        raise ValueError("strategy is 'auto', 'count' or 'sort'")
    s = _Spec()
    s.k = len(columns)
    for c in range(3):
        s.col[c] = scope.names.index(columns[c]) if c < len(columns) else -1
    s.weight = scope.names.index(weight) if weight is not None else -1
    s.repr, s.strategy = REPRS[repr], STRATEGIES[strategy]
    s.domain = 0
    if domain is not None:
        if repr != "dense":
            raise ValueError("an explicit domain applies to a dense histogram")
        domain = list(domain)
        if len(domain) != len(columns):
            raise ValueError("domain: one (lo, hi, null slot) per coordinate")
        s.domain = 1
        for c, d in enumerate(domain):
            lo, hi, null = d
            if not (I64[0] <= int(lo) <= I64[1] and I64[0] <= int(hi) <= I64[1]):
                raise OverflowError("domain bounds must fit int64")
            s.lo[c], s.hi[c], s.nullslot[c] = int(lo), int(hi), int(bool(null))
    return s, tuple(columns)


def need(scope, columns, *, weight=None, repr="sparse", domain=None, strategy="auto", backend="cpu"):
    """the needs checked before allocating: {resource: need} (CPU: the build's first stage; a sparse build adds its
    final state once the occupied cells are counted; "cuda": resource by resource the larger of the device copies
    and the adopted state, which for a sparse state is bounded by one cell per selected row)"""
    s, _ = _spec(scope, columns, weight, repr, domain, strategy)
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    targets = (_NEED_CPU,) if backend == "cpu" else (_NEED_DEVICE, _NEED_ADOPT)
    both = _Need()
    for target in targets:
        out, cap = _Need(), C.c_uint64()
        _check(_call("tr_dist_hist_need", scope._handle, C.byref(s), target, C.byref(out), C.byref(cap)),
               what="needs")
        for i in range(len(RESOURCES)):
            both.need[i] = max(both.need[i], out.need[i])
    return _needs_dict(both)


def histogram(scope, columns, *, weight=None, repr="sparse", domain=None, strategy="auto", backend="cpu", budget=None):
    """the histogram of scope over 1..3 columns (names of the scope) with an optional weight column. repr "dense"
    (domain: None = the scope's [min, max] per column with a NULL slot when a value is NULL, or [(lo, hi, null_slot)]
    per coordinate) or "sparse" (strategy auto / count / sort on the CPU). backend "cuda" runs the CUDA kernels."""
    if scope.source == "table":
        scope.require_fresh("histogram")
    s, names = _spec(scope, columns, weight, repr, domain, strategy)
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = _call("tr_dist_hist_build", scope._handle, C.byref(s), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Histogram._wrap(_made(h, error, refusal, "histogram"), scope, names, weight, "cpu", "native CPU")
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    gpu = _gpu()
    dev, cap = _Need(), C.c_uint64()
    _check(_call("tr_dist_hist_need", scope._handle, C.byref(s), _NEED_DEVICE, C.byref(dev), C.byref(cap)),
           what="histogram needs")
    _precheck(budget, dev, "histogram on the device")
    view = scope.view()
    k, weighted, sparse = s.k, s.weight >= 0, s.repr == REPRS["sparse"]
    G, capacity = scope.groups, int(cap.value)
    keep = dict(rows=np.zeros(max(G, 1), np.uint64), count=np.zeros(max(capacity, 1), np.uint64))
    st = _States()
    st.ngroups, st.capacity, st.k, st.repr, st.weighted = G, capacity, k, s.repr, int(weighted)
    st.rows, st.count = keep["rows"].ctypes.data, keep["count"].ctypes.data
    if weighted:
        keep["wcount"] = np.zeros(max(capacity, 1), np.uint64)
        keep["wsum"] = np.zeros(max(2 * capacity, 2), np.int64)
        st.wcount, st.wsum = keep["wcount"].ctypes.data, keep["wsum"].ctypes.data
    if sparse:
        keep["offset"] = np.zeros(G + 1, np.uint64)
        keep["nulls"] = np.zeros(max(capacity, 1), np.uint8)
        st.offset, st.nulls = keep["offset"].ctypes.data, keep["nulls"].ctypes.data
        for c in range(k):
            keep[f"v{c}"] = np.zeros(max(capacity, 1), np.int64)
            st.value[c] = keep[f"v{c}"].ctypes.data
    gpu.run("tr_mgpu_dist_hist_build", C.byref(view), C.byref(s), C.byref(_budget(budget)), C.byref(st),
            C.byref(refusal), refusal=refusal, what="histogram on the device")
    h = _call("tr_dist_hist_from_states", scope._handle, C.byref(s), C.byref(st), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Histogram._wrap(_made(h, error, refusal, "histogram states"), scope, names, weight, *_device_label())


def prefix(scope, column, *, weight=None, mass="count", degree=2, backend="cpu", budget=None):
    """the prefix state of one column (a sparse 1-D histogram, then its prefix sums)"""
    return histogram(scope, [column], weight=weight, backend=backend, budget=budget).prefix(mass, degree, budget=budget)


class Builder:
    """A chunked accumulation: accumulate(begin, end) over disjoint physical row ranges, merge(other builder of the
    same scope and spec), finish() once every row is covered. Not thread-safe: one owner."""

    def __init__(self, scope, columns, *, weight=None, repr="sparse", domain=None, strategy="auto", budget=None):
        if scope.source == "table":
            scope.require_fresh("histogram builder")
        s, self.names = _spec(scope, columns, weight, repr, domain, strategy)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dist_hist_init", scope._handle, C.byref(s), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        self._handle = _made(h, error, refusal, "histogram builder")
        self.scope, self.weight, self._budget = scope, weight, budget

    def _open(self):
        if not self._handle:
            raise ValueError("the builder is finished or closed")

    def accumulate(self, begin, end):
        self._open()
        _check(_call("tr_dist_hist_accumulate", self._handle, int(begin), int(end)),
               what="accumulate (rows within the scope, each range once)")
        return self

    def merge(self, other):
        self._open(); other._open()
        _check(_call("tr_dist_hist_merge_builder", self._handle, other._handle),
               what="merge (same scope and spec; disjoint rows)")
        return self

    def finish(self, budget=None):
        self._open()
        error, refusal = C.c_int32(), _Refusal()
        b = budget if budget is not None else self._budget
        h = _call("tr_dist_hist_finish", self._handle, C.byref(_budget(b)), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "finish (every row covered once)")
        out = Histogram._wrap(h, self.scope, self.names, self.weight, "cpu", "native CPU")
        self.close()
        return out

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_dist_hist_builder_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- histograms
class Histogram:
    """An immutable native histogram: per group the bag multiplicity and the cells (count, and wcount / wsum when
    weighted); NULL is a category of its own per coordinate; dense or sparse."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_dist.histogram(...), Builder(...).finish() or a Histogram method")

    @classmethod
    def _wrap(cls, handle, scope, names, weight, backend, device):
        self = object.__new__(cls)
        self._handle, self.scope, self.names, self.weight = handle, scope, tuple(names), weight
        self.backend, self.device = backend, device
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the histogram is closed")
        return self._handle

    def _raw_info(self):
        out = _HistInfo()
        _check(_call("tr_dist_hist_get_info", self._open(), C.byref(out)), what="histogram")
        return out

    @property
    def info(self):
        i = self._raw_info()
        k = int(i.k)
        repr_ = {v: n for n, v in REPRS.items()}[int(i.repr)]
        return dict(groups=int(i.ngroups), present=int(i.present), rows=int(i.rows), cells=int(i.ncells),
                    occupied=int(i.occupied), max_cells=int(i.max_cells), bytes=int(i.bytes), scans=int(i.scans),
                    passes=int(i.passes), k=k, repr=repr_, weighted=bool(i.weighted), scoped=bool(i.scoped),
                    strategy={v: n for n, v in STRATEGIES.items()}.get(int(i.strategy), "unknown"),
                    scales=tuple(int(i.scale[c]) for c in range(k)), wscale=int(i.wscale),
                    nonnegative_weights=bool(i.wnonneg),
                    domain=[(int(i.lo[c]), int(i.hi[c]), bool(i.nullslot[c])) for c in range(k)] if repr_ == "dense"
                    else None, backend=self.backend, device=self.device)

    @property
    def groups(self):
        return int(self._raw_info().ngroups)

    @property
    def present(self):
        return int(self._raw_info().present)

    @property
    def rows(self):
        return int(self._raw_info().rows)

    @property
    def bytes(self):
        return int(self._raw_info().bytes)

    @property
    def k(self):
        return int(self._raw_info().k)

    def _states(self):
        st = _States()
        _check(_call("tr_dist_hist_get_states", self._open(), C.byref(st)), what="histogram states")
        return st

    def arrays(self):
        """copies of the native states: rows [G]; count [n] (and wcount [n], wsum [n] exact ints when weighted);
        sparse: offset [G + 1], values [k][n], nulls [n]; dense: the domain (lo, span, dim) and cells per group"""
        st = self._states()
        G, n, k = int(st.ngroups), int(st.ncells), int(st.k)

        def arr(ptr, ctype, count):                      # count > 0
            return np.ctypeslib.as_array((ctype * count).from_address(ptr)).copy()
        out = dict(rows=np.ctypeslib.as_array((C.c_uint64 * G).from_address(st.rows)).copy(),
                   count=arr(st.count, C.c_uint64, n) if n else np.zeros(0, np.uint64), repr=int(st.repr), k=k)
        if st.weighted:
            out["wcount"] = arr(st.wcount, C.c_uint64, n) if n else np.zeros(0, np.uint64)
            words = arr(st.wsum, C.c_int64, 2 * n) if n else np.zeros(0, np.int64)
            out["wsum"] = [_i128(words[2 * i:2 * i + 2]) for i in range(n)]
        if st.repr == REPRS["sparse"]:
            out["offset"] = np.ctypeslib.as_array((C.c_uint64 * (G + 1)).from_address(st.offset)).copy()
            out["values"] = [arr(st.value[c], C.c_int64, n) if n else np.zeros(0, np.int64) for c in range(k)]
            out["nulls"] = arr(st.nulls, C.c_uint8, n) if n else np.zeros(0, np.uint8)
        else:
            out.update(cells=int(st.cells), lo=[int(st.lo[c]) for c in range(k)],
                       span=[int(st.span[c]) for c in range(k)], dim=[int(st.dim[c]) for c in range(k)])
        return out

    def _decode_dense(self, a, idx):
        key = []
        stride = 1
        strides = []
        for c in range(a["k"] - 1, -1, -1):
            strides.append(stride)
            stride *= a["dim"][c]
        strides = strides[::-1]
        for c in range(a["k"]):
            i = (idx // strides[c]) % a["dim"][c]
            key.append(None if i == a["span"][c] else a["lo"][c] + i)
        return tuple(key)

    def _cells(self, a, g):
        weighted = "wcount" in a
        out = []
        if a["repr"] == REPRS["sparse"]:
            b, e = int(a["offset"][g]), int(a["offset"][g + 1])
            for i in range(b, e):
                key = tuple(None if (int(a["nulls"][i]) >> c) & 1 else int(a["values"][c][i]) for c in range(a["k"]))
                out.append((key, int(a["count"][i])) + ((int(a["wcount"][i]), a["wsum"][i]) if weighted else ()))
        else:
            cells = a["cells"]
            for j in range(cells):
                i = g * cells + j
                if a["count"][i]:
                    out.append((self._decode_dense(a, j), int(a["count"][i])) +
                               ((int(a["wcount"][i]), a["wsum"][i]) if weighted else ()))
        return out

    def cells(self, g=0):
        """the occupied cells of group g in cell order: (key, count) or (key, count, wcount, wsum); a key is a tuple
        of logical ints, None for the NULL category"""
        a = self.arrays()
        if not 0 <= int(g) < len(a["rows"]):
            raise ValueError("group id out of range")
        return self._cells(a, int(g))

    def group(self, g):
        """{"present", "rows", "cells"} of group g"""
        a = self.arrays()
        g = int(g)
        if not 0 <= g < len(a["rows"]):
            raise ValueError("group id out of range")
        rows = int(a["rows"][g])
        return dict(present=rows > 0, rows=rows, cells=self._cells(a, g))

    def table(self):
        """the present groups: [{"group", "key" (decoded with a scope), "rows", "cells"}]"""
        a = self.arrays()
        out = []
        for g in range(len(a["rows"])):
            if a["rows"][g]:
                key = self.scope.key_values(g) if self.scope is not None else (g,)
                out.append(dict(group=g, key=key, rows=int(a["rows"][g]), cells=self._cells(a, g)))
        return out

    def equal(self, other):
        return bool(_check(_call("tr_dist_hist_equal", self._open(), other._open()), what="histogram comparison"))

    def __eq__(self, other):
        return isinstance(other, Histogram) and self.equal(other)

    __hash__ = None

    def _derived(self, handle, error, refusal, what, names=None, scope="same"):
        return Histogram._wrap(_made(handle, error, refusal, what), self.scope if scope == "same" else scope,
                               names if names is not None else self.names, self.weight, "cpu", "native CPU")

    def to_dense(self, domain=None, *, budget=None):
        """the dense image over domain [(lo, hi, null_slot)] per coordinate (None: the hull of the occupied cells);
        DomainError when an occupied cell lies outside it, AlgebraBudgetError over budget"""
        error, refusal = C.c_int32(), _Refusal()
        if domain is None:
            h = _call("tr_dist_hist_to_dense", self._open(), None, None, None, C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
        else:
            domain = list(domain)
            if len(domain) != self.k:
                raise ValueError("domain: one (lo, hi, null slot) per coordinate")
            lo = (C.c_int64 * 3)(*[int(d[0]) for d in domain])
            hi = (C.c_int64 * 3)(*[int(d[1]) for d in domain])
            ns = (C.c_int32 * 3)(*[int(bool(d[2])) for d in domain])
            h = _call("tr_dist_hist_to_dense", self._open(), lo, hi, ns, C.byref(_budget(budget)), C.byref(refusal),
                      C.byref(error))
        return self._derived(h, error, refusal, "dense histogram")

    def to_sparse(self, *, budget=None):
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dist_hist_to_sparse", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return self._derived(h, error, refusal, "sparse histogram")

    def merge(self, other, *, budget=None):
        """the bag union with another histogram of the same shape (disjoint rows, e.g. partitions); unscoped"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dist_hist_merge", self._open(), other._open(), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return self._derived(h, error, refusal, "histogram merge", scope=None)

    def rollup(self, coarse=None, *, map=None, groups=None, budget=None):
        """the coarse histogram along a refinement map: coarse (a Scope of the same rows: its map from this
        histogram's scope; the result is that scope's histogram) or map= (coarse id per fine group) with groups="""
        if coarse is not None:
            if self.scope is None:
                raise ValueError("this histogram has no scope: give map= and groups=")
            self.scope.require_fresh("rollup"); coarse.require_fresh("rollup")
            fmap, ngroups, origin = self.scope.rollup_map(coarse), coarse.groups, coarse
        else:
            if map is None or groups is None:
                raise ValueError("give a coarse scope, or map= and groups=")
            fmap = np.ascontiguousarray(map, dtype=np.uint32)
            ngroups, origin = int(groups), None
            if fmap.shape != (self.groups,):
                raise ValueError("map: one coarse id per fine group")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dist_hist_rollup", self._open(), fmap.ctypes.data, ngroups,
                  origin._handle if origin is not None else None, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return self._derived(h, error, refusal, "rollup", scope=origin)

    def marginal(self, columns, *, budget=None):
        """the histogram of a subset of the coordinates (in their order), the others summed out"""
        columns = [columns] if isinstance(columns, str) else list(columns)
        if not columns or any(c not in self.names for c in columns) or len(set(columns)) != len(columns):
            raise ValueError(f"choose distinct coordinates among {self.names}")
        mask = 0
        for c in columns:
            mask |= 1 << self.names.index(c)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dist_hist_marginal", self._open(), mask, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        kept = tuple(n for n in self.names if n in columns)
        return self._derived(h, error, refusal, "marginal", names=kept)

    def prefix(self, mass="count", degree=2, *, budget=None):
        """the prefix state of a 1-D histogram: mass "count" (rows) or "weight" (weight sums; every weight >= 0),
        prefix sums of mass * x^k for k = 0..degree (<= 4)"""
        if mass not in MASSES:
            raise ValueError("mass is 'count' or 'weight'")
        error, refusal = C.c_int32(), _Refusal()
        p = _call("tr_dist_prefix_build", self._open(), MASSES[mass], int(degree), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Prefix._wrap(_made(p, error, refusal, "prefix state"), self)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_dist_hist_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Histogram(closed)"
        i = self.info
        return (f"Histogram({i['repr']}, k={i['k']}, groups={i['groups']}, present={i['present']}, "
                f"cells={i['cells']}, {self.backend})")


# ---------------------------------------------------------------------- prefix states and requests
def requests(items):
    """a REQUEST_DTYPE array from dicts: {"op": "range" | "moment" | "quantile" | "rank" | "trimmed" |
    "trimmed_mean", "group", and lo / hi / closed (range, moment), p / rule (quantile), r (rank), r1 / r2
    (trimmed), k (moment, trimmed)}"""
    items = list(items)
    a = np.zeros(len(items), dtype=REQUEST_DTYPE)
    for i, q in enumerate(items):
        op = q["op"]
        if op not in OPS:
            raise ValueError(f"op is one of {', '.join(OPS)}")
        a["op"][i], a["group"][i] = OPS[op], int(q.get("group", 0))
        a["a_den"][i] = a["b_den"][i] = 1
        if op in ("range", "moment"):
            lo, hi = q.get("lo"), q.get("hi")
            closed = q.get("closed", (True, True))
            flags = (LO_NONE if lo is None else (LO_CLOSED if closed[0] else 0)) | \
                    (HI_NONE if hi is None else (HI_CLOSED if closed[1] else 0))
            a["flags"][i] = flags
            if lo is not None:
                a["a_num"][i], a["a_den"][i] = _rational(lo, "lo")
            if hi is not None:
                a["b_num"][i], a["b_den"][i] = _rational(hi, "hi")
            a["power"][i] = int(q.get("k", 0))
        elif op == "quantile":
            num, den = _rational(q["p"], "p")
            if not 0 <= num <= den:
                raise ValueError("a quantile level p lies in [0, 1]")
            a["a_num"][i], a["a_den"][i] = num, den
            rule = q.get("rule", "disc")
            if rule not in RULES:
                raise ValueError(f"rule is one of {', '.join(RULES)}")
            a["rule"][i] = RULES[rule]
        elif op == "rank":
            a["a_num"][i] = _rational(q["r"], "r")[0]
        else:
            a["a_num"][i], a["b_num"][i] = int(q["r1"]), int(q["r2"])
            a["power"][i] = int(q.get("k", 1))
    return a


class Prefix:
    """The immutable prefix state of a sorted 1-D support (native): per group the distinct present values of positive
    mass and the prefix sums of mass * x^k; every request is answered natively (CPU) or by the CUDA kernel."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Histogram.prefix(...) or math_dist.prefix(...)")

    @classmethod
    def _wrap(cls, handle, hist):
        self = object.__new__(cls)
        self._handle, self.histogram, self._device = handle, hist, None
        self.scope = hist.scope if hist is not None else None
        self.name = hist.names[0] if hist is not None else None
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the prefix state is closed")
        return self._handle

    def _raw_info(self):
        out = _PrefixInfo()
        _check(_call("tr_dist_prefix_get_info", self._open(), C.byref(out)), what="prefix state")
        return out

    @property
    def info(self):
        i = self._raw_info()
        return dict(groups=int(i.ngroups), present=int(i.present), support=int(i.nsupport),
                    max_support=int(i.max_support), rows=int(i.rows), nulls=int(i.nulls), bytes=int(i.bytes),
                    degree=int(i.degree), mass={v: k for k, v in MASSES.items()}[int(i.mass)], scale=int(i.scale),
                    wscale=int(i.wscale), scoped=bool(i.scoped),
                    device_bytes=int(_call("tr_dist_prefix_device_bytes", C.byref(self.view()))))

    def view(self):
        v = _PrefixView()
        _check(_call("tr_dist_prefix_get_view", self._open(), C.byref(v)), what="prefix view")
        return v

    def support(self, g=0):
        """[(value, mass)] of group g in increasing value order, and its NULL rows: {"values", "nulls", "rows"}"""
        v = self.view()
        G = int(v.ngroups)
        g = int(g)
        if not 0 <= g < G:
            raise ValueError("group id out of range")
        off = np.ctypeslib.as_array((C.c_uint64 * (G + 1)).from_address(v.offset))
        b, e = int(off[g]), int(off[g + 1])
        w = int(v.degree) + 1
        vals, masses = [], []
        if e > b:
            x = np.ctypeslib.as_array((C.c_int64 * int(v.nsupport)).from_address(v.value))
            px = np.ctypeslib.as_array((C.c_int64 * (2 * int(v.nsupport) * w)).from_address(v.prefix))
            prev = 0
            for i in range(b, e):
                cum = _i128(px[2 * (i * w):2 * (i * w) + 2])
                vals.append(int(x[i])); masses.append(cum - prev); prev = cum
        rows = int(np.ctypeslib.as_array((C.c_uint64 * G).from_address(v.rows))[g])
        nulls = int(np.ctypeslib.as_array((C.c_uint64 * G).from_address(v.nulls))[g])
        return dict(values=list(zip(vals, masses)), nulls=nulls, rows=rows)

    def equal(self, other):
        return bool(_check(_call("tr_dist_prefix_equal", self._open(), other._open()), what="prefix comparison"))

    # ------------------------------------------------------------------ batches
    def device(self, budget=None):
        """the device copy (uploaded once, freed with this object): its bytes"""
        if self._device is None:
            gpu = _gpu()
            error, refusal = C.c_int(), _Refusal()
            h = gpu.lib().tr_mgpu_dist_prefix_upload(C.byref(self.view()), C.byref(_budget(budget)),
                                                     C.byref(refusal), C.byref(error))
            gpu.check(error.value, refusal, "prefix upload")
            if not h:
                raise MemoryError("prefix upload: allocation failed")
            self._device = h
        return int(_gpu().lib().tr_mgpu_dist_prefix_bytes(self._device))

    def query_arrays(self, reqs, *, backend="cpu", budget=None):
        """the raw RESULT_DTYPE answers of a REQUEST_DTYPE array (natively, every answer's status included)"""
        reqs = np.ascontiguousarray(reqs, dtype=REQUEST_DTYPE)
        out = np.zeros(len(reqs), dtype=RESULT_DTYPE)
        refusal = _Refusal()
        rp = reqs.ctypes.data if len(reqs) else None
        op = out.ctypes.data if len(out) else None
        if backend == "cpu":
            _check(_call("tr_dist_query", self._open(), rp, len(reqs), op, C.byref(_budget(budget)), C.byref(refusal)),
                   refusal, "prefix requests")
        elif backend == "cuda":
            self.device(budget)
            _gpu().run("tr_mgpu_dist_query", self._device, rp, len(reqs), op, C.byref(_budget(budget)),
                       C.byref(refusal), refusal=refusal, what="prefix requests on the device")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return out

    @staticmethod
    def _format(op, r):
        if int(r["status"]):
            _check(int(r["status"]), what=f"{op} request")
        num, den = _i256(r["num"]), _i128(r["den"])
        empty = bool(r["empty"])
        out = dict(empty=empty, mass=_i128(r["mass"]), total=_i128(r["total"]), lo=int(r["lo"]), hi=int(r["hi"]))
        if op == "range":
            out["fraction"] = None if empty else Fraction(num, den)
        elif op == "moment":
            out["value"] = num
        elif op in ("quantile", "rank"):
            out["value"] = None if empty else (num if den == 1 else Fraction(num, den))
        elif op == "trimmed":
            out["value"] = num
        else:
            out["value"] = None if empty else Fraction(num, den)
        return out

    def query(self, items, *, backend="cpu", budget=None):
        """answers of request dicts (see requests()): [{"empty", "mass", "total", "lo", "hi", and "fraction" (range)
        or "value"}]; a request the native code refuses raises"""
        items = list(items)
        res = self.query_arrays(requests(items), backend=backend, budget=budget)
        return [self._format(q["op"], r) for q, r in zip(items, res)]

    # ------------------------------------------------------------------ single requests
    def _one(self, q, backend="cpu"):
        return self.query([q], backend=backend)[0]

    def count(self, lo=None, hi=None, *, closed=(True, True), group=0, backend="cpu"):
        """the mass of the values in the interval lo .. hi (None: unbounded; closed per end)"""
        r = self._one(dict(op="range", lo=lo, hi=hi, closed=closed, group=group), backend)
        return dict(mass=r["mass"], total=r["total"], fraction=r["fraction"])

    def cdf(self, t, *, closed=True, group=0, backend="cpu"):
        """mass of x <= t (closed) or x < t, its fraction of the total (None when the group has no mass)"""
        return self.count(None, t, closed=(True, closed), group=group, backend=backend)

    def tail(self, t, *, closed=True, group=0, backend="cpu"):
        """mass of x >= t (closed) or x > t"""
        return self.count(t, None, closed=(closed, True), group=group, backend=backend)

    def moment(self, k, lo=None, hi=None, *, closed=(True, True), group=0, backend="cpu"):
        """sum of mass * x^k over the values in the interval (exact int, logical units)"""
        return self._one(dict(op="moment", k=k, lo=lo, hi=hi, closed=closed, group=group), backend)["value"]

    def quantile(self, p, rule="disc", *, group=0, backend="cpu"):
        """the exact quantile at level p (rules: disc = PERCENTILE_DISC, cont = PERCENTILE_CONT, lower, higher);
        None when the group has no mass"""
        return self._one(dict(op="quantile", p=p, rule=rule, group=group), backend)["value"]

    def value_at_rank(self, r, *, group=0, backend="cpu"):
        """the order statistic y_r (1 <= r <= total mass), None past the total"""
        return self._one(dict(op="rank", r=r, group=group), backend)["value"]

    def trimmed(self, r1, r2, k=1, *, group=0, backend="cpu"):
        """sum of y_r^k over the ranks r1..r2 (1-based, inclusive; r1 = r2 + 1 sums nothing)"""
        return self._one(dict(op="trimmed", r1=r1, r2=r2, k=k, group=group), backend)["value"]

    def trimmed_mean(self, r1, r2, k=1, *, group=0, backend="cpu"):
        return self._one(dict(op="trimmed_mean", r1=r1, r2=r2, k=k, group=group), backend)["value"]

    def close(self):
        if getattr(self, "_device", None):
            try:
                _gpu().lib().tr_mgpu_dist_prefix_free(self._device)
            finally:
                self._device = None
        if getattr(self, "_handle", None):
            _call("tr_dist_prefix_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Prefix(closed)"
        i = self.info
        return f"Prefix(groups={i['groups']}, support={i['support']}, mass={i['mass']}, degree={i['degree']})"
