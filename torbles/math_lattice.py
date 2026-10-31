# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LATTICE-001: bounded k-condition occupancy histograms (the exact bag count of every cell of k Boolean axes over a
BOOL-001 domain) and LATTICE-002: their subset-lattice transforms (zeta sums, Moebius inversion, inclusion-exclusion
counts, Moebius coefficients up to an order), natively on the CPU or with the CUDA kernels (torbles/math_lattice.c,
math_lattice_gpu.cu).

    from torbles import math_bool as mb, math_lattice as ml
    masks = mb.Masks.from_scope(scope, [mb.condition("x", ">", 5), mb.condition("d", "between", "1.5", "3.25")])
    h = ml.Histogram.build(masks, nulls="apart", backend="cpu")   # or nulls="three", backend="cuda"
    h = ml.Histogram.build(masks, axes=[1, 0, mb.var(0) & ~mb.var(1)])   # axes: conditions or predicates, in order
    h.counts()           # the 2^k (or 3^k) exact cell counts; h.cell((True, False)), h.table(), h.axis_counts()
    h.vector("complete") # the 2^k projection ("true_sets", "false_sets", "unknown_sets" on a three-valued histogram)
    ml.zeta(h)["values"]                       # intersections: points TRUE on at least S, for every S
    ml.mobius(ml.zeta(h)["values"])["values"]  # back to the exclusive cells
    ml.transform(v, "zeta_down" | "mobius_down" | ..., carrier="i128" | "i64" | "bigint", backend="cpu" | "cuda")
    ml.coefficients(v, sets, op="zeta_up")     # a few coefficients by their direct sums (no full transform)
    ml.interactions(v, order=2)                # Moebius (Harsanyi) coefficients of every coalition of size <= 2
    ml.counts(h, [[0, 1], [2]], projection="true_sets")   # all / any (inclusion-exclusion) / none / exactly

NULL convention (declared, never silent): "apart" keeps 2^k cells of the complete points and counts the points with
an UNKNOWN axis apart (h.unknown, per axis in axis_counts); "three" keeps 3^k cells with UNKNOWN as a third value
(digit 2). Axis i is bit i (digit i) of a cell id. Counts are exact native integers; transforms are exact signed
integers in a checked carrier (128-bit by default: OverflowError instead of wrapping; "i64": every input and output
must fit int64; "bigint": FND-009 big integers, exact for any input). Budgets raise AlgebraBudgetError before anything
is allocated, a stale scope StaleStateError. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from numbers import Integral

import numpy as np

from . import algq
from . import math_bool as mb
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Histogram", "transform", "zeta", "mobius", "coefficients", "interactions", "order_sets", "counts",
           "histogram_need", "transform_need", "coefficients_need", "counts_need", "MAX_AXES"]

MAX_AXES = 30
CONDITION, PROGRAM = 1, 2
NULLS = {"apart": 0, "three": 1}
PROJECTIONS = {"complete": 1, "true_sets": 2, "false_sets": 3, "unknown_sets": 4}
OPS = {"zeta_up": 1, "zeta_down": 2, "mobius_up": 3, "mobius_down": 4}
CARRIERS = {"i64": 1, "i128": 2, "bigint": 3}
KINDS = {0: "counts", 1: "rows", 2: "assignments"}
_NEED_CPU, _NEED_DEVICE = 0, 1
_U64 = (1 << 64) - 1
P = C.POINTER


class _Axis(C.Structure):
    _native_ = "tr_lattice_axis"
    _fields_ = [("kind", C.c_int32), ("index", C.c_int32), ("reserved0", C.c_int32), ("reserved1", C.c_int32)]


class _HistInfo(C.Structure):
    _native_ = "tr_lattice_hist_info"
    _fields_ = [(n, C.c_uint64) for n in ("ncells", "points", "complete", "unknown", "occupied", "bytes")] + [
        ("k", C.c_int32), ("nulls", C.c_int32), ("kind", C.c_int32), ("reserved", C.c_int32)]


class _Count(C.Structure):
    _native_ = "tr_lattice_count"
    _fields_ = [("vars", C.c_uint64), ("order", C.c_int32), ("reserved", C.c_int32), ("terms", C.c_uint64)] + [
        (f"{n}_{w}", C.c_int64) for n in ("all", "any", "none", "exactly", "peak") for w in ("lo", "hi")]


for _s, _n in ((_Axis, 16), (_HistInfo, 64), (_Count, 104)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        D, V = P(mb._Domain), P(mb._ProgramView)
        sig = {
            "tr_lattice_hist_build": (vp, [D, P(_Axis), i32, V, i32, i32, P(_Budget), P(_Refusal), e]),
            "tr_lattice_hist_need": (i32, [D, P(_Axis), i32, V, i32, i32, i32, P(_Need)]),
            "tr_lattice_hist_from_counts": (vp, [i32, i32, i32, u64, vp, vp, P(_Budget), P(_Refusal), e]),
            "tr_lattice_hist_merge": (vp, [vp, vp, P(_Budget), P(_Refusal), e]),
            "tr_lattice_hist_get_info": (i32, [vp, P(_HistInfo)]),
            "tr_lattice_hist_cells": (i32, [vp, vp, u64]),
            "tr_lattice_hist_axis_counts": (i32, [vp, vp, u64]),
            "tr_lattice_hist_equal": (i32, [vp, vp]),
            "tr_lattice_hist_vector": (i32, [vp, i32, vp, u64]),
            "tr_lattice_hist_vector_i128": (i32, [vp, i32, vp, u64]),
            "tr_lattice_hist_free": (None, [vp]),
            "tr_lattice_transform_need": (i32, [i32, i32, i32, P(_Need)]),
            "tr_lattice_transform": (i32, [i32, i32, i32, vp, vp, P(_Budget), P(_Refusal)]),
            "tr_lattice_transform_bigint": (i32, [i32, i32, vp, vp, P(_Budget), P(_Refusal)]),
            "tr_lattice_coefficients": (i32, [i32, i32, i32, vp, vp, i32, P(_Budget), vp, P(_Refusal)]),
            "tr_lattice_coefficients_need": (i32, [i32, i32, vp, i32, P(_Need)]),
            "tr_lattice_order_sets": (i32, [i32, i32, i32, P(_Budget), vp, u64, P(u64), P(_Refusal)]),
            "tr_lattice_counts": (i32, [vp, i32, vp, i32, P(_Budget), P(_Count), P(_Refusal)]),
            "tr_lattice_counts_need": (i32, [vp, i32, vp, i32, P(_Need)]),
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
    math_gpu.bind("tr_mgpu_lattice_hist", C.c_int, [P(mb._Domain), P(_Axis), C.c_int32, P(mb._ProgramView),
                                                   C.c_int32, C.c_int32, P(_Budget), C.c_void_p, C.c_void_p,
                                                   P(_Refusal)])
    math_gpu.bind("tr_mgpu_lattice_transform", C.c_int, [C.c_int32, C.c_int32, C.c_int32, C.c_void_p, C.c_void_p,
                                                        P(_Budget), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _choice(table, value, what):
    if value not in table:
        raise ValueError(f"{what} is one of {', '.join(map(repr, table))}")
    return table[value]


def _device_label(gpu):
    d = gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


# ---------------------------------------------------------------------- exact integer vectors (tr_math_i128)
def _from_words(words):
    """Python ints of an (n, 2) int64 array of two's complement (lo, hi) words"""
    lo, hi = words[:, 0], words[:, 1]
    if np.array_equal(hi, lo >> 63):                       # every value fits int64
        return lo.tolist()
    return [(h << 64) | (l & _U64) for l, h in zip(lo.tolist(), hi.tolist())]


def _words_of(values, what="values"):
    """(n, 2) int64 words of n exact integers (|v| < 2^127): an int64 / uint64 ndarray or a sequence of ints"""
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise TypeError(f"{what}: a one-dimensional integer array, or a sequence of Python ints")
        out = np.zeros((len(values), 2), dtype=np.int64)
        if values.dtype.kind == "u":
            v = np.ascontiguousarray(values, dtype=np.uint64)
            out[:, 0] = v.view(np.int64)                   # < 2^64: the high word stays 0
        else:
            v = np.ascontiguousarray(values, dtype=np.int64)
            out[:, 0] = v
            out[:, 1] = v >> 63                            # sign extension
        return out
    vals = list(values)
    out = np.zeros((len(vals), 2), dtype=np.int64)
    lo, hi = [], []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, Integral):
            raise TypeError(f"{what}: {v!r} is not an exact integer")
        v = int(v)
        if not -(1 << 127) <= v < (1 << 127):
            raise OverflowError(f"{what}: {v} does not fit the 128-bit carrier (use carrier='bigint')")
        x = v & ((1 << 128) - 1)
        lo.append(x & _U64)
        hi.append(x >> 64)
    if vals:
        out[:, 0] = np.array(lo, dtype=np.uint64).view(np.int64)
        out[:, 1] = np.array(hi, dtype=np.uint64).view(np.int64)
    return out


def _k_of(n, what="values"):
    if n < 1 or n & (n - 1):
        raise ValueError(f"{what}: 2^k values (a power of two), got {n}")
    k = n.bit_length() - 1
    if k > MAX_AXES:
        raise UnsupportedError(f"{what}: k = {k} > {MAX_AXES}")
    return k


def _vector_words(values, projection="complete"):
    """(k, words) of a Histogram's projection or of 2^k exact integers"""
    if isinstance(values, Histogram):
        return values.k, values._vector_i128(projection)
    words = _words_of(values)
    return _k_of(len(words)), words


def _sets(sets, k):
    """uint64 masks of sets of axes (iterables of indices, or one index)"""
    out = []
    for s in sets:
        s = [s] if isinstance(s, Integral) and not isinstance(s, bool) else list(s)
        idx = sorted(set(int(i) for i in s))
        if len(idx) != len(s):
            raise ValueError(f"set {s!r}: an axis appears twice")
        if any(not 0 <= i < k for i in idx):
            raise ValueError(f"set {s!r}: axes are 0..{k - 1}")
        out.append(sum(1 << i for i in idx))
    arr = np.array(out if out else [0], dtype=np.uint64)
    return arr, len(out)


def _members(mask, k):
    return tuple(i for i in range(k) if (int(mask) >> i) & 1)


# ---------------------------------------------------------------------- histograms (LATTICE-001)
def _axes(masks, axes):
    """native axes and program views: an axis is a condition index of the masks, a math_bool.Program over the masks'
    conditions, or a math_bool.Expr (compiled over them); one Program object read by two axes is evaluated once"""
    if axes is None:
        axes = list(range(masks.k))
    axes = list(axes)
    programs, index, described = [], {}, []
    arr = (_Axis * max(len(axes), 1))()
    for i, a in enumerate(axes):
        if isinstance(a, Integral) and not isinstance(a, bool):
            if not 0 <= int(a) < masks.k:
                raise ValueError(f"axis {i}: condition {a} is not one of the domain's 0..{masks.k - 1}")
            arr[i] = _Axis(CONDITION, int(a), 0, 0)
            described.append(("condition", int(a)))
            continue
        if isinstance(a, mb.Expr):
            a = mb.Program(a, masks.k)
        if not isinstance(a, mb.Program):
            raise TypeError(f"axis {i}: a condition index, a math_bool.Program or a math_bool.Expr")
        if a.k != masks.k:
            from .math_domains import DomainError
            raise DomainError(f"axis {i}: the program reads {a.k} conditions, the domain has {masks.k}")
        if id(a) not in index:
            index[id(a)] = len(programs)
            programs.append(a)
        arr[i] = _Axis(PROGRAM, index[id(a)], 0, 0)
        described.append(("program", index[id(a)]))
    views = (mb._ProgramView * max(len(programs), 1))()
    for j, p in enumerate(programs):
        views[j] = p.view()
    return arr, len(axes), views, programs, tuple(described)


class Histogram:
    """An immutable native occupancy histogram: the exact count of every cell of k Boolean axes (math_lattice.h)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Histogram.build(masks, ...), Histogram.from_counts(...) or h.merge(other)")

    @classmethod
    def _wrap(cls, handle, masks, axes, programs, backend, device):
        self = object.__new__(cls)
        self._handle, self.masks, self.axes, self._programs = handle, masks, axes, programs
        self.backend, self.device = backend, device
        i = self._raw_info()
        self.k, self.nulls, self.ncells = int(i.k), ("apart", "three")[i.nulls], int(i.ncells)
        self.kind = KINDS[int(i.kind)]
        return self

    @classmethod
    def build(cls, masks, axes=None, *, nulls="apart", backend="cpu", budget=None):
        """the histogram of the axes (default: every condition of the masks, in order) over the masks' domain:
        nulls "apart" (2^k cells of the complete points, the others counted apart) or "three" (3^k cells)"""
        if not isinstance(masks, mb.Masks):
            raise TypeError("masks: a math_bool.Masks (rows of a scope, or the 2^k assignments)")
        masks._open()
        masks.require_fresh("occupancy histogram")
        conv = _choice(NULLS, nulls, "nulls")
        arr, k, views, programs, described = _axes(masks, axes)
        dom = masks.domain()
        error, refusal = C.c_int32(), _Refusal()
        if _backend(backend) == "cpu":
            h = _call("tr_lattice_hist_build", C.byref(dom), arr, k, views, len(programs), conv,
                      C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            _check(error.value, refusal, "occupancy histogram")
            if not h:
                raise MemoryError("occupancy histogram: native allocation failed")
            return cls._wrap(h, masks, described, programs, "cpu", "native CPU")
        gpu = _gpu()
        need = _Need()
        _check(_call("tr_lattice_hist_need", C.byref(dom), arr, k, views, len(programs), conv, _NEED_DEVICE,
                     C.byref(need)), what="histogram needs")
        from . import math_budgets
        math_budgets.check(budget or Budget(), "occupancy histogram on the device", **_needs_dict(need))
        if k > MAX_AXES:
            raise UnsupportedError(f"occupancy histogram: {k} axes > {MAX_AXES}")
        ncells = 3 ** k if conv else 1 << k                # the output buffer the device fills
        cells = np.zeros(max(ncells, 1), dtype=np.uint64)
        axis = np.zeros(max(3 * k, 1), dtype=np.uint64)
        gpu.run("tr_mgpu_lattice_hist", C.byref(dom), arr, k, views, len(programs), conv, C.byref(_budget(budget)),
                cells.ctypes.data, axis.ctypes.data, C.byref(refusal), refusal=refusal,
                what="occupancy histogram on the device")
        h = _call("tr_lattice_hist_from_counts", k, conv, int(dom.kind), int(dom.selected), cells.ctypes.data,
                  axis.ctypes.data, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "occupancy histogram (device counts)")
        if not h:
            raise MemoryError("occupancy histogram: native allocation failed")
        return cls._wrap(h, masks, described, programs, *_device_label(gpu))

    @classmethod
    def from_counts(cls, k, cells, *, nulls="apart", points=None, axis_counts=None, budget=None):
        """a histogram of given cell counts (2^k or 3^k exact non-negative integers): points = the total (default:
        the sum of the cells); axis_counts = per axis (TRUE, FALSE, UNKNOWN) counts or None (derived). Checked
        consistent natively."""
        conv = _choice(NULLS, nulls, "nulls")
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= int(k) <= MAX_AXES:
            raise ValueError(f"k is an integer in 0..{MAX_AXES}")
        k = int(k)
        c = [int(x) for x in cells]
        want = 3 ** k if conv else 1 << k
        if len(c) != want:
            raise ValueError(f"{want} cell counts expected, got {len(c)}")
        if any(x < 0 or x > _U64 for x in c):
            raise ValueError("cell counts are integers in 0..2^64-1")
        arr = np.array(c, dtype=np.uint64)
        ax = None
        if axis_counts is not None:
            rows = [tuple(int(x) for x in a) for a in axis_counts]
            if len(rows) != k or any(len(r) != 3 or min(r) < 0 or max(r) > _U64 for r in rows):
                raise ValueError("axis_counts: k (TRUE, FALSE, UNKNOWN) triples of counts")
            ax = np.array([x for r in rows for x in r] or [0], dtype=np.uint64)
        if points is None:
            points = _U64                                  # TR_LATTICE_POINTS_SUM: the native sum of the cells
        elif not 0 <= int(points) < _U64:
            raise ValueError("points is an integer in 0..2^64-2")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_lattice_hist_from_counts", k, conv, 0, int(points), arr.ctypes.data,
                  ax.ctypes.data if ax is not None else None, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        _check(error.value, refusal, "histogram from counts")
        if not h:
            raise MemoryError("histogram: native allocation failed")
        return cls._wrap(h, None, tuple(("given", i) for i in range(k)), (), "cpu", "native CPU")

    def merge(self, other, *, budget=None):
        """the histogram of the union of two disjoint point sets (same k and NULL convention)"""
        self._fresh("merge"); other._fresh("merge")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_lattice_hist_merge", self._handle, other._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        _check(error.value, refusal, "histogram merge")
        if not h:
            raise MemoryError("histogram merge: native allocation failed")
        return Histogram._wrap(h, None, self.axes if self.axes == other.axes else
                               tuple(("given", i) for i in range(self.k)), (), "cpu", "native CPU")

    # ------------------------------------------------------------------ reading
    def _open(self):
        if not self._handle:
            raise ValueError("the histogram is closed")

    def _fresh(self, what):
        self._open()
        if self.masks is not None:
            self.masks.require_fresh(what)

    def require_fresh(self, what="histogram"):
        self._fresh(what)

    def _raw_info(self):
        self._open()
        i = _HistInfo()
        _check(_call("tr_lattice_hist_get_info", self._handle, C.byref(i)), what="histogram")
        return i

    @property
    def info(self):
        i = self._raw_info()
        return dict(k=int(i.k), nulls=self.nulls, cells=int(i.ncells), points=int(i.points),
                    complete=int(i.complete), unknown=int(i.unknown), occupied=int(i.occupied), bytes=int(i.bytes),
                    kind=self.kind, axes=self.axes, backend=self.backend, device=self.device)

    @property
    def points(self):
        return int(self._raw_info().points)

    @property
    def complete(self):
        return int(self._raw_info().complete)

    @property
    def unknown(self):
        return int(self._raw_info().unknown)

    def counts(self):
        """the exact count of every cell (uint64 array of 2^k or 3^k; cell id: bit / base-3 digit i = axis i)"""
        self._fresh("histogram cells")
        out = np.zeros(self.ncells, dtype=np.uint64)
        _check(_call("tr_lattice_hist_cells", self._handle, out.ctypes.data, self.ncells), what="histogram cells")
        return out

    def axis_counts(self):
        """per axis: {"true", "false", "unknown"} over every point"""
        self._fresh("histogram axis counts")
        out = np.zeros(max(3 * self.k, 1), dtype=np.uint64)
        _check(_call("tr_lattice_hist_axis_counts", self._handle, out.ctypes.data, 3 * self.k),
               what="histogram axis counts")
        return [dict(true=int(out[3 * i]), false=int(out[3 * i + 1]), unknown=int(out[3 * i + 2]))
                for i in range(self.k)]

    def cell_id(self, values):
        """the cell id of an assignment: k values True / False (and None = UNKNOWN for a three-valued histogram)"""
        values = tuple(values)
        if len(values) != self.k:
            raise ValueError(f"{self.k} axis values expected")
        cid, mult = 0, 1
        for i, v in enumerate(values):
            if v is None and self.nulls == "apart":
                raise ValueError("an 'apart' histogram has no UNKNOWN cells (its unknown points are counted apart)")
            if v is not None and not isinstance(v, (bool, np.bool_)) and v not in (0, 1):
                raise ValueError("axis values are True, False or None")
            if self.nulls == "apart":
                cid |= int(bool(v)) << i
            else:
                cid += (2 if v is None else int(bool(v))) * mult
                mult *= 3
        return cid

    def cell(self, values):
        """the exact count of one cell (values as for cell_id)"""
        return int(self.counts()[self.cell_id(values)])

    def table(self):
        """the occupied cells: [(axis values, count)] in cell-id order (None = UNKNOWN)"""
        out = []
        for cid, n in enumerate(self.counts().tolist()):
            if not n:
                continue
            if self.nulls == "apart":
                vals = tuple(bool((cid >> i) & 1) for i in range(self.k))
            else:
                vals, c = [], cid
                for _ in range(self.k):
                    c, d = divmod(c, 3)
                    vals.append((False, True, None)[d])
                vals = tuple(vals)
            out.append((vals, int(n)))
        return out

    def vector(self, projection="complete"):
        """the 2^k projection (uint64 counts): "complete", or on a three-valued histogram "true_sets" (SQL WHERE
        semantics), "false_sets", "unknown_sets" (math_lattice.h, Projections)"""
        self._fresh("histogram projection")
        out = np.zeros(1 << self.k, dtype=np.uint64)
        _check(_call("tr_lattice_hist_vector", self._handle, _choice(PROJECTIONS, projection, "projection"),
                     out.ctypes.data, len(out)), what="histogram projection")
        return out

    def _vector_i128(self, projection):
        self._fresh("histogram projection")
        out = np.zeros((1 << self.k, 2), dtype=np.int64)
        _check(_call("tr_lattice_hist_vector_i128", self._handle, _choice(PROJECTIONS, projection, "projection"),
                     out.ctypes.data, len(out)), what="histogram projection")
        return out

    def equal(self, other):
        self._open(); other._open()
        return bool(_check(_call("tr_lattice_hist_equal", self._handle, other._handle), what="histogram comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_lattice_hist_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Histogram(closed)"
        return f"Histogram(k={self.k}, nulls={self.nulls!r}, points={self.points}, {self.backend})"


def histogram_need(masks, axes=None, *, nulls="apart", backend="cpu"):
    """the needs a histogram build checks before allocating: {resource: need}"""
    masks._open()
    arr, k, views, programs, _ = _axes(masks, axes)
    out = _Need()
    _check(_call("tr_lattice_hist_need", C.byref(masks.domain()), arr, k, views, len(programs),
                 _choice(NULLS, nulls, "nulls"), _NEED_CPU if _backend(backend) == "cpu" else _NEED_DEVICE,
                 C.byref(out)), what="histogram needs")
    return _needs_dict(out)


# ---------------------------------------------------------------------- transforms (LATTICE-002)
def transform_need(k, op="zeta_up", carrier="i128"):
    out = _Need()
    _check(_call("tr_lattice_transform_need", int(k), _choice(OPS, op, "op"), _choice(CARRIERS, carrier, "carrier"),
                 C.byref(out)), what="transform needs")
    return _needs_dict(out)


def _bigint_transform(values, op, budget, projection):
    from . import math_carriers as mc
    vals = values
    if isinstance(values, Histogram):
        vals = values.vector(projection).tolist()
    vals = [int(v) for v in vals]
    k = _k_of(len(vals))
    keep, arr = mc._batch_handles(vals, budget)
    out, refusal = (C.c_void_p * len(vals))(), _Refusal()
    _check(_call("tr_lattice_transform_bigint", k, OPS[op], arr, out, C.byref(_budget(budget)), C.byref(refusal)),
           refusal, f"{op} transform (big integers)")
    del keep
    return k, [mc.BigInt._wrap(out[i], budget).to_int() for i in range(len(vals))]


def transform(values, op="zeta_up", *, carrier="i128", backend="cpu", budget=None, projection="complete"):
    """The fast transform `op` of 2^k exact integers (a Histogram's projection, a sequence of ints or an integer
    ndarray): {"values", "k", "op", "carrier", "cost" (the needs: 2^k values, k passes, bytes), "backend", "device"}.
    carrier "i128" (Python ints), "i64" (an int64 ndarray; every input and output must fit int64) or "bigint"."""
    op_code = _choice(OPS, op, "op")
    carrier_code = _choice(CARRIERS, carrier, "carrier")
    if isinstance(values, Histogram):
        values._fresh(f"{op} transform")
    if carrier == "bigint":
        if _backend(backend) == "cuda":
            raise UnsupportedError("the big-integer carrier has no CUDA path (use carrier='i128' on the device)")
        k, vals = _bigint_transform(values, op, budget, projection)
        return dict(values=vals, k=k, op=op, carrier=carrier, cost=transform_need(k, op, carrier), backend="cpu",
                    device="native CPU")
    k, words = _vector_words(values, projection)
    out = np.zeros_like(words)
    refusal = _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_lattice_transform", k, op_code, carrier_code, words.ctypes.data, out.ctypes.data,
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, f"{op} transform")
        label = ("cpu", "native CPU")
    else:
        gpu = _gpu()
        gpu.run("tr_mgpu_lattice_transform", k, op_code, carrier_code, words.ctypes.data, out.ctypes.data,
                C.byref(_budget(budget)), C.byref(refusal), refusal=refusal, what=f"{op} transform on the device")
        label = _device_label(gpu)
    vals = out[:, 0].copy() if carrier == "i64" else _from_words(out)
    return dict(values=vals, k=k, op=op, carrier=carrier, cost=transform_need(k, op, carrier), backend=label[0],
                device=label[1])


def zeta(values, direction="up", **kw):
    """superset sums ("up": points TRUE on at least S) or subset sums ("down")"""
    return transform(values, {"up": "zeta_up", "down": "zeta_down"}[direction], **kw)


def mobius(values, direction="up", **kw):
    """the inverse of zeta(direction): "up" gives the exclusive regions back, "down" the Harsanyi dividends"""
    return transform(values, {"up": "mobius_up", "down": "mobius_down"}[direction], **kw)


def coefficients_need(values, sets, op="zeta_up", projection="complete"):
    k = values.k if isinstance(values, Histogram) else _k_of(len(values))
    arr, n = _sets(sets, k)
    out = _Need()
    _check(_call("tr_lattice_coefficients_need", k, _choice(OPS, op, "op"), arr.ctypes.data, n, C.byref(out)),
           what="coefficient needs")
    return _needs_dict(out)


def coefficients(values, sets, op="zeta_up", *, carrier="i128", budget=None, projection="complete"):
    """(op v)(S) for each requested set S by its direct sum (2^(k-|S|) terms for the UP ops, 2^|S| for the DOWN
    ones), without the full transform: a list of ints"""
    op_code = _choice(OPS, op, "op")
    if carrier not in ("i64", "i128"):
        raise ValueError("coefficients take carrier 'i64' or 'i128'")
    k, words = _vector_words(values, projection)
    arr, n = _sets(sets, k)
    out = np.zeros((max(n, 1), 2), dtype=np.int64)
    refusal = _Refusal()
    _check(_call("tr_lattice_coefficients", k, op_code, CARRIERS[carrier], words.ctypes.data, arr.ctypes.data, n,
                 C.byref(_budget(budget)), out.ctypes.data, C.byref(refusal)), refusal, f"{op} coefficients")
    return _from_words(out[:n])


def order_sets(k, max_order, min_order=0, *, budget=None):
    """the sets of min_order..max_order of k axes, by size then increasing mask (tuples of axes)"""
    n = C.c_uint64()
    refusal = _Refusal()
    _check(_call("tr_lattice_order_sets", int(k), int(min_order), int(max_order), C.byref(_budget(budget)), None, 0,
                 C.byref(n), C.byref(refusal)), refusal, "coalitions")
    out = np.zeros(max(int(n.value), 1), dtype=np.uint64)
    _check(_call("tr_lattice_order_sets", int(k), int(min_order), int(max_order), C.byref(_budget(budget)),
                 out.ctypes.data, len(out), C.byref(n), C.byref(refusal)), refusal, "coalitions")
    return [_members(m, int(k)) for m in out[:int(n.value)].tolist()]


def interactions(values, order, op="mobius_down", *, min_order=0, carrier="i128", budget=None,
                 projection="complete"):
    """the coefficients of every coalition of size min_order..order: [(axes, coefficient)]; the default op is the
    Moebius (Harsanyi) transform of a set function v (v(S) = sum of the coefficients of the subsets of S)"""
    k = values.k if isinstance(values, Histogram) else _k_of(len(values))
    sets = order_sets(k, order, min_order, budget=budget)
    return list(zip(sets, coefficients(values, sets, op, carrier=carrier, budget=budget, projection=projection)))


def counts_need(hist, sets, projection="complete"):
    arr, n = _sets(sets, hist.k)
    out = _Need()
    _check(_call("tr_lattice_counts_need", hist._handle, _choice(PROJECTIONS, projection, "projection"),
                 arr.ctypes.data, n, C.byref(out)), what="count needs")
    return _needs_dict(out)


def counts(hist, sets, projection="complete", *, budget=None):
    """For each requested set S of axes, on a projection of the histogram: "all" (points TRUE on every axis of S),
    "any" (on at least one, by inclusion-exclusion over the intersections), "none", "exactly" (TRUE on S and on no
    other axis), "terms" (2^|S| - 1) and "peak" (the largest |Bonferroni partial sum| of "any")"""
    if not isinstance(hist, Histogram):
        raise TypeError("hist: a math_lattice.Histogram")
    hist._fresh("inclusion-exclusion counts")
    arr, n = _sets(sets, hist.k)
    out = (_Count * max(n, 1))()
    refusal = _Refusal()
    _check(_call("tr_lattice_counts", hist._handle, _choice(PROJECTIONS, projection, "projection"), arr.ctypes.data,
                 n, C.byref(_budget(budget)), out, C.byref(refusal)), refusal, "inclusion-exclusion counts")

    def i128(r, name):
        return (int(getattr(r, name + "_hi")) << 64) | (int(getattr(r, name + "_lo")) & _U64)
    return [dict(conditions=_members(r.vars, hist.k), order=int(r.order), terms=int(r.terms), all=i128(r, "all"),
                 any=i128(r, "any"), none=i128(r, "none"), exactly=i128(r, "exactly"), peak=i128(r, "peak"),
                 projection=projection) for r in out[:n]]
