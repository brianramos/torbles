# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""TOPO-001: declared occupancy complexes and local topology; TOPO-002: bounded cycles, homology and persistence over
a declared field, natively on the CPU or with the CUDA kernels (torbles/math_topo.c, math_topo_gpu.cu).

    from torbles import math_bank as mbk, math_topo as mt
    scope = mbk.Scope.from_table(db["t"], ["x", "y", "v"], rows=mask)
    grid = mt.Grid(["x", "y"], extent=[(0, 15), (0, 15)], adjacency=8, value="v")   # declared geometry only
    cx = mt.Complex.build(scope, grid, backend="cpu")      # or backend="cuda"
    cx.rows, cx.counts, cx.euler                          # row accounting, cells per dimension, V - E + F (- C)
    cx.cells(1), cx.incidence(2), cx.boundary(2, "gf2")   # anchors / axes, faces and signs, FIELD-001 Matrix
    cx.local(backend="cuda"), cx.adjacency(), cx.check(backend="cuda")   # star / link counts, dual graph, d o d = 0
    mt.counts(scope, grid, backend="cuda")                # the Euler counts without the complex
    h = mt.homology(cx, field="gf2")                      # ranks, Betti numbers, H_1 basis as edge cycles
    h = mt.homology(cx, field="q", cycles=(1, 2)); h = mt.homology(cx, field=3); mt.homology(cx, field="gf2", method="packed")
    p = mt.persistence(cx, field="gf2", ties="index", thresholds=[1, 2, 3])   # sublevel sets of v at declared levels
    p.pairs, p.essential, p.diagram(1), p.betti(level), p.levels(0)
    cc = mt.ChainComplex([d1, d2]); mt.homology(cc, field="gf2")   # an explicit integral chain complex (RP^2 ...)

The geometry is declared: integer coordinate columns (their logical values; never the stored codes, bit planes, the
torus layout of the columns or the row order), an extent of the top cells, the dual-graph adjacency. A row inside
the extent occupies the closed unit cell [x, x + 1] x [y, y + 1] (x [z, z + 1]); NULL coordinates and rows outside
the extent are counted apart; duplicates collapse. Betti numbers come only from the boundary matrices reduced over the
declared field (GF(2) also through FIELD-001's packed matrices, GF(p) for a verified prime, Q exactly), never from
counts. Persistence is of a bounded filtration (declared levels, a declared tie rule). Budgets raise
AlgebraBudgetError before anything is allocated (reductions: when a step or a byte past the limit would be needed,
returning nothing), a stale scope StaleStateError. Nothing is computed in Python. Opt-in: `import torbles` does not
load it.
"""
import ctypes as C
from collections import namedtuple
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_field as mf
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_certificates import _Staleness, _freshness
from .math_domains import DomainError
from .math_errors import RESOURCES, StaleStateError, UnsupportedError, check as _check

__all__ = ["Grid", "Complex", "ChainComplex", "Homology", "Persistence", "counts", "need", "homology",
           "homology_need", "persistence", "persistence_need", "ADJACENCY", "NEVER", "MAX_CHAIN"]

ADJACENCY = {2: (4, 8), 3: (6, 18, 26)}
FIELDS = {1: "gf2", 2: "gfp", 3: "q"}
METHODS = {"reduction": 1, "packed": 2}
SOURCES = {"sublevel": 1, "superlevel": 2, "levels": 3}
TIES = {"index": 1, "reverse": 2}
NEVER = -1
MAX_CHAIN = 7
_NONE = (1 << 64) - 1
_I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER


class _Grid(C.Structure):
    _native_ = "tr_topo_grid"
    _fields_ = [("dim", C.c_int32), ("adjacency", C.c_int32), ("col", C.c_int32 * 3), ("value", C.c_int32),
                ("lo", C.c_int64 * 3), ("hi", C.c_int64 * 3)]


class _Rows(C.Structure):
    _native_ = "tr_topo_rows"
    _fields_ = [(n, C.c_uint64) for n in ("selected", "null_rows", "outside_rows", "valid_rows", "occupied",
                                          "duplicates", "value_nulls", "valued")]


class _Counts(C.Structure):
    _native_ = "tr_topo_counts"
    _fields_ = [("rows", _Rows), ("positions", C.c_uint64), ("candidates", C.c_uint64), ("n", C.c_uint64 * 4),
                ("boundary", C.c_uint64 * 4), ("euler", C.c_int64), ("boundary_euler", C.c_int64),
                ("dim", C.c_int32), ("adjacency", C.c_int32)]


class _View(C.Structure):
    _native_ = "tr_topo_view"
    _fields_ = [("dim", C.c_int32), ("adjacency", C.c_int32)] + [
        (n, C.c_uint64 * 3) for n in ("ext", "kdim", "kstride", "pstride")] + [
        ("positions", C.c_uint64), ("words", C.c_uint64), ("candidates", C.c_uint64), ("n", C.c_uint64 * 4),
        ("occupancy", C.c_void_p), ("cells", C.c_void_p * 4), ("face", C.c_void_p * 4), ("sign", C.c_void_p * 4)]


class _Info(C.Structure):
    _native_ = "tr_topo_info"
    _fields_ = [("counts", _Counts), ("cells", C.c_uint64), ("incidences", C.c_uint64), ("bytes", C.c_uint64),
                ("lo", C.c_int64 * 3), ("hi", C.c_int64 * 3), ("col", C.c_int32 * 3), ("value", C.c_int32),
                ("backend", C.c_int32), ("has_value", C.c_int32)]


class _Local(C.Structure):
    _native_ = "tr_topo_local"
    _fields_ = [("star", C.c_uint8 * 4), ("link_euler", C.c_int8), ("boundary", C.c_uint8), ("degree", C.c_uint8),
                ("dim", C.c_uint8)]


class _LocalTotals(C.Structure):
    _native_ = "tr_topo_local_totals"
    _fields_ = [("cells", C.c_uint64), ("boundary", C.c_uint64 * 4), ("dual_edges", C.c_uint64),
                ("link_euler_sum", C.c_int64 * 4)]


class _Adjacency(C.Structure):
    _native_ = "tr_topo_adjacency"
    _fields_ = [("tops", C.c_uint64), ("edges", C.c_uint64), ("components", C.c_uint64), ("adjacency", C.c_int32),
                ("reserved", C.c_int32)]


class _Dd(C.Structure):
    _native_ = "tr_topo_dd"
    _fields_ = [(n, C.c_uint64 * 4) for n in ("checked", "violations_z", "violations_gf2", "witness_z",
                                              "witness_gf2")] + [
        ("invalid", C.c_uint64), ("terms", C.c_uint64), ("holds_z", C.c_int32), ("holds_gf2", C.c_int32)]


class _Field(C.Structure):
    _native_ = "tr_topo_field"
    _fields_ = [("field", C.c_int32), ("method", C.c_int32), ("prime", C.c_uint64), ("cycles", C.c_uint32),
                ("reserved", C.c_int32)]


class _Filtration(C.Structure):
    _native_ = "tr_topo_filtration"
    _fields_ = [("source", C.c_int32), ("ties", C.c_int32), ("nlevels", C.c_uint64), ("thresholds", C.c_void_p),
                ("levels", C.c_void_p)]


class _HomologyInfo(C.Structure):
    _native_ = "tr_topo_homology_info"
    _fields_ = [(n, C.c_int32) for n in ("top", "field", "method", "kind")] + [
        ("prime", C.c_uint64), ("n", C.c_uint64 * 8), ("rank", C.c_uint64 * 9), ("betti", C.c_uint64 * 8),
        ("euler", C.c_int64), ("steps", C.c_uint64), ("peak_bytes", C.c_uint64), ("classes", C.c_uint64),
        ("entries", C.c_uint64), ("cycles", C.c_uint32), ("reserved", C.c_int32)]


class _Class(C.Structure):
    _native_ = "tr_topo_class"
    _fields_ = [("dim", C.c_int32), ("reserved", C.c_int32), ("generator", C.c_uint64), ("offset", C.c_uint64),
                ("length", C.c_uint64)]


class _PersistenceInfo(C.Structure):
    _native_ = "tr_topo_persistence_info"
    _fields_ = [(n, C.c_int32) for n in ("top", "field", "source", "ties")] + [
        ("prime", C.c_uint64), ("nlevels", C.c_uint64), ("entered", C.c_uint64 * 8), ("never", C.c_uint64 * 8),
        ("rank", C.c_uint64 * 9), ("betti", C.c_uint64 * 8)] + [
        (n, C.c_uint64) for n in ("pairs", "essential", "zero_length", "steps", "peak_bytes")]


class _Pair(C.Structure):
    _native_ = "tr_topo_pair"
    _fields_ = [("dim", C.c_int32), ("reserved", C.c_int32), ("birth", C.c_int64), ("death", C.c_int64),
                ("birth_cell", C.c_uint64), ("death_cell", C.c_uint64)]


class _ChainInfo(C.Structure):
    _native_ = "tr_topo_chain_info"
    _fields_ = [("top", C.c_int32), ("reserved", C.c_int32), ("n", C.c_uint64 * 8), ("nnz", C.c_uint64 * 8),
                ("euler", C.c_int64), ("bytes", C.c_uint64)]


class _States(C.Structure):
    _native_ = "tr_topo_states"
    _fields_ = [("dim", C.c_int32), ("has_value", C.c_int32), ("positions", C.c_uint64), ("words", C.c_uint64),
                ("n", C.c_uint64 * 4), ("occupancy", C.c_void_p), ("cells", C.c_void_p * 4), ("face", C.c_void_p * 4),
                ("sign", C.c_void_p * 4), ("vmin", C.c_void_p), ("vmax", C.c_void_p), ("rows", _Rows)]


for _s, _n in ((_Grid, 72), (_Rows, 64), (_Counts, 168), (_View, 264), (_Info, 264), (_Local, 8), (_LocalTotals, 80),
               (_Adjacency, 32), (_Dd, 184), (_Field, 24), (_Filtration, 32), (_HomologyInfo, 272), (_Class, 32),
               (_PersistenceInfo, 336), (_Pair, 40), (_ChainInfo, 152), (_States, 240)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_topo_grid_check": (i32, [vp, P(_Grid)]),
            "tr_topo_scan_needs": (i32, [vp, P(_Grid), i32, P(_Need)]),
            "tr_topo_build_needs": (i32, [vp, P(_Grid), P(_Counts), i32, P(_Need)]),
            "tr_topo_count": (i32, [vp, P(_Grid), B, P(_Counts), R]),
            "tr_topo_build": (vp, [vp, P(_Grid), B, R, e]),
            "tr_topo_adopt": (vp, [vp, P(_Grid), P(_States), B, R, e]),
            "tr_topo_free": (None, [vp]),
            "tr_topo_get_info": (i32, [vp, P(_Info)]),
            "tr_topo_get_view": (i32, [vp, P(_View)]),
            "tr_topo_get_grid": (i32, [vp, P(_Grid)]),
            "tr_topo_cells": (i32, [vp, i32, vp, vp, vp, u64]),
            "tr_topo_values": (i32, [vp, vp, vp, u64]),
            "tr_topo_local_get": (i32, [vp, B, vp, u64, P(_LocalTotals), R]),
            "tr_topo_local_view": (i32, [P(_View), vp, u64, P(_LocalTotals)]),
            "tr_topo_adjacency_get": (i32, [vp, i32, B, vp, vp, u64, P(_Adjacency), R]),
            "tr_topo_fresh": (i32, [vp, vp, P(_Staleness)]),
            "tr_topo_equal": (i32, [vp, vp]),
            "tr_topo_check": (i32, [vp, B, P(_Dd), R]),
            "tr_topo_check_view": (i32, [P(_View), B, P(_Dd), R]),
            "tr_topo_boundary": (i32, [vp, i32, u64, B, vp, vp, vp, u64, R]),
            "tr_topo_boundary_packed": (i32, [vp, vp, i32, B, P(mf._Matrix), R]),
            "tr_topo_chain_create": (vp, [i32, vp, vp, vp, vp, B, R, e]),
            "tr_topo_chain_free": (None, [vp]),
            "tr_topo_chain_get_info": (i32, [vp, P(_ChainInfo)]),
            "tr_topo_reduce_need": (i32, [vp, vp, P(_Field), P(_Filtration), P(_Need)]),
            "tr_topo_homology_build": (vp, [vp, vp, P(_Field), B, R, e]),
            "tr_topo_homology_get_info": (i32, [vp, P(_HomologyInfo)]),
            "tr_topo_homology_classes": (i32, [vp, vp, u64]),
            "tr_topo_homology_entries": (i32, [vp, vp, vp, u64]),
            "tr_topo_homology_free": (None, [vp]),
            "tr_topo_persistence_build": (vp, [vp, vp, P(_Field), P(_Filtration), B, R, e]),
            "tr_topo_persistence_get_info": (i32, [vp, P(_PersistenceInfo)]),
            "tr_topo_persistence_pairs": (i32, [vp, vp, u64]),
            "tr_topo_persistence_betti": (i32, [vp, C.c_int64, vp, u64]),
            "tr_topo_persistence_levels": (i32, [vp, i32, vp, u64]),
            "tr_topo_persistence_free": (None, [vp]),
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
    B, R = P(_Budget), P(_Refusal)
    math_gpu.bind("tr_mgpu_topo_counts", C.c_int, [P(mbk._View), P(_Grid), B, P(_Counts), R])
    math_gpu.bind("tr_mgpu_topo_build", C.c_int, [P(mbk._View), P(_Grid), B, P(_States), R])
    math_gpu.bind("tr_mgpu_topo_local", C.c_int, [P(_View), B, C.c_void_p, C.c_uint64, P(_LocalTotals), R])
    math_gpu.bind("tr_mgpu_topo_dd", C.c_int, [P(_View), B, P(_Dd), R])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _device_name():
    from . import math_gpu
    info = math_gpu.device()
    return ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]


def _ptr(a):
    return None if a is None else a.ctypes.data


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _int(value, what):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be an exact integer")
    value = int(value)
    if not _I64[0] <= value <= _I64[1]:
        raise OverflowError(f"{what} does not fit int64")
    return value


def _rows_dict(r):
    return {n: int(getattr(r, n)) for n, _ in _Rows._fields_}


def _counts_dict(c, backend="cpu", device="native CPU"):
    dim = int(c.dim)
    return dict(rows=_rows_dict(c.rows), positions=int(c.positions), candidates=int(c.candidates),
                cells=[int(c.n[k]) for k in range(dim + 1)], boundary=[int(c.boundary[k]) for k in range(dim + 1)],
                euler=int(c.euler), boundary_euler=int(c.boundary_euler), dim=dim, adjacency=int(c.adjacency),
                backend=backend, device=device)


# ---------------------------------------------------------------------- declared geometry
class Grid:
    """The declared geometry of a complex: coords (2 or 3 distinct integer columns of the scope, in axis order),
    extent ((lo, hi) per axis: the top cells lo <= x <= hi, inclusive, in logical units), adjacency (4 / 8 in 2D,
    6 / 18 / 26 in 3D: the dual graph), value (a column whose values filtrations threshold, or None)."""

    def __init__(self, coords, extent, adjacency, value=None):
        coords = list(coords)
        if len(coords) not in (2, 3):
            raise ValueError("declare 2 or 3 coordinate columns")
        if not all(isinstance(c, str) for c in coords) or len(set(coords)) != len(coords):
            raise ValueError("coordinates are distinct column names (the declared geometry; no implicit layout)")
        extent = [tuple(e) for e in extent]
        if len(extent) != len(coords) or any(len(e) != 2 for e in extent):
            raise ValueError("one (lo, hi) extent per coordinate")
        extent = [(_int(lo, "an extent bound"), _int(hi, "an extent bound")) for lo, hi in extent]
        if any(hi < lo for lo, hi in extent):
            raise ValueError("an extent has lo <= hi")
        if adjacency not in ADJACENCY[len(coords)]:
            raise ValueError(f"the adjacency of a {len(coords)}D grid is one of {ADJACENCY[len(coords)]}")
        if value is not None and not isinstance(value, str):
            raise ValueError("value is a column name or None")
        self.coords, self.extent, self.adjacency, self.value = tuple(coords), tuple(extent), int(adjacency), value
        self.dim = len(coords)

    def _raw(self, scope):
        names = list(scope.names)
        for c in self.coords + ((self.value,) if self.value else ()):
            if c not in names:
                raise ValueError(f"the scope has no column {c!r}")
        if scope.source == "table":
            for c in self.coords:
                codec = scope.table.codecs.get(c) if hasattr(scope.table, "codecs") else None
                if codec is not None and codec.kind != "int":
                    raise DomainError(f"{c!r}: a {codec.kind} column is not a coordinate (declare integer columns)")
        g = _Grid()
        g.dim, g.adjacency = self.dim, self.adjacency
        for a in range(3):
            g.col[a] = names.index(self.coords[a]) if a < self.dim else -1
            g.lo[a] = self.extent[a][0] if a < self.dim else 0
            g.hi[a] = self.extent[a][1] if a < self.dim else 0
        g.value = names.index(self.value) if self.value else -1
        return g

    def __repr__(self):
        return (f"Grid({list(self.coords)}, extent={list(self.extent)}, adjacency={self.adjacency}, "
                f"value={self.value!r})")


def counts(scope, grid, *, backend="cpu", budget=None):
    """the row accounting, the cells and boundary cells per dimension and the Euler characteristics (no complex)"""
    scope.require_fresh("topology counts")
    raw, b, refusal, out = grid._raw(scope), _budget(budget), _Refusal(), _Counts()
    if _backend(backend) == "cpu":
        _check(_call("tr_topo_count", scope._handle, C.byref(raw), C.byref(b), C.byref(out), C.byref(refusal)),
               refusal, "topology counts")
        return _counts_dict(out)
    gpu = _gpu()
    view = scope.view()
    gpu.run("tr_mgpu_topo_counts", C.byref(view), C.byref(raw), C.byref(b), C.byref(out), C.byref(refusal),
            refusal=refusal, what="topology counts on the device")
    return _counts_dict(out, *_device_name())


def need(scope, grid, *, backend="cpu"):
    """what each phase of a build checks: the scan, then (after a native scan) the complex with its FND-007 record
    (CPU), or the device copies (build) and the adopted complex (adopt) on CUDA"""
    scope.require_fresh("topology need")
    raw, device = grid._raw(scope), int(_backend(backend) == "cuda")
    scan, build, c, refusal = _Need(), _Need(), _Counts(), _Refusal()
    _check(_call("tr_topo_scan_needs", scope._handle, C.byref(raw), device, C.byref(scan)), what="topology need")
    huge = Budget(**{r: (1 << 64) - 1 for r in RESOURCES})
    _check(_call("tr_topo_count", scope._handle, C.byref(raw), C.byref(huge.raw), C.byref(c), C.byref(refusal)),
           refusal, "topology need")
    _check(_call("tr_topo_build_needs", scope._handle, C.byref(raw), C.byref(c), device, C.byref(build)),
           what="topology need")
    out = dict(scan=_needs(scan), build=_needs(build), counts=_counts_dict(c))
    if device:                                       # the device build's arrays are adopted into a host complex
        adopt = _Need()
        _check(_call("tr_topo_build_needs", scope._handle, C.byref(raw), C.byref(c), 0, C.byref(adopt)),
               what="topology need")
        out["adopt"] = _needs(adopt)
    return out


# ---------------------------------------------------------------------- complexes
Boundary = namedtuple("Boundary", "shape rows cols values field")


def _dense(self):
    out = np.zeros(self.shape, dtype=np.int64)
    out[self.rows.astype(np.int64), self.cols.astype(np.int64)] = self.values
    return out


Boundary.dense = _dense


class Complex:
    """An immutable cubical complex of a declared grid over the rows of an FND-004 scope (native)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Complex.build(scope, grid)")

    @classmethod
    def build(cls, scope, grid, *, backend="cpu", budget=None):
        scope.require_fresh("complex build")
        raw, b, refusal, error = grid._raw(scope), _budget(budget), _Refusal(), C.c_int32()
        self = object.__new__(cls)
        self._handle = None
        if _backend(backend) == "cpu":
            handle = _call("tr_topo_build", scope._handle, C.byref(raw), C.byref(b), C.byref(refusal), C.byref(error))
            self._handle = _made(handle, error, refusal, "complex")
            self.backend, self.device = "cpu", "native CPU"
        else:
            gpu = _gpu()
            view = scope.view()
            c = _Counts()
            gpu.run("tr_mgpu_topo_counts", C.byref(view), C.byref(raw), C.byref(b), C.byref(c), C.byref(refusal),
                    refusal=refusal, what="complex counts on the device")
            host = _Need()                      # the adopted complex, checked before the host arrays exist
            _check(_call("tr_topo_build_needs", scope._handle, C.byref(raw), C.byref(c), 0, C.byref(host)),
                   what="complex")
            _check(_call("tr_math_budget_check", C.byref(b), C.byref(host), C.byref(refusal)), refusal, "complex")
            st, keep = _States(), []
            st.dim, st.has_value, st.positions = raw.dim, int(raw.value >= 0), c.positions
            st.words = (c.positions + 63) // 64
            occ = np.zeros(max(st.words, 1), dtype=np.uint64)
            st.occupancy = occ.ctypes.data
            keep.append(occ)
            for k in range(raw.dim + 1):
                n = int(c.n[k])
                st.n[k] = n
                cells = np.zeros(max(n, 1), dtype=np.uint64)
                st.cells[k] = cells.ctypes.data
                keep.append(cells)
                if k >= 1:
                    face = np.zeros(max(2 * k * n, 1), dtype=np.uint32)
                    sign = np.zeros(max(2 * k * n, 1), dtype=np.int8)
                    st.face[k], st.sign[k] = face.ctypes.data, sign.ctypes.data
                    keep += [face, sign]
            if raw.value >= 0:
                tops = int(c.n[raw.dim])
                vmin, vmax = np.zeros(max(tops, 1), dtype=np.int64), np.zeros(max(tops, 1), dtype=np.int64)
                st.vmin, st.vmax = vmin.ctypes.data, vmax.ctypes.data
                keep += [vmin, vmax]
            gpu.run("tr_mgpu_topo_build", C.byref(view), C.byref(raw), C.byref(b), C.byref(st), C.byref(refusal),
                    refusal=refusal, what="complex build on the device")
            handle = _call("tr_topo_adopt", scope._handle, C.byref(raw), C.byref(st), C.byref(b), C.byref(refusal),
                           C.byref(error))
            self._handle = _made(handle, error, refusal, "complex (device build)")
            self.backend, self.device = _device_name()
        self.scope, self.grid = scope, grid
        return self

    # ------------------------------------------------------------------ freshness and reading
    def _open(self):
        if not self._handle:
            raise ValueError("the complex is closed")
        return self._handle

    def _require(self, what):
        self._open()
        self.scope.require_fresh(what)

    def fresh(self, scope=None):
        """FND-007 freshness against a scope (default: the build's scope as it is now): Freshness(fresh, reasons,
        first, column, ...); a scope over other columns reports "columns" (another layout)"""
        current = self.scope if scope is None else scope
        if not self.scope.fresh() and scope is None:
            current.require_fresh("complex freshness")
        st = _Staleness()
        rc = _call("tr_topo_fresh", self._open(), current._handle, C.byref(st))
        if rc not in (0, -13):
            _check(rc, what="complex freshness")
        return _freshness(st)

    def require_fresh(self, scope=None, what="complex"):
        """None, or StaleStateError (with .reasons) when the scope differs from the one the complex records"""
        f = self.fresh(scope)
        if not f.fresh:
            err = StaleStateError(f"{what}: the complex was built on another scope or layout ({', '.join(f.reasons)})")
            err.reasons, err.freshness = f.reasons, f
            raise err

    def _info(self):
        out = _Info()
        _check(_call("tr_topo_get_info", self._open(), C.byref(out)), what="complex")
        return out

    @property
    def info(self):
        i = self._info()
        d = _counts_dict(i.counts, self.backend, self.device)
        d.update(cells_total=int(i.cells), incidences=int(i.incidences), bytes=int(i.bytes),
                 has_value=bool(i.has_value), built_by=("cpu", "device")[i.backend])
        return d

    @property
    def dim(self):
        return self.grid.dim

    @property
    def rows(self):
        self._require("complex")
        return _rows_dict(self._info().counts.rows)

    @property
    def counts(self):
        """cells per dimension [n_0, .., n_dim]"""
        self._require("complex")
        c = self._info().counts
        return [int(c.n[k]) for k in range(self.dim + 1)]

    @property
    def euler(self):
        self._require("complex")
        return int(self._info().counts.euler)

    def view(self):
        """the native tr_topo_view (pointers into this complex; valid while it lives)"""
        v = _View()
        _check(_call("tr_topo_get_view", self._open(), C.byref(v)), what="complex view")
        return v

    def cells(self, k):
        """the k-cells in their canonical order: anchors (n_k x dim logical coordinates of the lower corner), axes
        (bit a: the cell spans [x_a, x_a + 1]), linear (Khalimsky linear indices)"""
        self._require("complex cells")
        k = self._dimension(k)
        n = self.counts[k]
        anchors = np.zeros((max(n, 1), self.dim), dtype=np.int64)
        axes = np.zeros(max(n, 1), dtype=np.uint8)
        linear = np.zeros(max(n, 1), dtype=np.uint64)
        _check(_call("tr_topo_cells", self._handle, k, anchors.ctypes.data, axes.ctypes.data, linear.ctypes.data,
                     max(n, 1)), what="complex cells")
        return dict(anchor=anchors[:n], axes=axes[:n], linear=linear[:n])

    def _dimension(self, k):
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= int(k) <= self.dim:
            raise ValueError(f"a dimension 0 .. {self.dim}")
        return int(k)

    def incidence(self, k):
        """the boundary relation of the k-cells (k >= 1): faces (n_k x 2k indices of (k-1)-cells: slot 2i the minus,
        2i + 1 the plus face of the i-th spanning axis) and their orientation signs -(-1)^i, (-1)^i"""
        self._require("complex incidence")
        k = self._dimension(k)
        if k < 1:
            raise ValueError("0-cells have no faces")
        v = self.view()
        n = int(v.n[k])
        faces = np.ctypeslib.as_array((C.c_uint32 * max(2 * k * n, 1)).from_address(v.face[k])) if n else \
            np.zeros(0, dtype=np.uint32)
        signs = np.ctypeslib.as_array((C.c_int8 * max(2 * k * n, 1)).from_address(v.sign[k])) if n else \
            np.zeros(0, dtype=np.int8)
        return faces[:2 * k * n].reshape(n, 2 * k).copy(), signs[:2 * k * n].reshape(n, 2 * k).copy()

    def boundary(self, k, field="z", *, budget=None):
        """d_k (rows: the (k-1)-cells, columns: the k-cells): field "z" / None (integer signs) or a prime p
        (residues) as a Boundary (shape, rows, cols, values; .dense()), or "gf2": FIELD-001's packed Matrix"""
        self._require("boundary matrix")
        k = self._dimension(k)
        if k < 1:
            raise ValueError("d_k for k >= 1")
        b, refusal = _budget(budget), _Refusal()
        nk1, nk = self.counts[k - 1], self.counts[k]
        if isinstance(field, str) and field.lower() == "gf2":
            m = mf.Matrix(nk1, nk)
            raw = m._raw()
            _check(_call("tr_topo_boundary_packed", self._handle, None, k, C.byref(b), C.byref(raw), C.byref(refusal)),
                   refusal, "packed boundary matrix")
            return m
        prime = 0
        if field not in (None, "z"):
            if isinstance(field, bool) or not isinstance(field, Integral) or int(field) < 2:
                raise ValueError("field is 'z', 'gf2' or a prime p")
            prime = int(field)
        m = 2 * k * nk
        rows, cols, vals = (np.zeros(max(m, 1), dtype=np.uint64), np.zeros(max(m, 1), dtype=np.uint64),
                            np.zeros(max(m, 1), dtype=np.int64))
        _check(_call("tr_topo_boundary", self._handle, k, prime, C.byref(b), rows.ctypes.data, cols.ctypes.data,
                     vals.ctypes.data, max(m, 1), C.byref(refusal)), refusal, "boundary matrix")
        return Boundary((nk1, nk), rows[:m], cols[:m], vals[:m], "z" if prime == 0 else prime)

    def values(self):
        """per top cell the extremes of its rows' present values (logical units) and whether it has one"""
        self._require("complex values")
        n = self.counts[self.dim]
        vmin, vmax = np.zeros(max(n, 1), dtype=np.int64), np.zeros(max(n, 1), dtype=np.int64)
        _check(_call("tr_topo_values", self._handle, vmin.ctypes.data, vmax.ctypes.data, max(n, 1)),
               what="complex values (declare a value column)")
        present = vmin[:n] <= vmax[:n]
        return dict(min=vmin[:n], max=vmax[:n], present=present)

    def local(self, *, backend="cpu", budget=None, cells=True):
        """per-cell local invariants of every cell (dimension-major: the 0-cells, then the 1-cells, ...): star (the
        j-cofaces in the complex, j = 0..3), link_euler, boundary, degree (top cells: dual-graph neighbours), dim;
        with the totals (boundary cells per dimension, dual-graph edges, link Euler sums)"""
        self._require("local invariants")
        total = sum(self.counts)
        out = (_Local * max(total, 1))()
        tot, b, refusal = _LocalTotals(), _budget(budget), _Refusal()
        ptr = C.addressof(out) if cells else None
        if _backend(backend) == "cpu":
            _check(_call("tr_topo_local_get", self._handle, C.byref(b), ptr, max(total, 1), C.byref(tot),
                         C.byref(refusal)), refusal, "local invariants")
            backend_name, device = "cpu", "native CPU"
        else:
            gpu = _gpu()
            v = self.view()
            gpu.run("tr_mgpu_topo_local", C.byref(v), C.byref(b), ptr, max(total, 1), C.byref(tot), C.byref(refusal),
                    refusal=refusal, what="local invariants on the device")
            backend_name, device = _device_name()
        result = dict(totals=dict(cells=int(tot.cells), boundary=[int(tot.boundary[k]) for k in range(self.dim + 1)],
                                  dual_edges=int(tot.dual_edges),
                                  link_euler_sum=[int(tot.link_euler_sum[k]) for k in range(self.dim + 1)]),
                      backend=backend_name, device=device)
        if cells:
            raw = np.ctypeslib.as_array(C.cast(out, P(C.c_uint8)), shape=(max(total, 1) * 8,)).reshape(-1, 8)[:total]
            result.update(star=raw[:, 0:4].copy(), link_euler=raw[:, 4].view(np.int8).copy(),
                          boundary=raw[:, 5].astype(bool), degree=raw[:, 6].copy(), dim=raw[:, 7].copy(),
                          offsets=np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64))
        return result

    def adjacency(self, adjacency=None, *, pairs=False, budget=None):
        """the dual graph of the occupied top cells under the declared (or the given) adjacency: edges, components,
        labels (the smallest top-cell index of each component), the adjacent pairs (i < j) when pairs=True"""
        self._require("dual graph")
        adj = 0 if adjacency is None else int(adjacency)
        if adjacency is not None and adj not in ADJACENCY[self.dim]:
            raise ValueError(f"the adjacency of a {self.dim}D grid is one of {ADJACENCY[self.dim]}")
        n = self.counts[self.dim]
        labels = np.zeros(max(n, 1), dtype=np.uint32)
        out, b, refusal = _Adjacency(), _budget(budget), _Refusal()
        _check(_call("tr_topo_adjacency_get", self._handle, adj, C.byref(b), labels.ctypes.data, None, 0, C.byref(out),
                     C.byref(refusal)), refusal, "dual graph")
        listed = None
        if pairs:
            e = int(out.edges)
            buf = np.zeros(max(2 * e, 2), dtype=np.uint32)
            _check(_call("tr_topo_adjacency_get", self._handle, adj, C.byref(b), labels.ctypes.data, buf.ctypes.data,
                         max(e, 1), C.byref(out), C.byref(refusal)), refusal, "dual graph")
            listed = buf[:2 * e].reshape(e, 2)
        return dict(adjacency=int(out.adjacency), tops=int(out.tops), edges=int(out.edges),
                    components=int(out.components), labels=labels[:n], pairs=listed)

    def check(self, *, backend="cpu", budget=None):
        """d_{k-1} d_k = 0 over Z and over GF(2) for every k-cell (k = 2 .. dim), from the stored incidences"""
        self._require("boundary check")
        return check_view(self.view(), backend=backend, budget=budget)

    def equal(self, other):
        return bool(_check(_call("tr_topo_equal", self._open(), other._open()), what="complex comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_topo_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not getattr(self, "_handle", None):
            return "Complex(closed)"
        return f"Complex({self.grid!r}, cells={self.counts}, euler={self.euler}, backend={self.backend!r})"


def check_view(view, *, backend="cpu", budget=None):
    """the composition check of a native tr_topo_view (a complex's, or one assembled from arrays)"""
    out, b, refusal = _Dd(), _budget(budget), _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_topo_check_view", C.byref(view), C.byref(b), C.byref(out), C.byref(refusal)), refusal,
               "boundary check")
        name, device = "cpu", "native CPU"
    else:
        gpu = _gpu()
        gpu.run("tr_mgpu_topo_dd", C.byref(view), C.byref(b), C.byref(out), C.byref(refusal), refusal=refusal,
                what="boundary check on the device")
        name, device = _device_name()

    def part(viol, wit):
        return dict(checked=[int(out.checked[k]) for k in range(4)], violations=[int(viol[k]) for k in range(4)],
                    witness=[None if wit[k] == _NONE else int(wit[k]) for k in range(4)])
    return dict(z=dict(part(out.violations_z, out.witness_z), holds=bool(out.holds_z)),
                gf2=dict(part(out.violations_gf2, out.witness_gf2), holds=bool(out.holds_gf2)),
                invalid=int(out.invalid), terms=int(out.terms), backend=name, device=device)


# ---------------------------------------------------------------------- explicit chain complexes
def _csc(matrix, rows, what):
    """CSC arrays (ptr uint64, row uint32, val int64) of a dense integer matrix (rows x cols) or {(i, j): v}"""
    if isinstance(matrix, dict):
        raise TypeError("give sizes= with dictionary matrices")
    a = np.asarray(matrix, dtype=object)
    if a.ndim != 2:
        if a.size == 0:
            a = a.reshape(rows, 0)
        else:
            raise ValueError(f"{what}: a 2-D matrix")
    if a.shape[0] != rows:
        raise ValueError(f"{what} has {a.shape[0]} rows, the previous dimension {rows} cells")
    ptr, row, val = [0], [], []
    for j in range(a.shape[1]):
        for i in range(a.shape[0]):
            v = a[i, j]
            if isinstance(v, (np.integer,)):
                v = int(v)
            v = _int(v, f"{what}[{i}, {j}]")
            if v:
                row.append(i)
                val.append(v)
        ptr.append(len(row))
    return (np.array(ptr, dtype=np.uint64), np.array(row or [0], dtype=np.uint32), np.array(val or [0], dtype=np.int64),
            a.shape[1])


class ChainComplex:
    """An explicit integral chain complex: boundaries [d_1, .., d_top] (d_k: n_{k-1} x n_k exact integers), or
    sizes=[n_0] alone for top 0; d_{k-1} d_k = 0 is verified over Z natively (DomainError otherwise)."""

    def __init__(self, boundaries=(), *, sizes=None, budget=None):
        boundaries = list(boundaries)
        top = len(boundaries)
        if top > MAX_CHAIN:
            raise UnsupportedError(f"at most {MAX_CHAIN} boundary maps")
        if top == 0:
            if not sizes or len(sizes) != 1:
                raise ValueError("a chain complex of dimension 0 needs sizes=[n_0]")
            n = [_int(sizes[0], "a size")]
        else:
            first = np.asarray(boundaries[0], dtype=object)
            n = [int(first.shape[0]) if first.ndim == 2 else (int(sizes[0]) if sizes else 0)]
        arrays, keep = [None], []
        for k, d in enumerate(boundaries, start=1):
            ptr, row, val, cols = _csc(d, n[k - 1], f"d_{k}")
            n.append(cols)
            arrays.append((ptr, row, val))
            keep += [ptr, row, val]
        if sizes is not None and [int(s) for s in sizes] != n:
            raise ValueError(f"sizes {list(sizes)} disagree with the matrices {n}")
        N = (C.c_uint64 * 8)(*(n + [0] * (8 - len(n))))
        PT, RW, VL = (C.c_void_p * 8)(), (C.c_void_p * 8)(), (C.c_void_p * 8)()
        for k in range(1, top + 1):
            PT[k], RW[k], VL[k] = arrays[k][0].ctypes.data, arrays[k][1].ctypes.data, arrays[k][2].ctypes.data
        refusal, error = _Refusal(), C.c_int32()
        handle = _call("tr_topo_chain_create", top, N, PT, RW, VL, C.byref(_budget(budget)), C.byref(refusal),
                       C.byref(error))
        if error.value == -6:
            raise DomainError("not a chain complex: d_{k-1} d_k != 0 over Z")
        self._handle = _made(handle, error, refusal, "chain complex")
        self.top, self.sizes = top, tuple(n)

    def _open(self):
        if not self._handle:
            raise ValueError("the chain complex is closed")
        return self._handle

    @property
    def info(self):
        i = _ChainInfo()
        _check(_call("tr_topo_chain_get_info", self._open(), C.byref(i)), what="chain complex")
        return dict(top=int(i.top), cells=[int(i.n[k]) for k in range(self.top + 1)],
                    nnz=[int(i.nnz[k]) for k in range(self.top + 1)], euler=int(i.euler), bytes=int(i.bytes))

    def boundary(self, k, field="gf2", *, budget=None):
        """d_k over GF(2) as FIELD-001's packed Matrix"""
        if field != "gf2":
            raise ValueError("an explicit chain's matrices are its input; packed GF(2) export only")
        if not 1 <= k <= self.top:
            raise ValueError(f"d_k for 1 <= k <= {self.top}")
        m = mf.Matrix(self.sizes[k - 1], self.sizes[k])
        raw, refusal = m._raw(), _Refusal()
        _check(_call("tr_topo_boundary_packed", None, self._open(), int(k), C.byref(_budget(budget)), C.byref(raw),
                     C.byref(refusal)), refusal, "packed boundary matrix")
        return m

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_topo_chain_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"ChainComplex(sizes={list(self.sizes)})"


# ---------------------------------------------------------------------- homology and persistence
def _field(field, method="reduction", cycles=()):
    """the native field spec of a declared field: 'gf2', 'q', a prime p (2 is GF(2)) or ('gfp', p)"""
    if field is None:
        raise ValueError("declare the coefficient field: 'gf2', 'q' or a prime p")
    if isinstance(field, tuple) and len(field) == 2 and field[0] == "gfp":
        field = field[1]
    spec = _Field()
    if isinstance(field, str):
        f = field.lower()
        if f in ("z", "integers"):
            raise UnsupportedError("integral homology (torsion) is not computed: declare a field ('gf2', 'q', p)")
        if f not in ("gf2", "q"):
            raise ValueError("field is 'gf2', 'q' or a prime p")
        spec.field = 1 if f == "gf2" else 3
    elif isinstance(field, Integral) and not isinstance(field, bool):
        p = int(field)
        if p < 2:
            raise ValueError("a prime p >= 2")
        if p == 2:
            spec.field = 1
        else:
            if p >= 1 << 64:
                raise UnsupportedError("GF(p) needs p < 2^62")
            spec.field, spec.prime = 2, p
    else:
        raise TypeError("field is 'gf2', 'q' or a prime p")
    if method not in METHODS:
        raise ValueError(f"method is one of {list(METHODS)}")
    spec.method = METHODS[method]
    mask = 0
    for k in cycles or ():
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= int(k) <= MAX_CHAIN:
            raise ValueError("cycles: dimensions 0 .. 7")
        mask |= 1 << int(k)
    if mask and method == "packed":
        raise ValueError("cycle representatives come from the reduction (method='reduction')")
    spec.cycles = mask
    return spec


def _source(source, what):
    if isinstance(source, Complex):
        source._require(what)
        return source._handle, None, source.dim
    if isinstance(source, ChainComplex):
        return None, source._open(), source.top
    raise TypeError("a Complex or a ChainComplex")


def homology_need(source, *, field, method="reduction"):
    """the static needs of a homology request (states, tensor, support, bytes)"""
    cx, ch, _ = _source(source, "homology need")
    out = _Need()
    _check(_call("tr_topo_reduce_need", cx, ch, C.byref(_field(field, method)), None, C.byref(out)),
           what="homology need")
    return _needs(out)


class Homology:
    """Ranks, Betti numbers and (cycles) a basis of H_k as k-cycles over a declared field."""

    def __init__(self, source, info, classes):
        self.source = source
        self.top = int(info.top)
        self.field = FIELDS[int(info.field)]
        self.prime = int(info.prime) or (2 if self.field == "gf2" else None)
        self.method = {1: "reduction", 2: "packed"}[int(info.method)]
        self.cells = [int(info.n[k]) for k in range(self.top + 1)]
        self.ranks = [int(info.rank[k]) for k in range(self.top + 2)]
        self.betti = [int(info.betti[k]) for k in range(self.top + 1)]
        self.euler = int(info.euler)
        self.steps, self.peak_bytes = int(info.steps), int(info.peak_bytes)
        self.cycles = classes
        self.info = dict(steps=self.steps, peak_bytes=self.peak_bytes, classes=int(info.classes),
                         entries=int(info.entries), cells=self.cells)

    def fresh(self, scope=None):
        if isinstance(self.source, Complex):
            return self.source.fresh(scope)
        return None

    def __repr__(self):
        return f"Homology(field={self.field!r}, prime={self.prime}, betti={self.betti}, ranks={self.ranks})"


def homology(source, *, field, method="reduction", cycles=None, budget=None):
    """ranks rank_k = rank d_k (k = 0 .. top + 1), Betti numbers b_k = n_k - rank_k - rank_{k+1} over the declared
    field, and for each k in cycles (default: 1 with the reduction) a basis of H_k: {k: [{"generator", "cells",
    "coefficients"}]} (GF(2): 1; GF(p): residues with the generator's 1; Q: primitive integers, the generator's
    positive); method "packed" (GF(2)): ranks through FIELD-001's packed matrices, no representatives"""
    cx, ch, top = _source(source, "homology")
    if cycles is None:                                  # default: H_1 edge cycles from the reduction
        cycles = (1,) if method == "reduction" and top >= 1 else ()
    cycles = tuple(cycles)
    if any(isinstance(k, Integral) and int(k) > top for k in cycles):
        raise ValueError(f"cycles: dimensions 0 .. {top}")
    spec = _field(field, method, cycles)
    refusal, error = _Refusal(), C.c_int32()
    handle = _call("tr_topo_homology_build", cx, ch, C.byref(spec), C.byref(_budget(budget)), C.byref(refusal),
                   C.byref(error))
    handle = _made(handle, error, refusal, "homology")
    try:
        info = _HomologyInfo()
        _check(_call("tr_topo_homology_get_info", handle, C.byref(info)), what="homology")
        nc, ne = int(info.classes), int(info.entries)
        classes = (_Class * max(nc, 1))()
        cells, coefs = np.zeros(max(ne, 1), dtype=np.uint64), np.zeros(max(ne, 1), dtype=np.int64)
        _check(_call("tr_topo_homology_classes", handle, C.addressof(classes), max(nc, 1)), what="homology")
        _check(_call("tr_topo_homology_entries", handle, cells.ctypes.data, coefs.ctypes.data, max(ne, 1)),
               what="homology")
    finally:
        _call("tr_topo_homology_free", handle)
    reps = {int(k): [] for k in cycles}
    for i in range(nc):
        c = classes[i]
        a, b = int(c.offset), int(c.offset + c.length)
        reps.setdefault(int(c.dim), []).append(dict(generator=int(c.generator), cells=[int(x) for x in cells[a:b]],
                                                    coefficients=[int(x) for x in coefs[a:b]]))
    return Homology(source, info, reps)


def _filtration(source, ties, thresholds, direction, levels, nlevels, keep):
    if ties not in TIES:
        raise ValueError(f"ties is one of {list(TIES)} (the declared rule within a level and dimension)")
    flt = _Filtration()
    flt.ties = TIES[ties]
    if thresholds is not None:
        if levels is not None:
            raise ValueError("give thresholds (a value filtration) or levels, not both")
        if not isinstance(source, Complex) or source.grid.value is None:
            raise ValueError("thresholds filter the declared value column of a Complex")
        if direction not in ("sublevel", "superlevel"):
            raise ValueError("direction is 'sublevel' or 'superlevel'")
        scale = source.scope.column(source.grid.value)["scale"]
        logical = []
        for t in thresholds:
            q = Fraction(t) if not isinstance(t, float) else None
            if q is None:
                raise TypeError("thresholds are exact (int, Fraction or a decimal string), not floats")
            x = q * 10 ** scale
            if x.denominator != 1:
                raise ValueError(f"threshold {t} is not a value of the column (scale {scale})")
            logical.append(_int(x.numerator, "a threshold"))
        if not logical:
            raise ValueError("at least one threshold")
        arr = np.array(logical, dtype=np.int64)
        keep.append(arr)
        flt.source, flt.nlevels, flt.thresholds = SOURCES[direction], len(logical), arr.ctypes.data
        return flt
    if levels is None or nlevels is None:
        raise ValueError("give thresholds=, or levels= with nlevels= (the declared bound)")
    m = _int(nlevels, "nlevels")
    if m < 1:
        raise ValueError("nlevels >= 1")
    if isinstance(source, Complex):
        flat = list(levels)
        if len(flat) != source.counts[source.dim]:
            raise ValueError(f"one level per top cell ({source.counts[source.dim]})")
    else:
        per = list(levels)
        if len(per) != source.top + 1 or any(len(list(l)) != n for l, n in zip(per, source.sizes)):
            raise ValueError(f"one list of levels per dimension, of sizes {list(source.sizes)}")
        flat = [x for l in per for x in l]
    arr = np.array([NEVER if x is None else _int(x, "a level") for x in flat] or [0], dtype=np.int64)
    keep.append(arr)
    flt.source, flt.nlevels, flt.levels = SOURCES["levels"], m, arr.ctypes.data
    return flt


def persistence_need(source, *, field, ties, thresholds=None, direction="sublevel", levels=None, nlevels=None):
    cx, ch, _ = _source(source, "persistence need")
    keep, out = [], _Need()
    flt = _filtration(source, ties, thresholds, direction, levels, nlevels, keep)
    _check(_call("tr_topo_reduce_need", cx, ch, C.byref(_field(field)), C.byref(flt), C.byref(out)),
           what="persistence need")
    return _needs(out)


class Persistence:
    """The persistence pairs of a bounded filtration over a declared field (native, kept for betti / levels)."""

    def __init__(self, source, handle, thresholds):
        self.source, self._handle, self.thresholds = source, handle, thresholds
        i = _PersistenceInfo()
        _check(_call("tr_topo_persistence_get_info", handle, C.byref(i)), what="persistence")
        self.top = int(i.top)
        self.field = FIELDS[int(i.field)]
        self.prime = int(i.prime) or (2 if self.field == "gf2" else None)
        self.ties = {1: "index", 2: "reverse"}[int(i.ties)]
        self.nlevels = int(i.nlevels)
        r = range(self.top + 1)
        self.info = dict(entered=[int(i.entered[k]) for k in r], never=[int(i.never[k]) for k in r],
                         ranks=[int(i.rank[k]) for k in range(self.top + 2)], betti=[int(i.betti[k]) for k in r],
                         pairs=int(i.pairs), essential=int(i.essential), zero_length=int(i.zero_length),
                         steps=int(i.steps), peak_bytes=int(i.peak_bytes), nlevels=self.nlevels)
        n = int(i.pairs) + int(i.essential)
        raw = (_Pair * max(n, 1))()
        _check(_call("tr_topo_persistence_pairs", handle, C.addressof(raw), max(n, 1)), what="persistence")
        self.pairs, self.essential = [], []
        for q in range(n):
            x = raw[q]
            if x.death == NEVER:
                self.essential.append(dict(dim=int(x.dim), birth=int(x.birth), cell=int(x.birth_cell)))
            else:
                self.pairs.append(dict(dim=int(x.dim), birth=int(x.birth), death=int(x.death),
                                       birth_cell=int(x.birth_cell), death_cell=int(x.death_cell)))

    def diagram(self, dim=None, *, zero_length=False):
        """[(birth, death)] level pairs (death None for an essential class), zero-length pairs only on request"""
        out = [(p["birth"], p["death"]) for p in self.pairs if (dim is None or p["dim"] == dim) and
               (zero_length or p["birth"] < p["death"])]
        return out + [(e["birth"], None) for e in self.essential if dim is None or e["dim"] == dim]

    def value(self, level):
        """the declared threshold of a level (value filtrations), as given"""
        return None if level is None or self.thresholds is None else self.thresholds[level]

    def betti(self, level):
        """the Betti numbers of the filtration's complex at a level (native)"""
        out = np.zeros(self.top + 1, dtype=np.uint64)
        _check(_call("tr_topo_persistence_betti", self._handle, _int(level, "level"), out.ctypes.data, self.top + 1),
               what="persistence Betti numbers (0 <= level < nlevels)")
        return [int(x) for x in out]

    def levels(self, k):
        """the entry level of every k-cell (NEVER = -1: it never enters)"""
        if not 0 <= k <= self.top:
            raise ValueError(f"a dimension 0 .. {self.top}")
        n = self.info["entered"][k] + self.info["never"][k]
        out = np.zeros(max(n, 1), dtype=np.int64)
        _check(_call("tr_topo_persistence_levels", self._handle, int(k), out.ctypes.data, max(n, 1)),
               what="persistence levels")
        return out[:n]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_topo_persistence_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"Persistence(field={self.field!r}, ties={self.ties!r}, pairs={len(self.pairs)}, "
                f"essential={len(self.essential)})")


def persistence(source, *, field, ties, thresholds=None, direction="sublevel", levels=None, nlevels=None,
                budget=None):
    """the persistence pairs of a bounded filtration by the standard column reduction over the declared field:
    thresholds (on the complex's declared value column: a top cell enters at the first level l with v <= t_l
    (sublevel, t increasing; v the minimum of its rows) or v >= t_l (superlevel, t decreasing; the maximum)), or
    explicit levels (per top cell of a complex, per cell of each dimension of a chain; None / -1 never) with nlevels;
    ties: "index" or "reverse" within a level and dimension"""
    cx, ch, _ = _source(source, "persistence")
    keep = []
    flt = _filtration(source, ties, thresholds, direction, levels, nlevels, keep)
    refusal, error = _Refusal(), C.c_int32()
    handle = _call("tr_topo_persistence_build", cx, ch, C.byref(_field(field)), C.byref(flt), C.byref(_budget(budget)),
                   C.byref(refusal), C.byref(error))
    if error.value == -6:
        raise DomainError("the levels are not a filtration (a face after its coface)")
    handle = _made(handle, error, refusal, "persistence")
    return Persistence(source, handle, list(thresholds) if thresholds is not None else None)
