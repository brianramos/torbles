# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GRAPH-001: declared graph layouts and reductions; GRAPH-002: bounded fixed points, paths and graph energies,
natively on the CPU or with the CUDA kernels (torbles/math_graph.c, math_graph_gpu.cu; contract in math_graph.h).

    from torbles import math_bank as mbk, math_graph as mg
    scope = mbk.Scope.from_table(db["e"], ["src", "dst", "w"])
    G = mg.Layout.build(scope, "src", "dst", weight="w", directed=True)     # dedup=, loops=, nulls=, vertices=
    G = mg.Layout.from_edges([1, 2, 3], [2, 3, 1], weight=[Fraction(1, 2), 2, -1], directed=False)
    G.keys(); G.ids([1, 3]); G.vertex_set([1]); G.adjacency("out"); G.edges_list(); G.incidence()
    G.degrees("in", weights="edge"); G.expand(G.vertex_set([1]), exclude=visited)
    G.spmv(x, "count" | "bool" | "gf2" | "min_plus" | "max_plus" | "float", direction="forward", weights="edge")
    G.aggregate(x, "sum" | "min" | "max" | "any" | "count"); G.gf2_matrix()            # a math_field.Matrix
    G.reach(sources); G.closure(); G.walks(sources, 8, bound=2**200); G.walks(sources, None)
    G.paths(sources, "min_plus", algorithm="bellman_ford"); G.energy(x); G.pagerank(); G.power()
    every request takes backend="cpu" | "cuda" (the CUDA kernels; "cuda-emulated" without a device)

Vertices are the distinct keys (exact logical integers; decimals as Fractions) in increasing order, dense ids their
ranks. Undirected edges appear in both directions, a self-loop once; multi-edges are multiplicities unless dedup is
declared. Vertex sets and vectors of a layout carry it: one of another layout (another dictionary or snapshot) is a
DomainError; a stale scope (the table changed) is a StaleStateError before every use. Exact results are ints and
Fractions; floating results come with their stated bounds. Nothing is computed in Python. Opt-in: `import torbles`
does not load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from . import math_bank as mbk
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import (NoSolutionError, NotConvergedError, StaleStateError, UnboundedError, UnsupportedError,
                          RESOURCES, check as _check)
from .math_domains import DomainError

__all__ = ["Layout", "VertexSet", "Vector", "SEMIRINGS", "MAX_LIMBS", "BLOCK"]

MAX_LIMBS, BLOCK, NONE = 32, 4096, (1 << 32) - 1
UNBOUNDED = NONE
DIRECTIONS = {"forward": 1, "backward": 2}
SEMIRINGS = {"count": 1, "bool": 2, "gf2": 3, "min_plus": 4, "max_plus": 5, "float": 6}
WEIGHTS = {"edge": 1, "one": 2, "distinct": 3}
DEGREES = {"out": 1, "in": 2, "total": 3, "loops": 4}
ALGORITHMS = {"auto": 0, "bellman_ford": 1, "dijkstra": 2}
LOOPS = {"keep": 0, "drop": 1}
COMBINE = {"refuse": 0, "min": 1, "max": 2, "sum": 3}
NULLS = {"skip": 0, "forbid": 1}
KINDS = {0: None, 1: "row", 2: "total", 3: "length", 4: "grand"}
ROLES = {0: "tail", 1: "head", 2: "loop"}
REQ = {"degrees": 1, "expand": 2, "spmv": 3, "gf2": 4, "reach": 5, "closure": 6, "paths": 8, "energy": 9,
       "iterate": 10}
_U64 = (1 << 64) - 1
P = C.POINTER


class _Spec(C.Structure):
    _native_ = "tr_graph_spec"
    _fields_ = [(n, C.c_int32) for n in ("direction", "multi", "loops", "combine", "nulls", "src", "dst", "weight",
                                          "vertex", "reserved")] + [("denominator", C.c_int64)]


class _Info(C.Structure):
    _native_ = "tr_graph_info"
    _fields_ = [(n, C.c_uint64) for n in ("vertices", "entries", "edges", "rows", "kept", "null_endpoints",
                                          "null_weights", "loops", "dropped_loops", "merged", "multiplicity",
                                          "vertex_rows", "isolated", "bytes")] + [
        ("wmin", C.c_int64), ("wmax", C.c_int64), ("denominator", C.c_int64), ("digest", C.c_uint64 * 2)] + [
        (n, C.c_int32) for n in ("directed", "weighted", "multi", "loops_policy", "combine", "nulls", "key_scale",
                                 "weight_scale")]


class _View(C.Structure):
    _native_ = "tr_graph_view"
    _fields_ = [("n", C.c_uint64), ("entries", C.c_uint64), ("edges", C.c_uint64)] + [
        (n, C.c_void_p) for n in ("in_off", "in_nbr", "in_w", "in_m", "out_off", "out_nbr", "out_w", "out_m",
                                  "e_tail", "e_head", "e_w", "e_m")] + [
        ("denominator", C.c_int64), ("directed", C.c_int32), ("weighted", C.c_int32), ("wmin", C.c_int64),
        ("wmax", C.c_int64), ("max_in", C.c_uint64), ("max_out", C.c_uint64)]


class _Rounds(C.Structure):
    _native_ = "tr_graph_rounds"
    _fields_ = [(n, C.c_uint64) for n in ("rounds", "reached", "frontier_max", "work")]


class _WalksInfo(C.Structure):
    _native_ = "tr_graph_walks_info"
    _fields_ = [("lengths", C.c_uint64), ("violation_length", C.c_uint64), ("violation_vertex", C.c_uint64),
                ("status", C.c_int32), ("kind", C.c_int32), ("limbs", C.c_uint32), ("unbounded", C.c_uint32),
                ("maxbits", C.c_uint64), ("work", C.c_uint64)]


class _PathsInfo(C.Structure):
    _native_ = "tr_graph_paths_info"
    _fields_ = [("rounds", C.c_uint64), ("reached", C.c_uint64), ("cycle_length", C.c_uint64),
                ("cycle_weight_lo", C.c_int64), ("cycle_weight_hi", C.c_int64), ("witness", C.c_uint64),
                ("status", C.c_int32), ("algorithm", C.c_int32), ("work", C.c_uint64)]


class _IterInfo(C.Structure):
    _native_ = "tr_graph_iter_info"
    _fields_ = [("iterations", C.c_uint64)] + [(n, C.c_double) for n in ("delta", "value", "bound", "residual",
                                                                         "radius")] + [
        ("status", C.c_int32), ("converged", C.c_int32), ("dangling", C.c_uint64), ("reserved", C.c_uint64)]


for _s, _n in ((_Spec, 48), (_Info, 184), (_View, 168), (_Rounds, 32), (_WalksInfo, 56), (_PathsInfo, 64),
               (_IterInfo, 72)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u32, u64, d, e = C.c_void_p, C.c_int32, C.c_uint32, C.c_uint64, C.c_double, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_graph_layout_build": (vp, [vp, vp, P(_Spec), B, R, e]),
            "tr_graph_layout_need": (i32, [vp, vp, P(_Spec), P(_Need)]),
            "tr_graph_layout_get_info": (i32, [vp, P(_Info)]),
            "tr_graph_layout_get_view": (i32, [vp, P(_View)]),
            "tr_graph_layout_keys": (i32, [vp, vp, u64]),
            "tr_graph_layout_ids": (i32, [vp, vp, u64, vp, P(u64)]),
            "tr_graph_layout_incidence": (i32, [vp, vp, vp, u64, P(u64)]),
            "tr_graph_layout_same": (i32, [vp, vp]),
            "tr_graph_layout_fresh": (i32, [vp, vp, vp, vp, vp, vp, vp]),
            "tr_graph_layout_free": (None, [vp]),
            "tr_graph_rationals": (i32, [u64, vp, vp, vp, P(C.c_int64)]),
            "tr_graph_degrees": (i32, [vp, i32, i32, vp, B, R]),
            "tr_graph_expand": (i32, [vp, i32, vp, vp, vp, P(u64), B, R]),
            "tr_graph_spmv": (i32, [vp, i32, i32, i32, vp, vp, vp, vp, vp, P(u64), B, R]),
            "tr_graph_need": (i32, [vp, i32, i32, u64, u64, u32, i32, P(_Need)]),
            "tr_graph_gf2_matrix": (i32, [vp, i32, vp, u64, B, R]),
            "tr_graph_reach": (i32, [vp, i32, vp, i32, u64, vp, vp, P(_Rounds), B, R]),
            "tr_graph_closure": (i32, [vp, i32, vp, u64, i32, u64, vp, u64, P(_Rounds), B, R]),
            "tr_graph_walks_need": (i32, [vp, i32, vp, u32, u32, i32, i32, P(_Need), P(u64)]),
            "tr_graph_walks": (i32, [vp, i32, vp, u32, vp, u32, vp, u64, vp, vp, u64, vp, P(_WalksInfo), B, R]),
            "tr_graph_paths": (i32, [vp, i32, i32, i32, vp, vp, vp, vp, vp, vp, u64, P(_PathsInfo), B, R]),
            "tr_graph_energy": (i32, [vp, vp, vp, B, R]),
            "tr_graph_pagerank": (i32, [vp, i32, d, vp, vp, d, u64, vp, P(_IterInfo), B, R]),
            "tr_graph_power": (i32, [vp, i32, vp, d, u64, vp, P(_IterInfo), B, R]),
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


_GPU = {
    "tr_mgpu_graph_degrees": [P(_View), C.c_int32, C.c_int32, C.c_void_p, P(_Budget), P(_Refusal)],
    "tr_mgpu_graph_expand": [P(_View), C.c_int32, C.c_void_p, C.c_void_p, C.c_void_p, P(C.c_uint64), P(_Budget),
                             P(_Refusal)],
    "tr_mgpu_graph_spmv": [P(_View), C.c_int32, C.c_int32, C.c_int32, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                           C.c_void_p, P(C.c_uint64), P(_Budget), P(_Refusal)],
    "tr_mgpu_graph_reach": [P(_View), C.c_int32, C.c_void_p, C.c_int32, C.c_uint64, C.c_void_p, C.c_void_p,
                            P(_Rounds), P(_Budget), P(_Refusal)],
    "tr_mgpu_graph_closure": [P(_View), C.c_int32, C.c_void_p, C.c_uint64, C.c_int32, C.c_uint64, C.c_void_p,
                              C.c_uint64, P(_Rounds), P(_Budget), P(_Refusal)],
    "tr_mgpu_graph_walks": [P(_View), C.c_int32, C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32, C.c_void_p,
                            C.c_uint64, C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p, P(_WalksInfo), P(_Budget),
                            P(_Refusal)],
    "tr_mgpu_graph_paths": [P(_View), C.c_int32, C.c_int32, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                            C.c_void_p, C.c_void_p, C.c_uint64, P(_PathsInfo), P(_Budget), P(_Refusal)],
    "tr_mgpu_graph_energy": [P(_View), C.c_void_p, C.c_void_p, P(_Budget), P(_Refusal)],
    "tr_mgpu_graph_pagerank": [P(_View), C.c_int32, C.c_double, C.c_void_p, C.c_void_p, C.c_double, C.c_uint64,
                               C.c_void_p, P(_IterInfo), P(_Budget), P(_Refusal)],
    "tr_mgpu_graph_power": [P(_View), C.c_int32, C.c_void_p, C.c_double, C.c_uint64, C.c_void_p, P(_IterInfo),
                            P(_Budget), P(_Refusal)],
}


def _gpu():
    from . import math_gpu
    for name, args in _GPU.items():
        math_gpu.bind(name, C.c_int, args)
    return math_gpu


def _device_label(gpu):
    dv = gpu.device()
    return ("cuda-emulated" if dv["emulated"] else "cuda"), dv["name"]


def _budget(budget):
    return (budget or Budget()).raw


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _choice(table, value, what):
    if value not in table:
        raise ValueError(f"{what} is one of {', '.join(map(repr, table))}")
    return table[value]


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _ptr(a):
    return None if a is None else a.ctypes.data


def _words(n):
    return (n + 63) // 64


def _from_i128(words):
    lo, hi = words[:, 0], words[:, 1]
    if np.array_equal(hi, lo >> 63):
        return [int(v) for v in lo.tolist()]
    return [(int(h) << 64) | (int(l) & _U64) for l, h in zip(lo.tolist(), hi.tolist())]


def _limbs_to_int(row):
    v = 0
    for i, w in enumerate(row.tolist()):
        v |= int(w) << (64 * i)
    return v


def _limbs_rows(buf, count, W):
    """count exact unsigned integers of W limbs each (least significant first) as Python ints"""
    a = np.asarray(buf[:count * W], dtype=np.uint64).reshape(count, W)
    if W == 1 or not a[:, 1:].any():
        return [int(v) for v in a[:, 0].tolist()]
    return [_limbs_to_int(r) for r in a]


def _int_to_limbs(x, limbs):
    return np.array([(x >> (64 * i)) & _U64 for i in range(limbs)], dtype=np.uint64)


def _exact(value, what):
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise TypeError(f"{what}: {value!r} is not an exact integer or Fraction")
    return Fraction(value)


def _i64(value, what):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{what}: {value!r} is not an exact integer")
    v = int(value)
    if not -(1 << 63) <= v < (1 << 63):
        raise OverflowError(f"{what}: {v} does not fit int64")
    return v


class VertexSet:
    """A set of vertices of one layout (a frontier, sources, a reach set): a packed bitmap over the dense ids."""

    def __init__(self, layout, bits):
        self.layout = layout
        self.bits = np.ascontiguousarray(bits, dtype=np.uint64)

    def ids(self):
        n = self.layout.n
        flat = np.unpackbits(self.bits.view(np.uint8), bitorder="little")[:n]
        return [int(i) for i in np.flatnonzero(flat)]

    def keys(self):
        k = self.layout.keys()
        return [k[i] for i in self.ids()]

    def __len__(self):
        return int(sum(bin(int(w)).count("1") for w in self.bits.tolist()))

    def __contains__(self, key):
        try:
            i = self.layout.ids([key])[0]
        except NoSolutionError:
            return False
        return bool((int(self.bits[i >> 6]) >> (i & 63)) & 1)

    def __eq__(self, other):
        return isinstance(other, VertexSet) and self.layout.same(other.layout) and np.array_equal(self.bits,
                                                                                                    other.bits)

    def __repr__(self):
        return f"VertexSet({len(self)} of {self.layout.n})"


def _mask_bits(mask, n):
    """a packed uint64 bitmap of a boolean mask of n entries"""
    padded = np.zeros(max(_words(n), 1) * 64, dtype=np.uint8)
    padded[:n] = np.asarray(mask, dtype=bool)
    return np.packbits(padded, bitorder="little").view(np.uint64).copy()


def _bits_mask(bits, n):
    return np.unpackbits(np.ascontiguousarray(bits, dtype=np.uint64).view(np.uint8), bitorder="little")[:n].astype(bool)


class Vector:
    """Per-vertex values of one layout (dense id order). .array is the native result (int64 / two-word int128 /
    float64 / bool ndarray), .values its exact Python view (ints, Fractions, bools, floats; None where invalid),
    .valid (a term exists / the value is present), .error (float: the rigorous per-vertex bound)."""

    def __init__(self, layout, raw, valid, kind, denominator=1, label=("cpu", "native CPU"), error=None,
                 semiring=None):
        self.layout, self.array, self._valid, self.kind, self.denominator = layout, raw, valid, kind, denominator
        self.backend, self.device = label
        self.error, self.semiring = error, semiring or kind
        self._values = None

    @property
    def valid(self):
        return [bool(v) for v in self._valid.tolist()]

    @property
    def values(self):
        if self._values is None:
            n, d = self.layout.n, self.denominator
            if self.kind == "i128":
                vals = _from_i128(self.array[:n])
                vals = [Fraction(v, d) for v in vals] if d != 1 else vals
            elif self.kind == "i64":
                vals = [(Fraction(v, d) if d != 1 else int(v)) if ok else None
                        for v, ok in zip(self.array[:n].tolist(), self._valid.tolist())]
            elif self.kind == "bool":
                vals = [bool(v) for v in self.array[:n].tolist()]
            else:
                vals = self.array[:n]
            self._values = vals
        return self._values

    def as_dict(self):
        keys, vals, ok = self.layout.keys(), self.values, self.valid
        return {keys[i]: (vals[i] if ok[i] else None) for i in range(self.layout.n)}

    def __repr__(self):
        return f"Vector({self.semiring}, n={self.layout.n}, {self.backend})"


class Layout:
    """An immutable native graph layout: the vertex dictionary, CSR / CSC / incidence forms (math_graph.h)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Layout.build(scope, ...), Layout.from_table(...) or Layout.from_edges(...)")

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _spec(scope, src, dst, weight, vertices, directed, dedup, loops, combine, nulls, denominator):
        names = list(scope.names)

        def col(name, what, sc=scope, nm=names):
            if name not in nm:
                raise ValueError(f"{what}: the scope has no column {name!r} ({nm})")
            return nm.index(name)
        vcol = 0
        if vertices is not None:
            vscope, vname = vertices
            vcol = col(vname, "vertices", vscope, list(vscope.names))
        den = _i64(denominator, "denominator")
        if den < 1:
            raise ValueError("denominator >= 1")
        return _Spec(1 if directed else 2, int(bool(dedup)), _choice(LOOPS, loops, "loops"),
                     _choice(COMBINE, combine, "combine"), _choice(NULLS, nulls, "nulls"), col(src, "src"),
                     col(dst, "dst"), -1 if weight is None else col(weight, "weight"), vcol, 0, den)

    @classmethod
    def build(cls, scope, src, dst, weight=None, *, directed=True, dedup=False, loops="keep", combine="refuse",
              nulls="skip", vertices=None, denominator=1, budget=None):
        """The layout of the edge rows of an FND-004 scope: src / dst key columns, an optional weight column
        (value = logical / (10^scale x denominator)); vertices = (scope, column) adds the keys of another scope;
        directed or not; dedup collapses parallel edges (their weights combined by `combine`); loops "keep" / "drop";
        nulls "skip" (rows with a NULL endpoint or weight are no edges, counted) / "forbid" (DomainError)."""
        if not isinstance(scope, mbk.Scope):
            raise TypeError("scope: a math_bank.Scope (Scope.from_table or Scope.from_arrays)")
        scope.require_fresh("graph layout")
        vscope = None
        if vertices is not None:
            vscope = vertices[0]
            if not isinstance(vscope, mbk.Scope):
                raise TypeError("vertices: (math_bank.Scope, column)")
            vscope.require_fresh("graph layout (vertices)")
        spec = cls._spec(scope, src, dst, weight, vertices, directed, dedup, loops, combine, nulls, denominator)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_graph_layout_build", scope._handle, vscope._handle if vscope else None, C.byref(spec),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "graph layout")
        if not h:
            raise MemoryError("graph layout: native allocation failed")
        self = object.__new__(cls)
        self._handle, self.scope, self.vscope, self.spec_raw = h, scope, vscope, spec
        self.columns = dict(src=src, dst=dst, weight=weight, vertices=None if vertices is None else vertices[1])
        self._keys = None
        i = self._raw_info()
        self.n, self.entries, self.edges = int(i.vertices), int(i.entries), int(i.edges)
        self.directed, self.weighted = bool(i.directed), bool(i.weighted)
        self.denominator, self.key_scale = int(i.denominator), int(i.key_scale)
        return self

    @classmethod
    def from_table(cls, table, src, dst, weight=None, *, rows=None, vertices=None, budget=None, **kw):
        """the layout of a table's rows (a table scope over the key and weight columns); vertices = (table,
        column) or (table, column, rows)"""
        cols = [src, dst] + ([weight] if weight is not None and weight not in (src, dst) else [])
        scope = mbk.Scope.from_table(table, cols, rows=rows, budget=budget)
        vs = None
        if vertices is not None:
            vt, vc = vertices[0], vertices[1]
            vs = (mbk.Scope.from_table(vt, [vc], rows=vertices[2] if len(vertices) > 2 else None, budget=budget), vc)
        return cls.build(scope, src, dst, weight, vertices=vs, budget=budget, **kw)

    @classmethod
    def from_edges(cls, src, dst, weight=None, *, vertices=None, budget=None, **kw):
        """the layout of explicit edges: src / dst exact integer keys; weight ints or Fractions (put on their least
        common denominator natively); vertices: extra keys (isolated vertices); an array scope is built"""
        def keys_of(a, what):
            if isinstance(a, np.ndarray) and a.ndim == 1 and a.dtype.kind in "iu":
                if a.dtype.kind == "u" and a.size and int(a.max()) >= 1 << 63:
                    raise OverflowError(f"{what}: a key does not fit int64")
                return np.ascontiguousarray(a, dtype=np.int64)
            return np.array([_i64(v, what) for v in a], dtype=np.int64)
        s, t = keys_of(src, "src key"), keys_of(dst, "dst key")
        if len(s) != len(t):
            raise ValueError("src and dst have one key per edge")
        cols = {"src": s, "dst": t}
        den = 1
        if isinstance(weight, np.ndarray) and weight.ndim == 1 and weight.dtype.kind in "iu":
            cols["w"] = keys_of(weight, "weight")                   # integers: the unit is 1
            if len(cols["w"]) != len(s):
                raise ValueError("one weight per edge")
        elif weight is not None:
            ws = [_exact(w, "weight") for w in weight]
            if len(ws) != len(s):
                raise ValueError("one weight per edge")
            num = np.array([_i64(f.numerator, "weight numerator") for f in ws] or [0], dtype=np.int64)
            dd = np.array([_i64(f.denominator, "weight denominator") for f in ws] or [1], dtype=np.int64)
            scaled, D = np.zeros(max(len(ws), 1), dtype=np.int64), C.c_int64()
            _check(_call("tr_graph_rationals", len(ws), num.ctypes.data, dd.ctypes.data, scaled.ctypes.data,
                         C.byref(D)), what="edge weights on a common denominator")
            cols["w"] = scaled[:len(ws)]
            den = int(D.value)
        scope = mbk.Scope.from_arrays(cols, budget=budget)
        vs = None
        if vertices is not None:
            vk = np.array([_i64(v, "vertex key") for v in vertices], dtype=np.int64)
            vs = (mbk.Scope.from_arrays({"key": vk}, budget=budget), "key")
        return cls.build(scope, "src", "dst", "w" if weight is not None else None, vertices=vs,
                         denominator=den * int(kw.pop("denominator", 1)), budget=budget, **kw)

    @staticmethod
    def build_need(scope, src, dst, weight=None, *, directed=True, dedup=False, loops="keep", combine="refuse",
                   nulls="skip", vertices=None, denominator=1):
        """the needs a layout build checks before allocating: {resource: need}"""
        spec = Layout._spec(scope, src, dst, weight, vertices, directed, dedup, loops, combine, nulls, denominator)
        out = _Need()
        _check(_call("tr_graph_layout_need", scope._handle, vertices[0]._handle if vertices else None, C.byref(spec),
                     C.byref(out)), what="graph layout needs")
        return _needs_dict(out)

    # ------------------------------------------------------------------ identity, freshness, reading
    def _open(self):
        if not getattr(self, "_handle", None):
            raise ValueError("the layout is closed")

    def fresh(self):
        """True while both scopes are fresh (the tables, selections and lookups unchanged) and still the recorded
        ones. The scopes and the recorded descriptors are immutable, so their exact comparison (natively, with the
        tables' freshness) is made once; the tables' revisions and selections are checked on every call."""
        self._open()
        for sc in (self.scope, self.vscope):
            if sc is not None and not sc.fresh():
                return False
        if getattr(self, "_descriptors_checked", False):
            return True

        def args(sc):
            if sc is None:
                return None, None, None
            if sc.source == "arrays":
                return sc._handle, None, _ptr(sc._rows)
            rows = sc.table.used_bm if sc._rows is None else sc._rows
            return sc._handle, sc.table.at, rows.ctypes.data
        rc = _call("tr_graph_layout_fresh", self._handle, *args(self.scope), *args(self.vscope))
        if rc == -13:
            return False
        _check(rc, what="layout freshness")
        self._descriptors_checked = True
        return True

    def require_fresh(self, what="graph layout"):
        if not self.fresh():
            raise StaleStateError(f"{what}: the table or the selection changed since the layout was built")

    def same(self, other):
        """the same identity: declaration, the exact scope descriptors (snapshot revision included), dictionary"""
        self._open(); other._open()
        return bool(_check(_call("tr_graph_layout_same", self._handle, other._handle), what="layout identity"))

    def _raw_info(self):
        self._open()
        i = _Info()
        _check(_call("tr_graph_layout_get_info", self._handle, C.byref(i)), what="graph layout")
        return i

    @property
    def info(self):
        i = self._raw_info()
        out = {n: int(getattr(i, n)) for n, _ in _Info._fields_ if n not in ("digest",)}
        out["digest"] = (int(i.digest[0]), int(i.digest[1]))
        out["directed"], out["weighted"] = bool(i.directed), bool(i.weighted)
        return out

    def view(self):
        self._open()
        v = _View()
        _check(_call("tr_graph_layout_get_view", self._handle, C.byref(v)), what="graph view")
        return v

    def _key_value(self, logical):
        return Fraction(logical, 10 ** self.key_scale) if self.key_scale else logical

    def keys(self):
        """the dictionary: the key of every dense id, increasing (Fractions for a decimal key column)"""
        if self._keys is None:
            self._open()
            out = np.zeros(max(self.n, 1), dtype=np.int64)
            _check(_call("tr_graph_layout_keys", self._handle, out.ctypes.data, self.n), what="graph keys")
            self._keys = [self._key_value(int(v)) for v in out[:self.n].tolist()]
        return list(self._keys)

    def _logical_keys(self, keys):
        out = []
        for k in keys:
            if self.key_scale:
                f = _exact(k, "a vertex key") * 10 ** self.key_scale
                if f.denominator != 1:
                    raise ValueError(f"{k!r} is not a key of scale {self.key_scale}")
                out.append(_i64(f.numerator, "a vertex key"))
            else:
                out.append(_i64(k, "a vertex key"))
        return out

    def ids(self, keys):
        """the dense ids of keys (NoSolutionError naming a key that is no vertex)"""
        self._open()
        keys = list(keys)
        lk = np.array(self._logical_keys(keys) or [0], dtype=np.int64)
        out, missing = np.zeros(max(len(keys), 1), dtype=np.uint32), C.c_uint64()
        rc = _call("tr_graph_layout_ids", self._handle, lk.ctypes.data, len(keys), out.ctypes.data, C.byref(missing))
        if rc == -10:
            raise NoSolutionError(f"{keys[int(missing.value)]!r} is not a vertex of the layout")
        _check(rc, what="vertex ids")
        return [int(i) for i in out[:len(keys)].tolist()]

    def vertex_set(self, keys=None, *, ids=None, mask=None):
        """a VertexSet of keys, of dense ids, or of a boolean mask over the dense ids"""
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self.n,):
                raise ValueError(f"mask: {self.n} flags (one per vertex) expected")
            return VertexSet(self, _mask_bits(m, self.n))
        idx = np.array(self.ids(keys) if keys is not None else list(ids if ids is not None else []), dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n):
            raise ValueError("a vertex id is out of range")
        m = np.zeros(self.n, dtype=bool)
        m[idx] = True
        return VertexSet(self, _mask_bits(m, self.n))

    def all_vertices(self):
        return self.vertex_set(mask=np.ones(self.n, dtype=bool))

    def _set(self, s, what):
        """the bitmap of a VertexSet of this layout, or of keys"""
        if isinstance(s, VertexSet):
            if s.layout is not self and not self.same(s.layout):
                raise DomainError(f"{what}: the vertex set belongs to another layout (another dictionary or "
                                  "snapshot)")
            s.layout.require_fresh(what)
            return s.bits
        return self.vertex_set(list(s)).bits

    def _arrays(self):
        v = self.view()
        n, E, M = self.n, self.entries, self.edges

        def arr(ptr, count, ctype, dtype):
            if not count:
                return np.zeros(0, dtype=dtype)
            return np.ctypeslib.as_array(C.cast(ptr, P(ctype)), shape=(count,)).astype(dtype, copy=True)
        return dict(in_off=arr(v.in_off, n + 1, C.c_uint64, np.uint64), in_nbr=arr(v.in_nbr, E, C.c_uint32, np.uint32),
                    in_w=arr(v.in_w, E, C.c_int64, np.int64), in_m=arr(v.in_m, E, C.c_uint64, np.uint64),
                    out_off=arr(v.out_off, n + 1, C.c_uint64, np.uint64),
                    out_nbr=arr(v.out_nbr, E, C.c_uint32, np.uint32), out_w=arr(v.out_w, E, C.c_int64, np.int64),
                    out_m=arr(v.out_m, E, C.c_uint64, np.uint64), e_tail=arr(v.e_tail, M, C.c_uint32, np.uint32),
                    e_head=arr(v.e_head, M, C.c_uint32, np.uint32), e_w=arr(v.e_w, M, C.c_int64, np.int64),
                    e_m=arr(v.e_m, M, C.c_uint64, np.uint64))

    def _weight_value(self, w):
        return Fraction(int(w), self.denominator) if self.denominator != 1 else int(w)

    def adjacency(self, form="out"):
        """the stored entries: "out" (CSR: the out-entries of each vertex) or "in" (CSC): {offsets, neighbors (ids),
        weights (values), multiplicity}"""
        self.require_fresh("adjacency")
        a = self._arrays()
        p = {"out": "out", "in": "in"}[form]
        return dict(offsets=[int(x) for x in a[p + "_off"].tolist()], neighbors=[int(x) for x in a[p + "_nbr"].tolist()],
                    weights=[self._weight_value(x) for x in a[p + "_w"].tolist()],
                    multiplicity=[int(x) for x in a[p + "_m"].tolist()])

    def edges_list(self):
        """the canonical edges (the incidence form): [(tail id, head id, weight, multiplicity)]"""
        self.require_fresh("edges")
        a = self._arrays()
        return [(int(t), int(h), self._weight_value(w), int(m)) for t, h, w, m in
                zip(a["e_tail"].tolist(), a["e_head"].tolist(), a["e_w"].tolist(), a["e_m"].tolist())]

    def incidence(self):
        """the vertex-incidence lists: per vertex [(edge id, role "tail" | "head" | "loop")] in edge order"""
        self.require_fresh("incidence")
        off = np.zeros(self.n + 1, dtype=np.uint64)
        ent, cnt = np.zeros(max(2 * self.edges, 1), dtype=np.uint64), C.c_uint64()
        _check(_call("tr_graph_layout_incidence", self._handle, off.ctypes.data, ent.ctypes.data, len(ent),
                     C.byref(cnt)), what="incidence")
        out = []
        for v in range(self.n):
            out.append([(int(x) >> 2, ROLES[int(x) & 3]) for x in ent[int(off[v]):int(off[v + 1])].tolist()])
        return out

    def need(self, request, semiring="count", a=0, b=0, *, backend="cpu"):
        """the needs of a request ("degrees", "expand", "spmv", "gf2", "reach" (a = max_rounds), "closure" (a = k,
        b = max_rounds), "paths", "energy", "iterate" (a = max_iter)) on a backend"""
        out = _Need()
        _check(_call("tr_graph_need", self._handle, _choice(REQ, request, "request"),
                     _choice(SEMIRINGS, semiring, "semiring"), int(a), int(b), 0, 0 if _backend(backend) == "cpu" else 1,
                     C.byref(out)), what="graph needs")
        return _needs_dict(out)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_graph_layout_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not getattr(self, "_handle", None):
            return "Layout(closed)"
        return (f"Layout({'directed' if self.directed else 'undirected'}, vertices={self.n}, entries={self.entries}, "
                f"edges={self.edges})")

    # ------------------------------------------------------------------ GRAPH-001 reductions
    def _run(self, backend, cpu_name, gpu_name, cpu_args, gpu_args, refusal, what):
        """one request on a backend: (backend label, device)"""
        if _backend(backend) == "cpu":
            _check(_call(cpu_name, self._handle, *cpu_args), refusal, what)
            return "cpu", "native CPU"
        gpu = _gpu()
        view = self.view()
        gpu.run(gpu_name, C.byref(view), *gpu_args, refusal=refusal, what=what + " on the device")
        return _device_label(gpu)

    def degrees(self, kind="out", *, weights="one", backend="cpu", budget=None):
        """per vertex (dense id order): "out" (row sums of A), "in" (column sums), "total" (directed: out + in;
        undirected: the row sum, a loop once), "loops" (A[v][v]); weights "one" (multiplicities), "edge" (m w, values),
        "distinct" (distinct neighbors)"""
        self.require_fresh("degrees")
        k, w = _choice(DEGREES, kind, "kind"), _choice(WEIGHTS, weights, "weights")
        out = np.zeros((max(self.n, 1), 2), dtype=np.int64)
        refusal = _Refusal()
        label = self._run(backend, "tr_graph_degrees", "tr_mgpu_graph_degrees", (k, w, out.ctypes.data,
                          C.byref(_budget(budget)), C.byref(refusal)), (k, w, out.ctypes.data, C.byref(_budget(budget)),
                          C.byref(refusal)), refusal, f"{kind} degrees")
        return Vector(self, out, np.ones(self.n, dtype=bool), "i128", self.denominator if w == 1 else 1, label,
                      semiring=f"degree:{kind}:{weights}")

    def expand(self, frontier, *, direction="forward", exclude=None, backend="cpu", budget=None):
        """one step: the vertices with an entry from the frontier (forward: heads of its out-entries; backward:
        tails of its in-entries), minus `exclude`; a VertexSet (.count = its size)"""
        self.require_fresh("frontier expansion")
        d = _choice(DIRECTIONS, direction, "direction")
        f = self._set(frontier, "frontier")
        x = None if exclude is None else self._set(exclude, "exclude")
        nxt, cnt, refusal = np.zeros(max(_words(self.n), 1), dtype=np.uint64), C.c_uint64(), _Refusal()
        args = (d, f.ctypes.data, _ptr(x), nxt.ctypes.data, C.byref(cnt), C.byref(_budget(budget)), C.byref(refusal))
        label = self._run(backend, "tr_graph_expand", "tr_mgpu_graph_expand", args, args, refusal, "frontier expansion")
        out = VertexSet(self, nxt)
        out.count, out.backend, out.device = int(cnt.value), label[0], label[1]
        return out

    def _values(self, x, what):
        """per-vertex values in dense id order: a sequence of n values, a {key: value} mapping (absent: no value),
        or a Vector of this layout; -> (list, validity list)"""
        if isinstance(x, Vector):
            if x.layout is not self and not self.same(x.layout):
                raise DomainError(f"{what}: the vector belongs to another layout")
            return list(x.values), [bool(v) for v in x.valid]
        if isinstance(x, np.ndarray):
            if x.shape != (self.n,):
                raise ValueError(f"{what}: {self.n} values (one per vertex, dense id order) expected")
            return x, None
        if isinstance(x, dict):
            ids = self.ids(list(x))
            vals, ok = [0] * self.n, [False] * self.n
            for (k, v), i in zip(x.items(), ids):
                if v is not None:
                    vals[i], ok[i] = v, True
            return vals, ok
        vals = list(x)
        if len(vals) != self.n:
            raise ValueError(f"{what}: {self.n} values (one per vertex, dense id order) expected, got {len(vals)}")
        return [0 if v is None else v for v in vals], [v is not None for v in vals]

    def _bits_of(self, flags):
        return _mask_bits(np.asarray(flags, dtype=bool) if len(flags) else np.zeros(self.n, dtype=bool), self.n)

    def _unpack(self, bits):
        return [bool(b) for b in _bits_mask(bits, self.n).tolist()]

    def _units(self, vals, ok, what):
        """exact values to int64 units of 1/denominator (the weights' unit)"""
        out = []
        for v, o in zip(vals, ok):
            if not o:
                out.append(0)
                continue
            f = _exact(v, what) * self.denominator
            if f.denominator != 1:
                raise ValueError(f"{what}: {v!r} is not a multiple of 1/{self.denominator} (the weight unit)")
            out.append(_i64(f.numerator, what))
        return np.array(out or [0], dtype=np.int64)

    def spmv(self, x, semiring="count", *, direction="forward", weights="edge", backend="cpu", budget=None):
        """y = A^T x (forward: v aggregates its in-entries) or A x (backward) in a declared semiring (math_graph.h):
        "count" exact integers (weights: m w / D, so Fractions when D != 1), "bool", "gf2", "min_plus" / "max_plus"
        (exact values in multiples of 1/D; None = no term: +inf / -inf), "float" (binary64 with .error: the
        rigorous per-vertex bound |y - y*| <= error). weights "edge", "one" (the semiring's one per parallel edge),
        "distinct" (one per distinct neighbor). A vertex without a term (or a NULL input) is not valid."""
        self.require_fresh("semiring product")
        sr, d, w = _choice(SEMIRINGS, semiring, "semiring"), _choice(DIRECTIONS, direction, "direction"), \
            _choice(WEIGHTS, weights, "weights")
        vals, ok = self._values(x, "x")
        n, nw = self.n, max(_words(self.n), 1)
        xv = None if ok is None or all(ok) else self._bits_of(ok)
        err = None
        if isinstance(vals, np.ndarray):                     # one exact value per vertex, no NULL
            if sr in (2, 3):
                xs = _mask_bits(vals.astype(bool), n)
            elif sr == 6:
                if vals.dtype.kind not in "fiu":
                    raise TypeError("float semiring: a numeric array")
                xs = np.ascontiguousarray(vals, dtype=np.float64)
                if vals.dtype.kind in "iu" and not np.array_equal(xs.astype(vals.dtype), vals):
                    raise ValueError("float semiring: an integer is not an exact double")
            else:
                if vals.dtype.kind not in "iu":
                    raise TypeError(f"{semiring}: an integer array of exact values")
                if vals.dtype.kind == "u" and vals.size and int(vals.max()) >= 1 << 63:
                    raise OverflowError("x: a value does not fit int64")
                xs = np.ascontiguousarray(vals, dtype=np.int64)
                if sr in (4, 5) and self.denominator != 1:
                    lim = ((1 << 63) - 1) // self.denominator
                    if xs.size and int(np.abs(xs).max()) > lim:
                        raise OverflowError("x: a value in units of 1/denominator does not fit int64")
                    xs = xs * self.denominator
        elif sr in (2, 3):
            xs = self._bits_of([bool(v) and o for v, o in zip(vals, ok)])
        elif sr == 1:
            xs = np.array([_i64(v, "x") for v in vals] or [0], dtype=np.int64)
        elif sr in (4, 5):
            xs = self._units(vals, ok, "x")
        else:
            for v in vals:
                if not isinstance(v, (float, int)) or isinstance(v, bool):
                    raise TypeError("float semiring: x values are floats (or ints, converted exactly)")
            if any(float(v) != v for v in vals if isinstance(v, int)):
                raise ValueError("float semiring: an int input is not an exact double")
            xs = np.array([float(v) for v in vals] or [0.0], dtype=np.float64)
        if sr in (2, 3):
            y = np.zeros(nw, dtype=np.uint64)
        elif sr == 1:
            y = np.zeros((max(n, 1), 2), dtype=np.int64)
        elif sr in (4, 5):
            y = np.zeros(max(n, 1), dtype=np.int64)
        else:
            y = np.zeros(max(n, 1), dtype=np.float64)
            err = np.zeros(max(n, 1), dtype=np.float64)
        yv, where, refusal = np.zeros(nw, dtype=np.uint64), C.c_uint64(NONE), _Refusal()
        args = (sr, d, w, xs.ctypes.data, _ptr(xv), y.ctypes.data, yv.ctypes.data, _ptr(err), C.byref(where),
                C.byref(_budget(budget)), C.byref(refusal))
        try:
            label = self._run(backend, "tr_graph_spmv", "tr_mgpu_graph_spmv", args, args, refusal, f"{semiring} product")
        except OverflowError as e:
            at = int(where.value)
            vk = self.keys()[at] if at < n else None
            err_ = OverflowError(f"{semiring} product: the value at vertex {vk!r} leaves its carrier" if at < n else str(e))
            err_.vertex = vk
            raise err_ from None
        valid = _bits_mask(yv, n)
        if sr in (2, 3):
            return Vector(self, _bits_mask(y, n), valid, "bool", 1, label, semiring=semiring)
        if sr == 1:
            return Vector(self, y, valid, "i128", self.denominator if w == 1 else 1, label, semiring=semiring)
        if sr in (4, 5):
            return Vector(self, y, valid, "i64", self.denominator, label, semiring=semiring)
        return Vector(self, y[:n].copy(), valid, "float", 1, label, error=err[:n].copy(), semiring=semiring)

    def aggregate(self, x, how="sum", *, direction="forward", backend="cpu", budget=None):
        """neighborhood aggregates over the in-neighbors (forward) or out-neighbors (backward): "sum" (with
        multiplicity), "min", "max" (exact), "any" (Boolean), "count" (entries with a value), "distinct_sum"""
        table = {"sum": ("count", "one"), "distinct_sum": ("count", "distinct"), "min": ("min_plus", "one"),
                 "max": ("max_plus", "one"), "any": ("bool", "one")}
        if how == "count":
            vals, ok = self._values(x, "x")
            ones = np.ones(self.n, dtype=np.int64) if ok is None else [1 if o else None for o in ok]
            return self.spmv(ones, "count", direction=direction, weights="one", backend=backend, budget=budget)
        sr, w = _choice(table, how, "how")
        return self.spmv(x, sr, direction=direction, weights=w, backend=backend, budget=budget)

    def gf2_matrix(self, *, weights="edge", budget=None):
        """A mod 2 as a FIELD-001 packed matrix (row u = the out-entries of u, entries (m w) mod 2): its matvec is the
        backward GF(2) product, its transpose's the forward one"""
        from . import math_field as mf
        self.require_fresh("GF(2) adjacency")
        w = _choice(WEIGHTS, weights, "weights")
        stride = max(_words(self.n), 1)
        words = np.zeros((self.n, stride), dtype=np.uint64)
        refusal = _Refusal()
        _check(_call("tr_graph_gf2_matrix", self._handle, w, words.ctypes.data if self.n else None, stride,
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "GF(2) adjacency")
        return mf.Matrix(self.n, self.n, words, stride)

    # ------------------------------------------------------------------ GRAPH-002
    def reach(self, sources, *, direction="forward", reflexive=True, max_rounds=None, backend="cpu", budget=None):
        """the vertices reachable from the sources by walks of length >= 0 (reflexive) or >= 1: {"reached" (a
        VertexSet), "level" (per vertex: the shortest such walk, None unreached), "rounds" (expansions to the fixed
        point), "reached_count", "frontier_max", "work", "backend"}; max_rounds (default n + 1, always enough):
        NotConvergedError when the frontier is not empty after that many rounds"""
        self.require_fresh("reachability")
        d = _choice(DIRECTIONS, direction, "direction")
        s = self._set(sources, "sources")
        mr = self.n + 1 if max_rounds is None else int(max_rounds)
        reached = np.zeros(max(_words(self.n), 1), dtype=np.uint64)
        level = np.zeros(max(self.n, 1), dtype=np.uint32)
        info, refusal = _Rounds(), _Refusal()
        args = (d, s.ctypes.data, int(bool(reflexive)), mr, reached.ctypes.data, level.ctypes.data, C.byref(info),
                C.byref(_budget(budget)), C.byref(refusal))
        label = self._run(backend, "tr_graph_reach", "tr_mgpu_graph_reach", args, args, refusal, "reachability")
        return dict(reached=VertexSet(self, reached), level=[None if v == NONE else int(v) for v in level[:self.n].tolist()],
                    rounds=int(info.rounds), reached_count=int(info.reached), frontier_max=int(info.frontier_max),
                    work=int(info.work), backend=label[0], device=label[1])

    def closure(self, sources=None, *, direction="forward", reflexive=False, max_rounds=None, backend="cpu",
                budget=None):
        """the reach set of each source (default every vertex: the transitive closure; reflexive adds the source
        itself): {"matrix" (a FIELD-001 Matrix, row i = source i), "sources" (keys), "rounds", ...}"""
        from . import math_field as mf
        self.require_fresh("transitive closure")
        d = _choice(DIRECTIONS, direction, "direction")
        ids = list(range(self.n)) if sources is None else (
            sources.ids() if isinstance(sources, VertexSet) and (self._set(sources, "sources") is not None) else
            self.ids(list(sources)))
        k, stride = len(ids), max(_words(self.n), 1)
        src = np.array(ids or [0], dtype=np.uint32)
        rows = np.zeros((max(k, 1), stride), dtype=np.uint64)
        mr = self.n + 1 if max_rounds is None else int(max_rounds)
        info, refusal = _Rounds(), _Refusal()
        args = (d, src.ctypes.data, k, int(bool(reflexive)), mr, rows.ctypes.data, stride, C.byref(info),
                C.byref(_budget(budget)), C.byref(refusal))
        label = self._run(backend, "tr_graph_closure", "tr_mgpu_graph_closure", args, args, refusal,
                          "transitive closure")
        keys = self.keys()
        return dict(matrix=mf.Matrix(k, self.n, rows[:k], stride), sources=[keys[i] for i in ids],
                    rounds=int(info.rounds), reached_count=int(info.reached), frontier_max=int(info.frontier_max),
                    work=int(info.work), backend=label[0], device=label[1])

    def walks_need(self, sources, length, *, bound=(1 << 63) - 1, rows=True, direction="forward", backend="cpu"):
        """the needs of a walk table and its lengths (length None: the horizon pre-pass runs, NoSolutionError when
        a cycle is reachable)"""
        d = _choice(DIRECTIONS, direction, "direction")
        s = self._set(sources, "sources")
        limbs = max(1, (int(bound).bit_length() + 63) // 64)
        out, lens = _Need(), C.c_uint64()
        _check(_call("tr_graph_walks_need", self._handle, d, s.ctypes.data, UNBOUNDED if length is None else int(length),
                     limbs, int(bool(rows)), 0 if _backend(backend) == "cpu" else 1, C.byref(out), C.byref(lens)),
               what="walk table needs")
        return dict(_needs_dict(out), lengths=int(lens.value))

    def walks(self, sources, length, *, bound=(1 << 63) - 1, rows=True, direction="forward", backend="cpu",
              budget=None):
        """walk counts from the sources: rows[l][v] = walks of exactly l steps ending at v (multiplicities count),
        totals[v] = sum over l <= length, lengths[l] = sum over v, grand; exact ints under the declared bound (every
        coefficient <= bound, else OverflowError with .length, .vertex, .kind); length None: every length (finite
        only when no cycle is reachable from the sources, else NoSolutionError)"""
        self.require_fresh("walk counts")
        d = _choice(DIRECTIONS, direction, "direction")
        s = self._set(sources, "sources")
        B = int(bound)
        if B < 0:
            raise ValueError("bound >= 0")
        limbs = max(1, (B.bit_length() + 63) // 64)
        if limbs > MAX_LIMBS:
            raise UnsupportedError(f"walk counts: a bound of {B.bit_length()} bits > {64 * MAX_LIMBS}")
        L = UNBOUNDED if length is None else int(length)
        if length is not None and not 0 <= L < UNBOUNDED:
            raise ValueError("length: an integer >= 0, or None (unbounded)")
        lens = self.walks_need(sources, length, bound=B, rows=rows, direction=direction, backend=backend)["lengths"]
        n, W = self.n, limbs
        bw = _int_to_limbs(B, W)
        rbuf = np.zeros(max(lens * n * W, 1), dtype=np.uint64) if rows else None
        tot = np.zeros(max(n * W, 1), dtype=np.uint64)
        lbuf = np.zeros(max(lens * W, 1), dtype=np.uint64)
        grand = np.zeros(W, dtype=np.uint64)
        info, refusal = _WalksInfo(), _Refusal()
        args = (d, s.ctypes.data, L, bw.ctypes.data, W, _ptr(rbuf), lens * n * W if rows else 0, tot.ctypes.data,
                lbuf.ctypes.data, lens * W, grand.ctypes.data, C.byref(info), C.byref(_budget(budget)),
                C.byref(refusal))
        try:
            label = self._run(backend, "tr_graph_walks", "tr_mgpu_graph_walks", args, args, refusal, "walk counts")
        except OverflowError:
            kind = KINDS.get(int(info.kind))
            vtx = int(info.violation_vertex)
            e = OverflowError(f"walk counts: a coefficient exceeds the declared bound {B} ({kind} at length "
                              f"{int(info.violation_length)}" + (f", vertex {self.keys()[vtx]!r})" if vtx < n else ")"))
            e.kind, e.length = kind, int(info.violation_length)
            e.vertex = self.keys()[vtx] if vtx < n else None
            raise e from None
        out = dict(totals=_limbs_rows(tot, n, W), lengths=_limbs_rows(lbuf, lens, W),
                   grand=_limbs_to_int(grand), length=lens - 1, unbounded=length is None, bound=B, limbs=W,
                   maxbits=int(info.maxbits), work=int(info.work), backend=label[0], device=label[1])
        if rows:
            flat = _limbs_rows(rbuf, lens * n, W)
            out["rows"] = [flat[l * n:(l + 1) * n] for l in range(lens)]
        return out

    def paths(self, sources, semiring="min_plus", *, algorithm="auto", direction="forward", backend="cpu",
              budget=None):
        """shortest ("min_plus") or longest ("max_plus") walk weights from the sources: {"distance" (exact, None
        unreached), "hops", "pred" (dense ids, None at a root), "reached", "rounds", "algorithm"}; a reachable cycle
        of negative (min_plus) / positive (max_plus) weight raises UnboundedError with .cycle (keys, in edge order)
        and .weight; Dijkstra only with every weight >= 0 (DomainError)"""
        self.require_fresh("paths")
        sr = _choice({"min_plus": 4, "max_plus": 5}, semiring, "semiring")
        alg = _choice(ALGORITHMS, algorithm, "algorithm")
        d = _choice(DIRECTIONS, direction, "direction")
        s = self._set(sources, "sources")
        n = self.n
        dist = np.zeros(max(n, 1), dtype=np.int64)
        reached = np.zeros(max(_words(n), 1), dtype=np.uint64)
        hops = np.zeros(max(n, 1), dtype=np.uint32)
        pred = np.zeros(max(n, 1), dtype=np.uint32)
        cycle = np.zeros(max(n, 1), dtype=np.uint32)
        info, refusal = _PathsInfo(), _Refusal()
        if _backend(backend) == "cuda" and alg == 2:
            raise UnsupportedError("Dijkstra has no CUDA path (Bellman-Ford rounds run on the device)")
        cpu = (sr, alg, d, s.ctypes.data, dist.ctypes.data, reached.ctypes.data, hops.ctypes.data, pred.ctypes.data,
               cycle.ctypes.data, n, C.byref(info), C.byref(_budget(budget)), C.byref(refusal))
        gpu = (sr, d, s.ctypes.data, dist.ctypes.data, reached.ctypes.data, hops.ctypes.data, pred.ctypes.data,
               cycle.ctypes.data, n, C.byref(info), C.byref(_budget(budget)), C.byref(refusal))
        try:
            label = self._run(backend, "tr_graph_paths", "tr_mgpu_graph_paths", cpu, gpu, refusal, f"{semiring} paths")
        except UnboundedError:
            keys = self.keys()
            ln = int(info.cycle_length)
            w = (int(info.cycle_weight_hi) << 64) | (int(info.cycle_weight_lo) & _U64)
            weight = Fraction(w, self.denominator) if self.denominator != 1 else w
            e = UnboundedError(f"{semiring} paths: a cycle of {'negative' if sr == 4 else 'positive'} weight "
                               f"{weight} is reachable from the sources: no finite answer")
            e.cycle = [keys[i] for i in cycle[:ln].tolist()]
            e.cycle_ids = [int(i) for i in cycle[:ln].tolist()]
            e.weight, e.rounds = weight, int(info.rounds)
            e.witness = keys[int(info.witness)] if int(info.witness) < n else None
            raise e from None
        ok = self._unpack(reached)
        return dict(distance=[self._weight_value(v) if o else None for v, o in zip(dist[:n].tolist(), ok)],
                    hops=[None if h == NONE else int(h) for h in hops[:n].tolist()],
                    pred=[None if p == NONE else int(p) for p in pred[:n].tolist()],
                    reached=VertexSet(self, reached), rounds=int(info.rounds), reached_count=int(info.reached),
                    work=int(info.work), algorithm={1: "bellman_ford", 2: "dijkstra"}[int(info.algorithm)],
                    backend=label[0], device=label[1])

    def energy(self, x, *, backend="cpu", budget=None):
        """the exact Dirichlet energy sum_e m w (x_head - x_tail)^2 = x^T L x of a vertex function (ints or
        Fractions; dense id order, a {key: value} mapping or a Vector): an int or a Fraction"""
        self.require_fresh("Dirichlet energy")
        vals, ok = self._values(x, "x")
        if ok is not None and not all(ok):
            raise ValueError("the Dirichlet energy needs a value at every vertex")
        D = C.c_int64(1)
        if isinstance(vals, np.ndarray):
            if vals.dtype.kind not in "iu" or (vals.dtype.kind == "u" and vals.size and int(vals.max()) >= 1 << 63):
                raise TypeError("the Dirichlet energy: exact int64 values")
            xs = np.ascontiguousarray(vals, dtype=np.int64)            # integers: the unit is 1
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            xs = np.array([_i64(v, "x") for v in vals] or [0], dtype=np.int64)
        else:
            fr = [_exact(v, "x") for v in vals]
            num = np.array([_i64(f.numerator, "x numerator") for f in fr] or [0], dtype=np.int64)
            den = np.array([_i64(f.denominator, "x denominator") for f in fr] or [1], dtype=np.int64)
            xs = np.zeros(max(self.n, 1), dtype=np.int64)
            _check(_call("tr_graph_rationals", self.n, num.ctypes.data, den.ctypes.data, xs.ctypes.data, C.byref(D)),
                   what="vertex values on a common denominator")
        out, refusal = np.zeros(5, dtype=np.uint64), _Refusal()
        args = (xs.ctypes.data, out.ctypes.data, C.byref(_budget(budget)), C.byref(refusal))
        label = self._run(backend, "tr_graph_energy", "tr_mgpu_graph_energy", args, args, refusal, "Dirichlet energy")
        v = _limbs_to_int(out)
        if v >> 319:
            v -= 1 << 320
        e = Fraction(v, self.denominator * int(D.value) ** 2)
        self._last_backend = label
        return int(e) if e.denominator == 1 else e

    def pagerank(self, damping=0.85, *, tol=1e-12, max_iter=200, teleport=None, start=None, direction="forward",
                 backend="cpu", budget=None):
        """PageRank in binary64 (math_graph.h): {"rank" (ndarray, dense id order), "iterations", "delta", "bound"
        (rigorous: >= ||rank - exact||_1; inf when damping = 1), "residual" (>= ||G r - r||_1), "mass", "dangling"};
        NotConvergedError when the change is above tol after max_iter iterations"""
        self.require_fresh("PageRank")
        d = _choice(DIRECTIONS, direction, "direction")
        t = None if teleport is None else self._set(teleport, "teleport")
        s = None if start is None else self._set(start, "start")
        rank = np.zeros(max(self.n, 1), dtype=np.float64)
        info, refusal = _IterInfo(), _Refusal()
        args = (d, float(damping), _ptr(t), _ptr(s), float(tol), int(max_iter), rank.ctypes.data, C.byref(info),
                C.byref(_budget(budget)), C.byref(refusal))
        try:
            label = self._run(backend, "tr_graph_pagerank", "tr_mgpu_graph_pagerank", args, args, refusal, "PageRank")
        except NotConvergedError:
            e = NotConvergedError(f"PageRank: the change {info.delta!r} is above tol {tol!r} after "
                                  f"{int(info.iterations)} iterations: no result")
            e.iterations, e.delta = int(info.iterations), float(info.delta)
            raise e from None
        return dict(rank=rank[:self.n].copy(), iterations=int(info.iterations), delta=float(info.delta),
                    bound=float(info.bound), residual=float(info.residual), mass=float(info.value),
                    dangling=int(info.dangling), backend=label[0], device=label[1])

    def power(self, *, tol=1e-12, max_iter=500, start=None, direction="forward", backend="cpu", budget=None):
        """power iteration x <- A^T x / ||A^T x||_1 (forward; A x backward) in binary64: {"vector", "eigenvalue",
        "iterations", "delta", "residual_l1" / "residual_l2" (rigorous upper bounds of ||A' x - lambda x||),
        "radius" (undirected: an eigenvalue of A lies within it of lambda; else inf)}; NotConvergedError at the cap,
        NoSolutionError when the iterate vanishes"""
        self.require_fresh("power iteration")
        d = _choice(DIRECTIONS, direction, "direction")
        s = None if start is None else self._set(start, "start")
        vec = np.zeros(max(self.n, 1), dtype=np.float64)
        info, refusal = _IterInfo(), _Refusal()
        args = (d, _ptr(s), float(tol), int(max_iter), vec.ctypes.data, C.byref(info), C.byref(_budget(budget)),
                C.byref(refusal))
        try:
            label = self._run(backend, "tr_graph_power", "tr_mgpu_graph_power", args, args, refusal, "power iteration")
        except NotConvergedError:
            e = NotConvergedError(f"power iteration: the change {info.delta!r} is above tol {tol!r} after "
                                  f"{int(info.iterations)} iterations: no result")
            e.iterations, e.delta = int(info.iterations), float(info.delta)
            raise e from None
        return dict(vector=vec[:self.n].copy(), eigenvalue=float(info.value), iterations=int(info.iterations),
                    delta=float(info.delta), residual_l1=float(info.bound), residual_l2=float(info.residual),
                    radius=float(info.radius), backend=label[0], device=label[1])
