# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""MULTI-001: hierarchies of group keys: certified refinement maps, per-level exact moment banks built once at the
finest level and rolled up exactly, localized updates, and exact Haar-like contrasts, natively on the CPU or with the
CUDA kernels (torbles/math_multi.c, math_multi_gpu.cu).

    from torbles import math_bank as mbk, math_multi as mm
    day = mbk.Scope.from_table(t, ["x", "d"], group_by=["day"])
    month, year = (mbk.Scope.from_table(t, ["x", "d"], group_by=[k]) for k in ("month", "year"))
    m = mm.certify(day, month, backend="cpu")        # Map (proven) or RefinementError with .witness
    m = mm.declare(month, year, {(201401,): (2014,)})  # a map given by the schema: declared, not proven
    m.verify(month, year)                            # the declared map proven on the data, or RefinementError
    h = mm.Hierarchy([day, month, year], ["1", "x", "x^2", "d"], mode="sql", backend="cpu")
    h.bank(2).state(g), h.table(1), h.status         # FND-004 banks per level; "proven" / "declared" / "carried"
    h.contrasts(0, "mean", backend="cuda")           # exact Fractions: child mean - parent mean, per term
    h.update(new_day_bank)                           # localized: only the coarse cells it must recompute

A level is a FND-004 scope with the logical codec descriptor of its keys (derived here from the table's codecs; empty
for array scopes). Maps are bound to both levels' groupings and codecs (and, proven, to the rows they were proven on):
a map built for another scope, grouping, selection or codec raises DomainError. Nothing is computed in Python. Opt-in:
`import torbles` does not load it.
"""
import ctypes as C
import json
from fractions import Fraction

import numpy as np

from . import algq
from . import math_bank as mbk
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Map", "Hierarchy", "Contrasts", "RefinementError", "certify", "declare", "map_need", "hierarchy_need",
           "codec_descriptor",
           "KINDS", "MAX_LEVELS", "UNMAPPED"]

MAX_LEVELS = 8
UNMAPPED = 0xFFFFFFFF
_NONE = (1 << 64) - 1
KINDS = {"mean": 1, "total": 2, "share": 3, "sibling_mean": 4, "sibling_total": 5}
STATUS = {1: "proven", 2: "declared", 3: "carried"}
CELL = {0: "defined", 1: "absent", 2: "undefined", 3: "no sibling"}
STRATEGY = {0: None, 1: "rollup", 2: "local", 3: "copy"}
P = C.POINTER


class _Level(C.Structure):
    _native_ = "tr_multi_level"
    _fields_ = [("scope", C.c_void_p), ("codec", C.c_void_p), ("codec_len", C.c_uint64)]


class _Witness(C.Structure):
    _native_ = "tr_multi_witness"
    _fields_ = [("row", C.c_uint64), ("first_row", C.c_uint64), ("fine", C.c_uint32), ("coarse", C.c_uint32),
                ("first_coarse", C.c_uint32), ("found", C.c_int32)]


class _MapInfo(C.Structure):
    _native_ = "tr_multi_map_info"
    _fields_ = [(n, C.c_uint64) for n in ("nfine", "ncoarse", "mapped", "checked", "bytes")] + [
        ("digest", C.c_uint64 * 2), ("status", C.c_int32), ("rows_bound", C.c_int32)]


class _HierInfo(C.Structure):
    _native_ = "tr_multi_hier_info"
    _fields_ = [(n, C.c_int32) for n in ("levels", "status", "ncols", "nterms", "mode", "reserved")] + [
        (n, C.c_uint64) for n in ("bytes", "rows", "work", "changed")] + [
        (n, C.c_uint64 * MAX_LEVELS) for n in ("groups", "present", "level_bytes", "level_work")] + [
        (n, C.c_int32 * MAX_LEVELS) for n in ("scoped", "strategy", "map_status")]


class _Q(C.Structure):
    _native_ = "tr_multi_q"
    _fields_ = [("num", C.c_uint64 * 4), ("den", C.c_uint64 * 2), ("child", C.c_uint32), ("other", C.c_uint32),
                ("status", C.c_int32), ("term", C.c_int32)]


class _Cert(C.Structure):
    _native_ = "tr_multi_cert"
    _fields_ = [("nfine", C.c_uint64), ("first", C.c_void_p), ("lo", C.c_void_p), ("hi", C.c_void_p),
                ("witness", C.c_uint64), ("checked", C.c_uint64)]


for _s, _n in ((_Level, 24), (_Witness, 32), (_MapInfo, 64), (_HierInfo, 408), (_Q, 64), (_Cert, 48)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R, Lv = P(_Budget), P(_Refusal), P(_Level)
        sig = {
            "tr_multi_map_certify": (vp, [Lv, Lv, B, R, P(_Witness), e]),
            "tr_multi_map_from_cert": (vp, [Lv, Lv, P(_Cert), B, R, P(_Witness), e]),
            "tr_multi_map_declare": (vp, [Lv, Lv, vp, u64, B, R, e]),
            "tr_multi_map_verify": (vp, [vp, Lv, Lv, B, R, P(_Witness), e]),
            "tr_multi_map_compose": (vp, [vp, vp, B, R, e]),
            "tr_multi_map_check": (i32, [vp, Lv, Lv]),
            "tr_multi_map_need": (i32, [Lv, Lv, i32, P(_Need)]),
            "tr_multi_map_get_info": (i32, [vp, P(_MapInfo)]),
            "tr_multi_map_get": (i32, [vp, vp, u64]),
            "tr_multi_map_equal": (i32, [vp, vp]),
            "tr_multi_map_free": (None, [vp]),
            "tr_multi_hier_build": (vp, [Lv, i32, vp, vp, B, R, P(u64), e]),
            "tr_multi_hier_update": (vp, [vp, vp, Lv, vp, B, R, P(u64), e]),
            "tr_multi_hier_need": (i32, [Lv, i32, vp, P(_Need)]),
            "tr_multi_hier_get_info": (i32, [vp, P(_HierInfo)]),
            "tr_multi_hier_states": (i32, [vp, i32, P(mbk._States)]),
            "tr_multi_hier_bank": (vp, [vp, i32, e]),
            "tr_multi_hier_map": (vp, [vp, i32, e]),
            "tr_multi_hier_links": (i32, [vp, i32, vp, vp, u64, B, R]),
            "tr_multi_contrasts": (i32, [vp, i32, i32, vp, i32, u64, u64, B, P(_Q), R]),
            "tr_multi_contrasts_need": (i32, [vp, i32, i32, u64, i32, P(_Need)]),
            "tr_multi_hier_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_multi_certify", C.c_int, [P(mbk._View), P(mbk._View), P(_Budget), P(_Cert), P(_Refusal)])
    math_gpu.bind("tr_mgpu_multi_contrasts", C.c_int, [P(mbk._States), P(mbk._States), C.c_void_p, C.c_void_p,
                                                       C.c_int32, C.c_void_p, C.c_int32, C.c_uint64, C.c_uint64,
                                                       P(_Budget), P(_Q), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _backend_name(gpu):
    d = gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


class RefinementError(DomainError):
    """A map is not a refinement of the data: .witness = {"row", "first_row", "fine", "coarse", "first_coarse",
    "fine_key", "coarse_key", "first_coarse_key"} (the smallest violating selected row and what it disagrees with)."""


def _witness(w, fine, coarse):
    def key(scope, g):
        return None if g == UNMAPPED else scope.key_values(g)
    return dict(row=None if w.row == _NONE else int(w.row), first_row=None if w.first_row == _NONE else int(w.first_row),
                fine=int(w.fine), coarse=int(w.coarse),
                first_coarse=None if w.first_coarse == UNMAPPED else int(w.first_coarse),
                fine_key=key(fine, w.fine), coarse_key=key(coarse, w.coarse),
                first_coarse_key=key(coarse, w.first_coarse))


def _made(handle, error, refusal, what, witness=None, fine=None, coarse=None):
    """the handle of a native constructor, or its refusal raised (a refinement violation with its witness)"""
    if error.value == -6 and witness is not None and witness.found:
        err = RefinementError(f"{what}: not a refinement: selected row {witness.row} of fine group {witness.fine} "
                              f"falls in coarse group {witness.coarse}, not {witness.first_coarse}")
        err.witness = _witness(witness, fine, coarse)
        raise err
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


# ---------------------------------------------------------------------- levels and codecs
def _codec_json(codec):
    if codec is None:
        return dict(kind="code")                  # a plain uint32 table: the code is the key
    d = dict(codec.to_json())
    if hasattr(codec, "values") and isinstance(getattr(codec, "values"), list):
        d["values"] = [repr(v) for v in codec.values]
    if hasattr(codec, "raw"):
        d["raw"] = [int(v) for v in codec.raw]
    return d


def codec_descriptor(scope):
    """the logical codec descriptor of a scope's keys (bytes): for each key its coltypes codec (kind, offset, scale,
    dictionary), canonical JSON; b"" for an array scope (positional ids). Part of every map binding."""
    if scope.source == "arrays":
        return b""
    return json.dumps([_codec_json(k["codec"]) for k in scope.keys], sort_keys=True,
                      separators=(",", ":")).encode()


class _Levels:
    """native tr_multi_level entries for scopes (the codec bytes kept alive with them)"""
    def __init__(self, scopes):
        self.scopes = list(scopes)
        self.codecs = [codec_descriptor(s) for s in self.scopes]
        self._bufs = [C.create_string_buffer(c, max(len(c), 1)) for c in self.codecs]
        self.array = (_Level * max(len(self.scopes), 1))()
        for i, (s, c, b) in enumerate(zip(self.scopes, self.codecs, self._bufs)):
            s._open()
            self.array[i] = _Level(s._handle, C.addressof(b) if c else None, len(c))

    def at(self, i):
        return C.cast(C.addressof(self.array) + i * C.sizeof(_Level), P(_Level))


def _fresh(*scopes, what):
    for s in scopes:
        if s.source == "table":
            s.require_fresh(what)


# ---------------------------------------------------------------------- maps
class Map:
    """An immutable refinement map from the groups of a fine level to those of a coarse level (native): proven on the
    rows of two levels (certify, verify) or declared (declare). ids[g] is the coarse group of fine group g, or
    UNMAPPED."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_multi.certify(...), math_multi.declare(...) or Map.verify / compose")

    @classmethod
    def _wrap(cls, handle, fine, coarse, backend="cpu", device="native CPU"):
        self = object.__new__(cls)
        self._handle, self.fine, self.coarse, self.backend, self.device = handle, fine, coarse, backend, device
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the map is closed")

    def _info(self):
        self._open()
        out = _MapInfo()
        _check(_call("tr_multi_map_get_info", self._handle, C.byref(out)), what="map")
        return out

    @property
    def status(self):
        return STATUS[self._info().status]

    @property
    def proven(self):
        return self.status == "proven"

    @property
    def info(self):
        i = self._info()
        return dict(fine_groups=int(i.nfine), coarse_groups=int(i.ncoarse), mapped=int(i.mapped),
                    checked=int(i.checked), bytes=int(i.bytes), status=STATUS[i.status], rows_bound=bool(i.rows_bound),
                    backend=self.backend, device=self.device)

    @property
    def ids(self):
        """copy of the entries: uint32 coarse group per fine group (UNMAPPED = no parent)"""
        i = self._info()
        out = np.zeros(max(int(i.nfine), 1), dtype=np.uint32)
        _check(_call("tr_multi_map_get", self._handle, out.ctypes.data, len(out)), what="map entries")
        return out[:int(i.nfine)]

    def parent(self, g):
        v = int(self.ids[int(g)])
        return None if v == UNMAPPED else v

    def check(self, fine, coarse):
        """True when the map is bound to exactly these levels (groupings, codecs, and for a proven map their rows)"""
        self._open()
        lv = _Levels([fine, coarse])
        rc = _call("tr_multi_map_check", self._handle, lv.at(0), lv.at(1))
        if rc == -6:
            return False
        _check(rc, what="map binding")
        return True

    def equal(self, other):
        self._open(); other._open()
        return bool(_check(_call("tr_multi_map_equal", self._handle, other._handle), what="map comparison"))

    def __eq__(self, other):
        return isinstance(other, Map) and self.equal(other)

    __hash__ = None

    def verify(self, fine, coarse, *, budget=None):
        """this map proven exhaustively on the data of two levels (a new proven Map), or RefinementError with the
        smallest selected row where it disagrees with the data"""
        self._open()
        _fresh(fine, coarse, what="map verification")
        lv = _Levels([fine, coarse])
        error, refusal, w = C.c_int32(), _Refusal(), _Witness()
        h = _call("tr_multi_map_verify", self._handle, lv.at(0), lv.at(1), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(w), C.byref(error))
        return Map._wrap(_made(h, error, refusal, "map verification", w, fine, coarse), fine, coarse)

    def compose(self, other, *, budget=None):
        """the map from this map's fine level to other's coarse level (proven when both are proven on the same rows)"""
        self._open(); other._open()
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_multi_map_compose", self._handle, other._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Map._wrap(_made(h, error, refusal, "map composition"), self.fine, other.coarse)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_multi_map_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Map(closed)"
        i = self.info
        return f"Map({i['fine_groups']} -> {i['coarse_groups']} groups, {i['status']}, {i['mapped']} mapped)"


def _map_need_raw(fine, coarse, backend):
    lv = _Levels([fine, coarse])
    out = _Need()
    _check(_call("tr_multi_map_need", lv.at(0), lv.at(1), 0, C.byref(out)), what="map needs")
    if backend == "cuda":                     # the device copies and the adopted map: resource by resource the larger
        dev = _Need()
        _check(_call("tr_multi_map_need", lv.at(0), lv.at(1), 1, C.byref(dev)), what="map needs")
        for i in range(len(RESOURCES)):
            out.need[i] = max(out.need[i], dev.need[i])
    elif backend != "cpu":
        raise ValueError("backend is 'cpu' or 'cuda'")
    return out


def map_need(fine, coarse, backend="cpu"):
    """the needs certify / declare / verify check before allocating ({resource: need}; cuda: the larger of the
    device copies and the adopted map, resource by resource)"""
    out = _map_need_raw(fine, coarse, backend)
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}


def certify(fine, coarse, *, backend="cpu", budget=None):
    """the refinement map from fine's groups to coarse's, proven exhaustively over the selected rows (two scopes of
    the same rows), or RefinementError with the smallest violating row (backend "cuda": the CUDA kernels, the same map
    and witness)"""
    _fresh(fine, coarse, what="certification")
    lv = _Levels([fine, coarse])
    error, refusal, w = C.c_int32(), _Refusal(), _Witness()
    if backend == "cpu":
        h = _call("tr_multi_map_certify", lv.at(0), lv.at(1), C.byref(_budget(budget)), C.byref(refusal), C.byref(w),
                  C.byref(error))
        return Map._wrap(_made(h, error, refusal, "certification", w, fine, coarse), fine, coarse)
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    gpu = _gpu()
    _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(_map_need_raw(fine, coarse, "cuda")),
                 C.byref(refusal)), refusal, "certification on the device")
    G = fine.groups
    first = np.zeros(max(G, 1), dtype=np.uint64)
    lo = np.zeros(max(G, 1), dtype=np.uint32)
    hi = np.zeros(max(G, 1), dtype=np.uint32)
    cert = _Cert(G, first.ctypes.data, lo.ctypes.data, hi.ctypes.data, 0, 0)
    fv, cv = fine.view(), coarse.view()
    gpu.run("tr_mgpu_multi_certify", C.byref(fv), C.byref(cv), C.byref(_budget(budget)), C.byref(cert),
            C.byref(refusal), refusal=refusal, what="certification on the device")
    h = _call("tr_multi_map_from_cert", lv.at(0), lv.at(1), C.byref(cert), C.byref(_budget(budget)), C.byref(refusal),
              C.byref(w), C.byref(error))
    name, dev = _backend_name(gpu)
    return Map._wrap(_made(h, error, refusal, "certification on the device", w, fine, coarse), fine, coarse, name, dev)


def _as_key(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,)


def declare(fine, coarse, mapping, *, budget=None):
    """a map given by the schema (declared, not proven): mapping is {fine key: coarse key} in logical key values (a
    tuple per level, or a scalar for a one-key level; None for a NULL key), translated to group ids through the
    levels' codecs, or an array of coarse group ids per fine group (UNMAPPED for no parent)"""
    if isinstance(mapping, dict):
        index = {coarse.key_values(q): q for q in range(coarse.groups)}
        wanted = {_as_key(k): _as_key(v) for k, v in mapping.items()}
        ids = np.full(fine.groups, UNMAPPED, dtype=np.uint32)
        for g in range(fine.groups):
            target = wanted.get(fine.key_values(g))
            if target is None:
                continue
            if target not in index:
                raise ValueError(f"coarse key {target!r} has no group in the coarse level (not representable in its codec)")
            ids[g] = index[target]
    else:
        ids = np.ascontiguousarray(np.asarray(mapping, dtype=np.int64))
        if ids.shape != (fine.groups,) or (ids.size and (ids.min() < 0 or ids.max() > UNMAPPED)):
            raise ValueError("map: one coarse group id per fine group (UNMAPPED for none)")
        ids = np.ascontiguousarray(ids, dtype=np.uint32)
    lv = _Levels([fine, coarse])
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_multi_map_declare", lv.at(0), lv.at(1), ids.ctypes.data if ids.size else None, len(ids),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Map._wrap(_made(h, error, refusal, "declared map (coarse ids < coarse groups)"), fine, coarse)


# ---------------------------------------------------------------------- hierarchies
def hierarchy_need(levels, bank):
    """the needs a hierarchy build checks before allocating ({resource: need}): the coarse banks and their scratch"""
    lv = _Levels(levels)
    out = _Need()
    _check(_call("tr_multi_hier_need", lv.array, len(levels), bank._handle, C.byref(out)), what="hierarchy needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}


def _q_value(q):
    num = sum(int(q.num[i]) << (64 * i) for i in range(4))
    if num >> 255:
        num -= 1 << 256
    den = int(q.den[0]) | (int(q.den[1]) << 64)
    return Fraction(num, den), num, den


class Hierarchy:
    """An immutable hierarchy of levels (finest first) with one FND-004 bank per level: the finest built once, the
    others its exact rollups along the maps (native)."""

    def __init__(self, levels, basis=None, mode="sql", *, maps=None, bank=None, backend="cpu", budget=None):
        levels = list(levels)
        if len(levels) > MAX_LEVELS:
            raise UnsupportedError(f"{len(levels)} levels: a hierarchy has 2..{MAX_LEVELS}")
        if len(levels) < 2:
            raise ValueError(f"a hierarchy has 2..{MAX_LEVELS} levels")
        _fresh(*levels, what="hierarchy")
        if bank is None:
            if basis is None:
                raise ValueError("give the basis of the finest bank (or bank=)")
            bank = mbk.build(levels[0], basis, mode, backend=backend, budget=budget)
        if maps is None:
            maps = [certify(levels[i], levels[i + 1], backend=backend, budget=budget) for i in range(len(levels) - 1)]
        maps = list(maps)
        if len(maps) != len(levels) - 1:
            raise ValueError("one map per pair of consecutive levels")
        lv = _Levels(levels)
        arr = (C.c_void_p * len(maps))(*[m._handle for m in maps])
        error, refusal, bad = C.c_int32(), _Refusal(), C.c_uint64(_NONE)
        h = _call("tr_multi_hier_build", lv.array, len(levels), bank._handle, arr, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(bad), C.byref(error))
        self._init(h, error, refusal, bad, levels, maps, bank, "hierarchy")

    def _init(self, h, error, refusal, bad, levels, maps, bank, what):
        if error.value == -6 and bad.value != _NONE:
            raise DomainError(f"{what}: group {bad.value} has rows but no parent in its map")
        self._handle = _made(h, error, refusal, what)
        self.levels, self.names = tuple(levels), tuple(bank.names)
        self.maps_given = tuple(maps)
        self.bank_backend = bank.backend

    def _open(self):
        if not self._handle:
            raise ValueError("the hierarchy is closed")

    def _info(self):
        self._open()
        out = _HierInfo()
        _check(_call("tr_multi_hier_get_info", self._handle, C.byref(out)), what="hierarchy")
        return out

    @property
    def status(self):
        return STATUS[self._info().status]

    @property
    def depth(self):
        return int(self._info().levels)

    @property
    def info(self):
        i = self._info()
        L = int(i.levels)
        return dict(levels=L, status=STATUS[i.status], bytes=int(i.bytes), rows=int(i.rows), work=int(i.work),
                    changed=int(i.changed), terms=int(i.nterms),
                    groups=[int(i.groups[k]) for k in range(L)], present=[int(i.present[k]) for k in range(L)],
                    level_bytes=[int(i.level_bytes[k]) for k in range(L)],
                    level_work=[int(i.level_work[k]) for k in range(L)],
                    scoped=[bool(i.scoped[k]) for k in range(L)],
                    strategy=[STRATEGY[i.strategy[k]] for k in range(L)],
                    maps=[STATUS[i.map_status[k]] for k in range(L - 1)])

    def _level(self, level):
        level = int(level)
        if not 0 <= level < self.depth:
            raise ValueError(f"level {level} out of range (0..{self.depth - 1})")
        return level

    def bank(self, level):
        """an independently owned FND-004 Bank of the level (scoped to the level's scope when proven so)"""
        level = self._level(level)
        error = C.c_int32()
        h = _call("tr_multi_hier_bank", self._handle, level, C.byref(error))
        _check(error.value, what="level bank")
        scoped = self._info().scoped[level]
        return mbk.Bank._wrap(h, self.levels[level] if scoped else None, self.names, "cpu", "native CPU")

    def state(self, level, g):
        return self.bank(level).state(g)

    def states(self, level):
        return self.bank(level).states()

    def table(self, level):
        """the present groups of a level with their keys decoded by the level's scope"""
        b = self.bank(level)
        scope = self.levels[self._level(level)]
        return [dict(row, key=scope.key_values(row["group"])) for row in b.table()]

    def map(self, level):
        """a copy of the map from level to level + 1"""
        level = self._level(level)
        if level + 1 >= self.depth:
            raise ValueError("the coarsest level has no map")
        error = C.c_int32()
        h = _call("tr_multi_hier_map", self._handle, level, C.byref(error))
        _check(error.value, what="hierarchy map")
        return Map._wrap(h, self.levels[level], self.levels[level + 1])

    def update(self, bank, *, levels=None, maps=None, backend="cpu", budget=None):
        """the hierarchy of a new finest bank (same basis, mode, columns and groups), recomputing only the coarse cells
        it must. Without levels the maps are carried (status "carried": proven on the earlier data); with levels (the
        scopes of the new data) the maps are those given or certified on them, and the result is their hierarchy."""
        self._open()
        error, refusal, bad = C.c_int32(), _Refusal(), C.c_uint64(_NONE)
        if levels is None:
            if maps is not None:
                raise ValueError("maps need the levels they are bound to")
            h = _call("tr_multi_hier_update", self._handle, bank._handle, None, None, C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(bad), C.byref(error))
            out = object.__new__(Hierarchy)
            out._init(h, error, refusal, bad, self.levels, self.maps_given, bank, "hierarchy update")
            return out
        levels = list(levels)
        _fresh(*levels, what="hierarchy update")
        if maps is None:
            maps = [certify(levels[i], levels[i + 1], backend=backend, budget=budget) for i in range(len(levels) - 1)]
        maps = list(maps)
        if len(levels) != self.depth or len(maps) != self.depth - 1:
            raise ValueError("the same number of levels and one map per pair of consecutive levels")
        lv = _Levels(levels)
        arr = (C.c_void_p * len(maps))(*[m._handle for m in maps])
        h = _call("tr_multi_hier_update", self._handle, bank._handle, lv.array, arr, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(bad), C.byref(error))
        out = object.__new__(Hierarchy)
        out._init(h, error, refusal, bad, levels, maps, bank, "hierarchy update")
        return out

    def links(self, level, *, budget=None):
        """(parent of each child group, next present sibling of each child group) of child level `level`, uint32 with
        UNMAPPED for none"""
        level = self._level(level)
        G = self._info().groups[level]
        parent = np.zeros(max(G, 1), dtype=np.uint32)
        nxt = np.zeros(max(G, 1), dtype=np.uint32)
        refusal = _Refusal()
        _check(_call("tr_multi_hier_links", self._handle, level, parent.ctypes.data, nxt.ctypes.data, len(parent),
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "links")
        return parent[:G], nxt[:G]

    def contrasts_need(self, level, terms=None, groups=None, backend="cpu"):
        level, sel, first, count = self._contrast_args(level, terms, groups)
        out = _Need()
        _check(_call("tr_multi_contrasts_need", self._handle, level, len(sel), count, 0 if backend == "cpu" else 1,
                     C.byref(out)), what="contrast needs")
        return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}

    def _contrast_args(self, level, terms, groups):
        level = self._level(level)
        info = self._info()
        if level + 1 >= int(info.levels):
            raise ValueError("contrasts need a child level with a parent level (level < levels - 1)")
        m = int(info.nterms)
        sel = list(range(m)) if terms is None else [int(t) for t in terms]
        G = int(info.groups[level])
        first, count = (0, G) if groups is None else (int(groups[0]), int(groups[1]) - int(groups[0]))
        if count < 0 or first < 0:
            raise ValueError("groups: a range (first, stop) of child group ids")
        return level, sel, first, count

    def contrasts(self, level, kind="mean", *, terms=None, groups=None, backend="cpu", budget=None):
        """exact contrasts of child level `level` (see docs/math/multi.md): kind "mean" (child mean - parent mean),
        "total" (child total - parent total), "share" (child total - its share of the parent's total), "sibling_mean"
        / "sibling_total" (against the next present sibling); per term (terms: indices into the basis, default all)
        and child group (groups: a range (first, stop), default all). Values are Fractions in logical units (term j
        carries the decimal scale of its columns); None where the status is not "defined"."""
        if kind not in KINDS:
            raise ValueError(f"kind is one of {', '.join(KINDS)}")
        level, sel, first, count = self._contrast_args(level, terms, groups)
        cells = (_Q * max(count * len(sel), 1))()
        tv = (C.c_int32 * max(len(sel), 1))(*sel)
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_multi_contrasts", self._handle, level, KINDS[kind], tv, len(sel), first, count,
                         C.byref(_budget(budget)), cells, C.byref(refusal)), refusal, "contrasts")
            name, dev = "cpu", "native CPU"
        elif backend == "cuda":
            gpu = _gpu()
            need = self.contrasts_need(level, sel, (first, first + count), backend="cuda")
            raw = _Need()
            for i, r in enumerate(RESOURCES):
                raw.need[i] = need.get(r, 0)
            _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw), C.byref(refusal)), refusal,
                   "contrasts on the device")
            parent, nxt = self.links(level, budget=budget)
            child_st, parent_st = mbk._States(), mbk._States()
            _check(_call("tr_multi_hier_states", self._handle, level, C.byref(child_st)), what="states")
            _check(_call("tr_multi_hier_states", self._handle, level + 1, C.byref(parent_st)), what="states")
            gpu.run("tr_mgpu_multi_contrasts", C.byref(child_st), C.byref(parent_st), parent.ctypes.data,
                    nxt.ctypes.data, KINDS[kind], tv, len(sel), first, count, C.byref(_budget(budget)), cells,
                    C.byref(refusal), refusal=refusal, what="contrasts on the device")
            name, dev = _backend_name(gpu)
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return Contrasts(self, level, kind, sel, cells, count, name, dev)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_multi_hier_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Hierarchy(closed)"
        i = self.info
        return f"Hierarchy({i['levels']} levels, groups={i['groups']}, {i['status']})"


class Contrasts:
    """The cells of one contrast request: rows() gives per child group its key, the group it is contrasted with, and
    per term the status and the exact value (Fraction, logical units) or None; raw() the native cells."""

    def __init__(self, hierarchy, level, kind, terms, cells, count, backend, device):
        self.level, self.kind, self.terms, self.backend, self.device = level, kind, tuple(terms), backend, device
        self._cells, self._count, self._h = cells, count, hierarchy

    def _columns(self):
        """the native cells as columns: child, other, term, status (names) and exact numerators / denominators
        (Python ints; 0 / 0 where not defined), read from a numpy view of the native buffer"""
        k = self._count * len(self.terms)
        a = np.ctypeslib.as_array(self._cells)[:k]
        num, den = a["num"], a["den"]
        status = a["status"].tolist()
        lo = num[:, 0].view(np.int64)
        ones = np.uint64(0xFFFFFFFFFFFFFFFF)
        small = (((num[:, 1] == 0) & (num[:, 2] == 0) & (num[:, 3] == 0) & (lo >= 0)) |
                 ((num[:, 1] == ones) & (num[:, 2] == ones) & (num[:, 3] == ones) & (lo < 0)))
        nums = lo.tolist()
        for i in np.flatnonzero(~small).tolist():                 # numerators past int64: their 256-bit words
            v = sum(int(w) << (64 * b) for b, w in enumerate(num[i].tolist()))
            nums[i] = v - (1 << 256) if v >> 255 else v
        dens = den[:, 0].tolist()
        for i in np.flatnonzero(den[:, 1] != 0).tolist():
            dens[i] = int(den[i, 0]) | (int(den[i, 1]) << 64)
        other = a["other"].tolist()
        return (a["child"].tolist(), [None if o == UNMAPPED else o for o in other], a["term"].tolist(),
                [CELL[q] for q in status], nums, dens)

    def raw(self):
        """[(child, other or None, term, status, numerator, denominator)] as native ints (canonical when defined;
        0, 0 otherwise)"""
        return list(zip(*self._columns()))

    def rows(self):
        h, n = self._h, len(self.terms)
        child_scope = h.levels[self.level]
        other_scope = h.levels[self.level if self.kind.startswith("sibling") else self.level + 1]
        child, other, _, status, nums, dens = self._columns()
        out = []
        for c in range(self._count):
            i = c * n
            g, o = child[i], other[i]
            out.append(dict(group=g, key=child_scope.key_values(g), other=o,
                            other_key=None if o is None else other_scope.key_values(o),
                            status=status[i:i + n],
                            value=[Fraction(nums[j], dens[j]) if status[j] == "defined" else None
                                   for j in range(i, i + n)]))
        return out

    def __len__(self):
        return self._count
