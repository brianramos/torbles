# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""MULTI-002: adaptive refinement over MULTI-001 hierarchies (requests answered from the coarsest level down with
sound per-region bounds, only undecided regions refined) and change localization (DELTA-001 updates rolled up along
the affected paths only), natively on the CPU or with the CUDA kernels (torbles/math_refine.c, math_refine_gpu.cu).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_multi as mm, math_delta as md, math_refine as mr
    h = mm.Hierarchy([day, month, year], ["1", "x", "x^2"])
    st = mr.State(h)                                           # cells, UNC-001 inputs and links of every level
    r = st.threshold(0, "sum:x", ">", 1000)                    # the leaves whose SUM(x) > 1000, refined adaptively
    r.answer, r.unresolved, r.entries, r.stats                 # target groups; every entry with its sound bound
    st.ratio(1, "sum:x", "count:x", ">=", Fraction(5, 2), backend="cuda")    # AVG(x) >= 5/2 (D = 0: undefined)
    st.topk(0, {"sum:x": 1, "rows": -3}, 10, expand_limit=50)  # unexpanded regions stay "unresolved" with bounds
    st.fine("threshold", 0, "sum:x", op=">", threshold=1000)   # the complete fine-grain answer (the baseline)
    st2 = st.update(md.Delta.from_arrays(...), levels=[day2, month2, year2])   # local, or rebuilt when needed
    st2.changes, st2.info["strategy"]                          # every changed region with its exact change

Aggregates are integer combinations of atoms over the hierarchy's FND-004 basis: "sum:<term>", "count:<term>" and
"rows" (a term in math_bank's syntax: "x", "x*d", "x^2", "1"), in the bank's logical integer units. Nothing is
computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
import re
from decimal import Decimal
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_delta as md
from . import math_multi as mm
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _Q
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check
from .math_unc import _Input

__all__ = ["State", "Result", "Fine", "RebuildRequired", "KINDS", "OPS", "DECISIONS", "MAX_ATOMS"]

MAX_ATOMS, MAX_CLASSES, MAX_NODES, MAX_LEVELS = 4, 18, 512, 8
KINDS = {"threshold": 1, "ratio": 2, "topk": 3}
OPS = {">": 1, ">=": 2, "<": 3, "<=": 4}
ATOMS = {"sum": 1, "count": 2, "rows": 3}
DECISIONS = {1: "all", 2: "none", 3: "unresolved", 4: "true", 5: "false", 6: "undefined", 7: "in", 8: "out"}
FLAGS = {"lo_inf": 1, "hi_inf": 2, "dlo_inf": 4, "dhi_inf": 8, "exact": 16, "merged": 32}
STRATEGIES = {0: None, 2: "local", 3: "copy", 4: "rebuild"}
REASONS = {0: None, 1: "schema", 2: "overflow", 3: "new group"}
STATUS = {1: "proven", 2: "declared", 3: "carried"}
PROMISES = {"nonnegative": 1, "signed": 2}
_U64 = (1 << 64) - 1
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Atom(C.Structure):
    _native_ = "tr_refine_atom"
    _fields_ = [("kind", C.c_int32), ("term", C.c_int32), ("coef", C.c_int64)]


class _Request(C.Structure):
    _native_ = "tr_refine_request"
    _fields_ = [(n, C.c_int32) for n in ("kind", "level", "op", "natoms", "ndatoms", "descending", "reserved")] + [
        ("k", C.c_uint32), ("threshold", _Q), ("expand_limit", C.c_uint64), ("atoms", _Atom * MAX_ATOMS),
        ("datoms", _Atom * MAX_ATOMS)]


class _Class(C.Structure):
    _native_ = "tr_refine_class"
    _fields_ = [(n, C.c_int32) for n in ("first_atom", "natoms", "count_term", "merged", "first_node", "nnodes",
                                          "root", "nlive")]


class _Plan(C.Structure):
    _native_ = "tr_refine_plan"
    _fields_ = [(n, C.c_int32) for n in ("naggs", "ncols", "nterms", "mode")] + [
        (n, C.c_int32 * 2) for n in ("first", "nclasses", "merged")] + [
        (n, C.c_int32) for n in ("natoms", "nclass", "nnodes", "nlive")] + [
        ("atoms", C.c_void_p), ("classes", C.c_void_p), ("nodes", C.c_void_p)]


class _LevelView(C.Structure):
    _native_ = "tr_refine_level_view"
    _fields_ = [("ngroups", C.c_uint64), ("nterms", C.c_int32), ("ncols", C.c_int32), ("rows", C.c_void_p),
                ("count", C.c_void_p), ("sum", C.c_void_p), ("inputs", C.c_void_p)]


class _Bound(C.Structure):
    _native_ = "tr_refine_bound"
    _fields_ = [("flags", C.c_int32), ("reserved", C.c_int32), ("group", C.c_uint32), ("level", C.c_uint32)] + [
        (n, C.c_uint64 * 4) for n in ("lo", "hi", "dlo", "dhi")]


class _Entry(C.Structure):
    _native_ = "tr_refine_entry"
    _fields_ = [(n, C.c_int32) for n in ("level", "decision", "flags", "reserved")] + [
        ("group", C.c_uint32), ("rank", C.c_uint32), ("rows", C.c_uint64)] + [
        (n, C.c_uint64 * 4) for n in ("lo", "hi", "dlo", "dhi")]


class _Stats(C.Structure):
    _native_ = "tr_refine_stats"
    _fields_ = [("evaluated", C.c_uint64 * MAX_LEVELS), ("expanded", C.c_uint64 * MAX_LEVELS)] + [
        (n, C.c_uint64) for n in ("entries", "answer", "unresolved", "unresolved_groups", "work", "fine_work",
                                  "enumerated")] + [
        (n, C.c_int32) for n in ("complete", "waves", "target", "levels")]


class _Info(C.Structure):
    _native_ = "tr_refine_info"
    _fields_ = [(n, C.c_int32) for n in ("levels", "ncols", "nterms", "mode", "status", "promise", "strategy",
                                          "reason")] + [
        (n, C.c_uint64 * MAX_LEVELS) for n in ("groups", "present", "level_bytes", "changed", "loose")] + [
        (n, C.c_uint64) for n in ("bytes", "rows", "step", "touched", "work", "rebuild_work")] + [
        ("identity", C.c_uint64 * 2)]


class _Change(C.Structure):
    _native_ = "tr_refine_change"
    _fields_ = [("level", C.c_int32), ("reserved", C.c_int32), ("group", C.c_uint32), ("reserved2", C.c_uint32),
                ("drows", C.c_int64)]


class _Ins(C.Structure):
    _native_ = "tr_refine_ins"
    _fields_ = [("count", C.c_uint64), ("min", C.c_int64), ("max", C.c_int64), ("mod", C.c_uint64),
                ("dels", C.c_uint64)]


class _UpdateView(C.Structure):
    _native_ = "tr_refine_update_view"
    _fields_ = [(n, C.c_int32) for n in ("levels", "nterms", "ncols", "reserved")] + [
        ("naff", C.c_uint64 * MAX_LEVELS), ("off", C.c_uint64 * MAX_LEVELS)] + [
        (n, C.c_void_p) for n in ("ids", "child_off", "child_idx", "drows", "dcount", "dsum", "dpresent", "ins")]


for _s, _n in ((_Atom, 16), (_Request, 184), (_Class, 32), (_Plan, 80), (_LevelView, 48), (_Bound, 144),
               (_Entry, 160), (_Stats, 200), (_Info, 416), (_Change, 24), (_Ins, 40), (_UpdateView, 208)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R, N, V = P(_Budget), P(_Refusal), P(_Need), P(md._Violation)
        sig = {
            "tr_refine_build": (vp, [vp, vp, i32, B, R, e]),
            "tr_refine_build_need": (i32, [vp, vp, N]),
            "tr_refine_get_info": (i32, [vp, P(_Info)]),
            "tr_refine_get_level": (i32, [vp, i32, P(_LevelView)]),
            "tr_refine_get_links": (i32, [vp, i32, P(vp), P(vp), P(vp)]),
            "tr_refine_tight": (i32, [vp, i32, u64]),
            "tr_refine_free": (None, [vp]),
            "tr_refine_begin": (vp, [vp, P(_Request), B, R, e]),
            "tr_refine_query_need": (i32, [vp, P(_Request), N]),
            "tr_refine_bounds_need": (i32, [P(_LevelView), P(_Plan), u64, N]),
            "tr_refine_wave": (i32, [vp, P(i32), P(i32), P(vp), P(u64)]),
            "tr_refine_supply": (i32, [vp, vp, u64]),
            "tr_refine_run": (i32, [vp]),
            "tr_refine_get_plan": (i32, [vp, P(_Plan)]),
            "tr_refine_get_stats": (i32, [vp, P(_Stats)]),
            "tr_refine_get_entries": (i32, [vp, vp, u64, P(u64)]),
            "tr_refine_get_answer": (i32, [vp, vp, vp, u64]),
            "tr_refine_query_free": (None, [vp]),
            "tr_refine_eval_bounds": (i32, [P(_LevelView), P(_Plan), i32, i32, vp, u64, vp]),
            "tr_refine_fine": (i32, [vp, P(_Request), vp, u64, vp, P(_Stats), B, R]),
            "tr_refine_update_state": (vp, [vp, vp, vp, i32, B, R, V, e]),
            "tr_refine_update_prepare": (vp, [vp, vp, B, R, V, e, e]),
            "tr_refine_update_need": (i32, [vp, vp, N]),
            "tr_refine_rollup_need": (i32, [P(_UpdateView), u64, N]),
            "tr_refine_update_view_get": (i32, [vp, P(_UpdateView), P(u64)]),
            "tr_refine_update_rollup": (i32, [vp]),
            "tr_refine_update_adopt": (i32, [vp]),
            "tr_refine_update_finish": (vp, [vp, B, R, e]),
            "tr_refine_update_free": (None, [vp]),
            "tr_refine_get_changes": (i32, [vp, vp, vp, vp, u64, P(u64)]),
            "tr_math_budget_check": (i32, [B, N, R]),
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
    math_gpu.bind("tr_mgpu_refine_bounds", C.c_int, [P(_LevelView), P(_Plan), C.c_int32, C.c_int32, C.c_void_p,
                                                    C.c_uint64, C.c_void_p, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_refine_rollup", C.c_int, [P(_UpdateView), C.c_uint64, P(_Budget), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _w256(words):
    """a 256-bit two's complement value (four uint64 words, least significant first) as a Python int"""
    v = 0
    for i in range(3, -1, -1):
        v = (v << 64) | (int(words[i]) & _U64)
    return v - (1 << 256) if v >> 255 else v


def _w192(words):
    v = (int(words[0]) & _U64) | ((int(words[1]) & _U64) << 64) | ((int(words[2]) & _U64) << 128)
    return v - (1 << 192) if v >> 191 else v


def _array(ptr, dtype, count):
    """a copy of count native values at ptr"""
    dtype = np.dtype(dtype)
    if not count:
        return np.zeros(0, dtype=dtype)
    buf = (C.c_char * (count * dtype.itemsize)).from_address(ptr)
    return np.frombuffer(buf, dtype=dtype).copy()


def _bits(words, n):
    """the set bits of a bitmap of n entries"""
    if not n:
        return frozenset()
    b = np.unpackbits(np.ascontiguousarray(words).view(np.uint8), bitorder="little")[:n]
    return frozenset(int(g) for g in np.flatnonzero(b))


def _rational(v, what):
    if isinstance(v, bool) or isinstance(v, float):
        raise TypeError(f"{what}: give an exact integer, Fraction, Decimal or string (not {type(v).__name__})")
    if isinstance(v, (Integral, Fraction, Decimal, str)):
        q = Fraction(v)
    else:
        raise TypeError(f"{what}: give an exact integer, Fraction, Decimal or string")
    lim = 1 << 63
    if not (-lim <= q.numerator < lim and q.denominator < lim):
        raise UnsupportedError(f"{what}: {q} does not fit an int64 rational")
    return q


def _bound(raw, *, entry=False):
    f = int(raw.flags)
    out = dict(level=int(raw.level), group=int(raw.group),
               lo=None if f & 1 else _w256(raw.lo), hi=None if f & 2 else _w256(raw.hi),
               dlo=None if f & 4 else _w256(raw.dlo), dhi=None if f & 8 else _w256(raw.dhi),
               exact=bool(f & 16), merged=bool(f & 32))
    if entry:
        out.update(decision=DECISIONS[int(raw.decision)], rank=int(raw.rank), rows=int(raw.rows))
    return out


class RebuildRequired(DomainError):
    """A localized update is not compatible with the state (.reason: "schema" or "new group"); pass the new levels
    (levels=) for a complete rebuild."""


_ATOM = re.compile(r"^\s*(sum|count)\s*(?::\s*(.+?)|\(\s*(.+?)\s*\))\s*$", re.I)


# ------------------------------------------------------------------ results
class Result:
    """the answer of an adaptive request: answer (target groups certainly in it), unresolved (target groups under
    unresolved entries), entries (each final region or group with its decision and sound bound: lo / hi None when
    unbounded; RATIO: lo / hi bound C = q A - p D and dlo / dhi bound D), stats"""

    def __init__(self, kind, level, entries, answer, unresolved, stats, backend, device, ngroups):
        self.kind, self.level, self.backend, self.device = kind, level, backend, device
        self._entries, self._answer, self._unresolved, self._stats = entries, answer, unresolved, stats
        self.answer, self.unresolved = _bits(answer, ngroups), _bits(unresolved, ngroups)
        self.entries = [_bound(e, entry=True) for e in entries]
        self.stats = _stats(stats)
        self.complete = bool(stats.complete)

    @property
    def raw(self):
        return bytes(self._entries) + self._answer.tobytes() + self._unresolved.tobytes() + bytes(self._stats)

    def same(self, other):
        """bit-for-bit equality with another result (entries, answer, unresolved, stats)"""
        return self.raw == other.raw

    def __repr__(self):
        return (f"Result({self.kind}, level {self.level}: {len(self.answer)} in, {len(self.unresolved)} unresolved, "
                f"{self.stats['work']} cells; {self.backend})")


def _stats(s):
    L = int(s.levels)
    return dict(evaluated=[int(s.evaluated[i]) for i in range(L)], expanded=[int(s.expanded[i]) for i in range(L)],
                entries=int(s.entries), answer=int(s.answer), unresolved=int(s.unresolved),
                unresolved_groups=int(s.unresolved_groups), work=int(s.work), fine_work=int(s.fine_work),
                enumerated=int(s.enumerated), complete=bool(s.complete), waves=int(s.waves), target=int(s.target),
                levels=L)


class Fine:
    """the complete fine-grain answer: answer (a frozenset), values {group: exact A} (RATIO: (C, D)), stats"""

    def __init__(self, kind, answer, values, stats, ngroups):
        self.kind = kind
        self.answer = _bits(answer, ngroups)
        self.values = values
        self.stats = _stats(stats)


# ------------------------------------------------------------------ states
class State:
    """An immutable refinement state of a MULTI-001 hierarchy (native): per level the exact cells, the UNC-001 abstract
    inputs of every group and column, the links, and the DELTA-001 state of the leaves."""

    def __init__(self, hierarchy, *, promise="nonnegative", budget=None):
        if promise not in PROMISES:
            raise ValueError("promise is 'nonnegative' (a signed measure is not supported: the bounds count rows)")
        hierarchy._open()
        scope0 = hierarchy.levels[0]
        if scope0.source == "table":
            scope0.require_fresh("refinement state")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_refine_build", hierarchy._handle, scope0._handle, PROMISES[promise], C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        self._init(_made(h, error, refusal, "refinement state"), hierarchy.names,
                   hierarchy.bank(0).terms, hierarchy.levels, "cpu", "native CPU")

    def _init(self, handle, names, terms, levels, backend, device):
        self._handle, self.names, self.terms, self.levels = handle, tuple(names), tuple(terms), tuple(levels)
        self.backend, self.device = backend, device

    @classmethod
    def _wrap(cls, handle, like, levels, backend, device):
        self = object.__new__(cls)
        self._init(handle, like.names, like.terms, levels, backend, device)
        return self

    @staticmethod
    def need(hierarchy):
        """the needs a build checks (they cover its UNC-001 and DELTA-001 steps)"""
        out = _Need()
        _check(_call("tr_refine_build_need", hierarchy._handle, hierarchy.levels[0]._handle, C.byref(out)),
               what="refinement state needs")
        return _needs(out)

    def _open(self):
        if not self._handle:
            raise ValueError("the refinement state is closed")
        return self._handle

    def _info(self):
        out = _Info()
        _check(_call("tr_refine_get_info", self._open(), C.byref(out)), what="refinement state")
        return out

    @property
    def info(self):
        i = self._info()
        L = int(i.levels)
        return dict(levels=L, columns=int(i.ncols), terms=int(i.nterms), mode={0: "sql", 1: "joint"}[i.mode],
                    status=STATUS[i.status], promise={1: "nonnegative", 2: "signed"}[i.promise],
                    strategy=STRATEGIES[i.strategy], reason=REASONS[i.reason],
                    groups=[int(i.groups[k]) for k in range(L)], present=[int(i.present[k]) for k in range(L)],
                    level_bytes=[int(i.level_bytes[k]) for k in range(L)],
                    changed=[int(i.changed[k]) for k in range(L)], loose=[int(i.loose[k]) for k in range(L)],
                    bytes=int(i.bytes), rows=int(i.rows), step=int(i.step), touched=int(i.touched),
                    work=int(i.work), rebuild_work=int(i.rebuild_work),
                    identity=(int(i.identity[1]) << 64) | int(i.identity[0]))

    @property
    def depth(self):
        return int(self._info().levels)

    def _level(self, level):
        level = int(level)
        if not 0 <= level < self.depth:
            raise ValueError(f"level {level} out of range (0..{self.depth - 1})")
        return level

    def _view(self, level):
        v = _LevelView()
        _check(_call("tr_refine_get_level", self._open(), self._level(level), C.byref(v)), what="refinement level")
        return v

    def cells(self, level):
        """copies of the level's cells: rows[G] (uint64), count[G, m] (uint64), sums [[int] * m] * G (exact)"""
        v = self._view(level)
        G, m = int(v.ngroups), int(v.nterms)
        rows = _array(v.rows, np.uint64, G)
        count = _array(v.count, np.uint64, G * m).reshape(G, m)
        raw = _array(v.sum, np.int64, 2 * G * m).reshape(G * m, 2)
        sums = [[(int(raw[g * m + j, 1]) << 64) | (int(raw[g * m + j, 0]) & _U64) for j in range(m)] for g in range(G)]
        return dict(rows=rows, count=count, sum=sums)

    def inputs(self, level):
        """the UNC-001 abstract inputs of the level: a list per group of (rows, present, min, max, modulus) per column"""
        v = self._view(level)
        G, c = int(v.ngroups), int(v.ncols)
        arr = (_Input * max(G * c, 1)).from_address(v.inputs) if G * c else []
        return [[(int(arr[g * c + j].rows), int(arr[g * c + j].present), int(arr[g * c + j].min),
                  int(arr[g * c + j].max), int(arr[g * c + j].modulus)) for j in range(c)] for g in range(G)]

    def links(self, level):
        """parent[G] of the level (None at the coarsest) and its children (offsets[G + 1], ids; None at level 0)"""
        level = self._level(level)
        parent, off, ids = C.c_void_p(), C.c_void_p(), C.c_void_p()
        _check(_call("tr_refine_get_links", self._open(), level, C.byref(parent), C.byref(off), C.byref(ids)),
               what="refinement links")
        i = self._info()
        G = int(i.groups[level])
        out = dict(parent=_array(parent.value, np.uint32, G) if parent.value else None, offsets=None, children=None)
        if off.value:
            offsets = _array(off.value, np.uint64, G + 1)
            out["offsets"] = offsets
            out["children"] = _array(ids.value, np.uint32, int(offsets[-1]))
        return out

    def tight(self, level, g):
        """True when the group's extrema are exact (no deletion since), False when they are enclosures"""
        return bool(_check(_call("tr_refine_tight", self._open(), self._level(level), int(g)), what="tightness"))

    # ---------------------------------------------------------------- requests
    def _term(self, t):
        e = tuple(int(k) for k in mbk._basis(self.names, [t])[0])
        if e not in self.terms:
            raise ValueError(f"term {t!r} is not in the hierarchy's basis")
        return self.terms.index(e)

    def _atom(self, spec):
        if isinstance(spec, tuple) and len(spec) == 2 and spec[0] in ATOMS:
            kind, term = spec
            return ATOMS[kind], (0 if kind == "rows" else self._term(term))
        if not isinstance(spec, str):
            raise TypeError(f"atom {spec!r}: 'sum:<term>', 'count:<term>' or 'rows'")
        if spec.strip().lower() in ("rows", "count(*)", "count:*"):
            return ATOMS["rows"], 0
        m = _ATOM.match(spec)
        if not m:
            raise ValueError(f"atom {spec!r}: 'sum:<term>', 'count:<term>' or 'rows'")
        return ATOMS[m.group(1).lower()], self._term(m.group(2) or m.group(3))

    def _aggregate(self, spec, what):
        if isinstance(spec, (str, tuple)):
            items = [(spec, 1)]
        elif isinstance(spec, dict):
            items = list(spec.items())
        else:
            items = [(a, 1) if not (isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], Integral)) else
                     (a[1], a[0]) for a in spec]
        out = []
        for atom, coef in items:
            if isinstance(coef, bool) or not isinstance(coef, Integral):
                raise TypeError(f"{what}: coefficients are exact integers")
            if not -(1 << 63) <= int(coef) < (1 << 63):
                raise UnsupportedError(f"{what}: coefficient {coef} does not fit int64")
            kind, term = self._atom(atom)
            out.append((kind, term, int(coef)))
        if not out:
            raise ValueError(f"{what}: at least one atom")
        if len(out) > MAX_ATOMS:
            raise UnsupportedError(f"{what}: {len(out)} atoms (at most {MAX_ATOMS})")
        return out

    def _request(self, kind, level, a, op=None, threshold=None, d=None, k=None, descending=True, expand_limit=0):
        if kind not in KINDS:
            raise ValueError(f"kind is one of {', '.join(KINDS)}")
        r = _Request()
        r.kind, r.level = KINDS[kind], self._level(level)
        atoms = self._aggregate(a, "aggregate")
        r.natoms = len(atoms)
        for i, (kd, t, c) in enumerate(atoms):
            r.atoms[i] = _Atom(kd, t, c)
        if kind == "ratio":
            datoms = self._aggregate(d, "denominator")
            r.ndatoms = len(datoms)
            for i, (kd, t, c) in enumerate(datoms):
                r.datoms[i] = _Atom(kd, t, c)
        elif d is not None:
            raise ValueError("a denominator belongs to a ratio")
        if kind in ("threshold", "ratio"):
            if op not in OPS:
                raise ValueError(f"op is one of {', '.join(OPS)}")
            q = _rational(threshold, "threshold")
            r.op, r.threshold = OPS[op], _Q(q.numerator, q.denominator)
        else:
            if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= int(k) < (1 << 32):
                raise ValueError("k: an integer 1 .. 2^32 - 1")
            r.k, r.descending = int(k), 1 if descending else 0
        if isinstance(expand_limit, bool) or not isinstance(expand_limit, Integral) or expand_limit < 0:
            raise ValueError("expand_limit: an integer >= 0 (0: no limit)")
        r.expand_limit = int(expand_limit)
        return r

    def request_need(self, kind, level, a, **kw):
        """the needs a request checks (support, circuit, output, iterations, bytes)"""
        out = _Need()
        _check(_call("tr_refine_query_need", self._open(), C.byref(self._request(kind, level, a, **kw)),
                     C.byref(out)), what="request needs")
        return _needs(out)

    def query(self, kind, level, a, *, op=None, threshold=None, d=None, k=None, descending=True, expand_limit=0,
              backend="cpu", budget=None):
        """an adaptive request about the groups of `level` (see threshold / ratio / topk)"""
        req = self._request(kind, level, a, op, threshold, d, k, descending, expand_limit)
        error, refusal = C.c_int32(), _Refusal()
        q = _made(_call("tr_refine_begin", self._open(), C.byref(req), C.byref(_budget(budget)), C.byref(refusal),
                        C.byref(error)), error, refusal, "refinement request")
        try:
            if backend == "cpu":
                _check(_call("tr_refine_run", q), what="refinement")
                name, dev = "cpu", "native CPU"
            elif backend == "cuda":
                g = _gpu()
                plan = _Plan()
                _check(_call("tr_refine_get_plan", q, C.byref(plan)), what="refinement plan")
                lvl, exact, groups, n = C.c_int32(), C.c_int32(), C.c_void_p(), C.c_uint64()
                while _check(_call("tr_refine_wave", q, C.byref(lvl), C.byref(exact), C.byref(groups), C.byref(n)),
                             what="refinement wave"):
                    v = self._view(lvl.value)
                    out = (_Bound * max(int(n.value), 1))()
                    ref = _Refusal()
                    g.run("tr_mgpu_refine_bounds", C.byref(v), C.byref(plan), lvl.value, exact.value, groups,
                          n.value, out, C.byref(_budget(budget)), C.byref(ref), refusal=ref,
                          what="region bounds on the device")
                    _check(_call("tr_refine_supply", q, out, n.value), what="refinement wave")
                d_ = g.device()
                name, dev = ("cuda-emulated" if d_["emulated"] else "cuda"), d_["name"]
            else:
                raise ValueError("backend is 'cpu' or 'cuda'")
            stats = _Stats()
            _check(_call("tr_refine_get_stats", q, C.byref(stats)), what="refinement stats")
            count = C.c_uint64()
            _check(_call("tr_refine_get_entries", q, None, 0, C.byref(count)), what="refinement entries")
            entries = (_Entry * max(int(count.value), 1))()
            _check(_call("tr_refine_get_entries", q, entries, count.value, C.byref(count)), what="refinement entries")
            G = int(self._info().groups[req.level])
            words = max((G + 63) // 64, 1)
            ans, unres = np.zeros(words, dtype=np.uint64), np.zeros(words, dtype=np.uint64)
            _check(_call("tr_refine_get_answer", q, ans.ctypes.data, unres.ctypes.data, words), what="answer")
            _check(_call("tr_refine_get_stats", q, C.byref(stats)), what="refinement stats")
            entries = (_Entry * int(count.value)).from_buffer_copy(bytes(entries)[:int(count.value) * C.sizeof(_Entry)])
            return Result(kind, req.level, entries, ans, unres, stats, name, dev, G)
        finally:
            _call("tr_refine_query_free", q)

    def threshold(self, level, a, op, threshold, **kw):
        """the groups of `level` whose aggregate `a` satisfies `op threshold` (exact rationals)"""
        return self.query("threshold", level, a, op=op, threshold=threshold, **kw)

    def ratio(self, level, a, d, op, threshold, **kw):
        """the groups of `level` whose a / d satisfies `op threshold` (d = 0: undefined, never in the answer)"""
        return self.query("ratio", level, a, op=op, threshold=threshold, d=d, **kw)

    def topk(self, level, a, k, *, descending=True, **kw):
        """the k groups of `level` with the largest (descending) or smallest a, ties by the smaller group id"""
        return self.query("topk", level, a, k=k, descending=descending, **kw)

    def fine(self, kind, level, a, *, op=None, threshold=None, d=None, k=None, descending=True, budget=None):
        """the same request answered from the complete fine-grain states of `level` (the baseline)"""
        req = self._request(kind, level, a, op, threshold, d, k, descending, 0)
        G = int(self._info().groups[req.level])
        words = max((G + 63) // 64, 1)
        ans = np.zeros(words, dtype=np.uint64)
        vals = (_Bound * max(G, 1))()
        stats, refusal = _Stats(), _Refusal()
        _check(_call("tr_refine_fine", self._open(), C.byref(req), ans.ctypes.data, words, vals, C.byref(stats),
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "fine-grain request")
        rows = self.cells(req.level)["rows"]
        values = {}
        for g in range(G):
            if rows[g]:
                b = _bound(vals[g])
                values[g] = (b["lo"], b["dlo"]) if kind == "ratio" else b["lo"]
        return Fine(kind, ans, values, stats, G)

    def bounds(self, kind, level, a, at, groups=None, *, op=">", threshold=0, d=None, k=1, descending=True,
               backend="cpu", budget=None):
        """the region bounds of the request's plan at level `at` (>= level; at == level: the exact values) for the
        given groups (default: the present groups): a list of dicts and the raw bytes. `budget` limits the bound
        evaluation (CUDA: its device copies, bounds_need)."""
        at = self._level(at)
        req = self._request(kind, level, a, op, threshold, d, k, descending, 0)
        if at < req.level:
            raise ValueError("bounds are evaluated at the target level or above")
        v = self._view(at)
        if groups is None:
            groups = np.flatnonzero(_array(v.rows, np.uint64, int(v.ngroups)))
        ids = np.ascontiguousarray(np.asarray(groups, dtype=np.uint32))
        n = len(ids)
        error, refusal = C.c_int32(), _Refusal()
        q = _made(_call("tr_refine_begin", self._open(), C.byref(req), None, C.byref(refusal), C.byref(error)),
                  error, refusal, "refinement request")
        try:
            plan = _Plan()
            _check(_call("tr_refine_get_plan", q, C.byref(plan)), what="refinement plan")
            out = (_Bound * max(n, 1))()
            exact = 1 if at == req.level else 0
            if backend == "cpu":
                _check(_call("tr_refine_eval_bounds", C.byref(v), C.byref(plan), at, exact,
                             ids.ctypes.data if n else None, n, out), what="region bounds")
            elif backend == "cuda":
                ref = _Refusal()
                _gpu().run("tr_mgpu_refine_bounds", C.byref(v), C.byref(plan), at, exact,
                           ids.ctypes.data if n else None, n, out, C.byref(_budget(budget)), C.byref(ref),
                           refusal=ref, what="region bounds on the device")
            else:
                raise ValueError("backend is 'cpu' or 'cuda'")
            return [_bound(out[i]) for i in range(n)], bytes(out)[:n * C.sizeof(_Bound)]
        finally:
            _call("tr_refine_query_free", q)

    def bounds_need(self, kind, level, a, at, n, **kw):
        """the needs tr_mgpu_refine_bounds checks for n regions of level `at`"""
        req = self._request(kind, level, a, kw.get("op", ">"), kw.get("threshold", 0), kw.get("d"), kw.get("k", 1),
                            kw.get("descending", True), 0)
        error, refusal = C.c_int32(), _Refusal()
        q = _made(_call("tr_refine_begin", self._open(), C.byref(req), None, C.byref(refusal), C.byref(error)),
                  error, refusal, "refinement request")
        try:
            plan, out = _Plan(), _Need()
            _check(_call("tr_refine_get_plan", q, C.byref(plan)), what="refinement plan")
            v = self._view(at)
            _check(_call("tr_refine_bounds_need", C.byref(v), C.byref(plan), int(n), C.byref(out)), what="needs")
            return _needs(out)
        finally:
            _call("tr_refine_query_free", q)

    # ---------------------------------------------------------------- change localization
    @property
    def changes(self):
        """the last update's changed regions: [{"level", "group", "rows", "count": [int] * m, "sum": [int] * m}]"""
        count = C.c_uint64()
        _check(_call("tr_refine_get_changes", self._open(), None, None, None, 0, C.byref(count)), what="changes")
        n, m = int(count.value), len(self.terms)
        if not n:
            return []
        ch = (_Change * n)()
        dc = np.zeros(max(n * m, 1), dtype=np.int64)
        ds = np.zeros(max(n * m * 3, 1), dtype=np.uint64)
        _check(_call("tr_refine_get_changes", self._handle, ch, dc.ctypes.data, ds.ctypes.data, n, C.byref(count)),
               what="changes")
        return [dict(level=int(ch[i].level), group=int(ch[i].group), rows=int(ch[i].drows),
                     count=[int(dc[i * m + j]) for j in range(m)],
                     sum=[_w192(ds[(i * m + j) * 3:(i * m + j) * 3 + 3]) for j in range(m)]) for i in range(n)]

    def update_need(self, delta, backend="cpu"):
        """the needs of a localized update (the largest of DELTA-001's apply, the update arrays and the new state;
        backend "cuda" adds the rollup's device copies)"""
        out = _Need()
        _check(_call("tr_refine_update_need", self._open(), delta._handle, C.byref(out)), what="update needs")
        need = _needs(out)
        if backend == "cuda":
            u, failed = self._prepare(delta, None)
            if u is None:
                viol, error, refusal, _ = failed
                md._made(None, error, refusal, "update needs", viol)
            try:
                view, nchild = _UpdateView(), C.c_uint64()
                _check(_call("tr_refine_update_view_get", u, C.byref(view), C.byref(nchild)), what="update view")
                dev = _Need()
                _check(_call("tr_refine_rollup_need", C.byref(view), nchild.value, C.byref(dev)), what="needs")
                for r, v in _needs(dev).items():
                    need[r] = max(need.get(r, 0), v)
            finally:
                _call("tr_refine_update_free", u)
        return need

    def _prepare(self, delta, budget):
        viol, error, refusal, reason = md._Violation(), C.c_int32(), _Refusal(), C.c_int32()
        u = _call("tr_refine_update_prepare", self._open(), delta._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(viol), C.byref(reason), C.byref(error))
        if u:
            return u, None
        return None, (viol, error, refusal, REASONS[reason.value])

    def update(self, delta, *, levels=None, backend="cpu", budget=None):
        """the state after a DELTA-001 delta of the leaves: the leaves' changes rolled up along the affected paths
        only (CPU or the CUDA rollup); when the local path is not compatible (the delta's schema, the overflow proof,
        a leaf that becomes present without a parent) a complete rebuild from `levels` (the new scopes, finest first;
        maps certified on them), else RebuildRequired / OverflowError. A contradicted nonnegative measure raises
        math_delta.NegativeMultiplicityError."""
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend is 'cpu' or 'cuda'")
        u, failed = self._prepare(delta, budget)
        if u is None:
            viol, error, refusal, reason = failed
            if reason and levels is not None:
                return self._rebuild(delta, levels, budget)
            if reason in ("schema", "new group"):
                err = RebuildRequired(f"refinement update: the localized path is not compatible ({reason}); "
                                      "pass levels= for a complete rebuild")
                err.reason = reason
                raise err
            md._made(None, error, refusal, "refinement update", viol)
            raise MemoryError("refinement update: native allocation failed")
        try:
            if backend == "cpu":
                _check(_call("tr_refine_update_rollup", u), what="localized rollup")
                name, dev = "cpu", "native CPU"
            else:
                g = _gpu()
                view, nchild = _UpdateView(), C.c_uint64()
                _check(_call("tr_refine_update_view_get", u, C.byref(view), C.byref(nchild)), what="update view")
                ref = _Refusal()
                g.run("tr_mgpu_refine_rollup", C.byref(view), nchild.value, C.byref(_budget(budget)), C.byref(ref),
                      refusal=ref, what="localized rollup on the device")
                _check(_call("tr_refine_update_adopt", u), what="adopting the device rollup")
                d_ = g.device()
                name, dev = ("cuda-emulated" if d_["emulated"] else "cuda"), d_["name"]
            error, refusal = C.c_int32(), _Refusal()
            h = _call("tr_refine_update_finish", u, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            return State._wrap(_made(h, error, refusal, "refinement update"), self, self.levels, name, dev)
        finally:
            _call("tr_refine_update_free", u)

    def _rebuild(self, delta, levels, budget):
        levels = list(levels)
        lv = mm._Levels(levels)
        viol, error, refusal = md._Violation(), C.c_int32(), _Refusal()
        h = _call("tr_refine_update_state", self._open(), delta._handle, lv.array, len(levels),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(viol), C.byref(error))
        h = md._made(h, error, refusal, "refinement rebuild", viol)
        return State._wrap(h, self, levels, "cpu", "native CPU (rebuild)")

    def update_native(self, delta, *, levels=None, budget=None):
        """tr_refine_update_state in one native call (prepare, CPU rollup, finish; or the rebuild)"""
        lv = mm._Levels(list(levels)) if levels is not None else None
        viol, error, refusal = md._Violation(), C.c_int32(), _Refusal()
        h = _call("tr_refine_update_state", self._open(), delta._handle, lv.array if lv else None,
                  len(levels) if lv else 0, C.byref(_budget(budget)), C.byref(refusal), C.byref(viol), C.byref(error))
        h = md._made(h, error, refusal, "refinement update", viol)
        return State._wrap(h, self, levels if levels is not None else self.levels, "cpu", "native CPU")

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_refine_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not getattr(self, "_handle", None):
            return "State(closed)"
        i = self.info
        return f"State({i['levels']} levels, groups {i['groups']}, {i['bytes']} bytes, {i['status']})"
