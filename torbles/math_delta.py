# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DELTA-001: maintained FND-004 bank states under signed linear updates (insert, delete, weight change, group
move), exact 128-bit with an overflow proof, natively on the CPU or with the CUDA kernels (torbles/math_delta.c,
math_delta_gpu.cu).

    from torbles import math_bank as mbk, math_delta as md
    scope = mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=mask)
    st = md.State.from_bank(mbk.build(scope, basis), promise="nonnegative")    # or "signed"
    d = md.Delta.from_arrays({"x": xs, "d": ds}, valid={"x": vx}, groups=g, ngroups=scope.groups, weights=w)
    d = md.Delta.from_scope(staged_rows_scope, sign=-1)          # the selected rows of a scope (same schema)
    d = md.Delta.between(scope, mbk.Scope.from_table(t, ["x", "d"], group_by=["k"], rows=mask2))   # a table diff
    st2 = st.apply(d, backend="cpu")                             # or "cuda"; NegativeMultiplicityError, OverflowError
    st2.state(g), st2.identity, st2.step, st2.bound
    st2.bank(new_scope)                                          # the FND-004 bank of exactly that scope (a diff)

Weights: +1 an insert, -1 a delete, any other int64 a weight change; a group move is a delete and an insert. A
nonnegative state refuses an update that contradicts an empirical measure (DomainError subclass
NegativeMultiplicityError with .group, .term, .reason, .have, .change, .need). Every new state has a new identity
(old identity, delta identity, step), so cached entries of the old state are never reused. Nothing is computed in
Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C

import numpy as np

from . import algq
from . import math_bank as mbk
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, check as _check

__all__ = ["Delta", "State", "NegativeMultiplicityError", "PROMISES", "REASONS"]

PROMISES = {"nonnegative": 1, "signed": 2}
KINDS = {1: "arrays", 2: "scope", 3: "diff"}
REASONS = {0: None, 1: "negative rows", 2: "negative count", 3: "count above rows", 4: "sum without rows",
           5: "constant term", 6: "negative even sum"}
_NONE = (1 << 64) - 1
_U64 = (1 << 64) - 1
P = C.POINTER


class _Violation(C.Structure):
    _native_ = "tr_delta_violation"
    _fields_ = [("group", C.c_uint64), ("term", C.c_int32), ("reason", C.c_int32), ("have", C.c_int64 * 2),
                ("change", C.c_int64 * 2), ("result", C.c_int64 * 2)]


class _Info(C.Structure):
    _native_ = "tr_delta_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "present", "step", "bytes")] + [
        (n, C.c_uint64 * 2) for n in ("identity", "base", "source")] + [
        ("revision", C.c_uint64), ("table", C.c_uint64), ("rows", C.c_int64)] + [
        (n, C.c_int32) for n in ("ncols", "nterms", "mode", "promise", "bound", "degree")]


class _BatchInfo(C.Structure):
    _native_ = "tr_delta_batch_info"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "ngroups", "touched", "inserted", "deleted", "wabs")] + [
        ("wnet", C.c_int64), ("identity", C.c_uint64 * 2), ("new_digest", C.c_uint64 * 2), ("revision", C.c_uint64),
        ("kind", C.c_int32), ("ncols", C.c_int32)]


class _Rows(C.Structure):
    _native_ = "tr_delta_rows"
    _fields_ = [("rows", C.c_uint64), ("touched", C.c_uint64), ("weights", C.c_void_p), ("local", C.c_void_p),
                ("groups", C.c_void_p), ("wabs", C.c_void_p)]


for _s, _n in ((_Violation, 64), (_Info, 128), (_BatchInfo, 104), (_Rows, 48)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R, N = P(_Budget), P(_Refusal), P(_Need)
        sig = {
            "tr_delta_from_arrays": (vp, [u64, P(mbk._Array), i32, vp, u64, vp, B, R, e]),
            "tr_delta_from_scope": (vp, [vp, vp, i32, B, R, e]),
            "tr_delta_between": (vp, [vp, vp, B, R, e]),
            "tr_delta_need_arrays": (i32, [u64, P(mbk._Array), i32, N]),
            "tr_delta_need_scope": (i32, [vp, N]),
            "tr_delta_need_between": (i32, [vp, vp, N]),
            "tr_delta_get_info": (i32, [vp, P(_BatchInfo)]),
            "tr_delta_get_view": (i32, [vp, P(mbk._View)]),
            "tr_delta_get_rows": (i32, [vp, P(_Rows)]),
            "tr_delta_get_groups": (i32, [vp, vp, u64]),
            "tr_delta_free": (None, [vp]),
            "tr_delta_state_from_bank": (vp, [vp, vp, i32, B, R, e]),
            "tr_delta_state_apply": (vp, [vp, vp, B, R, P(_Violation), e]),
            "tr_delta_state_gather": (i32, [vp, vp, P(mbk._States), B, R]),
            "tr_delta_state_from_cells": (vp, [vp, vp, P(mbk._States), u64, B, R, P(_Violation), e]),
            "tr_delta_state_need": (i32, [vp, vp, i32, N]),
            "tr_delta_state_to_bank": (vp, [vp, vp, B, R, e]),
            "tr_delta_state_get_info": (i32, [vp, P(_Info)]),
            "tr_delta_state_get_states": (i32, [vp, P(mbk._States)]),
            "tr_delta_state_get_exponents": (i32, [vp, vp, u64]),
            "tr_delta_state_equal_bank": (i32, [vp, vp]),
            "tr_delta_state_equal": (i32, [vp, vp]),
            "tr_delta_state_bound_to": (i32, [vp, vp]),
            "tr_delta_state_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_delta_apply", C.c_int, [P(mbk._View), P(_Rows), C.c_void_p, C.c_int32, C.c_int32, C.c_int32,
                                                  P(mbk._States), P(_Budget), P(mbk._States), P(C.c_uint64),
                                                  P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _digest(words):
    return (int(words[1]) << 64) | int(words[0])


def _i128(words):
    return (int(words[1]) << 64) | (int(words[0]) & _U64)


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


class NegativeMultiplicityError(DomainError):
    """An update contradicts the nonnegative empirical measure a state promises (deleting rows that are not there):
    .group, .term (None: the bag multiplicity, else the term index), .reason, .have (the old value), .change (the
    delta's net change), .result, .need (the multiplicity the delete needs, -change, when change < 0)."""


def _made(handle, error, refusal, what, violation=None):
    if error.value == -6 and violation is not None and violation.group != _NONE:
        have, change, result = _i128(violation.have), _i128(violation.change), _i128(violation.result)
        term = None if violation.term < 0 else int(violation.term)
        reason = REASONS[violation.reason]
        err = NegativeMultiplicityError(
            f"{what}: group {violation.group}, {'rows' if term is None else f'term {term}'}: {reason} "
            f"(have {have}, change {change}, result {result})")
        err.group, err.term, err.reason = int(violation.group), term, reason
        err.have, err.change, err.result = have, change, result
        err.need = -change if change < 0 else None
        raise err
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


# ---------------------------------------------------------------------- deltas
class Delta:
    """An immutable batch of weighted rows (native): an id, logical values with validity, a group and a weight per
    row. Weights: +1 insert, -1 delete, other values weight changes."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Delta.from_arrays(...), Delta.from_scope(...) or Delta.between(...)")

    @classmethod
    def _wrap(cls, handle, names, keep=()):
        self = object.__new__(cls)
        self._handle, self.names, self._keep = handle, tuple(names), keep
        return self

    @classmethod
    def from_arrays(cls, columns, *, valid=None, scales=None, groups=None, ngroups=None, weights=None, budget=None):
        """columns: {name: int64 logical values} (in the order of the state's columns); valid: {name: mask};
        scales: {name: decimal scale}; groups: an id per row (< ngroups) or None (one group); weights: int64 per row
        (default +1: inserts)"""
        names = list(columns)
        valid, scales = dict(valid or {}), dict(scales or {})
        arrays, flags, n = [], [], None
        for name in names:
            v = np.asarray(columns[name])
            if v.dtype.kind not in "iub":
                raise TypeError(f"{name!r}: logical values are exact integers (got {v.dtype})")
            a = np.ascontiguousarray(v, dtype=np.int64)
            n = len(a) if n is None else n
            if a.shape != (n,):
                raise ValueError("every column has one value per row")
            arrays.append(a)
            f = valid.get(name)
            flags.append(None if f is None else np.ascontiguousarray(np.asarray(f, dtype=bool), dtype=np.uint8))
            if flags[-1] is not None and flags[-1].shape != (n,):
                raise ValueError(f"{name!r}: one validity flag per row")
        if groups is not None:
            g = np.asarray(groups)
            n = len(g) if n is None else n
            if g.shape != (n,) or (g.size and (g.min() < 0 or g.max() > 0xFFFFFFFF)):
                raise ValueError("groups: one id per row, 0 <= id < 2^32")
            gids = np.ascontiguousarray(g, dtype=np.uint32)
            if ngroups is None:
                raise ValueError("groups need ngroups")
        else:
            gids, ngroups = None, 1 if ngroups is None else ngroups
        if weights is not None:
            w = np.asarray(weights)
            n = len(w) if n is None else n
            if w.dtype.kind not in "iu" or w.shape != (n,):
                raise ValueError("weights: one int64 per row")
            w = np.ascontiguousarray(w, dtype=np.int64)
        if n is None:
            raise ValueError("give the rows: columns, groups or weights")
        if weights is None:
            w = np.ones(n, dtype=np.int64)
        specs = (mbk._Array * max(len(names), 1))()
        for i, name in enumerate(names):
            specs[i] = mbk._Array(arrays[i].ctypes.data if n else None,
                                  flags[i].ctypes.data if flags[i] is not None else None, int(scales.get(name, 0)), 0)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_delta_from_arrays", n, specs, len(names), gids.ctypes.data if gids is not None else None,
                  int(ngroups), w.ctypes.data if n else None, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "delta"), names)

    @classmethod
    def from_scope(cls, scope, *, sign=1, weights=None, budget=None):
        """the selected rows of a FND-004 scope, each with weight `sign` (+1 insert, -1 delete) or weights[physical
        row] (int64)"""
        if scope.source == "table":
            scope.require_fresh("delta")
        w = None
        if weights is not None:
            w = np.ascontiguousarray(np.asarray(weights), dtype=np.int64)
            if w.shape != (scope.rows,):
                raise ValueError("weights: one int64 per physical row of the scope")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_delta_from_scope", scope._handle, w.ctypes.data if w is not None and w.size else None,
                  int(sign), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "delta (sign is +1 or -1)"), scope.names)

    @classmethod
    def between(cls, old, new, *, budget=None):
        """the diff of two scopes of the same schema and rows: for each row whose contribution differs, -1 for its old
        version and +1 for its new one (inserts, deletes, value changes and group moves)"""
        if new.source == "table":
            new.require_fresh("delta")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_delta_between", old._handle, new._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return cls._wrap(_made(h, error, refusal, "delta between scopes (same table, columns and grouping)"), old.names)

    @staticmethod
    def need(*, columns=None, valid=None, scope=None, old=None, new=None):
        """the needs a delta constructor checks: from_arrays (columns: {name: values}, valid), from_scope (scope),
        between (old, new)"""
        out = _Need()
        if scope is not None:
            _check(_call("tr_delta_need_scope", scope._handle, C.byref(out)), what="delta needs")
        elif old is not None:
            _check(_call("tr_delta_need_between", old._handle, new._handle, C.byref(out)), what="delta needs")
        else:
            names = list(columns or {})
            n = len(np.asarray(columns[names[0]])) if names else 0
            specs = (mbk._Array * max(len(names), 1))()
            dummy = np.zeros(1, dtype=np.uint8)
            for i, name in enumerate(names):
                specs[i] = mbk._Array(None, dummy.ctypes.data if (valid or {}).get(name) is not None else None, 0, 0)
            _check(_call("tr_delta_need_arrays", n, specs, len(names), C.byref(out)), what="delta needs")
        return _needs(out)

    def _open(self):
        if not self._handle:
            raise ValueError("the delta is closed")

    def _info(self):
        self._open()
        out = _BatchInfo()
        _check(_call("tr_delta_get_info", self._handle, C.byref(out)), what="delta")
        return out

    @property
    def info(self):
        i = self._info()
        return dict(rows=int(i.rows), groups=int(i.ngroups), touched=int(i.touched), inserted=int(i.inserted),
                    deleted=int(i.deleted), weight_abs=int(i.wabs), weight_net=int(i.wnet), kind=KINDS[i.kind],
                    identity=_digest(i.identity), revision=int(i.revision),
                    new_scope=_digest(i.new_digest) if i.kind == 3 else None)

    @property
    def identity(self):
        return _digest(self._info().identity)

    @property
    def rows(self):
        return int(self._info().rows)

    def groups(self):
        """the group id of each delta row (uint32)"""
        n = self.rows
        out = np.zeros(max(n, 1), dtype=np.uint32)
        _check(_call("tr_delta_get_groups", self._handle, out.ctypes.data, len(out)), what="delta groups")
        return out[:n]

    def weights(self):
        r = _Rows()
        _check(_call("tr_delta_get_rows", self._handle, C.byref(r)), what="delta rows")
        n = int(r.rows)
        return np.ctypeslib.as_array((C.c_int64 * n).from_address(r.weights)).copy() if n else np.zeros(0, np.int64)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_delta_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Delta(closed)"
        i = self.info
        return f"Delta({i['kind']}, rows={i['rows']}, +{i['inserted']} -{i['deleted']}, touched={i['touched']})"


# ---------------------------------------------------------------------- maintained states
class State:
    """An immutable maintained state (native): the FND-004 cells of a weighted bag of rows, its promise
    ("nonnegative" or "signed"), its identity, base and step, and its source scope when it is exactly that scope's."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use State.from_bank(...) or State.apply(...)")

    @classmethod
    def _wrap(cls, handle, names, terms, backend="cpu", device="native CPU"):
        self = object.__new__(cls)
        self._handle, self.names, self.terms, self.backend, self.device = handle, tuple(names), terms, backend, device
        return self

    @classmethod
    def from_bank(cls, bank, *, promise="nonnegative", budget=None):
        """the state of a FND-004 bank of a scope (bank.scope; a table scope must be fresh)"""
        if promise not in PROMISES:
            raise ValueError(f"promise is one of {', '.join(PROMISES)}")
        if bank.scope is None:
            raise ValueError("a maintained state starts from the bank of a scope")
        if bank.scope.source == "table":
            bank.scope.require_fresh("state")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_delta_state_from_bank", bank._handle, bank.scope._handle, PROMISES[promise],
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "state (the bank of its scope)"), bank.names, bank.terms)

    def _open(self):
        if not self._handle:
            raise ValueError("the state is closed")

    def _info(self):
        self._open()
        out = _Info()
        _check(_call("tr_delta_state_get_info", self._handle, C.byref(out)), what="state")
        return out

    @property
    def info(self):
        i = self._info()
        return dict(groups=int(i.ngroups), present=int(i.present), rows=int(i.rows), bytes=int(i.bytes),
                    step=int(i.step), identity=_digest(i.identity), base=_digest(i.base),
                    source=_digest(i.source) if i.bound else None, bound=bool(i.bound), revision=int(i.revision),
                    promise={v: k for k, v in PROMISES.items()}[i.promise], terms=int(i.nterms),
                    mode={v: k for k, v in mbk.MODES.items()}[i.mode], backend=self.backend, device=self.device)

    @property
    def identity(self):
        return _digest(self._info().identity)

    @property
    def step(self):
        return int(self._info().step)

    @property
    def bound(self):
        return bool(self._info().bound)

    @property
    def promise(self):
        return {v: k for k, v in PROMISES.items()}[self._info().promise]

    def _states(self):
        st = mbk._States()
        _check(_call("tr_delta_state_get_states", self._handle, C.byref(st)), what="state cells")
        return st

    def state(self, g):
        """{"present", "rows", "count": [...], "sum": [...]} of group g as exact (signed) ints"""
        self._open()
        st = self._states()
        n, m, g = int(st.ngroups), int(st.nterms), int(g)
        if not 0 <= g < n:
            raise ValueError("group id out of range")
        rows = int(C.c_int64.from_address(st.rows + 8 * g).value)
        count = np.ctypeslib.as_array((C.c_int64 * m).from_address(st.count + 8 * g * m)).tolist() if m else []
        words = np.ctypeslib.as_array((C.c_int64 * (2 * m)).from_address(st.sum + 16 * g * m)) if m else np.zeros(0)
        sums = mbk._i128(words[0::2], words[1::2]) if m else []
        return dict(present=rows != 0, rows=rows, count=[int(c) for c in count], sum=sums)

    def states(self):
        return [self.state(g) for g in range(int(self._info().ngroups))]

    def need(self, delta, backend="cpu"):
        """the needs an apply checks ({resource: need}; cuda: the device copies)"""
        out = _Need()
        _check(_call("tr_delta_state_need", self._handle, delta._handle, 0 if backend == "cpu" else 1, C.byref(out)),
               what="apply needs")
        return _needs(out)

    def apply(self, delta, *, backend="cpu", budget=None):
        """the new state of this state updated by delta (a new identity, step + 1); refused before anything is
        exposed: OverflowError (the proof fails), NegativeMultiplicityError (a nonnegative measure contradicted),
        DomainError (an incompatible delta), AlgebraBudgetError"""
        self._open(); delta._open()
        error, refusal, v = C.c_int32(), _Refusal(), _Violation()
        if backend == "cpu":
            h = _call("tr_delta_state_apply", self._handle, delta._handle, C.byref(_budget(budget)), C.byref(refusal),
                      C.byref(v), C.byref(error))
            return State._wrap(_made(h, error, refusal, "update", v), self.names, self.terms)
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        gpu = _gpu()
        raw = _Need()
        dev = self.need(delta, "cuda")
        for i, r in enumerate(RESOURCES):
            raw.need[i] = dev.get(r, 0)
        _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw), C.byref(refusal)), refusal,
               "update on the device")
        info = self._info()
        rows = _Rows()
        _check(_call("tr_delta_get_rows", delta._handle, C.byref(rows)), what="delta rows")
        T, m = int(rows.touched), int(info.nterms)
        cin, keep_in = mbk._alloc_states(max(T, 1), m)
        cout, keep_out = mbk._alloc_states(max(T, 1), m)
        cin.ngroups = cout.ngroups = T
        _check(_call("tr_delta_state_gather", self._handle, delta._handle, C.byref(cin), C.byref(_budget(budget)),
                     C.byref(refusal)), refusal, "update on the device")
        view = mbk._View()
        _check(_call("tr_delta_get_view", delta._handle, C.byref(view)), what="delta view")
        exps = np.zeros(max(m * int(info.ncols), 1), dtype=np.uint8)
        _check(_call("tr_delta_state_get_exponents", self._handle, exps.ctypes.data, exps.size), what="basis")
        key = C.c_uint64(_NONE)
        gpu.run("tr_mgpu_delta_apply", C.byref(view), C.byref(rows), exps.ctypes.data, m, int(info.mode),
                int(info.promise), C.byref(cin), C.byref(_budget(budget)), C.byref(cout), C.byref(key),
                C.byref(refusal), refusal=refusal, what="update on the device")
        h = _call("tr_delta_state_from_cells", self._handle, delta._handle, C.byref(cout), key.value,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(v), C.byref(error))
        d = gpu.device()
        return State._wrap(_made(h, error, refusal, "update on the device", v), self.names, self.terms,
                           "cuda-emulated" if d["emulated"] else "cuda", d["name"])

    def bank(self, scope=None, *, budget=None):
        """a FND-004 Bank of the cells (every count >= 0, else DomainError): scoped to `scope` when this state is the
        state of exactly that scope (else DomainError), unscoped without one"""
        self._open()
        if scope is not None and scope.source == "table":
            scope.require_fresh("state bank")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_delta_state_to_bank", self._handle, scope._handle if scope is not None else None,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return mbk.Bank._wrap(_made(h, error, refusal, "state bank"), scope, self.names, "cpu", "native CPU")

    def bound_to(self, scope):
        return bool(_check(_call("tr_delta_state_bound_to", self._handle, scope._handle), what="state binding"))

    def publish(self, cache, scope):
        """puts the bank of exactly `scope` into a FND-004 cache with refresh (dropping the table's other versions)"""
        cache.put(self.bank(scope), refresh=True)

    def equal_bank(self, bank):
        """the same cells as a FND-004 bank, bit for bit"""
        return bool(_check(_call("tr_delta_state_equal_bank", self._handle, bank._handle), what="state comparison"))

    def equal(self, other):
        return bool(_check(_call("tr_delta_state_equal", self._handle, other._handle), what="state comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_delta_state_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "State(closed)"
        i = self.info
        return f"State(groups={i['groups']}, step={i['step']}, {i['promise']}, bound={i['bound']}, {self.backend})"
