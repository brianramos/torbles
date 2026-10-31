# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-004: data scopes, grouped exact moment banks, their rollup and a bounded cache of banks, natively on the CPU
or with the CUDA kernels (torbles/math_scope.c, math_bank.c, math_bank_gpu.cu).

    from torbles import math_bank as mbk
    scope = mbk.Scope.from_table(db["f"], ["x", "d"], group_by=["k", ("c", db["dim"], "region")], rows=mask)
    bank = mbk.build(scope, ["1", "x", "d", "x^2", "x*d"], mode="sql", backend="cpu")    # or backend="cuda"
    bank.state(g)          # {"present", "rows", "count": [...], "sum": [...]}  exact ints, per term
    bank.table()           # the present groups with their decoded keys
    coarse = bank.rollup(mbk.Scope.from_table(db["f"], ["x", "d"], group_by=["k"], rows=mask))
    cache = mbk.Cache(capacity=16, max_bytes=8 << 20)
    cache.bank(scope, basis, mode="sql")     # a hit (an independent copy) or a build that is then cached

A scope is the snapshot the engine selected: rows (a bitmap; default the table's used rows), value columns as exact
logical integers (code + codec offset; decimal value = logical / 10^scale) with validity, and dense groups (the
mixed-radix number of the key codes, first key most significant; a nullable key has a NULL group of its own; a
dimension attribute is reached through a foreign key, unmatched references in its NULL group). A bank holds per
group the bag multiplicity (rows) and per term the count of rows where the term is valid and the exact sum of its
logical values: mode "sql" skips a row for a term when one of its columns is NULL, mode "joint" skips the row for every
term when any column of the basis is NULL. A group is present iff it has rows; an all-NULL group is present with count
0 and sum 0; an absent group has no rows. Overflow is refused before anything is accumulated (OverflowError), budgets
before anything is allocated (AlgebraBudgetError), a stale scope (the table mutated: t.flip, a new revision; or its
selection changed) with StaleStateError. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
import re
import threading
import uuid

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, StaleStateError, UnsupportedError, check as _check

__all__ = ["Scope", "Bank", "Builder", "Cache", "build", "need", "monomials", "MODES", "MAX_COLUMNS", "MAX_TERMS",
           "MAX_DEGREE"]

MAX_COLUMNS, MAX_KEYS, MAX_TERMS, MAX_DEGREE = 32, 12, 1024, 4
MODES = {"sql": 0, "joint": 1}
_NEED_CPU, _NEED_DEVICE, _NEED_ADOPT = 0, 1, 2
_U64 = (1 << 64) - 1
P = C.POINTER


class _Column(C.Structure):
    _fields_ = [("col", C.c_int32), ("valid", C.c_int32), ("hi", C.c_int32), ("shift", C.c_int32),
                ("offset", C.c_int64), ("scale", C.c_int32), ("reserved", C.c_int32)]


class _Key(C.Structure):
    _fields_ = [("col", C.c_int32), ("valid", C.c_int32), ("nullable", C.c_int32), ("null_code", C.c_uint32),
                ("radix", C.c_uint64), ("lookup", C.c_void_p), ("lookup_valid", C.c_void_p), ("lookup_n", C.c_uint64)]


class _Array(C.Structure):
    _fields_ = [("values", C.c_void_p), ("valid", C.c_void_p), ("scale", C.c_int32), ("reserved", C.c_int32)]


class _ViewColumn(C.Structure):
    _fields_ = [("codes", C.c_void_p), ("values", C.c_void_p), ("valid", C.c_void_p), ("offset", C.c_int64),
                ("min", C.c_int64), ("max", C.c_int64), ("nvalid", C.c_uint64), ("scale", C.c_int32),
                ("reserved", C.c_int32)]


class _View(C.Structure):
    _fields_ = [("rows", C.c_uint64), ("selected", C.c_uint64), ("ngroups", C.c_uint64), ("selection", C.c_void_p),
                ("groups", C.c_void_p), ("ncols", C.c_int32), ("reserved", C.c_int32),
                ("col", _ViewColumn * MAX_COLUMNS)]


class _ScopeInfo(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "ngroups", "bytes", "table", "revision")] + [
        ("digest", C.c_uint64 * 2), ("version", C.c_uint64 * 2), ("identity", C.c_uint8 * 16),
        ("source", C.c_int32), ("ncols", C.c_int32), ("nkeys", C.c_int32), ("reserved", C.c_int32)]


class _ColumnInfo(C.Structure):
    _fields_ = [("offset", C.c_int64), ("min", C.c_int64), ("max", C.c_int64), ("nvalid", C.c_uint64)] + [
        (n, C.c_int32) for n in ("scale", "storage", "col", "valid", "hi", "shift")]


class _States(C.Structure):
    _fields_ = [("ngroups", C.c_uint64), ("nterms", C.c_int32), ("reserved", C.c_int32), ("rows", C.c_void_p),
                ("count", C.c_void_p), ("sum", C.c_void_p)]


class _BankInfo(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "present", "rows", "bytes")] + [("digest", C.c_uint64 * 2)] + [
        (n, C.c_int32) for n in ("ncols", "nterms", "mode", "degree")] + [("joint", C.c_uint32), ("scoped", C.c_int32)]


class _CacheStats(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("hits", "misses", "puts", "evictions", "refreshes", "invalidations",
                                          "bytes", "max_bytes")] + [("capacity", C.c_int32), ("entries", C.c_int32)]


for _s, _n in ((_Column, 32), (_Key, 48), (_Array, 24), (_ViewColumn, 64), (_View, 2096), (_ScopeInfo, 112),
               (_ColumnInfo, 56), (_States, 40), (_BankInfo, 72), (_CacheStats, 72)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        sig = {
            "tr_scope_from_table": (vp, [vp, vp, vp, P(_Column), i32, P(_Key), i32, P(_Budget), P(_Refusal), e]),
            "tr_scope_from_arrays": (vp, [u64, vp, P(_Array), i32, vp, u64, P(_Budget), P(_Refusal), e]),
            "tr_scope_need_table": (i32, [vp, P(_Column), i32, P(_Key), i32, P(_Need)]),
            "tr_scope_need_arrays": (i32, [u64, P(_Array), i32, vp, u64, P(_Need)]),
            "tr_scope_free": (None, [vp]),
            "tr_scope_get_info": (i32, [vp, P(_ScopeInfo)]),
            "tr_scope_get_view": (i32, [vp, P(_View)]),
            "tr_scope_column_get": (i32, [vp, i32, P(_ColumnInfo)]),
            "tr_scope_fresh": (i32, [vp, vp, vp]),
            "tr_scope_same": (i32, [vp, vp]),
            "tr_scope_same_rows": (i32, [vp, vp]),
            "tr_scope_rollup_map": (i32, [vp, vp, vp, u64]),
            "tr_scope_key_size": (u64, [vp, i32]),
            "tr_bank_init": (vp, [vp, vp, i32, i32, P(_Budget), P(_Refusal), e]),
            "tr_bank_accumulate": (i32, [vp, u64, u64]),
            "tr_bank_merge": (i32, [vp, vp]),
            "tr_bank_finish": (vp, [vp, e]),
            "tr_bank_builder_free": (None, [vp]),
            "tr_bank_build": (vp, [vp, vp, i32, i32, P(_Budget), P(_Refusal), e]),
            "tr_bank_need": (i32, [vp, vp, i32, i32, i32, P(_Need)]),
            "tr_bank_from_states": (vp, [vp, i32, vp, i32, i32, P(_States), P(_Budget), P(_Refusal), e]),
            "tr_bank_rollup": (vp, [vp, vp, u64, vp, P(_Budget), P(_Refusal), e]),
            "tr_bank_rollup_need": (i32, [vp, u64, i32, P(_Need)]),
            "tr_bank_clone": (vp, [vp, e]),
            "tr_bank_equal": (i32, [vp, vp]),
            "tr_bank_get_info": (i32, [vp, P(_BankInfo)]),
            "tr_bank_get_states": (i32, [vp, P(_States)]),
            "tr_bank_get_exponents": (i32, [vp, vp, u64]),
            "tr_bank_free": (None, [vp]),
            "tr_bank_cache_create": (vp, [i32, u64, e]),
            "tr_bank_cache_get": (vp, [vp, vp, vp, i32, i32, e]),
            "tr_bank_cache_put": (i32, [vp, vp, vp, i32, P(_Refusal)]),
            "tr_bank_cache_refresh": (vp, [vp, vp, vp, i32, i32, P(_Budget), P(_Refusal), e]),
            "tr_bank_cache_get_stats": (i32, [vp, P(_CacheStats)]),
            "tr_bank_cache_clear": (i32, [vp]),
            "tr_bank_cache_free": (None, [vp]),
            "tr_math_budget_check": (i32, [P(_Budget), P(_Need), P(_Refusal)]),
            "tr_math_table_revision": (u64, [vp]),
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
    math_gpu.bind("tr_mgpu_bank_build", C.c_int, [P(_View), C.c_void_p, C.c_int32, C.c_int32, P(_Budget), P(_States),
                                                 P(_Refusal)])
    math_gpu.bind("tr_mgpu_bank_rollup", C.c_int, [P(_States), C.c_void_p, C.c_uint64, P(_Budget), P(_States),
                                                  P(_Refusal)])
    return math_gpu


def _made(handle, error, refusal, what):
    """the handle of a native constructor, or its refusal raised"""
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _budget(budget):
    return (budget or Budget()).raw


def _bitmap(rows, n):
    """(packed uint64 selection or None, the array whose contents are the selection, for freshness)"""
    if rows is None:
        return None, None
    a = np.asarray(rows)
    nw = (n + 63) // 64
    if a.dtype == np.bool_:
        if a.shape != (n,):
            raise ValueError(f"the row mask has {a.size} entries for {n} rows")
        padded = np.zeros(max(nw, 1) * 64, dtype=np.uint8)
        padded[:n] = a
        bits = np.packbits(padded, bitorder="little").view(np.uint64)
        return bits, bits
    if a.dtype == np.uint64 and a.ndim == 1:
        if len(a) < nw:
            raise ValueError(f"the row bitmap has {len(a)} words for {n} rows ({nw} needed)")
        bits = np.ascontiguousarray(a)
        return bits, a if bits is a else bits
    raise TypeError("rows: a boolean mask of the rows or a packed uint64 bitmap (bit r of word r // 64)")


def _ptr(a):
    return None if a is None else a.ctypes.data


def _decode(table, name):
    """the stored codes of a column, decoded natively now (at_decode_column)"""
    out = np.empty(max(table.n, 1), dtype=np.uint32)
    algq.lib.at_decode_column(table.at, table.idx(name), out.ctypes.data)
    return out[:table.n]


def _revision(table):
    return int(_call("tr_math_table_revision", table.at))


def _i128(lo, hi):
    """exact Python ints from (lo, hi) int64 words: value = hi * 2^64 + (lo mod 2^64)"""
    return [(int(h) << 64) | (int(l) & _U64) for l, h in zip(lo.tolist(), hi.tolist())]


class Scope:
    """An immutable snapshot of selected rows, exact value columns with validity, and dense group ids (native)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Scope.from_table(...) or Scope.from_arrays(...)")

    @classmethod
    def _new(cls):
        self = object.__new__(cls)
        self._handle = None
        return self

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_table(cls, table, columns, *, rows=None, group_by=(), budget=None):
        """columns: value columns (integer, decimal or boolean codecs; self-extending columns included).
        rows: None (the table's used rows, Table.used_bm), a boolean mask or a packed uint64 bitmap.
        group_by: keys, first most significant: a column name (its codes; NULL its own group when nullable), or
        (foreign key column, dimension Table, attribute) for a dimension attribute (unmatched references and NULL
        attributes form the key's NULL group)."""
        columns = list(columns)
        if len(columns) > MAX_COLUMNS or len(set(columns)) != len(columns):
            raise ValueError(f"choose at most {MAX_COLUMNS} distinct value columns")
        group_by = list(group_by)
        if len(group_by) > MAX_KEYS:
            raise ValueError(f"at most {MAX_KEYS} group keys")
        nullable, wide = getattr(table, "nullable", {}), getattr(table, "wide", {})
        specs = (_Column * max(len(columns), 1))()
        for i, name in enumerate(columns):
            codec = table.codecs.get(name) if hasattr(table, "codecs") else None
            if name not in table.cols:
                raise ValueError(f"{table.name} has no column {name!r}")
            if codec is not None and codec.affine is None:
                raise UnsupportedError(f"{name!r}: a {codec.kind} column has no exact logical value (affine codec)")
            offset, scale = codec.affine if codec is not None else (0, 0)
            if not -(1 << 63) <= int(offset) < (1 << 63):
                raise OverflowError(f"{name!r}: the codec offset does not fit int64")
            hi, shift = -1, 0
            if name in wide:
                shift, hi_name = wide[name][0], wide[name][1]
                hi = table.idx(hi_name)
            specs[i] = _Column(table.idx(name), table.idx(nullable[name]) if name in nullable else -1, hi, int(shift),
                               int(offset), int(scale), 0)
        keys = (_Key * max(len(group_by), 1))()
        described, keep, dims = [], [], []
        for i, key in enumerate(group_by):
            if isinstance(key, str):
                if key not in table.cols:
                    raise ValueError(f"{table.name} has no column {key!r}")
                if key in wide:
                    raise UnsupportedError(f"{key!r}: a self-extending column cannot be a group key")
                radix = int(table.maxcode(key)) + 1
                null = key in nullable
                keys[i] = _Key(table.idx(key), table.idx(nullable[key]) if null else -1, int(null),
                               radix if null else 0, radix + int(null), None, None, 0)
                codec = table.codecs.get(key) if hasattr(table, "codecs") else None
                described.append(dict(name=key, radix=radix + int(null), null_code=radix if null else None,
                                      codec=codec))
            else:
                fk, dim, attr = key
                if fk not in table.cols or attr not in dim.cols:
                    raise ValueError(f"no column {fk!r} in {table.name} or {attr!r} in {dim.name}")
                if getattr(table, "surr", {}).get(fk) is not None:
                    raise UnsupportedError(f"{fk!r}: references through a derived surrogate are not supported")
                if attr in getattr(dim, "wide", {}):
                    raise UnsupportedError(f"{attr!r}: a self-extending attribute cannot be a group key")
                lookup = _decode(dim, attr)               # decoded now, natively (Table.decoded() caches)
                flags = None
                dn = getattr(dim, "nullable", {})
                used = np.asarray(getattr(dim, "used", np.ones(dim.n, bool)), dtype=bool)
                if attr in dn or not used.all():          # NULL attributes and ids not in use: the NULL group
                    ok = used.copy()
                    if attr in dn:
                        ok &= _decode(dim, dn[attr]) != 0
                    flags = np.ascontiguousarray(ok, dtype=np.uint32)
                dims.append((dim, _revision(dim), np.array(dim.used_bm, copy=True)))
                radix = int(dim.maxcode(attr)) + 1
                keys[i] = _Key(table.idx(fk), table.idx(nullable[fk]) if fk in nullable else -1, 1, radix, radix + 1,
                               lookup.ctypes.data, _ptr(flags), len(lookup))
                keep += [lookup, flags]
                described.append(dict(name=f"{dim.name}.{attr}", radix=radix + 1, null_code=radix,
                                      codec=dim.codecs.get(attr) if hasattr(dim, "codecs") else None))
        bits, source = _bitmap(rows, table.n)
        if bits is None:
            bits = table.used_bm
        identity = table.__dict__.get("_math_scope_identity")
        if identity is None:            # a lifetime token: a later table at the same address is another source
            identity = table.__dict__.setdefault("_math_scope_identity", uuid.uuid4().bytes)
        token = (C.c_uint8 * 16)(*identity)
        error, refusal = C.c_int32(), _Refusal()
        handle = _call("tr_scope_from_table", table.at, token, bits.ctypes.data, specs, len(columns), keys,
                       len(group_by), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        self = cls._new()
        self._handle = _made(handle, error, refusal, "scope")
        self.source, self.table, self.names, self.keys = "table", table, tuple(columns), tuple(described)
        self._rows = source              # None: the table's current used rows
        self._dims = tuple(dims)         # the dimensions whose attributes the lookups copied, with their version
        return self

    @classmethod
    def from_arrays(cls, columns, *, valid=None, scales=None, groups=None, ngroups=None, rows=None, budget=None):
        """columns: {name: int64 logical values}; valid: {name: boolean mask (True = present)}; scales: {name:
        decimal scale}; groups: an id per row (< ngroups) or None (one group); rows: a boolean mask or a uint64 bitmap
        (None = every row)."""
        names = list(columns)
        if len(names) > MAX_COLUMNS:
            raise ValueError(f"at most {MAX_COLUMNS} value columns")
        valid, scales = dict(valid or {}), dict(scales or {})
        arrays, flags = [], []
        n = None
        for name in names:
            v = np.asarray(columns[name])
            if v.dtype.kind not in "iubO":
                raise TypeError(f"{name!r}: logical values are exact integers (got {v.dtype}); scale decimals first")
            if v.dtype.kind in "uO" and v.size and any(not -(1 << 63) <= int(x) < (1 << 63) for x in
                                                       ([v.max()] if v.dtype.kind == "u" else v.tolist())):
                raise OverflowError(f"{name!r}: a value does not fit int64")
            a = np.ascontiguousarray(v, dtype=np.int64)
            if n is None:
                n = len(a)
            if a.shape != (n,):
                raise ValueError("every column has one value per row")
            arrays.append(a)
            f = valid.get(name)
            flags.append(None if f is None else np.ascontiguousarray(np.asarray(f, dtype=bool), dtype=np.uint8))
            if flags[-1] is not None and flags[-1].shape != (n,):
                raise ValueError(f"{name!r}: the validity mask has one flag per row")
        if groups is not None:
            g = np.asarray(groups)
            if n is None:
                n = len(g)
            if g.shape != (n,) or (g.size and (g.min() < 0 or g.max() > 0xFFFFFFFF)):
                raise ValueError("groups: one id per row, 0 <= id < 2^32")
            gids = np.ascontiguousarray(g, dtype=np.uint32)
            if ngroups is None:
                raise ValueError("groups need ngroups")
        else:
            gids = None
            ngroups = 1 if ngroups is None else ngroups
        if n is None:
            raise ValueError("give at least one column or the group ids")
        specs = (_Array * max(len(names), 1))()
        for i, name in enumerate(names):
            specs[i] = _Array(arrays[i].ctypes.data if n else None, _ptr(flags[i]), int(scales.get(name, 0)), 0)
        bits, source = _bitmap(rows, n)
        error, refusal = C.c_int32(), _Refusal()
        handle = _call("tr_scope_from_arrays", n, _ptr(bits), specs, len(names), _ptr(gids), int(ngroups),
                       C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        self = cls._new()
        self._handle = _made(handle, error, refusal, "scope")
        self.source, self.table, self.names, self.keys = "arrays", None, tuple(names), ()
        self._rows, self._dims = source, ()
        return self

    # ------------------------------------------------------------------ reading
    def _info(self):
        self._open()
        out = _ScopeInfo()
        _check(_call("tr_scope_get_info", self._handle, C.byref(out)), what="scope")
        return out

    def _open(self):
        if not self._handle:
            raise ValueError("the scope is closed")

    def view(self):
        """the native tr_scope_view (pointers into this scope; valid while it lives)"""
        self._open()
        v = _View()
        _check(_call("tr_scope_get_view", self._handle, C.byref(v)), what="scope view")
        return v

    @property
    def rows(self):
        return int(self._info().rows)

    @property
    def selected(self):
        return int(self._info().selected)

    @property
    def groups(self):
        return int(self._info().ngroups)

    @property
    def bytes(self):
        return int(self._info().bytes)

    @property
    def revision(self):
        return int(self._info().revision)

    def column(self, name):
        """what the build decoded: min, max (None without present values), valid (present selected values), scale"""
        self._open()
        out = _ColumnInfo()
        _check(_call("tr_scope_column_get", self._handle, self.names.index(name), C.byref(out)), what="scope column")
        empty = out.nvalid == 0
        return dict(min=None if empty else int(out.min), max=None if empty else int(out.max), valid=int(out.nvalid),
                    scale=int(out.scale), offset=int(out.offset), storage=("codes", "values")[out.storage - 1])

    def fresh(self):
        """True while the table (revision, rows, sidecars), the selection and the dimensions of the lookups
        (revision, used rows) are unchanged since the build"""
        self._open()
        if self.source == "arrays":
            rc = _call("tr_scope_fresh", self._handle, None, _ptr(self._rows))
        else:
            rows = self.table.used_bm if self._rows is None else self._rows
            rc = _call("tr_scope_fresh", self._handle, self.table.at, rows.ctypes.data)
        if rc == -13:
            return False
        _check(rc, what="scope freshness")
        return all(_revision(d) == r and np.array_equal(d.used_bm, u) for d, r, u in getattr(self, "_dims", ()))

    def require_fresh(self, what="scope"):
        if not self.fresh():
            raise StaleStateError(f"{what}: the table or the selection changed since the scope was built")

    def same(self, other, rows_only=False):
        """exact descriptor equality (rows_only: everything but the grouping)"""
        self._open(); other._open()
        return bool(_check(_call("tr_scope_same_rows" if rows_only else "tr_scope_same", self._handle, other._handle),
                           what="scope comparison"))

    def rollup_map(self, coarse):
        """map[g]: the coarse group of fine group g (this scope is the fine one; same rows and columns)"""
        self._open(); coarse._open()
        out = np.zeros(max(self.groups, 1), dtype=np.uint32)
        _check(_call("tr_scope_rollup_map", self._handle, coarse._handle, out.ctypes.data, len(out)),
               what="rollup map")
        return out[:self.groups]

    def key_values(self, g):
        """the decoded key values of group g (None for a NULL key; codes when the column has no codec); for an
        array scope, (g,) (or () with one group)"""
        g = int(g)
        if not 0 <= g < self.groups:
            raise ValueError("group id out of range")
        if self.source == "arrays":
            return (g,) if self.groups > 1 else ()
        digits = []
        for spec in reversed(self.keys):
            digits.append(g % spec["radix"])
            g //= spec["radix"]
        out = []
        for spec, d in zip(self.keys, reversed(digits)):
            if spec["null_code"] is not None and d == spec["null_code"]:
                out.append(None)
            elif spec["codec"] is not None:
                out.append(spec["codec"].decode(np.array([d], dtype=np.int64))[0])
            else:
                out.append(int(d))
        return tuple(out)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_scope_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Scope(closed)"
        return (f"Scope({self.source}, rows={self.rows}, selected={self.selected}, columns={list(self.names)}, "
                f"groups={self.groups})")


# ---------------------------------------------------------------------- basis
_TERM = re.compile(r"^\s*([A-Za-z_][\w#.]*)\s*(?:(?:\^|\*\*)\s*(\d+))?\s*$")


def monomials(names, degree):
    """every monomial of total degree <= degree over names, graded (the constant first), as exponent tuples"""
    k, out = len(list(names)), []

    def rec(prefix, left, i):                # exponents over columns i.. of total exactly `left`
        if i == k - 1:
            out.append(tuple(prefix + [left]))
            return
        for e in range(left, -1, -1):
            rec(prefix + [e], left - e, i + 1)
    for d in range(int(degree) + 1):
        if k:
            rec([], d, 0)
        elif d == 0:
            out.append(())
    return out


def _basis(names, basis):
    """an exponent matrix (nterms x ncols, uint8) from terms: exponent tuples, {name: exponent} or 'x*y^2' / '1'"""
    names = list(names)
    rows = []
    for t in basis:
        e = [0] * len(names)
        if isinstance(t, str):
            if t.strip() != "1":
                for factor in re.split(r"(?<!\*)\*(?!\*)", t):     # x*y^2, x**2*y
                    m = _TERM.match(factor)
                    if not m or m.group(1) not in names:
                        raise ValueError(f"term {t!r}: {factor.strip()!r} is not a scope column (power: x^2 or x**2)")
                    e[names.index(m.group(1))] += int(m.group(2) or 1)
        elif isinstance(t, dict):
            for name, k in t.items():
                if name not in names:
                    raise ValueError(f"term {t!r}: {name!r} is not a scope column")
                e[names.index(name)] += int(k)
        else:
            e = [int(k) for k in t]
            if len(e) != len(names):
                raise ValueError(f"term {t!r}: one exponent per scope column ({len(names)})")
        if any(k < 0 or k > 255 for k in e):
            raise ValueError(f"term {t!r}: exponents are 0..255 here (the bank admits total degree <= {MAX_DEGREE})")
        rows.append(e)
    if len(rows) > MAX_TERMS:
        raise UnsupportedError(f"{len(rows)} terms: a bank has at most {MAX_TERMS}")
    return np.ascontiguousarray(np.array(rows, dtype=np.uint8).reshape(len(rows), len(names)))


def _exps_ptr(exps):
    return exps.ctypes.data if exps.size else None


def _mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode is one of {', '.join(MODES)}")
    return MODES[mode]


def _need(fn, *args):
    out = _Need()
    _check(_call(fn, *args, C.byref(out)), what="bank needs")
    return out


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def need(scope, basis, mode="sql", backend="cpu"):
    """the needs a build checks before allocating: {resource: need} (backend "cuda": its device copies and the
    adopted bank, resource by resource the larger)"""
    exps = _basis(scope.names, basis)
    m = _mode(mode)
    if backend == "cpu":
        return _needs_dict(_need("tr_bank_need", scope._handle, _exps_ptr(exps), len(exps), m, _NEED_CPU))
    if backend == "cuda":
        return _needs_dict(_cuda_need(scope, exps, m))
    raise ValueError("backend is 'cpu' or 'cuda'")


def _cuda_need(scope, exps, m):
    dev = _need("tr_bank_need", scope._handle, _exps_ptr(exps), len(exps), m, _NEED_DEVICE)
    adopt = _need("tr_bank_need", scope._handle, _exps_ptr(exps), len(exps), m, _NEED_ADOPT)
    both = _Need()
    for i in range(len(RESOURCES)):
        both.need[i] = max(dev.need[i], adopt.need[i])
    return both


def _precheck(budget, raw_need, what):
    refusal = _Refusal()
    _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw_need), C.byref(refusal)), refusal, what)


# ---------------------------------------------------------------------- banks
class Bank:
    """An immutable grouped exact moment bank (native). Read it with state(g), states(), table(); never mutated."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_bank.build(...), Builder(...).finish(), Bank.rollup(...) or Cache")

    @classmethod
    def _wrap(cls, handle, scope, names, backend, device, cache_hit=False):
        self = object.__new__(cls)
        self._handle, self.scope, self.names = handle, scope, tuple(names)
        self.backend, self.device, self.cache_hit = backend, device, cache_hit
        info = self._raw_info()
        exps = np.zeros(max(info.nterms * info.ncols, 1), dtype=np.uint8)
        _check(_call("tr_bank_get_exponents", handle, exps.ctypes.data, exps.size), what="bank basis")
        self.terms = tuple(tuple(int(k) for k in exps[j * info.ncols:(j + 1) * info.ncols])
                           for j in range(info.nterms))
        self.mode = {v: k for k, v in MODES.items()}[info.mode]
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the bank is closed")

    def _raw_info(self):
        self._open()
        out = _BankInfo()
        _check(_call("tr_bank_get_info", self._handle, C.byref(out)), what="bank")
        return out

    @property
    def info(self):
        i = self._raw_info()
        return dict(groups=int(i.ngroups), present=int(i.present), rows=int(i.rows), bytes=int(i.bytes),
                    columns=int(i.ncols), terms=int(i.nterms), mode=self.mode, degree=int(i.degree),
                    scoped=bool(i.scoped), backend=self.backend, device=self.device, cache_hit=self.cache_hit)

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

    def _states(self):
        self._open()
        st = _States()
        _check(_call("tr_bank_get_states", self._handle, C.byref(st)), what="bank states")
        return st

    def arrays(self):
        """copies of the native states: rows uint64[G], count uint64[G, m], sum_lo / sum_hi int64[G, m]"""
        st = self._states()
        g, m = int(st.ngroups), int(st.nterms)
        rows = np.ctypeslib.as_array((C.c_uint64 * g).from_address(st.rows)).copy()
        if m:
            count = np.ctypeslib.as_array((C.c_uint64 * (g * m)).from_address(st.count)).reshape(g, m).copy()
            words = np.ctypeslib.as_array((C.c_int64 * (2 * g * m)).from_address(st.sum)).reshape(g, m, 2).copy()
        else:
            count, words = np.zeros((g, 0), np.uint64), np.zeros((g, 0, 2), np.int64)
        return rows, count, words[:, :, 0], words[:, :, 1]

    def state(self, g):
        """{"present", "rows", "count": [per term], "sum": [per term]} of group g, exact ints"""
        g = int(g)
        st = self._states()
        n, m = int(st.ngroups), int(st.nterms)
        if not 0 <= g < n:
            raise ValueError("group id out of range")
        rows = int(C.c_uint64.from_address(st.rows + 8 * g).value)
        if m:
            count = np.ctypeslib.as_array((C.c_uint64 * m).from_address(st.count + 8 * g * m)).tolist()
            words = np.ctypeslib.as_array((C.c_int64 * (2 * m)).from_address(st.sum + 16 * g * m))
            sums = _i128(words[0::2], words[1::2])
        else:
            count, sums = [], []
        return dict(present=rows > 0, rows=rows, count=[int(c) for c in count], sum=sums)

    def states(self):
        """state(g) for every group g (absent groups included: present False, zeros)"""
        rows, count, lo, hi = self.arrays()
        out = []
        for g in range(len(rows)):
            out.append(dict(present=bool(rows[g]), rows=int(rows[g]), count=[int(c) for c in count[g]],
                            sum=_i128(lo[g], hi[g])))
        return out

    def table(self):
        """the present groups: [{"group", "key" (decoded, with a scope), "rows", "count", "sum"}]"""
        out = []
        for g, s in enumerate(self.states()):
            if s["present"]:
                key = self.scope.key_values(g) if self.scope is not None else (g,)
                out.append(dict(group=g, key=key, rows=s["rows"], count=s["count"], sum=s["sum"]))
        return out

    def equal(self, other):
        """same columns, basis, mode, groups and identical states (native comparison)"""
        self._open(); other._open()
        return bool(_check(_call("tr_bank_equal", self._handle, other._handle), what="bank comparison"))

    def __eq__(self, other):
        return isinstance(other, Bank) and self.equal(other)

    __hash__ = None

    def rollup(self, coarse=None, *, map=None, groups=None, backend="cpu", budget=None):
        """the coarse bank along a refinement map: coarse (a Scope of the same rows; its map from this bank's
        scope, and the result is that scope's bank) or map= (uint32 coarse id per fine group) with groups="""
        self._open()
        if coarse is not None:
            if self.scope is None:
                raise ValueError("this bank has no scope: give map= and groups=")
            self.scope.require_fresh("rollup"); coarse.require_fresh("rollup")
            fmap, ngroups, origin = self.scope.rollup_map(coarse), coarse.groups, coarse
        else:
            if map is None or groups is None:
                raise ValueError("give a coarse scope, or map= and groups=")
            fmap = np.ascontiguousarray(map, dtype=np.uint32)
            ngroups, origin = int(groups), None
            if fmap.shape != (self.groups,):
                raise ValueError("map: one coarse id per fine group")
        handle_origin = origin._handle if origin is not None else None
        error, refusal = C.c_int32(), _Refusal()
        if backend == "cpu":
            h = _call("tr_bank_rollup", self._handle, fmap.ctypes.data, ngroups, handle_origin,
                      C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            return Bank._wrap(_made(h, error, refusal, "rollup"), origin, self.names, "cpu", "native CPU")
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        gpu = _gpu()
        dev = _need("tr_bank_rollup_need", self._handle, ngroups, _NEED_DEVICE)
        adopt = _need("tr_bank_rollup_need", self._handle, ngroups, _NEED_CPU)
        both = _Need()
        for i in range(len(RESOURCES)):
            both.need[i] = max(dev.need[i], adopt.need[i])
        _precheck(budget, both, "rollup on the device")
        m = len(self.terms)
        out, keep = _alloc_states(ngroups, m)
        fine = self._states()
        gpu.run("tr_mgpu_bank_rollup", C.byref(fine), fmap.ctypes.data, ngroups, C.byref(_budget(budget)),
                C.byref(out), C.byref(refusal), refusal=refusal, what="rollup on the device")
        info = self._raw_info()
        exps = _basis(self.names, self.terms)
        h = _call("tr_bank_from_states", handle_origin, int(info.ncols), _exps_ptr(exps), m, int(info.mode),
                  C.byref(out), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        d = gpu.device()
        return Bank._wrap(_made(h, error, refusal, "rollup states"), origin, self.names,
                          "cuda-emulated" if d["emulated"] else "cuda", d["name"])

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_bank_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Bank(closed)"
        i = self.info
        return f"Bank(groups={i['groups']}, present={i['present']}, terms={i['terms']}, mode={self.mode}, {self.backend})"


def _alloc_states(ngroups, m):
    rows = np.zeros(ngroups, dtype=np.uint64)
    count = np.zeros(max(ngroups * m, 1), dtype=np.uint64)
    sums = np.zeros(max(2 * ngroups * m, 2), dtype=np.int64)
    st = _States(ngroups, m, 0, rows.ctypes.data, count.ctypes.data, sums.ctypes.data)
    return st, (rows, count, sums)


def build(scope, basis, mode="sql", *, backend="cpu", budget=None):
    """the bank of scope over basis (terms: exponent tuples, {name: exponent} or 'x*y^2' / '1'); backend "cuda" runs
    the CUDA kernels (Bank.backend says "cuda" on a device, "cuda-emulated" in the emulation)"""
    if scope.source == "table":
        scope.require_fresh("bank")
    exps = _basis(scope.names, basis)
    m = _mode(mode)
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = _call("tr_bank_build", scope._handle, _exps_ptr(exps), len(exps), m, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Bank._wrap(_made(h, error, refusal, "bank"), scope, scope.names, "cpu", "native CPU")
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    gpu = _gpu()
    _precheck(budget, _cuda_need(scope, exps, m), "bank on the device")
    view = scope.view()
    out, keep = _alloc_states(scope.groups, len(exps))
    gpu.run("tr_mgpu_bank_build", C.byref(view), _exps_ptr(exps), len(exps), m, C.byref(_budget(budget)),
            C.byref(out), C.byref(refusal), refusal=refusal, what="bank on the device")
    h = _call("tr_bank_from_states", scope._handle, len(scope.names), _exps_ptr(exps), len(exps), m, C.byref(out),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    d = gpu.device()
    return Bank._wrap(_made(h, error, refusal, "bank states"), scope, scope.names,
                      "cuda-emulated" if d["emulated"] else "cuda", d["name"])


class Builder:
    """A chunked accumulation: accumulate(begin, end) over disjoint physical row ranges, merge(other builder of the
    same scope, basis and mode), finish() once every row is covered. Not thread-safe: one owner."""

    def __init__(self, scope, basis, mode="sql", *, budget=None):
        if scope.source == "table":
            scope.require_fresh("bank builder")
        exps = _basis(scope.names, basis)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_bank_init", scope._handle, _exps_ptr(exps), len(exps), _mode(mode), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        self._handle = _made(h, error, refusal, "bank builder")
        self.scope = scope                 # the native builder reads the scope: keep it alive

    def _open(self):
        if not self._handle:
            raise ValueError("the builder is finished or closed")

    def accumulate(self, begin, end):
        self._open()
        _check(_call("tr_bank_accumulate", self._handle, int(begin), int(end)),
               what="accumulate (rows within the scope, each range once)")
        return self

    def merge(self, other):
        self._open(); other._open()
        _check(_call("tr_bank_merge", self._handle, other._handle),
               what="merge (same scope, basis and mode; disjoint rows)")
        return self

    def finish(self):
        self._open()
        error = C.c_int32()
        h = _call("tr_bank_finish", self._handle, C.byref(error))
        _check(error.value, what="finish (every row covered once)")
        bank = Bank._wrap(h, self.scope, self.scope.names, "cpu", "native CPU")
        self.close()
        return bank

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_bank_builder_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- the cache
class Cache:
    """A bounded LRU cache of immutable banks keyed by exact scope descriptors, bases and modes (native). Calls are
    serialized by a lock; returned banks are independent copies."""

    def __init__(self, capacity=16, max_bytes=8 << 20):
        error = C.c_int32()
        h = _call("tr_bank_cache_create", int(capacity), int(max_bytes), C.byref(error))
        self._handle = _made(h, error, None, "bank cache (capacity 1..128; max_bytes covers its fixed allocation)")
        self._lock = threading.Lock()

    def _open(self):
        if not self._handle:
            raise ValueError("the cache is closed")

    def get(self, scope, basis, mode="sql"):
        """an independent copy of the cached bank of exactly (scope, basis, mode), or None"""
        if scope.source == "table":
            scope.require_fresh("cache")
        exps = _basis(scope.names, basis)
        _need("tr_bank_need", scope._handle, _exps_ptr(exps), len(exps), _mode(mode), _NEED_CPU)   # a valid basis
        error = C.c_int32()
        with self._lock:
            self._open()
            h = _call("tr_bank_cache_get", self._handle, scope._handle, _exps_ptr(exps), len(exps), _mode(mode),
                      C.byref(error))
        _check(error.value, what="cache get")
        return Bank._wrap(h, scope, scope.names, "cache", "native CPU", cache_hit=True) if h else None

    def put(self, bank, *, refresh=False):
        """caches a copy of a bank under its scope (refresh: also drop entries of the table's other versions)"""
        if bank.scope is None:
            raise ValueError("an unscoped bank cannot be cached")
        if bank.scope.source == "table":
            bank.scope.require_fresh("cache")
        refusal = _Refusal()
        with self._lock:
            self._open()
            rc = _call("tr_bank_cache_put", self._handle, bank.scope._handle, bank._handle, int(bool(refresh)),
                       C.byref(refusal))
        _check(rc, refusal, "cache put")

    def bank(self, scope, basis, mode="sql", *, backend="cpu", budget=None, refresh=False):
        """the cached bank (a copy) or, on a miss, the bank built on backend and then cached; refresh=True always
        rebuilds and replaces"""
        if refresh:
            return self.refresh(scope, basis, mode, backend=backend, budget=budget)
        hit = self.get(scope, basis, mode)
        if hit is not None:
            return hit
        b = build(scope, basis, mode, backend=backend, budget=budget)
        self.put(b)
        return b

    def refresh(self, scope, basis, mode="sql", *, backend="cpu", budget=None):
        """rebuilds the bank of the scope (never a hit), replaces its entry and drops the table's other versions"""
        if backend != "cpu":
            b = build(scope, basis, mode, backend=backend, budget=budget)
            self.put(b, refresh=True)
            return b
        if scope.source == "table":
            scope.require_fresh("cache refresh")
        exps = _basis(scope.names, basis)
        error, refusal = C.c_int32(), _Refusal()
        with self._lock:
            self._open()
            h = _call("tr_bank_cache_refresh", self._handle, scope._handle, _exps_ptr(exps), len(exps), _mode(mode),
                      C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Bank._wrap(_made(h, error, refusal, "cache refresh"), scope, scope.names, "cpu", "native CPU")

    @property
    def stats(self):
        out = _CacheStats()
        with self._lock:
            self._open()
            _check(_call("tr_bank_cache_get_stats", self._handle, C.byref(out)), what="cache stats")
        return {name: int(getattr(out, name)) for name, _ in _CacheStats._fields_}

    def clear(self):
        with self._lock:
            self._open()
            _check(_call("tr_bank_cache_clear", self._handle), what="cache clear")

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            if getattr(self, "_handle", None):
                _call("tr_bank_cache_free", self._handle)
                self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
