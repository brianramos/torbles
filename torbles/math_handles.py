# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-013: state ownership, concurrency and the native ABI (torbles/math_handles.h/.c, math_handles_gpu.cu).

    from torbles import math_handles as mh
    with mh.HandleTable(capacity=1024) as table:                # caller-owned; no global table
        p = table.polynomial({(2, 0): 3, (1, 1): -5, (0, 0): 7})  # immutable 3*x0^2 - 5*x0*x1 + 7
        p.evaluate([[1, 2], [10**18, 3]])                        # {"values": [exact int or None], "overflow": ...}
        p.evaluate(points, backend="cuda")                       # the CUDA kernel (on a device or emulated)
        with p.retained():                                       # a live native reference
            p.close()                                            # "deferred": freed when the reference ends
        p.evaluate([[1, 2]])                                     # ClosedHandleError, always
        table.stats()                                            # live handles, payload bytes, native allocations
    mh.abi_check()                                               # every ctypes Structure of torbles.math* vs native

A handle is a 64-bit id: slot index (low 32 bits) and generation (high 32 bits). A closed handle raises
ClosedHandleError deterministically (also while its payload is still held by a live operation); a value that was
never issued, or names another kind, raises ValueError. Every native operation retains the handles it uses for its
duration, so closing from another thread never frees a payload under a live operation. Immutable payloads
(polynomials, domains, banks) are read concurrently; a state cache is mutable and every call on it is serialized by
its native mutex. Native calls release the GIL. Opt-in: `import torbles` does not load this module.

The ABI check: libalg.so and the CUDA library report every registered struct (TR_MATH_ABI / TR_MGPU_ABI, one line per
struct in the C file of its family); abi_check compares each with the ctypes Structures of every torbles.math* module
(registered by a class attribute `_native_ = "<C type>"` (and `_abi_version_` when not 1) or, for the modules that
predate the registry, by the table _EXISTING below) and raises AbiError on any mismatch or unregistered Structure.
"""
import contextlib
import ctypes as C
import importlib
import sys
import threading
from numbers import Integral
from pathlib import Path

import numpy as np

from . import algq
from .math_budgets import Budget, _Budget, _Refusal
from .math_domains import CARRIERS, GROUPING, KINDS as DOMAIN_KINDS, VALIDITY, _CacheStats, _DomainSpec, _key_bytes
from .math_errors import check as _math_check, ClosedHandleError

__all__ = ["HandleTable", "Handle", "Polynomial", "Domain", "Bank", "StateCache", "ClosedHandleError", "AbiError",
           "KINDS", "abi_check", "abi_entries", "python_structures"]

CLOSED = -17                       # TR_MATH_CLOSED (math_num.h)
KINDS = {"polynomial": 1, "domain": 2, "bank": 3, "state cache": 4}
_KIND_NAMES = {v: k for k, v in KINDS.items()}
M64 = (1 << 64) - 1
I64 = (-(1 << 63), (1 << 63) - 1)


class AbiError(RuntimeError):
    """A ctypes Structure differs from the struct the native library was compiled with (or is not registered)."""


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _AbiEntry(C.Structure):
    _native_ = "tr_math_abi_entry"
    _fields_ = [("name", C.c_char * 48), ("size", C.c_uint32), ("align", C.c_uint32), ("version", C.c_int32),
                ("reserved", C.c_int32)]


class _I128(C.Structure):
    _native_ = "tr_math_i128"
    _fields_ = [("lo", C.c_int64), ("hi", C.c_int64)]


class _View(C.Structure):
    _native_ = "tr_math_ipoly_view"
    _fields_ = [("nterms", C.c_uint64), ("nvars", C.c_int32), ("degree", C.c_int32), ("exponents", C.c_void_p),
                ("coefficients", C.c_void_p)]


class _Info(C.Structure):
    _native_ = "tr_math_handle_info"
    _fields_ = [("kind", C.c_int32), ("flags", C.c_int32), ("refs", C.c_uint64), ("bytes", C.c_uint64),
                ("slot", C.c_uint32), ("generation", C.c_uint32), ("reserved", C.c_uint64)]


_STAT_FIELDS = ("capacity", "live", "open", "deferred", "retired", "free_slots", "inserted", "closes",
                "deferred_closes", "destroyed", "retains", "releases", "refused_closed", "refused_invalid",
                "payload_bytes", "peak_payload_bytes", "table_bytes", "allocations", "allocations_total")


class _Stats(C.Structure):
    _native_ = "tr_math_handle_stats"
    _fields_ = [(name, C.c_uint64) for name in _STAT_FIELDS] + [("live_kind", C.c_uint64 * 8)]


# the ctypes Structures of the modules that predate the registry (their files are fixed): native name, ABI version
_EXISTING = {
    "torbles.mathematics._Rational": ("tr_math_rational", 1),
    "torbles.mathematics._Term": ("tr_math_term", 1),
    "torbles.math_domains._DomainSpec": ("tr_math_domain_spec", 1),
    "torbles.math_domains._ModuleInfo": ("tr_math_module_info", 1),
    "torbles.math_domains._ModuleTrace": ("tr_math_module_trace", 1),
    "torbles.math_domains._CacheStats": ("tr_math_state_stats", 1),
    "torbles.math_budgets._Budget": ("tr_math_budget", 1),
    "torbles.math_budgets._Need": ("tr_math_need", 1),
    "torbles.math_budgets._Refusal": ("tr_math_refusal", 1),
    "torbles.math_budgets._Choice": ("tr_math_choice", 1),
    "torbles.math_disc._View": ("tr_disc_view", 1),
    "torbles.math_disc._Evidence": ("tr_disc_evidence", 1),
}

_native = None
_gpu_ready = False
_init = threading.Lock()


def _api():
    global _native
    if _native is None:
        with _init:
            if _native is None:
                u64, i32, P, ptr, H = C.c_uint64, C.c_int32, C.POINTER, C.c_void_p, C.c_uint64
                sig = {
                    "abi": (i32, [P(_AbiEntry), i32, C.c_uint32]),
                    "table_bytes": (u64, [u64]),
                    "create": (ptr, [C.c_uint32, C.c_uint32, P(_Budget), P(i32), P(_Refusal)]),
                    "free": (i32, [ptr]),
                    "retain": (i32, [ptr, H, i32, P(ptr)]),
                    "release": (i32, [ptr, H]),
                    "close": (i32, [ptr, H]),
                    "info": (i32, [ptr, H, P(_Info)]),
                    "stats": (i32, [ptr, P(_Stats)]),
                    "ipoly_bytes": (u64, [i32, u64]),
                    "ipoly_create": (i32, [ptr, i32, u64, ptr, ptr, P(_Budget), P(H), P(_Refusal)]),
                    "ipoly_evaluate": (i32, [ptr, H, ptr, u64, i32, P(_I128), ptr, P(u64), P(_Budget), P(_Refusal)]),
                    "ipoly_view": (i32, [ptr, H, P(_View)]),
                    "view_evaluate": (i32, [P(_View), ptr, u64, P(_I128), ptr, P(u64)]),
                    "device_bytes": (u64, [P(_View), u64]),
                    "domain_create": (i32, [ptr, P(_DomainSpec), ptr, C.c_uint32, H, P(H), P(_Refusal)]),
                    "domain_spec": (i32, [ptr, H, P(_DomainSpec)]),
                    "bank_create": (i32, [ptr, i32, i32, ptr, ptr, P(H), P(_Refusal)]),
                    "bank_moment": (i32, [ptr, H, ptr, P(C.c_int64)]),
                    "cache_create": (i32, [ptr, i32, u64, P(H), P(_Refusal)]),
                    "cache_put": (i32, [ptr, H, H, ptr, C.c_uint32, H]),
                    "cache_get": (i32, [ptr, H, H, ptr, C.c_uint32, i32, P(H), P(_Refusal)]),
                    "cache_stats": (i32, [ptr, H, P(_CacheStats)]),
                }
                native = {}
                for name, (restype, args) in sig.items():
                    fn = getattr(algq.lib, "tr_math_handles_" + name)
                    fn.restype, fn.argtypes = restype, args
                    native[name] = fn
                problems, _ = _compare(python_structures(import_all=False), {"libalg": _entries(native["abi"])},
                                       strict=False)
                if problems:   # fail loudly before any struct of a loaded module crosses the boundary
                    raise AbiError("ctypes ABI mismatch with libalg.so:\n  " + "\n  ".join(problems))
                _native = native
    return _native


def _gpu():
    global _gpu_ready
    from . import math_gpu
    if not _gpu_ready:
        math_gpu.bind("tr_mgpu_handles_evaluate", C.c_int,
                      [C.POINTER(_View), C.c_void_p, C.c_uint64, C.POINTER(_I128), C.c_void_p,
                       C.POINTER(C.c_uint64), C.POINTER(_Budget), C.POINTER(_Refusal)])
        abi = math_gpu.bind("tr_mgpu_handles_abi", C.c_int, [C.POINTER(_AbiEntry), C.c_int, C.c_uint])
        problems, _ = _compare(python_structures(import_all=False), {"cuda": _entries(abi)}, strict=False)
        if problems:
            raise AbiError("ctypes ABI mismatch with the CUDA library:\n  " + "\n  ".join(problems))
        _gpu_ready = True
    return math_gpu


def _check(code, refusal=None, what=""):
    if code == CLOSED:
        raise ClosedHandleError(f"{what}: the handle is closed" if what else "the handle is closed")
    return _math_check(code, refusal, what)


# ------------------------------------------------------------------ the ABI check
def _entries(fn):
    n = fn(None, 0, C.sizeof(_AbiEntry))
    if n < 0:
        raise AbiError(f"the native ABI registry refused this tr_math_abi_entry ({C.sizeof(_AbiEntry)} bytes)")
    out = (_AbiEntry * max(n, 1))()
    fn(out, n, C.sizeof(_AbiEntry))
    return [dict(name=e.name.decode(), size=int(e.size), align=int(e.align), version=int(e.version),
                 reserved=int(e.reserved)) for e in out[:n]]


def abi_entries(library="cpu"):
    """the native registry of libalg.so ("cpu") or of the loaded CUDA library ("cuda"): name, size, align, version"""
    if library == "cpu":
        return _entries(_api()["abi"])
    if library == "cuda":
        return _entries(_gpu().lib().tr_mgpu_handles_abi)
    raise ValueError("library is 'cpu' or 'cuda'")


def python_structures(import_all=True):
    """every ctypes Structure / Union defined at module level in a torbles.math* module (all of them imported when
    import_all, else only those already loaded)"""
    found, seen = [], set()
    for path in sorted(Path(__file__).resolve().parent.glob("math*.py")):
        name = "torbles." + path.stem
        module = sys.modules.get(name)
        if module is None:
            if not import_all:
                continue
            module = importlib.import_module(name)
        for attr in sorted(vars(module)):
            cls = getattr(module, attr)
            if (isinstance(cls, type) and issubclass(cls, (C.Structure, C.Union)) and cls.__module__ == name
                    and id(cls) not in seen):
                seen.add(id(cls))
                found.append(cls)
    return found


def _registration(cls):
    if "_native_" in cls.__dict__:
        return cls.__dict__["_native_"], int(cls.__dict__.get("_abi_version_", 1))
    return _EXISTING.get(f"{cls.__module__}.{cls.__qualname__}", (None, None))


def _compare(classes, registries, strict=True):
    """(problems, rows): each Structure against every native registry that has its struct; strict also refuses an
    unregistered Structure, a Structure no registry knows, and registries that disagree"""
    problems, rows, tables = [], [], {}
    for lib, entries in registries.items():
        table = {}
        for e in entries:
            if e["reserved"] != 0 or not e["name"]:
                problems.append(f"{lib}: malformed registry entry {e}")
            if e["name"] in table and table[e["name"]] != e:
                problems.append(f"{lib}: {e['name']} registered twice with different layouts")
            table[e["name"]] = e
        tables[lib] = table
    for cls in classes:
        where = f"{cls.__module__}.{cls.__qualname__}"
        native, version = _registration(cls)
        if native is None:
            if strict:
                problems.append(f"{where}: ctypes Structure without a native registration (_native_)")
            continue
        found = [(lib, table[native]) for lib, table in tables.items() if native in table]
        if not found and strict:
            problems.append(f"{where} -> {native}: in no native ABI registry ({', '.join(tables)})")
        for lib, e in found:
            got = dict(size=C.sizeof(cls), align=C.alignment(cls), version=version)
            want = dict(size=e["size"], align=e["align"], version=e["version"])
            if got != want:
                problems.append(f"{where} -> {native} in {lib}: Python {got} != native {want}")
            rows.append(dict(structure=where, native=native, library=lib, **got, match=got == want))
    if strict:
        names = set().union(*(set(t) for t in tables.values())) if tables else set()
        for name in sorted(names):
            layouts = {lib: (t[name]["size"], t[name]["align"], t[name]["version"]) for lib, t in tables.items()
                       if name in t}
            if len(set(layouts.values())) > 1:
                problems.append(f"{name}: the libraries disagree {layouts}")
    return problems, rows


def abi_check(cuda=True, import_all=True):
    """Every ctypes Structure of the torbles.math* modules against the native registries of libalg.so and (cuda)
    the CUDA library: AbiError listing every mismatch, unregistered Structure or disagreement; else a report."""
    registries = {"libalg": abi_entries("cpu")}
    if cuda:
        registries["cuda"] = abi_entries("cuda")
    classes = python_structures(import_all)
    problems, rows = _compare(classes, registries)
    if problems:
        raise AbiError("ctypes ABI mismatch:\n  " + "\n  ".join(problems))
    mirrored = {r["native"] for r in rows}
    registered = {e["name"] for entries in registries.values() for e in entries}
    return dict(structures=len(classes), checked=rows,
                registries={lib: sorted(e["name"] for e in entries) for lib, entries in registries.items()},
                without_python_mirror=sorted(registered - mirrored))


# ------------------------------------------------------------------ argument conversion
def _i64(value, what):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be an exact integer")
    value = int(value)
    if not I64[0] <= value <= I64[1]:
        raise OverflowError(f"{what} does not fit int64")
    return value


def _points(points, nvars):
    # an int64/uint64 ndarray is used as it is; anything else is read as Python objects, so every coordinate is
    # checked exactly (no float or silently converted value reaches the native code)
    a = points if isinstance(points, np.ndarray) else np.asarray(points, dtype=object)
    if a.size == 0:
        a = np.zeros((a.shape[0] if a.ndim == 2 else 0, nvars), dtype=np.int64)
    elif a.dtype == object:
        flat = a.ravel().tolist()
        for v in flat:
            _i64(v, "a point coordinate")
        a = np.array(flat, dtype=np.int64).reshape(a.shape)
    elif a.dtype.kind not in "iu":
        raise TypeError("points must be exact integers (an int64 array or Python ints)")
    elif a.dtype.kind == "u" and a.max() > I64[1]:
        raise OverflowError("a point coordinate does not fit int64")
    if a.ndim == 1 and nvars == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[1] != nvars:
        raise ValueError(f"points must have shape (n, {nvars})")
    return np.ascontiguousarray(a, dtype=np.int64)


_NO_POINTS = np.zeros(1, dtype=np.int64)   # read-only stand-in buffer for points without coordinates


def _pointer(a):
    """a non-NULL data pointer, also for an empty array (the native code refuses NULL with points to read)"""
    return a.ctypes.data if a.size else _NO_POINTS.ctypes.data


# ------------------------------------------------------------------ tables and handles
class HandleTable:
    """A caller-owned native table of handles (capacity slots, each reused for at most generation_limit
    generations). Python serializes only its own lifetime: close() waits for the operations in flight and then
    frees every live payload; a closed table refuses everything with ClosedHandleError."""

    def __init__(self, capacity=1024, *, generation_limit=None, budget=None):
        native = _api()
        self._ptr = None
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or not 1 <= capacity <= 1 << 24:
            raise ValueError("capacity must be an integer between 1 and 2^24")
        if generation_limit is not None and (not isinstance(generation_limit, Integral)
                                             or not 1 <= generation_limit <= (1 << 32) - 1):
            raise ValueError("generation_limit must be an integer between 1 and 2^32-1")
        error, refusal = C.c_int32(), _Refusal()
        ptr = native["create"](int(capacity), int(generation_limit or 0), C.byref(budget.raw) if budget else None,
                               C.byref(error), C.byref(refusal))
        _check(error.value, refusal, "handle table")
        if not ptr:
            raise MemoryError("native handle table unavailable")
        self._native, self._ptr, self.capacity = native, ptr, int(capacity)
        self._lock = threading.RLock()        # re-entrant: a handle's __del__ may run inside an operation
        self._idle = threading.Condition(self._lock)
        self._active, self._closing, self._local = 0, False, threading.local()

    # the table's lifetime: every native call runs between _enter and _exit
    def _enter(self):
        with self._lock:
            if self._ptr is None or self._closing:
                raise ClosedHandleError("the handle table is closed")
            self._active += 1
            self._local.depth = getattr(self._local, "depth", 0) + 1
            return self._ptr

    def _exit(self):
        with self._lock:
            self._active -= 1
            self._local.depth -= 1
            if not self._active:
                self._idle.notify_all()

    @contextlib.contextmanager
    def _use(self):
        ptr = self._enter()
        try:
            yield ptr
        finally:
            self._exit()

    @property
    def closed(self):
        return self._ptr is None

    def close(self):
        """waits for the operations in flight (from other threads), then frees the table and every live payload"""
        if getattr(self._local, "depth", 0):
            raise RuntimeError("a handle table cannot be closed inside one of its own operations")
        with self._lock:
            if self._ptr is None:
                return
            self._closing = True
            while self._active:
                self._idle.wait()
            rc = self._native["free"](self._ptr)
            if rc:   # a native reference outside a Python operation: never expected
                self._closing = False
                _check(rc, what="handle table close")
            self._ptr = None

    def __del__(self):
        try:
            if getattr(self, "_ptr", None) and not self._active:
                self._native["free"](self._ptr)
                self._ptr = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return "HandleTable(closed)" if self.closed else f"HandleTable(capacity={self.capacity})"

    def stats(self):
        """the native counters (math_handles.h, tr_math_handle_stats) as a dict"""
        out = _Stats()
        with self._use() as t:
            _check(self._native["stats"](t, C.byref(out)))
        result = {name: int(getattr(out, name)) for name in _STAT_FIELDS}
        result["live_kind"] = {"other": int(out.live_kind[0]), **{k: int(out.live_kind[v]) for k, v in KINDS.items()}}
        return result

    def table_bytes(self):
        return int(self._native["table_bytes"](self.capacity))

    # constructors
    def _new(self, cls, fn, *args, what):
        out, refusal = C.c_uint64(), _Refusal()
        with self._use() as t:
            _check(fn(t, *args, C.byref(out), C.byref(refusal)), refusal, what)
        return cls(self, out.value, owned=True)

    def polynomial(self, terms, nvars=None, budget=None):
        """an immutable sparse integer polynomial: {exponent tuple: int64 coefficient} or (exponents, coefficient)
        pairs (repeated exponents are merged natively); exponents 0..255"""
        items = list(terms.items()) if isinstance(terms, dict) else [tuple(item) for item in terms]
        if nvars is None:
            nvars = len(tuple(items[0][0])) if items else 0
        if isinstance(nvars, bool) or not isinstance(nvars, Integral) or nvars < 0:
            raise ValueError("nvars must be a nonnegative integer")
        nvars = int(nvars)
        exponents = np.zeros((len(items), nvars), dtype=np.uint8)
        coefficients = np.zeros(len(items), dtype=np.int64)
        for i, item in enumerate(items):
            if len(item) != 2:
                raise ValueError("a term is (exponents, coefficient)")
            powers = tuple(item[0])
            if len(powers) != nvars:
                raise ValueError(f"every exponent tuple has {nvars} entries")
            for p in powers:
                if isinstance(p, bool) or not isinstance(p, Integral) or not 0 <= p <= 255:
                    raise ValueError("exponents are integers between 0 and 255")
            exponents[i] = powers
            coefficients[i] = _i64(item[1], "a coefficient")
        b = budget or Budget()
        return self._new(Polynomial, self._native["ipoly_create"], nvars, len(items),
                         exponents.ctypes.data if exponents.size else None,
                         coefficients.ctypes.data if len(items) else None, C.byref(b.raw), what="polynomial")

    def polynomial_from_arrays(self, exponents, coefficients, budget=None):
        """the same from arrays: exponents (nterms, nvars) integers 0..255, coefficients (nterms,) int64"""
        e, c = np.asarray(exponents), np.asarray(coefficients)
        if e.ndim != 2 or c.ndim != 1 or len(e) != len(c):
            raise ValueError("exponents is (nterms, nvars) and coefficients (nterms,)")
        if e.dtype.kind not in "iu" or (e.size and (int(e.min()) < 0 or int(e.max()) > 255)):
            raise ValueError("exponents are integers between 0 and 255")
        if c.dtype.kind not in "iu":
            raise TypeError("coefficients must be an integer array")
        if c.dtype.kind == "u" and c.size and int(c.max()) > I64[1]:
            raise OverflowError("a coefficient does not fit int64")
        e, c = np.ascontiguousarray(e, dtype=np.uint8), np.ascontiguousarray(c, dtype=np.int64)
        b = budget or Budget()
        return self._new(Polynomial, self._native["ipoly_create"], e.shape[1], len(c),
                         e.ctypes.data if e.size else None, c.ctypes.data if len(c) else None, C.byref(b.raw),
                         what="polynomial")

    def domain(self, kind="snapshot", key=b"domain", *, carrier="integer", nvars=1, ndata=None, validity="nonnull",
               grouping="scalar", revision=0, universe_rows=0, aggregate=False, origin=None):
        """an immutable tr_math_domain (math_domain.h; the semantics of torbles.math_domains.Domain)"""
        def choice(value, table, what):
            if value not in table:
                raise ValueError(f"unsupported {what} {value!r}")
            return table[value]
        if origin is not None and (not isinstance(origin, Domain) or origin.table is not self):
            raise TypeError("a domain origin must be a Domain handle of this table")
        ndata = nvars if ndata is None else ndata
        for value, what in ((nvars, "nvars"), (ndata, "ndata")):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{what} must be an exact integer")
        for value, what in ((revision, "revision"), (universe_rows, "universe_rows")):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= M64:
                raise ValueError(f"{what} must be an unsigned 64-bit integer")
        spec = _DomainSpec(1, choice(kind, DOMAIN_KINDS, "domain kind"), choice(carrier, CARRIERS, "carrier"),
                           int(nvars), int(ndata), choice(validity, VALIDITY, "validity policy"),
                           choice(grouping, GROUPING, "grouping policy"), int(bool(aggregate)), int(revision),
                           int(universe_rows))
        key = _key_bytes(key)
        buffer = C.create_string_buffer(key, len(key))
        return self._new(Domain, self._native["domain_create"], C.byref(spec), buffer, len(key),
                         origin.value if origin is not None else 0, what="domain")

    def bank(self, moments, ndata=None):
        """an immutable tr_math_bank (mathematics.h): {exponent tuple of ndata powers (total degree <= 2): int64}
        with the complete degree-two basis"""
        items = list(moments.items())
        if ndata is None:
            ndata = len(tuple(items[0][0])) if items else 0
        exponents = np.zeros((max(len(items), 1), max(int(ndata), 1)), dtype=np.uint8)
        values = np.zeros(max(len(items), 1), dtype=np.int64)
        for i, (powers, value) in enumerate(items):
            powers = tuple(powers)
            if len(powers) != ndata or any(isinstance(p, bool) or not isinstance(p, Integral) or not 0 <= p <= 2
                                           for p in powers):
                raise ValueError(f"every moment has {ndata} exponents between 0 and 2")
            exponents[i, :ndata] = powers
            values[i] = _i64(value, "a moment")
        packed = np.ascontiguousarray(exponents[:, :ndata]) if ndata else exponents
        return self._new(Bank, self._native["bank_create"], int(ndata), len(items),
                         packed.ctypes.data if ndata else None, values.ctypes.data, what="moment bank")

    def state_cache(self, capacity=64, max_bytes=8 * 1024 * 1024):
        """a mutable tr_math_state_cache (math_state.h); every call is serialized natively"""
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or not 1 <= capacity <= 128:
            raise ValueError("state cache capacity must be one to 128 entries")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral) or not 0 <= max_bytes <= M64:
            raise ValueError("max_bytes must be an unsigned 64-bit integer")
        return self._new(StateCache, self._native["cache_create"], int(capacity), int(max_bytes), what="state cache")

    def handle(self, value):
        """a borrowed wrapper of a raw handle value (never closes on garbage collection); ClosedHandleError for a
        closed handle, ValueError for one never issued"""
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= M64:
            raise ValueError("a handle is an unsigned 64-bit integer")
        info = _Info()
        with self._use() as t:
            _check(self._native["info"](t, int(value), C.byref(info)), what="handle")
        cls = {1: Polynomial, 2: Domain, 3: Bank, 4: StateCache}.get(info.kind, Handle)
        return cls(self, int(value), owned=False)


class Handle:
    """One handle of a HandleTable. Owned handles (returned by the table's constructors) are closed when collected;
    borrowed ones (HandleTable.handle) only by an explicit close()."""
    kind = None

    def __init__(self, table, value, owned=True):
        self.table, self.value, self._owned = table, int(value), owned

    def close(self):
        """"freed" (payload freed now), "deferred" (an operation still holds it: freed at its last release) or
        "closed" (already closed, or the table is closed)"""
        try:
            with self.table._use() as t:
                rc = self.table._native["close"](t, self.value)
        except ClosedHandleError:
            return "closed"
        if rc == CLOSED:
            return "closed"
        _check(rc, what="close")
        return "deferred" if rc == 1 else "freed"

    @property
    def closed(self):
        try:
            self.info()
        except ClosedHandleError:
            return True
        return False

    def info(self):
        """kind, mutable, refs (operations holding it now), declared bytes, slot, generation"""
        out = _Info()
        with self.table._use() as t:
            _check(self.table._native["info"](t, self.value, C.byref(out)), what="handle")
        return dict(kind=_KIND_NAMES.get(out.kind, out.kind), mutable=bool(out.flags & 1), refs=int(out.refs),
                    bytes=int(out.bytes), slot=int(out.slot), generation=int(out.generation))

    @property
    def slot(self):
        return self.value & 0xFFFFFFFF

    @property
    def generation(self):
        return self.value >> 32

    @contextlib.contextmanager
    def retained(self):
        """a native reference held for the block: close() inside it is deferred until the block ends"""
        with self.table._use() as t:
            payload = C.c_void_p()
            _check(self.table._native["retain"](t, self.value, KINDS.get(self.kind, 0), C.byref(payload)),
                   what="retain")
            try:
                yield self
            finally:
                _check(self.table._native["release"](t, self.value), what="release")

    def __del__(self):
        try:
            if self._owned and not self.table.closed:
                self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return f"{type(self).__name__}(slot={self.slot}, generation={self.generation})"


class Polynomial(Handle):
    """An immutable sparse integer polynomial; evaluated exactly (128 bits, overflow flagged per point)."""
    kind = "polynomial"

    def __init__(self, table, value, owned=True):
        super().__init__(table, value, owned)
        view = _View()
        with table._use() as t:
            _check(table._native["ipoly_view"](t, self.value, C.byref(view)), what="polynomial")
            _check(table._native["release"](t, self.value), what="release")
        self.nvars, self.degree, self.term_count = int(view.nvars), int(view.degree), int(view.nterms)

    @contextlib.contextmanager
    def retained(self):
        """a native reference and the immutable view of the terms (tr_math_ipoly_view), valid for the block"""
        view = _View()
        with self.table._use() as t:
            _check(self.table._native["ipoly_view"](t, self.value, C.byref(view)), what="polynomial")
            try:
                yield view
            finally:
                _check(self.table._native["release"](t, self.value), what="release")

    @property
    def terms(self):
        """the canonical terms {exponent tuple: coefficient}, read from the native state"""
        with self.retained() as view:                 # copied while the native reference is held
            n, k = int(view.nterms), int(view.nvars)
            exps = np.frombuffer(C.string_at(view.exponents, n * k), dtype=np.uint8) if n * k else np.zeros(0, np.uint8)
            coefs = np.frombuffer(C.string_at(view.coefficients, 8 * n), dtype=np.int64) if n else np.zeros(0, np.int64)
        return {tuple(int(e) for e in exps[i * k:(i + 1) * k]): int(coefs[i]) for i in range(n)}

    def device_bytes(self, npoints):
        with self.retained() as view:
            return int(self.table._native["device_bytes"](C.byref(view), int(npoints)))

    def evaluate_raw(self, points, backend="cpu", budget=None):
        """{"values": (n, 2) int64 (lo, hi words), "flags": uint8 (1 = overflow), "overflowed", "backend", "device"}"""
        pts = _points(points, self.nvars)
        n = len(pts)
        values = np.zeros((max(n, 1), 2), dtype=np.int64)
        flags = np.zeros(max(n, 1), dtype=np.uint8)
        over, refusal, b = C.c_uint64(), _Refusal(), budget or Budget()
        vptr = values.ctypes.data_as(C.POINTER(_I128))
        native = self.table._native
        if backend == "cpu":
            with self.table._use() as t:
                _check(native["ipoly_evaluate"](t, self.value, _pointer(pts), n, self.nvars, vptr, flags.ctypes.data,
                                                C.byref(over), C.byref(b.raw), C.byref(refusal)),
                       refusal, "polynomial evaluation")
            ran, device = "cpu", "native CPU"
        elif backend == "cuda":
            gpu = _gpu()
            view = _View()
            with self.table._use() as t:
                _check(native["ipoly_view"](t, self.value, C.byref(view)), what="polynomial evaluation")
                try:                                   # the handle stays retained for the whole device call
                    gpu.run("tr_mgpu_handles_evaluate", C.byref(view), _pointer(pts), n, vptr, flags.ctypes.data,
                            C.byref(over), C.byref(b.raw), C.byref(refusal), refusal=refusal,
                            what="polynomial evaluation on the device")
                finally:
                    _check(native["release"](t, self.value), what="release")
            info = gpu.device()
            ran, device = ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return dict(values=values[:n], flags=flags[:n], overflowed=int(over.value), backend=ran, device=device)

    def evaluate(self, points, backend="cpu", budget=None):
        """{"values": exact ints (None where flagged), "overflow": bools, "overflowed", "backend", "device"}"""
        raw = self.evaluate_raw(points, backend, budget)
        exact = [None if f else (int(hi) << 64) + (int(lo) & M64)
                 for (lo, hi), f in zip(raw["values"].tolist(), raw["flags"].tolist())]
        return dict(values=exact, overflow=[bool(f) for f in raw["flags"].tolist()], overflowed=raw["overflowed"],
                    backend=raw["backend"], device=raw["device"])


class Domain(Handle):
    """An immutable tr_math_domain owned by the table."""
    kind = "domain"

    @property
    def spec(self):
        out = _DomainSpec()
        with self.table._use() as t:
            _check(self.table._native["domain_spec"](t, self.value, C.byref(out)), what="domain")
        values = {name: getattr(out, name) for name, _ in out._fields_}
        for name, table in (("kind", DOMAIN_KINDS), ("carrier", CARRIERS), ("validity", VALIDITY),
                            ("grouping", GROUPING)):
            values[name] = {number: text for text, number in table.items()}[values[name]]
        values["aggregate"] = bool(values.pop("flags"))
        return values


class Bank(Handle):
    """An immutable tr_math_bank owned by the table."""
    kind = "bank"

    def moment(self, powers):
        powers = tuple(powers)
        if any(isinstance(p, bool) or not isinstance(p, Integral) or not 0 <= p <= 2 for p in powers):
            raise ValueError("moment exponents are integers between 0 and 2")
        exps = (C.c_uint8 * max(len(powers), 1))(*powers)
        out = C.c_int64()
        with self.table._use() as t:
            _check(self.table._native["bank_moment"](t, self.value, exps if powers else None, C.byref(out)),
                   what="moment")
        return int(out.value)


class StateCache(Handle):
    """A mutable tr_math_state_cache owned by the table; every call is serialized by its native mutex."""
    kind = "state cache"

    def put(self, domain, basis_key, bank):
        """"stored", or "skipped_budget" when the bounded cache declines the entry (the bank stays usable)"""
        if not isinstance(domain, Domain) or not isinstance(bank, Bank) or domain.table is not self.table \
                or bank.table is not self.table:
            raise TypeError("put takes a Domain and a Bank of the same table")
        key = _key_bytes(basis_key)
        with self.table._use() as t:
            rc = self.table._native["cache_put"](t, self.value, domain.value, C.create_string_buffer(key, len(key)),
                                                 len(key), bank.value)
        if rc == -5:
            return "skipped_budget"
        _check(rc, what="state cache put")
        return "stored"

    def get(self, domain, basis_key, ndata):
        """an owned Bank (a clone, independent of the cache) on a hit, None on a miss"""
        if not isinstance(domain, Domain) or domain.table is not self.table:
            raise TypeError("get takes a Domain of the same table")
        key = _key_bytes(basis_key)
        out, refusal = C.c_uint64(), _Refusal()
        with self.table._use() as t:
            _check(self.table._native["cache_get"](t, self.value, domain.value, C.create_string_buffer(key, len(key)),
                                                   len(key), int(ndata), C.byref(out), C.byref(refusal)),
                   refusal, "state cache get")
        return Bank(self.table, out.value, owned=True) if out.value else None

    @property
    def stats(self):
        out = _CacheStats()
        with self.table._use() as t:
            _check(self.table._native["cache_stats"](t, self.value, C.byref(out)), what="state cache")
        return {name: int(getattr(out, name)) for name, _ in out._fields_}
