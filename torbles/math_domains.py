"""Owned native mathematical domains, executable modules, and moment caches.

Descriptors declare the meaning of a mathematical state. Creating one does not
prove a relationship or provide an implementation for every named carrier.
Only registered native operators can execute a supported typed composition.
"""
import ctypes as C
from numbers import Integral
import threading

from . import algq


KINDS = {"physical": 1, "snapshot": 2, "assignments": 3, "formal": 4, "geometry": 5}
CARRIERS = {"integer": 1, "rational": 2, "boolean": 3, "gf2": 4}
VALIDITY = {"nonnull": 0, "joint": 1, "sql": 2}
GROUPING = {"scalar": 0, "dense": 1, "sparse": 2}
OPERATIONS = {"poly_add": 1, "poly_sub": 2, "poly_mul": 3, "poly_pow": 4,
              "poly_derivative": 5, "poly_integrate": 6, "poly_substitute": 7,
              "poly_evaluate": 8, "bank_reduce": 9}


class DomainError(ValueError):
    """Mathematical objects have incompatible semantic domains."""


class CarrierError(TypeError):
    """An operator does not implement the requested arithmetic carrier."""


class OperatorError(ValueError):
    """The native module does not implement the requested operator."""


class _DomainSpec(C.Structure):
    _fields_ = [(name, C.c_int32) for name in (
        "abi_version", "kind", "carrier", "nvars", "ndata", "validity", "grouping", "flags")] + [
        ("revision", C.c_uint64), ("universe_rows", C.c_uint64)]


class _ModuleInfo(C.Structure):
    _fields_ = [(name, C.c_int32) for name in (
        "abi_version", "id", "carrier", "backend", "variable_limit", "degree_limit", "term_limit", "data_limit")] + [
        ("operators", C.c_uint64), ("name", C.c_char * 32)]


class _ModuleTrace(C.Structure):
    _fields_ = [(name, C.c_int32) for name in (
        "module_id", "operation", "backend", "reason", "native_calls", "output_terms", "left_kind", "right_kind")] + [
        ("estimated_work", C.c_double)]


class _CacheStats(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "hits", "misses", "puts", "evictions", "bytes", "max_bytes")] + [
        ("capacity", C.c_int32), ("entries", C.c_int32)]


_native = None


def _check(code):
    if code == -6:
        raise DomainError("incompatible native mathematical domains")
    if code == -7:
        raise CarrierError("native operator does not support this carrier")
    if code == -8:
        raise OperatorError("native module does not implement this operator")
    from .mathematics import _check as math_check
    return math_check(code)


def _domain_api():
    global _native
    if _native is None:
        from .mathematics import _Rational
        ptr, err = C.c_void_p, C.POINTER(C.c_int32)
        bindings = {
            "domain_create": (ptr, [C.POINTER(_DomainSpec), ptr, C.c_uint32, ptr, err]),
            "domain_clone": (ptr, [ptr, err]),
            "domain_with_aggregate": (ptr, [ptr, C.c_int32, err]),
            "domain_origin": (ptr, [ptr, err]),
            "domain_get_spec": (C.c_int32, [ptr, C.POINTER(_DomainSpec)]),
            "domain_key_size": (C.c_int32, [ptr]),
            "domain_key_copy": (C.c_int32, [ptr, ptr, C.c_uint32]),
            "domain_bytes": (C.c_int64, [ptr]),
            "domain_equal": (C.c_int32, [ptr, ptr]),
            "domain_context_equal": (C.c_int32, [ptr, ptr]),
            "domain_free": (None, [ptr]),
            "module_count": (C.c_int32, []),
            "module_get": (C.c_int32, [C.c_int32, C.POINTER(_ModuleInfo)]),
            "module_apply": (ptr, [C.c_int32, C.c_int32, ptr, ptr, ptr, ptr, C.c_int32,
                                   C.c_uint32, C.c_int64, C.c_int64, C.POINTER(ptr), C.POINTER(_ModuleTrace), err]),
            "module_evaluate": (C.c_int32, [C.c_int32, ptr, ptr, C.POINTER(_Rational),
                                           C.POINTER(_Rational), C.POINTER(_ModuleTrace)]),
            "module_reduce": (ptr, [C.c_int32, ptr, ptr, ptr, ptr,
                                    C.POINTER(ptr), C.POINTER(_ModuleTrace), err]),
            "state_cache_create": (ptr, [C.c_int32, C.c_uint64, err]),
            "state_cache_get": (ptr, [ptr, ptr, ptr, C.c_uint32, C.c_int32, err]),
            "state_cache_put": (C.c_int32, [ptr, ptr, ptr, C.c_uint32, ptr]),
            "state_cache_clear": (C.c_int32, [ptr]),
            "state_cache_stats": (C.c_int32, [ptr, C.POINTER(_CacheStats)]),
            "state_cache_free": (None, [ptr]),
        }
        native = {}
        try:
            for name, (restype, arguments) in bindings.items():
                fn = getattr(algq.lib, "tr_math_" + name)
                fn.restype, fn.argtypes = restype, arguments
                native[name] = fn
        except AttributeError as exc:
            raise RuntimeError("native typed mathematics is unavailable; rebuild the prototype CPU engine") from exc
        _native = native
    return _native


def _u64(value, name):
    if not isinstance(value, Integral) or not 0 <= value <= (1 << 64)-1:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return int(value)


def _choice(value, choices, name):
    if value not in choices:
        raise ValueError(f"unsupported {name} {value!r}")
    return choices[value]


def _key_bytes(key):
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError("a domain or basis key must be bytes or a UTF-8 string")
    key = bytes(key)
    if not 1 <= len(key) <= 65536:
        raise ValueError("semantic keys require one to 65536 bytes")
    return key


class Domain:
    """An immutable, independently owned native semantic descriptor.

    A formal context normally has eight stable variable slots, with observed
    coordinates first and free parameters afterwards. Appending Python symbol
    names does not alter its identity. Observed coordinates require a copied
    physical/snapshot origin. Descriptor construction supplies no proof that
    arbitrary caller-provided identity bytes describe an actual dataset.
    """
    __hash__ = None

    def __init__(self, kind="formal", key=b"", *, carrier="rational", nvars=8, ndata=0,
                 validity="nonnull", grouping="scalar", revision=0, universe_rows=0,
                 aggregate=False, origin=None):
        self._handle = None
        self._native = _domain_api()
        if not isinstance(nvars, Integral) or not isinstance(ndata, Integral):
            raise TypeError("domain variable counts must be exact integers")
        if not 0 <= ndata <= 4 or not ndata <= nvars <= 8:
            raise ValueError("domain variable layout exceeds eight slots/four data coordinates")
        if not isinstance(aggregate, bool):
            raise TypeError("aggregate must be a Boolean flag")
        if origin is not None and not isinstance(origin, Domain):
            raise TypeError("a domain origin must be another Domain")
        key = _key_bytes(key)
        spec = _DomainSpec(1, _choice(kind, KINDS, "domain kind"),
                           _choice(carrier, CARRIERS, "carrier"), int(nvars), int(ndata),
                           _choice(validity, VALIDITY, "validity policy"),
                           _choice(grouping, GROUPING, "grouping policy"), int(aggregate),
                           _u64(revision, "revision"), _u64(universe_rows, "universe_rows"))
        error = C.c_int32()
        buffer = C.create_string_buffer(key)
        self._handle = self._native["domain_create"](C.byref(spec), buffer, len(key),
                                                     None if origin is None else origin._open(), C.byref(error))
        _check(error.value)
        if not self._handle:
            raise MemoryError("native domain descriptor unavailable")

    @classmethod
    def _from_handle(cls, handle):
        if not handle:
            raise MemoryError("native domain descriptor unavailable")
        result = cls.__new__(cls)
        result._handle, result._native = handle, _domain_api()
        return result

    def _open(self):
        if not self._handle:
            raise RuntimeError("mathematical domain is closed")
        return self._handle

    def _copy_call(self, operation, *arguments):
        error = C.c_int32()
        handle = self._native[operation](self._open(), *arguments, C.byref(error))
        _check(error.value)
        return None if not handle and operation == "domain_origin" else self._from_handle(handle)

    def clone(self):
        return self._copy_call("domain_clone")

    def with_aggregate(self, aggregate=True):
        if not isinstance(aggregate, bool):
            raise TypeError("aggregate must be a Boolean flag")
        return self._copy_call("domain_with_aggregate", int(aggregate))

    @property
    def origin(self):
        return self._copy_call("domain_origin")

    @property
    def spec(self):
        result = _DomainSpec()
        _check(self._native["domain_get_spec"](self._open(), C.byref(result)))
        values = {name: getattr(result, name) for name, _ in result._fields_}
        for name, choices in (("kind", KINDS), ("carrier", CARRIERS), ("validity", VALIDITY), ("grouping", GROUPING)):
            values[name] = {number: text for text, number in choices.items()}[values[name]]
        values["aggregate"] = bool(values.pop("flags"))
        return values

    @property
    def key(self):
        size = _check(self._native["domain_key_size"](self._open()))
        buffer = C.create_string_buffer(size)
        _check(self._native["domain_key_copy"](self._open(), buffer, size))
        return bytes(buffer.raw)

    @property
    def bytes(self):
        return _check(self._native["domain_bytes"](self._open()))

    def same_as(self, other):
        if not isinstance(other, Domain):
            return False
        return bool(_check(self._native["domain_equal"](self._open(), other._open())))

    def context_equal(self, other):
        if not isinstance(other, Domain):
            return False
        return bool(_check(self._native["domain_context_equal"](self._open(), other._open())))

    def __eq__(self, other):
        return self.same_as(other)

    def close(self):
        if self._handle:
            self._native["domain_free"](self._handle)
            self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None):
            self._native["domain_free"](self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return "Domain(closed)" if not self._handle else f"Domain({self.spec!r}, key_bytes={len(self.key)})"


def modules():
    """Describe only the native operator modules that can actually execute."""
    native = _domain_api()
    records = []
    for index in range(_check(native["module_count"]())):
        item = _ModuleInfo()
        _check(native["module_get"](index, C.byref(item)))
        record = {name: getattr(item, name) for name, _ in item._fields_}
        record["name"] = bytes(item.name).split(b"\0", 1)[0].decode("ascii")
        record["backend"] = "native CPU" if item.backend == 0 else "CUDA"
        record["carrier"] = {value: name for name, value in CARRIERS.items()}[item.carrier]
        record["operator_ids"] = [number for number in OPERATIONS.values() if item.operators >> number & 1]
        records.append(record)
    return tuple(records)


def _trace_dict(trace):
    result = {name: getattr(trace, name) for name, _ in trace._fields_}
    result["backend"] = "native CPU" if trace.backend == 0 else "CUDA"
    result["cost_unit"] = "relative shape work; not elapsed time"
    result["executed_via_registry"] = True
    return result


class MomentCache:
    """A bounded native cache of owned scalar moment snapshots.

    Cache calls are serialized by this wrapper. A returned bank is a new owned
    native object, so closing or evicting the cache does not invalidate a basis
    or a previously reduced function. ``clear`` resets counters and entries.
    """
    def __init__(self, capacity=64, *, max_bytes=8*1024*1024):
        self._handle = None
        self._native = _domain_api()
        self._lock = threading.RLock()
        if not isinstance(capacity, Integral) or not 1 <= capacity <= 128:
            raise ValueError("moment cache capacity must be one to 128 entries")
        error = C.c_int32()
        self._handle = self._native["state_cache_create"](
            int(capacity), _u64(max_bytes, "max_bytes"), C.byref(error))
        _check(error.value)
        if not self._handle:
            raise MemoryError("native moment cache unavailable")

    def _open(self):
        if not self._handle:
            raise RuntimeError("moment cache is closed")
        return self._handle

    def _get(self, domain, basis_key, ndata):
        if not isinstance(domain, Domain):
            raise TypeError("a native domain descriptor is required")
        key = _key_bytes(basis_key)
        buffer = C.create_string_buffer(key)
        error = C.c_int32()
        with self._lock:
            handle = self._native["state_cache_get"](
                self._open(), domain._open(), buffer, len(key), ndata, C.byref(error))
        _check(error.value)
        return handle

    def _put(self, domain, basis_key, bank):
        key = _key_bytes(basis_key)
        buffer = C.create_string_buffer(key)
        with self._lock:
            code = self._native["state_cache_put"](
                self._open(), domain._open(), buffer, len(key), bank)
        if code == -5:
            # A bounded optimization cache may decline a valid new snapshot.
            # The caller still owns that snapshot and can evaluate it normally.
            return "skipped_budget"
        _check(code)
        return "stored"

    @property
    def stats(self):
        result = _CacheStats()
        with self._lock:
            _check(self._native["state_cache_stats"](self._open(), C.byref(result)))
        return {name: getattr(result, name) for name, _ in result._fields_}

    def clear(self):
        with self._lock:
            _check(self._native["state_cache_clear"](self._open()))

    def close(self):
        with self._lock:
            if self._handle:
                self._native["state_cache_free"](self._handle)
                self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None):
            self._native["state_cache_free"](self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()
