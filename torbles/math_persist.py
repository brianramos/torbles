# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-011: versioned binary persistence of typed mathematical states (torbles/math_persist.h/.c,
math_persist_gpu.cu): FND-004 banks with their scope descriptor, DIST-001 histograms, FND-009 carriers (big integers,
big rationals, GF(p) elements), LIN-001 moment states and FND-007 certificates.

    from torbles import math_persist as mp
    data = mp.dumps(bank)                      # bytes: header, source identity, basis, metadata, payload, checksum
    mp.save(hist, "h.tms"); rec = mp.load("h.tms", expected="histogram")
    rec = mp.loads(data)                       # TruncatedStateError, CorruptStateError, UnknownVersionError, ...
    rec.type, rec.header, rec.source, rec.metadata
    rec.fresh(scope)                           # math_certificates.Freshness: the recorded source vs the current scope
    rec.bank(origin=scope)                     # the scope's bank again (StaleSourceError when the source changed)
    rec.bank()                                 # unscoped (no source can be claimed)
    rec.histogram(scope), rec.moments(scope), rec.value(), rec.certificate()
    mp.validate(data, backend="cuda")          # header and checksum (the checksum computed on the device)

A record keeps the source identity it was saved with (the table's lifetime token, revision, rows, selection digest,
column encodings and NULL domains, grouping); nothing re-stamps it, so a reloaded state can never be presented as a
refreshed view of a changed source. No address is serialized. Nothing is computed in Python. Opt-in: `import torbles`
does not load it.
"""
import ctypes as C
import json
from pathlib import Path

import numpy as np

from . import algq
from . import math_bank as _mbk
from . import math_certificates as _mcert
from .math_budgets import _Budget, _Refusal, Budget
from .math_carriers import _Elem as _GFPElem      # tr_gfp_elem (registered by FND-009)
from .math_errors import StaleStateError, check as _check

__all__ = ["dumps", "save", "loads", "load", "validate", "checksum", "need", "Record", "PersistError", "TruncatedStateError",
           "NotAStateError", "ByteOrderError", "UnknownVersionError", "MalformedStateError", "CorruptStateError",
           "WrongTypeError", "StaleSourceError", "TYPES", "FORMAT"]

FORMAT = 1
TYPES = {"bank": 1, "histogram": 2, "bigint": 3, "bigq": 4, "gfp": 5, "moments": 6, "certificate": 7}
_TYPE_NAMES = {v: k for k, v in TYPES.items()}
_CARRIERS = {1: "i64", 2: "i128", 3: "bigint", 4: "q64", 5: "bigq", 6: "gfp", 7: "binary64", 8: "interval",
             9: "cinterval", 10: "series"}
P = C.POINTER


class PersistError(ValueError):
    """A record that cannot be loaded; .status has the native reason and what was read."""


class TruncatedStateError(PersistError):
    """Fewer bytes than the header or than the total it declares."""


class NotAStateError(PersistError):
    """Not a torbles mathematical state (magic)."""


class ByteOrderError(PersistError):
    """Written in another byte order."""


class UnknownVersionError(PersistError):
    """An unknown format version, header size or type version."""


class MalformedStateError(PersistError):
    """Inconsistent sizes or sections (the checksum may be correct)."""


class CorruptStateError(PersistError):
    """The checksum differs: the bytes were changed."""


class WrongTypeError(PersistError):
    """Another state type than the one expected."""


class StaleSourceError(StaleStateError):
    """The scope given is not the recorded source (another revision, selection, NULL domain, grouping or source);
    .reasons, .freshness."""


_REASONS = {1: TruncatedStateError, 2: NotAStateError, 3: ByteOrderError, 4: UnknownVersionError,
            5: MalformedStateError, 6: CorruptStateError, 7: WrongTypeError}
_REASON_NAMES = {0: "ok", 1: "truncated", 2: "magic", 3: "byte order", 4: "version", 5: "malformed", 6: "corrupt",
                 7: "type"}


class _Header(C.Structure):
    _native_ = "tr_persist_header"
    _fields_ = [("format", C.c_uint32), ("header_bytes", C.c_uint32)] + [
        (n, C.c_int32) for n in ("type", "type_version", "carrier", "exact", "flags", "reserved")] + [
        (n, C.c_uint64) for n in ("source_bytes", "basis_bytes", "metadata_bytes", "payload_bytes", "total_bytes",
                                  "endian")] + [("checksum", C.c_uint64 * 2)]


class _Status(C.Structure):
    _native_ = "tr_persist_status"
    _fields_ = [("reason", C.c_int32), ("type", C.c_int32), ("format", C.c_uint32), ("type_version", C.c_uint32),
                ("bytes", C.c_uint64), ("needed", C.c_uint64), ("stored", C.c_uint64 * 2),
                ("computed", C.c_uint64 * 2)]


class _BankBasis(C.Structure):
    _native_ = "tr_persist_bank_basis"
    _fields_ = [(n, C.c_int32) for n in ("ncols", "nterms", "mode", "reserved")] + [
        ("ngroups", C.c_uint64), ("exponents", C.c_void_p)]


for _s, _n in ((_Header, 96), (_Status, 64), (_BankBasis, 32)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        blob = [vp, u64, P(_Budget), P(_Refusal), e]
        sig = {
            "tr_persist_bank": (vp, [vp, vp] + blob),
            "tr_persist_histogram": (vp, [vp, vp] + blob),
            "tr_persist_moments": (vp, [vp, vp] + blob),
            "tr_persist_bigint": (vp, [vp] + blob),
            "tr_persist_bigq": (vp, [vp] + blob),
            "tr_persist_gfp": (vp, [_GFPElem] + blob),
            "tr_persist_certificate": (vp, [vp] + blob),
            "tr_persist_save_need": (u64, [u64]),
            "tr_persist_checksum_need": (u64, [u64]),
            "tr_persist_load_need": (u64, [u64]),
            "tr_persist_blob_size": (u64, [vp]),
            "tr_persist_blob_copy": (i32, [vp, vp, u64]),
            "tr_persist_blob_free": (None, [vp]),
            "tr_persist_header_read": (i32, [vp, u64, P(_Header), P(_Status)]),
            "tr_persist_validate": (i32, [vp, u64, P(_Status)]),
            "tr_persist_validate_sum": (i32, [vp, u64, P(C.c_uint64 * 2), P(_Status)]),
            "tr_persist_checksum": (None, [vp, u64, P(C.c_uint64 * 2)]),
            "tr_persist_load": (vp, [vp, u64, i32, P(_Budget), P(_Refusal), P(_Status), e]),
            "tr_persist_get_header": (i32, [vp, P(_Header)]),
            "tr_persist_get_source": (vp, [vp]),
            "tr_persist_get_metadata": (i32, [vp, vp, u64, P(u64)]),
            "tr_persist_fresh": (i32, [vp, vp, P(_mcert._Dim), i32, P(_mcert._Staleness)]),
            "tr_persist_bank_view": (i32, [vp, P(_BankBasis), P(_mbk._States)]),
            "tr_persist_to_bank": (vp, [vp, vp, P(_Budget), P(_Refusal), P(_mcert._Staleness), e]),
            "tr_persist_to_histogram": (vp, [vp, vp, P(_Budget), P(_Refusal), P(_mcert._Staleness), e]),
            "tr_persist_to_moments": (vp, [vp, vp, P(_Budget), P(_Refusal), P(_mcert._Staleness), e]),
            "tr_persist_to_bigint": (vp, [vp, P(_Budget), P(_Refusal), e]),
            "tr_persist_to_bigq": (vp, [vp, P(_Budget), P(_Refusal), e]),
            "tr_persist_to_gfp": (i32, [vp, P(_GFPElem)]),
            "tr_persist_to_certificate": (vp, [vp, P(_Budget), P(_Refusal), e]),
            "tr_persist_record_free": (None, [vp]),
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


def _budget(budget):
    return (budget or Budget()).raw


def _buffer(data):
    data = bytes(data)
    return (C.c_uint8 * max(len(data), 1)).from_buffer_copy(data or b"\0"), len(data)


def _status_dict(st):
    return dict(reason=_REASON_NAMES.get(st.reason, str(st.reason)), type=_TYPE_NAMES.get(st.type, st.type),
                format=int(st.format), type_version=int(st.type_version), bytes=int(st.bytes), needed=int(st.needed),
                stored=(int(st.stored[0]), int(st.stored[1])), computed=(int(st.computed[0]), int(st.computed[1])))


def _raise_status(rc, st, refusal=None, what="state record"):
    if rc == 0:
        return
    cls = _REASONS.get(st.reason)
    if cls is not None:
        d = _status_dict(st)
        detail = {"truncated": f"{d['bytes']} bytes, {d['needed']} needed", "version":
                  f"format {d['format']}, type version {d['type_version']}", "type": f"a {d['type']} record"}.get(
                  d["reason"], d["reason"])
        err = cls(f"{what}: {detail}")
        err.status = d
        raise err
    _check(rc, refusal, what)


def _meta_bytes(meta):
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str).encode()
    return (C.c_uint8 * max(len(raw), 1)).from_buffer_copy(raw or b"\0"), len(raw)


def _blob(fn, *args, meta=None, budget=None, what="record"):
    buf, n = _meta_bytes(meta if meta is not None else {})
    error, refusal = C.c_int32(), _Refusal()
    h = _call(fn, *args, buf, n, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    _check(error.value, refusal, what)
    if not h:
        raise MemoryError(f"{what}: native allocation failed")
    try:
        size = int(_call("tr_persist_blob_size", h))
        out = (C.c_uint8 * max(size, 1))()
        _check(_call("tr_persist_blob_copy", h, out, size), what=what)
        return bytes(out[:size])
    finally:
        _call("tr_persist_blob_free", h)


def dumps(state, *, origin=None, metadata=None, budget=None):
    """the record bytes of a state: math_bank.Bank, math_dist.Histogram, math_lin.Moments (origin: the scope it was
    built from, default its own; an unscoped bank is saved without a source), math_carriers.BigInt / BigRational /
    GFElement, math_certificates.Certificate. metadata: JSON-serializable, stored with the type's own names."""
    from . import math_carriers as mc
    user = metadata if metadata is not None else None
    name = type(state).__name__
    if isinstance(state, _mbk.Bank):
        scope = origin if origin is not None else state.scope
        meta = dict(names=list(state.names), terms=[list(t) for t in state.terms], mode=state.mode, user=user)
        return _blob("tr_persist_bank", state._handle, scope._handle if scope is not None else None, meta=meta,
                     budget=budget, what="bank record")
    if name == "Histogram" and type(state).__module__.endswith("math_dist"):
        scope = origin if origin is not None else state.scope
        meta = dict(names=list(state.names), weight=state.weight, user=user)
        return _blob("tr_persist_histogram", state._open(), scope._handle if scope is not None else None, meta=meta,
                     budget=budget, what="histogram record")
    if name == "Moments" and type(state).__module__.endswith("math_lin"):
        scope = origin if origin is not None else state.scope
        meta = dict(features=list(state.features), target=state.target, user=user)
        state._open()
        return _blob("tr_persist_moments", state._handle, scope._handle if scope is not None else None, meta=meta,
                     budget=budget, what="moments record")
    if isinstance(state, mc.BigInt):
        return _blob("tr_persist_bigint", state._h, meta=dict(user=user), budget=budget, what="bigint record")
    if isinstance(state, mc.BigRational):
        return _blob("tr_persist_bigq", state._h, meta=dict(user=user), budget=budget, what="bigq record")
    if isinstance(state, mc.GFElement):
        return _blob("tr_persist_gfp", state._raw(), meta=dict(user=user), budget=budget, what="GF(p) record")
    if isinstance(state, _mcert.Certificate):
        meta = dict(names=list(state.names) if state.names is not None else None, user=user)
        return _blob("tr_persist_certificate", state._open(), meta=meta, budget=budget, what="certificate record")
    raise TypeError(f"no persisted form for {name}")


def need(operation, record_bytes):
    """the bytes budget a "save" or "load" of a record of record_bytes checks (checksum on the device: "checksum",
    8 bytes per word and 16)"""
    n = int(record_bytes)
    if operation == "save":
        return {"bytes": int(_call("tr_persist_save_need", n))}
    if operation == "load":
        return {"bytes": int(_call("tr_persist_load_need", n))}
    if operation == "checksum":
        return {"bytes": int(_call("tr_persist_checksum_need", (n + 7) // 8))}
    raise ValueError("operation is 'save', 'load' or 'checksum'")


def save(state, path, **options):
    """writes dumps(state) to path; returns the bytes written"""
    data = dumps(state, **options)
    Path(path).write_bytes(data)
    return len(data)


def loads(data, *, expected=None, budget=None):
    """a Record (expected: a type name or None); refusals raise the PersistError subclass of their reason"""
    if expected is not None and expected not in TYPES:
        raise ValueError(f"expected is one of {', '.join(TYPES)}")
    buf, n = _buffer(data)
    st, error, refusal = _Status(), C.c_int32(), _Refusal()
    h = _call("tr_persist_load", buf, n, TYPES.get(expected, 0), C.byref(_budget(budget)), C.byref(refusal),
              C.byref(st), C.byref(error))
    _raise_status(error.value, st, refusal)
    if not h:
        raise MemoryError("state record: native allocation failed")
    return Record._wrap(h, n)


def load(path, **options):
    return loads(Path(path).read_bytes(), **options)


def checksum(data, *, backend="cpu", budget=None):
    """the position-weighted sums (two lanes mod 2^64) of the 64-bit little-endian words of data (bytes, padded to 8
    with zeros) or of a uint64 array, on the CPU or the CUDA device"""
    if isinstance(data, np.ndarray):
        words = np.ascontiguousarray(data, dtype=np.uint64)
    else:
        raw = bytes(data)
        raw += b"\0" * (-len(raw) % 8)
        words = np.frombuffer(raw, dtype="<u8").astype(np.uint64)
    out = (C.c_uint64 * 2)()
    if backend == "cpu":
        _call("tr_persist_checksum", words.ctypes.data if len(words) else None, len(words), C.byref(out))
        return int(out[0]), int(out[1])
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    from . import math_gpu
    math_gpu.bind("tr_mgpu_persist_checksum", C.c_int, [C.c_void_p, C.c_uint64, P(_Budget), P(C.c_uint64 * 2),
                                                       P(_Refusal)])
    refusal = _Refusal()
    math_gpu.run("tr_mgpu_persist_checksum", words.ctypes.data if len(words) else None, len(words),
                 C.byref(_budget(budget)), C.byref(out), C.byref(refusal), refusal=refusal, what="checksum on the device")
    return int(out[0]), int(out[1])


def validate(data, *, backend="cpu", budget=None):
    """the header and checksum checks (no state is built): {"reason": "ok", ...}, or the PersistError of the
    reason. backend "cuda" computes the checksum on the device (for a large payload before it is uploaded and used)."""
    buf, n = _buffer(data)
    st = _Status()
    if backend == "cpu":
        rc = _call("tr_persist_validate", buf, n, C.byref(st))
    elif backend == "cuda":
        head = _Header()
        rc = _call("tr_persist_header_read", buf, n, C.byref(head), C.byref(st))
        _raise_status(rc, st)
        full = checksum(bytes(data)[:head.total_bytes], backend="cuda", budget=budget)
        sums = (C.c_uint64 * 2)(*full)
        rc = _call("tr_persist_validate_sum", buf, n, C.byref(sums), C.byref(st))
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    _raise_status(rc, st)
    out = _status_dict(st)
    out["backend"] = backend
    return out


class Record:
    """A loaded, immutable state record: its header, recorded source, metadata and payload (native); rehydrate it
    with bank / histogram / moments / value / certificate."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_persist.loads(...) or load(...)")

    @classmethod
    def _wrap(cls, handle, size):
        self = object.__new__(cls)
        self._handle, self.size = handle, size
        h = _Header()
        _check(_call("tr_persist_get_header", handle, C.byref(h)), what="record header")
        self._head = h
        length = C.c_uint64()
        _check(_call("tr_persist_get_metadata", handle, None, 0, C.byref(length)), what="record metadata")
        raw = (C.c_uint8 * max(int(length.value), 1))()
        _check(_call("tr_persist_get_metadata", handle, raw, length.value, C.byref(length)), what="record metadata")
        text = bytes(raw[:length.value])
        self.metadata = json.loads(text.decode()) if text else {}
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the record is closed")
        return self._handle

    @property
    def type(self):
        return _TYPE_NAMES[self._head.type]

    @property
    def header(self):
        h = self._head
        return dict(format=int(h.format), header_bytes=int(h.header_bytes), type=_TYPE_NAMES[h.type],
                    type_version=int(h.type_version), carrier=_CARRIERS.get(h.carrier, h.carrier), exact=bool(h.exact),
                    has_source=bool(h.flags & 1), immutable=bool(h.flags & 2), source_bytes=int(h.source_bytes),
                    basis_bytes=int(h.basis_bytes), metadata_bytes=int(h.metadata_bytes),
                    payload_bytes=int(h.payload_bytes), total_bytes=int(h.total_bytes),
                    endianness="little" if h.endian == 0x0102030405060708 else hex(h.endian),
                    checksum=(int(h.checksum[0]), int(h.checksum[1])))

    @property
    def source(self):
        """the recorded source (FND-007 record): identity, revision, rows, selected, selection digest, columns,
        grouping; None for a carrier or an unscoped state"""
        rec = _call("tr_persist_get_source", self._open())
        return _mcert._source(rec) if rec else None

    def fresh(self, scope, dims=()):
        """math_certificates.Freshness of the recorded source against the current scope"""
        arr, n = _mcert._dims(scope, dims) if dims else ((_mcert._Dim * 1)(), 0)
        st = _mcert._Staleness()
        scope._open()
        rc = _call("tr_persist_fresh", self._open(), scope._handle, arr, n, C.byref(st))
        if rc not in (0, -13):
            _check(rc, what="record freshness")
        return _mcert._freshness(st)

    def _stale(self, error, st, what):
        if error.value == -13:
            f = _mcert._freshness(st)
            err = StaleSourceError(f"{what}: the scope is not the recorded source ({', '.join(f.reasons)})")
            err.reasons, err.freshness = f.reasons, f
            raise err
        if error.value == -6:
            from .math_domains import DomainError
            raise DomainError(f"{what}: the record has no source to adopt into a scope")

    def bank(self, origin=None, *, budget=None):
        """the bank: adopted as origin's bank (origin must be the recorded source), or unscoped (origin None)"""
        error, refusal, st = C.c_int32(), _Refusal(), _mcert._Staleness()
        h = _call("tr_persist_to_bank", self._open(), origin._handle if origin is not None else None,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(st), C.byref(error))
        self._stale(error, st, "bank")
        h = _mbk._made(h, error, refusal, "bank from a record")
        names = self.metadata.get("names") or (list(origin.names) if origin is not None else [])
        return _mbk.Bank._wrap(h, origin, names, "persisted", "native CPU")

    def histogram(self, origin, *, budget=None):
        """the histogram, adopted as origin's (origin must be the recorded source)"""
        from . import math_dist as md
        error, refusal, st = C.c_int32(), _Refusal(), _mcert._Staleness()
        h = _call("tr_persist_to_histogram", self._open(), origin._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(st), C.byref(error))
        self._stale(error, st, "histogram")
        h = md._made(h, error, refusal, "histogram from a record")
        return md.Histogram._wrap(h, origin, self.metadata.get("names", []), self.metadata.get("weight"), "persisted",
                                  "native CPU")

    def moments(self, origin, *, budget=None):
        """the moment state, adopted as origin's (origin must be the recorded source)"""
        from . import math_lin as ml
        error, refusal, st = C.c_int32(), _Refusal(), _mcert._Staleness()
        h = _call("tr_persist_to_moments", self._open(), origin._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(st), C.byref(error))
        self._stale(error, st, "moments")
        h = ml._made(h, error, refusal, "moments from a record")
        return ml.Moments._wrap(h, origin, self.metadata.get("features", []), self.metadata.get("target"),
                                "persisted", "native CPU")

    def value(self, *, budget=None):
        """a carrier record's value: BigInt, BigRational or GFElement"""
        from . import math_carriers as mc
        error, refusal = C.c_int32(), _Refusal()
        if self.type == "bigint":
            h = _call("tr_persist_to_bigint", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            _check(error.value, refusal, "bigint from a record")
            return mc.BigInt._wrap(h, budget)
        if self.type == "bigq":
            h = _call("tr_persist_to_bigq", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            _check(error.value, refusal, "bigq from a record")
            return mc.BigRational._wrap(h, budget)
        if self.type == "gfp":
            e = _GFPElem()
            _check(_call("tr_persist_to_gfp", self._open(), C.byref(e)), what="GF(p) from a record")
            return mc.GFElement(mc.GF(int(e.p)), int(e.v))
        raise WrongTypeError(f"a {self.type} record holds no carrier value")

    def certificate(self, *, budget=None):
        """the FND-007 certificate (its own recorded scope decides its freshness)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_persist_to_certificate", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "certificate from a record")
        return _mcert.Certificate._wrap(h, self.metadata.get("names"))

    def bank_states(self):
        """copies of a bank record's states: basis (ncols, nterms, mode, groups) and rows / count / sum (exact ints)"""
        basis, st = _BankBasis(), _mbk._States()
        _check(_call("tr_persist_bank_view", self._open(), C.byref(basis), C.byref(st)), what="bank record")
        g, m = int(st.ngroups), int(st.nterms)
        rows = np.ctypeslib.as_array((C.c_uint64 * g).from_address(st.rows)).copy()
        count = (np.ctypeslib.as_array((C.c_uint64 * (g * m)).from_address(st.count)).reshape(g, m).copy()
                 if m else np.zeros((g, 0), np.uint64))
        words = (np.ctypeslib.as_array((C.c_int64 * (2 * g * m)).from_address(st.sum)).reshape(g, m, 2)
                 if m else np.zeros((g, 0, 2), np.int64))
        sums = [[(int(h) << 64) | (int(lo) & ((1 << 64) - 1)) for lo, h in row] for row in words.tolist()]
        return dict(columns=int(basis.ncols), terms=int(basis.nterms), mode=("sql", "joint")[basis.mode],
                    groups=g, rows=rows, count=count, sum=sums)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_persist_record_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Record(closed)"
        return f"Record({self.type}, {self.size} bytes, source={'yes' if self._head.flags & 1 else 'no'})"
