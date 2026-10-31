# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-005: the native module and operator registry (torbles/math_registry.c, math_registry_gpu.cu).

    from torbles import math_registry as reg
    reg.modules()                                   # every native module: carriers, laws, operators, state, proofs,
                                                    # CPU / CUDA entry points, cost coefficients
    reg.check_chain([("convert", "rational"), ("divide", "rational")], input="integer")   # or CarrierError
    col = reg.Column(db["t"], "x")                  # the engine column's codes, decoded natively
    req = reg.Request(col, [[1, 0, 3]], aggregate="sum", rows=mask)       # sum of 1 + 3x^2 over the selected rows
    reg.execute(req, "histogram", backend="cuda")   # {"values": [...], "selected", "valid", "nulls", "explanation"}

Values are exact Python ints (SUM: 128-bit exact; MIN/MAX: None when no selected row is valid). A refusal raises the
exception of its native code (math_errors.check) with .explanation (what was refused and why) attached; a
composition refusal names the missing law or conversion. Python marshals arguments and formats results only.
Opt-in: `import torbles` does not load it.
"""
import ctypes as C
import itertools
import os

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal as _BudgetRefusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["carriers", "laws", "conversions", "operators", "modules", "module", "chain", "check_chain",
           "explain_chain", "entry", "Column", "Request", "State", "execute", "build_state", "candidates", "verify",
           "REPRESENTATIONS"]

POLYS, DEGREE, COMPONENTS, MAX_CANDIDATES = 4, 8, 12, 16
REPRESENTATIONS = {"formal": 1, "moments": 2, "cells": 3, "verify": 4, "scan": 5, "histogram": 6, "power": 7,
                   "bitplane": 8, "empty": 9}
AGGREGATES = {"sum": 1, "min": 2, "max": 3}
COST_COMPONENTS = ("words", "scan", "chain", "combine", "construct", "reuse", "fusion", "materialize", "transfer",
                   "cpu_final", "gpu_final", "fixed")
REFUSALS = ("none", "law", "conversion", "operator", "carrier", "implementation", "budget", "overflow", "bound",
            "stale", "proof", "device", "invalid", "unsupported", "memory")
STATE_KINDS = ("none", "formal polynomial", "moment bank", "cell bitmap", "evidence", "histogram", "power sums",
               "plane counts", "empty proof")
PROOFS = ("domain", "snapshot", "scope", "bound", "exact_count", "catalog", "cell_domain")
REQUESTS = ("formal", "moments", "cells", "verify", "polysum")
CONVERSION_KINDS = {1: "exact", 2: "checked", 3: "enclosing"}
ON_CPU, ON_CUDA = 1, 2
_NAMES_BY_REPR = {v: k for k, v in REPRESENTATIONS.items()}


class _Step(C.Structure):
    _fields_ = [("op", C.c_int32), ("carrier", C.c_int32)]


class _Chain(C.Structure):
    _fields_ = [("input", C.c_int32), ("len", C.c_int32), ("step", _Step * 6)]


class _Operator(C.Structure):
    _fields_ = [("id", C.c_int32), ("input", C.c_int32), ("output", C.c_int32), ("required", C.c_uint32)]


class _Conversion(C.Structure):
    _fields_ = [("from_carrier", C.c_int32), ("to_carrier", C.c_int32), ("kind", C.c_int32)]


class _Refusal(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("kind", "code", "step", "op", "carrier", "law", "from_carrier", "to_carrier")]


class _Request(C.Structure):
    _fields_ = [("rows", C.c_uint64), ("ncodes", C.c_uint64), ("codes", C.c_void_p), ("validity", C.c_void_p),
                ("selected", C.c_void_p), ("offset", C.c_int64), ("source", C.c_uint64), ("revision", C.c_uint64),
                ("width", C.c_int32), ("aggregate", C.c_int32), ("npolys", C.c_int32), ("reserved", C.c_int32),
                ("degree", C.c_int32 * POLYS), ("coef", (C.c_int64 * (DEGREE + 1)) * POLYS)]


class _I128(C.Structure):
    _fields_ = [("lo", C.c_int64), ("hi", C.c_int64)]


class _Result(C.Structure):
    _fields_ = [("value", _I128 * POLYS), ("selected", C.c_uint64), ("valid", C.c_uint64), ("nulls", C.c_uint64),
                ("npolys", C.c_int32), ("aggregate", C.c_int32)]


class _Explain(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("abi", "status", "module", "representation", "device", "finalize", "reused",
                                         "code")] + [
        ("refusal", _Refusal), ("budget", _BudgetRefusal)] + [
        (n, C.c_uint64) for n in ("rows_scanned", "words_scanned", "state_elements", "state_bytes", "device_bytes",
                                  "native_calls")]


class _StateInfo(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("representation", "degree", "width", "has_validity")] + [
        ("support", C.c_uint32), ("has_selection", C.c_uint32)] + [
        (n, C.c_uint64) for n in ("elements", "bytes", "rows", "ncodes", "selected", "valid", "nulls", "source",
                                  "revision")] + [("offset", C.c_int64)]


class _Candidate(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("module", "representation", "device", "finalize", "reuse", "batch", "status",
                                         "reason", "code", "valid")] + [
        ("shared", C.c_uint32), ("reserved", C.c_uint32), ("refusal", _Refusal), ("budget", _BudgetRefusal),
        ("need", _Need), ("amount", C.c_uint64 * COMPONENTS), ("cost", C.c_double * COMPONENTS), ("total", C.c_double)]


class _Impl(C.Structure):
    _fields_ = [("op", C.c_int32), ("carrier", C.c_int32)]


class _Module(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("abi", "id", "representation", "state_kind", "requests", "devices", "nimpl",
                                         "finalize")] + [
        (n, C.c_uint32) for n in ("carriers", "laws_relied", "laws_guaranteed", "proofs")] + [
        ("operators", C.c_uint64), ("impl", _Impl * 12), ("cost", (C.c_double * COMPONENTS) * 2),
        ("name", C.c_char * 40), ("capability", C.c_char * 16), ("cpu_entry", (C.c_char * 32) * 2),
        ("cuda_entry", (C.c_char * 32) * 2)]


SIZES = {"refusal": 32, "chain": 56, "request": 384, "result": 96, "explanation": 136, "state_info": 104,
         "candidate": 376, "module": 528}
_native = None


def _api():
    global _native
    if _native is None:
        P, i32, u32, u64, i64, vp = C.POINTER, C.c_int32, C.c_uint32, C.c_uint64, C.c_int64, C.c_void_p
        sig = {
            "module_count": (i32, []), "module_get": (i32, [i32, P(_Module)]), "module_by_id": (i32, [i32, P(_Module)]),
            "chain_get": (i32, [i32, i32, P(_Chain)]),
            "carrier_info": (i32, [i32, P(u32), C.c_char_p, i32]),
            "operator_info": (i32, [i32, P(_Operator), C.c_char_p, i32]),
            "conversion_get": (i32, [i32, P(_Conversion)]), "law_name": (i32, [i32, C.c_char_p, i32]),
            "find": (i32, [i32, i32, i32]),
            "check_chain": (i32, [P(_Step), i32, i32, i32, i32, P(i32), P(_Refusal)]),
            "code_domain": (i32, [vp, u64, P(u64), P(i32)]),
            "column_create": (vp, [vp, i32, i32, i64, P(_Budget), P(_BudgetRefusal), P(i32)]),
            "column_request": (i32, [vp, vp, P(_Request)]), "column_bytes": (u64, [vp]), "column_free": (None, [vp]),
            "state_build": (vp, [P(_Request), i32, u32, i32, P(_Budget), P(_Explain), P(i32)]),
            "state_import": (vp, [P(_Request), P(_StateInfo), vp, u64, P(i32)]),
            "state_get_info": (i32, [vp, P(_StateInfo)]), "state_raw": (i32, [vp, vp, u64]),
            "state_size": (i32, [P(_Request), i32, u32, i32, i32, P(_Budget), P(u64), P(_BudgetRefusal)]),
            "state_matches": (i32, [vp, P(_Request)]), "state_free": (None, [vp]),
            "execute": (i32, [P(_Request), i32, vp, P(_Budget), P(_Result), P(_Explain)]),
            "candidates": (i32, [P(_Request), vp, i32, u64, u64, i32, P(_Budget), P(_Candidate), i32, P(i32),
                                 P(_BudgetRefusal)]),
            "verify": (i32, [vp, i64, i64, vp, P(_Explain)]),
            "candidates_verify": (i32, [vp, i32, P(_Budget), P(_Candidate), i32, P(i32), P(_BudgetRefusal)]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, "tr_reg_" + name)
            fn.restype, fn.argtypes = restype, args
            native[name] = fn
        _native = native
    return _native


def _gpu(name):
    from . import math_gpu
    P = C.POINTER
    sigs = {
        "tr_mgpu_registry_execute": [P(_Request), C.c_int32, C.c_int32, P(_Budget), P(_Result), P(_Explain)],
        "tr_mgpu_registry_state": [P(_Request), C.c_int32, C.c_uint32, C.c_int32, P(_Budget), C.c_void_p,
                                   C.c_uint64, P(_StateInfo), P(_Explain)],
        "tr_mgpu_registry_verify": [C.c_void_p, C.c_int64, C.c_int64, P(_Budget), C.c_void_p, P(_BudgetRefusal),
                                    P(_Explain)],
    }
    fn = math_gpu.bind(name, C.c_int, sigs[name])
    if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
        math_gpu.require_device()
    return fn


def _text(buffer):
    return bytes(buffer).split(b"\0", 1)[0].decode("ascii")


_names = None


def _tables():
    global _names
    if _names is None:
        api = _api()
        buf = C.create_string_buffer(32)
        carriers_, laws_, ops_ = [], [], []
        for c in range(13):
            laws_mask = C.c_uint32()
            _check(api["carrier_info"](c, C.byref(laws_mask), buf, 32))
            carriers_.append((buf.value.decode(), int(laws_mask.value)))
        for b in range(12):
            _check(api["law_name"](b, buf, 32))
            laws_.append(buf.value.decode())
        ops_.append(None)
        for op in range(1, 28):
            info = _Operator()
            _check(api["operator_info"](op, C.byref(info), buf, 32))
            ops_.append((buf.value.decode(), info.input, info.output, info.required))
        _names = dict(carriers=carriers_, laws=laws_, operators=ops_)
    return _names


def _carrier_id(name):
    names = [n for n, _ in _tables()["carriers"]]
    if isinstance(name, int):
        return name
    if name not in names or name == "any":
        raise ValueError(f"unknown carrier {name!r} (one of {', '.join(names[1:])})")
    return names.index(name)


def _op_id(name):
    if isinstance(name, int):
        return name
    names = [o[0] if o else None for o in _tables()["operators"]]
    if name not in names:
        raise ValueError(f"unknown operator {name!r}")
    return names.index(name)


def _carrier_name(c):
    t = _tables()["carriers"]
    return t[c][0] if 0 <= c < len(t) else f"carrier {c}"


def _op_name(op):
    t = _tables()["operators"]
    return t[op][0] if 0 < op < len(t) else f"operator {op}"


def _law_names(mask):
    return [n for b, n in enumerate(_tables()["laws"]) if mask >> b & 1]


def _law_name(bit):
    return _law_names(bit)[0] if bit else None


def laws():
    """the law names, in bit order"""
    return tuple(_tables()["laws"])


def carriers():
    """every carrier with the laws its operations satisfy as implemented"""
    return tuple(dict(id=i, name=n, laws=_law_names(m)) for i, (n, m) in enumerate(_tables()["carriers"]) if i)


def conversions():
    """the declared carrier conversions: exact (lossless), checked (lossless or refused), enclosing (not lossless)"""
    out = []
    for i in itertools.count():
        conv = _Conversion()
        if _api()["conversion_get"](i, C.byref(conv)) != 0:
            break
        out.append({"from": _carrier_name(conv.from_carrier), "to": _carrier_name(conv.to_carrier),
                    "kind": CONVERSION_KINDS[conv.kind]})
    return tuple(out)


def operators():
    """every operator: its fixed input / output carrier (None: the step's) and the laws it requires"""
    return tuple(dict(id=op, name=t[0], input=_carrier_name(t[1]) if t[1] else None,
                      output=_carrier_name(t[2]) if t[2] else None, requires=_law_names(t[3]))
                 for op, t in enumerate(_tables()["operators"]) if t)


def _module_dict(m):
    return dict(
        id=m.id, name=_text(m.name), capability=_text(m.capability), representation=_NAMES_BY_REPR[m.representation],
        state=STATE_KINDS[m.state_kind], requests=[r for b, r in enumerate(REQUESTS) if m.requests >> b & 1],
        devices=[d for b, d in enumerate(("cpu", "cuda")) if m.devices >> b & 1],
        cuda_finalize=[d for b, d in enumerate(("cpu", "gpu")) if m.finalize >> b & 1],
        carriers=[_carrier_name(c) for c in range(13) if m.carriers >> c & 1],
        laws_relied=_law_names(m.laws_relied), laws_guaranteed=_law_names(m.laws_guaranteed),
        proofs=[p for b, p in enumerate(PROOFS) if m.proofs >> b & 1],
        operator_ids=[op for op in range(64) if m.operators >> op & 1],
        operators=[_op_name(op) for op in range(1, 28) if m.operators >> op & 1],
        implements=[(_op_name(m.impl[i].op), _carrier_name(m.impl[i].carrier)) for i in range(m.nimpl)],
        cpu_entries=[e for e in (_text(m.cpu_entry[0]), _text(m.cpu_entry[1])) if e],
        cuda_entries=[e for e in (_text(m.cuda_entry[0]), _text(m.cuda_entry[1])) if e],
        costs={dev: dict(zip(COST_COMPONENTS, (float(v) for v in m.cost[i]))) for i, dev in enumerate(("cpu", "cuda"))},
        cost_unit="relative work units per unit amount; not time")


def modules():
    """every registered native module (the first-slice built-ins included), in id order"""
    out = []
    for i in range(_api()["module_count"]()):
        m = _Module()
        _check(_api()["module_get"](i, C.byref(m)))
        out.append(_module_dict(m))
    return tuple(out)


def module(id_or_name):
    """one module by id or representation name"""
    m = _Module()
    ident = REPRESENTATIONS[id_or_name] if isinstance(id_or_name, str) else int(id_or_name)
    _check(_api()["module_by_id"](ident, C.byref(m)), what="registry module")
    return _module_dict(m)


def chain(representation, aggregate="sum", multiplicities=False):
    """the steps a representation computes for an aggregate (family SUM, or MIN/MAX), or its multiplicity chain"""
    slot = 2 if multiplicities else (0 if aggregate == "sum" else 1)
    ch = _Chain()
    _check(_api()["chain_get"](REPRESENTATIONS[representation], slot, C.byref(ch)))
    return dict(input=_carrier_name(ch.input), steps=[(_op_name(ch.step[i].op), _carrier_name(ch.step[i].carrier))
                                                       for i in range(ch.len)])


def _refusal_dict(r):
    if r.kind == 0:
        return None
    return dict(kind=REFUSALS[r.kind], code=r.code, step=None if r.step < 0 else r.step,
                operator=_op_name(r.op) if r.op else None, carrier=_carrier_name(r.carrier) if r.carrier else None,
                law=_law_name(r.law), conversion=(_carrier_name(r.from_carrier), _carrier_name(r.to_carrier))
                if r.from_carrier else None)


def _refusal_text(d):
    if d is None:
        return ""
    where = f"step {d['step']} ({d['operator']} in {d['carrier']}): " if d.get("step") is not None else ""
    if d["kind"] == "law":
        return f"{where}needs law {d['law']}, which {d['carrier']} lacks"
    if d["kind"] == "conversion" and d.get("conversion") and d["step"] is not None:
        return f"{where}needs a lossless conversion {d['conversion'][0]} -> {d['conversion'][1]}, none is declared"
    if d["kind"] == "carrier":
        return f"{where}the operator runs only in another carrier"
    if d["kind"] == "implementation":
        return f"{where}no native module implements it"
    if d["kind"] == "operator":
        return f"{where}no such operator"
    if d["kind"] == "bound" and d.get("operator") == "eval":
        return "the checked int64 evaluation is not proven to stay within int64 over the code domain"
    if d["kind"] == "bound":
        return "|result| < 2^127 is not proven for the mod 2^128 representation (mod128 -> int128 is checked)"
    if d["kind"] == "proof":
        return "the selection is not proven empty (only an exact zero count is a proof)"
    if d["kind"] == "stale":
        return "the state's scope, source or revision differs from the request's"
    return d["kind"]


def _steps(steps):
    arr = (_Step * max(len(steps), 1))()
    for i, (op, carrier) in enumerate(steps):
        arr[i] = _Step(_op_id(op), _carrier_id(carrier))
    return arr


def explain_chain(steps, input="code", contract="exact", implemented=False):
    """{"accepted": bool, "output": carrier or None, "refusal": None or {kind, step, operator, carrier, law, conversion}}"""
    refusal, out = _Refusal(), C.c_int32()
    rc = _api()["check_chain"](_steps(steps), len(steps), _carrier_id(input),
                               {"exact": 0, "enclosure": 1}[contract], int(bool(implemented)), C.byref(out),
                               C.byref(refusal))
    return dict(accepted=rc == 0, code=rc, output=_carrier_name(out.value) if rc == 0 else None,
                refusal=_refusal_dict(refusal))


def check_chain(steps, input="code", contract="exact", implemented=False):
    """the carrier a composition ends in; CarrierError (missing law / conversion / implementation) or OperatorError
    (unknown operator) naming the refused step otherwise, with .refusal"""
    res = explain_chain(steps, input, contract, implemented)
    if res["accepted"]:
        return res["output"]
    try:
        _check(res["code"], None, _refusal_text(res["refusal"]))
    except Exception as exc:
        exc.refusal = res["refusal"]
        raise
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ requests and executions
_tokens = itertools.count(1)


class Column:
    """the stored codes (and validity flags) of one engine column, decoded natively (tr_reg_column_create); logical
    value = code + the codec's offset (a DECIMAL's value is logical / 10^scale)"""
    def __init__(self, table, name, budget=None):
        from .math_errors import UnsupportedError
        codec = getattr(table, "codecs", {}).get(name)
        offset = getattr(codec, "offset", 0) if codec is not None else 0
        if codec is not None and not isinstance(getattr(codec, "offset", None), int):
            raise UnsupportedError(f"{name}: a {type(codec).__name__} column has no logical integer values")
        if name in getattr(table, "wide", {}):
            raise UnsupportedError(f"{name}: a self-extending column's stored codes are only its low bits")
        valid = getattr(table, "nullable", {}).get(name)
        refusal, error = _BudgetRefusal(), C.c_int32()
        b = budget or Budget()
        handle = _api()["column_create"](table.at, table.idx(name), table.idx(valid) if valid else -1, int(offset),
                                         C.byref(b.raw), C.byref(refusal), C.byref(error))
        _check(error.value, refusal, f"column {name}")
        if not handle:
            raise MemoryError("native column unavailable")
        self._handle, self.table, self.name, self.offset = handle, table, name, int(offset)
        self.scale = (codec.affine[1] if codec is not None and codec.affine else 0)
        self.nullable = valid is not None
        self.token = next(_tokens)

    def _open(self):
        if not self._handle:
            raise RuntimeError("column is closed")
        return self._handle

    @property
    def bytes(self):
        return int(_api()["column_bytes"](self._open()))

    def close(self):
        if getattr(self, "_handle", None):
            _api()["column_free"](self._handle)
            self._handle = None

    __del__ = close


def _pack(rows, n):
    """a selection as packed uint64 words: a boolean mask of n rows, or an already packed uint64 bitmap"""
    a = np.asarray(rows)
    words = (n + 63) // 64
    if a.dtype == np.bool_:
        if len(a) != n:
            raise ValueError("the rows mask length differs from the rows")
        padded = np.zeros(max(words, 1) * 64, dtype=np.uint8)
        padded[:n] = a
        return np.packbits(padded, bitorder="little").view(np.uint64).copy()
    if a.dtype != np.uint64:
        raise ValueError("rows is a boolean mask or a packed uint64 bitmap (row ids: row_ids=)")
    if len(a) < words:
        raise ValueError("the rows bitmap is shorter than the rows")
    return np.ascontiguousarray(a[:max(words, 1)])


def _ids(row_ids, n):
    ids = np.asarray(row_ids, dtype=np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= n):
        raise ValueError("row ids outside the rows")
    mask = np.zeros(n, dtype=bool)
    mask[ids] = True
    return _pack(mask, n)


def _polys(polys):
    if len(polys) and isinstance(polys[0], (int, np.integer)):
        polys = [polys]
    out = []
    for p in polys:
        p = [int(c) for c in p] or [0]
        for c in p:
            if not -(1 << 63) <= c < (1 << 63):
                raise OverflowError("polynomial coefficients are int64")
        out.append(p)
    if not 1 <= len(out) <= POLYS:
        raise ValueError(f"one to {POLYS} polynomials")
    return out


class Request:
    """aggregate f_j(x) over the selected valid rows (x = code + offset), for one to four integer polynomials f_j
    given low to high; source: a Column, or raw uint32 codes (with offset=, validity=, ncodes=, source_id=)"""
    def __init__(self, source, polys, aggregate="sum", rows=None, row_ids=None, offset=0, validity=None, ncodes=None,
                 source_id=None):
        if aggregate not in AGGREGATES:
            raise ValueError("aggregate is 'sum', 'min' or 'max'")
        self.polys, self.aggregate = _polys(polys), aggregate
        raw = _Request()
        self._keep = []
        if isinstance(source, Column):
            _check(_api()["column_request"](source._open(), source.table.at, C.byref(raw)), what="column")
            raw.source = source.token
            self.column = source
            n = int(raw.rows)
            if rows is None and row_ids is None:
                used = np.asarray(source.table.used_bm)
                bits = np.ascontiguousarray(used[:max((n + 63) // 64, 1)])
                self._keep.append(bits)
                raw.selected = bits.ctypes.data
        else:
            codes = np.ascontiguousarray(source, dtype=np.uint32)
            n = len(codes)
            self._keep.append(codes)
            raw.rows, raw.codes, raw.offset = n, codes.ctypes.data if n else None, int(offset)
            if validity is not None:
                v = np.ascontiguousarray(validity, dtype=np.uint32)
                if len(v) != n:
                    raise ValueError("validity length differs from the codes")
                self._keep.append(v)
                raw.validity = v.ctypes.data if n else None
            nc, width = C.c_uint64(), C.c_int32()
            _check(_api()["code_domain"](codes.ctypes.data if n else None, n, C.byref(nc), C.byref(width)))
            if ncodes is not None:
                if int(ncodes) < int(nc.value):
                    raise ValueError("ncodes must exceed every code")
                nc = C.c_uint64(int(ncodes))
                width = C.c_int32(max(int(ncodes) - 1, 0).bit_length())
            raw.ncodes, raw.width = nc.value, width.value
            raw.source = int(source_id) if source_id is not None else next(_tokens)
            self.column = None
        if rows is not None or row_ids is not None:
            bits = _pack(rows, n) if rows is not None else _ids(row_ids, n)
            self._keep.append(bits)
            raw.selected = bits.ctypes.data
        raw.aggregate, raw.npolys = AGGREGATES[aggregate], len(self.polys)
        for j, p in enumerate(self.polys):
            raw.degree[j] = len(p) - 1
            for k, c in enumerate(p[:DEGREE + 1]):
                raw.coef[j][k] = c
        self.raw = raw
        self.rows = n

    def with_polys(self, polys, aggregate=None):
        """the same scope (source, selection) with other polynomials"""
        other = Request.__new__(Request)
        other.polys, other.aggregate = _polys(polys), aggregate or self.aggregate
        other._keep, other.column, other.rows = list(self._keep), self.column, self.rows
        raw = _Request()
        C.memmove(C.byref(raw), C.byref(self.raw), C.sizeof(_Request))
        raw.aggregate, raw.npolys = AGGREGATES[other.aggregate], len(other.polys)
        for j in range(POLYS):
            raw.degree[j] = 0
            for k in range(DEGREE + 1):
                raw.coef[j][k] = 0
        for j, p in enumerate(other.polys):
            raw.degree[j] = len(p) - 1
            for k, c in enumerate(p[:DEGREE + 1]):
                raw.coef[j][k] = c
        other.raw = raw
        return other


def _device(code):
    return {0: "cpu", 1: "cuda", 2: "cuda-emulated"}.get(code, f"device {code}")


def _explanation(e):
    budget = None if not e.budget.refused else dict(resource=RESOURCES[e.budget.resource], need=int(e.budget.need),
                                                    limit=int(e.budget.limit))
    device = _device(e.device)
    if device == "cpu":
        name = "native CPU"
    else:
        from . import math_gpu
        name = math_gpu.device()["name"]
    m = _Module()
    mname = cap = None
    if _api()["module_by_id"](e.module, C.byref(m)) == 0:
        mname, cap = _text(m.name), _text(m.capability)
    return dict(status="ran" if e.status == 0 else "refused", module=e.module, module_name=mname, capability=cap,
                representation=_NAMES_BY_REPR.get(e.representation), device=device, device_name=name,
                finalize="gpu" if e.finalize == 1 else "cpu", reused=bool(e.reused), code=e.code,
                refusal=_refusal_dict(e.refusal), budget=budget, rows_scanned=int(e.rows_scanned),
                words_scanned=int(e.words_scanned), state_elements=int(e.state_elements),
                state_bytes=int(e.state_bytes), device_bytes=int(e.device_bytes), native_calls=int(e.native_calls))


def _raise(code, explain, what):
    """map a refusal to its exception (DeviceError for the CUDA device), with .explanation attached"""
    detail = explain["refusal"]
    text = what + (": " + _refusal_text(detail) if detail and detail["kind"] not in ("budget", "invalid", "overflow")
                   else "")
    try:
        if code == -11:
            from . import math_gpu
            math_gpu.check(code, None, text)
        _check(code, explain_budget(explain), text)
    except Exception as exc:
        exc.explanation = explain
        exc.refusal = detail
        raise


def explain_budget(explain):
    b = explain.get("budget") if isinstance(explain, dict) else None
    if not b:
        return None
    r = _BudgetRefusal()
    r.resource, r.refused, r.need, r.limit = RESOURCES.index(b["resource"]), 1, b["need"], b["limit"]
    return r


def _i128(v):
    return (int(v.hi) << 64) + (int(v.lo) & ((1 << 64) - 1))


def _result(res, req, explain):
    values = [_i128(res.value[j]) for j in range(len(req.polys))]
    if req.aggregate != "sum" and res.valid == 0:
        values = [None] * len(values)
    return dict(values=values, aggregate=req.aggregate, selected=int(res.selected), valid=int(res.valid),
                nulls=int(res.nulls), explanation=explain)


class State:
    """a native state (HISTOGRAM counts, POWER sums, BITPLANE counts) of one request's scope, reusable by requests
    with the same scope (Request.with_polys)"""
    def __init__(self, handle, explanation):
        self._handle, self.explanation = handle, explanation

    def _open(self):
        if not self._handle:
            raise RuntimeError("state is closed")
        return self._handle

    @property
    def info(self):
        i = _StateInfo()
        _check(_api()["state_get_info"](self._open(), C.byref(i)))
        d = {n: getattr(i, n) for n, _ in i._fields_}
        d["representation"] = _NAMES_BY_REPR[i.representation]
        d["support"] = [k for k in range(DEGREE + 1) if i.support >> k & 1]
        return d

    def raw(self):
        """the raw state words (uint64): histogram counts, (lo, hi) power residues, plane-set counts"""
        i = _StateInfo()
        _check(_api()["state_get_info"](self._open(), C.byref(i)))
        words = int(i.elements) * (2 if i.representation == 7 else 1)
        out = np.zeros(max(words, 1), dtype=np.uint64)
        _check(_api()["state_raw"](self._open(), out.ctypes.data, out.nbytes))
        return out[:words]

    def matches(self, request):
        return bool(_check(_api()["state_matches"](self._open(), C.byref(request.raw))))

    def close(self):
        if getattr(self, "_handle", None):
            _api()["state_free"](self._handle)
            self._handle = None

    __del__ = close


def _final(finalize, representation):
    if finalize is None:
        return {"scan": 1, "histogram": 1}.get(representation, 0)
    return {"cpu": 0, "gpu": 1}[finalize]


_CPU_SIGNATURES = {
    "tr_reg_execute": (C.c_int32, [C.POINTER(_Request), C.c_int32, C.c_void_p, C.POINTER(_Budget), C.POINTER(_Result),
                                   C.POINTER(_Explain)]),
    "tr_reg_verify": (C.c_int32, [C.c_void_p, C.c_int64, C.c_int64, C.c_void_p, C.POINTER(_Explain)]),
}


def entry(representation, backend):
    """the native entry point the registry declares for a representation on a backend, bound by its name"""
    m = module(representation)
    names = m["cpu_entries"] if backend == "cpu" else m["cuda_entries"]
    if not names:
        from .math_errors import UnsupportedError
        raise UnsupportedError(f"the {representation} module has no {backend} entry point")
    name = names[0]
    if backend == "cuda":
        return name, _gpu(name)
    fn = getattr(algq.lib, name)
    fn.restype, fn.argtypes = _CPU_SIGNATURES[name]
    return name, fn


def execute(request, representation, backend="cpu", finalize=None, state=None, budget=None):
    """run one representation: {"values", "aggregate", "selected", "valid", "nulls", "explanation"}; backend "cuda"
    runs the device kernels (finalize "gpu" or "cpu" where the representation offers both), state: a matching State
    contracted without a scan (CPU). The entry point is the one the module's descriptor names."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    if backend == "cuda" and state is not None:
        raise ValueError("a cached state is contracted on the CPU (backend='cpu')")
    b = budget or Budget()
    res, ex = _Result(), _Explain()
    rid = REPRESENTATIONS[representation]
    _, fn = entry(representation, backend)
    if backend == "cpu":
        rc = fn(C.byref(request.raw), rid, state._open() if state is not None else None, C.byref(b.raw), C.byref(res),
                C.byref(ex))
    else:
        rc = fn(C.byref(request.raw), rid, _final(finalize, representation), C.byref(b.raw), C.byref(res), C.byref(ex))
    explain = _explanation(ex)
    if rc:
        _raise(rc, explain, f"{representation} ({backend})")
    return _result(res, request, explain)


def build_state(request, representation, backend="cpu", support=None, degree=None, budget=None):
    """the reusable state of a representation for the request's scope, built natively on the CPU or on the device"""
    if representation not in ("histogram", "power", "bitplane"):
        raise ValueError("states: histogram, power, bitplane")
    b = budget or Budget()
    mask = 0 if support is None else sum(1 << int(k) for k in support)
    D = int(degree or 0)
    rid = REPRESENTATIONS[representation]
    ex, error = _Explain(), C.c_int32()
    if backend == "cpu":
        handle = _api()["state_build"](C.byref(request.raw), rid, mask, D, C.byref(b.raw), C.byref(ex), C.byref(error))
        explain = _explanation(ex)
        if error.value:
            _raise(error.value, explain, f"{representation} state")
        return State(handle, explain)
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    size, refusal = C.c_uint64(), _BudgetRefusal()
    _check(_api()["state_size"](C.byref(request.raw), rid, mask, D, ON_CUDA, C.byref(b.raw), C.byref(size),
                                C.byref(refusal)), refusal, f"{representation} state on the device")
    raw = np.zeros(max(int(size.value) // 8, 1), dtype=np.uint64)
    info = _StateInfo()
    rc = _gpu("tr_mgpu_registry_state")(C.byref(request.raw), rid, mask, D, C.byref(b.raw), raw.ctypes.data,
                                        raw.nbytes, C.byref(info), C.byref(ex))
    explain = _explanation(ex)
    if rc:
        _raise(rc, explain, f"{representation} state on the device")
    handle = _api()["state_import"](C.byref(request.raw), C.byref(info), raw.ctypes.data, int(size.value),
                                    C.byref(error))
    _check(error.value, None, "state import")
    return State(handle, explain)


STATUS = {0: None, 1: "chosen", 2: "admissible", 3: "rejected"}


def _candidate_dict(c):
    return dict(module=c.module, representation=_NAMES_BY_REPR.get(c.representation),
                device={1: "cpu", 2: "cuda"}.get(c.device), finalize="gpu" if c.finalize == 1 else "cpu",
                reuse=bool(c.reuse), batch=c.batch, valid=bool(c.valid), code=c.code, refusal=_refusal_dict(c.refusal),
                need={n: int(c.need.need[i]) for i, n in enumerate(RESOURCES) if c.need.need[i]},
                amounts=dict(zip(COST_COMPONENTS, (int(v) for v in c.amount))),
                shared=[n for b, n in enumerate(COST_COMPONENTS) if c.shared >> b & 1])


def _devices(devices):
    mask = 0
    for d in devices:
        mask |= {"cpu": ON_CPU, "cuda": ON_CUDA}[d]
    return mask


def candidates(request, devices=("cpu",), state=None, estimate=None, rows_bound=None, empty_proven=False, budget=None):
    """the bounded candidate search: every representation x device the registry offers for the request, with its
    validity (law, conversion, proof, bound, stale), budget needs and work amounts per cost component"""
    b = budget or Budget()
    out = (_Candidate * MAX_CANDIDATES)()
    n, refusal = C.c_int32(), _BudgetRefusal()
    rows = request.rows
    _check(_api()["candidates"](C.byref(request.raw), state._open() if state is not None else None, _devices(devices),
                                rows if estimate is None else int(estimate), rows if rows_bound is None else int(rows_bound),
                                int(bool(empty_proven)), C.byref(b.raw), out, MAX_CANDIDATES, C.byref(n),
                                C.byref(refusal)), refusal, "candidate search")
    return [_candidate_dict(out[i]) for i in range(n.value)]


def verify(catalog, module, out, left, right=None, *, scale=1, bias=0, backend="cpu", budget=None):
    """DISC-001 verification through the registry: {"evidence": math_disc's evidence, "explanation": ...}"""
    from . import math_disc
    scale, bias = int(scale), int(bias)
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    with catalog._lock:
        v = math_disc.view(catalog, module, out, left, right)
        e, ex = math_disc._Evidence(), _Explain()
        _, fn = entry("verify", backend)
        if backend == "cpu":
            rc = fn(C.byref(v), scale, bias, C.byref(e), C.byref(ex))
        else:
            refusal = _BudgetRefusal()
            b = budget or Budget()
            rc = fn(C.byref(v), scale, bias, C.byref(b.raw), C.byref(e), C.byref(refusal), C.byref(ex))
        explain = _explanation(ex)
        if rc:
            _raise(rc, explain, f"relation verification ({backend})")
        name = explain["device"]
        return dict(evidence=math_disc._evidence(e, module, "cpu" if name == "cpu" else name, explain["device_name"]),
                    explanation=explain)
