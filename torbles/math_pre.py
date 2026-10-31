# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PRE-001: a predicate on a derived value y = f(x) transformed into the exact set of stored codes of the generator
column (torbles/math_pre.c, math_pre_gpu.cu), natively.

    from torbles import math_pre as mp, math_bank as mbk
    codec = mp.Codec.from_table(t, "x")                      # logical = code + offset, value = logical / 10^scale
    s = mp.preimage(codec, mp.affine(Fraction(-3, 2), 7), "<", Fraction(1, 3))   # negative slope: flipped, rounded
    s.spans, s.empty, s.all, s.codes, s.contains(code), s.certificates          # exact code intervals + FND-007 laws
    mp.preimage(codec, mp.square(), "<=", 50)                  # |L| <= isqrt(50 * 10^2s): a symmetric interval
    mp.preimage(codec, mp.residue(7), "in", [1, 3])            # residue classes of codes
    mp.preimage(codec, mp.pieces([(None, 0, "affine", -1, 0), (0.., None, "square", 1, 0)]), ">", 4)
    bits, stats = s.scan(mbk.Scope.from_table(t, ["x"]), "x", direct=True, backend="cuda")   # checked row by row

A zero slope or modulus, and a piecewise map whose pieces do not cover the code domain (a missing branch), are
refused (DomainError); overlapping pieces are ValueError. A NULL value never satisfies (the scan requires presence).
Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from .math_budgets import _Budget, _Refusal, Budget
from .math_carriers import _Q
from .math_errors import UnsupportedError, check as _check

__all__ = ["Codec", "Piece", "Map", "affine", "square", "residue", "pieces", "preimage", "PreimageSet", "OPS",
           "MAX_PIECES", "MAX_VALUES"]

MAX_PIECES, MAX_VALUES = 256, 4096
KINDS = {"affine": 1, "square": 2, "residue": 3, "pieces": 4}
OPS = {"<": 1, "<=": 2, "=": 3, "==": 3, "<>": 4, "!=": 4, ">=": 5, ">": 6, "in": 7}
_I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER


class _Codec(C.Structure):
    _native_ = "tr_pre_codec"
    _fields_ = [("offset", C.c_int64), ("scale", C.c_int32), ("nullable", C.c_int32), ("lo", C.c_int64),
                ("hi", C.c_int64)]


class _Piece(C.Structure):
    _native_ = "tr_pre_piece"
    _fields_ = [("kind", C.c_int32), ("reserved", C.c_int32), ("lo_inf", C.c_int32), ("hi_inf", C.c_int32),
                ("lo", _Q), ("hi", _Q), ("a", _Q), ("b", _Q)]


class _Map(C.Structure):
    _native_ = "tr_pre_map"
    _fields_ = [("kind", C.c_int32), ("npieces", C.c_int32), ("a", _Q), ("b", _Q), ("modulus", C.c_int64),
                ("reserved", C.c_int64), ("pieces", C.c_void_p)]


class _Predicate(C.Structure):
    _native_ = "tr_pre_predicate"
    _fields_ = [("op", C.c_int32), ("reserved", C.c_int32), ("nvalues", C.c_uint64), ("c", _Q), ("values", C.c_void_p)]


class _Span(C.Structure):
    _native_ = "tr_pre_span"
    _fields_ = [(n, C.c_int64) for n in ("lo", "hi", "modulus", "rlo", "rhi")]


class _SetInfo(C.Structure):
    _native_ = "tr_pre_set_info"
    _fields_ = [("nspans", C.c_uint64), ("codes", C.c_uint64)] + [(n, C.c_int32) for n in (
        "flags", "nullable", "branches", "decided", "certified", "kind")] + [("modulus", C.c_int64), ("codec", _Codec)]


class _ScanStats(C.Structure):
    _native_ = "tr_pre_scan_stats"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "nulls", "in_set", "direct_true", "mismatches",
                                          "first_mismatch", "overflow", "first_overflow", "outside")]


class _View(C.Structure):
    _native_ = "tr_pre_view"
    _fields_ = [("codec", _Codec), ("op", C.c_int32), ("residue", C.c_int32), ("nbranch", C.c_int32),
                ("reserved", C.c_int32), ("nspans", C.c_uint64), ("spans", C.c_void_p), ("c", _Q),
                ("nvalues", C.c_uint64), ("values", C.c_void_p), ("branch_lo", C.c_void_p), ("branch_hi", C.c_void_p),
                ("node_off", C.c_void_p), ("nodes", C.c_void_p), ("nnodes", C.c_uint64)]


for _s, _n in ((_Codec, 32), (_Piece, 80), (_Map, 64), (_Predicate, 40), (_Span, 40), (_SetInfo, 80),
               (_ScanStats, 80), (_View, 136)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, i64, e = C.c_void_p, C.c_int32, C.c_int64, P(C.c_int32)
        sig = {
            "tr_pre_codec_for_column": (i32, [vp, i32, i64, i32, i32, P(_Codec)]),
            "tr_pre_preimage": (vp, [P(_Codec), P(_Map), P(_Predicate), P(_Budget), P(_Refusal), e]),
            "tr_pre_set_get_info": (i32, [vp, P(_SetInfo)]),
            "tr_pre_set_spans": (i32, [vp, P(_Span), C.c_uint64]),
            "tr_pre_set_contains": (i32, [vp, i64]),
            "tr_pre_set_get_view": (i32, [vp, P(_View)]),
            "tr_pre_set_certificate": (vp, [vp, i32]),
            "tr_pre_set_free": (None, [vp]),
            "tr_pre_scan": (i32, [vp, vp, i32, i32, vp, P(_ScanStats), P(_Budget), P(_Refusal)]),
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


def _frac(v, what="number"):
    if isinstance(v, bool):
        raise TypeError(f"{what}: a boolean is not a number")
    if isinstance(v, Integral):
        f = Fraction(int(v))
    elif isinstance(v, Fraction):
        f = v
    elif isinstance(v, Decimal):
        f = Fraction(v)
    elif isinstance(v, str):
        f = Fraction(Decimal(v))
    else:
        raise TypeError(f"{what}: exact numbers only (int, Fraction, Decimal, str), got {type(v).__name__}")
    if not (_I64[0] <= f.numerator <= _I64[1] and f.denominator <= _I64[1]):
        raise OverflowError(f"{what} {f} does not fit an int64 rational")
    return f


def _q(v, what="number"):
    f = _frac(v, what)
    return _Q(f.numerator, f.denominator)


# ------------------------------------------------------------------ codecs and maps
class Codec:
    """value = (code + offset) / 10^scale over the stored codes lo..hi; nullable: the column may hold NULL"""

    def __init__(self, offset=0, scale=0, lo=0, hi=(1 << 32) - 1, nullable=False):
        self.raw = _Codec(int(offset), int(scale), int(bool(nullable)), int(lo), int(hi))

    @classmethod
    def from_table(cls, table, name):
        """the codec of an engine column (its coltypes affine codec, its width's code domain, its validity)"""
        codec = table.codecs.get(name) if hasattr(table, "codecs") else None
        if name not in table.cols:
            raise ValueError(f"{table.name} has no column {name!r}")
        if codec is not None and codec.affine is None:
            raise UnsupportedError(f"{name!r}: a {codec.kind} column has no affine codec")
        if name in getattr(table, "wide", {}):
            raise UnsupportedError(f"{name!r}: a self-extending (wide) column is not supported")
        offset, scale = codec.affine if codec is not None else (0, 0)
        self = cls.__new__(cls)
        self.raw = _Codec()
        nullable = name in getattr(table, "nullable", {})
        _check(_call("tr_pre_codec_for_column", table.at, table.idx(name), int(offset), int(scale), int(nullable),
                     C.byref(self.raw)), what="codec")
        return self

    offset = property(lambda self: int(self.raw.offset))
    scale = property(lambda self: int(self.raw.scale))
    lo = property(lambda self: int(self.raw.lo))
    hi = property(lambda self: int(self.raw.hi))
    nullable = property(lambda self: bool(self.raw.nullable))

    def value(self, code):
        """the exact value of a code (Fraction)"""
        return Fraction(int(code) + self.offset, 10 ** self.scale)

    def __repr__(self):
        return f"Codec(offset={self.offset}, scale={self.scale}, codes={self.lo}..{self.hi}, nullable={self.nullable})"


class Piece:
    """a piece of a piecewise map: x in [lo, hi] (None: unbounded), y = a x + b ("affine") or a x^2 + b ("square")"""

    def __init__(self, lo, hi, kind="affine", a=1, b=0):
        if kind not in ("affine", "square"):
            raise ValueError("a piece is 'affine' or 'square'")
        self.lo, self.hi, self.kind = (None if lo is None else _frac(lo, "piece bound"),
                                       None if hi is None else _frac(hi, "piece bound"), kind)
        self.a, self.b = _frac(a, "slope"), _frac(b, "intercept")

    def __repr__(self):
        return f"Piece([{self.lo}, {self.hi}], {self.kind}, a={self.a}, b={self.b})"


class Map:
    def __init__(self, kind, a=1, b=0, modulus=0, pieces=()):
        self.kind, self.a, self.b, self.modulus = kind, _frac(a, "a"), _frac(b, "b"), int(modulus)
        self.pieces = [p if isinstance(p, Piece) else Piece(*p) for p in pieces]
        if len(self.pieces) > MAX_PIECES:
            raise UnsupportedError(f"at most {MAX_PIECES} pieces")

    def native(self):
        raw = _Map()
        raw.kind, raw.npieces = KINDS[self.kind], len(self.pieces)
        raw.a, raw.b = _Q(self.a.numerator, self.a.denominator), _Q(self.b.numerator, self.b.denominator)
        raw.modulus = self.modulus
        arr = None
        if self.pieces:
            arr = (_Piece * len(self.pieces))()
            for i, p in enumerate(self.pieces):
                arr[i].kind = KINDS[p.kind]
                arr[i].lo_inf, arr[i].hi_inf = int(p.lo is None), int(p.hi is None)
                arr[i].lo = _Q(0, 1) if p.lo is None else _Q(p.lo.numerator, p.lo.denominator)
                arr[i].hi = _Q(0, 1) if p.hi is None else _Q(p.hi.numerator, p.hi.denominator)
                arr[i].a = _Q(p.a.numerator, p.a.denominator)
                arr[i].b = _Q(p.b.numerator, p.b.denominator)
            raw.pieces = C.cast(arr, C.c_void_p)
        return raw, arr

    def __repr__(self):
        if self.kind == "residue":
            return f"Map(residue, modulus={self.modulus})"
        if self.kind == "pieces":
            return f"Map(pieces, {self.pieces})"
        return f"Map({self.kind}, a={self.a}, b={self.b})"


def affine(a, b=0):
    """y = a x + b (a != 0)"""
    return Map("affine", a, b)


def square(a=1, b=0):
    """y = a x^2 + b (a != 0): both branches x <= 0 and x >= 0"""
    return Map("square", a, b)


def residue(m):
    """y = x mod m (x - m floor(x / m)) for an integer m >= 1"""
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError("the modulus is an integer")
    return Map("residue", modulus=m)


def pieces(items):
    """a piecewise map: Piece objects or tuples (lo, hi, kind, a, b) whose integer code ranges cover the domain"""
    return Map("pieces", pieces=list(items))


# ------------------------------------------------------------------ preimages
def _gpu():
    from . import math_gpu
    from .math_bank import _View as _ScopeView
    math_gpu.bind("tr_mgpu_pre_scan", C.c_int, [P(_View), P(_ScopeView), C.c_int32, C.c_int32, C.c_void_p,
                                               P(_ScanStats), P(_Budget), P(_Refusal)])
    return math_gpu


def _stats(s):
    out = {name: int(getattr(s, name)) for name, _ in _ScanStats._fields_}
    for k in ("first_mismatch", "first_overflow"):
        out[k] = None if getattr(s, k) == (1 << 64) - 1 else int(getattr(s, k))
    return out


class PreimageSet:
    """The exact set of codes (native): spans, flags, the codec, the laws of the transformation, the scan."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_pre.preimage(...)")

    @classmethod
    def _wrap(cls, handle, codec, mp, op):
        self = object.__new__(cls)
        self._handle, self.codec, self.map, self.op = handle, codec, mp, op
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the set is closed")
        return self._handle

    @property
    def info(self):
        out = _SetInfo()
        _check(_call("tr_pre_set_get_info", self._open(), C.byref(out)), what="preimage")
        return dict(spans=int(out.nspans), codes=None if out.codes == (1 << 64) - 1 else int(out.codes),
                    empty=bool(out.flags & 1), all=bool(out.flags & 2), residue=bool(out.flags & 4),
                    nullable=bool(out.nullable), branches=int(out.branches), decided=int(out.decided),
                    certified=int(out.certified), modulus=int(out.modulus))

    empty = property(lambda self: self.info["empty"])
    all = property(lambda self: self.info["all"])
    codes = property(lambda self: self.info["codes"])

    @property
    def spans(self):
        """[(lo, hi)] code intervals, or [(lo, hi, modulus, rlo, rhi)] residue spans"""
        n = self.info["spans"]
        arr = (_Span * max(n, 1))()
        _check(_call("tr_pre_set_spans", self._open(), arr, n), what="preimage spans")
        res = self.info["residue"]
        return [(int(s.lo), int(s.hi), int(s.modulus), int(s.rlo), int(s.rhi)) if res else (int(s.lo), int(s.hi))
                for s in arr[:n]]

    def contains(self, code):
        return bool(_check(_call("tr_pre_set_contains", self._open(), int(code)), what="membership"))

    @property
    def certificates(self):
        """the FND-007 laws of the transformation, one per branch (None: coefficients past int64)"""
        from . import math_certificates as mc
        out = []
        for i in range(self.info["branches"]):
            h = _call("tr_pre_set_certificate", self._open(), i)
            c = mc.Certificate._wrap(h, ["code"], owned=False) if h else None
            if c is not None:
                c._parent = self
            out.append(c)
        return out

    def view(self):
        v = _View()
        _check(_call("tr_pre_set_get_view", self._open(), C.byref(v)), what="preimage view")
        return v

    def scan(self, scope, column, *, direct=False, backend="cpu", budget=None):
        """bitmap of the selected rows whose present value's code is in the set; direct: also evaluate the original
        predicate on every row (UNC-001's exact row evaluator) and count mismatches"""
        if column not in scope.names:
            raise ValueError(f"{column!r} is not a column of the scope")
        col = scope.names.index(column)
        nw = (scope.rows + 63) // 64
        bits = np.zeros(max(nw, 1), dtype=np.uint64)
        st, refusal = _ScanStats(), _Refusal()
        braw = (budget or Budget()).raw
        if backend == "cpu":
            rc = _call("tr_pre_scan", self._open(), scope._handle, col, int(bool(direct)), bits.ctypes.data,
                       C.byref(st), C.byref(braw), C.byref(refusal))
        elif backend == "cuda":
            import os
            g = _gpu()
            if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
                g.require_device()
            v, sv = self.view(), scope.view()
            rc = g.lib().tr_mgpu_pre_scan(C.byref(v), C.byref(sv), col, int(bool(direct)), bits.ctypes.data,
                                          C.byref(st), C.byref(braw), C.byref(refusal))
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        stats = _stats(st)
        if rc == -3:
            err = OverflowError(f"scan: {stats['overflow']} rows leave the exact int64 rational evaluation")
            err.stats = stats
            raise err
        if rc and backend == "cuda":
            from . import math_gpu
            math_gpu.check(rc, refusal, "scan")
        _check(rc, refusal, "scan")
        if backend == "cpu":
            stats.update(backend="cpu", device="native CPU")
        else:
            from . import math_gpu
            d = math_gpu.device()
            stats.update(backend="cuda-emulated" if d["emulated"] else "cuda", device=d["name"])
        return bits[:nw], stats

    def plane_program(self, table, column):
        """the engine's bit-plane program (algq PInstr) of the code intervals: OR of BETWEENs, AND the column's
        validity (plain spans only; a residue set has no plane form here)"""
        if self.info["residue"]:
            raise UnsupportedError("residue spans have no plane program (use a code lookup)")
        P_ = algq.P
        c = table.idx(column)
        prog = []
        spans = self.spans
        if not spans:
            prog.append(algq.PInstr(P_["LT"], c, 0, 0))                 # code < 0: no row
        for i, (lo, hi) in enumerate(spans):
            prog.append(algq.PInstr(P_["BETWEEN"], c, int(lo), int(hi)))
            if i:
                prog.append(algq.PInstr(P_["OR"], 0, 0, 0))
        valid = getattr(table, "nullable", {}).get(column)
        if valid is not None:
            prog.append(algq.PInstr(P_["EQ"], table.idx(valid), 1, 0))
            prog.append(algq.PInstr(P_["AND"], 0, 0, 0))
        return prog

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_pre_set_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "PreimageSet(closed)"
        i = self.info
        return f"PreimageSet({self.map!r} {self.op} ..., spans={i['spans']}, codes={i['codes']})"


def preimage(codec, mp, op, c=None, *, values=None, budget=None):
    """the exact set of codes whose value satisfies f(x) op c (op <, <=, =, <>, >=, >) or f(x) in values (op "in")"""
    if op not in OPS:
        raise ValueError(f"op is one of {', '.join(OPS)}")
    pr = _Predicate()
    pr.op = OPS[op]
    keep = None
    if op == "in":
        vals = list(values if values is not None else (c if c is not None else []))
        if len(vals) > MAX_VALUES:
            raise UnsupportedError(f"at most {MAX_VALUES} values")
        keep = (_Q * max(len(vals), 1))(*[_q(v, "value") for v in vals])
        pr.nvalues, pr.values = len(vals), C.cast(keep, C.c_void_p)
        pr.c = _Q(0, 1)
    else:
        if c is None:
            raise ValueError("a threshold is needed")
        pr.c = _q(c, "threshold")
    raw, arr = mp.native()
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_pre_preimage", C.byref(codec.raw), C.byref(raw), C.byref(pr), C.byref((budget or Budget()).raw),
              C.byref(refusal), C.byref(error))
    _check(error.value, refusal, "preimage")
    if not h:
        raise MemoryError("preimage: native allocation failed")
    return PreimageSet._wrap(h, codec, mp, op)
