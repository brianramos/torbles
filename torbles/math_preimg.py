# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PRE-002: finite roots, residues and periodic preimages: a predicate on y = f(x) as the exact set of stored codes,
as intervals, residue classes, finite sets or Boolean cubes, with a budgeted lookup or direct fallback
(torbles/math_preimg.c, math_preimg_gpu.cu), natively.

    from fractions import Fraction
    from torbles import math_pre as mp, math_preimg as mq, math_bank as mbk
    codec = mp.Codec.from_table(t, "x")                       # PRE-001's codec: logical = code + offset
    s = mq.preimage(codec, mq.poly([-6, 11, -6, 1]), ">", 0)    # (x - 1)(x - 2)(x - 3) > 0: monotone pieces
    s = mq.preimage(codec, mq.powmod(2, 15), "=", 4)            # x^2 = 4 (mod 15): CRT over 3 and 5
    s = mq.preimage(codec, mq.affmod(6, 1, 20), "<", 7)         # (6x + 1) mod 20 < 7: classes of the period 10
    s = mq.preimage(codec, mq.xor(0b1010, 8), "<=", 40)         # Boolean cubes of the stored code's bits
    s.mode, s.repr, s.branches, s.codes, s.contains(code), s.info, s.certificates
    bits, stats = s.scan(mbk.Scope.from_table(t, ["x"]), "x", direct=True, backend="cuda")
    s.verify(backend="cuda")                                    # every code of the domain against the predicate
    s.plane_program(t, "x")                                      # intervals, cubes, residues mod 2^k

When the exact union has more branches than the output budget (or would exceed the iterations or bytes budget), the
set falls back to a lookup bitmap or the direct evaluation of the predicate (fallback="auto" | "lookup" | "direct" |
"none"); no solution is ever dropped. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_pre as _mp
from .math_budgets import _Budget, _Refusal, Budget
from .math_carriers import _Q
from .math_errors import UnsupportedError, check as _check

__all__ = ["Map", "poly", "polymod", "powmod", "affmod", "xor", "bitand", "bitor", "preimage", "PreimageSet", "MODES",
           "REPRS", "FALLBACKS", "MAX_DEGREE", "MAX_VALUES"]

MAX_DEGREE, MAX_VALUES, MAX_WORDS = 16, 4096, 8
KINDS = {"poly": 1, "polymod": 2, "affmod": 3, "xor": 4, "and": 5, "or": 6}
MODES = {1: "branches", 2: "lookup", 3: "direct"}
REPRS = {1: "intervals", 2: "residues", 3: "cubes"}
FALLBACKS = {"auto": 0, "lookup": 1, "direct": 2, "none": 3}
WHY = {0: None, 1: "output", 2: "iterations", 3: "bytes", 4: "period", 5: "classes"}
_NONE = (1 << 64) - 1
P = C.POINTER


class _Map(C.Structure):
    _native_ = "tr_preimg_map"
    _fields_ = [("kind", C.c_int32), ("degree", C.c_int32), ("width", C.c_int32), ("base_is_offset", C.c_int32),
                ("modulus", C.c_int64), ("a", C.c_int64), ("b", C.c_int64), ("base", C.c_int64), ("coef", C.c_void_p)]


class _Spec(C.Structure):
    _native_ = "tr_preimg_spec"
    _fields_ = [("fallback", C.c_int32), ("reserved", C.c_int32)]


class _Branch(C.Structure):
    _native_ = "tr_preimg_branch"
    _fields_ = [(n, C.c_int64) for n in ("lo", "hi", "modulus", "rlo", "rhi")] + [
        ("mask", C.c_uint64), ("value", C.c_uint64), ("reserved", C.c_int64)]


class _Info(C.Structure):
    _native_ = "tr_preimg_info"
    _fields_ = [(n, C.c_int32) for n in ("map", "op", "mode", "repr", "fallback", "why", "words", "certified", "pieces",
                                         "critical", "empty", "all")] + [
        ("nbranch", C.c_uint64), ("branches", C.c_uint64), ("codes", C.c_uint64), ("modulus", C.c_int64),
        ("nlookup", C.c_uint64), ("iterations", C.c_uint64), ("bytes", C.c_uint64), ("codec", _mp._Codec)]


class _View(C.Structure):
    _native_ = "tr_preimg_view"
    _fields_ = [("codec", _mp._Codec)] + [(n, C.c_int32) for n in ("map", "op", "mode", "repr", "words", "degree",
                                                                    "width", "reserved")] + [
        (n, C.c_int64) for n in ("modulus", "m", "a", "b", "base")] + [
        ("nbranch", C.c_uint64), ("branch", C.c_void_p), ("nlookup", C.c_uint64), ("lookup", C.c_void_p),
        ("icoef", C.c_void_p), ("wcoef", C.c_void_p), ("c", _Q), ("nvalues", C.c_uint64), ("values", C.c_void_p),
        ("wthreshold", C.c_void_p)]


class _Verification(C.Structure):
    _native_ = "tr_preimg_verification"
    _fields_ = [(n, C.c_uint64) for n in ("codes", "members", "truths", "mismatches")] + [
        ("first", C.c_int64), ("lo", C.c_int64), ("hi", C.c_int64), ("reserved", C.c_uint64)]


for _s, _n in ((_Map, 56), (_Spec, 8), (_Branch, 64), (_Info, 136), (_View, 192), (_Verification, 64)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, i64, u64, e = C.c_void_p, C.c_int32, C.c_int64, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_preimg_preimage": (vp, [P(_mp._Codec), P(_Map), P(_mp._Predicate), P(_Spec), B, R, e]),
            "tr_preimg_set_get_info": (i32, [vp, P(_Info)]),
            "tr_preimg_set_branches": (i32, [vp, P(_Branch), u64]),
            "tr_preimg_set_contains": (i32, [vp, i64]),
            "tr_preimg_set_get_view": (i32, [vp, P(_View)]),
            "tr_preimg_set_lookup": (i32, [vp, vp, u64]),
            "tr_preimg_set_cubes": (i32, [vp, i32, P(_Branch), u64, P(u64)]),
            "tr_preimg_set_certificate": (vp, [vp]),
            "tr_preimg_set_free": (None, [vp]),
            "tr_preimg_scan": (i32, [vp, vp, i32, i32, vp, P(_mp._ScanStats), B, R]),
            "tr_preimg_verify": (i32, [vp, B, P(_Verification), R]),
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
    from .math_bank import _View as _ScopeView
    math_gpu.bind("tr_mgpu_preimg_scan", C.c_int, [P(_View), P(_ScopeView), C.c_int32, C.c_int32, C.c_void_p,
                                                  P(_mp._ScanStats), P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_preimg_verify", C.c_int, [P(_View), P(_Budget), P(_Verification), P(_Refusal)])
    return math_gpu


def _int(v, what):
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError(f"{what}: an integer")
    return int(v)


# ------------------------------------------------------------------ maps
class Map:
    """a map y = f(x): kind poly (rational coefficients, low degree first), polymod (integer coefficients, mod m),
    affmod ((a x + b) mod m), xor / and / or (on the w low bits of logical - base; base None: the codec offset)"""

    def __init__(self, kind, coefficients=(), modulus=0, a=0, b=0, width=0, base=None):
        self.kind = kind
        self.coefficients = [_mp._frac(c, "coefficient") for c in coefficients]
        if len(self.coefficients) > MAX_DEGREE + 1:
            raise UnsupportedError(f"degree at most {MAX_DEGREE}")
        if kind == "polymod" and any(c.denominator != 1 for c in self.coefficients):
            raise ValueError("polymod coefficients are integers")
        self.modulus, self.a, self.b, self.width = int(modulus), int(a), int(b), int(width)
        self.base = None if base is None else _int(base, "base")

    def native(self):
        raw = _Map()
        raw.kind = KINDS[self.kind]
        raw.degree = max(len(self.coefficients) - 1, 0)
        raw.width, raw.modulus, raw.a, raw.b = self.width, self.modulus, self.a, self.b
        raw.base_is_offset = 1 if self.base is None else 0
        raw.base = 0 if self.base is None else self.base
        arr = None
        if self.coefficients:
            arr = (_Q * len(self.coefficients))(*[_Q(c.numerator, c.denominator) for c in self.coefficients])
            raw.coef = C.cast(arr, C.c_void_p)
        return raw, arr

    def __repr__(self):
        if self.kind in ("poly", "polymod"):
            body = " + ".join(f"{c}*x^{i}" for i, c in enumerate(self.coefficients) if c) or "0"
            return f"Map({body}" + (f" mod {self.modulus})" if self.kind == "polymod" else ")")
        if self.kind == "affmod":
            return f"Map(({self.a} x + {self.b}) mod {self.modulus})"
        return f"Map(x {self.kind} {self.b} on {self.width} bits, base {'offset' if self.base is None else self.base})"


def poly(coefficients):
    """y = sum_i coefficients[i] x^i (exact rationals, degree <= 16)"""
    coefficients = list(coefficients)
    if not coefficients:
        raise ValueError("at least one coefficient")
    return Map("poly", coefficients)


def polymod(coefficients, m):
    """y = (sum_i coefficients[i] x^i) mod m (integer coefficients, x an integer, 1 <= m <= 2^62)"""
    coefficients = list(coefficients)
    if not coefficients:
        raise ValueError("at least one coefficient")
    return Map("polymod", [_int(c, "coefficient") for c in coefficients], modulus=_int(m, "modulus"))


def powmod(k, m):
    """y = x^k mod m (a power residue)"""
    k = _int(k, "exponent")
    if not 0 <= k <= MAX_DEGREE:
        raise UnsupportedError(f"an exponent 0..{MAX_DEGREE}")
    return polymod([0] * k + [1], m)


def affmod(a, b, m):
    """y = (a x + b) mod m (integers)"""
    return Map("affmod", modulus=_int(m, "modulus"), a=_int(a, "a"), b=_int(b, "b"))


def _bits(kind, c, width, base):
    width = _int(width, "width")
    return Map(kind, b=_int(c, "constant"), width=width, base=base)


def xor(c, width, base=None):
    """y = u XOR c, u = the width low bits of (logical - base) (base None: the codec offset: the stored code's bits)"""
    return _bits("xor", c, width, base)


def bitand(c, width, base=None):
    """y = u AND c"""
    return _bits("and", c, width, base)


def bitor(c, width, base=None):
    """y = u OR c"""
    return _bits("or", c, width, base)


# ------------------------------------------------------------------ the set
class PreimageSet:
    """The exact set of codes (native): its representation (branches, a lookup or the direct predicate), counts, the
    law certificate of the transformation, the scan and the exhaustive verification."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_preimg.preimage(...)")

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
        i = _Info()
        _check(_call("tr_preimg_set_get_info", self._open(), C.byref(i)), what="preimage")
        return dict(mode=MODES[i.mode], repr=REPRS.get(i.repr) if i.mode == 1 else None,
                    fallback=MODES.get(i.fallback), why=WHY[i.why], words=int(i.words), certified=int(i.certified),
                    pieces=int(i.pieces), critical=int(i.critical), empty=bool(i.empty), all=bool(i.all),
                    nbranch=int(i.nbranch), branches=None if i.branches == _NONE else int(i.branches),
                    codes=None if i.codes == _NONE else int(i.codes), modulus=int(i.modulus), nlookup=int(i.nlookup),
                    iterations=int(i.iterations), bytes=int(i.bytes))

    mode = property(lambda self: self.info["mode"])
    repr = property(lambda self: self.info["repr"])
    codes = property(lambda self: self.info["codes"])
    empty = property(lambda self: self.info["empty"])
    all = property(lambda self: self.info["all"])

    @property
    def branches(self):
        """intervals [(lo, hi)], residue classes [(lo, hi, modulus, rlo, rhi)] or cubes [(lo, hi, mask, value)]"""
        i = self.info
        n = i["nbranch"]
        arr = (_Branch * max(n, 1))()
        _check(_call("tr_preimg_set_branches", self._open(), arr, n), what="preimage branches")
        if i["repr"] == "residues":
            return [(int(b.lo), int(b.hi), int(b.modulus), int(b.rlo), int(b.rhi)) for b in arr[:n]]
        if i["repr"] == "cubes":
            return [(int(b.lo), int(b.hi), int(b.mask), int(b.value)) for b in arr[:n]]
        return [(int(b.lo), int(b.hi)) for b in arr[:n]]

    def contains(self, code):
        return bool(_check(_call("tr_preimg_set_contains", self._open(), int(code)), what="membership"))

    @property
    def lookup(self):
        """the lookup bits of a lookup set (a uint64 array; key = code mod modulus, or code - lo)"""
        i = self.info
        if i["mode"] != "lookup":
            raise ValueError("not a lookup set")
        out = np.zeros(max((i["nlookup"] + 63) // 64, 1), dtype=np.uint64)
        _check(_call("tr_preimg_set_lookup", self._open(), out.ctypes.data, len(out)), what="lookup")
        return out

    @property
    def certificates(self):
        """the FND-007 law of the transformation ([] when its coefficients pass int64 or the map has none)"""
        from . import math_certificates as mc
        h = _call("tr_preimg_set_certificate", self._open())
        if not h:
            return []
        names = {"poly": ["code"], "polymod": ["r", "t", "m"], "affmod": ["r", "t"]}.get(self.map.kind, ["x"])
        c = mc.Certificate._wrap(h, names, owned=False)
        c._parent = self
        return [c]

    def view(self):
        v = _View()
        _check(_call("tr_preimg_set_get_view", self._open(), C.byref(v)), what="preimage view")
        return v

    def scan(self, scope, column, *, direct=False, backend="cpu", budget=None):
        """bitmap of the selected rows whose present value's code is in the set; direct: also evaluate the original
        predicate on every row and count mismatches (PRE-001's statistics)"""
        if column not in scope.names:
            raise ValueError(f"{column!r} is not a column of the scope")
        if getattr(scope, "source", None) == "table":
            scope.require_fresh("preimage scan")
        col = scope.names.index(column)
        nw = (scope.rows + 63) // 64
        bits = np.zeros(max(nw, 1), dtype=np.uint64)
        st, refusal = _mp._ScanStats(), _Refusal()
        braw = (budget or Budget()).raw
        if backend == "cpu":
            rc = _call("tr_preimg_scan", self._open(), scope._handle, col, int(bool(direct)), bits.ctypes.data,
                       C.byref(st), C.byref(braw), C.byref(refusal))
            label = ("cpu", "native CPU")
        elif backend == "cuda":
            import os
            g = _gpu()
            if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
                g.require_device()
            v, sv = self.view(), scope.view()
            rc = g.lib().tr_mgpu_preimg_scan(C.byref(v), C.byref(sv), col, int(bool(direct)), bits.ctypes.data,
                                             C.byref(st), C.byref(braw), C.byref(refusal))
            d = g.device()
            label = ("cuda-emulated" if d["emulated"] else "cuda", d["name"])
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        stats = _mp._stats(st)
        if rc == -3:
            err = OverflowError(f"scan: {stats['overflow']} rows the direct evaluation could not decide")
            err.stats = stats
            raise err
        if rc == -12:
            raise UnsupportedError("the predicate has no proven direct evaluation (its bound passes 512 bits)")
        if rc and backend == "cuda":
            from . import math_gpu
            math_gpu.check(rc, refusal, "scan")
        _check(rc, refusal, "scan")
        stats.update(backend=label[0], device=label[1])
        return bits[:nw], stats

    def verify(self, *, backend="cpu", budget=None):
        """every code of the domain: membership against the original predicate (codes, members, truths, mismatches,
        first: the smallest mismatching code)"""
        out, refusal = _Verification(), _Refusal()
        braw = (budget or Budget()).raw
        if backend == "cpu":
            rc = _call("tr_preimg_verify", self._open(), C.byref(braw), C.byref(out), C.byref(refusal))
            label = ("cpu", "native CPU")
        elif backend == "cuda":
            g = _gpu()
            import os
            if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
                g.require_device()
            v = self.view()
            rc = g.lib().tr_mgpu_preimg_verify(C.byref(v), C.byref(braw), C.byref(out), C.byref(refusal))
            d = g.device()
            label = ("cuda-emulated" if d["emulated"] else "cuda", d["name"])
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        if rc == -12:
            raise UnsupportedError("verification needs a domain of at most 2^26 codes with values within int64 and a "
                                   "direct evaluation")
        _check(rc, refusal, "verification")
        return dict(codes=int(out.codes), members=int(out.members), truths=int(out.truths),
                    mismatches=int(out.mismatches), first=int(out.first) if out.mismatches else None, lo=int(out.lo),
                    hi=int(out.hi), backend=label[0], device=label[1])

    def cubes(self, width):
        """the set as cubes (lo, hi, mask, value) on the code bits below width (cubes; residues of a power-of-two
        period as low-bit cubes)"""
        cnt = C.c_uint64()
        rc = _call("tr_preimg_set_cubes", self._open(), int(width), None, 0, C.byref(cnt))
        if rc == -12:
            raise UnsupportedError("only cubes and residues of a power-of-two period have a cube form")
        n = int(cnt.value)
        arr = (_Branch * max(n, 1))()
        _check(_call("tr_preimg_set_cubes", self._open(), int(width), arr, n, C.byref(cnt)), what="cubes")
        return [(int(b.lo), int(b.hi), int(b.mask), int(b.value)) for b in arr[:n]]

    def plane_program(self, table, column):
        """the engine's bit-plane program (algq PInstr): intervals as an OR of BETWEENs; cubes and residues mod 2^k as
        an OR of ANDs of bit literals (within the domain); AND the column's validity. A lookup, a direct set and residues
        of another period have no plane form (UnsupportedError: the lookup or the direct evaluation is the candidate)."""
        i = self.info
        if i["mode"] != "branches":
            raise UnsupportedError(f"a {i['mode']} set has no plane program")
        P_ = algq.P
        c = table.idx(column)
        width = int(table.width(column))
        prog = []
        if i["repr"] == "intervals":
            spans = self.branches
            if not spans:
                prog.append(algq.PInstr(P_["LT"], c, 0, 0))                # code < 0: no row
            for k, (lo, hi) in enumerate(spans):
                prog.append(algq.PInstr(P_["BETWEEN"], c, int(lo), int(hi)))
                if k:
                    prog.append(algq.PInstr(P_["OR"], 0, 0, 0))
        else:
            cubes = self.cubes(width)
            if not cubes:
                prog.append(algq.PInstr(P_["LT"], c, 0, 0))
            for k, (lo, hi, mask, value) in enumerate(cubes):
                bits = [j for j in range(width) if (mask >> j) & 1]
                if not bits:
                    prog.append(algq.PInstr(P_["TRUE"], 0, 0, 0))
                for q, j in enumerate(bits):
                    prog.append(algq.PInstr(P_["BIT"], c, j, 0))
                    if not (value >> j) & 1:
                        prog.append(algq.PInstr(P_["NOT"], 0, 0, 0))
                    if q:
                        prog.append(algq.PInstr(P_["AND"], 0, 0, 0))
                if k:
                    prog.append(algq.PInstr(P_["OR"], 0, 0, 0))
            lo, hi = self.codec.lo, self.codec.hi
            if lo > 0 or hi < (1 << width) - 1:                             # the domain restriction
                prog.append(algq.PInstr(P_["BETWEEN"], c, int(lo), int(hi)))
                prog.append(algq.PInstr(P_["AND"], 0, 0, 0))
        valid = getattr(table, "nullable", {}).get(column)
        if valid is not None:
            prog.append(algq.PInstr(P_["EQ"], table.idx(valid), 1, 0))
            prog.append(algq.PInstr(P_["AND"], 0, 0, 0))
        if len(prog) > 1024:
            raise UnsupportedError(f"the plane program has {len(prog)} instructions (the engine runs 1024)")
        return prog

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_preimg_set_free", self._handle)
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
        return (f"PreimageSet({self.map!r} {self.op} ..., {i['mode']}" + (f" {i['repr']}" if i["repr"] else "") +
                f", branches={i['branches']}, codes={i['codes']})")


def preimage(codec, mp, op, c=None, *, values=None, fallback="auto", budget=None):
    """the exact set of codes whose value satisfies f(x) op c (op <, <=, =, <>, >=, >) or f(x) in values ("in");
    past the output (branches), iterations or bytes budget the set falls back (fallback auto / lookup / direct /
    none: refused)"""
    if op not in _mp.OPS:
        raise ValueError(f"op is one of {', '.join(_mp.OPS)}")
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback is one of {', '.join(FALLBACKS)}")
    if not isinstance(mp, Map):
        raise TypeError("a math_preimg.Map (poly, polymod, powmod, affmod, xor, bitand, bitor)")
    pr = _mp._Predicate()
    pr.op = _mp.OPS[op]
    keep = None
    if op == "in":
        vals = list(values if values is not None else (c if c is not None else []))
        if len(vals) > MAX_VALUES:
            raise UnsupportedError(f"at most {MAX_VALUES} values")
        keep = (_Q * max(len(vals), 1))(*[_mp._q(v, "value") for v in vals])
        pr.nvalues, pr.values = len(vals), C.cast(keep, C.c_void_p)
        pr.c = _Q(0, 1)
    else:
        if c is None:
            raise ValueError("a threshold is needed")
        pr.c = _mp._q(c, "threshold")
    raw, arr = mp.native()
    spec = _Spec(FALLBACKS[fallback], 0)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_preimg_preimage", C.byref(codec.raw), C.byref(raw), C.byref(pr), C.byref(spec),
              C.byref((budget or Budget()).raw), C.byref(refusal), C.byref(error))
    _check(error.value, refusal, "preimage")
    if not h:
        raise MemoryError("preimage: native allocation failed")
    return PreimageSet._wrap(h, codec, mp, op)
