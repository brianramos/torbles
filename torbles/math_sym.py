# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SYM-001 / SYM-002: closed-basis transformation actions on retained moment states, and certified data symmetries
with orbit reductions, natively on the CPU or with the CUDA kernels (torbles/math_sym.c, math_sym_gpu.cu; every
contract in math_sym.h).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_sym as ms
    scope = mbk.Scope.from_table(db["p"], ["x", "y", "c"], group_by=["k"])
    geo = ms.geometry(scope, numeric={"x": "m", "y": "m"}, categorical={"c": "color"})   # the only coordinates
    B = ms.basis(geo, degree=2)                          # every monomial of degree <= 2 over the numeric coordinates
    R = ms.rotate("x", "y", cos=Fraction(3, 5), sin=Fraction(4, 5))     # exact; degrees=30: certified intervals
    A = ms.action(B, ms.compose(R, ms.translate(x=Fraction(1, 2))), geo)   # M_{R o T}
    A.matrix(); A.compose(other); A.equal(other); ms.identity_action(B)
    st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)    # or a TENSOR-001 moment tensor
    st.transformed(A, g)                                 # moments of the transformed rows, exact (no rescan)
    st.query([dict(action=0, group=g, term="x^2", mean=True)], actions=[A], backend="cuda")
    G = ms.group(geo, [ms.rotate("x", "y", degrees=90), ms.reflect(x=0), ms.values("c", {0: 1, 1: 0})])
    cert = ms.certify(scope, geo, G)                     # SymmetryViolated (.evidence, .witness) otherwise
    orb = cert.orbits(scope)                             # representatives, sizes, multiplicities
    orb.query([ms.total({"x^2": 1, "y^2": 1}), ms.count({"x^2": 1, "y^2": 1}, "<=", 25)])
    ms.evaluate(scope, geo, [...])                       # the exhaustive original-domain evaluation

Values are exact: int / Fraction, intervals as (lo, hi) with lo <= exact <= hi (binary64 endpoints, certified).
Floating-point parameters are refused. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
import re
from collections import namedtuple
from decimal import Decimal
from fractions import Fraction

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_tensor as mt
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _Interval, _Q
from .math_certificates import StaleCertificateError, _Staleness, _freshness
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Geometry", "Transform", "Map", "Basis", "Action", "State", "Group", "Support", "Certificate", "Orbits",
           "Query", "SymmetryViolated", "NotInvariantError", "geometry", "abstract", "identity", "translate", "scale",
           "reflect", "permute", "affine", "rotate", "values", "compose", "basis", "action", "identity_action",
           "state", "group", "support", "certify", "total", "count", "evaluate", "MAX_COORDS", "MAX_DEGREE",
           "MAX_BASIS", "MAX_GENERATORS"]

MAX_COORDS, MAX_DEGREE, MAX_BASIS, MAX_GENERATORS, MAX_VALUES, MAX_TERMS, WORDS = 32, 8, 4096, 16, 65536, 4096, 8
NUMERIC, CATEGORICAL = 1, 2
T_KINDS = {"identity": 0, "translate": 1, "scale": 2, "reflect": 3, "permute": 4, "affine": 5, "rotate": 6, "values": 7}
EXACT, INTERVAL = 1, 2
CARRIERS = {1: "exact", 2: "interval"}
TERM, FUNCTIONAL, MEAN = 1, 2, 1
NULLS = {"forbid": 1, "skip": 2}
KINDS = {"sum": 1, "count": 2}
OPS = {"<": 1, "<=": 2, "==": 3, "!=": 4, ">=": 5, ">": 6}
WITNESS_KINDS = {0: None, 1: "missing", 2: "broken"}
BY_CPU, BY_DEVICE = 0, 1
NONE = (1 << 64) - 1
_NEED_CPU, _NEED_DEVICE = 0, 1
_I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER


class SymmetryViolated(DomainError):
    """The declared group is not a symmetry of the rows (no certificate): .evidence (every generator's), .witness
    (the smallest violating row), .kind ("missing" orbit member or "broken" multiplicity), .generator."""


class NotInvariantError(DomainError):
    """A query is not proven invariant under the certified group: refused under the orbit reduction; .generator."""


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
_I128 = mt._I128


class _Geometry(C.Structure):
    _native_ = "tr_sym_geometry"
    _fields_ = [("n", C.c_int32), ("ncols", C.c_int32)] + [(f, C.c_int32 * MAX_COORDS) for f in
                                                           ("col", "kind", "unit", "scale")] + [
        ("digest", C.c_uint64 * 2)]


class _Transform(C.Structure):
    _native_ = "tr_sym_transform"
    _fields_ = [("kind", C.c_int32), ("k", C.c_int32), ("col", C.c_int32 * MAX_COORDS),
                ("perm", C.c_int32 * MAX_COORDS), ("mode", C.c_int32), ("reserved", C.c_int32),
                ("value", _Q * MAX_COORDS), ("cosine", _Q), ("sine", _Q), ("degrees", _Q), ("center", _Q * 2),
                ("nvalues", C.c_uint64), ("A", C.c_void_p), ("b", C.c_void_p), ("keys", C.c_void_p),
                ("targets", C.c_void_p)]


class _Closure(C.Structure):
    _native_ = "tr_sym_closure"
    _fields_ = [("term", C.c_int32), ("closed", C.c_int32), ("reserved", C.c_int32 * 2),
                ("missing", C.c_uint8 * MAX_COORDS)]


class _MapInfo(C.Structure):
    _native_ = "tr_sym_map_info"
    _fields_ = [(n, C.c_int32) for n in ("n", "kind", "carrier", "integral", "permutation", "invertible", "coord",
                                         "precision")] + [("touched", C.c_uint32), ("reserved", C.c_uint32),
                                                          ("nvalues", C.c_uint64), ("bytes", C.c_uint64)]


class _BasisInfo(C.Structure):
    _native_ = "tr_sym_basis_info"
    _fields_ = [(n, C.c_int32) for n in ("n", "nb", "degree", "constant", "homogeneous", "reserved")] + [
        ("coords", C.c_uint32), ("reserved2", C.c_uint32), ("bytes", C.c_uint64)]


class _ActionInfo(C.Structure):
    _native_ = "tr_sym_action_info"
    _fields_ = [(n, C.c_int32) for n in ("nb", "n", "carrier", "permutation", "monomial", "identity", "precision",
                                         "approximate")] + [("den", _I128), ("nonzeros", C.c_uint64),
                                                            ("bytes", C.c_uint64)]


class _ActionView(C.Structure):
    _native_ = "tr_sym_action_view"
    _fields_ = [(n, C.c_int32) for n in ("carrier", "permutation", "nb", "reserved")] + [
        ("den", _I128), ("num", C.c_void_p), ("iv", C.c_void_p), ("index", C.c_void_p)]


class _StateInfo(C.Structure):
    _native_ = "tr_sym_state_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "present", "rows", "used", "bytes")] + [
        (n, C.c_int32) for n in ("nb", "n", "source", "weighted", "constant", "reserved")] + [
        ("digest", C.c_uint64 * 2)]


class _QueryView(C.Structure):
    _native_ = "tr_sym_query_view"
    _fields_ = [("ngroups", C.c_uint64)] + [(n, C.c_int32) for n in ("nb", "nactions", "nfunc", "constant")] + [
        ("moments", C.c_void_p), ("actions", C.c_void_p), ("functionals", C.c_void_p)]


class _Request(C.Structure):
    _native_ = "tr_sym_request"
    _fields_ = [(n, C.c_int32) for n in ("action", "kind", "index", "flags")] + [("group", C.c_uint64)]


class _Wide(C.Structure):
    _native_ = "tr_sym_wide"
    _fields_ = [("w", C.c_uint64 * WORDS)]


class _Result(C.Structure):
    _native_ = "tr_sym_result"
    _fields_ = [(n, C.c_int32) for n in ("status", "empty", "carrier", "reserved")] + [
        ("num", _Wide), ("den", _Wide), ("iv", _Interval)]


class _Gen(C.Structure):
    _native_ = "tr_sym_gen"
    _fields_ = [("kind", C.c_int32), ("coord", C.c_int32), ("at", C.c_uint64), ("nvalues", C.c_uint64)]


class _GroupView(C.Structure):
    _native_ = "tr_sym_group_view"
    _fields_ = [("ngen", C.c_int32), ("n", C.c_int32)] + [(n, C.c_void_p) for n in ("gens", "coef", "keys", "targets")]


class _GroupInfo(C.Structure):
    _native_ = "tr_sym_group_info"
    _fields_ = [("ngen", C.c_int32), ("n", C.c_int32)] + [(n, C.c_uint64) for n in ("ncoef", "nvalues", "bytes")]


class _SupportView(C.Structure):
    _native_ = "tr_sym_support_view"
    _fields_ = [("distinct", C.c_uint64), ("n", C.c_int32), ("reserved", C.c_int32)] + [
        (n, C.c_void_p) for n in ("tuples", "mult", "first")]


class _SupportInfo(C.Structure):
    _native_ = "tr_sym_support_info"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "checked", "excluded", "distinct", "bytes")] + [
        ("n", C.c_int32), ("policy", C.c_int32)]


class _Evidence(C.Structure):
    _native_ = "tr_sym_evidence"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "tuples", "violations", "missing", "broken", "witness",
                                          "witness_tuple", "image_mult")] + [
        (n, C.c_int32) for n in ("generator", "kind", "holds", "reserved")]


class _CertInfo(C.Structure):
    _native_ = "tr_sym_cert_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "backend", "ngen", "n", "policy", "holds")] + [
        (n, C.c_uint64) for n in ("rows", "selected", "checked", "excluded", "distinct", "bytes")]


class _OrbitsInfo(C.Structure):
    _native_ = "tr_sym_orbits_info"
    _fields_ = [(n, C.c_uint64) for n in ("norbits", "distinct", "rows", "largest", "smallest", "bytes")] + [
        ("n", C.c_int32), ("ngen", C.c_int32)]


class _OrbitsView(C.Structure):
    _native_ = "tr_sym_orbits_view"
    _fields_ = [("norbits", C.c_uint64), ("distinct", C.c_uint64), ("n", C.c_int32), ("reserved", C.c_int32)] + [
        (n, C.c_void_p) for n in ("rep", "size", "mult", "tuples", "orbit")]


class _PolyQuery(C.Structure):
    _native_ = "tr_sym_poly_query"
    _fields_ = [(n, C.c_int32) for n in ("kind", "op", "nterms", "reserved")] + [
        ("constant", C.c_int64), ("exponents", C.c_void_p), ("coefficients", C.c_void_p)]


class _PolyAnswer(C.Structure):
    _native_ = "tr_sym_poly_answer"
    _fields_ = [(n, C.c_int32) for n in ("status", "invariant", "generator", "reserved")] + [
        ("evaluations", C.c_uint64), ("rows", C.c_uint64), ("value", _Wide)]


for _s, _n in ((_Geometry, 536), (_Transform, 904), (_Closure, 48), (_MapInfo, 56), (_BasisInfo, 40),
               (_ActionInfo, 64), (_ActionView, 56), (_StateInfo, 80), (_QueryView, 48), (_Request, 24), (_Wide, 64),
               (_Result, 160), (_Gen, 24), (_GroupView, 40), (_GroupInfo, 32), (_SupportView, 40),
               (_SupportInfo, 56), (_Evidence, 80), (_CertInfo, 72), (_OrbitsInfo, 56), (_OrbitsView, 64),
               (_PolyQuery, 40), (_PolyAnswer, 96)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u32, u64, e, R, B = C.c_void_p, C.c_int32, C.c_uint32, C.c_uint64, P(C.c_int32), P(_Refusal), P(_Budget)
        pi = P(C.c_int32)
        sig = {
            "tr_sym_geometry_make": (i32, [vp, pi, pi, pi, i32, P(_Geometry)]),
            "tr_sym_geometry_abstract": (i32, [i32, pi, pi, pi, P(_Geometry)]),
            "tr_sym_basis_create": (vp, [i32, vp, i32, B, R, e]),
            "tr_sym_basis_full": (vp, [i32, u32, i32, i32, B, R, e]),
            "tr_sym_basis_need": (i32, [i32, i32, i32, P(_Need)]),
            "tr_sym_basis_get_info": (i32, [vp, P(_BasisInfo)]),
            "tr_sym_basis_get_exponents": (i32, [vp, vp, u64]),
            "tr_sym_basis_find": (i32, [vp, vp]),
            "tr_sym_basis_check": (i32, [vp, P(_Geometry)]),
            "tr_sym_basis_equal": (i32, [vp, vp]),
            "tr_sym_basis_clone": (vp, [vp, e]),
            "tr_sym_basis_free": (None, [vp]),
            "tr_sym_map_make": (vp, [P(_Geometry), P(_Transform), B, R, e]),
            "tr_sym_map_identity": (vp, [i32, e]),
            "tr_sym_map_compose": (vp, [vp, vp, B, R, e]),
            "tr_sym_map_get_info": (i32, [vp, P(_MapInfo)]),
            "tr_sym_map_get_exact": (i32, [vp, vp, vp, vp]),
            "tr_sym_map_get_interval": (i32, [vp, vp, vp]),
            "tr_sym_map_get_values": (i32, [vp, vp, vp, u64]),
            "tr_sym_map_free": (None, [vp]),
            "tr_sym_action_need": (i32, [vp, vp, P(_Need)]),
            "tr_sym_action_make": (vp, [vp, vp, B, R, P(_Closure), e]),
            "tr_sym_action_identity": (vp, [vp, B, R, e]),
            "tr_sym_action_compose": (vp, [vp, vp, B, R, e]),
            "tr_sym_action_equal": (i32, [vp, vp]),
            "tr_sym_action_get_info": (i32, [vp, P(_ActionInfo)]),
            "tr_sym_action_get_view": (i32, [vp, P(_ActionView)]),
            "tr_sym_action_get_basis": (vp, [vp]),
            "tr_sym_action_free": (None, [vp]),
            "tr_sym_state_from_bank": (vp, [vp, P(_Geometry), B, R, e]),
            "tr_sym_state_from_tensor": (vp, [vp, P(_Geometry), B, R, e]),
            "tr_sym_state_get_info": (i32, [vp, P(_StateInfo)]),
            "tr_sym_state_get_basis": (vp, [vp]),
            "tr_sym_state_get_view": (i32, [vp, P(_QueryView)]),
            "tr_sym_state_get_rows": (i32, [vp, vp, vp, u64]),
            "tr_sym_state_free": (None, [vp]),
            "tr_sym_query_need": (i32, [P(_QueryView), u64, i32, P(_Need)]),
            "tr_sym_query_view_eval": (i32, [P(_QueryView), vp, u64, vp, B, R]),
            "tr_sym_query_bind": (i32, [vp, vp, i32, vp, i32, vp, P(_QueryView)]),
            "tr_sym_query": (i32, [vp, vp, i32, vp, i32, vp, u64, vp, B, R]),
            "tr_sym_group_create": (vp, [vp, i32, B, R, e]),
            "tr_sym_group_get_info": (i32, [vp, P(_GroupInfo)]),
            "tr_sym_group_get_view": (i32, [vp, P(_GroupView)]),
            "tr_sym_group_free": (None, [vp]),
            "tr_sym_support_build": (vp, [vp, P(_Geometry), i32, B, R, P(u64), e]),
            "tr_sym_support_need": (i32, [vp, P(_Geometry), P(_Need)]),
            "tr_sym_support_get_info": (i32, [vp, P(_SupportInfo)]),
            "tr_sym_support_get_view": (i32, [vp, P(_SupportView)]),
            "tr_sym_support_fresh": (i32, [vp, vp, P(_Staleness)]),
            "tr_sym_support_free": (None, [vp]),
            "tr_sym_verify_need": (i32, [P(_SupportView), P(_GroupView), u64, u64, i32, P(_Need)]),
            "tr_sym_verify_view": (i32, [P(_SupportView), P(_GroupView), u64, u64, B, vp, R]),
            "tr_sym_certify": (vp, [vp, vp, B, R, vp, e]),
            "tr_sym_adopt": (vp, [vp, vp, vp, i32, B, R, vp, e]),
            "tr_sym_cert_get_info": (i32, [vp, P(_CertInfo)]),
            "tr_sym_cert_get_evidence": (i32, [vp, vp, i32]),
            "tr_sym_cert_fresh": (i32, [vp, vp, P(_Staleness)]),
            "tr_sym_cert_free": (None, [vp]),
            "tr_sym_orbits_build": (vp, [vp, vp, vp, B, R, P(_Staleness), e]),
            "tr_sym_orbits_get_info": (i32, [vp, P(_OrbitsInfo)]),
            "tr_sym_orbits_get_view": (i32, [vp, P(_OrbitsView)]),
            "tr_sym_orbits_free": (None, [vp]),
            "tr_sym_poly_invariant": (i32, [vp, P(_PolyQuery), pi, B, R]),
            "tr_sym_orbit_query": (i32, [vp, vp, vp, i32, vp, B, R, P(_Staleness)]),
            "tr_sym_rows_query": (i32, [vp, P(_Geometry), i32, vp, i32, vp, B, R]),
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
    math_gpu.bind("tr_mgpu_sym_query", C.c_int, [P(_QueryView), C.c_void_p, C.c_uint64, C.c_void_p, P(_Budget),
                                                P(_Refusal)])
    math_gpu.bind("tr_mgpu_sym_verify", C.c_int, [P(_SupportView), P(_GroupView), C.c_uint64, C.c_uint64, P(_Budget),
                                                 C.c_void_p, P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _precheck(budget, raw_need, what):
    refusal = _Refusal()
    _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw_need), C.byref(refusal)), refusal, what)


def _int(lo, hi):
    return (int(hi) << 64) | (int(lo) & ((1 << 64) - 1))


def _wide(w):
    """the exact int of a 512-bit two's complement word array"""
    v = 0
    for i in reversed(range(WORDS)):
        v = (v << 64) | int(w.w[i])
    return v - (1 << (64 * WORDS)) if v >> (64 * WORDS - 1) else v


def _exact(v, what="a value"):
    """an exact rational of an int, Fraction, finite Decimal or decimal string (floats refused)"""
    if isinstance(v, bool):
        raise TypeError(f"{what} must be exact, not bool")
    if isinstance(v, (int, np.integer)):
        f = Fraction(int(v))
    elif isinstance(v, Fraction):
        f = v
    elif isinstance(v, Decimal):
        if not v.is_finite():
            raise ValueError(f"{what} must be finite")
        f = Fraction(v)
    elif isinstance(v, str):
        f = Fraction(v)
    else:
        raise TypeError(f"{what} must be exact (int, Fraction, Decimal or a decimal string), not {type(v).__name__}")
    return f


def _q(v, what="a value"):
    f = _exact(v, what)
    if not (_I64[0] <= f.numerator <= _I64[1] and f.denominator <= _I64[1]):
        raise OverflowError(f"{what} needs an int64 numerator and denominator")
    return _Q(f.numerator, f.denominator)


def _result_value(num, den):
    f = Fraction(num, den)
    return f.numerator if f.denominator == 1 else f


# ---------------------------------------------------------------------- geometry
class Geometry:
    """The declared coordinates: names (in order), their scope columns, kinds and unit labels (native struct)."""

    def __init__(self, raw, scope, names, kinds, units):
        self.raw, self.scope, self.names = raw, scope, tuple(names)
        self.kinds, self.units = tuple(kinds), tuple(units)

    @property
    def n(self):
        return len(self.names)

    def index(self, name):
        if name not in self.names:
            raise ValueError(f"{name!r} is not a declared coordinate")
        return self.names.index(name)

    def column(self, name):
        """the scope column a transformation names (any scope column: the native code refuses undeclared ones)"""
        cols = self.scope.names if self.scope is not None else self.names
        if name not in cols:
            raise ValueError(f"{name!r} is not a column of the scope")
        return cols.index(name)

    @property
    def scales(self):
        return tuple(int(self.raw.scale[i]) for i in range(self.n))

    def __repr__(self):
        return "Geometry(" + ", ".join(f"{n}: {k} [{u}]" for n, k, u in zip(self.names, self.kinds, self.units)) + ")"


def _unit_ids(labels):
    ids = {}
    for u in labels:
        ids.setdefault(u, len(ids))
    return ids


def geometry(scope, numeric=None, categorical=None):
    """declares coordinates over a FND-004 scope: numeric {column: unit label} (exact decimal values, the scale read
    from the scope), categorical {column: domain label} (codes with no metric). A code column (a dictionary or
    foreign-key code) cannot be declared numeric (DomainError)."""
    numeric, categorical = dict(numeric or {}), dict(categorical or {})
    names = list(numeric) + list(categorical)
    if not names or len(set(names)) != len(names):
        raise ValueError("declare at least one coordinate, each once")
    if len(names) > MAX_COORDS:
        raise UnsupportedError(f"at most {MAX_COORDS} coordinates")
    table = getattr(scope, "table", None)
    for name in numeric:
        if name not in scope.names:
            raise ValueError(f"{name!r} is not a scope column")
        codec = getattr(table, "codecs", {}).get(name) if table is not None else None
        if codec is not None and getattr(codec, "affine", None) is None:
            raise DomainError(f"{name!r} holds codes ({codec.kind}): code order has no metric; declare it categorical")
    for name in categorical:
        if name not in scope.names:
            raise ValueError(f"{name!r} is not a scope column")
    labels = [("n", str(numeric[n])) for n in numeric] + [("c", str(categorical[n])) for n in categorical]
    ids = _unit_ids(labels)
    k = len(names)
    cols = (C.c_int32 * k)(*[scope.names.index(n) for n in names])
    kinds = (C.c_int32 * k)(*([NUMERIC] * len(numeric) + [CATEGORICAL] * len(categorical)))
    units = (C.c_int32 * k)(*[ids[l] for l in labels])
    raw = _Geometry()
    _check(_call("tr_sym_geometry_make", scope._handle, cols, kinds, units, k, C.byref(raw)), what="geometry")
    return Geometry(raw, scope, names, ["numeric"] * len(numeric) + ["categorical"] * len(categorical),
                    [l for _, l in labels])


def abstract(names, units=None, scales=None, categorical=()):
    """a geometry without data (DYN-001, action matrices alone): coordinate names, unit labels, decimal scales"""
    names = list(names)
    k = len(names)
    if not k or len(set(names)) != k:
        raise ValueError("declare at least one coordinate, each once")
    if k > MAX_COORDS:
        raise UnsupportedError(f"at most {MAX_COORDS} coordinates")
    units = dict(units or {})
    scales = dict(scales or {})
    cat = set(categorical)
    labels = [("c" if n in cat else "n", str(units.get(n, "1"))) for n in names]
    ids = _unit_ids(labels)
    raw = _Geometry()
    _check(_call("tr_sym_geometry_abstract", k, (C.c_int32 * k)(*[CATEGORICAL if n in cat else NUMERIC for n in names]),
                 (C.c_int32 * k)(*[ids[l] for l in labels]), (C.c_int32 * k)(*[int(scales.get(n, 0)) for n in names]),
                 C.byref(raw)), what="geometry")
    return Geometry(raw, None, names, ["categorical" if n in cat else "numeric" for n in names], [l for _, l in labels])


# ---------------------------------------------------------------------- transformations and maps
class Transform:
    """A declared transformation in value units (exact rationals); .map(geometry) makes the native map."""

    def __init__(self, kind, **p):
        self.kind, self.p = kind, p

    def native(self, geo):
        t = _Transform()
        t.kind = T_KINDS[self.kind]
        keep = []
        p = self.p
        cols = list(p.get("columns", ()))
        if len(cols) > MAX_COORDS:
            raise UnsupportedError(f"at most {MAX_COORDS} coordinates")
        t.k = len(cols)
        for i, name in enumerate(cols):
            t.col[i] = geo.column(name)
        if self.kind in ("translate", "scale", "reflect"):
            for i, v in enumerate(p["values"]):
                t.value[i] = _q(v, f"{self.kind} of {cols[i]!r}")
        elif self.kind == "permute":
            for i, s in enumerate(p["sources"]):
                t.perm[i] = cols.index(s)
        elif self.kind == "affine":
            k = len(cols)
            A = [list(r) for r in p["A"]]
            if len(A) != k or any(len(r) != k for r in A) or len(p["b"]) != k:
                raise ValueError(f"A is {k} x {k} and b has {k} entries")
            Aa = (_Q * max(k * k, 1))(*[_q(x, "an affine coefficient") for r in A for x in r])
            ba = (_Q * max(k, 1))(*[_q(x, "an affine offset") for x in p["b"]])
            keep += [Aa, ba]
            t.A, t.b = C.cast(Aa, C.c_void_p), C.cast(ba, C.c_void_p)
        elif self.kind == "rotate":
            if p.get("degrees") is not None:
                t.mode, t.degrees = 2, _q(p["degrees"], "degrees")
            else:
                t.mode, t.cosine, t.sine = 1, _q(p["cos"], "cos"), _q(p["sin"], "sin")
            t.center[0], t.center[1] = _q(p["center"][0], "the center"), _q(p["center"][1], "the center")
        elif self.kind == "values":
            pairs = list(p["mapping"].items())
            if len(pairs) > MAX_VALUES:
                raise UnsupportedError(f"at most {MAX_VALUES} values")
            ka = (_Q * max(len(pairs), 1))(*[_q(a, "a value") for a, _ in pairs])
            ta = (_Q * max(len(pairs), 1))(*[_q(b, "a value") for _, b in pairs])
            keep += [ka, ta]
            t.nvalues, t.keys, t.targets = len(pairs), C.cast(ka, C.c_void_p), C.cast(ta, C.c_void_p)
        return t, keep

    def map(self, geo, budget=None):
        if self.kind == "compose":
            parts = [x.map(geo, budget) for x in self.p["parts"]]
            m = parts[-1]
            for x in reversed(parts[:-1]):
                m = x.after(m, budget)
            return m
        t, keep = self.native(geo)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_sym_map_make", C.byref(geo.raw), C.byref(t), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Map(_made(h, error, refusal, f"{self.kind} map"), geo)

    def __repr__(self):
        return f"Transform({self.kind}, {self.p})"


def identity():
    return Transform("identity")


def _named(mapping, kw, what):
    d = dict(mapping or {})
    d.update(kw)
    if not d:
        raise ValueError(f"{what}: name at least one coordinate")
    return d


def translate(offsets=None, **kw):
    """x_c += t_c (value units)"""
    d = _named(offsets, kw, "translate")
    return Transform("translate", columns=list(d), values=list(d.values()))


def scale(factors=None, **kw):
    """x_c *= f_c"""
    d = _named(factors, kw, "scale")
    return Transform("scale", columns=list(d), values=list(d.values()))


def reflect(centers=None, **kw):
    """x_c -> 2 a_c - x_c (the reflection about the point a_c)"""
    d = _named(centers, kw, "reflect")
    return Transform("reflect", columns=list(d), values=list(d.values()))


def permute(mapping=None, **kw):
    """x'_target = x_source for {target: source} (a permutation of the named coordinates)"""
    d = _named(mapping, kw, "permute")
    if sorted(d) != sorted(d.values()):
        raise ValueError("permute: the sources are a permutation of the targets")
    return Transform("permute", columns=list(d), sources=list(d.values()))


def affine(columns, A, b):
    """x'_C = A x_C + b over the named coordinates (value units; the others fixed)"""
    return Transform("affine", columns=list(columns), A=A, b=list(b))


def rotate(x, y, *, cos=None, sin=None, degrees=None, center=(0, 0)):
    """the rotation of the (x, y) plane about center: exact for a rational point (cos, sin) of the unit circle or a
    multiple of 90 degrees; any other angle in degrees gives certified interval enclosures"""
    if (degrees is None) == (cos is None or sin is None):
        raise ValueError("rotate: give cos= and sin=, or degrees=")
    return Transform("rotate", columns=[x, y], cos=cos, sin=sin, degrees=degrees, center=tuple(center))


def values(column, mapping):
    """a permutation of finitely many values of one coordinate (value units; every other value fixed): SYM-002"""
    return Transform("values", columns=[column], mapping=dict(mapping))


def compose(*transforms):
    """the composition: compose(T, S)(x) = T(S(x)) (S first)"""
    if not transforms:
        raise ValueError("compose at least one transformation")
    return Transform("compose", parts=list(transforms))


class Map:
    """A transformation in logical units (native): exact rows (a x + b) / q, interval rows, or a value permutation."""

    def __init__(self, handle, geo):
        self._handle, self.geometry = handle, geo

    def _open(self):
        if not self._handle:
            raise ValueError("the map is closed")
        return self._handle

    @property
    def info(self):
        i = _MapInfo()
        _check(_call("tr_sym_map_get_info", self._open(), C.byref(i)), what="map")
        return dict(n=int(i.n), kind={1: "affine", 2: "values"}[i.kind], carrier=CARRIERS[i.carrier],
                    integral=bool(i.integral), permutation=bool(i.permutation),
                    invertible={1: True, 0: False, -1: None}[i.invertible], precision=int(i.precision),
                    touched=[k for k in range(i.n) if i.touched >> k & 1], values=int(i.nvalues), bytes=int(i.bytes))

    def after(self, other, budget=None):
        """self o other (other first), natively"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_sym_map_compose", self._open(), other._open(), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Map(_made(h, error, refusal, "map composition"), self.geometry or other.geometry)

    def exact(self):
        """(A, b) in logical units as Fractions: x' = A x + b"""
        n = self.info["n"]
        a, b, q = (_I128 * (n * n))(), (_I128 * n)(), (_I128 * n)()
        _check(_call("tr_sym_map_get_exact", self._open(), a, b, q), what="exact map")
        qs = [_int(q[i].lo, q[i].hi) for i in range(n)]
        return ([[Fraction(_int(a[i * n + j].lo, a[i * n + j].hi), qs[i]) for j in range(n)] for i in range(n)],
                [Fraction(_int(b[i].lo, b[i].hi), qs[i]) for i in range(n)])

    def interval(self):
        n = self.info["n"]
        a, b = (_Interval * (n * n))(), (_Interval * n)()
        _check(_call("tr_sym_map_get_interval", self._open(), a, b), what="interval map")
        return [[(a[i * n + j].lo, a[i * n + j].hi) for j in range(n)] for i in range(n)], [(x.lo, x.hi) for x in b]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_sym_map_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _as_map(t, geo, budget=None):
    if isinstance(t, Map):
        return t, False
    if geo is None:
        raise ValueError("a transformation needs the geometry it is declared in")
    return t.map(geo, budget), True


# ---------------------------------------------------------------------- bases
_TERM = re.compile(r"^\s*([A-Za-z_][\w#.]*)\s*(?:(?:\^|\*\*)\s*(\d+))?\s*$")


def _exponents(names, t):
    """an exponent tuple over names from '1' / 'x*y^2', {name: exponent} or a tuple"""
    names = list(names)
    e = [0] * len(names)
    if isinstance(t, str):
        if t.strip() != "1":
            for factor in re.split(r"(?<!\*)\*(?!\*)", t):
                m = _TERM.match(factor)
                if not m or m.group(1) not in names:
                    raise ValueError(f"term {t!r}: {factor.strip()!r} is not a coordinate")
                e[names.index(m.group(1))] += int(m.group(2) or 1)
    elif isinstance(t, dict):
        for name, k in t.items():
            if name not in names:
                raise ValueError(f"term {t!r}: {name!r} is not a coordinate")
            e[names.index(name)] += int(k)
    else:
        e = [int(k) for k in t]
        if len(e) != len(names):
            raise ValueError(f"term {t!r}: one exponent per coordinate ({len(names)})")
    if any(k < 0 or k > 255 for k in e):
        raise ValueError(f"term {t!r}: exponents are 0..255")
    return tuple(e)


class Basis:
    """A monomial basis over the coordinates (native), in order; terms are exponent tuples."""

    def __init__(self, handle, names):
        self._handle, self.names = handle, tuple(names)

    def _open(self):
        if not self._handle:
            raise ValueError("the basis is closed")
        return self._handle

    @property
    def info(self):
        i = _BasisInfo()
        _check(_call("tr_sym_basis_get_info", self._open(), C.byref(i)), what="basis")
        return dict(n=int(i.n), size=int(i.nb), degree=int(i.degree), constant=None if i.constant < 0 else int(i.constant),
                    homogeneous=bool(i.homogeneous), bytes=int(i.bytes))

    def __len__(self):
        return self.info["size"]

    @property
    def terms(self):
        i = self.info
        out = (C.c_uint8 * max(i["size"] * i["n"], 1))()
        _check(_call("tr_sym_basis_get_exponents", self._open(), out, i["size"] * i["n"]), what="basis terms")
        return [tuple(out[j * i["n"]:(j + 1) * i["n"]]) for j in range(i["size"])]

    def find(self, term):
        """the index of a term (a string, dict or exponent tuple), or None"""
        e = _exponents(self.names, term)
        if any(k > MAX_DEGREE for k in e):
            return None
        r = _call("tr_sym_basis_find", self._open(), (C.c_uint8 * len(e))(*e))
        return None if r < 0 else int(r)

    def index(self, term):
        if isinstance(term, (int, np.integer)) and not isinstance(term, bool):
            if not 0 <= int(term) < len(self):
                raise ValueError("term index out of range")
            return int(term)
        i = self.find(term)
        if i is None:
            raise ValueError(f"{term!r} is not a term of the basis")
        return i

    def terms_for(self, scope):
        """the basis terms as FND-004 bank terms over the scope's columns ({column: exponent})"""
        return [{n: k for n, k in zip(self.names, t) if k} for t in self.terms]

    def equal(self, other):
        return bool(_check(_call("tr_sym_basis_equal", self._open(), other._open()), what="basis comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_sym_basis_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Basis({len(self)} monomials over {list(self.names)})"


def basis(geo, *, degree=None, terms=None, columns=None, homogeneous=False, budget=None):
    """every monomial of total degree <= degree (homogeneous: exactly degree) over the numeric coordinates (or
    columns), graded in FND-004's order; or explicit terms (strings, dicts, exponent tuples)"""
    names = geo.names
    error, refusal = C.c_int32(), _Refusal()
    if terms is not None:
        rows = [_exponents(names, t) for t in terms]
        if len(rows) > MAX_BASIS:
            raise UnsupportedError(f"at most {MAX_BASIS} monomials")
        flat = (C.c_uint8 * max(len(rows) * len(names), 1))(*[k for r in rows for k in r])
        h = _call("tr_sym_basis_create", len(names), flat, len(rows), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
    else:
        if degree is None:
            raise ValueError("give degree= or terms=")
        cols = list(columns) if columns is not None else [n for n, k in zip(names, geo.kinds) if k == "numeric"]
        mask = 0
        for c in cols:
            mask |= 1 << geo.index(c)
        h = _call("tr_sym_basis_full", len(names), mask, int(degree), int(bool(homogeneous)), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
    b = Basis(_made(h, error, refusal, "basis"), names)
    _check(_call("tr_sym_basis_check", b._open(), C.byref(geo.raw)),
           what="basis (a monomial over a categorical coordinate has no metric meaning)")
    return b


# ---------------------------------------------------------------------- actions
class Action:
    """The exact (N / D) or interval matrix M_T of a map on a closed basis: phi(T x) = M_T phi(x) (native)."""

    def __init__(self, handle, basis_, geo):
        self._handle, self.basis, self.geometry = handle, basis_, geo

    def _open(self):
        if not self._handle:
            raise ValueError("the action is closed")
        return self._handle

    @property
    def info(self):
        i = _ActionInfo()
        _check(_call("tr_sym_action_get_info", self._open(), C.byref(i)), what="action")
        return dict(size=int(i.nb), n=int(i.n), carrier=CARRIERS[i.carrier], permutation=bool(i.permutation),
                    monomial=bool(i.monomial), identity=bool(i.identity), precision=int(i.precision),
                    approximate=bool(i.approximate), certified=True, den=_int(i.den.lo, i.den.hi),
                    nonzeros=int(i.nonzeros), bytes=int(i.bytes))

    def _view(self):
        v = _ActionView()
        _check(_call("tr_sym_action_get_view", self._open(), C.byref(v)), what="action view")
        return v

    def numerators(self):
        """EXACT: (N as rows of ints, D); M = N / D"""
        v = self._view()
        if v.carrier != EXACT:
            raise UnsupportedError("an interval action has no exact numerators")
        nb = int(v.nb)
        w = np.ctypeslib.as_array((C.c_int64 * (2 * nb * nb)).from_address(v.num)).reshape(nb, nb, 2)
        return [[_int(w[i, j, 0], w[i, j, 1]) for j in range(nb)] for i in range(nb)], _int(v.den.lo, v.den.hi)

    def matrix(self):
        """the matrix M_T: rows of Fractions (EXACT) or of (lo, hi) enclosures (INTERVAL)"""
        v = self._view()
        nb = int(v.nb)
        if v.carrier == EXACT:
            N, D = self.numerators()
            return [[_result_value(x, D) for x in row] for row in N]
        a = np.ctypeslib.as_array((C.c_double * (2 * nb * nb)).from_address(v.iv)).reshape(nb, nb, 2)
        return [[(float(a[i, j, 0]), float(a[i, j, 1])) for j in range(nb)] for i in range(nb)]

    @property
    def index(self):
        """a permutation action's index map ((M m)_i = m_index[i]), or None"""
        v = self._view()
        if not v.permutation:
            return None
        return list(np.ctypeslib.as_array((C.c_int32 * int(v.nb)).from_address(v.index)))

    def compose(self, other, budget=None):
        """M_self M_other: the action of self's map after other's"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_sym_action_compose", self._open(), other._open(), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Action(_made(h, error, refusal, "action composition"), self.basis, self.geometry or other.geometry)

    def equal(self, other):
        return bool(_check(_call("tr_sym_action_equal", self._open(), other._open()), what="action comparison"))

    def __eq__(self, other):
        return isinstance(other, Action) and self.equal(other)

    __hash__ = None

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_sym_action_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        i = self.info
        return f"Action({i['size']} x {i['size']}, {i['carrier']}{', permutation' if i['permutation'] else ''})"


class NotClosedError(DomainError):
    """The basis is not closed under the map: .term (the basis term) and .missing (a monomial of its image)."""


def action(basis_, transform, geo=None, budget=None):
    """M_T on the basis for a transformation (with its geometry) or a Map; NotClosedError when not closed"""
    m, own = _as_map(transform, geo, budget)
    try:
        error, refusal, closure = C.c_int32(), _Refusal(), _Closure()
        h = _call("tr_sym_action_make", basis_._open(), m._open(), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(closure), C.byref(error))
        if error.value == -6 and closure.term >= 0:
            names = basis_.names
            missing = {n: int(closure.missing[i]) for i, n in enumerate(names) if closure.missing[i]}
            err = NotClosedError(f"the basis is not closed: the image of term {basis_.terms[closure.term]} has "
                                 f"{missing or '1'} outside it")
            err.term, err.missing = basis_.terms[closure.term], missing
            raise err
        return Action(_made(h, error, refusal, "action"), basis_, m.geometry)
    finally:
        if own:
            m.close()


def identity_action(basis_, budget=None):
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_sym_action_identity", basis_._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Action(_made(h, error, refusal, "identity action"), basis_, None)


# ---------------------------------------------------------------------- states and requests
class State:
    """Retained per-group moments over a basis of the geometry's numeric coordinates (native)."""

    def __init__(self, handle, geo, source):
        self._handle, self.geometry, self.source = handle, geo, source
        error = C.c_int32()
        bh = _call("tr_sym_basis_clone", _call("tr_sym_state_get_basis", handle), C.byref(error))
        _check(error.value, what="state basis")
        self.basis = Basis(bh, geo.names)

    def _open(self):
        if not self._handle:
            raise ValueError("the state is closed")
        return self._handle

    @property
    def info(self):
        i = _StateInfo()
        _check(_call("tr_sym_state_get_info", self._open(), C.byref(i)), what="state")
        return dict(groups=int(i.ngroups), present=int(i.present), rows=int(i.rows), used=int(i.used),
                    bytes=int(i.bytes), size=int(i.nb), source={1: "bank", 2: "tensor"}[i.source],
                    weighted=bool(i.weighted), constant=None if i.constant < 0 else int(i.constant))

    def moments(self, g):
        v = _QueryView()
        _check(_call("tr_sym_state_get_view", self._open(), C.byref(v)), what="state view")
        g = int(g)
        if not 0 <= g < int(v.ngroups):
            raise ValueError("group id out of range")
        nb = int(v.nb)
        w = np.ctypeslib.as_array((C.c_int64 * (2 * nb)).from_address(v.moments + 16 * nb * g))
        return [_int(w[2 * j], w[2 * j + 1]) for j in range(nb)]

    def group(self, g):
        G = self.info["groups"]
        rows, used = (C.c_uint64 * G)(), (C.c_uint64 * G)()
        _check(_call("tr_sym_state_get_rows", self._open(), rows, used, G), what="state rows")
        g = int(g)
        return dict(present=rows[g] > 0, rows=int(rows[g]), used=int(used[g]))

    def _functionals(self, functionals):
        nb = len(self.basis)
        rows = []
        for f in functionals:
            row = [0] * nb
            items = f.items() if isinstance(f, dict) else enumerate(f)
            for term, c in items:
                if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
                    raise TypeError("functional coefficients are exact integers")
                if not _I64[0] <= int(c) <= _I64[1]:
                    raise OverflowError("a functional coefficient does not fit int64")
                row[self.basis.index(term)] += int(c)
            rows.append(row)
        if any(not _I64[0] <= x <= _I64[1] for r in rows for x in r):
            raise OverflowError("a functional coefficient does not fit int64")
        return np.ascontiguousarray(np.array(rows or [[0]], dtype=np.int64))

    def query(self, requests, *, actions=(), functionals=(), backend="cpu", budget=None, raw=False):
        """answers of request dicts: action (an index into actions, or None: the moments themselves), group, term (a
        basis term) or functional (an index into functionals: {term: int coefficient} or rows), mean (divide by the
        constant term's moment). [{"status", "value" (int / Fraction / (lo, hi) / None when empty), "carrier",
        "num", "den", "empty"}]; raw=False raises on a request's error"""
        requests = list(requests)
        acts = list(actions)
        arr = (_Request * max(len(requests), 1))()
        for i, q in enumerate(requests):
            r = arr[i]
            a = q.get("action")
            r.action = -1 if a is None else int(a)
            r.group = int(q.get("group", 0))
            if q.get("functional") is not None:
                r.kind, r.index = FUNCTIONAL, int(q["functional"])
            else:
                r.kind, r.index = TERM, self.basis.index(q["term"])
            r.flags = MEAN if q.get("mean") else 0
        F = self._functionals(list(functionals))
        handles = (C.c_void_p * max(len(acts), 1))(*[a._open() for a in acts])
        out = (_Result * max(len(requests), 1))()
        refusal = _Refusal()
        nf = len(list(functionals))
        if backend == "cpu":
            _check(_call("tr_sym_query", self._open(), handles, len(acts), F.ctypes.data, nf, arr, len(requests), out,
                         C.byref(_budget(budget)), C.byref(refusal)), refusal, "requests")
        elif backend == "cuda":
            gpu = _gpu()
            views = (_ActionView * max(len(acts), 1))()
            v = _QueryView()
            _check(_call("tr_sym_query_bind", self._open(), handles, len(acts), F.ctypes.data, nf, views, C.byref(v)),
                   what="requests (actions on the state's basis and geometry)")
            gpu.run("tr_mgpu_sym_query", C.byref(v), arr, len(requests), out, C.byref(_budget(budget)),
                    C.byref(refusal), refusal=refusal, what="requests on the device")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        answers = []
        for i in range(len(requests)):
            r = out[i]
            if r.status:
                if not raw:
                    _check(r.status, what=f"request {i}")
                answers.append(dict(status=int(r.status), value=None, carrier=None, num=None, den=None, empty=False))
                continue
            if r.empty:
                answers.append(dict(status=0, value=None, carrier=CARRIERS[r.carrier], num=None, den=None, empty=True))
            elif r.carrier == INTERVAL:
                answers.append(dict(status=0, value=(r.iv.lo, r.iv.hi), carrier="interval", num=None, den=None,
                                    empty=False))
            else:
                num, den = _wide(r.num), _wide(r.den)
                answers.append(dict(status=0, value=_result_value(num, den), carrier="exact", num=num, den=den,
                                    empty=False))
        return answers

    def transformed(self, act, g, *, backend="cpu", mean=False):
        """the moments of the transformed rows of group g, one per basis term (exact values or enclosures)"""
        reqs = [dict(action=0, group=g, term=j, mean=mean) for j in range(len(self.basis))]
        return [r["value"] for r in self.query(reqs, actions=[act], backend=backend)]

    def close(self):
        if getattr(self, "basis", None) is not None:
            self.basis.close()
        if getattr(self, "_handle", None):
            _call("tr_sym_state_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        i = self.info
        return f"State({i['size']} moments, {i['groups']} groups, from a {i['source']})"


def state(source, geo, budget=None):
    """the retained moments of a FND-004 bank (JOINT, or SQL without NULLs) or a TENSOR-001 moment tensor over the
    geometry's numeric coordinates (LISTWISE, every axis the same columns, at most once the constant)"""
    error, refusal = C.c_int32(), _Refusal()
    if isinstance(source, mbk.Bank):
        h = _call("tr_sym_state_from_bank", source._handle, C.byref(geo.raw), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        kind = "bank"
    elif isinstance(source, mt.Tensor):
        h = _call("tr_sym_state_from_tensor", source._open(), C.byref(geo.raw), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        kind = "tensor"
    else:
        raise TypeError("a state is read from a math_bank.Bank or a math_tensor.Tensor")
    return State(_made(h, error, refusal, f"state from a {kind}"), geo, kind)


# ---------------------------------------------------------------------- SYM-002: groups, supports, certificates
class Group:
    """A declared group: its generators (integral invertible maps or value permutations, native)."""

    def __init__(self, handle, geo, transforms):
        self._handle, self.geometry, self.transforms = handle, geo, list(transforms)

    def _open(self):
        if not self._handle:
            raise ValueError("the group is closed")
        return self._handle

    @property
    def info(self):
        i = _GroupInfo()
        _check(_call("tr_sym_group_get_info", self._open(), C.byref(i)), what="group")
        return dict(generators=int(i.ngen), n=int(i.n), coefficients=int(i.ncoef), values=int(i.nvalues),
                    bytes=int(i.bytes))

    def _views(self):
        v = _GroupView()
        _check(_call("tr_sym_group_get_view", self._open(), C.byref(v)), what="group view")
        i = self.info
        return v, i["coefficients"], i["values"]

    def invariant(self, query, budget=None):
        """(True, None) when the query's polynomial is proven invariant under every generator, else (False, j)"""
        q, keep = query.native(self.geometry)
        gen, refusal = C.c_int32(-1), _Refusal()
        r = _check(_call("tr_sym_poly_invariant", self._open(), C.byref(q), C.byref(gen), C.byref(_budget(budget)),
                         C.byref(refusal)), refusal, "invariance")
        return (True, None) if r == 1 else (False, int(gen.value))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_sym_group_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def group(geo, generators, budget=None):
    """the group generated by transformations (or maps) of the geometry: each must map the integer domain into
    itself (integral and invertible in logical units) or permute values"""
    gens = list(generators)
    if not gens:
        raise ValueError("a group needs at least one generator")
    if len(gens) > MAX_GENERATORS:
        raise UnsupportedError(f"at most {MAX_GENERATORS} generators")
    maps = [_as_map(t, geo, budget) for t in gens]
    try:
        arr = (C.c_void_p * len(maps))(*[m._open() for m, _ in maps])
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_sym_group_create", arr, len(maps), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Group(_made(h, error, refusal, "group (generators integral and invertible in logical units)"), geo, gens)
    finally:
        for m, own in maps:
            if own:
                m.close()


class Support:
    """The sorted distinct coordinate tuples of a scope's selected rows with multiplicities and first rows (native)."""

    def __init__(self, handle, scope, geo, nulls):
        self._handle, self.scope, self.geometry, self.nulls = handle, scope, geo, nulls

    def _open(self):
        if not self._handle:
            raise ValueError("the support is closed")
        return self._handle

    @property
    def info(self):
        i = _SupportInfo()
        _check(_call("tr_sym_support_get_info", self._open(), C.byref(i)), what="support")
        return dict(rows=int(i.rows), selected=int(i.selected), checked=int(i.checked), excluded=int(i.excluded),
                    distinct=int(i.distinct), bytes=int(i.bytes), nulls={1: "forbid", 2: "skip"}[i.policy])

    def _view(self):
        v = _SupportView()
        _check(_call("tr_sym_support_get_view", self._open(), C.byref(v)), what="support view")
        return v

    def tuples(self):
        """[(tuple of logical values, multiplicity, smallest row)] in increasing order"""
        v = self._view()
        d, n = int(v.distinct), int(v.n)
        if not d:
            return []
        t = np.ctypeslib.as_array((C.c_int64 * (d * n)).from_address(v.tuples)).reshape(d, n)
        m = np.ctypeslib.as_array((C.c_uint64 * d).from_address(v.mult))
        f = np.ctypeslib.as_array((C.c_uint64 * d).from_address(v.first))
        return [(tuple(int(x) for x in t[i]), int(m[i]), int(f[i])) for i in range(d)]

    def fresh(self, scope=None):
        s = _Staleness()
        rc = _call("tr_sym_support_fresh", self._open(), (scope or self.scope)._handle, C.byref(s))
        if rc not in (0, -13):
            _check(rc, what="freshness")
        return _freshness(s)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_sym_support_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _nulls(nulls):
    if nulls not in NULLS:
        raise ValueError("nulls is 'skip' or 'forbid'")
    return NULLS[nulls]


def support(scope, geo, *, nulls="skip", budget=None):
    """the support of the scope's selected rows over every declared coordinate; nulls="forbid" refuses a NULL
    coordinate (DomainError with .row, the smallest such row)"""
    if scope.source == "table":
        scope.require_fresh("support")
    error, refusal, row = C.c_int32(), _Refusal(), C.c_uint64(NONE)
    h = _call("tr_sym_support_build", scope._handle, C.byref(geo.raw), _nulls(nulls), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(row), C.byref(error))
    if error.value == -6 and row.value != NONE:
        err = DomainError(f"support: row {row.value} has a NULL coordinate (nulls='forbid')")
        err.row = int(row.value)
        raise err
    return Support(_made(h, error, refusal, "support"), scope, geo, nulls)


def _evidence(e, names=None):
    return dict(generator=int(e.generator), rows=int(e.rows), tuples=int(e.tuples), violations=int(e.violations),
                missing=int(e.missing), broken=int(e.broken), witness=None if e.witness == NONE else int(e.witness),
                witness_tuple=None if e.witness_tuple == NONE else int(e.witness_tuple),
                image_multiplicity=int(e.image_mult), kind=WITNESS_KINDS[e.kind], holds=bool(e.holds))


def _violated(ev, what="symmetry"):
    bad = [e for e in ev if not e["holds"]]
    first = min(bad, key=lambda e: e["witness"])
    err = SymmetryViolated(f"{what}: generator {first['generator']} breaks the multiset of rows ({first['kind']} at "
                           f"row {first['witness']}; {sum(e['violations'] for e in bad)} violating rows)")
    err.evidence, err.witness, err.kind, err.generator = ev, first["witness"], first["kind"], first["generator"]
    return err


def certify(scope, geo, generators, *, nulls="skip", backend="cpu", budget=None, support_=None):
    """the FND-007 snapshot certificate that the declared group preserves the multiset of the selected rows
    (exhaustive and exact), or SymmetryViolated with every generator's evidence and the smallest witness row;
    backend "cuda" verifies with the CUDA kernel and adopts its evidence natively"""
    g = generators if isinstance(generators, Group) else group(geo, generators, budget)
    sp = support_ or support(scope, geo, nulls=nulls, budget=budget)
    error, refusal = C.c_int32(), _Refusal()
    ngen = g.info["generators"]
    ev = (_Evidence * ngen)()
    if backend == "cpu":
        h = _call("tr_sym_certify", sp._open(), g._open(), C.byref(_budget(budget)), C.byref(refusal), ev,
                  C.byref(error))
        how, device = "cpu", "native CPU"
    elif backend == "cuda":
        gpu = _gpu()
        gv, ncoef, nvalues = g._views()
        sv = sp._view()
        raw = (_Evidence * ngen)()
        gpu.run("tr_mgpu_sym_verify", C.byref(sv), C.byref(gv), ncoef, nvalues, C.byref(_budget(budget)), raw,
                C.byref(refusal), refusal=refusal, what="symmetry verification on the device")
        h = _call("tr_sym_adopt", sp._open(), g._open(), raw, BY_DEVICE, C.byref(_budget(budget)), C.byref(refusal), ev,
                  C.byref(error))
        d = gpu.device()
        how, device = ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    evidence = [_evidence(e) for e in ev]
    if error.value == -6:
        raise _violated(evidence)
    return Certificate(_made(h, error, refusal, "symmetry certificate"), sp, g, scope, geo, how, device, evidence)


class Certificate:
    """A symmetry certificate (FND-007 kind snapshot) of a group over a support: fresh only while the scope's table,
    selection, NULL domains and columns are unchanged (native)."""

    def __init__(self, handle, sp, g, scope, geo, backend, device, evidence):
        self._handle, self.support, self.group, self.scope, self.geometry = handle, sp, g, scope, geo
        self.backend, self.device = backend, device

    def _open(self):
        if not self._handle:
            raise ValueError("the certificate is closed")
        return self._handle

    @property
    def info(self):
        i = _CertInfo()
        _check(_call("tr_sym_cert_get_info", self._open(), C.byref(i)), what="certificate")
        return dict(kind="snapshot", verified_by="cuda library" if i.backend == BY_DEVICE else "cpu",
                    generators=int(i.ngen), n=int(i.n), nulls={1: "forbid", 2: "skip"}[i.policy], holds=bool(i.holds),
                    rows=int(i.rows), selected=int(i.selected), checked=int(i.checked), excluded=int(i.excluded),
                    distinct=int(i.distinct), bytes=int(i.bytes), backend=self.backend, device=self.device)

    @property
    def evidence(self):
        n = self.group.info["generators"]
        ev = (_Evidence * n)()
        _check(_call("tr_sym_cert_get_evidence", self._open(), ev, n), what="certificate evidence")
        return [_evidence(e) for e in ev]

    def fresh(self, scope=None):
        s = _Staleness()
        rc = _call("tr_sym_cert_fresh", self._open(), (scope or self.scope)._handle, C.byref(s))
        if rc not in (0, -13):
            _check(rc, what="freshness")
        return _freshness(s)

    def orbits(self, scope=None, budget=None):
        """the orbit summary under this certificate, which must be fresh for the scope (StaleCertificateError)"""
        cur = scope or self.scope
        error, refusal, s = C.c_int32(), _Refusal(), _Staleness()
        h = _call("tr_sym_orbits_build", self.support._open(), self._open(), cur._handle, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(s), C.byref(error))
        if error.value == -13:
            f = _freshness(s)
            err = StaleCertificateError(f"the certificate's scope changed: {', '.join(f.reasons)}")
            err.reasons, err.first, err.freshness = f.reasons, f.first, f
            raise err
        return Orbits(_made(h, error, refusal, "orbits"), self, cur)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_sym_cert_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- queries
class Query:
    """A polynomial query over the coordinates (logical units): SUM of p over the rows, or COUNT of rows with
    p OP constant. terms: {monomial: int coefficient} (monomials as strings, dicts or exponent tuples)."""

    def __init__(self, terms, kind="sum", op=None, constant=0):
        if kind not in KINDS:
            raise ValueError("kind is 'sum' or 'count'")
        if kind == "count" and op not in OPS:
            raise ValueError(f"op is one of {', '.join(OPS)}")
        self.terms, self.kind, self.op, self.constant = dict(terms), kind, op, constant

    def native(self, geo):
        rows, coefs = [], []
        for t, c in self.terms.items():
            if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
                raise TypeError("query coefficients are exact integers")
            if not _I64[0] <= int(c) <= _I64[1]:
                raise OverflowError("a query coefficient does not fit int64")
            rows.append(_exponents(geo.names, t))
            coefs.append(int(c))
        if len(rows) > MAX_TERMS:
            raise UnsupportedError(f"at most {MAX_TERMS} terms")
        n = geo.n
        e = (C.c_uint8 * max(len(rows) * n, 1))(*[k for r in rows for k in r])
        c = (C.c_int64 * max(len(coefs), 1))(*coefs)
        q = _PolyQuery()
        q.kind, q.op, q.nterms = KINDS[self.kind], OPS[self.op] if self.kind == "count" else 0, len(rows)
        if isinstance(self.constant, bool) or not isinstance(self.constant, (int, np.integer)):
            raise TypeError("the constant is an exact integer")
        q.constant = int(self.constant)
        q.exponents, q.coefficients = C.cast(e, C.c_void_p), C.cast(c, C.c_void_p)
        return q, (e, c)

    def __repr__(self):
        return f"Query({self.kind}, {self.terms}{'' if self.kind == 'sum' else f' {self.op} {self.constant}'})"


def total(terms):
    """sum over the rows of the polynomial"""
    return Query(terms, "sum")


def count(terms, op, constant):
    """the rows where the polynomial OP constant"""
    return Query(terms, "count", op, constant)


def _answers(out, nq, raw, what):
    res = []
    for i in range(nq):
        a = out[i]
        if a.status and not raw:
            if a.status == -6:
                err = NotInvariantError(f"{what}: query {i} is not proven invariant under generator {a.generator}: "
                                        f"refused under the orbit reduction")
                err.generator = int(a.generator)
                raise err
            _check(a.status, what=f"{what}: query {i}")
        res.append(dict(status=int(a.status), value=_wide(a.value) if not a.status else None,
                        invariant=bool(a.invariant), generator=None if a.generator < 0 else int(a.generator),
                        evaluations=int(a.evaluations), rows=int(a.rows)))
    return res


class Orbits:
    """The orbits of a certified group on its support: representatives, sizes, multiplicities (native)."""

    def __init__(self, handle, cert, scope):
        self._handle, self.certificate, self.scope = handle, cert, scope

    def _open(self):
        if not self._handle:
            raise ValueError("the orbits are closed")
        return self._handle

    @property
    def info(self):
        i = _OrbitsInfo()
        _check(_call("tr_sym_orbits_get_info", self._open(), C.byref(i)), what="orbits")
        return dict(orbits=int(i.norbits), distinct=int(i.distinct), rows=int(i.rows), largest=int(i.largest),
                    smallest=int(i.smallest), bytes=int(i.bytes))

    def summary(self):
        """[(representative tuple, orbit size, multiplicity)] (representatives increasing)"""
        v = _OrbitsView()
        _check(_call("tr_sym_orbits_get_view", self._open(), C.byref(v)), what="orbits view")
        k, d, n = int(v.norbits), int(v.distinct), int(v.n)
        if not k:
            return []
        t = np.ctypeslib.as_array((C.c_int64 * (d * n)).from_address(v.tuples)).reshape(d, n)
        rep = np.ctypeslib.as_array((C.c_uint64 * k).from_address(v.rep))
        size = np.ctypeslib.as_array((C.c_uint64 * k).from_address(v.size))
        mult = np.ctypeslib.as_array((C.c_uint64 * k).from_address(v.mult))
        return [(tuple(int(x) for x in t[int(rep[i])]), int(size[i]), int(mult[i])) for i in range(k)]

    def query(self, queries, scope=None, budget=None, raw=False):
        """invariant queries evaluated on the representatives (sum over orbits of size x multiplicity x f(rep));
        a query not proven invariant raises NotInvariantError (raw=True: status -6); a stale certificate
        StaleCertificateError"""
        queries = list(queries)
        geo = self.certificate.geometry
        natives = [q.native(geo) for q in queries]
        arr = (_PolyQuery * max(len(queries), 1))(*[q for q, _ in natives])
        out = (_PolyAnswer * max(len(queries), 1))()
        refusal, s = _Refusal(), _Staleness()
        rc = _call("tr_sym_orbit_query", self._open(), (scope or self.scope)._handle, arr, len(queries), out,
                   C.byref(_budget(budget)), C.byref(refusal), C.byref(s))
        if rc == -13:
            f = _freshness(s)
            err = StaleCertificateError(f"the certificate's scope changed: {', '.join(f.reasons)}")
            err.reasons, err.first, err.freshness = f.reasons, f.first, f
            raise err
        _check(rc, refusal, "orbit queries")
        return _answers(out, len(queries), raw, "orbit queries")

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_sym_orbits_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def evaluate(scope, geo, queries, *, nulls="skip", budget=None):
    """the exhaustive evaluation of any query over every selected row of the scope (complete rows under "skip")"""
    queries = list(queries)
    if scope.source == "table":
        scope.require_fresh("evaluation")
    natives = [q.native(geo) for q in queries]
    arr = (_PolyQuery * max(len(queries), 1))(*[q for q, _ in natives])
    out = (_PolyAnswer * max(len(queries), 1))()
    refusal = _Refusal()
    _check(_call("tr_sym_rows_query", scope._handle, C.byref(geo.raw), _nulls(nulls), arr, len(queries), out,
                 C.byref(_budget(budget)), C.byref(refusal)), refusal, "row queries")
    return _answers(out, len(queries), False, "row queries")


def need(what, *args, backend="cpu"):
    """the needs a request checks before allocating ({resource: need}): ("action", basis, map) or
    ("query", state, n_requests, actions, functionals)"""
    out = _Need()
    if what == "action":
        b, m = args
        _check(_call("tr_sym_action_need", b._open(), m._open(), C.byref(out)), what="action needs")
    elif what == "query":
        st, n, acts, funcs = args
        F = st._functionals(list(funcs))
        handles = (C.c_void_p * max(len(acts), 1))(*[a._open() for a in acts])
        views, v = (_ActionView * max(len(acts), 1))(), _QueryView()
        _check(_call("tr_sym_query_bind", st._open(), handles, len(acts), F.ctypes.data, len(list(funcs)), views,
                     C.byref(v)), what="query needs")
        _check(_call("tr_sym_query_need", C.byref(v), int(n), _NEED_DEVICE if backend == "cuda" else _NEED_CPU,
                     C.byref(out)), what="query needs")
    else:
        raise ValueError("need('action', basis, map) or need('query', state, n, actions, functionals)")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}
