# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DYN-001 / DYN-002: finite closed-observable dynamics on SYM-001's declared geometry, bases and actions, and repeated
operator powers, semigroups, recurrences and parameterized evolution, natively on the CPU or with the CUDA kernels
(torbles/math_dyn.c, math_dyn_gpu.cu; every contract in math_dyn.h).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_sym as ms, math_dyn as dyn
    geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
    B = ms.basis(geo, degree=2)
    F = dyn.affine(geo, ms.compose(ms.rotate("x", "y", cos=Fraction(3, 5), sin=Fraction(4, 5)), ms.translate(x=1)))
    K = dyn.operator(B, F)                      # exact transfer matrix (SYM-001's action); NotClosedError otherwise
    st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
    ev = dyn.evolve(st, K, 10)                  # moments after 10 steps, exact, no row touched (backend="cuda")
    P = dyn.polynomial(geo, {"x": {"x": Fraction(1, 2)}, "y": {"y": Fraction(1, 3), "x^2": 1}})
    dyn.operator(ms.basis(geo, terms=["1", "x", "y", "x^2"]), P)      # nonlinear, closed: exact
    Kt = dyn.operator(B, P, truncate=dyn.box(geo, {"x": (-1, 1), "y": (-2, 2)}))   # approximate, certified
    dyn.evolve(st, Kt, 1, scope=scope).errors(g)                        # |true - approx| <= error, per term
    K.power(1000, method="modular", backend="cuda"); K.charpoly(); K.recurrence({"x^2": 1}, state=st, group=0)
    K.float_power(50, tolerance=1e-9)           # rigorous floating powers; UnstableError past the tolerance
    dyn.mixture([(Fraction(1, 2), K1), (Fraction(1, 2), K2)]); dyn.semigroup(K, 3, 5)

Exact values are int / Fraction; floating results are (values, rigorous bounds). Floats are refused as parameters.
Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_sym as ms
from . import math_certificates as mcert
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _GFP, _I128, _Q
from .math_domains import DomainError
from .math_errors import RESOURCES, NotConvergedError, UnsupportedError, check as _check

__all__ = ["Dynamics", "Box", "Operator", "Power", "Evolution", "Recurrence", "FloatPower", "NotClosedError",
           "UnstableError", "affine", "polynomial", "box", "operator", "mixture", "evolve", "evolve_rows", "semigroup",
           "need", "primes", "MAX_DEGREE", "MAX_PDEGREE", "MAX_TERMS"]

MAX_COORDS, MAX_DEGREE, MAX_PDEGREE, MAX_TERMS, MAX_OPERATORS, MAX_PRIMES = 32, 8, 8, 4096, 64, 4096
METHODS = {"auto": 0, "squaring": 1, "modular": 2, "stepping": 3, "float": 4}
METHOD_NAMES = {v: k for k, v in METHODS.items()}
KINDS = {1: "exact", 2: "truncated"}
SOURCES = {1: "map", 2: "mixture", 3: "substitution"}
LABELS = {0: "exact", 5: "certified approximate"}
RECURRENCES = {"charpoly": 1, "minimal": 2}
NORMS = {"abs": 1, "rel": 2}
REQUESTS = {"power": 1, "evolve": 2, "float": 3, "charpoly": 4, "recurrence": 5, "apply": 6}
PARTS = dict(moments=1, denominators=2, errors=3, error_den=4, numerators=5, denominator=6, coefficients=7,
             initial=8, epsilon=9)
BY_CPU, BY_DEVICE = 0, 1
NotClosedError = ms.NotClosedError
NotAuthorizedError = mcert.NotAuthorizedError
P = C.POINTER


class UnstableError(NotConvergedError):
    """A floating power's certified bound exceeds the declared tolerance (refused): .info (exponent, reached, error,
    tolerance, magnitude, norm)."""


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Term(C.Structure):
    _native_ = "tr_dyn_term"
    _fields_ = [("coord", C.c_int32), ("pexp", C.c_int32), ("exps", C.c_uint8 * MAX_COORDS), ("coef", _Q)]


class _MapInfo(C.Structure):
    _native_ = "tr_dyn_map_info"
    _fields_ = [(n, C.c_int32) for n in ("n", "kind", "degree", "pdegree", "nterms", "parametric")] + [
        ("touched", C.c_uint32), ("reserved", C.c_uint32), ("digest", C.c_uint64 * 2), ("bytes", C.c_uint64)]


class _MapTerm(C.Structure):
    _native_ = "tr_dyn_map_term"
    _fields_ = [("coord", C.c_int32), ("pexp", C.c_int32), ("exps", C.c_uint8 * MAX_COORDS), ("num", _I128),
                ("den", _I128)]


class _Box(C.Structure):
    _native_ = "tr_dyn_box"
    _fields_ = [("n", C.c_int32), ("reserved", C.c_int32), ("declared", C.c_uint32), ("reserved2", C.c_uint32),
                ("lo", _Q * MAX_COORDS), ("hi", _Q * MAX_COORDS)]


class _OpInfo(C.Structure):
    _native_ = "tr_dyn_op_info"
    _fields_ = [(n, C.c_int32) for n in ("nb", "n", "kind", "label", "source", "closed", "pdegree", "invariant",
                                         "sym_action", "reserved")] + [
        ("den", _I128), ("nonzeros", C.c_uint64), ("dropped", C.c_uint64), ("norm_bits", C.c_uint64),
        ("digest", C.c_uint64 * 2), ("bytes", C.c_uint64)]


class _PowerInfo(C.Structure):
    _native_ = "tr_dyn_power_info"
    _fields_ = [("nb", C.c_uint64), ("exponent", C.c_uint64)] + [
        (n, C.c_int32) for n in ("method", "primes", "identity", "checked", "backend", "reserved")] + [
        (n, C.c_uint64) for n in ("bound_bits", "max_bits", "den_bits", "products", "bytes", "cost_squaring",
                                  "cost_modular")] + [("digest", C.c_uint64 * 2)]


class _EvolveInfo(C.Structure):
    _native_ = "tr_dyn_evolve_info"
    _fields_ = [("ngroups", C.c_uint64), ("steps", C.c_uint64)] + [
        (n, C.c_int32) for n in ("nb", "method", "label", "kind", "primes", "invariant", "pdegree", "checked",
                                 "backend", "weighted")] + [
        (n, C.c_uint64) for n in ("bound_bits", "max_bits", "bytes", "rows", "used", "cost_stepping",
                                  "cost_squaring", "cost_modular")]


class _FloatInfo(C.Structure):
    _native_ = "tr_dyn_float_info"
    _fields_ = [("exponent", C.c_uint64), ("reached", C.c_uint64)] + [
        (n, C.c_int32) for n in ("norm", "stable", "products", "groups")] + [
        (n, C.c_double) for n in ("tolerance", "error", "magnitude")]


class _RecInfo(C.Structure):
    _native_ = "tr_dyn_rec_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "degree", "nb", "primes", "sequence", "checked")] + [
        (n, C.c_uint64) for n in ("terms", "bound_bits", "bytes")]


class _Semigroup(C.Structure):
    _native_ = "tr_dyn_semigroup_evidence"
    _fields_ = [("m", C.c_uint64), ("n", C.c_uint64)] + [
        (n, C.c_int32) for n in ("identity", "one", "product", "holds", "method", "reserved")] + [
        ("products", C.c_uint64)]


class _ApplyView(C.Structure):
    _native_ = "tr_dyn_apply_view"
    _fields_ = [("nb", C.c_int32), ("nops", C.c_int32), ("ngroups", C.c_uint64), ("num", C.c_void_p),
                ("moments", C.c_void_p)]


class _ModView(C.Structure):
    _native_ = "tr_dyn_mod_view"
    _fields_ = [("nb", C.c_int32), ("nops", C.c_int32), ("nprimes", C.c_uint64), ("exponent", C.c_uint64),
                ("num", C.c_void_p), ("primes", C.c_void_p)]


for _s, _n in ((_Term, 56), (_MapInfo, 56), (_MapTerm, 72), (_Box, 1040), (_OpInfo, 104), (_PowerInfo, 112),
               (_EvolveInfo, 120), (_FloatInfo, 56), (_RecInfo, 48), (_Semigroup, 48), (_ApplyView, 32),
               (_ModView, 40)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e, R, B = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32), P(_Refusal), P(_Budget)
        sig = {
            "tr_dyn_map_from_sym": (vp, [vp, P(ms._Geometry), B, R, e]),
            "tr_dyn_map_polynomial": (vp, [P(ms._Geometry), vp, i32, i32, B, R, e]),
            "tr_dyn_map_get_info": (i32, [vp, P(_MapInfo)]),
            "tr_dyn_map_get_terms": (i32, [vp, vp, u64]),
            "tr_dyn_map_free": (None, [vp]),
            "tr_dyn_operator_need": (i32, [vp, vp, vp, P(_Need)]),
            "tr_dyn_operator_make": (vp, [vp, vp, vp, B, R, P(ms._Closure), e]),
            "tr_dyn_operator_mixture": (vp, [vp, vp, i32, B, R, e]),
            "tr_dyn_operator_substitute": (vp, [vp, _Q, B, R, e]),
            "tr_dyn_operator_get_info": (i32, [vp, P(_OpInfo)]),
            "tr_dyn_operator_get_view": (i32, [vp, i32, P(ms._ActionView)]),
            "tr_dyn_operator_get_box": (i32, [vp, P(_Box)]),
            "tr_dyn_operator_check_geometry": (i32, [vp, P(ms._Geometry)]),
            "tr_dyn_authorize_exact": (i32, [vp, P(mcert._Decision)]),
            "tr_dyn_operator_free": (None, [vp]),
            "tr_dyn_need": (i32, [i32, vp, vp, u64, i32, i32, P(_Need), P(i32)]),
            "tr_dyn_mod_plan": (i32, [vp, vp, u64, P(u64), P(u64)]),
            "tr_dyn_primes": (i32, [u64, vp]),
            "tr_dyn_mod_bind": (i32, [vp, u64, vp, u64, P(_ModView)]),
            "tr_dyn_apply_bind": (i32, [vp, vp, P(_ApplyView)]),
            "tr_dyn_apply_view_eval": (i32, [P(_ApplyView), vp, B, R]),
            "tr_dyn_powmod_view_eval": (i32, [P(_ModView), vp, B, R]),
            "tr_dyn_evolvemod_view_eval": (i32, [P(_ModView), vp, u64, vp, B, R]),
            "tr_dyn_evolve": (vp, [vp, vp, u64, i32, vp, B, R, e]),
            "tr_dyn_evolution_adopt": (vp, [vp, vp, u64, vp, u64, vp, i32, B, R, e]),
            "tr_dyn_evolution_at": (vp, [vp, _Q, B, R, e]),
            "tr_dyn_rows_need": (i32, [vp, vp, vp, u64, i32, P(_Need)]),
            "tr_dyn_rows_evolve": (vp, [vp, vp, vp, u64, i32, B, R, e]),
            "tr_dyn_evolution_get_info": (i32, [vp, P(_EvolveInfo)]),
            "tr_dyn_evolution_get_rows": (i32, [vp, vp, vp, u64]),
            "tr_dyn_evolution_authorize_exact": (i32, [vp, P(mcert._Decision)]),
            "tr_dyn_evolution_free": (None, [vp]),
            "tr_dyn_power_make": (vp, [vp, u64, i32, B, R, e]),
            "tr_dyn_power_adopt": (vp, [vp, u64, vp, u64, vp, i32, B, R, e]),
            "tr_dyn_power_compose": (vp, [vp, vp, B, R, e]),
            "tr_dyn_power_equal": (i32, [vp, vp]),
            "tr_dyn_power_get_info": (i32, [vp, P(_PowerInfo)]),
            "tr_dyn_power_free": (None, [vp]),
            "tr_dyn_semigroup": (i32, [vp, u64, u64, i32, P(_Semigroup), B, R]),
            "tr_dyn_charpoly": (vp, [vp, B, R, e]),
            "tr_dyn_recurrence_make": (vp, [vp, vp, vp, u64, vp, i32, B, R, e]),
            "tr_dyn_recurrence_get_info": (i32, [vp, P(_RecInfo)]),
            "tr_dyn_recurrence_term_need": (i32, [vp, u64, P(_Need)]),
            "tr_dyn_recurrence_term": (i32, [vp, u64, vp, u64, P(u64), B, R]),
            "tr_dyn_recurrence_free": (None, [vp]),
            "tr_dyn_float_power": (i32, [vp, u64, vp, C.c_double, i32, vp, vp, P(_FloatInfo), B, R]),
            "tr_dyn_export": (i32, [vp, i32, vp, u64, P(u64)]),
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
    math_gpu.bind("tr_mgpu_dyn_apply", C.c_int, [P(_ApplyView), C.c_void_p, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_dyn_powmod", C.c_int, [P(_ModView), C.c_void_p, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_dyn_evolvemod", C.c_int, [P(_ModView), C.c_void_p, C.c_uint64, C.c_void_p, P(_Budget),
                                                    P(_Refusal)])
    return math_gpu


def _precheck(request, op, n, state, budget):
    """the full needs of a device request (tr_dyn_need, TR_DYN_NEED_DEVICE) against the budget, natively, before
    anything is copied: the refusal of the request is that of its reported needs"""
    raw, chosen, refusal = _Need(), C.c_int32(), _Refusal()
    _check(_call("tr_dyn_need", REQUESTS[request], op._open(), state._open() if state is not None else None, n,
                 METHODS["modular"] if request != "apply" else METHODS["stepping"], 1, C.byref(raw), C.byref(chosen)),
           what="device needs")
    _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw), C.byref(refusal)), refusal,
           f"{request} on the device")


def _frame(state, op):
    """the state's geometry is exactly the operator's (natively; the scope digest and the basis are checked by every
    native entry point)"""
    if state is not None:
        _check(_call("tr_dyn_operator_check_geometry", op._open(), C.byref(state.geometry.raw)),
               what="the state's geometry (columns, kinds, units, scales) is not the operator's")


def _gpu_backend(gpu):
    return "cuda-emulated" if gpu.device()["emulated"] else "cuda"


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _int(lo, hi):
    return (int(hi) << 64) | (int(lo) & ((1 << 64) - 1))


def _value(f):
    f = Fraction(f)
    return f.numerator if f.denominator == 1 else f


def _method(method):
    if method not in METHODS:
        raise ValueError(f"method is one of {', '.join(METHODS)}")
    return METHODS[method]


def _steps(n, what="steps"):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 0 or int(n) >= 1 << 64:
        raise ValueError(f"{what} is an integer in [0, 2^64)")
    return int(n)


def _export(obj, part, what):
    """the packed exact integers of a part of a native object, as Python ints"""
    words = C.c_uint64()
    _check(_call("tr_dyn_export", obj, PARTS[part], None, 0, C.byref(words)), what=what)
    n = int(words.value)
    buf = np.zeros(max(n, 1), dtype=np.uint64)
    _check(_call("tr_dyn_export", obj, PARTS[part], buf.ctypes.data, n, C.byref(words)), what=what)
    out, i = [], 0
    raw = buf.tobytes()
    while i < n:
        h = int(buf[i])
        cnt, neg = h & ((1 << 63) - 1), h >> 63
        v = int.from_bytes(raw[8 * (i + 1):8 * (i + 1 + cnt)], "little") if cnt else 0
        out.append(-v if neg else v)
        i += 1 + cnt
    return out


def _pairs(values):
    return [_value(Fraction(values[2 * k], values[2 * k + 1])) for k in range(len(values) // 2)]


def _decision(d, rc):
    return dict(authorized=bool(d.authorized), reason=mcert.REFUSALS.get(d.reason),
                kind="law" if d.kind == 1 else LABELS.get(d.kind, d.kind), code=int(rc))


def _authorize(fn, handle, what):
    d = mcert._Decision()
    rc = _call(fn, handle, C.byref(d))
    if rc == -1:
        _check(rc, what=what)
    dec = _decision(d, rc)
    if dec["authorized"]:
        return dec
    err = NotAuthorizedError(f"{what} is approximate (a certified bound): it cannot authorize an exact rewrite")
    err.reason, err.decision = dec["reason"], dec
    raise err


class _Handle:
    _free = None

    def __init__(self, handle):
        self._handle = handle

    def _open(self):
        if not self._handle:
            raise ValueError(f"the {type(self).__name__.lower()} is closed")
        return self._handle

    def close(self):
        if getattr(self, "_handle", None):
            _call(self._free, self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- dynamics
class Dynamics(_Handle):
    """Declared dynamics x' = F(x) on a geometry's coordinates (native, logical units)."""
    _free = "tr_dyn_map_free"

    def __init__(self, handle, geo, parameter=None):
        super().__init__(handle)
        self.geometry, self.parameter = geo, parameter

    @property
    def info(self):
        i = _MapInfo()
        _check(_call("tr_dyn_map_get_info", self._open(), C.byref(i)), what="dynamics")
        return dict(n=int(i.n), kind={1: "affine", 2: "polynomial"}[i.kind], degree=int(i.degree),
                    pdegree=int(i.pdegree), terms=int(i.nterms), parametric=bool(i.parametric),
                    touched=[self.geometry.names[k] for k in range(i.n) if i.touched >> k & 1], bytes=int(i.bytes))

    def terms(self):
        """the stored terms in logical units: {coordinate: {(exponents, parameter exponent): Fraction}}"""
        n = self.info["terms"]
        arr = (_MapTerm * max(n, 1))()
        _check(_call("tr_dyn_map_get_terms", self._open(), arr, n), what="dynamics terms")
        out = {}
        for t in arr[:n]:
            key = (tuple(t.exps[:self.geometry.n]), int(t.pexp))
            out.setdefault(self.geometry.names[t.coord], {})[key] = _value(
                Fraction(_int(t.num.lo, t.num.hi), _int(t.den.lo, t.den.hi)))
        return out


def affine(geo, transform, budget=None):
    """exact affine dynamics from a SYM-001 transformation (or Map) of the geometry: translations, scalings,
    reflections, permutations, affine maps, exact rotations and their compositions (interval maps are refused)"""
    m, own = ms._as_map(transform, geo, budget)
    try:
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_map_from_sym", m._open(), C.byref((m.geometry or geo).raw), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Dynamics(_made(h, error, refusal, "affine dynamics"), m.geometry or geo)
    finally:
        if own:
            m.close()


def polynomial(geo, components, parameter=None, budget=None):
    """x'_c = sum of the terms of components[c] ({term: exact coefficient}, terms like "1", "x", "x*y^2" over the
    coordinates and the declared parameter); undeclared coordinates are fixed"""
    names = list(geo.names) + ([parameter] if parameter is not None else [])
    if parameter is not None and parameter in geo.names:
        raise ValueError(f"the parameter {parameter!r} is a coordinate")
    rows = []
    for coord, terms in dict(components).items():
        ci = geo.index(coord)
        items = terms.items() if isinstance(terms, dict) else list(terms)
        for term, coef in items:
            e = ms._exponents(names, term)
            t = _Term()
            t.coord = ci
            t.pexp = e[-1] if parameter is not None else 0
            for j in range(geo.n):
                t.exps[j] = e[j]
            t.coef = ms._q(coef, f"the coefficient of {term!r} in {coord!r}")
            rows.append(t)
    if len(rows) > MAX_TERMS:
        raise UnsupportedError(f"at most {MAX_TERMS} terms")
    arr = (_Term * max(len(rows), 1))(*rows)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_dyn_map_polynomial", C.byref(geo.raw), arr, len(rows), int(parameter is not None),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Dynamics(_made(h, error, refusal, "polynomial dynamics"), geo, parameter)


class Box:
    """The approximation contract of a truncation: lo <= x <= hi (exact, value units) per bounded coordinate."""

    def __init__(self, geo, bounds):
        self.geometry = geo
        raw = _Box()
        raw.n = geo.n
        for name, (lo, hi) in dict(bounds).items():
            i = geo.index(name)
            raw.declared |= 1 << i
            raw.lo[i], raw.hi[i] = ms._q(lo, f"the lower bound of {name!r}"), ms._q(hi, f"the upper bound of {name!r}")
        self.raw, self.bounds = raw, {k: (Fraction(v[0]), Fraction(v[1])) for k, v in dict(bounds).items()}

    def __repr__(self):
        return "Box(" + ", ".join(f"{k} in [{a}, {b}]" for k, (a, b) in self.bounds.items()) + ")"


def box(geo, bounds):
    return Box(geo, bounds)


# ---------------------------------------------------------------------- operators
class Operator(_Handle):
    """The transfer operator K of dynamics on a closed basis: phi(F(x)) = K phi(x), exact (N / D), or truncated
    under a box contract (approximate, with certified dropped-term bounds)."""
    _free = "tr_dyn_operator_free"

    def __init__(self, handle, basis_, geo, dynamics=None):
        super().__init__(handle)
        self.basis, self.geometry, self.dynamics = basis_, geo, dynamics

    @property
    def info(self):
        i = _OpInfo()
        _check(_call("tr_dyn_operator_get_info", self._open(), C.byref(i)), what="operator")
        return dict(size=int(i.nb), n=int(i.n), kind=KINDS[i.kind], label=LABELS[i.label],
                    approximate=i.kind == 2, source=SOURCES[i.source], closed=bool(i.closed), pdegree=int(i.pdegree),
                    invariant=bool(i.invariant), sym_action=bool(i.sym_action), den=_int(i.den.lo, i.den.hi),
                    nonzeros=int(i.nonzeros), dropped=int(i.dropped), norm_bits=int(i.norm_bits), bytes=int(i.bytes))

    def numerators(self, j=0):
        """(N_j as rows of ints, D): K = sum_j theta^j N_j / D"""
        v = ms._ActionView()
        _check(_call("tr_dyn_operator_get_view", self._open(), int(j), C.byref(v)), what="operator view")
        nb = int(v.nb)
        w = np.ctypeslib.as_array((C.c_int64 * (2 * nb * nb)).from_address(v.num)).reshape(nb, nb, 2)
        return [[_int(w[a, b, 0], w[a, b, 1]) for b in range(nb)] for a in range(nb)], _int(v.den.lo, v.den.hi)

    def matrix(self, j=0):
        N, D = self.numerators(j)
        return [[_value(Fraction(x, D)) for x in row] for row in N]

    def epsilon(self):
        """truncated: the certified bound of the dropped terms of every basis row (logical units)"""
        return _pairs(_export(self._open(), "epsilon", "dropped-term bounds"))

    @property
    def box(self):
        b = _Box()
        _check(_call("tr_dyn_operator_get_box", self._open(), C.byref(b)), what="the contract (a truncated operator)")
        return {self.geometry.names[i]: (Fraction(b.lo[i].numerator, b.lo[i].denominator),
                                         Fraction(b.hi[i].numerator, b.hi[i].denominator))
                for i in range(b.n) if b.declared >> i & 1}

    def authorize_exact(self):
        """the FND-007 decision for an exact operator; NotAuthorizedError (reason "approximate") for a truncation"""
        return _authorize("tr_dyn_authorize_exact", self._open(), "the operator")

    def at(self, theta, budget=None):
        """K(theta) of a parametric operator at an exact rational"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_operator_substitute", self._open(), ms._q(theta, "theta"), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Operator(_made(h, error, refusal, "substitution"), self.basis, self.geometry, self.dynamics)

    def power(self, n, method="auto", backend="cpu", budget=None):
        """K^n exactly: integer numerators N^n over D^n (squaring, or modular with CRT and a check)"""
        n = _steps(n, "the exponent")
        if backend == "cpu":
            error, refusal = C.c_int32(), _Refusal()
            h = _call("tr_dyn_power_make", self._open(), n, _method(method), C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
            return Power(_made(h, error, refusal, "power"), self, "cpu")
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        if method not in ("auto", "modular"):
            raise UnsupportedError("the CUDA powers are modular")
        gpu = _gpu()
        _precheck("power", self, n, None, budget)
        pr, view = self._mod_setup(n, None)
        e2 = int(view.nb) ** 2
        res = np.zeros(max(len(pr) * e2, 1), dtype=np.uint64)
        refusal = _Refusal()
        gpu.run("tr_mgpu_dyn_powmod", C.byref(view), res.ctypes.data, C.byref(_budget(budget)), C.byref(refusal),
                refusal=refusal, what="powers on the device")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_power_adopt", self._open(), n, pr, len(pr), res.ctypes.data, BY_DEVICE,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Power(_made(h, error, refusal, "adopted power"), self, _gpu_backend(gpu))

    def _mod_setup(self, n, state):
        np_, bound = C.c_uint64(), C.c_uint64()
        _check(_call("tr_dyn_mod_plan", self._open(), state._open() if state is not None else None, n, C.byref(np_),
                     C.byref(bound)), what="modular plan")
        pr = (_GFP * int(np_.value))()
        _check(_call("tr_dyn_primes", int(np_.value), pr), what="primes")
        view = _ModView()
        _check(_call("tr_dyn_mod_bind", self._open(), n, pr, int(np_.value), C.byref(view)), what="modular view")
        return pr, view

    def apply(self, state, backend="cpu", budget=None):
        """one step K m_g for every group: exact (512-bit numerators over D), CPU or CUDA"""
        _frame(state, self)
        view = _ApplyView()
        _check(_call("tr_dyn_apply_bind", state._open(), self._open(), C.byref(view)),
               what="application (the state's basis and scope)")
        total = int(view.ngroups) * int(view.nb)
        out = (ms._Wide * max(total, 1))()
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_dyn_apply_view_eval", C.byref(view), out, C.byref(_budget(budget)), C.byref(refusal)),
                   refusal, "application")
        elif backend == "cuda":
            gpu = _gpu()
            _precheck("apply", self, 1, state, budget)
            gpu.run("tr_mgpu_dyn_apply", C.byref(view), out, C.byref(_budget(budget)), C.byref(refusal),
                    refusal=refusal, what="application on the device")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        D = self.info["den"]
        nb = int(view.nb)
        return [[_value(Fraction(ms._wide(out[g * nb + i]), D)) for i in range(nb)] for g in range(int(view.ngroups))]

    def charpoly(self, budget=None):
        """the exact characteristic polynomial det(x I - K): coefficients, constant first (modular, CRT, checked)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_charpoly", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Recurrence(_made(h, error, refusal, "characteristic polynomial"), self)

    def recurrence(self, functional, state=None, group=0, vector=None, kind="minimal", budget=None):
        """the recurrence of s_t = c^T K^t v (c: {term: int} or a row of ints; v: a state group's moments or ints)"""
        nb = self.info["size"]
        c = (C.c_int64 * nb)(*ms.State._functionals(_FakeState(self.basis), [functional])[0])
        vec = None
        if state is None:
            if vector is None or len(vector) != nb:
                raise ValueError(f"give state= and group=, or a vector of {nb} integers")
            vec = (_I128 * nb)()
            for i, x in enumerate(vector):
                if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not -(1 << 127) < int(x) < 1 << 127:
                    raise ValueError("the vector holds exact int128 integers")
                vec[i].lo, vec[i].hi = int(x) & ((1 << 64) - 1), int(x) >> 64
        if kind not in RECURRENCES:
            raise ValueError("kind is 'minimal' or 'charpoly'")
        _frame(state, self)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_recurrence_make", self._open(), c, state._open() if state is not None else None,
                  int(group), vec, RECURRENCES[kind], C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Recurrence(_made(h, error, refusal, "recurrence"), self)

    def float_power(self, n, tolerance, norm="abs", state=None, budget=None):
        """K^n (or K^n m_g with state) in outward-rounded binary64 with a rigorous bound per value; UnstableError
        when the bound exceeds the tolerance (norm "abs": max bound; "rel": max bound / max |value|)"""
        n = _steps(n, "the exponent")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 < float(tolerance) < float("inf"):
            raise ValueError("the tolerance is a positive finite number")
        if norm not in NORMS:
            raise ValueError("norm is 'abs' or 'rel'")
        _frame(state, self)
        nb = self.info["size"]
        G = state.info["groups"] if state is not None else 0
        size = G * nb if state is not None else nb * nb
        mid, rad = np.zeros(max(size, 1)), np.zeros(max(size, 1))
        info, refusal = _FloatInfo(), _Refusal()
        rc = _call("tr_dyn_float_power", self._open(), n, state._open() if state is not None else None,
                   float(tolerance), NORMS[norm], mid.ctypes.data, rad.ctypes.data, C.byref(info),
                   C.byref(_budget(budget)), C.byref(refusal))
        d = dict(exponent=int(info.exponent), reached=int(info.reached), norm=norm, stable=bool(info.stable),
                 products=int(info.products), tolerance=float(info.tolerance), error=float(info.error),
                 magnitude=float(info.magnitude), approximate=True, certified=True, precision=53)
        if rc == -14:
            err = UnstableError(f"the floating power K^{n} is unstable: its certified bound {d['error']:.3g} exceeds "
                                f"the tolerance {d['tolerance']:.3g} (certified up to the exponent {d['reached']})")
            err.info = d
            raise err
        _check(rc, refusal, "floating power")
        shape = (G, nb) if state is not None else (nb, nb)
        return FloatPower(mid[:size].reshape(shape), rad[:size].reshape(shape), d)

    def __repr__(self):
        i = self.info
        return f"Operator({i['size']} x {i['size']}, {i['kind']}, den={i['den']})"


class _FakeState:
    """the functional parser of SYM-001's State over a basis (formatting only)"""

    def __init__(self, basis_):
        self.basis = basis_


def operator(basis_, dynamics, truncate=None, budget=None):
    """the transfer operator of the dynamics on the basis: exact when the basis is closed (NotClosedError with the
    witness otherwise), or truncated under truncate=box(...) (approximate, certified bounds)"""
    error, refusal, closure = C.c_int32(), _Refusal(), ms._Closure()
    bx = C.byref(truncate.raw) if truncate is not None else None
    h = _call("tr_dyn_operator_make", basis_._open(), dynamics._open(), bx, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(closure), C.byref(error))
    if error.value == -6 and closure.term >= 0:
        names = basis_.names
        missing = {n: int(closure.missing[i]) for i, n in enumerate(names) if closure.missing[i]}
        err = NotClosedError(f"the basis is not closed under the dynamics: the image of term "
                             f"{basis_.terms[closure.term]} has {missing or '1'} outside it")
        err.term, err.missing = basis_.terms[closure.term], missing
        raise err
    return Operator(_made(h, error, refusal, "operator"), basis_, dynamics.geometry, dynamics)


def mixture(pairs, budget=None):
    """the convex combination sum_j w_j K_j (exact weights >= 0 summing to 1): the expected moments of a dynamics
    choosing F_j with probability w_j independently at every step"""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("a mixture needs at least one operator")
    if len(pairs) > MAX_OPERATORS:
        raise UnsupportedError(f"at most {MAX_OPERATORS} operators")
    ops = (C.c_void_p * len(pairs))(*[k._open() for _, k in pairs])
    ws = (_Q * len(pairs))(*[ms._q(w, "a weight") for w, _ in pairs])
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_dyn_operator_mixture", ops, ws, len(pairs), C.byref(_budget(budget)), C.byref(refusal),
              C.byref(error))
    return Operator(_made(h, error, refusal, "mixture (convex weights, one basis)"), pairs[0][1].basis,
                    pairs[0][1].geometry)


# ---------------------------------------------------------------------- powers
class Power(_Handle):
    """K^n exactly: integer numerators N^n over the denominator D^n (composites: products)."""
    _free = "tr_dyn_power_free"

    def __init__(self, handle, op, backend):
        super().__init__(handle)
        self.operator, self.backend = op, backend

    @property
    def info(self):
        i = _PowerInfo()
        _check(_call("tr_dyn_power_get_info", self._open(), C.byref(i)), what="power")
        return dict(size=int(i.nb), exponent=int(i.exponent), method=METHOD_NAMES.get(i.method, "composite"),
                    primes=int(i.primes), identity=bool(i.identity), checked=bool(i.checked),
                    backend=self.backend, bound_bits=int(i.bound_bits), max_bits=int(i.max_bits),
                    den_bits=int(i.den_bits), products=int(i.products), bytes=int(i.bytes),
                    cost_squaring=int(i.cost_squaring), cost_modular=int(i.cost_modular))

    def numerators(self):
        nb = self.info["size"]
        num = _export(self._open(), "numerators", "power numerators")
        den = _export(self._open(), "denominator", "power denominator")[0]
        return [num[i * nb:(i + 1) * nb] for i in range(nb)], den

    def matrix(self):
        N, D = self.numerators()
        return [[_value(Fraction(x, D)) for x in row] for row in N]

    def compose(self, other, budget=None):
        """self other (the power of self's exponent after other's)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_power_compose", self._open(), other._open(), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Power(_made(h, error, refusal, "power composition"), self.operator, "cpu")

    def equal(self, other):
        return bool(_check(_call("tr_dyn_power_equal", self._open(), other._open()), what="power comparison"))

    def __eq__(self, other):
        return isinstance(other, Power) and self.equal(other)

    __hash__ = None


def semigroup(op, m, n, method="auto", budget=None):
    """K^0 = I, K^1 = K and K^m K^n = K^(m+n), verified exactly (native)"""
    ev, refusal = _Semigroup(), _Refusal()
    _check(_call("tr_dyn_semigroup", op._open(), _steps(m, "m"), _steps(n, "n"), _method(method), C.byref(ev),
                 C.byref(_budget(budget)), C.byref(refusal)), refusal, "semigroup check")
    return dict(m=int(ev.m), n=int(ev.n), identity=bool(ev.identity), one=bool(ev.one), product=bool(ev.product),
                holds=bool(ev.holds), method=method, products=int(ev.products))


# ---------------------------------------------------------------------- evolution
class Evolution(_Handle):
    """Per-group moments after t steps: exact numerators over a denominator per basis term; truncated: with the
    certified componentwise error bound; parametric: polynomials in the parameter."""
    _free = "tr_dyn_evolution_free"

    def __init__(self, handle, basis_, backend="cpu"):
        super().__init__(handle)
        self.basis, self.backend, self._cache = basis_, backend, {}

    @property
    def info(self):
        i = _EvolveInfo()
        _check(_call("tr_dyn_evolution_get_info", self._open(), C.byref(i)), what="evolution")
        return dict(groups=int(i.ngroups), steps=int(i.steps), size=int(i.nb), method=METHOD_NAMES[i.method],
                    label=LABELS[i.label], approximate=i.label != 0, kind={1: "operator", 2: "rows"}[i.kind],
                    primes=int(i.primes), invariant=bool(i.invariant), pdegree=int(i.pdegree),
                    checked=bool(i.checked), backend=self.backend, weighted=bool(i.weighted),
                    bound_bits=int(i.bound_bits), max_bits=int(i.max_bits), bytes=int(i.bytes), rows=int(i.rows),
                    used=int(i.used), cost_stepping=int(i.cost_stepping), cost_squaring=int(i.cost_squaring),
                    cost_modular=int(i.cost_modular))

    def _part(self, name):
        if name not in self._cache:
            self._cache[name] = _export(self._open(), name, f"evolution {name}")
        return self._cache[name]

    def moments(self, g):
        """the exact moments of group g, one per basis term (parametric: per term the coefficients of theta^d)"""
        i = self.info
        g = int(g)
        if not 0 <= g < i["groups"]:
            raise ValueError("group id out of range")
        num, den, nb, W = self._part("moments"), self._part("denominators"), i["size"], i["pdegree"] + 1
        if W == 1:
            return [_value(Fraction(num[g * nb + a], den[a])) for a in range(nb)]
        return [[_value(Fraction(num[(g * nb + a) * W + d], den[a])) for d in range(W)] for a in range(nb)]

    def errors(self, g):
        """truncated: the certified bound E_t (|true moment - moment| <= E_t) of every basis term of group g"""
        i = self.info
        if not i["approximate"]:
            raise ValueError("an exact evolution has no error bound")
        err, den, nb = self._part("errors"), self._part("error_den")[0], i["size"]
        return [_value(Fraction(err[int(g) * nb + a], den)) for a in range(nb)]

    def group(self, g):
        G = self.info["groups"]
        rows, used = (C.c_uint64 * G)(), (C.c_uint64 * G)()
        _check(_call("tr_dyn_evolution_get_rows", self._open(), rows, used, G), what="evolution rows")
        return dict(present=rows[int(g)] > 0, rows=int(rows[int(g)]), used=int(used[int(g)]))

    def at(self, theta, budget=None):
        """a parametric evolution at an exact rational theta"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_evolution_at", self._open(), ms._q(theta, "theta"), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Evolution(_made(h, error, refusal, "evaluation at theta"), self.basis, self.backend)

    def authorize_exact(self):
        return _authorize("tr_dyn_evolution_authorize_exact", self._open(), "the evolution")


def evolve(state, op, steps, method="auto", backend="cpu", scope=None, budget=None):
    """the moments after `steps` steps of every group of a SYM-001 state, exactly, from the operator alone (no row
    is read): stepping, squaring or modular (CPU), modular on the device (backend="cuda"); a truncated operator needs
    the state's scope (its rows must lie in the box) and returns certified error bounds"""
    t = _steps(steps)
    _frame(state, op)
    if backend == "cpu":
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_dyn_evolve", state._open(), op._open(), t, _method(method),
                  scope._handle if scope is not None else None, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Evolution(_made(h, error, refusal, "evolution"), state.basis, "cpu")
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    if method not in ("auto", "modular"):
        raise UnsupportedError("the CUDA evolution is modular")
    gpu = _gpu()
    _precheck("evolve", op, t, state, budget)
    pr, view = op._mod_setup(t, state)
    qv = ms._QueryView()
    _check(ms._call("tr_sym_state_get_view", state._open(), C.byref(qv)), what="state view")
    G, nb = int(qv.ngroups), int(qv.nb)
    res = np.zeros(max(len(pr) * G * nb, 1), dtype=np.uint64)
    refusal = _Refusal()
    gpu.run("tr_mgpu_dyn_evolvemod", C.byref(view), qv.moments, G, res.ctypes.data, C.byref(_budget(budget)),
            C.byref(refusal), refusal=refusal, what="evolution on the device")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_dyn_evolution_adopt", state._open(), op._open(), t, pr, len(pr), res.ctypes.data, BY_DEVICE,
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Evolution(_made(h, error, refusal, "adopted evolution"), state.basis, _gpu_backend(gpu))


def evolve_rows(scope, dynamics, basis_, steps, weight=None, budget=None):
    """the reference: every selected row (map and basis coordinates present) evolved exactly, then its moments"""
    w = -1 if weight is None else scope.names.index(weight)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_dyn_rows_evolve", scope._handle, dynamics._open(), basis_._open(), _steps(steps), w,
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Evolution(_made(h, error, refusal, "row evolution"), basis_, "cpu")


def need(request, op, steps=0, state=None, method="auto", backend="cpu"):
    """the needs a request would be checked against (and the method AUTO chooses)"""
    raw, chosen = _Need(), C.c_int32()
    _frame(state, op)
    _check(_call("tr_dyn_need", REQUESTS[request], op._open(), state._open() if state is not None else None,
                 _steps(steps), _method(method), 1 if backend == "cuda" else 0, C.byref(raw), C.byref(chosen)),
           what="needs")
    d = {RESOURCES[i]: int(raw.need[i]) for i in range(len(RESOURCES))}
    d["method"] = METHOD_NAMES.get(chosen.value)
    return d


def rows_need(scope, dynamics, basis_, steps, weight=None):
    raw = _Need()
    w = -1 if weight is None else scope.names.index(weight)
    _check(_call("tr_dyn_rows_need", scope._handle, dynamics._open(), basis_._open(), _steps(steps), w, C.byref(raw)),
           what="row evolution needs")
    return {RESOURCES[i]: int(raw.need[i]) for i in range(len(RESOURCES))}


def operator_need(basis_, dynamics, truncate=None):
    raw = _Need()
    _check(_call("tr_dyn_operator_need", basis_._open(), dynamics._open(),
                 C.byref(truncate.raw) if truncate is not None else None, C.byref(raw)), what="operator needs")
    return {RESOURCES[i]: int(raw.need[i]) for i in range(len(RESOURCES))}


def primes(count):
    """the first count primes below 2^62 (LIN-002's), each verified prime"""
    pr = (_GFP * int(count))()
    _check(_call("tr_dyn_primes", int(count), pr), what="primes")
    return [int(p.p) for p in pr]


# ---------------------------------------------------------------------- recurrences and floating powers
class Recurrence(_Handle):
    """chi_K's coefficients (charpoly), or the recurrence s_(t+d) = sum_k r_k s_(t+k) of a sequence c^T K^t v."""
    _free = "tr_dyn_recurrence_free"

    def __init__(self, handle, op):
        super().__init__(handle)
        self.operator = op

    @property
    def info(self):
        i = _RecInfo()
        _check(_call("tr_dyn_recurrence_get_info", self._open(), C.byref(i)), what="recurrence")
        return dict(kind={1: "charpoly", 2: "minimal"}[i.kind], degree=int(i.degree), size=int(i.nb),
                    primes=int(i.primes), sequence=bool(i.sequence), checked=bool(i.checked), terms=int(i.terms),
                    bound_bits=int(i.bound_bits), bytes=int(i.bytes))

    @property
    def coefficients(self):
        """charpoly: det(x I - K) constant first (the leading 1 last); a sequence's recurrence: r_0 .. r_(d-1)"""
        return _pairs(_export(self._open(), "coefficients", "coefficients"))

    @property
    def initial(self):
        return _pairs(_export(self._open(), "initial", "initial terms"))

    def term(self, T, budget=None):
        """s_T exactly by polynomial reduction (x^T mod the recurrence's polynomial)"""
        T = _steps(T, "T")
        words, refusal = C.c_uint64(), _Refusal()
        _check(_call("tr_dyn_recurrence_term", self._open(), T, None, 0, C.byref(words), C.byref(_budget(budget)),
                     C.byref(refusal)), refusal, "recurrence term")
        n = int(words.value)
        buf = np.zeros(max(n, 1), dtype=np.uint64)
        _check(_call("tr_dyn_recurrence_term", self._open(), T, buf.ctypes.data, n, C.byref(words),
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "recurrence term")
        vals, i, raw = [], 0, buf.tobytes()
        while i < n:
            h = int(buf[i])
            cnt, neg = h & ((1 << 63) - 1), h >> 63
            v = int.from_bytes(raw[8 * (i + 1):8 * (i + 1 + cnt)], "little") if cnt else 0
            vals.append(-v if neg else v)
            i += 1 + cnt
        return _value(Fraction(vals[0], vals[1]))

    def term_need(self, T):
        raw = _Need()
        _check(_call("tr_dyn_recurrence_term_need", self._open(), _steps(T, "T"), C.byref(raw)), what="term needs")
        return {RESOURCES[i]: int(raw.need[i]) for i in range(len(RESOURCES))}


class FloatPower:
    """Floating values (midpoints) with rigorous bounds: |exact - values| <= bounds elementwise (APPROXIMATE)."""

    def __init__(self, values, bounds, info):
        self.values, self.bounds, self.info = values, bounds, info

    def __repr__(self):
        return f"FloatPower(shape={self.values.shape}, error={self.info['error']:.3g}, {self.info['norm']})"
