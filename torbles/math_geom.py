# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEOM-001: certified sparse ideal reduction, natively on the CPU or with the CUDA kernels
(torbles/math_geom.h/.c, math_geom_gpu.cu).

    from torbles import math_bank as mbk, math_ident as mi, math_geom as mg
    found = mi.discover(scope, ["x", "y", "z"], degree=2)             # DISC-002: verified identities, certificates
    rels = [i["certificate"] for i in found.discoveries()]
    I = mg.ideal(rels, order="deglex", variables=["z", "y", "x"])     # reduced Groebner basis with cofactors
    r = I.reduce(mg.poly({(2, 1, 0): 1, (0, 0, 3): -2}, ["x", "y", "z"]))   # f = g + sum h_i r_i, exact
    cert = r.certify()                                                 # FND-007 derived certificate (tr_cert_derive)
    cert.authorize_exact(scope)                                        # on the relations' own scope only
    mg.agree(scope, [(r.f, r.g)], backend="cuda")                      # f and g evaluated on every certified row
    mg.choose(scope, [r.f, r.g], priority="bits", measure=True)        # the cheaper representation
    mg.check([r, (claim, premises, cofactors)], backend="cuda")        # batched exact certificate checks
    mg.derivative(f, "x", through=True, dependencies={"y": phi})       # only through declared dependencies

The remainder g equals f only where the relations hold: a snapshot-certified relation makes them equivalent on that
snapshot's rows and nowhere else (the derived certificate refuses any other scope), a hypothesis never authorizes. An
identity on the rows never makes a column a function of another: a derivative through a relation is refused unless
the dependency is declared. Nothing is computed in Python (marshalling and display only). Opt-in: `import torbles`
does not load it.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_bank as _mbk
from . import math_certificates as _mc
from . import math_ident as _mi
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Polynomial", "Ideal", "Reduction", "poly", "ideal", "cost", "choose", "agree", "agree_need", "check",
           "check_need", "unrank", "derivative", "dependency_check", "ORDERS", "PRIORITIES", "MAX_VARS",
           "MAX_RELATIONS", "MAX_BASIS", "MAX_PAIRS", "MAX_CHECKS", "MAX_DEPS"]

MAX_VARS, MAX_RELATIONS, MAX_BASIS, MAX_DEPS, MAX_PAIRS, MAX_CHECKS = 8, 16, 128, 8, 256, 256
ORDERS = {"lex": 1, "deglex": 2, "degrevlex": 3}
PRIORITIES = {"columns": 1, "bits": 2, "terms": 3}
REASONS = {0: None, 1: "undeclared", 2: "dependent", 3: "cycle"}
_ORDER_NAMES = {v: k for k, v in ORDERS.items()}
_NEED_CPU, _NEED_DEVICE = 0, 1
_I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Order(C.Structure):
    _native_ = "tr_geom_order"
    _fields_ = [("kind", C.c_int32), ("nvars", C.c_int32), ("col", C.c_int32 * 8)]


class _GPoly(C.Structure):
    _native_ = "tr_geom_poly"
    _fields_ = [("ncols", C.c_int32), ("reserved", C.c_int32), ("col", C.c_int32 * 8), ("nterms", C.c_uint64),
                ("exponents", C.c_void_p), ("num", C.c_void_p), ("den", C.c_void_p)]


class _IdealInfo(C.Structure):
    _native_ = "tr_geom_ideal_info"
    _fields_ = [(n, C.c_int32) for n in ("nrelations", "nvars", "kind", "completed")] + [("col", C.c_int32 * 8)] + [
        (n, C.c_int32) for n in ("nbasis", "policy", "weakest", "reserved")] + [
        (n, C.c_uint64) for n in ("pairs", "reductions", "max_terms", "max_degree", "cofactor_terms", "bytes")]


class _RedInfo(C.Structure):
    _native_ = "tr_geom_reduction_info"
    _fields_ = [("nvars", C.c_int32), ("nrelations", C.c_int32), ("col", C.c_int32 * 8)] + [
        (n, C.c_uint64) for n in ("f_terms", "g_terms", "cofactor_terms", "steps")] + [
        ("denominator", C.c_int64), ("claimable", C.c_int32), ("code", C.c_int32)]


class _Cost(C.Structure):
    _native_ = "tr_geom_cost"
    _fields_ = [(n, C.c_uint64) for n in ("terms", "columns", "degree", "multiplications", "bound_bits",
                                          "actual_bits")] + [
        ("denominator", C.c_int64), ("measured", C.c_int32), ("reserved", C.c_int32)]


class _PairView(C.Structure):
    _native_ = "tr_geom_pair_view"
    _fields_ = [("npairs", C.c_int32), ("reserved", C.c_int32), ("cand", _mi._CandView)]


class _Agreement(C.Structure):
    _native_ = "tr_geom_agreement"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "checked", "excluded", "disagreements", "witness",
                                          "undecided")] + [("bits_f", C.c_uint32), ("bits_g", C.c_uint32)]


class _Dep(C.Structure):
    _native_ = "tr_geom_dep"
    _fields_ = [("column", C.c_int32), ("reserved", C.c_int32), ("phi", _GPoly)]


class _CheckView(C.Structure):
    _native_ = "tr_geom_check_view"
    _fields_ = [("ncert", C.c_int32), ("reserved", C.c_int32)] + [
        (n, C.c_uint64) for n in ("npolys", "nterms", "nslots", "nwork")] + [
        (n, C.c_void_p) for n in ("nvars", "npremises", "maxdeg", "overflow", "poly0", "slot0", "work0", "term0",
                                  "exponents", "coefficients")]


class _CheckResult(C.Structure):
    _native_ = "tr_geom_check_result"
    _fields_ = [(n, C.c_uint64) for n in ("slots", "work", "nonzero", "first")] + [
        ("holds", C.c_int32), ("overflow", C.c_int32), ("reserved", C.c_uint64)]


for _s, _n in ((_Order, 40), (_GPoly, 72), (_IdealInfo, 112), (_RedInfo, 88), (_Cost, 64), (_PairView, 96),
               (_Agreement, 64), (_Dep, 80), (_CheckView, 120), (_CheckResult, 48)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_geom_pol_get": (i32, [vp, P(i32), vp, P(u64)]),
            "tr_geom_pol_export": (i32, [vp, vp, vp, vp, u64]),
            "tr_geom_pol_free": (None, [vp]),
            "tr_geom_ideal_make": (vp, [vp, i32, P(_Order), i32, B, R, e]),
            "tr_geom_ideal_get_info": (i32, [vp, P(_IdealInfo)]),
            "tr_geom_ideal_element": (vp, [vp, i32, i32, e]),
            "tr_geom_ideal_free": (None, [vp]),
            "tr_geom_reduce": (vp, [vp, P(_GPoly), B, R, e]),
            "tr_geom_reduction_get_info": (i32, [vp, P(_RedInfo)]),
            "tr_geom_reduction_part": (vp, [vp, i32, e]),
            "tr_geom_reduction_claims": (i32, [vp, P(_mc._Claim), P(_mc._Claim), i32]),
            "tr_geom_certify": (vp, [vp, B, R, e]),
            "tr_geom_reduction_free": (None, [vp]),
            "tr_geom_cost_of": (i32, [vp, P(_GPoly), P(_Cost)]),
            "tr_geom_choose": (i32, [P(_Cost), i32, i32, P(i32)]),
            "tr_geom_pairs_make": (vp, [vp, P(_GPoly), P(_GPoly), i32, B, R, e]),
            "tr_geom_pairs_get_view": (i32, [vp, P(_PairView)]),
            "tr_geom_pairs_free": (None, [vp]),
            "tr_geom_agree_need": (i32, [P(_mbk._View), P(_PairView), i32, P(_Need)]),
            "tr_geom_agree_view": (i32, [P(_mbk._View), P(_PairView), B, P(_Agreement), R]),
            "tr_geom_checks_make": (vp, [i32, P(_mc._Claim), vp, P(_mc._Claim), P(_mc._Claim), B, R, e]),
            "tr_geom_checks_get_view": (i32, [vp, P(_CheckView)]),
            "tr_geom_checks_columns": (i32, [vp, i32, vp, i32]),
            "tr_geom_checks_need": (i32, [vp, i32, P(_Need)]),
            "tr_geom_checks_run": (i32, [vp, B, P(_CheckResult), R]),
            "tr_geom_checks_free": (None, [vp]),
            "tr_geom_unrank": (i32, [u64, i32, vp]),
            "tr_geom_derivative": (vp, [P(_GPoly), i32, i32, P(_Dep), i32, B, P(i32), R, e]),
            "tr_geom_dependency_check": (i32, [vp, P(_Dep), i32, B, P(_mc._Evidence), R]),
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
    math_gpu.bind("tr_mgpu_geom_agree", C.c_int, [P(_mbk._View), P(_PairView), P(_Budget), P(_Agreement),
                                                 P(_Refusal)])
    math_gpu.bind("tr_mgpu_geom_check", C.c_int, [P(_CheckView), P(_Budget), P(_CheckResult), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _device_label():
    from . import math_gpu
    d = math_gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


def _fresh(scope, what):
    if getattr(scope, "source", None) == "table":
        scope.require_fresh(what)


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


# ------------------------------------------------------------------ polynomials (marshalling and display only)
def _coef(c):
    if isinstance(c, bool) or not isinstance(c, (int, Fraction)):
        raise TypeError("coefficients are exact integers or Fractions")
    return Fraction(c)


class Polynomial:
    """A polynomial over named columns with exact rational coefficients: columns (names), terms {exponent tuple:
    Fraction} (nonzero), ordered (the native order of a native result, else decreasing graded lexicographic)."""

    def __init__(self, terms, columns):
        columns = tuple(columns)
        if len(set(columns)) != len(columns):
            raise ValueError("distinct columns")
        if len(columns) > MAX_VARS:
            raise UnsupportedError(f"at most {MAX_VARS} columns")
        acc = {}
        items = terms.items() if isinstance(terms, dict) else [(None, t) for t in terms]
        for key, value in items:
            if key is None:
                coef, powers = value
                e = [0] * len(columns)
                for name, k in dict(powers).items():
                    if name not in columns:
                        raise ValueError(f"{name!r} is not a column ({', '.join(columns)})")
                    e[columns.index(name)] += int(k)
            else:
                e, coef = [int(k) for k in key], value
                if len(e) != len(columns):
                    raise ValueError("one exponent per column")
            if any(not 0 <= k <= 255 for k in e):
                raise ValueError("exponents are 0..255")
            acc[tuple(e)] = acc.get(tuple(e), Fraction(0)) + _coef(coef)
        self.columns = columns
        self.terms = {e: c for e, c in acc.items() if c}
        self.ordered = sorted(self.terms.items(), key=lambda t: (sum(t[0]), t[0]), reverse=True)

    @classmethod
    def _native(cls, handle, names):
        """a native tr_geom_pol (freed here) as a Polynomial over the names of its scope columns"""
        try:
            ncols, cols, n = C.c_int32(), (C.c_int32 * 8)(), C.c_uint64()
            _check(_call("tr_geom_pol_get", handle, C.byref(ncols), cols, C.byref(n)), what="polynomial")
            k, m = int(ncols.value), int(n.value)
            e = np.zeros(max(m * k, 1), np.uint8)
            num, den = np.zeros(max(m, 1), np.int64), np.zeros(max(m, 1), np.int64)
            _check(_call("tr_geom_pol_export", handle, e.ctypes.data, num.ctypes.data, den.ctypes.data, m),
                   what="polynomial")
        finally:
            _call("tr_geom_pol_free", handle)
        columns = [names[cols[i]] for i in range(k)]
        ordered = [(tuple(int(x) for x in e[t * k:(t + 1) * k]), Fraction(int(num[t]), int(den[t]))) for t in range(m)]
        p = cls({}, columns)
        p.terms = dict(ordered)
        p.ordered = ordered
        return p

    @classmethod
    def from_certificate(cls, cert, names=None):
        """the polynomial a certificate's claim states is zero (POLYNOMIAL; AFFINE / SUM / PRODUCT as FND-007 reads
        them: out - scale x - bias, out - x - y, out - x y)"""
        names = names if names is not None else cert.names
        c = cert.claim
        cols = [names[k] for k in c["columns"]]
        if c["kind"] == "polynomial":
            return cls(c["terms"], cols)
        if c["kind"] == "affine":
            return cls([(1, {cols[0]: 1}), (-c["scale"], {cols[1]: 1}), (-c["bias"], {})], cols)
        if c["kind"] == "sum":
            return cls([(1, {cols[0]: 1}), (-1, {cols[1]: 1}), (-1, {cols[2]: 1})], cols)
        if c["kind"] == "product":
            return cls([(1, {cols[0]: 1}), (-1, {cols[1]: 1, cols[2]: 1})], cols)
        raise DomainError(f"a {c['kind']} claim is not a polynomial relation")

    def aligned(self, columns):
        """the terms over another column sequence (a superset of the columns that appear)"""
        columns = tuple(columns)
        out = {}
        for e, c in self.terms.items():
            f = [0] * len(columns)
            for name, k in zip(self.columns, e):
                if k:
                    if name not in columns:
                        raise ValueError(f"{name!r} appears but is not among {columns}")
                    f[columns.index(name)] = k
            out[tuple(f)] = out.get(tuple(f), Fraction(0)) + c
        return {e: c for e, c in out.items() if c}

    @property
    def used(self):
        """the columns that appear"""
        return tuple(n for q, n in enumerate(self.columns) if any(e[q] for e in self.terms))

    def __eq__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        cols = tuple(dict.fromkeys(self.columns + other.columns))
        return self.aligned(cols) == other.aligned(cols)

    __hash__ = None

    def __len__(self):
        return len(self.terms)

    @property
    def text(self):
        out = []
        for e, c in self.ordered:
            m = _mi.monomial_text(e, self.columns)
            mag = abs(c)
            body = m if (mag == 1 and m != "1") else (str(mag) if m == "1" else f"{mag}*{m}")
            out.append(("- " if c < 0 else "+ ") + body)
        if not out:
            return "0"
        text = " ".join(out)
        return text[2:] if text.startswith("+ ") else "-" + text[2:]

    def __repr__(self):
        return f"Polynomial({self.text})"

    def native(self, names):
        """(the tr_geom_poly over scope columns, the arrays it points into)"""
        cols = []
        for name in self.columns:
            if name not in names:
                raise ValueError(f"{name!r} is not a column ({', '.join(names)})")
            cols.append(list(names).index(name))
        g = _GPoly()
        g.ncols = len(cols)
        for i in range(8):
            g.col[i] = cols[i] if i < len(cols) else -1
        items = self.ordered
        k = len(cols)
        exps = np.ascontiguousarray(np.array([e for e, _ in items], dtype=np.uint8).reshape(len(items), k))
        num = [c.numerator for _, c in items]
        den = [c.denominator for _, c in items]
        for v in num + den:
            if not _I64[0] <= v <= _I64[1]:
                raise OverflowError("a coefficient's numerator or denominator does not fit int64")
        num, den = np.ascontiguousarray(num, dtype=np.int64), np.ascontiguousarray(den, dtype=np.int64)
        g.nterms = len(items)
        if len(items):
            g.exponents, g.num, g.den = exps.ctypes.data, num.ctypes.data, den.ctypes.data
        return g, [exps, num, den]


def poly(terms, columns):
    """a Polynomial: terms {exponent tuple: int or Fraction} or [(coefficient, {column: exponent})]"""
    return Polynomial(terms, columns)


def _as_poly(f, what="polynomial"):
    if isinstance(f, Polynomial):
        return f
    if isinstance(f, tuple) and len(f) == 2:
        return Polynomial(*f)
    raise TypeError(f"{what}: a math_geom.Polynomial (math_geom.poly(terms, columns))")


def _claim_obj(raw, names):
    """a native POLYNOMIAL tr_cert_claim (pointers into native memory) copied into a math_certificates.Claim"""
    n, k = int(raw.nterms), int(raw.ncols)
    if n:
        exps = np.ctypeslib.as_array((C.c_uint8 * (n * k)).from_address(raw.exponents)).reshape(n, k).copy()
        coefs = np.ctypeslib.as_array((C.c_int64 * n).from_address(raw.coefficients)).copy()
    else:
        exps, coefs = np.zeros((0, k), np.uint8), np.zeros(0, np.int64)
    cols = [names[raw.col[i]] for i in range(k)]
    return _mc.Claim("polynomial", cols, nulls=_mc._POLICY_NAMES[raw.policy], exponents=exps, coefficients=coefs)


# ------------------------------------------------------------------ ideals and reductions
def _names_of(relations, names, scope):
    if scope is not None:
        return tuple(scope.names)
    if names is not None:
        return tuple(names)
    for r in relations:
        if getattr(r, "names", None) is not None:
            return tuple(r.names)
    raise ValueError("give the scope (or names) the relations' columns belong to")


class Ideal:
    """The ideal of relation certificates under a monomial order: its basis (with complete=True the reduced
    Groebner basis) and each element's cofactors over the relations, exact."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_geom.ideal(relations, ...)")

    @classmethod
    def _wrap(cls, handle, relations, names):
        self = object.__new__(cls)
        self._handle, self.relations, self.names = handle, tuple(relations), tuple(names)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the ideal is closed")
        return self._handle

    @property
    def info(self):
        i = _IdealInfo()
        _check(_call("tr_geom_ideal_get_info", self._open(), C.byref(i)), what="ideal")
        return dict(relations=int(i.nrelations), variables=[self.names[i.col[k]] for k in range(i.nvars)],
                    order=_ORDER_NAMES[i.kind], completed=bool(i.completed), basis=int(i.nbasis),
                    nulls=_mc._POLICY_NAMES.get(i.policy), weakest=_mc._KIND_NAMES.get(i.weakest),
                    pairs=int(i.pairs), reductions=int(i.reductions), max_terms=int(i.max_terms),
                    max_degree=int(i.max_degree), cofactor_terms=int(i.cofactor_terms), bytes=int(i.bytes))

    def _element(self, j, cofactor):
        error = C.c_int32()
        h = _call("tr_geom_ideal_element", self._open(), int(j), int(cofactor), C.byref(error))
        return Polynomial._native(_made(h, error, None, "ideal element"), self.names)

    @property
    def basis(self):
        """the basis elements (monic, increasing leading monomial for a completed basis)"""
        return [self._element(j, 0) for j in range(self.info["basis"])]

    def cofactors(self, j):
        """element j's cofactors over the relations: basis[j] = sum_i cofactors(j)[i] * relation_i"""
        return [self._element(j, i + 1) for i in range(len(self.relations))]

    def reduce(self, f, *, budget=None):
        """f = g + sum_i h_i r_i, g the normal form (no term divisible by a leading monomial of the basis)"""
        g, keep = _as_poly(f).native(self.names)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_geom_reduce", self._open(), C.byref(g), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Reduction._wrap(_made(h, error, refusal, "reduction"), self)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_geom_ideal_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        i = self.info
        return f"Ideal({i['relations']} relations, {i['basis']} basis elements, {i['order']} {i['variables']})"


def ideal(relations, *, order="deglex", variables=None, complete=True, scope=None, names=None, budget=None):
    """the ideal of FND-007 relation certificates (POLYNOMIAL, AFFINE, SUM or PRODUCT claims; one NULL policy):
    order lex / deglex / degrevlex over variables (column names, most significant first; the rest after, in scope
    column order); complete: Buchberger's completion within the budget (refused past it, never truncated)"""
    relations = list(relations)
    if not relations:
        raise ValueError("at least one relation")
    if len(relations) > MAX_RELATIONS:
        raise UnsupportedError(f"at most {MAX_RELATIONS} relations (FND-007 derivations: 16 premises)")
    for r in relations:
        if not isinstance(r, _mc.Certificate):
            raise TypeError("relations are math_certificates.Certificate objects")
    if order not in ORDERS:
        raise ValueError(f"order is one of {', '.join(ORDERS)}")
    nm = _names_of(relations, names, scope)
    for r in relations:
        if r.names is not None and _mc._KIND_NAMES.get(r._info().kind) != "law" and tuple(r.names) != nm:
            raise DomainError("the relations are certificates of scopes with different columns "
                              f"({', '.join(r.names)} against {', '.join(nm)})")
    o = _Order()
    o.kind = ORDERS[order]
    seq = list(variables or ())
    if len(seq) > MAX_VARS:
        raise UnsupportedError(f"at most {MAX_VARS} variables")
    for i in range(8):
        o.col[i] = -1
    for i, v in enumerate(seq):
        if v not in nm:
            raise ValueError(f"{v!r} is not a column ({', '.join(nm)})")
        o.col[i] = list(nm).index(v)
    o.nvars = len(seq)
    ptrs = (C.c_void_p * len(relations))(*[r._open() for r in relations])
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_geom_ideal_make", ptrs, len(relations), C.byref(o), 1 if complete else 0, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    if not h and error.value == -6:
        raise DomainError("the relations do not form an ideal here: a law or schema certificate, a tolerance or a "
                          "field claim, two NULL policies, or relations of different tables or column layouts")
    return Ideal._wrap(_made(h, error, refusal, "ideal"), relations, nm)


class Reduction:
    """f = g + sum_i h_i r_i: f, g (the remainder), cofactors [h_i], info; claims() and certify() (FND-007)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Ideal.reduce(f)")

    @classmethod
    def _wrap(cls, handle, ideal_):
        self = object.__new__(cls)
        self._handle, self.ideal, self.names = handle, ideal_, ideal_.names
        self.relations = ideal_.relations
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the reduction is closed")
        return self._handle

    @property
    def info(self):
        i = _RedInfo()
        _check(_call("tr_geom_reduction_get_info", self._open(), C.byref(i)), what="reduction")
        return dict(columns=[self.names[i.col[k]] for k in range(i.nvars)], relations=int(i.nrelations),
                    f_terms=int(i.f_terms), g_terms=int(i.g_terms), cofactor_terms=int(i.cofactor_terms),
                    steps=int(i.steps), denominator=int(i.denominator) or None, claimable=bool(i.claimable),
                    code=int(i.code))

    def _part(self, part):
        error = C.c_int32()
        h = _call("tr_geom_reduction_part", self._open(), int(part), C.byref(error))
        return Polynomial._native(_made(h, error, None, "reduction part"), self.names)

    @property
    def f(self):
        return self._part(0)

    @property
    def g(self):
        """the remainder (normal form)"""
        return self._part(1)

    @property
    def cofactors(self):
        return [self._part(2 + i) for i in range(len(self.relations))]

    def _raw_claims(self):
        n = len(self.relations)
        claim, cof = _mc._Claim(), (_mc._Claim * max(n, 1))()
        rc = _call("tr_geom_reduction_claims", self._open(), C.byref(claim), cof, n)
        if rc == -3:
            raise OverflowError("the certificate's integer coefficients do not fit int64 (no FND-007 claim)")
        if rc == -12:
            raise UnsupportedError("the certificate has more than 4096 terms (no FND-007 claim)")
        _check(rc, what="reduction claims")
        return claim, [cof[i] for i in range(n)]

    def claims(self):
        """(the claim P = D (f - g), [the cofactor claims D h_i]) as math_certificates Claims over the column names"""
        claim, cof = self._raw_claims()
        return _claim_obj(claim, self.names), [_claim_obj(c, self.names) for c in cof]

    def certify(self, *, budget=None):
        """the FND-007 derived certificate of P = sum D h_i r_i (verified natively by tr_cert_derive): its kind
        the weakest relation's, valid only where the relations are"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_geom_certify", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        if not h and error.value == -6:
            raise DomainError("the certificate does not follow from the relations")
        if not h and error.value == -3:
            raise OverflowError("the certificate's integer coefficients do not fit int64 (no FND-007 claim)")
        return _mc.Certificate._wrap(_made(h, error, refusal, "certificate"), self.names)

    def check(self, *, backend="cpu", budget=None):
        """the batched exact check of this certificate (math_geom.check)"""
        return check([self], backend=backend, budget=budget)[0]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_geom_reduction_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        i = self.info
        return f"Reduction({i['f_terms']} -> {i['g_terms']} terms, {i['steps']} steps, claimable={i['claimable']})"


# ------------------------------------------------------------------ costs and choices
def _cost_dict(c):
    return dict(terms=int(c.terms), columns=int(c.columns), degree=int(c.degree),
                multiplications=int(c.multiplications), bound_bits=int(c.bound_bits),
                actual_bits=int(c.actual_bits) if c.measured else None, denominator=int(c.denominator),
                measured=bool(c.measured))


def _costs(scope, polys):
    out = (_Cost * max(len(polys), 1))()
    for i, f in enumerate(polys):
        g, keep = _as_poly(f).native(scope.names)
        _check(_call("tr_geom_cost_of", scope._handle, C.byref(g), C.byref(out[i])), what="cost")
    return out


def cost(scope, f):
    """terms, columns read, total degree, multiplications of a naive evaluation, the bits of the integer form's
    evaluation bound on the scope (bound_bits) and its denominator"""
    return _cost_dict(_costs(scope, [f])[0])


def choose(scope, candidates, *, priority="columns", measure=False, backend="cpu", budget=None):
    """(the index of the cheapest candidate, [costs]): priority columns / bits / terms first, then the others in
    the order columns, bits, terms, ties to the earliest; measure: the actual bits on the rows (agree) and their
    agreement with the first candidate on every selected row"""
    candidates = [_as_poly(f) for f in candidates]
    if not candidates:
        raise ValueError("at least one candidate")
    if priority not in PRIORITIES:
        raise ValueError(f"priority is one of {', '.join(PRIORITIES)}")
    costs = _costs(scope, candidates)
    agreement = None
    if measure:
        agreement = agree(scope, [(c, candidates[0]) for c in candidates], backend=backend, budget=budget)
        for i, a in enumerate(agreement):
            costs[i].actual_bits, costs[i].measured = a["bits_f"], 1
    chosen = C.c_int32()
    _check(_call("tr_geom_choose", costs, len(candidates), PRIORITIES[priority], C.byref(chosen)), what="choose")
    out = [_cost_dict(costs[i]) for i in range(len(candidates))]
    if agreement is not None:
        for c, a in zip(out, agreement):
            c["agrees"] = a["disagreements"] == 0 and a["undecided"] == 0
            c["disagreements"] = a["disagreements"]
    return int(chosen.value), out


# ------------------------------------------------------------------ agreement on rows
def _agreement_dict(a, label, device):
    return dict(rows=int(a.rows), selected=int(a.selected), checked=int(a.checked), excluded=int(a.excluded),
                disagreements=int(a.disagreements), witness=None if a.witness == _mc.NONE else int(a.witness),
                undecided=int(a.undecided), bits_f=int(a.bits_f), bits_g=int(a.bits_g), backend=label, device=device)


def agree(scope, pairs, *, backend="cpu", budget=None):
    """for each (f, g): F = D_f f, G = D_g g and P = D (f - g) evaluated exactly on every selected row (a row where a
    column of f or g is NULL is excluded): checked, excluded, disagreements (P != 0), the smallest disagreeing row
    (witness), undecided rows, and the largest bit lengths of |F| and |G|"""
    _fresh(scope, "agreement")
    pairs = [(_as_poly(f), _as_poly(g)) for f, g in pairs]
    if len(pairs) > MAX_PAIRS:
        raise UnsupportedError(f"at most {MAX_PAIRS} pairs per batch")
    n = len(pairs)
    F, G = (_GPoly * max(n, 1))(), (_GPoly * max(n, 1))()
    keep = []
    for j, (f, g) in enumerate(pairs):
        F[j], k1 = f.native(scope.names)
        G[j], k2 = g.native(scope.names)
        keep += k1 + k2
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_geom_pairs_make", scope._handle, F, G, n, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    h = _made(h, error, refusal, "agreement pairs")
    try:
        pv = _PairView()
        _check(_call("tr_geom_pairs_get_view", h, C.byref(pv)), what="agreement pairs")
        view = scope.view()
        out = (_Agreement * max(n, 1))()
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_geom_agree_view", C.byref(view), C.byref(pv), C.byref(_budget(budget)), out,
                         C.byref(refusal)), refusal, "agreement")
            label, device = "cpu", "native CPU"
        elif backend == "cuda":
            _gpu().run("tr_mgpu_geom_agree", C.byref(view), C.byref(pv), C.byref(_budget(budget)), out,
                       C.byref(refusal), refusal=refusal, what="agreement on the device")
            label, device = _device_label()
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return [_agreement_dict(out[j], label, device) for j in range(n)]
    finally:
        _call("tr_geom_pairs_free", h)


def agree_need(scope, pairs, *, backend="cpu", budget=None):
    """the needs of an agreement batch: {resource: need}"""
    pairs = [(_as_poly(f), _as_poly(g)) for f, g in pairs]
    n = len(pairs)
    F, G = (_GPoly * max(n, 1))(), (_GPoly * max(n, 1))()
    keep = []
    for j, (f, g) in enumerate(pairs):
        F[j], k1 = f.native(scope.names)
        G[j], k2 = g.native(scope.names)
        keep += k1 + k2
    error, refusal = C.c_int32(), _Refusal()
    h = _made(_call("tr_geom_pairs_make", scope._handle, F, G, n, C.byref(_budget(budget)), C.byref(refusal),
                    C.byref(error)), error, refusal, "agreement pairs")
    try:
        pv, need = _PairView(), _Need()
        _call("tr_geom_pairs_get_view", h, C.byref(pv))
        _check(_call("tr_geom_agree_need", C.byref(scope.view()), C.byref(pv),
                     _NEED_DEVICE if backend == "cuda" else _NEED_CPU, C.byref(need)), what="needs")
        return _needs_dict(need)
    finally:
        _call("tr_geom_pairs_free", h)


# ------------------------------------------------------------------ batched certificate checks
def _native_claim(c, names, keep):
    if isinstance(c, _mc.Certificate):
        keep.append(c)
        return c._native_claim()
    if isinstance(c, _mc.Claim):
        raw, k = c.native(names)
        keep += k
        return raw
    if isinstance(c, _mc._Claim):
        return c
    raise TypeError("a claim is a math_certificates Claim or Certificate")


def _items(items, names):
    """[(claim, [premise], [cofactor])] as native claims, with the column names"""
    out, keep = [], []
    for it in items:
        if isinstance(it, Reduction):
            claim, cof = it._raw_claims()
            keep.append(it)
            prem = [r._native_claim() for r in it.relations]
            keep += list(it.relations)
            out.append((claim, prem, cof))
            names = names or it.names
            continue
        claim, premises, cofactors = it
        premises, cofactors = list(premises), list(cofactors)
        if len(premises) != len(cofactors):
            raise ValueError("one cofactor per premise")
        if names is None:
            for c in premises:
                if isinstance(c, _mc.Certificate) and c.names is not None:
                    names = c.names
        if names is None:
            raise ValueError("give the column names")
        out.append((_native_claim(claim, names, keep), [_native_claim(p, names, keep) for p in premises],
                    [_native_claim(q, names, keep) for q in cofactors]))
    return out, keep, names


def _checks(items, names, budget):
    raw, keep, names = _items(items, names)
    n = len(raw)
    if n > MAX_CHECKS:
        raise UnsupportedError(f"at most {MAX_CHECKS} certificates per batch")
    claims = (_mc._Claim * max(n, 1))()
    npremises = np.zeros(max(n, 1), np.int32)
    total = sum(len(p) for _, p, _ in raw)
    prem, cof = (_mc._Claim * max(total, 1))(), (_mc._Claim * max(total, 1))()
    at = 0
    for c, (claim, ps, qs) in enumerate(raw):
        claims[c] = claim
        npremises[c] = len(ps)
        for p, q in zip(ps, qs):
            prem[at], cof[at] = p, q
            at += 1
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_geom_checks_make", n, claims, npremises.ctypes.data, prem, cof, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return _made(h, error, refusal, "certificate checks"), names, n


def check(items, *, backend="cpu", budget=None, names=None):
    """the batched exact check of certificates P = sum_i q_i G_i: items are Reductions or (claim, [premise
    certificate or claim], [cofactor claim]); per certificate holds (decided and zero), decided, nonzero slots and
    the smallest nonzero monomial (first: {column: exponent}), slots, work"""
    items = list(items)
    h, names, n = _checks(items, names, budget)
    try:
        out = (_CheckResult * max(n, 1))()
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_geom_checks_run", h, C.byref(_budget(budget)), out, C.byref(refusal)), refusal,
                   "certificate checks")
            label, device = "cpu", "native CPU"
        elif backend == "cuda":
            view = _CheckView()
            _check(_call("tr_geom_checks_get_view", h, C.byref(view)), what="certificate checks")
            _gpu().run("tr_mgpu_geom_check", C.byref(view), C.byref(_budget(budget)), out, C.byref(refusal),
                       refusal=refusal, what="certificate checks on the device")
            label, device = _device_label()
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        res = []
        for c in range(n):
            r = out[c]
            cols = (C.c_int32 * 8)()
            k = _check(_call("tr_geom_checks_columns", h, c, cols, 8), what="certificate columns")
            first = None
            if r.first != _mc.NONE and not r.overflow:
                e = (C.c_uint8 * 8)()
                _check(_call("tr_geom_unrank", r.first, k, e), what="rank")
                first = {names[cols[q]]: int(e[q]) for q in range(k) if e[q]}
            res.append(dict(holds=bool(r.holds), decided=not r.overflow, nonzero=int(r.nonzero),
                            first_rank=None if r.first == _mc.NONE else int(r.first), first=first,
                            slots=int(r.slots), work=int(r.work), columns=[names[cols[q]] for q in range(k)],
                            backend=label, device=device))
        return res
    finally:
        _call("tr_geom_checks_free", h)


def check_need(items, *, backend="cpu", budget=None, names=None):
    """the needs of running a check batch: {resource: need}"""
    h, names, n = _checks(list(items), names, budget)
    try:
        need = _Need()
        _check(_call("tr_geom_checks_need", h, _NEED_DEVICE if backend == "cuda" else _NEED_CPU, C.byref(need)),
               what="needs")
        return _needs_dict(need)
    finally:
        _call("tr_geom_checks_free", h)


def unrank(rank, nvars):
    """the exponent vector of a graded rank (the dense index of the checks)"""
    e = (C.c_uint8 * 8)()
    _check(_call("tr_geom_unrank", int(rank), int(nvars), e), what="rank")
    return tuple(int(e[q]) for q in range(nvars))


# ------------------------------------------------------------------ derivatives and dependencies
def _deps(dependencies, names, keep):
    deps = dict(dependencies or {})
    if len(deps) > MAX_DEPS:
        raise UnsupportedError(f"at most {MAX_DEPS} declared dependencies")
    arr = (_Dep * max(len(deps), 1))()
    for i, (col, phi) in enumerate(deps.items()):
        if col not in names:
            raise ValueError(f"{col!r} is not a column ({', '.join(names)})")
        arr[i].column = list(names).index(col)
        arr[i].phi, k = _as_poly(phi, "a dependency").native(names)
        keep += k
    return arr, len(deps)


def derivative(f, column, *, through=False, dependencies=None, budget=None):
    """through=False: the formal partial derivative of f as written (its columns independent; no dependencies may
    be given). through=True: the total derivative d f / d column through the declared dependencies {v: phi} only
    (v = phi(other columns)); an identity on the rows is never a dependency: DomainError .reason "undeclared" with
    none declared, "dependent" for d/dv of a declared dependent column, "cycle" for a cyclic declaration."""
    f = _as_poly(f)
    deps = dict(dependencies or {})
    names = list(dict.fromkeys(list(f.columns) + [column] + list(deps) +
                               [c for phi in deps.values() for c in _as_poly(phi, "a dependency").columns]))
    keep = []
    g, k = f.native(names)
    keep += k
    arr, nd = _deps(deps, names, keep)
    reason, error, refusal = C.c_int32(), C.c_int32(), _Refusal()
    h = _call("tr_geom_derivative", C.byref(g), names.index(column), 1 if through else 0, arr, nd,
              C.byref(_budget(budget)), C.byref(reason), C.byref(refusal), C.byref(error))
    if not h and error.value == -6:
        why = REASONS[reason.value]
        text = {"undeclared": "a derivative through a relation needs a declared dependency (an identity on the rows "
                              "does not make one column a function of another)",
                "dependent": f"{column!r} is a declared dependent column",
                "cycle": "the declared dependencies are cyclic"}[why]
        err = DomainError(text)
        err.reason = why
        raise err
    if not h and error.value == -1 and not through and nd:
        raise ValueError("a formal derivative takes no dependencies (through=True uses them)")
    return Polynomial._native(_made(h, error, refusal, "derivative"), names)


def dependency_check(scope, dependencies, *, budget=None):
    """each declared dependency v = phi verified exhaustively on every selected row of the scope (DISC-002's exact
    evaluation of D v - D phi): [evidence], the smallest violating row as the witness"""
    _fresh(scope, "dependency check")
    keep = []
    arr, nd = _deps(dependencies, scope.names, keep)
    out = (_mc._Evidence * max(nd, 1))()
    refusal = _Refusal()
    _check(_call("tr_geom_dependency_check", scope._handle, arr, nd, C.byref(_budget(budget)), out,
                 C.byref(refusal)), refusal, "dependency check")
    return [_mc._evidence(out[i]) for i in range(nd)]
