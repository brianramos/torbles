# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-002: Moment-Gram polynomial identity discovery, natively on the CPU or with the CUDA kernels
(torbles/math_ident.h/.c, math_ident_gpu.cu).

    from torbles import math_bank as mbk, math_ident as mi
    scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
    found = mi.discover(scope, ["x", "y", "z"], degree=2)              # plan, Gram, nullspace, identities, certificates
    found.report                                                        # basis level: rank, nullity, points, forced
    for i in found.identities: i["text"], i["status"], i["underdetermined"], i["certificate"]
    plan = mi.plan(scope, ["x", "y"], degree=3, weight="w")            # basis, moments, EXACT or MODULAR
    gram = mi.gram(scope, plan, backend="cuda")                         # or source="bank" / "lin"
    kernel = gram.kernel(backend="cuda")                                # LIN-002 (exact) or modular + CRT
    ids = mi.identities(scope, gram, kernel, known=[{(1, 0, 0): 1, (0, 2, 0): -1}])
    ids = mi.candidates(scope, ["x", "y"], [{(2, 0): 1, (0, 1): -1}], gram=gram)   # given polynomials
    ids.certify(backend="cuda"); mi.hypothesis(scope, gram, ids, 0, tolerance=1e-12, size=1000)

A Gram built under signed weights (a negative weight, or a difference of Grams) never certifies an identity through
its zero quadratic form ("sos": "signed"); a numerical near-zero quadratic form gives at most a hypothesis certificate.
Every identity is verified exhaustively on every selected row (a single violating row rejects it); an identity whose
monomials outnumber the distinct points of its variables is labelled underdetermined, never a discovery. Nothing is
computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_bank as _mbk
from . import math_certificates as _mc
from . import math_lin as _lin
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _GFP
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Plan", "Gram", "Kernel", "Identities", "Discovery", "plan", "gram", "identities", "candidates", "discover",
           "quadratic", "hypothesis", "support", "need", "MAX_VARS", "MAX_BASIS", "MAX_DEGREE", "MAX_PRIMES",
           "STATUSES", "SOS"]

MAX_VARS, MAX_BASIS, MAX_DEGREE, MAX_PRIMES, MAX_WORDS, MAX_KNOWN = 8, 63, 16, 32, 8, 16
WIDTHS = {"auto": 0, "exact": 128, "modular": -1}
REPRS = {1: "exact", 2: "modular"}
SIGNS = {1: "nonnegative", 2: "signed"}
SOURCES = {1: "fused", 2: "bank", 3: "lin", 4: "adopted", 5: "combined"}
METHODS = {1: "lin-002", 2: "modular"}
SOS = {0: None, 1: "certified", 2: "nonzero", 3: "signed", 4: "modular"}
ORIGINS = {1: "kernel", 2: "given", 3: "declared"}
STATUSES = {0: "pending", 1: "verified", 2: "rejected", 3: "undecided", 4: "unrepresentable", 5: "unresolved"}
_NEED_CPU, _NEED_DEVICE, _NEED_ADOPT = 0, 1, 2
_I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Spec(C.Structure):
    _native_ = "tr_ident_spec"
    _fields_ = [("nvars", C.c_int32), ("var", C.c_int32 * 8), ("weight", C.c_int32), ("degree", C.c_int32),
                ("nterms", C.c_int32), ("width", C.c_int32), ("nprimes", C.c_int32), ("reserved", C.c_int32),
                ("exponents", C.c_void_p)]


class _MomentView(C.Structure):
    _native_ = "tr_ident_moment_view"
    _fields_ = [(n, C.c_int32) for n in ("nvars", "nbasis", "nmoments", "weight")] + [("var", C.c_int32 * 8)] + [
        ("repr", C.c_int32), ("nprimes", C.c_int32)] + [
        (n, C.c_void_p) for n in ("basis", "bparent", "bvar", "mleft", "mright", "primes")]


class _PlanInfo(C.Structure):
    _native_ = "tr_ident_plan_info"
    _fields_ = [(n, C.c_int32) for n in ("nvars", "nbasis", "nmoments", "degree", "repr", "nprimes", "proven",
                                         "weight")] + [("primes", C.c_uint64 * 32), ("bytes", C.c_uint64)]


class _GramStates(C.Structure):
    _native_ = "tr_ident_gram_states"
    _fields_ = [("nmoments", C.c_int32), ("nslots", C.c_int32)] + [
        (n, C.c_uint64) for n in ("domain", "positive", "zero", "negative")] + [
        ("wmin", C.c_int64), ("wmax", C.c_int64), ("acc", C.c_void_p)]


class _GramInfo(C.Structure):
    _native_ = "tr_ident_gram_info"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "domain", "positive", "zero", "negative")] + [
        ("wmin", C.c_int64), ("wmax", C.c_int64), ("bytes", C.c_uint64), ("digest", C.c_uint64 * 2)] + [
        (n, C.c_int32) for n in ("nvars", "nbasis", "nmoments", "degree", "repr", "nprimes", "sign", "source",
                                 "weight", "reserved")] + [("primes", C.c_uint64 * 32)]


class _KernelInfo(C.Structure):
    _native_ = "tr_ident_kernel_info"
    _fields_ = [(n, C.c_int32) for n in ("nbasis", "rank", "nullity", "method", "exact_rank", "primes", "agreeing",
                                         "unresolved")] + [(n, C.c_uint64) for n in ("pivots", "free", "bytes")]


class _Poly(C.Structure):
    _native_ = "tr_ident_poly"
    _fields_ = [("nterms", C.c_uint64), ("exponents", C.c_void_p), ("coefficients", C.c_void_p)]


class _CandView(C.Structure):
    _native_ = "tr_ident_cand_view"
    _fields_ = [("ncand", C.c_int32), ("nvars", C.c_int32), ("var", C.c_int32 * 8), ("nterms", C.c_uint64)] + [
        (n, C.c_void_p) for n in ("offset", "support", "words", "exponents", "coefficients")]


class _Item(C.Structure):
    _native_ = "tr_ident_item"
    _fields_ = [(n, C.c_int32) for n in ("index", "origin", "lead", "vector", "nterms", "ncols")] + [
        ("support", C.c_uint32)] + [
        (n, C.c_int32) for n in ("words", "representable", "resolved", "known", "consequence", "parent",
                                 "underdetermined", "sos", "degree")] + [
        (n, C.c_uint64) for n in ("points", "forced", "domain", "bound_bits")]


class _SetInfo(C.Structure):
    _native_ = "tr_ident_set_info"
    _fields_ = [(n, C.c_int32) for n in ("nvars", "nbasis", "rank", "nullity", "exact_rank", "repr", "sign", "ncand",
                                         "nknown", "known_rank", "generators", "consequences", "known",
                                         "underdetermined", "unrepresentable", "unresolved", "labels",
                                         "reserved")] + [
        (n, C.c_uint64) for n in ("points", "forced", "domain", "bytes")]


class _Quad(C.Structure):
    _native_ = "tr_ident_quad"
    _fields_ = [(n, C.c_int32) for n in ("exact", "sos", "sign", "nlimbs")] + [
        (n, C.c_double) for n in ("numeric", "scale", "relative")] + [("residues", C.c_uint64 * 32)]


for _s, _n in ((_Spec, 72), (_MomentView, 104), (_PlanInfo, 296), (_GramStates, 64), (_GramInfo, 384),
               (_KernelInfo, 56), (_Poly, 24), (_CandView, 88), (_Item, 96), (_SetInfo, 104), (_Quad, 296)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_ident_plan_make": (vp, [vp, P(_Spec), B, R, e]),
            "tr_ident_plan_get_info": (i32, [vp, P(_PlanInfo)]),
            "tr_ident_plan_get_view": (i32, [vp, P(_MomentView)]),
            "tr_ident_plan_basis": (i32, [vp, vp, u64]),
            "tr_ident_plan_moments": (i32, [vp, vp, u64]),
            "tr_ident_plan_need": (i32, [vp, vp, i32, P(_Need)]),
            "tr_ident_plan_free": (None, [vp]),
            "tr_ident_gram_build": (vp, [vp, vp, B, R, e]),
            "tr_ident_gram_from_states": (vp, [vp, vp, P(_GramStates), B, R, e]),
            "tr_ident_gram_from_bank": (vp, [vp, vp, vp, B, R, e]),
            "tr_ident_gram_from_lin": (vp, [vp, vp, vp, B, R, e]),
            "tr_ident_gram_combine": (vp, [vp, vp, i32, B, R, e]),
            "tr_ident_gram_get_info": (i32, [vp, P(_GramInfo)]),
            "tr_ident_gram_get_states": (i32, [vp, P(_GramStates), vp, u64]),
            "tr_ident_gram_matrix": (i32, [vp, i32, vp, u64]),
            "tr_ident_gram_equal": (i32, [vp, vp]),
            "tr_ident_gram_free": (None, [vp]),
            "tr_ident_gram_system": (vp, [vp, B, R, e]),
            "tr_ident_kernel_from_lin": (vp, [vp, vp, B, R, e]),
            "tr_ident_kernel_build": (vp, [vp, B, R, e]),
            "tr_ident_kernel_get_info": (i32, [vp, P(_KernelInfo)]),
            "tr_ident_kernel_export": (i32, [vp, P(_lin._QBuf)]),
            "tr_ident_kernel_limbs": (u64, [vp]),
            "tr_ident_kernel_resolved": (u64, [vp]),
            "tr_ident_kernel_free": (None, [vp]),
            "tr_ident_propose": (vp, [vp, vp, vp, P(_Poly), i32, B, R, e]),
            "tr_ident_candidates": (vp, [vp, i32, vp, P(_Poly), i32, vp, B, R, e]),
            "tr_ident_set_get_info": (i32, [vp, P(_SetInfo)]),
            "tr_ident_set_items": (i32, [vp, P(_Item), i32]),
            "tr_ident_set_get_view": (i32, [vp, P(_CandView)]),
            "tr_ident_set_poly": (i32, [vp, i32, vp, vp, u64, P(u64)]),
            "tr_ident_set_free": (None, [vp]),
            "tr_ident_quadratic": (i32, [vp, vp, i32, P(_Quad), vp, u64]),
            "tr_ident_verify_need": (i32, [P(_mbk._View), P(_CandView), i32, P(_Need)]),
            "tr_ident_verify_view": (i32, [P(_mbk._View), P(_CandView), B, P(_mc._Evidence), R]),
            "tr_ident_certify": (i32, [vp, vp, P(_mc._Evidence), i32, B, vp, vp, vp, R]),
            "tr_ident_claim": (i32, [vp, i32, P(_mc._Claim)]),
            "tr_ident_hypothesis": (vp, [vp, vp, vp, i32, C.c_double, P(_mc._Sample), P(_Quad), P(_mc._Evidence),
                                         B, R, e]),
            "tr_ident_support": (i32, [vp, i32, vp, C.c_uint32, i32, B, P(u64), P(u64), R]),
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
    math_gpu.bind("tr_mgpu_ident_gram", C.c_int, [P(_mbk._View), P(_MomentView), P(_Budget), P(_GramStates),
                                                 P(_Refusal)])
    math_gpu.bind("tr_mgpu_ident_verify", C.c_int, [P(_mbk._View), P(_CandView), P(_Budget), P(_mc._Evidence),
                                                   P(_Refusal)])
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


def _columns(scope, variables, what="variables"):
    variables = list(variables)
    for v in variables:
        if v not in scope.names:
            raise ValueError(f"{v!r} is not a column of the scope ({', '.join(scope.names)})")
    if len(set(variables)) != len(variables):
        raise ValueError(f"{what}: distinct columns")
    return [list(scope.names).index(v) for v in variables]


def _precheck(budget, needs, what):
    raw = _Need()
    for n in needs:
        for i in range(len(RESOURCES)):
            raw.need[i] = max(raw.need[i], n.need[i])
    refusal = _Refusal()
    _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw), C.byref(refusal)), refusal, what)


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


# ------------------------------------------------------------------ polynomials (marshalling and display only)
def monomial_text(e, names):
    parts = [n if k == 1 else f"{n}^{k}" for n, k in zip(names, e) if k]
    return "*".join(parts) if parts else "1"


def poly_text(terms, names):
    """a polynomial {exponent tuple: coefficient} as text, terms as given (decreasing graded order natively)"""
    out = []
    for e, c in terms:
        m = monomial_text(e, names)
        mag = abs(c)
        body = m if (mag == 1 and m != "1") else (str(mag) if m == "1" else f"{mag}*{m}")
        out.append(("- " if c < 0 else "+ ") + body)
    if not out:
        return "0"
    text = " ".join(out)
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def _poly_arrays(poly, names):
    """exponents (n x k uint8) and int64 coefficients from {tuple: int}, [(coef, {name: exp})] or [(tuple, coef)]"""
    k = len(names)
    items = poly.items() if isinstance(poly, dict) else list(poly)
    rows, coefs = [], []
    for item in items:
        a, b = item
        if isinstance(a, tuple) and not isinstance(b, dict):
            e, c = list(a), b
        elif isinstance(b, dict):
            c, e = a, [0] * k
            for name, p in b.items():
                if name not in names:
                    raise ValueError(f"{name!r} is not a variable ({', '.join(names)})")
                e[names.index(name)] += int(p)
        else:
            raise ValueError("a term is (exponent tuple, coefficient) or (coefficient, {variable: exponent})")
        if len(e) != k or any(not 0 <= int(x) <= 255 for x in e):
            raise ValueError(f"a term has {k} exponents in 0..255")
        if isinstance(c, bool) or not isinstance(c, int):
            if isinstance(c, Fraction) and c.denominator == 1:
                c = int(c)
            else:
                raise TypeError("coefficients are exact integers (clear denominators first)")
        if not _I64[0] <= c <= _I64[1]:
            raise OverflowError("a coefficient does not fit int64")
        rows.append([int(x) for x in e])
        coefs.append(int(c))
    exps = np.ascontiguousarray(np.array(rows, dtype=np.uint8).reshape(len(rows), k))
    co = np.ascontiguousarray(coefs, dtype=np.int64)
    return exps, co


def _polys(polys, names):
    arr = (_Poly * max(len(polys), 1))()
    keep = []
    for i, p in enumerate(polys):
        exps, co = _poly_arrays(p, names)
        arr[i] = _Poly(len(co), exps.ctypes.data if len(co) else None, co.ctypes.data if len(co) else None)
        keep += [exps, co]
    return arr, keep


# ------------------------------------------------------------------ plans
class Plan:
    """The canonical basis (increasing graded lexicographic order), the moments m_i m_j, and the representation
    ("exact": every moment proven to fit 128 bits; "modular": residues modulo the recorded primes)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_ident.plan(scope, variables, degree=...)")

    @classmethod
    def _wrap(cls, handle, scope, variables, weight):
        self = object.__new__(cls)
        self._handle, self.scope, self.variables, self.weight = handle, scope, tuple(variables), weight
        i = _PlanInfo()
        _check(_call("tr_ident_plan_get_info", handle, C.byref(i)), what="plan")
        self._info = i
        k, nb, nm = int(i.nvars), int(i.nbasis), int(i.nmoments)
        b = np.zeros(max(nb * k, 1), np.uint8)
        _check(_call("tr_ident_plan_basis", handle, b.ctypes.data, b.size), what="plan basis")
        m = np.zeros(max(nm * k, 1), np.uint8)
        _check(_call("tr_ident_plan_moments", handle, m.ctypes.data, m.size), what="plan moments")
        self.basis = [tuple(int(x) for x in b[j * k:(j + 1) * k]) for j in range(nb)]
        self.moments = [tuple(int(x) for x in m[j * k:(j + 1) * k]) for j in range(nm)]
        return self

    @property
    def info(self):
        i = self._info
        return dict(variables=list(self.variables), weight=self.weight, basis=int(i.nbasis), moments=int(i.nmoments),
                    degree=int(i.degree), repr=REPRS[i.repr], proven=bool(i.proven), primes=self.primes,
                    bytes=int(i.bytes))

    @property
    def repr(self):
        return REPRS[self._info.repr]

    @property
    def primes(self):
        return [int(self._info.primes[q]) for q in range(self._info.nprimes)]

    def monomial(self, j):
        return monomial_text(self.basis[j], self.variables)

    def view(self):
        v = _MomentView()
        _check(_call("tr_ident_plan_get_view", self._handle, C.byref(v)), what="plan view")
        return v

    def need(self, backend="cpu"):
        """the needs of building the Gram: {resource: need} (cuda: the device copies and the adopted Gram)"""
        if backend == "cpu":
            return _needs_dict(self._need(_NEED_CPU))
        if backend == "cuda":
            dev, adopt = self._need(_NEED_DEVICE), self._need(_NEED_ADOPT)
            return {k: max(v, _needs_dict(adopt).get(k, 0)) for k, v in _needs_dict(dev).items()}
        raise ValueError("backend is 'cpu' or 'cuda'")

    def _need(self, target):
        out = _Need()
        _check(_call("tr_ident_plan_need", self._handle, self.scope._handle, target, C.byref(out)), what="plan needs")
        return out

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_ident_plan_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Plan({list(self.variables)}, basis={len(self.basis)}, moments={len(self.moments)}, {self.repr})"


def plan(scope, variables, degree=None, *, basis=None, weight=None, width="auto", primes=None, budget=None):
    """the plan of a Gram: every monomial of total degree <= degree over the variables (or an explicit basis of
    exponent tuples), an optional weight column, width "auto" | "exact" (refuses an unproven plan: OverflowError) |
    "modular" (forces residues), primes: the modular representation's prime count"""
    _fresh(scope, "plan")
    cols = _columns(scope, variables)
    if not 1 <= len(cols) <= MAX_VARS:
        if len(cols) > MAX_VARS:
            raise UnsupportedError(f"{len(cols)} variables: at most {MAX_VARS} (an FND-007 claim reads 8 columns)")
        raise ValueError("give at least one variable")
    if width not in WIDTHS:
        raise ValueError(f"width is one of {', '.join(WIDTHS)}")
    if weight is not None and weight in variables:
        raise ValueError("the weight column is not a variable")
    s = _Spec()
    s.nvars = len(cols)
    for i in range(8):
        s.var[i] = cols[i] if i < len(cols) else -1
    s.weight = _columns(scope, [weight])[0] if weight is not None else -1
    s.width, s.nprimes = WIDTHS[width], int(primes or 0)
    keep = None
    if basis is not None:
        rows = [tuple(int(x) for x in t) for t in basis]
        if any(len(t) != len(cols) or any(not 0 <= x <= 255 for x in t) for t in rows):
            raise ValueError(f"basis terms are tuples of {len(cols)} exponents in 0..255")
        keep = np.ascontiguousarray(np.array(rows, dtype=np.uint8).reshape(len(rows), len(cols)))
        s.nterms, s.exponents = len(rows), keep.ctypes.data if len(rows) else None
    else:
        if degree is None or int(degree) < 0:
            raise ValueError("give degree >= 0 or an explicit basis")
        s.degree = int(degree)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_ident_plan_make", scope._handle, C.byref(s), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Plan._wrap(_made(h, error, refusal, "plan"), scope, variables, weight)


# ------------------------------------------------------------------ Grams
def _i128_words(lo, hi):
    return (int(hi) << 64) | (int(lo) & ((1 << 64) - 1))


class Gram:
    """The exact Gram matrix G = sum w_r m(x_r) m(x_r)^T of a plan (or its residues), with its domain: rows, weights
    by sign, extremes, the sign class ("nonnegative" / "signed") and the source."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_ident.gram(scope, plan)")

    @classmethod
    def _wrap(cls, handle, scope, plan_, backend, device):
        self = object.__new__(cls)
        self._handle, self.scope, self.plan, self.backend, self.device = handle, scope, plan_, backend, device
        return self

    def _info(self):
        i = _GramInfo()
        _check(_call("tr_ident_gram_get_info", self._open(), C.byref(i)), what="gram")
        return i

    def _open(self):
        if not self._handle:
            raise ValueError("the Gram is closed")
        return self._handle

    @property
    def info(self):
        i = self._info()
        return dict(rows=int(i.rows), selected=int(i.selected), domain=int(i.domain), positive=int(i.positive),
                    zero=int(i.zero), negative=int(i.negative), wmin=int(i.wmin), wmax=int(i.wmax), bytes=int(i.bytes),
                    basis=int(i.nbasis), moments=int(i.nmoments), degree=int(i.degree), repr=REPRS[i.repr],
                    primes=[int(i.primes[q]) for q in range(i.nprimes)], sign=SIGNS[i.sign],
                    source=SOURCES[i.source], backend=self.backend, device=self.device)

    @property
    def sign(self):
        return SIGNS[self._info().sign]

    @property
    def repr(self):
        return REPRS[self._info().repr]

    def matrix(self, prime=None):
        """the B x B matrix: exact ints (exact representation), or the residues modulo the prime-th recorded prime"""
        i = self._info()
        nb = int(i.nbasis)
        if prime is None:
            out = np.zeros(max(2 * nb * nb, 1), np.uint64)
            _check(_call("tr_ident_gram_matrix", self._open(), -1, out.ctypes.data, out.size), what="gram matrix")
            flat = [_i128_words(out[2 * t], out[2 * t + 1]) for t in range(nb * nb)]
            flat = [v - (1 << 128) if v >> 127 else v for v in flat]
        else:
            out = np.zeros(max(nb * nb, 1), np.uint64)
            _check(_call("tr_ident_gram_matrix", self._open(), int(prime), out.ctypes.data, out.size), what="gram residues")
            flat = [int(v) for v in out[:nb * nb]]
        return [flat[r * nb:(r + 1) * nb] for r in range(nb)]

    def equal(self, other):
        return bool(_check(_call("tr_ident_gram_equal", self._open(), other._open()), what="gram comparison"))

    def __eq__(self, other):
        return isinstance(other, Gram) and self.equal(other)

    __hash__ = None

    def combine(self, other, sign=1, budget=None):
        """self + sign * other (one plan): merges of disjoint selections; sign -1 (deleted rows) is SIGNED"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_ident_gram_combine", self._open(), other._open(), int(sign), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Gram._wrap(_made(h, error, refusal, "gram combination"), None, self.plan, "cpu", "native CPU")

    def __add__(self, other):
        return self.combine(other, 1)

    def __sub__(self, other):
        return self.combine(other, -1)

    def kernel(self, *, backend="cpu", budget=None, max_rounds=8):
        """the exact rank and nullspace basis: exact Gram through LIN-002 (backend "cuda": its modular elimination
        kernel), modular Gram from the recorded residues (CPU: CRT and rational reconstruction)"""
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend is 'cpu' or 'cuda'")
        error, refusal = C.c_int32(), _Refusal()
        if self.repr == "modular" or backend == "cpu":
            h = _call("tr_ident_kernel_build", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            return Kernel._wrap(_made(h, error, refusal, "nullspace"), self, "cpu", "native CPU")
        b = _call("tr_ident_gram_system", self._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        batch = _lin.Batch(_made(b, error, refusal, "Gram system"))
        results, label, device = batch.solve("cuda", budget=budget, max_rounds=max_rounds)
        h = _call("tr_ident_kernel_from_lin", self._open(), results[0]._h, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Kernel._wrap(_made(h, error, refusal, "nullspace"), self, label, device)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_ident_gram_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Gram(closed)"
        i = self.info
        return f"Gram(basis={i['basis']}, rows={i['domain']}, {i['repr']}, {i['sign']}, {i['source']})"


def gram(scope, plan_, *, backend="cpu", source="fused", bank=None, lin=None, budget=None):
    """the Gram of a plan over its scope: source "fused" (one pass; backend "cuda": the CUDA kernel), "bank" (a
    FND-004 bank of the scope in joint mode holding every moment: bank=) or "lin" (LIN-001 listwise moments of a
    degree <= 1 plan: lin=)"""
    _fresh(scope, "gram")
    error, refusal = C.c_int32(), _Refusal()
    if source == "bank":
        if bank is None:
            raise ValueError("source='bank' needs bank=")
        h = _call("tr_ident_gram_from_bank", scope._handle, plan_._handle, bank._handle, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Gram._wrap(_made(h, error, refusal, "gram from a bank"), scope, plan_, "cpu", "native CPU")
    if source == "lin":
        if lin is None:
            raise ValueError("source='lin' needs lin=")
        h = _call("tr_ident_gram_from_lin", scope._handle, plan_._handle, lin._handle, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Gram._wrap(_made(h, error, refusal, "gram from LIN-001 moments"), scope, plan_, "cpu", "native CPU")
    if source != "fused":
        raise ValueError("source is 'fused', 'bank' or 'lin'")
    if backend == "cpu":
        h = _call("tr_ident_gram_build", scope._handle, plan_._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Gram._wrap(_made(h, error, refusal, "gram"), scope, plan_, "cpu", "native CPU")
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    gpu = _gpu()
    _precheck(budget, [plan_._need(_NEED_DEVICE), plan_._need(_NEED_ADOPT)], "gram on the device")
    view, mv = scope.view(), plan_.view()
    slots = int(mv.nprimes) if mv.repr == 2 else 1
    acc = np.zeros(max(int(mv.nmoments) * slots * 2, 2), np.int64)
    st = _GramStates(int(mv.nmoments), slots, 0, 0, 0, 0, 0, 0, acc.ctypes.data)
    gpu.run("tr_mgpu_ident_gram", C.byref(view), C.byref(mv), C.byref(_budget(budget)), C.byref(st), C.byref(refusal),
            refusal=refusal, what="gram on the device")
    h = _call("tr_ident_gram_from_states", scope._handle, plan_._handle, C.byref(st), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    label, device = _device_label()
    return Gram._wrap(_made(h, error, refusal, "gram states"), scope, plan_, label, device)


# ------------------------------------------------------------------ the nullspace
class Kernel:
    """The rank and the canonical nullspace basis (one vector per free monomial, coefficient 1 there, other entries
    at smaller pivot monomials only); exact_rank: LIN-002 verified G c = 0 exactly (a modular kernel: a lower bound
    of the rank until its identities verify)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Gram.kernel()")

    @classmethod
    def _wrap(cls, handle, gram_, backend, device):
        self = object.__new__(cls)
        self._handle, self.gram, self.backend, self.device = handle, gram_, backend, device
        i = _KernelInfo()
        _check(_call("tr_ident_kernel_get_info", handle, C.byref(i)), what="nullspace")
        self.rank, self.nullity, self.nbasis = int(i.rank), int(i.nullity), int(i.nbasis)
        self.method, self.exact_rank = METHODS[i.method], bool(i.exact_rank)
        self.primes, self.agreeing, self.unresolved = int(i.primes), int(i.agreeing), int(i.unresolved)
        self.pivots = [j for j in range(self.nbasis) if (i.pivots >> j) & 1]
        self.free = [j for j in range(self.nbasis) if (i.free >> j) & 1]
        self.bytes = int(i.bytes)
        resolved = int(_call("tr_ident_kernel_resolved", handle))
        self.resolved = [bool(resolved >> f & 1) for f in range(self.nullity)]
        self.vectors = self._vectors()
        return self

    def _vectors(self):
        n = self.nullity * self.nbasis
        if not n:
            return []
        nl = int(_call("tr_ident_kernel_limbs", self._handle))
        sign, nlen, dlen = np.zeros(n, np.int8), np.zeros(n, np.uint32), np.zeros(n, np.uint32)
        limbs = np.zeros(max(nl, 1), np.uint64)
        q = _lin._QBuf(n, max(nl, 1), sign.ctypes.data, nlen.ctypes.data, dlen.ctypes.data, limbs.ctypes.data)
        _check(_call("tr_ident_kernel_export", self._handle, C.byref(q)), what="nullspace vectors")
        flat = _lin._decode(sign, nlen, dlen, limbs, n)
        rows = [flat[r * self.nbasis:(r + 1) * self.nbasis] for r in range(self.nullity)]
        return [row if ok else None for row, ok in zip(rows, self.resolved)]

    @property
    def info(self):
        return dict(rank=self.rank, nullity=self.nullity, basis=self.nbasis, method=self.method,
                    exact_rank=self.exact_rank, primes=self.primes, agreeing=self.agreeing, unresolved=self.unresolved,
                    pivots=self.pivots, free=self.free, bytes=self.bytes, backend=self.backend, device=self.device)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_ident_kernel_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Kernel(rank={self.rank}, nullity={self.nullity}, {self.method}, exact_rank={self.exact_rank})"


# ------------------------------------------------------------------ identities
class Identities:
    """The identities of a kernel (or given polynomials): exact labels, then exhaustive verification and FND-007
    certificates (certify)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_ident.identities(...) or math_ident.candidates(...)")

    @classmethod
    def _wrap(cls, handle, scope, variables, gram_, kernel_):
        self = object.__new__(cls)
        self._handle, self.scope, self.variables, self.gram, self.kernel = handle, scope, tuple(variables), gram_, kernel_
        si = _SetInfo()
        _check(_call("tr_ident_set_get_info", handle, C.byref(si)), what="identities")
        self._si = si
        n = int(si.ncand)
        items = (_Item * max(n, 1))()
        _check(_call("tr_ident_set_items", handle, items, n), what="identities")
        self._items = [items[i] for i in range(n)]
        self.status, self.certificates, self.codes, self.evidence = [None] * n, [None] * n, [None] * n, [None] * n
        self.verified_by = None
        return self

    def __len__(self):
        return len(self._items)

    def _open(self):
        if not self._handle:
            raise ValueError("the identities are closed")
        return self._handle

    def terms(self, i):
        """identity i's terms [(exponent tuple, coefficient)] in decreasing graded order (None: not representable)"""
        it = self._items[i]
        if not it.representable or not it.resolved:
            return None
        k = len(self.variables)
        cap = max(int(it.nterms), 1)
        e, c, n = np.zeros(cap * k, np.uint8), np.zeros(cap, np.int64), C.c_uint64()
        _check(_call("tr_ident_set_poly", self._open(), int(i), e.ctypes.data, c.ctypes.data, cap, C.byref(n)),
               what="identity terms")
        return [(tuple(int(x) for x in e[t * k:(t + 1) * k]), int(c[t])) for t in range(n.value)]

    def claim(self, i):
        """identity i's FND-007 POLYNOMIAL claim (policy skip, carrier i128) as a math_certificates.Claim"""
        terms = self.terms(i)
        if terms is None:
            raise UnsupportedError("the identity is not representable with int64 coefficients")
        sup = [v for q, v in enumerate(self.variables) if self._items[i].support >> q & 1]
        idx = [self.variables.index(v) for v in sup]
        return _mc.polynomial({tuple(e[q] for q in idx): c for e, c in terms}, sup, nulls="skip")

    def item(self, i):
        it = self._items[i]
        terms = self.terms(i)
        sup = [v for q, v in enumerate(self.variables) if it.support >> q & 1]
        plan_ = self.gram.plan if self.gram is not None else None
        lead = plan_.monomial(it.lead) if (plan_ is not None and it.lead >= 0 and it.origin == 1) else (
            monomial_text(terms[0][0], self.variables) if terms else None)
        exact = None
        if self.kernel is not None and it.vector >= 0 and self.kernel.vectors and self.kernel.vectors[it.vector] is not None:
            exact = {plan_.basis[j]: v for j, v in enumerate(self.kernel.vectors[it.vector]) if v}
        status = self.status[i]
        d = dict(index=int(it.index), origin=ORIGINS[it.origin], lead=lead, terms=terms,
                 text=poly_text(terms, self.variables) if terms is not None else None, exact=exact,
                 nterms=int(it.nterms), support=sup, degree=int(it.degree), points=int(it.points),
                 forced=int(it.forced), domain=int(it.domain), underdetermined=bool(it.underdetermined),
                 known=bool(it.known), consequence=bool(it.consequence),
                 parent=int(it.parent) if it.parent >= 0 else ("declared" if it.parent == -2 else None),
                 sos=SOS[it.sos], words=int(it.words), bound_bits=int(it.bound_bits),
                 representable=bool(it.representable), resolved=bool(it.resolved),
                 status=STATUSES[status] if status is not None else "pending", evidence=self.evidence[i],
                 certificate=self.certificates[i], certificate_code=self.codes[i])
        d["discovery"] = (d["status"] == "verified" and d["origin"] == "kernel" and not d["underdetermined"]
                          and not d["known"] and not d["consequence"])
        return d

    def __iter__(self):
        return (self.item(i) for i in range(len(self)))

    def __getitem__(self, i):
        return self.item(i)

    @property
    def report(self):
        """the basis level: rank, nullity, exact_rank, points S (Gram rows of nonzero total weight), forced
        max(0, B - S), generators / consequences / known, underdetermined, unrepresentable, unresolved"""
        s = self._si
        return dict(variables=list(self.variables), basis=int(s.nbasis), rank=int(s.rank), nullity=int(s.nullity),
                    exact_rank=bool(s.exact_rank), repr=REPRS.get(s.repr), sign=SIGNS.get(s.sign),
                    candidates=int(s.ncand), declared=int(s.nknown), declared_rank=int(s.known_rank),
                    generators=int(s.generators), consequences=int(s.consequences), known=int(s.known),
                    underdetermined=int(s.underdetermined), unrepresentable=int(s.unrepresentable),
                    unresolved=int(s.unresolved), labels_exact=bool(s.labels), points=int(s.points),
                    forced=int(s.forced), domain=int(s.domain), bytes=int(s.bytes))

    def view(self):
        v = _CandView()
        _check(_call("tr_ident_set_get_view", self._open(), C.byref(v)), what="candidates view")
        return v

    def verify(self, *, backend="cpu", budget=None):
        """the exhaustive evidence of every identity over every selected row of the scope: [dict]"""
        _fresh(self.scope, "verification")
        evs, label, device = self._evidence(backend, budget)
        return [_mc._evidence(e, label, device) for e in evs]

    def _evidence(self, backend, budget):
        view, cv = self.scope.view(), self.view()
        n = len(self)
        out = (_mc._Evidence * max(n, 1))()
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_ident_verify_view", C.byref(view), C.byref(cv), C.byref(_budget(budget)), out,
                         C.byref(refusal)), refusal, "verification")
            return list(out[:n]), "cpu", "native CPU"
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        need = _Need()
        _check(_call("tr_ident_verify_need", C.byref(view), C.byref(cv), _NEED_DEVICE, C.byref(need)), what="needs")
        _precheck(budget, [need], "verification on the device")
        _gpu().run("tr_mgpu_ident_verify", C.byref(view), C.byref(cv), C.byref(_budget(budget)), out, C.byref(refusal),
                   refusal=refusal, what="verification on the device")
        label, device = _device_label()
        return list(out[:n]), label, device

    def certify(self, *, backend="cpu", budget=None):
        """verify every identity exhaustively (backend "cuda": the kernel; the CPU library adopts its evidence) and
        make an FND-007 snapshot certificate of each verified one; returns self"""
        _fresh(self.scope, "certification")
        evs, label, device = self._evidence(backend, budget)
        n = len(self)
        arr = (_mc._Evidence * max(n, 1))(*evs) if n else (_mc._Evidence * 1)()
        status, codes = np.zeros(max(n, 1), np.int32), np.zeros(max(n, 1), np.int32)
        certs = (C.c_void_p * max(n, 1))()
        refusal = _Refusal()
        _check(_call("tr_ident_certify", self._open(), self.scope._handle, arr, 1 if backend == "cuda" else 0,
                     C.byref(_budget(budget)), status.ctypes.data, certs, codes.ctypes.data, C.byref(refusal)),
               refusal, "certification")
        names = self.scope.names
        for i in range(n):
            self.status[i] = int(status[i])
            self.codes[i] = int(codes[i])
            self.evidence[i] = _mc._evidence(evs[i], label, device)
            self.certificates[i] = (_mc.Certificate._wrap(certs[i], names, (), label, device) if certs[i] else None)
        self.verified_by = label
        return self

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_ident_set_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        r = self.report
        return f"Identities({len(self)}, rank={r['rank']}, nullity={r['nullity']}, points={r['points']})"


def identities(scope, gram_, kernel_, *, known=(), budget=None):
    """the identities of a kernel labelled over scope (the scope they are verified on; its columns of the plan's
    names), with declared relations known=[polynomial over the plan's variables] appended (restated definitions)"""
    _fresh(scope, "identities")
    plan_ = gram_.plan
    if list(scope.names) != list(plan_.scope.names):
        for v in plan_.variables:
            if v not in scope.names or scope.names.index(v) != plan_.scope.names.index(v):
                raise DomainError("the verification scope must hold the plan's variables at the same columns")
    if len(known) > MAX_KNOWN:
        raise UnsupportedError(f"at most {MAX_KNOWN} declared relations")
    arr, keep = _polys(list(known), list(plan_.variables))
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_ident_propose", scope._handle, gram_._open(), kernel_._handle, arr, len(known),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Identities._wrap(_made(h, error, refusal, "identities"), scope, plan_.variables, gram_, kernel_)


def candidates(scope, variables, polys, *, gram=None, budget=None):
    """given polynomials over variables as candidate identities (gram: optional, the same variables in the same
    order, for their sos verdicts)"""
    _fresh(scope, "candidates")
    cols = _columns(scope, variables)
    if not 1 <= len(cols) <= MAX_VARS:
        raise UnsupportedError(f"1..{MAX_VARS} variables")
    arr, keep = _polys(list(polys), list(variables))
    var = (C.c_int32 * 8)(*(cols + [-1] * (8 - len(cols))))
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_ident_candidates", scope._handle, len(cols), var, arr, len(polys),
              gram._open() if gram is not None else None, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Identities._wrap(_made(h, error, refusal, "candidates"), scope, variables, gram, None)


class Discovery:
    """plan, gram, kernel and identities of one discovery run (certified when certify=True)"""

    def __init__(self, plan_, gram_, kernel_, ids):
        self.plan, self.gram, self.kernel, self.identities = plan_, gram_, kernel_, ids

    @property
    def report(self):
        r = dict(self.identities.report)
        r.update(gram=self.gram.info, kernel=self.kernel.info)
        return r

    def discoveries(self):
        return [i for i in self.identities if i["discovery"]]

    def __repr__(self):
        return f"Discovery({self.identities!r})"


def discover(scope, variables, degree=None, *, basis=None, weight=None, known=(), width="auto", primes=None,
             backend="cpu", certify=True, budget=None):
    """one run: plan, Gram, nullspace, identities and (certify) their exhaustive verification and certificates"""
    p = plan(scope, variables, degree, basis=basis, weight=weight, width=width, primes=primes, budget=budget)
    g = gram(scope, p, backend=backend, budget=budget)
    k = g.kernel(backend=backend, budget=budget)
    ids = identities(scope, g, k, known=known, budget=budget)
    if certify:
        ids.certify(backend=backend, budget=budget)
    return Discovery(p, g, k, ids)


def quadratic(gram_, ids, i):
    """c^T G c of identity i: exact (an exact Gram: int and the sos verdict), residues (modular), and the binary64
    value a floating miner computes (numeric, scale, relative: APPROXIMATE, only ever a hypothesis)"""
    q = _Quad()
    cap = 64
    while True:
        limbs = np.zeros(cap, np.uint64)
        rc = _call("tr_ident_quadratic", gram_._open(), ids._open(), int(i), C.byref(q), limbs.ctypes.data, cap)
        if rc == -1 and q.exact and q.nlimbs > cap:
            cap = int(q.nlimbs)
            continue
        _check(rc, what="quadratic form")
        break
    value = None
    if q.exact:
        mag = sum(int(limbs[j]) << (64 * j) for j in range(q.nlimbs))
        value = -mag if q.sign < 0 else mag
    primes = gram_.info["primes"]
    return dict(exact=value, sos=SOS[q.sos], residues=[int(q.residues[j]) for j in range(len(primes))] if not q.exact else None,
                numeric=float(q.numeric), scale=float(q.scale), relative=float(q.relative), approximate=True)


def hypothesis(scope, gram_, ids, i, *, tolerance, size, seed=0, method="stride", budget=None):
    """an FND-007 HYPOTHESIS certificate of identity i when its numerical quadratic form is within tolerance
    (relative), checked on a sample of rows; never exact. DomainError when not near zero (or modular);
    ClaimViolated when the sample refutes it."""
    _fresh(scope, "hypothesis")
    if method not in _mc.SAMPLERS:
        raise ValueError(f"method is one of {', '.join(_mc.SAMPLERS)}")
    sample = _mc._Sample(_mc.SAMPLERS[method], 0, int(size), int(seed) & _mc.NONE, 0, 0)
    q, ev = _Quad(), _mc._Evidence()
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_ident_hypothesis", scope._handle, gram_._open(), ids._open(), int(i), float(tolerance),
              C.byref(sample), C.byref(q), C.byref(ev), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    if not h and error.value == -6 and not ev.rows:
        err = DomainError(f"the numerical quadratic form is not within the tolerance (relative {q.relative!r})")
        err.relative = float(q.relative)
        raise err
    cert = _mc._made(h, error, refusal, "hypothesis", ev)
    return _mc.Certificate._wrap(cert, scope.names)


def support(scope, variables, *, weight=None, budget=None):
    """(distinct points, rows): the exact distinct tuples of the variables over the selected rows where they (and
    the weight) are present; with a weight, points of nonzero total weight only"""
    _fresh(scope, "support")
    cols = _columns(scope, variables)
    if not 1 <= len(cols) <= MAX_VARS:
        raise UnsupportedError(f"1..{MAX_VARS} variables")
    var = (C.c_int32 * 8)(*(cols + [-1] * (8 - len(cols))))
    w = _columns(scope, [weight])[0] if weight is not None else -1
    pts, rows, refusal = C.c_uint64(), C.c_uint64(), _Refusal()
    _check(_call("tr_ident_support", scope._handle, len(cols), var, (1 << len(cols)) - 1, w, C.byref(_budget(budget)),
                 C.byref(pts), C.byref(rows), C.byref(refusal)), refusal, "distinct support")
    return int(pts.value), int(rows.value)


def need(scope, variables, degree=None, **options):
    """the needs of a plan's Gram build: {resource: need}"""
    backend = options.pop("backend", "cpu")
    return plan(scope, variables, degree, **options).need(backend)
