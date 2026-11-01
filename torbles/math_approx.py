# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""UNC-002: explicit approximations that carry their error (declared norm, assumptions, error budget, provenance and
the error spent term by term), natively on the CPU and with the CUDA row kernel (torbles/math_approx.c,
math_approx_gpu.cu).

    from fractions import Fraction
    from torbles import math_approx as ma
    f = ma.series("exp", 0, 1, degree=8)                       # Chebyshev (economized Taylor), certified sup-norm bound
    f.error, f.terms, f.value(Fraction(1, 3)), f.info          # exact Fractions; every error term with its origin
    g = ma.series("sin", 0, 1, degree=8, error_budget=Fraction(1, 10**9))   # NotConvergedError if not achieved
    h = (f * g + 3 * f).quantize(40)                           # errors propagate: sums add, products scale, no term dropped
    ma.series("log", 1, 2, degree=10)(f)                       # composition through the outer Lipschitz bound
    fl = f.to_float(); fl.doubles([0.25, 0.5])                 # binary64 Horner values with a certified bound
    lr = ma.lowrank(A, rank=2, method="cross")                 # exact residual; method="sketch", seed=1: probabilistic
    ev = f.evaluate(scope, "x", backend="cuda", fbits=40)      # per-row enclosures, exact group sums
    bits, undecided, stats = ma.filter(state, scope, "x", f, ">", 2)   # approximate predicates in UNC-001
    f.authorize_exact()                                        # NotAuthorizedError (reason "approximate"): never exact
    ma.relation(f, scope, "x", "y", tolerance=Fraction(1, 10**4))   # an FND-007 snapshot of an approximate claim

Nothing is computed in Python: rationals are marshalled as FND-009 big rationals. Opt-in: `import torbles` does not
load it.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, NotConvergedError, UnsupportedError, check as _check

__all__ = ["Approximation", "RowEvaluation", "series", "poly", "lowrank", "filter", "relation", "FUNCTIONS", "BASES",
           "NORMS", "LABELS", "TERMS", "ASSUMPTIONS", "MAX_DEGREE", "MAX_TERMS", "MAX_DIM", "MAX_PROBES"]

MAX_DEGREE, MAX_TERMS, MAX_DIM, MAX_PROBES, MAX_TAYLOR, MAX_FBITS = 64, 32, 64, 64, 160, 64
KINDS = {1: "series", 2: "lowrank", 3: "float"}
FUNCTIONS = {"exp": 1, "log": 2, "sin": 3, "cos": 4, "sqrt": 5, "recip": 6}
_FUNCTION_NAMES = {1: "exp", 2: "log", 3: "sin", 4: "cos", 5: "sqrt", 6: "recip", 7: "poly", 8: "composed",
                   9: "matrix"}
BASES = {"taylor": 1, "chebyshev": 2}
_BASIS_NAMES = {0: None, 1: "taylor", 2: "chebyshev", 3: "monomial_x"}
METHODS = {"cross": 1, "sketch": 2}
NORMS = {"abs": 1, "rel": 2, "l2": 3, "frobenius": 4, "spectral": 5, "max": 6}
_NORM_NAMES = {v: k for k, v in NORMS.items()}
LABELS = {5: "certified", 6: "probabilistic"}
PROVENANCE = {1: "deterministic", 2: "randomized"}
TERMS = {1: "remainder", 2: "coefficient", 3: "economization", 4: "quantization", 5: "rounding", 6: "truncation",
         7: "residual", 8: "probabilistic"}
ASSUMPTIONS = {"domain": 1, "derivative": 2, "lipschitz": 4, "range": 8, "independent": 16, "no_overflow": 32}
OPERATIONS = {0: None, 1: "add", 2: "mul", 3: "scale", 4: "compose", 5: "quantize", 6: "float"}
ROW_STATUS = {0: "ok", 1: "null", 2: "domain", 3: "overflow", 4: "unselected"}
OPS = {">": 1, ">=": 2, "<": 3, "<=": 4}
_Q = {"error": 1, "term": 2, "budget": 3, "a": 4, "b": 5, "coef": 6, "lipschitz": 7, "derivative": 8,
      "range_lo": 9, "range_hi": 10, "relative": 11, "l2": 12, "frobenius2": 13, "frobenius": 14, "spectral": 15,
      "maxentry": 16, "failure": 17, "probe": 18, "entry": 19}
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Info(C.Structure):
    _native_ = "tr_approx_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "function", "basis", "method", "norm", "label", "provenance")] + [
        ("assumptions", C.c_uint32)] + [(n, C.c_int32) for n in (
            "degree", "nterms", "taylor", "quantized", "rows", "cols", "rank", "probes", "has_budget", "has_relative",
            "exact", "operation")] + [
        ("seed", C.c_uint64), ("id", C.c_uint64), ("origins", C.c_uint64 * 2), ("bytes", C.c_uint64)] + [
        (n, C.c_double) for n in ("error_hi", "budget_hi", "failure_hi", "range_lo", "range_hi")]


class _Term(C.Structure):
    _native_ = "tr_approx_term"
    _fields_ = [("kind", C.c_int32), ("scaled", C.c_int32), ("origin", C.c_uint64), ("hi", C.c_double)]


class _IV(C.Structure):
    _native_ = "tr_math_interval"
    _fields_ = [("lo", C.c_double), ("hi", C.c_double)]


class _EvalView(C.Structure):
    _native_ = "tr_approx_eval_view"
    _fields_ = [("mode", C.c_int32), ("basis", C.c_int32), ("ncoef", C.c_int32), ("reserved", C.c_int32),
                ("mid", _IV), ("rad", _IV), ("dom", _IV), ("range", _IV), ("err", C.c_double), ("lip", C.c_double),
                ("coef", C.c_void_p)]


class _Group(C.Structure):
    _native_ = "tr_approx_group"
    _fields_ = [("lo", C.c_int64 * 2), ("hi", C.c_int64 * 2)] + [
        (n, C.c_uint64) for n in ("rows", "ok", "nulls", "refused")]


class _EvalStats(C.Structure):
    _native_ = "tr_approx_eval_stats"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "ok", "nulls", "domain", "overflow")] + [
        ("fbits", C.c_int32), ("groups", C.c_int32)]


class _FilterStats(C.Structure):
    _native_ = "tr_approx_filter_stats"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "groups_all", "groups_none", "groups_unknown",
                                          "groups_empty", "groups_outside", "rows_proven", "rows_evaluated",
                                          "rows_true", "rows_false", "rows_null", "rows_undecided", "rows_domain")] + [
        ("error_used", C.c_double), ("degree", C.c_int32), ("reserved", C.c_int32)]


for _s, _n in ((_Info, 160), (_Term, 24), (_EvalView, 104), (_Group, 64), (_EvalStats, 56), (_FilterStats, 128)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _mc():
    from . import math_certificates
    return math_certificates


def _carriers():
    from . import math_carriers
    return math_carriers


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        mc = _mc()
        from .math_unc import _Q as _UQ
        sig = {
            "tr_approx_series": (vp, [i32, vp, vp, i32, i32, i32, i32, vp, B, R, e]),
            "tr_approx_poly": (vp, [vp, i32, vp, vp, B, R, e]),
            "tr_approx_lowrank": (vp, [vp, i32, i32, i32, i32, u64, i32, i32, vp, B, R, e]),
            "tr_approx_add": (vp, [vp, vp, vp, B, R, e]),
            "tr_approx_mul": (vp, [vp, vp, vp, B, R, e]),
            "tr_approx_scale": (vp, [vp, vp, vp, B, R, e]),
            "tr_approx_compose": (vp, [vp, vp, vp, B, R, e]),
            "tr_approx_quantize": (vp, [vp, i32, vp, B, R, e]),
            "tr_approx_float": (vp, [vp, vp, B, R, e]),
            "tr_approx_free": (None, [vp]),
            "tr_approx_get_info": (i32, [vp, P(_Info)]),
            "tr_approx_get_term": (i32, [vp, i32, P(_Term)]),
            "tr_approx_get_q": (vp, [vp, i32, i32]),
            "tr_approx_get_pivots": (i32, [vp, vp, vp, i32]),
            "tr_approx_get_sketch": (i32, [vp, vp, i32, vp, i32]),
            "tr_approx_value": (vp, [vp, vp, i32, B, R, e]),
            "tr_approx_eval_doubles": (i32, [vp, vp, u64, vp, vp, vp]),
            "tr_approx_get_eval_view": (i32, [vp, P(_EvalView)]),
            "tr_approx_eval_rows": (i32, [vp, vp, i32, i32, vp, vp, vp, vp, P(_EvalStats), B, R]),
            "tr_approx_eval_need": (i32, [vp, vp, i32, i32, P(_Need)]),
            "tr_approx_eval_view_rows": (i32, [P(_EvalView), vp, i32, i32, vp, vp, vp, vp, P(_EvalStats)]),
            "tr_approx_filter": (i32, [vp, vp, i32, vp, i32, _UQ, vp, vp, P(_FilterStats), B, R]),
            "tr_approx_authorize_exact": (i32, [vp, u64, P(mc._Decision)]),
            "tr_approx_relation": (vp, [vp, vp, i32, i32, vp, i32, P(mc._Evidence), B, R, e]),
            "tr_bigq_numerator": (vp, [vp]),
            "tr_bigq_denominator": (vp, [vp]),
            "tr_bigq_free": (None, [vp]),
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
    from .math_bank import _View
    math_gpu.bind("tr_mgpu_approx_eval", C.c_int, [P(_View), C.c_int32, P(_EvalView), C.c_int32, C.c_void_p,
                                                  C.c_void_p, C.c_void_p, C.c_void_p, P(_EvalStats), P(_Budget),
                                                  P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


# ------------------------------------------------------------------ exact numbers
def _frac(v, what="value"):
    if isinstance(v, bool):
        raise TypeError(f"{what}: a boolean is not a number")
    if isinstance(v, Integral):
        return Fraction(int(v))
    if isinstance(v, Fraction):
        return v
    if isinstance(v, Decimal):
        if not v.is_finite():
            raise ValueError(f"{what}: {v} is not finite")
        return Fraction(v)
    if isinstance(v, str):
        return Fraction(Decimal(v))
    raise TypeError(f"{what}: exact numbers only (int, Fraction, Decimal, str), got {type(v).__name__}")


def _bigq(v, what="value"):
    """an FND-009 big rational for an exact number (the Python object keeps the native handle alive)"""
    return _carriers().BigRational(_frac(v, what))


def _fraction(ptr):
    """the Fraction of a borrowed native big rational (None for NULL)"""
    if not ptr:
        return None
    mcar = _carriers()
    n = mcar._bigint_value(_call("tr_bigq_numerator", ptr))
    d = mcar._bigint_value(_call("tr_bigq_denominator", ptr))
    return Fraction(n, d)


def _made(handle, error, refusal, what):
    if error.value == -14:
        raise NotConvergedError(f"{what}: the certified bound exceeds the error budget (refused, not reported as "
                                "certified)")
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


# ------------------------------------------------------------------ approximations
class Approximation:
    """An immutable native approximation (tr_approx) with its certified error record."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_approx.series / poly / lowrank or an operation on approximations")

    @classmethod
    def _wrap(cls, handle):
        self = object.__new__(cls)
        self._handle = handle
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the approximation is closed")
        return self._handle

    def _info(self):
        out = _Info()
        _check(_call("tr_approx_get_info", self._open(), C.byref(out)), what="approximation")
        return out

    def _q(self, which, index=0):
        return _fraction(_call("tr_approx_get_q", self._open(), _Q[which], int(index)))

    @property
    def kind(self):
        return KINDS[self._info().kind]

    @property
    def info(self):
        i = self._info()
        out = dict(kind=KINDS[i.kind], function=_FUNCTION_NAMES.get(i.function), basis=_BASIS_NAMES.get(i.basis),
                   method={0: None, 1: "cross", 2: "sketch"}[i.method], norm=_NORM_NAMES.get(i.norm),
                   label=LABELS[i.label], provenance=PROVENANCE[i.provenance],
                   assumptions=[k for k, b in ASSUMPTIONS.items() if i.assumptions & b], degree=int(i.degree),
                   terms=int(i.nterms), taylor=int(i.taylor) or None, quantized=int(i.quantized) or None,
                   exact=bool(i.exact), operation=OPERATIONS[i.operation], id=int(i.id),
                   origins=[int(o) for o in i.origins if o], bytes=int(i.bytes), error_hi=float(i.error_hi),
                   has_budget=bool(i.has_budget), relative_available=bool(i.has_relative))
        if i.kind == 2:
            out.update(rows=int(i.rows), cols=int(i.cols), rank=int(i.rank), probes=int(i.probes) or None,
                       seed=int(i.seed) if i.provenance == 2 else None,
                       failure_hi=float(i.failure_hi) if i.provenance == 2 else None)
        else:
            out.update(range_hi=float(i.range_hi), range_lo=float(i.range_lo))
        return out

    @property
    def label(self):
        return LABELS[self._info().label]

    @property
    def provenance(self):
        i = self._info()
        out = dict(kind=PROVENANCE[i.provenance])
        if i.provenance == 2:
            out.update(seed=int(i.seed), failure=self._q("failure"), probes=int(i.probes),
                       statement=f"the bound holds with probability >= 1 - {self._q('failure')} over the probes")
        return out

    @property
    def norm(self):
        return _NORM_NAMES[self._info().norm]

    @property
    def assumptions(self):
        i = self._info()
        out = {k: True for k, b in ASSUMPTIONS.items() if i.assumptions & b}
        if i.kind != 2:
            out["domain"] = (self._q("a"), self._q("b"))
            if i.assumptions & ASSUMPTIONS["lipschitz"]:
                out["lipschitz"] = self._q("lipschitz")
            if i.assumptions & ASSUMPTIONS["derivative"] and self._q("derivative") is not None:
                out["derivative"] = self._q("derivative")
        return out

    @property
    def error(self):
        """the certified error (an exact Fraction) in the declared norm"""
        return self._q("error")

    @property
    def error_budget(self):
        return self._q("budget")

    @property
    def terms(self):
        """the error spent, term by term: [{"kind", "origin", "scaled", "magnitude" (Fraction)}]; their sum is error"""
        out = []
        for i in range(self._info().nterms):
            t = _Term()
            _check(_call("tr_approx_get_term", self._open(), i, C.byref(t)), what="error term")
            out.append(dict(kind=TERMS[t.kind], origin=int(t.origin), scaled=bool(t.scaled), magnitude=self._q("term", i)))
        return out

    @property
    def interval(self):
        return self._q("a"), self._q("b")

    @property
    def lipschitz(self):
        return self._q("lipschitz")

    @property
    def range(self):
        """a certified enclosure of the exact function's values on the interval"""
        return self._q("range_lo"), self._q("range_hi")

    @property
    def relative(self):
        """the relative bound (error / a certified lower bound of |f|), or None when |f| may vanish"""
        return self._q("relative")

    @property
    def l2(self):
        """the certified L2(dx) bound on the interval (<= sqrt(b - a) error)"""
        return self._q("l2")

    @property
    def coefficients(self):
        i = self._info()
        return [self._q("coef", j) for j in range(i.degree + 1)] if i.kind != 2 else None

    def bounds(self):
        """every applicable certified bound by norm"""
        i = self._info()
        if i.kind == 2:
            out = dict(frobenius=self._q("frobenius"), spectral=self._q("spectral"), max=self._q("maxentry"))
            if i.method == 1:
                out["frobenius_squared_exact"] = self._q("frobenius2")
            return out
        out = dict(abs=self.error, l2=self.l2)
        if i.has_relative:
            out["rel"] = self.relative
        return out

    def value(self, x):
        """the exact approximant at a rational x of the interval (FLOAT: the polynomial of its binary64 coefficients)"""
        if self._info().kind == 2:
            raise ValueError("a low-rank approximation has entries: use entry(i, j) or entries()")
        q = _bigq(x, "x")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_approx_value", self._open(), q._h, 0, None, C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "approximant value")
        try:
            return _fraction(h)
        finally:
            _call("tr_bigq_free", h)

    # ---------------------------------------------------------------- low rank
    def entry(self, i, j):
        info = self._info()
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_approx_value", self._open(), None, int(i) * info.cols + int(j), None, C.byref(refusal),
                  C.byref(error))
        _check(error.value, refusal, "approximant entry")
        try:
            return _fraction(h)
        finally:
            _call("tr_bigq_free", h)

    def entries(self):
        i = self._info()
        return [[self._q("entry", r * i.cols + c) for c in range(i.cols)] for r in range(i.rows)]

    def pivots(self):
        """CROSS: (pivot rows, pivot columns) in elimination order"""
        i = self._info()
        rows = (C.c_int32 * max(i.rank, 1))()
        cols = (C.c_int32 * max(i.rank, 1))()
        _check(_call("tr_approx_get_pivots", self._open(), rows, cols, max(i.rank, 1)), what="pivots")
        return [int(rows[k]) for k in range(i.rank)], [int(cols[k]) for k in range(i.rank)]

    def sketch(self):
        """SKETCH: (Omega as a cols x k int8 array of +-1, the kept columns of Y = A Omega)"""
        i = self._info()
        k = i.degree
        om = np.zeros(max(i.cols * k, 1), dtype=np.int8)
        kept = (C.c_int32 * max(k, 1))()
        n = _check(_call("tr_approx_get_sketch", self._open(), om.ctypes.data, om.size, kept, max(k, 1)), what="sketch")
        return om[:i.cols * k].reshape(i.cols, k).copy(), [int(kept[j]) for j in range(n)]

    def probes(self):
        """SKETCH: the exact ||E w_i||^2 of each probe"""
        return [self._q("probe", j) for j in range(self._info().probes)]

    # ---------------------------------------------------------------- operations
    def _op(self, name, other, what, error_budget=None, budget=None):
        eb = None if error_budget is None else _bigq(error_budget, "error budget")
        error, refusal = C.c_int32(), _Refusal()
        h = _call(name, self._open(), other._open() if isinstance(other, Approximation) else other,
                  eb._h if eb else None, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Approximation._wrap(_made(h, error, refusal, what))

    def add(self, other, *, error_budget=None, budget=None):
        return self._op("tr_approx_add", other, "sum", error_budget, budget)

    def mul(self, other, *, error_budget=None, budget=None):
        return self._op("tr_approx_mul", other, "product", error_budget, budget)

    def scale(self, k, *, error_budget=None, budget=None):
        q = _bigq(k, "scale")
        return self._op("tr_approx_scale", q._h, "scale", error_budget, budget)

    def compose(self, inner, *, error_budget=None, budget=None):
        """self(inner(x)): the inner range must lie in this approximation's interval (else DomainError)"""
        return self._op("tr_approx_compose", inner, "composition", error_budget, budget)

    def quantize(self, bits, *, error_budget=None, budget=None):
        """the coefficients rounded to multiples of 2^-bits, the induced error added"""
        eb = None if error_budget is None else _bigq(error_budget, "error budget")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_approx_quantize", self._open(), int(bits), eb._h if eb else None, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Approximation._wrap(_made(h, error, refusal, "quantization"))

    def to_float(self, *, error_budget=None, budget=None):
        """the floating evaluation (binary64 Horner in x) with its certified rounding bound"""
        eb = None if error_budget is None else _bigq(error_budget, "error budget")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_approx_float", self._open(), eb._h if eb else None, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Approximation._wrap(_made(h, error, refusal, "floating evaluation"))

    def __add__(self, other):
        if isinstance(other, Approximation):
            return self.add(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Approximation):
            return self.add(other.scale(-1))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Approximation):
            return self.mul(other)
        if isinstance(other, (Integral, Fraction, Decimal)) and not isinstance(other, bool):
            return self.scale(other)
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        return self.scale(-1)

    def __call__(self, inner):
        return self.compose(inner)

    # ---------------------------------------------------------------- evaluation
    def doubles(self, xs):
        """FLOAT: binary64 values at binary64 points and their certified bounds: (values, bounds, status names)"""
        x = np.ascontiguousarray(np.asarray(xs, dtype=np.float64))
        n = len(x)
        val, bnd = np.zeros(max(n, 1)), np.zeros(max(n, 1))
        st = np.zeros(max(n, 1), dtype=np.int8)
        _check(_call("tr_approx_eval_doubles", self._open(), x.ctypes.data, n, val.ctypes.data, bnd.ctypes.data,
                     st.ctypes.data), what="floating evaluation")
        return val[:n], bnd[:n], [ROW_STATUS[int(s)] for s in st[:n]]

    def view(self):
        v = _EvalView()
        _check(_call("tr_approx_get_eval_view", self._open(), C.byref(v)), what="evaluation view")
        return v

    def evaluate(self, scope, column, *, fbits=40, groups=True, backend="cpu", budget=None):
        """f(x) over the selected rows of a scope column: per-row enclosures and the exact group sums"""
        if column not in scope.names:
            raise ValueError(f"{column!r} is not a column of the scope")
        if scope.source == "table":
            scope.require_fresh("approximate evaluation")
        col = scope.names.index(column)
        n, G = scope.rows, scope.groups
        lo, hi = np.zeros(max(n, 1)), np.zeros(max(n, 1))
        st = np.zeros(max(n, 1), dtype=np.int8)
        garr = (_Group * max(G, 1))() if groups else None
        stats, refusal = _EvalStats(), _Refusal()
        if backend == "cpu":
            rc = _call("tr_approx_eval_rows", self._open(), scope._handle, col, int(fbits), lo.ctypes.data,
                       hi.ctypes.data, st.ctypes.data, garr, C.byref(stats), C.byref(_budget(budget)), C.byref(refusal))
            _check(rc, refusal, "approximate evaluation")
            name, dev = "cpu", "native CPU"
        elif backend == "cuda":
            g = _gpu()
            v = self.view()
            sv = scope.view()
            g.run("tr_mgpu_approx_eval", C.byref(sv), col, C.byref(v), int(fbits), lo.ctypes.data, hi.ctypes.data,
                  st.ctypes.data, garr, C.byref(stats), C.byref(_budget(budget)), C.byref(refusal), refusal=refusal,
                  what="approximate evaluation on the device")
            d = g.device()
            name, dev = ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return RowEvaluation(lo[:n], hi[:n], st[:n], garr, G, stats, name, dev)

    def need(self, scope, column, backend="cpu"):
        out = _Need()
        _check(_call("tr_approx_eval_need", self._open(), scope._handle, scope.names.index(column),
                     0 if backend == "cpu" else 1, C.byref(out)), what="evaluation needs")
        return {r: int(out.need[i]) for i, r in enumerate(RESOURCES) if out.need[i]}

    # ---------------------------------------------------------------- FND-007
    def decide(self, field=0):
        """the native refusal of an exact rewrite as a dict (never authorized)"""
        mc = _mc()
        d = mc._Decision()
        rc = _call("tr_approx_authorize_exact", self._open(), int(field), C.byref(d))
        return dict(authorized=bool(d.authorized), reason=mc.REFUSALS[d.reason], label=LABELS.get(d.kind),
                    field=int(d.field), code=int(rc))

    def authorize_exact(self, field=0):
        """an approximation never authorizes an exact rewrite: NotAuthorizedError ("approximate", or "statistical"
        for a probabilistic bound)"""
        d = self.decide(field)
        mc = _mc()
        err = mc.NotAuthorizedError(f"an approximation ({d['label']}) cannot authorize an exact rewrite: {d['reason']}")
        err.reason, err.decision = d["reason"], d
        raise err

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_approx_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Approximation(closed)"
        i = self.info
        return (f"Approximation({i['kind']}, {i['function']}, degree={i['degree']}, error<={i['error_hi']:.3g}, "
                f"{i['label']})")


class RowEvaluation:
    """per-row enclosures (lo, hi, status) and the exact group sums [sum_lo / 2^F, sum_hi / 2^F]"""

    def __init__(self, lo, hi, status, groups, ngroups, stats, backend, device):
        self.lo, self.hi, self.status = lo.copy(), hi.copy(), status.copy()
        self.backend, self.device = backend, device
        self.stats = {n: int(getattr(stats, n)) for n, _ in _EvalStats._fields_}
        self.fbits = self.stats["fbits"]
        self._groups = groups
        self.ngroups = ngroups

    def statuses(self):
        return [ROW_STATUS[int(s)] for s in self.status]

    @staticmethod
    def _i128(words):
        return (int(words[1]) << 64) | (int(words[0]) & ((1 << 64) - 1))

    def group(self, g):
        """{"rows", "ok", "nulls", "refused", "sum": (lo, hi) exact Fractions or None when a row was refused}"""
        if self._groups is None:
            return None
        G = self._groups[int(g)]
        lo, hi = self._i128(G.lo), self._i128(G.hi)
        scale = 1 << self.fbits
        out = dict(rows=int(G.rows), ok=int(G.ok), nulls=int(G.nulls), refused=int(G.refused),
                   raw=(lo, hi))
        out["sum"] = (Fraction(lo, scale), Fraction(hi, scale)) if not G.refused else None
        return out

    def groups(self):
        return [self.group(g) for g in range(self.ngroups)]

    def same(self, other):
        """bit-for-bit equality with another evaluation (rows and group sums)"""
        same = (np.array_equal(self.lo.view(np.int64), other.lo.view(np.int64)) and
                np.array_equal(self.hi.view(np.int64), other.hi.view(np.int64)) and
                np.array_equal(self.status, other.status))
        if self._groups is not None and other._groups is not None:
            same = same and all(bytes(a) == bytes(b) for a, b in zip(self._groups, other._groups))
        return bool(same) and {k: v for k, v in self.stats.items()} == other.stats


# ------------------------------------------------------------------ constructors
def series(function, a, b, degree, *, basis="chebyshev", taylor=None, norm="abs", error_budget=None, budget=None):
    """a certified polynomial approximation of exp, log, sin, cos, sqrt or recip on [a, b] (exact rationals):
    "chebyshev" (an economized Taylor series of degree `taylor`, default min(n + 24, 2n + 8)) or "taylor" (monomials in
    t = (x - mid) / rad); error_budget in the declared norm ("abs", "rel", "l2"): NotConvergedError if not achieved"""
    if function not in FUNCTIONS:
        raise ValueError(f"function is one of {', '.join(FUNCTIONS)}")
    if basis not in BASES:
        raise ValueError("basis is 'chebyshev' or 'taylor'")
    if norm not in ("abs", "rel", "l2"):
        raise ValueError("a series is bounded in the 'abs', 'rel' or 'l2' norm")
    qa, qb = _bigq(a, "a"), _bigq(b, "b")
    eb = None if error_budget is None else _bigq(error_budget, "error budget")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_approx_series", FUNCTIONS[function], qa._h, qb._h, int(degree), BASES[basis], int(taylor or 0),
              NORMS[norm], eb._h if eb else None, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Approximation._wrap(_made(h, error, refusal, f"{function} series"))


def poly(coefficients, a, b, *, budget=None):
    """an exact polynomial sum_j c_j x^j on [a, b] as a Chebyshev series (error 0): for compositions"""
    qs = [_bigq(c, "coefficient") for c in coefficients]
    if not qs:
        raise ValueError("a polynomial has at least one coefficient")
    arr = (C.c_void_p * len(qs))(*[q._h for q in qs])
    qa, qb = _bigq(a, "a"), _bigq(b, "b")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_approx_poly", arr, len(qs) - 1, qa._h, qb._h, C.byref(_budget(budget)), C.byref(refusal),
              C.byref(error))
    return Approximation._wrap(_made(h, error, refusal, "polynomial"))


def lowrank(matrix, rank, *, method="cross", seed=0, probes=32, norm="frobenius", error_budget=None, budget=None):
    """a rank-k approximation of a small integer matrix: "cross" (exact Bareiss elimination with complete pivoting:
    exact residual, certified norms) or "sketch" (a seeded Rademacher range finder and the exact projection, with a
    probabilistic bound from independent probes); norm "frobenius", "spectral" or "max" for the error budget"""
    A = np.asarray(matrix)
    if A.ndim != 2 or A.dtype.kind not in "iu":
        raise TypeError("a two-dimensional integer matrix")
    A = np.ascontiguousarray(A, dtype=np.int64)
    if method not in METHODS:
        raise ValueError("method is 'cross' or 'sketch'")
    if norm not in ("frobenius", "spectral", "max"):
        raise ValueError("norm is 'frobenius', 'spectral' or 'max'")
    eb = None if error_budget is None else _bigq(error_budget, "error budget")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_approx_lowrank", A.ctypes.data, A.shape[0], A.shape[1], int(rank), METHODS[method], int(seed),
              int(probes) if method == "sketch" else 0, NORMS[norm], eb._h if eb else None, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Approximation._wrap(_made(h, error, refusal, f"rank-{rank} approximation"))


def filter(state, scope, column, approx, op, threshold, *, budget=None):
    """f(x) op T over the groups of a UNC-001 State (built from `scope`; x its column `column`): groups decided by the
    binary64 UNC-001 proof of p(t) op T -+ e, the other rows by their certified enclosure; (true bitmap, undecided
    bitmap, stats). A row the approximation cannot decide is undecided, never guessed."""
    from . import math_unc as mu
    if op not in OPS:
        raise ValueError(f"op is one of {', '.join(OPS)}")
    if approx.kind != "series":
        raise UnsupportedError("the approximate filter proves a series (its binary64 form is evaluated per row: "
                               "Approximation.evaluate)")
    if column not in state.columns:
        raise ValueError(f"{column!r} is not a column of the state")
    T = mu._frac(threshold, "threshold")
    nw = (scope.rows + 63) // 64
    bits = np.zeros(max(nw, 1), dtype=np.uint64)
    und = np.zeros(max(nw, 1), dtype=np.uint64)
    st, refusal = _FilterStats(), _Refusal()
    rc = _call("tr_approx_filter", state._open(), scope._handle, list(state.columns).index(column), approx._open(),
               OPS[op], mu._q(T), bits.ctypes.data, und.ctypes.data, C.byref(st), C.byref(_budget(budget)),
               C.byref(refusal))
    _check(rc, refusal, "approximate filter")
    stats = {n: (float(getattr(st, n)) if n == "error_used" else int(getattr(st, n))) for n, _ in _FilterStats._fields_
             if n != "reserved"}
    return bits[:nw], und[:nw], stats


def relation(approx, scope, x, y, *, tolerance=0, bits=30, budget=None):
    """the approximate relationship y ~ p(x) as an FND-007 snapshot certificate (a POLYNOMIAL claim with a tolerance:
    the certified error, the rounding of p to multiples of 2^-bits and the data tolerance), verified on every selected
    row; it never authorizes an exact rewrite"""
    mc = _mc()
    q = _bigq(tolerance, "tolerance")
    ev, error, refusal = mc._Evidence(), C.c_int32(), _Refusal()
    h = _call("tr_approx_relation", approx._open(), scope._handle, scope.names.index(x), scope.names.index(y), q._h,
              int(bits), C.byref(ev), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    h = mc._made(h, error, refusal, "approximate relation", ev)
    return mc.Certificate._wrap(h, scope.names)
