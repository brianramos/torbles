# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEN-002: convolutions of generating states with an explicit, checked independence assumption, the conditional
(per-key) form, joint states and bounded truncated series (torbles/math_conv.c on GEN-001's generating polynomials;
CUDA: math_conv_gpu.cu).

    from torbles import math_bank as mbk, math_dist as md, math_conv as mc
    scope = mbk.Scope.from_table(db["t"], ["x", "y"], group_by=["k"])
    X, Y = mc.factor(md.histogram(scope, ["x"])), mc.factor(md.histogram(scope, ["y"]))   # marginals over the keys
    mc.product([X, Y])                     # DomainError: same keyed source, no independence declaration
    d = mc.independent(scope)              # the caller's declaration (its scope: that scope's source)
    mc.product([X, Y], declaration=d)      # X + Y for independent draws (the Cartesian product of the bags)
    mc.product([X, Y], declaration=d, combine="outer")   # the joint of (X, Y) under independence
    mc.power(X, 8, declaration=d, order=16)               # sums of 8 i.i.d. draws below 16 (exponents of P)
    mc.joint_sum(mc.factor(md.histogram(scope, ["x", "y"])))   # per row x + y (the dependent sum)
    mc.conditional(mc.family(hx), mc.family(hy), backend="cuda")   # per key products summed: the key-equijoin
    mc.convolve([[1, 2], [3]], [[1, 1], [5, 6]], backend="cuda")   # exact 192-bit batched convolutions
    s = mc.Series([0, 1, 1], order=16, bound=10**6); s.walks(8); s.walks(None)   # walks of length <= 8 / any length

Every result is computed natively; Python marshals and formats. A Factor's requests (terms, coefficient, total,
moments, evaluation, series) are GEN-001's on its polynomial (`f.gen`). Refusals: DomainError (linked inputs
without a covering declaration: `.reason`, `.factors`, `.keyed`; key domains or scales that differ), AlgebraBudgetError
(before allocation), OverflowError (a value past int64, a coefficient past its declared bound, a summed state past
int128 per key), NoSolutionError (an unbounded walk sum that diverges), CarrierError (series of different orders),
UnsupportedError. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from numbers import Integral

import numpy as np

from . import algq
from . import math_gen as _mg
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_dist import Histogram
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Factor", "Family", "Declaration", "Series", "Batch", "independent", "factor", "from_terms", "from_gen", "product",
           "power", "need", "joint_sum", "marginal", "family", "conditional", "conditional_need", "convolve", "series",
           "ALL", "UNBOUNDED"]

ALL = (1 << 64) - 1
UNBOUNDED = (1 << 32) - 1
NONE = (1 << 64) - 1
COMBINE = {"sum": 1, "outer": 2}
KINDS = {0: "free", 1: "data", 2: "derived"}
MODES = {0: "base", 1: "independent", 2: "conditional", 3: "joint", 4: "marginal", 5: "series"}
REASONS = {0: None, 1: "needs_declaration", 2: "not_covered", 3: "keys_differ", 4: "grouping_differs",
           5: "scales_differ"}
P = C.POINTER
i32, u32, i64, u64, vp = C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_void_p


class _Info(C.Structure):
    _native_ = "tr_conv_info"
    _fields_ = [("sources", u64), ("keys", u64), ("key", u64), ("tuples", u64), ("nulls", u64), ("draws", u64),
                ("digest", u64 * 2), ("kind", i32), ("mode", i32), ("keyed", i32), ("declared", i32),
                ("combine", i32), ("counted", i32), ("nvars", i32), ("mass", i32), ("col", i32 * 2),
                ("scale", i32 * 2), ("reserved", i32 * 4)]


class _Check(C.Structure):
    _native_ = "tr_conv_check"
    _fields_ = [("reason", i32), ("first", i32), ("second", i32), ("keyed", i32), ("digest", u64 * 2)]


class _DeclInfo(C.Structure):
    _native_ = "tr_conv_decl_info"
    _fields_ = [("sources", u64), ("digest", u64 * 2)]


class _FamilyView(C.Structure):
    _native_ = "tr_conv_family_view"
    _fields_ = [("keys", u64), ("ncoef", u64), ("maxspan", u64), ("offset", vp), ("shift", vp), ("coef", vp),
                ("mass", i32), ("scale", i32), ("reserved", u64)]


class _FamilyInfo(C.Structure):
    _native_ = "tr_conv_family_info"
    _fields_ = [(n, u64) for n in ("keys", "present", "ncoef", "maxspan", "rows", "nulls", "bytes")] + [
        ("digest", u64 * 2)]


class _Batch(C.Structure):
    _native_ = "tr_conv_batch"
    _fields_ = [("npairs", u64), ("order", u64), ("total", u64), ("pairs", u64), ("aoff", vp), ("boff", vp),
                ("ooff", vp), ("a", vp), ("b", vp)]


class _SeriesInfo(C.Structure):
    _native_ = "tr_conv_series_info"
    _fields_ = [("order", u64), ("sources", u64), ("bound", i64), ("maxabs", i64), ("shift", i64)]


class _Walks(C.Structure):
    _native_ = "tr_conv_walks"
    _fields_ = [("lengths", u64), ("violation", u64), ("status", i32), ("reserved", i32), ("bound", i64),
                ("maxabs", i64)]


assert (C.sizeof(_Info), C.sizeof(_Check), C.sizeof(_DeclInfo), C.sizeof(_FamilyView), C.sizeof(_FamilyInfo),
        C.sizeof(_Batch), C.sizeof(_SeriesInfo), C.sizeof(_Walks)) == (128, 32, 24, 64, 72, 72, 40, 40)

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R, K = P(i32), P(_Budget), P(_Refusal), P(_Check)
        sig = {
            "tr_conv_declare": (vp, [vp, u32, B, R, e]),
            "tr_conv_decl_get_info": (i32, [vp, P(_DeclInfo)]),
            "tr_conv_decl_covers": (i32, [vp, vp]),
            "tr_conv_decl_free": (None, [vp]),
            "tr_conv_factor_from_hist": (vp, [vp, vp, u64, i32, i32, vp, B, R, e]),
            "tr_conv_factor_from_terms": (vp, [i32, u64, vp, vp, i32, vp, B, R, e]),
            "tr_conv_factor_from_gen": (vp, [vp, B, R, e]),
            "tr_conv_factor_get_info": (i32, [vp, P(_Info)]),
            "tr_conv_factor_poly_copy": (vp, [vp, i32, B, R, e]),
            "tr_conv_factor_source": (i32, [vp, u64, P(u64), P(i32)]),
            "tr_conv_factor_free": (None, [vp]),
            "tr_conv_check_product": (i32, [vp, vp, u32, vp, K]),
            "tr_conv_product": (vp, [vp, vp, u32, i32, u64, vp, B, R, K, e]),
            "tr_conv_product_need": (i32, [vp, vp, u32, i32, u64, B, P(_Need), P(u64)]),
            "tr_conv_joint_sum": (vp, [vp, B, R, e]),
            "tr_conv_marginal": (vp, [vp, i32, B, R, e]),
            "tr_conv_family_from_hist": (vp, [vp, vp, i32, B, R, e]),
            "tr_conv_family_get_info": (i32, [vp, P(_FamilyInfo)]),
            "tr_conv_family_get_view": (i32, [vp, P(_FamilyView)]),
            "tr_conv_family_factor": (vp, [vp, u64, i32, B, R, e]),
            "tr_conv_family_free": (None, [vp]),
            "tr_conv_conditional": (vp, [vp, vp, i32, i32, B, R, K, e]),
            "tr_conv_conditional_need": (i32, [vp, vp, i32, i32, P(_Need)]),
            "tr_conv_conditional_check": (i32, [vp, vp, i32, K]),
            "tr_conv_conditional_plan": (i32, [vp, vp, vp, P(u64), P(u64)]),
            "tr_conv_conditional_adopt": (vp, [vp, vp, vp, vp, i32, B, R, K, e]),
            "tr_conv_batch_run": (i32, [P(_Batch), vp, B, R]),
            "tr_conv_batch_layout": (i32, [u64, vp, vp, u64, vp, P(u64), P(u64)]),
            "tr_conv_series_create": (vp, [vp, u32, u32, i64, B, R, e]),
            "tr_conv_series_from_factor": (vp, [vp, u32, i64, B, R, e]),
            "tr_conv_series_get_info": (i32, [vp, P(_SeriesInfo)]),
            "tr_conv_series_coefficients": (i32, [vp, vp, u64]),
            "tr_conv_series_add": (vp, [vp, vp, i64, B, R, P(u64), e]),
            "tr_conv_series_mul": (vp, [vp, vp, i64, vp, B, R, K, P(u64), e]),
            "tr_conv_series_pow": (vp, [vp, u32, i64, vp, B, R, K, P(u64), e]),
            "tr_conv_series_walks": (i32, [vp, u32, i64, vp, vp, u64, P(vp), P(_Walks), B, R, K]),
            "tr_conv_series_walks_need": (i32, [vp, u32, i32, P(_Need), P(u64)]),
            "tr_conv_walks_run": (i32, [vp, u64, u64, i64, vp, vp, P(_Walks), B, R]),
            "tr_conv_series_check": (i32, [vp, vp, u32, vp, K]),
            "tr_conv_series_walks_adopt": (vp, [vp, vp, i64, B, R, e]),
            "tr_conv_series_to_factor": (vp, [vp, i32, B, R, e]),
            "tr_conv_series_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_conv_batch", C.c_int, [P(_Batch), vp, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_conv_conditional", C.c_int, [P(_FamilyView), P(_FamilyView), vp, vp, P(_Budget),
                                                         P(_Refusal)])
    math_gpu.bind("tr_mgpu_conv_walks", C.c_int, [vp, u64, u64, i64, vp, vp, P(_Walks), P(_Budget), P(_Refusal)])
    return math_gpu


def _device_label():
    from . import math_gpu
    info = math_gpu.device()
    return "cuda-emulated" if info["emulated"] else "cuda"


def _budget(budget):
    return (budget or Budget()).raw


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _hex(d):
    return f"{int(d[0]):016x}{int(d[1]):016x}"


def _domain_error(check, what):
    reason = REASONS.get(int(check.reason), "unknown")
    i, j = int(check.first), int(check.second)
    src = _hex(check.digest)
    if reason in ("needs_declaration", "not_covered"):
        pair = f"copy {i} of one factor (several draws)" if i == j else f"factors {i} and {j}"
        keyed = "the same keyed source" if check.keyed else "the same source"
        missing = ("no independence declaration was given" if reason == "needs_declaration"
                   else "the given independence declaration does not cover it")
        msg = (f"{what}: {pair} come from {keyed} (rows linked by a row or key: source {src}) and {missing}; declare "
               f"the independence of that source (math_conv.independent(scope)), use the joint state "
               f"(math_conv.joint_sum) or the conditional form (math_conv.conditional)")
    elif reason == "keys_differ":
        msg = f"{what}: the families have different key domains (numbers of keys)"
    elif reason == "grouping_differs":
        msg = f"{what}: the families come from one source with different groupings: their keys do not correspond"
    elif reason == "scales_differ":
        msg = f"{what}: values of different decimal scales do not add"
    else:
        msg = f"{what}: incompatible domains"
    err = DomainError(msg)
    err.reason, err.factors, err.keyed, err.source = reason, (i, j), bool(check.keyed), src
    return err


def _checked(rc, refusal, check, what):
    if rc == -6 and check is not None and check.reason:
        raise _domain_error(check, what)
    _check(rc, refusal, what)


def _made(handle, error, refusal, what, check=None):
    _checked(error.value, refusal, check, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _i64(v, what):
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError(f"{what} must be an exact integer")
    v = int(v)
    if not -(1 << 63) <= v < (1 << 63):
        raise OverflowError(f"{what} does not fit int64")
    return v


def _words(arr, k):
    """the 192-bit two's complement integer of words 3k .. 3k + 2"""
    v = int(arr[3 * k]) & ((1 << 64) - 1) | (int(arr[3 * k + 1]) & ((1 << 64) - 1)) << 64 | \
        (int(arr[3 * k + 2]) & ((1 << 64) - 1)) << 128
    return v - (1 << 192) if v >> 191 else v


# ---------------------------------------------------------------------- declarations
class Declaration:
    """An independence declaration over the sources of some scopes: factors whose values come from those sources may
    be multiplied as independent draws (the product of their marginals, the full Cartesian product of the bags)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_conv.independent(scope, ...)")

    def _open(self):
        if not self._handle:
            raise ValueError("the declaration is closed")
        return self._handle

    @property
    def info(self):
        out = _DeclInfo()
        _check(_call("tr_conv_decl_get_info", self._open(), C.byref(out)), what="declaration")
        return dict(sources=int(out.sources), digest=_hex(out.digest))

    def covers(self, scope):
        return bool(_check(_call("tr_conv_decl_covers", self._open(), _scope_handle(scope)), what="declaration"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_conv_decl_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def independent(*scopes, budget=None):
    """declare that factors of these scopes' sources are independent draws (one declaration may cover several)"""
    if not scopes:
        raise ValueError("independent(scope, ...) needs at least one scope")
    arr = (vp * len(scopes))(*[_scope_handle(s) for s in scopes])
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_conv_declare", arr, len(scopes), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    d = object.__new__(Declaration)
    d._handle, d._scopes = _made(h, error, refusal, "independence declaration"), scopes
    return d


def _decl(declaration):
    if declaration is None:
        return None
    if not isinstance(declaration, Declaration):
        raise TypeError("declaration: a math_conv.Declaration (math_conv.independent(scope))")
    return declaration._open()


# ---------------------------------------------------------------------- factors
class Factor:
    """A generating state with the data sources of its values (GEN-001 polynomial + origin), immutable."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_conv.factor, from_terms, product, conditional, ...")

    @classmethod
    def _wrap(cls, handle, names=None, backend="cpu", keep=()):
        self = object.__new__(cls)
        self._handle, self.names, self.backend, self._gen, self._keep = handle, names, backend, None, keep
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the factor is closed")
        return self._handle

    def _raw(self):
        out = _Info()
        _check(_call("tr_conv_factor_get_info", self._open(), C.byref(out)), what="factor")
        return out

    @property
    def info(self):
        i = self._raw()
        nv = int(i.nvars)
        return dict(kind=KINDS[int(i.kind)], mode=MODES[int(i.mode)], variables=nv, sources=int(i.sources),
                    keyed=bool(i.keyed), declared=bool(i.declared), keys=int(i.keys),
                    key=None if i.key == ALL else int(i.key), draws=int(i.draws),
                    combine={0: None, 1: "sum", 2: "outer"}[int(i.combine)],
                    tuples=int(i.tuples) if i.counted else None, nulls=int(i.nulls) if i.counted else None,
                    mass={0: None, 1: "count", 2: "weight"}[int(i.mass)],
                    columns=tuple(int(i.col[c]) for c in range(nv)), scales=tuple(int(i.scale[c]) for c in range(nv)),
                    digest=_hex(i.digest) if i.sources else None, backend=self.backend)

    @property
    def sources(self):
        out = []
        for k in range(self.info["sources"]):
            d, keyed = (u64 * 2)(), i32()
            _check(_call("tr_conv_factor_source", self._open(), k, d, C.byref(keyed)), what="factor source")
            out.append(dict(digest=_hex(d), keyed=bool(keyed.value)))
        return out

    @property
    def gen(self):
        """the factor's GEN-001 generating polynomial (an owned copy): every GEN-001 request reads it"""
        if self._gen is None:
            error, refusal = C.c_int32(), _Refusal()
            h = _call("tr_conv_factor_poly_copy", self._open(), _mg.REPRS["sparse"], None, C.byref(refusal),
                      C.byref(error))
            self._gen = _mg.GeneratingPolynomial._wrap(_made(h, error, refusal, "factor polynomial"), self.names)
        return self._gen

    @property
    def nvars(self):
        return int(self._raw().nvars)

    def terms(self):
        return self.gen.terms()

    def coefficient(self, value):
        return self.gen.coefficient(value)

    def total(self):
        return self.gen.total()

    def moment(self, k, *, falling=False, budget=None):
        return self.gen.moment(k, falling=falling, budget=budget)

    def evaluate(self, z, *, budget=None):
        return self.gen.evaluate(z, budget=budget)

    __call__ = evaluate

    def series(self, order, *, budget=None):
        """P mod z^order as an FND-009 truncated series (exact rationals; GEN-001)"""
        return self.gen.series(order, budget=budget)

    def bounded(self, order, bound, *, budget=None):
        """P mod z^order as a bounded series (int64 coefficients, |c| <= bound), with this factor's sources"""
        return series(self, order, bound, budget=budget)

    def equal(self, other):
        return self.gen.equal(other.gen if isinstance(other, Factor) else other)

    def __eq__(self, other):
        return isinstance(other, (Factor, _mg.GeneratingPolynomial)) and self.equal(other)

    __hash__ = None

    def __mul__(self, other):
        return product([self, other])

    def __pow__(self, k):
        return power(self, k)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_conv_factor_free", self._handle)
            self._handle = None
        if getattr(self, "_gen", None) is not None:
            self._gen.close()
            self._gen = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Factor(closed)"
        i = self.info
        return f"Factor({i['kind']}, {i['mode']}, {i['variables']} var, sources={i['sources']}, keyed={i['keyed']})"


def _scope_handle(scope):
    scope._open()
    return scope._handle


def _hist_scope(hist, what):
    if not isinstance(hist, Histogram):
        raise TypeError(f"{what}: a math_dist.Histogram")
    if hist.scope is None:
        raise ValueError(f"{what}: the histogram has no scope (merged histograms have no source)")
    if hist.scope.source == "table":
        hist.scope.require_fresh(what)
    return hist.scope


def factor(hist, key=None, *, mass="count", repr="sparse", shift=None, budget=None):
    """the factor of one key (a group of the histogram's scope) or of every key (key=None: the marginal over the keys)
    of a 1-D or 2-D histogram; its source is the scope's (rows of one table, or one array scope)"""
    scope = _hist_scope(hist, "factor")
    if mass not in _mg.MASSES or repr not in _mg.REPRS:
        raise ValueError("mass is 'count' or 'weight'; repr is 'sparse' or 'dense'")
    s = _mg._shift(shift, hist.k)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_conv_factor_from_hist", hist._open(), _scope_handle(scope), ALL if key is None else int(key),
              _mg.MASSES[mass], _mg.REPRS[repr], s, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Factor._wrap(_made(h, error, refusal, "factor"), hist.names)


def from_terms(terms, *, repr="sparse", shift=None, budget=None):
    """a factor without a source: explicit terms {value: coefficient} or {(v1, v2): coefficient} (an abstract
    distribution: it combines with anything without a declaration)"""
    g = _mg.from_terms(terms, repr=repr, shift=shift, budget=budget)
    return from_gen(g, budget=budget)


def from_gen(poly, *, budget=None):
    """a factor without a source from a copy of a GEN-001 generating polynomial"""
    if not isinstance(poly, _mg.GeneratingPolynomial):
        raise TypeError("a math_gen.GeneratingPolynomial")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_conv_factor_from_gen", poly._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Factor._wrap(_made(h, error, refusal, "factor"), poly.names)


def _factor_args(factors, mult):
    fs = list(factors)
    if not fs or not all(isinstance(f, Factor) for f in fs):
        raise TypeError("a non-empty list of math_conv.Factor")
    arr = (vp * len(fs))(*[f._open() for f in fs])
    if mult is None:
        return fs, arr, None
    ks = [int(k) for k in mult]
    if len(ks) != len(fs) or any(k < 0 or k >= (1 << 32) for k in ks):
        raise ValueError("one nonnegative multiplicity per factor")
    return fs, arr, (u32 * len(ks))(*ks)


def product(factors, *, mult=None, combine="sum", order=None, declaration=None, budget=None):
    """prod f_i^(mult_i): combine "sum" (values add: the sum of one independent draw per copy, the Cartesian product of
    the bags) or "outer" (two one-variable factors: their joint as a two-variable state); order: exponents < order of
    P. Copies sharing a source need an independence declaration covering it (DomainError otherwise)."""
    if combine not in COMBINE:
        raise ValueError("combine is 'sum' or 'outer'")
    fs, arr, ks = _factor_args(factors, mult)
    error, refusal, check = C.c_int32(), _Refusal(), _Check()
    h = _call("tr_conv_product", arr, ks, len(fs), COMBINE[combine], 0 if order is None else int(order),
              _decl(declaration), C.byref(_budget(budget)), C.byref(refusal), C.byref(check), C.byref(error))
    return Factor._wrap(_made(h, error, refusal, "product", check), fs[0].names)


def power(f, k, *, order=None, declaration=None, budget=None):
    """f^k: the distribution of the sum of k i.i.d. draws (a data factor needs a declaration of its source)"""
    return product([f], mult=[k], order=order, declaration=declaration, budget=budget)


def check(factors, *, mult=None, declaration=None):
    """the independence check alone: None when the product may run, else the DomainError it would raise"""
    fs, arr, ks = _factor_args(factors, mult)
    c = _Check()
    rc = _call("tr_conv_check_product", arr, ks, len(fs), _decl(declaration), C.byref(c))
    if rc == -6:
        return _domain_error(c, "product")
    _check(rc, what="independence check")
    return None


def need(factors, *, mult=None, combine="sum", order=None, budget=None):
    """what the product checks before allocating (support growth, pairs; bytes for a power or two factors) and the
    bound on its terms ("terms")"""
    fs, arr, ks = _factor_args(factors, mult)
    raw, terms = _Need(), C.c_uint64()
    _check(_call("tr_conv_product_need", arr, ks, len(fs), COMBINE[combine], 0 if order is None else int(order),
                 C.byref(_budget(budget)), C.byref(raw), C.byref(terms)), what="product needs")
    return dict(_needs(raw), terms=int(terms.value))


def joint_sum(joint, *, budget=None):
    """G(z, z) of a two-variable factor: per tuple, the sum of its coordinates (dependent coordinates of one row)"""
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_conv_joint_sum", joint._open(), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Factor._wrap(_made(h, error, refusal, "joint sum"), joint.names)


def marginal(joint, var, *, budget=None):
    """G(z_var, 1) of a two-variable factor (same source: linked with the other marginal)"""
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_conv_marginal", joint._open(), int(var), C.byref(_budget(budget)), C.byref(refusal),
              C.byref(error))
    return Factor._wrap(_made(h, error, refusal, "marginal"), joint.names)


# ---------------------------------------------------------------------- keyed families and the conditional form
class Family:
    """Per key (group of a 1-D histogram's scope) the dense int64 coefficients of that key's distribution."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_conv.family(histogram)")

    def _open(self):
        if not self._handle:
            raise ValueError("the family is closed")
        return self._handle

    @property
    def info(self):
        out = _FamilyInfo()
        _check(_call("tr_conv_family_get_info", self._open(), C.byref(out)), what="family")
        return dict(keys=int(out.keys), present=int(out.present), coefficients=int(out.ncoef),
                    max_span=int(out.maxspan), rows=int(out.rows), nulls=int(out.nulls), bytes=int(out.bytes),
                    digest=_hex(out.digest))

    def _view(self):
        v = _FamilyView()
        _check(_call("tr_conv_family_get_view", self._open(), C.byref(v)), what="family view")
        return v

    def key(self, g, *, repr="sparse", budget=None):
        """key g's factor (None: the marginal over every key)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_conv_family_factor", self._open(), ALL if g is None else int(g), _mg.REPRS[repr],
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Factor._wrap(_made(h, error, refusal, "family factor"), self.names)

    def marginal(self, *, repr="sparse", budget=None):
        return self.key(None, repr=repr, budget=budget)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_conv_family_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def family(hist, *, mass="count", budget=None):
    """the keyed family of a 1-D histogram: key g's coefficients (counts, or weight sums that fit int64)"""
    scope = _hist_scope(hist, "family")
    if mass not in _mg.MASSES:
        raise ValueError("mass is 'count' or 'weight'")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_conv_family_from_hist", hist._open(), _scope_handle(scope), _mg.MASSES[mass], C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    f = object.__new__(Family)
    f._handle, f.names, f.scope = _made(h, error, refusal, "family"), hist.names, scope
    return f


def conditional_need(a, b, *, combine="sum", backend="cpu"):
    """what the conditional form checks before allocating (backend "cuda": bytes = the larger of the host stage and
    the device copies)"""
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    raw = _Need()
    _check(_call("tr_conv_conditional_need", a._open(), b._open(), COMBINE[combine], int(backend == "cuda"),
                 C.byref(raw)), what="conditional needs")
    return _needs(raw)


def conditional(a, b, *, combine="sum", repr="sparse", backend="cpu", budget=None):
    """the conditional form: per key g, A_g x B_g (sum: values add; outer: pairs), summed over the keys (the
    key-equijoin's distribution). backend "cuda": the per-key products in parallel over keys (sum)"""
    if not isinstance(a, Family) or not isinstance(b, Family):
        raise TypeError("two math_conv.Family")
    if combine not in COMBINE or repr not in _mg.REPRS:
        raise ValueError("combine is 'sum' or 'outer'; repr is 'sparse' or 'dense'")
    error, refusal, check = C.c_int32(), _Refusal(), _Check()
    if backend == "cpu":
        h = _call("tr_conv_conditional", a._open(), b._open(), COMBINE[combine], _mg.REPRS[repr],
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(check), C.byref(error))
        return Factor._wrap(_made(h, error, refusal, "conditional product", check), a.names)
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    if combine != "sum":
        raise UnsupportedError("the CUDA conditional product adds the values (combine='sum')")
    _checked(_call("tr_conv_conditional_check", a._open(), b._open(), 1, C.byref(check)), refusal, check,
             "conditional product")
    from .math_budgets import check as _bcheck
    _bcheck(budget or Budget(), "conditional product", **conditional_need(a, b, backend="cuda"))
    va, vb = a._view(), b._view()
    ooff = np.zeros(int(va.keys) + 1, dtype=np.uint64)
    total, pairs = C.c_uint64(), C.c_uint64()
    _check(_call("tr_conv_conditional_plan", a._open(), b._open(), ooff.ctypes.data, C.byref(total), C.byref(pairs)),
           what="conditional plan")
    out = np.zeros(max(3 * int(total.value), 3), dtype=np.int64)
    _gpu().run("tr_mgpu_conv_conditional", C.byref(va), C.byref(vb), ooff.ctypes.data, out.ctypes.data,
               C.byref(_budget(budget)), C.byref(refusal), refusal=refusal, what="conditional product on the device")
    h = _call("tr_conv_conditional_adopt", a._open(), b._open(), ooff.ctypes.data, out.ctypes.data, _mg.REPRS[repr],
              C.byref(_budget(budget)), C.byref(refusal), C.byref(check), C.byref(error))
    f = Factor._wrap(_made(h, error, refusal, "conditional product", check), a.names, backend=_device_label())
    f._raw_out, f._raw_ooff = out[:3 * int(total.value)], ooff
    return f


# ---------------------------------------------------------------------- batched convolutions
class Batch(list):
    """the exact convolutions of a batch (one list of ints per pair); .backend: cpu, cuda or cuda-emulated"""
    backend = "cpu"


def _csr(lists, what):
    off = np.zeros(len(lists) + 1, dtype=np.uint64)
    flat = []
    for i, xs in enumerate(lists):
        xs = [_i64(x, what) for x in xs]
        flat.extend(xs)
        off[i + 1] = off[i] + len(xs)
    return off, np.array(flat if flat else [0], dtype=np.int64)


def convolve(a, b, *, order=None, backend="cpu", budget=None, raw=False):
    """exact convolutions of pairs (a[i], b[i]) of int64 coefficient lists (exponents 0, 1, ...): one list of exact ints
    per pair (empty when either is empty), at most `order` coefficients each. CPU and CUDA give the same 192-bit words
    (raw=True returns them)."""
    if len(a) != len(b):
        raise ValueError("one b per a")
    aoff, af = _csr(a, "a coefficient")
    boff, bf = _csr(b, "a coefficient")
    n = len(a)
    ooff = np.zeros(n + 1, dtype=np.uint64)
    total, pairs = C.c_uint64(), C.c_uint64()
    _check(_call("tr_conv_batch_layout", n, aoff.ctypes.data, boff.ctypes.data, 0 if order is None else int(order),
                 ooff.ctypes.data, C.byref(total), C.byref(pairs)), what="batch layout")
    out = np.zeros(max(3 * int(total.value), 3), dtype=np.int64)
    bt = _Batch(npairs=n, order=0 if order is None else int(order), total=total.value, pairs=pairs.value,
                aoff=aoff.ctypes.data, boff=boff.ctypes.data, ooff=ooff.ctypes.data, a=af.ctypes.data,
                b=bf.ctypes.data)
    refusal = _Refusal()
    if backend == "cpu":
        _check(_call("tr_conv_batch_run", C.byref(bt), out.ctypes.data, C.byref(_budget(budget)), C.byref(refusal)),
               refusal, "batched convolution")
        label = "cpu"
    elif backend == "cuda":
        _gpu().run("tr_mgpu_conv_batch", C.byref(bt), out.ctypes.data, C.byref(_budget(budget)), C.byref(refusal),
                   refusal=refusal, what="batched convolution on the device")
        label = _device_label()
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    words = out[:3 * int(total.value)]
    if raw:
        return words.copy()
    res = Batch([_words(words, int(ooff[i]) + e) for e in range(int(ooff[i + 1] - ooff[i]))] for i in range(n))
    res.backend = label
    return res


# ---------------------------------------------------------------------- bounded series
class Series:
    """A bounded truncated series: c_0 + ... + c_(N-1) z^(N-1) + O(z^N), int64 coefficients with the declared bound
    |c| <= bound, exact up to the order; it keeps the sources of the factor it was made from."""

    def __init__(self, coefficients, order=None, bound=(1 << 63) - 1, *, budget=None):
        cs = [_i64(c, "a coefficient") for c in coefficients]
        n = len(cs) if order is None else int(order)
        if n < 1 or len(cs) > n:
            raise ValueError("order >= 1 and at most order coefficients")
        arr = np.array(cs if cs else [0], dtype=np.int64)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_conv_series_create", arr.ctypes.data, len(cs), n, _i64(bound, "bound"),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        self._handle, self._keep = _made(h, error, refusal, "bounded series"), ()

    @classmethod
    def _wrap(cls, handle, keep=()):
        self = object.__new__(cls)
        self._handle, self._keep = handle, keep
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the series is closed")
        return self._handle

    @property
    def info(self):
        out = _SeriesInfo()
        _check(_call("tr_conv_series_get_info", self._open(), C.byref(out)), what="series")
        return dict(order=int(out.order), sources=int(out.sources), bound=int(out.bound), maxabs=int(out.maxabs),
                    shift=int(out.shift))

    @property
    def order(self):
        return self.info["order"]

    @property
    def bound(self):
        return self.info["bound"]

    @property
    def coefficients(self):
        n = self.order
        out = np.zeros(n, dtype=np.int64)
        _check(_call("tr_conv_series_coefficients", self._open(), out.ctypes.data, n), what="series")
        return [int(x) for x in out]

    def _bound(self, other, bound):
        if bound is not None:
            return _i64(bound, "bound")
        return max(self.bound, other.bound) if other is not None else self.bound

    def add(self, other, *, bound=None, budget=None):
        error, refusal, viol = C.c_int32(), _Refusal(), C.c_uint64()
        h = _call("tr_conv_series_add", self._open(), other._open(), self._bound(other, bound),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(viol), C.byref(error))
        return Series._wrap(_made(h, error, refusal, _bound_what("series sum", viol)))

    __add__ = lambda self, other: self.add(other)                 # noqa: E731

    def mul(self, other, *, bound=None, declaration=None, budget=None):
        """the truncated product, every coefficient checked against the bound (default: the larger declared bound)"""
        error, refusal, check, viol = C.c_int32(), _Refusal(), _Check(), C.c_uint64()
        h = _call("tr_conv_series_mul", self._open(), other._open(), self._bound(other, bound), _decl(declaration),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(check), C.byref(viol), C.byref(error))
        return Series._wrap(_made(h, error, refusal, _bound_what("series product", viol), check))

    __mul__ = lambda self, other: self.mul(other)                 # noqa: E731

    def pow(self, k, *, bound=None, declaration=None, budget=None):
        """a^k by sequential products, every power a^j (j <= k) within the bound"""
        error, refusal, check, viol = C.c_int32(), _Refusal(), _Check(), C.c_uint64()
        h = _call("tr_conv_series_pow", self._open(), int(k), self._bound(None, bound), _decl(declaration),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(check), C.byref(viol), C.byref(error))
        return Series._wrap(_made(h, error, refusal, _bound_what("series power", viol), check))

    __pow__ = lambda self, k: self.pow(k)                         # noqa: E731

    def walks_need(self, max_length=None, *, rows=True):
        raw, lengths = _Need(), C.c_uint64()
        _check(_call("tr_conv_series_walks_need", self._open(), UNBOUNDED if max_length is None else int(max_length),
                     int(bool(rows)), C.byref(raw), C.byref(lengths)), what="walk table needs")
        return dict(_needs(raw), lengths=int(lengths.value))

    def walks(self, max_length=None, *, bound=None, rows=True, declaration=None, backend="cpu", budget=None):
        """the walk table of this step series: rows[l] = s^l mod z^N (walks of exactly l steps by total), total = their
        sum for l <= max_length (None: any length, which needs a zero constant term: else NoSolutionError)"""
        L = UNBOUNDED if max_length is None else int(max_length)
        b = self._bound(None, bound)
        lengths = self.walks_need(max_length, rows=rows)["lengths"]
        n = self.order
        info, refusal, check = _Walks(), _Refusal(), _Check()
        if backend == "cpu":
            buf = np.zeros(max(lengths * n, 1), dtype=np.int64) if rows else None
            tot = vp()
            rc = _call("tr_conv_series_walks", self._open(), L, b, _decl(declaration),
                       buf.ctypes.data if rows else None, lengths * n if rows else 0, C.byref(tot), C.byref(info),
                       C.byref(_budget(budget)), C.byref(refusal), C.byref(check))
            _checked(rc, refusal, check, _walk_what(info, n))
            total, label = Series._wrap(tot.value), "cpu"
        elif backend == "cuda":
            k = 2 if lengths > 2 else lengths - 1
            rc = _call("tr_conv_series_check", self._open(), None, k, _decl(declaration), C.byref(check))
            _checked(rc, refusal, check, "walk table")
            buf = np.zeros(max(lengths * n, 1), dtype=np.int64)
            step = np.array(self.coefficients, dtype=np.int64)
            tot = np.zeros(n, dtype=np.int64)
            try:
                _gpu().run("tr_mgpu_conv_walks", step.ctypes.data, n, lengths, b, buf.ctypes.data, tot.ctypes.data,
                           C.byref(info), C.byref(_budget(budget)), C.byref(refusal), refusal=refusal,
                           what="walk table on the device")
            except OverflowError:
                raise OverflowError(_walk_what(info, n) + " on the device") from None
            error = C.c_int32()
            h = _call("tr_conv_series_walks_adopt", self._open(), tot.ctypes.data, b, C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
            total, label = Series._wrap(_made(h, error, refusal, "walk totals")), _device_label()
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        out = dict(lengths=int(info.lengths), total=total, maxabs=int(info.maxabs), bound=b, backend=label)
        if rows:
            out["rows"] = [[int(x) for x in buf[l * n:(l + 1) * n]] for l in range(lengths)]
        self._last_walks = info
        return out

    def to_factor(self, *, repr="sparse", budget=None):
        """the one-variable factor of the coefficients (at the values shift + e), with the series' sources"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_conv_series_to_factor", self._open(), _mg.REPRS[repr], C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Factor._wrap(_made(h, error, refusal, "series factor"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_conv_series_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __eq__(self, other):
        return isinstance(other, Series) and self.order == other.order and self.coefficients == other.coefficients

    __hash__ = None

    def __repr__(self):
        return f"Series({self.coefficients}, order={self.order}, bound={self.bound})" if self._handle else "Series(closed)"


def _bound_what(what, viol):
    v = int(viol.value)
    return what if v == NONE else f"{what} (first coefficient past the bound: {v})"


def _walk_what(info, n):
    v = int(info.violation)
    if v == NONE or not n:
        return "walk table"
    l, e = divmod(v, n)
    if l >= int(info.lengths):
        return f"walk table (total coefficient {e} exceeds the declared bound)"
    return f"walk table (row {l}, coefficient {e} exceeds the declared bound)"


def series(f, order, bound=(1 << 63) - 1, *, budget=None):
    """P mod z^order of a one-variable factor as a bounded series (int64, |c| <= bound), keeping its sources"""
    if not isinstance(f, Factor):
        raise TypeError("a math_conv.Factor")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_conv_series_from_factor", f._open(), int(order), _i64(bound, "bound"), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Series._wrap(_made(h, error, refusal, "bounded series"))
