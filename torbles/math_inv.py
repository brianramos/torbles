# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""INV-001: linear and univariate polynomial inverse answers ("which parameters give this target"), natively on the
CPU or with the CUDA kernels (torbles/math_inv.c, math_inv_gpu.cu).

    from fractions import Fraction
    from torbles import math_inv as mi
    from torbles.mathematics import Context
    ctx = Context(); t, u = ctx.symbols("t", "u")
    mi.solve([(2*t + u, 3), (t - u, 0)])                  # kind "unique", point {"t": 1, "u": 1}
    mi.solve([(t + u, 2)], domain={"t": (0, 1), "u": (None, 1)})   # "unique" (1, 1): the box admits one point
    mi.solve_affine([[1, -1]], [0])                       # "family": witness, directions, parameter ranges
    r = mi.roots(t**3 - 2*t, 0, lo=-2, hi=2, width=Fraction(1, 2**40))   # 0 exact; +-sqrt(2) isolated
    r = mi.roots([-2, 0, 1], lo=0, backend="cuda", width=Fraction(1, 10**12))   # coefficients, low degree first
    mi.signs([[-2, 0, 1]], [Fraction(1), Fraction(3, 2)], which=[0, 0])  # exact signs (CPU or CUDA)
    s = mi.sweep(f, {"t": range(8), "u": [Fraction(k, 2) for k in range(8)]}, target=3)
    s.complete, s.searched, s.next, s.matches, s.bound_bits

Every answer is exact. A linear answer is "unique" (the point), "empty" (reason "inconsistent" or "outside" the
domain), "whole" (the entire domain) or "family" (a witness inside the domain, directions, the exact range of each
parameter). A root is exact only when an exact evaluation proved it (Root.how); every other root is an open interval
(lo, hi) with a certified sign change and exactly one root inside. A finite-domain search that its iteration or output
budget truncates is reported incomplete with the index to resume from. Python marshals values (Fraction <-> native
big rationals of math_carriers) and maps native codes to exceptions; it computes nothing. Opt-in: `import torbles`
does not load this module.
"""
import ctypes as C
import os
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from . import math_carriers as _mc
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check

__all__ = ["solve", "solve_affine", "roots", "signs", "sweep", "primes", "Linear", "Roots", "Root", "Signs", "Sweep",
           "KINDS", "REASONS", "PRIMES_MAX", "SWEEP_DIMS"]

PRIMES_MAX, POINT_LIMBS_MAX, SWEEP_DIMS, LINEAR_MAX = 128, 128, 3, 64
KINDS = {0: "empty", 1: "unique", 2: "whole", 3: "family", 4: "finite"}
REASONS = {0: None, 1: "inconsistent", 2: "outside", 3: "constant"}
SEARCH = {0: "complete", 1: "iterations", 2: "output"}
HOW = {0: "isolated", 1: "evaluated", 2: "linear factor", 3: "rational candidate", 4: "irrational"}
P = C.POINTER


class _LinearInfo(C.Structure):
    _native_ = "tr_inv_linear_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "reason", "unknowns", "equations", "rank", "dims", "fixed",
                                         "reserved")] + [("iterations", C.c_uint64)]


class _RootsInfo(C.Structure):
    _native_ = "tr_inv_roots_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "reason", "degree", "nroots", "nfactors", "exact", "lo_bounded",
                                         "hi_bounded")] + [("iterations", C.c_uint64), ("rounds", C.c_uint64)]


class _RootInfo(C.Structure):
    _native_ = "tr_inv_root_info"
    _fields_ = [(n, C.c_int32) for n in ("exact", "multiplicity", "boundary", "sign_lo", "factor", "how")]


class _SignsView(C.Structure):
    _native_ = "tr_inv_signs_view"
    _fields_ = [("npoints", C.c_uint64), ("nprimes", C.c_uint32), ("npolys", C.c_uint32), ("limbs", C.c_uint32),
                ("total", C.c_uint32)] + [(n, C.c_void_p) for n in ("fields", "garner", "half", "offset", "coef",
                                                                    "poly", "sign", "num", "den")]


class _SweepView(C.Structure):
    _native_ = "tr_inv_sweep_view"
    _fields_ = [(n, C.c_uint64) for n in ("start", "count", "total", "nvalues")] + [
        ("size", C.c_uint64 * 3), ("voff", C.c_uint64 * 3), ("nprimes", C.c_uint32), ("nterms", C.c_uint32),
        ("ndims", C.c_int32), ("reserved", C.c_int32)] + [(n, C.c_void_p) for n in ("fields", "exps", "cres", "vres")]


class _SweepInfo(C.Structure):
    _native_ = "tr_inv_sweep_info"
    _fields_ = [(n, C.c_uint64) for n in ("total", "start", "searched", "next", "matches", "bound_bits")] + [
        (n, C.c_int32) for n in ("complete", "reason", "ndims", "nprimes")]


for _s, _n in ((_LinearInfo, 40), (_RootsInfo, 48), (_RootInfo, 24), (_SignsView, 96), (_SweepView, 128),
               (_SweepInfo, 64)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u32, u64, B, R, E = C.c_void_p, C.c_int32, C.c_uint32, C.c_uint64, P(_Budget), P(_Refusal), P(C.c_int32)
        sig = {
            "tr_inv_linear_solve": (vp, [i32, i32, vp, vp, vp, vp, B, R, E]),
            "tr_inv_linear_from_polys": (vp, [i32, vp, vp, i32, vp, vp, vp, B, R, E]),
            "tr_inv_linear_get_info": (i32, [vp, P(_LinearInfo)]),
            "tr_inv_linear_point": (vp, [vp, i32]),
            "tr_inv_linear_direction": (vp, [vp, i32, i32]),
            "tr_inv_linear_free_unknown": (i32, [vp, i32]),
            "tr_inv_linear_range": (i32, [vp, i32, P(vp), P(vp)]),
            "tr_inv_linear_free": (None, [vp]),
            "tr_inv_signs_create": (vp, [u32, vp, vp, u64, vp, vp, B, R, E]),
            "tr_inv_signs_get_view": (i32, [vp, P(_SignsView)]),
            "tr_inv_signs_eval": (i32, [P(_SignsView), vp, B, R]),
            "tr_inv_signs_bound_bits": (u64, [vp]),
            "tr_inv_signs_device_need": (u64, [P(_SignsView)]),
            "tr_inv_signs_free": (None, [vp]),
            "tr_inv_isolate": (vp, [i32, vp, vp, vp, vp, B, R, E]),
            "tr_inv_isolate_poly": (vp, [vp, i32, vp, vp, vp, B, R, E]),
            "tr_inv_roots_refine": (vp, [vp, vp, vp, B, R, E]),
            "tr_inv_roots_get_info": (i32, [vp, P(_RootsInfo)]),
            "tr_inv_roots_root": (i32, [vp, i32, P(_RootInfo), P(vp), P(vp)]),
            "tr_inv_roots_factor_degree": (i32, [vp, i32]),
            "tr_inv_roots_factor_coef": (vp, [vp, i32, i32]),
            "tr_inv_roots_bound": (vp, [vp, i32]),
            "tr_inv_roots_free": (None, [vp]),
            "tr_inv_sweep_prepare": (vp, [vp, vp, i32, vp, vp, vp, u64, B, R, E]),
            "tr_inv_sweep_get_view": (i32, [vp, P(_SweepView)]),
            "tr_inv_sweep_eval": (i32, [P(_SweepView), vp]),
            "tr_inv_sweep_finish": (vp, [vp, vp, B, R, E]),
            "tr_inv_sweep_get_info": (i32, [vp, P(_SweepInfo)]),
            "tr_inv_sweep_match": (i32, [vp, u64, P(u64), vp]),
            "tr_inv_sweep_device_need": (u64, [P(_SweepView)]),
            "tr_inv_sweep_plan_free": (None, [vp]),
            "tr_inv_sweep_free": (None, [vp]),
            "tr_inv_primes_verify": (i32, []),
            "tr_inv_prime": (i32, [i32, P(u64)]),
        }
        native = {}
        for name, (restype, argtypes) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            native[name[7:]] = fn
        _native = native
    return _native


def primes():
    """the native moduli table of the multimodular evaluations and its native verification (distinct primes in
    (2^61, 2^62), decreasing, deterministic Miller-Rabin)"""
    api = _api()
    out = []
    for i in range(PRIMES_MAX):
        v = C.c_uint64()
        _check(api["prime"](i, C.byref(v)))
        out.append(int(v.value))
    return out, bool(api["primes_verify"]())


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_inv_signs", C.c_int, [P(_SignsView), C.c_void_p, P(_Budget), P(_Refusal)])
        math_gpu.bind("tr_mgpu_inv_sweep", C.c_int, [P(_SweepView), C.c_void_p, P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _device_label(gpu):
    info = gpu.device()
    return ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]


def _raw(budget):
    return C.byref((budget or Budget()).raw)


# ---------------------------------------------------------------------------------------------------- exact values
def _exact(value, what="value"):
    """an exact rational from int, Fraction (any numbers.Rational) or a finite Decimal; floats are refused"""
    if isinstance(value, bool):
        raise TypeError(f"{what}: bool is not an exact number")
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{what}: a finite Decimal is required")
        return Fraction(value)
    if isinstance(value, _mc.BigRational):
        return value.to_fraction()
    raise TypeError(f"{what}: use an int, a Fraction or a finite Decimal (floats are not exact input)")


class _Values:
    """native big rationals of exact Python values, alive while this object is (handles in a void* array)"""
    def __init__(self, values, what="value", allow_none=False):
        self.keep = []
        values = list(values)
        self.array = (C.c_void_p * max(len(values), 1))()
        for i, v in enumerate(values):
            if v is None:
                if not allow_none:
                    raise TypeError(f"{what}: a value is required")
                self.array[i] = None
                continue
            q = _mc.BigRational(_exact(v, what))
            self.keep.append(q)
            self.array[i] = q._h
        self.n = len(values)


def _one(value, what):
    """one native big rational (or None) and the object that keeps it alive"""
    if value is None:
        return None, None
    q = _mc.BigRational(_exact(value, what))
    return q._h, q


def _fraction(handle):
    """a Fraction from a borrowed tr_bigq handle"""
    if not handle:
        return None
    api = _mc._api()
    n, d = _mc._bigint_value(api["bigq_numerator"](handle)), _mc._bigint_value(api["bigq_denominator"](handle))
    return Fraction(n, d)


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _expression_handle(expr):
    from .mathematics import Expression
    if not isinstance(expr, Expression):
        raise TypeError("a first-slice Expression (torbles.mathematics) is required")
    return expr._open()


# ---------------------------------------------------------------------------------------------------- linear targets
class Linear:
    """The exact solution set of linear (affine) equations inside a box domain (native, immutable once built).

    kind: "unique" (point), "empty" (reason "inconsistent" or "outside"), "whole" (every point of the domain) or
    "family" (point = a witness inside the domain; directions[d] = {unknown: coordinate}; free[d] = the unknown whose
    offset from the witness is parameter y_d; ranges[d] = the exact (lo, hi) of y_d over the family in the domain,
    None for an unbounded side). Solutions are point + sum_d y_d directions[d] with y in the ranges' polytope."""

    def __init__(self, handle, names):
        api = _api()
        try:
            info = _LinearInfo()
            _check(api["linear_get_info"](handle, C.byref(info)), what="linear answer")
            self.names = tuple(names)
            self.kind, self.reason = KINDS[info.kind], REASONS[info.reason] if info.kind == 0 else None
            self.rank, self.dims, self.fixed, self.iterations = int(info.rank), int(info.dims), int(info.fixed), \
                int(info.iterations)
            self.equations = int(info.equations)
            n = len(self.names)
            self.point = None
            if info.kind in (1, 3):
                self.point = {self.names[j]: _fraction(api["linear_point"](handle, j)) for j in range(n)}
            self.directions, self.free, self.ranges = [], [], []
            for d in range(self.dims):
                self.directions.append({self.names[j]: _fraction(api["linear_direction"](handle, d, j))
                                        for j in range(n)})
                self.free.append(self.names[_check(api["linear_free_unknown"](handle, d), what="linear answer")])
                lo, hi = C.c_void_p(), C.c_void_p()
                _check(api["linear_range"](handle, d, C.byref(lo), C.byref(hi)), what="linear answer")
                self.ranges.append((_fraction(lo.value), _fraction(hi.value)))
        finally:
            api["linear_free"](handle)

    def __repr__(self):
        extra = f", reason={self.reason!r}" if self.reason else ""
        if self.kind == "unique":
            extra = f", point={self.point}"
        elif self.kind == "family":
            extra = f", point={self.point}, free={self.free}, ranges={self.ranges}"
        return f"Linear({self.kind!r}{extra})"


def _bounds(names, domain):
    domain = dict(domain or {})
    unknown = set(domain) - set(names)
    if unknown:
        raise ValueError(f"domain bounds for unknowns that are not solved for: {sorted(unknown)}")
    lo, hi = [], []
    for n in names:
        b = domain.get(n, (None, None))
        if b is None:
            b = (None, None)
        if len(b) != 2:
            raise ValueError(f"the domain of {n!r} is a (lo, hi) pair (None = unbounded)")
        lo.append(b[0]); hi.append(b[1])
    return _Values(lo, "lower bound", True), _Values(hi, "upper bound", True)


def solve(equations, unknowns=None, domain=None, budget=None):
    """The exact solutions of equations f_i = target_i (first-slice Expressions of degree <= 1 in the unknowns) in a
    box domain {name: (lo, hi)} (None = unbounded). equations: (expression, target) pairs or expressions (= 0).
    unknowns: names of the context (default: the variables the expressions use, in context order)."""
    pairs = []
    for e in equations:
        if isinstance(e, tuple):
            if len(e) != 2:
                raise ValueError("an equation is (expression, target)")
            pairs.append(e)
        else:
            pairs.append((e, 0))
    if not pairs and unknowns is None:
        raise ValueError("give the unknowns when there are no equations")
    context = None
    for expr, _ in pairs:
        _expression_handle(expr)
        if context is None:
            context = expr.context
        elif expr.context is not context:
            from .math_domains import DomainError
            raise DomainError("the equations belong to different variable contexts")
    if unknowns is None:
        used = set()
        for expr, _ in pairs:
            used |= set(expr.support)
        names = [n for n in context.names if n in used]
        if not names:
            raise ValueError("the equations use no variable: give the unknowns explicitly")
    else:
        names = list(unknowns)
        if context is None:
            raise ValueError("solve needs at least one equation to know the variable context (use solve_affine)")
    slots = []
    for n in names:
        if n not in context.names:
            raise ValueError(f"unknown variable {n!r} in this context")
        slots.append(context.names.index(n))
    if len(set(slots)) != len(slots):
        raise ValueError("the unknowns must be distinct")
    polys = (C.c_void_p * max(len(pairs), 1))(*[expr._open() for expr, _ in pairs])
    targets = _Values([t for _, t in pairs], "target")
    lo, hi = _bounds(names, domain)
    idx = (C.c_int32 * max(len(slots), 1))(*slots)
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["linear_from_polys"](len(pairs), polys, targets.array, len(slots), idx, lo.array, hi.array,
                                    _raw(budget), C.byref(refusal), C.byref(error))
    return Linear(_made(h, error, refusal, "linear target"), names)


def solve_affine(A, b, lo=None, hi=None, names=None, budget=None):
    """The exact solutions of A x = b (rational rows) with lo <= x <= hi (lists with None = unbounded)."""
    rows = [list(r) for r in A]
    m = len(rows)
    n = len(rows[0]) if rows else (len(names) if names else (len(lo) if lo is not None else 0))
    if any(len(r) != n for r in rows):
        raise ValueError("every row of A has one coefficient per unknown")
    if len(list(b)) != m:
        raise ValueError("b has one value per equation")
    names = list(names) if names is not None else [f"x{j}" for j in range(n)]
    if len(names) != n or n < 1:
        raise ValueError("one name per unknown (at least one unknown)")
    a = _Values([v for r in rows for v in r], "coefficient")
    bb = _Values(list(b), "right-hand side")
    lv = _Values(lo if lo is not None else [None] * n, "lower bound", True)
    hv = _Values(hi if hi is not None else [None] * n, "upper bound", True)
    if lv.n != n or hv.n != n:
        raise ValueError("lo and hi have one bound per unknown")
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["linear_solve"](m, n, a.array, bb.array, lv.array, hv.array, _raw(budget), C.byref(refusal),
                               C.byref(error))
    return Linear(_made(h, error, refusal, "affine target"), names)


# ---------------------------------------------------------------------------------------------------- roots
class Root:
    """One real root of f - target: exact (value) or an open isolating interval (lo, hi) with exactly one root of
    f - target inside and a certified sign change of its square-free factor (sign_lo at lo, -sign_lo at hi)."""
    __slots__ = ("exact", "lo", "hi", "multiplicity", "boundary", "sign_lo", "how")

    def __init__(self, exact, lo, hi, multiplicity, boundary, sign_lo, how):
        self.exact, self.lo, self.hi, self.multiplicity = exact, lo, hi, multiplicity
        self.boundary, self.sign_lo, self.how = boundary, sign_lo, how

    @property
    def value(self):
        """the exact root (None for an isolating interval: an irrational root is never given as a number)"""
        return self.lo if self.exact else None

    @property
    def width(self):
        return self.hi - self.lo

    def __repr__(self):
        b = f", boundary={self.boundary!r}" if self.boundary else ""
        if self.exact:
            return f"Root(exact {self.lo}, multiplicity={self.multiplicity}{b})"
        return f"Root(({self.lo}, {self.hi}), multiplicity={self.multiplicity}, {self.how})"


class Roots:
    """Every real root of f - target in the declared bounds (native). kind "finite" (roots), "empty" (no root; reason
    "constant" for a nonzero constant) or "whole" (f - target is the zero polynomial: every point of the bounds)."""

    def __init__(self, handle, backend="cpu", device="native CPU"):
        self._h, self.backend, self.device = handle, backend, device
        api = _api()
        info = _RootsInfo()
        _check(api["roots_get_info"](handle, C.byref(info)), what="roots")
        self.kind, self.reason = KINDS[info.kind], REASONS[info.reason] if info.kind == 0 else None
        self.degree, self.iterations, self.rounds = int(info.degree), int(info.iterations), int(info.rounds)
        self.lo = _fraction(api["roots_bound"](handle, 0))
        self.hi = _fraction(api["roots_bound"](handle, 1))
        out = []
        for i in range(int(info.nroots)):
            ri, lo, hi = _RootInfo(), C.c_void_p(), C.c_void_p()
            _check(api["roots_root"](handle, i, C.byref(ri), C.byref(lo), C.byref(hi)), what="root")
            b = {0: None, 1: "lo", 2: "hi", 3: "lo and hi"}[ri.boundary]
            out.append(Root(bool(ri.exact), _fraction(lo.value), _fraction(hi.value), int(ri.multiplicity), b,
                            int(ri.sign_lo), HOW[ri.how]))
        self.roots = tuple(out)
        self.factors = []
        for i in range(int(info.nfactors)):
            d = api["roots_factor_degree"](handle, i)
            self.factors.append([_mc._bigint_value(api["roots_factor_coef"](handle, i, j)) for j in range(d + 1)])

    def refine(self, width, backend="cpu", budget=None):
        """a new answer whose isolating intervals have width <= width; the signs at the split points are computed by
        the native CPU evaluator or the CUDA kernel (the same exact signs, so the same intervals)"""
        backend = _backend(backend)
        w, keep = _one(width, "width")
        if w is None or keep.to_fraction() <= 0:
            raise ValueError("the width is a positive exact number")
        fn, device = None, "native CPU"
        if backend == "cuda":
            gpu = _gpu()
            if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
                gpu.require_device()
            fn = C.cast(gpu.lib().tr_mgpu_inv_signs, C.c_void_p).value
            backend, device = _device_label(gpu)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["roots_refine"](self._open(), w, fn, _raw(budget), C.byref(refusal), C.byref(error))
        return Roots(_made(h, error, refusal, "root refinement"), backend, device)

    def _open(self):
        if not self._h:
            raise ValueError("the roots are closed")
        return self._h

    @property
    def exact(self):
        return [r.value for r in self.roots if r.exact]

    def close(self):
        if getattr(self, "_h", None):
            _api()["roots_free"](self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.roots)

    def __iter__(self):
        return iter(self.roots)

    def __repr__(self):
        return f"Roots({self.kind!r}, {list(self.roots)})"


def roots(f, target=0, lo=None, hi=None, variable=None, width=None, backend="cpu", budget=None):
    """Every real root of f(t) = target with lo <= t <= hi (None = unbounded), exactly: f is a first-slice Expression
    in one variable (another variable in its support is DomainError) or its coefficients, low degree first.
    width: refine every isolating interval to at most this width (signs by the backend: "cpu" or "cuda")."""
    backend = _backend(backend)
    tq, k1 = _one(target, "target")
    lq, k2 = _one(lo, "lower bound")
    hq, k3 = _one(hi, "upper bound")
    error, refusal = C.c_int32(), _Refusal()
    from .mathematics import Expression
    if isinstance(f, Expression):
        if variable is None:
            support = f.support
            if len(support) > 1:
                raise ValueError(f"f uses {list(support)}: name the variable (the others must be substituted)")
            slot = f.context.names.index(support[0]) if support else 0
        else:
            slot = f.context._index(variable)
        h = _api()["isolate_poly"](f._open(), slot, tq, lq, hq, _raw(budget), C.byref(refusal), C.byref(error))
    else:
        coef = _Values(list(f), "coefficient", True)
        if coef.n < 1:
            raise ValueError("give at least one coefficient")
        h = _api()["isolate"](coef.n, coef.array, tq, lq, hq, _raw(budget), C.byref(refusal), C.byref(error))
    out = Roots(_made(h, error, refusal, "root isolation"))
    del k1, k2, k3
    if width is not None:
        refined = out.refine(width, backend, budget)
        out.close()
        return refined
    return out


# ---------------------------------------------------------------------------------------------------- batched signs
class Signs:
    """exact signs (-1, 0, 1) of polynomials at rational points, with the backend that evaluated them and the proof:
    the value bound (|v^d P(u/v)| < 2^bound_bits) and the number of primes of the multimodular evaluation"""
    __slots__ = ("values", "backend", "device", "primes", "bound_bits")

    def __init__(self, values, backend, device, primes, bound_bits):
        self.values, self.backend, self.device = values, backend, device
        self.primes, self.bound_bits = primes, bound_bits

    def __repr__(self):
        return f"Signs({self.values}, backend={self.backend!r}, primes={self.primes})"


def signs(polys, points, which=None, backend="cpu", budget=None):
    """The exact sign of polys[which[i]] at points[i] (polys: coefficient lists, low degree first; which defaults to
    0 for every point), natively on the CPU or with the CUDA kernel."""
    backend = _backend(backend)
    polys = [list(p) for p in polys]
    if not polys or any(not p for p in polys):
        raise ValueError("give at least one polynomial, each with at least one coefficient")
    pts = list(points)
    which = [0] * len(pts) if which is None else [int(w) for w in which]
    if len(which) != len(pts) or any(not 0 <= w < len(polys) for w in which):
        raise ValueError("which: one polynomial index per point")
    ncoef = (C.c_uint32 * len(polys))(*[len(p) for p in polys])
    coef = _Values([c for p in polys for c in p], "coefficient", True)
    pv = _Values(pts, "point")
    idx = (C.c_uint32 * max(len(pts), 1))(*which)
    error, refusal = C.c_int32(), _Refusal()
    api = _api()
    job = _made(api["signs_create"](len(polys), ncoef, coef.array, len(pts), pv.array, idx, _raw(budget),
                                    C.byref(refusal), C.byref(error)), error, refusal, "sign job")
    try:
        view = _SignsView()
        _check(api["signs_get_view"](job, C.byref(view)), what="sign job")
        out = np.zeros(max(len(pts), 1), dtype=np.int8)
        refusal = _Refusal()
        if backend == "cpu":
            _check(api["signs_eval"](C.byref(view), out.ctypes.data, _raw(budget), C.byref(refusal)), refusal,
                   "signs")
            label, device = "cpu", "native CPU"
        else:
            gpu = _gpu()
            gpu.run("tr_mgpu_inv_signs", C.byref(view), out.ctypes.data, _raw(budget), C.byref(refusal),
                    refusal=refusal, what="signs on the device")
            label, device = _device_label(gpu)
        return Signs([int(s) for s in out[:len(pts)]], label, device, int(view.nprimes),
                     int(api["signs_bound_bits"](job)))
    finally:
        api["signs_free"](job)


def signs_device_bytes(polys, points, which=None):
    """the bytes the CUDA sign evaluation copies to the device for this job (its budget need)"""
    polys = [list(p) for p in polys]
    pts = list(points)
    which = [0] * len(pts) if which is None else [int(w) for w in which]
    ncoef = (C.c_uint32 * len(polys))(*[len(p) for p in polys])
    coef = _Values([c for p in polys for c in p], "coefficient", True)
    pv = _Values(pts, "point")
    idx = (C.c_uint32 * max(len(pts), 1))(*which)
    error, refusal = C.c_int32(), _Refusal()
    api = _api()
    job = _made(api["signs_create"](len(polys), ncoef, coef.array, len(pts), pv.array, idx, None, C.byref(refusal),
                                    C.byref(error)), error, refusal, "sign job")
    try:
        view = _SignsView()
        _check(api["signs_get_view"](job, C.byref(view)))
        return int(api["signs_device_need"](C.byref(view)))
    finally:
        api["signs_free"](job)


# ---------------------------------------------------------------------------------------------------- finite domains
class Sweep:
    """The exact solutions of f = target on a finite grid, examined from index start in lexicographic order (the first
    coordinate most significant). complete: every grid point was examined; otherwise reason "iterations" (the budget
    truncated the search) or "output" (more matches than the output budget: the search stops before the first match
    it cannot list), and next is the index to resume from. matches: [{name: value}] in index order, all of those in
    start .. next - 1. bound_bits / primes: the proof (|H| < 2^bound_bits, residues modulo `primes` primes > 2^61)."""

    def __init__(self, handle, names, grid, backend, device):
        api = _api()
        try:
            info = _SweepInfo()
            _check(api["sweep_get_info"](handle, C.byref(info)), what="sweep")
            self.names, self.backend, self.device = tuple(names), backend, device
            self.total, self.start, self.searched, self.next = int(info.total), int(info.start), int(info.searched), \
                int(info.next)
            self.complete, self.reason = bool(info.complete), SEARCH[info.reason]
            self.bound_bits, self.primes = int(info.bound_bits), int(info.nprimes)
            self.indices, self.matches = [], []
            coords = (C.c_uint64 * 3)()
            index = C.c_uint64()
            for i in range(int(info.matches)):
                _check(api["sweep_match"](handle, i, C.byref(index), coords), what="sweep")
                self.indices.append(int(index.value))
                self.matches.append({n: grid[d][int(coords[d])] for d, n in enumerate(self.names)})
        finally:
            api["sweep_free"](handle)

    def __repr__(self):
        state = "complete" if self.complete else f"incomplete ({self.reason}, next={self.next})"
        return f"Sweep({len(self.matches)} matches, {self.searched}/{self.total - self.start} examined, {state})"


def sweep(f, grid, target=0, start=0, backend="cpu", budget=None):
    """Exhaustive exact search of f = target over grid = {name: values} (1..3 coordinates, first most significant;
    values exact) from grid index start. The budget's iterations bound the points examined in this call and its
    output the matches listed; a truncated search is incomplete, never silently partial (resume with start=next)."""
    backend = _backend(backend)
    fh = _expression_handle(f)
    items = list(grid.items()) if isinstance(grid, dict) else [(n, v) for n, v in grid]
    names = [n for n, _ in items]
    values = [[_exact(v, f"value of {n!r}") for v in vals] for n, vals in items]
    slots = [f.context._index(n) for n in names]
    flat = _Values([v for vals in values for v in vals], "grid value")
    sizes = (C.c_uint64 * max(len(items), 1))(*[len(v) for v in values])
    idx = (C.c_int32 * max(len(items), 1))(*slots)
    tq, keep = _one(target, "target")
    error, refusal = C.c_int32(), _Refusal()
    api = _api()
    start = int(start)
    if start < 0:
        raise ValueError("start is a grid index >= 0")
    plan = _made(api["sweep_prepare"](fh, tq, len(items), idx, sizes, flat.array, start, _raw(budget),
                                      C.byref(refusal), C.byref(error)), error, refusal, "finite-domain search")
    try:
        view = _SweepView()
        _check(api["sweep_get_view"](plan, C.byref(view)), what="sweep")
        bits = np.zeros(max((int(view.count) + 63) // 64, 1), dtype=np.uint64)
        if backend == "cpu":
            _check(api["sweep_eval"](C.byref(view), bits.ctypes.data), what="sweep")
            label, device = "cpu", "native CPU"
        else:
            gpu = _gpu()
            refusal = _Refusal()
            gpu.run("tr_mgpu_inv_sweep", C.byref(view), bits.ctypes.data, _raw(budget), C.byref(refusal),
                    refusal=refusal, what="sweep on the device")
            label, device = _device_label(gpu)
        error, refusal = C.c_int32(), _Refusal()
        res = _made(api["sweep_finish"](plan, bits.ctypes.data, _raw(budget), C.byref(refusal), C.byref(error)),
                    error, refusal, "sweep")
        out = Sweep(res, names, values, label, device)
        out.bitmap_words = bits.copy()
        return out
    finally:
        api["sweep_plan_free"](plan)
        del keep


def sweep_device_bytes(f, grid, target=0, start=0, budget=None):
    """the bytes the CUDA search copies to the device (its budget need)"""
    fh = _expression_handle(f)
    items = list(grid.items()) if isinstance(grid, dict) else [(n, v) for n, v in grid]
    values = [[_exact(v) for v in vals] for _, vals in items]
    flat = _Values([v for vals in values for v in vals], "grid value")
    sizes = (C.c_uint64 * max(len(items), 1))(*[len(v) for v in values])
    idx = (C.c_int32 * max(len(items), 1))(*[f.context._index(n) for n, _ in items])
    tq, keep = _one(target, "target")
    error, refusal = C.c_int32(), _Refusal()
    api = _api()
    plan = _made(api["sweep_prepare"](fh, tq, len(items), idx, sizes, flat.array, int(start), _raw(budget),
                                      C.byref(refusal), C.byref(error)), error, refusal, "finite-domain search")
    try:
        view = _SweepView()
        _check(api["sweep_get_view"](plan, C.byref(view)))
        return int(api["sweep_device_need"](C.byref(view)))
    finally:
        api["sweep_plan_free"](plan)
        del keep
