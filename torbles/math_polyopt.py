# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""OPT-002: bounded convex and polynomial optimization with feasibility and optimality evidence, natively on the CPU
(torbles/math_polyopt.c) with the enclosure of branch-and-bound boxes also on the CUDA kernel (math_polyopt_gpu.cu).

    from fractions import Fraction
    from torbles import math_polyopt as po, math_opt as opt
    r = po.lp([1, 1], A_ub=[[-1, -2]], b_ub=[-4], bounds=[(0, None), (0, None)])   # exact simplex (Bland)
    r.status, r.point, r.value, r.multipliers, r.kkt, r.unique, r.face             # optimal: KKT verified natively
    r = po.qp(H, g, A_ub=A, b_ub=b, A_eq=E, b_eq=e)       # x^T H x / 2 + g^T x, H PSD: exact active set
    r = po.qp_from(opt.Quadratic.from_moments(state, 0), A_eq=[[0, 1, 1]], b_eq=[1])   # an OPT-001 objective
    r.farkas                # infeasible: y >= 0, z with A^T y + E^T z = 0, b^T y + e^T z < 0 (verified)
    r.ray                   # unbounded: feasible anchor, direction d (A d <= 0, E d = 0), curvature d^T H d < 0,
                            # or curvature 0 and rate g(anchor)^T d < 0
    r = po.polynomial(f, {"x": (-2, 2), "y": (-2, 2)}, [(g, "<=", 0)], tolerance=Fraction(1, 10**6), backend="cuda")
    r.status                # "complete" (upper - lower <= tolerance), "incomplete" (budget / time), "infeasible"
    r.lower, r.upper, r.point, r.feasibility, r.leaves

An indefinite QP is not convex. On an affine feasible set (no inequality varying on it) it is decided exactly from the
inertia of the reduced Hessian (a curvature ray, or the convex reduced problem's optimum or rate ray); otherwise a
negative-curvature recession direction certifies unboundedness when one is found, else it is handed to the certified
branch and bound (handed=True) when its feasible polyhedron is bounded and of dimension <= 4 after the equalities are
eliminated; otherwise UnsupportedError. A result
is never a local optimum presented as a global one: "optimal" is exact with its KKT certificate; "complete" encloses
the global minimum in [lower, upper] within the tolerance with a feasible witness attaining upper; "incomplete"
reports the certified bounds reached. Python marshals values (Fraction <-> native big rationals) and maps native codes
to exceptions; it computes nothing. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_carriers as _mc
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check
from .math_inv import _Values, _one, _fraction, _exact, _made

__all__ = ["lp", "qp", "qp_from", "polynomial", "Problem", "Result", "boxes", "STATUSES", "METHODS", "MAX_VARS",
           "BB_VARS", "BB_CONS", "DENSE_MAX"]

MAX_VARS, MAX_ROWS, BB_VARS, BB_CONS, DENSE_MAX, MAX_DEGREE = 64, 256, 4, 64, 256, 8
STATUSES = {1: "optimal", 2: "infeasible", 3: "unbounded", 4: "complete", 5: "incomplete"}
METHODS = {0: None, 1: "simplex", 2: "active set", 3: "branch and bound", 4: "roots"}
FEASIBILITY = {0: None, 1: "feasible", 2: "infeasible", 3: "unknown"}
REASONS = {0: None, 1: "nodes", 2: "time", 3: "depth"}
BOX_STATES = {0: "keep", 1: "infeasible", 2: "bound"}
KINDS = {"<=": 0, ">=": 1, "=": 2, "==": 2}
PART = dict(point=0, value=1, lower=2, upper=3, lam=4, mu=5, lam_lo=6, lam_hi=7, farkas_y=8, farkas_z=9,
            farkas_lo=10, farkas_hi=11, anchor=12, direction=13, rate=14, face=15, gradient=16, slack=17, tolerance=18,
            curvature=19)
P = C.POINTER


class _Info(C.Structure):
    _native_ = "tr_polyopt_info"
    _fields_ = [(n, C.c_int32) for n in ("status", "method", "n", "m", "p", "unique", "convex", "kkt", "feasibility",
                                         "reason", "certificate", "handed", "active", "reserved")] + [
        (n, C.c_uint64) for n in ("iterations", "lps", "pruned_infeasible", "pruned_bound", "pruned_tolerance", "open",
                                  "leaves", "incumbents", "batches")]


class _Spec(C.Structure):
    _native_ = "tr_polyopt_spec"
    _fields_ = [("slice", C.c_void_p), ("nterms", C.c_int32), ("kind", C.c_int32), ("exps", C.c_void_p),
                ("coef", C.c_void_p), ("rhs", C.c_void_p)]


class _View(C.Structure):
    _native_ = "tr_polyopt_encl_view"
    _fields_ = [("n", C.c_int32), ("npolys", C.c_int32), ("ncons", C.c_int32), ("neq", C.c_int32),
                ("offset", C.c_uint32 * (BB_CONS + 2)), ("deg", C.c_uint8 * ((BB_CONS + 1) * BB_VARS)),
                ("clo", C.c_void_p), ("chi", C.c_void_p)]


class _BoxOut(C.Structure):
    _native_ = "tr_polyopt_box_out"
    _fields_ = [("flo", C.c_double), ("fhi", C.c_double), ("status", C.c_int32), ("violated", C.c_int32),
                ("inner", C.c_int32), ("reserved", C.c_int32)]


for _s, _n in ((_Info, 128), (_Spec, 40), (_View, 560), (_BoxOut, 32)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, dbl, B, R, E = C.c_void_p, C.c_int32, C.c_uint64, C.c_double, P(_Budget), P(_Refusal), P(C.c_int32)
        sig = {
            "tr_polyopt_qp_create": (vp, [i32, vp, vp, vp, i32, vp, vp, i32, vp, vp, vp, vp, B, R, E]),
            "tr_polyopt_qp_from_quad": (vp, [vp, vp, i32, vp, vp, i32, vp, vp, vp, vp, B, R, E]),
            "tr_polyopt_qp_dims": (i32, [vp, P(i32), P(i32), P(i32)]),
            "tr_polyopt_qp_solve": (vp, [vp, i32, vp, u64, vp, dbl, B, R, E]),
            "tr_polyopt_qp_value": (vp, [vp, vp, B, R, E]),
            "tr_polyopt_qp_free": (None, [vp]),
            "tr_polyopt_poly_create": (vp, [i32, vp, P(_Spec), i32, vp, vp, vp, B, R, E]),
            "tr_polyopt_poly_get_view": (i32, [vp, P(_View)]),
            "tr_polyopt_poly_dims": (i32, [vp, P(i32), P(i32), P(i32)]),
            "tr_polyopt_poly_value": (vp, [vp, i32, vp, B, R, E]),
            "tr_polyopt_boxes": (i32, [P(_View), u64, vp, vp, dbl, vp, B, R]),
            "tr_polyopt_minimize": (vp, [vp, vp, u64, vp, dbl, B, R, E]),
            "tr_polyopt_poly_free": (None, [vp]),
            "tr_polyopt_result_get_info": (i32, [vp, P(_Info)]),
            "tr_polyopt_result_part": (vp, [vp, i32, P(u64)]),
            "tr_polyopt_result_value": (vp, [vp, i32, u64]),
            "tr_polyopt_result_leaf": (i32, [vp, u64, vp, vp, P(i32)]),
            "tr_polyopt_result_active": (i32, [vp, vp, i32]),
            "tr_polyopt_result_free": (None, [vp]),
        }
        native = {}
        for name, (restype, argtypes) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            native[name[11:]] = fn
        _native = native
    return _native


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_polyopt_boxes", C.c_int, [P(_View), C.c_uint64, C.c_void_p, C.c_void_p, C.c_double,
                                                         C.c_void_p, P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _raw(budget):
    return C.byref((budget or Budget()).raw)


def _fn(backend):
    """the enclosure evaluator: NULL (the native CPU one) or the CUDA entry point, and the backend's label"""
    if _backend(backend) == "cpu":
        return None, "cpu", "native CPU"
    gpu = _gpu()
    import os
    if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
        gpu.require_device()
    info = gpu.device()
    return C.cast(gpu.lib().tr_mgpu_polyopt_boxes, C.c_void_p).value, \
        ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]


# ---------------------------------------------------------------------------------------------------- results
class Result:
    """An exact optimization answer (native, verified before it is returned).

    status: "optimal" (exact optimum with KKT multipliers, uniqueness decided), "infeasible" (a Farkas certificate,
    or the leaves of a branch and bound that each violate a constraint), "unbounded" (a feasible ray), "complete"
    (the global minimum lies in [lower, upper], upper - lower <= tolerance, point attains upper exactly) or
    "incomplete" (a budget or time limit stopped the search: the certified [lower, upper] reached, upper None when
    no feasible point was found). feasibility: "feasible", "infeasible" or "unknown"."""

    def __init__(self, handle, names, backend="cpu", device="native CPU"):
        api = _api()
        try:
            i = _Info()
            _check(api["result_get_info"](handle, C.byref(i)), what="optimization result")
            self.names, self.backend, self.device = tuple(names), backend, device
            self.status, self.method = STATUSES.get(i.status), METHODS[i.method]
            self.feasibility, self.reason = FEASIBILITY[i.feasibility], REASONS[i.reason]
            self.unique = {1: True, 0: False}.get(int(i.unique))
            self.convex, self.kkt, self.certificate, self.handed = bool(i.convex), bool(i.kkt), bool(i.certificate), \
                bool(i.handed)
            self.iterations, self.lps, self.batches = int(i.iterations), int(i.lps), int(i.batches)
            self.pruned = dict(infeasible=int(i.pruned_infeasible), bound=int(i.pruned_bound),
                               tolerance=int(i.pruned_tolerance))
            self.open, self.incumbents = int(i.open), int(i.incumbents)
            part = lambda name: self._part(handle, name)
            one = lambda name: (part(name) or [None])[0]
            self.point, self.value = part("point"), one("value")
            self.lower, self.upper, self.tolerance = one("lower"), one("upper"), one("tolerance")
            self.gap = None if self.lower is None or self.upper is None else self.upper - self.lower
            self.gradient, self.slack, self.face = part("gradient"), part("slack"), part("face")
            self.multipliers = None
            if self.status == "optimal":
                self.multipliers = dict(ineq=part("lam") or [], eq=part("mu") or [], lower=part("lam_lo"),
                                        upper=part("lam_hi"))
            self.farkas = None
            if self.status == "infeasible" and self.certificate and part("farkas_lo") is not None:
                self.farkas = dict(y=part("farkas_y") or [], z=part("farkas_z") or [], lower=part("farkas_lo"),
                                   upper=part("farkas_hi"))
            self.ray = None
            if self.status == "unbounded":
                self.ray = dict(anchor=part("anchor"), direction=part("direction"), rate=one("rate"),
                                curvature=one("curvature"))
            na = api["result_active"](handle, None, 0)
            rows = (C.c_int32 * max(na, 1))()
            api["result_active"](handle, rows, na)
            self.active = [int(rows[k]) for k in range(max(na, 0))]
            self.leaves = []
            n = len(self.names)
            lo, hi, con = (C.c_void_p * max(n, 1))(), (C.c_void_p * max(n, 1))(), C.c_int32()
            for k in range(int(i.leaves) if self.status == "infeasible" and self.method == "branch and bound" else 0):
                _check(api["result_leaf"](handle, k, lo, hi, C.byref(con)), what="leaf")
                self.leaves.append((tuple(_fraction(lo[v]) for v in range(n)),
                                    tuple(_fraction(hi[v]) for v in range(n)), int(con.value)))
        finally:
            api["result_free"](handle)

    @staticmethod
    def _part(handle, name):
        api = _api()
        count = C.c_uint64()
        arr = api["result_part"](handle, PART[name], C.byref(count))
        if not arr:
            return None
        ptrs = C.cast(arr, P(C.c_void_p))
        return [_fraction(ptrs[k]) for k in range(int(count.value))]

    @property
    def exact(self):
        """the minimum is known exactly: an optimal answer, or bounds that meet"""
        return self.status == "optimal" or (self.status == "complete" and self.gap == 0)

    def same(self, other):
        keys = ("status", "method", "feasibility", "reason", "unique", "point", "value", "lower", "upper", "face",
                "multipliers", "farkas", "ray", "active", "leaves", "iterations", "pruned", "open", "incumbents")
        return all(getattr(self, k) == getattr(other, k) for k in keys)

    def __repr__(self):
        if self.status == "optimal":
            return f"Result(optimal, value={self.value}, point={self.point}, unique={self.unique})"
        if self.status in ("complete", "incomplete"):
            return (f"Result({self.status}, lower={self.lower}, upper={self.upper}, point={self.point}, "
                    f"feasibility={self.feasibility!r})")
        return f"Result({self.status})"


# ---------------------------------------------------------------------------------------------------- convex problems
def _matrix(rows, n, what):
    rows = [list(r) for r in (rows or [])]
    if any(len(r) != n for r in rows):
        raise ValueError(f"every row of {what} has {n} coefficients")
    return rows


def _box(bounds, n):
    bounds = [None] * n if bounds is None else list(bounds)
    if len(bounds) != n:
        raise ValueError(f"one bound pair (or None) per variable ({n})")
    lo, hi = [], []
    for b in bounds:
        l, h = (None, None) if b is None else b
        lo.append(l); hi.append(h)
    return _Values(lo, "lower bound", True), _Values(hi, "upper bound", True)


def _constraints(n, A_ub, b_ub, A_eq, b_eq):
    A = _matrix(A_ub, n, "A_ub")
    E = _matrix(A_eq, n, "A_eq")
    b = list(b_ub or [])
    e = list(b_eq or [])
    if len(b) != len(A) or len(e) != len(E):
        raise ValueError("one right-hand side per constraint row")
    return (_Values([v for r in A for v in r], "A_ub"), _Values(b, "b_ub"), len(A),
            _Values([v for r in E for v in r], "A_eq"), _Values(e, "b_eq"), len(E))


def _solve(handle, names, tolerance, uniqueness, backend, batch, time_limit, budget):
    fn, label, device = _fn(backend)
    tq, keep = _one(tolerance, "tolerance")
    error, refusal = C.c_int32(), _Refusal()
    try:
        h = _api()["qp_solve"](handle, 0 if uniqueness else 1, tq, int(batch), fn, float(time_limit or 0) * 1000.0,
                               _raw(budget), C.byref(refusal), C.byref(error))
        return Result(_made(h, error, refusal, "optimization"), names, label, device)
    finally:
        _api()["qp_free"](handle)
        del keep


def qp(H, g, c0=0, *, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, names=None, uniqueness=True,
       tolerance=Fraction(1, 2 ** 20), backend="cpu", batch=32, time_limit=None, budget=None):
    """Minimize x^T H x / 2 + g^T x + c0 subject to A_ub x <= b_ub, A_eq x = b_eq and bounds [(lo, hi) or None per
    variable; either side None]. H None is a linear program. An indefinite H is handed to the certified branch and
    bound (tolerance, backend of its box enclosures, batch, time_limit in seconds)."""
    g = list(g)
    n = len(g)
    if n < 1:
        raise ValueError("at least one variable")
    hv = None
    if H is not None:
        rows = _matrix(H, n, "H")
        if len(rows) != n:
            raise ValueError("H is n x n")
        hv = _Values([v for r in rows for v in r], "H")
    gv = _Values(g, "g")
    cq, kc = _one(c0, "c0")
    av, bv, m, ev, fv, p = _constraints(n, A_ub, b_ub, A_eq, b_eq)
    lo, hi = _box(bounds, n)
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["qp_create"](n, hv.array if hv else None, gv.array, cq, m, av.array, bv.array, p, ev.array, fv.array,
                            lo.array, hi.array, _raw(budget), C.byref(refusal), C.byref(error))
    h = _made(h, error, refusal, "problem")
    return _solve(h, names or [f"x{i}" for i in range(n)], tolerance, uniqueness, backend, batch, time_limit, budget)


def lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, *, names=None, uniqueness=True, budget=None):
    """Minimize c^T x subject to A_ub x <= b_ub, A_eq x = b_eq and bounds, by the exact simplex method."""
    return qp(None, c, 0, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, names=names,
              uniqueness=uniqueness, budget=budget)


def qp_from(quadratic, *, linear=None, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, uniqueness=True,
            tolerance=Fraction(1, 2 ** 20), backend="cpu", batch=32, time_limit=None, budget=None):
    """An OPT-001 objective q(t) = t^T A t / 2 - b^T t + c (math_opt.Quadratic: from matrices, a LIN-001 moment
    state or a first-slice polynomial) plus `linear`^T t, under general linear constraints."""
    n = quadratic.n
    lv = _Values(list(linear) if linear is not None else [None] * n, "linear", True)
    if lv.n != n:
        raise ValueError(f"linear has {n} coefficients")
    av, bv, m, ev, fv, p = _constraints(n, A_ub, b_ub, A_eq, b_eq)
    lo, hi = _box(bounds, n)
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["qp_from_quad"](quadratic._h, lv.array, m, av.array, bv.array, p, ev.array, fv.array, lo.array,
                               hi.array, _raw(budget), C.byref(refusal), C.byref(error))
    h = _made(h, error, refusal, "problem")
    return _solve(h, quadratic.names, tolerance, uniqueness, backend, batch, time_limit, budget)


# ---------------------------------------------------------------------------------------------------- polynomials
class _PolySpec:
    """a native tr_polyopt_spec of a polynomial (a first-slice Expression or {exponents: coefficient}) and its
    keepalive objects"""

    def __init__(self, poly, n, kind=0, rhs=None):
        from .mathematics import Expression
        self.keep = []
        self.spec = _Spec()
        self.spec.kind = kind
        if rhs is not None:
            q, k = _one(rhs, "right-hand side")
            self.spec.rhs = q
            self.keep.append(k)
        if isinstance(poly, Expression):
            self.spec.slice = poly._open()
            self.keep.append(poly)
            return
        if not isinstance(poly, dict):
            raise TypeError("a polynomial is a first-slice Expression or a dict {exponent tuple: coefficient}")
        items = list(poly.items())
        exps = np.zeros(max(len(items) * n, 1), np.uint8)
        for t, (e, _) in enumerate(items):
            e = tuple(e)
            if len(e) != n or any(not isinstance(x, int) or x < 0 or x > 255 for x in e):
                raise ValueError(f"an exponent tuple has {n} small nonnegative ints")
            exps[t * n:(t + 1) * n] = e
        coef = _Values([c for _, c in items], "coefficient")
        self.keep += [exps, coef]
        self.spec.slice = None
        self.spec.nterms = len(items)
        self.spec.exps = exps.ctypes.data
        self.spec.coef = C.cast(coef.array, C.c_void_p).value


def _names_of(objective, box, variables):
    from .mathematics import Expression
    if variables is not None:
        return list(variables)
    if isinstance(box, dict):
        return list(box)
    if isinstance(objective, Expression):
        return [n for n in objective.context.names if n in objective.support]
    if isinstance(objective, dict) and objective:
        return [f"x{i}" for i in range(len(next(iter(objective))))]
    raise ValueError("name the variables")


class Problem:
    """A native polynomial problem: minimize objective over a box subject to (poly, op, rhs) constraints, op "<=",
    ">=" or "=" (n <= 4 variables, each polynomial's dense size prod (deg_v + 1) <= 256, at most 64 constraints)."""

    def __init__(self, objective, box, constraints=(), *, variables=None, budget=None):
        from .mathematics import Expression
        self.names = tuple(_names_of(objective, box, variables))
        n = len(self.names)
        if isinstance(box, dict):
            unknown = set(box) - set(self.names)
            if unknown:
                raise ValueError(f"box bounds for unknown variables {sorted(unknown)}")
            pairs = [box.get(name) for name in self.names]
        else:
            pairs = list(box)
        if len(pairs) != n or any(p is None or len(p) != 2 for p in pairs):
            raise ValueError("a finite (lo, hi) pair for every variable")
        self._lo = _Values([p[0] for p in pairs], "lower bound")
        self._hi = _Values([p[1] for p in pairs], "upper bound")
        contexts = [p.context for p in [objective] + [c[0] for c in constraints] if isinstance(p, Expression)]
        slots = (C.c_int32 * max(n, 1))()
        if contexts:
            ctx = contexts[0]
            if any(c is not ctx for c in contexts):
                from .math_domains import DomainError
                raise DomainError("the polynomials belong to different variable contexts")
            for i, name in enumerate(self.names):
                slots[i] = ctx._index(name)
        self._obj = _PolySpec(objective, n)
        specs = []
        for item in constraints:
            if len(item) != 3 or item[1] not in KINDS:
                raise ValueError("a constraint is (polynomial, '<=' | '>=' | '=', rhs)")
            specs.append(_PolySpec(item[0], n, KINDS[item[1]], item[2]))
        self._specs = specs
        arr = (_Spec * max(len(specs), 1))(*[s.spec for s in specs])
        self.ncons = len(specs)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["poly_create"](n, slots if contexts else None, C.byref(self._obj.spec), len(specs), arr,
                                  self._lo.array, self._hi.array, _raw(budget), C.byref(refusal), C.byref(error))
        self._h = _made(h, error, refusal, "polynomial problem")
        self.n = n

    def view(self):
        v = _View()
        _check(_api()["poly_get_view"](self._h, C.byref(v)), what="enclosure view")
        return v

    def value(self, x, which=0, budget=None):
        """the exact value of the objective (which=0) or of constraint which-1 (left side minus rhs) at x"""
        xv = _Values(list(x), "point")
        if xv.n != self.n:
            raise ValueError(f"a point has {self.n} coordinates")
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["poly_value"](self._h, int(which), xv.array, _raw(budget), C.byref(refusal), C.byref(error))
        h = _made(h, error, refusal, "value")
        try:
            return _fraction(h)
        finally:
            _mc._api()["bigq_free"](h)

    def minimize(self, tolerance=Fraction(1, 2 ** 20), *, backend="cpu", batch=32, time_limit=None, budget=None):
        fn, label, device = _fn(backend)
        tq, keep = _one(tolerance, "tolerance")
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["minimize"](self._h, tq, int(batch), fn, float(time_limit or 0) * 1000.0, _raw(budget),
                               C.byref(refusal), C.byref(error))
        del keep
        return Result(_made(h, error, refusal, "branch and bound"), self.names, label, device)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _native is not None:
            _native["poly_free"](h)
            self._h = None


def polynomial(objective, box, constraints=(), *, variables=None, tolerance=Fraction(1, 2 ** 20), backend="cpu",
               batch=32, time_limit=None, budget=None):
    """The certified global minimum of a small polynomial problem over a box (see Problem): Result with status
    "complete" / "incomplete" / "infeasible", [lower, upper], a feasible witness and the search statistics. backend
    "cuda" evaluates the box enclosures on the device (the same answers)."""
    pr = Problem(objective, box, constraints, variables=variables, budget=budget)
    return pr.minimize(tolerance, backend=backend, batch=batch, time_limit=time_limit, budget=budget)


def boxes(problem, boxes, ubound=None, backend="cpu", budget=None):
    """The enclosure of each box [(lo, hi) per variable, exact values: enclosed outward natively] as computed by the
    native CPU evaluator or the CUDA kernel: dicts status ("keep" / "infeasible" / "bound"), objective (flo, fhi),
    violated (internal constraint index or None), inner."""
    _backend(backend)
    n = problem.n
    boxes = [list(b) for b in boxes]
    xlo, xhi = np.zeros(max(len(boxes) * n, 1)), np.zeros(max(len(boxes) * n, 1))
    for i, b in enumerate(boxes):
        if len(b) != n:
            raise ValueError(f"a box has {n} (lo, hi) pairs")
        for v, (l, h) in enumerate(b):
            xlo[i * n + v] = _mc.BigRational(_exact(l, "box bound")).enclose().lo
            xhi[i * n + v] = _mc.BigRational(_exact(h, "box bound")).enclose().hi
    ub = float("inf") if ubound is None else _mc.BigRational(_exact(ubound, "bound")).enclose().hi
    out = (_BoxOut * max(len(boxes), 1))()
    view = problem.view()
    refusal = _Refusal()
    if backend == "cpu":
        _check(_api()["boxes"](C.byref(view), len(boxes), xlo.ctypes.data, xhi.ctypes.data, ub, out, _raw(budget),
                               C.byref(refusal)), refusal, "box enclosures")
    else:
        gpu = _gpu()
        gpu.run("tr_mgpu_polyopt_boxes", C.byref(view), len(boxes), xlo.ctypes.data, xhi.ctypes.data, ub, out,
                _raw(budget), C.byref(refusal), refusal=refusal, what="box enclosures on the device")
    return [dict(status=BOX_STATES[o.status], objective=(o.flo, o.fhi), violated=None if o.violated < 0 else o.violated,
                 inner=bool(o.inner)) for o in out[:len(boxes)]]
