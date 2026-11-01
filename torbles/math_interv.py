# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""INV-002: constrained minimal interventions, natively on the CPU (torbles/math_interv.c, with OPT-002's exact LPs
and QPs and INV-001's roots) and, for finite grids, with the CUDA kernels (math_interv_gpu.cu).

    from fractions import Fraction
    from torbles import math_interv as iv
    r = iv.linear(2*t + u, {"t": 1, "u": 0}, ">=", 7, cost="l1", d_bounds={"u": (0, None)})   # an exact LP
    r = iv.linear(([2, 1], 0), [1, 0], "=", 7, cost="l2", A_ub=[[1, 1]], b_ub=[5])            # the exact projection
    r = iv.linear(f, x0, ">=", 0, flip=True, cost="l0")      # a membership change: every minimal support (ties)
    r = iv.univariate(t**2, {"t": 1}, "=", 2, cost="l1")      # nearest roots by INV-001: exact or isolated
    g = iv.Grid(t*t + u, {"t": 0, "u": 0}, {"t": range(-4, 5), "u": range(-4, 5)}, "=", 8, cost="l1")
    r = g.search(backend="cuda")                              # every tie of the minimal cost, or proven bounds
    st = iv.Aggregate(scope, "x", "v")                        # a query whose rows cross its predicate when x shifts
    r = st.minimal("sum", (">=", 10), (">=", 500), cost="l1") # SUM(v) WHERE x + d >= 10 attains >= 500
    r.status, r.cost, r.solutions[0].change, r.unique, r.claim, r.causal

Every answer is an algebraic intervention on the function as given (Result.claim == "algebraic", Result.causal is
False): the cheapest change of the inputs that makes the function attain the target, not a statement about what would
happen if the world's inputs were changed; semantics="causal" is refused (UnsupportedError). Exact values are
Fraction; a change at an irrational root is an open isolating interval (lo, hi). Python marshals values and maps
native codes to exceptions; it computes nothing. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
import os
from fractions import Fraction

import numpy as np

from . import algq
from . import math_carriers as _mc
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check
from .math_inv import _Values, _one, _fraction, _exact, _made
from .math_polyopt import _Spec, _PolySpec

__all__ = ["linear", "univariate", "Grid", "grid", "Aggregate", "Result", "Solution", "CLAIM", "COSTS", "OPS",
           "AGGREGATES", "MAX_DIMS", "GRID_DIMS"]

MAX_DIMS, GRID_DIMS, CONDS = 16, 8, 16
CLAIM = ("an algebraic intervention on the function as given: the cheapest change of the declared inputs that makes "
         "the function attain the target; no causal model is involved and no claim is made about what would happen "
         "if the inputs were changed in the world")
COSTS = {"l1": 1, "linf": 2, "l2": 3, "l0": 4}
OPS = {"=": 0, "==": 0, "!=": 1, "<>": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}
_OPNAME = {0: "=", 1: "!=", 2: "<", 3: "<=", 4: ">", 5: ">="}
SEMANTICS = {"algebraic": 1, "causal": 2}
AGGREGATES = {"count": 1, "count_value": 2, "sum": 3, "avg": 4}
STATUSES = {1: "optimal", 2: "unattained", 3: "infeasible", 4: "incomplete"}
METHODS = {1: "lp", 2: "qp", 3: "supports", 4: "roots", 5: "grid", 6: "aggregate"}
REASONS = {0: None, 1: "iterations", 2: "output"}
HOW = {0: None, 1: "evaluated", 2: "linear factor", 3: "rational candidate", 4: "irrational"}
PART = dict(lower=0, upper=1, base=2, face=3, face_cost=4)
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
P = C.POINTER
vp = C.c_void_p


class _Linear(C.Structure):
    _native_ = "tr_interv_linear_problem"
    _fields_ = [(n, C.c_int32) for n in ("n", "cost", "op", "flip", "m", "p", "semantics", "reserved")] + [
        (n, vp) for n in ("x0", "weight", "a", "a0", "t", "A", "b", "E", "e", "xlo", "xhi", "dlo", "dhi", "l1", "linf")]


class _Univariate(C.Structure):
    _native_ = "tr_interv_univariate_problem"
    _fields_ = [("slice", vp), ("variable", C.c_int32), ("ncoef", C.c_int32), ("coef", vp), ("x0", vp), ("t", vp),
                ("weight", vp)] + [(n, C.c_int32) for n in ("cost", "op", "flip", "semantics")] + [
        (n, vp) for n in ("dlo", "dhi", "xlo", "xhi", "width")]


class _GridProblem(C.Structure):
    _native_ = "tr_interv_grid_problem"
    _fields_ = [(n, C.c_int32) for n in ("ndims", "cost", "op", "flip", "ncons", "semantics", "nrows", "reserved")] + [
        (n, vp) for n in ("vars", "sizes", "values", "x0", "weight")] + [("target", _Spec), ("t", vp), ("cons", vp),
                                                                        ("A", vp), ("b", vp)]


class _GridView(C.Structure):
    _native_ = "tr_interv_grid_view"
    _fields_ = [("total", C.c_uint64), ("nvalues", C.c_uint64), ("size", C.c_uint64 * GRID_DIMS),
                ("voff", C.c_uint64 * GRID_DIMS)] + [(n, C.c_int32) for n in ("ndims", "nconds", "nprimes", "combine")] + [
        ("nterms", C.c_uint32), ("reserved", C.c_uint32), ("cond_off", C.c_uint32 * (CONDS + 1)),
        ("allowed", C.c_uint8 * CONDS)] + [(n, vp) for n in ("fields", "garner", "half", "exps", "cres", "vres", "ctab",
                                                             "order")]


class _GridInfo(C.Structure):
    _native_ = "tr_interv_grid_info"
    _fields_ = [(n, C.c_uint64) for n in ("grid", "total", "pruned", "blocks", "blocks_pruned", "bound_bits")] + [
        (n, C.c_int32) for n in ("ndims", "nconds", "nprimes", "base_member", "op", "reserved")]


class _Info(C.Structure):
    _native_ = "tr_interv_info"
    _fields_ = [(n, C.c_int32) for n in ("method", "status", "cost", "op", "n", "nsolutions", "unique", "base_member",
                                         "claim", "complete", "reason", "exact", "nprimes", "reserved")] + [
        (n, C.c_uint64) for n in ("examined", "pruned", "total", "next", "iterations", "bound_bits", "crossed")]


class _Solution(C.Structure):
    _native_ = "tr_interv_solution"
    _fields_ = [(n, C.c_int32) for n in ("exact", "attained", "side", "how", "multiplicity", "null")] + [
        (n, C.c_uint64) for n in ("index", "rows", "entered", "left")]


class _AggInfo(C.Structure):
    _native_ = "tr_interv_agg_info"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "present", "distinct", "vpresent", "bytes")] + [
        (n, C.c_int32) for n in ("xscale", "vscale", "has_value", "shifted")]


class _AggQuery(C.Structure):
    _native_ = "tr_interv_agg_query"
    _fields_ = [(n, C.c_int32) for n in ("aggregate", "pred_op", "op", "flip", "cost", "semantics", "r0", "r1")] + [
        (n, vp) for n in ("c", "t", "weight", "dlo", "dhi")]


for _s, _n in ((_Linear, 152), (_Univariate, 104), (_GridProblem, 144), (_GridView, 320), (_GridInfo, 72),
               (_Info, 112), (_Solution, 56), (_AggInfo, 56), (_AggQuery, 72)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        i32, u64, B, R, E = C.c_int32, C.c_uint64, P(_Budget), P(_Refusal), P(C.c_int32)
        sig = {
            "tr_interv_linear": (vp, [P(_Linear), B, R, E]),
            "tr_interv_linear_poly": (vp, [P(_Linear), vp, vp, B, R, E]),
            "tr_interv_univariate": (vp, [P(_Univariate), B, R, E]),
            "tr_interv_grid_prepare": (vp, [P(_GridProblem), B, R, E]),
            "tr_interv_grid_get_view": (i32, [vp, P(_GridView)]),
            "tr_interv_grid_get_info": (i32, [vp, P(_GridInfo)]),
            "tr_interv_grid_eval": (C.c_int, [P(_GridView), u64, u64, C.c_int64, vp, vp, P(C.c_int64), B, R]),
            "tr_interv_grid_search": (vp, [vp, u64, vp, B, R, E]),
            "tr_interv_grid_coords": (i32, [vp, u64, vp]),
            "tr_interv_grid_plan_free": (None, [vp]),
            "tr_interv_grid_device_need": (u64, [P(_GridView), u64]),
            "tr_interv_agg_build": (vp, [vp, i32, i32, B, R, E]),
            "tr_interv_agg_get_info": (i32, [vp, P(_AggInfo)]),
            "tr_interv_agg_solve": (vp, [vp, P(_AggQuery), B, R, E]),
            "tr_interv_agg_value": (i32, [vp, i32, i32, vp, vp, P(vp), P(i32), P(u64), B, R]),
            "tr_interv_agg_free": (None, [vp]),
            "tr_interv_result_get_info": (i32, [vp, P(_Info)]),
            "tr_interv_result_solution": (i32, [vp, i32, P(_Solution), vp, vp, vp, vp, P(vp), P(vp), P(vp), P(vp)]),
            "tr_interv_result_support": (u64, [vp, i32]),
            "tr_interv_result_part": (vp, [vp, i32, P(u64)]),
            "tr_interv_result_free": (None, [vp]),
        }
        native = {}
        for name, (restype, argtypes) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            native[name[10:]] = fn
        _native = native
    return _native


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_interv_grid", C.c_int, [P(_GridView), C.c_uint64, C.c_uint64, C.c_int64, vp, vp,
                                                       P(C.c_int64), P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _raw(budget):
    return C.byref((budget or Budget()).raw)


def _code(table, value, what):
    if value not in table:
        raise ValueError(f"{what} is one of {sorted(table)}")
    return table[value]


def _bigq_free(h):
    if h:
        _mc._api()["bigq_free"](h)


# ---------------------------------------------------------------------------------------------------- results
class Solution:
    """One minimal change: change {name: Fraction or (lo, hi)} (the open isolating interval of an irrational change),
    point = x0 + change, cost (Fraction or (lo, hi)), value (the function at the point; None for SQL NULL), exact,
    attained (False: the infimum of a strict target, not attained), side (-1 / +1: the direction of a one-input
    change), and per method: how / multiplicity (a root), index (a grid point), support (L0: the changed inputs),
    rows / entered / left (an aggregate: rows in the predicate at the change, rows that entered and left it)."""
    __slots__ = ("change", "point", "cost", "value", "exact", "attained", "side", "how", "multiplicity", "index",
                 "support", "rows", "entered", "left")

    def __repr__(self):
        a = "" if self.attained else ", infimum (not attained)"
        return f"Solution(change={self.change}, cost={self.cost}{a})"

    def key(self):
        return tuple(getattr(self, k) if not isinstance(getattr(self, k), dict) else tuple(getattr(self, k).items())
                     for k in self.__slots__)


class Result:
    """A minimal intervention (native): status "optimal" (every minimal change listed; unique: True / False / None
    undecided), "unattained" (a strict target: the exact infimum and its boundary change), "infeasible" (no change in
    the declared domain attains the target) or "incomplete" (a budget stopped the search: lower <= minimum <= upper,
    proven; upper None when nothing feasible was found yet; resume a grid at next). claim == "algebraic", causal is
    False: see CLAIM."""
    claim, causal, statement = "algebraic", False, CLAIM

    def __init__(self, handle, names, backend="cpu", device="native CPU"):
        api = _api()
        try:
            i = _Info()
            _check(api["result_get_info"](handle, C.byref(i)), what="intervention")
            if i.claim != 1:
                raise RuntimeError("the native result is not labelled as an algebraic intervention")
            self.names, self.backend, self.device = tuple(names), backend, device
            self.status, self.method = STATUSES.get(i.status), METHODS.get(i.method)
            self.cost_kind = {v: k for k, v in COSTS.items()}[i.cost]
            self.op, self.base_member = _OPNAME[i.op], bool(i.base_member)
            self.unique = {1: True, 0: False}.get(int(i.unique))
            self.complete, self.reason, self.exact = bool(i.complete), REASONS[i.reason], bool(i.exact)
            self.examined, self.pruned, self.total, self.next = int(i.examined), int(i.pruned), int(i.total), int(i.next)
            self.iterations, self.bound_bits, self.primes, self.crossed = int(i.iterations), int(i.bound_bits), \
                int(i.nprimes), int(i.crossed)
            part = lambda name: self._part(handle, name)
            one = lambda name: (part(name) or [None])[0]
            self.lower, self.upper, self.base_value = one("lower"), one("upper"), one("base")
            face = part("face")
            self.face = dict(zip(self.names, face)) if face else None
            self.face_cost = one("face_cost")
            n = len(self.names)
            self.solutions = []
            arrs = [(vp * max(n, 1))() for _ in range(4)]
            for k in range(int(i.nsolutions)):
                s, cl, ch, vl, vh = _Solution(), vp(), vp(), vp(), vp()
                _check(api["result_solution"](handle, k, C.byref(s), *arrs, C.byref(cl), C.byref(ch), C.byref(vl),
                                              C.byref(vh)), what="solution")
                sol = Solution()
                both = lambda lo, hi: (lambda a, b: a if a == b else (a, b))(_fraction(lo), _fraction(hi))
                sol.change = {self.names[j]: both(arrs[0][j], arrs[1][j]) for j in range(n)}
                sol.point = {self.names[j]: both(arrs[2][j], arrs[3][j]) for j in range(n)}
                sol.cost = both(cl.value, ch.value)
                sol.value = both(vl.value, vh.value) if vl.value else None
                sol.exact, sol.attained, sol.side = bool(s.exact), bool(s.attained), int(s.side)
                sol.how, sol.multiplicity = HOW.get(int(s.how)), int(s.multiplicity)
                sol.index, sol.rows, sol.entered, sol.left = int(s.index), int(s.rows), int(s.entered), int(s.left)
                mask = int(api["result_support"](handle, k))
                sol.support = tuple(self.names[j] for j in range(n) if (mask >> j) & 1)
                self.solutions.append(sol)
            self.cost = self.solutions[0].cost if self.solutions else None
        finally:
            api["result_free"](handle)

    @staticmethod
    def _part(handle, name):
        count = C.c_uint64()
        arr = _api()["result_part"](handle, PART[name], C.byref(count))
        if not arr:
            return None
        ptrs = C.cast(arr, P(vp))
        return [_fraction(ptrs[k]) for k in range(int(count.value))]

    def same(self, other):
        keys = ("status", "method", "op", "base_member", "unique", "complete", "reason", "exact", "examined", "pruned",
                "total", "next", "lower", "upper", "base_value", "face", "bound_bits", "primes")
        return all(getattr(self, k) == getattr(other, k) for k in keys) and \
            [s.key() for s in self.solutions] == [s.key() for s in other.solutions]

    def __repr__(self):
        return (f"Result({self.status}, cost={self.cost}, solutions={len(self.solutions)}, unique={self.unique}, "
                f"claim='algebraic')")


# ---------------------------------------------------------------------------------------------------- linear targets
def _names_and_values(x0, names):
    if isinstance(x0, dict):
        names = list(x0) if names is None else list(names)
        if set(names) != set(x0):
            raise ValueError("x0 gives a value for every input")
        return names, [x0[n] for n in names]
    x0 = list(x0)
    names = [f"x{i}" for i in range(len(x0))] if names is None else list(names)
    if len(names) != len(x0):
        raise ValueError("one name per input")
    return names, x0


def _per_input(spec, names, what, pair=False):
    """a per-input list from None, a list, or {name: value}"""
    if spec is None:
        return [None] * len(names)
    if isinstance(spec, dict):
        unknown = set(spec) - set(names)
        if unknown:
            raise ValueError(f"{what} for unknown inputs {sorted(unknown)}")
        return [spec.get(n) for n in names]
    spec = list(spec)
    if len(spec) != len(names):
        raise ValueError(f"{what}: one entry per input ({len(names)})")
    return spec


def _bounds(spec, names, what):
    pairs = _per_input(spec, names, what)
    lo, hi = [], []
    for p in pairs:
        l, h = (None, None) if p is None else p
        lo.append(l); hi.append(h)
    return _Values(lo, f"{what} lower bound", True), _Values(hi, f"{what} upper bound", True)


def linear(target, x0, op, value=0, *, cost="l1", weights=None, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
           x_bounds=None, d_bounds=None, l1=None, linf=None, flip=False, names=None, semantics="algebraic", budget=None):
    """The minimal change d of x0 with target(x0 + d) op value: target a first-slice Expression of degree <= 1 (its
    variables named by x0's keys) or (a, a0) coefficients; constraints A_ub x <= b_ub, A_eq x = b_eq on the point,
    boxes x_bounds / d_bounds ({name: (lo, hi)} or lists, None for no bound), ||d||_1 <= l1, ||d||_inf <= linf.
    cost "l1", "linf" (weighted), "l2" (weighted squared), "l0" (weighted count, weights > 0). flip: the target is the
    complement of the predicate at x0 (a membership change)."""
    from .mathematics import Expression
    names, xs = _names_and_values(x0, names)
    n = len(names)
    keep = []
    pr = _Linear()
    pr.n, pr.cost, pr.op, pr.flip = n, _code(COSTS, cost, "cost"), _code(OPS, op, "op"), int(bool(flip))
    pr.semantics = _code(SEMANTICS, semantics, "semantics")
    xv = _Values(xs, "x0")
    wv = _Values(_per_input(weights, names, "weights"), "weight", True)
    keep += [xv, wv]
    pr.x0, pr.weight = C.cast(xv.array, vp).value, C.cast(wv.array, vp).value
    A = [list(r) for r in (A_ub or [])]
    E = [list(r) for r in (A_eq or [])]
    b, e = list(b_ub or []), list(b_eq or [])
    if any(len(r) != n for r in A + E) or len(A) != len(b) or len(E) != len(e):
        raise ValueError("each constraint row has one coefficient per input and a right-hand side")
    av, bv = _Values([v for r in A for v in r], "A_ub"), _Values(b, "b_ub")
    ev, fv = _Values([v for r in E for v in r], "A_eq"), _Values(e, "b_eq")
    keep += [av, bv, ev, fv]
    pr.m, pr.p = len(A), len(E)
    pr.A, pr.b, pr.E, pr.e = (C.cast(x.array, vp).value for x in (av, bv, ev, fv))
    xl, xh = _bounds(x_bounds, names, "x_bounds")
    dl, dh = _bounds(d_bounds, names, "d_bounds")
    keep += [xl, xh, dl, dh]
    pr.xlo, pr.xhi, pr.dlo, pr.dhi = (C.cast(x.array, vp).value for x in (xl, xh, dl, dh))
    tq, k1 = _one(value, "value")
    l1q, k2 = _one(l1, "l1")
    lfq, k3 = _one(linf, "linf")
    pr.t, pr.l1, pr.linf = tq, l1q, lfq
    error, refusal = C.c_int32(), _Refusal()
    if isinstance(target, Expression):
        slots = (C.c_int32 * n)(*[target.context._index(nm) for nm in names])
        h = _api()["linear_poly"](C.byref(pr), target._open(), slots, _raw(budget), C.byref(refusal), C.byref(error))
    else:
        a, a0 = target
        av2 = _Values(list(a), "coefficient", True)
        if av2.n != n:
            raise ValueError(f"the target has {n} coefficients")
        a0q, k4 = _one(a0, "a0")
        keep += [av2, k4]
        pr.a, pr.a0 = C.cast(av2.array, vp).value, a0q
        h = _api()["linear"](C.byref(pr), _raw(budget), C.byref(refusal), C.byref(error))
    out = Result(_made(h, error, refusal, "minimal intervention"), names)
    del keep, k1, k2, k3
    return out


# ---------------------------------------------------------------------------------------------------- univariate
def univariate(f, x0, op, value=0, *, cost="l1", weight=1, d_bounds=None, x_bounds=None, width=None, flip=False,
               variable=None, semantics="algebraic", budget=None):
    """The minimal change of one input with f(x0 + d) op value: f a first-slice Expression in one variable or its
    coefficients (low degree first); x0 a value or {name: value}; d_bounds / x_bounds (lo, hi) with None for no
    bound. The roots of f - value come from INV-001: a change at an irrational root is an isolating interval (refined
    to width when given), and ties between the two sides are decided exactly."""
    from .mathematics import Expression
    if isinstance(x0, dict):
        if len(x0) != 1:
            raise ValueError("one input")
        (name, xval), = x0.items()
    else:
        name, xval = (variable if isinstance(variable, str) else "x"), x0
    pr = _Univariate()
    keep = []
    if isinstance(f, Expression):
        nm = name if isinstance(x0, dict) else (variable or (f.support[0] if f.support else f.context.names[0]))
        pr.slice, pr.variable = f._open(), f.context._index(nm)
        keep.append(f)
        name = nm
    else:
        cv = _Values(list(f), "coefficient", True)
        if cv.n < 1:
            raise ValueError("at least one coefficient")
        pr.slice, pr.ncoef, pr.coef = None, cv.n, C.cast(cv.array, vp).value
        keep.append(cv)
    handles = []
    for field, v, what in (("x0", xval, "x0"), ("t", value, "value"), ("weight", weight, "weight"),
                           ("width", width, "width")):
        q, k = _one(v, what)
        setattr(pr, field, q)
        handles.append(k)
    for (lo_f, hi_f), b, what in ((("dlo", "dhi"), d_bounds, "d_bounds"), (("xlo", "xhi"), x_bounds, "x_bounds")):
        lo, hi = (None, None) if b is None else b
        for field, v in ((lo_f, lo), (hi_f, hi)):
            q, k = _one(v, what)
            setattr(pr, field, q)
            handles.append(k)
    pr.cost, pr.op, pr.flip = _code(COSTS, cost, "cost"), _code(OPS, op, "op"), int(bool(flip))
    pr.semantics = _code(SEMANTICS, semantics, "semantics")
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["univariate"](C.byref(pr), _raw(budget), C.byref(refusal), C.byref(error))
    out = Result(_made(h, error, refusal, "minimal intervention"), [name])
    del keep, handles
    return out


# ---------------------------------------------------------------------------------------------------- finite grids
class Grid:
    """The plan of a finite-grid search (native): the conditions (f op value, polynomial constraints (g, op, rhs) and
    linear rows A_ub x <= b_ub on the point) as exact integer forms with their multimodular proof, the cost tables,
    the blocks pruned by interval enclosures and the cost order of the remaining points."""

    def __init__(self, f, x0, grid, op, value=0, *, cost="l1", weights=None, constraints=(), A_ub=None, b_ub=None,
                 flip=False, names=None, semantics="algebraic", budget=None):
        from .mathematics import Expression
        items = list(grid.items()) if isinstance(grid, dict) else [(nm, v) for nm, v in grid]
        self.names = [nm for nm, _ in items]
        nd = len(self.names)
        self.values = [[_exact(v, f"value of {nm!r}") for v in vals] for nm, vals in items]
        if isinstance(x0, dict):
            if set(x0) != set(self.names):
                raise ValueError("x0 gives a value for every grid coordinate")
            xs = [x0[nm] for nm in self.names]
        else:
            xs = list(x0)
        self._keep = []
        pr = _GridProblem()
        pr.ndims, pr.cost, pr.op, pr.flip = nd, _code(COSTS, cost, "cost"), _code(OPS, op, "op"), int(bool(flip))
        pr.semantics = _code(SEMANTICS, semantics, "semantics")
        contexts = [p.context for p in [f] + [c[0] for c in constraints] if isinstance(p, Expression)]
        slots = (C.c_int32 * max(nd, 1))()
        if contexts:
            for i, nm in enumerate(self.names):
                slots[i] = contexts[0]._index(nm)
        sizes = (C.c_uint64 * max(nd, 1))(*[len(v) for v in self.values])
        vals = _Values([v for vv in self.values for v in vv], "grid value")
        xv = _Values(xs, "x0")
        wv = _Values(_per_input(weights, self.names, "weights"), "weight", True)
        tspec = _PolySpec(f, nd)
        cspecs = []
        for item in constraints:
            if len(item) != 3 or item[1] not in ("<=", ">=", "=", "=="):
                raise ValueError("a constraint is (polynomial, '<=' | '>=' | '=', rhs)")
            cspecs.append(_PolySpec(item[0], nd, {"<=": 0, ">=": 1, "=": 2, "==": 2}[item[1]], item[2]))
        carr = (_Spec * max(len(cspecs), 1))(*[s.spec for s in cspecs])
        A = [list(r) for r in (A_ub or [])]
        b = list(b_ub or [])
        if any(len(r) != nd for r in A) or len(A) != len(b):
            raise ValueError("each row of A_ub has one coefficient per coordinate and a right-hand side")
        av, bv = _Values([v for r in A for v in r], "A_ub"), _Values(b, "b_ub")
        tq, kt = _one(value, "value")
        self._keep += [slots, sizes, vals, xv, wv, tspec, cspecs, carr, av, bv, kt]
        pr.vars = C.cast(slots, vp).value if contexts else None
        pr.sizes, pr.values, pr.x0, pr.weight = C.cast(sizes, vp).value, C.cast(vals.array, vp).value, \
            C.cast(xv.array, vp).value, C.cast(wv.array, vp).value
        pr.target, pr.t, pr.ncons, pr.cons = tspec.spec, tq, len(cspecs), C.cast(carr, vp).value
        pr.nrows, pr.A, pr.b = len(A), C.cast(av.array, vp).value, C.cast(bv.array, vp).value
        error, refusal = C.c_int32(), _Refusal()
        self._h = _made(_api()["grid_prepare"](C.byref(pr), _raw(budget), C.byref(refusal), C.byref(error)), error,
                        refusal, "grid plan")
        i = _GridInfo()
        _check(_api()["grid_get_info"](self._h, C.byref(i)), what="grid plan")
        self.info = dict(grid=int(i.grid), total=int(i.total), pruned=int(i.pruned), blocks=int(i.blocks),
                         blocks_pruned=int(i.blocks_pruned), bound_bits=int(i.bound_bits), primes=int(i.nprimes),
                         conditions=int(i.nconds), base_member=bool(i.base_member), op=_OPNAME[i.op])

    def view(self):
        v = _GridView()
        _check(_api()["grid_get_view"](self._h, C.byref(v)), what="grid view")
        return v

    def order(self):
        """the grid indices of the points kept after the enclosure pruning, in increasing (cost, index) order"""
        v = self.view()
        if not v.total:
            return np.zeros(0, np.uint64)
        return np.ctypeslib.as_array(C.cast(v.order, P(C.c_uint64)), shape=(int(v.total),)).copy()

    def coords(self, index):
        out = (C.c_uint64 * GRID_DIMS)()
        _check(_api()["grid_coords"](self._h, int(index), out), what="grid coordinates")
        return {nm: self.values[d][int(out[d])] for d, nm in enumerate(self.names)}

    def evaluate(self, start=0, count=None, match=None, backend="cpu", budget=None):
        """the raw batched evaluation of ranks start .. start + count - 1 (CPU evaluator or CUDA kernels): the
        feasibility and tie bitmaps (uint64 words) and the minimal feasible integer cost (None: none)"""
        v = self.view()
        count = int(v.total) - int(start) if count is None else int(count)
        words = max((count + 63) // 64, 1)
        fe, ti, mc = np.zeros(words, np.uint64), np.zeros(words, np.uint64), C.c_int64()
        m = INT64_MIN if match is None else int(match)
        refusal = _Refusal()
        if backend == "cpu":
            _check(_api()["grid_eval"](C.byref(v), int(start), count, m, fe.ctypes.data, ti.ctypes.data, C.byref(mc),
                                       _raw(budget), C.byref(refusal)), refusal, "grid evaluation")
        elif backend == "cuda":
            _gpu().run("tr_mgpu_interv_grid", C.byref(v), int(start), count, m, fe.ctypes.data, ti.ctypes.data,
                       C.byref(mc), _raw(budget), C.byref(refusal), refusal=refusal, what="grid evaluation on the device")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return fe, ti, (None if mc.value == INT64_MAX else int(mc.value))

    def device_bytes(self, count=None):
        v = self.view()
        return int(_api()["grid_device_need"](C.byref(v), int(v.total) if count is None else int(count)))

    def search(self, start=0, backend="cpu", budget=None):
        """The exact search in cost order from rank start (backend "cuda": the device evaluates each chunk)."""
        fn, label, device = None, "cpu", "native CPU"
        if backend == "cuda":
            gpu = _gpu()
            if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
                gpu.require_device()
            fn = C.cast(gpu.lib().tr_mgpu_interv_grid, vp).value
            info = gpu.device()
            label, device = ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]
        elif backend != "cpu":
            raise ValueError("backend is 'cpu' or 'cuda'")
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["grid_search"](self._h, int(start), fn, _raw(budget), C.byref(refusal), C.byref(error))
        return Result(_made(h, error, refusal, "grid search"), self.names, label, device)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _native is not None:
            _native["grid_plan_free"](h)
            self._h = None


def grid(f, x0, grid, op, value=0, *, start=0, backend="cpu", budget=None, **kwargs):
    """Grid(...).search(start, backend): the exhaustive minimal change over a finite grid (see Grid)."""
    return Grid(f, x0, grid, op, value, budget=budget, **kwargs).search(start, backend, budget)


# ---------------------------------------------------------------------------------------------------- aggregates
class Aggregate:
    """The state of a query whose rows move across its predicate when the feature x is shifted by d (native, built
    once from an FND-004 scope): the sorted distinct values of x over the selected rows with their row counts and
    exact sums (of the value column, and of x itself), as prefix sums. value None: COUNT(*) only; value == feature:
    the aggregate sums the shifted feature."""

    def __init__(self, scope, feature, value=None, budget=None):
        self.scope, self.feature, self.value_column = scope, feature, value
        names = list(scope.names)
        f = names.index(feature)
        v = -1 if value is None else names.index(value)
        error, refusal = C.c_int32(), _Refusal()
        self._h = _made(_api()["agg_build"](scope._handle, f, v, _raw(budget), C.byref(refusal), C.byref(error)), error,
                        refusal, "aggregate state")
        i = _AggInfo()
        _check(_api()["agg_get_info"](self._h, C.byref(i)), what="aggregate state")
        self.info = dict(rows=int(i.rows), present=int(i.present), distinct=int(i.distinct), vpresent=int(i.vpresent),
                         bytes=int(i.bytes), xscale=int(i.xscale), vscale=int(i.vscale), shifted=bool(i.shifted))

    def _fresh(self):
        self.scope.require_fresh("intervention aggregate state")

    def value(self, aggregate, predicate, d, budget=None):
        """(the aggregate at the shift d exactly, None for SQL NULL; the rows in the predicate x + d op c)"""
        self._fresh()
        op, c = predicate
        cq, k1 = _one(c, "threshold")
        dq, k2 = _one(d, "shift")
        out, null, rows, refusal = vp(), C.c_int32(), C.c_uint64(), _Refusal()
        _check(_api()["agg_value"](self._h, _code(AGGREGATES, aggregate, "aggregate"), _code(OPS, op, "predicate op"),
                                   cq, dq, C.byref(out), C.byref(null), C.byref(rows), _raw(budget), C.byref(refusal)),
               refusal, "aggregate value")
        try:
            return (None if null.value else _fraction(out.value)), int(rows.value)
        finally:
            _bigq_free(out.value)
            del k1, k2

    def minimal(self, aggregate, predicate, target, *, cost="l1", weight=1, d_bounds=None, flip=False,
                semantics="algebraic", budget=None):
        """The minimal shift d of the feature such that aggregate(... WHERE x + d op c) attains the target (op, t)
        (or, flip, leaves its base membership): the piecewise aggregate decided exactly on every piece, rows entering
        and leaving the predicate included."""
        self._fresh()
        pop, c = predicate
        top, t = target
        q = _AggQuery()
        q.aggregate, q.pred_op, q.op, q.flip = _code(AGGREGATES, aggregate, "aggregate"), _code(OPS, pop, "predicate op"), \
            _code(OPS, top, "target op"), int(bool(flip))
        q.cost, q.semantics = _code(COSTS, cost, "cost"), _code(SEMANTICS, semantics, "semantics")
        lo, hi = (None, None) if d_bounds is None else d_bounds
        handles = []
        for field, v, what in (("c", c, "threshold"), ("t", t, "target"), ("weight", weight, "weight"),
                               ("dlo", lo, "d lower bound"), ("dhi", hi, "d upper bound")):
            qh, k = _one(v, what)
            setattr(q, field, qh)
            handles.append(k)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["agg_solve"](self._h, C.byref(q), _raw(budget), C.byref(refusal), C.byref(error))
        out = Result(_made(h, error, refusal, "minimal shift"), ["d"])
        del handles
        return out

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _native is not None:
            _native["agg_free"](h)
            self._h = None
