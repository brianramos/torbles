# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ENV-002: bounded multivariate parameter regions: for group functions affine in a parameter vector theta of
dimension d (1..4), the exact regions of a declared box or polytope where each group attains the minimum (or the
maximum), their vertices, facets and ties, and point location of many parameter values, natively on the CPU and with
the CUDA kernel (torbles/math_region.c, math_region_gpu.cu).

    from fractions import Fraction
    from torbles import math_region as mr
    R = mr.regions([(0, 1, 1), (1, -1, 0), None, (2, 0, -1)], box=[(-2, 2), (-2, 2)], rows=[((1, 1), 3)],
                   upper=False, tie="smallest")
    R.classes, R.regions, R.vertices, R.boundaries, R.info
    R.locate([(0, 0), (Fraction(1, 3), 1)], backend="cuda", mode="index" | "regions" | "all", ties=8)
    R.locate_arrays(mr.Values(numerators, 100))          # many values as int64 arrays, the answers as int64 arrays
    R.evaluate_all(points)                               # every class at every value (the verification path)
    mr.regions(quadratics, box, contract=mr.PiecewiseLinear(pieces=4, error_budget=Fraction(1, 10)))  # approximate
    mr.from_expressions([e0, e1], ["a", "b"], box)       # first-slice polynomials (RESP-001 functions)

functions[g]: None (an empty group), (a0, a1, .., ad) for a0 + sum a_k theta_k, or with the quadratic coefficients
h_kl of theta_k theta_l (k <= l) appended (refused with UnsupportedError unless a PiecewiseLinear contract is given;
then the result is labelled approximate with its certified error). Every number is exact (int, Fraction, Decimal;
floats are refused). Python marshals values and formats results; nothing is computed here. Opt-in: `import torbles`
does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_carriers as _mc
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check
from .math_inv import _Values, _exact, _fraction as _borrowed

__all__ = ["regions", "from_expressions", "Regions", "PiecewiseLinear", "Values", "need", "MAX_DIMS", "TIES", "MODES"]

NONE = (1 << 64) - 1
MAX_DIMS, MAX_ROWS, MAX_PIECES = 4, 64, 16
TIES = {"smallest": 0, "largest": 1}
MODES = {"index": 1, "regions": 2, "all": 3}
STATUS = {0: "ok", 1: "outside", 2: "empty"}
KINDS = {1: "interior", 2: "boundary"}
SOURCES = {1: "lower", 2: "upper", 3: "row", 4: "class"}
BOUNDARY = {1: "wall", 2: "cell", 3: "inner"}
CLAIMS = {1: "exact", 2: "approximate"}
P = C.POINTER


class _Problem(C.Structure):
    _native_ = "tr_region_problem"
    _fields_ = [(n, C.c_int32) for n in ("d", "ncoef", "nrows", "upper", "tie", "contract", "pieces", "reserved")] + [
        ("ngroups", C.c_uint64)] + [(n, C.c_void_p) for n in ("coef", "present", "lo", "hi", "A", "b", "error_budget")]


class _Info(C.Structure):
    _native_ = "tr_region_info"
    _fields_ = [(n, C.c_uint64) for n in (
        "groups", "present", "classes", "cells", "regions", "full", "lower", "vertices", "incidences", "boundaries",
        "nodes", "leaves", "candidates", "lps", "pivots", "clips", "iterations", "bytes")] + [
        (n, C.c_int32) for n in ("d", "kind", "upper", "tie", "claim", "contract", "pieces", "degree", "nrows",
                                 "unverified")]


class _ClassInfo(C.Structure):
    _native_ = "tr_region_class_info"
    _fields_ = [(n, C.c_uint64) for n in ("cell", "members", "winner", "region")] + [
        ("degree", C.c_int32), ("exact", C.c_int32), ("reserved", C.c_int32 * 2)]


class _RegionInfo(C.Structure):
    _native_ = "tr_region_region_info"
    _fields_ = [(n, C.c_uint64) for n in ("cls", "cell", "winner", "members", "vertices", "inequalities", "equalities",
                                          "lp_rows")] + [
        (n, C.c_int32) for n in ("dim", "full", "lp_sign", "reserved")]


class _VertexInfo(C.Structure):
    _native_ = "tr_region_vertex_info"
    _fields_ = [(n, C.c_uint64) for n in ("cell", "classes", "winner", "walls")]


class _BoundaryInfo(C.Structure):
    _native_ = "tr_region_boundary_info"
    _fields_ = [(n, C.c_uint64) for n in ("cell", "classes", "winner", "vertices", "region")] + [
        ("kind", C.c_int32), ("reserved", C.c_int32)]


class _Answer(C.Structure):
    _native_ = "tr_region_answer"
    _fields_ = [("status", C.c_int32), ("kind", C.c_int32)] + [
        (n, C.c_uint64) for n in ("cell", "winner", "cls", "region", "tied")]


class _View(C.Structure):
    _native_ = "tr_region_view"
    _fields_ = [(n, C.c_uint64) for n in ("npoints", "nclasses", "nregions", "nnodes", "ncells", "ncons", "ncand",
                                          "max_ties")] + [
        (n, C.c_int32) for n in ("d", "nprimes", "mode", "tie", "kind", "nwalls", "pieces", "limbs")] + [
        (n, C.c_void_p) for n in ("fields", "garner", "half", "wres", "gres", "cell_root", "cell_class", "cell_region",
                                  "naxis", "nlink", "nres", "cand", "cres", "cwinner", "cregion", "rclass", "rcons",
                                  "ceq", "kres", "qsign", "qnum", "qden")]


for _s, _n in ((_Problem, 96), (_Info, 184), (_ClassInfo, 48), (_RegionInfo, 80), (_VertexInfo, 32),
               (_BoundaryInfo, 48), (_Answer, 48), (_View, 272)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R, N = P(_Budget), P(_Refusal), P(_Need)
        sig = {
            "tr_region_need": (i32, [P(_Problem), N]),
            "tr_region_build": (vp, [P(_Problem), B, R, e]),
            "tr_region_from_polys": (vp, [P(_Problem), vp, vp, B, R, e]),
            "tr_region_get_info": (i32, [vp, P(_Info)]),
            "tr_region_bound": (vp, [vp, i32, i32]),
            "tr_region_error": (vp, [vp, i32]),
            "tr_region_class_get": (i32, [vp, u64, P(_ClassInfo)]),
            "tr_region_class_members": (i32, [vp, u64, vp, u64]),
            "tr_region_class_coef": (vp, [vp, u64, i32]),
            "tr_region_class_eps": (vp, [vp, u64]),
            "tr_region_region_get": (i32, [vp, u64, P(_RegionInfo)]),
            "tr_region_region_vertices": (i32, [vp, u64, vp, u64]),
            "tr_region_region_constraint": (i32, [vp, u64, u64, vp, P(vp), P(i32), P(u64)]),
            "tr_region_region_sources": (i32, [vp, u64, u64, vp, vp, u64]),
            "tr_region_region_lp": (i32, [vp, u64, vp, P(vp)]),
            "tr_region_vertex_get": (i32, [vp, u64, P(_VertexInfo)]),
            "tr_region_vertex_point": (i32, [vp, u64, vp]),
            "tr_region_vertex_classes": (i32, [vp, u64, vp, u64]),
            "tr_region_vertex_walls": (i32, [vp, u64, vp, u64]),
            "tr_region_boundary_get": (i32, [vp, u64, P(_BoundaryInfo)]),
            "tr_region_boundary_plane": (i32, [vp, u64, vp, P(vp)]),
            "tr_region_boundary_vertices": (i32, [vp, u64, vp, u64]),
            "tr_region_boundary_classes": (i32, [vp, u64, vp, u64]),
            "tr_region_authorize": (i32, [vp, vp]),
            "tr_region_free": (None, [vp]),
            "tr_region_lookup_need": (i32, [vp, u64, vp, i32, u64, i32, N]),
            "tr_region_lookup_create": (vp, [vp, u64, vp, i32, u64, B, R, e]),
            "tr_region_lookup_need_q64": (i32, [vp, u64, vp, vp, i32, u64, i32, N]),
            "tr_region_lookup_create_q64": (vp, [vp, u64, vp, vp, i32, u64, B, R, e]),
            "tr_region_lookup_get_view": (i32, [vp, P(_View)]),
            "tr_region_lookup_bound_bits": (u64, [vp]),
            "tr_region_lookup_eval": (i32, [P(_View), vp, vp, B, R]),
            "tr_region_lookup_free": (None, [vp]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, args
            native[name[10:]] = fn
        _native = native
    return _native


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_region_locate", C.c_int, [P(_View), C.c_void_p, C.c_void_p, P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _raw(budget):
    return C.byref((budget or Budget()).raw)


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _int(handle):
    return _mc._bigint_value(handle)


def _u64s(n):
    return np.zeros(max(int(n), 1), dtype=np.uint64)


def _int64_array(a, what):
    arr = np.asarray(a)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{what}: an integer array (floats are refused: values are exact)")
    if arr.dtype.kind == "u" and arr.size and int(arr.max()) >= 1 << 63:
        raise OverflowError(f"{what}: outside int64")
    return np.ascontiguousarray(arr, dtype=np.int64)


class Values:
    """Many parameter values as int64 arrays: numerators (m x d) over denominators (one positive integer, or m x d),
    handed to the native lookup as they are (marshalling only: no per-value conversion; the native code refuses a
    denominator below 1). Values that do not fit int64 go through the exact per-value path (a list of tuples)."""

    def __init__(self, numerators, denominators=1):
        num = _int64_array(numerators, "numerators")
        if num.ndim != 2:
            raise ValueError("numerators: an m x d array")
        den = _int64_array(denominators, "denominators")
        try:
            den = np.ascontiguousarray(np.broadcast_to(den, num.shape))
        except ValueError:
            raise ValueError("denominators: one integer or an m x d array") from None
        self.num, self.den = num, den

    def __len__(self):
        return int(self.num.shape[0])

    def __repr__(self):
        return f"Values({self.num.shape[0]} x {self.num.shape[1]})"


class PiecewiseLinear:
    """The explicit permitted contract for quadratic group functions: the box is divided into pieces^d equal cells and
    each function is replaced on each cell by an affine enclosure with a certified sup-norm error eps; the largest eps
    must not exceed error_budget (else NotConvergedError). The regions are then those of the enclosures, labelled
    approximate: the winner's true value is within 2 eps of the true envelope at every parameter value."""

    def __init__(self, pieces=1, error_budget=None):
        if isinstance(pieces, bool) or not isinstance(pieces, int) or not 1 <= pieces <= MAX_PIECES:
            raise ValueError(f"pieces is an integer in 1..{MAX_PIECES}")
        if error_budget is None:
            raise ValueError("the contract needs an explicit error_budget (an exact number >= 0)")
        self.pieces, self.error_budget = pieces, _exact(error_budget, "error_budget")
        if self.error_budget < 0:
            raise ValueError("error_budget >= 0")

    def __repr__(self):
        return f"PiecewiseLinear(pieces={self.pieces}, error_budget={self.error_budget})"


class _Marshal:
    """the native problem of the Python arguments (handles kept alive with it)"""

    def __init__(self, functions, box, rows, upper, tie, contract, d=None):
        box = [tuple(b) for b in box]
        d = len(box) if d is None else d
        if not 1 <= d:
            raise ValueError("the box needs at least one (lo, hi) pair")
        if any(len(b) != 2 for b in box):
            raise ValueError("box: one (lo, hi) pair per coordinate")
        if tie not in TIES:
            raise ValueError("tie is 'smallest' or 'largest'")
        if contract is not None and not isinstance(contract, PiecewiseLinear):
            raise TypeError("contract: a math_region.PiecewiseLinear (or None)")
        rows = [] if rows is None else list(rows)
        self.keep = []
        self.d = d
        aff, full = 1 + d, 1 + d + d * (d + 1) // 2
        if functions is not None:
            functions = list(functions)
            width = max([len(f) for f in functions if f is not None] + [aff])
            if width not in (aff, full):
                raise ValueError(f"a function has {aff} (affine) or {full} (with the quadratic monomials) coefficients")
            flat, present = [], np.zeros(max(len(functions), 1), dtype=np.uint8)
            for g, f in enumerate(functions):
                if f is None:
                    flat += [None] * width
                    continue
                f = list(f)
                if len(f) not in (aff, full):
                    raise ValueError(f"group {g}: {aff} or {full} coefficients expected, got {len(f)}")
                present[g] = 1
                flat += f + [None] * (width - len(f))
            self.coef = _Values(flat, "coefficient", allow_none=True)
            self.present = present
            self.ngroups, self.ncoef = len(functions), width
        else:
            self.coef, self.present, self.ngroups, self.ncoef = None, None, 0, aff
        self.lo = _Values([b[0] for b in box], "lo")
        self.hi = _Values([b[1] for b in box], "hi")
        A, b = [], []
        for r in rows:
            coeffs, rhs = r
            coeffs = list(coeffs)
            if len(coeffs) != d:
                raise ValueError(f"a row has {d} coefficients and a right-hand side: ((a_1, .., a_d), b)")
            A += coeffs
            b.append(rhs)
        self.A, self.b = _Values(A, "row coefficient", allow_none=True), _Values(b, "row bound")
        self.nrows = len(rows)
        self.budget_value = None
        if contract is not None:
            self.budget_value = _mc.BigRational(contract.error_budget)
        p = _Problem(d=d, ncoef=self.ncoef, nrows=self.nrows, upper=int(bool(upper)), tie=TIES[tie],
                     contract=1 if contract is not None else 0, pieces=contract.pieces if contract is not None else 1,
                     ngroups=self.ngroups)
        p.coef = C.cast(self.coef.array, C.c_void_p) if self.coef is not None else None
        p.present = self.present.ctypes.data if self.present is not None else None
        p.lo, p.hi = C.cast(self.lo.array, C.c_void_p), C.cast(self.hi.array, C.c_void_p)
        p.A = C.cast(self.A.array, C.c_void_p) if self.nrows else None
        p.b = C.cast(self.b.array, C.c_void_p) if self.nrows else None
        p.error_budget = self.budget_value._h if self.budget_value is not None else None
        self.problem = p


def need(functions, box, rows=None, *, upper=False, tie="smallest", contract=None):
    """the a-priori needs of a build ({resource: need}: variables, degree, states, bytes)"""
    m = _Marshal(functions, box, rows, upper, tie, contract)
    out = _Need()
    _check(_api()["need"](C.byref(m.problem), C.byref(out)), what="region needs")
    return _needs(out)


def regions(functions, box, rows=None, *, upper=False, tie="smallest", contract=None, budget=None):
    """the exact regions of the declared box (and rows: ((a_1, .., a_d), b) for a.theta <= b) where each group's
    function attains the minimum (upper=True: the maximum)"""
    m = _Marshal(functions, box, rows, upper, tie, contract)
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["build"](C.byref(m.problem), _raw(budget), C.byref(refusal), C.byref(error))
    return Regions(_made(h, error, refusal, "parameter regions"))


def from_expressions(expressions, variables, box, rows=None, *, upper=False, tie="smallest", contract=None,
                     budget=None):
    """the regions of first-slice polynomials (torbles.mathematics Expressions, e.g. RESP-001 response functions) in
    the parameter variables (one name per box coordinate); None for an empty group; a term in another variable raises
    DomainError, a degree above 2 UnsupportedError (2: only under a PiecewiseLinear contract)"""
    exprs = list(expressions)
    context = next((e.context for e in exprs if e is not None), None)
    if context is None:
        raise ValueError("give at least one expression")
    variables = list(variables)
    if len(variables) != len(list(box)):
        raise ValueError("one parameter variable per box coordinate")
    for v in variables:
        if v not in context.names:
            raise ValueError(f"{v!r} is not a variable of the expressions' context")
    m = _Marshal(None, box, rows, upper, tie, contract, d=len(variables))
    arr = (C.c_void_p * max(len(exprs), 1))(*[(e._open() if e is not None else None) for e in exprs])
    m.problem.ngroups = len(exprs)
    vars_ = (C.c_int32 * len(variables))(*[context.names.index(v) for v in variables])
    error, refusal = C.c_int32(), _Refusal()
    h = _api()["from_polys"](C.byref(m.problem), arr, vars_, _raw(budget), C.byref(refusal), C.byref(error))
    return Regions(_made(h, error, refusal, "parameter regions of expressions"))


class Regions:
    """The exact regions of the parameter set where each class of identical group functions attains the minimum (native,
    immutable): classes, regions (vertices, irredundant inequalities, LP evidence), vertices and boundaries with their
    ties, and point location on the CPU or the CUDA kernel."""

    def __init__(self, handle):
        self._h = handle
        api = _api()
        info = _Info()
        _check(api["get_info"](handle, C.byref(info)), what="regions")
        d = self.d = int(info.d)
        self.kind = "empty" if info.kind == 0 else "finite"
        self.claim = CLAIMS[int(info.claim)]
        self.upper, self.tie = bool(info.upper), {v: k for k, v in TIES.items()}[int(info.tie)]
        self.info = {n: int(getattr(info, n)) for n, _ in _Info._fields_}
        self.info.update(kind=self.kind, claim=self.claim, upper=self.upper, tie=self.tie,
                         contract="piecewise_linear" if info.contract else "exact")
        self.box = [(_borrowed(api["bound"](handle, 0, k)), _borrowed(api["bound"](handle, 1, k))) for k in range(d)]
        err = [api["error"](handle, i) for i in range(3)]
        self.error = _borrowed(err[0]) if err[0] else Fraction(0)
        self.envelope_error = _borrowed(err[1]) if err[1] else Fraction(0)
        self.error_budget = _borrowed(err[2]) if err[2] else None
        self.contract = None if not info.contract else dict(kind="piecewise_linear", pieces=int(info.pieces),
                                                            cells=int(info.cells), error_budget=self.error_budget)
        self.statement = (
            "exact: the regions of the declared functions" if self.claim == "exact" else
            f"approximate: the regions of affine enclosures (|f - L| <= {self.error} per cell); at every parameter "
            f"value the winner's true value is within {self.envelope_error} of the true envelope")
        self.classes = []
        for c in range(int(info.classes)):
            ci = _ClassInfo()
            _check(api["class_get"](handle, c, C.byref(ci)), what="region class")
            mem = _u64s(ci.members)
            _check(api["class_members"](handle, c, mem.ctypes.data, len(mem)), what="region class")
            self.classes.append(dict(
                cell=int(ci.cell), coefficients=tuple(_borrowed(api["class_coef"](handle, c, k)) for k in range(d + 1)),
                groups=[int(g) for g in mem[:int(ci.members)]], winner=int(ci.winner),
                region=None if ci.region == NONE else int(ci.region), degree=int(ci.degree),
                eps=_borrowed(api["class_eps"](handle, c)), exact=bool(ci.exact)))
        self.vertices = []
        pts = (C.c_void_p * d)()
        for v in range(int(info.vertices)):
            vi = _VertexInfo()
            _check(api["vertex_get"](handle, v, C.byref(vi)), what="region vertex")
            _check(api["vertex_point"](handle, v, pts), what="region vertex")
            cls, walls = _u64s(vi.classes), _u64s(vi.walls)
            _check(api["vertex_classes"](handle, v, cls.ctypes.data, len(cls)), what="region vertex")
            _check(api["vertex_walls"](handle, v, walls.ctypes.data, len(walls)), what="region vertex")
            classes = [int(x) for x in cls[:int(vi.classes)]]
            groups = sorted(g for c in classes for g in self.classes[c]["groups"])
            self.vertices.append(dict(point=tuple(_borrowed(pts[k]) for k in range(d)), cell=int(vi.cell),
                                      classes=classes, groups=groups, winner=int(vi.winner),
                                      walls=[self._wall(int(w)) for w in walls[:int(vi.walls)]]))
        self.regions = []
        normal = (C.c_void_p * d)()
        for r in range(int(info.regions)):
            ri = _RegionInfo()
            _check(api["region_get"](handle, r, C.byref(ri)), what="region")
            vid = _u64s(ri.vertices)
            _check(api["region_vertices"](handle, r, vid.ctypes.data, len(vid)), what="region")
            ineq, eq = [], []
            for i in range(int(ri.inequalities + ri.equalities)):
                rhs, equality, nsrc = C.c_void_p(), C.c_int32(), C.c_uint64()
                _check(api["region_constraint"](handle, r, i, normal, C.byref(rhs), C.byref(equality), C.byref(nsrc)),
                       what="region constraint")
                kinds = np.zeros(max(int(nsrc.value), 1), dtype=np.int32)
                idx = _u64s(nsrc.value)
                _check(api["region_sources"](handle, r, i, kinds.ctypes.data, idx.ctypes.data, len(idx)),
                       what="region constraint")
                entry = dict(normal=tuple(_int(normal[k]) for k in range(d)), rhs=_int(rhs.value),
                             sources=[(SOURCES[int(kinds[q])], int(idx[q])) for q in range(int(nsrc.value))])
                (eq if equality.value else ineq).append(entry)
            point, margin = (C.c_void_p * d)(), C.c_void_p()
            _check(api["region_lp"](handle, r, point, C.byref(margin)), what="region")
            lp = dict(rows=int(ri.lp_rows), sign=int(ri.lp_sign),
                      point=tuple(_borrowed(point[k]) for k in range(d)) if margin.value else None,
                      margin=_borrowed(margin.value) if margin.value else None)
            cls = self.classes[int(ri.cls)]
            self.regions.append(dict(cls=int(ri.cls), cell=int(ri.cell), groups=list(cls["groups"]),
                                     winner=int(ri.winner), dimension=int(ri.dim), full=bool(ri.full),
                                     vertices=[int(x) for x in vid[:int(ri.vertices)]], inequalities=ineq,
                                     equalities=eq, lp=lp))
        self.boundaries = []
        for b in range(int(info.boundaries)):
            bi = _BoundaryInfo()
            _check(api["boundary_get"](handle, b, C.byref(bi)), what="region boundary")
            rhs = C.c_void_p()
            _check(api["boundary_plane"](handle, b, normal, C.byref(rhs)), what="region boundary")
            vid, cls = _u64s(bi.vertices), _u64s(bi.classes)
            _check(api["boundary_vertices"](handle, b, vid.ctypes.data, len(vid)), what="region boundary")
            _check(api["boundary_classes"](handle, b, cls.ctypes.data, len(cls)), what="region boundary")
            classes = [int(x) for x in cls[:int(bi.classes)]]
            self.boundaries.append(dict(kind=BOUNDARY[int(bi.kind)], cell=int(bi.cell), region=int(bi.region),
                                        normal=tuple(_int(normal[k]) for k in range(d)), rhs=_int(rhs.value),
                                        vertices=[int(x) for x in vid[:int(bi.vertices)]], classes=classes,
                                        groups=sorted(g for c in classes for g in self.classes[c]["groups"]),
                                        winner=int(bi.winner)))

    def _wall(self, w):
        d = self.d
        if w < 2 * d:
            return ("lower" if w % 2 == 0 else "upper", w // 2)
        return ("row", w - 2 * d)

    def _open(self):
        if not self._h:
            raise ValueError("the regions are closed")
        return self._h

    # ---------------------------------------------------------------- point location
    def _points(self, points):
        """(kind, data): int64 numerators / denominators when every value fits them, else native big rationals
        (marshaling only: the exact values are converted, never rounded)"""
        if isinstance(points, Values):
            if points.num.shape[1] != self.d:
                raise ValueError(f"a parameter value has {self.d} coordinates")
            m = len(points)
            if not m:
                return "q64", (np.zeros(1, np.int64), np.ones(1, np.int64), 0)
            return "q64", (points.num.reshape(-1), points.den.reshape(-1), m)
        qs = []
        for p in points:
            p = list(p)
            if len(p) != self.d:
                raise ValueError(f"a parameter value has {self.d} coordinates")
            qs.append([_exact(x, "parameter value") for x in p])
        lim = 1 << 63
        flat = [x for p in qs for x in p]
        if all(-lim <= q.numerator < lim and q.denominator < lim for q in flat):
            num = np.array([q.numerator for q in flat] or [0], dtype=np.int64)
            den = np.array([q.denominator for q in flat] or [1], dtype=np.int64)
            return "q64", (num, den, len(qs))
        return "big", (_Values(flat, "parameter value"), len(qs))

    def locate_need(self, points, mode="index", backend="cpu", ties=0):
        kind, data = self._points(points if isinstance(points, Values) else list(points))
        out = _Need()
        target = 0 if backend == "cpu" else 1
        if kind == "q64":
            num, den, n = data
            rc = _api()["lookup_need_q64"](self._open(), n, num.ctypes.data, den.ctypes.data, MODES[mode], int(ties),
                                           target, C.byref(out))
        else:
            vals, n = data
            rc = _api()["lookup_need"](self._open(), n, vals.array, MODES[mode], int(ties), target, C.byref(out))
        _check(rc, what="lookup needs")
        return _needs(out)

    def locate(self, points, *, backend="cpu", mode="index", ties=0, budget=None):
        """per parameter value: {"status" ("ok" / "outside" / "empty"), "kind" ("interior" / "boundary"), "cell",
        "winner" (the tie policy's group), "class", "region", "tied" (classes attaining the minimum), and with
        ties = k > 0 "classes": the first k tied classes}; mode "index" (kd-tree), "regions" (every region's
        inequalities) or "all" (every class evaluated); points: tuples of exact numbers, or a Values"""
        out, tie_ids, n = self._run(points, backend, mode, ties, budget)
        res = []
        for i, a in enumerate(out[:n]):
            r = dict(status=STATUS[a.status], kind=KINDS.get(a.kind), cell=None if a.cell == NONE else int(a.cell),
                     winner=None if a.winner == NONE else int(a.winner), cls=None if a.cls == NONE else int(a.cls),
                     region=None if a.region == NONE else int(a.region), tied=int(a.tied))
            if ties:
                r["classes"] = [int(x) for x in tie_ids[i * ties:(i + 1) * ties] if x != NONE]
            res.append(r)
        return res

    def locate_arrays(self, points, *, backend="cpu", mode="index", ties=0, budget=None):
        """locate() as int64 arrays (one entry per value, -1 for none): status (0 ok, 1 outside, 2 empty), kind (1
        interior, 2 boundary, 0 none), cell, winner, cls, region, tied, and with ties = k > 0 "classes" (m x k)"""
        out, tie_ids, n = self._run(points, backend, mode, ties, budget)
        raw = np.ctypeslib.as_array(out)[:n] if n else None
        def col(name):
            if raw is None:
                return np.zeros(0, dtype=np.int64)
            return raw[name].astype(np.int64)            # NONE (2^64 - 1) reads as -1
        res = {name: col(name) for name in ("status", "kind", "cell", "winner", "cls", "region", "tied")}
        if ties:
            res["classes"] = tie_ids[:n * ties].astype(np.int64).reshape(n, ties)
        return res

    def _run(self, points, backend, mode, ties, budget):
        """the native lookup: (answers, tie ids, n)"""
        if mode not in MODES:
            raise ValueError("mode is 'index', 'regions' or 'all'")
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend is 'cpu' or 'cuda'")
        if isinstance(ties, bool) or not isinstance(ties, int) or ties < 0:
            raise ValueError("ties: the number of tied classes listed per value (>= 0)")
        if not isinstance(points, Values):
            points = list(points)
        if backend == "cuda":                      # the host job and its device copies, checked together
            from . import math_budgets
            math_budgets.check(budget or Budget(), "region lookup on the device",
                               **self.locate_need(points, mode=mode, backend="cuda", ties=ties))
        kind, data = self._points(points)
        error, refusal = C.c_int32(), _Refusal()
        if kind == "q64":
            num, den, n = data
            job = _api()["lookup_create_q64"](self._open(), n, num.ctypes.data, den.ctypes.data, MODES[mode], ties,
                                              _raw(budget), C.byref(refusal), C.byref(error))
        else:
            vals, n = data
            job = _api()["lookup_create"](self._open(), n, vals.array, MODES[mode], ties, _raw(budget),
                                          C.byref(refusal), C.byref(error))
        job = _made(job, error, refusal, "region lookup")
        try:
            view = _View()
            _check(_api()["lookup_get_view"](job, C.byref(view)), what="region lookup")
            out = (_Answer * max(n, 1))()
            tie_ids = np.full(max(n * ties, 1), NONE, dtype=np.uint64)
            refusal = _Refusal()
            if backend == "cpu":
                _check(_api()["lookup_eval"](C.byref(view), out, tie_ids.ctypes.data if ties else None, _raw(budget),
                                             C.byref(refusal)), refusal, "region lookup")
                label = "cpu"
            else:
                gpu = _gpu()
                gpu.run("tr_mgpu_region_locate", C.byref(view), out, tie_ids.ctypes.data if ties else None,
                        _raw(budget), C.byref(refusal), refusal=refusal, what="region lookup on the device")
                label = "cuda-emulated" if gpu.device()["emulated"] else "cuda"
            self.last_lookup = dict(backend=label, mode=mode, primes=int(view.nprimes),
                                    bound_bits=int(_api()["lookup_bound_bits"](job)))
        finally:
            _api()["lookup_free"](job)
        for a in out[:n]:
            if a.status < 0:
                _check(int(a.status), what="region lookup")
        return out, tie_ids, n

    def evaluate_all(self, points, *, backend="cpu", ties=0, budget=None):
        """every class of the value's cell evaluated exactly (the verification path)"""
        return self.locate(points, backend=backend, mode="all", ties=ties, budget=budget)

    # ---------------------------------------------------------------- FND-007
    def decide(self):
        """the FND-007 decision on using the regions as an exact result (a dict, never raising)"""
        from . import math_certificates as mc
        d = mc._Decision()
        rc = _api()["authorize"](self._open(), C.byref(d))
        return dict(authorized=bool(d.authorized), reason=mc.REFUSALS.get(int(d.reason)), claim=self.claim,
                    code=int(rc))

    def authorize_exact(self):
        """an exact build is authorized (the decision); an approximate one raises NotAuthorizedError ("approximate")"""
        d = self.decide()
        if d["authorized"]:
            return d
        from . import math_certificates as mc
        err = mc.NotAuthorizedError(f"approximate regions ({self.statement}) cannot authorize an exact result")
        err.reason, err.decision = d["reason"], d
        raise err

    def close(self):
        if getattr(self, "_h", None):
            _api()["free"](self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._h:
            return "Regions(closed)"
        return (f"Regions(d={self.d}, {'upper' if self.upper else 'lower'}, {self.claim}, classes={len(self.classes)}, "
                f"regions={len(self.regions)}, vertices={len(self.vertices)})")
