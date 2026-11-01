# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROB-002: factorized probabilistic inference over bounded factor graphs of discrete variables with declared finite
supports, natively on the CPU or with the CUDA kernels (torbles/math_factor.c, math_factor_gpu.cu).

    from fractions import Fraction
    from torbles import math_factor as mf, math_tensor as mt, math_prob as mp, math_conv as mc
    V = {"x": 2, "y": [0, 1, 2], "z": mf.values([10, 20, None])}     # declared supports (None: the NULL category)
    F = [mf.table(["x", "y"], [[1, 2, 0], [Fraction(1, 2), 1, 3]]),   # exact rationals, C order
         mf.tensor(mt.counts(scope, ["y", "z"]), ["y", "z"]),         # a TENSOR-001 count tensor (by value)
         mf.categorical(mp.categorical(S), ["x"], [1, 3])]            # a PROB-001 categorical model's weights
    G = mf.graph(V, F, declaration=None)                              # linked data factors need mc.independent(scope)
    G.partition(); G.partition(evidence={"z": 20})                    # Z, Z_e, P(e)
    G.likelihood([{"z": 10}, {"z": 20}], backend="cuda")              # a batch of evidence requests
    G.marginal(["x", "y"], evidence={"z": None})                      # P(x, y | e), exact
    G.conditional(["x"], given=["y"]); G.map(evidence=...); G.kbest(3)
    G.plan(["x"], evidence=["z"], devices=("cpu", "cuda"), budget=Budget(degree=2, tensor=4096))
    G.count(method="symcount" | "brute" | "diagram" | "tensor")      # COUNT-002 / TENSOR-002 alternative backends

Every value is an exact rational (Fraction) computed natively; the elimination order is chosen by the FND-008 planner
over costed candidate orders under the budget's degree (treewidth), tensor (largest joint table) and support (largest
intermediate table) limits; when no candidate fits, AlgebraBudgetError carries the plan (.plan, .every_order: proven
for every order). Marginals of a variable set come from the factor graph, never from products of single marginals;
factors of one data source multiply only under a GEN-002 independence declaration. Opt-in: `import torbles` does not
load this module.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, UNKNOWN, budget_error, check as _check

__all__ = ["graph", "values", "table", "tensor", "categorical", "stats", "Graph", "Plan", "Query", "HEURISTICS",
           "METHODS", "default_rates"]

KINDS = {"table": 1, "tensor": 2, "categorical": 3, "stats": 4}
_KIND_NAMES = {v: k for k, v in KINDS.items()}
MODES = {"sum": 1, "max": 2, "kbest": 3}
HEURISTICS = {"exact": 0, "min-fill": 1, "min-degree": 2, "min-weight": 3, "weighted-fill": 4, "natural": 5,
              "reverse": 6, "given": 7}
_HEURISTIC_NAMES = {v: k for k, v in HEURISTICS.items()}
METHODS = {"symcount": 1, "brute": 2, "diagram": 3, "tensor": 4}
ANSWERS = {"partition": 1, "marginal": 2, "conditional": 3, "max": 4, "kbest": 5, "count": 6}
_OUT_REASONS = {1: "value", 2: "scale", 3: "dims", 4: "negative"}
_CHECK_REASONS = {0: None, 1: "needs_declaration", 2: "not_covered", 3: "keys_differ", 4: "grouping_differs",
                  5: "scales_differ"}
BY_CPU, BY_DEVICE = 1, 2
ON_CPU, ON_CUDA = 1, 2
MAX_SCOPE = 31
I64 = (-(1 << 63), (1 << 63) - 1)
P = C.POINTER
i32, u32, i64, u64, vp, dbl = C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_void_p, C.c_double
COMPONENTS = 12


class _Spec(C.Structure):
    _native_ = "tr_factor_spec"
    _fields_ = [("kind", i32), ("mode", i32), ("nscope", u32), ("reserved", u32), ("scope_at", u64),
                ("table_at", u64), ("group", u64), ("object", vp), ("scope", vp)]


class _Outside(C.Structure):
    _native_ = "tr_factor_outside"
    _fields_ = [("factor", u32), ("axis", u32), ("reason", i32), ("null", i32), ("value", i64), ("mass", u64)]


class _Check(C.Structure):
    _native_ = "tr_factor_check"
    _fields_ = [("reason", i32), ("first", i32), ("second", i32), ("keyed", i32), ("digest", u64 * 2)]


class _GraphInfo(C.Structure):
    _native_ = "tr_factor_graph_info"
    _fields_ = [(n, u64) for n in ("entries", "bytes", "assignments_lo", "assignments_hi")] + [
        ("digest", u64 * 2)] + [(n, u32) for n in ("nvars", "nfactors", "max_scope", "wide")] + [
        (n, i32) for n in ("declared", "linked", "sources", "data")] + [
        ("limbs", u32), ("reserved", u32), ("max_table", u64), ("reserved2", u64)]


class _FactorInfo(C.Structure):
    _native_ = "tr_factor_info"
    _fields_ = [("kind", i32), ("mode", i32), ("nscope", u32), ("linked", u32), ("entries", u64), ("group", u64),
                ("nonzero", u64), ("denom", u64 * 2), ("maxent", u64 * 2), ("source_digest", u64 * 2),
                ("source_kind", i32), ("keyed", i32)]


class _Shape(C.Structure):
    _native_ = "tr_factor_shape"
    _fields_ = [("mode", i32), ("nkeep", u32), ("nfixed", u32), ("reserved", u32), ("keep", vp), ("fixed", vp),
                ("requests", u64), ("kbest", u64)]


class _Options(C.Structure):
    _native_ = "tr_factor_plan_options"
    _fields_ = [(n, i32) for n in ("devices", "heuristics", "force", "custom_rates")] + [
        ("norder", u32), ("dp_limit", u32), ("order", vp), ("rates", (dbl * COMPONENTS) * 2)]


def _reg():
    from . import math_registry as reg
    return reg


class _Candidate(C.Structure):
    _native_ = "tr_factor_candidate"
    _fields_ = [("reg", _reg()._Candidate)] + [(n, i32) for n in ("heuristic", "device", "nsteps", "order")] + [
        (n, u64) for n in ("width", "joint", "support", "products", "arena", "bytes", "device_bytes")] + [
        ("limbs", u32), ("device_ok", u32)]


class _PlanInfo(C.Structure):
    _native_ = "tr_factor_plan_info"
    _fields_ = [(n, i32) for n in ("ncandidates", "chosen", "code", "reason", "complete", "every_order", "forced",
                                   "devices", "bounds_exact", "mode", "norders", "reserved")] + [
        (n, u64) for n in ("eliminate", "keep", "fixed", "free_vars", "dp_states", "evaluated", "min_width",
                           "min_joint", "min_support")] + [("refusal", _Refusal), ("bytes", u64)]


class _Step(C.Structure):
    _native_ = "tr_factor_step"
    _fields_ = [(n, u32) for n in ("final", "var", "nout", "nopd")] + [
        (n, u64) for n in ("cells", "nterms", "out_at", "opd_at", "dim_at")]


class _Operand(C.Structure):
    _native_ = "tr_factor_operand"
    _fields_ = [("kind", u32), ("index", u32), ("stride_at", u64)]


class _Program(C.Structure):
    _native_ = "tr_factor_program"
    _fields_ = [(n, u32) for n in ("nsteps", "limbs", "mode", "nfactors")] + [
        (n, u64) for n in ("arena", "nopd", "nstrides", "ndims", "inputs")] + [
        (n, vp) for n in ("step", "opd", "stride", "dims", "input")]


class _AnswerInfo(C.Structure):
    _native_ = "tr_factor_answer_info"
    _fields_ = [(n, i32) for n in ("kind", "backend", "mode", "method")] + [
        (n, u64) for n in ("requests", "nvalues", "cells", "assignments")] + [
        (n, u32) for n in ("nvars", "limbs", "nkeep", "ngiven")]


for _s, _n in ((_Spec, 56), (_Outside, 32), (_Check, 32), (_GraphInfo, 104), (_FactorInfo, 96), (_Shape, 48),
               (_Options, 224), (_Candidate, 456), (_PlanInfo, 152), (_Step, 56), (_Operand, 16), (_Program, 96),
               (_AnswerInfo, 64)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_factor_graph_create": (vp, [u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, u64, vp, B, R, P(_Outside),
                                            P(_Check), e]),
            "tr_factor_graph_get_info": (i32, [vp, P(_GraphInfo)]),
            "tr_factor_graph_factor": (i32, [vp, u32, P(_FactorInfo)]),
            "tr_factor_graph_table": (i32, [vp, u32, vp, vp, u64, vp]),
            "tr_factor_graph_free": (None, [vp]),
            "tr_factor_plan_create": (i32, [vp, P(_Shape), P(_Options), B, P(vp), R]),
            "tr_factor_plan_get_info": (i32, [vp, P(_PlanInfo)]),
            "tr_factor_plan_get": (i32, [vp, i32, P(_Candidate)]),
            "tr_factor_plan_order": (i32, [vp, i32, vp, u64, P(u64)]),
            "tr_factor_plan_free": (None, [vp]),
            "tr_factor_default_rates": (None, [vp]),
            "tr_factor_prog_create": (vp, [vp, vp, i32, B, R, e]),
            "tr_factor_prog_get_view": (i32, [vp, P(_Program)]),
            "tr_factor_prog_candidate": (i32, [vp, P(i32), P(i32)]),
            "tr_factor_prog_bases": (i32, [vp, u64, vp, vp]),
            "tr_factor_prog_need": (i32, [vp, u64, i32, P(_Need)]),
            "tr_factor_eliminate": (i32, [P(_Program), u64, vp, vp, vp, vp, vp, B, R]),
            "tr_factor_prog_free": (None, [vp]),
            "tr_factor_answer_create": (vp, [vp, i32, u32, u64, vp, vp, vp, vp, vp, vp, i32, B, R, e]),
            "tr_factor_kbest": (vp, [vp, u64, vp, u64, vp, B, R, e]),
            "tr_factor_alternative": (vp, [vp, i32, u32, vp, vp, B, R, e]),
            "tr_factor_answer_get_info": (i32, [vp, P(_AnswerInfo)]),
            "tr_factor_answer_status": (i32, [vp, u64]),
            "tr_factor_answer_value": (i32, [vp, u64, u64, vp, vp, u32, P(u32), P(u32)]),
            "tr_factor_answer_column": (i32, [vp, u64, u64, u64, vp, vp, u32, vp, vp]),
            "tr_factor_answer_assignments": (i32, [vp, u64, vp, u64, P(u64)]),
            "tr_factor_answer_tied": (i32, [vp, u64, P(u64)]),
            "tr_factor_answer_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_factor_eliminate", C.c_int, [P(_Program), u64, vp, vp, vp, vp, vp, P(_Budget), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _precheck(budget, raw_need, what):
    refusal = _Refusal()
    raw = _budget(budget)
    _check(_call("tr_math_budget_check", C.byref(raw), C.byref(raw_need), C.byref(refusal)), refusal, what)
    for i in range(len(RESOURCES)):
        if raw_need.need[i] == UNKNOWN:
            raise budget_error(i, UNKNOWN, int(raw.limit[i]), what)


def _limbs_int(words, n):
    return sum(int(words[i]) << (64 * i) for i in range(n))


def _exact(x, what):
    """an exact nonnegative rational with int64 numerator and denominator (floats and bools are refused)"""
    if isinstance(x, bool) or isinstance(x, (float, np.floating)):
        raise TypeError(f"{what} must be exact (int, Fraction, Decimal), not {type(x).__name__}")
    if isinstance(x, np.integer):
        x = int(x)
    if isinstance(x, Decimal):
        x = Fraction(x)
    if not isinstance(x, (Integral, Rational)):
        raise TypeError(f"{what} must be exact (int, Fraction, Decimal), not {type(x).__name__}")
    q = Fraction(x)
    if not (I64[0] <= q.numerator <= I64[1] and q.denominator <= I64[1]):
        raise OverflowError(f"{what}: numerator and denominator must fit int64")
    return q


def default_rates():
    """the candidates' relative work units per amount ({"cpu": {...}, "cuda": {...}}, NOT time)"""
    reg = _reg()
    t = ((dbl * COMPONENTS) * 2)()
    _call("tr_factor_default_rates", t)
    return {dev: dict(zip(reg.COST_COMPONENTS, (float(t[i][k]) for k in range(COMPONENTS))))
            for i, dev in enumerate(("cpu", "cuda"))}


# ---------------------------------------------------------------------- declared supports
class Values:
    """A declared support of one variable: values (logical integers; Decimal / Fraction values scaled by 10^scale, which
    must be integral; None: the NULL category) in the given order."""

    def __init__(self, values, scale=0):
        self.labels = tuple(values)
        if not self.labels:
            raise ValueError("a variable has at least one declared value")
        self.scale = int(scale)
        if not 0 <= self.scale <= 18:
            raise ValueError("a decimal scale is 0..18")
        logical, nulls = [], []
        for x in self.labels:
            if x is None:
                logical.append(0); nulls.append(1)
                continue
            if isinstance(x, bool) or isinstance(x, (float, np.floating)):
                raise TypeError("declared values are exact integers, Decimals or Fractions (or None)")
            if isinstance(x, (Integral, np.integer)):
                v = int(x)                                  # a logical integer, as DIST-001 / TENSOR-001 report it
            else:
                q = Fraction(x) * (10 ** self.scale)
                if q.denominator != 1:
                    raise ValueError(f"{x} is not a multiple of 10^-{self.scale}")
                v = int(q)
            if not I64[0] <= v <= I64[1]:
                raise OverflowError("a declared value does not fit int64")
            logical.append(v); nulls.append(0)
        self.logical, self.nulls = logical, nulls
        if len(set(zip(logical, nulls))) != len(logical):
            raise ValueError("declared values must be distinct")
        self.index = {lab: i for i, lab in enumerate(self.labels)}

    def __len__(self):
        return len(self.labels)


def values(values, scale=0):
    """a declared support (logical integers: the values DIST-001 / TENSOR-001 report; None: the NULL category)"""
    return Values(values, scale)


def _variable(spec):
    if isinstance(spec, Values):
        return spec
    if isinstance(spec, (Integral, np.integer)) and not isinstance(spec, bool):
        d = int(spec)
        if d < 1:
            raise ValueError("a domain has at least one value")
        return Values(range(d))
    try:
        from .math_prob import Support
        if isinstance(spec, Support):
            if spec.k != 1:
                raise ValueError("a variable's support has one coordinate")
            return Values(spec.points, spec.scales[0])
    except ImportError:
        pass
    return Values(list(spec))


# ---------------------------------------------------------------------- factor definitions
class _FactorDef:
    def __init__(self, kind, scope, **kw):
        self.kind, self.scope = kind, tuple(scope)
        self.__dict__.update(kw)


def table(scope, entries):
    """an explicit factor over the scope's variables: nested lists of exact nonnegative rationals (shape = the domains,
    C order), a flat sequence in C order, or a dict {(labels...): value} (missing assignments are 0)"""
    return _FactorDef(KINDS["table"], scope, entries=entries)


def tensor(t, scope, group=0):
    """group `group` of a TENSOR-001 tensor (math_tensor.Tensor) as a factor, axis k on variable scope[k]: count axes
    map by value to the variable's declared support, index axes by index"""
    from .math_tensor import Tensor
    if not isinstance(t, Tensor):
        raise TypeError("tensor() takes a math_tensor.Tensor")
    return _FactorDef(KINDS["tensor"], scope, obj=t, group=int(group))


def categorical(model, scope, weights, mode="weights"):
    """a PROB-001 categorical model (math_prob.Model) with one request's weights (per support point: a sequence in the
    support's order, a dict or a callable; mode "probabilities": exact probabilities summing to 1); coordinate c on
    variable scope[c]"""
    from .math_prob import Model
    if not isinstance(model, Model):
        raise TypeError("categorical() takes a math_prob.Model")
    if mode not in ("weights", "probabilities"):
        raise ValueError("mode is 'weights' or 'probabilities'")
    w = model.support._aligned(weights, "weights", default=0)
    return _FactorDef(KINDS["categorical"], scope, obj=model, weights=[_exact(x, "a weight") for x in w],
                      mode=0 if mode == "weights" else 1)


def stats(st, scope, group=0):
    """group `group` of PROB-001 retained statistics (math_prob.Stats: the masses of the declared support) as a factor"""
    from .math_prob import Stats
    if not isinstance(st, Stats):
        raise TypeError("stats() takes math_prob.Stats")
    return _FactorDef(KINDS["stats"], scope, obj=st, group=int(group))


def _table_entries(entries, dims, labels, what):
    size = 1
    for d in dims:
        size *= d
    if isinstance(entries, dict):
        flat = [0] * size
        for key, val in entries.items():
            key = key if isinstance(key, tuple) else (key,)
            if len(key) != len(dims):
                raise ValueError(f"{what}: an assignment of {len(dims)} values")
            idx = 0
            for lab, d, vl in zip(key, dims, labels):
                if lab not in vl.index:
                    raise ValueError(f"{what}: {lab!r} is not a declared value")
                idx = idx * d + vl.index[lab]
            flat[idx] = val
    else:
        arr = np.array(entries, dtype=object)
        if arr.shape == tuple(dims) or (len(dims) == 0 and arr.shape in ((), (1,))):
            flat = list(arr.reshape(-1))
        elif arr.ndim == 1 and arr.shape[0] == size:
            flat = list(arr)
        else:
            raise ValueError(f"{what}: a table of shape {tuple(dims)}, not {arr.shape}")
    return [_exact(v, f"{what} entry") for v in flat]


def _outside_error(out, graph_names, factors, what):
    f = int(out.factor)
    reason = _OUT_REASONS.get(int(out.reason))
    fac = factors[f] if f < len(factors) else None
    var = fac.scope[int(out.axis)] if fac is not None and int(out.axis) < len(fac.scope) else None
    if reason == "value":
        val = None if out.null else int(out.value)
        msg = (f"{what}: factor {f} holds {val!r} for variable {var!r} (mass {int(out.mass)}), which is not a declared "
               f"value of the variable: an observed value outside the declared support")
    elif reason == "scale":
        msg = f"{what}: factor {f}: the decimal scale of variable {var!r} differs ({int(out.value)})"
    elif reason == "dims":
        msg = f"{what}: factor {f}: an index axis of dimension {int(out.value)} on variable {var!r} of another domain"
    elif reason == "negative":
        msg = f"{what}: factor {f} has a negative entry"
    else:
        msg = f"{what}: factor {f}: incompatible domain (a negative weight, probabilities not summing to 1, ...)"
    err = DomainError(msg)
    err.factor, err.variable, err.reason = f, var, reason
    err.value = None if out.null else int(out.value)
    return err


def _check_error(ck, what):
    reason = _CHECK_REASONS.get(int(ck.reason), "unknown")
    i, j = int(ck.first), int(ck.second)
    src = f"{int(ck.digest[0]):016x}{int(ck.digest[1]):016x}"
    missing = ("no independence declaration was given" if reason == "needs_declaration"
               else "the given independence declaration does not cover it")
    err = DomainError(f"{what}: factors {i} and {j} come from the same {'keyed ' if ck.keyed else ''}data source "
                      f"(source {src}) and {missing}: their product would factor the joint of linked rows into "
                      f"marginals; use the joint state as one factor or declare the independence "
                      f"(math_conv.independent(scope))")
    err.reason, err.factors, err.keyed, err.source = reason, (i, j), bool(ck.keyed), src
    return err


# ---------------------------------------------------------------------- the graph
def graph(variables, factors, *, declaration=None, budget=None):
    """a factor graph: variables {name: domain} (an int d: values 0..d-1; a sequence of declared values; values(...);
    a one-coordinate PROB-001 Support) and factors (table, tensor, categorical, stats); declaration: a GEN-002
    independence declaration (math_conv.independent(scope, ...)) admitting data factors of one source"""
    return Graph(variables, factors, declaration=declaration, budget=budget)


class Graph:
    """An immutable factor graph over declared finite supports (native), with its queries."""

    def __init__(self, variables, factors, *, declaration=None, budget=None):
        from . import math_conv as mcv
        items = list(variables.items()) if isinstance(variables, dict) else list(variables)
        self.names = tuple(str(n) for n, _ in items)
        if len(set(self.names)) != len(self.names):
            raise ValueError("variable names must be distinct")
        self.vars = tuple(_variable(s) for _, s in items)
        self.pos = {n: i for i, n in enumerate(self.names)}
        self.domains = tuple(len(v) for v in self.vars)
        n = len(self.names)
        doms = np.array(self.domains or [1], dtype=np.uint32)
        vals = np.array([x for v in self.vars for x in v.logical] or [0], dtype=np.int64)
        nulls = np.array([x for v in self.vars for x in v.nulls] or [0], dtype=np.uint8)
        scales = np.array([v.scale for v in self.vars] or [0], dtype=np.int32)
        self.factors = list(factors)
        specs = (_Spec * max(len(self.factors), 1))()
        scope_ids, num, den = [], [], []
        keep = []                                         # objects whose handles the specs borrow
        self._scopes = []
        for j, fd in enumerate(self.factors):
            if not isinstance(fd, _FactorDef):
                raise TypeError(f"factor {j}: use math_factor.table / tensor / categorical / stats")
            if any(v not in self.pos for v in fd.scope):
                raise ValueError(f"factor {j}: a scope variable is not a variable of the graph")
            if len(set(fd.scope)) != len(fd.scope):
                raise ValueError(f"factor {j}: scope variables must be distinct")
            sp = specs[j]
            sp.kind, sp.nscope, sp.scope_at = fd.kind, len(fd.scope), len(scope_ids)
            scope_ids += [self.pos[v] for v in fd.scope]
            if fd.kind == KINDS["table"]:
                dims = [self.domains[self.pos[v]] for v in fd.scope]
                ent = _table_entries(fd.entries, dims, [self.vars[self.pos[v]] for v in fd.scope], f"factor {j}")
                sp.table_at = len(num)
                num += [q.numerator for q in ent]
                den += [q.denominator for q in ent]
            elif fd.kind == KINDS["tensor"]:
                sp.object, sp.group = fd.obj._open(), fd.group
                if fd.obj.scope is not None:
                    sp.scope = fd.obj.scope._handle
                    self._scopes.append(fd.obj.scope)
                keep.append(fd.obj)
            elif fd.kind == KINDS["categorical"]:
                sp.object, sp.mode, sp.table_at = fd.obj._open(), fd.mode, len(num)
                num += [q.numerator for q in fd.weights]
                den += [q.denominator for q in fd.weights]
                keep.append(fd.obj)
            else:
                sp.object, sp.group = fd.obj._open(), fd.group
                scope = fd.obj.scope
                if scope is None or isinstance(scope, tuple):
                    raise ValueError(f"factor {j}: statistics need their data source (a scoped histogram, no product)")
                sp.scope = scope._handle
                self._scopes.append(scope)
                keep.append(fd.obj)
        for s in self._scopes:
            if s.source == "table":
                s.require_fresh("factor graph")
        ids = np.array(scope_ids or [0], dtype=np.uint32)
        nn, dd = np.array(num or [0], dtype=np.int64), np.array(den or [1], dtype=np.int64)
        error, refusal, out, ck = C.c_int32(), _Refusal(), _Outside(), _Check()
        h = _call("tr_factor_graph_create", n, doms.ctypes.data, vals.ctypes.data, nulls.ctypes.data,
                  scales.ctypes.data, len(self.factors), specs, ids.ctypes.data, nn.ctypes.data, dd.ctypes.data,
                  len(num), mcv._decl(declaration), C.byref(_budget(budget)), C.byref(refusal), C.byref(out),
                  C.byref(ck), C.byref(error))
        if error.value == -6 and ck.reason:
            raise _check_error(ck, "factor graph")
        if error.value == -6:
            raise _outside_error(out, self.names, self.factors, "factor graph")
        if error.value == -12:
            from .math_errors import UnsupportedError
            raise UnsupportedError("factor graph: outside the declared scope (an exponential-family model, a moment, "
                                   "factored or counted tensor, a product of statistics, more than 31 scope "
                                   "variables, a domain past 2^16, denominators or entries past 2^126)")
        self._handle = _made(h, error, refusal, "factor graph")
        self._keep, self._declaration = keep, declaration
        self._norm = None
        self.budget = budget

    # ------------------------------------------------------------------ description
    def _open(self):
        if not self._handle:
            raise ValueError("the factor graph is closed")
        return self._handle

    @property
    def info(self):
        i = _GraphInfo()
        _check(_call("tr_factor_graph_get_info", self._open(), C.byref(i)), what="factor graph")
        return dict(variables=int(i.nvars), factors=int(i.nfactors), entries=int(i.entries), bytes=int(i.bytes),
                    max_scope=int(i.max_scope), max_table=int(i.max_table),
                    assignments=None if i.wide else int(i.assignments_lo) | int(i.assignments_hi) << 64,
                    declared=bool(i.declared), linked=int(i.linked), sources=int(i.sources), data_factors=int(i.data),
                    limbs=int(i.limbs), digest=f"{int(i.digest[0]):016x}{int(i.digest[1]):016x}")

    def factor(self, f):
        i = _FactorInfo()
        _check(_call("tr_factor_graph_factor", self._open(), int(f), C.byref(i)), what="factor")
        return dict(kind=_KIND_NAMES[int(i.kind)], scope=self.factors[f].scope, entries=int(i.entries),
                    nonzero=int(i.nonzero), group=int(i.group), linked=bool(i.linked),
                    denominator=int(i.denom[0]) | int(i.denom[1]) << 64,
                    max_entry=int(i.maxent[0]) | int(i.maxent[1]) << 64,
                    source={0: None, 1: "table", 2: "arrays"}.get(int(i.source_kind)), keyed=bool(i.keyed),
                    source_digest=f"{int(i.source_digest[0]):016x}{int(i.source_digest[1]):016x}"
                    if i.source_kind else None)

    def table(self, f):
        """factor f's table as exact Fractions {(labels...): value} (zeros included), from the native scaled table"""
        fi = self.factor(f)
        scope = np.zeros(max(len(fi["scope"]), 1), np.uint32)
        ent = np.zeros(2 * fi["entries"], np.uint64)
        den = np.zeros(2, np.uint64)
        _check(_call("tr_factor_graph_table", self._open(), int(f), scope.ctypes.data, ent.ctypes.data, fi["entries"],
                     den.ctypes.data), what="factor table")
        D = int(den[0]) | int(den[1]) << 64
        labels = [self.vars[self.pos[v]].labels for v in fi["scope"]]
        out = {}
        import itertools
        for e, key in enumerate(itertools.product(*labels)):
            out[key] = Fraction(int(ent[2 * e]) | int(ent[2 * e + 1]) << 64, D)
        return out

    def _require_fresh(self, what):
        for s in self._scopes:
            if s.source == "table":
                s.require_fresh(what)

    def _vars(self, names, what):
        names = [names] if isinstance(names, str) else list(names)
        for v in names:
            if v not in self.pos:
                raise ValueError(f"{what}: {v!r} is not a variable of the graph")
        if len(set(names)) != len(names):
            raise ValueError(f"{what}: variables must be distinct")
        return names

    def _evidence(self, evidence, what="evidence"):
        """(fixed variable names in graph order, value indices [R, nfixed]) of one evidence dict or a batch"""
        if evidence is None:
            return [], np.zeros((1, 0), np.uint32)
        if isinstance(evidence, dict):
            batch = [evidence]
        else:
            batch = list(evidence)
            if not batch:
                raise ValueError(f"{what}: at least one request")
        keys = set(batch[0])
        for e in batch:
            if set(e) != keys:
                raise ValueError(f"{what}: every request of a batch observes the same variables")
        fixed = [v for v in self.names if v in keys]
        if len(fixed) != len(keys):
            raise ValueError(f"{what}: {sorted(keys - set(fixed))} are not variables of the graph")
        ev = np.zeros((len(batch), len(fixed)), np.uint32)
        for r, e in enumerate(batch):
            for i, v in enumerate(fixed):
                vl = self.vars[self.pos[v]]
                lab = e[v]
                if lab not in vl.index:
                    raise ValueError(f"{what}: {lab!r} is not a declared value of {v!r}")
                ev[r, i] = vl.index[lab]
        return fixed, ev

    # ------------------------------------------------------------------ plans
    def _options(self, devices, order, force, heuristics, dp_limit, rates, elim):
        o = _Options()
        mask = 0
        for d in devices:
            mask |= {"cpu": ON_CPU, "cuda": ON_CUDA}[d]
        o.devices, o.force = mask, -1 if force is None else int(force)
        o.dp_limit = int(dp_limit or 0)
        if heuristics is not None:
            for h in heuristics:
                o.heuristics |= 1 << HEURISTICS[h]
        keepalive = None
        if order is not None:
            order = [self.pos[v] for v in self._vars(order, "order")]
            keepalive = np.array(order or [0], dtype=np.uint32)
            o.norder, o.order = len(order), keepalive.ctypes.data
        if rates is not None:
            reg = _reg()
            o.custom_rates = 1
            for i, dev in enumerate(("cpu", "cuda")):
                base = default_rates()[dev]
                base.update(rates.get(dev, {}))
                for k, comp in enumerate(reg.COST_COMPONENTS):
                    o.rates[i][k] = float(base[comp])
        return o, keepalive

    def plan(self, keep=(), *, evidence=(), mode="sum", requests=1, k=1, devices=("cpu",), order=None, force=None,
             heuristics=None, dp_limit=None, rates=None, budget=None, refuse=True):
        """the costed plan of a query shape: kept variables (sum), the evidence variables (names or a dict), mode
        "sum" / "max" / "kbest", the batch size; candidates on `devices`; order: a given order of the eliminated
        variables (an extra candidate); force: the candidate that must run. A plan with no admissible candidate raises
        (AlgebraBudgetError .plan, .every_order) unless refuse=False."""
        keep = self._vars(keep, "kept variables")
        fixed = list(evidence) if isinstance(evidence, dict) else self._vars(evidence, "evidence variables")
        fixed = [v for v in self.names if v in set(fixed)]
        return Plan._make(self, keep, fixed, mode, requests, k, devices, order, force, heuristics, dp_limit, rates,
                          budget, refuse)

    def query(self, keep=(), *, evidence=(), mode="sum", kind=None, given=0, requests=1, k=1, backend="cpu",
              devices=None, order=None, force=None, heuristics=None, dp_limit=None, rates=None, budget=None):
        """a retained query (its plan and program) run on batches of evidence values: Query.run(evidence)"""
        devices = devices or (("cpu",) if backend == "cpu" else ("cuda",) if backend == "cuda" else ("cpu", "cuda"))
        plan = self.plan(keep, evidence=evidence, mode=mode, requests=requests, k=k, devices=devices, order=order,
                         force=force, heuristics=heuristics, dp_limit=dp_limit, rates=rates, budget=budget)
        if kind is None:
            kind = {"sum": "marginal" if plan.keep else "partition", "max": "max", "kbest": "kbest"}[mode]
        return Query(self, plan, kind, given, k, budget)

    def normalizer(self, *, budget=None):
        """the PARTITION answer without evidence (Z), computed once natively and retained"""
        if self._norm is None:
            q = self.query(budget=budget)
            self._norm = q._run(np.zeros((1, 0), np.uint32), normalizer=None)
        return self._norm

    # ------------------------------------------------------------------ queries
    def _go(self, keep, evidence, mode, kind, given=0, k=1, backend="cpu", devices=None, order=None, force=None,
            heuristics=None, dp_limit=None, rates=None, budget=None):
        self._require_fresh(kind)
        fixed, ev = self._evidence(evidence)
        norm = self.normalizer(budget=budget) if kind == "partition" else None    # P(e) needs Z; the others Z_e
        q = self.query(keep, evidence=fixed, mode=mode, kind=kind, given=given, requests=ev.shape[0], k=k,
                       backend=backend, devices=devices, order=order, force=force, heuristics=heuristics,
                       dp_limit=dp_limit, rates=rates, budget=budget)
        return q._run(ev, normalizer=norm)

    def partition(self, evidence=None, **kw):
        """Z_e (Z without evidence) and P(e) = Z_e / Z, exact (one request)"""
        return self._go((), evidence, "sum", "partition", **kw)[0]

    def likelihood(self, evidence, **kw):
        """a batch of evidence requests (dicts observing the same variables): Z_e and P(e) per request"""
        return self._go((), evidence, "sum", "partition", **kw)

    def marginal(self, variables, evidence=None, **kw):
        """P(q | e) for every assignment q of the variables (in their order), exact, from the factor graph"""
        keep = self._vars(variables, "marginal variables")
        return self._go(keep, evidence, "sum", "marginal", **kw)[0]

    def conditional(self, variables, given, evidence=None, **kw):
        """P(q | r, e) for every q of the variables and r of the given variables (None where P(r | e) = 0)"""
        keep = self._vars(list(self._vars(variables, "variables")) + list(self._vars(given, "given")), "conditional")
        return self._go(keep, evidence, "sum", "conditional", given=len(self._vars(given, "given")), **kw)[0]

    def map(self, evidence=None, **kw):
        """the maximum weight w*, P(x* | e) and EVERY tied maximizer (sorted by variable index)"""
        return self._go((), evidence, "max", "max", **kw)[0]

    def kbest(self, k, evidence=None, **kw):
        """the k assignments of largest weight (weight decreasing, then lexicographic), with how many tie beyond"""
        kw.pop("backend", None)
        return self._go((), evidence, "kbest", "kbest", k=int(k), **kw)[0]

    def count(self, evidence=None, *, method="symcount", budget=None):
        """Z_e by an alternative backend: COUNT-002 variable elimination ("symcount"), COUNT-002 brute force over the
        factor universe ("brute"), COUNT-002 decision diagram of the parameter encoding ("diagram", binary variables),
        TENSOR-002 contraction ("tensor")"""
        if method not in METHODS:
            raise ValueError(f"method is one of {sorted(METHODS)}")
        self._require_fresh("count")
        fixed, ev = self._evidence(evidence)
        if ev.shape[0] != 1:
            raise ValueError("count takes one evidence")
        fx = np.array([self.pos[v] for v in fixed] or [0], dtype=np.uint32)
        vv = np.ascontiguousarray(ev[0] if len(fixed) else np.zeros(1, np.uint32))
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_factor_alternative", self._open(), METHODS[method], len(fixed), fx.ctypes.data, vv.ctypes.data,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        a = _Answer(_made(h, error, refusal, f"count ({method})"))
        try:
            return a.value(0, 0)
        finally:
            a.close()

    def close(self):
        if getattr(self, "_norm", None) is not None:
            self._norm.close()
            self._norm = None
        if getattr(self, "_handle", None):
            _call("tr_factor_graph_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Graph({len(self.names)} variables, {len(self.factors)} factors)"


# ---------------------------------------------------------------------- plans
class Plan:
    """A costed elimination plan (native) with the FND-008 evidence of every candidate."""

    @classmethod
    def _make(cls, g, keep, fixed, mode, requests, k, devices, order, force, heuristics, dp_limit, rates, budget, refuse):
        if mode not in MODES:
            raise ValueError("mode is 'sum', 'max' or 'kbest'")
        self = object.__new__(cls)
        self.graph, self.keep, self.fixed, self.mode = g, tuple(keep), tuple(fixed), mode
        self._keep_ids = np.array([g.pos[v] for v in keep] or [0], dtype=np.uint32)
        self._fixed_ids = np.array([g.pos[v] for v in fixed] or [0], dtype=np.uint32)
        sh = _Shape()
        sh.mode, sh.nkeep, sh.nfixed = MODES[mode], len(keep), len(fixed)
        sh.keep, sh.fixed = self._keep_ids.ctypes.data, self._fixed_ids.ctypes.data
        sh.requests, sh.kbest = int(requests), int(k)
        eliminated = [v for v in g.names if v not in set(keep) | set(fixed)]
        if order is not None and set(order) != set(eliminated):
            raise ValueError(f"order: a permutation of the eliminated variables {eliminated}")
        opts, self._order_keep = g._options(devices, order, force, heuristics, dp_limit, rates, eliminated)
        handle, refusal = vp(), _Refusal()
        rc = _call("tr_factor_plan_create", g._open(), C.byref(sh), C.byref(opts), C.byref(_budget(budget)),
                   C.byref(handle), C.byref(refusal))
        if not handle.value:
            _check(rc, refusal, "factor plan")
            raise MemoryError("factor plan: native allocation failed")
        self._handle = handle.value
        self.code = rc
        if rc and refuse:
            try:
                _check(rc, refusal if rc == -5 else None, "factor plan")
            except Exception as err:
                info = self.info
                err.plan = self
                err.every_order = info["every_order"]
                err.minima = dict(width=info["min_width"], joint=info["min_joint"], support=info["min_support"],
                                  exact=info["bounds_exact"])
                raise
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the plan is closed")
        return self._handle

    @property
    def info(self):
        i = _PlanInfo()
        _check(_call("tr_factor_plan_get_info", self._open(), C.byref(i)), what="plan")
        r = i.refusal
        return dict(candidates=int(i.ncandidates), chosen=None if i.chosen < 0 else int(i.chosen), code=int(i.code),
                    complete=bool(i.complete), every_order=bool(i.every_order), forced=bool(i.forced),
                    bounds_exact=bool(i.bounds_exact), orders=int(i.norders), eliminate=int(i.eliminate),
                    keep=int(i.keep), fixed=int(i.fixed), free_variables=int(i.free_vars),
                    dp_states=int(i.dp_states), evaluated=int(i.evaluated), min_width=int(i.min_width),
                    min_joint=int(i.min_joint), min_support=int(i.min_support), bytes=int(i.bytes),
                    refusal=None if not r.refused else dict(resource=RESOURCES[r.resource], need=int(r.need),
                                                            limit=int(r.limit)))

    def candidate(self, i):
        from . import math_planner as mpl
        c = _Candidate()
        _check(_call("tr_factor_plan_get", self._open(), int(i), C.byref(c)), what="plan candidate")
        m = np.zeros(max(len(self.graph.names), 1), np.uint32)
        n = u64()
        _check(_call("tr_factor_plan_order", self._open(), int(i), m.ctypes.data, len(m), C.byref(n)), what="order")
        d = mpl._candidate(c.reg)
        d.update(index=int(i), heuristic=_HEURISTIC_NAMES.get(int(c.heuristic)),
                 device={1: "cpu", 2: "cuda"}[int(c.device)], steps=int(c.nsteps),
                 order=tuple(self.graph.names[int(v)] for v in m[:int(n.value)]), width=int(c.width),
                 joint=int(c.joint), support=int(c.support), products=int(c.products), arena=int(c.arena),
                 bytes=int(c.bytes), device_bytes=int(c.device_bytes), limbs=int(c.limbs),
                 device_ok=bool(c.device_ok))
        return d

    @property
    def candidates(self):
        return [self.candidate(i) for i in range(self.info["candidates"])]

    @property
    def chosen(self):
        c = self.info["chosen"]
        return None if c is None else self.candidate(c)

    def summary(self):
        c = self.chosen
        info = self.info
        if c is None:
            return dict(chosen=None, every_order=info["every_order"], refusal=info["refusal"])
        return dict(chosen=c["index"], heuristic=c["heuristic"], device=c["device"], order=c["order"],
                    width=c["width"], joint=c["joint"], support=c["support"], products=c["products"],
                    limbs=c["limbs"], total=c["total"], candidates=info["candidates"], complete=info["complete"])

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_factor_plan_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- answers
class _Answer:
    """a native answer handle with exact readouts"""

    def __init__(self, handle):
        self._handle = handle
        i = _AnswerInfo()
        _check(_call("tr_factor_answer_get_info", handle, C.byref(i)), what="answer")
        self.kind = {v: k for k, v in ANSWERS.items()}[int(i.kind)]
        self.requests, self.nvalues, self.cells = int(i.requests), int(i.nvalues), int(i.cells)
        self.nvars, self.limbs, self.backend_code = int(i.nvars), max(int(i.limbs), 1), int(i.backend)
        self._buf = (np.zeros(self.limbs, np.uint64), np.zeros(self.limbs, np.uint64))

    def status(self, r):
        return int(_call("tr_factor_answer_status", self._handle, int(r)))

    def value(self, r, i):
        num, den = self._buf
        nl, dl = u32(), u32()
        rc = _call("tr_factor_answer_value", self._handle, int(r), int(i), num.ctypes.data, den.ctypes.data,
                   self.limbs, C.byref(nl), C.byref(dl))
        _check(rc, what="answer value")
        if rc == 0:
            return None
        n, d = _limbs_int(num, int(nl.value)), _limbs_int(den, int(dl.value))
        return Fraction(n, d) if d != 1 else Fraction(n)

    def column(self, i, r0=0, n=None):
        """value i of every request (from r0, n of them) in one native call: Fractions (None: undefined)"""
        n = self.requests - r0 if n is None else int(n)
        cap = self.limbs
        num, den = np.zeros(max(n * cap, 1), np.uint64), np.zeros(max(n * cap, 1), np.uint64)
        nl, dl = np.zeros(max(n, 1), np.uint32), np.zeros(max(n, 1), np.uint32)
        _check(_call("tr_factor_answer_column", self._handle, int(i), int(r0), n, num.ctypes.data, den.ctypes.data, cap,
                     nl.ctypes.data, dl.ctypes.data), what="answer column")
        out = []
        num, den = num.reshape(-1, cap), den.reshape(-1, cap)
        for k in range(n):
            d = int(dl[k])
            if not d:
                out.append(None)
                continue
            a = int(num[k, 0]) if int(nl[k]) <= 1 else _limbs_int(num[k], int(nl[k]))
            b = int(den[k, 0]) if d == 1 else _limbs_int(den[k], d)
            out.append(Fraction(a, b))
        return out

    def assignments(self, r):
        cnt = u64()
        # the count first (capacity 0 is refused when there are assignments, but the count is written)
        rc = _call("tr_factor_answer_assignments", self._handle, int(r), None, 0, C.byref(cnt))
        n = int(cnt.value)
        out = np.zeros(max(n * self.nvars, 1), np.uint32)
        if n:
            _check(_call("tr_factor_answer_assignments", self._handle, int(r), out.ctypes.data, len(out), C.byref(cnt)),
                   what="assignments")
        elif rc < 0:
            _check(rc, what="assignments")
        return out[:n * self.nvars].reshape(n, self.nvars)

    def tied(self, r):
        t = u64()
        _check(_call("tr_factor_answer_tied", self._handle, int(r), C.byref(t)), what="tied")
        return int(t.value)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_factor_answer_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _ev2d(evidence, nfixed):
    """value indices as a C-contiguous uint32 array [R, nfixed] (R >= 1)"""
    ev = np.ascontiguousarray(evidence, dtype=np.uint32)
    if ev.ndim == 2 and ev.shape[1] == nfixed and ev.shape[0] >= 1:
        return ev
    if nfixed == 0:
        return np.zeros((max(ev.shape[0] if ev.ndim else 1, 1), 0), np.uint32)
    return np.ascontiguousarray(ev.reshape(-1, nfixed))


class Query:
    """A retained query: its plan and program (native), run on batches of evidence values."""

    def __init__(self, g, plan, kind, given, k, budget):
        self.graph, self.plan, self.kind, self.given, self.k, self.budget = g, plan, kind, int(given), int(k), budget
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_factor_prog_create", g._open(), plan._open(), -1, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        self._handle = _made(h, error, refusal, "elimination program")
        idx, dev = i32(), i32()
        _check(_call("tr_factor_prog_candidate", self._handle, C.byref(idx), C.byref(dev)), what="program")
        self.candidate, self.device = int(idx.value), {1: "cpu", 2: "cuda"}[int(dev.value)]
        v = _Program()
        _check(_call("tr_factor_prog_get_view", self._handle, C.byref(v)), what="program")
        self.view = v

    def raw(self, evidence, backend=None):
        """the raw arena of a batch (value indices [R, nfixed]) on the plan's device (or `backend`): (vals, maxv, cnts,
        arg) as the CPU loop or the device kernels wrote them, and the device label"""
        ev = _ev2d(evidence, len(self.plan.fixed))
        R = ev.shape[0]
        v = self.view
        nf = int(v.nfactors)
        bases = np.zeros(max(R * nf, 1), np.uint64)
        _check(_call("tr_factor_prog_bases", self._handle, R, ev.ctypes.data if ev.size else None, bases.ctypes.data),
               what="evidence values")
        backend = backend or self.device
        raw = _Need()
        _check(_call("tr_factor_prog_need", self._handle, R, BY_DEVICE if backend == "cuda" else BY_CPU,
                     C.byref(raw)), what="elimination needs")
        _precheck(self.budget, raw, "elimination")
        L, A = int(v.limbs), int(v.arena)
        mx = int(v.mode) == MODES["max"]
        vals = np.zeros(max(R * A * L, 1), np.uint64)
        maxv = np.zeros(max(R * A * L, 1), np.uint64) if mx else None
        cnts = np.zeros(max(R * A * L, 1), np.uint64) if mx else None
        arg = np.zeros(max(R * A, 1), np.uint32) if mx else None
        ptr = (lambda a: a.ctypes.data if a is not None else None)
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_factor_eliminate", C.byref(v), R, bases.ctypes.data, vals.ctypes.data, ptr(maxv),
                         ptr(cnts), ptr(arg), C.byref(_budget(self.budget)), C.byref(refusal)), refusal, "elimination")
            label = "cpu"
        elif backend == "cuda":
            gpu = _gpu()
            gpu.run("tr_mgpu_factor_eliminate", C.byref(v), R, bases.ctypes.data, vals.ctypes.data, ptr(maxv),
                    ptr(cnts), ptr(arg), C.byref(_budget(self.budget)), C.byref(refusal), refusal=refusal,
                    what="elimination on the device")
            label = "cuda-emulated" if gpu.device()["emulated"] else "cuda"
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return dict(vals=vals, maxv=maxv, cnts=cnts, arg=arg, evidence=ev, backend=label)

    def _run(self, ev, normalizer=None, backend=None):
        R = ev.shape[0]
        norm = normalizer._answer._handle if normalizer is not None else None
        error, refusal = C.c_int32(), _Refusal()
        if self.kind == "kbest":
            h = _call("tr_factor_kbest", self._handle, R, ev.ctypes.data if ev.size else None, self.k, norm,
                      C.byref(_budget(self.budget)), C.byref(refusal), C.byref(error))
            raw = dict(backend="cpu", evidence=ev)
        else:
            raw = self.raw(ev, backend)
            ptr = (lambda a: a.ctypes.data if a is not None else None)
            h = _call("tr_factor_answer_create", self._handle, ANSWERS[self.kind], self.given, R,
                      ev.ctypes.data if ev.size else None, raw["vals"].ctypes.data, ptr(raw["maxv"]), ptr(raw["cnts"]),
                      ptr(raw["arg"]), norm, BY_CPU if raw["backend"] == "cpu" else BY_DEVICE,
                      C.byref(_budget(self.budget)), C.byref(refusal), C.byref(error))
        ans = _Answer(_made(h, error, refusal, f"{self.kind} answer"))
        return Batch(self, ans, raw, normalizer)

    def run(self, evidence, backend=None, normalizer=None):
        """the answers of a batch: evidence dicts (the plan's evidence variables) or value indices [R, nfixed];
        normalizer (default: for partition queries only) adds P(e) = Z_e / Z from the graph's retained Z"""
        if isinstance(evidence, dict) or (isinstance(evidence, (list, tuple)) and evidence and isinstance(evidence[0], dict)):
            fixed, ev = self.graph._evidence(evidence)
            if tuple(fixed) != self.plan.fixed:
                raise ValueError(f"the query observes {self.plan.fixed}")
        else:
            ev = _ev2d(evidence, len(self.plan.fixed))
        self.graph._require_fresh("query")
        if normalizer is None:
            normalizer = self.kind == "partition"
        norm = self.graph.normalizer(budget=self.budget) if normalizer else None
        return self._run(ev, norm, backend)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_factor_prog_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Batch:
    """The answers of a batch of requests (exact, natively computed): batch[r] is request r's Result (read lazily),
    batch.column("Z_e" | "probability") a whole column in one native call. .raw: the arena the answers were adopted
    from (CPU or device), .plan: the plan's summary, .backend."""

    def __init__(self, query, answer, raw, normalizer=None):
        self.query, self._answer, self.raw = query, answer, raw
        self.backend = raw["backend"]
        self.plan = query.plan.summary()
        self._Z = normalizer[0].Z_e if normalizer is not None else None
        self._items = {}

    def __len__(self):
        return self._answer.requests

    def __getitem__(self, r):
        r = int(r)
        if r < 0:
            r += len(self)
        if not 0 <= r < len(self):
            raise IndexError("request index out of range")
        if r not in self._items:
            ev = self.raw["evidence"]
            self._items[r] = Result(self.query.graph, self.query, self._answer, r, ev[r] if ev.size else (), self._Z)
        return self._items[r]

    def __iter__(self):
        return (self[r] for r in range(len(self)))

    def column(self, name):
        """an exact value of every request: "Z_e" or "probability" (P(e); None without the normalizer)"""
        return self._answer.column({"Z_e": 0, "probability": 1}[name])

    def statuses(self):
        return [self._answer.status(r) for r in range(len(self))]

    def equal(self, other):
        """the same raw arena bit for bit (values, maxima, counts, argmax) and the same exact answers"""
        a, b = self.raw, other.raw
        for k in ("vals", "maxv", "cnts", "arg"):
            x, y = a.get(k), b.get(k)
            if (x is None) != (y is None) or (x is not None and not np.array_equal(x, y)):
                return False
        return len(self) == len(other) and all(p._cmp == q._cmp for p, q in zip(self, other))

    def close(self):
        self._answer.close()


class Result:
    """One request's exact answer (Fractions; labels of the declared supports)."""

    def __init__(self, g, query, ans, r, ev, Z=None):
        self.status = ans.status(r)
        self.kind = query.kind
        self.evidence = {v: g.vars[g.pos[v]].labels[int(i)] for v, i in zip(query.plan.fixed, ev)}
        self.Z_e = ans.value(r, 0)
        self.probability = ans.value(r, 1)              # P(e) = Z_e / Z (None without the normalizer)
        self.Z = Z if Z is not None else (self.Z_e if not query.plan.fixed else None)
        keep = query.plan.keep
        self.table = self.weight = self.map_probability = self.count = self.assignments = self.best = self.tied = None
        if self.kind in ("marginal", "conditional"):
            self.variables = keep
            if self.kind == "conditional":
                self.given = keep[len(keep) - query.given:]
                self.variables = keep[:len(keep) - query.given]
        if self.kind in ("marginal", "conditional") and self.status == 0:
            import itertools
            labels = [g.vars[g.pos[v]].labels for v in keep]
            self.table = {key: ans.value(r, 2 + i) for i, key in enumerate(itertools.product(*labels))}
        if self.kind == "max" and self.status == 0:
            self.weight, self.map_probability, self.count = ans.value(r, 2), ans.value(r, 3), ans.value(r, 4)
            self.count = int(self.count) if self.count is not None else None
            rows = ans.assignments(r)
            self.assignments = [self._labels(g, row) for row in rows]
            self.indices = rows
        if self.kind == "kbest" and self.status == 0:
            rows = ans.assignments(r)
            self.best = [(self._labels(g, row), ans.value(r, 2 + 2 * j), ans.value(r, 3 + 2 * j))
                         for j, row in enumerate(rows)]
            self.indices = rows
            self.tied = ans.tied(r)
        self._cmp = (self.status, self.Z_e, self.probability, self.table, self.weight, self.count, self.best,
                     self.tied, tuple(map(tuple, getattr(self, "indices", np.zeros((0, 0), np.uint32)))))

    @staticmethod
    def _labels(g, row):
        return {v: g.vars[i].labels[int(row[i])] for i, v in enumerate(g.names)}

    def require(self):
        if self.status:
            _check(self.status, what=f"{self.kind} request (the evidence has probability zero)")
        return self

    def __repr__(self):
        return f"Result({self.kind}, Z_e={self.Z_e}, P(e)={self.probability})"
