# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""JET-001 / JET-002: exact jets of bounded polynomial circuits, natively (torbles/math_jet.c, math_jet_gpu.cu).

    from fractions import Fraction
    from torbles import math_jet as mj
    b = mj.Builder(params=["a", "c"], data=["x", "y"])
    a, c = b.params; x, y = b.data
    f = (a * x + c) ** 3 - Fraction(1, 2) * y * a
    circuit = b.build(f)                                       # immutable, budgeted
    plan = circuit.plan(order=2, center={"a": 1, "c": Fraction(1, 3)})   # directions: the parameters (U = I)
    jet = plan.at({"x": 3, "y": -2})                           # value, gradient(), hessian(), third(i, j, k), ...
    duals = circuit.plan(order=1, center={"a": 1}, directions=[{"a": 1, "c": 2}, [0, 1]])   # 2 directional derivatives
    jets = plan.evaluate({"x": xs, "y": ys}, backend="cuda")  # one jet per point (JetSet)
    agg = plan.aggregate(scope)                                # one jet per group of a math_bank.Scope: sum over rows
    circuit.aggregate(db["t"], order=1, center=..., group_by=["k"])   # builds the scope (the circuit's data columns)
    agg + other                                                # the sum law, natively (every component kept)
    agg.values("bigq" | "q64" | "gfp" | "interval", convention="taylor" | "derivative", p=...)   # all components

A plan expands g(t) = f(c + U^T t, x) to order d in the m directions t: the stored coefficients are Taylor (Hasse)
coefficients J[alpha] = d^alpha g(0) / alpha! ("taylor"); "derivative" readouts are alpha! J[alpha]. Data columns are
fixed observations (logical integers, value = logical / 10^scale), never differentiated. Everything is computed in
C (fixed-denominator int64 jets with checked 128-bit intermediates; exact 128-bit sums over rows) or in the CUDA
kernel (backend="cuda": "cuda" on a device, "cuda-emulated" in the emulation), with the same states bit for bit.
Python marshals the circuit's nodes and the expansion point, and formats exact results as int / Fraction. Refusals:
OverflowError (a value leaves the exact carrier), AlgebraBudgetError (budgets, before allocation), UnsupportedError
(representation limits), DomainError (a scope or a state of another plan), CarrierError, NoSolutionError,
StaleStateError (the scope's table changed). Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from .math_bank import Scope, _View as _ScopeView
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _Interval, _I128, _Q, _api as _carrier_api, _bigint_value
from .math_errors import RESOURCES, check as _check

__all__ = ["Builder", "Expr", "Circuit", "Plan", "AggregateBuilder", "JetSet", "Jet", "JetValues", "OPS",
           "CONVENTIONS", "CARRIERS", "MAX_DIRECTIONS", "MAX_ORDER", "MAX_PARAMS", "MAX_COLUMNS", "MAX_SUPPORT",
           "Budget"]

OPS = {"const": 1, "param": 2, "data": 3, "add": 4, "sub": 5, "mul": 6, "pow": 7, "scale": 8, "neg": 9}
_OP_NAMES = {v: k for k, v in OPS.items()}
CONVENTIONS = {"taylor": 0, "derivative": 1}
CARRIERS = {"q64": 4, "bigq": 5, "gfp": 6, "interval": 8}
_CARRIER_NAMES = {v: k for k, v in CARRIERS.items()}
MAX_DIRECTIONS, MAX_ORDER, MAX_PARAMS, MAX_COLUMNS, MAX_SUPPORT, MAX_EXPONENT = 64, 16, 64, 32, 65535, 255
_NEED_CPU, _NEED_DEVICE, _NEED_ADOPT = 0, 1, 2
I64 = (-(1 << 63), (1 << 63) - 1)
_M64 = (1 << 64) - 1
P = C.POINTER
vp, i32, u32, i64, u64, u8 = C.c_void_p, C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_uint8


class _Node(C.Structure):
    _native_ = "tr_jet_node"
    _fields_ = [("op", i32), ("a", i32), ("b", i32), ("index", i32), ("value", _Q)]


class _CircuitInfo(C.Structure):
    _native_ = "tr_jet_circuit_info"
    _fields_ = [(n, i32) for n in ("nodes", "nparams", "ncols", "live", "degree", "param_degree")] + [
        ("columns", u32), ("reserved", i32), ("params", u64), ("bytes", u64)]


class _Step(C.Structure):
    _native_ = "tr_jet_step"
    _fields_ = [("op", u8), ("jet", u8), ("pdeg", u8), ("pdeg_a", u8), ("pdeg_b", u8), ("reserved", u8 * 3),
                ("index", i32), ("dst", u32), ("a", u32), ("b", u32), ("t0", u32), ("t1", u32), ("dep", u64),
                ("dep_a", u64), ("dep_b", u64), ("ma", i64), ("mb", i64)]


class _JetView(C.Structure):
    _native_ = "tr_jet_view"
    _fields_ = [(n, u32) for n in ("support", "nsteps", "nregs", "nslots")] + [
        (n, i32) for n in ("order", "directions", "nparams", "ncols")] + [
        ("columns", u32), ("out", u32), ("out_jet", i32), ("out_pdeg", i32), ("denominator", i64), ("pairs", u64),
        ("workspace", u64)] + [(n, vp) for n in ("steps", "leaf", "degree", "mask", "conv_start", "conv_pairs",
                                                 "upto", "scales")] + [("out_dep", u64)]


class _PlanInfo(C.Structure):
    _native_ = "tr_jet_plan_info"
    _fields_ = [(n, u64) for n in ("support", "pairs", "steps", "registers", "slots", "workspace", "bytes",
                                   "multiplications", "additions")] + [("digest", u64 * 2), ("denominator", i64)] + [
        (n, i32) for n in ("order", "directions", "nparams", "ncols", "nodes", "jets", "out_jet", "out_pdeg")] + [
        ("columns", u32), ("reserved", i32), ("out_dep", u64)]


class _States(C.Structure):
    _native_ = "tr_jet_states"
    _fields_ = [("ngroups", u64), ("support", u64), ("rows", vp), ("count", vp), ("sum", vp)]


class _AggInfo(C.Structure):
    _native_ = "tr_jet_agg_info"
    _fields_ = [(n, u64) for n in ("ngroups", "present", "rows", "count", "support", "bytes")] + [
        ("digest", u64 * 2), ("scope_digest", u64 * 2), ("denominator", i64)] + [
        (n, i32) for n in ("order", "directions", "scoped", "reserved")]


class _ValuesInfo(C.Structure):
    _native_ = "tr_jet_values_info"
    _fields_ = [(n, u64) for n in ("njets", "support", "bytes", "p")] + [
        (n, i32) for n in ("carrier", "convention", "order", "directions", "exact", "reserved")]


for _s, _n in ((_Node, 32), (_CircuitInfo, 48), (_Step, 72), (_JetView, 144), (_PlanInfo, 144), (_States, 40),
               (_AggInfo, 104), (_ValuesInfo, 56)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        B, R, E, N = P(_Budget), P(_Refusal), P(i32), P(_Need)
        sig = {
            "tr_jet_circuit_bytes": (u64, [i32]),
            "tr_jet_circuit_need": (i32, [P(_Node), i32, i32, i32, N]),
            "tr_jet_circuit_create": (vp, [P(_Node), i32, i32, i32, B, R, E]),
            "tr_jet_circuit_get_info": (i32, [vp, P(_CircuitInfo)]),
            "tr_jet_circuit_get_nodes": (i32, [vp, P(_Node), i32]),
            "tr_jet_circuit_free": (None, [vp]),
            "tr_jet_plan_create": (vp, [vp, i32, i32, P(_Q), P(_Q), P(i32), B, R, E]),
            "tr_jet_plan_need": (i32, [vp, i32, i32, N]),
            "tr_jet_plan_get_info": (i32, [vp, P(_PlanInfo)]),
            "tr_jet_plan_get_view": (i32, [vp, P(_JetView)]),
            "tr_jet_plan_monomials": (i32, [vp, vp, u64]),
            "tr_jet_plan_steps": (i32, [vp, P(_Step), u64]),
            "tr_jet_plan_denominators": (i32, [vp, P(i64), u64]),
            "tr_jet_plan_equal": (i32, [vp, vp]),
            "tr_jet_plan_free": (None, [vp]),
            "tr_jet_agg_init": (vp, [vp, vp, B, R, E]),
            "tr_jet_agg_accumulate": (i32, [vp, u64, u64]),
            "tr_jet_agg_merge": (i32, [vp, vp]),
            "tr_jet_agg_finish": (vp, [vp, E]),
            "tr_jet_builder_free": (None, [vp]),
            "tr_jet_agg_build": (vp, [vp, vp, B, R, E]),
            "tr_jet_agg_need": (i32, [vp, vp, i32, N]),
            "tr_jet_agg_from_states": (vp, [vp, vp, P(_States), B, R, E]),
            "tr_jet_agg_add": (vp, [vp, vp, B, R, E]),
            "tr_jet_agg_get_info": (i32, [vp, P(_AggInfo)]),
            "tr_jet_agg_get_states": (i32, [vp, P(_States)]),
            "tr_jet_agg_monomials": (i32, [vp, vp, u64]),
            "tr_jet_agg_equal": (i32, [vp, vp]),
            "tr_jet_agg_read": (i32, [vp, u64, u64, i32, P(_I128), P(i64)]),
            "tr_jet_agg_free": (None, [vp]),
            "tr_jet_values_from_agg": (vp, [vp, i32, i32, u64, B, R, E]),
            "tr_jet_values_convert": (vp, [vp, i32, u64, B, R, E]),
            "tr_jet_values_add": (vp, [vp, vp, B, R, E]),
            "tr_jet_values_get_info": (i32, [vp, P(_ValuesInfo)]),
            "tr_jet_values_read_q64": (i32, [vp, P(_Q), u64]),
            "tr_jet_values_read_gfp": (i32, [vp, P(u64), u64]),
            "tr_jet_values_read_interval": (i32, [vp, P(_Interval), u64]),
            "tr_jet_values_bigq": (vp, [vp, u64]),
            "tr_jet_values_equal": (i32, [vp, vp]),
            "tr_jet_values_free": (None, [vp]),
            "tr_math_budget_check": (i32, [B, N, R]),
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
    math_gpu.bind("tr_mgpu_jet_aggregate", C.c_int, [P(_JetView), P(_ScopeView), P(_Budget), P(_States), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return C.byref((budget or Budget()).raw)


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _i64(value, what):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be an exact integer")
    value = int(value)
    if not I64[0] <= value <= I64[1]:
        raise OverflowError(f"{what} does not fit int64")
    return value


def _rational(value, what="a rational"):
    """(numerator, denominator) of an exact int / Fraction / finite Decimal, both fitting int64 (no float)"""
    if isinstance(value, bool):
        raise TypeError(f"{what}: bool is not a number here")
    if isinstance(value, Integral):
        n, d = int(value), 1
    elif isinstance(value, Fraction):
        n, d = value.numerator, value.denominator
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{what}: a finite exact value is required")
        q = Fraction(value)
        n, d = q.numerator, q.denominator
    else:
        raise TypeError(f"{what}: use an int, Fraction or finite Decimal (floats are not exact input)")
    if not (I64[0] <= n <= I64[1] and d <= I64[1]):
        raise OverflowError(f"{what}: the numerator and denominator must fit int64")
    return _Q(n, d)


def _fraction(n, d):
    make = getattr(Fraction, "_from_coprime_ints", None)
    return make(n, d) if make else Fraction(n, d)


def _i128_value(x):
    return (int(x.hi) << 64) | (int(x.lo) & _M64)


# ---------------------------------------------------------------------- circuits
class Expr:
    """A node of a Builder's circuit (operators append nodes; constants are exact int / Fraction / Decimal)."""
    __slots__ = ("builder", "index")

    def __init__(self, builder, index):
        self.builder, self.index = builder, index

    def _other(self, other):
        if isinstance(other, Expr):
            if other.builder is not self.builder:
                raise ValueError("the expressions belong to different circuit builders")
            return other
        return self.builder.constant(other)

    def _is_constant(self, other):
        return not isinstance(other, Expr)

    def __add__(self, other): return self.builder._node("add", self, self._other(other))
    def __radd__(self, other): return self.builder._node("add", self._other(other), self)
    def __sub__(self, other): return self.builder._node("sub", self, self._other(other))
    def __rsub__(self, other): return self.builder._node("sub", self._other(other), self)

    def __mul__(self, other):
        if self._is_constant(other):
            return self.builder._node("scale", self, value=other)
        return self.builder._node("mul", self, self._other(other))

    def __rmul__(self, other):
        if self._is_constant(other):
            return self.builder._node("scale", self, value=other)
        return self.builder._node("mul", self._other(other), self)

    def __neg__(self): return self.builder._node("neg", self)
    def __pos__(self): return self

    def __pow__(self, n):
        if isinstance(n, bool) or not isinstance(n, Integral) or not 0 <= int(n) <= MAX_EXPONENT:
            raise ValueError(f"exponents are integers 0..{MAX_EXPONENT}")
        return self.builder._node("pow", self, index=int(n))

    def scale(self, c):
        """the SCALE node c * self"""
        return self.builder._node("scale", self, value=c)

    def __repr__(self):
        return f"Expr(node {self.index})"


class Builder:
    """Records a circuit's nodes (Python marshals the node list; the native constructor validates and freezes it)."""

    def __init__(self, params=(), data=()):
        self.param_names, self.data_names = tuple(params), tuple(data)
        if len(set(self.param_names + self.data_names)) != len(self.param_names) + len(self.data_names):
            raise ValueError("parameter and data names must be distinct")
        self.nodes = []
        self._cache = {}
        self.params = tuple(self.param(n) for n in self.param_names)
        self.data = tuple(self.data_column(n) for n in self.data_names)

    def _node(self, op, a=None, b=None, index=0, value=None):
        q = _rational(value, "a constant") if value is not None else _Q(0, 1)
        self.nodes.append((OPS[op], -1 if a is None else a.index, -1 if b is None else b.index, int(index),
                           (q.numerator, q.denominator)))
        return Expr(self, len(self.nodes) - 1)

    def param(self, name):
        key = ("param", name)
        if key not in self._cache:
            self._cache[key] = self._node("param", index=self.param_names.index(name))
        return self._cache[key]

    def data_column(self, name):
        key = ("data", name)
        if key not in self._cache:
            self._cache[key] = self._node("data", index=self.data_names.index(name))
        return self._cache[key]

    def constant(self, value):
        return self._node("const", value=value)

    def __getitem__(self, name):
        return self.param(name) if name in self.param_names else self.data_column(name)

    def build(self, output, budget=None):
        """the immutable native circuit whose output is `output` (an Expr of this builder; a constant is allowed)"""
        if not isinstance(output, Expr):
            output = self.constant(output)
        if output.builder is not self:
            raise ValueError("the output belongs to another builder")
        nodes = self.nodes[:output.index + 1]
        return Circuit.from_nodes(nodes, self.param_names, self.data_names, budget=budget)


def _node_array(nodes):
    arr = (_Node * max(len(nodes), 1))()
    for i, (op, a, b, index, value) in enumerate(nodes):
        op = OPS[op] if isinstance(op, str) else int(op)
        q = value if isinstance(value, _Q) else (_rational(value) if not isinstance(value, tuple) else
                                                 _Q(_i64(value[0], "a numerator"), _i64(value[1], "a denominator")))
        arr[i] = _Node(op, int(a), int(b), int(index), q)
    return arr


class Circuit:
    """An immutable bounded polynomial circuit (native tr_jet_circuit): f(params, data)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Builder(...).build(expr) or Circuit.from_nodes(nodes, params, data)")

    @classmethod
    def from_nodes(cls, nodes, params, data=(), budget=None):
        """nodes: (op, a, b, index, value) with op a name of OPS (or its code), a / b operand indices (-1 unused),
        index (PARAM p, DATA c, POW n), value an exact rational (CONST, SCALE) or (numerator, denominator)"""
        params, data = tuple(params), tuple(data)
        arr = _node_array(nodes)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_jet_circuit_create", arr, len(nodes), len(params), len(data), _budget(budget), C.byref(refusal),
                  C.byref(error))
        self = object.__new__(cls)
        self._handle = _made(h, error, refusal, "circuit")
        self.param_names, self.data_names = params, data
        return self

    @staticmethod
    def need(nodes, nparams, ncols):
        arr, out = _node_array(nodes), _Need()
        _check(_call("tr_jet_circuit_need", arr, len(nodes), int(nparams), int(ncols), C.byref(out)), what="circuit needs")
        return _needs_dict(out)

    def _open(self):
        if not self._handle:
            raise ValueError("the circuit is closed")
        return self._handle

    @property
    def info(self):
        out = _CircuitInfo()
        _check(_call("tr_jet_circuit_get_info", self._open(), C.byref(out)), what="circuit")
        return dict(nodes=out.nodes, params=out.nparams, columns=out.ncols, live=out.live, degree=out.degree,
                    param_degree=out.param_degree, reads_columns=[n for i, n in enumerate(self.data_names)
                                                                  if out.columns >> i & 1],
                    reads_params=[n for i, n in enumerate(self.param_names) if out.params >> i & 1], bytes=out.bytes)

    def nodes(self):
        """the stored canonical nodes: (op name, a, b, index, Fraction value)"""
        n = self.info["nodes"]
        arr = (_Node * n)()
        _check(_call("tr_jet_circuit_get_nodes", self._open(), arr, n), what="circuit nodes")
        return [(_OP_NAMES[x.op], x.a, x.b, x.index, _fraction(x.value.numerator, x.value.denominator)) for x in arr]

    # the expansion point
    def _vector(self, v, what):
        names = self.param_names
        if isinstance(v, dict):
            unknown = set(v) - set(names)
            if unknown:
                raise ValueError(f"{what}: unknown parameters {sorted(unknown)}")
            return [v.get(n, 0) for n in names]
        v = list(v)
        if len(v) != len(names):
            raise ValueError(f"{what}: one value per parameter ({len(names)})")
        return v

    def plan(self, order=1, center=None, directions=None, scales=None, budget=None):
        """the jet plan: order d, center c (dict or sequence; default 0), directions (None: the parameters, U = I;
        else m vectors, each a dict {param: rational} or a sequence of nparams rationals), scales (dict or sequence
        of the data columns' decimal scales; default 0)"""
        np_, nc = len(self.param_names), len(self.data_names)
        cen = (_Q * max(np_, 1))()
        for i, x in enumerate(self._vector(center or {}, "center")):
            cen[i] = _rational(x, "a center value")
        if directions is None:
            m, seeds = np_, None
        else:
            dirs = [self._vector(u, "a direction") for u in directions]
            m = len(dirs)
            seeds = (_Q * max(m * np_, 1))()
            for j, u in enumerate(dirs):
                for i, x in enumerate(u):
                    seeds[j * np_ + i] = _rational(x, "a seed")
        vals = [0] * nc
        if scales is not None:
            vals = [scales.get(n, 0) for n in self.data_names] if isinstance(scales, dict) else list(scales)
            if len(vals) != nc:
                raise ValueError(f"scales: one per data column ({nc})")
        sc = (i32 * max(nc, 1))(*[int(v) for v in vals])
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_jet_plan_create", self._open(), int(order), int(m), cen, seeds, sc, _budget(budget),
                  C.byref(refusal), C.byref(error))
        return Plan._wrap(_made(h, error, refusal, "jet plan"), self, [int(v) for v in vals])

    def plan_need(self, order, directions):
        out = _Need()
        _check(_call("tr_jet_plan_need", self._open(), int(order), int(directions), C.byref(out)), what="plan needs")
        return _needs_dict(out)

    # conveniences
    def jet(self, data=None, center=None, order=1, directions=None, scales=None, backend="cpu", budget=None):
        """the jet at one point (data: {column: logical value})"""
        return self.plan(order, center, directions, scales, budget).at(data or {}, backend=backend, budget=budget)

    def dual(self, data=None, center=None, directions=None, scales=None, backend="cpu", budget=None):
        """JET-001: the value and the k directional derivatives (an order-1 jet)"""
        return self.jet(data, center, 1, directions, scales, backend, budget)

    def hvp(self, v, data=None, center=None, scales=None, backend="cpu", budget=None):
        """the Hessian-vector product H v at the center: the mixed coefficients J[e_i + e_v] of the order-2 jet in the
        directions [the parameters; v] (no factorial: alpha! = 1)"""
        np_ = len(self.param_names)
        dirs = [[1 if j == i else 0 for j in range(np_)] for i in range(np_)] + [self._vector(v, "v")]
        jet = self.jet(data, center, 2, dirs, scales, backend, budget)
        return [jet.coefficient(_unit(np_ + 1, i, np_)) for i in range(np_)]

    def aggregate(self, source, center=None, order=1, directions=None, *, rows=None, group_by=(), backend="cpu",
                  budget=None):
        """the jets of sum_r f(theta, x_r) per group: source a math_bank.Scope (its columns the circuit's data
        columns) or an engine table (the scope is built with the circuit's data columns, rows= and group_by=)"""
        scope = source if isinstance(source, Scope) else Scope.from_table(source, list(self.data_names), rows=rows,
                                                                          group_by=group_by, budget=budget)
        if list(scope.names) != list(self.data_names):
            raise ValueError(f"the scope's columns {list(scope.names)} are not the circuit's data {list(self.data_names)}")
        scales = [scope.column(n)["scale"] for n in self.data_names]
        return self.plan(order, center, directions, scales, budget).aggregate(scope, backend=backend, budget=budget)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_jet_circuit_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Circuit(closed)"
        i = self.info
        return f"Circuit(nodes={i['nodes']}, params={list(self.param_names)}, data={list(self.data_names)}, degree={i['degree']})"


def _unit(m, *ones):
    e = [0] * m
    for i in ones:
        e[i] += 1
    return tuple(e)


# ---------------------------------------------------------------------- plans
class Plan:
    """An immutable compiled jet plan: circuit, order, directions, center, seeds and column scales (native)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Circuit.plan(...)")

    @classmethod
    def _wrap(cls, handle, circuit, scales):
        self = object.__new__(cls)
        self._handle, self.circuit, self.scales = handle, circuit, tuple(scales)
        self.param_names, self.data_names = circuit.param_names, circuit.data_names
        info = self._raw_info()
        m, S = info.directions, int(info.support)
        exps = np.zeros(max(m * S, 1), dtype=np.uint8)
        _check(_call("tr_jet_plan_monomials", handle, exps.ctypes.data, exps.size), what="plan monomials")
        self.monomials = tuple(tuple(int(k) for k in exps[i * m:(i + 1) * m]) for i in range(S)) if m else ((),)
        self.index = {e: i for i, e in enumerate(self.monomials)}
        self.order, self.directions, self.support = info.order, m, S
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the plan is closed")
        return self._handle

    def _raw_info(self):
        out = _PlanInfo()
        _check(_call("tr_jet_plan_get_info", self._open(), C.byref(out)), what="plan")
        return out

    @property
    def info(self):
        i = self._raw_info()
        return dict(order=i.order, directions=i.directions, support=int(i.support), pairs=int(i.pairs),
                    steps=int(i.steps), registers=int(i.registers), slots=int(i.slots), jets=i.jets,
                    workspace_words=int(i.workspace), bytes=int(i.bytes), multiplications=int(i.multiplications),
                    additions=int(i.additions), denominator=int(i.denominator), output_is_jet=bool(i.out_jet),
                    output_param_degree=i.out_pdeg, output_directions=[j for j in range(i.directions)
                                                                         if i.out_dep >> j & 1])

    def view(self):
        v = _JetView()
        _check(_call("tr_jet_plan_get_view", self._open(), C.byref(v)), what="plan view")
        return v

    def program(self):
        """the compiled steps (op name, result/operand registers or slots, direction masks, degrees, multipliers)"""
        n = int(self._raw_info().steps)
        arr = (_Step * max(n, 1))()
        _check(_call("tr_jet_plan_steps", self._open(), arr, n), what="plan steps")
        return [dict(op=_OP_NAMES[s.op], jet=s.jet, dst=s.dst, a=s.a, b=s.b, index=s.index, dep=s.dep, pdeg=s.pdeg,
                     ma=s.ma, mb=s.mb) for s in arr[:n]]

    def denominators(self):
        """D_v of every circuit node (0 for a node that is not evaluated)"""
        n = int(self._raw_info().nodes)                 # the plan is independent of the circuit object
        arr = (i64 * n)()
        _check(_call("tr_jet_plan_denominators", self._open(), arr, n), what="plan denominators")
        return [int(x) for x in arr]

    def equal(self, other):
        return bool(_check(_call("tr_jet_plan_equal", self._open(), other._open()), what="plan comparison"))

    # evaluation
    def need(self, scope, backend="cpu"):
        """the needs an aggregate over scope checks before allocating (backend "cuda": the larger of its device copies
        and the adopted aggregate, per resource)"""
        if backend == "cpu":
            return _needs_dict(self._need(scope, _NEED_CPU))
        if backend == "cuda":
            return _needs_dict(self._cuda_need(scope))
        raise ValueError("backend is 'cpu' or 'cuda'")

    def _need(self, scope, target):
        out = _Need()
        _check(_call("tr_jet_agg_need", self._open(), scope._handle, target, C.byref(out)), what="aggregate needs")
        return out

    def _cuda_need(self, scope):
        dev, adopt, both = self._need(scope, _NEED_DEVICE), self._need(scope, _NEED_ADOPT), _Need()
        for i in range(len(RESOURCES)):
            both.need[i] = max(dev.need[i], adopt.need[i])
        return both

    def aggregate(self, scope, backend="cpu", budget=None):
        """the jets of sum_r f(theta, x_r) over the selected valid rows of each group of scope (a math_bank.Scope
        whose columns are the circuit's data columns with this plan's scales)"""
        if scope.source == "table":
            scope.require_fresh("jet aggregate")
        error, refusal = C.c_int32(), _Refusal()
        if backend == "cpu":
            h = _call("tr_jet_agg_build", self._open(), scope._handle, _budget(budget), C.byref(refusal), C.byref(error))
            return JetSet._wrap(_made(h, error, refusal, "jet aggregate"), self, scope, "cpu", "native CPU")
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        gpu = _gpu()
        _check(_call("tr_math_budget_check", _budget(budget), C.byref(self._cuda_need(scope)), C.byref(refusal)),
               refusal, "jet aggregate on the device")
        G, S = scope.groups, self.support
        rows, count = np.zeros(G, dtype=np.uint64), np.zeros(G, dtype=np.uint64)
        sums = np.zeros(2 * G * S, dtype=np.int64)
        st = _States(G, S, rows.ctypes.data, count.ctypes.data, sums.ctypes.data)
        view, sview = self.view(), scope.view()
        gpu.run("tr_mgpu_jet_aggregate", C.byref(view), C.byref(sview), _budget(budget), C.byref(st), C.byref(refusal),
                refusal=refusal, what="jet aggregate on the device")
        h = _call("tr_jet_agg_from_states", self._open(), scope._handle, C.byref(st), _budget(budget),
                  C.byref(refusal), C.byref(error))
        d = gpu.device()
        return JetSet._wrap(_made(h, error, refusal, "jet aggregate states"), self, scope,
                            "cuda-emulated" if d["emulated"] else "cuda", d["name"])

    def builder(self, scope, budget=None):
        return AggregateBuilder(self, scope, budget)

    def points_scope(self, points, valid=None, n=None, budget=None):
        """the array scope with one group (and one row) per point: points {column: logical values}; n: the number
        of points of a circuit without data columns"""
        names = list(self.data_names)
        unknown = set(points) - set(names)
        if unknown:
            raise ValueError(f"points: unknown columns {sorted(unknown)}")
        cols = {}
        for name in names:
            if name not in points:
                raise ValueError(f"points: the column {name!r} is missing")
            cols[name] = np.asarray(points[name])
            if n is not None and len(cols[name]) != n:
                raise ValueError("points: every column has one value per point")
            n = len(cols[name])
        n = 1 if n is None else int(n)
        if n < 1:
            raise ValueError("points: at least one point")
        return Scope.from_arrays(cols, valid=valid, scales=dict(zip(names, self.scales)),
                                 groups=np.arange(n, dtype=np.uint32), ngroups=n, budget=budget)

    def evaluate(self, points, valid=None, backend="cpu", n=None, budget=None):
        """one jet per point (a JetSet whose group g is point g); points {column: logical values}; valid {column:
        mask}: a point with a NULL value has count 0 and the zero jet"""
        return self.aggregate(self.points_scope(points, valid, n, budget), backend=backend, budget=budget)

    def at(self, point=None, backend="cpu", budget=None):
        """the jet at one point {column: logical value}"""
        point = point or {}
        missing = [name for name in self.data_names if name not in point]
        if missing:
            raise ValueError(f"the point has no value for {missing}")
        return self.evaluate({name: [point[name]] for name in self.data_names}, backend=backend, n=1,
                             budget=budget).jet(0)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_jet_plan_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return "Plan(closed)" if not self._handle else f"Plan(order={self.order}, directions={self.directions}, support={self.support})"


class AggregateBuilder:
    """Chunked accumulation: accumulate(begin, end) over disjoint physical row ranges, merge(builder of the same
    scope and plan), finish() once every row is covered. One owner (not thread-safe)."""

    def __init__(self, plan, scope, budget=None):
        if scope.source == "table":
            scope.require_fresh("jet aggregate builder")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_jet_agg_init", plan._open(), scope._handle, _budget(budget), C.byref(refusal), C.byref(error))
        self._handle = _made(h, error, refusal, "jet aggregate builder")
        self.plan, self.scope = plan, scope            # the native builder reads both: keep them alive

    def _open(self):
        if not self._handle:
            raise ValueError("the builder is finished or closed")
        return self._handle

    def accumulate(self, begin, end):
        _check(_call("tr_jet_agg_accumulate", self._open(), int(begin), int(end)),
               what="accumulate (rows within the scope, each range once)")
        return self

    def merge(self, other):
        _check(_call("tr_jet_agg_merge", self._open(), other._open()),
               what="merge (same scope and plan; disjoint rows)")
        return self

    def finish(self):
        error = C.c_int32()
        h = _call("tr_jet_agg_finish", self._open(), C.byref(error))
        _check(error.value, what="finish (every row covered once)")
        out = JetSet._wrap(h, self.plan, self.scope, "cpu", "native CPU")
        self.close()
        return out

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_jet_builder_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- aggregates and jets
class JetSet:
    """Immutable exact jets of one plan, one per group (native tr_jet_agg): rows, count and the jet's numerators."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Plan.aggregate / Plan.evaluate / AggregateBuilder.finish")

    @classmethod
    def _wrap(cls, handle, plan, scope, backend, device):
        self = object.__new__(cls)
        self._handle, self.plan, self.scope, self.backend, self.device = handle, plan, scope, backend, device
        info = self._raw_info()
        self.groups, self.support = int(info.ngroups), int(info.support)
        self.order, self.directions = info.order, info.directions
        m, S = info.directions, self.support
        exps = np.zeros(max(m * S, 1), dtype=np.uint8)
        _check(_call("tr_jet_agg_monomials", handle, exps.ctypes.data, exps.size), what="jet monomials")
        self.monomials = tuple(tuple(int(k) for k in exps[i * m:(i + 1) * m]) for i in range(S)) if m else ((),)
        self.index = {e: i for i, e in enumerate(self.monomials)}
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the jet set is closed")
        return self._handle

    def _raw_info(self):
        out = _AggInfo()
        _check(_call("tr_jet_agg_get_info", self._open(), C.byref(out)), what="jet set")
        return out

    @property
    def info(self):
        i = self._raw_info()
        return dict(groups=int(i.ngroups), present=int(i.present), rows=int(i.rows), count=int(i.count),
                    support=int(i.support), bytes=int(i.bytes), denominator=int(i.denominator), order=i.order,
                    directions=i.directions, scoped=bool(i.scoped), backend=self.backend, device=self.device)

    def arrays(self):
        """copies of the native states: rows uint64[G], count uint64[G], sum_lo / sum_hi int64[G, S] (numerators over
        info["denominator"])"""
        st = _States()
        _check(_call("tr_jet_agg_get_states", self._open(), C.byref(st)), what="jet states")
        G, S = int(st.ngroups), int(st.support)
        rows = np.ctypeslib.as_array((C.c_uint64 * G).from_address(st.rows)).copy()
        count = np.ctypeslib.as_array((C.c_uint64 * G).from_address(st.count)).copy()
        words = np.ctypeslib.as_array((C.c_int64 * (2 * G * S)).from_address(st.sum)).reshape(G, S, 2).copy()
        return rows, count, words[:, :, 0], words[:, :, 1]

    def _read(self, first, n, convention):
        S = self.support
        num, den = (_I128 * max(n * S, 1))(), (i64 * max(n * S, 1))()
        _check(_call("tr_jet_agg_read", self._open(), int(first), int(n), CONVENTIONS[convention], num, den),
               what="jet readout")
        return [_fraction(_i128_value(num[i]), int(den[i])) for i in range(n * S)]

    def coefficients(self, convention="taylor"):
        """every group's components as exact Fractions: [[S values] per group]"""
        flat, S = self._read(0, self.groups, convention), self.support
        return [flat[g * S:(g + 1) * S] for g in range(self.groups)]

    def jet(self, g):
        g = int(g)
        if not 0 <= g < self.groups:
            raise ValueError("group id out of range")
        rows, count, _, _ = self._state_row(g)
        return Jet(self, g, self._read(g, 1, "taylor"), self._read(g, 1, "derivative"), rows, count)

    def _state_row(self, g):
        st = _States()
        _check(_call("tr_jet_agg_get_states", self._open(), C.byref(st)), what="jet states")
        rows = int(C.c_uint64.from_address(st.rows + 8 * g).value)
        count = int(C.c_uint64.from_address(st.count + 8 * g).value)
        return rows, count, None, None

    def jets(self):
        taylor, deriv = self._read(0, self.groups, "taylor"), self._read(0, self.groups, "derivative")
        rows, count, _, _ = self.arrays()
        S = self.support
        return [Jet(self, g, taylor[g * S:(g + 1) * S], deriv[g * S:(g + 1) * S], int(rows[g]), int(count[g]))
                for g in range(self.groups)]

    def values(self, carrier="bigq", convention="taylor", p=None, budget=None):
        """every component in a carrier (native, all or nothing): "q64" (narrowing: OverflowError), "bigq"
        (lossless), "gfp" (reduction mod the prime p: NoSolutionError when p divides the denominator), "interval"
        (the tight enclosure: approximate)"""
        if carrier not in CARRIERS:
            raise ValueError(f"carrier is one of {', '.join(CARRIERS)}")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_jet_values_from_agg", self._open(), CARRIERS[carrier], CONVENTIONS[convention], int(p or 0),
                  _budget(budget), C.byref(refusal), C.byref(error))
        return JetValues._wrap(_made(h, error, refusal, f"jet values ({carrier})"), self.monomials)

    def add(self, other, budget=None):
        """the sum law: the aggregate of the bag union of both row sets (same plan and group count)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_jet_agg_add", self._open(), other._open(), _budget(budget), C.byref(refusal), C.byref(error))
        return JetSet._wrap(_made(h, error, refusal, "jet aggregate sum"), self.plan, None, "cpu", "native CPU")

    __add__ = add

    def equal(self, other):
        return bool(_check(_call("tr_jet_agg_equal", self._open(), other._open()), what="jet set comparison"))

    def __eq__(self, other):
        return isinstance(other, JetSet) and self.equal(other)

    __hash__ = None

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_jet_agg_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return "JetSet(closed)" if not self._handle else (
            f"JetSet(groups={self.groups}, order={self.order}, directions={self.directions}, {self.backend})")


class Jet:
    """One exact jet (formatted): Taylor coefficients J[alpha] and derivatives alpha! J[alpha], read natively."""
    __slots__ = ("taylor", "derivatives", "monomials", "index", "rows", "count", "group", "order", "directions")

    def __init__(self, owner, group, taylor, derivatives, rows, count):
        self.taylor, self.derivatives = list(taylor), list(derivatives)
        self.monomials, self.index = owner.monomials, owner.index
        self.rows, self.count, self.group = rows, count, group
        self.order, self.directions = owner.order, owner.directions

    @property
    def value(self):
        return self.taylor[0]

    @property
    def present(self):
        return self.rows > 0

    def _at(self, alpha):
        alpha = tuple(int(k) for k in alpha)
        if alpha not in self.index:
            raise ValueError(f"{alpha} is not a monomial of this jet (order {self.order}, {self.directions} directions)")
        return self.index[alpha]

    def coefficient(self, alpha):
        """the Taylor (Hasse) coefficient J[alpha]"""
        return self.taylor[self._at(alpha)]

    def derivative(self, alpha):
        """the derivative d^alpha g(0) = alpha! J[alpha]"""
        return self.derivatives[self._at(alpha)]

    def coefficients(self, convention="taylor"):
        vals = self.taylor if convention == "taylor" else self.derivatives
        return dict(zip(self.monomials, vals))

    def gradient(self):
        """the first derivatives along the directions (the gradient when the directions are the parameters)"""
        return [self.derivatives[1 + j] for j in range(self.directions)] if self.order >= 1 else []

    directional = gradient

    def hessian(self):
        """the matrix of second derivatives along the directions (derivative convention)"""
        m = self.directions
        return [[self.derivative(_unit(m, i, j)) for j in range(m)] for i in range(m)]

    def third(self, i, j, k):
        return self.derivative(_unit(self.directions, i, j, k))

    def __repr__(self):
        return f"Jet(value={self.value}, order={self.order}, directions={self.directions})"


class JetValues:
    """Every component of a JetSet in one carrier (native tr_jet_values, immutable)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use JetSet.values(...) or JetValues.convert(...)")

    @classmethod
    def _wrap(cls, handle, monomials):
        self = object.__new__(cls)
        self._handle, self.monomials = handle, monomials
        info = self._raw_info()
        self.carrier, self.convention = _CARRIER_NAMES[info.carrier], {v: k for k, v in CONVENTIONS.items()}[info.convention]
        self.p = int(info.p) or None
        self.exact = bool(info.exact)
        self.approximate = not self.exact
        self.jets, self.support = int(info.njets), int(info.support)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the values are closed")
        return self._handle

    def _raw_info(self):
        out = _ValuesInfo()
        _check(_call("tr_jet_values_get_info", self._open(), C.byref(out)), what="jet values")
        return out

    @property
    def info(self):
        i = self._raw_info()
        return dict(carrier=self.carrier, convention=self.convention, p=self.p, exact=self.exact, jets=int(i.njets),
                    support=int(i.support), bytes=int(i.bytes))

    def components(self):
        """[[S components] per jet]: Fraction (q64, bigq), int residues in [0, p) (gfp), (lo, hi) enclosures with
        lo <= x <= hi (interval: approximate, certified)"""
        n, S = self.jets * self.support, self.support
        h = self._open()
        if self.carrier == "q64":
            arr = (_Q * max(n, 1))()
            _check(_call("tr_jet_values_read_q64", h, arr, n), what="q64 values")
            flat = [_fraction(q.numerator, q.denominator) for q in arr[:n]]
        elif self.carrier == "gfp":
            arr = (u64 * max(n, 1))()
            _check(_call("tr_jet_values_read_gfp", h, arr, n), what="gfp values")
            flat = [int(x) for x in arr[:n]]
        elif self.carrier == "interval":
            arr = (_Interval * max(n, 1))()
            _check(_call("tr_jet_values_read_interval", h, arr, n), what="interval values")
            flat = [(float(x.lo), float(x.hi)) for x in arr[:n]]
        else:                                         # borrowed big rationals, read natively (canonical)
            capi, flat = _carrier_api(), []
            for k in range(n):
                q = _call("tr_jet_values_bigq", h, k)
                flat.append(_fraction(_bigint_value(capi["bigq_numerator"](q)), _bigint_value(capi["bigq_denominator"](q))))
        return [flat[j * S:(j + 1) * S] for j in range(self.jets)]

    def convert(self, carrier, p=None, budget=None):
        """the explicit conversion of every component (math_carriers' matrix): q64 -> bigq lossless, bigq -> q64
        narrowing, -> gfp reduction, -> interval enclosing; from gfp / interval to an exact carrier: CarrierError"""
        if carrier not in CARRIERS:
            raise ValueError(f"carrier is one of {', '.join(CARRIERS)}")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_jet_values_convert", self._open(), CARRIERS[carrier], int(p or 0), _budget(budget),
                  C.byref(refusal), C.byref(error))
        return JetValues._wrap(_made(h, error, refusal, f"jet values {self.carrier} -> {carrier}"), self.monomials)

    def add(self, other, budget=None):
        """the sum law in this carrier (same plan, convention, carrier and field)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_jet_values_add", self._open(), other._open(), _budget(budget), C.byref(refusal), C.byref(error))
        return JetValues._wrap(_made(h, error, refusal, "jet values sum"), self.monomials)

    __add__ = add

    def equal(self, other):
        return bool(_check(_call("tr_jet_values_equal", self._open(), other._open()), what="jet values comparison"))

    def __eq__(self, other):
        return isinstance(other, JetValues) and self.equal(other)

    __hash__ = None

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_jet_values_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return "JetValues(closed)" if not self._handle else f"JetValues({self.carrier}, jets={self.jets}, support={self.support})"
