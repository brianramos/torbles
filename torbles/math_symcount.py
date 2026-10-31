# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""COUNT-002: symbolic counting over a declared full assignment domain, natively on the CPU or with the CUDA kernels
(torbles/math_symcount.c, math_symcount_gpu.cu).

    from torbles import math_bool as mb, math_symcount as ms
    x = mb.variables(4)
    f = (x[0] & x[1]) | (x[2] ^ x[3])                     # a BOOL-001 predicate (UNKNOWN constants allowed)
    d = ms.compile(f, k=4, order="dfs")                    # "natural", "dfs" or a permutation; budget circuit / iterations
    c = d.count()                                          # Count over the 2^4 assignments: true, false, unknown, total
    c = d.count(weights={0: (Fraction(1, 3), Fraction(2, 3)), 2: (2, 1)}, backend="cuda")   # w(x=1), w(x=0)
    d.witnesses(5); d.witnesses(5, value="false"); d.info; d.order
    ms.brute(f, k=4, weights=w, backend="cuda")            # every assignment enumerated (verification)
    ms.search(f, k=4, max_evaluations=100)                 # exact, or certified lo <= count <= hi (incomplete)
    fs = ms.Factors([2, 3, 2], [((0, 1), [[1, 0, 2], [Fraction(1, 2), 1, 1]]), ((1, 2), table)])
    fs.count(order="min-fill", weighted=True, witnesses=3) # variable elimination within a width budget (tensor)
    fs.brute(); fs.width("min-fill"); fs.need(...)

Every count is over the DECLARED assignment universe (2^k assignments of k variables, or prod d_i assignments of
finite-domain variables), never over stored rows: COUNT-001 (math_count) counts observed rows. Results are exact
(int or Fraction, reduced natively); an incomplete search gives (lo, hi) bounds that are certified, never an
extrapolation. Budgets raise AlgebraBudgetError before anything is allocated: compilation past its node or step
budget, an elimination wider than its tensor budget (with the exact need). Nothing is computed in Python. Opt-in:
`import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from .math_bool import Expr, Program, _ProgramView
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["compile", "Diagram", "Count", "brute", "search", "Factors", "ORDERS", "METHODS"]

ORDERS = {"natural": 0, "given": 1, "dfs": 2, "min-fill": 3, "min-degree": 4}
METHODS = {1: "decision diagram", 2: "brute force", 3: "bounded search", 4: "variable elimination",
           5: "brute force over factors"}
STATUS = {1: "exact", 2: "incomplete"}
_BY_CPU, _BY_DEVICE = 1, 2
_NEED_CPU, _NEED_DEVICE = 0, 1
_WEIGHTED, _INDICATOR = 1, 2
P = C.POINTER
u32, i32, u64, i64, vp = C.c_uint32, C.c_int32, C.c_uint64, C.c_int64, C.c_void_p


class _DDNode(C.Structure):
    _native_ = "tr_symc_node"
    _fields_ = [("level", u32), ("lo", u32), ("hi", u32), ("reserved", u32)]


class _Weight(C.Structure):
    _native_ = "tr_symc_weight"
    _fields_ = [("num1", i64), ("den1", i64), ("num0", i64), ("den0", i64)]


class _DDView(C.Structure):
    _native_ = "tr_symc_dd_view"
    _fields_ = [("node", vp), ("roots", vp), ("order", vp), ("first", vp), ("count", vp), ("nnodes", u64),
                ("nlevels", u32), ("nroots", u32), ("nvars", u32), ("three_valued", u32)]


class _DDInfo(C.Structure):
    _native_ = "tr_symc_dd_info"
    _fields_ = [(n, u64) for n in ("nnodes", "peak_nodes", "steps", "bytes", "width")] + [
        (n, u32) for n in ("nlevels", "nroots", "nvars", "three_valued")]


class _AnswerInfo(C.Structure):
    _native_ = "tr_symc_answer_info"
    _fields_ = [(n, i32) for n in ("method", "status", "weighted", "three_valued", "backend", "nvalues")] + [
        ("limbs", u32), ("universe_vars", u32)] + [
        (n, u64) for n in ("witnesses", "witness_width", "nodes", "width", "evaluations", "open")]


class _Factor(C.Structure):
    _native_ = "tr_symc_factor"
    _fields_ = [("scope_at", u64), ("table_at", u64), ("nscope", u32), ("reserved", u32)]


class _FactorsInfo(C.Structure):
    _native_ = "tr_symc_factors_info"
    _fields_ = [(n, u64) for n in ("entries", "bytes", "assignments_lo", "assignments_hi")] + [
        (n, u32) for n in ("nvars", "nfactors", "max_scope", "assignments_wide")]


for _s, _n in ((_DDNode, 16), (_Weight, 32), (_DDView, 64), (_DDInfo, 56), (_AnswerInfo, 80), (_Factor, 24),
               (_FactorsInfo, 48)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        pv = P(_ProgramView)
        sig = {
            "tr_symc_compile": (vp, [pv, i32, vp, B, R, e]),
            "tr_symc_program_order": (i32, [pv, i32, vp, vp]),
            "tr_symc_dd_get_view": (i32, [vp, P(_DDView)]),
            "tr_symc_dd_get_info": (i32, [vp, P(_DDInfo)]),
            "tr_symc_dd_free": (None, [vp]),
            "tr_symc_dd_prepare": (i32, [P(_DDView), vp, vp, P(u32)]),
            "tr_symc_dd_count_need": (i32, [vp, vp, i32, P(_Need)]),
            "tr_symc_dd_levels": (i32, [P(_DDView), vp, u32, vp, B, R]),
            "tr_symc_dd_adopt": (vp, [vp, vp, u32, vp, i32, B, R, e]),
            "tr_symc_dd_count": (vp, [vp, vp, B, R, e]),
            "tr_symc_dd_witnesses": (i32, [vp, u32, u64, vp, P(u64), B, R]),
            "tr_symc_brute_prepare": (i32, [pv, vp, vp, P(u32)]),
            "tr_symc_brute_need": (i32, [pv, vp, i32, P(_Need)]),
            "tr_symc_brute_raw": (i32, [pv, vp, u32, vp, vp, B, R]),
            "tr_symc_brute_adopt": (vp, [pv, vp, u32, vp, vp, i32, B, R, e]),
            "tr_symc_brute": (vp, [pv, vp, B, R, e]),
            "tr_symc_search": (vp, [pv, i32, vp, vp, u64, B, R, e]),
            "tr_symc_factors_create": (vp, [u32, vp, u32, vp, vp, vp, vp, u64, B, R, e]),
            "tr_symc_factors_get_info": (i32, [vp, P(_FactorsInfo)]),
            "tr_symc_factors_order": (i32, [vp, i32, vp, vp, P(u64), B, R]),
            "tr_symc_factors_need": (i32, [vp, i32, vp, i32, u64, P(_Need)]),
            "tr_symc_factors_count": (vp, [vp, i32, vp, i32, u64, B, R, e]),
            "tr_symc_factors_brute": (vp, [vp, i32, u64, B, R, e]),
            "tr_symc_factors_free": (None, [vp]),
            "tr_symc_answer_get_info": (i32, [vp, P(_AnswerInfo)]),
            "tr_symc_answer_value": (i32, [vp, u32, vp, vp, u32, P(u32), P(u32)]),
            "tr_symc_answer_witnesses": (i32, [vp, vp, u64]),
            "tr_symc_answer_order": (i32, [vp, vp, u64]),
            "tr_symc_answer_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_symcount_levels", C.c_int, [P(_DDView), vp, u32, vp, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_symcount_brute", C.c_int, [P(_ProgramView), vp, u32, vp, vp, P(_Budget), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _rational(x, what):
    """an exact nonnegative rational with int64 numerator and denominator (floats are refused)"""
    if isinstance(x, bool) or not isinstance(x, (Integral, Rational)):
        raise TypeError(f"{what} must be an exact int or Fraction, not {type(x).__name__}")
    q = Fraction(x)
    if q < 0:
        raise ValueError(f"{what} must be nonnegative")
    if q.numerator >= (1 << 63) or q.denominator >= (1 << 63):
        raise OverflowError(f"{what}: numerator and denominator must fit int64")
    return q


def _program(f, k):
    """a math_bool.Program from a Program or an Expr (k required for an Expr)"""
    if isinstance(f, Program):
        if k is not None and int(k) != f.k:
            raise ValueError(f"the program has k = {f.k}, not {k}")
        return f
    if isinstance(f, (Expr, bool)) or f is None:
        if k is None:
            raise ValueError("k (the number of variables of the universe) is required with an expression")
        return Program(f, int(k))
    raise TypeError("a math_bool.Program or a math_bool.Expr")


def _order(order, n):
    """(kind, uint32 array or None)"""
    if order is None or (isinstance(order, str) and order == "natural"):
        return ORDERS["natural"], None
    if isinstance(order, str):
        if order not in ORDERS or order == "given":
            raise ValueError(f"order is one of {sorted(k for k in ORDERS if k != 'given')} or a permutation")
        return ORDERS[order], None
    arr = np.array([int(v) for v in order], dtype=np.uint32)
    if len(arr) != n:
        raise ValueError(f"the order is a permutation of the {n} variables")
    return ORDERS["given"], arr


def _weights(weights, n):
    """an array of n native weights (w(x=1), w(x=0)) or None for unit weights (a native array passes through)"""
    if weights is None:
        return None
    if isinstance(weights, C.Array) and weights._type_ is _Weight:
        if len(weights) < n:
            raise ValueError("native weights: one per variable")
        return weights
    items = weights.items() if isinstance(weights, dict) else enumerate(weights)
    arr = (_Weight * max(n, 1))()
    for i in range(n):
        arr[i] = _Weight(1, 1, 1, 1)
    for v, pair in items:
        v = int(v)
        if not 0 <= v < n:
            raise ValueError(f"weights: variable {v} is outside 0..{n - 1}")
        w1, w0 = pair
        q1, q0 = _rational(w1, f"w(x{v}=1)"), _rational(w0, f"w(x{v}=0)")
        arr[v] = _Weight(q1.numerator, q1.denominator, q0.numerator, q0.denominator)
    return arr


def _limbs_int(a, n):
    return sum(int(a[i]) << (64 * i) for i in range(n))


def _exact(num, den):
    return num if den == 1 else Fraction(num, den)


class Count:
    """A count over the declared ASSIGNMENT UNIVERSE (not observed rows).

    domain: "universe"; universe: the variables counted; assignments: their number of assignments (2^k, or prod d_i);
    method: "decision diagram", "brute force", "bounded search", "variable elimination" or "brute force over factors";
    status: "exact" or "incomplete"; weighted: False for model counts (numbers of assignments), True for weighted
    counts (sums of products of literal weights or factor entries). Programs: true, false, unknown, total (exact; for
    an incomplete search: true, false, unknown are certified (lo, hi) pairs). Factors: value (Z, the weighted sum or
    the number of models) and assignments. witnesses: assignments of the class asked for (masks as dicts
    {variable: 0/1}, or tuples of values per variable)."""
    domain = "universe"

    def __init__(self, handle, backend, device=None):
        try:
            info = _AnswerInfo()
            _check(_call("tr_symc_answer_get_info", handle, C.byref(info)), what="count")
            n, cap = int(info.nvalues), max(int(info.limbs), 1)
            num, den = np.zeros(cap, np.uint64), np.zeros(cap, np.uint64)
            values = []
            for i in range(n):
                nl, dl = u32(), u32()
                _check(_call("tr_symc_answer_value", handle, i, num.ctypes.data, den.ctypes.data, cap, C.byref(nl),
                             C.byref(dl)), what="count value")
                values.append(_exact(_limbs_int(num, int(nl.value)), _limbs_int(den, int(dl.value))))
            nw, width = int(info.witnesses), int(info.witness_width)
            wit = np.zeros(max(nw * width, 1), np.uint64)
            _check(_call("tr_symc_answer_witnesses", handle, wit.ctypes.data, len(wit)), what="witnesses")
            k = int(info.universe_vars)
            order = np.zeros(max(k, 1), np.uint32)
            _check(_call("tr_symc_answer_order", handle, order.ctypes.data, len(order)), what="order")
        finally:
            _call("tr_symc_answer_free", handle)
        self.method, self.status = METHODS[int(info.method)], STATUS[int(info.status)]
        self.weighted, self.three_valued = bool(info.weighted), bool(info.three_valued)
        self.backend = backend if backend == "cpu" else ("cuda-emulated" if device and device.get("emulated") else "cuda")
        self.device = device
        self.universe = k
        self.order = tuple(int(v) for v in order[:k]) if int(info.method) in (1, 3, 4) else None
        self.info = dict(nodes=int(info.nodes), width=int(info.width), evaluations=int(info.evaluations),
                         open=int(info.open), limbs=int(info.limbs))
        self.witnesses = [tuple(int(v) for v in wit[i * width:(i + 1) * width]) for i in range(nw)]
        self.values = values
        if int(info.method) in (4, 5):
            self.value, self.assignments = values[0], values[1]
        elif int(info.method) == 3:
            self.true, self.false, self.unknown = (values[0], values[1]), (values[2], values[3]), (values[4], values[5])
            self.total, self.assignments = values[6], 1 << k
        else:
            self.true, self.false, self.unknown, self.total = values[:4]
            self.assignments = 1 << k

    @property
    def exact(self):
        return self.status == "exact"

    def __repr__(self):
        kind = "weighted count" if self.weighted else "model count"
        return f"Count({kind} over the universe of {self.universe} variables, {self.method}, {self.status})"


# ---------------------------------------------------------------------- decision diagrams
class Diagram:
    """An immutable reduced ordered BDD of a program over all k variables of its universe (math_symcount.h)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_symcount.compile")

    def _open(self):
        if not self._handle:
            raise ValueError("the diagram is closed")
        return self._handle

    def view(self):
        v = _DDView()
        _check(_call("tr_symc_dd_get_view", self._open(), C.byref(v)), what="diagram view")
        return v

    @property
    def info(self):
        i = _DDInfo()
        _check(_call("tr_symc_dd_get_info", self._open(), C.byref(i)), what="diagram")
        v = self.view()
        per = np.ctypeslib.as_array(C.cast(v.count, P(u64)), shape=(int(v.nlevels),)) if int(v.nlevels) else []
        return dict(nodes=int(i.nnodes), peak_nodes=int(i.peak_nodes), steps=int(i.steps), bytes=int(i.bytes),
                    width=int(i.width), levels=int(i.nlevels), roots=int(i.nroots), variables=int(i.nvars),
                    three_valued=bool(i.three_valued), level_nodes=[int(x) for x in per])

    @property
    def order(self):
        v = self.view()
        n = int(v.nlevels)
        return tuple(int(x) for x in np.ctypeslib.as_array(C.cast(v.order, P(u32)), shape=(n,))) if n else ()

    def need(self, weights=None, backend="cpu"):
        w = _weights(weights, int(self.view().nvars))
        raw = _Need()
        _check(_call("tr_symc_dd_count_need", self._open(), w, _NEED_DEVICE if _backend(backend) == "cuda" else
                     _NEED_CPU, C.byref(raw)), what="count needs")
        return _needs(raw)

    def raw(self, weights=None, backend="cpu", budget=None):
        """the raw root totals (L-limb words) as the CPU loop or the device kernels wrote them, and L"""
        v = self.view()
        w = _weights(weights, int(v.nvars))
        scaled = np.zeros(max(4 * int(v.nlevels), 4), np.uint64)
        L = u32()
        _check(_call("tr_symc_dd_prepare", C.byref(v), w, scaled.ctypes.data, C.byref(L)), what="count weights")
        out = np.zeros(max(int(v.nroots), 1) * int(L.value), np.uint64)
        refusal = _Refusal()
        if _backend(backend) == "cpu":
            _check(_call("tr_symc_dd_levels", C.byref(v), scaled.ctypes.data, L, out.ctypes.data,
                         C.byref(_budget(budget)), C.byref(refusal)), refusal, "diagram count")
            dev = None
        else:
            gpu = _gpu()
            gpu.run("tr_mgpu_symcount_levels", C.byref(v), scaled.ctypes.data, L, out.ctypes.data,
                    C.byref(_budget(budget)), C.byref(refusal), refusal=refusal, what="diagram count on the device")
            dev = gpu.device()
        return out, int(L.value), w, dev

    def count(self, weights=None, backend="cpu", budget=None):
        """the TRUE / FALSE / UNKNOWN counts (or weights) of the whole universe: CPU, or the level kernels"""
        out, L, w, dev = self.raw(weights, backend, budget)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_symc_dd_adopt", self._open(), w, L, out.ctypes.data, _BY_CPU if backend == "cpu" else _BY_DEVICE,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Count(_made(h, error, refusal, "diagram count"), backend, dev)

    def witnesses(self, m=1, value="true", budget=None):
        """the first m assignments of the class (lexicographic in the order: level 0 most significant, 0 before 1)
        as dicts {variable: 0 / 1} over all k variables"""
        v = self.view()
        roots = {"true": 0, "false": 1}
        if value not in roots:
            raise ValueError("value is 'true' or 'false'")
        r = roots[value]
        if r == 1 and not int(v.three_valued):
            raise ValueError("a two-valued diagram has one root: its FALSE assignments are those not TRUE (use "
                             "math_bool's ~ on the predicate)")
        out = np.zeros(max(int(m), 1), np.uint64)
        cnt, refusal = u64(), _Refusal()
        _check(_call("tr_symc_dd_witnesses", self._open(), r, int(m), out.ctypes.data, C.byref(cnt),
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "witnesses")
        k = int(v.nvars)
        return [{i: int((int(out[j]) >> i) & 1) for i in range(k)} for j in range(int(cnt.value))]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_symc_dd_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Diagram({self.info['nodes']} nodes)" if self._handle else "Diagram(closed)"


def compile(f, k=None, *, order=None, budget=None):
    """the reduced ordered BDD of a BOOL-001 predicate over all k variables (order: "natural", "dfs" or a
    permutation; budget: circuit = nodes, iterations = apply steps, bytes)"""
    prog = _program(f, k)
    kind, arr = _order(order, prog.k)
    v = prog.view()
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_symc_compile", C.byref(v), kind, arr.ctypes.data if arr is not None else None,
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    d = object.__new__(Diagram)
    d._handle, d.program = _made(h, error, refusal, "decision diagram"), prog
    return d


def brute(f, k=None, *, weights=None, backend="cpu", budget=None):
    """every assignment of the k variables enumerated natively (verification; support budget 2^k)"""
    prog = _program(f, k)
    v = prog.view()
    w = _weights(weights, prog.k)
    scaled = np.zeros(max(4 * prog.k, 4), np.uint64)
    L = u32()
    _check(_call("tr_symc_brute_prepare", C.byref(v), w, scaled.ctypes.data, C.byref(L)), what="brute force")
    counts, wsum = np.zeros(3, np.uint64), np.zeros(2 * int(L.value), np.uint64)
    refusal = _Refusal()
    sp = scaled.ctypes.data if w is not None else None
    if _backend(backend) == "cpu":
        _check(_call("tr_symc_brute_raw", C.byref(v), sp, L, counts.ctypes.data, wsum.ctypes.data,
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "brute force")
        dev = None
    else:
        gpu = _gpu()
        gpu.run("tr_mgpu_symcount_brute", C.byref(v), sp, L, counts.ctypes.data, wsum.ctypes.data,
                C.byref(_budget(budget)), C.byref(refusal), refusal=refusal, what="brute force on the device")
        dev = gpu.device()
    error = C.c_int32()
    h = _call("tr_symc_brute_adopt", C.byref(v), w, L, counts.ctypes.data, wsum.ctypes.data,
              _BY_CPU if backend == "cpu" else _BY_DEVICE, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    c = Count(_made(h, error, refusal, "brute force"), backend, dev)
    c.raw = (counts.copy(), wsum.copy() if w is not None else None)
    return c


def brute_need(f, k=None, *, weights=None, backend="cpu"):
    prog = _program(f, k)
    raw = _Need()
    _check(_call("tr_symc_brute_need", C.byref(prog.view()), _weights(weights, prog.k),
                 _NEED_DEVICE if _backend(backend) == "cuda" else _NEED_CPU, C.byref(raw)), what="brute-force needs")
    return _needs(raw)


def search(f, k=None, *, weights=None, order=None, max_evaluations=1 << 12, budget=None):
    """bounded Shannon expansion: exact counts, or (status "incomplete") certified bounds lo <= count <= hi per class
    after max_evaluations branch evaluations (the open branches' weight is never extrapolated)"""
    prog = _program(f, k)
    kind, arr = _order(order, prog.k)
    if kind > 2:
        raise ValueError("a program's order is 'natural', 'dfs' or a permutation")
    w = _weights(weights, prog.k)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_symc_search", C.byref(prog.view()), kind, arr.ctypes.data if arr is not None else None, w,
              int(max_evaluations), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Count(_made(h, error, refusal, "bounded search"), "cpu")


# ---------------------------------------------------------------------- factor systems
def _table(table, dims, what):
    """the C-order entries of a factor table: an array / nested lists of shape dims, or a dict {assignment: value}"""
    size = 1
    for d in dims:
        size *= d
    if isinstance(table, dict):
        flat = [0] * size
        for key, val in table.items():
            key = (key,) if not isinstance(key, tuple) else key
            if len(key) != len(dims) or any(not 0 <= int(a) < d for a, d in zip(key, dims)):
                raise ValueError(f"{what}: assignment {key} outside the scope's domains")
            idx = 0
            for a, d in zip(key, dims):
                idx = idx * d + int(a)
            flat[idx] = val
    else:
        arr = np.array(table, dtype=object)
        if arr.shape != tuple(dims) and not (len(dims) == 0 and arr.shape in ((), (1,))):
            raise ValueError(f"{what}: a table of shape {tuple(dims)}, not {arr.shape}")
        flat = list(arr.reshape(-1))
    return [_rational(v, f"{what} entry") for v in flat]


class Factors:
    """Finite-domain variables (variable i takes 0 .. domains[i] - 1) and factors: (scope, table) with nonnegative
    exact entries, one per joint assignment of the scope (C order: the last scope variable fastest). The weight of an
    assignment is the product of its factors' entries; a zero entry is a hard constraint."""

    def __init__(self, domains, factors=(), *, budget=None):
        self.domains = tuple(int(d) for d in domains)
        n = len(self.domains)
        fs = (_Factor * max(len(factors), 1))()
        scope, num, den = [], [], []
        self.factors = []
        for j, (sc, table) in enumerate(factors):
            sc = tuple(int(v) for v in ((sc,) if isinstance(sc, Integral) else sc))
            if any(not 0 <= v < n for v in sc):
                raise ValueError(f"factor {j}: a scope variable outside 0..{n - 1}")
            entries = _table(table, [self.domains[v] for v in sc], f"factor {j}")
            fs[j] = _Factor(len(scope), len(num), len(sc), 0)
            scope += sc
            num += [q.numerator for q in entries]
            den += [q.denominator for q in entries]
            self.factors.append((sc, entries))
        self._keep = (np.array(self.domains or [1], np.uint32), np.array(scope or [0], np.uint32),
                      np.array(num or [0], np.int64), np.array(den or [1], np.int64))
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_symc_factors_create", n, self._keep[0].ctypes.data, len(factors), fs, self._keep[1].ctypes.data,
                  self._keep[2].ctypes.data, self._keep[3].ctypes.data, len(num), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        self._handle = _made(h, error, refusal, "factor system")

    def _open(self):
        if not self._handle:
            raise ValueError("the factor system is closed")
        return self._handle

    @property
    def info(self):
        i = _FactorsInfo()
        _check(_call("tr_symc_factors_get_info", self._open(), C.byref(i)), what="factors")
        return dict(entries=int(i.entries), bytes=int(i.bytes), variables=int(i.nvars), factors=int(i.nfactors),
                    max_scope=int(i.max_scope),
                    assignments=None if i.assignments_wide else int(i.assignments_lo) | int(i.assignments_hi) << 64)

    def _order_args(self, order):
        kind, arr = _order(order, len(self.domains))
        if kind == ORDERS["dfs"]:
            raise ValueError("an elimination order is 'natural', 'min-fill', 'min-degree' or a permutation")
        return kind, arr

    def order(self, order="min-fill", budget=None):
        """(the elimination order, its width: the largest joint table)"""
        kind, arr = self._order_args(order)
        out, width, refusal = np.zeros(max(len(self.domains), 1), np.uint32), u64(), _Refusal()
        _check(_call("tr_symc_factors_order", self._open(), kind, arr.ctypes.data if arr is not None else None,
                     out.ctypes.data, C.byref(width), C.byref(_budget(budget)), C.byref(refusal)), refusal,
               "elimination order")
        return tuple(int(v) for v in out[:len(self.domains)]), int(width.value)

    def width(self, order="min-fill"):
        return self.order(order)[1]

    def need(self, order="min-fill", weighted=True, witnesses=0):
        kind, arr = self._order_args(order)
        raw = _Need()
        _check(_call("tr_symc_factors_need", self._open(), kind, arr.ctypes.data if arr is not None else None,
                     _WEIGHTED if weighted else _INDICATOR, int(witnesses), C.byref(raw)), what="elimination needs")
        return _needs(raw)

    def count(self, order="min-fill", weighted=True, witnesses=0, budget=None):
        """Z by variable elimination (weighted: the sum of the products of the entries; else the number of models),
        with the first `witnesses` models in lexicographic order of the reverse elimination order"""
        kind, arr = self._order_args(order)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_symc_factors_count", self._open(), kind, arr.ctypes.data if arr is not None else None,
                  _WEIGHTED if weighted else _INDICATOR, int(witnesses), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Count(_made(h, error, refusal, "variable elimination"), "cpu")

    def brute(self, weighted=True, witnesses=0, budget=None):
        """every assignment enumerated natively (verification; witnesses in lexicographic order, variable 0 first)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_symc_factors_brute", self._open(), _WEIGHTED if weighted else _INDICATOR, int(witnesses),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Count(_made(h, error, refusal, "brute force over factors"), "cpu")

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_symc_factors_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Factors({len(self.domains)} variables, {len(self.factors)} factors)"
