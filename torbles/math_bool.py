# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""BOOL-001: predicate DAGs over k conditions, their cofactors, XOR derivatives and signed first and mixed
differences with exact counts, natively on the CPU or with the CUDA kernels (torbles/math_bool.c, math_bool_gpu.cu).

    from torbles import math_bank as mbk, math_bool as mb
    x = mb.variables(3)
    shared = x[0] & x[1]
    f = (shared | ~x[2]) ^ (shared & mb.UNKNOWN)             # AND OR NOT XOR, constants, shared subexpressions
    prog = mb.Program(f, k=3)
    uniform = mb.Masks.assignments(3)                        # the 2^3 assignments (uniform measure)
    scope = mbk.Scope.from_table(db["t"], ["x", "d", "flag"], rows=mask)
    rows = mb.Masks.from_scope(scope, [mb.condition("x", ">", 5), mb.condition("d", "between", "1.5", "3.25"),
                                       mb.condition("flag", "is not null")], backend="cpu")   # or "cuda"
    mb.differences(rows, prog, [[0], [1], [0, 2]], cofactors=True, backend="cpu")      # or "cuda"
    # [{"conditions": (0,), "order": 1, "points", "f": {"true", "false", "unknown"}, "known", "unknown",
    #   "xor": {"true", "false"}, "signed": {-1: n, 0: n, 1: n}, "sum", "nonzero",
    #   "cofactors": [{"assignment": {0: 0}, "true", "false", "unknown"}, {"assignment": {0: 1}, ...}], ...}, ...]
    mb.direct(prog, {0: 1}, scope=scope, conditions=conds)  # one cofactor, evaluated independently (the baseline)

Domains: Masks.assignments(k) (encoded variables: the 2^k assignments, condition i = bit i, uniform measure) or
Masks.from_scope (logical features: each condition a native row predicate over the scope's exact values, evaluated
once to a value and a valid bitmap; the empirical measure of the selected rows). SQL three-valued logic: a condition
over a NULL is UNKNOWN; AND / OR / NOT are Kleene's (SQL's), XOR(a, b) is SQL's `a <> b`. A cofactor replaces the
conditions of S by constants and never reads them at the point; a difference is UNKNOWN at a point where any of its
cofactors is, and is then counted in "unknown" only (never as FALSE or 0). XOR derivative (parity of the cofactors)
and signed difference (sum of (-1)^(m - |t|) f|S:=t) are different numbers: the XOR derivative is the signed one mod 2.
Counts are exact native integers; the probability under the domain's measure is count / points. Budgets raise
AlgebraBudgetError before anything is allocated, a stale scope StaleStateError. Nothing is computed in Python.
Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from .math_bank import Scope, _View as _ScopeView
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["Expr", "var", "variables", "TRUE", "FALSE", "UNKNOWN", "Program", "Condition", "condition", "column",
           "Masks", "differences", "first_differences", "cofactors", "direct", "masks_need", "differences_need",
           "MAX_CONDITIONS", "MAX_ORDER"]

MAX_CONDITIONS, MAX_ASSIGN_VARS, MAX_ORDER, MAX_QUERIES = 64, 63, 20, 1 << 16
ROWS, ASSIGNMENTS = 1, 2
CONST, VAR, NOT, AND, OR, XOR = 1, 2, 3, 4, 5, 6
_OPS = {"const": CONST, "var": VAR, "not": NOT, "and": AND, "or": OR, "xor": XOR}
_COND = {"=": 1, "==": 1, "<>": 2, "!=": 2, "<": 3, "<=": 4, ">": 5, ">=": 6, "between": 7, "is null": 8,
         "is not null": 9, "bit": 10}
_COL_BASE = 10                     # COL_EQ = EQ + 10 .. COL_GE = GE + 10
COFACTORS = 1
_NEED_CPU, _NEED_DEVICE, _NEED_ADOPT = 0, 1, 2
_NONE = (1 << 64) - 1
_U64 = (1 << 64) - 1
P = C.POINTER


class _Condition(C.Structure):
    _native_ = "tr_bool_condition"
    _fields_ = [("op", C.c_int32), ("col", C.c_int32), ("col2", C.c_int32), ("reserved", C.c_int32),
                ("num", C.c_int64 * 2), ("den", C.c_int64 * 2)]


class _Node(C.Structure):
    _native_ = "tr_bool_node"
    _fields_ = [("op", C.c_int32), ("a", C.c_int32), ("b", C.c_int32), ("reserved", C.c_int32)]


class _ProgramView(C.Structure):
    _native_ = "tr_bool_program_view"
    _fields_ = [("nodes", C.c_void_p), ("deps", C.c_void_p), ("schedule", C.c_void_p), ("nnodes", C.c_int32),
                ("nsched", C.c_int32), ("root", C.c_int32), ("k", C.c_int32)]


class _Domain(C.Structure):
    _native_ = "tr_bool_domain"
    _fields_ = [("points", C.c_uint64), ("nw", C.c_uint64), ("selected", C.c_uint64), ("kind", C.c_int32),
                ("k", C.c_int32), ("selection", C.c_void_p), ("value", C.c_void_p), ("valid", C.c_void_p)]


class _Query(C.Structure):
    _native_ = "tr_bool_query"
    _fields_ = [("vars", C.c_uint64), ("flags", C.c_int32), ("reserved", C.c_int32)]


class _Result(C.Structure):
    _native_ = "tr_bool_result"
    _fields_ = [("vars", C.c_uint64), ("order", C.c_int32), ("nbins", C.c_int32), ("lo", C.c_int64),
                ("points", C.c_uint64)] + [(n, C.c_uint64) for n in (
                    "f_true", "f_false", "f_unknown", "known", "unknown", "xor_true", "xor_false", "nonzero")] + [
                ("sum_lo", C.c_int64), ("sum_hi", C.c_int64), ("hist", C.c_uint64), ("cof", C.c_uint64)]


class _MasksInfo(C.Structure):
    _native_ = "tr_bool_masks_info"
    _fields_ = [(n, C.c_uint64) for n in ("points", "nw", "selected", "bytes")] + [
        ("scope_digest", C.c_uint64 * 2), ("kind", C.c_int32), ("k", C.c_int32), ("has_valid", C.c_int32),
        ("reserved", C.c_int32)]


class _ProgramInfo(C.Structure):
    _native_ = "tr_bool_program_info"
    _fields_ = [("nnodes", C.c_int32), ("nsched", C.c_int32), ("root", C.c_int32), ("k", C.c_int32),
                ("support", C.c_uint64), ("bytes", C.c_uint64)]


for _s, _n in ((_Condition, 48), (_Node, 16), (_ProgramView, 40), (_Domain, 56), (_Query, 16), (_Result, 128),
               (_MasksInfo, 64), (_ProgramInfo, 32)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        sig = {
            "tr_bool_masks_build": (vp, [vp, P(_Condition), i32, P(_Budget), P(_Refusal), e]),
            "tr_bool_masks_assignments": (vp, [i32, P(_Budget), P(_Refusal), e]),
            "tr_bool_masks_adopt": (vp, [vp, P(_Condition), i32, vp, vp, P(_Budget), P(_Refusal), e]),
            "tr_bool_masks_need": (i32, [vp, P(_Condition), i32, i32, P(_Need)]),
            "tr_bool_masks_get_info": (i32, [vp, P(_MasksInfo)]),
            "tr_bool_masks_get_domain": (i32, [vp, P(_Domain)]),
            "tr_bool_masks_counts": (i32, [vp, i32, P(C.c_uint64 * 3)]),
            "tr_bool_masks_equal": (i32, [vp, vp]),
            "tr_bool_masks_free": (None, [vp]),
            "tr_bool_program_create": (vp, [i32, P(_Node), i32, i32, P(_Budget), P(_Refusal), e]),
            "tr_bool_program_need": (i32, [i32, i32, P(_Need)]),
            "tr_bool_program_get_view": (i32, [vp, P(_ProgramView)]),
            "tr_bool_program_get_info": (i32, [vp, P(_ProgramInfo)]),
            "tr_bool_program_free": (None, [vp]),
            "tr_bool_layout": (i32, [vp, P(_Query), i32, P(C.c_uint64), P(C.c_uint64)]),
            "tr_bool_differences_need": (i32, [vp, vp, P(_Query), i32, i32, P(_Need)]),
            "tr_bool_differences": (i32, [vp, vp, P(_Query), i32, P(_Budget), P(_Result), vp, vp, P(_Refusal)]),
            "tr_bool_differences_view": (i32, [P(_Domain), P(_ProgramView), P(_Query), i32, P(_Budget), P(_Result),
                                               vp, vp, P(_Refusal)]),
            "tr_bool_direct": (i32, [vp, P(_Condition), vp, u64, u64, P(_Budget), P(C.c_uint64 * 3), P(_Refusal)]),
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
    math_gpu.bind("tr_mgpu_bool_masks", C.c_int, [P(_ScopeView), P(_Condition), C.c_int32, P(_Budget), C.c_void_p,
                                                 C.c_void_p, P(_Refusal)])
    math_gpu.bind("tr_mgpu_bool_differences", C.c_int, [P(_Domain), P(_ProgramView), P(_Query), C.c_int32,
                                                       P(_Budget), P(_Result), C.c_void_p, C.c_void_p, P(_Refusal)])
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


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


# ---------------------------------------------------------------------- predicate expressions
class Expr:
    """A predicate over conditions 0..k-1: var(i), TRUE / FALSE / UNKNOWN, and & | ^ ~ (AND, OR, XOR, NOT). A Python
    object used twice is one shared node of the program."""
    __slots__ = ("op", "a", "b")

    def __init__(self, op, a=-1, b=-1):
        self.op, self.a, self.b = op, a, b

    def __and__(self, o): return Expr(AND, self, _expr(o))
    def __rand__(self, o): return Expr(AND, _expr(o), self)
    def __or__(self, o): return Expr(OR, self, _expr(o))
    def __ror__(self, o): return Expr(OR, _expr(o), self)
    def __xor__(self, o): return Expr(XOR, self, _expr(o))
    def __rxor__(self, o): return Expr(XOR, _expr(o), self)
    def __invert__(self): return Expr(NOT, self)

    def __bool__(self):
        raise TypeError("a predicate has no Python truth value: combine with & | ^ ~, not and / or / not")

    def __repr__(self):
        if self.op == VAR:
            return f"x{self.a}"
        if self.op == CONST:
            return ("FALSE", "TRUE", "UNKNOWN")[self.a]
        if self.op == NOT:
            return f"~{self.a!r}"
        return f"({self.a!r} {dict([(AND, '&'), (OR, '|'), (XOR, '^')])[self.op]} {self.b!r})"


def _expr(o):
    if isinstance(o, Expr):
        return o
    if o is None:
        return UNKNOWN
    if isinstance(o, bool):
        return TRUE if o else FALSE
    raise TypeError(f"{o!r} is not a predicate (use var(i), TRUE, FALSE, UNKNOWN)")


def var(i):
    """condition i"""
    if isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= int(i) < MAX_CONDITIONS:
        raise ValueError(f"a condition index is an integer in 0..{MAX_CONDITIONS - 1}")
    return Expr(VAR, int(i))


def variables(k):
    return [var(i) for i in range(int(k))]


TRUE, FALSE, UNKNOWN = Expr(CONST, 1), Expr(CONST, 0), Expr(CONST, 2)


def _compile(expr):
    """the nodes of an expression DAG (post-order; a Python object reached twice is one node) and the root index"""
    index, nodes, stack = {}, [], [(expr, False)]
    while stack:
        e, ready = stack.pop()
        if not isinstance(e, Expr):
            raise TypeError(f"{e!r} is not a predicate")
        if id(e) in index:
            continue
        kids = [] if e.op in (CONST, VAR) else ([e.a] if e.op == NOT else [e.a, e.b])
        if not ready:
            stack.append((e, True))
            stack.extend((c, False) for c in reversed(kids) if id(c) not in index)
            continue
        a = e.a if e.op in (CONST, VAR) else index[id(e.a)]
        b = index[id(e.b)] if e.op in (AND, OR, XOR) else -1
        index[id(e)] = len(nodes)
        nodes.append((e.op, a, b))
    return nodes, index[id(expr)]


class Program:
    """An immutable native predicate DAG over k conditions (math_bool.h, Programs)."""

    def __init__(self, expr, k, *, budget=None):
        nodes, root = _compile(_expr(expr))
        self._create(int(k), nodes, root, budget)

    @classmethod
    def from_nodes(cls, k, nodes, root=None, *, budget=None):
        """nodes: (op, a, b) triples, op a name ("const", "var", "not", "and", "or", "xor") or its code; node i may
        only refer to nodes < i; unused operands -1; root: default the last node"""
        self = object.__new__(cls)
        triples = []
        for n in nodes:
            op, a, b = (tuple(n) + (-1, -1))[:3]
            triples.append((_OPS[op] if isinstance(op, str) else int(op), int(a), int(b)))
        self._create(int(k), triples, len(triples) - 1 if root is None else int(root), budget)
        return self

    def _create(self, k, nodes, root, budget):
        self._handle = None
        n = len(nodes)
        arr = (_Node * max(n, 1))()
        for i, (op, a, b) in enumerate(nodes):
            for v in (op, a, b):
                if not -(1 << 31) <= v < (1 << 31):
                    raise ValueError("node fields are int32")
            arr[i] = _Node(op, a, b, 0)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_bool_program_create", k, arr, n, root, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        self._handle = _made(h, error, refusal, "predicate program")
        self.k, self.nodes, self.root = k, tuple(nodes), root

    def _open(self):
        if not self._handle:
            raise ValueError("the program is closed")

    def view(self):
        self._open()
        v = _ProgramView()
        _check(_call("tr_bool_program_get_view", self._handle, C.byref(v)), what="program view")
        return v

    @property
    def info(self):
        self._open()
        i = _ProgramInfo()
        _check(_call("tr_bool_program_get_info", self._handle, C.byref(i)), what="program")
        return dict(nodes=int(i.nnodes), evaluated=int(i.nsched), root=int(i.root), k=int(i.k),
                    support=tuple(j for j in range(int(i.k)) if (int(i.support) >> j) & 1), bytes=int(i.bytes))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_bool_program_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Program(k={self.k}, nodes={len(self.nodes)})" if self._handle else "Program(closed)"


# ---------------------------------------------------------------------- conditions
class column:
    """the other operand of a column comparison: condition("x", "<", column("y"))"""
    def __init__(self, name):
        self.name = str(name)

    def __repr__(self):
        return f"column({self.name!r})"


def _rational(v, what):
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, float):
        raise TypeError(f"{what}: a float is not an exact constant (give an int, Fraction, Decimal or a decimal string)")
    if isinstance(v, (Integral, Rational)):
        f = Fraction(v)
    elif isinstance(v, (Decimal, str)):
        f = Fraction(v)
    else:
        raise TypeError(f"{what}: {v!r} is not an exact constant")
    if not (-(1 << 63) <= f.numerator < (1 << 63) and f.denominator < (1 << 63)):
        raise OverflowError(f"{what}: {v!r} does not fit an int64 numerator and denominator")
    return f.numerator, f.denominator


class Condition:
    """A native row predicate over a scope column (math_bool.h, Conditions); resolved against a scope by Masks."""
    __slots__ = ("column", "op", "args", "other")

    def __init__(self, column_name, op, args=(), other=None):
        self.column, self.op, self.args, self.other = column_name, op, args, other

    def native(self, names):
        names = list(names)
        if self.column not in names:
            raise ValueError(f"the scope has no column {self.column!r}")
        c = _Condition()
        c.op, c.col, c.col2, c.reserved = _COND[self.op], names.index(self.column), -1, 0
        c.num[0] = c.num[1] = 0
        c.den[0] = c.den[1] = 1
        if self.other is not None:
            if self.other not in names:
                raise ValueError(f"the scope has no column {self.other!r}")
            c.op += _COL_BASE
            c.col2 = names.index(self.other)
        elif self.op == "bit":
            c.num[0], c.num[1] = self.args
        else:
            for j, (n, d) in enumerate(self.args):
                c.num[j], c.den[j] = n, d
        return c

    def __repr__(self):
        rhs = f"column({self.other!r})" if self.other is not None else ", ".join(map(str, self.args))
        return f"condition({self.column!r}, {self.op!r}{', ' + rhs if rhs else ''})"


def condition(column_name, op, value=None, high=None, *, base=0):
    """column op value: "=", "<>" ("!="), "<", "<=", ">", ">=" with an exact constant (int, Fraction, Decimal or a
    decimal string, in value units) or column("y"); "between" value and high (inclusive); "is null", "is not null";
    "bit" value = the bit index 0..63 of (logical - base) mod 2^64 (base: an int64 logical value)."""
    op = op.lower().strip()
    if op not in _COND:
        raise ValueError(f"unknown condition operator {op!r}")
    if op in ("is null", "is not null"):
        if value is not None or high is not None:
            raise ValueError(f"{op} takes no constant")
        return Condition(column_name, op)
    if op == "bit":
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) <= 63:
            raise ValueError("bit: the bit index is an integer in 0..63")
        if isinstance(base, bool) or not isinstance(base, Integral) or not -(1 << 63) <= int(base) < (1 << 63):
            raise ValueError("bit: base is an int64 logical value")
        return Condition(column_name, op, (int(base), int(value)))
    if op == "between":
        if value is None or high is None:
            raise ValueError("between needs a low and a high constant")
        return Condition(column_name, op, (_rational(value, "between"), _rational(high, "between")))
    if high is not None:
        raise ValueError(f"{op} takes one operand")
    if isinstance(value, column):
        return Condition(column_name, op, (), value.name)
    if value is None:
        raise ValueError(f"{op} needs a constant or column(name) (a NULL constant: use the UNKNOWN constant)")
    return Condition(column_name, op, (_rational(value, op),))


def _conditions(conds, names):
    conds = list(conds)
    arr = (_Condition * max(len(conds), 1))()
    for i, c in enumerate(conds):
        if not isinstance(c, Condition):
            raise TypeError(f"{c!r} is not a condition (use condition(...))")
        arr[i] = c.native(names)
    return arr, len(conds)


# ---------------------------------------------------------------------- domains
class Masks:
    """An immutable native domain: the 2^k assignments, or k condition bitmaps over the selected rows of a scope."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Masks.from_scope(...) or Masks.assignments(k)")

    @classmethod
    def _wrap(cls, handle, scope, conditions, backend, device):
        self = object.__new__(cls)
        self._handle, self.scope, self.conditions, self.backend, self.device = handle, scope, conditions, backend, device
        i = self._raw_info()
        self.kind = "rows" if i.kind == ROWS else "assignments"
        self.k = int(i.k)
        return self

    @classmethod
    def assignments(cls, k, *, budget=None):
        """the 2^k assignments of k two-valued variables (encoded variables, uniform measure)"""
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k is an integer")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_bool_masks_assignments", int(k), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "assignment domain"), None, (), "cpu", "native CPU")

    @classmethod
    def from_scope(cls, scope, conditions, *, backend="cpu", budget=None):
        """k = len(conditions) row predicates over the selected rows of a math_bank.Scope (logical features,
        empirical measure); backend "cuda" builds the words with the CUDA kernel and adopts them"""
        if not isinstance(scope, Scope):
            raise TypeError("scope: a math_bank.Scope")
        if scope.source == "table":
            scope.require_fresh("predicate masks")
        conditions = tuple(conditions)
        arr, k = _conditions(conditions, scope.names)
        error, refusal = C.c_int32(), _Refusal()
        if _backend(backend) == "cpu":
            h = _call("tr_bool_masks_build", scope._handle, arr, k, C.byref(_budget(budget)), C.byref(refusal),
                      C.byref(error))
            return cls._wrap(_made(h, error, refusal, "predicate masks"), scope, conditions, "cpu", "native CPU")
        gpu = _gpu()
        need = masks_need(scope, conditions, backend="cuda")
        from . import math_budgets
        math_budgets.check(budget or Budget(), "predicate masks on the device", **need)
        nw = (scope.rows + 63) // 64
        value = np.zeros(max(k * nw, 1), dtype=np.uint64)
        valid = np.zeros(max(k * nw, 1), dtype=np.uint64)
        view = scope.view()
        gpu.run("tr_mgpu_bool_masks", C.byref(view), arr, k, C.byref(_budget(budget)), value.ctypes.data,
                valid.ctypes.data, C.byref(refusal), refusal=refusal, what="predicate masks on the device")
        h = _call("tr_bool_masks_adopt", scope._handle, arr, k, value.ctypes.data, valid.ctypes.data,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        d = gpu.device()
        return cls._wrap(_made(h, error, refusal, "predicate masks (device words)"), scope, conditions,
                         "cuda-emulated" if d["emulated"] else "cuda", d["name"])

    def _open(self):
        if not self._handle:
            raise ValueError("the masks are closed")

    def _raw_info(self):
        self._open()
        i = _MasksInfo()
        _check(_call("tr_bool_masks_get_info", self._handle, C.byref(i)), what="masks")
        return i

    @property
    def info(self):
        i = self._raw_info()
        return dict(kind=self.kind, k=int(i.k), points=int(i.points), selected=int(i.selected), words=int(i.nw),
                    bytes=int(i.bytes), has_unknown=bool(i.has_valid), backend=self.backend, device=self.device)

    @property
    def points(self):
        """the points that count: selected rows, or 2^k"""
        return int(self._raw_info().selected)

    def domain(self):
        """the native tr_bool_domain view (pointers into these masks; valid while they live)"""
        self._open()
        d = _Domain()
        _check(_call("tr_bool_masks_get_domain", self._handle, C.byref(d)), what="masks domain")
        return d

    def counts(self, i):
        """condition i over the points that count: {"true", "false", "unknown"}"""
        self._open()
        out = (C.c_uint64 * 3)()
        _check(_call("tr_bool_masks_counts", self._handle, int(i), C.byref(out)), what="condition counts")
        return dict(true=int(out[0]), false=int(out[1]), unknown=int(out[2]))

    def words(self, i):
        """copies of condition i's value and valid words (ROWS; valid equals the selection without UNKNOWN)"""
        d = self.domain()
        if self.kind != "rows":
            raise ValueError("the words of the assignment domain are generated, not stored")
        if not 0 <= int(i) < self.k:
            raise ValueError("condition index out of range")
        nw = int(d.nw)
        if not nw:
            return np.zeros(0, np.uint64), np.zeros(0, np.uint64)
        def read(ptr, offset):
            return np.ctypeslib.as_array((C.c_uint64 * nw).from_address(ptr + 8 * offset)).copy()
        value = read(d.value, int(i) * nw)
        valid = read(d.valid, int(i) * nw) if d.valid else read(d.selection, 0)
        return value, valid

    def selection(self):
        d = self.domain()
        nw = int(d.nw)
        if self.kind != "rows" or not nw:
            return np.zeros(0, np.uint64)
        return np.ctypeslib.as_array((C.c_uint64 * nw).from_address(d.selection)).copy()

    def equal(self, other):
        self._open(); other._open()
        return bool(_check(_call("tr_bool_masks_equal", self._handle, other._handle), what="masks comparison"))

    def require_fresh(self, what="masks"):
        if self.scope is not None and self.scope.source == "table":
            self.scope.require_fresh(what)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_bool_masks_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Masks(closed)"
        return f"Masks({self.kind}, k={self.k}, points={self.points}, {self.backend})"


def masks_need(scope, conditions, backend="cpu"):
    """the needs a masks build checks: {resource: need} (scope None: the assignment domain of len(conditions) or k
    variables; backend "cuda": the device copies and the adopted masks, resource by resource the larger)"""
    if scope is None:
        out = _Need()
        _check(_call("tr_bool_masks_need", None, None, int(conditions), _NEED_CPU, C.byref(out)), what="masks needs")
        return _needs_dict(out)
    arr, k = _conditions(conditions, scope.names)
    targets = (_NEED_CPU,) if _backend(backend) == "cpu" else (_NEED_DEVICE, _NEED_ADOPT)
    best = [0] * len(RESOURCES)
    for t in targets:
        out = _Need()
        _check(_call("tr_bool_masks_need", scope._handle, arr, k, t, C.byref(out)), what="masks needs")
        best = [max(a, int(b)) for a, b in zip(best, out.need)]
    return {name: v for name, v in zip(RESOURCES, best) if v}


# ---------------------------------------------------------------------- differences
def _queries(queries, k, cofactors):
    out = []
    for q in queries:
        q = [q] if isinstance(q, Integral) and not isinstance(q, bool) else list(q)
        conds = sorted(set(int(i) for i in q))
        if len(conds) != len(q):
            raise ValueError(f"query {q!r}: a condition appears twice")
        if any(not 0 <= i < k for i in conds):
            raise ValueError(f"query {q!r}: conditions are 0..{k - 1}")
        out.append(conds)
    arr = (_Query * max(len(out), 1))()
    for j, conds in enumerate(out):
        arr[j] = _Query(sum(1 << i for i in conds), COFACTORS if cofactors else 0, 0)
    return arr, out


def _format(results, hist, cof, conds, backend, device):
    out = []
    for r, cs in zip(results, conds):
        nbins, lo = int(r.nbins), int(r.lo)
        bins = hist[int(r.hist):int(r.hist) + nbins].tolist()
        rec = dict(conditions=tuple(cs), order=int(r.order), points=int(r.points),
                   f=dict(true=int(r.f_true), false=int(r.f_false), unknown=int(r.f_unknown)),
                   known=int(r.known), unknown=int(r.unknown), xor=dict(true=int(r.xor_true), false=int(r.xor_false)),
                   signed={lo + b: int(c) for b, c in enumerate(bins)},
                   sum=(int(r.sum_hi) << 64) | (int(r.sum_lo) & _U64), nonzero=int(r.nonzero),
                   cofactors=None, backend=backend, device=device)
        if int(r.cof) != _NONE:
            base = int(r.cof)
            rec["cofactors"] = [dict(assignment={c: (t >> j) & 1 for j, c in enumerate(cs)},
                                     true=int(cof[base + 3 * t]), false=int(cof[base + 3 * t + 1]),
                                     unknown=int(cof[base + 3 * t + 2])) for t in range(1 << len(cs))]
        out.append(rec)
    return out


def differences(masks, program, queries, *, cofactors=False, backend="cpu", budget=None):
    """For each query (an iterable of condition indices S, or one index): the counts of the program f, of the XOR
    derivative and of the signed difference over S (a value histogram, the exact sum, the nonzero points), and with
    cofactors=True the TRUE / FALSE / UNKNOWN counts of every cofactor f|S:=t. Shared-mask execution, natively."""
    masks._open(); program._open()
    masks.require_fresh("differences")
    if masks.k != program.k:
        from .math_domains import DomainError
        raise DomainError(f"the program reads {program.k} conditions, the domain has {masks.k}")
    arr, conds = _queries(queries, masks.k, cofactors)
    nq = len(conds)
    hist_total, cof_total = C.c_uint64(), C.c_uint64()
    _check(_call("tr_bool_layout", program._handle, arr, nq, C.byref(hist_total), C.byref(cof_total)),
           what="queries")
    results = (_Result * max(nq, 1))()
    hist = np.zeros(max(int(hist_total.value), 1), dtype=np.uint64)
    cof = np.zeros(max(int(cof_total.value), 1), dtype=np.uint64)
    refusal = _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_bool_differences", masks._handle, program._handle, arr, nq, C.byref(_budget(budget)), results,
                     hist.ctypes.data, cof.ctypes.data, C.byref(refusal)), refusal, "differences")
        return _format(results[:nq], hist, cof, conds, "cpu", "native CPU")
    gpu = _gpu()
    dom, view = masks.domain(), program.view()
    gpu.run("tr_mgpu_bool_differences", C.byref(dom), C.byref(view), arr, nq, C.byref(_budget(budget)), results,
            hist.ctypes.data, cof.ctypes.data, C.byref(refusal), refusal=refusal, what="differences on the device")
    d = gpu.device()
    return _format(results[:nq], hist, cof, conds, "cuda-emulated" if d["emulated"] else "cuda", d["name"])


def first_differences(masks, program, *, cofactors=True, backend="cpu", budget=None):
    """differences over each single condition 0..k-1 (cofactors f|x_i=0, f|x_i=1, XOR derivative, f|1 - f|0)"""
    return differences(masks, program, [[i] for i in range(masks.k)], cofactors=cofactors, backend=backend,
                       budget=budget)


def cofactors(masks, program, i, *, backend="cpu", budget=None):
    """(f|x_i=0, f|x_i=1) as {"true", "false", "unknown"} counts"""
    r = differences(masks, program, [[i]], cofactors=True, backend=backend, budget=budget)[0]["cofactors"]
    return tuple({k: c[k] for k in ("true", "false", "unknown")} for c in r)


def differences_need(masks, program, queries, *, cofactors=False, backend="cpu"):
    """the needs a differences request checks before allocating: {resource: need}"""
    masks._open(); program._open()
    arr, conds = _queries(queries, masks.k, cofactors)
    out = _Need()
    _check(_call("tr_bool_differences_need", masks._handle, program._handle, arr, len(conds),
                 _NEED_CPU if _backend(backend) == "cpu" else _NEED_DEVICE, C.byref(out)), what="differences needs")
    return _needs_dict(out)


def direct(program, assignment, *, scope=None, conditions=None, budget=None):
    """one cofactor evaluated independently, point by point (no masks, no sharing): assignment {condition: 0 or 1}
    (empty: f itself); over the selected rows of scope with the conditions read from its values, or (scope None)
    over the 2^k assignments. {"true", "false", "unknown"} counts."""
    program._open()
    vars_ = values = 0
    for i, v in dict(assignment).items():
        i = int(i)
        if not 0 <= i < program.k or v not in (0, 1, True, False):
            raise ValueError("assignment: {condition index: 0 or 1}")
        vars_ |= 1 << i
        values |= int(v) << i
    out, refusal = (C.c_uint64 * 3)(), _Refusal()
    if scope is None:
        if conditions is not None:
            raise ValueError("conditions need a scope")
        rc = _call("tr_bool_direct", None, None, program._handle, vars_, values, C.byref(_budget(budget)),
                   C.byref(out), C.byref(refusal))
    else:
        if scope.source == "table":
            scope.require_fresh("direct evaluation")
        arr, k = _conditions(conditions or (), scope.names)
        if k != program.k:
            raise ValueError(f"the program reads {program.k} conditions, {k} given")
        rc = _call("tr_bool_direct", scope._handle, arr, program._handle, vars_, values, C.byref(_budget(budget)),
                   C.byref(out), C.byref(refusal))
    _check(rc, refusal, "direct evaluation")
    return dict(true=int(out[0]), false=int(out[1]), unknown=int(out[2]))
