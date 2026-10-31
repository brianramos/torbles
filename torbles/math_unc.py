# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""UNC-001: certified abstract states (intervals, affine forms, congruences, nullness, exact-zero facts) over the
groups of a scope, natively on the CPU or with the CUDA kernels (torbles/math_unc.c, math_unc_gpu.cu).

    from torbles import math_unc as mu, math_bank as mbk
    x, y = mu.column("x"), mu.column("y")
    prog = mu.Program((x + 1) > x, ["x", "y"])                 # carrier="exact" | "binary64", precision, rank, budget
    state = mu.State.from_scope(mbk.Scope.from_table(t, ["x", "y"], group_by=["k"]), ["x", "y"])
    ev = state.evaluate(prog, backend="cpu")                   # or backend="cuda"; ev.results[g]: decision, ...
    f = state.filter(scope, prog, backend="cuda")              # groups decided by proof, the rest row by row
    mu.identity(mu.Program(x - x, ["x"]))                      # an FND-007 law: x - x = 0 is a ring identity
    state.certify(scope, mu.Program(x, ["x"]), "congruence", modulus=7)   # an FND-007 snapshot, re-verified
    mu.meet((1, 3), (2, 5))                                    # CRT: (Fraction(7), Fraction(15))
    mu.sum_fits(ev.results[g], bits=64, unit=1)                # an overflow decision for SUM over the group

Every fact is sound: a concrete row's value lies in its node's interval, form range and congruence class; a predicate
is decided ALL / NONE for a group only when every row is TRUE / no row can be TRUE; everything else is "unknown"
and the filter evaluates those rows exactly. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _Q, _I128
from .math_errors import RESOURCES, StaleStateError, check as _check

__all__ = ["Expr", "column", "const", "min_", "max_", "Program", "State", "Evaluation", "rows", "symbolic",
           "identity", "meet", "sum_fits", "need", "OPS", "CARRIERS", "DECISIONS", "FLAGS", "MAX_NODES", "MAX_TERMS",
           "MAX_COLUMNS"]

MAX_NODES, MAX_TERMS, MAX_COLUMNS, FRESH = 64, 16, 32, 32
OPS = {"col": 1, "const": 2, "add": 3, "sub": 4, "mul": 5, "neg": 6, "div": 7, "min": 8, "max": 9, "abs": 10,
       "sqr": 11, "mod": 12, "lt": 13, "le": 14, "eq": 15, "ne": 16, "ge": 17, "gt": 18, "and": 19, "or": 20,
       "not": 21, "is_null": 22, "not_null": 23}
CARRIERS = {"exact": 1, "binary64": 2}
DOMAINS = {"affine": 1, "congruence": 2}
DECISIONS = {0: "unknown", 1: "all", 2: "none", 3: "empty"}
FLAGS = {"lo_inf": 1, "hi_inf": 2, "affine_top": 4, "congruence_top": 8, "widened": 16, "condensed": 32,
         "budget": 64, "rounded": 128}
FACTS = {"not_null": 1, "congruence": 2, "constant": 3}
ZERO = {0: "unknown", 1: "zero", 2: "nonzero"}
_I64 = (-(1 << 63), (1 << 63) - 1)
_UNARY = {"neg", "div", "abs", "sqr", "mod", "not", "is_null", "not_null"}
_BOOL = {"lt", "le", "eq", "ne", "ge", "gt", "and", "or", "not", "is_null", "not_null"}
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Node(C.Structure):
    _native_ = "tr_unc_node"
    _fields_ = [("op", C.c_int32), ("a", C.c_int32), ("b", C.c_int32), ("reserved", C.c_int32), ("k", _Q)]


class _Program(C.Structure):
    _native_ = "tr_unc_program"
    _fields_ = [(n, C.c_int32) for n in ("nnodes", "ncols", "carrier", "precision", "rank", "domains", "root",
                                          "reserved")] + [("budget", _Q), ("nodes", C.c_void_p)]


class _Input(C.Structure):
    _native_ = "tr_unc_input"
    _fields_ = [("rows", C.c_uint64), ("present", C.c_uint64), ("min", C.c_int64), ("max", C.c_int64),
                ("modulus", C.c_uint64), ("scale", C.c_int32), ("reserved", C.c_int32)]


class _Term(C.Structure):
    _native_ = "tr_unc_term"
    _fields_ = [("sym", C.c_int32), ("reserved", C.c_int32), ("num", C.c_int64), ("den", C.c_int64), ("d", C.c_double)]


class _Value(C.Structure):
    _native_ = "tr_unc_value"
    _fields_ = [(n, C.c_int32) for n in ("kind", "null", "outcomes", "zero", "flags", "nterms")] + [
        ("lo", _Q), ("hi", _Q), ("dlo", C.c_double), ("dhi", C.c_double), ("cr", _Q), ("cm", _Q), ("center", _Q),
        ("dcenter", C.c_double), ("reserved", C.c_double), ("term", _Term * MAX_TERMS)]


class _Result(C.Structure):
    _native_ = "tr_unc_result"
    _fields_ = [(n, C.c_int32) for n in ("kind", "decision", "outcomes", "null", "zero", "flags", "nterms",
                                          "fresh")] + [
        ("lo", _Q), ("hi", _Q), ("dlo", C.c_double), ("dhi", C.c_double), ("cr", _Q), ("cm", _Q), ("center", _Q),
        ("radius", _Q), ("error", _Q), ("dcenter", C.c_double), ("dradius", C.c_double), ("derror", C.c_double),
        ("rows", C.c_uint64), ("reserved", C.c_uint64)]


class _Stats(C.Structure):
    _native_ = "tr_unc_filter_stats"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "rows_proven", "rows_evaluated", "rows_true",
                                          "overflow", "first_overflow", "groups_all", "groups_none",
                                          "groups_unknown", "groups_empty")]


class _Sum(C.Structure):
    _native_ = "tr_unc_sum"
    _fields_ = [("proven", C.c_int32), ("integral", C.c_int32), ("bits", C.c_int32), ("bounded", C.c_int32),
                ("lo", _I128), ("hi", _I128)]


class _StateInfo(C.Structure):
    _native_ = "tr_unc_state_info"
    _fields_ = [("ngroups", C.c_uint64), ("bytes", C.c_uint64), ("ncols", C.c_int32), ("scoped", C.c_int32),
                ("cols", C.c_int32 * MAX_COLUMNS)]


for _s, _n in ((_Node, 32), (_Program, 56), (_Input, 48), (_Term, 32), (_Value, 648), (_Result, 200), (_Stats, 88),
               (_Sum, 48), (_StateInfo, 152)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _mc():
    from . import math_certificates
    return math_certificates


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        mc = _mc()
        sig = {
            "tr_unc_program_check": (i32, [P(_Program), vp, P(_Need)]),
            "tr_unc_state_from_scope": (vp, [vp, P(C.c_int32), i32, P(_Budget), P(_Refusal), e]),
            "tr_unc_state_from_inputs": (vp, [u64, i32, P(_Input), P(_Budget), P(_Refusal), e]),
            "tr_unc_state_need": (i32, [u64, i32, u64, i32, P(_Need)]),
            "tr_unc_state_get_info": (i32, [vp, P(_StateInfo)]),
            "tr_unc_state_inputs": (vp, [vp]),
            "tr_unc_state_fresh": (i32, [vp, vp, P(mc._Staleness)]),
            "tr_unc_state_free": (None, [vp]),
            "tr_unc_eval": (i32, [vp, P(_Program), u64, u64, P(_Result), P(_Budget), P(_Refusal)]),
            "tr_unc_eval_need": (i32, [P(_Program), u64, i32, i32, P(_Need)]),
            "tr_unc_trace": (i32, [vp, P(_Program), u64, P(_Value), i32]),
            "tr_unc_symbolic": (i32, [P(_Program), i32, P(_Value), P(C.c_int32)]),
            "tr_unc_rows": (i32, [vp, P(C.c_int32), P(_Program), vp, vp, P(_Stats), P(_Budget), P(_Refusal)]),
            "tr_unc_filter": (i32, [vp, vp, P(_Program), vp, P(_Stats), P(_Budget), P(_Refusal)]),
            "tr_unc_sum_fits": (i32, [P(_Result), i32, _Q, P(_Sum)]),
            "tr_unc_cong_meet": (i32, [_Q, _Q, _Q, _Q, P(_Q), P(_Q)]),
            "tr_unc_identity": (vp, [P(_Program), i32, P(_Budget), P(_Refusal), e]),
            "tr_unc_certify": (vp, [vp, vp, P(_Program), i32, u64, P(mc._Evidence), P(_Budget), P(_Refusal), e]),
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
    math_gpu.bind("tr_mgpu_unc_eval", C.c_int, [P(_Input), C.c_uint64, C.c_int32, P(_Program), P(_Result),
                                               P(_Budget), P(_Refusal)])
    from .math_bank import _View
    math_gpu.bind("tr_mgpu_unc_rows", C.c_int, [P(_View), P(C.c_int32), P(_Program), C.c_void_p, C.c_void_p,
                                               P(_Stats), P(_Budget), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _gpu_code(name, *args):
    """the raw code of a CUDA entry point (the caller maps it; TORBLES_MATH_REQUIRE_DEVICE=1 refuses the emulation)"""
    import os
    g = _gpu()
    if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
        g.require_device()
    return getattr(g.lib(), name)(*args)


def _raise(rc, refusal, what, stats, backend):
    if rc == -3:
        err = OverflowError(f"{what}: {stats['overflow']} rows leave the exact int64 rational evaluation "
                            f"(smallest {stats['first_overflow']})")
        err.stats = stats
        raise err
    if rc:
        if backend == "cuda":
            from . import math_gpu
            math_gpu.check(rc, refusal, what)
        _check(rc, refusal, what)


def _backend_name(backend):
    if backend == "cpu":
        return "cpu", "native CPU"
    from . import math_gpu
    d = math_gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


# ------------------------------------------------------------------ exact constants
def _frac(v, what="constant"):
    if isinstance(v, bool):
        raise TypeError(f"{what}: a boolean is not a number")
    if isinstance(v, Integral):
        f = Fraction(int(v))
    elif isinstance(v, Fraction):
        f = v
    elif isinstance(v, Decimal):
        if not v.is_finite():
            raise ValueError(f"{what}: {v} is not finite")
        f = Fraction(v)
    elif isinstance(v, str):
        f = Fraction(Decimal(v))
    else:
        raise TypeError(f"{what}: exact numbers only (int, Fraction, Decimal, str), got {type(v).__name__}")
    if not (_I64[0] <= f.numerator <= _I64[1] and f.denominator <= _I64[1]):
        raise OverflowError(f"{what} {f} does not fit an int64 rational")
    return f


def _q(f):
    f = Fraction(f)
    return _Q(f.numerator, f.denominator)


def _fq(q):
    return Fraction(int(q.numerator), int(q.denominator))


# ------------------------------------------------------------------ expressions
class Expr:
    """A node of an expression DAG (built by operators; shared subexpressions keep their identity, so x - x cancels)."""
    __slots__ = ("op", "args", "k", "name")
    __hash__ = object.__hash__

    def __init__(self, op, args=(), k=None, name=None):
        self.op, self.args, self.k, self.name = op, tuple(args), k, name

    @property
    def boolean(self):
        return self.op in _BOOL

    def _num(self, o):
        if isinstance(o, Expr):
            if o.boolean:
                raise TypeError("a boolean expression is not a number")
            return o
        return const(o)

    def _need_num(self):
        if self.boolean:
            raise TypeError("a boolean expression is not a number")
        return self

    def __add__(self, o): return Expr("add", (self._need_num(), self._num(o)))
    def __radd__(self, o): return Expr("add", (self._num(o), self._need_num()))
    def __sub__(self, o): return Expr("sub", (self._need_num(), self._num(o)))
    def __rsub__(self, o): return Expr("sub", (self._num(o), self._need_num()))
    def __mul__(self, o): return Expr("mul", (self._need_num(), self._num(o)))
    def __rmul__(self, o): return Expr("mul", (self._num(o), self._need_num()))
    def __neg__(self): return Expr("neg", (self._need_num(),))
    def __abs__(self): return Expr("abs", (self._need_num(),))

    def __truediv__(self, k):
        """division by a nonzero exact constant (a zero divisor is refused natively)"""
        if isinstance(k, Expr):
            raise TypeError("only a division by a constant is supported (scalar division)")
        return Expr("div", (self._need_num(),), k=_frac(k, "divisor"))

    def __mod__(self, k):
        """x - k floor(x / k) for a positive exact constant k (Python's %, not C's)"""
        if isinstance(k, Expr):
            raise TypeError("only a constant modulus is supported")
        return Expr("mod", (self._need_num(),), k=_frac(k, "modulus"))

    def sqr(self): return Expr("sqr", (self._need_num(),))
    def __lt__(self, o): return Expr("lt", (self._need_num(), self._num(o)))
    def __le__(self, o): return Expr("le", (self._need_num(), self._num(o)))
    def __gt__(self, o): return Expr("gt", (self._need_num(), self._num(o)))
    def __ge__(self, o): return Expr("ge", (self._need_num(), self._num(o)))
    def eq(self, o): return Expr("eq", (self._need_num(), self._num(o)))
    def ne(self, o): return Expr("ne", (self._need_num(), self._num(o)))
    __eq__ = eq
    __ne__ = ne

    def _need_bool(self, o=None):
        for e in (self,) + ((o,) if o is not None else ()):
            if not isinstance(e, Expr) or not e.boolean:
                raise TypeError("AND / OR / NOT take boolean expressions")
        return self

    def __and__(self, o): self._need_bool(o); return Expr("and", (self, o))
    def __or__(self, o): self._need_bool(o); return Expr("or", (self, o))
    def __invert__(self): self._need_bool(); return Expr("not", (self,))
    def is_null(self): return Expr("is_null", (self,))
    def not_null(self): return Expr("not_null", (self,))

    def __bool__(self):
        raise TypeError("an expression has no truth value (use & | ~ and evaluate it)")

    def __repr__(self):
        if self.op == "col":
            return self.name
        if self.op == "const":
            return str(self.k)
        if self.op in ("div", "mod"):
            return f"{self.op}({self.args[0]!r}, {self.k})"
        return f"{self.op}({', '.join(repr(a) for a in self.args)})"


def column(name):
    """a value column of the state (its name in the state's column list)"""
    return Expr("col", name=str(name))


def const(value):
    """an exact constant (int, Fraction, Decimal or a decimal string; it must fit an int64 rational)"""
    return Expr("const", k=_frac(value))


def min_(a, b):
    """the smaller value (NULL when either is NULL; unlike SQL's LEAST, which skips NULLs)"""
    a = a if isinstance(a, Expr) else const(a)
    return Expr("min", (a._need_num(), a._num(b)))


def max_(a, b):
    a = a if isinstance(a, Expr) else const(a)
    return Expr("max", (a._need_num(), a._num(b)))


class Program:
    """An expression compiled into native nodes (topological order, shared subexpressions once).

    columns: the state's columns (the program reads them by name); carrier "exact" (int64 rationals, precision None)
    or "binary64" (outward-rounded doubles, precision 2..53 coefficient bits); rank: terms per affine form (1..16);
    budget: the error budget (an exact number: the approximation radius a form may carry; None: unlimited);
    domains: the optional domains ("affine", "congruence"; intervals and nullness always)."""

    def __init__(self, expr, columns, *, carrier="exact", precision=None, rank=MAX_TERMS, budget=None,
                 domains=("affine", "congruence")):
        if not isinstance(expr, Expr):
            raise TypeError("a Program compiles an Expr")
        if carrier not in CARRIERS:
            raise ValueError(f"carrier is one of {', '.join(CARRIERS)}")
        self.columns = [str(c) for c in columns]
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("columns: one or more distinct names")
        if precision is None:
            precision = 0 if carrier == "exact" else 53
        self.carrier, self.precision, self.rank = carrier, int(precision), int(rank)
        self.budget = None if budget is None else _frac(budget, "error budget")
        if self.budget is not None and self.budget < 0:
            raise ValueError("the error budget is >= 0")
        unknown = set(domains) - set(DOMAINS)
        if unknown:
            raise ValueError(f"unknown domains {sorted(unknown)} (affine, congruence)")
        self.domains = tuple(domains)
        order, index = [], {}
        stack = [(expr, False)]
        while stack:                                     # iterative post-order (deep expressions)
            e, done = stack.pop()
            if id(e) in index:
                continue
            if done:
                index[id(e)] = len(order); order.append(e); continue
            stack.append((e, True))
            for a in reversed(e.args):
                if id(a) not in index:
                    stack.append((a, False))
        self.exprs = order
        n = len(order)
        self._nodes = (_Node * max(n, 1))()
        for i, e in enumerate(order):
            nd = self._nodes[i]
            nd.op = OPS[e.op]
            nd.a = nd.b = -1
            nd.k = _Q(0, 1)
            if e.op == "col":
                if e.name not in self.columns:
                    raise ValueError(f"column {e.name!r} is not one of {self.columns}")
                nd.a = self.columns.index(e.name)
            elif e.op == "const":
                nd.k = _q(e.k)
            else:
                nd.a = index[id(e.args[0])]
                if len(e.args) == 2:
                    nd.b = index[id(e.args[1])]
                if e.op in ("div", "mod"):
                    nd.k = _q(e.k)
        self.root = index[id(expr)]
        raw = _Program()
        raw.nnodes, raw.ncols, raw.carrier, raw.precision = n, len(self.columns), CARRIERS[carrier], self.precision
        raw.rank, raw.root, raw.reserved = self.rank, self.root, 0
        raw.domains = sum(DOMAINS[d] for d in set(self.domains))
        raw.budget = _Q(0, 0) if self.budget is None else _q(self.budget)
        raw.nodes = C.cast(self._nodes, C.c_void_p)
        self.raw = raw
        self.boolean = expr.boolean
        self.check()

    def check(self):
        """the native validation and its needs (circuit = nodes, variables = columns, support = rank)"""
        need = _Need()
        _check(_call("tr_unc_program_check", C.byref(self.raw), None, C.byref(need)), what="program")
        return {r: int(need.need[i]) for i, r in enumerate(RESOURCES) if need.need[i]}

    def node(self, expr):
        """the node index of a subexpression of this program"""
        for i, e in enumerate(self.exprs):
            if e is expr:
                return i
        raise ValueError("not a node of this program")

    def __len__(self):
        return len(self.exprs)

    def __repr__(self):
        return f"Program({self.exprs[self.root]!r}, carrier={self.carrier}, rank={self.rank}, nodes={len(self)})"


# ------------------------------------------------------------------ results
def _flags(f):
    return [name for name, bit in FLAGS.items() if f & bit]


def _null(n):
    return dict(maybe_null=bool(n & 1), maybe_value=bool(n & 2))


def _outcomes(o):
    return {name for name, bit in (("true", 1), ("false", 2), ("null", 4)) if o & bit}


def _interval(r, carrier):
    if not r.null & 2:
        return None
    if carrier == "exact":
        lo = None if r.flags & 1 else _fq(r.lo)
        hi = None if r.flags & 2 else _fq(r.hi)
        return (lo, hi)
    return (float(r.dlo), float(r.dhi))


def _congruence(r):
    if not r.null & 2 or r.flags & 8:
        return None
    return (_fq(r.cr), _fq(r.cm))


def _result(r, carrier, backend="cpu"):
    out = dict(kind=("numeric", "boolean")[r.kind - 1], decision=DECISIONS[r.decision], rows=int(r.rows),
               null=_null(r.null), flags=_flags(r.flags), carrier=carrier, backend=backend)
    if r.kind == 2:
        out["outcomes"] = _outcomes(r.outcomes)
        return out
    out["interval"] = _interval(r, carrier)
    out["congruence"] = _congruence(r)
    out["zero"] = ZERO[r.zero]
    if r.null & 2 and not r.flags & 4:
        if carrier == "exact":
            out["form"] = dict(center=_fq(r.center), radius=_fq(r.radius), error=_fq(r.error), terms=int(r.nterms),
                               approximation_terms=int(r.fresh))
        else:
            out["form"] = dict(center=float(r.dcenter), radius=float(r.dradius), error=float(r.derror),
                               terms=int(r.nterms), approximation_terms=int(r.fresh))
    else:
        out["form"] = None
    out["_raw"] = r
    return out


def _value(v, carrier, names):
    out = dict(kind=("numeric", "boolean")[v.kind - 1], null=_null(v.null), flags=_flags(v.flags))
    if v.kind == 2:
        out["outcomes"] = _outcomes(v.outcomes)
        return out
    out["interval"] = _interval(v, carrier)
    out["congruence"] = _congruence(v)
    out["zero"] = ZERO[v.zero]
    if v.null & 2 and not v.flags & 4:
        terms = []
        for i in range(v.nterms):
            t = v.term[i]
            sym = int(t.sym)
            label = names[sym] if sym < FRESH and names and sym < len(names) else (sym if sym < FRESH else f"node {sym - FRESH}")
            coef = Fraction(int(t.num), int(t.den)) if carrier == "exact" else float(t.d)
            terms.append((sym, label, coef))
        out["form"] = dict(center=_fq(v.center) if carrier == "exact" else float(v.dcenter), terms=terms)
    else:
        out["form"] = None
    return out


class Evaluation:
    """the per-group results of one evaluation: .results (list of dicts), .decisions (int32 array), .backend"""

    def __init__(self, raw, carrier, backend, device):
        self.raw, self.carrier, self.backend, self.device = raw, carrier, backend, device
        self.results = [_result(r, carrier, backend) for r in raw]
        self.decisions = np.array([r.decision for r in raw], dtype=np.int32)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, g):
        return self.results[g]

    def counts(self):
        return {name: int((self.decisions == k).sum()) for k, name in DECISIONS.items()}

    def same(self, other):
        """bit-for-bit equality of the native results"""
        return len(self.raw) == len(other.raw) and all(bytes(a) == bytes(b) for a, b in zip(self.raw, other.raw))


def _stats(s):
    out = {name: int(getattr(s, name)) for name, _ in _Stats._fields_}
    out["first_overflow"] = None if s.first_overflow == (1 << 64) - 1 else int(s.first_overflow)
    return out


# ------------------------------------------------------------------ states
def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _input(spec, scale=0):
    """an input from a dict (rows, present, min, max, modulus, scale)"""
    d = dict(spec)
    return _Input(int(d.get("rows", 0)), int(d.get("present", d.get("rows", 0))), int(d.get("min", 0)),
                  int(d.get("max", 0)), int(d.get("modulus", 0)), int(d.get("scale", scale)), 0)


class State:
    """Abstract inputs per group and column: rows, present rows, min, max, gcd of differences (native)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use State.from_scope(...) or State.from_inputs(...)")

    @classmethod
    def _wrap(cls, handle, columns, scope=None):
        self = object.__new__(cls)
        self._handle, self.columns, self._scope = handle, tuple(columns), scope
        return self

    @classmethod
    def from_scope(cls, scope, columns=None, *, budget=None):
        """the state of an FND-004 scope (math_bank.Scope) over its columns (names; default every column)"""
        scope.require_fresh("abstract state")
        columns = list(scope.names if columns is None else columns)
        cols = (C.c_int32 * max(len(columns), 1))()
        for j, name in enumerate(columns):
            if name not in scope.names:
                raise ValueError(f"{name!r} is not a column of the scope ({', '.join(scope.names)})")
            cols[j] = scope.names.index(name)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_unc_state_from_scope", scope._handle, cols, len(columns), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "abstract state"), columns, scope)

    @classmethod
    def from_inputs(cls, groups, columns, *, budget=None):
        """groups: one list per group of one dict per column {rows, present, min, max, modulus, scale}"""
        columns = list(columns)
        groups = list(groups)
        if not groups or any(len(g) != len(columns) for g in groups):
            raise ValueError("one input per column in every group")
        arr = (_Input * (len(groups) * len(columns)))()
        for g, row in enumerate(groups):
            for j, spec in enumerate(row):
                arr[g * len(columns) + j] = _input(spec)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_unc_state_from_inputs", len(groups), len(columns), arr, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "abstract state"), columns)

    def _open(self):
        if not self._handle:
            raise ValueError("the state is closed")
        return self._handle

    def info(self):
        out = _StateInfo()
        _check(_call("tr_unc_state_get_info", self._open(), C.byref(out)), what="state")
        return dict(groups=int(out.ngroups), bytes=int(out.bytes), columns=int(out.ncols), scoped=bool(out.scoped),
                    scope_columns=[int(out.cols[j]) for j in range(out.ncols)])

    @property
    def groups(self):
        return self.info()["groups"]

    def inputs(self, g):
        """the inputs of group g, one dict per column"""
        info = self.info()
        if not 0 <= g < info["groups"]:
            raise ValueError("group out of range")
        ptr = _call("tr_unc_state_inputs", self._open())
        n = info["columns"]
        arr = (_Input * n).from_address(ptr + g * n * C.sizeof(_Input))
        return [dict(rows=int(a.rows), present=int(a.present), min=int(a.min), max=int(a.max),
                     modulus=int(a.modulus), scale=int(a.scale)) for a in arr]

    def _inputs_array(self):
        info = self.info()
        ptr = _call("tr_unc_state_inputs", self._open())
        n = info["groups"] * info["columns"]
        return (_Input * n).from_address(ptr), info

    def _program(self, prog):
        if list(prog.columns) != list(self.columns):
            raise ValueError(f"the program reads columns {prog.columns}, the state has {list(self.columns)}")
        return prog

    def fresh(self, scope):
        """FND-007's freshness of the recorded scope against the current scope: (fresh, reasons)"""
        mc = _mc()
        st = mc._Staleness()
        rc = _call("tr_unc_state_fresh", self._open(), scope._handle, C.byref(st))
        if rc == -13:
            return False, mc._reasons(st.reasons)
        _check(rc, what="state freshness")
        return True, []

    def require_fresh(self, scope, what="abstract state"):
        ok, reasons = self.fresh(scope)
        if not ok:
            err = StaleStateError(f"{what}: the scope changed since the state was built ({', '.join(reasons)})")
            err.reasons = reasons
            raise err

    def evaluate(self, prog, *, first=0, count=None, backend="cpu", budget=None):
        """per-group results of the program (CPU or the CUDA kernel: one work item per group)"""
        self._program(prog)
        groups = self.groups
        count = groups - first if count is None else int(count)
        out = (_Result * max(count, 1))()
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_unc_eval", self._open(), C.byref(prog.raw), int(first), count, out,
                         C.byref(_budget(budget)), C.byref(refusal)), refusal, "abstract evaluation")
        elif backend == "cuda":
            arr, info = self._inputs_array()
            if first != 0 or count != groups:
                raise ValueError("the CUDA path evaluates every group")
            g = _gpu()
            g.run("tr_mgpu_unc_eval", arr, groups, info["columns"], C.byref(prog.raw), out, C.byref(_budget(budget)),
                  C.byref(refusal), refusal=refusal, what="abstract evaluation")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        name, device = _backend_name(backend)
        return Evaluation(list(out)[:count], prog.carrier, name, device)

    def trace(self, prog, group):
        """every node's abstract value for one group (forms with their terms)"""
        self._program(prog)
        vals = (_Value * len(prog))()
        _check(_call("tr_unc_trace", self._open(), C.byref(prog.raw), int(group), vals, len(prog)), what="trace")
        return [_value(v, prog.carrier, prog.columns) for v in vals]

    def filter(self, scope, prog, *, backend="cpu", budget=None):
        """the rows of the scope whose predicate is TRUE: groups decided by the proof (ALL kept, NONE pruned), the
        rows of UNKNOWN groups evaluated exactly; (bitmap words, stats)"""
        self._program(prog)
        if not prog.boolean:
            raise ValueError("the filter needs a boolean program (a predicate)")
        nw = (scope.rows + 63) // 64
        bits = np.zeros(max(nw, 1), dtype=np.uint64)
        st, refusal = _Stats(), _Refusal()
        if backend == "cpu":
            rc = _call("tr_unc_filter", self._open(), scope._handle, C.byref(prog.raw), bits.ctypes.data, C.byref(st),
                       C.byref(_budget(budget)), C.byref(refusal))
        elif backend == "cuda":
            self.require_fresh(scope, "filter")
            ev = self.evaluate(prog, backend="cuda", budget=budget)
            info = self.info()
            cols = (C.c_int32 * max(info["columns"], 1))(*info["scope_columns"])
            view = scope.view()
            dec = np.ascontiguousarray(ev.decisions, dtype=np.int32)
            rc = _gpu_code("tr_mgpu_unc_rows", C.byref(view), cols, C.byref(prog.raw), dec.ctypes.data,
                           bits.ctypes.data, C.byref(st), C.byref(_budget(budget)), C.byref(refusal))
            if rc in (0, -3):
                c = ev.counts()
                st.groups_all, st.groups_none, st.groups_unknown, st.groups_empty = (c["all"], c["none"],
                                                                                     c["unknown"], c["empty"])
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        stats = _stats(st)
        _raise(rc, refusal, "filter", stats, backend)
        name, device = _backend_name(backend)
        stats.update(backend=name, device=device)
        return bits[:nw], stats

    def certify(self, scope, prog, fact, *, modulus=0, budget=None):
        """an FND-007 snapshot certificate of a fact the analysis proved for the program's root in every group:
        "not_null" (a column), "congruence" (modulo a prime), "constant"; FND-007 re-verifies it on every row"""
        mc = _mc()
        self._program(prog)
        if fact not in FACTS:
            raise ValueError(f"fact is one of {', '.join(FACTS)}")
        ev, error, refusal = mc._Evidence(), C.c_int32(), _Refusal()
        h = _call("tr_unc_certify", self._open(), scope._handle, C.byref(prog.raw), FACTS[fact], int(modulus),
                  C.byref(ev), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        if error.value == -6 and not h:
            if ev.checked or ev.violations:
                return mc._made(h, error, refusal, f"{fact} certificate", ev)
            from .math_domains import DomainError
            raise DomainError(f"{fact}: the analysis did not prove it in every group")
        h = mc._made(h, error, refusal, f"{fact} certificate", ev)
        return mc.Certificate._wrap(h, scope.names)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_unc_state_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "State(closed)"
        i = self.info()
        return f"State(groups={i['groups']}, columns={list(self.columns)}, scoped={i['scoped']})"


# ------------------------------------------------------------------ module functions
def rows(scope, prog, columns=None, *, decisions=None, backend="cpu", budget=None):
    """the exact evaluation of a predicate program on every selected row of a scope (SQL NULLs): (bitmap, stats);
    columns: the scope columns of the program's columns (default the same names); decisions: per group (from an
    evaluation of the same scope's state) or None"""
    columns = list(prog.columns if columns is None else columns)
    if len(columns) != len(prog.columns):
        raise ValueError("one scope column per program column")
    cols = (C.c_int32 * max(len(columns), 1))()
    for j, name in enumerate(columns):
        if name not in scope.names:
            raise ValueError(f"{name!r} is not a column of the scope")
        cols[j] = scope.names.index(name)
    nw = (scope.rows + 63) // 64
    bits = np.zeros(max(nw, 1), dtype=np.uint64)
    dec = None if decisions is None else np.ascontiguousarray(decisions, dtype=np.int32)
    if dec is not None and len(dec) != scope.groups:
        raise ValueError("one decision per group of the scope")
    st, refusal = _Stats(), _Refusal()
    if backend == "cpu":
        rc = _call("tr_unc_rows", scope._handle, cols, C.byref(prog.raw), None if dec is None else dec.ctypes.data,
                   bits.ctypes.data, C.byref(st), C.byref(_budget(budget)), C.byref(refusal))
    elif backend == "cuda":
        view = scope.view()
        rc = _gpu_code("tr_mgpu_unc_rows", C.byref(view), cols, C.byref(prog.raw),
                       None if dec is None else dec.ctypes.data, bits.ctypes.data, C.byref(st),
                       C.byref(_budget(budget)), C.byref(refusal))
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    stats = _stats(st)
    _raise(rc, refusal, "rows", stats, backend)
    name, device = _backend_name(backend)
    stats.update(backend=name, device=device)
    return bits[:nw], stats


def symbolic(prog, node=None):
    """the symbolic affine form of a node (columns as exact symbols): (exact, center, {column: coefficient},
    approximation terms)"""
    node = prog.root if node is None else int(node)
    v, exact = _Value(), C.c_int32()
    _check(_call("tr_unc_symbolic", C.byref(prog.raw), node, C.byref(v), C.byref(exact)), what="symbolic form")
    if v.kind != 1:
        return dict(exact=False, kind="boolean")
    if v.flags & 4:
        return dict(exact=False, kind="numeric", center=None, coefficients=None, approximation=None)
    coefs, approx = {}, 0
    for i in range(v.nterms):
        t = v.term[i]
        if t.sym < FRESH:
            coefs[prog.columns[t.sym]] = Fraction(int(t.num), int(t.den))
        else:
            approx += 1
    return dict(exact=bool(exact.value), kind="numeric", center=_fq(v.center), coefficients=coefs,
                approximation=approx)


def identity(prog, node=None, *, budget=None):
    """the FND-007 LAW certificate (ring identity) that the node equals its exact affine form: D E - D L = 0 in Z[x]
    (DomainError when the symbolic form is not exact; UnsupportedError for a non-polynomial operation)"""
    mc = _mc()
    node = prog.root if node is None else int(node)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_unc_identity", C.byref(prog.raw), node, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    if error.value == -6:
        from .math_domains import DomainError
        raise DomainError("not an exact affine identity (the symbolic form carries an approximation term)")
    h = mc._made(h, error, refusal, "identity")
    used = [c for c in prog.columns if any(e.op == "col" and e.name == c for e in prog.exprs)] or prog.columns[:1]
    return mc.Certificate._wrap(h, used)


def meet(a, b):
    """the CRT meet of two classes (r, m) (value in r + m Z; m = 0: exactly r): (r, m), or None when disjoint"""
    r1, m1 = (_frac(v, "congruence") for v in a)
    r2, m2 = (_frac(v, "congruence") for v in b)
    r, m = _Q(), _Q()
    rc = _call("tr_unc_cong_meet", _q(r1), _q(m1), _q(r2), _q(m2), C.byref(r), C.byref(m))
    if rc == -10:
        return None
    _check(rc, what="congruence meet")
    return _fq(r), _fq(m)


def sum_fits(result, bits=64, unit=1):
    """the overflow decision of SUM over a group of a numeric result: every partial sum of value / unit fits a
    signed `bits`-bit accumulator (proven), with the bounds of the partial sums in units"""
    raw = result["_raw"] if isinstance(result, dict) else result
    out = _Sum()
    _check(_call("tr_unc_sum_fits", C.byref(raw), int(bits), _q(_frac(unit, "unit")), C.byref(out)), what="sum bound")
    lo = (int(out.lo.hi) << 64) | (int(out.lo.lo) & ((1 << 64) - 1))
    hi = (int(out.hi.hi) << 64) | (int(out.hi.lo) & ((1 << 64) - 1))
    return dict(proven=bool(out.proven), integral=bool(out.integral), bits=int(out.bits),
                bounds=(lo, hi) if out.bounded else None)


def need(prog, groups, *, backend="cpu"):
    """the needs of an evaluation of `groups` groups (CPU or CUDA device copies)"""
    raw = _Need()
    _check(_call("tr_unc_eval_need", C.byref(prog.raw), int(groups), len(prog.columns),
                 0 if backend == "cpu" else 1, C.byref(raw)), what="evaluation needs")
    return {r: int(raw.need[i]) for i, r in enumerate(RESOURCES) if raw.need[i]}
