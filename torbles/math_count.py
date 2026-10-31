# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""COUNT-001: observed-domain constraint counts and witnesses, natively on the CPU or with the CUDA kernels
(torbles/math_count.c, math_count_gpu.cu).

    from torbles import math_bank as mbk, math_bool as mb, math_count as mc
    x = mb.variables(3)
    cs = mc.Constraints(3, [mc.implies(x[0], x[1]), mc.equiv(x[1], x[0] | x[2]), mc.excludes(x[0], x[2]),
                            mc.holds(x[0] ^ x[2])])                 # a BOOL-001 predicate is a HOLDS constraint
    scope = mbk.Scope.from_table(db["t"], ["a", "b", "d"], rows=mask)
    dom = mc.Domain.observe(scope, [mb.condition("a", ">", 5), mb.condition("b", "is not null"),
                                    mb.condition("d", "bit", 3)], backend="cpu")     # or "cuda"
    ans = mc.count(dom, cs, rows=True, witnesses=5, backend="cpu")                  # or "cuda"
    ans.results        # [{"constraint", "a", "b", "points", "satisfied", "violated", "unknown", "witness",
                       #   "rows" (every satisfying row, with rows=True), "witnesses" (the smallest violating rows)}]
    ans.certificate    # {"kind": "snapshot", "domain": "observed", "points" (rows covered), "support",
                       #  "complete", "observed" (distinct assignments seen), "universe" (2^s), "unobserved", ...}
    ans.fresh(scope); ans.authorize("snapshot", scope); ans.authorize("universe", scope)   # the latter is refused
    uni = mc.count(mc.Domain.universe(3), cs)                    # the 2^k assignment universe: another result

SQL three-valued logic: a constraint is satisfied (TRUE), violated (FALSE) or UNKNOWN at each point; UNKNOWN is never
counted as satisfied or violated. The observed snapshot covers exactly the selected rows of the scope and never the
assignments that do not occur there; the universe count enumerates the 2^k assignments (small k only) and is labelled
as such. Budgets raise AlgebraBudgetError before anything is allocated (the satisfying rows are refused past the output
budget with the exact need), a stale scope StaleStateError. Nothing is computed in Python. Opt-in: `import torbles`
does not load it.
"""
import ctypes as C
from numbers import Integral

import numpy as np

from . import algq
from . import math_bool as _mb
from .math_bank import Scope
from .math_bool import Expr, _Condition, _Domain, _Node, _conditions, _expr
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_certificates import (Freshness, NotAuthorizedError, StaleCertificateError, _Staleness, _freshness,
                                _source)
from .math_domains import DomainError
from .math_errors import RESOURCES, check as _check

__all__ = ["Constraint", "Constraints", "Domain", "Answer", "holds", "implies", "equiv", "excludes", "count",
           "predicate", "domain_need", "count_need", "KINDS", "PATTERN_BITS"]

KINDS = {"holds": 1, "implies": 2, "equiv": 3, "excludes": 4}
_KIND_NAMES = {v: k for k, v in KINDS.items()}
DOMAINS = {1: "observed", 2: "universe"}
CERT_KINDS = {1: "law", 3: "snapshot"}
TARGETS = {"snapshot": 1, "universe": 2}
REFUSALS = {0: None, 1: "stale", 2: "unobserved", 3: "domain", 4: "no scope"}
ROWS = 1
PATTERN_BITS = 24
NONE = (1 << 64) - 1
_NEED_CPU, _NEED_DEVICE, _NEED_ADOPT = 0, 1, 2
P = C.POINTER


class _Constraint(C.Structure):
    _native_ = "tr_count_constraint"
    _fields_ = [("kind", C.c_int32), ("a", C.c_int32), ("b", C.c_int32), ("reserved", C.c_int32)]


class _Request(C.Structure):
    _native_ = "tr_count_request"
    _fields_ = [("flags", C.c_int32), ("reserved", C.c_int32), ("witnesses", C.c_uint64)]


class _Result(C.Structure):
    _native_ = "tr_count_result"
    _fields_ = [("kind", C.c_int32), ("a", C.c_int32), ("b", C.c_int32), ("reserved", C.c_int32)] + [
        (n, C.c_uint64) for n in ("points", "satisfied", "violated", "unknown", "witness", "rows_at", "rows_n",
                                  "wit_at", "wit_n")]


class _SetView(C.Structure):
    _native_ = "tr_count_set_view"
    _fields_ = [(n, C.c_void_p) for n in ("nodes", "deps", "schedule", "constraints", "cones", "cone_at")] + [
        (n, C.c_int32) for n in ("nnodes", "nsched", "k", "nc")]


class _SetInfo(C.Structure):
    _native_ = "tr_count_set_info"
    _fields_ = [(n, C.c_int32) for n in ("k", "nnodes", "nsched", "nc")] + [
        (n, C.c_uint64) for n in ("support", "cones", "bytes")]


class _DomainInfo(C.Structure):
    _native_ = "tr_count_domain_info"
    _fields_ = [(n, C.c_uint64) for n in ("points", "nw", "selected", "bytes")] + [
        (n, C.c_int32) for n in ("domain", "k", "has_unknown", "ncolumns")]


class _Certificate(C.Structure):
    _native_ = "tr_count_certificate"
    _fields_ = [(n, C.c_int32) for n in ("kind", "domain", "backend", "support_size")] + [
        (n, C.c_uint64) for n in ("support", "points", "complete", "observed", "universe", "rows")] + [
        ("nconstraints", C.c_int32), ("exact_selection", C.c_int32)]


class _AnswerView(C.Structure):
    _native_ = "tr_count_answer_view"
    _fields_ = [("results", C.c_void_p), ("points", C.c_void_p), ("witnesses", C.c_void_p),
                ("points_total", C.c_uint64), ("witnesses_total", C.c_uint64), ("nc", C.c_int32),
                ("flags", C.c_int32)]


class _Decision(C.Structure):
    _native_ = "tr_count_decision"
    _fields_ = [(n, C.c_int32) for n in ("authorized", "reason", "target", "domain")] + [
        ("unobserved", C.c_uint64), ("stale", _Staleness)]


for _s, _n in ((_Constraint, 16), (_Request, 16), (_Result, 88), (_SetView, 64), (_SetInfo, 40), (_DomainInfo, 48),
               (_Certificate, 72), (_AnswerView, 48), (_Decision, 56)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, e, pu = C.c_void_p, C.c_int32, P(C.c_int32), P(C.c_uint64)
        sig = {
            "tr_count_set_create": (vp, [i32, P(_Node), i32, P(_Constraint), i32, P(_Budget), P(_Refusal), e]),
            "tr_count_set_need": (i32, [i32, i32, i32, P(_Need)]),
            "tr_count_set_get_view": (i32, [vp, P(_SetView)]),
            "tr_count_set_get_info": (i32, [vp, P(_SetInfo)]),
            "tr_count_set_free": (None, [vp]),
            "tr_count_observe": (vp, [vp, P(_Condition), i32, P(_Budget), P(_Refusal), e]),
            "tr_count_observe_adopt": (vp, [vp, P(_Condition), i32, vp, vp, P(_Budget), P(_Refusal), e]),
            "tr_count_universe": (vp, [i32, P(_Budget), P(_Refusal), e]),
            "tr_count_domain_need": (i32, [vp, P(_Condition), i32, i32, P(_Need)]),
            "tr_count_domain_get_info": (i32, [vp, P(_DomainInfo)]),
            "tr_count_domain_get_domain": (i32, [vp, P(_Domain)]),
            "tr_count_domain_get_record": (vp, [vp]),
            "tr_count_domain_fresh": (i32, [vp, vp, P(_Staleness)]),
            "tr_count_domain_free": (None, [vp]),
            "tr_count_run_need": (i32, [vp, vp, P(_Request), i32, P(_Need)]),
            "tr_count_run": (vp, [vp, vp, P(_Request), P(_Budget), P(_Refusal), e]),
            "tr_count_counts_view": (i32, [P(_Domain), P(_SetView), P(_Budget), P(_Result), P(_Refusal)]),
            "tr_count_plan": (i32, [P(_Result), i32, P(_Request), pu, pu]),
            "tr_count_adopt": (vp, [vp, vp, P(_Request), P(_Result), vp, vp, i32, P(_Budget), P(_Refusal), e]),
            "tr_count_answer_get_view": (i32, [vp, P(_AnswerView)]),
            "tr_count_answer_certificate": (i32, [vp, P(_Certificate)]),
            "tr_count_answer_get_record": (vp, [vp]),
            "tr_count_answer_fresh": (i32, [vp, vp, P(_Staleness)]),
            "tr_count_answer_authorize": (i32, [vp, i32, vp, P(_Decision)]),
            "tr_count_answer_free": (None, [vp]),
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
    gpu = _mb._gpu()                                         # binds tr_mgpu_bool_masks (the condition words)
    gpu.bind("tr_mgpu_count_counts", C.c_int, [P(_Domain), P(_SetView), P(_Budget), P(_Result), P(_Refusal)])
    gpu.bind("tr_mgpu_count_extract", C.c_int, [P(_Domain), P(_SetView), P(_Request), P(_Result), P(_Budget),
                                                C.c_void_p, C.c_void_p, P(_Refusal)])
    return gpu


def _budget(budget):
    return (budget or Budget()).raw


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _label(gpu):
    d = gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


# ---------------------------------------------------------------------- constraints
class Constraint:
    """kind ("holds", "implies", "equiv", "excludes") over math_bool expressions a and b (b None for holds)"""
    __slots__ = ("kind", "a", "b")

    def __init__(self, kind, a, b=None):
        if kind not in KINDS:
            raise ValueError(f"a constraint kind is one of {', '.join(KINDS)}")
        if (kind == "holds") != (b is None):
            raise ValueError("holds takes one predicate, the other kinds two")
        self.kind, self.a, self.b = kind, _expr(a), None if b is None else _expr(b)

    def __repr__(self):
        return f"{self.kind}({self.a!r}{'' if self.b is None else ', ' + repr(self.b)})"


def holds(a):
    """the predicate a itself (satisfied where TRUE, violated where FALSE)"""
    return Constraint("holds", a)


def implies(a, b):
    """a -> b = NOT a OR b: violated where a is TRUE and b FALSE"""
    return Constraint("implies", a, b)


def equiv(a, b):
    """a <-> b = NOT (a XOR b): violated where both are known and differ"""
    return Constraint("equiv", a, b)


def excludes(a, b):
    """NOT (a AND b): violated where both are TRUE"""
    return Constraint("excludes", a, b)


def _compile_many(exprs):
    """the nodes of several expression DAGs (post-order; a Python object reached twice is one node) and each root"""
    index, nodes = {}, []
    for root in exprs:
        stack = [(root, False)]
        while stack:
            e, ready = stack.pop()
            if not isinstance(e, Expr):
                raise TypeError(f"{e!r} is not a predicate")
            if id(e) in index:
                continue
            kids = [] if e.op in (_mb.CONST, _mb.VAR) else ([e.a] if e.op == _mb.NOT else [e.a, e.b])
            if not ready:
                stack.append((e, True))
                stack.extend((c, False) for c in reversed(kids) if id(c) not in index)
                continue
            a = e.a if e.op in (_mb.CONST, _mb.VAR) else index[id(e.a)]
            b = index[id(e.b)] if e.op in (_mb.AND, _mb.OR, _mb.XOR) else -1
            index[id(e)] = len(nodes)
            nodes.append((e.op, a, b))
    return nodes, [index[id(e)] for e in exprs]


class Constraints:
    """An immutable native constraint set over k conditions: one shared predicate DAG and its constraints."""

    def __init__(self, k, constraints, *, budget=None):
        cs = [holds(c) if isinstance(c, Expr) else c for c in constraints]
        for c in cs:
            if not isinstance(c, Constraint):
                raise TypeError(f"{c!r} is not a constraint (use holds, implies, equiv, excludes)")
        operands = [c.a for c in cs] + [c.b for c in cs if c.b is not None]
        nodes, roots = _compile_many(operands)
        it = iter(roots[len(cs):])
        triples = [(KINDS[c.kind], roots[j], next(it) if c.b is not None else -1) for j, c in enumerate(cs)]
        self._create(int(k), nodes, triples, budget)
        self.constraints = tuple(cs)

    @classmethod
    def from_nodes(cls, k, nodes, constraints, *, budget=None):
        """nodes: math_bool (op, a, b) triples; constraints: (kind, a, b) with node indices (b -1 or None for holds)"""
        self = object.__new__(cls)
        triples = []
        for n in nodes:
            op, a, b = (tuple(n) + (-1, -1))[:3]
            triples.append((_mb._OPS[op] if isinstance(op, str) else int(op), int(a), int(b)))
        cons = []
        for c in constraints:
            kind, a, b = (tuple(c) + (-1,))[:3]
            cons.append((KINDS[kind] if isinstance(kind, str) else int(kind), int(a), -1 if b is None else int(b)))
        self._create(int(k), triples, cons, budget)
        self.constraints = None
        return self

    def _create(self, k, nodes, cons, budget):
        self._handle = None
        arr = (_Node * max(len(nodes), 1))()
        for i, (op, a, b) in enumerate(nodes):
            for v in (op, a, b):
                if not -(1 << 31) <= v < (1 << 31):
                    raise ValueError("node fields are int32")
            arr[i] = _Node(op, a, b, 0)
        carr = (_Constraint * max(len(cons), 1))()
        for j, (kind, a, b) in enumerate(cons):
            for v in (kind, a, b):
                if not -(1 << 31) <= v < (1 << 31):
                    raise ValueError("constraint fields are int32")
            carr[j] = _Constraint(kind, a, b, 0)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_count_set_create", k, arr, len(nodes), carr, len(cons), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        self._handle = _made(h, error, refusal, "constraint set")
        self.k, self.nodes, self.triples = k, tuple(nodes), tuple(cons)

    def _open(self):
        if not self._handle:
            raise ValueError("the constraint set is closed")

    def view(self):
        self._open()
        v = _SetView()
        _check(_call("tr_count_set_get_view", self._handle, C.byref(v)), what="constraint set view")
        return v

    @property
    def info(self):
        self._open()
        i = _SetInfo()
        _check(_call("tr_count_set_get_info", self._handle, C.byref(i)), what="constraint set")
        return dict(k=int(i.k), nodes=int(i.nnodes), evaluated=int(i.nsched), constraints=int(i.nc),
                    support=tuple(j for j in range(int(i.k)) if (int(i.support) >> j) & 1), cones=int(i.cones),
                    bytes=int(i.bytes))

    def __len__(self):
        return len(self.triples)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_count_set_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Constraints(k={self.k}, {len(self.triples)} constraints)" if self._handle else "Constraints(closed)"


def set_need(k, nodes, constraints):
    """the needs a constraint set checks before allocating (the cones are budgeted once counted)"""
    out = _Need()
    _check(_call("tr_count_set_need", int(k), int(nodes), int(constraints), C.byref(out)), what="set needs")
    return _needs_dict(out)


# ---------------------------------------------------------------------- domains
class Domain:
    """An immutable native domain: the selected rows of a scope with k condition bitmaps and the FND-007 record of
    the scope (observe), or the 2^k assignments of k conditions (universe)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Domain.observe(scope, conditions) or Domain.universe(k)")

    @classmethod
    def _wrap(cls, handle, scope, conditions, backend, device):
        self = object.__new__(cls)
        self._handle, self.scope, self.conditions, self.backend, self.device = handle, scope, conditions, backend, device
        i = self._raw_info()
        self.kind, self.k = DOMAINS[i.domain], int(i.k)
        return self

    @classmethod
    def observe(cls, scope, conditions, *, backend="cpu", budget=None):
        """the observed snapshot: k = len(conditions) math_bool conditions over the selected rows of scope; backend
        "cuda" builds the condition words with BOOL-001's kernel and adopts them"""
        if not isinstance(scope, Scope):
            raise TypeError("scope: a math_bank.Scope")
        if scope.source == "table":
            scope.require_fresh("observed domain")
        conditions = tuple(conditions)
        arr, k = _conditions(conditions, scope.names)
        error, refusal = C.c_int32(), _Refusal()
        if _backend(backend) == "cpu":
            h = _call("tr_count_observe", scope._handle, arr, k, C.byref(_budget(budget)), C.byref(refusal),
                      C.byref(error))
            return cls._wrap(_made(h, error, refusal, "observed domain"), scope, conditions, "cpu", "native CPU")
        gpu = _gpu()
        from . import math_budgets
        math_budgets.check(budget or Budget(), "observed domain on the device", **domain_need(scope, conditions,
                                                                                              backend="cuda"))
        nw = (scope.rows + 63) // 64
        value = np.zeros(max(k * nw, 1), dtype=np.uint64)
        valid = np.zeros(max(k * nw, 1), dtype=np.uint64)
        view = scope.view()
        gpu.run("tr_mgpu_bool_masks", C.byref(view), arr, k, C.byref(_budget(budget)), value.ctypes.data,
                valid.ctypes.data, C.byref(refusal), refusal=refusal, what="condition words on the device")
        h = _call("tr_count_observe_adopt", scope._handle, arr, k, value.ctypes.data, valid.ctypes.data,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        label, device = _label(gpu)
        return cls._wrap(_made(h, error, refusal, "observed domain (device words)"), scope, conditions, label, device)

    @classmethod
    def universe(cls, k, *, budget=None):
        """the 2^k assignments of k two-valued conditions (bit i of an assignment = condition i): not data"""
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k is an integer")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_count_universe", int(k), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "assignment universe"), None, (), "cpu", "native CPU")

    def _open(self):
        if not self._handle:
            raise ValueError("the domain is closed")

    def _raw_info(self):
        self._open()
        i = _DomainInfo()
        _check(_call("tr_count_domain_get_info", self._handle, C.byref(i)), what="domain")
        return i

    @property
    def info(self):
        i = self._raw_info()
        return dict(kind=self.kind, k=int(i.k), points=int(i.points), selected=int(i.selected), words=int(i.nw),
                    bytes=int(i.bytes), has_unknown=bool(i.has_unknown), columns=int(i.ncolumns),
                    backend=self.backend, device=self.device)

    @property
    def points(self):
        """the points that count: the selected rows, or 2^k"""
        return int(self._raw_info().selected)

    def domain(self):
        """the BOOL-001 tr_bool_domain view (pointers into this domain; valid while it lives)"""
        self._open()
        d = _Domain()
        _check(_call("tr_count_domain_get_domain", self._handle, C.byref(d)), what="domain view")
        return d

    def source(self):
        """the FND-007 record of the observed scope (None for the universe)"""
        self._open()
        rec = _call("tr_count_domain_get_record", self._handle)
        return _source(rec) if rec else None

    def fresh(self, scope=None):
        """Freshness of the domain's record against the current scope (default: the scope it was built from, as it
        is now); the universe is always fresh"""
        self._open()
        st = _Staleness()
        current = scope if scope is not None else self.scope
        rc = _call("tr_count_domain_fresh", self._handle, current._handle if current is not None else None,
                   C.byref(st))
        if rc not in (0, -13):
            _check(rc, what="domain freshness")
        return _freshness(st)

    def require_fresh(self, what="domain"):
        if self.scope is not None and self.scope.source == "table":
            self.scope.require_fresh(what)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_count_domain_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Domain(closed)"
        return f"Domain({self.kind}, k={self.k}, points={self.points}, {self.backend})"


def domain_need(scope, conditions, backend="cpu"):
    """the needs a domain build checks: {resource: need} (scope None: the universe of `conditions` = k variables;
    backend "cuda": the device copies of the condition words and the adopted domain, resource by resource the larger)"""
    if scope is None:
        out = _Need()
        _check(_call("tr_count_domain_need", None, None, int(conditions), _NEED_CPU, C.byref(out)),
               what="domain needs")
        return _needs_dict(out)
    arr, k = _conditions(conditions, scope.names)
    targets = (_NEED_CPU,) if _backend(backend) == "cpu" else (_NEED_DEVICE, _NEED_ADOPT)
    best = [0] * len(RESOURCES)
    for t in targets:
        out = _Need()
        _check(_call("tr_count_domain_need", scope._handle, arr, k, t, C.byref(out)), what="domain needs")
        best = [max(a, int(b)) for a, b in zip(best, out.need)]
    return {name: v for name, v in zip(RESOURCES, best) if v}


# ---------------------------------------------------------------------- answers
class Answer:
    """An immutable native answer: the counts, points and witnesses of every constraint, and its certificate."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_count.count(domain, constraints)")

    @classmethod
    def _wrap(cls, handle, domain, constraints, backend, device):
        self = object.__new__(cls)
        self._handle, self.domain, self.constraints, self.backend, self.device = (handle, domain, constraints,
                                                                                  backend, device)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the answer is closed")
        return self._handle

    def _view(self):
        v = _AnswerView()
        _check(_call("tr_count_answer_get_view", self._open(), C.byref(v)), what="answer")
        return v

    @property
    def results(self):
        v = self._view()
        nc = int(v.nc)
        res = (_Result * nc).from_address(v.results) if nc else []

        def words(ptr, at, n):
            if not n:
                return np.zeros(0, dtype=np.uint64)
            return np.ctypeslib.as_array((C.c_uint64 * n).from_address(ptr + 8 * at)).copy()
        out = []
        for r in res:
            out.append(dict(constraint=_KIND_NAMES[r.kind], a=int(r.a), b=None if r.b < 0 else int(r.b),
                            points=int(r.points), satisfied=int(r.satisfied), violated=int(r.violated),
                            unknown=int(r.unknown), witness=None if r.witness == NONE else int(r.witness),
                            rows=words(v.points, int(r.rows_at), int(r.rows_n)) if v.flags & ROWS else None,
                            witnesses=[int(x) for x in words(v.witnesses, int(r.wit_at), int(r.wit_n))],
                            backend=self.backend, device=self.device))
        return out

    @property
    def certificate(self):
        c = _Certificate()
        _check(_call("tr_count_answer_certificate", self._open(), C.byref(c)), what="certificate")
        s = int(c.support_size)
        universe = 1 << s
        return dict(kind=CERT_KINDS[c.kind], domain=DOMAINS[c.domain], verified_by=("cpu", "cuda library")[c.backend],
                    points=int(c.points), rows=int(c.rows), support=tuple(i for i in range(64) if (int(c.support) >> i) & 1),
                    complete=int(c.complete), observed=int(c.observed), universe=universe,
                    unobserved=universe - int(c.observed), covers_universe=int(c.observed) == universe,
                    constraints=int(c.nconstraints), exact_selection=bool(c.exact_selection), backend=self.backend,
                    device=self.device)

    def source(self):
        """the FND-007 record of the rows covered (None for the universe)"""
        rec = _call("tr_count_answer_get_record", self._open())
        return _source(rec) if rec else None

    def fresh(self, scope=None):
        """Freshness (math_certificates.Freshness) of the certificate against the current scope (default: the
        domain's scope as it is now); a universe certificate is always fresh"""
        st = _Staleness()
        current = scope if scope is not None else getattr(self.domain, "scope", None)
        rc = _call("tr_count_answer_fresh", self._open(), current._handle if current is not None else None,
                   C.byref(st))
        if rc not in (0, -13):
            _check(rc, what="answer freshness")
        return _freshness(st)

    def decide(self, target, scope=None):
        """the native decision as a dict (never raises for a refusal): target "snapshot" (a statement about the
        covered rows of scope) or "universe" (about every assignment of the support)"""
        if target not in TARGETS:
            raise ValueError("target is 'snapshot' or 'universe'")
        d = _Decision()
        rc = _call("tr_count_answer_authorize", self._open(), TARGETS[target],
                   scope._handle if scope is not None else None, C.byref(d))
        if rc == -1:
            _check(rc, what="authorization")
        return dict(authorized=bool(d.authorized), reason=REFUSALS[d.reason], target=target,
                    domain=DOMAINS[d.domain], unobserved=None if d.unobserved == NONE else int(d.unobserved),
                    stale=_freshness(d.stale) if d.stale.reasons else None, code=int(rc))

    def authorize(self, target, scope=None):
        """the decision when authorized, else StaleCertificateError (with .reasons) or NotAuthorizedError (with
        .reason: "unobserved" (and .unobserved), "domain", "no scope")"""
        d = self.decide(target, scope)
        if d["authorized"]:
            return d
        if d["reason"] == "stale":
            f = d["stale"]
            err = StaleCertificateError(f"the counted rows changed: {', '.join(f.reasons)}")
            err.reasons, err.first, err.freshness, err.decision = f.reasons, f.first, f, d
            raise err
        what = {"unobserved": f"{d['unobserved']} assignments of the support were never observed",
                "domain": "a universe count covers no table rows", "no scope": "give the current scope"}[d["reason"]]
        err = NotAuthorizedError(f"a {d['domain']} count cannot support a {target} statement: {what}")
        err.reason, err.unobserved, err.decision = d["reason"], d["unobserved"], d
        raise err

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_count_answer_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Answer(closed)"
        c = self.certificate
        return f"Answer({c['domain']}, {c['constraints']} constraints over {c['points']} points, {self.backend})"


def _request(rows, witnesses):
    if isinstance(witnesses, bool) or not isinstance(witnesses, Integral) or not 0 <= int(witnesses) < (1 << 64):
        raise ValueError("witnesses is a count >= 0")
    return _Request(ROWS if rows else 0, 0, int(witnesses))


def count(domain, constraints, *, rows=False, witnesses=1, backend="cpu", budget=None):
    """The exact satisfied / violated / UNKNOWN counts of every constraint over the domain's points, the smallest
    violating point (witness), with rows=True every satisfying point (refused past the output budget with the exact
    need), and the `witnesses` smallest violating points; with the certificate of the points covered."""
    if not isinstance(domain, Domain) or not isinstance(constraints, Constraints):
        raise TypeError("count(Domain, Constraints)")
    domain._open(); constraints._open()
    domain.require_fresh("counts")
    if domain.k != constraints.k:
        raise DomainError(f"the constraints read {constraints.k} conditions, the domain has {domain.k}")
    q = _request(rows, witnesses)
    error, refusal = C.c_int32(), _Refusal()
    if _backend(backend) == "cpu":
        h = _call("tr_count_run", domain._handle, constraints._handle, C.byref(q), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Answer._wrap(_made(h, error, refusal, "constraint counts"), domain, constraints, "cpu", "native CPU")
    gpu = _gpu()
    d, s = domain.domain(), constraints.view()
    nc = len(constraints)
    results = (_Result * nc)()
    gpu.run("tr_mgpu_count_counts", C.byref(d), C.byref(s), C.byref(_budget(budget)), results, C.byref(refusal),
            refusal=refusal, what="constraint counts on the device")
    pt, wt = C.c_uint64(), C.c_uint64()
    _check(_call("tr_count_plan", results, nc, C.byref(q), C.byref(pt), C.byref(wt)), what="output layout")
    points = np.zeros(max(int(pt.value), 1), dtype=np.uint64)
    wits = np.zeros(max(int(wt.value), 1), dtype=np.uint64)
    gpu.run("tr_mgpu_count_extract", C.byref(d), C.byref(s), C.byref(q), results, C.byref(_budget(budget)),
            points.ctypes.data, wits.ctypes.data, C.byref(refusal), refusal=refusal,
            what="points and witnesses on the device")
    h = _call("tr_count_adopt", domain._handle, constraints._handle, C.byref(q), results, points.ctypes.data,
              wits.ctypes.data, 1, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    label, device = _label(gpu)
    return Answer._wrap(_made(h, error, refusal, "adopted counts"), domain, constraints, label, device)


def predicate(domain, expr, *, rows=False, witnesses=1, backend="cpu", budget=None):
    """the counts of one BOOL-001 predicate (a HOLDS constraint): TRUE / FALSE / UNKNOWN points, its TRUE points with
    rows=True, its smallest FALSE points"""
    return count(domain, Constraints(domain.k, [holds(expr)]), rows=rows, witnesses=witnesses, backend=backend,
                 budget=budget)


def count_need(domain, constraints, *, rows=False, witnesses=1, backend="cpu"):
    """the needs checked before counting (the outputs are checked after the counts, with their exact size)"""
    q = _request(rows, witnesses)
    out = _Need()
    _check(_call("tr_count_run_need", domain._handle, constraints._handle, C.byref(q),
                 _NEED_CPU if _backend(backend) == "cpu" else _NEED_DEVICE, C.byref(out)), what="count needs")
    return _needs_dict(out)
