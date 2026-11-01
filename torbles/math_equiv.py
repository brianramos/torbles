# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""META-001: scoped equivalence, implication and containment of query parts, with concrete counterexamples,
natively on the CPU or with the CUDA kernels (torbles/math_equiv.h/.c, math_equiv_gpu.cu).

    from torbles import math_bank as mbk, math_geom as mg, math_equiv as meq
    p0 = meq.cond("x", "<", 5) & meq.cond("y", ">=", 0)                  # BOOL-001 conditions, SQL 3VL
    p1 = ~(meq.cond("x", ">=", 5) | meq.cond("y", "<", 0))
    f, g = mg.poly({(1, 1): 1}, ["x", "y"]), mg.poly({(1, 0): 1, (0, 1): 1}, ["x", "y"])
    s = meq.Set(["x", "y"], predicates=[p0, p1], measures=[f, g], queries=[(0, 0), (1, 1), (0, None)])
    s.info()                                      # distinct conditions, shared DAG nodes, distinct filters/measures
    qs = [meq.predicates(0, 1), meq.predicates(0, 1, relation="implies"), meq.measures(0, 1, nulls="sql"),
          meq.queries(0, 1)]
    r = s.check(scope, qs, backend="cuda")        # SNAPSHOT: exhaustive over the scope's rows (or "cpu")
    r[0]["verdict"], r[0]["label"], r[2]["witness"], r[2]["point"], r.certificate(2), r.authorize(0, scope2)
    d = meq.domain({"x": (-10, 10), "y": (-3, 3)}, nullable=["y"])          # declared box (or {"x": [1, 5, 9]})
    u = s.decide(qs, d)                           # UNIVERSAL / DOMAIN proofs, counterexample points, or UNKNOWN
    answers, stats = s.run(scope, mode="shared")  # COUNT / exact SUM per query; stats["scans"] actually performed
    better, trace = meq.rewrite(catalog, scope, f)                  # relationships.Catalog's tr_optimize, lifted back

Verdicts are "proven", "refuted" (with a witness: the smallest differing row, or a point of the domain) or
"unknown" (why: "budget" with the resource, need and limit, "unsupported", "undecided"); labels say where a proof
holds: "snapshot" (these rows only; authorizes only on this exact scope), "relations" (where its GEOM-001 derived
certificate holds), "domain" (the declared domain: authorizes on a scope whose values lie in it), "universal".
Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_bank as _mbk
from . import math_bool as _mb
from . import math_certificates as _mc
from . import math_geom as _mg
from . import math_ident as _mi
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["Pred", "cond", "column", "TRUE", "FALSE", "UNKNOWN", "Set", "Result", "Domain", "domain", "predicates",
           "measures", "queries", "rewrite", "measure_from_math", "VERDICTS", "LABELS", "METHODS", "MAX_CONDITIONS",
           "MAX_CONE"]

MAX_CONDITIONS, MAX_PARTS, MAX_CONE, MAX_COLUMNS = 64, 4096, 128, 32
NONE = (1 << 64) - 1
PREDICATE, MEASURE, QUERY = 1, 2, 3
_RELATIONS = {"equivalent": 1, "implies": 2}
_SEMANTICS = {"filter": 1, "kleene": 2}
_NULLS = {"sql": 1, "skip": 2}
VERDICTS = {0: "unknown", 1: "proven", 2: "refuted"}
LABELS = {0: None, 1: "snapshot", 2: "domain", 3: "universal", 4: "relations"}
METHODS = {0: None, 1: "rows", 2: "enumerate", 3: "bdd", 4: "normal form", 5: "grid", 6: "reduction", 7: "constant"}
WHY = {0: None, 1: "budget", 2: "unsupported", 3: "undecided"}
AUTH = {0: None, 1: "not proven", 2: "stale", 3: "outside", 4: "relations", 5: "no scope"}
_MODES = {"enumerate": 1, "decide": 2}
_RUNS = {"shared": 1, "separate": 2}
_TRUTH = ("true", "false", "unknown")
_NEED_CPU, _NEED_DEVICE = 0, 1
_BY_CPU, _BY_DEVICE = 0, 1
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Pred(C.Structure):
    _native_ = "tr_equiv_pred"
    _fields_ = [("nconds", C.c_int32), ("nnodes", C.c_int32), ("root", C.c_int32), ("reserved", C.c_int32),
                ("conds", C.c_void_p), ("nodes", C.c_void_p)]


class _Query(C.Structure):
    _native_ = "tr_equiv_query"
    _fields_ = [("filter", C.c_int32), ("measure", C.c_int32)]


class _Question(C.Structure):
    _native_ = "tr_equiv_question"
    _fields_ = [(n, C.c_int32) for n in ("kind", "relation", "semantics", "nulls", "a", "b")]


class _Evidence(C.Structure):
    _native_ = "tr_equiv_evidence"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "checked", "excluded", "differ", "undecided", "witness",
                                          "null_mismatch", "nulls")] + [("cells", C.c_uint64 * 9)]


class _Plan(C.Structure):
    _native_ = "tr_equiv_plan"
    _fields_ = [(n, C.c_int32) for n in ("kind", "relation", "semantics", "nulls", "ra", "rb")] + [
        ("cone", C.c_uint32), ("ncone", C.c_uint32)] + [(n, C.c_int32) for n in ("pair", "hasa", "hasb", "constant")] + [
        ("supa", C.c_uint32), ("supb", C.c_uint32)]


class _Entry(C.Structure):
    _native_ = "tr_equiv_entry"
    _fields_ = [(n, C.c_int32) for n in ("op", "a", "b", "reserved")]


class _CheckView(C.Structure):
    _native_ = "tr_equiv_check_view"
    _fields_ = [("rows", _mbk._View), ("masks", _mb._Domain), ("cand", _mi._CandView), ("nq", C.c_int32),
                ("reserved", C.c_int32), ("nentries", C.c_uint64), ("plans", C.c_void_p), ("entries", C.c_void_p)]


class _Column(C.Structure):
    _native_ = "tr_equiv_column"
    _fields_ = [(n, C.c_int32) for n in ("col", "scale", "nullable", "kind")] + [
        ("lo", C.c_int64), ("hi", C.c_int64), ("nvalues", C.c_uint64), ("values", C.c_void_p)]


class _DomainS(C.Structure):
    _native_ = "tr_equiv_domain"
    _fields_ = [("ncols", C.c_int32), ("mode", C.c_int32), ("cols", C.c_void_p)]


class _Verdict(C.Structure):
    _native_ = "tr_equiv_verdict"
    _fields_ = [(n, C.c_int32) for n in ("verdict", "label", "method", "why", "resource", "point", "r0", "r1")] + [
        (n, C.c_uint64) for n in ("need", "limit", "work", "nodes", "patterns", "witness")]


class _Decision(C.Structure):
    _native_ = "tr_equiv_decision"
    _fields_ = [(n, C.c_int32) for n in ("authorized", "reason", "label", "verdict")] + [
        ("stale", _mc._Staleness), ("cert", _mc._Decision)]


class _SetInfo(C.Structure):
    _native_ = "tr_equiv_set_info"
    _fields_ = [(n, C.c_int32) for n in ("npred", "nmeasure", "nquery", "ncond", "nnodes", "ncols", "r0", "r1")] + [
        (n, C.c_uint64) for n in ("given_preds", "given_conds", "given_nodes", "given_measures", "given_queries",
                                  "filters", "measures", "pairs", "shared_nodes", "bytes")]


class _Answer(C.Structure):
    _native_ = "tr_equiv_answer"
    _fields_ = [("rows", C.c_uint64), ("nonnull", C.c_uint64)] + [
        (n, C.c_int32) for n in ("words", "filter", "measure", "pair")] + [("denominator", C.c_int64),
                                                                          ("sum", C.c_uint64 * 8)]


class _RunStats(C.Structure):
    _native_ = "tr_equiv_run_stats"
    _fields_ = [(n, C.c_uint64) for n in ("scans", "condition_rows", "node_words", "measure_rows", "queries",
                                          "filters", "measures", "pairs", "bytes")]


class _RewriteInfo(C.Structure):
    _native_ = "tr_equiv_rewrite_info"
    _fields_ = [(n, C.c_int32) for n in ("changed", "steps", "explored", "truncated")] + [
        ("before_cost", C.c_double), ("after_cost", C.c_double), ("ncols", C.c_int32), ("reserved", C.c_int32),
        ("col", C.c_int32 * 8), ("nterms", C.c_uint64)]


for _s, _n in ((_Pred, 32), (_Query, 8), (_Question, 24), (_Evidence, 144), (_Plan, 56), (_Entry, 16),
               (_CheckView, 2272), (_Column, 48), (_DomainS, 16), (_Verdict, 80), (_Decision, 112), (_SetInfo, 112),
               (_Answer, 104), (_RunStats, 72), (_RewriteInfo, 80)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_equiv_set_make": (vp, [i32, P(_Pred), i32, P(_mg._GPoly), i32, P(_Query), B, R, e]),
            "tr_equiv_set_get_info": (i32, [vp, P(_SetInfo)]),
            "tr_equiv_set_ids": (i32, [vp, vp, vp, vp, vp, vp]),
            "tr_equiv_set_conditions": (i32, [vp, P(_mb._Condition), i32]),
            "tr_equiv_set_nodes": (i32, [vp, P(_mb._Node), i32]),
            "tr_equiv_set_free": (None, [vp]),
            "tr_equiv_poly_from_math": (i32, [vp, P(C.c_int32), P(C.c_int32), P(C.c_int32), vp, vp, vp, u64, P(u64)]),
            "tr_equiv_check_make": (vp, [vp, P(_Question), i32, vp, vp, vp, B, R, e]),
            "tr_equiv_check_get_view": (i32, [vp, P(_CheckView)]),
            "tr_equiv_check_need": (i32, [vp, i32, P(_Need)]),
            "tr_equiv_check_run": (i32, [vp, P(_Evidence)]),
            "tr_equiv_check_decide": (vp, [vp, vp, P(_Evidence), i32, P(vp), i32, B, R, e]),
            "tr_equiv_check_free": (None, [vp]),
            "tr_equiv_decide": (vp, [vp, P(_Question), i32, P(_DomainS), B, R, e]),
            "tr_equiv_result_count": (i32, [vp]),
            "tr_equiv_result_get": (i32, [vp, i32, P(_Verdict), P(_Evidence)]),
            "tr_equiv_result_point": (i32, [vp, i32, vp, vp, i32]),
            "tr_equiv_result_certificate": (vp, [vp, i32]),
            "tr_equiv_authorize": (i32, [vp, i32, vp, P(_Decision)]),
            "tr_equiv_result_free": (None, [vp]),
            "tr_equiv_family_run": (i32, [vp, vp, i32, B, P(_Answer), P(_RunStats), R]),
            "tr_equiv_rewrite": (i32, [vp, vp, vp, P(_mg._GPoly), P(C.c_int32), i32, P(_RewriteInfo), vp, vp, u64]),
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
    gpu = _mb._gpu()                                  # tr_mgpu_bool_masks
    gpu.bind("tr_mgpu_equiv_check", C.c_int, [P(_CheckView), P(_Budget), P(_Evidence), P(_Refusal)])
    return gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _device_label():
    from . import math_gpu
    d = math_gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


def _opt(v):
    return None if v == NONE else int(v)


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


# ------------------------------------------------------------------ predicates
column = _mb.column
_CONST, _VAR, _NOT, _AND, _OR, _XOR = 1, 2, 3, 4, 5, 6


class Pred:
    """A predicate over named scope columns: cond(...) leaves, TRUE / FALSE / UNKNOWN, and & | ^ ~ (SQL's Kleene
    AND / OR / NOT; XOR is SQL's `a <> b` on booleans). Structurally equal parts are one node of the native DAG."""
    __slots__ = ("op", "a", "b")

    def __init__(self, op, a=-1, b=-1):
        self.op, self.a, self.b = op, a, b

    def __and__(self, o): return Pred(_AND, self, _pred(o))
    def __rand__(self, o): return Pred(_AND, _pred(o), self)
    def __or__(self, o): return Pred(_OR, self, _pred(o))
    def __ror__(self, o): return Pred(_OR, _pred(o), self)
    def __xor__(self, o): return Pred(_XOR, self, _pred(o))
    def __rxor__(self, o): return Pred(_XOR, _pred(o), self)
    def __invert__(self): return Pred(_NOT, self)

    def __bool__(self):
        raise TypeError("a predicate has no Python truth value: combine with & | ^ ~")

    def __repr__(self):
        if self.op == _VAR:
            return repr(self.a)
        if self.op == _CONST:
            return ("FALSE", "TRUE", "UNKNOWN")[self.a]
        if self.op == _NOT:
            return f"~{self.a!r}"
        return f"({self.a!r} {({_AND: '&', _OR: '|', _XOR: '^'})[self.op]} {self.b!r})"


def _pred(o):
    if isinstance(o, Pred):
        return o
    if isinstance(o, bool):
        return TRUE if o else FALSE
    if o is None:
        return UNKNOWN
    raise TypeError(f"{o!r} is not a predicate (use cond(...), TRUE, FALSE, UNKNOWN)")


TRUE, FALSE, UNKNOWN = Pred(_CONST, 1), Pred(_CONST, 0), Pred(_CONST, 2)


def cond(column_name, op, value=None, high=None, *, base=0):
    """a BOOL-001 condition as a predicate leaf: math_bool.condition's operators and exact constants"""
    return Pred(_VAR, _mb.condition(column_name, op, value, high, base=base))


def _compile(p, names):
    """(conditions, nodes, root) of a predicate (post-order; a Python object reached twice is one node)"""
    p = _pred(p)
    index, nodes, conds, cidx, stack = {}, [], [], {}, [(p, False)]
    while stack:
        e, ready = stack.pop()
        if id(e) in index:
            continue
        kids = [] if e.op in (_CONST, _VAR) else ([e.a] if e.op == _NOT else [e.a, e.b])
        if not ready:
            stack.append((e, True))
            stack.extend((k, False) for k in reversed(kids) if id(k) not in index)
            continue
        if e.op == _VAR:
            if id(e.a) not in cidx:
                cidx[id(e.a)] = len(conds)
                conds.append(e.a.native(names))
            a, b = cidx[id(e.a)], -1
        elif e.op == _CONST:
            a, b = e.a, -1
        else:
            a, b = index[id(e.a)], (index[id(e.b)] if e.op in (_AND, _OR, _XOR) else -1)
        index[id(e)] = len(nodes)
        nodes.append((e.op, a, b))
    return conds, nodes, index[id(p)]


def _poly(f):
    if isinstance(f, _mg.Polynomial):
        return f
    if isinstance(f, tuple) and len(f) == 2:
        return _mg.Polynomial(*f)
    if isinstance(f, (int, Fraction)) and not isinstance(f, bool):
        return _mg.Polynomial({(): f} if f else {}, [])
    raise TypeError(f"{f!r}: a measure is a math_geom.Polynomial (math_geom.poly(terms, columns)) or a constant")


# ------------------------------------------------------------------ questions
def predicates(a, b, *, relation="equivalent", semantics="filter"):
    """predicate a vs b: "equivalent" (semantics "filter": the same TRUE rows; "kleene": the same truth value) or
    "implies" (every TRUE row of a is TRUE in b: containment of the filtered rows)"""
    return (PREDICATE, _RELATIONS[relation], _SEMANTICS[semantics], 0, int(a), int(b))


def measures(a, b, *, nulls="sql"):
    """measure a vs b: equal values; nulls "sql" (exactly one side NULL differs) or "skip" (such rows excluded)"""
    return (MEASURE, 1, 0, _NULLS[nulls], int(a), int(b))


def queries(a, b, *, relation="equivalent", nulls="sql"):
    """query a vs b: the same filtered rows and, on them, equal measures ("implies": the rows of a are rows of b)"""
    return (QUERY, _RELATIONS[relation], 0, _NULLS[nulls], int(a), int(b))


def _questions(qs):
    qs = list(qs)
    arr = (_Question * max(len(qs), 1))()
    for i, q in enumerate(qs):
        if not (isinstance(q, tuple) and len(q) == 6):
            raise TypeError("questions come from predicates(...), measures(...) or queries(...)")
        arr[i] = _Question(*q)
    return arr, len(qs)


# ------------------------------------------------------------------ domains
class Domain:
    """A declared domain per column: an integer range (lo, hi) or a finite list of logical values, a decimal scale
    and whether NULL is a value; mode "decide" (BDDs, integer difference logic, Nullstellensatz grids) or
    "enumerate" (every point of the product)."""

    def __init__(self, spec, *, nullable=(), scales=None, mode="decide"):
        self.spec, self.nullable = dict(spec), frozenset(nullable)
        self.scales, self.mode = dict(scales or {}), mode
        if mode not in _MODES:
            raise ValueError(f"mode: one of {', '.join(_MODES)}")
        for n in self.nullable:
            if n not in self.spec:
                raise ValueError(f"nullable column {n!r} has no domain")

    def native(self, names):
        names = list(names)
        cols = (_Column * max(len(self.spec), 1))()
        keep = []
        for i, (name, s) in enumerate(self.spec.items()):
            if name not in names:
                raise ValueError(f"{name!r} is not a column ({', '.join(names)})")
            c = cols[i]
            c.col, c.scale, c.nullable = names.index(name), int(self.scales.get(name, 0)), int(name in self.nullable)
            if isinstance(s, tuple) and len(s) == 2:
                c.kind, c.lo, c.hi, c.nvalues, c.values = 1, int(s[0]), int(s[1]), 0, None
            else:
                v = np.ascontiguousarray([int(x) for x in s], dtype=np.int64)
                keep.append(v)
                c.kind, c.lo, c.hi, c.nvalues, c.values = 2, 0, 0, len(v), v.ctypes.data
        d = _DomainS(len(self.spec), _MODES[self.mode], C.cast(cols, C.c_void_p))
        keep.append(cols)
        return d, keep


def domain(spec, *, nullable=(), scales=None, mode="decide"):
    """Domain({name: (lo, hi) or [values]}, nullable=[names], scales={name: s}, mode="decide" | "enumerate")"""
    return Domain(spec, nullable=nullable, scales=scales, mode=mode)


# ------------------------------------------------------------------ results
def _evidence(e):
    return dict(rows=int(e.rows), selected=int(e.selected), checked=int(e.checked), excluded=int(e.excluded),
                differ=int(e.differ), undecided=int(e.undecided), witness=_opt(e.witness),
                null_mismatch=int(e.null_mismatch), nulls=int(e.nulls),
                cells={(_TRUTH[i // 3], _TRUTH[i % 3]): int(e.cells[i]) for i in range(9) if e.cells[i]})


class Result:
    """The verdicts of questions (snapshot checks or universal decisions); r[i] is a dict, r.certificate(i) the
    FND-007 certificate of a proof, r.authorize(i, scope) whether it authorizes an exact rewrite there."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Set.check(...) or Set.decide(...)")

    @classmethod
    def _wrap(cls, handle, names, questions, backend, device):
        self = object.__new__(cls)
        self._handle, self.names, self.questions = handle, tuple(names), tuple(questions)
        self.backend, self.device = backend, device
        self._items = [self._read(i) for i in range(len(self.questions))]
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the result is closed")
        return self._handle

    def _read(self, i):
        v, e = _Verdict(), _Evidence()
        _check(_call("tr_equiv_result_get", self._open(), i, C.byref(v), C.byref(e)), what="result")
        point = None
        if v.point:
            n = len(self.names)
            vals, pres = np.zeros(max(n, 1), np.int64), np.zeros(max(n, 1), np.uint8)
            _check(_call("tr_equiv_result_point", self._handle, i, vals.ctypes.data, pres.ctypes.data, n), what="point")
            point = {self.names[c]: (int(vals[c]) if pres[c] else None) for c in range(n)}
        out = dict(verdict=VERDICTS[v.verdict], label=LABELS[v.label], method=METHODS[v.method], why=WHY[v.why],
                   budget=(dict(resource=RESOURCES[v.resource], need=int(v.need), limit=int(v.limit))
                           if v.why == 1 else None),
                   work=int(v.work), nodes=int(v.nodes), patterns=int(v.patterns), witness=_opt(v.witness),
                   point=point, evidence=_evidence(e) if e.rows or v.label in (1, 4) else None,
                   certified=bool(_call("tr_equiv_result_certificate", self._handle, i)),
                   backend=self.backend, device=self.device)
        return out

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def certificate(self, i):
        """the FND-007 certificate of question i's proof (an owned clone), or None"""
        h = _call("tr_equiv_result_certificate", self._open(), int(i))
        if not h:
            return None
        error = C.c_int32()
        clone = _mc._call("tr_cert_clone", h, C.byref(error))
        return _mc.Certificate._wrap(_mc._made(clone, error, _Refusal(), "certificate"), self.names, (),
                                     self.backend, self.device)

    def authorize(self, i, scope=None):
        """whether question i's proof authorizes an exact rewrite on scope (None: universal proofs only)"""
        if scope is not None and not isinstance(scope, _mbk.Scope):
            raise TypeError("scope: a math_bank.Scope")
        d = _Decision()
        _check(_call("tr_equiv_authorize", self._open(), int(i), scope._handle if scope is not None else None,
                     C.byref(d)), what="authorization")
        stale = None
        if d.reason == 2 and d.stale.first:
            stale = dict(reasons=int(d.stale.reasons), first=int(d.stale.first),
                         recorded=int(d.stale.recorded), current=int(d.stale.current))
        return dict(authorized=bool(d.authorized), reason=AUTH[d.reason], label=LABELS[d.label],
                    verdict=VERDICTS[d.verdict], stale=stale)

    def close(self):
        if self._handle:
            _call("tr_equiv_result_free", self._handle)
            self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None):
            _call("tr_equiv_result_free", self._handle)
            self._handle = None

    def __repr__(self):
        return f"<META-001 result: {[(r['verdict'], r['label']) for r in self._items]}>"


# ------------------------------------------------------------------ sets
class Set:
    """Query parts over named scope columns, shared natively: predicates (Pred), measures (math_geom.Polynomial or a
    constant) and queries (filter predicate index or None, measure index or None: COUNT)."""

    def __init__(self, names, predicates=(), measures=(), queries=(), *, budget=None):
        self.names = tuple(names)
        if len(self.names) > MAX_COLUMNS or len(set(self.names)) != len(self.names):
            raise ValueError(f"at most {MAX_COLUMNS} distinct column names")
        self.predicates, self.measures = tuple(_pred(p) for p in predicates), tuple(_poly(f) for f in measures)
        self.queries = tuple((None if f is None else int(f), None if m is None else int(m)) for f, m in queries)
        keep = []
        parr = (_Pred * max(len(self.predicates), 1))()
        for i, p in enumerate(self.predicates):
            conds, nodes, root = _compile(p, self.names)
            ca = (_mb._Condition * max(len(conds), 1))(*conds)
            na = (_mb._Node * max(len(nodes), 1))(*[_mb._Node(o, a, b, 0) for o, a, b in nodes])
            keep += [ca, na]
            parr[i] = _Pred(len(conds), len(nodes), root, 0, C.cast(ca, C.c_void_p), C.cast(na, C.c_void_p))
        marr = (_mg._GPoly * max(len(self.measures), 1))()
        for i, f in enumerate(self.measures):
            g, arrays = f.native(self.names)
            marr[i] = g
            keep.append(arrays)
        qarr = (_Query * max(len(self.queries), 1))()
        for i, (f, m) in enumerate(self.queries):
            qarr[i] = _Query(-1 if f is None else f, -1 if m is None else m)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_equiv_set_make", len(self.predicates), parr, len(self.measures), marr, len(self.queries), qarr,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        self._handle = _made(h, error, refusal, "query parts")
        i = self._raw_info()
        self.ncond, self.nnodes = int(i.ncond), int(i.nnodes)

    def _open(self):
        if not self._handle:
            raise ValueError("the set is closed")
        return self._handle

    def _raw_info(self):
        out = _SetInfo()
        _check(_call("tr_equiv_set_get_info", self._open(), C.byref(out)), what="query parts")
        return out

    def info(self):
        i = self._raw_info()
        return dict(predicates=int(i.npred), measures=int(i.nmeasure), queries=int(i.nquery), conditions=int(i.ncond),
                    nodes=int(i.nnodes), columns=int(i.ncols), given_predicates=int(i.given_preds),
                    given_conditions=int(i.given_conds), given_nodes=int(i.given_nodes),
                    given_measures=int(i.given_measures), given_queries=int(i.given_queries),
                    distinct_filters=int(i.filters), distinct_measures=int(i.measures), distinct_pairs=int(i.pairs),
                    shared_nodes=int(i.shared_nodes), bytes=int(i.bytes))

    def ids(self):
        """the shared ids: per predicate its DAG root, per measure its distinct id, per query its distinct id, filter
        root and measure id (-1: none)"""
        n_p, n_m, n_q = max(len(self.predicates), 1), max(len(self.measures), 1), max(len(self.queries), 1)
        pr, mi = np.zeros(n_p, np.int32), np.zeros(n_m, np.int32)
        qi, qf, qm = np.zeros(n_q, np.int32), np.zeros(n_q, np.int32), np.zeros(n_q, np.int32)
        _check(_call("tr_equiv_set_ids", self._open(), pr.ctypes.data, mi.ctypes.data, qi.ctypes.data, qf.ctypes.data,
                     qm.ctypes.data), what="query parts")
        L = len(self.queries)
        return dict(predicate_roots=pr[:len(self.predicates)].tolist(), measure_ids=mi[:len(self.measures)].tolist(),
                    query_ids=qi[:L].tolist(), query_filters=qf[:L].tolist(), query_measures=qm[:L].tolist())

    def conditions(self):
        arr = (_mb._Condition * max(self.ncond, 1))()
        _check(_call("tr_equiv_set_conditions", self._open(), arr, max(self.ncond, 1)), what="query parts")
        return arr, self.ncond

    def nodes(self):
        arr = (_mb._Node * max(self.nnodes, 1))()
        _check(_call("tr_equiv_set_nodes", self._open(), arr, max(self.nnodes, 1)), what="query parts")
        return [(int(arr[i].op), int(arr[i].a), int(arr[i].b)) for i in range(self.nnodes)]

    def _scope(self, scope, what):
        if not isinstance(scope, _mbk.Scope):
            raise TypeError("scope: a math_bank.Scope")
        if tuple(scope.names) != self.names:
            raise ValueError(f"{what}: the scope's columns {scope.names} are not the set's {self.names}")
        if scope.source == "table":
            scope.require_fresh(what)

    def check(self, scope, questions, *, backend="cpu", relations=(), budget=None):
        """SNAPSHOT verdicts: every question decided exhaustively on the scope's selected rows (backend "cuda":
        the condition words and the questions on the device, adopted after consistency checks); relations:
        FND-007 certificates of polynomial facts, tried first on measure questions (GEOM-001 reductions)"""
        self._scope(scope, "snapshot check")
        qs = tuple(questions)
        arr, nq = _questions(qs)
        rels = tuple(relations)
        rarr = (C.c_void_p * max(len(rels), 1))(*[r._open() for r in rels])
        error, refusal = C.c_int32(), _Refusal()
        if backend == "cpu":
            k = _call("tr_equiv_check_make", self._open(), arr, nq, scope._handle, None, None, C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
            k = _made(k, error, refusal, "snapshot check")
            try:
                h = _call("tr_equiv_check_decide", k, scope._handle, None, _BY_CPU, rarr, len(rels),
                          C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
                h = _made(h, error, refusal, "snapshot verdicts")
            finally:
                _call("tr_equiv_check_free", k)
            return Result._wrap(h, self.names, qs, "cpu", "native CPU")
        if backend != "cuda":
            raise ValueError("backend: 'cpu' or 'cuda'")
        gpu = _gpu()
        conds, k_cond = self.conditions()
        nw = (scope.rows + 63) // 64
        value = np.zeros(max(k_cond * nw, 1), dtype=np.uint64)
        valid = np.zeros(max(k_cond * nw, 1), dtype=np.uint64)
        if k_cond:
            view = scope.view()
            gpu.run("tr_mgpu_bool_masks", C.byref(view), conds, k_cond, C.byref(_budget(budget)), value.ctypes.data,
                    valid.ctypes.data, C.byref(refusal), refusal=refusal, what="condition words on the device")
        k = _call("tr_equiv_check_make", self._open(), arr, nq, scope._handle, value.ctypes.data if k_cond else None,
                  valid.ctypes.data if k_cond else None, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        k = _made(k, error, refusal, "snapshot check (device words)")
        try:
            v = _CheckView()
            _check(_call("tr_equiv_check_get_view", k, C.byref(v)), what="check view")
            ev = (_Evidence * max(nq, 1))()
            gpu.run("tr_mgpu_equiv_check", C.byref(v), C.byref(_budget(budget)), ev, C.byref(refusal),
                    refusal=refusal, what="snapshot check on the device")
            h = _call("tr_equiv_check_decide", k, scope._handle, ev, _BY_DEVICE, rarr, len(rels),
                      C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            h = _made(h, error, refusal, "snapshot verdicts (device evidence)")
        finally:
            _call("tr_equiv_check_free", k)
        label, device = _device_label()
        return Result._wrap(h, self.names, qs, label, device)

    def check_need(self, scope, questions, backend="cpu"):
        """the needs of evaluating a check (CPU loop or tr_mgpu_equiv_check's device copies)"""
        self._scope(scope, "check need")
        arr, nq = _questions(questions)
        error, refusal = C.c_int32(), _Refusal()
        k = _made(_call("tr_equiv_check_make", self._open(), arr, nq, scope._handle, None, None, None, C.byref(refusal),
                        C.byref(error)), error, refusal, "check need")
        try:
            need = _Need()
            _check(_call("tr_equiv_check_need", k, _NEED_DEVICE if backend == "cuda" else _NEED_CPU, C.byref(need)))
        finally:
            _call("tr_equiv_check_free", k)
        return _needs_dict(need)

    def decide(self, questions, domain, *, budget=None):
        """universal verdicts over a declared Domain: proofs labelled "universal" or "domain", counterexample
        points, or "unknown" with the reason (never a guess); no row is read"""
        if not isinstance(domain, Domain):
            raise TypeError("domain: meq.domain(...)")
        qs = tuple(questions)
        arr, nq = _questions(qs)
        d, keep = domain.native(self.names)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_equiv_decide", self._open(), arr, nq, C.byref(d), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        del keep
        return Result._wrap(_made(h, error, refusal, "universal decision"), self.names, qs, "cpu", "native CPU")

    def run(self, scope, *, mode="shared", budget=None):
        """every query's COUNT, measure rows present and exact SUM over the scope: "shared" (one scan: conditions,
        DAG nodes and distinct measures once) or "separate" (one scan per query); (answers, stats)"""
        self._scope(scope, "query family")
        out = (_Answer * max(len(self.queries), 1))()
        st, refusal = _RunStats(), _Refusal()
        _check(_call("tr_equiv_family_run", self._open(), scope._handle, _RUNS[mode], C.byref(_budget(budget)), out,
                     C.byref(st), C.byref(refusal)), refusal, "query family")
        answers = []
        for i in range(len(self.queries)):
            a = out[i]
            W = int(a.words)
            s = 0
            if W:
                s = sum(int(a.sum[w]) << (64 * w) for w in range(W))
                if s >> (64 * W - 1):
                    s -= 1 << (64 * W)
            answers.append(dict(count=int(a.rows), nonnull=int(a.nonnull),
                                sum=(Fraction(s, int(a.denominator)) if a.measure >= 0 and a.nonnull else None),
                                integer_sum=s, denominator=int(a.denominator), filter=int(a.filter),
                                measure=int(a.measure), shared=int(a.pair)))
        stats = {n: int(getattr(st, n)) for n, _ in _RunStats._fields_}
        stats["mode"] = mode
        return answers, stats

    def close(self):
        if self._handle:
            _call("tr_equiv_set_free", self._handle)
            self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None):
            _call("tr_equiv_set_free", self._handle)
            self._handle = None

    def __repr__(self):
        return f"<META-001 set: {self.info()}>"


# ------------------------------------------------------------------ the relationship optimizer and the first slice
def rewrite(catalog, scope, measure):
    """the relationship optimizer's rewrite (relationships.Catalog: tr_optimize under its verified rules, lowered to
    stored codes and lifted back exactly) of an integer measure of degree <= 2 over the scope's table columns:
    (the rewritten math_geom.Polynomial, the trace); the rewrite is a candidate, to be checked like any measure"""
    if not isinstance(scope, _mbk.Scope) or scope.source != "table":
        raise TypeError("scope: a math_bank.Scope of the catalog's table")
    table = catalog.table
    f = _poly(measure)
    names = list(scope.names)
    g, keep = f.native(names)
    stored = (C.c_int32 * len(names))(*[table.idx(n) for n in names])
    info = _RewriteInfo()
    cap = 256
    exps, coefs = np.zeros(cap * 8, np.uint8), np.zeros(cap, np.int64)
    catalog._open()
    _check(_call("tr_equiv_rewrite", catalog._handle, table.at, table.used_bm.ctypes.data, C.byref(g), stored,
                 len(names), C.byref(info), exps.ctypes.data, coefs.ctypes.data, cap), what="relationship rewrite")
    del keep
    k, m = int(info.ncols), int(info.nterms)
    cols = [names[info.col[i]] for i in range(k)]
    terms = {tuple(int(x) for x in exps[t * k:(t + 1) * k]): int(coefs[t]) for t in range(m)}
    trace = dict(changed=bool(info.changed), steps=int(info.steps), explored=int(info.explored),
                 truncated=bool(info.truncated), before_cost=float(info.before_cost), after_cost=float(info.after_cost))
    return _mg.Polynomial(terms, cols), trace


def measure_from_math(poly, columns):
    """a first-slice mathematics polynomial (its variable v bound to columns[v]) as a math_geom.Polynomial"""
    cols_in = list(columns) + [None] * (8 - len(columns))
    names = [c for c in columns if c is not None]
    index = {n: i for i, n in enumerate(names)}
    binding = (C.c_int32 * 8)(*[index[c] if c is not None else -1 for c in cols_in[:8]])
    ncols, out_cols, nterms = C.c_int32(), (C.c_int32 * 8)(), C.c_uint64()
    cap = 256
    exps, num, den = np.zeros(cap * 8, np.uint8), np.zeros(cap, np.int64), np.zeros(cap, np.int64)
    _check(_call("tr_equiv_poly_from_math", poly._open(), binding, C.byref(ncols), out_cols, exps.ctypes.data,
                 num.ctypes.data, den.ctypes.data, cap, C.byref(nterms)), what="first-slice polynomial")
    k, m = int(ncols.value), int(nterms.value)
    cols = [names[out_cols[i]] for i in range(k)]
    terms = {}
    for t in range(m):
        e = tuple(int(x) for x in exps[t * k:(t + 1) * k])
        terms[e] = terms.get(e, Fraction(0)) + Fraction(int(num[t]), int(den[t]))
    return _mg.Polynomial(terms, cols)
