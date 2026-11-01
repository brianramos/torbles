# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ATTR-002: bounded explanation and repair search: redundant predicates, minimal distinguishing sets and least-cost
feasible repairs, exact where the search finishes and explicitly incomplete with proven bounds where its declared
state budget stops it; natively on the CPU and with the CUDA kernels (torbles/math_repair.c, math_repair_gpu.cu);
continuous repairs through INV-002 (math_interv).

    from torbles import math_bool as mb, math_repair as mr
    masks = mb.Masks.from_scope(scope, [mb.condition("x", ">", 5), mb.condition("y", "<", 3), ...])
    P = mr.Problem.keep(masks, [0, 1, 2])            # the conjunction c0 AND c1 AND c2: minimum keep sets
    P = mr.Problem.program(masks, prog)             # a BOOL-001 circuit: conditions outside S at their baseline
    P = mr.Problem.separate(masks, rows_a, rows_b)  # or ("group", g) with the masks' scope groups
    P = mr.Problem.explain(masks, prog)             # the smallest condition sets on which membership depends
    P = mr.Problem.game(game, "explanation", ">=", 10)   # an ATTR-001 game
    P = mr.Problem.conjunction(hist, [0, 1, 2])     # LATTICE-002: v(S) = v(C) over a histogram
    P = mr.Problem.delete(masks, rows=[3, 8, 11], aggregate="sum", value="v", op="<=", target=100)
    mr.search(P, cost=None, method="auto", order="shapley", game=g, max_states=10_000, backend="cuda")
    mr.redundant(P, witnesses=8)                    # removal of each condition: certified by exact counts
    mr.edit(masks, rows, "v", "sum", ">=", 100, cost="l1")            # an INV-002 result
    mr.threshold(scope, conditions, which=[1], aggregate="count", op="<=", target=4)

A result says "found" (the minimum cost, every minimum-cost subset and their exact number), "none" (proven) or
"incomplete" (lower <= minimum <= upper, proven, with the states explored and the open nodes; never a claimed
minimum; when lower == upper, multiplicity_bounds = (ties found, at most: None if unknown), else (0, None)).
Costs are exact positive numbers (Fractions). Python marshals and formats; nothing is computed here. Opt-in:
`import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_attr as _ma
from . import math_bool as _mb
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check
from .math_inv import _Values, _exact, _one

__all__ = ["Problem", "search", "redundant", "edit", "threshold", "MAX_ELEMENTS", "MAX_SORTED"]

NONE = (1 << 64) - 1
MAX_ELEMENTS, MAX_SORTED, BATCH = 62, 20, 1024
KINDS = {1: "hit", 2: "circuit", 3: "game", 4: "delete"}
SOURCES = {1: "masks", 2: "separate", 3: "explain", 4: "keep", 5: "program", 6: "game", 7: "delete"}
CONVENTIONS = {"values": 1, "selected": 2}
OPS = {"=": 1, "==": 1, "!=": 2, "<>": 2, "<": 3, "<=": 4, ">": 5, ">=": 6}
AGGREGATES = {"count(*)": 1, "count_star": 1, "count": 2, "sum": 3, "avg": 4}
METHODS = {"auto": 0, "exhaustive": 1, "branch_bound": 2}
ORDERS = {"index": 0, "cost": 1, "shapley": 2, "banzhaf": 3}
STATUS = {1: "found", 2: "none", 3: "incomplete"}
REASONS = {0: None, 1: "complete", 2: "states", 3: "exhausted"}
GAME_KINDS = {"explanation": 1, "repair": 2}
COSTS = {"l1": 1, "linf": 2, "l2": 3, "l0": 4}
P = C.POINTER


_I128 = _ma._I128                   # tr_math_i128


def _i128(s):
    return (int(s.hi) << 64) + (int(s.lo) & ((1 << 64) - 1))


def _set_i128(s, v):
    v = int(v)
    if not -(1 << 127) <= v < (1 << 127):
        raise OverflowError("an exact value outside 128 bits")
    s.lo = ((v & ((1 << 64) - 1)) ^ (1 << 63)) - (1 << 63)
    s.hi = v >> 64


class _View(C.Structure):
    _native_ = "tr_repair_view"
    _fields_ = [(n, C.c_int32) for n in ("kind", "n", "op", "agg", "explain", "monotone", "vscale", "reserved")] + [
        ("allowed", C.c_uint64), ("nmasks", C.c_uint64), ("masks", C.c_void_p), ("dom", _mb._Domain),
        ("prog", _mb._ProgramView), ("baseline", C.c_uint64), ("target", C.c_void_p), ("game", _ma._GameView),
        ("threshold", _I128), ("qual", C.c_void_p), ("vvalid", C.c_void_p), ("vval", C.c_void_p),
        ("base_count", C.c_uint64), ("base_vcount", C.c_uint64), ("base_sum", _I128), ("tnum", _I128),
        ("tden", C.c_int64), ("reserved2", C.c_int64)]


class _Out(C.Structure):
    _native_ = "tr_repair_out"
    _fields_ = [("value", _I128), ("count", C.c_uint64), ("feasible", C.c_int32), ("null", C.c_int32)]


class _Query(C.Structure):
    _native_ = "tr_repair_query"
    _fields_ = [(n, C.c_int32) for n in ("method", "order", "reserved0", "reserved1")] + [
        ("max_states", C.c_uint64), ("max_solutions", C.c_uint64), ("cost", C.c_void_p)]


class _Plan(C.Structure):
    _native_ = "tr_repair_plan"
    _fields_ = [(n, C.c_int32) for n in ("method", "order", "n", "m")] + [
        (n, C.c_uint64) for n in ("allowed", "max_states", "max_solutions")] + [
        ("den", C.c_int64), ("unit", C.c_int64), ("cost", C.c_void_p), ("perm", C.c_void_p)]


class _Info(C.Structure):
    _native_ = "tr_repair_info"
    _fields_ = [(n, C.c_int32) for n in ("status", "reason", "method", "order", "n", "m", "complete", "reserved")] + [
        (n, C.c_int64) for n in ("lower", "upper", "den", "reserved2")] + [
        (n, C.c_uint64) for n in ("multiplicity", "mult_hi", "explored", "open", "pruned", "listed", "total")]


class _ProblemInfo(C.Structure):
    _native_ = "tr_repair_problem_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "source", "n", "convention")] + [
        (n, C.c_uint64) for n in ("allowed", "nmasks", "patterns_a", "patterns_b", "pairs", "witness_a", "witness_b",
                                  "points", "selected", "bytes")] + [("infeasible", C.c_int32), ("monotone", C.c_int32)]


class _Removal(C.Structure):
    _native_ = "tr_repair_removal"
    _fields_ = [("element", C.c_int32), ("redundant", C.c_int32)] + [
        (n, C.c_uint64) for n in ("count", "gained", "lost", "witnesses")] + [("value", _I128)]


for _s, _n in ((_View, 472), (_Out, 32), (_Query, 40), (_Plan, 72), (_Info, 120), (_ProblemInfo, 104),
               (_Removal, 56)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R, N = P(_Budget), P(_Refusal), P(_Need)
        sig = {
            "tr_repair_masks": (vp, [i32, u64, vp, u64, B, R, e]),
            "tr_repair_separate": (vp, [P(_mb._Domain), vp, vp, i32, u64, B, R, e]),
            "tr_repair_explain": (vp, [P(_mb._Domain), P(_mb._ProgramView), i32, u64, B, R, e]),
            "tr_repair_keep": (vp, [P(_mb._Domain), u64, B, R, e]),
            "tr_repair_program": (vp, [P(_mb._Domain), P(_mb._ProgramView), u64, u64, B, R, e]),
            "tr_repair_game": (vp, [vp, i32, i32, _I128, u64, B, R, e]),
            "tr_repair_delete": (vp, [vp, P(_mb._Domain), vp, i32, i32, i32, vp, i32, vp, B, R, e]),
            "tr_repair_group_words": (i32, [vp, u64, vp, u64]),
            "tr_repair_problem_get_info": (i32, [vp, P(_ProblemInfo)]),
            "tr_repair_problem_get_view": (i32, [vp, P(_View)]),
            "tr_repair_problem_masks": (i32, [vp, vp, u64]),
            "tr_repair_problem_free": (None, [vp]),
            "tr_repair_evaluate": (i32, [P(_View), vp, u64, vp, B, R]),
            "tr_repair_scan": (i32, [P(_View), vp, vp, u64, C.c_int64, vp, vp, P(C.c_int64), B, R]),
            "tr_repair_plan_need": (i32, [vp, P(_Query), N]),
            "tr_repair_prepare": (i32, [vp, P(_Query), vp, vp, vp, P(_Plan), B, R]),
            "tr_repair_search": (i32, [vp, P(_Plan), vp, vp, P(_Info), B, R]),
            "tr_repair_redundancy": (i32, [vp, vp, u64, vp, u64, B, R]),
            "tr_repair_edit": (vp, [vp, P(_mb._Domain), vp, i32, i32, i32, vp, i32, vp, i32, vp, vp, vp, B, R, e]),
            "tr_repair_threshold": (i32, [vp, vp, i32, u64, i32, i32, i32, vp, vp, vp, P(u64), B, R]),
            "tr_repair_threshold_after": (vp, [P(_mb._Condition), vp, B, R, e]),
            "tr_repair_device_need": (i32, [P(_View), u64, vp, N]),
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
        math_gpu.bind("tr_mgpu_repair_eval", C.c_int, [P(_View), C.c_void_p, C.c_uint64, C.c_void_p, P(_Budget),
                                                       P(_Refusal)])
        math_gpu.bind("tr_mgpu_repair_scan", C.c_int, [P(_View), C.c_void_p, C.c_void_p, C.c_uint64, C.c_int64,
                                                       C.c_void_p, C.c_void_p, P(C.c_int64), P(_Budget), P(_Refusal)])
        math_gpu.bind("tr_mgpu_repair_search", C.c_int, [P(_View), P(_Plan), C.c_void_p, C.c_void_p, P(_Info),
                                                         P(_Budget), P(_Refusal)])
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


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _label(backend):
    if backend == "cpu":
        return "cpu"
    return "cuda-emulated" if _gpu().device()["emulated"] else "cuda"


def _code(table, value, what):
    if value not in table:
        raise ValueError(f"{what}: one of {', '.join(map(str, table))}")
    return table[value]


def _mask(elements, n, what="elements"):
    if elements is None:
        return 0
    if isinstance(elements, Integral) and not isinstance(elements, bool):
        m = int(elements)
    else:
        m = 0
        for e in elements:
            if isinstance(e, bool) or not isinstance(e, Integral) or not 0 <= int(e) < n:
                raise ValueError(f"{what}: element indices in 0..{n - 1}")
            m |= 1 << int(e)
    if m >> n:
        raise ValueError(f"{what}: outside the {n} elements")
    return m


def _members(mask, n):
    return tuple(i for i in range(n) if (int(mask) >> i) & 1)


def _rows_words(masks, rows, what):
    """the bitmap words of a row set over the masks' points: row indices, a boolean array, or ("group", g)"""
    d = masks.domain()
    nw = int(d.nw)
    words = np.zeros(max(nw, 1), dtype=np.uint64)
    if isinstance(rows, tuple) and len(rows) == 2 and rows[0] == "group":
        if masks.scope is None:
            raise ValueError(f"{what}: groups need a rows domain built over a scope")
        _check(_api()["group_words"](masks.scope._handle, int(rows[1]), words.ctypes.data, nw), what=what)
        return words
    arr = np.asarray(rows)
    if arr.dtype == bool:
        if arr.shape != (int(d.points),):
            raise ValueError(f"{what}: one flag per point ({int(d.points)})")
        idx = np.nonzero(arr)[0]
    else:
        idx = [int(r) for r in rows]
    for r in idx:
        r = int(r)
        if not 0 <= r < int(d.points):
            raise ValueError(f"{what}: row {r} outside the domain")
        words[r >> 6] |= np.uint64(1 << (r & 63))
    return words


class Problem:
    """An immutable native subset problem (math_repair.h): n elements (conditions, players or rows), the allowed ones,
    and an exact feasibility test of every subset: a hitting set (separation, explanation of membership, conjunction
    keep sets), a BOOL-001 circuit, an ATTR-001 game or row deletions with an aggregate target."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Problem.masks, .separate, .explain, .keep, .program, .game, .conjunction or .delete")

    @classmethod
    def _wrap(cls, handle, names, keep=(), fresh=None, convention=None):
        self = object.__new__(cls)
        self._handle, self._keep, self._fresh = handle, tuple(keep), fresh
        i = _ProblemInfo()
        _check(_api()["problem_get_info"](handle, C.byref(i)), what="repair problem")
        self.kind, self.source, self.n = KINDS[int(i.kind)], SOURCES[int(i.source)], int(i.n)
        self.names = tuple(names) if names is not None else tuple(range(self.n))
        self.convention = convention or {}
        self.info = dict(kind=self.kind, source=self.source, n=self.n, allowed=_members(i.allowed, self.n),
                         masks=int(i.nmasks), patterns_a=int(i.patterns_a), patterns_b=int(i.patterns_b),
                         pairs=int(i.pairs), points=int(i.points), selected=int(i.selected), bytes=int(i.bytes),
                         infeasible=bool(i.infeasible), monotone=bool(i.monotone),
                         witness=None if i.witness_a == NONE else (int(i.witness_a), int(i.witness_b)))
        self.allowed = int(i.allowed)
        return self

    # ------------------------------------------------------------------ constructors
    @classmethod
    def masks(cls, n, masks, *, allowed=None, names=None, budget=None):
        """declared masks: S is feasible when it meets every mask (a hitting set; only the minimal masks are kept)"""
        ms = [_mask(m, n, "mask") for m in masks]
        arr = np.array(ms or [0], dtype=np.uint64)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["masks"](int(n), len(ms), arr.ctypes.data, _mask(allowed, n, "allowed"), _raw(budget),
                            C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "hitting set"), names,
                         convention=dict(problem="S meets every declared mask"))

    @classmethod
    def _domain(cls, masks, what):
        if not isinstance(masks, _mb.Masks):
            raise TypeError("masks: a math_bool.Masks")
        masks._open()
        masks.require_fresh(what)
        return masks.domain()

    @classmethod
    def _names(cls, masks):
        return tuple(repr(c) for c in masks.conditions) if masks.kind == "rows" and masks.conditions else None

    @classmethod
    def separate(cls, masks, a, b, *, convention="values", allowed=None, budget=None):
        """the condition sets S on which every point of a differs from every point of b (a, b: rows, a boolean
        array, or ("group", g)); convention "values" (TRUE / FALSE / UNKNOWN differ) or "selected" (TRUE vs not)"""
        d = cls._domain(masks, "separation")
        wa, wb = _rows_words(masks, a, "a"), _rows_words(masks, b, "b")
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["separate"](C.byref(d), wa.ctypes.data, wb.ctypes.data, _code(CONVENTIONS, convention, "convention"),
                               _mask(allowed, masks.k, "allowed"), _raw(budget), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "separation"), cls._names(masks), keep=(masks,),
                         fresh=lambda what: masks.require_fresh(what),
                         convention=dict(problem="S separates a from b: every pair differs on a condition of S",
                                         values=convention))

    @classmethod
    def explain(cls, masks, program, *, convention="selected", allowed=None, budget=None):
        """the condition sets S on which membership depends: S separates the points the program selects (TRUE) from
        the other points"""
        d = cls._domain(masks, "explanation")
        if not isinstance(program, _mb.Program):
            raise TypeError("program: a math_bool.Program")
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["explain"](C.byref(d), C.byref(program.view()), _code(CONVENTIONS, convention, "convention"),
                              _mask(allowed, masks.k, "allowed"), _raw(budget), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "membership explanation"), cls._names(masks), keep=(masks, program),
                         fresh=lambda what: masks.require_fresh(what),
                         convention=dict(problem="S explains membership: the selected and the other points differ "
                                                 "on a condition of S", values=convention))

    @classmethod
    def keep(cls, masks, conjunction, *, budget=None):
        """the subsets S of the conjunction's conditions whose conjunction selects exactly the same points (SQL WHERE:
        TRUE on every condition); the conditions outside a minimum S are jointly redundant"""
        d = cls._domain(masks, "conjunction keep sets")
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["keep"](C.byref(d), _mask(conjunction, masks.k, "conjunction"), _raw(budget), C.byref(refusal),
                           C.byref(error))
        return cls._wrap(_made(h, error, refusal, "conjunction keep sets"), cls._names(masks), keep=(masks,),
                         fresh=lambda what: masks.require_fresh(what),
                         convention=dict(problem="AND over S selects exactly the points AND over the conjunction "
                                                 "selects"))

    @classmethod
    def program(cls, masks, program, *, baseline=True, allowed=None, budget=None):
        """the condition sets S such that the program with every condition outside S at its baseline constant (True:
        TRUE, False: FALSE; one or per condition) selects exactly the points it selects (SQL three-valued logic);
        allowed by default: the conditions the program reads"""
        d = cls._domain(masks, "program keep sets")
        if not isinstance(program, _mb.Program):
            raise TypeError("program: a math_bool.Program")
        k = program.k
        base = [bool(baseline)] * k if isinstance(baseline, (bool, np.bool_)) else [bool(x) for x in baseline]
        if len(base) != k:
            raise ValueError(f"baseline: one constant per condition ({k})")
        bits = sum(1 << i for i, x in enumerate(base) if x)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["program"](C.byref(d), C.byref(program.view()), bits, _mask(allowed, k, "allowed"), _raw(budget),
                              C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "program keep sets"), cls._names(masks), keep=(masks, program),
                         fresh=lambda what: masks.require_fresh(what),
                         convention=dict(problem="the program with the conditions outside S at their baselines "
                                                 "selects the same points", baseline=tuple(base)))

    @classmethod
    def game(cls, game, kind="explanation", op=">=", threshold=0, *, allowed=None, budget=None):
        """an ATTR-001 game: "explanation" v(S) op t, "repair" v(P - S) op t"""
        if not isinstance(game, _ma.Game):
            raise TypeError("game: a math_attr.Game")
        if isinstance(threshold, bool) or not isinstance(threshold, Integral):
            raise TypeError("threshold: an exact integer")
        t = _I128()
        _set_i128(t, threshold)
        error, refusal = C.c_int32(), _Refusal()
        h = _api()["game"](game._open("repair search"), _code(GAME_KINDS, kind, "kind"), _code(OPS, op, "op"), t,
                           _mask(allowed, game.n, "allowed"), _raw(budget), C.byref(refusal), C.byref(error))
        what = "v(S)" if kind == "explanation" else "v(P - S)"
        return cls._wrap(_made(h, error, refusal, "game search problem"), game.players, keep=(game,),
                         fresh=game._fresh, convention=dict(game.convention, problem=f"{what} {op} {int(threshold)}"))

    @classmethod
    def conjunction(cls, hist, conjunction, *, budget=None):
        """over a LATTICE-001 histogram: the subsets S of the conjunction C with v(S) = v(C) for the LATTICE-002
        conjunction game v(S) = COUNT(*) WHERE AND over S (equal counts: the same points, S being part of C)"""
        g = _ma.Game.conjunction(hist, budget=budget)
        C_ = _mask(conjunction, g.n, "conjunction")
        target = g.value(C_)
        return cls.game(g, "explanation", "=", target, allowed=C_, budget=budget)

    @classmethod
    def delete(cls, masks, rows, *, aggregate="count(*)", value=None, op="<=", target=0, predicate=None, budget=None):
        """deleting a subset S of the declared rows (the others fixed) so that aggregate(value) over the qualifying
        remaining rows op target (qualifying: selected, and TRUE under the predicate program if given)"""
        d = cls._domain(masks, "deletion repair")
        if masks.scope is None:
            raise ValueError("deletions need a rows domain built over a scope")
        names = list(masks.scope.names)
        agg = _code(AGGREGATES, aggregate, "aggregate")
        v = -1 if value is None else names.index(value)
        rows = [int(r) for r in rows]
        arr = np.array(rows or [0], dtype=np.uint64)
        tq, keep_t = _one(target, "target")
        error, refusal = C.c_int32(), _Refusal()
        pv = C.byref(predicate.view()) if predicate is not None else None
        h = _api()["delete"](masks.scope._handle, C.byref(d), pv, v, agg, _code(OPS, op, "op"), tq, len(rows),
                             arr.ctypes.data, _raw(budget), C.byref(refusal), C.byref(error))
        p = cls._wrap(_made(h, error, refusal, "deletion repair"), tuple(rows), keep=(masks, predicate, keep_t),
                      fresh=lambda what: masks.require_fresh(what),
                      convention=dict(problem=f"{aggregate}({value or '*'}) {op} {Fraction(_exact(target))} over the "
                                              "qualifying rows that remain after deleting S"))
        return p

    # ------------------------------------------------------------------ reading
    def _open(self, what="repair problem"):
        if not self._handle:
            raise ValueError("the problem is closed")
        if self._fresh is not None:
            self._fresh(what)
        return self._handle

    def view(self):
        v = _View()
        _check(_api()["problem_get_view"](self._open(), C.byref(v)), what="repair problem")
        return v

    def minimal_masks(self):
        """a hitting set's minimal masks, as element tuples"""
        if self.kind != "hit":
            raise ValueError("only hitting sets have masks")
        n = self.info["masks"]
        out = np.zeros(max(n, 1), dtype=np.uint64)
        _check(_api()["problem_masks"](self._open(), out.ctypes.data, len(out)), what="repair masks")
        return [_members(m, self.n) for m in out[:n].tolist()]

    def evaluate(self, subsets, *, backend="cpu", budget=None):
        """per subset (element tuples or masks): {"feasible", "count", "value", "null"}"""
        ms = [_mask(s, self.n, "subset") for s in subsets]
        k = len(ms)
        arr = np.array(ms or [0], dtype=np.uint64)
        out = (_Out * max(k, 1))()
        refusal = _Refusal()
        v = self.view()
        if _backend(backend) == "cpu":
            _check(_api()["evaluate"](C.byref(v), arr.ctypes.data, k, out, _raw(budget), C.byref(refusal)), refusal,
                   "repair evaluation")
        else:
            _gpu().run("tr_mgpu_repair_eval", C.byref(v), arr.ctypes.data, k, out, _raw(budget), C.byref(refusal),
                       refusal=refusal, what="repair evaluation on the device")
        return [dict(feasible=bool(o.feasible), count=int(o.count), value=_i128(o.value), null=bool(o.null))
                for o in out[:k]]

    def device_need(self, count):
        out = _Need()
        _check(_api()["device_need"](C.byref(self.view()), int(count), None, C.byref(out)), what="device needs")
        return _needs(out)

    def close(self):
        if getattr(self, "_handle", None):
            _api()["problem_free"](self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Problem({self.source}, n={self.n})" if self._handle else "Problem(closed)"


def _query(problem, cost, method, order, max_states, max_solutions, budget):
    b = budget or Budget()
    if max_states is None:
        max_states = b.limits["iterations"]
    if isinstance(max_states, bool) or not isinstance(max_states, Integral) or max_states < 1:
        raise ValueError("max_states: a positive integer")
    q = _Query(method=_code(METHODS, method, "method"), order=_code(ORDERS, order, "order"),
               max_states=int(max_states), max_solutions=int(max_solutions))
    keep = None
    if cost is not None:
        cost = list(cost)
        if len(cost) != problem.n:
            raise ValueError(f"cost: one positive exact number per element ({problem.n})")
        keep = _Values(cost, "cost")
        q.cost = C.cast(keep.array, C.c_void_p)
    return q, keep, b


def search(problem, *, cost=None, method="auto", order="index", game=None, max_states=None, max_solutions=64,
           backend="cpu", budget=None):
    """The minimum-cost feasible subsets of the problem's allowed elements: "exhaustive" (increasing cost order),
    "branch_bound" (best-first with admissible bounds) or "auto"; order "index", "cost", "shapley" / "banzhaf"
    (decreasing ATTR-001 index of `game`, or of the problem's own game) for the branching. max_states: the declared
    exploration budget (incomplete with proven bounds past it)."""
    if not isinstance(problem, Problem):
        raise TypeError("a math_repair.Problem")
    q, keep, b = _query(problem, cost, method, order, max_states, max_solutions, budget)
    if order in ("shapley", "banzhaf") and game is None:
        game = next((k for k in problem._keep if isinstance(k, _ma.Game)), None)
        if game is None:
            raise ValueError(f"order {order!r} needs an ATTR-001 game over the same elements (game=...)")
    gh = game._open("search order") if game is not None else None
    n = problem.n
    costs, perm = np.zeros(max(n, 1), dtype=np.int64), np.zeros(max(n, 1), dtype=np.uint32)
    plan, refusal = _Plan(), _Refusal()
    _check(_api()["prepare"](problem._open("repair search"), C.byref(q), gh, costs.ctypes.data, perm.ctypes.data,
                             C.byref(plan), C.byref(b.raw), C.byref(refusal)), refusal, "repair search")
    k = int(max_solutions)
    sols, scost = np.zeros(max(k, 1), dtype=np.uint64), np.zeros(max(k, 1), dtype=np.int64)
    info, refusal = _Info(), _Refusal()
    if _backend(backend) == "cpu":
        _check(_api()["search"](problem._open(), C.byref(plan), sols.ctypes.data, scost.ctypes.data, C.byref(info),
                                C.byref(b.raw), C.byref(refusal)), refusal, "repair search")
    else:
        v = problem.view()
        _gpu().run("tr_mgpu_repair_search", C.byref(v), C.byref(plan), sols.ctypes.data, scost.ctypes.data,
                   C.byref(info), C.byref(b.raw), C.byref(refusal), refusal=refusal,
                   what="repair search on the device")
    den = int(info.den)
    frac = lambda x: Fraction(int(x), den)
    status = STATUS[int(info.status)]
    lower = None if info.lower < 0 else frac(info.lower)
    upper = None if info.upper < 0 else frac(info.upper)
    listed = int(info.listed)
    names = problem.names
    out = dict(status=status, exact=status != "incomplete", reason=REASONS[int(info.reason)],
               method={1: "exhaustive", 2: "branch_bound"}[int(info.method)], order=order,
               minimum=lower if status == "found" else None, lower=lower, upper=upper,
               solutions=[tuple(names[i] for i in _members(m, n)) for m in sols[:listed].tolist()],
               masks=[int(m) for m in sols[:listed].tolist()], costs=[frac(c) for c in scost[:listed].tolist()],
               multiplicity=int(info.multiplicity) if status == "found" else None,
               multiplicity_bounds=(int(info.multiplicity), None if info.mult_hi == NONE else int(info.mult_hi)),
               listed=listed, explored=int(info.explored), open=int(info.open), pruned=int(info.pruned),
               total=int(info.total), elements=int(info.m), max_states=int(q.max_states), backend=_label(backend),
               order_elements=tuple(names[int(i)] for i in perm[:int(info.m)]), convention=dict(problem.convention),
               causal=False)
    out["statement"] = (f"exact: the minimum cost is {lower}, attained by {out['multiplicity']} subset(s)"
                        if status == "found" else "proven: no subset of the allowed elements is feasible"
                        if status == "none" else
                        f"incomplete after {out['explored']} states: {lower} <= minimum <= "
                        f"{'unknown (no feasible subset found yet)' if upper is None else upper} (proven); "
                        "not a claimed minimum")
    del keep
    return out


def redundant(problem, *, witnesses=8, backend="cpu", budget=None):
    """per allowed element i: removing it (S = allowed - {i}) leaves the result unchanged? Certified by the exact
    count of differing points (0); for a program, the first `witnesses` rows where the selections differ (gained:
    selected only without the condition, then lost)"""
    if not isinstance(problem, Problem):
        raise TypeError("a math_repair.Problem")
    m = bin(problem.allowed).count("1")
    out = (_Removal * max(m, 1))()
    rows = np.zeros(max(m * int(witnesses), 1), dtype=np.uint64)
    refusal = _Refusal()
    _check(_api()["redundancy"](problem._open("redundancy"), out, m, rows.ctypes.data if witnesses else None,
                                int(witnesses), _raw(budget), C.byref(refusal)), refusal, "redundancy")
    res = []
    for q, r in enumerate(out[:m]):
        w = rows[q * witnesses:q * witnesses + int(r.witnesses)].tolist() if witnesses else []
        gained = min(int(r.gained), len(w))
        res.append(dict(element=problem.names[int(r.element)], index=int(r.element), redundant=bool(r.redundant),
                        differ=int(r.count), gained=int(r.gained), lost=int(r.lost), value=_i128(r.value),
                        witnesses=dict(gained=[int(x) for x in w[:gained]], lost=[int(x) for x in w[gained:]])))
    if _backend(backend) == "cuda":                   # the certifying counts from the kernel
        subsets = [problem.allowed & ~(1 << r["index"]) for r in res]
        dev = problem.evaluate(subsets, backend="cuda", budget=budget)
        for r, e in zip(res, dev):
            r["device"] = dict(redundant=e["feasible"], differ=e["count"], value=e["value"])
    return res


def edit(masks, rows, value, aggregate, op, target, *, cost="l1", weights=None, d_bounds=None, predicate=None,
         budget=None):
    """value edits of the declared rows (present values of `value`) so that SUM / AVG(value) over the qualifying rows
    op target: INV-002's exact LP / projection (math_interv.Result: the cheapest change, every tie, algebraic)"""
    from . import math_interv as iv
    d = Problem._domain(masks, "value-edit repair")
    if masks.scope is None:
        raise ValueError("edits need a rows domain built over a scope")
    names = list(masks.scope.names)
    rows = [int(r) for r in rows]
    arr = np.array(rows or [0], dtype=np.uint64)
    tq, keep_t = _one(target, "target")
    w = _Values(weights if weights is not None else [1] * len(rows), "weight")
    lo, hi = (None, None) if d_bounds is None else d_bounds
    dlo = _Values([lo] * len(rows) if not isinstance(lo, (list, tuple)) else lo, "edit bound", allow_none=True)
    dhi = _Values([hi] * len(rows) if not isinstance(hi, (list, tuple)) else hi, "edit bound", allow_none=True)
    error, refusal = C.c_int32(), _Refusal()
    pv = C.byref(predicate.view()) if predicate is not None else None
    h = _api()["edit"](masks.scope._handle, C.byref(d), pv, names.index(value),
                       _code({"sum": 3, "avg": 4}, aggregate, "aggregate"), _code(OPS, op, "op"), tq, len(rows),
                       arr.ctypes.data, _code(COSTS, cost, "cost"), w.array, dlo.array, dhi.array, _raw(budget),
                       C.byref(refusal), C.byref(error))
    res = iv.Result(_made(h, error, refusal, "value-edit repair"), [f"row {r}" for r in rows])
    del keep_t
    return res


def threshold(scope, conditions, which, *, aggregate="count(*)", value=None, op="<=", target=0, weights=None,
              budget=None):
    """changing the constant of one declared comparison condition of the conjunction (the others fixed) so that
    aggregate(value) over the rows then selected op target: per declared condition INV-002's aggregate solution
    (the rows that enter and leave the predicate decided exactly; the new constant c - d), and the minimum over them
    with every condition attaining it"""
    from . import math_interv as iv
    from .math_bank import Scope
    if not isinstance(scope, Scope):
        raise TypeError("scope: a math_bank.Scope")
    if scope.source == "table":
        scope.require_fresh("threshold repair")
    conditions = tuple(conditions)
    arr, k = _mb._conditions(conditions, scope.names)
    which_mask = _mask(which, k, "which")
    if not which_mask:
        raise ValueError("which: at least one changeable condition")
    names = list(scope.names)
    v = -1 if value is None else names.index(value)
    tq, keep_t = _one(target, "target")
    w = _Values(weights if weights is not None else [1] * k, "weight")
    nres = bin(which_mask).count("1")
    results = (C.c_void_p * nres)()
    attaining, refusal = C.c_uint64(), _Refusal()
    _check(_api()["threshold"](scope._handle, arr, k, which_mask, v, _code(AGGREGATES, aggregate, "aggregate"),
                               _code(OPS, op, "op"), tq, w.array, results, C.byref(attaining), _raw(budget),
                               C.byref(refusal)), refusal, "threshold repair")
    out, idx = {}, [i for i in range(k) if (which_mask >> i) & 1]
    new = {}
    api = _api()
    for j, h in zip(idx, results):
        res = None
        thr = []
        if h:
            # the new constants of every solution, natively (c - d), before the result is consumed
            from .math_interv import _api as _iv_api, _Solution
            n_sol = _iv_info_solutions(h)
            for s in range(n_sol):
                sol, lo_, hi_, a, b2, cl, ch, vl, vh = _Solution(), (C.c_void_p * 1)(), (C.c_void_p * 1)(), \
                    (C.c_void_p * 1)(), (C.c_void_p * 1)(), C.c_void_p(), C.c_void_p(), C.c_void_p(), C.c_void_p()
                _check(_iv_api()["result_solution"](h, s, C.byref(sol), lo_, hi_, a, b2, C.byref(cl), C.byref(ch),
                                                     C.byref(vl), C.byref(vh)), what="threshold solution")
                e = C.c_int32()
                nv = api["threshold_after"](C.byref(arr[j]), lo_[0], _raw(budget), None, C.byref(e))
                _check(e.value, what="threshold value")
                from .math_env import _owned
                thr.append(_owned(nv))
            res = iv.Result(h, ["d"])
        out[j] = res
        new[j] = thr
    best = None
    att = [j for j in idx if (attaining.value >> j) & 1]
    if att:
        best = out[att[0]].lower
    del keep_t
    return dict(results=out, thresholds=new, minimum=best, attaining=att,
                conditions={j: repr(conditions[j]) for j in idx}, causal=False,
                statement="algebraic: the cheapest change of one declared threshold (rows entering and leaving the "
                          "predicate included); no causal claim")


def _iv_info_solutions(h):
    from .math_interv import _api as _iv_api, _Info as _IvInfo
    i = _IvInfo()
    _check(_iv_api()["result_get_info"](h, C.byref(i)), what="threshold result")
    return int(i.nsolutions)
