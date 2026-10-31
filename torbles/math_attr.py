# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ATTR-001: sparse interaction and contribution attribution of declared cooperative games, exactly (Fractions
formed from native canonical 128-bit rationals), natively on the CPU or with the CUDA kernels (torbles/math_attr.c,
math_attr_gpu.cu).

    from torbles import math_attr as ma
    g = ma.Game.provenance(G, tuples=[0], semiring="count", players="tuples")   # PROV-001: v(S) = COUNT(*) of G[0]
    g = ma.Game.provenance(J, weights="a", players="sources")                   # SUM(a) over J, one player per source
    g = ma.Game.conjunction(hist)        # LATTICE-002: v(S) = COUNT(*) WHERE the conditions of S (AND)
    g = ma.Game.disjunction(hist)        # v(S) = COUNT(*) WHERE c_i OR ... for i in S
    g = ma.Game.predicate(masks, program, baseline=True)   # v(S) = the predicate with the absent conditions constant
    g = ma.Game.table(values)            # 2^n declared exact values (index = coalition mask)
    g.values([(), (0, 2), 5], backend="cuda"); g.convention; g.info
    s = g.support()                      # the sparse Moebius (Harsanyi) support: s.terms, s.info
    ma.shapley(g); ma.banzhaf(s); ma.shapley(g, method="exhaustive", backend="cuda")
    ma.interactions(s, order=2, index="shapley" | "banzhaf" | "moebius" | "shapley_taylor")
    ma.explore(g, max_coalitions=5000)   # exact, or certified bounds with the explored states counted
    ma.search(g, "repair", "<=", 0, max_coalitions=10_000)    # minimum repairs / explanations, exact or bounded

Every result carries its convention (the game, the players, the baseline v(empty), the weighting) and says
causal=False: an algebraic attribution splits v(P) - v(empty) under the declared game; it has no automatic causal
interpretation. Budgets raise AlgebraBudgetError before anything is allocated, an exact value outside the 128-bit
carriers OverflowError, a stale scope StaleStateError. Nothing is computed in Python. Opt-in: `import torbles` does not
load this module.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_bool as mb
from . import math_lattice as ml
from . import math_prov as mp
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Game", "Support", "shapley", "banzhaf", "interactions", "interactions_need", "exhaustive", "explore",
           "search", "INTERPRETATION", "WEIGHTING", "INDICES", "MAX_ENUM", "MAX_TABLE", "MAX_EXHAUSTIVE"]

MAX_ENUM, MAX_TABLE, MAX_EXHAUSTIVE = 62, 24, 32
TABLE, PROV = 1, 2
SOURCES = {1: "values", 2: "conjunction", 3: "disjunction", 4: "predicate", 5: "provenance"}
INDICES = {"shapley": 1, "banzhaf": 2, "moebius": 3, "shapley_interaction": 4, "banzhaf_interaction": 5,
           "shapley_taylor": 6}
INTERACTION_INDICES = {"moebius": 3, "shapley": 4, "banzhaf": 5, "shapley_taylor": 6}
METHODS = {"auto": 0, "expansion": 1, "witnesses": 2, "transform": 3}
METHOD_NAMES = {0: "auto", 1: "expansion", 2: "witnesses", 3: "transform", 4: "terms"}
IDENTITIES = {0: None, 1: "efficiency", 2: "shapley_taylor_efficiency", 3: "moebius_sum", 4: "moebius_empty"}
OPS = {"=": 1, "==": 1, "<>": 2, "!=": 2, "<": 3, "<=": 4, ">": 5, ">=": 6}
KINDS = {"explanation": 1, "explain": 1, "repair": 2}
STATUS = {1: "found", 2: "none", 3: "incomplete"}
REASONS = {1: "minimum level complete", 2: "coalition budget exhausted", 3: "every coalition explored",
           4: "monotone game: the extreme value fails"}
OP_VALUES, OP_EXHAUSTIVE, OP_EXPLORE, OP_SEARCH = 1, 2, 3, 4
_NEED_CPU, _NEED_DEVICE = 0, 1
_U64, _M128 = (1 << 64) - 1, (1 << 128) - 1
P = C.POINTER
i32, u32, i64, u64, vp = C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_void_p

INTERPRETATION = ("an algebraic decomposition of v(P) - v(empty) among the declared players (or their sets) under the "
                  "declared game, baseline and weighting; it has no automatic causal interpretation: it is not the "
                  "effect of an intervention on the data, and another game, baseline or weighting attributes "
                  "differently")
WEIGHTING = {
    "shapley": "phi_i = sum over coalitions S without i of |S|! (n-|S|-1)! / n! (v(S + i) - v(S)) "
               "= sum over T containing i of m(T) / |T| (efficient: sum_i phi_i = v(P) - v(empty))",
    "banzhaf": "beta_i = 2^-(n-1) sum over coalitions S without i of (v(S + i) - v(S)) "
               "= sum over T containing i of m(T) / 2^(|T|-1) (every coalition weighted equally; not efficient)",
    "moebius": "m(S) = sum over T subset of S of (-1)^|S - T| v(T) (Harsanyi dividends: v(S) = sum over T subset of "
               "S of m(T))",
    "shapley_interaction": "I(S) = sum over T superset of S of m(T) / (|T| - |S| + 1) (Grabisch; I({i}) = phi_i)",
    "banzhaf_interaction": "B(S) = sum over T superset of S of m(T) / 2^(|T| - |S|) (B({i}) = beta_i)",
    "shapley_taylor": "ST(S) = m(S) for |S| < k, sum over T superset of S of m(T) / C(|T|, k) for |S| = k "
                      "(Sundararajan et al.; efficient: sum over 1 <= |S| <= k of ST(S) = v(P) - v(empty))",
}


class _I128(C.Structure):
    _native_ = "tr_math_i128"
    _fields_ = [("lo", i64), ("hi", i64)]


class _Q(C.Structure):
    _native_ = "tr_attr_q"
    _fields_ = [("num", _I128), ("den", _I128)]


class _Identity(C.Structure):
    _native_ = "tr_attr_identity"
    _fields_ = [("lo", _Q), ("hi", _Q), ("expected", _Q), ("kind", i32), ("holds", i32), ("exact", i32),
                ("reserved", i32)]


class _GameView(C.Structure):
    _native_ = "tr_attr_game_view"
    _fields_ = [(n, i32) for n in ("kind", "semiring", "n", "monotone", "sign", "degree", "source", "reserved")] + [
        ("nvars", u64), ("plan", mp._PlanView), ("weights", vp), ("var_player", vp), ("table", vp)] + [
        (n, _I128) for n in ("lo", "hi", "bound", "empty", "full")]


class _GameInfo(C.Structure):
    _native_ = "tr_attr_game_info"
    _fields_ = [(n, i32) for n in ("kind", "semiring", "n", "source", "monotone", "sign", "degree", "projection")] + [
        (n, u64) for n in ("baseline", "nvars", "present", "absent", "noutputs", "nodes", "bytes")] + [
        ("signed_weights", i32), ("players_used", i32)] + [(n, _I128) for n in ("lo", "hi", "bound", "empty", "full")]


class _SupportInfo(C.Structure):
    _native_ = "tr_attr_support_info"
    _fields_ = [(n, i32) for n in ("n", "method", "degree", "nonnegative")] + [
        (n, u64) for n in ("terms", "pairs", "bytes", "candidates")] + [("from_game", i32), ("reconstructs", i32)] + [
        (n, _I128) for n in ("sum", "empty", "game_full", "game_empty")]


class _TableInfo(C.Structure):
    _native_ = "tr_attr_table_info"
    _fields_ = [(n, i32) for n in ("index", "min_order", "max_order", "n")] + [
        (n, u64) for n in ("sets", "pairs", "candidates", "bytes")] + [("identity", _Identity)]


class _ExhaustiveInfo(C.Structure):
    _native_ = "tr_attr_exhaustive_info"
    _fields_ = [("n", i32), ("reserved", i32), ("coalitions", u64), ("shapley_den", _I128), ("banzhaf_den", _I128),
                ("efficiency", _Identity)]


class _ExploreInfo(C.Structure):
    _native_ = "tr_attr_explore_info"
    _fields_ = [(n, i32) for n in ("regime", "exact", "n", "degree", "bottom", "levels", "reserved0", "reserved1")] + [
        (n, u64) for n in ("explored", "total", "level_mask", "budget")] + [
        (n, _I128) for n in ("lo", "hi", "empty", "full")] + [("efficiency", _Identity)]


class _SearchQuery(C.Structure):
    _native_ = "tr_attr_search_query"
    _fields_ = [(n, i32) for n in ("kind", "op", "reserved0", "reserved1")] + [
        ("threshold", _I128), ("max_coalitions", u64), ("max_solutions", u64)]


class _SearchInfo(C.Structure):
    _native_ = "tr_attr_search_info"
    _fields_ = [(n, i32) for n in ("status", "reason", "size_lo", "size_hi")] + [
        (n, u64) for n in ("mult_lo", "mult_hi", "explored", "listed")] + [
        (n, i32) for n in ("level", "complete", "monotone", "n")] + [
        ("level_explored", u64), ("level_total", u64), ("empty", _I128), ("full", _I128)]


for _s, _n in ((_I128, 16), (_Q, 32), (_Identity, 112), (_GameView, 200), (_GameInfo, 176), (_SupportInfo, 120),
               (_TableInfo, 160), (_ExhaustiveInfo, 160), (_ExploreInfo, 240), (_SearchQuery, 48), (_SearchInfo, 112)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_attr_game_table": (vp, [i32, vp, B, R, e]),
            "tr_attr_game_lattice": (vp, [vp, i32, i32, B, R, e]),
            "tr_attr_game_predicate": (vp, [P(mb._Domain), P(mb._ProgramView), u64, B, R, e]),
            "tr_attr_game_prov": (vp, [P(mp._PlanView), i32, vp, vp, i32, B, R, e]),
            "tr_attr_game_need": (i32, [i32, i32, P(mp._PlanView), P(_Need)]),
            "tr_attr_game_get_info": (i32, [vp, P(_GameInfo)]),
            "tr_attr_game_get_view": (i32, [vp, P(_GameView)]),
            "tr_attr_game_free": (None, [vp]),
            "tr_attr_plan_variables": (i32, [P(mp._PlanView), vp, u64, P(u64)]),
            "tr_attr_values": (i32, [vp, vp, u64, vp, B, R]),
            "tr_attr_exhaustive": (i32, [vp, vp, vp, vp, vp, P(_ExhaustiveInfo), B, R]),
            "tr_attr_exhaustive_adopt": (i32, [vp, vp, vp, vp, vp, P(_ExhaustiveInfo)]),
            "tr_attr_explore": (i32, [vp, u64, vp, vp, vp, vp, P(_ExploreInfo), B, R]),
            "tr_attr_search": (i32, [vp, P(_SearchQuery), vp, P(_SearchInfo), B, R]),
            "tr_attr_enum_need": (i32, [vp, i32, u64, i32, P(_Need)]),
            "tr_attr_support_build": (vp, [vp, i32, B, R, e]),
            "tr_attr_support_need": (i32, [vp, i32, P(_Need)]),
            "tr_attr_support_terms": (vp, [i32, u64, vp, vp, vp, B, R, e]),
            "tr_attr_support_get_info": (i32, [vp, P(_SupportInfo)]),
            "tr_attr_support_get": (i32, [vp, vp, vp, vp, u64, u64]),
            "tr_attr_support_free": (None, [vp]),
            "tr_attr_support_values": (i32, [vp, i32, vp, P(_Identity), B, R]),
            "tr_attr_interactions": (vp, [vp, i32, i32, i32, B, R, e]),
            "tr_attr_interactions_need": (i32, [vp, i32, i32, i32, P(_Need)]),
            "tr_attr_table_get_info": (i32, [vp, P(_TableInfo)]),
            "tr_attr_table_get": (i32, [vp, vp, vp, vp, u64, u64]),
            "tr_attr_table_free": (None, [vp]),
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
    G, B, R = P(_GameView), P(_Budget), P(_Refusal)
    math_gpu.bind("tr_mgpu_attr_values", C.c_int, [G, vp, u64, vp, B, R])
    math_gpu.bind("tr_mgpu_attr_exhaustive", C.c_int, [G, vp, vp, B, R])
    math_gpu.bind("tr_mgpu_attr_explore", C.c_int, [G, u64, vp, vp, vp, vp, P(_ExploreInfo), B, R])
    math_gpu.bind("tr_mgpu_attr_search", C.c_int, [G, P(_SearchQuery), vp, P(_SearchInfo), B, R])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _device_label(gpu):
    d = gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


# ---------------------------------------------------------------------- exact numbers across the boundary
def _int(s):
    return (int(s.hi) << 64) | (int(s.lo) & _U64)


def _q(s):
    return Fraction(_int(s.num), _int(s.den))


def _set_i128(s, v):
    x = int(v) & _M128
    lo, hi = x & _U64, x >> 64
    s.lo = lo - (1 << 64) if lo >> 63 else lo
    s.hi = hi - (1 << 64) if hi >> 63 else hi


def _words(values, what="values"):
    """(n, 2) int64 words of exact integers |v| < 2^127"""
    vals = list(values)
    out = np.zeros((max(len(vals), 1), 2), dtype=np.int64)
    for i, v in enumerate(vals):
        if isinstance(v, bool) or not isinstance(v, Integral):
            raise TypeError(f"{what}: {v!r} is not an exact integer")
        v = int(v)
        if not -(1 << 127) < v < (1 << 127):
            raise OverflowError(f"{what}: {v} does not fit the 128-bit carrier")
        x = v & _M128
        out[i, 0] = np.array(x & _U64, dtype=np.uint64).view(np.int64)
        out[i, 1] = np.array(x >> 64, dtype=np.uint64).view(np.int64)
    return out


def _from_words(words, n):
    return [(int(h) << 64) | (int(l) & _U64) for l, h in words[:n].tolist()]


def _qs(arr, n):
    return [_q(arr[i]) for i in range(n)]


def _identity(raw):
    kind = IDENTITIES.get(int(raw.kind))
    if kind is None:
        return None
    lo, hi, want = _q(raw.lo), _q(raw.hi), _q(raw.expected)
    out = dict(identity=kind, expected=want, holds=bool(raw.holds), exact=bool(raw.exact))
    if raw.exact:
        out["sum"] = lo
    else:
        out["sum_lo"], out["sum_hi"] = lo, hi
    return out


def _result(game, index, extra):
    conv = dict(game.convention, index=index, weighting=WEIGHTING[index])
    out = dict(index=index, players=game.players, convention=conv, causal=False, interpretation=INTERPRETATION)
    out.update(extra)
    return out


# ---------------------------------------------------------------------- games
def _names(players, n):
    if players is None:
        return tuple(range(n))
    names = tuple(players)
    if len(names) != n:
        raise ValueError(f"{n} player names expected, got {len(names)}")
    return names


class Game:
    """An immutable native cooperative game (math_attr.h): n players, an exact value v(S) for every coalition S, the
    baseline v(empty). Built only by its declared constructors; the convention says what v is."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Game.table, .conjunction, .disjunction, .predicate or .provenance")

    @classmethod
    def _wrap(cls, handle, players, convention, keep=(), fresh=None):
        self = object.__new__(cls)
        self._handle, self._keep, self._fresh = handle, tuple(keep), fresh
        info = self._raw_info()
        self.n = int(info.n)
        self.players = _names(players, self.n)
        self.convention = dict(convention)
        return self

    # ------------------------------------------------------------------ constructors
    @classmethod
    def table(cls, values, *, players=None, description=None, budget=None):
        """a game of 2^n declared exact values: values[S] = v(S), S the coalition mask (bit i = player i)"""
        vals = list(values)
        n = len(vals).bit_length() - 1
        if len(vals) < 2 or len(vals) != 1 << n:
            raise ValueError(f"2^n values (n >= 1) expected, got {len(vals)}")
        words = _words(vals)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_attr_game_table", n, words.ctypes.data, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        conv = dict(game=description or "declared coalition values v(S) (2^n exact integers, index = coalition mask)",
                    players="the declared players 0..n-1", baseline="v(empty): the declared value of the empty "
                    "coalition", value="as declared")
        return cls._wrap(_made(h, error, refusal, "table game"), players, conv)

    @classmethod
    def _lattice(cls, hist, rule, projection, players, budget):
        if not isinstance(hist, ml.Histogram):
            raise TypeError("hist: a math_lattice.Histogram")
        hist.require_fresh("attribution game")
        if projection is None:
            projection = "true_sets" if hist.nulls == "three" else "complete"
        code = {"complete": 1, "true_sets": 2}.get(projection)
        if code is None:
            raise ValueError("projection is 'complete' or 'true_sets'")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_attr_game_lattice", hist._handle, code, rule, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        where = ("over the complete points (an UNKNOWN condition excludes the point from every coalition)"
                 if projection == "complete" else "SQL WHERE semantics: a point counts when its conditions are TRUE "
                 "(FALSE and UNKNOWN do not count)")
        if rule == 2:
            conv = dict(game="v(S) = the points TRUE on every condition of S: COUNT(*) WHERE the conditions of S "
                             "joined by AND (LATTICE-002 zeta transform of the histogram's "
                             f"{projection} projection; {where})",
                        players="the histogram's conditions (axes)",
                        baseline="v(empty): no condition applied (every point of the projection); an absent "
                                 "condition is dropped from the conjunction",
                        value="COUNT(*)")
        else:
            conv = dict(game="v(S) = the points TRUE on at least one condition of S: COUNT(*) WHERE the conditions of "
                             f"S joined by OR (LATTICE-002 zeta transform at the complement; {where})",
                        players="the histogram's conditions (axes)",
                        baseline="v(empty) = 0: an absent condition is FALSE (an empty disjunction holds nowhere)",
                        value="COUNT(*)")
        names = players
        if names is None and hist.masks is not None and len(hist.masks.conditions) == hist.k and \
                all(a[0] == "condition" for a in hist.axes):
            names = tuple(repr(hist.masks.conditions[a[1]]) for a in hist.axes)
        return cls._wrap(_made(h, error, refusal, "lattice game"), names, conv, keep=(hist,),
                         fresh=lambda what: hist.require_fresh(what))

    @classmethod
    def conjunction(cls, hist, *, projection=None, players=None, budget=None):
        """v(S) = COUNT(*) WHERE the conditions of S (AND): the zeta-up coalition values of a LATTICE-001 histogram
        (projection "true_sets" of a three-valued histogram by default, "complete" of an "apart" one)"""
        return cls._lattice(hist, 2, projection, players, budget)

    @classmethod
    def disjunction(cls, hist, *, projection=None, players=None, budget=None):
        """v(S) = COUNT(*) WHERE the conditions of S joined by OR (v(empty) = 0)"""
        return cls._lattice(hist, 3, projection, players, budget)

    @classmethod
    def predicate(cls, masks, program, *, baseline=True, players=None, budget=None):
        """v(S) = the points where the BOOL-001 program is TRUE with the conditions outside S replaced by their
        baseline constants (True: TRUE, False: FALSE; one bool or one per condition); SQL three-valued logic"""
        if not isinstance(masks, mb.Masks) or not isinstance(program, mb.Program):
            raise TypeError("masks: a math_bool.Masks; program: a math_bool.Program")
        masks._open()
        masks.require_fresh("attribution game")
        k = program.k
        base = [bool(baseline)] * k if isinstance(baseline, (bool, np.bool_)) else [bool(b) for b in baseline]
        if len(base) != k:
            raise ValueError(f"baseline: one constant per condition ({k})")
        bits = sum(1 << i for i, b in enumerate(base) if b)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_attr_game_predicate", C.byref(masks.domain()), C.byref(program.view()), bits,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        measure = "the selected rows of the scope" if masks.kind == "rows" else "the 2^k assignments"
        conv = dict(game=f"v(S) = the points of {measure} where the predicate is TRUE with every condition outside S "
                         "replaced by its baseline constant (SQL three-valued logic: UNKNOWN does not count)",
                    players="the predicate's conditions",
                    baseline="v(empty): every condition at its baseline constant ("
                             + ", ".join(f"c{i}={'TRUE' if b else 'FALSE'}" for i, b in enumerate(base)) + ")",
                    value="COUNT(*)", baseline_constants=tuple(base))
        names = players
        if names is None and masks.kind == "rows" and len(masks.conditions) == k:
            names = tuple(repr(c) for c in masks.conditions)
        return cls._wrap(_made(h, error, refusal, "predicate game"), names, conv, keep=(masks, program),
                         fresh=lambda what: masks.require_fresh(what))

    @classmethod
    def provenance(cls, relation, *, tuples=None, weights=None, semiring="count", players="tuples",
                   others="present", budget=None):
        """A PROV-001 game: v(S) = sum_j w_j a_j(S), a_j the annotation of tuple j of the relation (multiplicity in
        "count", existence 0/1 in "bool") when exactly the source tuples of the players in S are present.
        tuples: indices (default all); weights: None (1 each: COUNT(*)), a list, or a column name (SUM(column): a NULL
        value contributes nothing); players: "tuples" (each source tuple of the cone), "sources" (one per source
        relation), or a list whose entries are a label (name, row), a source name, or a collection of those (a group);
        others: the source tuples of no player stay "present" or are "absent"."""
        if not isinstance(relation, mp.Relation):
            raise TypeError("relation: a math_prov.Relation")
        st = relation.store
        anns = relation.annotations()
        idx = list(range(len(anns))) if tuples is None else [int(i) for i in tuples]
        if not idx or any(not 0 <= i < len(anns) for i in idx):
            raise ValueError("tuples: indices of the relation's tuples (at least one)")
        if semiring not in ("count", "bool", "N", "B"):
            raise ValueError("semiring is 'count' (N) or 'bool' (B)")
        sr = 2 if semiring in ("count", "N") else 1
        if weights is None:
            w = [1] * len(idx)
            value = "COUNT(*)" if sr == 2 else "the number of these output tuples that exist"
        elif isinstance(weights, str):
            col = relation._col(weights)
            tps = relation.tuples()
            w = [tps[i][col] if tps[i][col] is not None else 0 for i in idx]
            scale = relation.scales[col]
            value = (f"SUM({weights}) in logical units" + (f" (value x 10^{scale})" if scale else "") +
                     "; a NULL value contributes nothing, the sum of nothing is 0")
        else:
            w = [int(x) for x in weights]
            if len(w) != len(idx):
                raise ValueError("weights: one per selected tuple")
            value = "sum_j w_j a_j (declared weights)"
        for x in w:
            if isinstance(x, bool) or not isinstance(x, Integral) or not -(1 << 63) <= int(x) < (1 << 63):
                raise TypeError("weights are int64 exact integers")
        if others not in ("present", "absent"):
            raise ValueError("others is 'present' or 'absent'")
        nodes = np.array([int(anns[i]) for i in idx], dtype=np.uint32)
        error, refusal = C.c_int32(), _Refusal()
        ph = mp._call("tr_prov_plan_build", st._open(), nodes.ctypes.data, len(nodes), C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
        ph = _made(ph, error, refusal, "provenance plan")
        try:
            view = mp._PlanView()
            _check(mp._call("tr_prov_plan_get_view", ph, C.byref(view)), what="plan view")
            labels = st._labels()
            cnt = u64()
            _check(_call("tr_attr_plan_variables", C.byref(view), None, 0, C.byref(cnt)), what="plan variables")
            cone = np.zeros(max(int(cnt.value), 1), dtype=np.uint32)
            _check(_call("tr_attr_plan_variables", C.byref(view), cone.ctypes.data, len(cone), C.byref(cnt)),
                   what="plan variables")
            cone = [int(x) for x in cone[:int(cnt.value)]]
            fixed = 0xFFFFFFFF if others == "present" else 0xFFFFFFFE
            vp_map = np.full(max(int(view.nvars), 1), fixed, dtype=np.uint32)
            if players == "tuples":
                names = tuple(labels[x] for x in cone)
                for p, x in enumerate(cone):
                    vp_map[x] = p
                pdesc = "every source tuple of the cone is one player"
            elif players == "sources":
                order = []
                for x in cone:
                    if labels[x][0] not in order:
                        order.append(labels[x][0])
                names = tuple(order)
                for x in cone:
                    vp_map[x] = order.index(labels[x][0])
                pdesc = "one player per source relation (all its tuples)"
            else:
                entries = list(players)
                by_label = {lab: i for i, lab in enumerate(labels)}
                names = []
                for p, ent in enumerate(entries):
                    items = [ent] if isinstance(ent, str) or (isinstance(ent, tuple) and len(ent) == 2 and
                                                              isinstance(ent[0], str)) else list(ent)
                    for it in items:
                        if isinstance(it, str):
                            hit = [i for i, lab in enumerate(labels) if lab[0] == it]
                            if not hit:
                                raise ValueError(f"no source {it!r}")
                        else:
                            if tuple(it) not in by_label:
                                raise ValueError(f"no source tuple {it!r}")
                            hit = [by_label[tuple(it)]]
                        for x in hit:
                            if vp_map[x] != fixed:
                                raise ValueError(f"source tuple {labels[x]!r} belongs to two players")
                            vp_map[x] = p
                    names.append(ent if not isinstance(ent, (list, set, frozenset)) else tuple(sorted(ent, key=repr)))
                names = tuple(names)
                pdesc = "the declared groups of source tuples"
            n = len(names)
            if n < 1:
                raise ValueError("the game has no player (the cone reaches no source tuple)")
            wa = np.array(w, dtype=np.int64)
            h = _call("tr_attr_game_prov", C.byref(view), sr, wa.ctypes.data, vp_map.ctypes.data, n,
                      C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            h = _made(h, error, refusal, "provenance game")
        finally:
            mp._call("tr_prov_plan_free", ph)
        conv = dict(game=f"PROV-001 provenance of {len(idx)} output tuple(s) in "
                         f"{'N (multiplicities)' if sr == 2 else 'B (existence, 0/1)'}: v(S) = sum_j w_j a_j(S), "
                         "a_j(S) the annotation of output tuple j under the 0/1 valuation where exactly the source "
                         f"tuples of the players in S are present (the others' fixed {others})",
                    players=pdesc,
                    baseline=f"v(empty): every player's source tuples deleted (the source tuples of no player "
                             f"{others})",
                    value=value, semiring="N" if sr == 2 else "B")
        return cls._wrap(h, names, conv, keep=(st,))

    # ------------------------------------------------------------------ reading
    def _open(self, what="game"):
        if not self._handle:
            raise ValueError("the game is closed")
        if self._fresh is not None:
            self._fresh(what)
        return self._handle

    def _raw_info(self):
        out = _GameInfo()
        _check(_call("tr_attr_game_get_info", self._handle, C.byref(out)), what="game")
        return out

    def view(self):
        v = _GameView()
        _check(_call("tr_attr_game_get_view", self._open(), C.byref(v)), what="game view")
        return v

    @property
    def info(self):
        i = self._raw_info()
        out = dict(kind=("table", "provenance")[int(i.kind) - 1], source=SOURCES[int(i.source)], n=int(i.n),
                   monotone={1: "increasing", -1: "decreasing", 0: None}[int(i.monotone)],
                   moebius_nonnegative=bool(i.sign), degree=int(i.degree), lo=_int(i.lo), hi=_int(i.hi),
                   bound=_int(i.bound), empty=_int(i.empty), full=_int(i.full), bytes=int(i.bytes))
        if int(i.kind) == PROV:
            out.update(semiring="N" if i.semiring == 2 else "B", variables=int(i.nvars), fixed_present=int(i.present),
                       fixed_absent=int(i.absent), outputs=int(i.noutputs), nodes=int(i.nodes),
                       signed_weights=bool(i.signed_weights), players_used=int(i.players_used))
        return out

    def _mask(self, coalition):
        if isinstance(coalition, Integral) and not isinstance(coalition, bool):
            m = int(coalition)
            if not 0 <= m < (1 << self.n):
                raise ValueError(f"coalition mask {m} outside the {self.n} players")
            return m
        m = 0
        for p in coalition:
            i = self.players.index(p) if p in self.players else (int(p) if isinstance(p, Integral) and
                                                                     not isinstance(p, bool) else None)
            if i is None or not 0 <= i < self.n:
                raise ValueError(f"no player {p!r}")
            m |= 1 << i
        return m

    def values(self, coalitions, *, backend="cpu", budget=None):
        """v(S) for each coalition (a mask or an iterable of players): exact ints"""
        if self.n > MAX_ENUM:
            raise UnsupportedError(f"coalition values: {self.n} players > {MAX_ENUM}")
        ms = [self._mask(c) for c in coalitions]
        k = len(ms)
        masks = np.array(ms or [0], dtype=np.uint64)
        out = np.zeros((max(k, 1), 2), dtype=np.int64)
        refusal = _Refusal()
        if _backend(backend) == "cpu":
            _check(_call("tr_attr_values", self._open("coalition values"), masks.ctypes.data, k, out.ctypes.data,
                         C.byref(_budget(budget)), C.byref(refusal)), refusal, "coalition values")
        else:
            gpu = _gpu()
            v = self.view()
            gpu.run("tr_mgpu_attr_values", C.byref(v), masks.ctypes.data, k, out.ctypes.data, C.byref(_budget(budget)),
                    C.byref(refusal), refusal=refusal, what="coalition values on the device")
        return _from_words(out, k)

    def value(self, coalition, **kw):
        return self.values([coalition], **kw)[0]

    def need(self, op, arg=0, *, backend="cpu"):
        """the needs of "values" (arg coalitions), "exhaustive", "explore" / "search" (arg = max_coalitions)"""
        code = {"values": OP_VALUES, "exhaustive": OP_EXHAUSTIVE, "explore": OP_EXPLORE, "search": OP_SEARCH}[op]
        raw = _Need()
        _check(_call("tr_attr_enum_need", self._open(), code, int(arg), _NEED_CPU if _backend(backend) == "cpu"
                     else _NEED_DEVICE, C.byref(raw)), what=f"{op} needs")
        return _needs(raw)

    def support_need(self, method="auto"):
        raw = _Need()
        _check(_call("tr_attr_support_need", self._open(), _method(method), C.byref(raw)), what="support needs")
        return _needs(raw)

    def support(self, method="auto", *, budget=None):
        """the sparse Moebius (Harsanyi) support of v: "expansion" (provenance in N), "witnesses" (in B),
        "transform" (2^n values: tables, or a small provenance game), "auto" (the first that applies)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_attr_support_build", self._open("support"), _method(method), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Support._wrap(_made(h, error, refusal, "Moebius support"), self)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_attr_game_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Game(closed)"
        return f"Game({self.info['source']}, n={self.n})"


def _method(method):
    if method not in METHODS:
        raise ValueError(f"method is one of {', '.join(METHODS)}")
    return METHODS[method]


class Support:
    """An immutable sparse Moebius support: the nonzero m(T) (T a set of players), canonical order (by size, then
    the largest player first, as LATTICE-002 orders its sets)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Game.support(...) or Support.from_terms(...)")

    @classmethod
    def _wrap(cls, handle, game, players=None, convention=None):
        self = object.__new__(cls)
        self._handle, self.game = handle, game
        i = self._raw_info()
        self.n = int(i.n)
        self.players = game.players if game is not None else _names(players, self.n)
        self.convention = dict(game.convention) if game is not None else dict(convention)
        return self

    @classmethod
    def from_terms(cls, n, terms, *, players=None, budget=None):
        """a support from declared terms {set of players: coefficient} (merged exactly, zeros dropped); the game it
        defines is v(S) = sum of the coefficients of the subsets of S"""
        items = list(terms.items()) if isinstance(terms, dict) else list(terms)
        off, pl, coef = [0], [], []
        for s, c in items:
            ps = sorted(int(p) for p in s)
            if len(set(ps)) != len(ps):
                raise ValueError(f"term {s!r}: a player appears twice")
            pl += ps
            off.append(len(pl))
            coef.append(c)
        offa = np.array(off, dtype=np.uint64)
        pla = np.array(pl or [0], dtype=np.uint32)
        words = _words(coef, "coefficients")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_attr_support_terms", int(n), len(items), offa.ctypes.data, pla.ctypes.data, words.ctypes.data,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        conv = dict(game="v(S) = the sum of the declared Moebius coefficients of the subsets of S",
                    players="the declared players 0..n-1", baseline="v(empty) = the coefficient of the empty set",
                    value="as declared")
        return cls._wrap(_made(h, error, refusal, "Moebius support"), None, players, conv)

    def _open(self):
        if not self._handle:
            raise ValueError("the support is closed")
        if self.game is not None and self.game._fresh is not None:
            self.game._fresh("Moebius support")
        return self._handle

    def _raw_info(self):
        out = _SupportInfo()
        _check(_call("tr_attr_support_get_info", self._handle, C.byref(out)), what="support")
        return out

    @property
    def info(self):
        i = self._raw_info()
        return dict(n=int(i.n), method=METHOD_NAMES[int(i.method)], terms=int(i.terms), pairs=int(i.pairs),
                    degree=int(i.degree), nonnegative=bool(i.nonnegative), candidates=int(i.candidates),
                    bytes=int(i.bytes), sum=_int(i.sum), empty=_int(i.empty), from_game=bool(i.from_game),
                    game_full=_int(i.game_full), game_empty=_int(i.game_empty),
                    reconstructs=bool(i.reconstructs))

    @property
    def terms(self):
        """{(players...): m(T)} in canonical order (players as the game names them)"""
        i = self._raw_info()
        t, pr = int(i.terms), int(i.pairs)
        off = np.zeros(t + 1, dtype=np.uint64)
        pl = np.zeros(max(pr, 1), dtype=np.uint32)
        coef = np.zeros((max(t, 1), 2), dtype=np.int64)
        _check(_call("tr_attr_support_get", self._open(), off.ctypes.data, pl.ctypes.data, coef.ctypes.data, max(t, 1),
                     max(pr, 1)), what="support terms")
        cs = _from_words(coef, t)
        return {tuple(self.players[int(p)] for p in pl[int(off[k]):int(off[k + 1])]): cs[k] for k in range(t)}

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_attr_support_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Support({self._raw_info().terms} terms, n={self.n})" if self._handle else "Support(closed)"


# ---------------------------------------------------------------------- indices
def _support_values(sup, index, budget):
    out = (_Q * max(sup.n, 1))()
    ident, refusal = _Identity(), _Refusal()
    _check(_call("tr_attr_support_values", sup._open(), INDICES[index], out, C.byref(ident), C.byref(_budget(budget)),
                 C.byref(refusal)), refusal, f"{index} values")
    vals = _qs(out, sup.n)
    idn = _identity(ident)
    info = sup.info
    rec = dict(reconstruction=dict(moebius_sum=info["sum"], game_full=info["game_full"],
                                   moebius_empty=info["empty"], game_empty=info["game_empty"],
                                   holds=info["reconstructs"]))
    return vals, ({"efficiency": idn} if idn else {}) | rec, dict(terms=info["terms"], degree=info["degree"],
                                                                     method=info["method"])


def _values(index, target, method, backend, budget):
    if index not in ("shapley", "banzhaf"):
        raise ValueError("index is 'shapley' or 'banzhaf'")
    if method not in ("auto", "support", "exhaustive"):
        raise ValueError("method is 'auto', 'support' or 'exhaustive'")
    _backend(backend)
    if isinstance(target, Support):
        if method == "exhaustive":
            raise ValueError("a support has no coalitions to enumerate: pass its Game")
        if backend == "cuda":
            raise UnsupportedError("the sparse support path runs on the CPU (method='exhaustive' has a CUDA kernel)")
        vals, ids, sup = _support_values(target, index, budget)
        return _result(target, index, dict(values=vals, by_player=dict(zip(target.players, vals)), exact=True,
                                           method="support", identities=ids, support=sup, backend="cpu",
                                           device="native CPU"))
    if not isinstance(target, Game):
        raise TypeError("a math_attr.Game or Support")
    if method == "exhaustive" or (method == "auto" and backend == "cuda"):
        r = exhaustive(target, backend=backend, budget=budget)
        vals = r["shapley"] if index == "shapley" else r["banzhaf"]
        ids = {"efficiency": r["identities"]["efficiency"]} if index == "shapley" else {}
        return _result(target, index, dict(values=vals, by_player=dict(zip(target.players, vals)), exact=True,
                                           method="exhaustive", identities=ids, coalitions=r["coalitions"],
                                           numerators=r["numerators"][index], denominator=r["denominators"][index],
                                           backend=r["backend"], device=r["device"]))
    if backend == "cuda":
        raise UnsupportedError("the sparse support path runs on the CPU (method='exhaustive' has a CUDA kernel)")
    sup = target.support(budget=budget)
    try:
        vals, ids, sinfo = _support_values(sup, index, budget)
    finally:
        sup.close()
    return _result(target, index, dict(values=vals, by_player=dict(zip(target.players, vals)), exact=True,
                                       method="support", identities=ids, support=sinfo, backend="cpu",
                                       device="native CPU"))


def shapley(target, *, method="auto", backend="cpu", budget=None):
    """Shapley values (exact Fractions) of a Game or a Support: "support" (from the sparse Moebius support, CPU),
    "exhaustive" (every coalition's weighted marginal contribution; CPU or CUDA), "auto" (support on the CPU,
    exhaustive on CUDA)"""
    return _values("shapley", target, method, backend, budget)


def banzhaf(target, *, method="auto", backend="cpu", budget=None):
    """Banzhaf values (exact Fractions), as shapley()"""
    return _values("banzhaf", target, method, backend, budget)


def interactions(target, order=2, *, index="shapley", min_order=1, budget=None):
    """Every set S of players with min_order <= |S| <= order and a nonzero index (the others are exactly 0):
    index "moebius" (Harsanyi dividends), "shapley" (Shapley interaction), "banzhaf" (Banzhaf interaction),
    "shapley_taylor" (of order `order`, efficient). From the sparse Moebius support (CPU)."""
    if index not in INTERACTION_INDICES:
        raise ValueError(f"index is one of {', '.join(INTERACTION_INDICES)}")
    sup, own = (target, False) if isinstance(target, Support) else (None, True)
    if own:
        if not isinstance(target, Game):
            raise TypeError("a math_attr.Game or Support")
        sup = target.support(budget=budget)
    try:
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_attr_interactions", sup._open(), INTERACTION_INDICES[index], int(min_order), int(order),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        h = _made(h, error, refusal, f"{index} interactions")
        try:
            ti = _TableInfo()
            _check(_call("tr_attr_table_get_info", h, C.byref(ti)), what="interactions")
            k, pr = int(ti.sets), int(ti.pairs)
            off = np.zeros(k + 1, dtype=np.uint64)
            pl = np.zeros(max(pr, 1), dtype=np.uint32)
            vals = (_Q * max(k, 1))()
            _check(_call("tr_attr_table_get", h, off.ctypes.data, pl.ctypes.data, vals, max(k, 1), max(pr, 1)),
                   what="interactions")
        finally:
            _call("tr_attr_table_free", h)
        sets = [tuple(sup.players[int(p)] for p in pl[int(off[j]):int(off[j + 1])]) for j in range(k)]
        values = _qs(vals, k)
        idn = _identity(ti.identity)
        name = {"moebius": "moebius", "shapley": "shapley_interaction", "banzhaf": "banzhaf_interaction",
                "shapley_taylor": "shapley_taylor"}[index]
        return _result(sup, name, dict(order=int(order), min_order=int(min_order), sets=sets, values=values,
                                       by_set=dict(zip(sets, values)), exact=True, method="support",
                                       candidates=int(ti.candidates), identities={idn["identity"]: idn} if idn else {},
                                       note="every set not listed has index exactly 0", backend="cpu",
                                       device="native CPU"))
    finally:
        if own:
            sup.close()


def interactions_need(support, order=2, *, index="shapley", min_order=1):
    """the needs interactions() checks before allocating (tensor = output = the (set, term) candidates)"""
    if not isinstance(support, Support):
        raise TypeError("a math_attr.Support")
    if index not in INTERACTION_INDICES:
        raise ValueError(f"index is one of {', '.join(INTERACTION_INDICES)}")
    raw = _Need()
    _check(_call("tr_attr_interactions_need", support._open(), INTERACTION_INDICES[index], int(min_order), int(order),
                 C.byref(raw)), what="interaction needs")
    return _needs(raw)


def exhaustive(game, *, backend="cpu", budget=None):
    """Shapley and Banzhaf values by every coalition's weighted marginal contribution (exact numerators over n! and
    2^(n-1), 128-bit; CUDA: the accumulation kernel), with the efficiency identity"""
    if not isinstance(game, Game):
        raise TypeError("a math_attr.Game")
    n = game.n
    if n > MAX_EXHAUSTIVE:
        raise UnsupportedError(f"exhaustive enumeration: {n} players > {MAX_EXHAUSTIVE}")
    shn, bzn = np.zeros((n, 2), dtype=np.int64), np.zeros((n, 2), dtype=np.int64)
    sh, bz = (_Q * n)(), (_Q * n)()
    info, refusal = _ExhaustiveInfo(), _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_attr_exhaustive", game._open("exhaustive attribution"), shn.ctypes.data, bzn.ctypes.data, sh,
                     bz, C.byref(info), C.byref(_budget(budget)), C.byref(refusal)), refusal, "exhaustive attribution")
        label = ("cpu", "native CPU")
    else:
        gpu = _gpu()
        v = game.view()
        gpu.run("tr_mgpu_attr_exhaustive", C.byref(v), shn.ctypes.data, bzn.ctypes.data, C.byref(_budget(budget)),
                C.byref(refusal), refusal=refusal, what="exhaustive attribution on the device")
        _check(_call("tr_attr_exhaustive_adopt", game._open(), shn.ctypes.data, bzn.ctypes.data, sh, bz,
                     C.byref(info)), what="exhaustive attribution (device numerators)")
        label = _device_label(gpu)
    out = dict(players=game.players, shapley=_qs(sh, n), banzhaf=_qs(bz, n), coalitions=int(info.coalitions),
               numerators=dict(shapley=_from_words(shn, n), banzhaf=_from_words(bzn, n)),
               denominators=dict(shapley=_int(info.shapley_den), banzhaf=_int(info.banzhaf_den)),
               identities=dict(efficiency=_identity(info.efficiency)), exact=True, method="exhaustive",
               convention=dict(game.convention, weighting={k: WEIGHTING[k] for k in ("shapley", "banzhaf")}),
               causal=False, interpretation=INTERPRETATION, backend=label[0], device=label[1])
    return out


def explore(game, max_coalitions=None, *, backend="cpu", budget=None):
    """Shapley and Banzhaf values within an explicit budget of evaluated coalitions (whole levels in a fixed order;
    v(empty), v(P) are the game's): exact ("values") or certified bounds ("bounds": lo <= index <= hi), never
    presented as exact, with the explored states counted (math_attr.h, explore)"""
    if not isinstance(game, Game):
        raise TypeError("a math_attr.Game")
    n = game.n
    if n > MAX_ENUM:
        raise UnsupportedError(f"exploration: {n} players > {MAX_ENUM}")
    b = budget or Budget()
    if max_coalitions is None:
        max_coalitions = b.limits["iterations"]
    arrs = [(_Q * n)() for _ in range(4)]
    info, refusal = _ExploreInfo(), _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_attr_explore", game._open("bounded attribution"), int(max_coalitions), *arrs, C.byref(info),
                     C.byref(b.raw), C.byref(refusal)), refusal, "bounded attribution")
        label = ("cpu", "native CPU")
    else:
        gpu = _gpu()
        v = game.view()
        gpu.run("tr_mgpu_attr_explore", C.byref(v), int(max_coalitions), *arrs, C.byref(info), C.byref(b.raw),
                C.byref(refusal), refusal=refusal, what="bounded attribution on the device")
        label = _device_label(gpu)
    sl, sh, bl, bh = (_qs(a, n) for a in arrs)
    exact = bool(info.exact)
    mask = int(info.level_mask)
    out = dict(players=game.players, exact=exact, status="exact" if exact else "bounds",
               regime={1: "moebius", 2: "range"}[int(info.regime)],
               bounds=dict(shapley=list(zip(sl, sh)), banzhaf=list(zip(bl, bh))),
               explored=int(info.explored), known=int(info.explored) + 2, total=int(info.total),
               levels=[s for s in range(n + 1) if (mask >> s) & 1], bottom=int(info.bottom), degree=int(info.degree),
               max_coalitions=int(info.budget), range=(_int(info.lo), _int(info.hi)),
               identities=dict(efficiency=_identity(info.efficiency)),
               convention=dict(game.convention, weighting={k: WEIGHTING[k] for k in ("shapley", "banzhaf")}),
               causal=False, interpretation=INTERPRETATION, backend=label[0], device=label[1],
               note="levels 0 and n are the game's own v(empty), v(P) (not counted in explored)")
    if exact:
        out["values"] = dict(shapley=sl, banzhaf=bl)
    return out


def search(game, kind="repair", op="<=", threshold=0, *, max_coalitions=None, max_solutions=16, backend="cpu",
           budget=None):
    """The minimum coalitions C with a declared property: kind "explanation" (v(C) op threshold: C alone suffices)
    or "repair" (v(P - C) op threshold: removing C suffices), levels |C| = 0, 1, ... within max_coalitions
    evaluations: "found" (minimum size and multiplicity exact), "none" (proven), or "incomplete" with certified
    bounds on the minimum size and multiplicity, and the explored states counted"""
    if not isinstance(game, Game):
        raise TypeError("a math_attr.Game")
    n = game.n
    if n > MAX_ENUM:
        raise UnsupportedError(f"search: {n} players > {MAX_ENUM}")
    if kind not in KINDS or op not in OPS:
        raise ValueError(f"kind is one of {', '.join(KINDS)}; op one of {', '.join(OPS)}")
    b = budget or Budget()
    if max_coalitions is None:
        max_coalitions = b.limits["iterations"]
    q = _SearchQuery(kind=KINDS[kind], op=OPS[op], max_coalitions=int(max_coalitions),
                     max_solutions=int(max_solutions))
    if isinstance(threshold, bool) or not isinstance(threshold, Integral):
        raise TypeError("threshold is an exact integer")
    _set_i128(q.threshold, int(threshold))
    sols = np.zeros(max(int(max_solutions), 1), dtype=np.uint64)
    info, refusal = _SearchInfo(), _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_attr_search", game._open("minimum search"), C.byref(q), sols.ctypes.data, C.byref(info),
                     C.byref(b.raw), C.byref(refusal)), refusal, "minimum search")
        label = ("cpu", "native CPU")
    else:
        gpu = _gpu()
        v = game.view()
        gpu.run("tr_mgpu_attr_search", C.byref(v), C.byref(q), sols.ctypes.data, C.byref(info), C.byref(b.raw),
                C.byref(refusal), refusal=refusal, what="minimum search on the device")
        label = _device_label(gpu)
    status = STATUS[int(info.status)]
    lo, hi = int(info.size_lo), int(info.size_hi)
    known = lo == hi and hi >= 0
    mult = (int(info.mult_lo), None if int(info.mult_hi) == _U64 else int(info.mult_hi)) if known else None
    sols = [tuple(game.players[i] for i in range(n) if (int(m) >> i) & 1) for m in sols[:int(info.listed)].tolist()]
    what = "v(C)" if KINDS[kind] == 1 else "v(P - C)"
    out = dict(kind="explanation" if KINDS[kind] == 1 else "repair", op=op, threshold=int(threshold),
               property=f"{what} {op} {int(threshold)}", status=status, exact=status != "incomplete",
               reason=REASONS[int(info.reason)], min_size=(lo, hi if hi >= 0 else None) if status != "none" else None,
               multiplicity=mult, solutions=sols, listed=int(info.listed), explored=int(info.explored),
               level=int(info.level), complete_levels=int(info.complete), level_explored=int(info.level_explored),
               level_total=int(info.level_total), max_coalitions=int(max_coalitions),
               monotone={1: "increasing", -1: "decreasing", 0: None}[int(info.monotone)],
               convention=dict(game.convention, weighting="none: a search for the coalitions of minimum size with "
                               f"the property {what} {op} {int(threshold)} (levels |C| = 0, 1, ... in increasing mask "
                               "order; every minimum solution counts once)"),
               causal=False, interpretation=INTERPRETATION, backend=label[0], device=label[1],
               note="C = empty and C = P are decided from the game's v(empty), v(P) (not counted in explored)")
    if status == "found":
        out["minimum"], out["count"] = lo, mult[0]
    return out
