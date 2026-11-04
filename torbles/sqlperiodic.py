# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Weighted SQL aggregation over certified periodic native planes."""

import ctypes as C
from math import lcm

import numpy as np
import sqlir as I

from . import algq
from .sqlsnapshot import _revision

_MAX_WORDS = 128
_MIN_REPEATS = 8
_PREDICATES = {
    "and", "or", "not", "=", "<>", "<", "<=", ">", ">=", "between",
    "is_null", "is_not_null", "is_distinct", "is_not_distinct",
}


def _anchor(fact):
    at = fact.at
    return fact._native, at, _revision(at), fact.n, tuple(fact.cols)


def prepare(fact, max_words=_MAX_WORDS):
    """Certify the shortest packed-word periods of the fact's physical columns."""
    if type(max_words) is not int or not 1 <= max_words <= _MAX_WORDS:
        raise ValueError(f"max_words must be an integer from 1 through {_MAX_WORDS}")
    if fact.parts is not None or getattr(fact, "_vflag", None):
        return {}
    native = getattr(algq.lib, "at_periodic_periods", None)
    if native is None:
        return {}
    anchor = _anchor(fact)
    cached = getattr(fact, "_periodic", None)
    if cached is not None and cached[:2] == (anchor, max_words):
        return dict(cached[2])

    native.restype = C.c_int
    native.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_uint16), C.c_uint32]
    columns = tuple(fact.cols)
    periods = (C.c_uint16 * len(columns))()
    if native(anchor[1], max_words, periods, len(columns)) or _anchor(fact) != anchor:
        return {}
    result = {name: period for name, period in zip(columns, periods) if period}
    fact._periodic = anchor, max_words, result
    return dict(result)


def _full_scope(fact, full, tail):
    used = fact.used_bm
    all_bits = np.uint64((1 << 64) - 1)
    return (used.ndim == 1 and used.dtype == np.uint64 and len(used) >= full + bool(tail)
            and np.bitwise_and.reduce(used[:full], initial=all_bits) == all_bits
            and (not tail or int(used[full]) == (1 << tail) - 1)
            and not used[full + bool(tail):].any())


def _safe_predicate(expr, strip):
    expr = strip(expr)
    if isinstance(expr, I.Lit):
        return expr.v is None or type(expr.v) in (bool, int)
    if isinstance(expr, I.Col):
        return expr.t.integer
    return (isinstance(expr, I.Func) and expr.fn in _PREDICATES
            and all(_safe_predicate(arg, strip) for arg in expr.args))


def run(planner, rest, need, aggs, extra):
    """Return weighted representative rows for the usual SQL aggregate stage."""
    from . import sqlexec as E

    s, cx, binding = planner.s, planner.cx, planner.fact
    fact = planner.tab[binding]
    cached = getattr(fact, "_periodic", None)
    if (cached is None or cx.engine != "alg" or s.from_ is not binding or not s.aggregate
            or planner.joins or planner.left or planner.nulled or extra
            or planner.fact_sample is not None or s.sample is not None
            or getattr(binding, "writable", None) is not None
            or fact.parts is not None or getattr(fact, "_vflag", None)
            or s.sets is not None or s.qualify is not None or s.distinct or s.distinct_on is not None):
        return None

    top = [expr for expr, _ in s.items] + [s.having]
    top += [spec[1] for spec, _, _ in s.order if spec[0] == "expr"]
    if any(expr is not None and I.has(expr, (I.Win, I.Subq, I.Outer)) for expr in top):
        return None
    if any(not isinstance(expr, I.Col) or not expr.t.integer for expr in s.group):
        return None

    predicates = [E._fold_consts(expr) for expr in rest]
    atoms = dict(need)
    for agg in aggs.values():
        if (agg.name not in ("count_star", "count", "sum", "avg", "min", "max")
                or agg.order or agg.distinct or len(agg.args) != int(agg.name != "count_star")):
            return None
        for arg in agg.args:
            col = E._strip_cast(arg)
            if not arg.t.integer or not isinstance(col, I.Col) or not col.t.integer:
                return None
            atoms[col.key()] = col
    filters = predicates + [E._fold_consts(agg.filter) for agg in aggs.values() if agg.filter is not None]
    if any(not _safe_predicate(expr, E._strip_cast) for expr in filters):
        return None
    for expr in filters:
        atoms.update((col.key(), col) for col in I.cols_of(expr))
    if not atoms:
        return None

    anchor, _, periods = cached
    if _anchor(fact) != anchor:
        return None
    validity = getattr(fact, "nullable", {})
    refs = getattr(fact, "_refdim", {})
    moved = getattr(fact, "moved", {})
    cycle = 1
    for col in atoms.values():
        name = col.name
        if (col.b is not binding or not col.t.integer or name in refs or name in moved
                or name in fact.wide or getattr(fact.codecs.get(name), "kind", None) != "int"):
            return None
        for physical in (name, validity.get(name)):
            if physical is None:
                continue
            period = periods.get(physical)
            if period is None:
                return None
            cycle = lcm(cycle, period)
            if cycle > _MAX_WORDS:
                return None

    full, tail = divmod(fact.n, 64)
    repeats, remainder = divmod(full, cycle)
    if repeats < _MIN_REPEATS or not _full_scope(fact, full, tail):
        return None
    representatives = cycle * 64
    ids = np.arange(representatives, dtype=np.uint64)
    rel = planner.rows_rel(atoms, [], False, ids=ids)
    if _anchor(fact) != anchor:
        return None
    weights = np.full(representatives, repeats, dtype=np.int64)
    weights[:remainder * 64 + tail] += 1
    rel.cnt = weights.tolist()
    if predicates:
        rel = rel.take(E.rows_where(E._and(predicates), rel, cx))
    cx.stats["engine_queries"] += 1
    cx.stats["periodic_rows"] = cx.stats.get("periodic_rows", 0) + fact.n
    cx.stats["periodic_representatives"] = cx.stats.get("periodic_representatives", 0) + representatives
    return rel
