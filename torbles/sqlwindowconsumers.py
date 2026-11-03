# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Exact relational consumers of one shared global CUDA rank temporary."""

import os

from torbles.sqlwindowplanes import _COMPARISONS, _REVERSE


def _enabled(cx):
    return cx.engine == "gpu" and os.environ.get("TORBLES_CUDA_WINDOWS") != "0"


def _clean(select, E, *, aggregate):
    return (isinstance(select, E.I.Select) and select.aggregate == aggregate
            and not select.group and select.sets is None and select.having is None
            and select.qualify is None and not select.distinct and select.distinct_on is None
            and not select.order and select.limit is None and select.offset is None
            and not select.with_ties and not select.percent and select.sample is None)


def _shared_program(left, right, cx, E):
    if not all(isinstance(side, E.I.Binding) and side.sample is None
               and not getattr(side, "rec", None) and not side.outer_null
               for side in (left, right)):
        return None
    shared = getattr(left, "shared", None)
    if shared is None or shared is not getattr(right, "shared", None) or id(shared) in cx.ctes:
        return None
    producer = left.q
    if (producer is not right.q or not _clean(producer, E, aggregate=False)
            or producer.where is not None or not isinstance(producer.from_, E.I.Binding)
            or producer.from_.table is None or producer.from_.sample is not None):
        return None
    # A global row number is unique by construction, even with ordering ties.
    # The two aliases therefore meet at one common physical row per rank.
    if len(producer.items) != 1 or len(left.cols) != 1 or len(right.cols) != 1:
        return None
    window = producer.items[0][0]
    if (not isinstance(window, E.I.Win) or window.name != "row_number" or window.partition
            or window.args or window.rpr is not None or window.filter is not None
            or window.agg_order or window.exclude is not None or window.distinct):
        return None
    return producer


def _rank_column(expr, binding, E):
    while isinstance(expr, E.I.Cast) and not expr.try_ and E._widening(expr.arg, expr.t):
        expr = expr.arg
    return (isinstance(expr, E.I.Col) and expr.b is binding
            and expr.name == binding.cols[0][0] and expr.t.integer)


def _comparison(term, binding, E):
    if term is None:
        return 0, 0
    if not isinstance(term, E.I.Func) or term.fn not in _COMPARISONS or len(term.args) != 2:
        return None
    left, right = (E._fold_consts(arg) for arg in term.args)
    op = _COMPARISONS[term.fn]
    if isinstance(left, E.I.Lit):
        left, right, op = right, left, _REVERSE[op]
    if (not _rank_column(left, binding, E) or not isinstance(right, E.I.Lit)
            or type(right.v) is not int or not -(1 << 63) <= right.v < 1 << 63):
        return None
    return op, right.v


def _count_star(expr, E):
    return (isinstance(expr, E.I.Agg) and expr.name == "count_star" and not expr.args
            and not expr.distinct and not expr.order and expr.filter is None)


def run_select(select, cx, execute):
    """Reduce an inner join of two aliases on the same unique window rank."""
    from torbles import sqlexec as E

    if (not _enabled(cx) or not _clean(select, E, aggregate=True)
            or not isinstance(select.from_, E.I.Join) or len(select.items) != 1):
        return None
    join = select.from_
    expr, label = select.items[0]
    if (join.kind != "inner" or getattr(join, "partition", None) is not None
            or not _count_star(expr, E)):
        return None
    left, right = join.left, join.right
    producer = _shared_program(left, right, cx, E)
    if producer is None:
        return None
    cond = join.cond
    if not isinstance(cond, E.I.Func) or cond.fn != "=" or len(cond.args) != 2:
        return None
    a, b = cond.args
    if not ((_rank_column(a, left, E) and _rank_column(b, right, E))
            or (_rank_column(b, left, E) and _rank_column(a, right, E))):
        return None
    terms = E._conjuncts(select.where)
    if len(terms) != 2:
        return None
    first, second = (_comparison(terms[0], left, E), _comparison(terms[1], right, E))
    if first is None or second is None:
        first, second = (_comparison(terms[1], left, E), _comparison(terms[0], right, E))
    if first is None or second is None:
        return None
    result = execute(producer, cx, "join_count", (first, second))
    if result is None:
        return None
    values, _ = result
    return E.Relation([label], [expr.t], [(int(values[0]),)])


def run_setop(query, cx, execute):
    """Combine two masks over one shared rank domain, or count both masks."""
    from torbles import sqlexec as E

    if (not _enabled(cx) or len(query.items) != 1 or query.order or query.limit is not None
            or query.offset is not None or query.with_ties or query.percent):
        return None
    counts = query.op == "union" and query.all
    kind = "counts" if counts else {"intersect": "intersection", "except": "difference", "union": "union"}.get(query.op)
    if kind is None or (query.all and not counts):
        return None
    left, right = query.left, query.right
    if not all(_clean(side, E, aggregate=counts) and len(side.items) == 1 for side in (left, right)):
        return None
    producer = _shared_program(left.from_, right.from_, cx, E)
    if producer is None:
        return None
    name, typ = query.items[0]
    comparisons = []
    for side in (left, right):
        expr = side.items[0][0]
        if expr.t != typ or not (_count_star(expr, E) if counts else
                                 _rank_column(expr, side.from_, E)):
            return None
        comparison = _comparison(side.where, side.from_, E)
        if comparison is None:
            return None
        comparisons.append(comparison)
    result = execute(producer, cx, kind, tuple(comparisons))
    if result is None:
        return None
    values, _ = result
    return E.Relation([name], [typ], [(int(value),) for value in values])
