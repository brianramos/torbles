# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Pull exact inner-join filters back through a stored hierarchy."""

from copy import copy
from time import perf_counter_ns

import keydesign
import numpy as np
import sqlir as I

from . import algq, joinheader


_CELL_OPERATORS = {
    "and", "or", "not", "=", "<>", "<", "<=", ">", ">=", "between",
    "is_null", "is_not_null", "is_distinct", "is_not_distinct",
}


def _cell_expression(expr, binding, attrs, E):
    expr = E._strip_cast(expr)
    if isinstance(expr, I.Col):
        return expr.b is binding and expr.name in attrs
    if isinstance(expr, I.Lit):
        return expr.v is None or type(expr.v) in (int, bool, str)
    return (isinstance(expr, I.Func) and expr.fn in _CELL_OPERATORS
            and all(_cell_expression(arg, binding, attrs, E) for arg in expr.args))


def _predicate(term, target, fact, fk, record, cx, E):
    attrs = record["attrs"]
    if fk not in fact.msb_cols or not _cell_expression(term, target, attrs, E):
        return None
    cells = E.Rel(len(record["codes"]))
    for col in I.cols_of(term):
        cells.cols[col.key()] = attrs[col.name]
    selected = np.zeros(cells.n, bool)
    selected[E.rows_where(term, cells, cx)] = True
    width = fact.width(fk)
    bits, codes = record["bits"], np.asarray(record["codes"], np.int64)
    if bits and bits[0] >= width:
        # A fact with no rows in some of the dimension's high cells can
        # have narrower storage. Use its ordinary planner in that case.
        return None
    # The query may need fewer bits than the complete hierarchy cell.
    projection = keydesign.cell_bits(codes, selected)
    if projection is not None and len(projection) < len(bits):
        projected = np.zeros(len(codes), np.int64)
        for bit in projection:
            projected = (projected << 1) | ((codes >> bit) & 1)
        codes, first = np.unique(projected, return_index=True)
        bits = [bits[len(bits) - 1 - bit] for bit in projection]
        selected = selected[first]
    poly = algq.JoinPoly.from_cells(bits, width, codes, selected)
    planned = poly.plan("auto") or poly.plan("factored")
    if planned is None:
        return None
    predicate = algq._OnKey(fact, fk, target.table, None, cx.engine)
    predicate._planned = poly, planned
    predicate.sql = I.render(term)
    return predicate


def run(s, bindings, joins, conj, edges, need, aggs, cx):
    from . import sqlexec as E

    if (cx.engine not in ("alg", "gpu") or not getattr(cx.db, "_join_header", None)
            or not s.aggregate or not aggs or len(bindings) < 2 or s.sample is not None
            or s.sets is not None or s.distinct_on is not None
            or any(b.sample is not None or getattr(b, "writable", None) is not None for b in bindings)
            or len({id(b.table) for b in bindings}) != len(bindings)
            or any(j.kind not in ("inner", "cross") for j in joins)):
        return None

    started = perf_counter_ns()
    incoming = {}
    for cond, source, target in edges:
        if not all(E._join_key_term(arg) and E._strip_cast(arg).t.integer for arg in cond.args):
            continue
        if target.b in incoming:
            return None
        incoming[target.b] = cond, source, target
    roots = [b for b in bindings if b not in incoming]
    if len(roots) != 1:
        return None
    root = roots[0]
    fact = root.table
    paths = {root: ()}
    while len(paths) < len(bindings):
        before = len(paths)
        for binding, (_, source, _) in incoming.items():
            if binding not in paths and source.b in paths:
                paths[binding] = (*paths[source.b], source.name)
        if len(paths) == before:
            return None
    consumed = {id(edge[0]) for edge in incoming.values()}
    if any(id(c) not in consumed for join in joins for c in E._conjuncts(join.cond)):
        return None
    outputs = [expr for expr, _ in s.items] + list(s.group) + [s.having, s.qualify]
    outputs += [spec[1] for spec, _, _ in s.order if spec[0] == "expr"]
    if (any(c.b is not root for c in need.values()) or
            any(I.has(expr, (I.Subq, I.Outer, I.Win)) or any(c.b is not root for c in I.cols_of(expr))
                for expr in outputs if expr is not None)):
        return None

    records = {}
    for binding, path in paths.items():
        if not path:
            continue
        record = joinheader.record(cx.db, fact, path)
        if record is None or record["target"] != binding.table.name:
            return None
        records[binding] = record
    fact_terms, predicates = [], []
    try:
        for term in conj:
            if id(term) in consumed:
                continue
            scope = {col.b for col in I.cols_of(term)}
            if scope <= {root}:
                fact_terms.append(term)
                continue
            if len(scope) != 1:
                return None
            target = next(iter(scope))
            if target not in records:
                return None
            predicate = _predicate(term, target, fact, paths[target][0], records[target], cx, E)
            if predicate is None:
                return None
            predicates.append(predicate)
    except (TypeError, ValueError, OverflowError, E.SqlError):
        return None

    extra = [algq.RefOk(fact, fk) for fk in dict.fromkeys(path[0] for path in paths.values() if path)]
    extra.extend(predicates)
    # JoinPoly's program is small enough for the ordinary native path too.
    if len(extra) + sum(len(p._planned[0].program(0, p._planned[1])) for p in predicates) + len(extra) > 768:
        return None
    block = copy(s)
    block.from_ = root
    planner = E._Planner(block, root, {}, {}, cx)
    plan_ms = (perf_counter_ns() - started) / 1e6
    try:
        result = planner.run_pass(fact_terms, need, aggs, extra)
    except (TypeError, ValueError, OverflowError, E.SqlError, E._NotStarNow):
        return None
    cx.stats["star_blocks"] += 1
    cx.stats["join_header_paths"] = cx.stats.get("join_header_paths", 0) + len(records)
    cx.stats["join_header_predicates"] = cx.stats.get("join_header_predicates", 0) + len(predicates)
    cx.stats["join_header_plan_ms"] = cx.stats.get("join_header_plan_ms", 0.0) + plan_ms
    if any(len(path) > 1 for path in paths.values()):
        cx.stats["snowflake_blocks"] += 1
        cx.stats["snowflake_edges"] += sum(len(path) > 1 for path in paths.values())
    return result
