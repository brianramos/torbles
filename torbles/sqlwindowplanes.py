# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""CUDA rank and integer prefix reductions over physical bitplanes."""

from copy import copy
import ctypes as C
import os

import numpy as np

from torbles import algq
from torbles.sqlrowplanes import _ordered_columns


_COMPARISONS = {"=": 1, "<>": 2, "<": 3, "<=": 4, ">": 5, ">=": 6}
_REVERSE = {1: 1, 2: 2, 3: 5, 4: 6, 5: 3, 6: 4}


def _qualification(expr, window, E):
    if not isinstance(expr, E.I.Func) or expr.fn not in _COMPARISONS or len(expr.args) != 2:
        return None
    left, right = (E._fold_consts(arg) for arg in expr.args)
    op = _COMPARISONS[expr.fn]
    if isinstance(left, E.I.Lit):
        left, right, op = right, left, _REVERSE[op]
    while isinstance(left, E.I.Cast) and not left.try_ and E._widening(left.arg, left.t):
        left = left.arg
    if (left.key() != window.key() or not isinstance(right, E.I.Lit)
            or type(right.v) is not int or not -(1 << 63) <= right.v < (1 << 63)):
        return None
    return op, right.v



def _terminal(select, window, items, qualify, E):
    if (select.distinct or select.distinct_on is not None or select.percent or select.with_ties
            or (select.qualify is not None and qualify is None)
            or any(not isinstance(expr, (E.I.Col, E.I.Lit)) and expr.key() != window.key()
                   for expr in items)):
        return None
    limit = select.limit.v if isinstance(select.limit, E.I.Lit) else select.limit
    offset = select.offset.v if isinstance(select.offset, E.I.Lit) else select.offset
    if ((limit is not None and (type(limit) is not int or not 0 <= limit < (1 << 64) - 1))
            or (offset is not None and (type(offset) is not int or not 0 <= offset < 1 << 64))):
        return None
    descending = -1
    if select.order:
        if len(select.order) != 1:
            return None
        spec, desc, _ = select.order[0]
        expr = items[spec[1]] if spec[0] == "pos" else spec[1]
        if expr.key() != window.key():
            return None
        descending = int(desc)
    elif limit is None and not offset:
        return None
    return descending, offset or 0, (1 << 64) - 1 if limit is None else limit


def _source(fact, keys, directions, measure):
    source = algq._plan_table(fact)
    cols = (C.c_int32 * len(keys))(*(source.idx(key) for key in keys))
    desc = (C.c_int32 * len(keys))(*directions)
    measure_col = -1 if measure is None else source.idx(measure)
    return algq._run_on(source), cols, desc, measure_col


def _reduce_values(fact, keys, directions, npart, mode, measure, offset, bitmap, stats,
                   qualify, second, aggregate):
    native = getattr(algq.gpu_lib(), "gpu_plane_window_aggregate", None)
    if native is None:
        return None
    native.restype = C.c_int
    native.argtypes = [
        algq.vp, algq.vp, C.c_int32, algq.vp, algq.vp, C.c_int32, C.c_int32, C.c_int32, C.c_int64,
        C.POINTER(C.c_int64), C.POINTER(C.c_uint64), C.POINTER(C.c_double),
        C.c_int32, C.c_int64, C.c_int32, C.c_int64, C.c_int32, C.POINTER(C.c_uint32),
    ]
    table, cols, desc, measure_col = _source(fact, keys, directions, measure)
    value, nout, elapsed, planes = C.c_int64(), C.c_uint64(), C.c_double(), C.c_uint32()
    with algq._GPU_LOCK:
        try:
            handle = table.gpu_handle()
        except RuntimeError:
            return None
        rc = native(handle, None if bitmap is None else bitmap.ctypes.data, len(keys), cols, desc, npart,
                    mode, measure_col, offset, value, nout, elapsed,
                    *(qualify or (0, 0)), *(second or (0, 0)), aggregate, planes)
    if rc:
        return None
    stats["cuda_window_aggregate"] = stats.get("cuda_window_aggregate", 0) + 1
    stats["cuda_ordered_window_ms"] = elapsed.value
    stats["cuda_window_download_bytes"] = nout.value * 8
    if planes.value:
        stats["cuda_window_torus_planes"] = planes.value
    return (value.value if nout.value else None,)


def _window_values(fact, keys, directions, npart, mode, measure, offset, bitmap, stats, qualify=None, terminal=None, second=None):
    torus = second is not None or (qualify is not None and os.environ.get("TORBLES_WINDOW_PLANES") == "1")
    symbol = ("gpu_plane_window_torus_terminal" if torus else
              "gpu_plane_window_terminal" if terminal is not None else
              "gpu_plane_window_qualify" if qualify else "gpu_plane_window")
    native = getattr(algq.gpu_lib(), symbol, None)
    if native is None:
        return None
    native.restype = C.c_int
    native.argtypes = [
        algq.vp, algq.vp, C.c_int32, algq.vp, algq.vp, C.c_int32, C.c_int32,
        C.c_int32, C.c_int64, algq.vp, algq.vp, C.c_uint64,
        C.POINTER(C.c_uint64), C.POINTER(C.c_double),
    ]
    if qualify or terminal is not None:
        native.argtypes += [C.c_int32, C.c_int64]
    if terminal is not None or torus:
        native.argtypes += [C.c_int32, C.c_uint64, C.c_uint64]
    plane_width = C.c_uint32()
    if torus:
        native.argtypes += [C.POINTER(C.c_uint32), C.c_int32, C.c_int64]
    table, cols, desc, measure_col = _source(fact, keys, directions, measure)
    cap = fact.n if terminal is None else min(fact.n, terminal[2])
    ids = None if bitmap is None and qualify is None and terminal is None else np.empty(cap, np.uint32)
    values = np.empty(cap, np.int64)
    extra = qualify or ((0, 0) if terminal is not None else ())
    if terminal is not None or torus:
        extra += terminal if terminal is not None else (-1, 0, (1 << 64) - 1)
    if torus:
        extra += (C.byref(plane_width), *(second or (0, 0)))
    nout, elapsed = C.c_uint64(), C.c_double()
    with algq._GPU_LOCK:
        try:
            handle = table.gpu_handle()
        except RuntimeError:
            return None
        rc = native(handle, None if bitmap is None else bitmap.ctypes.data, len(keys), cols, desc, npart,
                    mode, measure_col, offset, None if ids is None else ids.ctypes.data,
                    values.ctypes.data, len(values), nout, elapsed, *extra)
    if rc:
        return None
    stats["cuda_ordered_window_ms"] = elapsed.value
    stats["cuda_ordered_window_rows"] = nout.value
    stats["cuda_window_download_bytes"] = nout.value * (8 if ids is None else 12)
    if plane_width.value:
        stats["cuda_window_torus_planes"] = plane_width.value
    if qualify:
        stats["cuda_window_qualify_rows"] = nout.value
    if terminal is not None:
        stats["cuda_window_terminal_rows"] = nout.value
    return (range(fact.n) if ids is None else ids[:nout.value]), values[:nout.value]


def run(planner, atoms, preds):
    from torbles import sqlexec as E

    s, cx, fact = planner.s, planner.cx, planner.tab[planner.fact]
    if (cx.engine != "gpu" or not E.RAW_PLANE_ORDER or not E.COLUMNAR or not E.VEC_WINDOWS
            or os.environ.get("TORBLES_CUDA_WINDOWS") == "0"
            or planner.joins or planner.left or planner.fact_sample is not None or s.sample is not None
            or not (1 << 15) <= fact.n < 1 << 31):
        return None
    items = [expr for expr, _ in s.items]
    outputs = items + [s.qualify] + [items[spec[1]] if spec[0] == "pos" else spec[1] for spec, _, _ in s.order]
    if E.subqueries_in(outputs + [s.where]):
        return None
    windows = list(E.collect(outputs, E.I.Win).values())
    if len(windows) != 1:
        return None
    w = windows[0]
    if w.rpr is not None or w.filter is not None or w.agg_order or w.exclude is not None or w.distinct:
        return None

    mode, measure, offset = 0, None, 0
    if w.name == "sum" and len(w.args) == 1 and w.order and w.frame == (
        "rows", ("unbounded_preceding", None), ("current", None)
    ):
        arg = w.args[0]
        if not isinstance(arg, E.I.Col) or not arg.t.integer or w.t.id not in E.T.INT_RANGE:
            return None
        binding, measure = planner.canon(arg)
        codec = fact.codecs.get(measure)
        if (binding is not planner.fact or codec is None or codec.kind != "int"
                or measure in fact.nullable or measure in fact.wide or measure in fact.sidecars):
            return None
        offset = int(codec.offset)
        hi = offset + (1 << fact.width(measure)) - 1
        lo_bound, hi_bound = E.T.INT_RANGE[w.t.id]
        if (min(0, offset) * fact.n < max(-(1 << 63), lo_bound)
                or max(0, hi) * fact.n > min((1 << 63) - 1, hi_bound)):
            return None
        mode = 1
    elif (w.name != "row_number" or w.args
          or (w.frame is not None and any(bound[1] is not None for bound in w.frame[1:]))):
        return None

    keys, directions, npart = [], [], 0
    for expr, descending, partition in (
        [(expr, False, True) for expr in w.partition]
        + [(expr, desc, False) for expr, desc, _ in w.order]
    ):
        physical = _ordered_columns(planner, fact, expr)
        if not physical:
            return None
        keys.extend(physical)
        directions.extend([bool(descending)] * len(physical))
        if partition:
            npart += len(physical)
    if len(keys) > 12:
        return None

    bitmap = None if fact.used.all() else fact.used_bm
    if preds:
        from torbles.sqlgroupplans import bitmap as predicate_bitmap
        bitmap = predicate_bitmap(planner, preds)
        if bitmap is None:
            return None
    conditions = E._conjuncts(s.qualify)
    qualify = _qualification(s.qualify, w, E)
    second = None
    if len(conditions) == 2:
        left, right = (_qualification(condition, w, E) for condition in conditions)
        if left is not None and right is not None:
            qualify, second = left, right
    aggregate = getattr(s, "_cuda_window_reduction", None)
    if aggregate is not None and (s.qualify is None or qualify is not None):
        scalar = _reduce_values(fact, keys, directions, npart, mode, measure, offset, bitmap,
                                cx.stats, qualify, second, aggregate)
        if scalar is not None:
            rel = E.Rel(1)
            rel.cols[w.key()] = list(scalar)
            rel.plane_ordered = True
            cx.stats["engine_queries"] += 1
            return rel
    terminal = _terminal(s, w, items, qualify, E)
    result = _window_values(fact, keys, directions, npart, mode, measure, offset, bitmap, cx.stats, qualify, terminal, second)
    if result is None and terminal is not None:
        terminal = None
        result = _window_values(fact, keys, directions, npart, mode, measure, offset, bitmap, cx.stats, qualify, second=second)
    if result is None and qualify is not None:
        qualify = None
        result = _window_values(fact, keys, directions, npart, mode, measure, offset, bitmap, cx.stats)
    if result is None:
        return None
    ids, values = result
    payload = {col.key() for expr in outputs if expr is not None
               for col in E.I.find(expr, E.I.Col, stop=(E.I.Win,))}
    rel = planner.rows_rel({key: col for key, col in atoms.items() if key in payload}, preds, False, ids=ids,
                           fact_order=terminal is None or terminal[0] < 0)
    rel.cols.put(w.key(), E.ColArr(values))
    rel.native_window = True
    rel.native_qualified = qualify is not None
    rel.native_terminal = terminal is not None
    cx.stats["engine_queries"] += 1
    cx.stats["cuda_running_sum" if mode else "cuda_row_number"] = len(values)
    if not mode:
        cx.stats["raw_plane_row_number_rows"] = len(values)
    return rel



def push_derived_qualify(binding, conditions):
    """Move one exact outer window comparison through a simple derived table."""
    from torbles import sqlexec as E

    inner = binding.q
    if (not isinstance(inner, E.I.Select) or not isinstance(inner.from_, E.I.Binding)
            or inner.from_.table is None or inner.aggregate or inner.qualify is not None
            or inner.distinct or inner.distinct_on is not None or inner.order
            or inner.limit is not None or inner.offset is not None or inner.sample is not None
            or inner.from_.sample is not None
            or any(not isinstance(expr, (E.I.Col, E.I.Lit, E.I.Win)) for expr, _ in inner.items)):
        return inner
    names = [name for name, _, _ in binding.cols]
    for condition in conditions:
        cols = E.I.cols_of(condition)
        if len(cols) != 1:
            continue
        col = next(iter(cols))
        if col.b is not binding or col.name not in names:
            continue
        window = inner.items[names.index(col.name)][0]
        if not isinstance(window, E.I.Win):
            continue
        qualify = E.I.subst(condition, lambda expr: window if expr.key() == col.key() else None)
        if _qualification(qualify, window, E) is not None:
            derived = copy(inner)
            derived.qualify = qualify
            return derived
    return inner



def reduce_derived(select, cx):
    """Compose one scalar aggregate with a simple single-window derived query."""
    from torbles import sqlexec as E

    if (not select.aggregate or not isinstance(select.from_, E.I.Binding) or len(select.items) != 1
            or select.group or select.sets is not None or select.having is not None or select.qualify is not None
            or select.distinct or select.order or select.limit is not None or select.offset is not None
            or select.sample is not None):
        return None
    binding = select.from_
    inner = binding.q
    if (getattr(binding, "shared", None) is not None or binding.sample is not None
            or not isinstance(inner, E.I.Select) or not isinstance(inner.from_, E.I.Binding)
            or inner.from_.table is None or inner.from_.sample is not None or inner.aggregate
            or inner.distinct or inner.order or inner.limit is not None or inner.offset is not None
            or inner.sample is not None
            or any(not isinstance(expr, (E.I.Col, E.I.Lit, E.I.Win)) for expr, _ in inner.items)):
        return None
    windows = [(i, expr) for i, (expr, _) in enumerate(inner.items) if isinstance(expr, E.I.Win)]
    if len(windows) != 1:
        return None
    index, window = windows[0]
    name, _, _ = binding.cols[index]
    agg, label = select.items[0]
    modes = {"count_star": 1, "count": 1, "min": 2, "max": 3, "sum": 4}
    if (not isinstance(agg, E.I.Agg) or agg.name not in modes or agg.filter is not None
            or agg.order or agg.distinct or agg.t.id not in E.T.INT_RANGE):
        return None
    if agg.name != "count_star":
        if len(agg.args) != 1 or not isinstance(agg.args[0], E.I.Col):
            return None
        arg = agg.args[0]
        if arg.b is not binding or arg.name != name:
            return None

    prepared = copy(inner)
    if select.where is not None:
        if inner.qualify is not None:
            return None
        cols = E.I.cols_of(select.where)
        if not cols or any(col.b is not binding or col.name != name for col in cols):
            return None
        predicate = E.I.subst(select.where, lambda node: window if isinstance(node, E.I.Col) else None)
        terms = E._conjuncts(predicate)
        if not 1 <= len(terms) <= 2 or any(_qualification(term, window, E) is None for term in terms):
            return None
        prepared.qualify = predicate
    prepared.items = [inner.items[index]]
    prepared._cuda_window_reduction = modes[agg.name]
    before = cx.stats.get("cuda_window_aggregate", 0)
    result = E.run_query(prepared, cx)
    if cx.stats.get("cuda_window_aggregate", 0) == before:
        return None
    value = result.scalar()
    low, high = E.T.INT_RANGE[agg.t.id]
    if value is not None and not low <= value <= high:
        return None
    return E.Relation([label], [agg.t], [(value,)])
