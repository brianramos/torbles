# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Fused scalar counts and exact integer moments over stored planes."""

import ctypes as C

import algq
import sqlir as I
import sqltypes as T
from . import sqlgroupplans
from .sqlgroupkeys import plane_group_key


_moments = getattr(algq.lib, "at_scalar_plane_moments", None)
if _moments is not None:
    _moments.restype = C.c_int
    _moments.argtypes = [algq.vp, algq.vp, C.c_int32, C.POINTER(algq._PlaneGroupKey),
                         C.POINTER(C.c_uint64)]


def _gpu_moments():
    fn = getattr(algq.gpu_lib(), "gpu_scalar_plane_moments", None)
    if fn is not None:
        fn.restype = C.c_int
        fn.argtypes = [algq.vp, algq.vp, C.c_int32, C.POINTER(algq._PlaneGroupKey),
                       C.POINTER(C.c_uint64), C.POINTER(C.c_double)]
    return fn


def run(planner, preds, aggs):
    from . import sqlexec as E

    s, cx = planner.s, planner.cx
    fact = planner.tab[planner.fact]
    if (not sqlgroupplans.single_fact(planner) or cx.engine not in ("alg", "gpu")
            or not E.ALG_MIN_ROWS <= fact.n < 1 << 32 or not s.aggregate or s.group or s.sets is not None
            or s.having is not None or s.order or s.limit is not None or s.offset is not None
            or s.qualify is not None or s.distinct or not aggs
            or any(expr.key() not in aggs for expr, _ in s.items)):
        return None
    source = algq._plan_table(fact)
    columns, plans = {}, {}
    for key, agg in aggs.items():
        if agg.distinct or agg.filter is not None or agg.order:
            return None
        if agg.name == "count_star":
            plans[key] = (agg.name, 0, 0)
            continue
        if agg.name not in ("sum", "avg", "count") or len(agg.args) != 1:
            return None
        if agg.name != "count" and not agg.args[0].t.integer:
            return None
        col = E._strip_cast(agg.args[0])
        if not isinstance(col, I.Col) or not col.t.integer:
            return None
        descriptor = plane_group_key(col, planner.fact, fact)
        if descriptor is None:
            return None
        if agg.name == "count":
            plans[key] = ("count_star", 0, 0)
            continue
        low, high, shift, transform, _, _ = descriptor
        codec = fact.codecs.get(low)
        if getattr(codec, "kind", None) != "int" or transform:
            return None
        offset = int(codec.offset)
        if agg.name == "sum":
            lo, hi = T.INT_RANGE[agg.t.id]
            if min(0, offset) * fact.n < lo or max(0, offset + fact.maxcode(low)) * fact.n > hi:
                return None
        if low not in columns:
            if len(columns) == 2:
                return None
            columns[low] = algq._PlaneGroupKey(source.idx(low), source.idx(high) if high else -1, shift, 0, 0)
        plans[key] = (agg.name, list(columns).index(low), offset)
    if not columns and (cx.engine != "gpu" or not (preds or planner.joins)):
        return None
    native = _gpu_moments() if cx.engine == "gpu" else _moments
    if native is None:
        return None
    scope = sqlgroupplans.bitmap(planner, preds)
    if scope is None:
        return None
    specs = (algq._PlaneGroupKey * len(columns))(*columns.values())
    out = (C.c_uint64 * 5)()
    view = algq._run_on(source)
    if cx.engine == "gpu":
        ms = C.c_double()
        with algq._GPU_LOCK:
            try:
                handle = view.gpu_handle()
            except RuntimeError:
                return None
            if native(handle, scope.ctypes.data, len(columns), specs, out, ms):
                return None
        cx.stats["cuda_scalar_moments"] = len(columns)
    elif native(view.at, scope.ctypes.data, len(columns), specs, out):
        return None
    count = out[0]
    rel = E.Rel(1)
    for key, (kind, index, offset) in plans.items():
        if kind == "count_star":
            result = count
        elif not count:
            result = None
        else:
            exact = (out[1 + 2 * index] | (out[2 + 2 * index] << 64)) + offset * count
            result = exact if kind == "sum" else exact / count
        rel.cols[key] = [result]
    rel.plane_ordered = True
    cx.stats["engine_queries"] += 1
    cx.stats["scalar_plane_moments"] = len(columns)
    return rel


_find_equal = getattr(algq.lib, "at_plane_equal_ids", None)
if _find_equal is not None:
    _find_equal.restype = C.c_int
    _find_equal.argtypes = [algq.vp, algq.vp, C.POINTER(algq._PlaneGroupKey), C.c_uint64, C.c_uint32,
                            algq.vp, C.POINTER(C.c_uint32)]


def point_ids(planner):
    """Bound a simple equality result directly on a split-wide column."""
    import numpy as np

    s = planner.s
    fact = planner.tab[planner.fact]
    term = s.where
    if (planner.cx.engine not in ("alg", "gpu") or not sqlgroupplans.single_fact(planner)
            or planner.joins or s.aggregate or s.order or s.limit is not None or s.offset is not None
            or s.qualify is not None or s.distinct or not isinstance(term, I.Func)
            or term.fn != "=" or len(term.args) != 2
            or any(not isinstance(expr, (I.Col, I.Lit)) for expr, _ in s.items)):
        return None
    col, literal = term.args
    if isinstance(col, I.Lit):
        col, literal = literal, col
    if not isinstance(col, I.Col) or not isinstance(literal, I.Lit) or type(literal.v) is not int:
        return None
    descriptor = plane_group_key(col, planner.fact, fact)
    if descriptor is None or descriptor[1] is None or not col.t.integer:
        return None
    low, high, shift, _, _, _ = descriptor
    codec = fact.codecs[low]
    if codec.kind != "int":
        return None
    code = literal.v - codec.offset
    if not 0 <= code < 1 << 64:
        return np.empty(0, np.uint64), fact.n
    source = algq._plan_table(fact)
    key = algq._PlaneGroupKey(source.idx(low), source.idx(high), shift, 0, 0)
    ids = np.empty(256, np.uint64)
    count = C.c_uint32()
    view = algq._run_on(source)
    if planner.cx.engine == "gpu":
        native = getattr(algq.gpu_lib(), "gpu_plane_equal_ids", None)
        if native is None:
            return None
        native.restype = C.c_int
        native.argtypes = [algq.vp, algq.vp, C.POINTER(algq._PlaneGroupKey), C.c_uint64,
                           algq.vp, C.POINTER(C.c_uint32)]
        with algq._GPU_LOCK:
            try:
                handle = view.gpu_handle()
            except RuntimeError:
                return None
            if native(handle, fact.used_bm.ctypes.data, key, code, ids.ctypes.data, count):
                return None
        ids[:count.value].sort()
        planner.cx.stats["cuda_plane_point_lookup"] = count.value
    elif _find_equal is None or _find_equal(view.at, fact.used_bm.ctypes.data, key, code,
                                           len(ids), ids.ctypes.data, count):
        return None
    planner.cx.stats["engine_queries"] += 1
    planner.cx.stats["raw_plane_point_lookup"] = count.value
    return ids[:count.value], fact.n
