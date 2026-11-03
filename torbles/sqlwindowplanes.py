# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""CUDA rank and integer prefix reductions over physical bitplanes."""

import ctypes as C
import os

import numpy as np

from torbles import algq
from torbles.sqlrowplanes import _ordered_columns


def _window_values(fact, keys, directions, npart, mode, measure, offset, bitmap, stats):
    native = getattr(algq.gpu_lib(), "gpu_plane_window", None)
    if native is None:
        return None
    native.restype = C.c_int
    native.argtypes = [
        algq.vp, algq.vp, C.c_int32, algq.vp, algq.vp, C.c_int32, C.c_int32,
        C.c_int32, C.c_int64, algq.vp, algq.vp, C.c_uint64,
        C.POINTER(C.c_uint64), C.POINTER(C.c_double),
    ]
    source = algq._plan_table(fact)
    cols = (C.c_int32 * len(keys))(*(source.idx(key) for key in keys))
    desc = (C.c_int32 * len(keys))(*directions)
    measure_col = -1 if measure is None else source.idx(measure)
    table = algq._run_on(source)
    ids = None if bitmap is None else np.empty(fact.n, np.uint32)
    values = np.empty(fact.n, np.int64)
    nout, elapsed = C.c_uint64(), C.c_double()
    with algq._GPU_LOCK:
        try:
            handle = table.gpu_handle()
        except RuntimeError:
            return None
        rc = native(handle, None if bitmap is None else bitmap.ctypes.data, len(keys), cols, desc, npart,
                    mode, measure_col, offset, None if ids is None else ids.ctypes.data,
                    values.ctypes.data, len(values), nout, elapsed)
    if rc:
        return None
    stats["cuda_ordered_window_ms"] = elapsed.value
    stats["cuda_ordered_window_rows"] = nout.value
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
    result = _window_values(fact, keys, directions, npart, mode, measure, offset, bitmap, cx.stats)
    if result is None:
        return None
    ids, values = result
    payload = {col.key() for expr in outputs if expr is not None
               for col in E.I.find(expr, E.I.Col, stop=(E.I.Win,))}
    rel = planner.rows_rel({key: col for key, col in atoms.items() if key in payload}, preds, False, ids=ids)
    rel.cols.put(w.key(), E.ColArr(values))
    rel.native_window = True
    cx.stats["engine_queries"] += 1
    cx.stats["cuda_running_sum" if mode else "cuda_row_number"] = len(values)
    if not mode:
        cx.stats["raw_plane_row_number_rows"] = len(values)
    return rel
