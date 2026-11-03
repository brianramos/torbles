# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Bounded SQL row selection directly from stored order-key planes."""

import ctypes

import numpy as np

import algq
import sqlir as I
import sqltypes as T


_MAX_KEYS = 12
_MAX_ROWS = 1024
_topk = getattr(algq.lib, "at_plane_topk", None)
if _topk is not None:
    _topk.restype = ctypes.c_int
    _topk.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
    ]


def _ordered_columns(planner, fact, col):
    if not isinstance(col, I.Col) or T.coll_key(col.t) is not None:
        return ()
    binding, name = planner.canon(col)
    if binding is not planner.fact or name not in fact.cols or name in fact.nullable or fact.fk_nulls.get(name, 0):
        return ()
    codec = fact.codecs.get(name)
    if codec is None or codec.kind not in ("int", "bool", "date", "timestamp", "decimal", "dict", "rank"):
        return ()
    if codec.kind == "dict" and col.t.id != "VARCHAR":
        return ()
    if codec.kind == "rank" and not col.t.integer:
        return ()
    wide = fact.wide.get(name)
    return (wide[1], name) if wide else (name,)


def bounded_ids(planner, preds):
    """Return ordered, post-OFFSET row IDs and qualifying row count, or None."""
    s, cx = planner.s, planner.cx
    fact = planner.tab[planner.fact]
    if (_topk is None or cx.engine != "alg" or s.from_ is not planner.fact
            or planner.left or planner.fact_sample is not None or s.sample is not None
            or s.aggregate or s.distinct or s.distinct_on is not None or s.qualify is not None
            or s.percent or s.with_ties or not s.order
            or not (1 << 17) <= fact.n < 1 << 32
            or any(not getattr(planner.tab[b], "generated", None) for b in planner.joins)):
        return None
    lim = s.limit.v if isinstance(s.limit, I.Lit) else s.limit
    off = s.offset.v if isinstance(s.offset, I.Lit) else s.offset
    off = 0 if off is None else off
    if type(lim) is not int or type(off) is not int or lim <= 0 or off < 0 or lim + off > _MAX_ROWS:
        return None
    if any(not isinstance(expr, (I.Col, I.Lit)) for expr, _ in s.items):
        return None

    keys, desc = [], []
    for spec, descending, _ in s.order:
        expr = s.items[spec[1]][0] if spec[0] == "pos" else spec[1]
        physical = _ordered_columns(planner, fact, expr)
        if not physical:
            return None
        keys.extend(physical)
        desc.extend([bool(descending)] * len(physical))
    if len(keys) > _MAX_KEYS:
        return None
    flags = [flag for key in keys if (flag := fact._part_of(key)._vflag_of(key)) is not None]
    source = fact if fact.parts is None else fact.view(tuple(dict.fromkeys(keys + flags)))
    bitmap = fact.used_bm
    if preds or planner.joins:
        query = algq.Query(fact, join_mode="auto")
        for b, fk in planner.joins.items():
            query.join(planner.tab[b], fk)
        for pred in preds:
            query.where(pred)
        query._rows = query._rowbits = True
        try:
            bitmap = query._run_codes("alg") & bitmap
        except (TypeError, ValueError, OverflowError):
            return None
        cx.stats["engine_queries"] += 1

    count, inputs = ctypes.c_uint32(), ctypes.c_uint64()
    out = np.empty(lim + off, np.uint64)
    cols = (ctypes.c_int32 * len(keys))(*(source.idx(key) for key in keys))
    directions = (ctypes.c_int32 * len(keys))(*desc)
    rc = _topk(source.at, bitmap.ctypes.data, len(keys), cols, directions,
               len(out), out.ctypes.data, count, inputs)
    if rc:
        return None
    cx.stats["engine_queries"] += 1
    return out[off:count.value], inputs.value
