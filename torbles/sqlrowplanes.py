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
_TOPK_ARGS = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
    ]
if _topk is not None:
    _topk.restype = ctypes.c_int
    _topk.argtypes = _TOPK_ARGS


_payload_columns = getattr(algq.lib, "at_get_rows_columns", None)
if _payload_columns is not None:
    _payload_columns.restype = ctypes.c_int
    _payload_columns.argtypes = [
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
    ]


def bounded_payload(planner, atoms, ids):
    """Gather raw physical codes for a small, wide, join-free projection."""
    fact = planner.tab[planner.fact]
    if (_payload_columns is None or fact.parts is None or not 0 < len(ids) <= 64
            or planner.joins or planner.snowflake_keys or planner.fact_sample is not None):
        return None
    physical = {}
    moved = getattr(fact, "moved", {})
    for col in atoms.values():
        name = col.name
        if col.b is not planner.fact or name in fact.refbit or name in moved:
            return None
        physical[name] = None
        if wide := fact.wide.get(name):
            physical[wide[1]] = None
        if valid := fact.nullable.get(name):
            physical[valid] = None
    if len(physical) < 16 or not physical.keys() <= set(fact.cols):
        return None

    parts = [fact._part_of(name) for name in physical]
    # This batch ABI requires dense inputs. Let per-column sparse gathering
    # read requested words directly when an encoded part is still compact.
    if any(part._storage().needs_dense and not algq._at_has_dense(part._raw_at) for part in set(parts)):
        return None
    tables = (ctypes.c_void_p * len(parts))(*(part._raw_at for part in parts))
    columns = (ctypes.c_int32 * len(parts))(*(part.idx(name) for part, name in zip(parts, physical)))
    rows = np.ascontiguousarray(ids, dtype=np.uint64)
    codes = np.empty((len(parts), len(rows)), dtype=np.uint32)
    if _payload_columns(len(parts), tables, columns, rows.ctypes.data, len(rows), codes.ctypes.data):
        return None
    planner.cx.stats["bounded_payload_native_columns"] = len(parts)
    return dict(zip(physical, codes))


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


def _order_ids(fact, ids, keys, desc):
    order = [ids]
    for name, descending in reversed(list(zip(keys, desc))):
        codes = fact.get_rows(name, ids)
        order.append(~codes if descending else codes)
    return ids[np.lexsort(tuple(order))]


def bounded_ids(planner, preds):
    """Return ordered, post-OFFSET row IDs and qualifying row count, or None."""
    s, cx = planner.s, planner.cx
    fact = planner.tab[planner.fact]
    if (cx.engine not in ("alg", "gpu") or s.from_ is not planner.fact
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
    native = getattr(algq.gpu_lib(), "gpu_topk_bitmap", None) if cx.engine == "gpu" else _topk
    # The legacy top-k also runs in emulation, which lacks gpu_topk_bitmap.
    if cx.engine == "gpu":
        legacy = planner.topk_ids(preds)
        if legacy is not None:
            return _order_ids(fact, legacy, keys, desc)[off:], None
    from . import sqlgroupplans
    bitmap = sqlgroupplans.bitmap(planner, preds)
    if bitmap is None:
        return None
    sparse = algq._sparse_bitmap_ids(bitmap, fact.n) if preds else None
    if sparse is not None:
        cx.stats["raw_sparse_topk_matches"] = len(sparse)
        return _order_ids(fact, sparse, keys, desc)[off:off + lim], len(sparse)
    if native is None:
        return None
    flags = [flag for key in keys if (flag := fact._part_of(key)._vflag_of(key)) is not None]
    source = fact if fact.parts is None else fact.view(tuple(dict.fromkeys(keys + flags)))

    count, inputs = ctypes.c_uint32(), ctypes.c_uint64()
    out = np.empty(lim + off, np.uint64)
    cols = (ctypes.c_int32 * len(keys))(*(source.idx(key) for key in keys))
    directions = (ctypes.c_int32 * len(keys))(*desc)
    if cx.engine == "gpu":
        native.restype = ctypes.c_int
        native.argtypes = [*_TOPK_ARGS, ctypes.POINTER(ctypes.c_double)]
        ms = ctypes.c_double()
        with algq._GPU_LOCK:
            try:
                handle = source.gpu_handle()
            except RuntimeError:
                return None
            rc = native(handle, bitmap.ctypes.data, len(keys), cols, directions,
                        len(out), out.ctypes.data, count, inputs, ms)
        if not rc:
            out = _order_ids(fact, out[:count.value], keys, desc)
            cx.stats["cuda_raw_topk_rows"] = count.value
    else:
        rc = native(source.at, bitmap.ctypes.data, len(keys), cols, directions,
                    len(out), out.ctypes.data, count, inputs)
    if rc:
        return None
    cx.stats["engine_queries"] += 1
    return out[off:count.value], inputs.value
