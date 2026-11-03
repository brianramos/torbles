# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Native sparse grouping over physical Torbles equality codes."""

import ctypes as C

import numpy as np

from torbles import algq

GROUP_LIMIT = 1024
BY_COUNT, BY_DISTINCT, BY_KEY, BY_UNORDERED = range(4)
PREFIX_LIMIT = 16

_group = getattr(algq.lib, "at_group_encoded", None)
if _group is not None:
    _group.restype = C.c_int
    _group.argtypes = [
        algq.vp, algq.vp, C.c_int32, C.POINTER(algq._PlaneGroupKey),
        C.c_int32, algq.vp, C.c_uint32, algq.vp, algq.vp, algq.vp,
        C.POINTER(C.c_uint32), C.c_int32, C.c_uint32, algq.vp,
    ]

_projected = getattr(algq.lib, "at_group_encoded_projected", None)
if _projected is not None and _group is not None:
    _projected.restype = C.c_int
    _projected.argtypes = [*_group.argtypes, algq.vp, C.c_uint64, C.c_uint64, C.c_uint32]


def encoded_groups(fact, keys, measures, limit, bitmap, *, distinct=None, rank=BY_COUNT, projection=None):
    if _group is None or (projection is not None and _projected is None):
        return None
    source = algq._plan_table(fact)
    specs = [*keys, distinct] if distinct is not None else keys
    key_args = (algq._PlaneGroupKey * len(specs))(*(
        algq._PlaneGroupKey(source.idx(low), -1 if high is None else source.idx(high), shift, transform, offset)
        for low, high, shift, transform, offset in specs
    ))
    measure_args = (C.c_int32 * len(measures))(*(source.idx(name) for name in measures))
    codes = np.empty((limit, len(keys)), np.uint64)
    counts = np.empty(limit, np.uint64)
    sums = np.empty((limit, len(measures)), np.uint64) if measures else None
    uniques = np.empty(limit, np.uint64) if distinct is not None else None
    n = C.c_uint32()
    args = (
        algq._run_on(source).at, None if bitmap is None else bitmap.ctypes.data,
        len(keys), key_args, len(measures), measure_args, limit, codes.ctypes.data,
        counts.ctypes.data, None if sums is None else sums.ctypes.data, n,
        distinct is not None, rank, None if uniques is None else uniques.ctypes.data,
    )
    if projection is None:
        rc = _group(*args)
    else:
        lookup, mask, width = projection
        lookup = np.ascontiguousarray(lookup, np.uint32)
        rc = _projected(*args, lookup.ctypes.data, len(lookup), mask, width)
    if rc:
        return None
    n = n.value
    return codes[:n], counts[:n], None if sums is None else sums[:n], None if uniques is None else uniques[:n]

