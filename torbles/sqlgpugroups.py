# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Bounded CUDA reductions over the fact's resident encoded planes."""

import ctypes as C

import numpy as np

from torbles import algq

LIMIT = 1024


def _group(fact, keys, measures, limit, bitmap, stats, distinct=None, rank=0, projection=None, interchange=None, rank_switch=False):
    if not 1 <= limit <= LIMIT or not 1 <= len(keys) <= 5 or len(measures) > 2:
        return None
    if any(transform not in (0, 1, 3, 4) for _, _, _, transform, _ in keys):
        return None
    if rank not in (0, 1) or (distinct is not None and distinct[3] not in (0, 1)):
        return None

    library = algq.gpu_lib()
    extra = distinct is not None or projection is not None
    name = "gpu_group_distinct_planes" if extra else "gpu_group_planes"
    if (interchange is not None or rank_switch) and hasattr(library, "gpu_group_distinct_sector_planes"):
        name = "gpu_group_distinct_sector_planes"
    else:
        interchange = None
        rank_switch = False
    run = getattr(library, name, None)
    if run is None:
        return None
    run.restype = C.c_int
    run.argtypes = [
        algq.vp, algq.vp, C.c_int32, C.POINTER(algq._PlaneGroupKey),
        C.c_int32, algq.vp, C.c_uint32, algq.vp, algq.vp, algq.vp, C.POINTER(C.c_uint32),
    ]

    if extra:
        run.argtypes += [C.POINTER(algq._PlaneGroupKey), C.c_uint32, algq.vp, algq.vp, C.c_uint64,
                         C.c_uint64, C.c_uint32]
    if interchange is not None or rank_switch:
        run.argtypes += [algq.vp, C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint64)]
    plan = algq._plan_table(fact)

    def descriptor(spec):
        low, high, shift, transform, offset = spec
        return algq._PlaneGroupKey(plan.idx(low), -1 if high is None else plan.idx(high), shift, transform, offset)

    specs = (algq._PlaneGroupKey * len(keys))(*(descriptor(spec) for spec in keys))
    unique_spec = descriptor(distinct) if distinct is not None else None
    columns = (C.c_int32 * len(measures))(*(plan.idx(name) for name in measures))
    table = algq._run_on(plan)
    codes = np.empty((limit, len(keys)), np.uint64)
    counts = np.empty(limit, np.uint64)
    sums = np.empty((limit, len(measures)), np.uint64) if measures else None
    nout, selected = C.c_uint32(), C.c_uint64()
    uniques = np.empty(limit, np.uint64) if distinct is not None else None
    lookup = None
    mask = width = 0
    if projection is not None:
        lookup, mask, width = projection
        lookup = np.ascontiguousarray(lookup, np.uint32)
    with algq._GPU_LOCK:
        try:
            handle = table.gpu_handle()
        except RuntimeError:
            return None
        args = (handle, None if bitmap is None else bitmap.ctypes.data,
                len(keys), specs, len(measures), columns if measures else None, limit,
                codes.ctypes.data, counts.ctypes.data, None if sums is None else sums.ctypes.data, nout)
        if extra:
            args += (None if unique_spec is None else C.byref(unique_spec), rank,
                     None if uniques is None else uniques.ctypes.data,
                     None if lookup is None else lookup.ctypes.data, 0 if lookup is None else len(lookup), mask, width)
        if interchange is not None or rank_switch:
            args += ((None, 0, 0, 0) if interchange is None else
                     (interchange.sectors.ctypes.data, interchange.shift, interchange.bits,
                      interchange.sectors.shape[1]))
            args += (C.byref(selected) if rank_switch else None,)
        rc = run(*args)
    if rc:
        return None
    if interchange is not None:
        stats["cuda_interchange_proven_sectors"] = int(np.count_nonzero(interchange.sectors[:, 0]))
        stats["cuda_interchange_dimension_rows"] = interchange.covered
    if selected.value:
        stats["cuda_interchange_rank_rows"] = selected.value
    stats["cuda_encoded_group"] = stats.get("cuda_encoded_group", 0) + 1
    n = nout.value
    if distinct is not None:
        stats["cuda_encoded_group_distinct"] = stats.get("cuda_encoded_group_distinct", 0) + 1
    return codes[:n], counts[:n], None if sums is None else sums[:n], None if uniques is None else uniques[:n]


def group(fact, keys, measures, limit, bitmap, stats):
    result = _group(fact, keys, measures, limit, bitmap, stats)
    return None if result is None else result[:3]


def group_distinct(fact, keys, measures, limit, bitmap, stats, *, distinct, rank, projection=None, interchange=None, rank_switch=False):
    return _group(fact, keys, measures, limit, bitmap, stats, distinct, rank, projection, interchange, rank_switch)



def distinct(fact, spec, bitmap, stats):
    library = algq.gpu_lib()
    run = getattr(library, "gpu_distinct_plane_count", None)
    if run is None:
        return None
    run.restype = C.c_int
    run.argtypes = [algq.vp, algq.vp, C.POINTER(algq._PlaneGroupKey), C.POINTER(C.c_uint64)]

    low, high, shift, transform, offset = spec
    if transform not in (0, 1):
        return None
    plan = algq._plan_table(fact)
    key = algq._PlaneGroupKey(plan.idx(low), -1 if high is None else plan.idx(high), shift, transform, offset)
    table = algq._run_on(plan)
    count = C.c_uint64()
    with algq._GPU_LOCK:
        try:
            handle = table.gpu_handle()
        except RuntimeError:
            return None
        rc = run(handle, None if bitmap is None else bitmap.ctypes.data, C.byref(key), C.byref(count))
    if rc:
        return None
    stats["cuda_encoded_distinct"] = stats.get("cuda_encoded_distinct", 0) + 1
    return count.value

