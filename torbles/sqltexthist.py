# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Reduce regex text groups from native per-code multiplicities."""

import ctypes as C

import numpy as np

from . import algq
from . import sqltextplanes as text


_code_histogram = getattr(algq.lib, "at_plane_code_histogram", None)
_REGEX_DOMAIN_LIMIT = 1 << 27
_REGEX_OUTPUT_LIMIT = 1 << 20
if _code_histogram is not None:
    _code_histogram.restype = C.c_int
    _code_histogram.argtypes = [algq.vp, algq.vp, C.c_int32, C.c_uint32, algq.vp]


def _cuda_call(cx, table, name, args, types, stat):
    try:
        fn = getattr(algq.gpu_lib(), name, None)
        if fn is None:
            return False
        fn.restype = C.c_int
        fn.argtypes = [algq.vp, *types, C.POINTER(C.c_double)]
        elapsed = C.c_double()
        handle = table.gpu_handle()
        with algq._GPU_LOCK:
            rc = fn(handle, *args, C.byref(elapsed))
    except (OSError, RuntimeError):
        return False
    if rc:
        return False
    cx.stats[stat] = cx.stats.get(stat, 0) + 1
    cx.stats["gpu_text_kernel_ms"] = cx.stats.get("gpu_text_kernel_ms", 0.0) + elapsed.value
    return True


def code_counts(fact, column, scope, cx=None):
    """Return one exact count per physical code in a packed row scope."""
    source = fact._part_of(column)
    domain = fact.maxcode(column) + 1
    counts = np.empty(domain, dtype=np.uint32)
    args = (scope.ctypes.data, source.idx(column), domain, counts.ctypes.data)
    if cx is not None and cx.engine == "gpu":
        if not _cuda_call(cx, source, "gpu_plane_code_histogram", args,
                          (algq.vp, C.c_int32, C.c_uint32, algq.vp), "gpu_text_histograms"):
            return None
    elif _code_histogram is None or _code_histogram(source.at, *args):
        return None
    return counts


def lookup_sums(cx, table, scope, key, value, mask, lookup, codes, sums):
    args = (scope.ctypes.data, key, value, mask, lookup.ctypes.data, len(lookup),
            codes.ctypes.data, len(codes), sums.ctypes.data)
    types = (algq.vp, C.c_int32, C.c_int32, C.c_uint32, algq.vp, C.c_uint64,
             algq.vp, C.c_uint32, algq.vp)
    return _cuda_call(cx, table, "gpu_plane_lookup_sums", args, types, "gpu_text_lookup_sums")


def _result(group, aggs, labels, counts, averages, minima):
    from . import sqlexec as E

    out = E.Rel(len(counts))
    out.grouped = True
    out.cols[group.key()] = labels
    for key, agg in aggs.items():
        if agg.name == "count_star":
            out.cols.put(key, E.ColArr(counts))
        elif agg.name == "avg":
            out.cols.put(key, E.ColArr(np.asarray(averages, np.float64)))
        else:
            out.cols[key] = minima
    return out


def _gpu_regex(cx, fact, scope, group, col, source, table, aggs, length, having):
    name, codec = source
    domain = fact.maxcode(name) + 1
    if not 0 < domain <= min(_REGEX_DOMAIN_LIMIT, len(codec.values)):
        return None
    minimum = 1
    if having is not None:
        compare, boundary = having
        if compare in (np.greater, np.greater_equal, np.equal):
            minimum = max(1, boundary + (compare is np.greater))
    if minimum > fact.n:
        return _result(group, aggs, [], np.empty(0, np.int64), [], [])
    mapping = text._regex_group(group, col, table, source)
    if mapping is None:
        return None
    group_codes, values = mapping
    cap = min(len(values), fact.n // minimum)
    if cap > _REGEX_OUTPUT_LIMIT:
        return None

    byte_length = length.fn in text._BYTE_LENGTH

    def length_codes():
        lengths = text._length_map(table, source, byte_length)
        if lengths.max(initial=0) > np.iinfo(np.uint32).max:
            return None
        return lengths.astype(np.uint32)

    try:
        lengths = text._cache(table, "_gpu_regex_lengths", (name, codec, byte_length), length_codes, 2)
    except UnicodeError:
        return None
    if lengths is None:
        return None
    groups, counts, minima = (np.empty(cap, np.uint32) for _ in range(3))
    sums = np.empty(cap, np.uint64)
    size = C.c_uint32()
    part = fact._part_of(name)
    args = (scope.ctypes.data, part.idx(name), domain, group_codes.ctypes.data, lengths.ctypes.data,
            len(values), minimum, cap, groups.ctypes.data, counts.ctypes.data, sums.ctypes.data,
            minima.ctypes.data, C.byref(size))
    types = (algq.vp, C.c_int32, C.c_uint32, algq.vp, algq.vp, C.c_uint32, C.c_uint32, C.c_uint32,
             algq.vp, algq.vp, algq.vp, algq.vp, C.POINTER(C.c_uint32))
    if not _cuda_call(cx, part, "gpu_plane_regex_moments", args, types, "gpu_regex_moments"):
        return None
    groups, counts, sums, minima = (arr[:size.value] for arr in (groups, counts, sums, minima))
    if having is not None:
        keep = compare(counts, boundary)
        groups, counts, sums, minima = (arr[keep] for arr in (groups, counts, sums, minima))
    cx.stats["engine_queries"] += 1
    cx.stats["gpu_text_histograms"] = cx.stats.get("gpu_text_histograms", 0) + 1
    cx.stats["gpu_regex_input_codes"] = domain
    cx.stats["gpu_regex_output_groups"] = len(groups)
    averages = [total / count for total, count in zip(sums.tolist(), counts.tolist())]
    return _result(group, aggs, [values[i] for i in groups.tolist()], counts.astype(np.int64),
                   averages, codec.decode(minima))


def project_group(planner, preds, aggs, group, col, source, table, fk, length, having):
    """Reduce a validated regex projection using one native code histogram."""
    from . import sqlgroupplans as P

    cx, fact = planner.cx, planner.tab[planner.fact]
    name, codec = source
    column = name if fk is None else fk
    domain = len(codec.values) if fk is None else table.n
    # Bound scratch and avoid building one histogram cell per nearly unique row.
    # FK surrogates can also have gaps, so consider their physical domain separately.
    legacy = domain * 2 <= fact.n and fact.maxcode(column) + 1 <= min(1 << 23, fact.n)
    direct_gpu = cx.engine == "gpu" and fk is None
    if not legacy and not direct_gpu:
        return None
    scope = P.bitmap(planner, preds)
    if scope is None:
        return None
    if direct_gpu:
        result = _gpu_regex(cx, fact, scope, group, col, source, table, aggs, length, having)
        if result is not None:
            return result
    if not legacy:
        return None
    histogram = code_counts(fact, column, scope, cx)
    if histogram is None:
        return None
    code = np.flatnonzero(histogram)
    multiplicity = histogram[code]
    if fk is not None:
        mask = fact.kmask(fk)
        ids = code & mask if mask else code
        code = table.decoded(name)[ids]
    mapping = text._regex_group(group, col, table, source)
    if mapping is None:
        return None
    group_codes, values = mapping
    mapped = group_codes[code]
    n = int(multiplicity.sum())
    # The SQL planner bounds fact rows to uint32, so count weights are exactly representable.
    counts = np.bincount(mapped, weights=multiplicity, minlength=len(values)).astype(np.int64)
    eligible = counts != 0
    if having is not None:
        compare, boundary = having
        eligible &= compare(counts, boundary)
    groups = np.flatnonzero(eligible)
    count = counts[groups]
    averages, minima = [], []
    if len(groups):
        chosen = eligible[mapped]
        targets, chosen_codes = mapped[chosen], code[chosen]
        try:
            lens = text._length_map(table, source, length.fn in text._BYTE_LENGTH)
        except UnicodeError:
            return None
        if n * int(lens.max(initial=0)) >= 1 << 53:
            return None
        weights = lens[chosen_codes] * multiplicity[chosen].astype(np.uint64)
        totals = np.bincount(targets, weights=weights, minlength=len(values))[groups].astype(np.int64)
        averages = [v / c for v, c in zip(totals.tolist(), count.tolist())]
        if any(a.name == "min" for a in aggs.values()):
            minima_codes = np.full(len(values), np.iinfo(np.uint32).max, np.uint32)
            np.minimum.at(minima_codes, targets, chosen_codes.astype(np.uint32, copy=False))
            minima = codec.decode(minima_codes[groups])
    cx.stats["engine_queries"] += 1
    cx.stats["dictionary_projection_rows"] = cx.stats.get("dictionary_projection_rows", 0) + n
    cx.stats["dictionary_histogram_codes"] = cx.stats.get("dictionary_histogram_codes", 0) + len(code)
    if fk is not None:
        cx.stats["dictionary_dimension_projection_rows"] = (
            cx.stats.get("dictionary_dimension_projection_rows", 0) + n
        )
    return _result(group, aggs, [values[i] for i in groups.tolist()], count, averages, minima)

