# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Reduce regex text groups from native per-code multiplicities."""

import ctypes as C
from time import perf_counter
from weakref import finalize

import numpy as np

from . import algq, coltypes, sqlir as I, sqltypes as T
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


_REFERRER_PATTERN = r"^https?://(?:www\.)?([^/]+)/.*$"
_REFERRER_REPLACEMENT = r"\1"


class _DeviceReferrer:
    def __init__(self, handle, release, run, wall_ms, kernel_ms):
        self.handle, self.run = handle, run
        self.wall_ms, self.kernel_ms = wall_ms, kernel_ms
        self._release = finalize(self, release, handle)
        self._release.atexit = False


def _referrer_projection(table, name, codec):
    dictionary = codec.values
    if not isinstance(dictionary, coltypes.PackedStrings) or not 0 < len(dictionary) <= _REGEX_DOMAIN_LIMIT:
        return None, False
    key = name, codec
    try:
        library = algq.gpu_lib()
        prepare = getattr(library, "gpu_referrer_prepare", None)
        release = getattr(library, "gpu_referrer_free", None)
        run = getattr(library, "gpu_referrer_moments", None)
        if prepare is None or release is None or run is None:
            return None, False
        vp, u32, elapsed = algq.vp, C.c_uint32, C.POINTER(C.c_double)
        prepare.restype = vp
        prepare.argtypes = [vp, C.c_uint64, vp, u32, elapsed]
        release.restype, release.argtypes = None, [vp]
        run.restype = C.c_int
        run.argtypes = [vp, vp, vp, C.c_int32, C.c_int32, u32, u32, *([vp] * 7),
                        C.POINTER(u32), elapsed]
        with algq._GPU_LOCK:
            cache = table.__dict__.setdefault("_gpu_referrer_projections", {})
            state = cache.get(key)
            if state is not None:
                return state, False
            kernel = C.c_double()
            start = perf_counter()
            handle = prepare(dictionary.blob.ctypes.data, dictionary.blob.nbytes,
                             dictionary.offsets.ctypes.data, len(dictionary), C.byref(kernel))
            if not handle:
                return None, False
            state = _DeviceReferrer(handle, release, run, (perf_counter() - start) * 1000, kernel.value)
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            cache[key] = state
            return state, True
    except (OSError, RuntimeError):
        return None, False


def prepare_referrer(table, column):
    """Derive the registered referrer projection on CUDA for a packed dictionary.

    Call once after opening a packed-text archive to stage the projection before
    queries. Normal execution also prepares the same projection on first use.
    """
    if table.closed:
        raise ValueError("cannot prepare a projection on a closed table")
    codec = table.codecs.get(column)
    if codec is None or codec.kind != "dict" or codec.sql_type != "VARCHAR":
        return None
    from .referrer_index import index_for

    saved = index_for(table, column)
    if saved is not None:
        state, built = _counted_cuda(saved)
        if state is not None:
            return dict(codes=len(codec.values), bytes=sum(a.nbytes for a in saved.arrays),
                        cached=not built, prepare_ms=state.wall_ms if built else 0.,
                        kernel_ms=0., persisted=True)
        return None
    state, built = _referrer_projection(table, column, codec)
    if state is None:
        return None
    return dict(codes=len(codec.values), bytes=codec.values.blob.nbytes,
                cached=not built, prepare_ms=state.wall_ms if built else 0.,
                kernel_ms=state.kernel_ms if built else 0.)


def _registered_referrer(group, col):
    if (not isinstance(group, I.Func) or group.fn != "regexp_replace"
            or not group.args or group.args[0].key() != col.key() or T.coll_key(group.t) is not None):
        return False
    args = group.args[1:]
    return (all(isinstance(arg, I.Lit) for arg in args)
            and tuple(arg.v for arg in args) in ((_REFERRER_PATTERN, _REFERRER_REPLACEMENT),
                                                (_REFERRER_PATTERN, _REFERRER_REPLACEMENT, "")))


def _minimum_count(having):
    if having is not None:
        compare, boundary = having
        if compare in (np.greater, np.greater_equal, np.equal):
            return max(1, boundary + (compare is np.greater))
    return 1


def _counted_cuda(state):
    with algq._GPU_LOCK:
        if state.cuda is not None:
            return state.cuda, False
        library = algq.gpu_lib()
        prepare = getattr(library, "gpu_referrer_counted_prepare", None)
        release = getattr(library, "gpu_referrer_counted_free", None)
        run = getattr(library, "gpu_referrer_counted_moments", None)
        if prepare is None or release is None or run is None:
            return None, False
        vp, u32 = algq.vp, C.c_uint32
        prepare.restype, prepare.argtypes = vp, [u32, vp, vp, vp, vp]
        release.restype, release.argtypes = None, [vp]
        run.restype = C.c_int
        run.argtypes = [vp, C.c_int32, u32, u32, *([vp] * 7), C.POINTER(u32), C.POINTER(C.c_double)]
        start = perf_counter()
        handle = prepare(len(state.arrays[0]), *(a.ctypes.data for a in state.arrays))
        if not handle:
            return None, False
        state.cuda = _DeviceReferrer(handle, release, run, (perf_counter() - start) * 1000, 0.)
        return state.cuda, True


def _persisted_referrer(planner, fact, group, col, source, aggs, length, having):
    from .referrer_index import index_for

    if planner.joins or length.fn not in text._BYTE_LENGTH or not _registered_referrer(group, col):
        return None
    term = planner.s.where
    if term is not None:
        if not isinstance(term, I.Func) or term.fn not in ("!=", "<>") or len(term.args) != 2:
            return None
        left, right = term.args
        if isinstance(left, I.Lit):
            left, right = right, left
        if left.key() != col.key() or not isinstance(right, I.Lit) or right.v != "":
            return None

    name, codec = source
    state = index_for(fact, name)
    if state is None:
        return None
    minimum = _minimum_count(having)
    cap = min(len(codec.values), fact.n // minimum)
    if cap > _REGEX_OUTPUT_LIMIT:
        return None
    if not cap:
        planner.cx.stats["referrer_fact_rows_scanned"] = 0
        return _result(group, aggs, [], np.empty(0, np.int64), [], [])
    reps, counts, minima, starts, lengths = (np.empty(cap, np.uint32) for _ in range(5))
    sums, suffix = np.empty(cap, np.uint64), np.empty(cap, np.uint8)
    size = C.c_uint32()
    output = (reps, counts, sums, minima, starts, lengths, suffix)
    args = (int(term is not None), minimum, cap, *(a.ctypes.data for a in output), C.byref(size))
    stats = planner.cx.stats
    try:
        if planner.cx.engine == "gpu":
            device, built = _counted_cuda(state)
            if device is None:
                return None
            kernel = C.c_double()
            with algq._GPU_LOCK:
                rc = device.run(device.handle, *args, C.byref(kernel))
            if not rc:
                stats["gpu_referrer_counted_moments"] = 1
                stats["gpu_text_kernel_ms"] = stats.get("gpu_text_kernel_ms", 0.) + kernel.value
                if built:
                    stats["gpu_referrer_counted_prepare_ms"] = device.wall_ms
        else:
            native = getattr(algq.lib, "at_referrer_counted_moments", None)
            if native is None:
                return None
            vp, u32 = algq.vp, C.c_uint32
            native.restype = C.c_int
            native.argtypes = [u32, *([vp] * 4), C.c_int32, u32, u32, *([vp] * 7), C.POINTER(u32)]
            rc = native(len(state.arrays[0]), *(a.ctypes.data for a in state.arrays), *args)
            if not rc:
                stats["cpu_referrer_counted_moments"] = 1
    except (OSError, RuntimeError):
        return None
    if rc or size.value > cap:
        return None
    indexes = np.arange(size.value)
    if having is not None:
        compare, boundary = having
        indexes = indexes[compare(counts[:size.value], boundary)]
    labels = [codec.values.span(int(reps[i]), int(starts[i]), int(lengths[i]), bool(suffix[i])) for i in indexes]
    averages = [int(sums[i]) / int(counts[i]) for i in indexes]
    stats["engine_queries"] += 1
    stats["referrer_index_codes"] = len(codec.values)
    stats["referrer_fact_rows_scanned"] = 0
    return _result(group, aggs, labels, counts[indexes].astype(np.int64), averages, codec.decode(minima[indexes]))



def _gpu_referrer(cx, fact, scope, group, col, source, table, aggs, byte_length, having, minimum):
    if not _registered_referrer(group, col):
        return None
    name, codec = source
    cap = min(len(codec.values), fact.n // minimum)
    if cap > _REGEX_OUTPUT_LIMIT or not cap:
        return None
    state, built = _referrer_projection(table, name, codec)
    if state is None:
        return None
    reps, counts, minima, starts, lengths = (np.empty(cap, np.uint32) for _ in range(5))
    sums = np.empty(cap, np.uint64)
    suffix = np.empty(cap, np.uint8)
    size, kernel = C.c_uint32(), C.c_double()
    try:
        part = fact._part_of(name)
        handle = part.gpu_handle()
        with algq._GPU_LOCK:
            rc = state.run(
                state.handle, handle, scope.ctypes.data, part.idx(name), byte_length, minimum, cap,
                reps.ctypes.data, counts.ctypes.data, sums.ctypes.data, minima.ctypes.data,
                starts.ctypes.data, lengths.ctypes.data, suffix.ctypes.data, C.byref(size), C.byref(kernel),
            )
    except (OSError, RuntimeError):
        return None
    if rc or size.value > cap:
        return None
    indexes = np.arange(size.value)
    if having is not None:
        compare, boundary = having
        indexes = indexes[compare(counts[:size.value], boundary)]
    try:
        labels = [codec.values.span(int(reps[i]), int(starts[i]), int(lengths[i]), bool(suffix[i]))
                  for i in indexes]
        values = codec.decode(minima[indexes])
    except (ValueError, UnicodeError):
        return None
    averages = [int(sums[i]) / int(counts[i]) for i in indexes]
    cx.stats["engine_queries"] += 1
    cx.stats["gpu_referrer_moments"] = cx.stats.get("gpu_referrer_moments", 0) + 1
    cx.stats["gpu_text_kernel_ms"] = cx.stats.get("gpu_text_kernel_ms", 0.) + kernel.value
    cx.stats["gpu_regex_input_codes"] = len(codec.values)
    cx.stats["gpu_regex_output_groups"] = len(indexes)
    if built:
        cx.stats["gpu_referrer_prepare_ms"] = state.wall_ms
        cx.stats["gpu_referrer_prepare_kernel_ms"] = state.kernel_ms
    return _result(group, aggs, labels, counts[indexes].astype(np.int64), averages, values)



def _gpu_regex(cx, fact, scope, group, col, source, table, aggs, length, having):
    name, codec = source
    domain = fact.maxcode(name) + 1
    if not 0 < domain <= min(_REGEX_DOMAIN_LIMIT, len(codec.values)):
        return None
    minimum = _minimum_count(having)
    if minimum > fact.n:
        return _result(group, aggs, [], np.empty(0, np.int64), [], [])
    byte_length = length.fn in text._BYTE_LENGTH
    projected = _gpu_referrer(cx, fact, scope, group, col, source, table, aggs, byte_length, having, minimum)
    if projected is not None:
        return projected
    mapping = text._regex_group(group, col, table, source)
    if mapping is None:
        return None
    group_codes, values = mapping
    cap = min(len(values), fact.n // minimum)
    if cap > _REGEX_OUTPUT_LIMIT:
        return None

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
    if fk is None:
        persisted = _persisted_referrer(planner, fact, group, col, source, aggs, length, having)
        if persisted is not None:
            return persisted
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

