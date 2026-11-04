# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""SQL planning for native reductions over certified periodic planes."""

import ctypes as C
from functools import cache
from math import lcm
import os
from time import perf_counter_ns
import weakref

import numpy as np
import sqlir as I
import sqltypes as T

from . import algq
from .sqlsnapshot import _revision

_MAX_WORDS = 128
_MAX_CYCLE_WORDS = 4096
_MIN_REPEATS = 8
_SECTOR_WORDS = 4096
_MIN_AUTO_ROWS = 1 << 16
_AUTO_CACHE_SIZE = 16
_GPU_SECTOR_BYTES = 32 << 20
_ENABLED = os.environ.get("TORBLES_PERIODIC", "auto") != "0"
_MAX_KEYS = 3
_MAX_MEASURES = 2
_MAX_GROUPS = 4096
_MAX_PROGRAM = 1024
_SUM, _MIN, _MAX = 1, 2, 4
_PREDICATES = {
    "and", "or", "not", "=", "<>", "<", "<=", ">", ">=", "between",
    "is_null", "is_not_null", "is_distinct", "is_not_distinct",
}
_NON_COLUMN_OPS = {algq.P[name] for name in ("TRUE", "AND", "OR", "NOT", "XOR")}


class _Key(C.Structure):
    _fields_ = [("col", C.c_int32), ("valid", C.c_int32), ("radix", C.c_uint32)]


class _Measure(C.Structure):
    _fields_ = [("col", C.c_int32), ("valid", C.c_int32), ("flags", C.c_uint32)]


class _Plan(C.Structure):
    _fields_ = [("nkeys", C.c_uint32), ("nmeasures", C.c_uint32), ("groups", C.c_uint32),
                ("keys", _Key * _MAX_KEYS), ("measures", _Measure * _MAX_MEASURES)]


class _State(C.Structure):
    _fields_ = [("count", C.c_uint64), ("valid", C.c_uint64 * _MAX_MEASURES),
                ("sums", C.c_uint64 * _MAX_MEASURES), ("mins", C.c_uint32 * _MAX_MEASURES),
                ("maxs", C.c_uint32 * _MAX_MEASURES)]


class _Sector(C.Structure):
    _fields_ = [("first_word", C.c_uint32), ("full_words", C.c_uint32),
                ("cycle_words", C.c_uint32), ("tail_rows", C.c_uint32)]


class _Metrics(C.Structure):
    _fields_ = [("periodic_sectors", C.c_uint64), ("ordinary_sectors", C.c_uint64),
                ("representative_words", C.c_uint64)]


def _bind(lib, name, restype, argtypes):
    fn = getattr(lib, name, None)
    if fn is not None:
        fn.restype, fn.argtypes = restype, argtypes
    return fn


_prog_args = [C.POINTER(algq.PInstr), C.c_int32, C.POINTER(_Plan), C.POINTER(_State)]
_cpu = _bind(algq.lib, "at_periodic_reduce", C.c_int, [algq.vp, C.c_uint32, *_prog_args])
_certify = _bind(algq.lib, "at_periodic_periods", C.c_int,
                 [algq.vp, C.c_uint32, C.POINTER(C.c_uint16), C.c_uint32])
_sector_args = [C.POINTER(_Sector), C.c_uint32]
_discover = _bind(algq.lib, "at_periodic_plan_sectors", C.c_int,
                  [algq.vp, C.c_uint32, C.c_uint32, C.c_uint32, *_sector_args, C.POINTER(_Metrics)])
_cpu_sectors = _bind(algq.lib, "at_periodic_reduce_sectors", C.c_int,
                     [algq.vp, *_sector_args, *_prog_args, C.POINTER(_Metrics)])


def _close_cycle(close, handle, lock=algq._GPU_LOCK):
    with lock:
        close(handle)


class _DeviceCycle:
    def __init__(self, handle, size, close):
        self.handle, self.size = handle, size
        self._release = weakref.finalize(self, _close_cycle, close, handle)


def _anchor(fact):
    at = fact.at
    return fact._native, at, _revision(at), fact.n, tuple(fact.cols)


def _anchors(parts):
    return tuple(_anchor(part) for part in parts)


class _Candidate:
    def __init__(self, anchor, storage, cycle=0, sectors=None, metrics=None, resident=None):
        self.anchor, self.storage = anchor, storage
        self.cycle, self.sectors, self.metrics = cycle, sectors, metrics
        self.resident = {} if resident is None else resident


def prepare(fact, max_words=_MAX_WORDS):
    """Certify the shortest packed-word periods of the fact's physical columns."""
    if type(max_words) is not int or not 1 <= max_words <= _MAX_WORDS:
        raise ValueError(f"max_words must be an integer from 1 through {_MAX_WORDS}")
    if fact.parts is not None or getattr(fact, "_vflag", None):
        return {}
    if _certify is None:
        return {}
    anchor = _anchor(fact)
    cached = getattr(fact, "_periodic", None)
    if cached is not None and cached[:2] == (anchor, max_words):
        return dict(cached[2])

    columns = tuple(fact.cols)
    periods = (C.c_uint16 * len(columns))()
    if _certify(anchor[1], max_words, periods, len(columns)) or _anchor(fact) != anchor:
        return {}
    result = {name: period for name, period in zip(columns, periods) if period}
    fact._periodic = anchor, max_words, result, {}
    return dict(result)


def _full_scope(fact, full, tail):
    used = fact.used_bm
    all_bits = np.uint64((1 << 64) - 1)
    return (used.ndim == 1 and used.dtype == np.uint64 and len(used) >= full + bool(tail)
            and np.bitwise_and.reduce(used[:full], initial=all_bits) == all_bits
            and (not tail or int(used[full]) == (1 << tail) - 1)
            and not used[full + bool(tail):].any())


def _safe_predicate(expr, strip):
    expr = strip(expr)
    if isinstance(expr, I.Lit):
        return expr.v is None or type(expr.v) in (bool, int)
    if isinstance(expr, I.Col):
        return expr.t.integer
    return (isinstance(expr, I.Func) and expr.fn in _PREDICATES
            and all(_safe_predicate(arg, strip) for arg in expr.args))


def _automatic(fact, source, mask, parts, stats):
    if _discover is None or fact.n < _MIN_AUTO_ROWS:
        return None
    key = mask if source is fact else tuple(source.cols)
    resident = fact.__dict__.setdefault("_periodic_auto", {})
    anchor = _anchors(parts)
    entry = resident.get(key)
    if entry is not None and entry.anchor == anchor:
        stats["periodic_proof_cache_hits"] = stats.get("periodic_proof_cache_hits", 0) + 1
        return entry

    storage = algq._run_on(source)
    started = perf_counter_ns()
    metrics = _Metrics()
    whole = (_Sector * 1)()
    rc = _discover(storage.at, mask, 0, _MAX_WORDS, whole, 1, metrics)
    words = (fact.n + 63) // 64
    count = (words + _SECTOR_WORDS - 1) // _SECTOR_WORDS
    stats["periodic_probes"] = stats.get("periodic_probes", 0) + 1
    entry = _Candidate(anchor, storage)
    if not rc and whole[0].cycle_words:
        entry.cycle = whole[0].cycle_words
    elif not rc and count > 1:
        sectors = (_Sector * count)()
        rc = _discover(storage.at, mask, _SECTOR_WORDS, _MAX_WORDS, sectors, count, metrics)
        stats["periodic_probes"] += 1
        if not rc and metrics.periodic_sectors and metrics.representative_words * 4 <= words:
            entry.sectors, entry.metrics = sectors, metrics
    stats["periodic_probe_ms"] = stats.get("periodic_probe_ms", 0.0) + (perf_counter_ns() - started) / 1e6
    if rc or _anchors(parts) != anchor:
        return None
    if key not in resident and len(resident) >= _AUTO_CACHE_SIZE:
        resident.pop(next(iter(resident), None), None)
    resident[key] = entry
    return entry


@cache
def _cuda_functions(lib):
    open_cycle = _bind(lib, "gpu_periodic_open", algq.vp,
                       [algq.vp, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint64)])
    reduce = _bind(lib, "gpu_periodic_reduce", C.c_int,
                   [algq.vp, *_prog_args, C.POINTER(C.c_double)])
    close = _bind(lib, "gpu_periodic_close", None, [algq.vp])
    open_sectors = _bind(lib, "gpu_periodic_open_sectors", algq.vp,
                         [algq.vp, *_sector_args, C.c_uint32, C.POINTER(C.c_uint64)])
    return open_cycle, reduce, close, open_sectors


def _cuda(candidate, mask, program, plan, states, stats):
    try:
        open_cycle, reduce, close, open_sectors = _cuda_functions(algq.gpu_lib())
    except (RuntimeError, OSError):
        return False
    sectors, cycle, resident = candidate.sectors, candidate.cycle, candidate.resident
    open_handle = open_cycle if sectors is None else open_sectors
    if open_handle is None or reduce is None or close is None:
        return False
    tag = cycle, mask
    uploaded = 0
    retired = None
    elapsed = C.c_double()
    with algq._GPU_LOCK:
        entry = resident.get(tag)
        if entry is None:
            if sectors is not None:
                # Packed planes and one device weight per representative word.
                width = sum(candidate.storage.width(name)
                            for i, name in enumerate(candidate.storage.cols) if mask >> i & 1)
                if candidate.metrics.representative_words * (8 * width + 4) > _GPU_SECTOR_BYTES:
                    stats["periodic_gpu_budget_fallback"] = True
                    return False
            size = C.c_uint64()
            geometry = (cycle,) if sectors is None else (sectors, len(sectors))
            handle = open_handle(candidate.storage.at, *geometry, mask, size)
            if not handle:
                return False
            entry = _DeviceCycle(handle, size.value, close)
            uploaded = entry.size
            if len(resident) >= 8:
                retired = resident.pop(next(iter(resident)))
            resident[tag] = entry
        rc = reduce(entry.handle, program, len(program), plan, states, elapsed)
    del retired
    if rc:
        return False
    stats["periodic_uploaded_bytes"] = stats.get("periodic_uploaded_bytes", 0) + uploaded
    stats["periodic_gpu_plane_bytes"] = entry.size
    stats["periodic_kernel_ms"] = stats.get("periodic_kernel_ms", 0.0) + elapsed.value
    stats["periodic_native_device"] = "cuda"
    return True


def _format_result(E, groups, aggregates, states):
    present = [i for i, state in enumerate(states) if state.count] if groups else [0]
    selected = [states[i] for i in present]
    rel = E.Rel(len(present))
    rel.grouped = True
    for expr, radix, valid, offset in reversed(groups):
        codes = [index % radix for index in present]
        rel.cols[expr.key()] = [None if valid >= 0 and code == radix - 1 else code + offset
                                for code in codes]
        present = [index // radix for index in present]
    for key, kind, index, offset in aggregates:
        def value(state):
            if kind == "count_star":
                return state.count
            count = state.valid[index]
            if kind == "count":
                return count
            if not count:
                return None
            if kind == "min":
                return state.mins[index] + offset
            if kind == "max":
                return state.maxs[index] + offset
            total = state.sums[index] + offset * count
            return total if kind == "sum" else total / count
        rel.cols[key] = [value(state) for state in selected]
    return rel


def run(planner, rest, aggs, extra):
    """Lower supported aggregates and return their native group states."""
    from . import sqlexec as E

    s, cx, binding = planner.s, planner.cx, planner.fact
    fact = planner.tab[binding]
    if (not _ENABLED or cx.engine not in ("alg", "gpu") or s.from_ is not binding or not s.aggregate
            or planner.joins or planner.left or planner.nulled or extra
            or planner.fact_sample is not None or s.sample is not None
            or getattr(binding, "writable", None) is not None
            or not 0 < fact.n < 1 << 32 or len(s.group) > _MAX_KEYS or not aggs
            or s.sets is not None or s.qualify is not None or s.distinct or s.distinct_on is not None):
        return None

    top = [expr for expr, _ in s.items] + [s.having]
    top += [spec[1] for spec, _, _ in s.order if spec[0] == "expr"]
    if any(expr is not None and I.has(expr, (I.Win, I.Subq, I.Outer)) for expr in top):
        return None
    source = algq._plan_table(fact)
    validity = getattr(fact, "nullable", {})
    refs = getattr(fact, "_refdim", {})
    moved = getattr(fact, "moved", {})
    dependencies = set()

    def eligible(col):
        return (isinstance(col, I.Col) and col.b is binding and col.t.integer and col.name in fact.cols
                and col.name not in refs and col.name not in moved and col.name not in fact.wide
                and fact._vflag_of(col.name) is None
                and getattr(fact.codecs.get(col.name), "kind", None) == "int")

    def valid_index(name):
        physical = validity.get(name)
        if physical is None:
            return -1
        dependencies.add(physical)
        return source.idx(physical)

    groups, aggregates, measures = [], [], {}
    plan = _Plan()
    plan.groups = 1
    for i, expr in enumerate(s.group):
        if not eligible(expr):
            return None
        name = expr.name
        dependencies.add(name)
        valid = valid_index(name)
        radix = int(fact.maxcode(name)) + 1 + (valid >= 0)
        if radix * plan.groups > _MAX_GROUPS:
            return None
        plan.groups *= radix
        plan.keys[i] = _Key(source.idx(name), valid, radix)
        groups.append((expr, radix, valid, int(fact.codecs[name].offset)))
    plan.nkeys = len(groups)
    flags = {"count": 0, "sum": _SUM, "avg": _SUM, "min": _MIN, "max": _MAX}
    for key, agg in aggs.items():
        kind = agg.name
        if agg.filter is not None or agg.distinct or agg.order:
            return None
        if kind == "count_star":
            aggregates.append((key, kind, 0, 0))
            continue
        if kind not in flags or len(agg.args) != 1 or not agg.args[0].t.integer:
            return None
        col = E._strip_cast(agg.args[0])
        if not eligible(col):
            return None
        name, flag = col.name, flags[kind]
        offset = int(fact.codecs[name].offset)
        if kind == "sum":
            low, high = T.INT_RANGE[agg.t.id]
            if min(0, offset) * fact.n < low or max(0, offset + int(fact.maxcode(name))) * fact.n > high:
                return None
        if name not in measures:
            if len(measures) == _MAX_MEASURES:
                return None
            index = measures[name] = len(measures)
            plan.measures[index] = _Measure(-1, valid_index(name), 0)
        index = measures[name]
        if flag:
            dependencies.add(name)
            plan.measures[index].col = source.idx(name)
            plan.measures[index].flags |= flag
        aggregates.append((key, kind, index, offset))
    plan.nmeasures = len(measures)

    program = []
    for term in rest:
        term = E._fold_consts(term)
        if isinstance(term, I.Lit) and term.v is True:
            continue
        if not _safe_predicate(term, E._strip_cast) or not all(eligible(c) for c in I.cols_of(term)):
            return None
        predicate = planner.native(term)
        if predicate is None:
            return None
        prior = len(program)
        try:
            algq.compile_pred(predicate, source, program)
        except (TypeError, algq._TooWide):
            return None
        if prior:
            program.append(algq.PInstr(algq.P["AND"], 0, 0, 0))
        if len(program) > _MAX_PROGRAM:
            return None
    dependencies.update(source.cols[op.col] for op in program if op.op not in _NON_COLUMN_OPS)
    if not dependencies:
        return None
    mask = sum(1 << source.idx(name) for name in dependencies)
    parts = (fact,) if fact.parts is None else tuple(dict.fromkeys(fact._part_of(name) for name in source.cols))
    full, tail = divmod(fact.n, 64)
    candidate = None
    cached = getattr(fact, "_periodic", None)
    if cached is not None and fact.parts is None:
        anchor, _, periods, resident = cached
        if dependencies <= periods.keys() and _anchor(fact) == anchor:
            cycle = lcm(*(periods[name] for name in dependencies))
            if cycle <= _MAX_CYCLE_WORDS and full // cycle >= _MIN_REPEATS:
                candidate = _Candidate((anchor,), fact, cycle, resident=resident)
    if candidate is None:
        candidate = _automatic(fact, source, mask, parts, cx.stats)
    if candidate is None or (not candidate.cycle and candidate.sectors is None):
        return None
    if not _full_scope(fact, full, tail):
        return None
    instructions = (algq.PInstr * len(program))(*program)
    states = (_State * plan.groups)()
    cycle, sectors, metrics = candidate.cycle, candidate.sectors, candidate.metrics
    if cx.engine == "gpu":
        if not _cuda(candidate, mask, instructions, plan, states, cx.stats):
            return None
    else:
        args = instructions, len(instructions), plan, states
        if sectors is None:
            rc = -1 if _cpu is None else _cpu(candidate.storage.at, cycle, *args)
        else:
            rc = -1 if _cpu_sectors is None else _cpu_sectors(
                candidate.storage.at, sectors, len(sectors), *args, None)
        if rc:
            return None
        cx.stats["periodic_native_device"] = "cpu"
    if _anchors(parts) != candidate.anchor:
        return None
    cx.stats["engine_queries"] += 1
    cx.stats["periodic_rows"] = cx.stats.get("periodic_rows", 0) + fact.n
    representatives = cycle if metrics is None else metrics.representative_words
    cx.stats["periodic_representatives"] = cx.stats.get("periodic_representatives", 0) + representatives * 64
    cx.stats["periodic_certified_sectors"] = cx.stats.get("periodic_certified_sectors", 0) + (
        1 if metrics is None else metrics.periodic_sectors)
    cx.stats["periodic_ordinary_sectors"] = cx.stats.get("periodic_ordinary_sectors", 0) + (
        0 if metrics is None else metrics.ordinary_sectors)
    cx.stats["periodic_scope"] = "table" if sectors is None else "sectors"
    return _format_result(E, groups, aggregates, states)
