# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""SQL planning for native reductions over certified periodic planes."""

import ctypes as C
from functools import cache
from math import lcm
import weakref

import numpy as np
import sqlir as I
import sqltypes as T

from . import algq
from .sqlsnapshot import _revision

_MAX_WORDS = 128
_MAX_CYCLE_WORDS = 4096
_MIN_REPEATS = 8
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


def _bind(lib, name, restype, argtypes):
    fn = getattr(lib, name, None)
    if fn is not None:
        fn.restype, fn.argtypes = restype, argtypes
    return fn


_prog_args = [C.POINTER(algq.PInstr), C.c_int32, C.POINTER(_Plan), C.POINTER(_State)]
_cpu = _bind(algq.lib, "at_periodic_reduce", C.c_int, [algq.vp, C.c_uint32, *_prog_args])
_certify = _bind(algq.lib, "at_periodic_periods", C.c_int,
                 [algq.vp, C.c_uint32, C.POINTER(C.c_uint16), C.c_uint32])


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


@cache
def _cuda_functions(lib):
    open_cycle = _bind(lib, "gpu_periodic_open", algq.vp,
                       [algq.vp, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint64)])
    reduce = _bind(lib, "gpu_periodic_reduce", C.c_int,
                   [algq.vp, *_prog_args, C.POINTER(C.c_double)])
    close = _bind(lib, "gpu_periodic_close", None, [algq.vp])
    return open_cycle, reduce, close


def _cuda(fact, cycle, mask, program, plan, states, resident, stats):
    try:
        open_cycle, reduce, close = _cuda_functions(algq.gpu_lib())
    except (RuntimeError, OSError):
        return False
    if open_cycle is None or reduce is None or close is None:
        return False
    tag = cycle, mask
    uploaded = 0
    retired = None
    elapsed = C.c_double()
    with algq._GPU_LOCK:
        entry = resident.get(tag)
        if entry is None:
            size = C.c_uint64()
            handle = open_cycle(fact.at, cycle, mask, size)
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
    cached = getattr(fact, "_periodic", None)
    if (cached is None or cx.engine not in ("alg", "gpu") or s.from_ is not binding or not s.aggregate
            or planner.joins or planner.left or planner.nulled or extra
            or planner.fact_sample is not None or s.sample is not None
            or getattr(binding, "writable", None) is not None
            or fact.parts is not None or getattr(fact, "_vflag", None)
            or not 0 < fact.n < 1 << 32 or len(s.group) > _MAX_KEYS or not aggs
            or s.sets is not None or s.qualify is not None or s.distinct or s.distinct_on is not None):
        return None

    top = [expr for expr, _ in s.items] + [s.having]
    top += [spec[1] for spec, _, _ in s.order if spec[0] == "expr"]
    if any(expr is not None and I.has(expr, (I.Win, I.Subq, I.Outer)) for expr in top):
        return None
    anchor, _, periods, cache = cached
    if _anchor(fact) != anchor:
        return None
    validity = getattr(fact, "nullable", {})
    refs = getattr(fact, "_refdim", {})
    moved = getattr(fact, "moved", {})
    dependencies = set()

    def eligible(col):
        return (isinstance(col, I.Col) and col.b is binding and col.t.integer and col.name in fact.cols
                and col.name not in refs and col.name not in moved and col.name not in fact.wide
                and getattr(fact.codecs.get(col.name), "kind", None) == "int")

    def valid_index(name):
        physical = validity.get(name)
        if physical is None:
            return -1
        dependencies.add(physical)
        return fact.idx(physical)

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
        plan.keys[i] = _Key(fact.idx(name), valid, radix)
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
            plan.measures[index].col = fact.idx(name)
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
            algq.compile_pred(predicate, fact, program)
        except TypeError:
            return None
        if prior:
            program.append(algq.PInstr(algq.P["AND"], 0, 0, 0))
        if len(program) > _MAX_PROGRAM:
            return None
    dependencies.update(fact.cols[op.col] for op in program if op.op not in _NON_COLUMN_OPS)
    if not dependencies or not dependencies <= periods.keys():
        return None
    cycle = lcm(*(periods[name] for name in dependencies))
    if cycle > _MAX_CYCLE_WORDS:
        return None

    full, tail = divmod(fact.n, 64)
    if full // cycle < _MIN_REPEATS or not _full_scope(fact, full, tail):
        return None
    instructions = (algq.PInstr * len(program))(*program)
    states = (_State * plan.groups)()
    if cx.engine == "gpu":
        mask = sum(1 << fact.idx(name) for name in dependencies)
        if not _cuda(fact, cycle, mask, instructions, plan, states, cache, cx.stats):
            return None
    else:
        if _cpu is None or _cpu(fact.at, cycle, instructions, len(instructions), plan, states):
            return None
        cx.stats["periodic_native_device"] = "cpu"
    if _anchor(fact) != anchor:
        return None
    cx.stats["engine_queries"] += 1
    cx.stats["periodic_rows"] = cx.stats.get("periodic_rows", 0) + fact.n
    cx.stats["periodic_representatives"] = cx.stats.get("periodic_representatives", 0) + cycle * 64
    return _format_result(E, groups, aggregates, states)
