# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""BOOL-002: influences and interactions of a predicate (a math_bool.Program over k conditions) under the uniform
measure of the 2^k assignments and under the empirical measure of the stored rows of a scope, as separate results,
natively on the CPU or with the CUDA kernels (torbles/math_influence.c, math_influence_gpu.cu).

    from torbles import math_bool as mb, math_influence as mi
    prog = mb.Program((x[0] & x[1]) | ~x[2], k=3)
    rows = mb.Masks.from_scope(scope, conditions)              # the stored rows (empirical measure)
    r = mi.influence(prog, rows=rows, order=2)                 # {"uniform": {...}, "empirical": {...}}
    r["uniform"]["influences"][i]["influence"]                 # P(d_i f = TRUE) over the 2^k assignments (Fraction)
    r["empirical"]["influences"][i]["influence"]               # the same over the selected rows
    r["empirical"]["total_influence"], ["sensitivity"], ["interactions"] (|S| = 2..order: mixed differences)
    mi.uniform(prog, order=2), mi.empirical(prog, rows, terms=[[0, 2]], backend="cuda")

What is measured is coordinate sensitivity of the predicate: how often substituting a condition (or a set of them),
with the other conditions held at the point's own values, changes the predicate's value. It is a property of the
predicate over a declared set of points, not a causal effect of the condition. The uniform measure is over every
assignment of the k conditions (the truth table); the empirical measure is over the stored rows only; the two are
returned separately and labelled, and a rows domain is never read as the assignments (or the reverse). SQL
three-valued logic (BOOL-001): a difference is UNKNOWN where a cofactor is, and such points are reported apart.
Counts are exact native integers, measures exact Fractions formed natively. Budgets raise AlgebraBudgetError before
anything is allocated, a stale scope StaleStateError. Nothing is computed in Python. Opt-in: `import torbles` does not
load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_bool as mb
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, check as _check

__all__ = ["influence", "uniform", "empirical", "influence_need", "interaction_sets", "INTERPRETATION", "MEASURES"]

FIRST = 1
_NEED_CPU, _NEED_DEVICE = 0, 1
_U64 = (1 << 64) - 1
P = C.POINTER
INTERPRETATION = ("coordinate sensitivity of the predicate: how often substituting the condition(s), with every other "
                  "condition held at the point's own value, changes the predicate's value; a property of the "
                  "predicate over the declared points, not a causal effect of the condition")
MEASURES = {"uniform": "every one of the 2^k assignments of the conditions, each once (the truth table)",
            "empirical": "the selected stored rows of the scope, each once (only the assignments the rows hold)"}


class _Term(C.Structure):
    _native_ = "tr_influence_term"
    _fields_ = [("vars", C.c_uint64), ("order", C.c_int32), ("measure", C.c_int32)] + [
        (n, C.c_uint64) for n in ("points", "known", "unknown", "sensitive", "positive", "negative")] + [
        ("sum_lo", C.c_int64), ("sum_hi", C.c_int64), ("measure_num", C.c_uint64), ("measure_den", C.c_uint64)]


class _Summary(C.Structure):
    _native_ = "tr_influence_summary"
    _fields_ = [(n, C.c_uint64) for n in ("points", "f_true", "f_false", "f_unknown", "all_known",
                                          "max_sensitivity")] + [
        ("total_lo", C.c_int64), ("total_hi", C.c_int64), ("total_num_lo", C.c_int64), ("total_num_hi", C.c_int64),
        ("total_den", C.c_uint64), ("k", C.c_int32), ("measure", C.c_int32), ("nterms", C.c_int32),
        ("flags", C.c_int32)]


for _s, _n in ((_Term, 96), (_Summary, 104)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64 = C.c_void_p, C.c_int32, C.c_uint64
        D, V = P(mb._Domain), P(mb._ProgramView)
        sig = {
            "tr_influence_need": (i32, [D, V, vp, i32, i32, i32, P(_Need)]),
            "tr_influence_compute": (i32, [D, V, vp, i32, i32, P(_Budget), P(_Summary), P(_Term), vp, P(_Term),
                                           P(_Refusal)]),
            "tr_influence_sets": (i32, [i32, i32, i32, P(_Budget), vp, u64, P(u64), P(_Refusal)]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, args
            native[name] = fn
        _native = native
    return _native


def _call(name, *args):
    return _api()[name](*args)


def _gpu():
    from . import math_gpu
    math_gpu.bind("tr_mgpu_influence_compute", C.c_int, [P(mb._Domain), P(mb._ProgramView), C.c_void_p, C.c_int32,
                                                        C.c_int32, P(_Budget), P(_Summary), P(_Term), C.c_void_p,
                                                        P(_Term), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _members(mask, k):
    return tuple(i for i in range(k) if (int(mask) >> i) & 1)


def _i128(lo, hi):
    return (int(hi) << 64) | (int(lo) & _U64)


def interaction_sets(k, max_order, min_order=2, *, budget=None):
    """the sets of min_order..max_order of k conditions, by size then increasing mask (natively enumerated)"""
    n, refusal = C.c_uint64(), _Refusal()
    _check(_call("tr_influence_sets", int(k), int(min_order), int(max_order), C.byref(_budget(budget)), None, 0,
                 C.byref(n), C.byref(refusal)), refusal, "interaction sets")
    out = np.zeros(max(int(n.value), 1), dtype=np.uint64)
    _check(_call("tr_influence_sets", int(k), int(min_order), int(max_order), C.byref(_budget(budget)),
                 out.ctypes.data, len(out), C.byref(n), C.byref(refusal)), refusal, "interaction sets")
    return out[:int(n.value)]


def _terms(k, order, terms, budget):
    """uint64 masks of the interaction terms: every set of 2..order conditions, then the given sets"""
    masks = list(interaction_sets(k, order, 2, budget=budget).tolist()) if order and order >= 2 else []
    for s in terms or ():
        s = [s] if isinstance(s, Integral) and not isinstance(s, bool) else list(s)
        idx = sorted(set(int(i) for i in s))
        if len(idx) != len(s):
            raise ValueError(f"term {s!r}: a condition appears twice")
        if any(not 0 <= i < k for i in idx):
            raise ValueError(f"term {s!r}: conditions are 0..{k - 1}")
        masks.append(sum(1 << i for i in idx))
    return np.array(masks or [0], dtype=np.uint64), len(masks)


def _term(t, k):
    known, sens = int(t.known), int(t.sensitive)
    return dict(conditions=_members(t.vars, k), order=int(t.order), sensitive=sens, insensitive=known - sens,
                unknown=int(t.unknown), known=known, increase=int(t.positive), decrease=int(t.negative),
                signed_sum=_i128(t.sum_lo, t.sum_hi), measure=Fraction(int(t.measure_num), int(t.measure_den)))


def _run(masks, program, measure, order, terms, first, backend, budget):
    masks._open(); program._open()
    masks.require_fresh("influence")
    if masks.k != program.k:
        raise DomainError(f"the program reads {program.k} conditions, the domain has {masks.k}")
    k = program.k
    arr, nterms = _terms(k, order, terms, budget)
    flags = FIRST if first else 0
    summary = _Summary()
    first_out = (_Term * max(k, 1))()
    sens = np.zeros(k + 1, dtype=np.uint64)
    term_out = (_Term * max(nterms, 1))()
    refusal = _Refusal()
    dom, view = masks.domain(), program.view()
    args = (C.byref(dom), C.byref(view), arr.ctypes.data, nterms, flags, C.byref(_budget(budget)), C.byref(summary),
            first_out, sens.ctypes.data, term_out, C.byref(refusal))
    if _backend(backend) == "cpu":
        _check(_call("tr_influence_compute", *args), refusal, f"{measure} influence")
        label = ("cpu", "native CPU")
    else:
        gpu = _gpu()
        gpu.run("tr_mgpu_influence_compute", *args, refusal=refusal, what=f"{measure} influence on the device")
        d = gpu.device()
        label = ("cuda-emulated" if d["emulated"] else "cuda", d["name"])
    expected = 2 if measure == "uniform" else 1                # the native measure is the domain's kind
    if int(summary.measure) != expected:
        raise AssertionError("native measure differs from the requested one")
    out = dict(measure=measure, description=MEASURES[measure], interpretation=INTERPRETATION, causal=False,
               points=int(summary.points),
               f=dict(true=int(summary.f_true), false=int(summary.f_false), unknown=int(summary.f_unknown)),
               interactions=[_term(term_out[j], k) for j in range(nterms)], backend=label[0], device=label[1])
    if first:
        infl = []
        for i in range(k):
            t = _term(first_out[i], k)
            t["condition"] = i
            t["influence"] = t.pop("measure")
            infl.append(t)
        out.update(influences=infl, total_sensitive=_i128(summary.total_lo, summary.total_hi),
                   total_influence=Fraction(_i128(summary.total_num_lo, summary.total_num_hi), int(summary.total_den)),
                   sensitivity={b: int(sens[b]) for b in range(k + 1)}, sensitivity_known=int(summary.all_known),
                   max_sensitivity=int(summary.max_sensitivity))
    return out


def uniform(program, *, order=None, terms=None, first=True, backend="cpu", budget=None):
    """influences under the uniform measure of the 2^k assignments (the truth table): see influence()"""
    if not isinstance(program, mb.Program):
        raise TypeError("program: a math_bool.Program")
    masks = mb.Masks.assignments(program.k, budget=budget)
    return _run(masks, program, "uniform", order, terms, first, backend, budget)


def empirical(program, rows, *, order=None, terms=None, first=True, backend="cpu", budget=None):
    """influences under the empirical measure of the selected rows of a rows domain (math_bool.Masks.from_scope);
    an assignment domain is refused (it is not stored rows)"""
    if not isinstance(rows, mb.Masks):
        raise TypeError("rows: a math_bool.Masks over the rows of a scope")
    if rows.kind != "rows":
        raise DomainError("the empirical measure needs stored rows (Masks.from_scope), not the 2^k assignments")
    return _run(rows, program, "empirical", order, terms, first, backend, budget)


def influence(program, *, rows=None, measures=None, order=None, terms=None, backend="cpu", budget=None):
    """The influence of every condition (P(d_i f = TRUE), with the UNKNOWN points apart), the total influence, the
    sensitivity histogram and the interactions (every set of 2..order conditions, plus `terms`: the measures of
    their mixed XOR derivatives and signed mixed differences), separately under each measure:
    {"uniform": ..., "empirical": ..., "interpretation": coordinate sensitivity, not a causal effect}.
    measures default: ("uniform", "empirical") with rows, ("uniform",) without."""
    if measures is None:
        measures = ("uniform", "empirical") if rows is not None else ("uniform",)
    measures = tuple(measures)
    if not measures or any(m not in MEASURES for m in measures) or len(set(measures)) != len(measures):
        raise ValueError("measures: 'uniform' and / or 'empirical'")
    out = dict(interpretation=INTERPRETATION, causal=False)
    for m in measures:
        if m == "uniform":
            out[m] = uniform(program, order=order, terms=terms, backend=backend, budget=budget)
        else:
            if rows is None:
                raise ValueError("the empirical measure needs rows (Masks.from_scope)")
            out[m] = empirical(program, rows, order=order, terms=terms, backend=backend, budget=budget)
    return out


def influence_need(program, *, rows=None, order=None, terms=None, first=True, backend="cpu"):
    """the needs an influence request checks before allocating (rows None: the uniform measure): {resource: need}"""
    masks = rows if rows is not None else mb.Masks.assignments(program.k, budget=Budget(support=(1 << 64) - 1))
    arr, nterms = _terms(program.k, order, terms, None)
    out = _Need()
    _check(_call("tr_influence_need", C.byref(masks.domain()), C.byref(program.view()), arr.ctypes.data, nterms,
                 FIRST if first else 0, _NEED_CPU if _backend(backend) == "cpu" else _NEED_DEVICE, C.byref(out)),
           what="influence needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}
