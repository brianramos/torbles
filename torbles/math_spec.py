# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SPEC-002: exact Walsh-Hadamard spectra of bounded Boolean histograms (math_lattice.Histogram projections, or any
2^k exact integers), natively on the CPU or with the CUDA kernels (torbles/math_spec.c, math_spec_gpu.cu).

    from torbles import math_lattice as ml, math_spec as ms
    h = ml.Histogram.build(masks)                      # k conditions over the rows of a scope
    f = ms.walsh(h)                                    # W(S) = sum_x h(x) (-1)^{|S and x|}, exact; f["cost"]: 2^k states
    ms.inverse(f["spectrum"])["values"]                # 2^-k sum_S W(S) (-1)^{|S and x|}: exact Fractions
    ms.parseval(h, f["spectrum"])                      # sum W^2 == 2^k sum h^2, exact (256-bit)
    ms.correlations(f["spectrum"], [[0], [0, 2]])      # W(S), even / odd weight, W(S) / N as a Fraction
    ms.coefficients(h, [[0, 2]])                       # the same for a few S, without the full transform
    t = ms.truncate(f["spectrum"], 8)                  # an explicit approximation with its exact error
    ms.evaluate(t, [0, 5, 7])                          # the kept expansion at points (exact dyadic values)
    ms.constant_parities(h)                            # |W(S)| = N: FIELD-001 affine parity relations (cross-checked)

Normalization (declared): forward unnormalized (exact integers), inverse divided by 2^k (exact dyadic rationals);
Parseval: sum_S W(S)^2 = 2^k sum_x v(x)^2. Every transform reports its full-domain cost (2^k states, k passes,
k 2^(k-1) butterflies), whatever the occupancy. Values are exact signed integers in a checked 128-bit carrier
(OverflowError instead of wrapping). Truncation is marked approximate with its exact residual energy and error bound.
Budgets raise AlgebraBudgetError before anything is allocated. Nothing is computed in Python. Opt-in: `import torbles`
does not load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral

import numpy as np

from . import algq
from . import math_lattice as ml
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["walsh", "inverse", "parseval", "correlations", "coefficients", "truncate", "evaluate",
           "constant_parities", "walsh_need", "coefficients_need", "MAX_K", "FORWARD", "INVERSE"]

MAX_K = 30
FORWARD = "forward: W(S) = sum_x v(x) (-1)^|S and x| (unnormalized, exact integers)"
INVERSE = "inverse: v(x) = 2^-k sum_S W(S) (-1)^|S and x| (divided by 2^k, exact dyadic rationals)"
PARSEVAL = "sum_S W(S)^2 = 2^k sum_x v(x)^2 (with p = v / N: sum_S (W(S) / N)^2 = 2^k sum_x p(x)^2)"
_U64 = (1 << 64) - 1
P = C.POINTER


class _Dyadic(C.Structure):
    _native_ = "tr_spec_dyadic"
    _fields_ = [("num_lo", C.c_int64), ("num_hi", C.c_int64), ("den_log2", C.c_int64)]


class _Cost(C.Structure):
    _native_ = "tr_spec_cost"
    _fields_ = [(n, C.c_uint64) for n in ("states", "passes", "butterflies", "bytes")] + [
        ("k", C.c_int32), ("normalization", C.c_int32)]


class _Energy(C.Structure):
    _native_ = "tr_spec_energy"
    _fields_ = [("cells", C.c_uint64 * 4), ("spectrum", C.c_uint64 * 4), ("k", C.c_int32), ("holds", C.c_int32)]


class _Parity(C.Structure):
    _native_ = "tr_spec_parity"
    _fields_ = [("vars", C.c_uint64), ("order", C.c_int32), ("reserved", C.c_int32)] + [
        (f"{n}_{w}", C.c_int64) for n in ("coefficient", "even", "odd", "corr_num", "corr_den") for w in ("lo", "hi")]


class _Truncation(C.Structure):
    _native_ = "tr_spec_truncation"
    _fields_ = [(n, C.c_uint64) for n in ("states", "kept", "nonzero", "dropped")] + [
        ("residual", C.c_uint64 * 4), ("total", C.c_uint64 * 4), ("l2_num", C.c_uint64 * 4),
        ("residual_l1_lo", C.c_int64), ("residual_l1_hi", C.c_int64), ("bound_num_lo", C.c_int64),
        ("bound_num_hi", C.c_int64), ("l2_den_log2", C.c_int32), ("bound_den_log2", C.c_int32), ("k", C.c_int32),
        ("exact", C.c_int32)]


class _Relation(C.Structure):
    _native_ = "tr_spec_relation"
    _fields_ = [("vars", C.c_uint64), ("constant", C.c_int32), ("reserved", C.c_int32)]


class _Parities(C.Structure):
    _native_ = "tr_spec_parities"
    _fields_ = [("total_lo", C.c_int64), ("total_hi", C.c_int64), ("support", C.c_uint64), ("constant", C.c_uint64),
                ("k", C.c_int32), ("rank", C.c_int32), ("nbasis", C.c_int32), ("agree", C.c_int32)]


for _s, _n in ((_Dyadic, 24), (_Cost, 40), (_Energy, 72), (_Parity, 96), (_Truncation, 176), (_Relation, 16),
               (_Parities, 48)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64 = C.c_void_p, C.c_int32, C.c_uint64
        sig = {
            "tr_spec_need": (i32, [i32, P(_Need)]),
            "tr_spec_walsh": (i32, [i32, vp, vp, P(_Budget), P(_Cost), P(_Refusal)]),
            "tr_spec_walsh_inverse": (i32, [i32, vp, P(_Dyadic), P(_Budget), P(_Cost), P(_Refusal)]),
            "tr_spec_parseval": (i32, [i32, vp, vp, P(_Energy)]),
            "tr_spec_correlations": (i32, [i32, vp, vp, i32, P(_Parity)]),
            "tr_spec_coefficients": (i32, [i32, vp, vp, i32, P(_Budget), P(_Parity), P(_Refusal)]),
            "tr_spec_coefficients_need": (i32, [i32, vp, i32, P(_Need)]),
            "tr_spec_truncate": (i32, [i32, vp, u64, P(_Budget), vp, vp, P(_Truncation), P(_Refusal)]),
            "tr_spec_evaluate": (i32, [i32, vp, vp, u64, vp, u64, P(_Budget), P(_Dyadic), P(_Refusal)]),
            "tr_spec_constant_parities": (i32, [i32, vp, P(_Budget), P(_Relation), u64, P(_Relation), i32,
                                                P(_Parities), P(_Refusal)]),
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
    math_gpu.bind("tr_mgpu_spec_walsh", C.c_int, [C.c_int32, C.c_void_p, C.c_void_p, P(_Budget), P(_Cost),
                                                 P(_Refusal)])
    math_gpu.bind("tr_mgpu_spec_walsh_inverse", C.c_int, [C.c_int32, C.c_void_p, P(_Dyadic), P(_Budget), P(_Cost),
                                                         P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _i128(lo, hi):
    return (int(hi) << 64) | (int(lo) & _U64)


def _u256(words):
    return sum(int(w) << (64 * i) for i, w in enumerate(words))


def _fraction(num, den_log2):
    return Fraction(num, 1 << int(den_log2))                # num / 2^d, already canonical natively


def _dyadics(arr):
    return [_fraction(_i128(d.num_lo, d.num_hi), d.den_log2) for d in arr]


def _cost(c):
    return dict(states=int(c.states), passes=int(c.passes), butterflies=int(c.butterflies), bytes=int(c.bytes),
                normalization=(FORWARD, INVERSE)[c.normalization])


def _values(values, projection="complete"):
    """(k, (2^k, 2) int64 words) of a Histogram's projection, of 2^k exact integers or of an integer ndarray"""
    return ml._vector_words(values, projection)


def _members(mask, k):
    return tuple(i for i in range(k) if (int(mask) >> i) & 1)


def walsh_need(k):
    out = _Need()
    _check(_call("tr_spec_need", int(k), C.byref(out)), what="Walsh needs")
    return _needs_dict(out)


def walsh(values, *, backend="cpu", budget=None, projection="complete"):
    """The forward Walsh-Hadamard transform: {"spectrum" (2^k ints, index S), "k", "normalization", "cost" (the
    full 2^k domain: states, passes, butterflies, bytes), "backend", "device"}"""
    if isinstance(values, ml.Histogram):
        values._fresh("Walsh spectrum")
    k, words = _values(values, projection)
    out = np.zeros_like(words)
    cost, refusal = _Cost(), _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_spec_walsh", k, words.ctypes.data, out.ctypes.data, C.byref(_budget(budget)),
                     C.byref(cost), C.byref(refusal)), refusal, "Walsh spectrum")
        label = ("cpu", "native CPU")
    else:
        gpu = _gpu()
        gpu.run("tr_mgpu_spec_walsh", k, words.ctypes.data, out.ctypes.data, C.byref(_budget(budget)), C.byref(cost),
                C.byref(refusal), refusal=refusal, what="Walsh spectrum on the device")
        label = ml._device_label(gpu)
    return dict(spectrum=ml._from_words(out), k=k, normalization=FORWARD, cost=_cost(cost), backend=label[0],
                device=label[1])


def inverse(spectrum, *, backend="cpu", budget=None, arrays=False):
    """The inverse transform 2^-k sum_S W(S) (-1)^{|S and x|}: {"values" (2^k exact Fractions), "k",
    "normalization", "cost", "backend", "device"}; arrays=True gives the same exact values as arrays instead:
    "numerators" ((2^k, 2) int64: 128-bit two's complement low and high words) and "den_log2" (value = numerator /
    2^den_log2, canonical)"""
    k, words = _values(spectrum)
    raw = np.zeros((len(words), 3), dtype=np.int64)            # the tr_spec_dyadic layout: lo, hi, den_log2
    out = raw.ctypes.data_as(P(_Dyadic))
    cost, refusal = _Cost(), _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_spec_walsh_inverse", k, words.ctypes.data, out, C.byref(_budget(budget)), C.byref(cost),
                     C.byref(refusal)), refusal, "inverse Walsh transform")
        label = ("cpu", "native CPU")
    else:
        gpu = _gpu()
        gpu.run("tr_mgpu_spec_walsh_inverse", k, words.ctypes.data, out, C.byref(_budget(budget)), C.byref(cost),
                C.byref(refusal), refusal=refusal, what="inverse Walsh transform on the device")
        label = ml._device_label(gpu)
    res = dict(k=k, normalization=INVERSE, cost=_cost(cost), backend=label[0], device=label[1])
    if arrays:
        return dict(res, numerators=raw[:, :2].copy(), den_log2=raw[:, 2].copy())
    num = ml._from_words(np.ascontiguousarray(raw[:, :2]))
    return dict(res, values=[_fraction(n, d) for n, d in zip(num, raw[:, 2].tolist())])


def parseval(values, spectrum, *, projection="complete"):
    """{"cells": sum v^2, "spectrum": sum W^2, "k", "holds": the identity sum W^2 == 2^k sum v^2, exactly}"""
    k, vw = _values(values, projection)
    k2, sw = _values(spectrum)
    if k != k2:
        raise ValueError(f"values over 2^{k} points, spectrum over 2^{k2}")
    out = _Energy()
    _check(_call("tr_spec_parseval", k, vw.ctypes.data, sw.ctypes.data, C.byref(out)), what="Parseval energies")
    return dict(cells=_u256(out.cells), spectrum=_u256(out.spectrum), k=k, holds=bool(out.holds), identity=PARSEVAL)


def _sets(sets, k):
    return ml._sets(sets, k)


def _parities(arr, k):
    return [dict(conditions=_members(r.vars, k), order=int(r.order),
                 coefficient=_i128(r.coefficient_lo, r.coefficient_hi), even=_i128(r.even_lo, r.even_hi),
                 odd=_i128(r.odd_lo, r.odd_hi),
                 correlation=Fraction(_i128(r.corr_num_lo, r.corr_num_hi), _i128(r.corr_den_lo, r.corr_den_hi)))
            for r in arr]


def correlations(spectrum, sets):
    """parity correlations of condition subsets from a full spectrum (N = W(empty)): per S {"conditions", "order",
    "coefficient" W(S), "even", "odd" (the weight with an even / odd number of TRUE conditions in S), "correlation"
    W(S) / N = P(even) - P(odd) (an exact Fraction)}"""
    k, words = _values(spectrum)
    arr, n = _sets(sets, k)
    out = (_Parity * max(n, 1))()
    _check(_call("tr_spec_correlations", k, words.ctypes.data, arr.ctypes.data, n, out), what="parity correlations")
    return _parities(out[:n], k)


def coefficients_need(values, sets):
    k = values.k if isinstance(values, ml.Histogram) else ml._k_of(len(values))
    arr, n = _sets(sets, k)
    out = _Need()
    _check(_call("tr_spec_coefficients_need", k, arr.ctypes.data, n, C.byref(out)), what="coefficient needs")
    return _needs_dict(out)


def coefficients(values, sets, *, budget=None, projection="complete"):
    """correlations() for a few sets S with W(S) and N summed directly from the values (2^k terms per S: the
    construction of new frequencies, without the full transform)"""
    if isinstance(values, ml.Histogram):
        values._fresh("Walsh coefficients")
    k, words = _values(values, projection)
    arr, n = _sets(sets, k)
    out, refusal = (_Parity * max(n, 1))(), _Refusal()
    _check(_call("tr_spec_coefficients", k, words.ctypes.data, arr.ctypes.data, n, C.byref(_budget(budget)), out,
                 C.byref(refusal)), refusal, "Walsh coefficients")
    return _parities(out[:n], k)


def truncate(spectrum, m, *, budget=None):
    """Keep the m coefficients of largest |W(S)| (ties: the smaller S): an explicit approximation. {"approximation":
    True unless nothing nonzero was dropped, "exact", "states" (2^k), "kept" [(conditions, W(S))], "kept_sets",
    "nonzero", "dropped", "residual_energy" (sum of the dropped W^2), "total_energy", "l2_error_squared" (= residual /
    2^k = sum_x (v - v_m)^2, exact), "max_error_bound" (sum of the dropped |W| / 2^k >= max_x |v - v_m|)}"""
    k, words = _values(spectrum)
    if isinstance(m, bool) or not isinstance(m, Integral) or not 0 <= int(m) <= len(words):
        raise ValueError(f"m is an integer in 0..2^{k}")
    m = int(m)
    sets = np.zeros(max(m, 1), dtype=np.uint64)
    coef = np.zeros((max(m, 1), 2), dtype=np.int64)
    out, refusal = _Truncation(), _Refusal()
    _check(_call("tr_spec_truncate", k, words.ctypes.data, m, C.byref(_budget(budget)), sets.ctypes.data,
                 coef.ctypes.data, C.byref(out), C.byref(refusal)), refusal, "spectrum truncation")
    values = ml._from_words(coef[:m])
    kept_sets = [int(s) for s in sets[:m].tolist()]
    return dict(approximation=not out.exact, exact=bool(out.exact), k=k, states=int(out.states), m=int(out.kept),
                kept=[(_members(s, k), v) for s, v in zip(kept_sets, values)], kept_sets=kept_sets,
                kept_coefficients=values, nonzero=int(out.nonzero), dropped=int(out.dropped),
                residual_energy=_u256(out.residual), total_energy=_u256(out.total),
                l2_error_squared=_fraction(_u256(out.l2_num), out.l2_den_log2),
                max_error_bound=_fraction(_i128(out.bound_num_lo, out.bound_num_hi), out.bound_den_log2),
                residual_l1=_i128(out.residual_l1_lo, out.residual_l1_hi))


def evaluate(truncation, points, *, budget=None):
    """the kept expansion v_m(x) = 2^-k sum_{S kept} W(S) (-1)^{|S and x|} at each point x (an int < 2^k): exact
    Fractions (the fixed-basis action: m terms per point)"""
    k = int(truncation["k"])
    sets = np.array(truncation["kept_sets"] or [0], dtype=np.uint64)
    m = len(truncation["kept_sets"])
    coef = ml._words_of(truncation["kept_coefficients"]) if m else np.zeros((1, 2), dtype=np.int64)
    pts = [int(p) for p in points]
    if any(not 0 <= p < (1 << k) for p in pts):
        raise ValueError(f"points are integers in 0..2^{k} - 1")
    parr = np.array(pts or [0], dtype=np.uint64)
    out, refusal = (_Dyadic * max(len(pts), 1))(), _Refusal()
    _check(_call("tr_spec_evaluate", k, sets.ctypes.data, coef.ctypes.data, m, parr.ctypes.data, len(pts),
                 C.byref(_budget(budget)), out, C.byref(refusal)), refusal, "truncated expansion")
    return _dyadics(out[:len(pts)])


def constant_parities(values, *, budget=None, projection="complete"):
    """The condition subsets S whose parity is constant over the weighted points (|W(S)| = N: XOR over S = 0 when
    W(S) = N, = 1 when W(S) = -N), and FIELD-001's affine GF(2) relation basis of the occupied cells, natively
    cross-checked: {"relations" [(conditions, constant)], "basis", "rank", "agree", "total", "support"}"""
    if isinstance(values, ml.Histogram):
        values._fresh("constant parities")
    k, words = _values(values, projection)
    rel = (_Relation * len(words))()
    basis = (_Relation * (k + 1))()
    out, refusal = _Parities(), _Refusal()
    _check(_call("tr_spec_constant_parities", k, words.ctypes.data, C.byref(_budget(budget)), rel, len(words), basis,
                 k + 1, C.byref(out), C.byref(refusal)), refusal, "constant parities")
    return dict(relations=[(_members(r.vars, k), int(r.constant)) for r in rel[:int(out.constant)]],
                basis=[(_members(r.vars, k), int(r.constant)) for r in basis[:int(out.nbasis)]],
                rank=int(out.rank), agree=bool(out.agree), total=_i128(out.total_lo, out.total_hi),
                support=int(out.support), k=k)
