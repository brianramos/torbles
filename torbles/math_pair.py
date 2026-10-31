# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PAIR-001: sorted-distribution pairwise summaries from DIST-002 prefix states, natively on the CPU or with the CUDA
kernel (torbles/math_pair.c, math_pair_gpu.cu), without enumerating pairs.

    from torbles import math_dist as md, math_pair as mp
    p = md.histogram(scope, ["x"]).prefix(degree=0)          # classes = groups (mass "count" or "weight")
    mp.abs_within(p, group=0, ordered=False, self_pairs=False, mean=True)   # the Gini mean difference
    mp.abs_cross(p, 0, 1); mp.mann_whitney(p, 0, 1, auc=True)   # ties count half
    mp.ks(p, 0, 1); mp.wasserstein(p, 0, 1); mp.cramer_von_mises(p, 0, 1, other=q)  # class 1 of another state q
    mp.query(p, [dict(op="ks", a=0, b=1), ...], backend="cuda")  # one CUDA work item per request

A class is one group of a prefix state: the expanded multiset of its present values (each repeated by its mass: rows,
or units of weight). Results are exact ints / Fractions computed natively; Python only marshals and formats. The
conventions (self-pairs, ordering, normalization, empty classes) are those of torbles/math_pair.h and
docs/math/pair.md. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from .math_budgets import _Budget, _Refusal, Budget
from .math_dist import Prefix, _i128, _i256
from .math_errors import check as _check

__all__ = ["OPS", "REQUEST_DTYPE", "RESULT_DTYPE", "requests", "query", "query_arrays", "abs_within", "abs_cross",
           "mann_whitney", "ks", "wasserstein", "cramer_von_mises"]

OPS = {"abs_within": 1, "abs_cross": 2, "mann_whitney": 3, "ks": 4, "wasserstein": 5, "cramer_von_mises": 6}
ORDERED, SELF, MEAN = 1, 2, 4
P = C.POINTER
i32, i64, u64, vp = C.c_int32, C.c_int64, C.c_uint64, C.c_void_p


class _Request(C.Structure):
    _native_ = "tr_pair_request"
    _fields_ = [("op", i32), ("flags", i32), ("a", u64), ("b", u64), ("reserved", i64)]


class _Result(C.Structure):
    _native_ = "tr_pair_result"
    _fields_ = [("status", i32), ("empty", i32), ("num", u64 * 4), ("den", u64 * 4), ("pairs", u64 * 4),
                ("mass_a", i64 * 2), ("mass_b", i64 * 2), ("at", i64), ("has_at", i32), ("reserved", i32)]


assert C.sizeof(_Request) == 32 and C.sizeof(_Result) == 152
REQUEST_DTYPE = np.dtype([("op", "<i4"), ("flags", "<i4"), ("a", "<u8"), ("b", "<u8"), ("reserved", "<i8")])
RESULT_DTYPE = np.dtype([("status", "<i4"), ("empty", "<i4"), ("num", "<u8", (4,)), ("den", "<u8", (4,)),
                         ("pairs", "<u8", (4,)), ("mass_a", "<i8", (2,)), ("mass_b", "<i8", (2,)), ("at", "<i8"),
                         ("has_at", "<i4"), ("reserved", "<i4")])
assert REQUEST_DTYPE.itemsize == 32 and RESULT_DTYPE.itemsize == 152

_native = None


def _api():
    global _native
    if _native is None:
        fn = algq.lib.tr_pair_query
        fn.restype, fn.argtypes = i32, [vp, vp, vp, u64, vp, P(_Budget), P(_Refusal)]
        _native = fn
    return _native


def _gpu():
    from . import math_dist
    gpu = math_dist._gpu()
    gpu.bind("tr_mgpu_pair_query", C.c_int, [vp, vp, vp, u64, vp, P(_Budget), P(_Refusal)])
    return gpu


def _prefix(state, what):
    if not isinstance(state, Prefix):
        raise TypeError(f"{what}: a math_dist.Prefix state (Histogram.prefix(...))")
    return state


def requests(items):
    """a REQUEST_DTYPE array from dicts {"op", "a" (group of the first state), "b" (group of the second: two-class
    ops), "ordered" (abs_within), "self_pairs", "mean" (abs_within, abs_cross; mann_whitney: the AUC)}"""
    items = list(items)
    a = np.zeros(len(items), dtype=REQUEST_DTYPE)
    for i, q in enumerate(items):
        op = q["op"]
        if op not in OPS:
            raise ValueError(f"op is one of {', '.join(OPS)}")
        flags = (ORDERED if q.get("ordered") else 0) | (SELF if q.get("self_pairs") else 0) | \
                (MEAN if q.get("mean") or q.get("auc") else 0)
        a["op"][i], a["flags"][i], a["a"][i] = OPS[op], flags, int(q.get("a", q.get("group", 0)))
        a["b"][i] = int(q.get("b", 0))
    return a


def query_arrays(state, reqs, *, other=None, backend="cpu", budget=None):
    """the raw RESULT_DTYPE answers of a REQUEST_DTYPE array against state (class A) and other (class B; None: state),
    natively; every answer carries its own status"""
    _prefix(state, "query")
    if other is not None:
        _prefix(other, "query (other)")
    reqs = np.ascontiguousarray(reqs, dtype=REQUEST_DTYPE)
    out = np.zeros(len(reqs), dtype=RESULT_DTYPE)
    refusal = _Refusal()
    raw = (budget or Budget()).raw
    rp = reqs.ctypes.data if len(reqs) else None
    op = out.ctypes.data if len(out) else None
    if backend == "cpu":
        _check(_api()(state._open(), other._open() if other is not None else None, rp, len(reqs), op, C.byref(raw),
                      C.byref(refusal)), refusal, "pairwise requests")
    elif backend == "cuda":
        gpu = _gpu()
        state.device(budget)
        if other is not None:
            other.device(budget)
        gpu.run("tr_mgpu_pair_query", state._device, other._device if other is not None else None, rp, len(reqs), op,
                C.byref(raw), C.byref(refusal), refusal=refusal, what="pairwise requests on the device")
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    return out


def _format(name, r):
    if int(r["status"]):
        _check(int(r["status"]), what=f"{name} request")
    num, den = _i256(r["num"]), _i256(r["den"])
    empty = bool(r["empty"])
    out = dict(empty=empty, pairs=_i256(r["pairs"]), mass_a=_i128(r["mass_a"]), mass_b=_i128(r["mass_b"]))
    if empty:
        out["value"] = None
    else:
        v = Fraction(num, den)
        out["value"] = v.numerator if v.denominator == 1 else v
    if name == "ks":
        out["at"] = int(r["at"]) if r["has_at"] else None
    return out


def query(state, items, *, other=None, backend="cpu", budget=None):
    """answers of request dicts (see requests()): [{"value" (int / Fraction, None when empty), "empty", "pairs",
    "mass_a", "mass_b", and "at" for ks}]; a request the native code refuses raises"""
    items = list(items)
    res = query_arrays(state, requests(items), other=other, backend=backend, budget=budget)
    return [_format(q["op"], r) for q, r in zip(items, res)]


def _one(state, q, other, backend):
    return query(state, [q], other=other, backend=backend)[0]


def abs_within(state, group=0, *, ordered=False, self_pairs=False, mean=False, backend="cpu"):
    """sum (mean: average) of |y_i - y_j| over the pairs of one class: unordered {i, j} unless ordered, self-pairs
    (i, i) excluded unless self_pairs; mean with self-pairs excluded is the Gini mean difference"""
    return _one(state, dict(op="abs_within", a=group, ordered=ordered, self_pairs=self_pairs, mean=mean), None,
                backend)


def abs_cross(state, a, b, *, other=None, self_pairs=False, mean=False, backend="cpu"):
    """sum (mean) of |x_i - y_j| over the pairs i in class a, j in class b (of other, or of state)"""
    return _one(state, dict(op="abs_cross", a=a, b=b, self_pairs=self_pairs, mean=mean), other, backend)


def mann_whitney(state, a, b, *, other=None, auc=False, self_pairs=False, backend="cpu"):
    """U = #{x_i > y_j} + 1/2 #{x_i = y_j} over i in class a, j in class b; auc: U / (pairs)"""
    return _one(state, dict(op="mann_whitney", a=a, b=b, self_pairs=self_pairs, auc=auc), other, backend)


def ks(state, a, b, *, other=None, backend="cpu"):
    """the two-sample Kolmogorov-Smirnov statistic sup |F_a - F_b| (exact Fraction; "at": where it is attained)"""
    return _one(state, dict(op="ks", a=a, b=b), other, backend)


def wasserstein(state, a, b, *, other=None, backend="cpu"):
    """the 1-Wasserstein distance of the normalized distributions (logical units)"""
    return _one(state, dict(op="wasserstein", a=a, b=b), other, backend)


def cramer_von_mises(state, a, b, *, other=None, backend="cpu"):
    """the two-sample Cramer-von Mises statistic T (Anderson 1962; ties at the pooled values)"""
    return _one(state, dict(op="cramer_von_mises", a=a, b=b), other, backend)
