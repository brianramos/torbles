# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Search, representation and output budgets of the mathematical modules (FND-010), natively.

    from torbles import math_budgets as mb
    b = mb.Budget(output=10_000)                     # the defaults, with one limit changed
    mb.check(b, variables=6, degree=4, support=mb.monomials(6, 4))
    mb.choose(b, dict(support=mb.monomials(8, 6)), 210.0, dict(support=48), 960.0)
    mb.count_cells(cells, ncells, rows=None, budget=b, backend="cpu" | "cuda")

Sizes are exact integers (UNKNOWN = 2^64-1 when a size does not fit 64 bits). A refusal raises AlgebraBudgetError
with .resource, .need and .limit, before anything is computed or returned. Opt-in: `import torbles` does not load it.
"""
import ctypes as C

import numpy as np

from . import algq
from .math_errors import RESOURCES, UNKNOWN, check as _check

__all__ = ["Budget", "RESOURCES", "UNKNOWN", "monomials", "binomial", "product", "power2", "check", "choose",
           "count_cells"]


class _Budget(C.Structure):
    _fields_ = [("limit", C.c_uint64 * len(RESOURCES))]


class _Need(C.Structure):
    _fields_ = [("need", C.c_uint64 * len(RESOURCES))]


class _Refusal(C.Structure):
    _fields_ = [("resource", C.c_int32), ("refused", C.c_int32), ("need", C.c_uint64), ("limit", C.c_uint64)]


class _Choice(C.Structure):
    _fields_ = [("choice", C.c_int32), ("reason", C.c_int32), ("expanded", _Refusal), ("factored", _Refusal),
                ("expanded_cost", C.c_double), ("factored_cost", C.c_double)]


_native = None


def _api():
    global _native
    if _native is None:
        u64, P = C.c_uint64, C.POINTER
        sig = {
            "default": (None, [P(_Budget)]), "polynomial_kernel": (None, [P(_Budget)]),
            "binomial": (u64, [u64, u64]), "monomials": (u64, [C.c_uint32, C.c_uint32]),
            "homogeneous": (u64, [C.c_uint32, C.c_uint32]), "product": (u64, [P(u64), C.c_uint32]),
            "power2": (u64, [C.c_uint32]),
            "check": (C.c_int32, [P(_Budget), P(_Need), P(_Refusal)]),
            "choose": (C.c_int32, [P(_Budget), P(_Need), C.c_double, P(_Need), C.c_double, C.c_int32, P(_Choice)]),
            "count_cells": (C.c_int32, [P(_Budget), C.c_void_p, C.c_void_p, u64, u64, P(u64), P(_Refusal)]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, "tr_math_budget_" + name)
            fn.restype, fn.argtypes = restype, args
            native[name] = fn
        _native = native
    return _native


def _u64(value, what):
    value = int(value)
    if not 0 <= value <= UNKNOWN:
        raise ValueError(f"{what} must be between 0 and 2^64-1")
    return value


class Budget:
    """limits per resource (variables, degree, support, tensor, circuit, iterations, output, states, bytes)"""
    def __init__(self, base=None, **limits):
        raw = _Budget()
        if base is None:
            _api()["default"](C.byref(raw))
        else:
            raw.limit[:] = list(base.raw.limit)
        for name, value in limits.items():
            if name not in RESOURCES:
                raise ValueError(f"unknown budget resource {name!r} (one of {', '.join(RESOURCES)})")
            raw.limit[RESOURCES.index(name)] = _u64(value, name)
        self.raw = raw

    @classmethod
    def polynomial_kernel(cls):
        """the fixed limits of the native polynomial kernel: 8 variables, degree 8, 128 terms"""
        b = cls.__new__(cls); b.raw = _Budget(); _api()["polynomial_kernel"](C.byref(b.raw))
        return b

    def replace(self, **limits):
        return Budget(self, **limits)

    @property
    def limits(self):
        return dict(zip(RESOURCES, (int(v) for v in self.raw.limit)))

    def __repr__(self):
        return f"Budget({', '.join(f'{k}={v}' for k, v in self.limits.items())})"


def _need(needs):
    raw = _Need()
    for name, value in needs.items():
        if name not in RESOURCES:
            raise ValueError(f"unknown budget resource {name!r}")
        raw.need[RESOURCES.index(name)] = _u64(value, name)
    return raw


def binomial(n, k):
    return int(_api()["binomial"](_u64(n, "n"), _u64(k, "k")))


def monomials(variables, degree, homogeneous=False):
    """monomials in `variables` variables of total degree <= degree (exactly degree when homogeneous)"""
    fn = _api()["homogeneous" if homogeneous else "monomials"]
    return int(fn(int(variables), int(degree)))


def product(dims):
    """dense elements of a tensor / cells of a joint domain with these dimensions"""
    a = (C.c_uint64 * max(len(dims), 1))(*[_u64(d, "dimension") for d in dims])
    return int(_api()["product"](a, len(dims)))


def power2(k):
    return int(_api()["power2"](int(k)))


def check(budget, what="", **needs):
    """None, or AlgebraBudgetError at the first resource (in RESOURCES order) over its limit"""
    refusal = _Refusal()
    _check(_api()["check"](C.byref(budget.raw), C.byref(_need(needs)), C.byref(refusal)), refusal, what)


def choose(budget, expanded, expanded_cost, factored=None, factored_cost=0.0, allow_factored=True, what=""):
    """which representation runs: {"choice": "expanded"|"factored", "reason": "cheaper"|"only fit", costs, refusals}"""
    out = _Choice()
    allow = bool(allow_factored and factored is not None)
    rc = _api()["choose"](C.byref(budget.raw), C.byref(_need(expanded)), float(expanded_cost),
                          C.byref(_need(factored or {})) if allow else None, float(factored_cost if allow else 0.0),
                          int(allow), C.byref(out))
    if rc == -5:
        first = out.expanded if out.expanded.refused else out.factored
        _check(rc, first, what or "no representation fits")
    _check(rc)

    def refusal(r):
        return None if not r.refused else dict(resource=RESOURCES[r.resource], need=int(r.need), limit=int(r.limit))
    return dict(choice=("expanded", "factored")[out.choice], reason=("cheaper", "only fit")[out.reason],
                expanded_cost=out.expanded_cost, factored_cost=out.factored_cost if allow else None,
                expanded_refusal=refusal(out.expanded), factored_refusal=refusal(out.factored) if allow else None)


def count_cells(cells, ncells, rows=None, budget=None, backend="cpu"):
    """exact number of distinct cell ids among the selected rows, refused before any output past the output limit

    cells: uint32 cell id per row (< ncells); rows: optional boolean mask or packed uint64 bitmap of the rows counted.
    backend "cuda" runs the CUDA kernel (math_gpu.device() says whether on a device or in emulation).
    """
    budget = budget or Budget()
    c = np.ascontiguousarray(cells, dtype=np.uint32)
    n = len(c)
    bits = None
    if rows is not None:
        r = np.asarray(rows)
        if r.dtype == np.bool_:
            if len(r) != n:
                raise ValueError("rows mask length differs from cells")
            padded = np.zeros(((n + 63) // 64) * 64, dtype=np.uint8); padded[:n] = r
            bits = np.packbits(padded, bitorder="little").view(np.uint64)
        else:
            bits = np.ascontiguousarray(r, dtype=np.uint64)
            if len(bits) < (n + 63) // 64:
                raise ValueError("rows bitmap is shorter than the cells")
    out, refusal = C.c_uint64(), _Refusal()
    args = (C.byref(budget.raw), c.ctypes.data if n else None, bits.ctypes.data if bits is not None else None,
            n, _u64(ncells, "ncells"), C.byref(out), C.byref(refusal))
    if backend == "cpu":
        _check(_api()["count_cells"](*args), refusal, "distinct cells")
    elif backend == "cuda":
        from . import math_gpu
        math_gpu.bind("tr_mgpu_budgets_count_cells", C.c_int,
                      [C.POINTER(_Budget), C.c_void_p, C.c_void_p, C.c_uint64, C.c_uint64,
                       C.POINTER(C.c_uint64), C.POINTER(_Refusal)])
        math_gpu.run("tr_mgpu_budgets_count_cells", *args, refusal=refusal, what="distinct cells")
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    return int(out.value)
