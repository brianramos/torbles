# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-001: exhaustive verification of affine, sum and product hypotheses of a relationship catalog, natively on
the CPU or with the CUDA kernel (torbles/math_disc.c, math_disc_gpu.cu).

    from torbles.relationships import discover
    from torbles import math_disc
    cat = discover(db, "t", ["x", "y", "s"])
    math_disc.verify(cat, "sum", "s", "x", "y", backend="cuda")

The evidence equals Catalog.probe's (checked, excluded, violations, smallest violating row, usable) but registers
nothing. A proof is usable as an exact rewrite only over every physical row, with no NULL, no violation, varied
inputs and a non-degenerate scale; a catalog whose table changed since discovery (revision, rows or used rows) is
refused with StaleStateError. Opt-in: `import torbles` does not load it.
"""
import ctypes as C

from . import algq
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import check as _check

MODULES = {"affine": 1, "product": 2, "sum": 4}


class _View(C.Structure):
    _fields_ = [("rows", C.c_uint64), ("used_count", C.c_uint64), ("used", C.c_void_p),
                ("module", C.c_int32), ("arity", C.c_int32),
                ("codes", C.c_void_p * 3), ("validity", C.c_void_p * 3), ("offset", C.c_int64 * 3)]


class _Evidence(C.Structure):
    _fields_ = [("checked", C.c_uint64), ("excluded", C.c_uint64), ("violations", C.c_uint64),
                ("witness", C.c_uint64), ("min_left", C.c_int64), ("max_left", C.c_int64),
                ("min_right", C.c_int64), ("max_right", C.c_int64), ("overflow", C.c_int32), ("usable", C.c_int32)]


_native = None


def _api():
    global _native
    if _native is None:
        P = C.POINTER
        get = algq.lib.tr_disc_view_get
        get.restype, get.argtypes = C.c_int32, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.c_int32, C.c_int32,
                                                C.c_int32, P(_View)]
        ver = algq.lib.tr_disc_verify
        ver.restype, ver.argtypes = C.c_int32, [P(_View), C.c_int64, C.c_int64, P(_Evidence)]
        nbytes = algq.lib.tr_disc_device_bytes
        nbytes.restype, nbytes.argtypes = C.c_uint64, [P(_View)]
        prop = algq.lib.tr_disc_propose_affine
        prop.restype, prop.argtypes = C.c_int32, [P(_View), P(C.c_int64), P(C.c_int64)]
        _native = dict(get=get, verify=ver, bytes=nbytes, propose=prop)
    return _native


def _gpu():
    from . import math_gpu
    math_gpu.bind("tr_mgpu_disc_verify", C.c_int, [C.POINTER(_View), C.c_int64, C.c_int64, C.POINTER(_Budget),
                                                  C.POINTER(_Evidence), C.POINTER(_Refusal)])
    return math_gpu


def _evidence(e, module, backend, device):
    return dict(module=module, checked=int(e.checked), excluded=int(e.excluded), violations=int(e.violations),
                witness=None if e.witness == (1 << 64) - 1 else int(e.witness), usable=bool(e.usable),
                varied_left=bool(e.checked and e.min_left != e.max_left),
                varied_right=bool(module == "affine" or (e.checked and e.min_right != e.max_right)),
                backend=backend, device=device)


def view(catalog, module, out, left, right=None):
    """the native view of one hypothesis (catalog-owned buffers): rows, used rows, arity"""
    if module not in MODULES:
        raise ValueError(f"unknown relationship module {module!r}")
    if module != "affine" and right is None:
        raise ValueError("product/sum need right=")
    t = catalog.table
    v = _View()
    with catalog._lock:
        catalog._open()
        _check(_api()["get"](catalog._handle, t.at, t.used_bm.ctypes.data, MODULES[module], t.idx(out), t.idx(left),
                             t.idx(right) if right is not None else -1, C.byref(v)), what="relationship view")
    return v


def verify(catalog, module, out, left, right=None, *, scale=1, bias=0, backend="cpu", budget=None):
    """exhaustive evidence of out = scale*left + bias / left*right / left + right over the catalog's used rows"""
    scale, bias = int(scale), int(bias)
    for value in (scale, bias):
        if not -(1 << 63) <= value < (1 << 63):
            raise OverflowError("scale and bias are int64")
    with catalog._lock:
        v = view(catalog, module, out, left, right)
        e = _Evidence()
        if backend == "cpu":
            _check(_api()["verify"](C.byref(v), scale, bias, C.byref(e)), what="relationship verification")
            return _evidence(e, module, "cpu", "native CPU")
        if backend == "cuda":
            gpu = _gpu()
            refusal = _Refusal()
            b = budget or Budget()
            gpu.run("tr_mgpu_disc_verify", C.byref(v), scale, bias, C.byref(b.raw), C.byref(e), C.byref(refusal),
                    refusal=refusal, what="relationship verification on the device")
            info = gpu.device()
            return _evidence(e, module, "cuda-emulated" if info["emulated"] else "cuda", info["name"])
        raise ValueError("backend is 'cpu' or 'cuda'")


def propose_affine(catalog, out, left):
    """the affine proposal discovery would verify for out = scale*left + bias: (scale, bias), or None"""
    with catalog._lock:
        v = view(catalog, "affine", out, left)
        scale, bias = C.c_int64(), C.c_int64()
        if not _api()["propose"](C.byref(v), C.byref(scale), C.byref(bias)):
            return None
        return int(scale.value), int(bias.value)


def device_bytes(catalog, module, out, left, right=None):
    """bytes the CUDA verification copies to the device for this hypothesis"""
    with catalog._lock:
        v = view(catalog, module, out, left, right)
        return int(_api()["bytes"](C.byref(v)))
