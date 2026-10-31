# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DELTA-002: bilinear (join) and nonlinear update laws over DELTA-001 maintained states, natively on the CPU or with
the CUDA kernels (torbles/math_bilinear.c, math_bilinear_gpu.cu).

    from torbles import math_bank as mbk, math_delta as md, math_bilinear as mbl
    sa = md.State.from_bank(mbk.build(scope_a, ["1", "x", "x^2"]))     # groups: key * GA + g (group_by=[key, g])
    sb = md.State.from_bank(mbk.build(scope_b, ["1", "y"]))            # groups: key * GB + h
    j = mbl.Join.build(sa, sb, keys=K, groups_a=GA, groups_b=GB, pairs=[("1", "1"), ("x", "1"), ("x", "y")])
    j.cell(g, h)              # {"rows", "count": [per pair], "sum": [per pair]}: COUNT(*), COUNT / SUM(a_i * b_j)
    j2 = j.apply(a=delta_a, b=delta_b, backend="cpu")    # or "cuda"; either side may be None
    j2.terms                  # the update's cross terms {"dA_B", "A_dB", "dA_dB"}: d(AB) = dA B + A dB + dA dB
    j2.finalize(g, h, "avg", pair=1)                     # exact Fraction (None for SQL NULL)
    mbl.matrix_delta(A, B, dA, dB, backend="cuda")       # AB, dA B, A dB, dA dB, the updated product (exact ints)
    v = mbl.Values.from_scope(scope, "x", kind="multiset")   # or "extremum"
    v2 = v.apply(delta); v2.finalize(g, "count_distinct"), v2.finalize(g, "min")
    mbl.finalize(state, g, "var_samp", "x", "x^2")       # ratios and variances from their sufficient sums
    mbl.requirement("count_distinct")                    # the retained state a finalizer requests

The update calculus is the exact finite difference of states, not a derivative (JET-001): the second-order cross
term dA dB is always kept. Nonlinear finalizers keep their own typed retained state (per-value multiplicities, the
extremum with its multiplicity and a rebuild refusal, sufficient sums, per-key cells); none inherits the SUM
deletion rule, and asking one of a state that lacks what it needs raises DomainError. Python marshals arguments and
formats results; nothing is computed here. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_carriers as _mc
from . import math_delta as md
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, check as _check

__all__ = ["Join", "Values", "matrix_delta", "finalize", "requirement", "RebuildRequired", "FINALIZERS", "WORDS"]

WORDS = 5
NONE = (1 << 64) - 1
FINALIZERS = {"sum": 1, "count": 2, "avg": 3, "ratio": 4, "var_pop": 5, "var_samp": 6, "count_distinct": 7,
              "sum_distinct": 8, "min": 9, "max": 10, "join_sum": 11}
RETAINED = {1: "linear", 2: "sufficient sums", 3: "multiplicities", 4: "extremum", 5: "per-key cells"}
DELETES = {1: "subtract", 2: "sufficient sums", 3: "multiplicity", 4: "rebuild when the extreme goes",
           5: "per-key cross terms"}
KINDS = {"multiset": 1, "extremum": 2}
REASONS = {0: None, 1: "negative rows", 2: "negative nulls", 3: "negative value multiplicity",
           4: "present above rows", 5: "extreme deleted"}
P = C.POINTER


class _Wide(C.Structure):
    _native_ = "tr_bilin_wide"
    _fields_ = [("w", C.c_uint64 * WORDS)]


class _Spec(C.Structure):
    _native_ = "tr_bilin_spec"
    _fields_ = [(n, C.c_uint64) for n in ("nkeys", "groups_a", "groups_b", "null_key")] + [
        ("npairs", C.c_int32), ("reserved", C.c_int32)]


class _Info(C.Structure):
    _native_ = "tr_bilin_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "nkeys", "groups_a", "groups_b", "null_key", "step", "bytes",
                                          "touched")] + [
        (n, C.c_uint64 * 2) for n in ("identity", "base", "side_a", "side_b")] + [
        (n, C.c_int32) for n in ("npairs", "values", "terms_a", "terms_b")]


class _CrossView(C.Structure):
    _native_ = "tr_bilin_cross_view"
    _fields_ = [(n, C.c_uint64) for n in ("keys", "groups_a", "groups_b", "ngroups", "skip")] + [
        (n, C.c_int32) for n in ("npairs", "values", "terms_a", "terms_b")] + [("pairs", C.c_void_p)] + [
        (n, C.c_void_p * 2) for n in ("a_rows", "a_count", "a_sum", "b_rows", "b_count", "b_sum")]


class _Requirement(C.Structure):
    _native_ = "tr_bilin_requirement"
    _fields_ = [(n, C.c_int32) for n in ("finalizer", "retained", "linear", "deletes", "sufficient",
                                         "needs_both_sides", "alternative", "alternative_deletes")] + [
        ("name", C.c_char * 32)]


class _ValuesInfo(C.Structure):
    _native_ = "tr_bilin_values_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "entries", "step", "bytes")] + [
        ("identity", C.c_uint64 * 2), ("base", C.c_uint64 * 2)] + [
        (n, C.c_int32) for n in ("kind", "column", "ncols", "scale")]


class _GroupValues(C.Structure):
    _native_ = "tr_bilin_group_values"
    _fields_ = [("rows", C.c_int64), ("nulls", C.c_int64), ("present", C.c_int64), ("distinct", C.c_uint64),
                ("min", C.c_int64), ("max", C.c_int64), ("min_count", C.c_int64), ("max_count", C.c_int64),
                ("sum_distinct", C.c_int64 * 2)]


class _Violation(C.Structure):
    _native_ = "tr_bilin_violation"
    _fields_ = [("group", C.c_uint64), ("reason", C.c_int32), ("has_value", C.c_int32), ("value", C.c_int64),
                ("have", C.c_int64), ("change", C.c_int64)]


for _s, _n in ((_Wide, 40), (_Spec, 40), (_Info, 144), (_CrossView, 160), (_Requirement, 64), (_ValuesInfo, 80),
               (_GroupValues, 80), (_Violation, 40)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R, N = P(_Budget), P(_Refusal), P(_Need)
        sig = {
            "tr_bilin_join_need": (i32, [vp, vp, P(_Spec), vp, i32, N]),
            "tr_bilin_join_build": (vp, [vp, vp, P(_Spec), vp, B, R, e]),
            "tr_bilin_update_need": (i32, [vp, vp, vp, vp, vp, vp, vp, i32, N]),
            "tr_bilin_join_update": (vp, [vp, vp, vp, vp, vp, vp, vp, B, R, e]),
            "tr_bilin_job_build": (vp, [vp, vp, P(_Spec), vp, i32, B, R, e]),
            "tr_bilin_job_update": (vp, [vp, vp, vp, vp, vp, vp, vp, i32, B, R, e]),
            "tr_bilin_job_get_view": (i32, [vp, P(_CrossView)]),
            "tr_bilin_cross_eval": (i32, [P(_CrossView), vp, B, R]),
            "tr_bilin_join_finish": (vp, [vp, vp, B, R, e]),
            "tr_bilin_job_free": (None, [vp]),
            "tr_bilin_join_get_info": (i32, [vp, P(_Info)]),
            "tr_bilin_join_get_values": (i32, [vp, vp, u64]),
            "tr_bilin_join_get_terms": (i32, [vp, vp, u64]),
            "tr_bilin_join_get_pairs": (i32, [vp, vp, u64]),
            "tr_bilin_join_equal": (i32, [vp, vp]),
            "tr_bilin_join_summarizes": (i32, [vp, vp, vp]),
            "tr_bilin_join_finalize": (i32, [vp, u64, i32, i32, i32, P(vp), P(i32), B, R]),
            "tr_bilin_join_free": (None, [vp]),
            "tr_bilin_matrix_need": (i32, [u64, u64, u64, i32, i32, i32, N]),
            "tr_bilin_matrix_delta": (i32, [u64, u64, u64, vp, vp, vp, vp, vp, B, R]),
            "tr_bilin_requirement_get": (i32, [i32, P(_Requirement)]),
            "tr_bilin_finalize_state": (i32, [vp, u64, i32, i32, i32, P(vp), P(i32), B, R]),
            "tr_bilin_values_need_scope": (i32, [vp, i32, i32, N]),
            "tr_bilin_values_from_scope": (vp, [vp, i32, i32, B, R, e]),
            "tr_bilin_values_need_apply": (i32, [vp, vp, N]),
            "tr_bilin_values_apply": (vp, [vp, vp, B, R, P(_Violation), e]),
            "tr_bilin_values_get_info": (i32, [vp, P(_ValuesInfo)]),
            "tr_bilin_values_group": (i32, [vp, u64, P(_GroupValues)]),
            "tr_bilin_values_entries": (i32, [vp, u64, vp, vp, u64, P(u64)]),
            "tr_bilin_values_finalize": (i32, [vp, u64, i32, P(vp), P(i32), B, R]),
            "tr_bilin_values_equal": (i32, [vp, vp]),
            "tr_bilin_values_free": (None, [vp]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, args
            native[name[9:]] = fn
        _native = native
    return _native


def _call(name, *args):
    return _api()[name](*args)


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_bilin_cross", C.c_int, [P(_CrossView), C.c_void_p, P(_Budget), P(_Refusal)])
        math_gpu.bind("tr_mgpu_bilin_matrix", C.c_int, [C.c_uint64, C.c_uint64, C.c_uint64, C.c_void_p, C.c_void_p,
                                                       C.c_void_p, C.c_void_p, C.c_void_p, P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _digest(words):
    return (int(words[1]) << 64) | int(words[0])


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _wide_ints(words):
    """exact Python ints from an (n, 5) uint64 array of 320-bit two's complement values"""
    flat = np.ascontiguousarray(words, dtype=np.uint64).reshape(-1, WORDS)
    return [int.from_bytes(row.tobytes(), "little", signed=True) for row in flat]


def _fraction(handle):
    """a Fraction from an owned tr_bigq handle (freed here)"""
    api = _mc._api()
    try:
        n = _mc._bigint_value(api["bigq_numerator"](handle))
        d = _mc._bigint_value(api["bigq_denominator"](handle))
    finally:
        api["bigq_free"](handle)
    make = getattr(Fraction, "_from_coprime_ints", None)
    return make(n, d) if make else Fraction(n, d)


def _finalized(code, out, null, refusal, what):
    _check(code, refusal, what)
    if null.value:
        return None
    if not out.value:
        raise MemoryError(f"{what}: native allocation failed")
    return _fraction(out.value)


def _finalizer(name):
    if name not in FINALIZERS:
        raise ValueError(f"finalizer is one of {', '.join(FINALIZERS)}")
    return FINALIZERS[name]


def _device(gpu):
    d = gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


def requirement(finalizer):
    """the retained state a finalizer requests under updates and how it handles a delete (a native table)"""
    out = _Requirement()
    _check(_call("requirement_get", _finalizer(finalizer), C.byref(out)), what="requirement")
    return dict(finalizer=out.name.decode(), retained=RETAINED[out.retained], linear=bool(out.linear),
                deletes=DELETES[out.deletes], sufficient=int(out.sufficient),
                needs_both_sides=bool(out.needs_both_sides),
                alternative=RETAINED.get(out.alternative), alternative_deletes=DELETES.get(out.alternative_deletes))


def _term_index(state, spec, what):
    """the index of a term in a maintained state's basis: an index, an exponent tuple, {name: exponent} or 'x*y'"""
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        if not 0 <= int(spec) < len(state.terms):
            raise ValueError(f"{what}: term index {spec} out of range")
        return int(spec)
    row = tuple(int(k) for k in mbk._basis(state.names, [spec])[0])
    if row not in state.terms:
        raise ValueError(f"{what}: {spec!r} is not a term of the state's basis")
    return state.terms.index(row)


# ---------------------------------------------------------------------- bilinear join states
class Join:
    """An immutable bilinear join state (native): per output cell (g, h) and pair (A term i, B term j), the exact
    COUNT(*), COUNT(a_i * b_j) and SUM(a_i * b_j) of the inner join of two DELTA-001 states on their key, with the
    side states it is the join of."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Join.build(...) or Join.apply(...)")

    @classmethod
    def _wrap(cls, handle, a, b, pair_specs, backend, device):
        self = object.__new__(cls)
        self._handle, self.a, self.b, self.pairs = handle, a, b, tuple(pair_specs)
        self.backend, self.device = backend, device
        return self

    @staticmethod
    def _spec(a, b, keys, groups_a, groups_b, pairs, null_key):
        idx = []
        for p in pairs:
            if len(p) != 2:
                raise ValueError("a pair is (A term, B term)")
            idx += [_term_index(a, p[0], "A term"), _term_index(b, p[1], "B term")]
        arr = np.ascontiguousarray(np.array(idx, dtype=np.int32))
        spec = _Spec(int(keys), int(groups_a), int(groups_b), NONE if null_key is None else int(null_key),
                     len(pairs), 0)
        return spec, arr, [(a.terms[idx[2 * i]], b.terms[idx[2 * i + 1]]) for i in range(len(pairs))]

    @staticmethod
    def need(a, b, *, keys, groups_a=1, groups_b=1, pairs, null_key=None, backend="cpu"):
        """the needs a build checks ({resource: need}; cuda: with the device copies)"""
        spec, arr, _ = Join._spec(a, b, keys, groups_a, groups_b, pairs, null_key)
        out = _Need()
        _check(_call("join_need", a._handle, b._handle, C.byref(spec), arr.ctypes.data, 0 if backend == "cpu" else 1,
                     C.byref(out)), what="join needs")
        return _needs(out)

    @classmethod
    def build(cls, a, b, *, keys, groups_a=1, groups_b=1, pairs=(("1", "1"),), null_key=None, backend="cpu",
              budget=None):
        """the join of side states a and b (md.State, key-major groups: key * groups + g) on their keys"""
        a._open(); b._open()
        spec, arr, specs = cls._spec(a, b, keys, groups_a, groups_b, pairs, null_key)
        error, refusal = C.c_int32(), _Refusal()
        if backend == "cpu":
            h = _call("join_build", a._handle, b._handle, C.byref(spec), arr.ctypes.data, C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
            return cls._wrap(md._made(h, error, refusal, "join"), a, b, specs, "cpu", "native CPU")
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        job = _call("job_build", a._handle, b._handle, C.byref(spec), arr.ctypes.data, 1, C.byref(_budget(budget)),
                    C.byref(refusal), C.byref(error))
        md._made(job, error, refusal, "join on the device")
        return cls._from_job(job, a, b, specs, budget)

    @classmethod
    def _from_job(cls, job, a, b, specs, budget):
        gpu = _gpu()
        try:
            view = _CrossView()
            _check(_call("job_get_view", job, C.byref(view)), what="join job")
            nv = int(view.ngroups) * int(view.values)
            terms = np.zeros(max(3 * nv * WORDS, 1), dtype=np.uint64)
            refusal, error = _Refusal(), C.c_int32()
            gpu.run("tr_mgpu_bilin_cross", C.byref(view), terms.ctypes.data, C.byref(_budget(budget)),
                    C.byref(refusal), refusal=refusal, what="join cross terms on the device")
            h = _call("join_finish", job, terms.ctypes.data, C.byref(_budget(budget)), C.byref(refusal),
                      C.byref(error))
            backend, device = _device(gpu)
            return cls._wrap(md._made(h, error, refusal, "join from the device terms"), a, b, specs, backend, device)
        finally:
            _call("job_free", job)

    def _open(self):
        if not self._handle:
            raise ValueError("the join is closed")

    def _info(self):
        self._open()
        out = _Info()
        _check(_call("join_get_info", self._handle, C.byref(out)), what="join")
        return out

    @property
    def info(self):
        i = self._info()
        return dict(cells=int(i.ngroups), keys=int(i.nkeys), groups_a=int(i.groups_a), groups_b=int(i.groups_b),
                    null_key=None if i.null_key == NONE else int(i.null_key), step=int(i.step), bytes=int(i.bytes),
                    touched=int(i.touched), identity=_digest(i.identity), base=_digest(i.base),
                    side_a=_digest(i.side_a), side_b=_digest(i.side_b), pairs=int(i.npairs), values=int(i.values),
                    backend=self.backend, device=self.device)

    @property
    def identity(self):
        return _digest(self._info().identity)

    @property
    def step(self):
        return int(self._info().step)

    def update_need(self, a=None, b=None, backend="cpu", new_a=None, new_b=None):
        """the needs an update checks (the side deltas a / b and the new side states)"""
        out = _Need()
        _check(_call("update_need", self._handle, self.a._handle, a._handle if a else None,
                     new_a._handle if new_a else None, self.b._handle, b._handle if b else None,
                     new_b._handle if new_b else None, 0 if backend == "cpu" else 1, C.byref(out)),
               what="join update needs")
        return _needs(out)

    def apply(self, a=None, b=None, *, backend="cpu", budget=None, new_a=None, new_b=None):
        """the join of the updated sides: a / b are DELTA-001 deltas of side A / B (None: unchanged); each side is
        updated by DELTA-001 on the same backend (or new_a / new_b: an update computed already), then the join by
        its three cross terms d(AB) = dA B + A dB + dA dB of the touched keys"""
        self._open()
        if a is None and b is None:
            raise ValueError("give a delta of side A, of side B or of both")
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend is 'cpu' or 'cuda'")
        sa2 = (new_a if new_a is not None else self.a.apply(a, backend=backend, budget=budget)) if a is not None else None
        sb2 = (new_b if new_b is not None else self.b.apply(b, backend=backend, budget=budget)) if b is not None else None
        error, refusal = C.c_int32(), _Refusal()
        args = (self._handle, self.a._handle, a._handle if a else None, sa2._handle if sa2 else None, self.b._handle,
                b._handle if b else None, sb2._handle if sb2 else None)
        na, nb = sa2 if sa2 is not None else self.a, sb2 if sb2 is not None else self.b
        if backend == "cpu":
            h = _call("join_update", *args, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            return Join._wrap(md._made(h, error, refusal, "join update"), na, nb, self.pairs, "cpu", "native CPU")
        job = _call("job_update", *args, 1, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        md._made(job, error, refusal, "join update on the device")
        return Join._from_job(job, na, nb, self.pairs, budget)

    def _values(self):
        i = self._info()
        n = int(i.ngroups) * int(i.values)
        out = np.zeros(max(n * WORDS, 1), dtype=np.uint64)
        _check(_call("join_get_values", self._handle, out.ctypes.data, n), what="join values")
        return _wide_ints(out[:n * WORDS]), int(i.values)

    @staticmethod
    def _cell(vals, values, cell):
        row = vals[cell * values:(cell + 1) * values]
        return dict(rows=row[0], count=row[1::2], sum=row[2::2])

    def cell(self, g=0, h=0):
        """{"rows", "count": [per pair], "sum": [per pair]} of output cell (g, h), exact ints"""
        i = self._info()
        if not (0 <= int(g) < i.groups_a and 0 <= int(h) < i.groups_b):
            raise ValueError("output group out of range")
        vals, values = self._values()
        return self._cell(vals, values, int(g) * int(i.groups_b) + int(h))

    def cells(self):
        """every output cell, cell index g * groups_b + h"""
        vals, values = self._values()
        return [self._cell(vals, values, c) for c in range(len(vals) // values)]

    @property
    def terms(self):
        """the cross terms of the update that made this state: {"dA_B", "A_dB", "dA_dB"} (cells as in cells();
        a build: 0, 0 and the whole product)"""
        i = self._info()
        nv = int(i.ngroups) * int(i.values)
        out = np.zeros(max(3 * nv * WORDS, 1), dtype=np.uint64)
        _check(_call("join_get_terms", self._handle, out.ctypes.data, 3 * nv), what="join terms")
        allv = _wide_ints(out[:3 * nv * WORDS])
        res = {}
        for k, name in enumerate(("dA_B", "A_dB", "dA_dB")):
            part = allv[k * nv:(k + 1) * nv]
            res[name] = [self._cell(part, int(i.values), c) for c in range(int(i.ngroups))]
        return res

    def finalize(self, g, h, finalizer, pair=0, pair2=None, *, budget=None):
        """sum, count, avg (of a pair) or ratio (sum of pair / sum of pair2) of cell (g, h): an exact Fraction, or
        None (SQL NULL); another finalizer raises DomainError (a join holds per-pair sums only)"""
        i = self._info()
        out, null, refusal = C.c_void_p(), C.c_int32(), _Refusal()
        code = _call("join_finalize", self._handle, int(g) * int(i.groups_b) + int(h), _finalizer(finalizer),
                     int(pair), -1 if pair2 is None else int(pair2), C.byref(out), C.byref(null),
                     C.byref(_budget(budget)), C.byref(refusal))
        return _finalized(code, out, null, refusal, f"join {finalizer}")

    def summarizes(self, a, b):
        """True when a and b are exactly the side states this is the join of"""
        return bool(_check(_call("join_summarizes", self._handle, a._handle, b._handle), what="join sides"))

    def equal(self, other):
        return bool(_check(_call("join_equal", self._handle, other._handle), what="join comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("join_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Join(closed)"
        i = self.info
        return f"Join(cells={i['cells']}, keys={i['keys']}, pairs={i['pairs']}, step={i['step']}, {self.backend})"


# ---------------------------------------------------------------------- dense matrices
def _matrix(m, rows, cols, what):
    a = np.asarray(m)
    if a.dtype == object or a.dtype.kind not in "iu":
        vals = [int(v) for v in np.asarray(m, dtype=object).ravel()]
        if any(not -(1 << 63) <= v < (1 << 63) for v in vals):
            raise OverflowError(f"{what}: an entry does not fit int64")
        a = np.array(vals, dtype=np.int64).reshape(np.asarray(m, dtype=object).shape)
    elif a.dtype.kind == "u" and a.size and int(a.max()) >= (1 << 63):
        raise OverflowError(f"{what}: an entry does not fit int64")
    a = np.ascontiguousarray(a, dtype=np.int64)
    if a.shape != (rows, cols):
        raise ValueError(f"{what}: shape {a.shape}, expected ({rows}, {cols})")
    return a


def matrix_delta(A, B, dA=None, dB=None, *, backend="cpu", budget=None):
    """the exact products of int64 matrices A (G x K), B (K x H) and updates dA, dB (None = 0):
    {"AB", "dA_B", "A_dB", "dA_dB", "updated"} (G x H lists of ints; updated = AB + dA B + A dB + dA dB)"""
    a0 = np.asarray(A if not isinstance(A, list) else np.asarray(A, dtype=object))
    b0 = np.asarray(B if not isinstance(B, list) else np.asarray(B, dtype=object))
    if a0.ndim != 2 or b0.ndim != 2 or a0.shape[1] != b0.shape[0]:
        raise ValueError("A is G x K and B is K x H")
    G, K = a0.shape
    H = b0.shape[1]
    a, b = _matrix(A, G, K, "A"), _matrix(B, K, H, "B")
    da = _matrix(dA, G, K, "dA") if dA is not None else None
    db = _matrix(dB, K, H, "dB") if dB is not None else None
    out = np.zeros(max(5 * G * H * WORDS, 1), dtype=np.uint64)
    refusal = _Refusal()
    args = (G, K, H, a.ctypes.data, b.ctypes.data, da.ctypes.data if da is not None else None,
            db.ctypes.data if db is not None else None, out.ctypes.data, C.byref(_budget(budget)), C.byref(refusal))
    if backend == "cpu":
        _check(_call("matrix_delta", *args), refusal, "matrix delta")
        where = ("cpu", "native CPU")
    elif backend == "cuda":
        gpu = _gpu()
        gpu.run("tr_mgpu_bilin_matrix", *args, refusal=refusal, what="matrix delta on the device")
        where = _device(gpu)
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    vals = _wide_ints(out[:5 * G * H * WORDS])
    res = {}
    for k, name in enumerate(("AB", "dA_B", "A_dB", "dA_dB", "updated")):
        part = vals[k * G * H:(k + 1) * G * H]
        res[name] = [part[g * H:(g + 1) * H] for g in range(G)]
    res["backend"], res["device"] = where
    return res


def matrix_need(G, K, H, *, with_da=True, with_db=True, backend="cpu"):
    out = _Need()
    _check(_call("matrix_need", int(G), int(K), int(H), int(bool(with_da)), int(bool(with_db)),
                 0 if backend == "cpu" else 1, C.byref(out)), what="matrix needs")
    return _needs(out)


# ---------------------------------------------------------------------- finalizers of linear states
def finalize(state, group, finalizer, term, term2=None, *, budget=None):
    """a finalizer of a DELTA-001 state's group from its cells: sum, count, avg (of a term), ratio (sum of term / sum
    of term2), var_pop / var_samp (term = t, term2 = t^2). An exact Fraction in logical units, or None (SQL NULL).
    count_distinct, sum_distinct, min, max (and join_sum) need their own retained state: DomainError."""
    state._open()
    t0 = _term_index(state, term, "term")
    t1 = _term_index(state, term2, "term2") if term2 is not None else -1
    out, null, refusal = C.c_void_p(), C.c_int32(), _Refusal()
    code = _call("finalize_state", state._handle, int(group), _finalizer(finalizer), t0, t1, C.byref(out),
                 C.byref(null), C.byref(_budget(budget)), C.byref(refusal))
    return _finalized(code, out, null, refusal, f"{finalizer} of linear cells")


# ---------------------------------------------------------------------- value states
class RebuildRequired(DomainError):
    """An EXTREMUM state cannot follow an update (the last copy of its minimum or maximum was deleted while values it
    does not retain remain): rebuild it from the data. .group, .value (the deleted extreme), .have, .change."""


def _values_made(handle, error, refusal, what, v):
    if error.value == -6 and v.group != NONE:
        reason = REASONS[v.reason]
        value = int(v.value) if v.has_value else None
        cls = RebuildRequired if v.reason == 5 else md.NegativeMultiplicityError
        where = f"value {value}, " if value is not None else ""
        err = cls(f"{what}: group {v.group}: {reason} ({where}have {v.have}, change {v.change})")
        err.group, err.reason, err.value, err.term = int(v.group), reason, value, None
        err.have, err.change, err.result = int(v.have), int(v.change), int(v.have) + int(v.change)
        err.need = -int(v.change) if v.change < 0 else None
        raise err
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


class Values:
    """An immutable typed retained state of one value column per group (native): "multiset" (per-value
    multiplicities: exact COUNT DISTINCT, SUM DISTINCT, MIN, MAX under every update) or "extremum" (the minimum and
    maximum with their multiplicities: MIN / MAX, refusing with RebuildRequired when the last copy of an extreme is
    deleted and the next one is not retained). Updated by DELTA-001 deltas."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Values.from_scope(...) or Values.apply(...)")

    @classmethod
    def _wrap(cls, handle, names, column, kind):
        self = object.__new__(cls)
        self._handle, self.names, self.column, self.kind = handle, tuple(names), column, kind
        return self

    @staticmethod
    def need(scope, column, kind="multiset"):
        if kind not in KINDS:
            raise ValueError("kind is 'multiset' or 'extremum'")
        out = _Need()
        _check(_call("values_need_scope", scope._handle, list(scope.names).index(column), KINDS[kind], C.byref(out)),
               what="value state needs")
        return _needs(out)

    @classmethod
    def from_scope(cls, scope, column, kind="multiset", *, budget=None):
        """the values of `column` per group of a FND-004 scope (selected rows; NULL values counted apart)"""
        if kind not in KINDS:
            raise ValueError("kind is 'multiset' or 'extremum'")
        if scope.source == "table":
            scope.require_fresh("value state")
        if column not in scope.names:
            raise ValueError(f"{column!r} is not a column of the scope")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("values_from_scope", scope._handle, list(scope.names).index(column), KINDS[kind],
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(md._made(h, error, refusal, "value state"), scope.names, column, kind)

    def _open(self):
        if not self._handle:
            raise ValueError("the value state is closed")

    def apply_need(self, delta):
        out = _Need()
        _check(_call("values_need_apply", self._handle, delta._handle, C.byref(out)), what="value update needs")
        return _needs(out)

    def apply(self, delta, *, budget=None):
        """the state updated by a DELTA-001 delta (same column count, scale and groups): NegativeMultiplicityError
        for a value, row or NULL that is not there; RebuildRequired (EXTREMUM) when the next extreme is unknown"""
        self._open(); delta._open()
        error, refusal, v = C.c_int32(), _Refusal(), _Violation()
        h = _call("values_apply", self._handle, delta._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(v), C.byref(error))
        return Values._wrap(_values_made(h, error, refusal, "value update", v), self.names, self.column, self.kind)

    @property
    def info(self):
        self._open()
        out = _ValuesInfo()
        _check(_call("values_get_info", self._handle, C.byref(out)), what="value state")
        return dict(groups=int(out.ngroups), entries=int(out.entries), step=int(out.step), bytes=int(out.bytes),
                    identity=_digest(out.identity), base=_digest(out.base), kind=self.kind, column=self.column)

    def group(self, g):
        """rows, nulls, present (with multiplicity), distinct (multiset), min, max and their multiplicities (None
        without present values), sum_distinct (multiset)"""
        self._open()
        out = _GroupValues()
        _check(_call("values_group", self._handle, int(g), C.byref(out)), what="value group")
        has = out.present > 0
        res = dict(rows=int(out.rows), nulls=int(out.nulls), present=int(out.present),
                   min=int(out.min) if has else None, max=int(out.max) if has else None,
                   min_count=int(out.min_count) if has else 0, max_count=int(out.max_count) if has else 0)
        if self.kind == "multiset":
            res["distinct"] = int(out.distinct)
            res["sum_distinct"] = (int(out.sum_distinct[1]) << 64) | (int(out.sum_distinct[0]) & ((1 << 64) - 1))
        return res

    def entries(self, g):
        """multiset: [(value, multiplicity)] of group g in increasing order"""
        self._open()
        count = C.c_uint64()
        _check(_call("values_entries", self._handle, int(g), None, None, 0, C.byref(count)), what="value entries")
        n = int(count.value)
        vals, mult = np.zeros(max(n, 1), np.int64), np.zeros(max(n, 1), np.int64)
        if n:
            _check(_call("values_entries", self._handle, int(g), vals.ctypes.data, mult.ctypes.data, n,
                         C.byref(count)), what="value entries")
        return list(zip(vals[:n].tolist(), mult[:n].tolist()))

    def finalize(self, g, finalizer, *, budget=None):
        """count, count_distinct, sum_distinct, min or max of group g: an exact int (as a Fraction's numerator) or
        None (SQL NULL); count_distinct / sum_distinct of an extremum state, or any linear finalizer: DomainError"""
        self._open()
        out, null, refusal = C.c_void_p(), C.c_int32(), _Refusal()
        code = _call("values_finalize", self._handle, int(g), _finalizer(finalizer), C.byref(out), C.byref(null),
                     C.byref(_budget(budget)), C.byref(refusal))
        r = _finalized(code, out, null, refusal, f"{finalizer} of a value state")
        return None if r is None else int(r)

    def equal(self, other):
        return bool(_check(_call("values_equal", self._handle, other._handle), what="value comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("values_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Values(closed)"
        i = self.info
        return f"Values({self.kind}, {self.column!r}, groups={i['groups']}, entries={i['entries']}, step={i['step']})"
