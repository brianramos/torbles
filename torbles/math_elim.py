# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEOM-002: bounded elimination and generator selection, natively on the CPU or with the CUDA kernel
(torbles/math_elim.h/.c, math_elim_gpu.cu).

    from torbles import math_bank as mbk, math_certificates as mc, math_geom as mg, math_elim as me
    sel = me.select(scope, relations)                  # the smallest generator set (bounded search by size)
    sel.generators, sel.eliminated                     # ["x", "y"], {"z": Polynomial(x*y), "s": Polynomial(x + y)}
    sel.columns; sel.candidates; sel.info              # per variable: role, read path, costs, its row check
    sel.certificate("z")                               # FND-007 derived certificate D (z - f) = sum D h_k r_k
    sel.check(backend="cuda")                          # every reconstruction checked on every row: values, NULLs, range
    me.select(scope, relations, mode="order", order=["z", "s", "x", "y"])     # one elimination order
    f = me.forms(scope, sel, mg.poly({...}, names))     # local elimination forms of a query expression
    f.chosen, f.best, f.candidates                      # the direct form is kept whenever expansion loses
    me.check(scope, [("z", f_z), (q, p)], backend="cuda")   # pairs of a target and a form
    me.cost(scope, p)                                   # scan, work and actual bit widths on the rows

A reconstruction is kept only when its row check holds on the scope (values, NULL mask and range: checked, never
assumed from the algebra). Nothing is computed in Python (marshalling and display only). Opt-in: `import torbles`
does not load it.
"""
import ctypes as C

import numpy as np

from . import algq
from . import math_bank as _mbk
from . import math_certificates as _mc
from . import math_geom as _mg
from . import math_ident as _mi
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Selection", "Forms", "select", "forms", "check", "check_need", "cost", "choose", "MODES", "PRIORITIES",
           "MAX_VARS", "MAX_RELATIONS", "MAX_PAIRS"]

MAX_VARS, MAX_RELATIONS, MAX_PAIRS, MAX_FORMS = 8, 16, 256, 257
MODES = {"search": 1, "order": 2}
PRIORITIES = {"total": 1, "scan": 2, "work": 3, "bits": 4}
ROLES = {1: "generator", 2: "eliminated"}
READS = {1: "direct", 2: "reconstruct"}
STATUSES = {1: "chosen", 2: "admissible", 3: "rejected"}
REASONS = {0: None, 1: "cheapest", 2: "costlier", 3: "not reconstructible", 4: "null mismatch", 5: "value mismatch",
           6: "range", 7: "budget", 8: "unclaimable", 9: "undecided", 10: "direct"}
KINDS = {1: "direct", 2: "local", 3: "normal"}
_NEED_CPU, _NEED_DEVICE = 0, 1
_NONE = (1 << 64) - 1
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Spec(C.Structure):
    _native_ = "tr_elim_spec"
    _fields_ = [("mode", C.c_int32), ("nvars", C.c_int32), ("col", C.c_int32 * 8), ("norder", C.c_int32),
                ("order", C.c_int32 * 8), ("max_size", C.c_int32), ("rank_all", C.c_int32), ("priority", C.c_int32),
                ("reserved", C.c_int32 * 2)]


class _Cost(C.Structure):
    _native_ = "tr_elim_cost"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "columns", "scan_bits", "terms", "multiplications", "degree",
                                          "bound_bits", "actual_bits", "words", "scan", "work", "total")] + [
        ("denominator", C.c_int64), ("measured", C.c_int32), ("reserved", C.c_int32)]


class _Check(C.Structure):
    _native_ = "tr_elim_check"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "compared", "nulls", "null_lost", "null_extra", "values",
                                          "below", "above", "at_lo", "at_hi", "undecided", "first", "first_null",
                                          "first_value", "first_range")] + [
        ("bits_target", C.c_uint32), ("bits_form", C.c_uint32), ("ranged", C.c_int32), ("holds", C.c_int32)]


class _PairView(C.Structure):
    _native_ = "tr_elim_pair_view"
    _fields_ = [("npairs", C.c_int32), ("reserved", C.c_int32), ("mag", C.c_uint64 * 8), ("cand", _mi._CandView),
                ("target", C.c_void_p), ("den", C.c_void_p)]


class _Candidate(C.Structure):
    _native_ = "tr_elim_candidate"
    _fields_ = [("generators", C.c_uint32), ("eliminated", C.c_uint32)] + [
        (n, C.c_int32) for n in ("size", "status", "reason", "solve", "failed", "reserved")] + [
        (n, C.c_uint64) for n in ("stored_bits", "storage", "recon_scan", "recon_work", "total", "terms", "degree",
                                  "bits")] + [("refusal", _Refusal)]


class _Column(C.Structure):
    _native_ = "tr_elim_column"
    _fields_ = [(n, C.c_int32) for n in ("column", "role", "read", "claimable")] + [
        ("support", C.c_uint32), ("reserved", C.c_uint32), ("width", C.c_uint64), ("direct", C.c_uint64),
        ("reconstruction", _Cost), ("check", _Check)]


class _Info(C.Structure):
    _native_ = "tr_elim_info"
    _fields_ = [(n, C.c_int32) for n in ("nvars", "mode", "ngenerators", "neliminated")] + [
        ("col", C.c_int32 * 8), ("generators", C.c_uint32), ("eliminated", C.c_uint32)] + [
        (n, C.c_int32) for n in ("minimum", "minimal", "undecided", "undecided_size", "ncandidates", "chosen",
                                 "nrelations", "priority")] + [
        (n, C.c_uint64) for n in ("solves", "max_degree", "max_terms", "rows", "direct_storage", "bytes")]


class _FormSpec(C.Structure):
    _native_ = "tr_elim_form_spec"
    _fields_ = [(n, C.c_int32) for n in ("max_eliminate", "normal_form", "priority", "reserved")]


class _Form(C.Structure):
    _native_ = "tr_elim_form"
    _fields_ = [(n, C.c_int32) for n in ("kind", "status", "reason", "depth")] + [
        ("eliminated", C.c_uint32), ("columns", C.c_uint32), ("claimable", C.c_int32), ("reserved", C.c_int32),
        ("cost", _Cost), ("check", _Check), ("refusal", _Refusal)]


class _FormsInfo(C.Structure):
    _native_ = "tr_elim_forms_info"
    _fields_ = [(n, C.c_int32) for n in ("ncandidates", "chosen", "nvars", "priority")] + [
        ("reads", C.c_uint32), ("reserved", C.c_uint32)] + [(n, C.c_uint64) for n in ("solves", "rows", "bytes")]


for _s, _n in ((_Spec, 96), (_Cost, 112), (_Check, 144), (_PairView, 176), (_Candidate, 120), (_Column, 296),
               (_Info, 136), (_FormSpec, 16), (_Form, 312), (_FormsInfo, 48)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_elim_select": (vp, [vp, vp, i32, P(_Spec), B, R, e]),
            "tr_elim_selection_get_info": (i32, [vp, P(_Info)]),
            "tr_elim_selection_candidates": (i32, [vp, P(_Candidate), i32]),
            "tr_elim_selection_column": (i32, [vp, i32, P(_Column)]),
            "tr_elim_selection_form": (vp, [vp, i32, e]),
            "tr_elim_selection_certify": (vp, [vp, i32, B, R, e]),
            "tr_elim_selection_pairs": (vp, [vp, vp, B, R, e]),
            "tr_elim_selection_free": (None, [vp]),
            "tr_elim_pairs_make": (vp, [vp, P(_mg._GPoly), P(_mg._GPoly), vp, i32, B, R, e]),
            "tr_elim_pairs_get_view": (i32, [vp, P(_PairView)]),
            "tr_elim_pairs_bound_bits": (u64, [vp, i32]),
            "tr_elim_pairs_free": (None, [vp]),
            "tr_elim_check_need": (i32, [P(_mbk._View), P(_PairView), i32, P(_Need)]),
            "tr_elim_check_view": (i32, [P(_mbk._View), P(_PairView), B, P(_Check), R]),
            "tr_elim_cost_of": (i32, [vp, P(_mg._GPoly), B, P(_Cost), R]),
            "tr_elim_choose": (i32, [P(_Cost), i32, i32, P(i32)]),
            "tr_elim_forms_make": (vp, [vp, vp, P(_mg._GPoly), P(_FormSpec), B, R, e]),
            "tr_elim_forms_get_info": (i32, [vp, P(_FormsInfo)]),
            "tr_elim_forms_candidate": (i32, [vp, i32, P(_Form)]),
            "tr_elim_forms_poly": (vp, [vp, i32, e]),
            "tr_elim_forms_certify": (vp, [vp, i32, B, R, e]),
            "tr_elim_forms_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_elim_check", C.c_int, [P(_mbk._View), P(_PairView), P(_Budget), P(_Check), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    if not handle and error.value == -6:
        raise DomainError(f"{what}: the relations or pairs do not apply here (GEOM-001 refused the relations, or the "
                          "view's extremes exceed the magnitudes the bounds were proven for)")
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _device_label():
    from . import math_gpu
    d = math_gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


def _fresh(scope, what):
    if getattr(scope, "source", None) == "table":
        scope.require_fresh(what)


def _needs_dict(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _row(x):
    return None if x == _NONE else int(x)


def _cost_dict(c):
    return dict(rows=int(c.rows), columns=int(c.columns), scan_bits=int(c.scan_bits), terms=int(c.terms),
                multiplications=int(c.multiplications), degree=int(c.degree), bound_bits=int(c.bound_bits),
                actual_bits=int(c.actual_bits) if c.measured else None, words=int(c.words), scan=int(c.scan),
                work=int(c.work), total=int(c.total), denominator=int(c.denominator), measured=bool(c.measured))


def _check_dict(k, label="cpu", device="native CPU"):
    return dict(rows=int(k.rows), selected=int(k.selected), compared=int(k.compared), nulls=int(k.nulls),
                null_lost=int(k.null_lost), null_extra=int(k.null_extra), values=int(k.values), below=int(k.below),
                above=int(k.above), at_lo=int(k.at_lo), at_hi=int(k.at_hi), undecided=int(k.undecided),
                first=_row(k.first), first_null=_row(k.first_null), first_value=_row(k.first_value),
                first_range=_row(k.first_range), bits_target=int(k.bits_target), bits_form=int(k.bits_form),
                ranged=bool(k.ranged), holds=bool(k.holds), backend=label, device=device)


def _refusal_dict(r):
    return dict(resource=RESOURCES[r.resource], need=int(r.need), limit=int(r.limit)) if r.refused else None


def _mask_names(mask, names):
    return [n for i, n in enumerate(names) if (mask >> i) & 1]


def _relations(relations, scope):
    relations = list(relations)
    if not relations:
        raise ValueError("at least one relation")
    if len(relations) > MAX_RELATIONS:
        raise UnsupportedError(f"at most {MAX_RELATIONS} relations")
    for r in relations:
        if not isinstance(r, _mc.Certificate):
            raise TypeError("relations are math_certificates.Certificate objects")
        if r.names is not None and _mc._KIND_NAMES.get(r._info().kind) != "law" and tuple(r.names) != tuple(scope.names):
            raise DomainError("the relations are certificates of scopes with other columns "
                              f"({', '.join(r.names)} against {', '.join(scope.names)})")
    return relations


def _index(scope, name):
    if name not in scope.names:
        raise ValueError(f"{name!r} is not a column of the scope ({', '.join(scope.names)})")
    return list(scope.names).index(name)


# ------------------------------------------------------------------ pairs and checks
def _target(t, scope):
    """a target: a column name (a range pair) or a polynomial"""
    if isinstance(t, str):
        i = _index(scope, t)
        return _mg.poly({(1,): 1}, [t]), i
    return _mg._as_poly(t, "a target"), -1


def _pairs(scope, pairs, budget):
    pairs = list(pairs)
    if len(pairs) > MAX_PAIRS:
        raise UnsupportedError(f"at most {MAX_PAIRS} pairs per batch")
    n = len(pairs)
    T, F = (_mg._GPoly * max(n, 1))(), (_mg._GPoly * max(n, 1))()
    ranged = np.full(max(n, 1), -1, dtype=np.int32)
    keep = []
    for j, (t, f) in enumerate(pairs):
        tp, ranged[j] = _target(t, scope)
        T[j], k1 = tp.native(scope.names)
        F[j], k2 = _mg._as_poly(f, "a form").native(scope.names)
        keep += k1 + k2
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_elim_pairs_make", scope._handle, T, F, ranged.ctypes.data, n, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return _made(h, error, refusal, "check pairs"), n


def _run_checks(scope, h, n, backend, budget):
    pv = _PairView()
    _check(_call("tr_elim_pairs_get_view", h, C.byref(pv)), what="check pairs")
    view = scope.view()
    out = (_Check * max(n, 1))()
    refusal = _Refusal()
    if backend == "cpu":
        rc = _call("tr_elim_check_view", C.byref(view), C.byref(pv), C.byref(_budget(budget)), out, C.byref(refusal))
        if rc == -6:
            raise DomainError("the scope's extremes exceed the magnitudes the pairs were proven for")
        _check(rc, refusal, "reconstruction check")
        label, device = "cpu", "native CPU"
    elif backend == "cuda":
        g = _gpu()
        import os
        if os.environ.get("TORBLES_MATH_REQUIRE_DEVICE") == "1":
            g.require_device()
        rc = g.lib().tr_mgpu_elim_check(C.byref(view), C.byref(pv), C.byref(_budget(budget)), out, C.byref(refusal))
        if rc == -6:
            raise DomainError("the scope's extremes exceed the magnitudes the pairs were proven for")
        g.check(rc, refusal, "reconstruction check on the device")
        label, device = _device_label()
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    return [_check_dict(out[j], label, device) for j in range(n)]


def check(scope, pairs, *, backend="cpu", budget=None):
    """for each (target, form): the target (a column name: also its range, or a Polynomial) and the form evaluated
    exactly on every selected row: compared, nulls (both NULL), null_lost / null_extra (one NULL), values (different),
    below / above / at_lo / at_hi (a column target's range), undecided, the first mismatching rows, the largest bit
    lengths, holds"""
    _fresh(scope, "reconstruction check")
    h, n = _pairs(scope, pairs, budget)
    try:
        return _run_checks(scope, h, n, backend, budget)
    finally:
        _call("tr_elim_pairs_free", h)


def check_need(scope, pairs, *, backend="cpu", budget=None):
    """the needs of a check batch: {resource: need}"""
    h, n = _pairs(scope, pairs, budget)
    try:
        pv, need = _PairView(), _Need()
        _call("tr_elim_pairs_get_view", h, C.byref(pv))
        _check(_call("tr_elim_check_need", C.byref(scope.view()), C.byref(pv),
                     _NEED_DEVICE if backend == "cuda" else _NEED_CPU, C.byref(need)), what="needs")
        return _needs_dict(need)
    finally:
        _call("tr_elim_pairs_free", h)


def cost(scope, f, *, budget=None):
    """the cost of a polynomial on the scope, measured on every row: columns, scan_bits, terms, multiplications, degree,
    bound_bits, actual_bits, words, scan, work, total (relative work units, not time)"""
    _fresh(scope, "cost")
    g, keep = _mg._as_poly(f).native(scope.names)
    out, refusal = _Cost(), _Refusal()
    _check(_call("tr_elim_cost_of", scope._handle, C.byref(g), C.byref(_budget(budget)), C.byref(out), C.byref(refusal)),
           refusal, "cost")
    return _cost_dict(out)


def choose(scope, candidates, *, priority="total", budget=None):
    """(the index of the cheapest candidate polynomial, [costs]): priority total / scan / work / bits first, ties to the
    earliest (candidate 0 is kept unless another is strictly cheaper)"""
    if priority not in PRIORITIES:
        raise ValueError(f"priority is one of {', '.join(PRIORITIES)}")
    candidates = [_mg._as_poly(f) for f in candidates]
    if not candidates:
        raise ValueError("at least one candidate")
    raw = (_Cost * len(candidates))()
    for i, f in enumerate(candidates):
        g, keep = f.native(scope.names)
        refusal = _Refusal()
        _check(_call("tr_elim_cost_of", scope._handle, C.byref(g), C.byref(_budget(budget)), C.byref(raw[i]),
                     C.byref(refusal)), refusal, "cost")
    chosen = C.c_int32()
    _check(_call("tr_elim_choose", raw, len(candidates), PRIORITIES[priority], C.byref(chosen)), what="choose")
    return int(chosen.value), [_cost_dict(raw[i]) for i in range(len(candidates))]


# ------------------------------------------------------------------ generator selection
class Selection:
    """A generator set with the exact reconstruction of every eliminated variable (checked on the scope's rows)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_elim.select(scope, relations, ...)")

    @classmethod
    def _wrap(cls, handle, scope, relations):
        self = object.__new__(cls)
        self._handle, self.scope, self.relations, self.names = handle, scope, tuple(relations), tuple(scope.names)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the selection is closed")
        return self._handle

    def _raw_info(self):
        i = _Info()
        _check(_call("tr_elim_selection_get_info", self._open(), C.byref(i)), what="selection")
        return i

    @property
    def variables(self):
        i = self._raw_info()
        return [self.names[i.col[k]] for k in range(i.nvars)]

    @property
    def info(self):
        i = self._raw_info()
        var = [self.names[i.col[k]] for k in range(i.nvars)]
        return dict(variables=var, mode={1: "search", 2: "order"}[i.mode], generators=_mask_names(i.generators, var),
                    eliminated=_mask_names(i.eliminated, var), minimum=int(i.minimum), minimal=bool(i.minimal),
                    undecided=int(i.undecided), undecided_size=None if i.undecided_size < 0 else int(i.undecided_size),
                    candidates=int(i.ncandidates), chosen=int(i.chosen), relations=int(i.nrelations),
                    priority={v: k for k, v in PRIORITIES.items()}[i.priority], solves=int(i.solves),
                    max_degree=int(i.max_degree), max_terms=int(i.max_terms), rows=int(i.rows),
                    direct_storage=int(i.direct_storage), bytes=int(i.bytes))

    @property
    def generators(self):
        return self.info["generators"]

    @property
    def eliminated(self):
        """{variable: its reconstruction (a math_geom.Polynomial over the generators)}"""
        return {v: self.form(v) for v in self.info["eliminated"]}

    def _var(self, name):
        var = self.variables
        if name not in var:
            raise ValueError(f"{name!r} is not a variable of the selection ({', '.join(var)})")
        return var.index(name)

    def form(self, name):
        """the reconstruction of an eliminated variable"""
        error = C.c_int32()
        h = _call("tr_elim_selection_form", self._open(), self._var(name), C.byref(error))
        if not h and error.value == -1:
            raise ValueError(f"{name!r} is a generator (not reconstructed)")
        return _mg.Polynomial._native(_made(h, error, None, "reconstruction"), self.names)

    def certificate(self, name, *, budget=None):
        """the FND-007 derived certificate of D (x - f) = sum D h_k r_k (GEOM-001's reduction certificate): its kind
        the weakest relation's, valid only where the relations are"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_elim_selection_certify", self._open(), self._var(name), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        if not h and error.value == -3:
            raise OverflowError("the certificate's integer coefficients do not fit int64 (no FND-007 claim)")
        return _mc.Certificate._wrap(_made(h, error, refusal, "reconstruction certificate"), self.names)

    @property
    def columns(self):
        """per variable: name, role, read path, claimable, width, direct (scanning the stored column), support (the
        generators its reconstruction reads), reconstruction (cost), check (its row check on the selection's scope)"""
        out = []
        var = self.variables
        for i, name in enumerate(var):
            k = _Column()
            _check(_call("tr_elim_selection_column", self._open(), i, C.byref(k)), what="selection column")
            elim = k.role == 2
            out.append(dict(name=name, role=ROLES[k.role], read=READS[k.read], claimable=bool(k.claimable),
                            width=int(k.width), direct=int(k.direct), support=_mask_names(k.support, var),
                            reconstruction=_cost_dict(k.reconstruction) if elim else None,
                            check=_check_dict(k.check) if elim else None))
        return out

    @property
    def candidates(self):
        """every generator set examined: generators, eliminated, size, status, reason, solve, failed (the first variable
        that failed), stored_bits, storage, recon_scan, recon_work, total, terms, degree, bits, refusal (an undecided
        set refused by the budget: resource, need, limit)"""
        n = self._raw_info().ncandidates
        arr = (_Candidate * max(n, 1))()
        _check(_call("tr_elim_selection_candidates", self._open(), arr, n), what="candidates")
        var = self.variables
        return [dict(generators=_mask_names(c.generators, var), eliminated=_mask_names(c.eliminated, var),
                     size=int(c.size), status=STATUSES.get(c.status, "rejected"), reason=REASONS[c.reason],
                     solve=None if c.solve < 0 else int(c.solve), failed=None if c.failed < 0 else var[c.failed],
                     stored_bits=int(c.stored_bits), storage=int(c.storage), recon_scan=int(c.recon_scan),
                     recon_work=int(c.recon_work), total=int(c.total), terms=int(c.terms), degree=int(c.degree),
                     bits=int(c.bits), refusal=_refusal_dict(c.refusal)) for c in arr[:n]]

    def check(self, scope=None, *, backend="cpu", budget=None):
        """{eliminated variable: its check}: the reconstruction evaluated on every selected row of the scope (default:
        the selection's) against the stored column: values, NULL mask and range"""
        scope = scope or self.scope
        _fresh(scope, "reconstruction check")
        if tuple(scope.names) != self.names:
            raise DomainError("the scope's columns differ from the selection's")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_elim_selection_pairs", self._open(), scope._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        h = _made(h, error, refusal, "reconstruction pairs")
        try:
            elim = self.info["eliminated"]
            res = _run_checks(scope, h, len(elim), backend, budget)
            return dict(zip(elim, res))
        finally:
            _call("tr_elim_pairs_free", h)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_elim_selection_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Selection(closed)"
        i = self.info
        return f"Selection(generators={i['generators']}, eliminated={i['eliminated']}, solves={i['solves']})"


def select(scope, relations, *, columns=None, mode="search", order=None, max_size=None, rank_all=True,
           priority="total", budget=None):
    """the generator set of the relations' variables (or `columns`): mode "search" (the minimum size by a bounded search
    over candidate sets, every set of that size costed with rank_all) or "order" (one elimination under the lex
    sequence `order`); priority total / scan / work / bits ranks the sets of the minimum size. Budget: iterations =
    solves (refused past them, never truncated); degree and support bound every completion and reduction: a search set
    past them is undecided (rejected with reason "budget" and its refusal; "unclaimable" past int64) and the search goes
    on, info["minimal"] only when every smaller set was decided; an "order" selection past them is refused."""
    _fresh(scope, "generator selection")
    relations = _relations(relations, scope)
    if mode not in MODES:
        raise ValueError(f"mode is one of {', '.join(MODES)}")
    if priority not in PRIORITIES:
        raise ValueError(f"priority is one of {', '.join(PRIORITIES)}")
    sp = _Spec()
    sp.mode, sp.priority = MODES[mode], PRIORITIES[priority]
    cols = list(columns or ())
    order = list(order or ())
    if len(cols) > MAX_VARS or len(order) > MAX_VARS:
        raise UnsupportedError(f"at most {MAX_VARS} variables")
    for i in range(8):
        sp.col[i] = sp.order[i] = -1
    for i, c in enumerate(cols):
        sp.col[i] = _index(scope, c)
    for i, c in enumerate(order):
        sp.order[i] = _index(scope, c)
    sp.nvars, sp.norder = len(cols), len(order)
    sp.max_size = -1 if max_size is None else int(max_size)
    sp.rank_all = 1 if rank_all else 0
    ptrs = (C.c_void_p * len(relations))(*[r._open() for r in relations])
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_elim_select", scope._handle, ptrs, len(relations), C.byref(sp), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Selection._wrap(_made(h, error, refusal, "generator selection"), scope, relations)


# ------------------------------------------------------------------ local elimination forms
class Forms:
    """The direct expression and its local elimination forms, each certified, checked on every row and costed."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_elim.forms(scope, selection, q)")

    @classmethod
    def _wrap(cls, handle, selection, q, names):
        self = object.__new__(cls)
        self._handle, self.selection, self.q, self.names = handle, selection, q, tuple(names)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the forms are closed")
        return self._handle

    @property
    def info(self):
        i = _FormsInfo()
        _check(_call("tr_elim_forms_get_info", self._open(), C.byref(i)), what="forms")
        var = self.selection.variables
        return dict(candidates=int(i.ncandidates), chosen=int(i.chosen), reads=_mask_names(i.reads, var),
                    priority={v: k for k, v in PRIORITIES.items()}[i.priority], solves=int(i.solves), rows=int(i.rows),
                    bytes=int(i.bytes))

    def polynomial(self, i):
        """candidate i's polynomial (0: the direct expression)"""
        if i == 0:
            return self.q
        error = C.c_int32()
        h = _call("tr_elim_forms_poly", self._open(), int(i), C.byref(error))
        if not h and error.value == -1:
            raise ValueError(f"candidate {i} was rejected before its form was built")
        return _mg.Polynomial._native(_made(h, error, None, "form"), self.names)

    def certificate(self, i, *, budget=None):
        """the FND-007 derived certificate of D (q - p) for candidate i >= 1 (local forms: through the reconstruction
        certificates; the normal form: from the relations)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_elim_forms_certify", self._open(), int(i), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        if not h and error.value == -3:
            raise OverflowError("the certificate's integer coefficients do not fit int64 (no FND-007 claim)")
        return _mc.Certificate._wrap(_made(h, error, refusal, "form certificate"), self.names)

    @property
    def candidates(self):
        """every candidate: index, kind (direct / local / normal), status, reason, eliminated, columns, claimable, depth,
        cost, check (against the direct expression), refusal (a budget rejection), polynomial"""
        n = self.info["candidates"]
        var = self.selection.variables
        out = []
        for i in range(n):
            f = _Form()
            _check(_call("tr_elim_forms_candidate", self._open(), i, C.byref(f)), what="form")
            built = i == 0 or f.reason != 7
            refusal = _refusal_dict(f.refusal)
            poly = None
            if built:
                try:
                    poly = self.polynomial(i)
                except ValueError:
                    poly = None
            measured = f.cost.measured
            out.append(dict(index=i, kind=KINDS[f.kind], status=STATUSES.get(f.status, "rejected"),
                            reason=REASONS[f.reason], eliminated=_mask_names(f.eliminated, var),
                            columns=_mask_names(f.columns, self.names), claimable=bool(f.claimable) if i else None,
                            depth=int(f.depth), cost=_cost_dict(f.cost) if measured else None,
                            check=_check_dict(f.check) if measured else None, refusal=refusal, polynomial=poly))
        return out

    @property
    def chosen(self):
        return self.info["chosen"]

    @property
    def best(self):
        return self.candidates[self.chosen]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_elim_forms_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Forms(closed)"
        i = self.info
        return f"Forms({i['candidates']} candidates, chosen {i['chosen']}, solves={i['solves']})"


def forms(scope, selection, q, *, max_eliminate=None, normal_form=True, priority="total", budget=None):
    """the direct expression q, its local elimination forms (q with every nonempty set of at most max_eliminate of the
    eliminated variables it reads substituted by their reconstructions) and the normal form under the selection's
    elimination ideal; each checked on every selected row against q and costed; the cheapest admissible chosen by the
    priority, ties to the earliest (the direct form). Budget: iterations = the alternatives (solves), degree and
    support reject an expansion past them (the direct form is kept)."""
    _fresh(scope, "elimination forms")
    if not isinstance(selection, Selection):
        raise TypeError("a math_elim.Selection")
    if tuple(scope.names) != selection.names:
        raise DomainError("the scope's columns differ from the selection's")
    if priority not in PRIORITIES:
        raise ValueError(f"priority is one of {', '.join(PRIORITIES)}")
    q = _mg._as_poly(q)
    g, keep = q.native(scope.names)
    fs = _FormSpec()
    fs.max_eliminate = -1 if max_eliminate is None else int(max_eliminate)
    fs.normal_form = 1 if normal_form else 0
    fs.priority = PRIORITIES[priority]
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_elim_forms_make", scope._handle, selection._open(), C.byref(g), C.byref(fs), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Forms._wrap(_made(h, error, refusal, "elimination forms"), selection, q, scope.names)
