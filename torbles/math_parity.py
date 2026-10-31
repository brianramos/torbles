# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FIELD-002: parity relationship discovery, certification and substitution, natively on the CPU or with the CUDA
kernels (torbles/math_parity.c, math_parity_gpu.cu; FIELD-001's discovery, FND-007's certificates).

    from torbles import math_bank as mbk, math_bool as mb, math_parity as mp
    scope = mbk.Scope.from_table(db["t"], ["a", "b", "c", "code"])
    found = mp.discover(scope, ["a", "b", "c", ("code", "code", 0)], sample=256, backend="cpu")   # or "cuda"
    found.relations     # [{"vars", "labels", "constant", "checked", "violations", "witness", "holds", "status",
                        #   "certified" (FND-007 evidence), "certificate" (math_certificates.Certificate or None)}]
    found.rejected      # the proposals that failed, with their witness rows (one-row near misses)
    conds = [mb.condition("c", "<>", 0), mb.condition("d", ">", 5)]
    x = mb.variables(2)
    rw = mp.substitute(scope, conds, x[0] & x[1], found.relations[0]["certificate"])   # c AND d -> kept or rewritten
    rw.chosen, rw.costs, rw.program, rw.original, rw.conditions
    mp.verify(rw, scope, backend="cuda")     # the substituted predicate against the original on every row

A relation XOR x_j = constant over bit variables (Boolean columns, (name, "code", b) planes, (name, "bit", b, base))
is found by FIELD-001 and certified by FND-007 (exhaustive over the selected rows; the certificate records the scope,
the revision, each column's encoding and NULL domain). A predicate that reads one side of a certified relation is
rewritten over the other when the certificate is fresh and the rewrite scans fewer columns and planes (mode "auto"), or
always (mode "all"); a stale certificate refuses the substitution with its staleness reasons. Nothing is computed in
Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from numbers import Integral

import numpy as np

from . import algq
from . import math_bool as _mb
from . import math_certificates as _mc
from . import math_field as _mf
from .math_bank import Scope
from .math_bool import Expr, Program, _Condition, _Domain, _ProgramView, _conditions
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_certificates import (Certificate, NotAuthorizedError, StaleCertificateError, _Bit, _Claim,
                                _Decision as _CertDecision, _Evidence as _CertEvidence, _freshness)
from .math_errors import RESOURCES, check as _check
from .math_field import _BitSpec, _Discovery, _DiscoverSpec, _Evidence as _FieldEvidence, _Relation

__all__ = ["Found", "Rewrite", "discover", "substitute", "verify", "verify_programs", "bit_condition",
           "discover_need", "STATUS", "MODES"]

STATUS = {1: "certified", 2: "violated", 3: "too wide", 4: "empty"}
MODES = {"auto": 0, "all": 1}
CHOSEN = {0: "kept", 1: "rewritten"}
REFUSALS = {0: None, 1: "certificate", 2: "nulls", 3: "claim"}
NULLS = {"forbid": 1, "skip": 2}
NONE = (1 << 64) - 1
_NEED_CPU, _NEED_DEVICE = 0, 1
P = C.POINTER


class _Spec(C.Structure):
    _native_ = "tr_parity_spec"
    _fields_ = [("sample", C.c_uint64), ("affine", C.c_int32), ("policy", C.c_int32)]


class _ParityRelation(C.Structure):
    _native_ = "tr_parity_relation"
    _fields_ = [("relation", _Relation), ("evidence", _FieldEvidence), ("certified", _CertEvidence),
                ("status", C.c_int32), ("ncols", C.c_int32)]


class _FoundInfo(C.Structure):
    _native_ = "tr_parity_found_info"
    _fields_ = [("stats", _Discovery)] + [(n, C.c_int32) for n in ("nrelations", "nrejected", "ncertified", "policy",
                                                                   "nvars", "affine", "backend", "reserved")]


class _RewriteInfo(C.Structure):
    _native_ = "tr_parity_rewrite_info"
    _fields_ = [(n, C.c_int32) for n in ("chosen", "reason", "k", "nnodes", "root", "forms", "cost_original",
                                         "cost_rewritten", "candidates", "appended", "mode", "k_original")] + [
        ("decision", _CertDecision)]


class _Check(C.Structure):
    _native_ = "tr_parity_check"
    _fields_ = [(n, C.c_uint64) for n in ("points", "mismatches", "witness", "a_true", "a_false", "a_unknown",
                                          "b_true", "b_false", "b_unknown")] + [
        ("equivalent", C.c_int32), ("reserved", C.c_int32)]


for _s, _n in ((_Spec, 16), (_ParityRelation, 128), (_FoundInfo, 96), (_RewriteInfo, 112), (_Check, 80)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, e, pi = C.c_void_p, C.c_int32, P(C.c_int32), P(C.c_int32)
        sig = {
            "tr_parity_claim": (i32, [P(_BitSpec), i32, P(_Relation), i32, P(_Claim), P(_Bit), pi, pi]),
            "tr_parity_discover_need": (i32, [vp, P(_BitSpec), i32, P(_Spec), P(_Need)]),
            "tr_parity_discover": (vp, [vp, P(_BitSpec), i32, P(_Spec), P(_Budget), P(_Refusal), e]),
            "tr_parity_adopt": (vp, [vp, P(_BitSpec), i32, P(_Spec), P(_Discovery), P(_Relation), P(_FieldEvidence),
                                     i32, P(_Relation), P(_FieldEvidence), i32, P(_CertEvidence), P(_Budget),
                                     P(_Refusal), e]),
            "tr_parity_found_get_info": (i32, [vp, P(_FoundInfo)]),
            "tr_parity_found_relation": (i32, [vp, i32, P(_ParityRelation)]),
            "tr_parity_found_rejected": (i32, [vp, i32, P(_Relation), P(_FieldEvidence)]),
            "tr_parity_found_certificate": (vp, [vp, i32]),
            "tr_parity_found_free": (None, [vp]),
            "tr_parity_substitute": (vp, [vp, P(_Condition), i32, vp, vp, i32, P(_Budget), P(_Refusal),
                                          P(_RewriteInfo), e]),
            "tr_parity_rewrite_get_info": (i32, [vp, P(_RewriteInfo)]),
            "tr_parity_rewrite_conditions": (i32, [vp, P(_Condition), i32]),
            "tr_parity_rewrite_program": (vp, [vp, i32]),
            "tr_parity_rewrite_free": (None, [vp]),
            "tr_parity_verify_need": (i32, [P(_Domain), P(_ProgramView), P(_ProgramView), i32, P(_Need)]),
            "tr_parity_verify": (i32, [vp, vp, vp, P(_Budget), P(_Check), P(_Refusal)]),
            "tr_parity_verify_view": (i32, [P(_Domain), P(_ProgramView), P(_ProgramView), P(_Budget), P(_Check),
                                            P(_Refusal)]),
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


def _budget(budget):
    return (budget or Budget()).raw


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _gpu():
    gpu = _mf._gpu()                                       # FIELD-001's device discovery
    _mc._gpu()                                             # FND-007's device verification of the claims
    gpu.bind("tr_mgpu_parity_verify", C.c_int, [P(_Domain), P(_ProgramView), P(_ProgramView), P(_Budget), P(_Check),
                                               P(_Refusal)])
    return gpu


def _label(gpu):
    d = gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


def _spec(sample, affine, nulls):
    if isinstance(sample, bool) or not isinstance(sample, Integral) or sample < 0:
        raise ValueError("sample is a row count >= 0")
    if nulls not in NULLS:
        raise ValueError("nulls is 'forbid' or 'skip'")
    return _Spec(int(sample), int(bool(affine)), NULLS[nulls])


def _field_evidence(e):
    return dict(checked=int(e.checked), excluded=int(e.excluded), violations=int(e.violations),
                witness=None if e.witness == NONE else int(e.witness), holds=bool(e.holds))


# ---------------------------------------------------------------------- discovery and certificates
class Found:
    """An immutable native discovery: FIELD-001's relations with their FND-007 certificates, and the rejected
    proposals (near misses with their witness rows)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_parity.discover(scope, specs)")

    @classmethod
    def _wrap(cls, handle, scope, labels, backend, device):
        self = object.__new__(cls)
        self._handle, self.scope, self.labels, self.backend, self.device = handle, scope, tuple(labels), backend, device
        self._certs = {}
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the discovery is closed")
        return self._handle

    def _raw_info(self):
        i = _FoundInfo()
        _check(_call("tr_parity_found_get_info", self._open(), C.byref(i)), what="discovery")
        return i

    @property
    def info(self):
        i = self._raw_info()
        stats = {n: int(getattr(i.stats, n)) for n, _ in _Discovery._fields_}
        return dict(stats, nrelations=int(i.nrelations), nrejected=int(i.nrejected), certified=int(i.ncertified),
                    nulls={v: k for k, v in NULLS.items()}[i.policy], variables=int(i.nvars), affine=bool(i.affine),
                    verified_by=("cpu", "cuda library")[i.backend], labels=self.labels, backend=self.backend,
                    device=self.device)

    def _relation(self, r):
        vars_ = tuple(j for j in range(len(self.labels)) if (int(r.vars) >> j) & 1)
        return dict(vars=vars_, labels=tuple(self.labels[j] for j in vars_), constant=int(r.constant))

    def certificate(self, i):
        """the FND-007 certificate of relation i (an owned copy), or None"""
        if i not in self._certs:
            h = _call("tr_parity_found_certificate", self._open(), int(i))
            if not h:
                self._certs[i] = None
            else:
                error = C.c_int32()
                copy = _mc._call("tr_cert_clone", h, C.byref(error))
                _check(error.value, what="certificate copy")
                self._certs[i] = Certificate._wrap(copy, self.scope.names, (), self.backend, self.device)
        return self._certs[i]

    @property
    def relations(self):
        out = []
        for i in range(int(self._raw_info().nrelations)):
            r = _ParityRelation()
            _check(_call("tr_parity_found_relation", self._open(), i, C.byref(r)), what="relation")
            rec = dict(self._relation(r.relation), **_field_evidence(r.evidence), status=STATUS[r.status],
                       columns=int(r.ncols), certified=_mc._evidence(r.certified, self.backend, self.device)
                       if r.status in (1, 2) else None, certificate=self.certificate(i))
            out.append(rec)
        return out

    @property
    def rejected(self):
        out = []
        for i in range(int(self._raw_info().nrejected)):
            r, e = _Relation(), _FieldEvidence()
            _check(_call("tr_parity_found_rejected", self._open(), i, C.byref(r), C.byref(e)), what="rejected")
            out.append(dict(self._relation(r), **_field_evidence(e)))
        return out

    @property
    def certificates(self):
        return [self.certificate(i) for i in range(int(self._raw_info().nrelations))]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_parity_found_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Found(closed)"
        i = self._raw_info()
        return f"Found({i.nrelations} relations, {i.ncertified} certified, {i.nrejected} rejected, {self.backend})"


def discover(scope, specs, *, sample=256, affine=True, nulls="forbid", backend="cpu", budget=None):
    """Every GF(2) relation among the bit variables that holds on every complete selected row (FIELD-001: proposals
    from a sample, exhaustive verification, near misses rejected with their witnesses), each certified as an FND-007
    snapshot certificate (nulls: the certificates' NULL policy). backend "cuda": FIELD-001's device discovery and
    FND-007's device verification of the claims, adopted natively."""
    if not isinstance(scope, Scope):
        raise TypeError("scope: a math_bank.Scope")
    if scope.source == "table":
        scope.require_fresh("parity discovery")
    arr, labels = _mf._specs(scope, specs)
    n = len(labels)
    spec = _spec(sample, affine, nulls)
    error, refusal = C.c_int32(), _Refusal()
    if _backend(backend) == "cpu":
        h = _call("tr_parity_discover", scope._handle, arr, n, C.byref(spec), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        _check(error.value, refusal, "parity discovery")
        if not h:
            raise MemoryError("parity discovery: native allocation failed")
        return Found._wrap(h, scope, labels, "cpu", "native CPU")
    gpu = _gpu()
    from . import math_budgets
    math_budgets.check(budget or Budget(), "parity discovery on the device", **discover_need(scope, specs,
                                                                                             sample=sample,
                                                                                             affine=affine))
    bits = _mf.Bits.from_scope(scope, specs, budget=budget)
    view = bits.view()
    cols = n + spec.affine
    fcap, rcap = max(cols, 1), max(cols * (cols + 1) // 2, 1)
    found, fev = (_Relation * fcap)(), (_FieldEvidence * fcap)()
    rej, rev = (_Relation * rcap)(), (_FieldEvidence * rcap)()
    stats = _Discovery()
    fspec = _DiscoverSpec(spec.sample, spec.affine, 0)
    gpu.run("tr_mgpu_field_discover", C.byref(view), C.byref(fspec), C.byref(_budget(budget)), found, fev, fcap, rej,
            rev, rcap, C.byref(stats), C.byref(refusal), refusal=refusal, what="parity discovery on the device")
    nrel, nrej = int(stats.nrelations), int(stats.nrejected)
    claims, keep, where = [], [], []
    for i in range(nrel):                                    # the claims, built natively
        c, bitbuf, st, nc = _Claim(), (_Bit * 64)(), C.c_int32(), C.c_int32()
        _check(_call("tr_parity_claim", arr, n, C.byref(found[i]), spec.policy, C.byref(c), bitbuf, C.byref(st),
                     C.byref(nc)), what="parity claim")
        if st.value == 0:
            need = _Need()
            _check(_mc._call("tr_cert_claim_check", C.byref(c), len(scope.names), C.byref(need)), what="claim")
            claims.append(c); keep.append(bitbuf); where.append(i)
    claim_ev = (_CertEvidence * max(nrel, 1))()
    if claims:
        raw = (_Claim * len(claims))(*claims)
        out = (_CertEvidence * len(claims))()
        sview = scope.view()
        gpu.run("tr_mgpu_cert_verify", C.byref(sview), raw, len(claims), C.byref(_budget(budget)), out,
                C.byref(refusal), refusal=refusal, what="parity certificates on the device")
        for j, i in enumerate(where):
            claim_ev[i] = out[j]
    h = _call("tr_parity_adopt", scope._handle, arr, n, C.byref(spec), C.byref(stats), found, fev, nrel, rej, rev,
              nrej, claim_ev, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    _check(error.value, refusal, "adopted parity discovery")
    if not h:
        raise MemoryError("parity discovery: native allocation failed")
    label, device = _label(gpu)
    return Found._wrap(h, scope, labels, label, device)


def discover_need(scope, specs, *, sample=256, affine=True, nulls="forbid"):
    """the needs a discovery checks before allocating (FIELD-001's bits and discovery, the found object)"""
    arr, labels = _mf._specs(scope, specs)
    out = _Need()
    _check(_call("tr_parity_discover_need", scope._handle, arr, len(labels), C.byref(_spec(sample, affine, nulls)),
                 C.byref(out)), what="discovery needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}


def bit_condition(scope, spec):
    """the BOOL-001 condition of a bit variable: a column (name or (name, "nonzero")): `name <> 0`; (name, "bit", b[,
    base]): bit b of (logical - base); (name, "code", b): bit plane b of the stored code (base = the codec offset)"""
    spec = (spec, "nonzero") if isinstance(spec, str) else tuple(spec)
    name, kind = spec[0], spec[1]
    if kind == "nonzero":
        return _mb.condition(name, "<>", 0)
    if kind == "code":
        return _mb.condition(name, "bit", int(spec[2]), base=scope.column(name)["offset"])
    if kind == "bit":
        return _mb.condition(name, "bit", int(spec[2]), base=int(spec[3]) if len(spec) > 3 else 0)
    raise ValueError(f"{spec!r}: a column name, (name, 'nonzero'), (name, 'bit', b[, base]) or (name, 'code', b)")


# ---------------------------------------------------------------------- substitution
def _program_copy(handle, k):
    """a math_bool.Program over k conditions with the nodes of a borrowed native program"""
    v = _ProgramView()
    _check(_mb._call("tr_bool_program_get_view", handle, C.byref(v)), what="program view")
    nodes = (_mb._Node * v.nnodes).from_address(v.nodes)
    return Program.from_nodes(k, [(int(x.op), int(x.a), int(x.b)) for x in nodes], root=int(v.root))


def _condition_from_native(c, names):
    if c.op == 2 and c.col2 == -1 and c.num[0] == 0 and c.den[0] == 1:
        return _mb.condition(names[c.col], "<>", 0)
    if c.op == 10:
        return _mb.condition(names[c.col], "bit", int(c.num[1]), base=int(c.num[0]))
    raise ValueError(f"unexpected appended condition (op {c.op})")


class Rewrite:
    """A substitution: the conditions (the predicate's and the relation bits appended), the original program over them,
    the rewritten program, the choice and its costs (columns and planes scanned)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_parity.substitute(...)")

    def _open(self):
        if not self._handle:
            raise ValueError("the rewrite is closed")
        return self._handle

    @property
    def info(self):
        i = _RewriteInfo()
        _check(_call("tr_parity_rewrite_get_info", self._open(), C.byref(i)), what="rewrite")
        return _rewrite_info(i)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_parity_rewrite_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Rewrite({self.chosen}, cost {self.costs})" if self._handle else "Rewrite(closed)"


def _rewrite_info(i):
    return dict(chosen=CHOSEN[i.chosen], reason=REFUSALS[i.reason], conditions=int(i.k), nodes=int(i.nnodes),
                root=int(i.root), forms=int(i.forms), cost_original=int(i.cost_original),
                cost_rewritten=int(i.cost_rewritten), candidates=int(i.candidates), appended=int(i.appended),
                mode={v: k for k, v in MODES.items()}[i.mode], k_original=int(i.k_original))


def substitute(scope, conditions, predicate, certificate, *, mode="auto", budget=None):
    """The predicate (a math_bool expression or Program over the conditions) rewritten with a certified parity relation
    (an FND-007 Certificate) when the certificate is fresh for scope: mode "auto" keeps the cheaper form (columns and
    planes scanned), "all" rewrites every XOR form meeting the relation. StaleCertificateError (with .reasons) when the
    certificate is stale; NotAuthorizedError (.reason "certificate", "nulls", "claim") for the other refusals."""
    if not isinstance(scope, Scope):
        raise TypeError("scope: a math_bank.Scope")
    if not isinstance(certificate, Certificate):
        raise TypeError("certificate: a math_certificates.Certificate")
    if mode not in MODES:
        raise ValueError("mode is 'auto' or 'all'")
    if scope.source == "table":
        scope.require_fresh("parity substitution")
    conditions = tuple(conditions)
    arr, k = _conditions(conditions, scope.names)
    program = predicate if isinstance(predicate, Program) else Program(predicate, k)
    program._open()
    info, error, refusal = _RewriteInfo(), C.c_int32(), _Refusal()
    h = _call("tr_parity_substitute", scope._handle, arr, k, program._handle, certificate._open(), MODES[mode],
              C.byref(_budget(budget)), C.byref(refusal), C.byref(info), C.byref(error))
    if not h:
        code = error.value
        if code == -13:
            f = _freshness(info.decision.stale)
            err = StaleCertificateError(f"the parity certificate is stale: {', '.join(f.reasons)}")
            err.reasons, err.first, err.freshness, err.info = f.reasons, f.first, f, _rewrite_info(info)
            raise err
        if code == -6:
            reason = REFUSALS[info.reason]
            detail = _mc.REFUSALS.get(info.decision.reason) if reason == "certificate" else None
            what = {"certificate": f"the certificate cannot authorize the rewrite ({detail})",
                    "nulls": "the certificate covers the complete rows only (NULLs were skipped)",
                    "claim": "the certificate is not a parity claim"}.get(reason, "domain error")
            err = NotAuthorizedError(f"parity substitution refused: {what}")
            err.reason, err.detail, err.info = reason, detail, _rewrite_info(info)
            raise err
        _check(code, refusal, "parity substitution")
        raise MemoryError("parity substitution: native allocation failed")
    rw = object.__new__(Rewrite)
    rw._handle = h
    i = rw.info
    raw = (_Condition * max(i["conditions"], 1))()
    _check(_call("tr_parity_rewrite_conditions", h, raw, i["conditions"]), what="rewrite conditions")
    rw.conditions = conditions + tuple(_condition_from_native(raw[j], scope.names) for j in range(k, i["conditions"]))
    rw.original = _program_copy(_call("tr_parity_rewrite_program", h, 0), i["conditions"])
    rw.program = _program_copy(_call("tr_parity_rewrite_program", h, 1), i["conditions"])
    rw.chosen = i["chosen"]
    rw.costs = dict(original=i["cost_original"], rewritten=i["cost_rewritten"])
    rw.scope, rw.certificate = scope, certificate
    return rw


def _check_dict(c, backend, device):
    return dict(points=int(c.points), mismatches=int(c.mismatches),
                witness=None if c.witness == NONE else int(c.witness), equivalent=bool(c.equivalent),
                original=dict(true=int(c.a_true), false=int(c.a_false), unknown=int(c.a_unknown)),
                substituted=dict(true=int(c.b_true), false=int(c.b_false), unknown=int(c.b_unknown)),
                backend=backend, device=device)


def verify_programs(masks, a, b, *, backend="cpu", budget=None):
    """two math_bool Programs (over the masks' conditions) on every point of the masks: their counts, the points where
    they differ and the smallest one"""
    masks._open(); a._open(); b._open()
    masks.require_fresh("predicate verification")
    out, refusal = _Check(), _Refusal()
    if _backend(backend) == "cpu":
        _check(_call("tr_parity_verify", masks._handle, a._handle, b._handle, C.byref(_budget(budget)), C.byref(out),
                     C.byref(refusal)), refusal, "predicate verification")
        return _check_dict(out, "cpu", "native CPU")
    gpu = _gpu()
    gpu.run("tr_mgpu_parity_verify", C.byref(masks.domain()), C.byref(a.view()), C.byref(b.view()),
            C.byref(_budget(budget)), C.byref(out), C.byref(refusal), refusal=refusal,
            what="predicate verification on the device")
    return _check_dict(out, *_label(gpu))


def verify(rewrite, scope=None, *, backend="cpu", budget=None):
    """the substituted predicate against the original on every selected row of scope (default: the rewrite's)"""
    scope = scope if scope is not None else rewrite.scope
    masks = _mb.Masks.from_scope(scope, rewrite.conditions, backend=backend, budget=budget)
    return verify_programs(masks, rewrite.original, rewrite.program, backend=backend, budget=budget)


def verify_need(masks, a, b, *, backend="cpu"):
    """the needs a verification checks before allocating"""
    out = _Need()
    _check(_call("tr_parity_verify_need", C.byref(masks.domain()), C.byref(a.view()), C.byref(b.view()),
                 _NEED_CPU if _backend(backend) == "cpu" else _NEED_DEVICE, C.byref(out)), what="verification needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}
