# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-003: conservation and stability of invariants across revisions, natively on the CPU or with the CUDA kernels
(torbles/math_conserve.h/.c, math_conserve_gpu.cu).

    from torbles import math_bank as mbk, math_ident as mi, math_delta as md, math_conserve as mcons
    scope = mbk.Scope.from_table(t, ["x", "y", "s"])
    ids = mi.discover(scope, ["x", "y", "s"], degree=1).identities          # or mi.candidates(...)
    tr = mcons.track(scope, ["x", "y", "s"], degree=1, identities=ids)       # step 0: verified and certified
    scope2 = mbk.Scope.from_table(t, ["x", "y", "s"])                        # after a mutation
    tr = tr.update(md.Delta.between(scope, scope2), scope=scope2, backend="cuda")   # incremental, bound: certified
    tr = tr.update(md.Delta.from_arrays({...}, weights=[1, -1]))             # unbound: held, never certified
    tr = tr.rebind(scope3)                                                   # recomputed from a scope
    tr.record(i); tr.steps(); tr.certificate(i); tr.gram(scope2); tr.moments()
    tr.classes(i, given=[hypothesis_certificate]); tr.authorize(i, scope2, given=[...])
    law = mcons.conserve({(1, 0, 0): 1, (0, 1, 0): 1}, {"x": {(1, 0, 0): 1, (0, 0, 1): 1},
                                                       "y": {(0, 1, 0): 1, (0, 0, 1): -1}}, ["x", "y", "t"])
    law.conserved, law.certificate, law.residual, law.counterexample
    law.check(scope, {"x": "x", "y": "y", "t": "t"}, backend="cuda")       # snapshot certificate or witness rows

Evidence classes are kept apart and never converted into one another: DECLARED (a schema guarantee: a declaration,
FND-007 schema facts), SNAPSHOT (the tracker's FND-007 certificate of one revision), STABILITY (the record of the
steps it held on: not a proof about another revision) and HYPOTHESIS (DISC-002 sampled hypotheses). Only a fresh
snapshot authorizes an exact rewrite. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C

import numpy as np

from . import algq
from . import math_bank as _mbk
from . import math_certificates as _mc
from . import math_delta as _md
from . import math_ident as _mi
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_domains import DomainError
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Tracker", "Law", "track", "conserve", "STATUSES", "STEPS", "CLASSES", "REFUSALS", "MAX_IDENTITIES"]

MAX_VARS, MAX_IDENTITIES = 8, 256
NONE = (1 << 64) - 1
STATUSES = {0: "pending", 1: "certified", 2: "verified", 3: "held", 4: "failed", 5: "undecided", 6: "untracked"}
STEPS = {1: "build", 2: "diff", 3: "scope", 4: "arrays", 5: "rebind"}
WITNESS = {0: None, 1: "scope", 2: "inserted", 3: "deleted"}
CLASSES = {0: None, 1: "declared", 2: "snapshot", 3: "stability", 4: "hypothesis"}
REFUSALS = {0: None, 1: "stability", 2: "statistical", 3: "declared", 4: "failed", 5: "snapshot", 6: "none",
            7: "untracked"}
_NEED_CPU, _NEED_DEVICE = 0, 1
_BY_CPU, _BY_DEVICE = 0, 1
P = C.POINTER


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _DeltaEvidence(C.Structure):
    _native_ = "tr_cons_delta_evidence"
    _fields_ = [(n, C.c_uint64) for n in ("checked_ins", "checked_del", "excluded_ins", "excluded_del",
                                          "violations_ins", "violations_del", "overflow_ins", "overflow_del",
                                          "witness_ins", "witness_del")]


class _DeltaOut(C.Structure):
    _native_ = "tr_cons_delta_out"
    _fields_ = [("nmoments", C.c_int32), ("nident", C.c_int32), ("gram_ins", C.c_uint64), ("gram_del", C.c_uint64),
                ("acc", C.c_void_p), ("ev", C.c_void_p)]


class _JobView(C.Structure):
    _native_ = "tr_cons_job_view"
    _fields_ = [("rows", _mbk._View), ("weights", C.c_void_p), ("plan", _mi._MomentView), ("cand", _mi._CandView)]


class _Info(C.Structure):
    _native_ = "tr_cons_info"
    _fields_ = [(n, C.c_int32) for n in ("nvars", "nbasis", "nmoments", "nident", "tracked", "bound", "degree",
                                         "ncols")] + [
        (n, C.c_uint64) for n in ("step", "revision", "domain", "bytes", "certified", "verified", "held", "failed",
                                  "undecided")] + [("scope_digest", C.c_uint64 * 2), ("state_identity", C.c_uint64 * 2)]


class _Record(C.Structure):
    _native_ = "tr_cons_record"
    _fields_ = [(n, C.c_int32) for n in ("index", "status", "tracked", "declared", "snapshot", "certificate_code")] + [
        (n, C.c_uint64) for n in ("certified", "verified", "held", "failed", "undecided", "first_certified",
                                  "last_certified", "runs", "streak", "fail_step", "fail_revision", "fail_violations",
                                  "fail_row")] + [
        ("fail_source", C.c_int32), ("fail_ncols", C.c_int32), ("fail_weight", C.c_int64),
        ("fail_col", C.c_int32 * 8), ("fail_value", C.c_int64 * 8), ("fail_valid", C.c_uint8 * 8),
        ("witness_ins", C.c_uint64), ("witness_del", C.c_uint64), ("evidence", _mc._Evidence)]


class _Step(C.Structure):
    _native_ = "tr_cons_step"
    _fields_ = [(n, C.c_uint64) for n in ("step", "revision", "rows", "inserted", "deleted")] + [
        ("kind", C.c_int32), ("bound", C.c_int32), ("certified", C.c_uint64), ("failed", C.c_uint64),
        ("domain", C.c_uint64), ("digest", C.c_uint64 * 2), ("delta", C.c_uint64 * 2)]


class _Classes(C.Structure):
    _native_ = "tr_cons_classes"
    _fields_ = [(n, C.c_int32) for n in ("declared", "snapshot", "stability", "hypothesis", "schema", "status")] + [
        ("reserved", C.c_int32 * 2), ("certified", C.c_uint64), ("snapshot_step", C.c_uint64),
        ("hypothesis_size", C.c_uint64)]


class _Decision(C.Structure):
    _native_ = "tr_cons_decision"
    _fields_ = [(n, C.c_int32) for n in ("authorized", "reason", "cls", "status")] + [("snapshot", _mc._Decision)]


class _LawInfo(C.Structure):
    _native_ = "tr_cons_law_info"
    _fields_ = [(n, C.c_int32) for n in ("nvars", "conserved", "certified", "searched", "found", "words", "degree",
                                         "reserved")] + [
        (n, C.c_uint64) for n in ("q_terms", "expanded_terms", "residual_terms", "points")] + [
        ("point", C.c_int64 * 8), ("value", C.c_uint64 * 8), ("bytes", C.c_uint64)]


for _s, _n in ((_DeltaEvidence, 80), (_DeltaOut, 40), (_JobView, 2296), (_Info, 136), (_Record, 328), (_Step, 104),
               (_Classes, 56), (_Decision, 80), (_LawInfo, 200)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, i64, u64, e = C.c_void_p, C.c_int32, C.c_int64, C.c_uint64, P(C.c_int32)
        B, R = P(_Budget), P(_Refusal)
        sig = {
            "tr_cons_tracker_make": (vp, [vp, P(_mi._Spec), vp, vp, B, R, e]),
            "tr_cons_job_make": (vp, [vp, vp, B, R, e]),
            "tr_cons_job_get_view": (i32, [vp, P(_JobView)]),
            "tr_cons_job_need": (i32, [vp, i32, P(_Need)]),
            "tr_cons_job_run": (i32, [vp, P(_DeltaOut)]),
            "tr_cons_job_free": (None, [vp]),
            "tr_cons_tracker_update": (vp, [vp, vp, P(_DeltaOut), i32, vp, B, R, P(_md._Violation), e]),
            "tr_cons_tracker_rebind": (vp, [vp, vp, B, R, e]),
            "tr_cons_tracker_get_info": (i32, [vp, P(_Info)]),
            "tr_cons_tracker_record": (i32, [vp, i32, P(_Record)]),
            "tr_cons_tracker_runs": (i64, [vp, i32, vp, u64]),
            "tr_cons_tracker_steps": (i64, [vp, P(_Step), u64]),
            "tr_cons_tracker_certificate": (vp, [vp, i32]),
            "tr_cons_tracker_claim": (i32, [vp, i32, P(_mc._Claim)]),
            "tr_cons_tracker_classes": (i32, [vp, i32, P(vp), i32, P(_Classes)]),
            "tr_cons_authorize": (i32, [vp, i32, P(vp), i32, vp, u64, P(_Decision)]),
            "tr_cons_tracker_moments": (i32, [vp, vp, vp, u64]),
            "tr_cons_tracker_gram": (vp, [vp, vp, B, R, e]),
            "tr_cons_tracker_state": (vp, [vp]),
            "tr_cons_tracker_free": (None, [vp]),
            "tr_cons_conserve": (vp, [i32, P(_mi._Poly), P(_mi._Poly), B, R, e]),
            "tr_cons_law_get_info": (i32, [vp, P(_LawInfo)]),
            "tr_cons_law_residual": (i32, [vp, vp, vp, u64, P(u64)]),
            "tr_cons_law_certificate": (vp, [vp]),
            "tr_cons_law_candidates": (vp, [vp, vp, P(C.c_int32), B, R, e]),
            "tr_cons_law_witnesses": (i32, [vp, vp, P(C.c_int32), B, P(_mc._Evidence), vp, u64, P(u64), R]),
            "tr_cons_law_free": (None, [vp]),
            "tr_math_budget_check": (i32, [B, P(_Need), R]),
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
    math_gpu.bind("tr_mgpu_cons_delta", C.c_int, [P(_JobView), P(_Budget), P(_DeltaOut), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
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


def _opt(v):
    return None if v == NONE else int(v)


def _i128(lo, hi):
    return (int(hi) << 64) + (int(lo) & ((1 << 64) - 1))      # hi signed


def _words(ws):
    v = 0
    for i, w in enumerate(ws):
        v |= int(w) << (64 * i)
    n = 64 * len(ws)
    return v - (1 << n) if ws and v >> (n - 1) else v


def _spec(scope, variables, degree, basis):
    cols = _mi._columns(scope, variables)
    if not 1 <= len(cols) <= MAX_VARS:
        if len(cols) > MAX_VARS:
            raise UnsupportedError(f"{len(cols)} variables: at most {MAX_VARS}")
        raise ValueError("give at least one variable")
    s = _mi._Spec()
    s.nvars = len(cols)
    for i in range(8):
        s.var[i] = cols[i] if i < len(cols) else -1
    s.weight, s.width, s.nprimes = -1, 0, 0
    keep = None
    if basis is not None:
        rows = [tuple(int(x) for x in t) for t in basis]
        if any(len(t) != len(cols) or any(not 0 <= x <= 255 for x in t) for t in rows):
            raise ValueError(f"basis terms are tuples of {len(cols)} exponents in 0..255")
        keep = np.ascontiguousarray(np.array(rows, dtype=np.uint8).reshape(len(rows), len(cols)))
        s.nterms, s.exponents = len(rows), keep.ctypes.data if len(rows) else None
    else:
        if degree is None or int(degree) < 0:
            raise ValueError("give degree >= 0 or an explicit basis")
        s.degree = int(degree)
    return s, keep


# ------------------------------------------------------------------ trackers
class Tracker:
    """An immutable tracker at one revision: the maintained DISC-002 Gram, the tracked identities with their
    incremental evidence, records and certificates, and the companion DELTA-001 state."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_conserve.track(scope, variables, ...)")

    @classmethod
    def _wrap(cls, handle, scope, names, variables, spec_args, backend="cpu", device="native CPU"):
        self = object.__new__(cls)
        self._handle, self.scope, self.names, self.variables = handle, scope, tuple(names), tuple(variables)
        self._spec_args, self.backend, self.device = spec_args, backend, device
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the tracker is closed")
        return self._handle

    def _info(self):
        i = _Info()
        _check(_call("tr_cons_tracker_get_info", self._open(), C.byref(i)), what="tracker")
        return i

    @property
    def info(self):
        i = self._info()
        return dict(variables=list(self.variables), basis=int(i.nbasis), moments=int(i.nmoments),
                    identities=int(i.nident), bound=bool(i.bound), degree=int(i.degree), step=int(i.step),
                    revision=int(i.revision), domain=int(i.domain), bytes=int(i.bytes), certified=int(i.certified),
                    verified=int(i.verified), held=int(i.held), failed=int(i.failed), undecided=int(i.undecided),
                    backend=self.backend, device=self.device)

    @property
    def step(self):
        return int(self._info().step)

    @property
    def bound(self):
        return bool(self._info().bound)

    def __len__(self):
        return int(self._info().nident)

    def claim(self, i):
        """identity i's FND-007 POLYNOMIAL claim as a math_certificates.Claim (columns: its support)"""
        c = _mc._Claim()
        _check(_call("tr_cons_tracker_claim", self._open(), int(i), C.byref(c)), what="identity claim")
        n, k = int(c.nterms), int(c.ncols)
        exps = np.ctypeslib.as_array((C.c_uint8 * (n * k)).from_address(c.exponents)).reshape(n, k).copy() if n else np.zeros((0, k), np.uint8)
        coefs = np.ctypeslib.as_array((C.c_int64 * n).from_address(c.coefficients)).copy() if n else np.zeros(0, np.int64)
        cols = [self.names[c.col[q]] for q in range(k)]
        return _mc.Claim("polynomial", cols, nulls="skip", exponents=exps, coefficients=coefs)

    def identity(self, i):
        """{columns, terms {exponent tuple over columns: coefficient}, text} of identity i"""
        cl = self.claim(i)
        exps, coefs = cl.params["exponents"], cl.params["coefficients"]
        terms = {tuple(int(x) for x in exps[t]): int(coefs[t]) for t in range(len(coefs))}
        ordered = sorted(terms.items(), key=lambda t: (sum(t[0]), t[0]), reverse=True)
        return dict(columns=list(cl.columns), terms=terms, text=_mi.poly_text(ordered, list(cl.columns)))

    def record(self, i):
        """the persistence record of identity i (see math_conserve.h): statuses per step, certified runs, the first
        failure with its witness row, the current evidence"""
        r = _Record()
        _check(_call("tr_cons_tracker_record", self._open(), int(i), C.byref(r)), what="record")
        n = int(r.runs)
        pairs = np.zeros(max(2 * n, 2), np.uint64)
        _check(_call("tr_cons_tracker_runs", self._handle, int(i), pairs.ctypes.data, n), what="runs")
        fail = None
        if r.fail_step != NONE:
            values = {}
            for q in range(int(r.fail_ncols)):
                if r.fail_col[q] >= 0:
                    values[self.names[r.fail_col[q]]] = int(r.fail_value[q]) if r.fail_valid[q] else None
            fail = dict(step=int(r.fail_step), revision=int(r.fail_revision), violations=int(r.fail_violations),
                        row=_opt(r.fail_row), source=WITNESS[r.fail_source], weight=int(r.fail_weight), values=values)
        ev = _mc._evidence(r.evidence, self.backend, self.device)
        ev.pop("witness", None)
        return dict(index=int(r.index), status=STATUSES[r.status], declared=bool(r.declared),
                    snapshot=bool(r.snapshot), certificate_code=int(r.certificate_code) or None,
                    certified=int(r.certified), verified=int(r.verified), held=int(r.held), failed=int(r.failed),
                    undecided=int(r.undecided), first_certified=_opt(r.first_certified),
                    last_certified=_opt(r.last_certified),
                    runs=[(int(pairs[2 * j]), int(pairs[2 * j + 1])) for j in range(n)], streak=int(r.streak),
                    first_failure=fail, witness_inserted=_opt(r.witness_ins), witness_deleted=_opt(r.witness_del),
                    evidence=ev)

    def records(self):
        return [self.record(i) for i in range(len(self))]

    def steps(self):
        """one dict per step: kind (build, diff, scope, arrays, rebind), bound, table revision, delta rows,
        inserted / deleted, identities certified / failed, Gram rows after it"""
        n = int(_call("tr_cons_tracker_steps", self._open(), None, 0))
        arr = (_Step * max(n, 1))()
        _check(_call("tr_cons_tracker_steps", self._handle, arr, n), what="steps")
        return [dict(step=int(s.step), kind=STEPS[s.kind], bound=bool(s.bound), revision=int(s.revision),
                     rows=int(s.rows), inserted=int(s.inserted), deleted=int(s.deleted), certified=int(s.certified),
                     failed=int(s.failed), domain=int(s.domain),
                     delta=(int(s.delta[1]) << 64) | int(s.delta[0]) if s.kind != 1 and s.kind != 5 else None)
                for s in arr[:n]]

    def certificate(self, i):
        """the FND-007 snapshot certificate of identity i at the current revision (an owned clone), or None"""
        h = _call("tr_cons_tracker_certificate", self._open(), int(i))
        if not h:
            return None
        error = C.c_int32()
        clone = _mc._call("tr_cert_clone", h, C.byref(error))
        return _mc.Certificate._wrap(_mc._made(clone, error, _Refusal(), "certificate"), self.names, (),
                                     self.backend, self.device)

    def _given(self, given):
        given = list(given or ())
        for g in given:
            if not isinstance(g, _mc.Certificate):
                raise TypeError("given: math_certificates.Certificate objects")
        return (C.c_void_p * max(len(given), 1))(*[g._open() for g in given]), len(given), given

    def classes(self, i, given=()):
        """identity i's evidence classes: declared (schema guarantee), snapshot, stability (record), hypothesis;
        given: FND-007 certificates sorted into them by kind (DomainError for one that cannot belong here)"""
        arr, n, keep = self._given(given)
        out = _Classes()
        rc = _call("tr_cons_tracker_classes", self._open(), int(i), arr, n, C.byref(out))
        if rc == -6:
            raise DomainError("a given certificate is a snapshot (not the tracker's own), a law, a hypothesis of "
                              "another claim, or a schema fact about another column")
        _check(rc, what="evidence classes")
        return dict(declared=bool(out.declared), schema=int(out.schema), snapshot=bool(out.snapshot),
                    stability=bool(out.stability), hypothesis=int(out.hypothesis), status=STATUSES[out.status],
                    certified=int(out.certified), snapshot_step=_opt(out.snapshot_step),
                    hypothesis_size=int(out.hypothesis_size))

    def authorize(self, i, scope=None, given=(), field=0):
        """the exact rewrite decision for identity i on the current scope (never raises for a refusal): authorized,
        reason (stability, statistical, declared, failed, snapshot, none), class, status, the snapshot's FND-007
        decision"""
        arr, n, keep = self._given(given)
        d = _Decision()
        rc = _call("tr_cons_authorize", self._open(), int(i), arr, n, scope._handle if scope is not None else None,
                   int(field), C.byref(d))
        if rc == -1:
            _check(rc, what="authorization")
        if rc == -6 and not d.reason:
            raise DomainError("a given certificate cannot belong to this identity's classes")
        snap = None
        if d.cls == 2:
            s = d.snapshot
            snap = dict(authorized=bool(s.authorized), reason=_mc.REFUSALS[s.reason],
                        stale=_mc._freshness(s.stale) if s.stale.reasons else None)
        return dict(authorized=bool(d.authorized), reason=REFUSALS[d.reason], cls=CLASSES[d.cls],
                    status=STATUSES[d.status], snapshot=snap, code=int(rc))

    def moments(self):
        """the maintained Gram: {moment exponent tuple over the variables: exact sum}"""
        i = self._info()
        nm, k = int(i.nmoments), int(i.nvars)
        e = np.zeros(max(nm * k, 1), np.uint8)
        s = np.zeros(max(2 * nm, 2), np.int64)
        _check(_call("tr_cons_tracker_moments", self._handle, e.ctypes.data, s.ctypes.data, nm), what="moments")
        return {tuple(int(x) for x in e[m * k:(m + 1) * k]): _i128(s[2 * m], s[2 * m + 1]) for m in range(nm)}

    def gram(self, scope, *, budget=None):
        """the DISC-002 Gram of the current revision adopted on its scope (a math_ident.Gram; the revision must be
        that scope: DomainError otherwise), ready for kernel() and math_ident.identities() without a pass over the
        data"""
        _fresh(scope, "tracker Gram")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_cons_tracker_gram", self._open(), scope._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        g = _made(h, error, refusal, "tracker Gram")
        degree, basis = self._spec_args
        try:
            plan = _mi.plan(scope, self.variables, degree, basis=basis, budget=budget)
        except BaseException:
            _mi._call("tr_ident_gram_free", g)
            raise
        return _mi.Gram._wrap(g, scope, plan, "cpu", "native CPU")

    def evaluate(self, delta, *, backend="cpu", budget=None):
        """the incremental step's raw result for a delta (no new tracker): gram_ins, gram_del, the moment increments
        and per identity the evidence of the delta rows (CPU and CUDA give the same dict)"""
        job = self._job(delta, budget)
        try:
            out, keep, label, device = self._run(job, backend, budget)
            nm, n = out.nmoments, out.nident
            acc, ev = keep
            return dict(gram_ins=int(out.gram_ins), gram_del=int(out.gram_del),
                        moments=[_i128(acc[2 * m], acc[2 * m + 1]) for m in range(nm)],
                        evidence=[{f: (_opt(getattr(e, f)) if f.startswith("witness") else int(getattr(e, f)))
                                   for f, _ in _DeltaEvidence._fields_} for e in ev[:n]],
                        backend=label, device=device)
        finally:
            _call("tr_cons_job_free", job)

    def need(self, delta, backend="cpu"):
        """the needs of an update's evaluation: {resource: need}"""
        job = self._job(delta, None)
        try:
            raw = _Need()
            _check(_call("tr_cons_job_need", job, _NEED_CPU if backend == "cpu" else _NEED_DEVICE, C.byref(raw)),
                   what="needs")
            return _needs_dict(raw)
        finally:
            _call("tr_cons_job_free", job)

    def _job(self, delta, budget):
        if not isinstance(delta, _md.Delta):
            raise TypeError("delta: a math_delta.Delta")
        delta._open()
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_cons_job_make", self._open(), delta._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return _made(h, error, refusal, "update job")

    def _run(self, job, backend, budget):
        i = self._info()
        nm, n = int(i.nmoments), int(i.nident)
        acc = np.zeros(max(2 * nm, 2), np.int64)
        ev = (_DeltaEvidence * max(n, 1))()
        out = _DeltaOut(nm, n, 0, 0, acc.ctypes.data, C.cast(ev, C.c_void_p).value)
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_cons_job_run", job, C.byref(out)), what="incremental verification")
            return out, (acc, ev), "cpu", "native CPU"
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        raw = _Need()
        _check(_call("tr_cons_job_need", job, _NEED_DEVICE, C.byref(raw)), what="needs")
        _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw), C.byref(refusal)), refusal,
               "incremental verification on the device")
        view = _JobView()
        _check(_call("tr_cons_job_get_view", job, C.byref(view)), what="job view")
        _gpu().run("tr_mgpu_cons_delta", C.byref(view), C.byref(_budget(budget)), C.byref(out), C.byref(refusal),
                   refusal=refusal, what="incremental verification on the device")
        label, device = _device_label()
        return out, (acc, ev), label, device

    def update(self, delta, *, scope=None, backend="cpu", budget=None):
        """the next revision: the delta's rows evaluated (backend "cuda": the kernel; the CPU library adopts its
        result after consistency checks), the Gram and every identity's evidence updated exactly, the companion
        DELTA-001 state updated; a DIFF (Delta.between) needs scope= (the new scope) and certifies; another delta
        gives an unbound revision (held, never certified). Refusals: DomainError (an incompatible delta, a negative
        multiplicity: NegativeMultiplicityError), OverflowError (the update's proof), AlgebraBudgetError."""
        if scope is not None:
            _fresh(scope, "update")
        job = self._job(delta, budget)
        try:
            error, refusal, v = C.c_int32(), _Refusal(), _md._Violation()
            if backend == "cpu":
                h = _call("tr_cons_tracker_update", self._handle, job, None, _BY_CPU,
                          scope._handle if scope is not None else None, C.byref(_budget(budget)), C.byref(refusal),
                          C.byref(v), C.byref(error))
                label, device = "cpu", "native CPU"
            else:
                out, keep, label, device = self._run(job, backend, budget)
                h = _call("tr_cons_tracker_update", self._handle, job, C.byref(out), _BY_DEVICE,
                          scope._handle if scope is not None else None, C.byref(_budget(budget)), C.byref(refusal),
                          C.byref(v), C.byref(error))
            if error.value == -1 and (scope is None) == (delta.info["kind"] == "diff"):
                raise ValueError("a diff (Delta.between) needs scope=the new scope; other deltas take none")
            h = _md._made(h, error, refusal, "tracker update", v)
        finally:
            _call("tr_cons_job_free", job)
        return Tracker._wrap(h, scope, self.names, self.variables, self._spec_args, label, device)

    def rebind(self, scope, *, budget=None):
        """a new step recomputed from a scope (Gram, exhaustive evidence, companion): the lineage bound to it"""
        _fresh(scope, "rebind")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_cons_tracker_rebind", self._open(), scope._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Tracker._wrap(_made(h, error, refusal, "rebind"), scope, self.names, self.variables, self._spec_args)

    def locate(self, i, scope, *, backend="cpu", budget=None):
        """identity i verified exhaustively on a scope (DISC-002's verification): its evidence with the smallest
        violating physical row (the witness of a failure recorded as a delta row)"""
        _fresh(scope, "locate")
        cl = self.claim(i)
        exps, coefs = cl.params["exponents"], cl.params["coefficients"]
        terms = {tuple(int(x) for x in exps[t]): int(coefs[t]) for t in range(len(coefs))}
        return _mi.candidates(scope, list(cl.columns), [terms], budget=budget).verify(backend=backend, budget=budget)[0]

    def state(self):
        """the companion DELTA-001 state's info (identity, step, bound, rows)"""
        h = _call("tr_cons_tracker_state", self._open())
        out = _md._Info()
        _check(_md._call("tr_delta_state_get_info", h, C.byref(out)), what="companion state")
        return dict(identity=_md._digest(out.identity), step=int(out.step), bound=bool(out.bound),
                    rows=int(out.rows), groups=int(out.ngroups))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_cons_tracker_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Tracker(closed)"
        i = self.info
        return (f"Tracker(step={i['step']}, identities={i['identities']}, certified={i['certified']}, "
                f"failed={i['failed']}, bound={i['bound']})")


def track(scope, variables, degree=None, *, basis=None, identities=None, declared=None, budget=None):
    """a tracker at step 0 over scope: the DISC-002 plan of variables (degree or an explicit basis; unweighted, EXACT
    on the scope: OverflowError otherwise), its Gram, the identities of a DISC-002 set (math_ident.Identities over
    this scope's columns; representable, resolved items) verified exhaustively and certified, declared: a flag per set
    item (default: DISC-002's declared items and known labels)"""
    _fresh(scope, "tracker")
    s, keep = _spec(scope, variables, degree, basis)
    dec = None
    if identities is not None:
        if not isinstance(identities, _mi.Identities):
            raise TypeError("identities: a math_ident.Identities set")
        if identities.scope is not None and tuple(identities.scope.names) != tuple(scope.names):
            raise DomainError("the identities are over another scope's columns")
        if declared is not None:
            flags = [1 if bool(f) else 0 for f in declared]
            if len(flags) != len(identities):
                raise ValueError("declared: one flag per identity of the set")
            dec = (C.c_int32 * max(len(flags), 1))(*flags)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_cons_tracker_make", scope._handle, C.byref(s), identities._open() if identities is not None else None,
              dec, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Tracker._wrap(_made(h, error, refusal, "tracker"), scope, scope.names, variables, (degree, basis))


# ------------------------------------------------------------------ conservation
class Law:
    """A declared transformation T and a quantity q: P = q(T(x)) - q(x) expanded exactly; conserved (P == 0, with the
    FND-007 ring-identity certificate) or not (a counterexample assignment from the Nullstellensatz grid)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_conserve.conserve(q, images, variables)")

    @classmethod
    def _wrap(cls, handle, variables):
        self = object.__new__(cls)
        self._handle, self.variables = handle, tuple(variables)
        i = _LawInfo()
        _check(_call("tr_cons_law_get_info", handle, C.byref(i)), what="conservation")
        self._i = i
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the law is closed")
        return self._handle

    @property
    def info(self):
        i = self._i
        return dict(variables=list(self.variables), conserved=bool(i.conserved), certified=bool(i.certified),
                    searched=bool(i.searched), found=bool(i.found), degree=int(i.degree), q_terms=int(i.q_terms),
                    expanded_terms=int(i.expanded_terms), residual_terms=int(i.residual_terms),
                    points=int(i.points), bytes=int(i.bytes))

    @property
    def conserved(self):
        return bool(self._i.conserved)

    @property
    def residual(self):
        """P = q(T(x)) - q(x): {exponent tuple over the variables: coefficient} (empty when conserved)"""
        n, k = int(self._i.residual_terms), len(self.variables)
        e, c, m = np.zeros(max(n * k, 1), np.uint8), np.zeros(max(n, 1), np.int64), C.c_uint64()
        _check(_call("tr_cons_law_residual", self._open(), e.ctypes.data, c.ctypes.data, n, C.byref(m)),
               what="residual")
        return {tuple(int(x) for x in e[t * k:(t + 1) * k]): int(c[t]) for t in range(m.value)}

    @property
    def text(self):
        r = sorted(self.residual.items(), key=lambda t: (sum(t[0]), t[0]), reverse=True)
        return _mi.poly_text(r, list(self.variables))

    @property
    def certificate(self):
        """the universal FND-007 law (ring identity of q(T(x)) - q(x)): an owned clone, or None"""
        h = _call("tr_cons_law_certificate", self._open())
        if not h:
            return None
        error = C.c_int32()
        clone = _mc._call("tr_cert_clone", h, C.byref(error))
        return _mc.Certificate._wrap(_mc._made(clone, error, _Refusal(), "law"), self.variables)

    @property
    def counterexample(self):
        """{variable: value} where q(T(x)) != q(x), with "value": P there (exact), or None"""
        i = self._i
        if not i.found:
            return None
        point = {v: int(i.point[q]) for q, v in enumerate(self.variables)}
        return dict(point=point, value=_words([i.value[w] for w in range(int(i.words))]))

    def _cols(self, scope, columns):
        columns = dict(columns or {})
        cols = []
        for v in self.variables:
            name = columns.get(v)
            if name is None:
                cols.append(-1)
                continue
            if name not in scope.names:
                raise ValueError(f"{name!r} is not a column of the scope ({', '.join(scope.names)})")
            cols.append(list(scope.names).index(name))
        unknown = set(columns) - set(self.variables)
        if unknown:
            raise ValueError(f"not variables of the law: {', '.join(sorted(unknown))}")
        return (C.c_int32 * 8)(*(cols + [-1] * (8 - len(cols))))

    def check(self, scope, columns, *, backend="cpu", witnesses=8, budget=None):
        """P = 0 verified exhaustively on every selected row of scope (variables bound to columns: {variable: column};
        a NULL bound column excludes the row): evidence, the FND-007 snapshot certificate when it holds (DISC-002's
        verification; backend "cuda": its kernel), and the first violating rows (the perturbations)"""
        _fresh(scope, "conservation check")
        cols = self._cols(scope, columns)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_cons_law_candidates", self._open(), scope._handle, cols, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        names = [scope.names[c] for c in cols[:len(self.variables)] if c >= 0]
        ids = _mi.Identities._wrap(_made(h, error, refusal, "conservation check"), scope, names, None, None)
        ids.certify(backend=backend, budget=budget)
        item = ids[0]
        rows = np.zeros(max(int(witnesses), 1), np.uint64)
        found, ev = C.c_uint64(), _mc._Evidence()
        _check(_call("tr_cons_law_witnesses", self._handle, scope._handle, cols, C.byref(_budget(budget)),
                     C.byref(ev), rows.ctypes.data, int(witnesses), C.byref(found), C.byref(refusal)), refusal,
               "conservation witnesses")
        return dict(holds=item["status"] == "verified", status=item["status"], evidence=item["evidence"],
                    certificate=item["certificate"], certificate_code=item["certificate_code"],
                    witnesses=[int(r) for r in rows[:found.value]], violations=int(ev.violations),
                    label="snapshot")

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_cons_law_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        i = self.info
        return f"Law(conserved={i['conserved']}, residual={self.text})"


def conserve(q, images, variables, *, budget=None):
    """the conservation check of a quantity q by the transformation x_v -> images[v] (variables not given map to
    themselves): q and the images are polynomials {exponent tuple over variables: int} or [(coefficient,
    {variable: exponent})]"""
    variables = list(variables)
    if not 1 <= len(variables) <= MAX_VARS:
        raise UnsupportedError(f"1..{MAX_VARS} variables")
    if len(set(variables)) != len(variables):
        raise ValueError("distinct variables")
    images = dict(images or {})
    unknown = set(images) - set(variables)
    if unknown:
        raise ValueError(f"images of unknown variables: {', '.join(sorted(map(str, unknown)))}")
    arr, keep = _mi._polys([q], variables)
    imgs = (_mi._Poly * len(variables))()
    for i, v in enumerate(variables):
        if v in images:
            exps, co = _mi._poly_arrays(images[v], variables)
            if not len(co):
                raise ValueError(f"the image of {v!r} is the zero polynomial: give a polynomial (the identity map is "
                                 "the default)")
            imgs[i] = _mi._Poly(len(co), exps.ctypes.data, co.ctypes.data)
            keep += [exps, co]
        else:
            imgs[i] = _mi._Poly(0, None, None)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_cons_conserve", len(variables), arr, imgs, C.byref(_budget(budget)), C.byref(refusal),
              C.byref(error))
    return Law._wrap(_made(h, error, refusal, "conservation"), variables)
