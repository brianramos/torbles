# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-008: complete-plan choice and cost evidence (torbles/math_planner.c, math_planner_gpu.cu).

    from torbles import math_registry as reg, math_planner as mp
    req = reg.Request(reg.Column(db["t"], "x"), [[0, 1, 1]], rows=mask)
    mp.sample(req, backend="cuda")                 # selected rows in a deterministic sample of words, or exactly
    plan = mp.plan(req, devices=("cpu", "cuda"))   # every candidate with its cost components, chosen/rejected and why
    out = mp.run(req, force="histogram@cuda")      # runs exactly the forced representation, or refuses
    mp.verify(catalog, "sum", "s", "x", "y", force="verify@cuda")   # DISC-001 through the planner

Costs are relative work units (amount x coefficient per component), not time. A sample with no selected row is no
evidence of emptiness: only exact=True with a zero count proves it. TORBLES_MATH_PLAN_FORCE="<repr>[@cpu|@cuda|@reuse]
[:cpu|:gpu]" (comma-separated, one per request kind) forces natively; force= does it for one call. A plan the native
planner refuses raises the exception of its code with .plan (the evidence) attached. Opt-in: `import torbles` does
not load it.
"""
import ctypes as C

import numpy as np

from . import algq
from . import math_registry as reg
from .math_budgets import _Budget, _Refusal as _BudgetRefusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["sample", "coefficients", "plan", "run", "plan_verify", "verify", "parse_force"]

STATUSES = {1: "chosen", 2: "admissible", 3: "rejected"}
REASONS = {0: None, 1: "cheapest", 2: "forced", 3: "costlier", 4: "not forced", 5: "refused by the registry",
           6: "over budget", 7: "forced representation refused", 8: "no candidate matches the force",
           9: "no admissible candidate", 10: "request or candidate search refused"}
METHODS = {0: "sampled", 1: "exact", 2: "covered"}
POLYSUM, VERIFY = 16, 8


class _Selectivity(C.Structure):
    _native_ = "tr_planner_selectivity"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "words", "sample_words", "sample_rows", "sample_selected",
                                          "estimate", "lower", "upper")] + [
        (n, C.c_int32) for n in ("exact", "empty_proven", "zero_sample", "method")]


class _Force(C.Structure):
    _native_ = "tr_planner_force"
    _fields_ = [(n, C.c_int32) for n in ("representation", "device", "finalize", "reuse")]


class _Evidence(C.Structure):
    _native_ = "tr_planner_evidence"
    _fields_ = [(n, C.c_int32) for n in ("abi", "request", "ncandidates", "chosen", "forced", "reason", "code",
                                         "devices")] + [
        ("force", _Force), ("refusal", _BudgetRefusal), ("selectivity", _Selectivity),
        ("candidates", reg._Candidate * reg.MAX_CANDIDATES)]


SIZES = {"selectivity": 80, "force": 16, "evidence": 6168}
_native = None


def _api():
    global _native
    if _native is None:
        P, i32, u64, vp = C.POINTER, C.c_int32, C.c_uint64, C.c_void_p
        sig = {
            "sample": (i32, [vp, u64, u64, i32, P(_Selectivity)]),
            "force_parse": (i32, [C.c_char_p, i32, P(_Force)]),
            "choose": (i32, [P(reg._Candidate), i32, P(_Budget), vp, P(_Force), P(_Evidence)]),
            "polysum": (i32, [P(reg._Request), vp, i32, P(_Budget), P(_Selectivity), P(_Force), vp, P(_Evidence)]),
            "verify": (i32, [vp, i32, P(_Budget), P(_Force), vp, P(_Evidence)]),
        }
        native = {}
        for name, (restype, args) in sig.items():
            fn = getattr(algq.lib, "tr_planner_" + name)
            fn.restype, fn.argtypes = restype, args
            native[name] = fn
        _native = native
    return _native


def _selectivity_dict(s, backend=None):
    d = {n: int(getattr(s, n)) for n, _ in s._fields_}
    for k in ("exact", "empty_proven", "zero_sample"):
        d[k] = bool(d[k])
    d["method"] = METHODS[s.method]
    if backend:
        d["backend"] = backend
    return d


def _sample(request, bits, rows, sample_words, exact, backend, budget):
    keep = None
    if request is not None:
        sel, n = request.raw.selected, request.rows
    else:
        if rows is None:
            raise ValueError("rows= (the number of rows) with bits=")
        n = int(rows)
        keep = None if bits is None else reg._pack(bits, n)
        sel = None if keep is None else keep.ctypes.data
    out = _Selectivity()
    if backend == "cpu":
        _check(_api()["sample"](sel, n, int(sample_words), int(bool(exact)), C.byref(out)), what="selectivity sample")
        return out, "cpu"
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    from . import math_gpu
    math_gpu.bind("tr_mgpu_planner_sample", C.c_int, [C.c_void_p, C.c_uint64, C.c_uint64, C.c_int32,
                                                      C.POINTER(_Budget), C.POINTER(_Selectivity),
                                                      C.POINTER(_BudgetRefusal)])
    refusal, b = _BudgetRefusal(), budget or Budget()
    math_gpu.run("tr_mgpu_planner_sample", sel, n, int(sample_words), int(bool(exact)), C.byref(b.raw), C.byref(out),
                 C.byref(refusal), refusal=refusal, what="selectivity sample on the device")
    del keep
    return out, "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def sample(request=None, bits=None, rows=None, sample_words=32, exact=False, backend="cpu", budget=None):
    """the selected rows of a deterministic sample of the selection's words (every word when exact): counts, exact
    bounds, the cost estimate, and whether emptiness is proven (an exact zero only) or merely unobserved"""
    out, where = _sample(request, bits, rows, sample_words, exact, backend, budget)
    return _selectivity_dict(out, where)


def coefficients(overrides=None):
    """the cost coefficient table [module id][cpu, cuda][component] of the registry's planned modules (ids 1 .. 9:
    the modules with candidates; the family modules have none and are not in the table), with overrides
    {(representation, "cpu" | "cuda"): {component: rate}}"""
    table = np.zeros((10, 2, len(reg.COST_COMPONENTS)), dtype=np.float64)
    for m in reg.modules():
        if m["id"] >= len(table):
            continue
        for d, dev in enumerate(("cpu", "cuda")):
            table[m["id"], d] = [m["costs"][dev][c] for c in reg.COST_COMPONENTS]
    for (representation, dev), rates in (overrides or {}).items():
        for comp, value in rates.items():
            table[reg.REPRESENTATIONS[representation], ("cpu", "cuda").index(dev), reg.COST_COMPONENTS.index(comp)] = value
    return table


def parse_force(text, request=0):
    """the native reading of a force text (request 0: strictly its first entry)"""
    f = _Force()
    _check(_api()["force_parse"](None if text is None else text.encode(), request, C.byref(f)),
           what=f"force {text!r}")
    return f


def _force_dict(f):
    if not f.representation:
        return None
    return dict(representation=reg._NAMES_BY_REPR.get(f.representation),
                device={-1: None, 1: "cpu", 2: "cuda"}.get(f.device), finalize={-1: None, 0: "cpu", 1: "gpu"}[f.finalize],
                reuse={-1: None, 0: False, 1: True}[f.reuse])


def _candidate(c):
    d = reg._candidate_dict(c)
    d.update(status=STATUSES.get(c.status), reason=REASONS.get(c.reason),
             budget=None if not c.budget.refused else dict(resource=RESOURCES[c.budget.resource],
                                                            need=int(c.budget.need), limit=int(c.budget.limit)),
             costs=dict(zip(reg.COST_COMPONENTS, (float(v) for v in c.cost))), total=float(c.total))
    return d


def _evidence(ev, selectivity_backend=None):
    cands = [_candidate(ev.candidates[i]) for i in range(ev.ncandidates)]
    return dict(request={POLYSUM: "polysum", VERIFY: "verify"}.get(ev.request), code=ev.code,
                reason=REASONS.get(ev.reason), chosen=None if ev.chosen < 0 else ev.chosen,
                choice=None if ev.chosen < 0 else cands[ev.chosen], forced=bool(ev.forced), force=_force_dict(ev.force),
                devices=[d for b, d in ((1, "cpu"), (2, "cuda")) if ev.devices & b],
                refusal=None if not ev.refusal.refused else dict(resource=RESOURCES[ev.refusal.resource],
                                                                need=int(ev.refusal.need), limit=int(ev.refusal.limit)),
                selectivity=_selectivity_dict(ev.selectivity, selectivity_backend), candidates=cands,
                cost_unit="relative work units; not time")


def _refuse(ev, evidence, what):
    """raise the plan's refusal: its code's exception, with .plan attached"""
    text = what + ": " + (evidence["reason"] or "refused")
    if evidence["chosen"] is None and evidence["force"] is not None:
        match = next((c for c in evidence["candidates"] if c["representation"] == evidence["force"]["representation"]),
                     None)
        if match is not None and match["refusal"]:
            text += " (" + reg._refusal_text(match["refusal"]) + ")"
    try:
        _check(ev.code, ev.refusal if ev.refusal.refused else None, text)
    except Exception as exc:
        exc.plan = evidence
        raise


def _force_arg(force):
    return None if force is None else C.byref(parse_force(force, 0))


def _coef_arg(table):
    if table is None:
        return None, None
    t = np.ascontiguousarray(table, dtype=np.float64)
    if t.shape != (10, 2, len(reg.COST_COMPONENTS)):
        raise ValueError("coefficients: a (10, 2, 12) table (coefficients())")
    return t, t.ctypes.data


def plan(request, devices=("cpu",), state=None, budget=None, force=None, sample_words=32, exact=False,
         sample_backend="cpu", coefficients=None):
    """the plan of a request: the selectivity sample (CPU or CUDA kernel), then every candidate of the registry with
    its cost components, status and reason; refused plans raise with .plan"""
    raw, where = _sample(request, None, None, sample_words, exact, sample_backend, budget)
    b = budget or Budget()
    ev = _Evidence()
    keep, coef = _coef_arg(coefficients)
    rc = _api()["polysum"](C.byref(request.raw), state._open() if state is not None else None, reg._devices(devices),
                           C.byref(b.raw), C.byref(raw), _force_arg(force), coef, C.byref(ev))
    evidence = _evidence(ev, where)
    del keep
    if rc:
        _refuse(ev, evidence, "plan")
    return evidence


def run(request, devices=("cpu",), state=None, budget=None, force=None, sample_words=32, exact=False,
        sample_backend="cpu", coefficients=None):
    """plan, then execute the chosen candidate through the entry point its module declares:
    {"values", "selected", "valid", "nulls", "explanation", "plan"}"""
    evidence = plan(request, devices, state, budget, force, sample_words, exact, sample_backend, coefficients)
    c = evidence["choice"]
    out = reg.execute(request, c["representation"], backend=c["device"], finalize=c["finalize"],
                      state=state if c["reuse"] else None, budget=budget)
    out["plan"] = evidence
    return out


def plan_verify(catalog, module, out, left, right=None, devices=("cpu",), force=None, budget=None, coefficients=None):
    """the plan of a DISC-001 verification: the CPU verifier or the CUDA kernel (device copies budgeted)"""
    from . import math_disc
    b = budget or Budget()
    ev = _Evidence()
    keep, coef = _coef_arg(coefficients)
    with catalog._lock:
        v = math_disc.view(catalog, module, out, left, right)
        rc = _api()["verify"](C.byref(v), reg._devices(devices), C.byref(b.raw), _force_arg(force), coef, C.byref(ev))
    evidence = _evidence(ev, "cpu")
    del keep
    if rc:
        _refuse(ev, evidence, "verification plan")
    return evidence


def verify(catalog, module, out, left, right=None, *, scale=1, bias=0, devices=("cpu",), force=None, budget=None,
           coefficients=None):
    """plan and run a DISC-001 verification: {"evidence", "explanation", "plan"}"""
    evidence = plan_verify(catalog, module, out, left, right, devices, force, budget, coefficients)
    c = evidence["choice"]
    result = reg.verify(catalog, module, out, left, right, scale=scale, bias=bias, backend=c["device"], budget=budget)
    result["plan"] = evidence
    return result
