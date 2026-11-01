# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-006: fused multi-output basis reduction (torbles/math_reduce.c, math_reduce_gpu.cu).

    from torbles import math_bank as mbk, math_bool as mb, math_reduce as mr
    scope = mbk.Scope.from_table(db["f"], ["x", "d", "y", "w"], group_by=["k"], rows=mask)
    req = mr.Request(scope, conditions=[mb.condition("x", ">", 5), mb.condition("y", "is null")],
                     groupings={"c": mbk.Scope.from_table(db["f"], [], group_by=["c"], rows=mask),
                                "all": mr.rollup(mbk.Scope.from_table(db["f"], ["x", "d", "y", "w"], rows=mask))})
    big, nul = mb.variables(2)                       # predicates: math_bool expressions over the conditions
    req.bank("A", ["1", "x", "x*d", "d^2"], where=big)              # a FND-004 basis, mode "sql" or "joint"
    req.count("B", where=big & ~nul, group="c")                     # rows only
    req.histogram("H", ["x"], weight="w", where=nul, group="all")   # a DIST-001 histogram (sparse or dense)
    plan = req.compile()                             # validated, merged, proven, budgeted natively
    plan.choose(devices=("cpu", "cuda"))             # FND-008-style evidence: fused / repeated x CPU / CUDA, costed
    res = plan.run()                                 # the cheapest admissible candidate; force="repeated@cpu" etc.
    res["A"].state(g); res["H"].cells(g); res["B"].table(); res.stats; res.plan

Every output equals the independent reduction of its derived scope: the scope's rows where its predicate is TRUE
(SQL WHERE), grouped by its grouping (FND-004 bank, DIST-001 histogram). The fused plan traverses the rows once,
sharing the selection, decoded columns, predicate words, group ids and products; the repeated plan runs the
independent reductions one by one. Refusals are atomic (no output of a refused run exists) and carry .failed (the
output's name) and .plan (the evidence). TORBLES_MATH_REDUCE_FORCE="fused|repeated[@cpu|@cuda]" forces natively.
Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_bool as mb
from . import math_dist as md
from . import math_registry as reg
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["Request", "Plan", "Result", "rollup", "PLANS", "MAX_OUTPUTS", "MAX_GROUPINGS", "MAX_CONDITIONS"]

MAX_OUTPUTS, MAX_GROUPINGS, MAX_CONDITIONS, MAX_NODES, MAX_TERMS, MAX_PRODUCTS = 1024, 16, 64, 65536, 65536, 4096
BANK, HIST = 1, 2
PLANS = {"fused": 1, "repeated": 2}
DEVICES = {"cpu": 1, "cuda": 2}
NEEDS = {("fused", "cpu"): 0, ("fused", "cuda"): 1, ("repeated", "cpu"): 2, ("repeated", "cuda"): 3,
         ("fused", "kernel"): 4}
_U64 = (1 << 64) - 1
P = C.POINTER


class _Grouping(C.Structure):
    _native_ = "tr_reduce_grouping"
    _fields_ = [("scope", C.c_void_p), ("map", C.c_void_p), ("ngroups", C.c_uint64)]


class _Output(C.Structure):
    _native_ = "tr_reduce_output"
    _fields_ = [(n, C.c_int32) for n in ("kind", "filter", "grouping", "mode", "first", "nterms")] + [
        ("reserved", C.c_int32 * 2), ("hist", md._Spec)]


class _Request(C.Structure):
    _native_ = "tr_reduce_request"
    _fields_ = [(n, C.c_int32) for n in ("nconds", "nnodes", "ngroupings", "noutputs", "nexps", "reserved")] + [
        (n, C.c_void_p) for n in ("conds", "nodes", "groupings", "outputs", "exps")]


class _GroupView(C.Structure):
    _native_ = "tr_reduce_group_view"
    _fields_ = [("ids", C.c_void_p), ("map", C.c_void_p), ("source_groups", C.c_uint64), ("ngroups", C.c_uint64)]


class _View(C.Structure):
    _native_ = "tr_reduce_view"
    _fields_ = [("scope", mbk._View)] + [(n, C.c_void_p) for n in ("conds", "nodes", "schedule", "condlist",
                                                                  "products", "terms", "outs", "hplans")] + [
        ("groups", _GroupView * MAX_GROUPINGS)] + [
        (n, C.c_int32) for n in ("nconds", "nnodes", "nsched", "ncondlist", "nproducts", "nterms", "noutputs", "nhist",
                                 "ngroupings", "degree")] + [
        ("used", C.c_uint32), ("nullable", C.c_uint32), ("arena64", C.c_uint64), ("arena128", C.c_uint64)]


class _Arena(C.Structure):
    _native_ = "tr_reduce_arena"
    _fields_ = [("a64", C.c_void_p), ("a128", C.c_void_p), ("n64", C.c_uint64), ("n128", C.c_uint64)]


_STAT_FIELDS = ("scans", "kernels", "native_calls", "decoded_bytes", "workspace_bytes", "transfer_bytes", "products",
                "accumulations", "rows_visited", "words_visited", "items", "state_bytes")


class _Stats(C.Structure):
    _native_ = "tr_reduce_stats"
    _fields_ = [(n, C.c_uint64) for n in _STAT_FIELDS] + [("plan", C.c_int32), ("device", C.c_int32)]


class _PlanInfo(C.Structure):
    _native_ = "tr_reduce_plan_info"
    _fields_ = [(n, C.c_int32) for n in ("noutputs", "nconds", "nnodes", "nsched", "nproducts", "nterms",
                                         "ngroupings", "nhist", "merged_conds", "merged_nodes")] + [
        ("used", C.c_uint32), ("nullable", C.c_uint32)] + [
        (n, C.c_uint64) for n in ("rows", "selected", "words", "arena64", "arena128", "states", "bytes", "term_rows")]


class _OutputInfo(C.Structure):
    _native_ = "tr_reduce_output_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "mode", "grouping", "filter", "nterms", "k", "repr", "weighted")] + [
        (n, C.c_uint64) for n in ("ngroups", "present", "rows", "cells", "state_bytes")] + [
        ("lo", C.c_int64 * 3), ("span", C.c_uint64 * 3), ("dim", C.c_uint64 * 3), ("nullslot", C.c_int32 * 3),
        ("scale", C.c_int32 * 3), ("col", C.c_int32 * 3), ("weight", C.c_int32), ("wscale", C.c_int32),
        ("reserved", C.c_int32)]


class _Force(C.Structure):
    _native_ = "tr_reduce_force"
    _fields_ = [("plan", C.c_int32), ("device", C.c_int32), ("reserved", C.c_int32 * 2)]


class _Evidence(C.Structure):
    _native_ = "tr_reduce_evidence"
    _fields_ = [(n, C.c_int32) for n in ("abi", "ncandidates", "chosen", "forced", "reason", "code", "devices",
                                         "module")] + [
        ("force", _Force), ("refusal", _Refusal), ("candidates", reg._Candidate * 4)]


for _s, _n in ((_Grouping, 24), (_Output, 128), (_Request, 64), (_GroupView, 32), (_View, 2736), (_Arena, 32),
               (_Stats, 104), (_PlanInfo, 112), (_OutputInfo, 192), (_Force, 16), (_Evidence, 72 + 4 * 376)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        sig = {
            "tr_reduce_compile": (vp, [vp, P(_Request), P(_Budget), P(_Refusal), e, e]),
            "tr_reduce_plan_free": (None, [vp]),
            "tr_reduce_plan_get_info": (i32, [vp, P(_PlanInfo)]),
            "tr_reduce_get_view": (i32, [vp, P(_View)]),
            "tr_reduce_need": (i32, [vp, i32, P(_Need)]),
            "tr_reduce_choose": (i32, [vp, i32, P(_Budget), vp, P(_Force), P(_Evidence)]),
            "tr_reduce_force_parse": (i32, [C.c_char_p, P(_Force)]),
            "tr_reduce_coefficients": (i32, [vp, i32]),
            "tr_reduce_run": (vp, [vp, i32, P(_Budget), P(_Refusal), e, e]),
            "tr_reduce_stage_create": (vp, [vp, i32, P(_Budget), P(_Refusal), e]),
            "tr_reduce_stage_arena": (i32, [vp, P(_Arena)]),
            "tr_reduce_stage_put_bank": (i32, [vp, i32, vp]),
            "tr_reduce_stage_put_hist": (i32, [vp, i32, vp, P(_Refusal)]),
            "tr_reduce_stage_finish": (vp, [vp, i32, P(_Stats), P(_Budget), P(_Refusal), e, e]),
            "tr_reduce_stage_free": (None, [vp]),
            "tr_reduce_derive": (vp, [vp, i32, vp, u64, P(md._Spec), P(i32), P(_Budget), P(_Refusal), e]),
            "tr_reduce_result_get_stats": (i32, [vp, P(_Stats)]),
            "tr_reduce_result_output": (i32, [vp, i32, P(_OutputInfo)]),
            "tr_reduce_result_bank": (i32, [vp, i32, P(mbk._States)]),
            "tr_reduce_result_hist": (i32, [vp, i32, P(md._States)]),
            "tr_reduce_result_equal": (i32, [vp, vp]),
            "tr_reduce_result_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_reduce_fused", C.c_int, [P(_View), P(_Need), P(_Budget), P(_Arena), P(_Stats), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _raise(code, refusal, what, *, failed=None, plan=None):
    """the exception of a native refusal, with .failed (the output's name or None) and .plan (evidence or None)"""
    try:
        _check(code, refusal if refusal is not None and refusal.resource >= 0 else None, what)
    except Exception as exc:
        exc.failed, exc.plan = failed, plan
        raise
    raise RuntimeError(f"{what}: native call returned no object (code {code})")


# ---------------------------------------------------------------------- groupings and predicates
class rollup:
    """a grouping that maps groups onto coarser ones: rollup(coarse_scope) (the map from the base scope's grouping to
    coarse_scope's, natively: Scope.rollup_map), or rollup(map=ids, groups=G[, source=scope]) (a map over the groups of
    source, default the base scope; keys of a raw map are (g,))"""

    def __init__(self, coarse=None, *, map=None, groups=None, source=None):
        if (coarse is None) == (map is None):
            raise ValueError("give a coarse scope, or map= with groups=")
        self.coarse, self.map, self.groups, self.source = coarse, map, groups, source


def _compile_nodes(roots):
    """the nodes of several predicate expressions (math_bool Expr) as one DAG: post-order, a Python object reached
    twice is one node; returns (nodes, [root index or -1 per root])"""
    index, nodes = {}, []
    out = []
    for root in roots:
        if root is None:
            out.append(-1)
            continue
        stack = [(mb._expr(root), False)]
        while stack:
            e, ready = stack.pop()
            if not isinstance(e, mb.Expr):
                raise TypeError(f"{e!r} is not a predicate (math_bool.var(i), &, |, ^, ~)")
            if id(e) in index:
                continue
            kids = [] if e.op in (mb.CONST, mb.VAR) else ([e.a] if e.op == mb.NOT else [e.a, e.b])
            if not ready:
                stack.append((e, True))
                stack.extend((c, False) for c in reversed(kids) if id(c) not in index)
                continue
            a = e.a if e.op in (mb.CONST, mb.VAR) else index[id(e.a)]
            b = index[id(e.b)] if e.op in (mb.AND, mb.OR, mb.XOR) else -1
            index[id(e)] = len(nodes)
            nodes.append((e.op, a, b))
        out.append(index[id(mb._expr(root))])
    return nodes, out


# ---------------------------------------------------------------------- the request
class Request:
    """Outputs over one scope: banks (FND-004 bases), counts and histograms (DIST-001), each with a predicate over the
    request's conditions (where=, a math_bool expression; None: every selected row) and a grouping (group=: None for
    the scope's own grouping, or a name of groupings)."""

    def __init__(self, scope, *, conditions=(), groupings=None):
        self.scope = scope
        self.conditions = list(conditions)
        if len(self.conditions) > MAX_CONDITIONS:
            raise ValueError(f"at most {MAX_CONDITIONS} conditions")
        self.groupings = {}
        for name, g in (groupings or {}).items():
            if not isinstance(g, (mbk.Scope, rollup)):
                raise TypeError(f"grouping {name!r}: a Scope of the same rows or a rollup(...)")
            self.groupings[str(name)] = g
        if len(self.groupings) + 1 > MAX_GROUPINGS:
            raise ValueError(f"at most {MAX_GROUPINGS - 1} named groupings")
        self.outputs = []
        self._names = set()

    def _add(self, name, kind, where, group, **spec):
        name = str(name)
        if name in self._names:
            raise ValueError(f"output {name!r} exists")
        if group is not None and group not in self.groupings:
            raise ValueError(f"no grouping {group!r} (groupings: {', '.join(self.groupings) or 'none'})")
        if len(self.outputs) >= MAX_OUTPUTS:
            raise ValueError(f"at most {MAX_OUTPUTS} outputs")
        self._names.add(name)
        self.outputs.append(dict(name=name, kind=kind, where=where, group=group, **spec))
        return self

    def bank(self, name, basis, *, where=None, mode="sql", group=None):
        """a FND-004 moment basis (exponent tuples, {name: exponent} or 'x*y^2' / '1'); mode "sql" or "joint" """
        exps = mbk._basis(self.scope.names, basis)
        return self._add(name, BANK, where, group, exps=exps, mode=mbk._mode(mode))

    def count(self, name, *, where=None, group=None):
        """the rows per group (a bank without terms)"""
        return self._add(name, BANK, where, group, exps=np.zeros((0, len(self.scope.names)), np.uint8), mode=0)

    def histogram(self, name, columns, *, weight=None, where=None, repr="sparse", domain=None, group=None):
        """a DIST-001 histogram over 1..3 scope columns (dense: the base scope's domain, or domain=[(lo, hi, null)])"""
        spec, cols = md._spec(self.scope, columns, weight, repr, domain, "auto")
        return self._add(name, HIST, where, group, spec=spec, columns=cols, weight=weight)

    def _native(self):
        """the native request and the objects it points into (kept alive by the caller)"""
        names = self.scope.names
        conds, nconds = mb._conditions(self.conditions, names)
        nodes, roots = _compile_nodes([o["where"] for o in self.outputs])
        if len(nodes) > MAX_NODES:
            raise ValueError(f"at most {MAX_NODES} predicate nodes")
        narr = (mb._Node * max(len(nodes), 1))()
        for i, (op, a, b) in enumerate(nodes):
            narr[i] = mb._Node(op, a, b, 0)
        order = [None] + list(self.groupings)
        garr = (_Grouping * len(order))()
        keep = [conds, narr, garr]
        self._keys = [self.scope]
        for i, name in enumerate(order):
            if name is None:
                garr[i] = _Grouping(None, None, 0)
                continue
            g = self.groupings[name]
            if isinstance(g, mbk.Scope):
                g._open()
                garr[i] = _Grouping(g._handle, None, 0)
                self._keys.append(g)
                keep.append(g)
                continue
            if g.coarse is not None:
                fmap, ngroups, src, keys = self.scope.rollup_map(g.coarse), g.coarse.groups, None, g.coarse
                keep.append(g.coarse)
            else:
                src = g.source
                fmap = np.ascontiguousarray(g.map, dtype=np.uint32)
                ngroups, keys = int(g.groups), None
                base = src if src is not None else self.scope
                if fmap.shape != (base.groups,):
                    raise ValueError(f"grouping {name!r}: the map has one coarse id per group of its source")
            fmap = np.ascontiguousarray(fmap, dtype=np.uint32)
            keep.append(fmap)
            if src is not None:
                src._open()
                keep.append(src)
            garr[i] = _Grouping(None if src is None else src._handle, fmap.ctypes.data, int(ngroups))
            self._keys.append(keys)
        outs = (_Output * max(len(self.outputs), 1))()
        mats, first = [], 0
        for i, o in enumerate(self.outputs):
            x = outs[i]
            x.kind, x.filter = o["kind"], roots[i]
            x.grouping = 0 if o["group"] is None else order.index(o["group"])
            if o["kind"] == BANK:
                x.mode, x.first, x.nterms = o["mode"], first, len(o["exps"])
                mats.append(o["exps"])
                first += len(o["exps"])
            else:
                x.hist = o["spec"]
        exps = (np.ascontiguousarray(np.concatenate(mats, axis=0), dtype=np.uint8) if mats else
                np.zeros((0, len(names)), np.uint8))
        keep += [outs, exps]
        rq = _Request(nconds, len(nodes), len(order), len(self.outputs), first, 0, C.cast(conds, C.c_void_p),
                      C.cast(narr, C.c_void_p), C.cast(garr, C.c_void_p), C.cast(outs, C.c_void_p),
                      exps.ctypes.data if exps.size else None)
        return rq, keep

    def compile(self, *, budget=None):
        """the native plan (validated, merged, proven, the fused CPU need checked); raises the refusal with .failed"""
        self._fresh()
        rq, keep = self._native()
        error, failed, refusal = C.c_int32(), C.c_int32(-1), _Refusal()
        h = _call("tr_reduce_compile", self.scope._handle, C.byref(rq), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(failed), C.byref(error))
        if not h or error.value:
            if h:
                _call("tr_reduce_plan_free", h)
            _raise(error.value or -2, refusal, "fused reduction",
                   failed=self.outputs[failed.value]["name"] if 0 <= failed.value < len(self.outputs) else None)
        return Plan(h, self, keep)

    def _fresh(self):
        scopes = [self.scope] + [g for g in self.groupings.values() if isinstance(g, mbk.Scope)]
        scopes += [g.coarse for g in self.groupings.values() if isinstance(g, rollup) and g.coarse is not None]
        scopes += [g.source for g in self.groupings.values() if isinstance(g, rollup) and g.source is not None]
        for s in scopes:
            if s.source == "table":
                s.require_fresh("fused reduction")

    def run(self, **kwargs):
        return self.compile(budget=kwargs.get("budget")).run(**kwargs)


def _plan_info(raw):
    d = {n: int(getattr(raw, n)) for n, _ in raw._fields_}
    d["shared_products"] = d["term_rows"] - d["nproducts"]
    return d


def _candidate(c):
    return dict(plan={1: "fused", 2: "repeated"}.get(c.representation), device={1: "cpu", 2: "cuda"}.get(c.device),
                module=int(c.module) or None, batch=int(c.batch), status=reg_status(c.status),
                reason=reg_reason(c.reason), code=int(c.code),
                budget=None if not c.budget.refused else dict(resource=RESOURCES[c.budget.resource],
                                                               need=int(c.budget.need), limit=int(c.budget.limit)),
                need={n: int(c.need.need[i]) for i, n in enumerate(RESOURCES) if c.need.need[i]},
                amounts=dict(zip(reg.COST_COMPONENTS, (int(v) for v in c.amount))),
                costs=dict(zip(reg.COST_COMPONENTS, (float(v) for v in c.cost))), total=float(c.total),
                shared=[n for b, n in enumerate(reg.COST_COMPONENTS) if c.shared >> b & 1])


def reg_status(s):
    return {1: "chosen", 2: "admissible", 3: "rejected"}.get(s)


def reg_reason(r):
    return {0: None, 1: "cheapest", 2: "forced", 3: "costlier", 4: "not forced", 5: "refused", 6: "over budget",
            7: "forced plan refused", 8: "no candidate matches the force", 9: "no admissible candidate",
            10: "request or force refused"}.get(r)


def _evidence(ev):
    cands = [_candidate(ev.candidates[i]) for i in range(ev.ncandidates)]
    force = None if not ev.force.plan else dict(plan={1: "fused", 2: "repeated"}[ev.force.plan],
                                                device={-1: None, 1: "cpu", 2: "cuda"}.get(ev.force.device))
    return dict(code=int(ev.code), reason=reg_reason(ev.reason), chosen=None if ev.chosen < 0 else int(ev.chosen),
                choice=None if ev.chosen < 0 else cands[ev.chosen], forced=bool(ev.forced), force=force,
                devices=[d for b, d in ((1, "cpu"), (2, "cuda")) if ev.devices & b], module=int(ev.module) or None,
                refusal=None if not ev.refusal.refused else dict(resource=RESOURCES[ev.refusal.resource],
                                                                need=int(ev.refusal.need), limit=int(ev.refusal.limit)),
                candidates=cands, cost_unit="relative work units; not time", cost_basis="estimate")


def coefficients(overrides=None):
    """the built-in coefficient table [plan - 1][device - 1][component] with overrides {(plan, device): {component:
    rate}} (plan "fused" / "repeated", device "cpu" / "cuda")"""
    table = np.zeros((2, 2, len(reg.COST_COMPONENTS)), dtype=np.float64)
    _check(_call("tr_reduce_coefficients", table.ctypes.data, table.size), what="coefficients")
    for (plan, dev), rates in (overrides or {}).items():
        for comp, value in rates.items():
            table[PLANS[plan] - 1, DEVICES[dev] - 1, reg.COST_COMPONENTS.index(comp)] = float(value)
    return table


def parse_force(text):
    f = _Force()
    _check(_call("tr_reduce_force_parse", None if text is None else str(text).encode(), C.byref(f)),
           what=f"force {text!r}")
    return f


# ---------------------------------------------------------------------- plans
class Plan:
    """A compiled native plan (immutable): every output resolved, conditions and nodes merged, products shared,
    bounds proven. It reads its scopes, which it keeps alive."""

    def __init__(self, handle, request, keep):
        self._handle, self.request, self._keep = handle, request, keep
        self.names = [o["name"] for o in request.outputs]

    def _open(self):
        if not self._handle:
            raise ValueError("the plan is closed")
        return self._handle

    @property
    def info(self):
        raw = _PlanInfo()
        _check(_call("tr_reduce_plan_get_info", self._open(), C.byref(raw)), what="plan")
        return _plan_info(raw)

    def need(self, plan="fused", backend="cpu"):
        """{resource: need} of the fused / repeated plan on the CPU / CUDA: what execute checks before it allocates
        anything, an upper bound of every later check of the run but the occupied cells of sparse histograms (checked
        once counted); ("fused", "kernel"): what the fused CUDA entry point itself checks (its device copies, scratch)"""
        raw = _Need()
        _check(_call("tr_reduce_need", self._open(), NEEDS[(plan, backend)], C.byref(raw)), what="needs")
        return {n: int(raw.need[i]) for i, n in enumerate(RESOURCES) if raw.need[i]}

    def _choose(self, devices, force, coefficients, budget):
        mask = 0
        for d in devices:
            mask |= DEVICES[d]
        ev = _Evidence()
        f = None if force is None else C.byref(parse_force(force))
        table = None
        if coefficients is not None:
            table = np.ascontiguousarray(coefficients, dtype=np.float64)
            if table.shape != (2, 2, len(reg.COST_COMPONENTS)):
                raise ValueError("coefficients: a [2 plans][2 devices][12 components] table (math_reduce.coefficients)")
        code = _call("tr_reduce_choose", self._open(), mask, C.byref(_budget(budget)),
                     None if table is None else table.ctypes.data, f, C.byref(ev))
        return code, ev

    def choose(self, devices=("cpu",), force=None, coefficients=None, budget=None):
        """the costed candidates and the choice (never raises for a refused plan: see "code" and "reason")"""
        _, ev = self._choose(devices, force, coefficients, budget)
        return _evidence(ev)

    def run(self, *, devices=("cpu",), force=None, coefficients=None, budget=None):
        """runs the chosen candidate (the cheapest admissible one, or the force's) and returns the Result; a refused
        plan or run raises with .plan and .failed; nothing of a refused run is exposed"""
        self.request._fresh()
        code, ev = self._choose(devices, force, coefficients, budget)
        evidence = _evidence(ev)
        if code or ev.chosen < 0:
            _raise(code or -12, ev.refusal, "fused reduction plan: " + (evidence["reason"] or "refused"), plan=evidence)
        choice = evidence["choice"]
        return self.execute(choice["plan"], choice["device"], budget=budget, evidence=evidence)

    def execute(self, plan="fused", backend="cpu", *, budget=None, evidence=None):
        """runs exactly this plan on this backend (no choice); its need (Plan.need) is checked before anything is
        allocated (natively: tr_reduce_run on the CPU, tr_math_budget_check of tr_reduce_need for the CUDA paths)"""
        self.request._fresh()
        if plan not in PLANS or backend not in DEVICES:
            raise ValueError("plan is 'fused' or 'repeated', backend 'cpu' or 'cuda'")
        b = _budget(budget)
        error, failed, refusal = C.c_int32(), C.c_int32(-1), _Refusal()
        if backend == "cuda":
            raw = _Need()
            _check(_call("tr_reduce_need", self._open(), NEEDS[(plan, "cuda")], C.byref(raw)), what="needs")
            code = mbk._call("tr_math_budget_check", C.byref(b), C.byref(raw), C.byref(refusal))
            if code:
                _raise(code, refusal, f"fused reduction ({plan}@cuda)", plan=evidence)
        if backend == "cpu":
            h = _call("tr_reduce_run", self._open(), PLANS[plan], C.byref(b), C.byref(refusal), C.byref(failed),
                      C.byref(error))
            device = dict(name="native CPU", emulated=False)
        elif plan == "fused":
            h, device = self._fused_cuda(b, refusal, failed, error)
        else:
            h, device = self._repeated_cuda(budget, refusal, failed, error, raw)
        if not h or error.value:
            if h:
                _call("tr_reduce_result_free", h)
            _raise(error.value or -2, refusal, f"fused reduction ({plan}@{backend})", plan=evidence,
                   failed=self.names[failed.value] if 0 <= failed.value < len(self.names) else None)
        label = backend if backend == "cpu" else ("cuda-emulated" if device["emulated"] else "cuda")
        return Result(h, self, plan, label, device["name"], evidence)

    def _fused_cuda(self, b, refusal, failed, error):
        gpu = _gpu()
        view, need = _View(), _Need()
        _check(_call("tr_reduce_get_view", self._open(), C.byref(view)), what="plan view")
        _check(_call("tr_reduce_need", self._open(), NEEDS[("fused", "cuda")], C.byref(need)), what="needs")
        base = _Need()
        _check(_call("tr_reduce_need", self._open(), NEEDS[("fused", "cpu")], C.byref(base)), what="needs")
        stage = _call("tr_reduce_stage_create", self._open(), PLANS["fused"], C.byref(b), C.byref(refusal),
                      C.byref(error))
        if not stage:
            return None, gpu.device()
        try:
            arena, stats = _Arena(), _Stats()
            _check(_call("tr_reduce_stage_arena", stage, C.byref(arena)), what="stage")
            gpu.run("tr_mgpu_reduce_fused", C.byref(view), C.byref(base), C.byref(b), C.byref(arena), C.byref(stats),
                    C.byref(refusal), refusal=refusal, what="fused reduction on the device")
        except Exception:
            _call("tr_reduce_stage_free", stage)
            raise
        h = _call("tr_reduce_stage_finish", stage, DEVICES["cuda"], C.byref(stats), C.byref(b), C.byref(refusal),
                  C.byref(failed), C.byref(error))
        return h, gpu.device()

    def _repeated_cuda(self, budget, refusal, failed, error, need):
        """the independent reductions one by one on the device: each output's derived scope (natively), then FND-004's
        or DIST-001's own CUDA build of it (math_bank.build / math_dist.histogram, backend "cuda"), its states put
        into the stage; a refusal of any output frees everything"""
        from . import math_gpu
        b = _budget(budget)
        stage = _call("tr_reduce_stage_create", self._open(), PLANS["repeated"], C.byref(b), C.byref(refusal),
                      C.byref(error))
        if not stage:
            return None, math_gpu.device()
        stats = _Stats()
        stats.workspace_bytes = need.need[RESOURCES.index("bytes")]          # the plan's need (Plan.need)
        names = self.request.scope.names
        current = [-1]
        try:
            for o, spec in enumerate(self.request.outputs):
                current[0] = o
                exps = np.zeros(1024 * 32, dtype=np.uint8)
                dspec, nc, err, r = md._Spec(), C.c_int32(), C.c_int32(), _Refusal()
                d = _call("tr_reduce_derive", self._open(), o, exps.ctypes.data, exps.size, C.byref(dspec),
                          C.byref(nc), C.byref(b), C.byref(r), C.byref(err))
                if not d or err.value:
                    failed.value = o
                    _raise(err.value or -2, r, f"derived scope of {spec['name']!r}", failed=spec["name"])
                ds = mbk.Scope._new()
                ds._handle, ds.source, ds.table, ds.keys, ds._rows, ds._dims = d, "arrays", None, (), None, ()
                cols = [i for i in range(len(names)) if _reads(spec, names, i)]
                ds.names = tuple(names[i] for i in cols)
                try:
                    dev, put = _Need(), _Refusal(resource=-1)
                    if spec["kind"] == BANK:
                        m = len(spec["exps"])
                        terms = [tuple(int(v) for v in exps[j * nc.value:(j + 1) * nc.value]) for j in range(m)]
                        dexps = mbk._basis(ds.names, terms)
                        _check(mbk._call("tr_bank_need", ds._handle, mbk._exps_ptr(dexps), m, spec["mode"],
                                         mbk._NEED_DEVICE, C.byref(dev)), what="bank needs")
                        out = mbk.build(ds, terms, {0: "sql", 1: "joint"}[spec["mode"]], backend="cuda", budget=budget)
                        rc = _call("tr_reduce_stage_put_bank", stage, o, out._handle)
                        # math_bank_gpu.cu: the rows, the joint flags (JOINT), the terms; nothing for no selected row
                        launches = (1 + (spec["mode"] == 1) + (m > 0)) if ds.selected else 0
                    else:
                        coords = [ds.names[dspec.col[c]] for c in range(dspec.k)]
                        weight = ds.names[dspec.weight] if dspec.weight >= 0 else None
                        dom = ([(int(dspec.lo[c]), int(dspec.hi[c]), int(dspec.nullslot[c])) for c in range(dspec.k)]
                               if dspec.repr == md.REPRS["dense"] else None)
                        hs, cap = md._spec(ds, coords, weight, "dense" if dom else "sparse", dom, "auto")[0], C.c_uint64()
                        _check(md._call("tr_dist_hist_need", ds._handle, C.byref(hs), md._NEED_DEVICE, C.byref(dev),
                                        C.byref(cap)), what="histogram needs")
                        out = md.histogram(ds, coords, weight=weight, repr="dense" if dom else "sparse", domain=dom,
                                           backend="cuda", budget=budget)
                        rc = _call("tr_reduce_stage_put_hist", stage, o, out._handle, C.byref(put))
                        # math_dist_gpu.cu: the rows and the cells, sparse: occupancy, scan, compaction
                        launches = (2 + 3 * (dom is None)) if ds.selected else 0
                    out.close()
                    if rc:
                        failed.value = o
                        _raise(rc, put, f"output {spec['name']!r}", failed=spec["name"])
                    passes = 3 if spec["where"] is not None else 2       # masks (when filtered), derive, reduction
                    stats.scans += passes
                    stats.native_calls += passes
                    stats.kernels += launches
                    stats.decoded_bytes += ds.rows * nc.value * 9
                    stats.transfer_bytes += int(dev.need[RESOURCES.index("bytes")])
                    stats.rows_visited += ds.selected
                    stats.words_visited += (ds.rows + 63) // 64 * passes
                finally:
                    ds.close()
        except Exception as exc:
            _call("tr_reduce_stage_free", stage)            # nothing of a refused run is exposed
            if getattr(exc, "failed", None) is None and 0 <= current[0] < len(self.names):
                exc.failed = self.names[current[0]]
            raise
        h = _call("tr_reduce_stage_finish", stage, DEVICES["cuda"], C.byref(stats), C.byref(b), C.byref(refusal),
                  C.byref(failed), C.byref(error))
        return h, math_gpu.device()

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_reduce_plan_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _reads(spec, names, i):
    """whether output spec reads scope column i (its derived scope's columns, in base order)"""
    if spec["kind"] == BANK:
        return bool(len(spec["exps"])) and bool(np.any(spec["exps"][:, i]))
    s = spec["spec"]
    return any(s.col[c] == i for c in range(s.k)) or s.weight == i


# ---------------------------------------------------------------------- results
def _i128(lo, hi):
    return [(int(h) << 64) | (int(l) & _U64) for l, h in zip(lo.tolist(), hi.tolist())]


_NP = {C.c_uint64: np.uint64, C.c_int64: np.int64, C.c_uint8: np.uint8}


def _arr(ptr, ctype, n):
    """a copy of n native values (empty without values)"""
    if not n or not ptr:
        return np.zeros(0, dtype=_NP[ctype])
    return np.ctypeslib.as_array((ctype * n).from_address(ptr)).copy()


class Output:
    """One output of a Result: its states exactly (read with state / states / table for a bank, cells / group / table
    for a histogram), its info and its keys."""

    def __init__(self, result, index, spec, keys):
        self._result, self.index, self.spec, self._keys = result, index, spec, keys
        self.name, self.kind = spec["name"], "bank" if spec["kind"] == BANK else "histogram"

    @property
    def info(self):
        raw = _OutputInfo()
        _check(_call("tr_reduce_result_output", self._result._open(), self.index, C.byref(raw)), what="output")
        d = dict(kind=self.kind, groups=int(raw.ngroups), present=int(raw.present), rows=int(raw.rows),
                 state_bytes=int(raw.state_bytes), grouping=self.spec["group"])
        if self.kind == "bank":
            d.update(terms=int(raw.nterms), mode={0: "sql", 1: "joint"}[raw.mode])
        else:
            k = int(raw.k)
            d.update(k=k, repr={1: "dense", 2: "sparse"}[raw.repr], weighted=bool(raw.weighted), cells=int(raw.cells))
            if raw.repr == 1:
                d["domain"] = [(int(raw.lo[c]), int(raw.lo[c]) + int(raw.span[c]) - 1, bool(raw.nullslot[c]))
                               for c in range(k)]
        return d

    def key(self, g):
        keys = self._keys
        if keys is None:
            return (int(g),)
        return keys.key_values(g)

    # banks
    def _bank(self):
        st = mbk._States()
        _check(_call("tr_reduce_result_bank", self._result._open(), self.index, C.byref(st)), what="bank output")
        return st

    def arrays(self):
        if self.kind == "bank":
            st = self._bank()
            g, m = int(st.ngroups), int(st.nterms)
            rows = _arr(st.rows, C.c_uint64, g)
            count = _arr(st.count, C.c_uint64, g * m).reshape(g, m)
            words = _arr(st.sum, C.c_int64, 2 * g * m).reshape(g, m, 2)
            return rows, count, words[:, :, 0], words[:, :, 1]
        st = md._States()
        _check(_call("tr_reduce_result_hist", self._result._open(), self.index, C.byref(st)), what="histogram output")
        G, n, k = int(st.ngroups), int(st.ncells), int(st.k)
        out = dict(rows=_arr(st.rows, C.c_uint64, G), count=_arr(st.count, C.c_uint64, n), repr=int(st.repr), k=k)
        if st.weighted:
            out["wcount"] = _arr(st.wcount, C.c_uint64, n)
            words = _arr(st.wsum, C.c_int64, 2 * n)
            out["wsum"] = _i128(words[0::2], words[1::2])
        if st.repr == md.REPRS["sparse"]:
            out["offset"] = _arr(st.offset, C.c_uint64, G + 1)
            out["values"] = [_arr(st.value[c], C.c_int64, n) for c in range(k)]
            out["nulls"] = _arr(st.nulls, C.c_uint8, n)
        else:
            out.update(cells=int(st.cells), lo=[int(st.lo[c]) for c in range(k)],
                       span=[int(st.span[c]) for c in range(k)], dim=[int(st.dim[c]) for c in range(k)])
        return out

    def state(self, g):
        """bank: {"present", "rows", "count": [per term], "sum": [per term]}; histogram: group(g)"""
        if self.kind != "bank":
            return self.group(g)
        rows, count, lo, hi = self.arrays()
        g = int(g)
        if not 0 <= g < len(rows):
            raise ValueError("group id out of range")
        return dict(present=bool(rows[g]), rows=int(rows[g]), count=[int(c) for c in count[g]], sum=_i128(lo[g], hi[g]))

    def states(self):
        if self.kind != "bank":
            a = self.arrays()
            return [self._group(a, g) for g in range(len(a["rows"]))]
        rows, count, lo, hi = self.arrays()
        return [dict(present=bool(rows[g]), rows=int(rows[g]), count=[int(c) for c in count[g]], sum=_i128(lo[g], hi[g]))
                for g in range(len(rows))]

    # histograms
    def _cells(self, a, g):
        weighted = "wcount" in a
        out = []
        if a["repr"] == md.REPRS["sparse"]:
            b, e = int(a["offset"][g]), int(a["offset"][g + 1])
            for i in range(b, e):
                key = tuple(None if (int(a["nulls"][i]) >> c) & 1 else int(a["values"][c][i]) for c in range(a["k"]))
                out.append((key, int(a["count"][i])) + ((int(a["wcount"][i]), a["wsum"][i]) if weighted else ()))
            return out
        cells, strides, stride = a["cells"], [], 1
        for c in range(a["k"] - 1, -1, -1):
            strides.append(stride)
            stride *= a["dim"][c]
        strides = strides[::-1]
        for j in range(cells):
            i = g * cells + j
            if a["count"][i]:
                key = []
                for c in range(a["k"]):
                    q = (j // strides[c]) % a["dim"][c]
                    key.append(None if q == a["span"][c] else a["lo"][c] + q)
                out.append((tuple(key), int(a["count"][i])) + ((int(a["wcount"][i]), a["wsum"][i]) if weighted else ()))
        return out

    def _group(self, a, g):
        rows = int(a["rows"][g])
        return dict(present=rows > 0, rows=rows, cells=self._cells(a, g))

    def cells(self, g=0):
        """the occupied cells of group g in cell order: (key, count) or (key, count, wcount, wsum)"""
        a = self.arrays()
        if not 0 <= int(g) < len(a["rows"]):
            raise ValueError("group id out of range")
        return self._cells(a, int(g))

    def group(self, g):
        a = self.arrays()
        if not 0 <= int(g) < len(a["rows"]):
            raise ValueError("group id out of range")
        return self._group(a, int(g))

    def table(self):
        """the present groups with their decoded keys"""
        out = []
        for g, s in enumerate(self.states()):
            if s["present"]:
                out.append(dict(group=g, key=self.key(g), **s))
        return out

    def __repr__(self):
        return f"Output({self.name!r}, {self.kind})"


class Result:
    """The immutable result of one run: every output, the run's record (stats) and the plan evidence."""

    def __init__(self, handle, plan, which, backend, device, evidence):
        self._handle, self._plan = handle, plan
        self.plan_kind, self.backend, self.device_name, self.plan = which, backend, device, evidence
        keys = plan.request._keys
        self.outputs = {}
        order = [None] + list(plan.request.groupings)
        for i, spec in enumerate(plan.request.outputs):
            gi = 0 if spec["group"] is None else order.index(spec["group"])
            self.outputs[spec["name"]] = Output(self, i, spec, keys[gi])

    def _open(self):
        if not self._handle:
            raise ValueError("the result is closed")
        return self._handle

    def __getitem__(self, name):
        return self.outputs[name]

    def __iter__(self):
        return iter(self.outputs)

    def __len__(self):
        return len(self.outputs)

    @property
    def stats(self):
        raw = _Stats()
        _check(_call("tr_reduce_result_get_stats", self._open(), C.byref(raw)), what="result")
        d = {n: int(getattr(raw, n)) for n in _STAT_FIELDS}
        d.update(plan={1: "fused", 2: "repeated"}[raw.plan], device=self.backend, device_name=self.device_name)
        return d

    def equal(self, other):
        """the same outputs with identical states (native comparison; the run records are not compared)"""
        return bool(_check(_call("tr_reduce_result_equal", self._open(), other._open()), what="result comparison"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_reduce_result_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Result({len(self.outputs)} outputs, {self.plan_kind}@{self.backend})" if self._handle else "Result(closed)"


def need(request, plan="fused", backend="cpu", budget=None):
    """the needs of a request's plan (compiles it first)"""
    return request.compile(budget=budget).need(plan, backend)
