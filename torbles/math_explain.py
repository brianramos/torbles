# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROV-002: explanation specialization and deletion propagation over retained PROV-001 provenance, natively on the
CPU or with the CUDA kernel (torbles/math_explain.c, math_explain_gpu.cu; probabilities compile the lineage with
COUNT-002's decision diagrams, math_symcount).

    from torbles import math_prov as mp, math_explain as mx
    st = mp.Store(); R = st.source("R", {...}); S = st.source("S", {...})
    G = R.join(S, on=[("k", "k")]).group(["a"])
    ex = mx.Explainer(st, [G], base={("R", 3): 0})            # the retained DAG, a base valuation (here R row 3 absent)
    d = ex.delete([[("R", 0)], [("S", 1), ("S", 2)]], backend="cuda")   # one change set per deletion request
    d.counts[w], d.exists[w], d.status[w], d.affected[w], d.changed[w], d.effects(w)   # per answer, per set
    ex.revalue([{("R", 1): 3}])                               # multiplicities (any naturals)
    ex.lineage(0); ex.relevant(0); ex.necessary(0); ex.sufficient(0); ex.smallest(0); ex.why_not(1)
    p = ex.probability(0, {("R", 0): Fraction(1, 2)}, default=Fraction(9, 10), method="exact" | "bounds" | "auto")
    p.value (Fraction), p.bounds ((lo, hi) certified), p.status, p.diagram

Answers are the tuples of the given relations in order (an int, or (relation index, tuple index)). Deletion
propagation visits only the nodes above the changed tuples (the update laws of N and B, with an early cutoff); the
result equals a complete re-execution of the query without them. Explanations are relative to the base: present
sources (base value != 0) explain an existing answer, absent ones restore a missing one. Probabilities are exact
rationals on the DAG with its shared variables (never a product of per-derivation probabilities) within the
diagram's budget; past it, a refusal (AlgebraBudgetError) or, with method="bounds" / "auto", a certified enclosure.
Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from . import math_prov as mp
from . import math_symcount as ms
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check
from .mathematics import AlgebraBudgetError

__all__ = ["Explainer", "Propagation", "Explanation", "Probability", "KINDS"]

KINDS = {"lineage": 1, "relevant": 2, "necessary": 3, "sufficient": 4, "smallest": 5, "why_not": 6}
STATUS = {1: "found", 2: "absent", 3: "present", 4: "underivable"}
_NEED_CPU, _NEED_DEVICE = 0, 1
_FREE = 0
P = C.POINTER
u32, i32, u64, i64, vp = C.c_uint32, C.c_int32, C.c_uint64, C.c_int64, C.c_void_p


class _View(C.Structure):
    _native_ = "tr_expl_view"
    _fields_ = [(n, u64) for n in ("nodes", "nchild", "noutputs", "nvars")] + [
        (n, vp) for n in ("node", "child", "output", "rev_at", "rev", "out_at", "outs", "var_node", "base_n", "base_b",
                          "base_c")] + [("count_ok", i32), ("reserved", i32)]


class _SetResult(C.Structure):
    _native_ = "tr_expl_set_result"
    _fields_ = [("status", i32), ("reserved", i32), ("affected", u64), ("changed", u64)]


class _Info(C.Structure):
    _native_ = "tr_expl_info"
    _fields_ = [(n, u64) for n in ("nodes", "children", "outputs", "variables", "cone_variables", "present",
                                   "bytes")] + [("count_ok", i32), ("reserved", i32)]


class _SetsInfo(C.Structure):
    _native_ = "tr_expl_sets_info"
    _fields_ = [("kind", i32), ("status", i32)] + [(n, u64) for n in ("nsets", "entries", "diagram_nodes", "levels")]


class _Interval(C.Structure):
    _native_ = "tr_math_interval"
    _fields_ = [("lo", C.c_double), ("hi", C.c_double)]


for _s, _n in ((_View, 128), (_SetResult, 24), (_Info, 64), (_SetsInfo, 40)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_expl_create": (vp, [P(mp._PlanView), P(mp._Valuations), B, R, e]),
            "tr_expl_create_need": (i32, [P(mp._PlanView), P(_Need)]),
            "tr_expl_get_view": (i32, [vp, P(_View)]),
            "tr_expl_get_info": (i32, [vp, P(_Info)]),
            "tr_expl_base": (i32, [vp, vp, u64]),
            "tr_expl_free": (None, [vp]),
            "tr_expl_propagate_need": (i32, [vp, P(mp._Valuations), i32, P(_Need)]),
            "tr_expl_propagate": (i32, [vp, P(mp._Valuations), vp, vp, vp, B, R]),
            "tr_expl_lineage": (vp, [vp, u32, i32, i32, vp, B, R, e]),
            "tr_expl_explain": (vp, [vp, u32, i32, u64, B, R, e]),
            "tr_expl_sets_get_info": (i32, [vp, P(_SetsInfo)]),
            "tr_expl_sets_get": (i32, [vp, vp, vp, u64, u64]),
            "tr_expl_sets_free": (None, [vp]),
            "tr_expl_probability_weights": (i32, [vp, vp, vp, vp]),
            "tr_expl_probability_bounds": (i32, [vp, u32, vp, vp, P(_Interval), P(i32), B, R]),
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
    math_gpu.bind("tr_mgpu_explain_propagate", C.c_int, [P(_View), P(mp._Valuations), vp, vp, vp, P(_Budget),
                                                        P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _probability(p, what):
    if isinstance(p, bool) or not isinstance(p, (Integral, Rational)):
        raise TypeError(f"{what} must be an exact int or Fraction, not {type(p).__name__}")
    q = Fraction(p)
    if not 0 <= q <= 1:
        raise ValueError(f"{what} must lie in [0, 1]")
    if q.denominator >= (1 << 63):
        raise OverflowError(f"{what}: the denominator must fit int64")
    return q


class Propagation:
    """The effect of change sets on every answer: counts[w][j] (N: multiplicity; None when the set's status is
    "overflow"), exists[w][j] (B), status[w] (None or "overflow"), affected[w] (DAG nodes recomputed), changed[w]
    (answers whose N or B changed), baseline (counts, exists under the base); backend "cpu", "cuda" or
    "cuda-emulated"."""

    def __init__(self, explainer, counts, exists, status, affected, changed, backend):
        self.counts, self.exists, self.status, self.affected, self.changed = counts, exists, status, affected, changed
        self.backend, self._ex = backend, explainer

    def of(self, w, k=0):
        """set w's (counts, exists) for the k-th relation's tuples"""
        a, b = self._ex.splits[k], self._ex.splits[k + 1]
        return (None if self.counts[w] is None else self.counts[w][a:b]), self.exists[w][a:b]

    def effects(self, w):
        """the answers set w changes: [{"answer", "relation", "tuple", "values", "before", "after", "existed",
        "exists"}] (before / after: multiplicities; None after an overflow)"""
        base_n, base_b = self._ex.baseline
        out = []
        for j, (b1, b0) in enumerate(zip(self.exists[w], base_b)):
            n1 = None if self.counts[w] is None else self.counts[w][j]
            if b1 == b0 and (n1 is None or n1 == base_n[j]):
                continue
            k, i = self._ex.locate(j)
            out.append(dict(answer=j, relation=k, tuple=i, values=self._ex.tuples[j], before=base_n[j], after=n1,
                            existed=b0, exists=b1))
        return out

    def __repr__(self):
        return f"Propagation({len(self.exists)} change sets, backend={self.backend})"


class Explanation:
    """An explanation of one answer relative to the base: kind, status ("found", "absent", "present",
    "underivable"), sets (frozensets of source labels), diagram (nodes, levels of the compiled lineage)."""

    def __init__(self, kind, status, sets, diagram):
        self.kind, self.status, self.sets, self.diagram = kind, status, sets, diagram

    @property
    def set(self):
        """the single set of LINEAGE / RELEVANT / NECESSARY / SMALLEST (None when there is none)"""
        return self.sets[0] if self.sets else None

    def __repr__(self):
        return f"Explanation({self.kind}, {self.status}, {len(self.sets)} sets)"


class Probability:
    """P(answer exists) under independent source tuples: status "exact" (value: a Fraction from the compiled
    lineage; diagram: its nodes, levels, apply steps) or "bounds" (bounds: (lo, hi) outward-rounded doubles with
    lo <= P <= hi; read_once: the DAG had disjoint children everywhere, so the enclosure is of exact independence);
    refusal: the diagram's budget refusal that led "auto" to the bounds."""

    def __init__(self, status, value=None, bounds=None, diagram=None, read_once=None, refusal=None, backend="cpu"):
        self.status, self.value, self.bounds, self.diagram = status, value, bounds, diagram
        self.read_once, self.refusal, self.backend = read_once, refusal, backend
        self.method = "decision diagram" if status == "exact" else "certified bounds"
        self.contract = ("exact rational" if status == "exact" else
                         "certified enclosure: lo <= P <= hi (outward-rounded doubles), not an estimate")

    def __repr__(self):
        return f"Probability({self.status}, {self.value if self.value is not None else self.bounds})"


class Explainer:
    """The retained provenance of some answers (a PROV-001 store's relations) specialized natively: deletion
    propagation, explanations and probabilities, relative to a base valuation ({label: value}, the others `default`;
    1 = present, 0 = absent, k = multiplicity k)."""

    def __init__(self, store, relations, *, base=None, default=1, budget=None):
        rels = [relations] if isinstance(relations, mp.Relation) else list(relations)
        if not rels:
            raise ValueError("at least one relation")
        if any(r.store is not store for r in rels):
            raise ValueError("every relation belongs to the store")
        self.store, self.relations = store, rels
        plan = store.plan(rels, budget=budget)
        self.splits = list(plan.splits)
        self.tuples = [tp for r in rels for tp in r.tuples()]
        self.labels = store._labels()
        self._index = {lab: i for i, lab in enumerate(self.labels)}
        raw, keep = store._valuations(None, [dict(base or {})], int(default))
        view = plan.view()
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_expl_create", C.byref(view), C.byref(raw), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        del keep
        self._handle = _made(h, error, refusal, "explanation state")
        plan.close()
        n = self.info["outputs"]
        res = self._run([{}], "cpu", None)
        self.baseline = (res.counts[0] if res.counts[0] is not None else [None] * n, res.exists[0])

    def _open(self):
        if not self._handle:
            raise ValueError("the explainer is closed")
        return self._handle

    @property
    def info(self):
        i = _Info()
        _check(_call("tr_expl_get_info", self._open(), C.byref(i)), what="explanation state")
        return dict(nodes=int(i.nodes), children=int(i.children), outputs=int(i.outputs), variables=int(i.variables),
                    cone_variables=int(i.cone_variables), present=int(i.present), bytes=int(i.bytes),
                    count_ok=bool(i.count_ok))

    def view(self):
        v = _View()
        _check(_call("tr_expl_get_view", self._open(), C.byref(v)), what="explanation view")
        return v

    def base(self):
        """{label: base value} of every source tuple"""
        n = len(self.labels)
        out = np.zeros(max(n, 1), np.uint64)
        _check(_call("tr_expl_base", self._open(), out.ctypes.data, len(out)), what="base")
        return {lab: int(out[i]) for i, lab in enumerate(self.labels)}

    def locate(self, j):
        """(relation index, tuple index) of answer j"""
        for k in range(len(self.relations)):
            if self.splits[k] <= j < self.splits[k + 1]:
                return k, j - self.splits[k]
        raise IndexError("no such answer")

    def _answer(self, a):
        if isinstance(a, tuple):
            k, i = a
            j = self.splits[int(k)] + int(i)
            if not self.splits[int(k)] <= j < self.splits[int(k) + 1]:
                raise IndexError("no such tuple")
            return j
        j = int(a)
        if not 0 <= j < self.splits[-1]:
            raise IndexError("no such answer")
        return j

    def _var(self, label):
        lab = tuple(label)
        if lab not in self._index:
            raise ValueError(f"no source tuple {label!r}")
        return self._index[lab]

    # ------------------------------------------------------------------ deletion propagation
    def _changes(self, sets):
        offset, var, value = [0], [], []
        for over in sets:
            ids = {}
            for lab, x in over.items():
                x = int(x)
                if not 0 <= x < (1 << 64):
                    raise ValueError("values are naturals below 2^64")
                ids[self._var(lab)] = x
            for v in sorted(ids):
                var.append(v)
                value.append(ids[v])
            offset.append(len(var))
        keep = (np.array(offset, np.uint64), np.array(var or [0], np.uint32), np.array(value or [0], np.uint64))
        raw = mp._Valuations(count=len(sets), offset=keep[0].ctypes.data, value=keep[2].ctypes.data, fallback=None,
                             var=keep[1].ctypes.data)
        return raw, keep

    def _run(self, sets, backend, budget):
        raw, keep = self._changes(sets)
        m, n = len(sets), int(self.view().noutputs)
        out_n, out_b = np.zeros(max(2 * m * n, 1), np.uint64), np.zeros(max(m * n, 1), np.uint8)
        res, refusal = (_SetResult * max(m, 1))(), _Refusal()
        if backend == "cpu":
            _check(_call("tr_expl_propagate", self._open(), C.byref(raw), out_n.ctypes.data, out_b.ctypes.data, res,
                         C.byref(_budget(budget)), C.byref(refusal)), refusal, "deletion propagation")
            label = "cpu"
        elif backend == "cuda":
            gpu = _gpu()
            v = self.view()
            gpu.run("tr_mgpu_explain_propagate", C.byref(v), C.byref(raw), out_n.ctypes.data, out_b.ctypes.data, res,
                    C.byref(_budget(budget)), C.byref(refusal), refusal=refusal,
                    what="deletion propagation on the device")
            label = "cuda-emulated" if gpu.device()["emulated"] else "cuda"
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        del keep
        self._last = (out_n[:2 * m * n].copy(), out_b[:m * n].copy(), bytes(res)[:m * C.sizeof(_SetResult)])
        counts, exists, status = [], [], []
        for w in range(m):
            st = int(res[w].status)
            status.append(None if st == 0 else "overflow")
            row = out_n[2 * w * n:2 * (w + 1) * n]
            counts.append(None if st else [int(row[2 * j]) | int(row[2 * j + 1]) << 64 for j in range(n)])
            exists.append([bool(x) for x in out_b[w * n:(w + 1) * n]])
        return Propagation(self, counts, exists, status, [int(res[w].affected) for w in range(m)],
                           [int(res[w].changed) for w in range(m)], label)

    def delete(self, sets, *, backend="cpu", budget=None):
        """one change set per request: a source label or a list of labels deleted together (value 0)"""
        reqs = []
        for d in sets:
            items = [d] if isinstance(d, tuple) and len(d) == 2 and isinstance(d[0], str) else list(d)
            reqs.append({lab: 0 for lab in items})
        return self._run(reqs, backend, budget)

    def revalue(self, changes, *, backend="cpu", budget=None):
        """change sets as {label: value} (deletions 0, restorations 1, multiplicities k); the rest keep the base"""
        return self._run([dict(c) for c in changes], backend, budget)

    def need(self, sets, backend="cpu"):
        raw, keep = self._changes([{lab: 0 for lab in ([d] if isinstance(d, tuple) and len(d) == 2 and
                                                        isinstance(d[0], str) else list(d))} for d in sets])
        need = _Need()
        _check(_call("tr_expl_propagate_need", self._open(), C.byref(raw), _NEED_DEVICE if backend == "cuda" else
                     _NEED_CPU, C.byref(need)), what="propagation needs")
        del keep
        return _needs(need)

    # ------------------------------------------------------------------ explanations
    def _explain(self, a, kind, max_sets, budget):
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_expl_explain", self._open(), self._answer(a), KINDS[kind], int(max_sets),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        h = _made(h, error, refusal, f"explanation ({kind})")
        try:
            info = _SetsInfo()
            _check(_call("tr_expl_sets_get_info", h, C.byref(info)), what="explanation")
            ns, ne = int(info.nsets), int(info.entries)
            off, var = np.zeros(ns + 1, np.uint64), np.zeros(max(ne, 1), np.uint32)
            _check(_call("tr_expl_sets_get", h, off.ctypes.data, var.ctypes.data, ns, max(ne, 1)), what="explanation")
        finally:
            _call("tr_expl_sets_free", h)
        sets = [frozenset(self.labels[int(x)] for x in var[int(off[s]):int(off[s + 1])]) for s in range(ns)]
        return Explanation(kind, STATUS[int(info.status)], sets,
                           dict(nodes=int(info.diagram_nodes), levels=int(info.levels)))

    def lineage(self, a, *, budget=None):
        """the source tuples of the answer's cone (every one occurs in a derivation: its multiplicity depends on it)"""
        return self._explain(a, "lineage", 1, budget)

    def relevant(self, a, *, budget=None):
        """the present source tuples the answer's existence depends on (the support of its Boolean lineage)"""
        return self._explain(a, "relevant", 1, budget)

    def necessary(self, a, *, budget=None):
        """the counterfactual sources: present tuples whose deletion alone removes the answer"""
        return self._explain(a, "necessary", 1, budget)

    def sufficient(self, a, *, max_sets=1 << 16, budget=None):
        """the minimal sets of present source tuples that suffice for the answer (refused past max_sets)"""
        return self._explain(a, "sufficient", max_sets, budget)

    def smallest(self, a, *, budget=None):
        """one minimum-cardinality sufficient set of present source tuples"""
        return self._explain(a, "smallest", 1, budget)

    def why_not(self, a, *, max_sets=1 << 16, budget=None):
        """for an answer absent under the base: the minimal sets of absent source tuples whose restoration makes it
        exist (status "present" when it exists, "underivable" when no recorded derivation can)"""
        return self._explain(a, "why_not", max_sets, budget)

    # ------------------------------------------------------------------ probabilities
    def _weights(self, probabilities, default):
        n = len(self.labels)
        dq = _probability(default, "default probability")
        num, den = np.full(max(n, 1), dq.numerator, np.int64), np.full(max(n, 1), dq.denominator, np.int64)
        for lab, p in (probabilities or {}).items():
            q = _probability(p, f"P{lab!r}")
            v = self._var(lab)
            num[v], den[v] = q.numerator, q.denominator
        return num, den

    def probability(self, a, probabilities=None, *, default=Fraction(1, 2), method="exact", order="dfs",
                    backend="cpu", budget=None):
        """P(answer a exists) when every source tuple is present independently with its probability ({label: p},
        the others `default`): "exact" (the compiled lineage, refused past its budget), "bounds" (certified, no
        compilation) or "auto" (exact within the budget, else the bounds with the refusal recorded)"""
        if method not in ("exact", "bounds", "auto"):
            raise ValueError("method is 'exact', 'bounds' or 'auto'")
        j = self._answer(a)
        num, den = self._weights(probabilities, default)
        if method in ("exact", "auto"):
            try:
                return self._exact(j, num, den, order, backend, budget)
            except AlgebraBudgetError as e:
                if method == "exact":
                    raise
                refusal = dict(resource=e.resource, need=e.need, limit=e.limit)
        else:
            refusal = None
        iv, once, rf = _Interval(), i32(), _Refusal()
        _check(_call("tr_expl_probability_bounds", self._open(), j, num.ctypes.data, den.ctypes.data, C.byref(iv),
                     C.byref(once), C.byref(_budget(budget)), C.byref(rf)), rf, "probability bounds")
        return Probability("bounds", bounds=(float(iv.lo), float(iv.hi)), read_once=bool(once.value), refusal=refusal)

    def _exact(self, j, num, den, order, backend, budget):
        kinds = {"dfs": 2, "natural": 0}
        if isinstance(order, str):
            if order not in kinds:
                raise ValueError("order is 'dfs', 'natural' or a list of source labels")
            kind, arr = kinds[order], None
        else:
            ids = [self._var(lab) for lab in order]
            given = set(ids)
            if len(given) != len(ids):
                raise ValueError("the order names a source tuple twice")
            rest = [v for v in range(len(self.labels)) if v not in given]
            kind, arr = 1, np.array(ids + rest, np.uint32)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_expl_lineage", self._open(), j, _FREE, kind, arr.ctypes.data if arr is not None else None,
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        d = object.__new__(ms.Diagram)
        d._handle, d.program = _made(h, error, refusal, "lineage diagram"), None
        try:
            w = (ms._Weight * max(len(self.labels), 1))()
            _check(_call("tr_expl_probability_weights", self._open(), num.ctypes.data, den.ctypes.data, w),
                   what="probability weights")
            c = d.count(weights=w, backend=backend, budget=budget)
            info = d.info
        finally:
            d.close()
        return Probability("exact", value=Fraction(c.true), backend=c.backend,
                           diagram=dict(nodes=info["nodes"], levels=info["levels"], steps=info["steps"],
                                        width=info["width"], peak_nodes=info["peak_nodes"]))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_expl_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Explainer({self.splits[-1]} answers)" if self._handle else "Explainer(closed)"
