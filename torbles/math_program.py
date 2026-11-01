# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""META-002: a family of related queries compiled into one multi-output program (torbles/math_program.c,
math_program_gpu.cu), executed through FND-006's fused reduction (math_reduce).

    from torbles import math_bank as mbk, math_bool as mb, math_program as mp
    scope = mbk.Scope.from_table(db["f"], ["x", "d", "y"], group_by=["k"])
    fam = mp.Family(scope, conditions=[mb.condition("x", ">", 5), mb.condition("y", "is null")],
                    groupings={"c": mbk.Scope.from_table(db["f"], [], group_by=["c"]), "all": mp.rollup(coarse)})
    big, nul = mb.variables(2)
    fam.query("q1", [mp.count(), mp.sum("x"), mp.avg({"x": 2, "x*y": -1})], where=big)        # GROUP BY the scope
    fam.query("q2", [mp.count("y"), mp.histogram(["y"])], where=big & ~nul, group="c")
    fam.query("q3", [mp.sum("x*d")], where=~~big, group="all", scalar=True)                    # no GROUP BY
    prog = fam.compile(prove=True, lifetime="run")    # canonical conditions and predicates, shared states, proofs
    res = prog.run(backend="cuda", finalize="gpu")    # or "cpu"; plan="fused" | "repeated" | None (the planner)
    res["q1"].rows()          # [{"key": (...), "count(*)": 3, "sum(x)": 12, "avg(...)": Fraction(...)}, ...]
    res["q1"].explanation     # the shared parts it used, its devices, memory and dispatch costs
    prog.compare("q1", "q3"); prog.snapshot_compare("q1", "q3")

SQL semantics per query (WHERE keeps TRUE rows; a measure skips a row where one of its columns is NULL; SUM / AVG of
no valid row is None; a grouped query lists its present groups, a scalar one exactly one row). Exact results: int,
Fraction, None. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_bool as mb
from . import math_dist as md
from . import math_reduce as mr
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, StaleStateError, check as _check

__all__ = ["Family", "Program", "ProgramResult", "QueryResult", "count", "sum", "avg", "histogram", "rollup",
           "LIFETIMES", "OUTCOMES"]

rollup = mr.rollup
COUNT_STAR, COUNT, SUM, AVG, HIST = 1, 2, 3, 4, 5
KINDS = {COUNT_STAR: "count(*)", COUNT: "count", SUM: "sum", AVG: "avg", HIST: "histogram"}
LIFETIMES = {"run": 0, "snapshot": 1}
OUTCOMES = {1: "equivalent", 2: "different", 3: "unknown", 4: "budget"}
HOW = {0: None, 1: "canonical", 2: "search", 3: "witness"}
MAX_QUERIES, MAX_MEASURES = 4096, 16384
P = C.POINTER


class _Measure(C.Structure):
    _native_ = "tr_prog_measure"
    _fields_ = [(n, C.c_int32) for n in ("kind", "first", "nterms", "reserved")] + [("hist", md._Spec)]


class _Query(C.Structure):
    _native_ = "tr_prog_query"
    _fields_ = [(n, C.c_int32) for n in ("filter", "grouping", "first", "nmeasures", "scalar")] + [
        ("reserved", C.c_int32 * 3)]


class _Family(C.Structure):
    _native_ = "tr_prog_family"
    _fields_ = [(n, C.c_int32) for n in ("nconds", "nnodes", "ngroupings", "nqueries", "nmeasures", "nterms", "flags",
                                         "lifetime")] + [
        (n, C.c_void_p) for n in ("conds", "nodes", "groupings", "queries", "measures", "exps", "coefs")]


class _FMeasure(C.Structure):
    _native_ = "tr_prog_fmeasure"
    _fields_ = [(n, C.c_int32) for n in ("kind", "output", "first", "nterms")] + [
        ("count_term", C.c_uint32), ("query", C.c_int32), ("groups", C.c_uint64), ("slot", C.c_uint64)]


class _FTerm(C.Structure):
    _native_ = "tr_prog_fterm"
    _fields_ = [("term", C.c_uint32), ("reserved", C.c_uint32), ("coef", C.c_int64)]


class _FOut(C.Structure):
    _native_ = "tr_prog_fout"
    _fields_ = [("rows", C.c_void_p), ("count", C.c_void_p), ("sum", C.c_void_p), ("ngroups", C.c_uint64),
                ("nterms", C.c_int32), ("reserved", C.c_int32)]


class _FinalView(C.Structure):
    _native_ = "tr_prog_final_view"
    _fields_ = [("measures", C.c_void_p), ("terms", C.c_void_p), ("outs", C.c_void_p)] + [
        (n, C.c_int32) for n in ("nmeasures", "nterms", "noutputs", "reserved")] + [("nslots", C.c_uint64)]


class _Value(C.Structure):
    _native_ = "tr_prog_value"
    _fields_ = [("status", C.c_int32), ("kind", C.c_int32), ("rows", C.c_uint64), ("count", C.c_uint64),
                ("num", C.c_uint64 * 4), ("den", C.c_uint64), ("reserved", C.c_uint64)]


class _Info(C.Structure):
    _native_ = "tr_prog_info"
    _fields_ = [(n, C.c_int32) for n in ("nqueries", "nmeasures", "noutputs", "nclasses", "conds_in", "conds",
                                         "nodes_in", "nodes", "merged_canonical", "merged_search", "search_unknown",
                                         "search_budget", "lifetime", "flags", "nvalue_measures", "nhist")] + [
        (n, C.c_uint64) for n in ("search_iterations", "slots", "bytes")]


class _QueryInfo(C.Structure):
    _native_ = "tr_prog_query_info"
    _fields_ = [(n, C.c_int32) for n in ("filter", "klass", "proof", "grouping", "nmeasures", "scalar")] + [
        (n, C.c_uint64) for n in ("conditions", "slots", "state_bytes")] + [
        (n, C.c_int32) for n in ("outputs", "sharers", "class_size", "reserved")]


class _MeasureInfo(C.Structure):
    _native_ = "tr_prog_measure_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "query", "output", "nterms")] + [
        ("count_term", C.c_uint32), ("reserved", C.c_uint32), ("slot", C.c_uint64), ("groups", C.c_uint64)]


class _OutputInfo(C.Structure):
    _native_ = "tr_prog_output_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "mode", "filter", "grouping", "nterms", "readers")] + [
        ("colset", C.c_uint32), ("reserved", C.c_uint32), ("hist", md._Spec)]


class _Compare(C.Structure):
    _native_ = "tr_prog_compare_result"
    _fields_ = [(n, C.c_int32) for n in ("outcome", "how", "grouping_equal", "measures_equal")] + [
        (n, C.c_uint64) for n in ("iterations", "support", "assign_true", "assign_unknown")] + [
        ("accepts", C.c_int32), ("ncols", C.c_int32), ("null_mask", C.c_uint64), ("set_mask", C.c_uint64),
        ("value", C.c_int64 * 32)]


class _Snapshot(C.Structure):
    _native_ = "tr_prog_snapshot_result"
    _fields_ = [(n, C.c_int32) for n in ("equal", "grouping_equal", "measures_equal", "reserved")] + [
        (n, C.c_uint64) for n in ("rows_a", "rows_b", "differ", "first", "words")]


for _s, _n in ((_Measure, 112), (_Query, 32), (_Family, 88), (_FMeasure, 40), (_FTerm, 16), (_FOut, 40),
               (_FinalView, 48), (_Value, 72), (_Info, 88), (_QueryInfo, 64), (_MeasureInfo, 40), (_OutputInfo, 128),
               (_Compare, 328), (_Snapshot, 56)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        sig = {
            "tr_prog_compile": (vp, [vp, P(_Family), P(_Budget), P(_Refusal), e, e]),
            "tr_prog_free": (None, [vp]),
            "tr_prog_get_info": (i32, [vp, P(_Info)]),
            "tr_prog_query_get": (i32, [vp, i32, P(_QueryInfo)]),
            "tr_prog_measure_get": (i32, [vp, i32, P(_MeasureInfo)]),
            "tr_prog_output_get": (i32, [vp, i32, P(_OutputInfo), vp, u64]),
            "tr_prog_reduce_plan": (vp, [vp]),
            "tr_prog_final_view_get": (i32, [vp, vp, vp, i32, P(_FinalView)]),
            "tr_prog_finalize": (i32, [vp, vp, vp, u64, P(_Budget), P(_Refusal)]),
            "tr_prog_need": (i32, [vp, i32, i32, i32, P(_Need)]),
            "tr_prog_final_need": (i32, [vp, i32, i32, P(_Need)]),
            "tr_prog_compare": (i32, [vp, i32, i32, P(_Budget), P(_Compare), P(_Refusal)]),
            "tr_prog_snapshot_compare": (i32, [vp, i32, i32, P(_Snapshot)]),
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
    math_gpu.bind("tr_mgpu_program_finalize", C.c_int, [P(_FinalView), P(_Budget), C.c_void_p, P(_Refusal)])
    math_gpu.bind("tr_mgpu_program_run", C.c_int, [P(mr._View), P(_Need), P(_FinalView), P(_Budget), P(mr._Arena),
                                                   C.c_void_p, P(mr._Stats), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


# ---------------------------------------------------------------------- measures
class Measure:
    """COUNT(*), COUNT(p), SUM(p), AVG(p) of an integer polynomial p over the scope's columns, or a DIST-001 histogram;
    p: a monomial string ("x", "x*y^2", "1"), {monomial: integer coefficient}, or [(coefficient, monomial), ...]"""

    def __init__(self, kind, poly=None, *, name=None, columns=None, weight=None, repr="sparse", domain=None):
        self.kind, self.poly, self.columns, self.weight, self.repr, self.domain = kind, poly, columns, weight, repr, domain
        self.name = name

    def terms(self):
        """[(monomial, coefficient)] as given (merged natively)"""
        p = self.poly
        if p is None:
            return []
        if isinstance(p, str):
            return [(p, 1)]
        if isinstance(p, dict):
            return list(p.items())
        return [(m, c) for c, m in p]

    def label(self):
        if self.name:
            return self.name
        if self.kind == COUNT_STAR:
            return "count(*)"
        if self.kind == HIST:
            return f"histogram({', '.join(self.columns)})"
        body = " + ".join((m if c == 1 else f"{c}*{m}") for m, c in self.terms()) or "0"
        return f"{KINDS[self.kind]}({body})"


def count(poly=None, *, name=None):
    """COUNT(*) without a polynomial, COUNT(p) with one"""
    return Measure(COUNT_STAR if poly is None else COUNT, poly, name=name)


def sum(poly, *, name=None):  # noqa: A001 - the SQL aggregate's name
    return Measure(SUM, poly, name=name)


def avg(poly, *, name=None):
    return Measure(AVG, poly, name=name)


def histogram(columns, *, weight=None, repr="sparse", domain=None, name=None):
    columns = [columns] if isinstance(columns, str) else list(columns)
    return Measure(HIST, None, name=name, columns=columns, weight=weight, repr=repr, domain=domain)


def _i256(words):
    v = int(words[0]) | (int(words[1]) << 64) | (int(words[2]) << 128) | (int(words[3]) << 192)
    return v - (1 << 256) if v >> 255 else v


# ---------------------------------------------------------------------- the family
class Family:
    """Related queries over one scope: each a predicate (where=, a math_bool expression over the family's conditions),
    a grouping (group=: None for the scope's own, or a name of groupings), scalar (no GROUP BY) and measures."""

    def __init__(self, scope, *, conditions=(), groupings=None):
        self.scope, self.conditions = scope, list(conditions)
        self.groupings = dict(groupings or {})
        self.queries = []
        self._names = set()

    def query(self, name, measures, *, where=None, group=None, scalar=None):
        name = str(name)
        if name in self._names:
            raise ValueError(f"query {name!r} exists")
        if group is not None and group not in self.groupings:
            raise ValueError(f"no grouping {group!r}")
        measures = list(measures)
        for m in measures:
            if not isinstance(m, Measure):
                raise TypeError(f"{m!r} is not a measure (math_program.count / sum / avg / histogram)")
        if len(self.queries) >= MAX_QUERIES:
            raise ValueError(f"at most {MAX_QUERIES} queries")
        self._names.add(name)
        self.queries.append(dict(name=name, measures=measures, where=where, group=group, scalar=scalar))
        return self

    def compile(self, *, prove=False, lifetime="run", budget=None):
        """the native program: canonical conditions and predicates, merged classes (prove=True: the bounded universal
        search too), shared FND-006 states, finalization tables; refusals raise with .failed (the query's name)"""
        if lifetime not in LIFETIMES:
            raise ValueError(f"lifetime is one of {', '.join(LIFETIMES)}")
        greq = mr.Request(self.scope, conditions=self.conditions, groupings=self.groupings)
        greq._fresh()
        grq, keep = greq._native()
        names = self.scope.names
        conds, nconds = mb._conditions(self.conditions, names)
        nodes, roots = mr._compile_nodes([q["where"] for q in self.queries])
        narr = (mb._Node * max(len(nodes), 1))()
        for i, (op, a, b) in enumerate(nodes):
            narr[i] = mb._Node(op, a, b, 0)
        order = [None] + list(self.groupings)
        allm = [m for q in self.queries for m in q["measures"]]
        if len(allm) > MAX_MEASURES:
            raise ValueError(f"at most {MAX_MEASURES} measures")
        qarr = (_Query * max(len(self.queries), 1))()
        marr = (_Measure * max(len(allm), 1))()
        rows, coefs, mi = [], [], 0
        for qi, q in enumerate(self.queries):
            gi = 0 if q["group"] is None else order.index(q["group"])
            scalar = q["scalar"] if q["scalar"] is not None else (q["group"] is None and self.scope.groups == 1)
            qarr[qi] = _Query(roots[qi], gi, mi, len(q["measures"]), int(bool(scalar)), (C.c_int32 * 3)())
            for m in q["measures"]:
                x = marr[mi]
                x.kind = m.kind
                if m.kind == HIST:
                    x.hist = md._spec(self.scope, m.columns, m.weight, m.repr, m.domain, "auto")[0]
                elif m.kind != COUNT_STAR:
                    terms = m.terms()
                    x.first, x.nterms = len(rows), len(terms)
                    if terms:
                        exps = mbk._basis(names, [t for t, _ in terms])
                        for j, (_, c) in enumerate(terms):
                            if isinstance(c, bool) or not isinstance(c, int) or not -(1 << 63) <= int(c) < (1 << 63):
                                raise ValueError(f"{m.label()}: coefficients are int64 integers")
                            rows.append(exps[j])
                            coefs.append(int(c))
                mi += 1
        exps = (np.ascontiguousarray(np.array(rows, dtype=np.uint8).reshape(len(rows), len(names))) if rows else
                np.zeros((0, len(names)), np.uint8))
        coef = np.ascontiguousarray(np.array(coefs, dtype=np.int64)) if coefs else np.zeros(1, np.int64)
        fam = _Family(nconds, len(nodes), grq.ngroupings, len(self.queries), len(allm), len(rows),
                      1 if prove else 0, LIFETIMES[lifetime], C.cast(conds, C.c_void_p), C.cast(narr, C.c_void_p),
                      grq.groupings, C.cast(qarr, C.c_void_p), C.cast(marr, C.c_void_p),
                      exps.ctypes.data if exps.size else None, coef.ctypes.data)
        error, failed, refusal = C.c_int32(), C.c_int32(-1), _Refusal()
        h = _call("tr_prog_compile", self.scope._handle, C.byref(fam), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(failed), C.byref(error))
        if not h or error.value:
            if h:
                _call("tr_prog_free", h)
            mr._raise(error.value or -2, refusal, "query program",
                      failed=self.queries[failed.value]["name"] if 0 <= failed.value < len(self.queries) else None)
        return Program(h, self, greq, keep + [conds, narr, qarr, marr, exps, coef], lifetime)


class _BorrowedPlan(mr.Plan):
    """the program's FND-006 plan (owned by the native program: never freed here)"""

    def close(self):
        self._handle = None


# ---------------------------------------------------------------------- programs
class Program:
    """A compiled native program (immutable). run() executes its shared FND-006 pass and the per-query finalization;
    lifetime "run" keeps nothing after a run, "snapshot" keeps the last run for later runs while its scopes are fresh."""

    def __init__(self, handle, family, greq, keep, lifetime):
        self._handle, self.family, self.lifetime = handle, family, lifetime
        self.names = [q["name"] for q in family.queries]
        self._kept = None
        info = self.info
        names = family.scope.names
        outputs = []
        order = [None] + list(family.groupings)
        for o in range(info["outputs"]):
            oi = _OutputInfo()
            exps = np.zeros(max(1024 * max(len(names), 1), 1), dtype=np.uint8)
            _check(_call("tr_prog_output_get", handle, o, C.byref(oi), exps.ctypes.data, exps.size), what="output")
            spec = dict(name=f"state{o}", kind=oi.kind, where=None if oi.filter < 0 else True,
                        group=order[oi.grouping], readers=int(oi.readers))
            if oi.kind == mr.BANK:
                spec.update(exps=exps[:oi.nterms * len(names)].reshape(oi.nterms, len(names)).copy(), mode=oi.mode)
            else:
                s = md._Spec()
                C.memmove(C.byref(s), C.byref(oi.hist), C.sizeof(s))
                spec.update(spec=s, columns=tuple(names[s.col[c]] for c in range(s.k)),
                            weight=names[s.weight] if s.weight >= 0 else None)
            outputs.append(spec)
        greq.outputs = outputs
        self._rplan = _BorrowedPlan(_call("tr_prog_reduce_plan", handle), greq, keep)   # results keep the program
        self._greq = greq
        self.measures = []
        n = 0
        for qi, q in enumerate(family.queries):
            for m in q["measures"]:
                raw = _MeasureInfo()
                _check(_call("tr_prog_measure_get", handle, n, C.byref(raw)), what="measure")
                self.measures.append(dict(query=qi, measure=m, output=int(raw.output), slot=int(raw.slot),
                                          groups=int(raw.groups), count_term=int(raw.count_term)))
                n += 1

    def _open(self):
        if not self._handle:
            raise ValueError("the program is closed")
        return self._handle

    @property
    def info(self):
        raw = _Info()
        _check(_call("tr_prog_get_info", self._open(), C.byref(raw)), what="program")
        d = {n: int(getattr(raw, n)) for n, _ in raw._fields_}
        d["outputs"] = d.pop("noutputs")
        d["lifetime"] = {v: k for k, v in LIFETIMES.items()}[d["lifetime"]]
        return d

    def _index(self, name):
        if name not in self.names:
            raise KeyError(f"no query {name!r}")
        return self.names.index(name)

    def query_info(self, name):
        raw = _QueryInfo()
        _check(_call("tr_prog_query_get", self._open(), self._index(name), C.byref(raw)), what="query")
        d = {n: int(getattr(raw, n)) for n, _ in raw._fields_ if n != "reserved"}
        d["klass"] = self.names[d["klass"]]
        d["proof"] = HOW.get(d["proof"])
        d["conditions"] = [i for i in range(64) if (d["conditions"] >> i) & 1]
        return d

    def readers(self):
        """per FND-006 state: the queries reading it"""
        out = {}
        for m in self.measures:
            out.setdefault(m["output"], [])
            name = self.names[m["query"]]
            if name not in out[m["output"]]:
                out[m["output"]].append(name)
        return out

    def explain(self, name):
        """what the compiled query shares: its predicate's canonical node and class (with the proof), the conditions
        it reads, the FND-006 states it reads (kind, mode, terms, groups, the queries sharing each), its value slots"""
        qi = self.query_info(name)
        readers = self.readers()
        states = []
        for o in sorted({m["output"] for m in self.measures if self.names[m["query"]] == name}):
            spec = self._greq.outputs[o]
            states.append(dict(state=o, kind="bank" if spec["kind"] == mr.BANK else "histogram",
                               mode={0: "sql", 1: "joint"}.get(spec.get("mode")) if spec["kind"] == mr.BANK else None,
                               terms=len(spec["exps"]) if spec["kind"] == mr.BANK else None, grouping=spec["group"],
                               shared_with=[r for r in readers[o] if r != name]))
        return dict(query=name, predicate=dict(node=qi["filter"], klass=qi["klass"], proof=qi["proof"],
                                               class_size=qi["class_size"], conditions=qi["conditions"]),
                    states=states, value_slots=qi["slots"], value_bytes=qi["slots"] * C.sizeof(_Value),
                    state_bytes_planned=qi["state_bytes"], sharers=qi["sharers"])

    # ------------------------------------------------------------------ comparisons
    def compare(self, a, b, *, budget=None):
        """the universal outcome of two queries' WHERE predicates (canonical / search / witness; unknown; budget),
        with whether their groupings and measures are equal"""
        r = _Compare()
        _check(_call("tr_prog_compare", self._open(), self._index(a), self._index(b), C.byref(_budget(budget)),
                     C.byref(r), C.byref(_Refusal())), what="compare")
        names = self.family.scope.names
        d = dict(outcome=OUTCOMES[r.outcome], how=HOW.get(r.how), iterations=int(r.iterations),
                 support=[i for i in range(64) if (int(r.support) >> i) & 1], grouping_equal=bool(r.grouping_equal),
                 measures_equal=bool(r.measures_equal), witness=None)
        d["equivalent_queries"] = d["outcome"] == "equivalent" and d["grouping_equal"] and d["measures_equal"]
        if r.outcome == 2:
            row = {}
            for c, nm in enumerate(names):
                if (r.null_mask >> c) & 1:
                    row[nm] = None
                elif (r.set_mask >> c) & 1:
                    scale = self.family.scope.column(nm)["scale"]
                    row[nm] = Fraction(int(r.value[c]), 10 ** scale) if scale else int(r.value[c])
            d["witness"] = dict(row=row, accepted_by=a if r.accepts == 1 else b,
                                true=[i for i in range(64) if (int(r.assign_true) >> i) & 1],
                                unknown=[i for i in range(64) if (int(r.assign_unknown) >> i) & 1])
        return d

    def snapshot_compare(self, a, b):
        """both predicates on this snapshot's selected rows: the rows each keeps, the rows where they differ and the
        first one; equal holds only for this snapshot"""
        self.family_fresh()
        r = _Snapshot()
        _check(_call("tr_prog_snapshot_compare", self._open(), self._index(a), self._index(b), C.byref(r)),
               what="snapshot compare")
        return dict(equal=bool(r.equal), grouping_equal=bool(r.grouping_equal), measures_equal=bool(r.measures_equal),
                    rows=(int(r.rows_a), int(r.rows_b)), differ=int(r.differ),
                    first=None if r.first == (1 << 64) - 1 else int(r.first), words=int(r.words), scope="snapshot")

    def family_fresh(self):
        self._greq._fresh()

    # ------------------------------------------------------------------ running
    def _need_raw(self, plan, backend, finalize):
        if plan not in mr.PLANS or backend not in mr.DEVICES or finalize not in ("cpu", "gpu"):
            raise ValueError("plan is 'fused' or 'repeated', backend 'cpu' or 'cuda', finalize 'cpu' or 'gpu'")
        raw = _Need()
        _check(_call("tr_prog_need", self._open(), mr.PLANS[plan], mr.DEVICES[backend], 0 if finalize == "cpu" else 1,
                     C.byref(raw)), what="program needs")
        return raw

    def need(self, plan="fused", backend="cpu", finalize=None):
        """{resource: need} of a run (what run checks before anything is allocated, an upper bound of every later
        check but the occupied cells of sparse histograms): FND-006's pass (Plan.need) and the finalization"""
        raw = self._need_raw(plan, backend, finalize or ("gpu" if backend == "cuda" else "cpu"))
        return {n: int(raw.need[i]) for i, n in enumerate(RESOURCES) if raw.need[i]}

    def run(self, *, backend="cpu", finalize=None, plan=None, devices=None, force=None, budget=None,
            coefficients=None):
        """the program's values: the shared FND-006 pass (plan "fused" / "repeated", or the planner's choice among
        devices / force) on backend "cpu" or "cuda", then the finalization on "cpu" or "gpu" (default: the backend's);
        the run's need (Program.need) is checked before anything is allocated"""
        try:
            self._greq._fresh()
        except StaleStateError:
            self._kept = None                       # a stale snapshot's states are never reused
            raise
        finalize = finalize or ("gpu" if backend == "cuda" else "cpu")
        if finalize not in ("cpu", "gpu") or backend not in ("cpu", "cuda"):
            raise ValueError("backend is 'cpu' or 'cuda', finalize 'cpu' or 'gpu'")
        if self.lifetime == "snapshot" and self._kept is not None:
            res, values, how = self._kept
            return ProgramResult(self, res, values, how, reused=True)
        evidence = None
        if plan is None:
            ev = self._rplan.choose(devices=devices or (backend,), force=force, coefficients=coefficients, budget=budget)
            if ev["chosen"] is None:
                mr._raise(ev["code"] or -12, None, "query program plan: " + (ev["reason"] or "refused"), plan=ev)
            plan, backend, evidence = ev["choice"]["plan"], ev["choice"]["device"], ev
        refusal = _Refusal()
        code = mbk._call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(self._need_raw(plan, backend, finalize)),
                         C.byref(refusal))
        if code:
            mr._raise(code, refusal, f"query program ({plan}@{backend}, finalized on the {finalize})", plan=evidence)
        nslots = int(self.info["slots"])
        values = (_Value * max(nslots, 1))()
        if backend == "cuda" and finalize == "gpu" and plan == "fused":
            res = self._run_device(budget, values, evidence)
        else:
            res = self._rplan.execute(plan, backend, budget=budget, evidence=evidence)
            if finalize == "cpu":
                refusal = _Refusal()
                _check(_call("tr_prog_finalize", self._open(), res._open(), C.cast(values, C.c_void_p), nslots,
                             C.byref(_budget(budget)), C.byref(refusal)), refusal, what="finalization")
            else:
                gpu = _gpu()
                outs = (_FOut * max(self.info["outputs"], 1))()
                fv = _FinalView()
                _check(_call("tr_prog_final_view_get", self._open(), res._open(), outs, len(outs), C.byref(fv)),
                       what="final view")
                refusal = _Refusal()
                gpu.run("tr_mgpu_program_finalize", C.byref(fv), C.byref(_budget(budget)), C.cast(values, C.c_void_p),
                        C.byref(refusal), refusal=refusal, what="finalization on the device")
        from . import math_gpu
        fin = finalize if finalize == "cpu" else ("cuda-emulated" if math_gpu.device()["emulated"] else "cuda")
        how = dict(plan=res.plan_kind, pass_device=res.backend, finalize_device=fin)
        if self.lifetime == "snapshot":
            self._kept = (res, values, how)
        return ProgramResult(self, res, values, how, reused=False)

    def _run_device(self, budget, values, evidence):
        """the fused pass on the device with its states kept there, finalized there (tr_mgpu_program_run)"""
        gpu = _gpu()
        rp = self._rplan
        b = _budget(budget)
        view, base = mr._View(), _Need()
        _check(mr._call("tr_reduce_get_view", rp._open(), C.byref(view)), what="plan view")
        _check(mr._call("tr_reduce_need", rp._open(), 0, C.byref(base)), what="needs")
        outs = (_FOut * max(self.info["outputs"], 1))()
        fv = _FinalView()
        _check(_call("tr_prog_final_view_get", self._open(), None, outs, len(outs), C.byref(fv)), what="final view")
        error, failed, refusal = C.c_int32(), C.c_int32(-1), _Refusal()
        stage = mr._call("tr_reduce_stage_create", rp._open(), 1, C.byref(b), C.byref(refusal), C.byref(error))
        if not stage:
            mr._raise(error.value or -2, refusal, "query program stage")
        arena, stats = mr._Arena(), mr._Stats()
        try:
            _check(mr._call("tr_reduce_stage_arena", stage, C.byref(arena)), what="stage")
            gpu.run("tr_mgpu_program_run", C.byref(view), C.byref(base), C.byref(fv), C.byref(b), C.byref(arena),
                    C.cast(values, C.c_void_p), C.byref(stats), C.byref(refusal), refusal=refusal,
                    what="query program on the device")
        except Exception:
            mr._call("tr_reduce_stage_free", stage)
            raise
        h = mr._call("tr_reduce_stage_finish", stage, 2, C.byref(stats), C.byref(b), C.byref(refusal),
                     C.byref(failed), C.byref(error))
        if not h or error.value:
            mr._raise(error.value or -2, refusal, "query program states")
        d = gpu.device()
        return mr.Result(h, rp, "fused", "cuda-emulated" if d["emulated"] else "cuda", d["name"], evidence)

    def drop(self):
        """ends a snapshot lifetime now: the kept states are released"""
        self._kept = None

    def close(self):
        self._kept = None
        if getattr(self, "_handle", None):
            if getattr(self, "_rplan", None) is not None:
                self._rplan.close()
            _call("tr_prog_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- results
class ProgramResult:
    """The values of one run (or a declared reuse) of a program: one QueryResult per query, the run's record."""

    def __init__(self, program, res, values, how, reused):
        self.program, self.result, self._values, self.how, self.reused = program, res, values, how, reused
        self.queries = {name: QueryResult(self, i, name) for i, name in enumerate(program.names)}

    def __getitem__(self, name):
        return self.queries[name]

    def __iter__(self):
        return iter(self.queries)

    @property
    def stats(self):
        s = self.result.stats
        if self.reused:
            s = dict(s, scans=0, kernels=0, native_calls=0, reused=True)
        else:
            s = dict(s, reused=False)
        return s


class QueryResult:
    """One query's result: rows() (SQL-like: present groups, or the single scalar row), values(g), explanation"""

    def __init__(self, pr, index, name):
        self._pr, self.index, self.name = pr, index, name
        prog = pr.program
        self.spec = prog.family.queries[index]
        self._measures = [m for m in prog.measures if m["query"] == index]
        qi = prog.query_info(name)
        self.scalar = bool(qi["scalar"])

    def _value(self, m, g):
        if m["measure"].kind == HIST:
            return self._pr.result[f"state{m['output']}"].cells(g)
        v = self._pr._values[m["slot"] + g]
        if v.status == 1:
            return None
        num = _i256(v.num)
        if m["measure"].kind == AVG:
            return Fraction(num, int(v.den))
        return num

    def _rows_of(self, g):
        m = self._measures[0] if self._measures else None
        if m is None:
            return 0
        if m["measure"].kind == HIST:
            return self._pr.result[f"state{m['output']}"].group(g)["rows"]
        return int(self._pr._values[m["slot"] + g].rows)

    def groups(self):
        """the group ids it reports: the present ones (rows > 0), or [0] for a scalar query"""
        if self.scalar:
            return [0]
        if not self._measures:
            return []
        G = self._measures[0]["groups"]
        return [g for g in range(G) if self._rows_of(g) > 0]

    def values(self, g):
        """{label: value} of group g (int, Fraction, None for SQL NULL; histogram cells)"""
        return {m["measure"].label(): self._value(m, g) for m in self._measures}

    def rows(self):
        """the SQL-like result: [{"key": decoded group key, label: value, ...}]"""
        out = []
        prog = self._pr.program
        keys = prog._greq._keys[0 if self.spec["group"] is None else
                                [None, *prog.family.groupings].index(self.spec["group"])]
        for g in self.groups():
            row = dict(key=() if self.scalar else (keys.key_values(g) if keys is not None else (g,)))
            row.update(self.values(g))
            out.append(row)
        return out

    @property
    def explanation(self):
        """the compiled sharing (Program.explain), the devices of this run, its memory and dispatch costs, and per
        measure its path: the device whose pass built its state and the one that finalized its values (a histogram's
        cells are its state as the pass built it: no finalization)"""
        prog = self._pr.program
        e = prog.explain(self.name)
        res = self._pr.result
        for s in e["states"]:
            s["state_bytes"] = res[f"state{s['state']}"].info["state_bytes"]
        st = self._pr.stats
        how = self._pr.how
        e["measures"] = [dict(measure=m["measure"].label(), kind=KINDS[m["measure"].kind], state=m["output"],
                              slots=0 if m["measure"].kind == HIST else m["groups"],
                              path=dict(state=how["pass_device"],
                                        values=None if m["measure"].kind == HIST else how["finalize_device"]))
                         for m in self._measures]
        e.update(devices=dict(how), reused=self._pr.reused,
                 dispatch=dict(shared_pass=dict(plan=self._pr.how["plan"], scans=st["scans"], kernels=st["kernels"],
                                                native_calls=st["native_calls"], queries=len(prog.names)),
                               finalize=dict(device=self._pr.how["finalize_device"], slots=e["value_slots"])),
                 memory=dict(states=[(s["state"], s["state_bytes"], 1 + len(s["shared_with"])) for s in e["states"]],
                             value_bytes=e["value_bytes"]))
        return e

    def __repr__(self):
        return f"QueryResult({self.name!r})"
