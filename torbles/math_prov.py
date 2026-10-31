# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROV-001: factorized provenance of small relational queries (torbles/math_prov.c; the evaluation of a provenance DAG
under many valuations in CUDA: math_prov_gpu.cu).

    from torbles import math_prov as mp
    st = mp.Store(budget=Budget(circuit=1 << 16))          # annotation limits: nodes, child entries, bytes
    R = st.source("R", {"k": [1, 1, 2, None], "a": [10, 20, 30, 40]})   # each row a variable ("R", i)
    S = st.source("S", {"k": [1, 2, 2], "b": [5, 6, 7]})
    J = R.join(S, on=[("k", "k")])                          # columns k, a, k_r, b; annotations x_R * x_S
    G = J.group(["a"])                                      # GROUP BY a (present groups only); group([]): global
    U = R.union(R2); F = R.where(("a", ">", 15), ("k", "is not null"))
    G.tuples(); G.polynomial(0); G.witnesses(0)             # N[X] (explicit), B's minimal witnesses
    G.counts(); G.exists(deletions=[("R", 0)])              # N / B under a valuation
    st.evaluate([G, J], "count", deletions=[[("R", 0)], [("S", 1)]], backend="cuda")   # many valuations at once
    st.info                                                 # nodes, children, shared nodes, hash-consing hits

Every result names its semiring ("B", "N", "N[X]"). The DAG is factorized (hash-consed, never expanded unless a
polynomial is asked for). Refusals: AlgebraBudgetError (annotation limits, expansions, outputs; before allocation),
DomainError (columns of different decimal scales, unions of different arities), OverflowError (a coefficient past
64 bits; an evaluation past 128 bits is reported per valuation), UnsupportedError, ValueError. Opt-in: `import
torbles` does not load this module.
"""
import ctypes as C
from numbers import Integral

import numpy as np

from . import algq
from .math_bank import _Array
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["Store", "Relation", "Plan", "Polynomial", "Witnesses", "Annotations", "monomial", "SEMIRINGS"]

SEMIRINGS = {"bool": 1, "B": 1, "count": 2, "N": 2, "poly": 3, "N[X]": 3}
NAMES = {1: "B", 2: "N", 3: "N[X]"}
OPS = {"=": 1, "==": 1, "<>": 2, "!=": 2, "<": 3, "<=": 4, ">": 5, ">=": 6, "is null": 7, "is not null": 8}
REL_OPS = {1: "source", 2: "select", 3: "union", 4: "join", 5: "group"}
P = C.POINTER
i32, u32, i64, u64, vp = C.c_int32, C.c_uint32, C.c_int64, C.c_uint64, C.c_void_p


class _Node(C.Structure):
    _native_ = "tr_prov_node"
    _fields_ = [("op", u32), ("n", u32), ("arg", u64)]


class _Pred(C.Structure):
    _native_ = "tr_prov_pred"
    _fields_ = [("op", i32), ("col", i32), ("col2", i32), ("reserved", i32), ("value", i64)]


class _Valuations(C.Structure):
    _native_ = "tr_prov_valuations"
    _fields_ = [("count", u64), ("offset", vp), ("value", vp), ("fallback", vp), ("var", vp)]


class _PlanView(C.Structure):
    _native_ = "tr_prov_plan_view"
    _fields_ = [("nodes", u64), ("nchild", u64), ("noutputs", u64), ("nvars", u64), ("node", vp), ("child", vp),
                ("output", vp)]


class _StoreInfo(C.Structure):
    _native_ = "tr_prov_store_info"
    _fields_ = [(n, u64) for n in ("nodes", "children", "variables", "sources", "shared", "hits", "bytes",
                                   "limit_nodes", "limit_children", "limit_bytes")]


class _RelInfo(C.Structure):
    _native_ = "tr_prov_rel_info"
    _fields_ = [(n, u64) for n in ("tuples", "bytes", "new_nodes", "store_nodes")] + [
        (n, i32) for n in ("ncols", "global_", "op", "reserved")]


class _PolyInfo(C.Structure):
    _native_ = "tr_prov_poly_info"
    _fields_ = [("terms", u64), ("pairs", u64), ("derivations", u64), ("degree", u32), ("semiring", u32)]


class _EvalInfo(C.Structure):
    _native_ = "tr_prov_eval_info"
    _fields_ = [("count", u64), ("noutputs", u64), ("overflowed", u64), ("chunk", u64), ("semiring", i32),
                ("reserved", i32)]


assert (C.sizeof(_Node), C.sizeof(_Pred), C.sizeof(_Valuations), C.sizeof(_PlanView), C.sizeof(_StoreInfo),
        C.sizeof(_RelInfo), C.sizeof(_PolyInfo), C.sizeof(_EvalInfo)) == (16, 24, 40, 56, 80, 48, 32, 40)

_native = None


def _api():
    global _native
    if _native is None:
        e, B, R = P(i32), P(_Budget), P(_Refusal)
        sig = {
            "tr_prov_store_create": (vp, [B, R, e]),
            "tr_prov_store_get_info": (i32, [vp, P(_StoreInfo)]),
            "tr_prov_store_nodes": (i32, [vp, u64, u64, vp]),
            "tr_prov_store_children": (i32, [vp, u64, u64, vp]),
            "tr_prov_store_variables": (i32, [vp, u64, u64, vp, vp]),
            "tr_prov_store_parents": (i32, [vp, u32, P(u64)]),
            "tr_prov_store_free": (None, [vp]),
            "tr_prov_source": (vp, [vp, u64, i32, vp, vp, P(u32), P(u32), B, R, e]),
            "tr_prov_source_scope": (vp, [vp, vp, vp, i32, P(u32), P(u32), B, R, e]),
            "tr_prov_select": (vp, [vp, vp, vp, u32, B, R, e]),
            "tr_prov_union": (vp, [vp, vp, vp, B, R, e]),
            "tr_prov_join": (vp, [vp, vp, vp, vp, vp, u32, B, R, e]),
            "tr_prov_group": (vp, [vp, vp, vp, i32, B, R, e]),
            "tr_prov_need": (i32, [vp, i32, vp, vp, vp, vp, u32, P(_Need), P(_Need)]),
            "tr_prov_rel_get_info": (i32, [vp, P(_RelInfo)]),
            "tr_prov_rel_get": (i32, [vp, vp, vp, vp, vp, u64]),
            "tr_prov_rel_free": (None, [vp]),
            "tr_prov_plan_build": (vp, [vp, vp, u64, B, R, e]),
            "tr_prov_plan_get_view": (i32, [vp, P(_PlanView)]),
            "tr_prov_plan_free": (None, [vp]),
            "tr_prov_evaluate": (i32, [vp, i32, P(_Valuations), vp, vp, P(_EvalInfo), B, R]),
            "tr_prov_expand": (vp, [vp, u32, B, R, e]),
            "tr_prov_witnesses": (vp, [vp, u32, B, R, e]),
            "tr_prov_expand_need": (i32, [vp, u32, i32, P(_Need)]),
            "tr_prov_poly_get_info": (i32, [vp, P(_PolyInfo)]),
            "tr_prov_poly_get": (i32, [vp, vp, vp, vp, vp, u64, u64]),
            "tr_prov_poly_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_prov_evaluate", C.c_int, [P(_PlanView), i32, P(_Valuations), vp, vp, P(_EvalInfo),
                                                    P(_Budget), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _needs(raw):
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _i64(v, what):
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError(f"{what} must be an exact integer (a logical value)")
    v = int(v)
    if not -(1 << 63) <= v < (1 << 63):
        raise OverflowError(f"{what} does not fit int64")
    return v


def _semiring(s):
    if s not in SEMIRINGS:
        raise ValueError(f"semiring is one of {sorted(SEMIRINGS)}")
    return SEMIRINGS[s]


def monomial(*labels):
    """the canonical monomial of the source tuples given (repeats are powers): a sorted tuple of labels"""
    return tuple(sorted(labels))


# ---------------------------------------------------------------------- results (each in its semiring)
class Polynomial:
    """An expanded provenance polynomial in N[X]: {monomial: coefficient}, a monomial being a sorted tuple of source
    labels (name, row), repeated for powers."""
    semiring = "N[X]"

    def __init__(self, terms, derivations, degree):
        self.terms, self.derivations, self.degree = terms, derivations, degree

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return self.terms == other.terms
        if isinstance(other, dict):
            return self.terms == other
        return NotImplemented

    __hash__ = None

    def __repr__(self):
        return f"Polynomial(N[X], {len(self.terms)} monomials, {self.derivations} derivations)"


class Witnesses:
    """The minimal sets of source tuples that suffice (the Boolean provenance), as frozensets of labels."""
    semiring = "B"

    def __init__(self, sets):
        self.sets = sets

    def __eq__(self, other):
        if isinstance(other, Witnesses):
            other = other.sets
        return set(self.sets) == set(other) if isinstance(other, (list, set, tuple)) else NotImplemented

    __hash__ = None

    def __repr__(self):
        return f"Witnesses(B, {len(self.sets)} minimal sets)"


class Annotations:
    """Annotations under valuations, in one semiring ("B" or "N"): values[w][j] for valuation w and output j (the
    tuples of the evaluated relations in order); status[w] is None or "overflow" (N past 128 bits)."""

    def __init__(self, semiring, values, status, backend, chunk, splits):
        self.semiring, self.values, self.status, self.backend, self.chunk = semiring, values, status, backend, chunk
        self._splits = splits

    def of(self, w, k=0):
        """valuation w's values for the k-th evaluated relation"""
        a, b = self._splits[k], self._splits[k + 1]
        return self.values[w][a:b]

    def __repr__(self):
        return f"Annotations({self.semiring}, {len(self.values)} valuations, backend={self.backend})"


# ---------------------------------------------------------------------- the store
class Store:
    """A shared provenance DAG (append-only, hash-consed) with its annotation limits (budget: circuit = nodes,
    support = child entries, bytes)."""

    def __init__(self, budget=None):
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prov_store_create", C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        self._handle = _made(h, error, refusal, "provenance store")
        self._sources = []                    # (name, column names)

    def _open(self):
        if not self._handle:
            raise ValueError("the store is closed")
        return self._handle

    @property
    def info(self):
        out = _StoreInfo()
        _check(_call("tr_prov_store_get_info", self._open(), C.byref(out)), what="store")
        return {n: int(getattr(out, n)) for n, _ in _StoreInfo._fields_}

    def _labels(self):
        n = self.info["variables"]
        src, row = np.zeros(max(n, 1), np.uint32), np.zeros(max(n, 1), np.uint64)
        _check(_call("tr_prov_store_variables", self._open(), 0, n, src.ctypes.data, row.ctypes.data), what="store")
        return [(self._sources[int(src[i])][0], int(row[i])) for i in range(n)]

    def variable(self, label):
        """the variable id of a source tuple (name, row)"""
        labels = self._labels()
        try:
            return labels.index(tuple(label))
        except ValueError:
            raise ValueError(f"no source tuple {label!r}") from None

    def parents(self, node):
        out = u64()
        _check(_call("tr_prov_store_parents", self._open(), int(node), C.byref(out)), what="store")
        return int(out.value)

    def source(self, name, columns, *, valid=None, scales=None, rows=None, budget=None):
        """a source relation: each selected row is a new variable (name, row); equal rows are one tuple, annotated
        with the sum of their variables. columns: {name: values} (None for NULL), logical int64 values"""
        names = list(columns)
        if not names:
            raise ValueError("a source has at least one column")
        n = len(columns[names[0]])
        keep, arrs = [], (_Array * len(names))()
        for c, cn in enumerate(names):
            vs = list(columns[cn])
            if len(vs) != n:
                raise ValueError("every column has the same length")
            ok = np.array([v is not None for v in vs] if (valid is None or cn not in valid) else
                          [bool(x) and v is not None for x, v in zip(valid[cn], vs)], dtype=np.uint8)
            vals = np.array([_i64(v, f"{cn}") if v is not None else 0 for v in vs] or [0], dtype=np.int64)
            keep += [vals, ok]
            arrs[c].values, arrs[c].valid = vals.ctypes.data, ok.ctypes.data
            arrs[c].scale = int((scales or {}).get(cn, 0))
        bm = None
        if rows is not None:
            mask = np.asarray(rows, dtype=bool)
            if len(mask) != n:
                raise ValueError("rows: one flag per row")
            bm = np.zeros((n + 63) // 64 or 1, dtype=np.uint64)
            for i in np.nonzero(mask)[0]:
                bm[i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        first, src, error, refusal = u32(), u32(), C.c_int32(), _Refusal()
        h = _call("tr_prov_source", self._open(), n, len(names), arrs, bm.ctypes.data if bm is not None else None,
                  C.byref(first), C.byref(src), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        rel = Relation._wrap(self, _made(h, error, refusal, f"source {name}"), names)
        self._sources.append((name, names))
        return rel

    def source_scope(self, name, scope, columns, *, budget=None):
        """a source relation from a FND-004 scope: its selected rows (row = the physical row) and columns"""
        if scope.source == "table":
            scope.require_fresh("provenance source")
        idx = (i32 * len(columns))(*[scope.names.index(c) for c in columns])
        scope._open()
        first, src, error, refusal = u32(), u32(), C.c_int32(), _Refusal()
        h = _call("tr_prov_source_scope", self._open(), scope._handle, idx, len(columns), C.byref(first),
                  C.byref(src), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        rel = Relation._wrap(self, _made(h, error, refusal, f"source {name}"), list(columns))
        self._sources.append((name, list(columns)))
        return rel

    def plan(self, relations, *, budget=None):
        """the immutable evaluation plan of the relations' annotations (their cone of the DAG)"""
        rels = [relations] if isinstance(relations, Relation) else list(relations)
        anns = np.concatenate([r.annotations() for r in rels] or [np.zeros(0, np.uint32)]).astype(np.uint32)
        splits = [0]
        for r in rels:
            splits.append(splits[-1] + r.info["tuples"])
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prov_plan_build", self._open(), anns.ctypes.data if len(anns) else None, len(anns),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        p = object.__new__(Plan)
        p._handle, p.store, p.splits = _made(h, error, refusal, "evaluation plan"), self, splits
        return p

    def _valuations(self, deletions, valuations, default):
        labels = {lab: i for i, lab in enumerate(self._labels())}

        def vid(lab):
            if tuple(lab) not in labels:
                raise ValueError(f"no source tuple {lab!r}")
            return labels[tuple(lab)]
        rows = []
        if deletions is not None:
            for d in deletions:
                items = [d] if isinstance(d, tuple) and len(d) == 2 and isinstance(d[0], str) else list(d)
                rows.append(({vid(lab): 0 for lab in items}, 1))
        else:
            for v in valuations:
                rows.append(({vid(lab): int(x) for lab, x in v.items()}, int(default)))
        offset = np.zeros(len(rows) + 1, np.uint64)
        var, value, fallback = [], [], np.zeros(max(len(rows), 1), np.uint64)
        for w, (over, fb) in enumerate(rows):
            for x in sorted(over):
                if not 0 <= over[x] < (1 << 64):
                    raise ValueError("valuation values are naturals below 2^64")
                var.append(x)
                value.append(over[x])
            offset[w + 1] = len(var)
            fallback[w] = fb
        keep = (offset, np.array(var or [0], np.uint32), np.array(value or [0], np.uint64), fallback)
        raw = _Valuations(count=len(rows), offset=keep[0].ctypes.data, value=keep[2].ctypes.data,
                          fallback=keep[3].ctypes.data, var=keep[1].ctypes.data)
        return raw, keep

    def evaluate(self, relations, semiring="count", *, deletions=None, valuations=None, default=1, backend="cpu",
                 budget=None, plan=None):
        """the annotations of the relations' tuples under each valuation, in B ("bool") or N ("count"): deletions
        (lists of source labels set to 0, the others 1) or valuations ({label: value}, the others `default`)"""
        sr = _semiring(semiring)
        if sr == 3:
            raise ValueError("N[X] is not evaluated under valuations: use Relation.polynomial")
        if (deletions is None) == (valuations is None):
            raise ValueError("give deletions= or valuations=")
        pl = plan or self.plan(relations, budget=budget)
        return pl.evaluate(sr, *self._valuations(deletions, valuations, default), backend=backend, budget=budget)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_prov_store_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Plan:
    """An immutable evaluation plan (the cone of some annotations), for any number of valuation batches."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Store.plan(relations)")

    def _open(self):
        if not self._handle:
            raise ValueError("the plan is closed")
        return self._handle

    def view(self):
        v = _PlanView()
        _check(_call("tr_prov_plan_get_view", self._open(), C.byref(v)), what="plan")
        return v

    @property
    def info(self):
        v = self.view()
        return dict(nodes=int(v.nodes), children=int(v.nchild), outputs=int(v.noutputs), variables=int(v.nvars))

    def evaluate(self, semiring, raw, keep, *, backend="cpu", budget=None):
        sr = SEMIRINGS[semiring] if isinstance(semiring, str) else int(semiring)
        n, m = int(raw.count), int(self.view().noutputs)
        out = np.zeros(max(n * m * (2 if sr == 2 else 1), 1), dtype=np.uint64 if sr == 2 else np.uint8)
        status = np.zeros(max(n, 1), dtype=np.int32)
        info, refusal = _EvalInfo(), _Refusal()
        if backend == "cpu":
            _check(_call("tr_prov_evaluate", self._open(), sr, C.byref(raw), out.ctypes.data, status.ctypes.data,
                         C.byref(info), C.byref(_budget(budget)), C.byref(refusal)), refusal, "provenance evaluation")
            label = "cpu"
        elif backend == "cuda":
            gpu = _gpu()
            v = self.view()
            gpu.run("tr_mgpu_prov_evaluate", C.byref(v), sr, C.byref(raw), out.ctypes.data, status.ctypes.data,
                    C.byref(info), C.byref(_budget(budget)), C.byref(refusal), refusal=refusal,
                    what="provenance evaluation on the device")
            label = "cuda-emulated" if gpu.device()["emulated"] else "cuda"
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        del keep
        self._last = (out, status)
        values = []
        for w in range(n):
            if sr == 2:
                row = out[2 * w * m:2 * (w + 1) * m]
                values.append([int(row[2 * j]) | int(row[2 * j + 1]) << 64 for j in range(m)])
            else:
                values.append([bool(x) for x in out[w * m:(w + 1) * m]])
        st = [None if s == 0 else "overflow" for s in status[:n]]
        return Annotations(NAMES[int(info.semiring)], values, st, label, int(info.chunk), self.splits)

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_prov_plan_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- relations
class Relation:
    """An immutable K-relation over the store's DAG: distinct tuples (NULL = NULL, ordered, NULL last), each with its
    annotation."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Store.source and the relational operators")

    @classmethod
    def _wrap(cls, store, handle, columns):
        self = object.__new__(cls)
        self.store, self._handle, self.columns = store, handle, tuple(columns)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the relation is closed")
        return self._handle

    @property
    def info(self):
        out = _RelInfo()
        _check(_call("tr_prov_rel_get_info", self._open(), C.byref(out)), what="relation")
        return dict(tuples=int(out.tuples), bytes=int(out.bytes), new_nodes=int(out.new_nodes),
                    store_nodes=int(out.store_nodes), columns=int(out.ncols), global_group=bool(out.global_),
                    op=REL_OPS.get(int(out.op)))

    def _raw(self):
        i = self.info
        n, k = i["tuples"], i["columns"]
        vals, ok = np.zeros(max(n * k, 1), np.int64), np.zeros(max(n * k, 1), np.uint8)
        ann, sc = np.zeros(max(n, 1), np.uint32), np.zeros(max(k, 1), np.int32)
        _check(_call("tr_prov_rel_get", self._open(), vals.ctypes.data, ok.ctypes.data, ann.ctypes.data, sc.ctypes.data,
                     max(n, 1)), what="relation")
        return n, k, vals, ok, ann[:n], sc[:k]

    def tuples(self):
        """the tuples in order, None for NULL"""
        n, k, vals, ok, _, _ = self._raw()
        return [tuple(int(vals[i * k + c]) if ok[i * k + c] else None for c in range(k)) for i in range(n)]

    def annotations(self):
        """the DAG node of each tuple"""
        return self._raw()[4].copy()

    @property
    def scales(self):
        return tuple(int(s) for s in self._raw()[5])

    def __len__(self):
        return self.info["tuples"]

    def _col(self, c):
        if isinstance(c, Integral) and not isinstance(c, bool):
            return int(c)
        if c not in self.columns:
            raise ValueError(f"no column {c!r} in {self.columns}")
        return self.columns.index(c)

    def _new(self, h, error, refusal, what, columns):
        return Relation._wrap(self.store, _made(h, error, refusal, what), columns)

    def where(self, *preds, budget=None):
        """SQL WHERE with a conjunction of (column, op, value or ("col", name)) / (column, "is [not] null")"""
        arr = (_Pred * max(len(preds), 1))()
        for i, p in enumerate(preds):
            op = OPS.get(p[1].lower() if isinstance(p[1], str) else p[1])
            if op is None:
                raise ValueError(f"unknown predicate operator {p[1]!r}")
            arr[i].op, arr[i].col, arr[i].col2 = op, self._col(p[0]), -1
            if op < 7:
                rhs = p[2]
                if isinstance(rhs, tuple) and len(rhs) == 2 and rhs[0] == "col":
                    arr[i].col2 = self._col(rhs[1])
                else:
                    arr[i].value = _i64(rhs, "a predicate constant")
            elif len(p) != 2:
                raise ValueError("is [not] null takes no operand")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prov_select", self.store._open(), self._open(), arr, len(preds), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return self._new(h, error, refusal, "selection", self.columns)

    def union(self, other, *, budget=None):
        """UNION ALL (equal tuples merge: annotations add)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prov_union", self.store._open(), self._open(), other._open(), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return self._new(h, error, refusal, "union", self.columns)

    def join(self, other, on=(), *, budget=None):
        """the equi-join on column pairs [(left, right), ...] (none: the Cartesian product); annotations multiply"""
        pairs = list(on)
        lc = (i32 * max(len(pairs), 1))(*[self._col(a) for a, _ in pairs])
        rc = (i32 * max(len(pairs), 1))(*[other._col(b) for _, b in pairs])
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prov_join", self.store._open(), self._open(), other._open(), lc, rc, len(pairs),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        cols = list(self.columns)
        for c in other.columns:
            cols.append(c if c not in cols else c + "_r")
        return self._new(h, error, refusal, "join", cols)

    def group(self, columns, *, budget=None):
        """GROUP BY columns: one tuple per present group (annotation: the sum of its members); [] = the global group"""
        cols = [self._col(c) for c in columns]
        arr = (i32 * max(len(cols), 1))(*cols)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prov_group", self.store._open(), self._open(), arr, len(cols), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return self._new(h, error, refusal, "group", [self.columns[c] for c in cols])

    def need(self, op, other=None, on=(), columns=()):
        """what join / union / group check: {"call": the operator's needs, "store": the store's after its bound}"""
        call, store = _Need(), _Need()
        ops = {"union": 3, "join": 4, "group": 5}
        if op == "join":
            pairs = list(on)
            lc = (i32 * max(len(pairs), 1))(*[self._col(a) for a, _ in pairs])
            rc = (i32 * max(len(pairs), 1))(*[other._col(b) for _, b in pairs])
            args = (other._open(), lc, rc, len(pairs))
        elif op == "group":
            cols = [self._col(c) for c in columns]
            args = (None, (i32 * max(len(cols), 1))(*cols), None, len(cols))
        elif op == "union":
            args = (other._open(), None, None, 0)
        else:
            raise ValueError("op is 'join', 'union' or 'group'")
        _check(_call("tr_prov_need", self.store._open(), ops[op], self._open(), *args, C.byref(call), C.byref(store)),
               what=f"{op} needs")
        return dict(call=_needs(call), store=_needs(store))

    def _poly(self, i, witnesses, budget):
        ann = self.annotations()
        if not 0 <= i < len(ann):
            raise IndexError("no such tuple")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_prov_witnesses" if witnesses else "tr_prov_expand", self.store._open(), int(ann[i]),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        h = _made(h, error, refusal, "witnesses" if witnesses else "expansion")
        try:
            info = _PolyInfo()
            _check(_call("tr_prov_poly_get_info", h, C.byref(info)), what="polynomial")
            t, pr = int(info.terms), int(info.pairs)
            coef, off = np.zeros(max(t, 1), np.uint64), np.zeros(t + 1, np.uint64)
            var, exp = np.zeros(max(pr, 1), np.uint32), np.zeros(max(pr, 1), np.uint32)
            _check(_call("tr_prov_poly_get", h, coef.ctypes.data, off.ctypes.data, var.ctypes.data, exp.ctypes.data,
                         max(t, 1), max(pr, 1)), what="polynomial")
        finally:
            _call("tr_prov_poly_free", h)
        labels = self.store._labels()
        terms = {}
        for k in range(t):
            mono = []
            for q in range(int(off[k]), int(off[k + 1])):
                mono += [labels[int(var[q])]] * int(exp[q])
            terms[tuple(sorted(mono))] = int(coef[k])
        return terms, info

    def polynomial(self, i, *, budget=None):
        """tuple i's derivation polynomial in N[X] (expanded; budgeted by its number of derivations)"""
        terms, info = self._poly(i, False, budget)
        assert NAMES[int(info.semiring)] == "N[X]"
        return Polynomial(terms, int(info.derivations), int(info.degree))

    def witnesses(self, i, *, budget=None):
        """tuple i's minimal sets of source tuples that suffice (Boolean provenance)"""
        terms, info = self._poly(i, True, budget)
        assert NAMES[int(info.semiring)] == "B"
        return Witnesses([frozenset(m) for m in terms])

    def expand_need(self, i, *, witnesses=False):
        raw = _Need()
        _check(_call("tr_prov_expand_need", self.store._open(), int(self.annotations()[i]), int(bool(witnesses)),
                     C.byref(raw)), what="expansion needs")
        return _needs(raw)

    def counts(self, *, deletions=(), valuation=None, default=1, backend="cpu", budget=None):
        """N: every tuple's multiplicity under one valuation (default: every source tuple once)"""
        a = self.store.evaluate(self, "count", **_one(deletions, valuation), default=default, backend=backend,
                                budget=budget)
        if a.status[0]:
            raise OverflowError("the multiplicity exceeds 128 bits")
        return a.values[0]

    def exists(self, *, deletions=(), valuation=None, backend="cpu", budget=None):
        """B: which tuples exist when the given sources are deleted (or under a 0/1 valuation)"""
        return self.store.evaluate(self, "bool", **_one(deletions, valuation), backend=backend, budget=budget).values[0]

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_prov_rel_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Relation({self.columns}, {len(self) if self._handle else 'closed'} tuples)"


def _one(deletions, valuation):
    if valuation is not None:
        return dict(valuations=[valuation])
    return dict(deletions=[list(deletions)])
