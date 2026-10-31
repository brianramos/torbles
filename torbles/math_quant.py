# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""QUANT-001 (existential and universal group predicates) and QUANT-002 (relational division and required-category
coverage): exact per-group states over the child rows of a FND-004 scope, natively on the CPU or with the CUDA kernels
(torbles/math_quant.c, math_quant_gpu.cu).

    from torbles import math_bank as mbk, math_bool as mb, math_quant as mq
    lines = mbk.Scope.from_table(db["lines"], ["qty"], group_by=[("oid", db["orders"], "id")])   # children by parent
    orders = mbk.Scope.from_table(db["orders"], [], group_by=["id"])                            # the parent rows
    q = mq.Quantifier.build(lines, [mb.condition("qty", ">", 5)], mb.var(0), parents=orders, backend="cpu")  # or "cuda"
    q.state(g)          # {"present", "rows", "true", "false", "unknown", "parents"}  exact ints
    q.answer("all", g)  # True / False / None (UNKNOWN) / mq.ABSENT; q.answers("some") codes; q.table(["some", "all"])
    cats = mbk.Scope.from_table(db["lines"], [], group_by=["product"])          # the same rows, grouped by category
    cov = mq.Coverage.build(lines, cats, ["apple", "pear"], parents=orders, representation="auto", backend="cpu")
    cov.state(g); cov.categories(g, "missing"); cov.answer("divides", g); cov.require(["fig"]); cov.masks()

Groups: the child scope's dense group ids (the parent key, possibly a dimension attribute). Presence is declared:
parents=None (a group exists iff it has child rows, SQL GROUP BY), "ids" (every id is a parent) or a Scope of the
parent rows (SQL's correlated subquery over a parent table: a present group may be empty; a NULL or unmatched child
key belongs to no parent). Quantifiers (SQL three-valued): some, all, none, not_all, exists, not_exists,
no_counterexample, all_true, bool_or, bool_and; vacuous truth for empty groups; an absent group answers ABSENT. Coverage:
the categories of a second scope over the same rows (its group ids), a dictionary (codes or keys) and a required set
(None = a NULL requirement, never satisfied); a NULL category never satisfies a requirement and is never extra;
duplicates count once. Every count is exact and native; budgets raise AlgebraBudgetError before anything is allocated,
a stale scope StaleStateError. Nothing is computed in Python. Opt-in: `import torbles` does not load it.
"""
import ctypes as C
import struct
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from .math_bank import Scope, _View as _ScopeView
from .math_bool import Expr, Program, _Condition, _ProgramView, _conditions
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Quantifier", "Dictionary", "Coverage", "ABSENT", "QUANTIFIERS", "QUESTIONS", "groups_need", "cover_need",
           "dictionary_need"]

QUANTIFIERS = {"some": 1, "all": 2, "none": 3, "not_all": 4, "exists": 5, "not_exists": 6, "no_counterexample": 7,
               "all_true": 8, "bool_or": 9, "bool_and": 10}
QUESTIONS = {"divides": 1, "exactly": 2, "overlaps": 3, "disjoint": 4}
GROUPED, IDS, PARENTS = 0, 1, 2
REPRESENTATIONS = {"auto": 0, "masks": 1, "sets": 2}
_REPR_NAMES = {1: "masks", 2: "sets"}
CODES, KEYS = 1, 2
LIST = {"present": 1, "missing": 2, "extra": 3}
NO_NULL = 0xFFFFFFFF
_NEED_CPU, _NEED_DEVICE, _NEED_ADOPT = 0, 1, 2
P = C.POINTER


class _Absent:
    """the answer of a group that does not exist (never FALSE, never UNKNOWN)"""
    __slots__ = ()

    def __repr__(self):
        return "ABSENT"

    def __bool__(self):
        raise TypeError("ABSENT has no truth value: the group does not exist")


ABSENT = _Absent()
_VALUE = {0: False, 1: True, 2: None, 3: ABSENT}


class _Counts(C.Structure):
    _native_ = "tr_quant_counts"
    _fields_ = [("ngroups", C.c_uint64)] + [(n, C.c_void_p) for n in ("rows", "qualifying", "failing", "unknown",
                                                                      "parents")]


class _GroupsInfo(C.Structure):
    _native_ = "tr_quant_groups_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "present", "empty", "absent", "rows", "qualifying", "failing",
                                          "unknown", "orphans", "parent_rows", "bytes", "identity_bytes")] + [
        ("digest", C.c_uint64 * 2)] + [(n, C.c_int32) for n in ("domain", "k", "nnodes", "parts")]


class _Keys(C.Structure):
    _native_ = "tr_quant_keys"
    _fields_ = [("bytes", C.c_void_p), ("offsets", C.c_void_p), ("n", C.c_uint64), ("null_key", C.c_int32),
                ("reserved", C.c_int32)]


class _DictInfo(C.Structure):
    _native_ = "tr_quant_dict_info"
    _fields_ = [(n, C.c_uint64) for n in ("size", "bytes", "key_bytes")] + [("digest", C.c_uint64 * 2),
                                                                             ("offset", C.c_int64), ("kind", C.c_int32),
                                                                             ("scale", C.c_int32),
                                                                             ("null_code", C.c_uint32),
                                                                             ("reserved", C.c_uint32)]


class _CoverArrays(C.Structure):
    _native_ = "tr_quant_cover_arrays"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "size", "words")] + [
        (n, C.c_void_p) for n in ("rows", "nulls", "parents", "distinct", "matched", "missing", "extra", "masks",
                                  "offsets", "codes")]


class _CoverInfo(C.Structure):
    _native_ = "tr_quant_cover_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "size", "words", "present", "empty", "absent", "rows", "nulls",
                                          "orphans", "parent_rows", "pairs", "required", "outside", "divides", "bytes",
                                          "identity_bytes")] + [
        ("digest", C.c_uint64 * 2), ("masks_cost", C.c_double), ("sets_cost", C.c_double)] + [
        (n, C.c_int32) for n in ("domain", "repr", "reason", "null_required")] + [("null_code", C.c_uint32),
                                                                                 ("parts", C.c_int32)]


for _s, _n in ((_Counts, 48), (_GroupsInfo, 128), (_Keys, 32), (_DictInfo, 64), (_CoverArrays, 104),
               (_CoverInfo, 184)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u32, u64, e = C.c_void_p, C.c_int32, C.c_uint32, C.c_uint64, P(C.c_int32)
        B, R, N = P(_Budget), P(_Refusal), P(_Need)
        sig = {
            "tr_quant_groups_need": (i32, [vp, P(_Condition), i32, vp, i32, vp, i32, N]),
            "tr_quant_groups_build": (vp, [vp, P(_Condition), i32, vp, i32, vp, B, R, e]),
            "tr_quant_groups_adopt": (vp, [vp, P(_Condition), i32, vp, i32, vp, P(_Counts), B, R, e]),
            "tr_quant_groups_merge": (vp, [vp, vp, B, R, e]),
            "tr_quant_groups_get_info": (i32, [vp, P(_GroupsInfo)]),
            "tr_quant_groups_get_counts": (i32, [vp, P(_Counts)]),
            "tr_quant_groups_answer": (i32, [vp, i32, vp, u64]),
            "tr_quant_groups_tally": (i32, [vp, i32, P(C.c_uint64 * 4)]),
            "tr_quant_groups_identity_size": (u64, [vp]),
            "tr_quant_groups_identity": (i32, [vp, vp, u64]),
            "tr_quant_groups_same": (i32, [vp, vp]),
            "tr_quant_groups_matches": (i32, [vp, vp, P(_Condition), i32, vp, i32, vp]),
            "tr_quant_groups_free": (None, [vp]),
            "tr_quant_dict_codes": (vp, [u64, u32, C.c_int64, i32, B, R, e]),
            "tr_quant_dict_keys": (vp, [u64, u32, P(_Keys), B, R, e]),
            "tr_quant_dict_need": (i32, [i32, u64, P(_Keys), N]),
            "tr_quant_dict_get_info": (i32, [vp, P(_DictInfo)]),
            "tr_quant_dict_find": (i32, [vp, vp, u64, P(C.c_uint32)]),
            "tr_quant_dict_mask": (i32, [vp, P(_Keys), vp, u64, P(C.c_uint64), P(C.c_uint64), B, R]),
            "tr_quant_dict_same": (i32, [vp, vp]),
            "tr_quant_dict_free": (None, [vp]),
            "tr_quant_cover_need": (i32, [vp, vp, vp, P(_Keys), i32, vp, i32, i32, N]),
            "tr_quant_cover_build": (vp, [vp, vp, vp, P(_Keys), i32, vp, i32, B, R, e]),
            "tr_quant_cover_adopt": (vp, [vp, vp, vp, P(_Keys), i32, vp, P(_CoverArrays), B, R, e]),
            "tr_quant_cover_require": (vp, [vp, P(_Keys), P(_CoverArrays), B, R, e]),
            "tr_quant_cover_merge": (vp, [vp, vp, B, R, e]),
            "tr_quant_cover_get_info": (i32, [vp, P(_CoverInfo)]),
            "tr_quant_cover_get_arrays": (i32, [vp, P(_CoverArrays)]),
            "tr_quant_cover_required_mask": (i32, [vp, vp, u64]),
            "tr_quant_cover_masks": (i32, [vp, vp]),
            "tr_quant_cover_list": (i32, [vp, u64, i32, vp, u64, P(C.c_uint64)]),
            "tr_quant_cover_required_codes": (i32, [vp, vp, u64, P(C.c_uint64)]),
            "tr_quant_cover_outside": (i32, [vp, vp, u64, vp, u64, P(C.c_uint64), P(C.c_uint64)]),
            "tr_quant_cover_answer": (i32, [vp, i32, vp, u64]),
            "tr_quant_cover_tally": (i32, [vp, i32, P(C.c_uint64 * 4)]),
            "tr_quant_cover_identity_size": (u64, [vp]),
            "tr_quant_cover_identity": (i32, [vp, vp, u64]),
            "tr_quant_cover_same": (i32, [vp, vp]),
            "tr_quant_cover_matches": (i32, [vp, vp, vp, vp, P(_Keys), i32, vp]),
            "tr_quant_cover_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_quant_groups", C.c_int, [P(_ScopeView), P(_Condition), C.c_int32, P(_ProgramView),
                                                   P(_ScopeView), P(_Budget), P(_Counts), P(_Refusal)])
    math_gpu.bind("tr_mgpu_quant_cover", C.c_int, [P(_ScopeView), P(_ScopeView), C.c_uint32, C.c_void_p, C.c_uint64,
                                                  C.c_uint64, P(_ScopeView), P(_Budget), P(_CoverArrays),
                                                  P(_Refusal)])
    math_gpu.bind("tr_mgpu_quant_cover_test", C.c_int, [P(_CoverArrays), C.c_void_p, C.c_uint64, C.c_uint64,
                                                       P(_Budget), P(_CoverArrays), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _ref(view):
    return None if view is None else C.byref(view)


def _needs_max(fn, targets):
    best = [0] * len(RESOURCES)
    for t in targets:
        out = _Need()
        _check(fn(t, C.byref(out)), what="needs")
        best = [max(a, int(b)) for a, b in zip(best, out.need)]
    return {name: v for name, v in zip(RESOURCES, best) if v}


def _u64_array(ptr, n):
    if not n or not ptr:
        return np.zeros(n, dtype=np.uint64)
    return np.ctypeslib.as_array((C.c_uint64 * n).from_address(ptr)).copy()


def _fresh(*scopes, what):
    for s in scopes:
        if isinstance(s, Scope) and s.source == "table":
            s.require_fresh(what)


def _codec_key(codec):
    """a comparable description of a key codec (kind, affine parameters, dictionary values)"""
    if codec is None:
        return None
    return (codec.kind, getattr(codec, "affine", None), getattr(codec, "offset", None),
            tuple(getattr(codec, "values", ()) or ()), getattr(codec, "sql_type", None))


def _domain(child, parents):
    """(native domain, parent scope or None), after the Python check that the groupings align"""
    if parents is None:
        return GROUPED, None
    if isinstance(parents, str):
        if parents != "ids":
            raise ValueError("parents: None (grouped), 'ids' or a math_bank.Scope of the parent rows")
        return IDS, None
    if not isinstance(parents, Scope):
        raise TypeError("parents: None (grouped), 'ids' or a math_bank.Scope of the parent rows")
    if parents.groups > child.groups:
        from .math_domains import DomainError
        raise DomainError(f"the parents have {parents.groups} groups, the children {child.groups}")
    if child.source == "table" and parents.source == "table":
        ck, pk = child.keys, parents.keys
        if len(ck) != len(pk):
            raise ValueError("the parent scope's grouping has another number of keys than the child's")
        for i, (a, b) in enumerate(zip(ck, pk)):
            if _codec_key(a["codec"]) != _codec_key(b["codec"]):
                raise ValueError(f"group key {i}: the parent key {b['name']!r} has another codec than the child key "
                                 f"{a['name']!r}: their codes do not name the same values")
            if (i and a["radix"] != b["radix"]) or b["radix"] > a["radix"]:
                raise ValueError(f"group key {i}: the parent radix {b['radix']} does not align with the child radix "
                                 f"{a['radix']} (parent group g must be child group g)")
    elif child.source != parents.source:
        raise ValueError("the parents and the children are both table scopes or both array scopes")
    return PARENTS, parents


def _predicate(predicate, k):
    if isinstance(predicate, Program):
        if predicate.k != k:
            from .math_domains import DomainError
            raise DomainError(f"the predicate reads {predicate.k} conditions, {k} given")
        return predicate
    if isinstance(predicate, Expr):
        return Program(predicate, k)
    raise TypeError("predicate: a math_bool.Program or a math_bool expression (var(i), & | ^ ~)")


# ====================================================================== QUANT-001
class Quantifier:
    """An immutable native quantifier state: per group the selected child rows and the rows where the predicate is
    TRUE, FALSE and UNKNOWN (SQL three-valued), the parent rows, and the group's presence (math_quant.h)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Quantifier.build(...) or Quantifier.merge(...)")

    @classmethod
    def _wrap(cls, handle, child, conditions, program, parents, backend, device, scopes):
        self = object.__new__(cls)
        self._handle, self.child, self.conditions, self.program = handle, child, conditions, program
        self.parents, self.backend, self.device, self._scopes = parents, backend, device, scopes
        return self

    @classmethod
    def build(cls, child, conditions, predicate, *, parents=None, backend="cpu", budget=None):
        """child: a math_bank.Scope of the child rows grouped by the parent key; conditions: math_bool.condition(...)
        over its columns; predicate: a math_bool.Program over them or an expression (var(i), & | ^ ~); parents: None,
        "ids" or a Scope of the parent rows (see the module doc). backend "cuda": the per-group scan on the device."""
        if not isinstance(child, Scope):
            raise TypeError("child: a math_bank.Scope")
        domain, pscope = _domain(child, parents)
        _fresh(child, pscope, what="quantifier state")
        conditions = tuple(conditions)
        arr, k = _conditions(conditions, child.names)
        program = _predicate(predicate, k)
        error, refusal = C.c_int32(), _Refusal()
        ph = pscope._handle if pscope is not None else None
        if _backend(backend) == "cpu":
            h = _call("tr_quant_groups_build", child._handle, arr, k, program._handle, domain, ph,
                      C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            return cls._wrap(_made(h, error, refusal, "quantifier state"), child, conditions, program, parents, "cpu",
                             "native CPU", (child, pscope))
        gpu = _gpu()
        need = groups_need(child, conditions, program, parents=parents, backend="cuda")
        from . import math_budgets
        math_budgets.check(budget or Budget(), "quantifier state on the device", **need)
        G = child.groups
        out = np.zeros((5, max(G, 1)), dtype=np.uint64)
        counts = _Counts(G, *[out[i].ctypes.data for i in range(5)])
        view, pview, prog = child.view(), (pscope.view() if pscope is not None else None), program.view()
        gpu.run("tr_mgpu_quant_groups", C.byref(view), arr, k, C.byref(prog), _ref(pview), C.byref(_budget(budget)),
                C.byref(counts), C.byref(refusal), refusal=refusal, what="quantifier scan on the device")
        h = _call("tr_quant_groups_adopt", child._handle, arr, k, program._handle, domain, ph, C.byref(counts),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        d = gpu.device()
        return cls._wrap(_made(h, error, refusal, "quantifier state (device counts)"), child, conditions, program,
                         parents, "cuda-emulated" if d["emulated"] else "cuda", d["name"], (child, pscope))

    def merge(self, other, *, budget=None):
        """the bag union with another state of the same predicate, domain and parents (e.g. another partition of the
        children): counts add exactly, nothing is ever erased"""
        self._use(); other._use()
        if list(self.child.names) != list(other.child.names):
            raise ValueError("merge states whose child scopes read the same columns in the same order")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_quant_groups_merge", self._handle, other._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Quantifier._wrap(_made(h, error, refusal, "merged quantifier state"), self.child, self.conditions,
                                self.program, self.parents, "cpu", "native CPU", self._scopes + other._scopes)

    # ------------------------------------------------------------------ reading
    def _open(self):
        if not self._handle:
            raise ValueError("the quantifier state is closed")

    def _use(self, what="quantifier state"):
        self._open()
        _fresh(*self._scopes, what=what)

    def fresh(self):
        self._open()
        return all(s.fresh() for s in self._scopes if isinstance(s, Scope) and s.source == "table")

    @property
    def info(self):
        self._open()
        i = _GroupsInfo()
        _check(_call("tr_quant_groups_get_info", self._handle, C.byref(i)), what="quantifier state")
        return dict(groups=int(i.ngroups), present=int(i.present), empty=int(i.empty), absent=int(i.absent),
                    rows=int(i.rows), true=int(i.qualifying), false=int(i.failing), unknown=int(i.unknown),
                    orphans=int(i.orphans), parent_rows=int(i.parent_rows), bytes=int(i.bytes),
                    identity_bytes=int(i.identity_bytes), digest=(int(i.digest[0]), int(i.digest[1])),
                    domain=("grouped", "ids", "parents")[i.domain], conditions=int(i.k), nodes=int(i.nnodes),
                    parts=int(i.parts), backend=self.backend, device=self.device)

    def states(self):
        """copies of the per-group arrays: {"rows", "true", "false", "unknown", "parents", "present"}"""
        self._use()
        c = _Counts()
        _check(_call("tr_quant_groups_get_counts", self._handle, C.byref(c)), what="quantifier state")
        G = int(c.ngroups)
        out = dict(rows=_u64_array(c.rows, G), true=_u64_array(c.qualifying, G), false=_u64_array(c.failing, G),
                   unknown=_u64_array(c.unknown, G), parents=_u64_array(c.parents, G))
        out["present"] = self.answers("exists") != 3
        return out

    def state(self, g):
        s = self.states()
        g = int(g)
        if not 0 <= g < len(s["rows"]):
            raise ValueError("group id out of range")
        return {k: (bool(v[g]) if k == "present" else int(v[g])) for k, v in s.items()}

    def answers(self, quantifier):
        """the answer codes of every group: 0 FALSE, 1 TRUE, 2 UNKNOWN, 3 ABSENT (numpy uint8)"""
        self._use()
        q = QUANTIFIERS.get(quantifier)
        if q is None:
            raise ValueError(f"quantifier: one of {', '.join(QUANTIFIERS)}")
        G = self.child.groups
        out = np.zeros(max(G, 1), dtype=np.uint8)
        _check(_call("tr_quant_groups_answer", self._handle, q, out.ctypes.data, len(out)), what="answers")
        return out[:G]

    def answer(self, quantifier, g):
        """True, False, None (UNKNOWN) or ABSENT"""
        a = self.answers(quantifier)
        if not 0 <= int(g) < len(a):
            raise ValueError("group id out of range")
        return _VALUE[int(a[int(g)])]

    def tally(self, quantifier):
        """the number of groups answering TRUE, FALSE, UNKNOWN and ABSENT"""
        self._use()
        q = QUANTIFIERS.get(quantifier)
        if q is None:
            raise ValueError(f"quantifier: one of {', '.join(QUANTIFIERS)}")
        out = (C.c_uint64 * 4)()
        _check(_call("tr_quant_groups_tally", self._handle, q, C.byref(out)), what="tally")
        return dict(false=int(out[0]), true=int(out[1]), unknown=int(out[2]), absent=int(out[3]))

    def table(self, quantifiers=tuple(QUANTIFIERS)):
        """one record per present group: its id, decoded key, counts and the answers asked for"""
        s = self.states()
        answers = {q: self.answers(q) for q in quantifiers}
        rows = []
        for g in np.flatnonzero(s["present"]).tolist():
            rec = dict(group=g, key=self.child.key_values(g), rows=int(s["rows"][g]), true=int(s["true"][g]),
                       false=int(s["false"][g]), unknown=int(s["unknown"][g]), parents=int(s["parents"][g]))
            rec.update({q: _VALUE[int(a[g])] for q, a in answers.items()})
            rows.append(rec)
        return rows

    @property
    def identity(self):
        """the exact identity (bytes): domain, the child and parent scope descriptors, the conditions and the program"""
        self._open()
        n = int(_call("tr_quant_groups_identity_size", self._handle))
        buf = (C.c_uint8 * max(n, 1))()
        _check(_call("tr_quant_groups_identity", self._handle, buf, n), what="identity")
        return bytes(buf[:n])

    def same(self, other):
        self._open(); other._open()
        return bool(_check(_call("tr_quant_groups_same", self._handle, other._handle), what="identity"))

    def matches(self, child, conditions, predicate, *, parents=None):
        """True when this state is exactly the build of these arguments (identity equality, never a hash alone)"""
        self._open()
        domain, pscope = _domain(child, parents)
        arr, k = _conditions(tuple(conditions), child.names)
        program = _predicate(predicate, k)
        return bool(_check(_call("tr_quant_groups_matches", self._handle, child._handle, arr, k, program._handle,
                                 domain, pscope._handle if pscope is not None else None), what="identity"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_quant_groups_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Quantifier(closed)"
        i = self.info
        return f"Quantifier(groups={i['groups']}, present={i['present']}, rows={i['rows']}, {self.backend})"


def groups_need(child, conditions, predicate, *, parents=None, backend="cpu"):
    """the needs a quantifier build checks: {resource: need} (cuda: the device copies and the adopted state, resource by
    resource the larger)"""
    domain, pscope = _domain(child, parents)
    arr, k = _conditions(tuple(conditions), child.names)
    program = _predicate(predicate, k)
    targets = (_NEED_CPU,) if _backend(backend) == "cpu" else (_NEED_DEVICE, _NEED_ADOPT)
    ph = pscope._handle if pscope is not None else None
    return _needs_max(lambda t, out: _call("tr_quant_groups_need", child._handle, arr, k, program._handle, domain, ph, t,
                                           out), targets)


# ====================================================================== QUANT-002: dictionaries
def _key_list(encoded, null_key=0):
    """(tr_quant_keys, the buffers it points into)"""
    data = b"".join(encoded)
    offs = np.zeros(len(encoded) + 1, dtype=np.uint64)
    if encoded:
        offs[1:] = np.cumsum([len(x) for x in encoded], dtype=np.uint64)
    buf = np.frombuffer(data, dtype=np.uint8).copy() if data else np.zeros(1, dtype=np.uint8)
    keys = _Keys(buf.ctypes.data, offs.ctypes.data, len(encoded), int(null_key), 0)
    return keys, (buf, offs)


class Dictionary:
    """The category dictionary (codec) of a coverage state, native: CODES (integer codes: key = code + offset, value =
    key / 10^scale) or KEYS (one exact byte string per code: the UTF-8 values of a string dictionary). One code may be
    the NULL category. Part of the state identity: another dictionary is another state."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Dictionary.codes(...), Dictionary.keys(...) or Dictionary.from_scope(scope)")

    @classmethod
    def _wrap(cls, handle, kind, values, null):
        self = object.__new__(cls)
        self._handle, self.kind, self.values, self.null = handle, kind, values, null
        i = self._raw_info()
        self.size, self.offset, self.scale = int(i.size), int(i.offset), int(i.scale)
        return self

    @classmethod
    def codes(cls, size, *, null=None, offset=0, scale=0, budget=None):
        """integer codes 0 .. size-1 meaning the values (code + offset) / 10^scale; null: the code of the NULL category"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_quant_dict_codes", int(size), NO_NULL if null is None else int(null), int(offset), int(scale),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "category dictionary"), CODES, None, null)

    @classmethod
    def keys(cls, values, *, null=None, budget=None):
        """one key per code (strings or bytes, distinct); the entry of the NULL code (null) must be None"""
        values = list(values)
        if null is not None and not (0 <= int(null) < len(values) and values[int(null)] is None):
            raise ValueError("the NULL code's entry is None")
        enc = []
        for c, v in enumerate(values):
            if c == null:
                enc.append(b"")
            elif isinstance(v, str):
                enc.append(v.encode("utf-8"))
            elif isinstance(v, (bytes, bytearray)):
                enc.append(bytes(v))
            else:
                raise TypeError(f"dictionary keys are strings or bytes (code {c}: {v!r})")
        keys, keep = _key_list(enc)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_quant_dict_keys", len(values), NO_NULL if null is None else int(null), C.byref(keys),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return cls._wrap(_made(h, error, refusal, "category dictionary"), KEYS, values, null)

    @classmethod
    def from_scope(cls, categories, *, budget=None):
        """the dictionary of a category scope's codes: its key's codec (a string dictionary: KEYS; an integer, decimal or
        boolean codec: CODES with its offset and scale; none: the codes themselves) and its NULL code; an array scope's
        group ids are the codes"""
        if not isinstance(categories, Scope):
            raise TypeError("categories: a math_bank.Scope")
        D = categories.groups
        if categories.source == "arrays":
            return cls.codes(D, budget=budget)
        if len(categories.keys) != 1:
            raise UnsupportedError("a category scope has one group key (its codes are the categories)")
        key = categories.keys[0]
        null, codec = key["null_code"], key["codec"]
        if codec is None:
            return cls.codes(D, null=null, budget=budget)
        if codec.kind == "dict":
            if getattr(codec, "numeric", False) or codec.sql_type == "BLOB":
                raise UnsupportedError(f"a {codec.sql_type} dictionary is not a supported category dictionary "
                                       "(strings or integer codes)")
            vals = list(codec.values)
            entries = [None if c == null else (vals[c] if c < len(vals) else None) for c in range(D)]
            if any(e is None for c, e in enumerate(entries) if c != null):
                raise ValueError("a category code has no dictionary value")
            return cls.keys(entries, null=null, budget=budget)
        if getattr(codec, "affine", None) is not None:
            offset, scale = codec.affine
            return cls.codes(D, null=null, offset=int(offset), scale=int(scale), budget=budget)
        raise UnsupportedError(f"a {codec.kind} column is not a supported category (strings or integer codes)")

    def _open(self):
        if not self._handle:
            raise ValueError("the dictionary is closed")

    def _raw_info(self):
        self._open()
        i = _DictInfo()
        _check(_call("tr_quant_dict_get_info", self._handle, C.byref(i)), what="dictionary")
        return i

    @property
    def info(self):
        i = self._raw_info()
        return dict(kind=("codes", "keys")[i.kind - 1], size=int(i.size), null=None if i.null_code == NO_NULL else
                    int(i.null_code), offset=int(i.offset), scale=int(i.scale), bytes=int(i.bytes),
                    key_bytes=int(i.key_bytes), digest=(int(i.digest[0]), int(i.digest[1])))

    def encode(self, value):
        """the canonical key bytes of a category value (CODES: the 8-byte little-endian logical int64; KEYS: UTF-8)"""
        if self.kind == KEYS:
            if isinstance(value, str):
                return value.encode("utf-8")
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            raise TypeError(f"a key of this dictionary is a string (got {value!r})")
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, float):
            raise TypeError(f"{value!r}: a float is not an exact category (give an int, Fraction, Decimal or string)")
        if isinstance(value, (Integral, Rational, Decimal, str)):
            f = Fraction(value) * 10 ** self.scale
        else:
            raise TypeError(f"{value!r} is not an exact category value")
        if f.denominator != 1:
            raise ValueError(f"{value!r} is not a value of this dictionary (scale {self.scale})")
        if not -(1 << 63) <= f.numerator < (1 << 63):
            raise OverflowError(f"{value!r} does not fit an int64 logical value")
        return struct.pack("<q", f.numerator)

    def decode_key(self, key):
        if self.kind == KEYS:
            return key.decode("utf-8")
        v = struct.unpack("<q", key)[0]
        return v if not self.scale else Fraction(v, 10 ** self.scale)

    def value(self, code):
        """the category value of a code (None for the NULL code)"""
        code = int(code)
        if not 0 <= code < self.size:
            raise ValueError("code out of range")
        if self.null is not None and code == int(self.null):
            return None
        if self.kind == KEYS:
            return self.values[code]
        v = code + self.offset
        return v if not self.scale else Fraction(v, 10 ** self.scale)

    def find(self, value):
        """the code of a category value, or None (not in the dictionary)"""
        self._open()
        key = self.encode(value)
        buf = np.frombuffer(key, dtype=np.uint8).copy() if key else np.zeros(1, np.uint8)
        code = C.c_uint32()
        r = _check(_call("tr_quant_dict_find", self._handle, buf.ctypes.data, len(key), C.byref(code)), what="find")
        return int(code.value) if r else None

    def required(self, values):
        """(tr_quant_keys, buffers) of a required set: None is the NULL requirement"""
        values = list(values)
        null = any(v is None for v in values)
        return _key_list([self.encode(v) for v in values if v is not None], null_key=int(null))

    def same(self, other):
        self._open(); other._open()
        return bool(_check(_call("tr_quant_dict_same", self._handle, other._handle), what="dictionary"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_quant_dict_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Dictionary({('codes', 'keys')[self.kind - 1]}, size={self.size})" if self._handle else "Dictionary(closed)"


# ====================================================================== QUANT-002: coverage states
class Coverage:
    """An immutable native coverage state: per group the present categories (bit masks or sorted sets), the rows and
    NULL-category rows, the parent rows, and the coverage of a required set (math_quant.h)."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Coverage.build(...), .require(...) or .merge(...)")

    @classmethod
    def _wrap(cls, handle, child, categories, dictionary, parents, required, backend, device, scopes):
        self = object.__new__(cls)
        self._handle, self.child, self.category_scope, self.dictionary = handle, child, categories, dictionary
        self.parents, self.required_values, self.backend, self.device = parents, required, backend, device
        self._scopes = scopes
        return self

    @classmethod
    def build(cls, child, categories, required, *, dictionary=None, parents=None, representation="auto",
              backend="cpu", budget=None):
        """child: the child rows grouped by the parent key; categories: a Scope of the same rows whose group ids are the
        categories (e.g. group_by=["product"]); required: category values (None = a NULL requirement); dictionary:
        default Dictionary.from_scope(categories); parents as for Quantifier; representation "auto" (costed choice),
        "masks" or "sets"; backend "cuda": the masks and the coverage test on the device (masks only)."""
        if not isinstance(child, Scope) or not isinstance(categories, Scope):
            raise TypeError("child and categories: math_bank.Scope objects over the same rows")
        domain, pscope = _domain(child, parents)
        _fresh(child, categories, pscope, what="coverage state")
        if dictionary is None:
            dictionary = Dictionary.from_scope(categories)
        repr_ = REPRESENTATIONS.get(representation)
        if repr_ is None:
            raise ValueError("representation: 'auto', 'masks' or 'sets'")
        required = tuple(required)
        keys, keep = dictionary.required(required)
        error, refusal = C.c_int32(), _Refusal()
        ph = pscope._handle if pscope is not None else None
        scopes = (child, categories, pscope)
        if _backend(backend) == "cpu":
            h = _call("tr_quant_cover_build", child._handle, categories._handle, dictionary._handle, C.byref(keys),
                      domain, ph, repr_, C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
            return cls._wrap(_made(h, error, refusal, "coverage state"), child, categories, dictionary, parents,
                             required, "cpu", "native CPU", scopes)
        if repr_ == REPRESENTATIONS["sets"]:
            raise UnsupportedError("the CUDA path builds masks (representation 'masks' or 'auto')")
        gpu = _gpu()
        need = cover_need(child, categories, required, dictionary=dictionary, parents=parents, representation="masks",
                          backend="cuda")
        from . import math_budgets
        math_budgets.check(budget or Budget(), "coverage state on the device", **need)
        G, D = child.groups, dictionary.size
        W = (D + 63) // 64
        mask = np.zeros(max(W, 1), dtype=np.uint64)
        nreq, cm = C.c_uint64(), C.c_uint64()
        _check(_call("tr_quant_dict_mask", dictionary._handle, C.byref(keys), mask.ctypes.data, W, C.byref(nreq),
                     C.byref(cm), C.byref(_budget(budget)), C.byref(refusal)), refusal, "required set")
        cols = np.zeros((7, max(G, 1)), dtype=np.uint64)
        masks = np.zeros(max(G * W, 1), dtype=np.uint64)
        arrays = _CoverArrays(G, D, W, *[cols[i].ctypes.data for i in range(7)], masks.ctypes.data, None, None)
        view, kview = child.view(), categories.view()
        pview = pscope.view() if pscope is not None else None
        gpu.run("tr_mgpu_quant_cover", C.byref(view), C.byref(kview), NO_NULL if dictionary.null is None else
                int(dictionary.null), mask.ctypes.data, nreq.value, cm.value, _ref(pview), C.byref(_budget(budget)),
                C.byref(arrays), C.byref(refusal), refusal=refusal, what="coverage scan on the device")
        h = _call("tr_quant_cover_adopt", child._handle, categories._handle, dictionary._handle, C.byref(keys), domain,
                  ph, C.byref(arrays), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        d = gpu.device()
        return cls._wrap(_made(h, error, refusal, "coverage state (device arrays)"), child, categories, dictionary,
                         parents, required, "cuda-emulated" if d["emulated"] else "cuda", d["name"], scopes)

    def require(self, required, *, backend="cpu", budget=None):
        """the state of another required set over the same presence states (no rescan), under its own identity;
        backend "cuda": the coverage test on the device (masks)"""
        self._use()
        required = tuple(required)
        keys, keep = self.dictionary.required(required)
        error, refusal = C.c_int32(), _Refusal()
        if _backend(backend) == "cpu":
            h = _call("tr_quant_cover_require", self._handle, C.byref(keys), None, C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
            return Coverage._wrap(_made(h, error, refusal, "coverage state"), self.child, self.category_scope,
                                  self.dictionary, self.parents, required, "cpu", "native CPU", self._scopes)
        info = self._raw_info()
        if info.repr != REPRESENTATIONS["masks"]:
            raise UnsupportedError("the CUDA coverage test reads masks (this state holds sets)")
        gpu = _gpu()
        G, D, W = int(info.ngroups), int(info.size), int(info.words)
        mask = np.zeros(max(W, 1), dtype=np.uint64)
        nreq, cm = C.c_uint64(), C.c_uint64()
        _check(_call("tr_quant_dict_mask", self.dictionary._handle, C.byref(keys), mask.ctypes.data, W, C.byref(nreq),
                     C.byref(cm), C.byref(_budget(budget)), C.byref(refusal)), refusal, "required set")
        src = _CoverArrays()
        _check(_call("tr_quant_cover_get_arrays", self._handle, C.byref(src)), what="coverage state")
        cols = np.zeros((4, max(G, 1)), dtype=np.uint64)
        out = _CoverArrays(G, D, W, None, None, None, *[cols[i].ctypes.data for i in range(4)], None, None, None)
        gpu.run("tr_mgpu_quant_cover_test", C.byref(src), mask.ctypes.data, nreq.value, cm.value,
                C.byref(_budget(budget)), C.byref(out), C.byref(refusal), refusal=refusal,
                what="coverage test on the device")
        h = _call("tr_quant_cover_require", self._handle, C.byref(keys), C.byref(out), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        d = gpu.device()
        return Coverage._wrap(_made(h, error, refusal, "coverage state (device test)"), self.child, self.category_scope,
                              self.dictionary, self.parents, required, "cuda-emulated" if d["emulated"] else "cuda",
                              d["name"], self._scopes)

    def merge(self, other, *, budget=None):
        """the bag union with another state of the same dictionary, required set, domain and parents: rows add, a
        category present in either is present (masks OR, sets union)"""
        self._use(); other._use()
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_quant_cover_merge", self._handle, other._handle, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Coverage._wrap(_made(h, error, refusal, "merged coverage state"), self.child, self.category_scope,
                              self.dictionary, self.parents, self.required_values, "cpu", "native CPU",
                              self._scopes + other._scopes)

    # ------------------------------------------------------------------ reading
    def _open(self):
        if not self._handle:
            raise ValueError("the coverage state is closed")

    def _use(self, what="coverage state"):
        self._open()
        _fresh(*self._scopes, what=what)

    def fresh(self):
        self._open()
        return all(s.fresh() for s in self._scopes if isinstance(s, Scope) and s.source == "table")

    def _raw_info(self):
        self._open()
        i = _CoverInfo()
        _check(_call("tr_quant_cover_get_info", self._handle, C.byref(i)), what="coverage state")
        return i

    @property
    def info(self):
        i = self._raw_info()
        return dict(groups=int(i.ngroups), categories=int(i.size), words=int(i.words), present=int(i.present),
                    empty=int(i.empty), absent=int(i.absent), rows=int(i.rows), nulls=int(i.nulls),
                    orphans=int(i.orphans), parent_rows=int(i.parent_rows), pairs=int(i.pairs),
                    required=int(i.required), outside=int(i.outside), null_required=bool(i.null_required),
                    divides=int(i.divides), bytes=int(i.bytes), identity_bytes=int(i.identity_bytes),
                    digest=(int(i.digest[0]), int(i.digest[1])), domain=("grouped", "ids", "parents")[i.domain],
                    representation=_REPR_NAMES[i.repr],
                    choice={0: "cheaper", 1: "only fit"}.get(int(i.reason), "forced"),
                    masks_cost=float(i.masks_cost), sets_cost=float(i.sets_cost), parts=int(i.parts),
                    backend=self.backend, device=self.device)

    def states(self):
        """copies of the per-group arrays and presence"""
        self._use()
        a = _CoverArrays()
        _check(_call("tr_quant_cover_get_arrays", self._handle, C.byref(a)), what="coverage state")
        G = int(a.ngroups)
        out = {n: _u64_array(getattr(a, n), G) for n in ("rows", "nulls", "parents", "distinct", "matched", "missing",
                                                         "extra")}
        out["present"] = self.answers("divides") != 3
        return out

    def state(self, g):
        s = self.states()
        g = int(g)
        if not 0 <= g < len(s["rows"]):
            raise ValueError("group id out of range")
        return {k: (bool(v[g]) if k == "present" else int(v[g])) for k, v in s.items()}

    def codes(self, g, which="present"):
        """the category codes of group g (present, missing among the in-dictionary required codes, or extra),
        increasing"""
        self._use()
        w = LIST.get(which)
        if w is None:
            raise ValueError("which: 'present', 'missing' or 'extra'")
        n = C.c_uint64()
        _check(_call("tr_quant_cover_list", self._handle, int(g), w, None, 0, C.byref(n)), what="categories")
        out = np.zeros(max(int(n.value), 1), dtype=np.uint32)
        _check(_call("tr_quant_cover_list", self._handle, int(g), w, out.ctypes.data, len(out), C.byref(n)),
               what="categories")
        return out[:int(n.value)].tolist()

    def outside(self):
        """the distinct required keys outside the dictionary (decoded) and whether NULL is required"""
        self._open()
        n, total = C.c_uint64(), C.c_uint64()
        _check(_call("tr_quant_cover_outside", self._handle, None, 0, None, 0, C.byref(n), C.byref(total)),
               what="required set")
        buf = np.zeros(max(int(total.value), 1), dtype=np.uint8)
        offs = np.zeros(int(n.value) + 1, dtype=np.uint64)
        _check(_call("tr_quant_cover_outside", self._handle, buf.ctypes.data, len(buf), offs.ctypes.data, len(offs),
                     C.byref(n), C.byref(total)), what="required set")
        raw = bytes(buf[:int(total.value)])
        keys = [self.dictionary.decode_key(raw[int(offs[i]):int(offs[i + 1])]) for i in range(int(n.value))]
        return keys, bool(self._raw_info().null_required)

    def categories(self, g, which="present"):
        """the category values of group g: present, extra, or missing (the in-dictionary required categories it lacks,
        then the required keys outside the dictionary, then None for a NULL requirement)"""
        values = [self.dictionary.value(c) for c in self.codes(g, which)]
        if which == "missing":
            out, null = self.outside()
            values += out + ([None] if null else [])
        return values

    def required(self):
        """the canonical required set: (in-dictionary values, outside values, NULL required)"""
        self._open()
        n = C.c_uint64()
        _check(_call("tr_quant_cover_required_codes", self._handle, None, 0, C.byref(n)), what="required set")
        out = np.zeros(max(int(n.value), 1), dtype=np.uint32)
        _check(_call("tr_quant_cover_required_codes", self._handle, out.ctypes.data, len(out), C.byref(n)),
               what="required set")
        keys, null = self.outside()
        return [self.dictionary.value(c) for c in out[:int(n.value)].tolist()], keys, null

    def answers(self, question):
        """the answer codes of every group: 0 FALSE, 1 TRUE, 3 ABSENT (numpy uint8)"""
        self._use()
        q = QUESTIONS.get(question)
        if q is None:
            raise ValueError(f"question: one of {', '.join(QUESTIONS)}")
        G = int(self._raw_info().ngroups)
        out = np.zeros(max(G, 1), dtype=np.uint8)
        _check(_call("tr_quant_cover_answer", self._handle, q, out.ctypes.data, len(out)), what="answers")
        return out[:G]

    def answer(self, question, g):
        a = self.answers(question)
        if not 0 <= int(g) < len(a):
            raise ValueError("group id out of range")
        return _VALUE[int(a[int(g)])]

    def tally(self, question):
        self._use()
        q = QUESTIONS.get(question)
        if q is None:
            raise ValueError(f"question: one of {', '.join(QUESTIONS)}")
        out = (C.c_uint64 * 4)()
        _check(_call("tr_quant_cover_tally", self._handle, q, C.byref(out)), what="tally")
        return dict(false=int(out[0]), true=int(out[1]), absent=int(out[3]))

    def table(self, questions=tuple(QUESTIONS)):
        s = self.states()
        answers = {q: self.answers(q) for q in questions}
        rows = []
        for g in np.flatnonzero(s["present"]).tolist():
            rec = dict(group=g, key=self.child.key_values(g), **{k: int(s[k][g]) for k in (
                "rows", "nulls", "parents", "distinct", "matched", "missing", "extra")})
            rec.update({q: _VALUE[int(a[g])] for q, a in answers.items()})
            rows.append(rec)
        return rows

    def masks(self):
        """the per-group category masks as a FIELD-001 matrix (groups x categories bits; a copy) - masks only"""
        self._use()
        from .math_field import Matrix, _Matrix
        view = _Matrix()
        _check(_call("tr_quant_cover_masks", self._handle, C.byref(view)), what="masks")
        rows, cols, stride = int(view.rows), int(view.cols), int(view.stride)
        return Matrix(rows, cols, _u64_array(view.words, rows * stride), stride)

    @property
    def identity(self):
        """the exact identity (bytes): the presence part (domain, scope descriptors, dictionary) and the requirement
        part (required codes, outside keys, NULL flag)"""
        self._open()
        n = int(_call("tr_quant_cover_identity_size", self._handle))
        buf = (C.c_uint8 * max(n, 1))()
        _check(_call("tr_quant_cover_identity", self._handle, buf, n), what="identity")
        return bytes(buf[:n])

    def same(self, other):
        self._open(); other._open()
        return bool(_check(_call("tr_quant_cover_same", self._handle, other._handle), what="identity"))

    def matches(self, child, categories, required, *, dictionary=None, parents=None):
        """True when this state is exactly the build of these arguments (identity equality)"""
        self._open()
        domain, pscope = _domain(child, parents)
        if dictionary is None:
            dictionary = Dictionary.from_scope(categories)
        keys, keep = dictionary.required(tuple(required))
        return bool(_check(_call("tr_quant_cover_matches", self._handle, child._handle, categories._handle,
                                 dictionary._handle, C.byref(keys), domain,
                                 pscope._handle if pscope is not None else None), what="identity"))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_quant_cover_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Coverage(closed)"
        i = self.info
        return (f"Coverage(groups={i['groups']}, categories={i['categories']}, required={i['required']}, "
                f"{i['representation']}, {self.backend})")


def cover_need(child, categories, required, *, dictionary=None, parents=None, representation="masks", backend="cpu"):
    """the needs of a coverage build of one representation ("masks" or "sets"; cuda: masks, the device copies and the
    adopted state, resource by resource the larger)"""
    domain, pscope = _domain(child, parents)
    if dictionary is None:
        dictionary = Dictionary.from_scope(categories)
    repr_ = REPRESENTATIONS.get(representation)
    if repr_ not in (1, 2):
        raise ValueError("representation: 'masks' or 'sets'")
    keys, keep = dictionary.required(tuple(required))
    targets = (_NEED_CPU,) if _backend(backend) == "cpu" else (_NEED_DEVICE, _NEED_ADOPT)
    ph = pscope._handle if pscope is not None else None
    return _needs_max(lambda t, out: _call("tr_quant_cover_need", child._handle, categories._handle, dictionary._handle,
                                           C.byref(keys), domain, ph, repr_, t, out), targets)


def dictionary_need(values=None, *, size=None):
    """the needs of a dictionary: KEYS (values: one key per code, None for the NULL code) or CODES (size)"""
    out = _Need()
    if values is None:
        _check(_call("tr_quant_dict_need", CODES, int(size), None, C.byref(out)), what="dictionary needs")
    else:
        values = list(values)
        keys, keep = _key_list([b"" if v is None else (v.encode("utf-8") if isinstance(v, str) else bytes(v))
                                for v in values])
        _check(_call("tr_quant_dict_need", KEYS, len(values), C.byref(keys), C.byref(out)), what="dictionary needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}
