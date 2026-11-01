# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PAIR-002: structured pairwise query planning, natively on the CPU or with the CUDA kernels
(torbles/math_pairplan.c, math_pairplan_gpu.cu; every contract in math_pairplan.h).

    from torbles import math_bank as mbk, math_kernel as mk, math_pairplan as pp
    scope = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=G)
    reqs = [dict(a=0, b=1), dict(a=0, ordered=True, self_pairs=False, mean=True)]
    plan = pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", devices=("cpu", "cuda"))
    plan.candidates            # support / features / prefix / direct x device: amounts, costs, needs, status, why
    out = plan.run()           # the cheapest admissible candidate: {"results": [{"value", "pairs", ...}], ...}
    pp.query(scope, ["x", "y"], pp.features(mk.polynomial(2, c=1)), reqs, force="direct@cuda")
    st = pp.retain(scope, ["x"], "support", weight="w")   # retained state: reused by later plans (retained=[st])
    pp.kernel("rbf", gamma=1)  # no exact value: refused natively (UnsupportedError), never approximated

S = sum over the declared pair set of w_i w_j k(x_i, x_j): within one group (a == b: ordered or unordered pairs,
self-pairs included or excluded) or between two (each cross pair once, or both orientations with ordered=True);
mean=True divides by the weighted pair count. Answers are exact ints / Fractions computed natively; Python only
marshals and formats. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

from . import algq
from . import math_kernel as mk
from . import math_registry as reg
from . import math_tensor as mt
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import RESOURCES, UnsupportedError, check as _check

__all__ = ["Kernel", "Plan", "State", "absdiff", "mann_whitney", "features", "table", "band", "sqdist", "kernel",
           "plan", "query", "retain", "requests", "eval_table", "default_rates", "REQUEST_DTYPE", "RESULT_DTYPE", "STRATEGIES",
           "MAX_COLUMNS", "TILE", "BITS_LIMIT"]

MAX_COLUMNS, TILE, BITS_LIMIT, WORDS = 8, 32, 1020, 16
KINDS = {"abs": 1, "mann_whitney": 2, "feature": 3, "table": 4, "band": 5, "sqdist": 6}
_REFUSED = {"rbf": 16, "gaussian": 16, "laplace": 17, "laplacian": 17}
ORDERED, SELF, MEAN = 1, 2, 4
STRATEGIES = {"support": 1, "features": 2, "prefix": 3, "direct": 4}
_STRATEGY_NAMES = {v: k for k, v in STRATEGIES.items()}
_RETAIN = {"support": 1, "features": 2, "prefix": 3, "rows": 4, "direct": 4}
_FORMS = {1: "support", 2: "rows"}
_POLICIES = {1: "listwise", 2: "category"}
_DEVICES = {1: "cpu", 2: "cuda"}
_REPRS = {0: None, 1: "dense", 2: "sparse"}
_STATUS = {1: "chosen", 2: "admissible", 3: "rejected"}
_REASON = {0: None, 1: "cheapest", 2: "forced", 3: "costlier", 4: "not forced", 5: "refused", 6: "over budget",
           7: "forced candidate refused", 8: "no candidate matches the force", 9: "no admissible candidate",
           10: "search refused"}
P = C.POINTER
i32, i64, u64, u8, vp = C.c_int32, C.c_int64, C.c_uint64, C.c_uint8, C.c_void_p


class _Spec(C.Structure):
    _native_ = "tr_pairplan_spec"
    _fields_ = [("kind", i32), ("ncols", i32), ("cols", i32 * MAX_COLUMNS), ("weight", i32), ("symmetric", i32),
                ("h", i64), ("dflt", i64), ("ntable", u64), ("table", vp), ("kernel", mk._Spec)]


class _Request(C.Structure):
    _native_ = "tr_pairplan_request"
    _fields_ = [("a", u64), ("b", u64), ("flags", i32), ("reserved", i32)]


class _Result(C.Structure):
    _native_ = "tr_pairplan_result"
    _fields_ = [("status", i32), ("empty", i32), ("evaluations", u64), ("num", mk._Wide), ("den", mk._Wide),
                ("pairs", mk._Wide)]


class _Totals(C.Structure):
    _native_ = "tr_pairplan_totals"
    _fields_ = [("rows", u64), ("entries", u64), ("W", mt._I128), ("A", mt._I128), ("Q", u64 * 4)]


class _TableView(C.Structure):
    _native_ = "tr_pairplan_table_view"
    _fields_ = [(n, i32) for n in ("form", "policy", "ncols", "reserved")] + [("ngroups", u64), ("nentries", u64)] + [
        (n, vp) for n in ("offset", "value", "nulls", "W", "Q", "count", "totals")] + [
        ("lo", i64 * MAX_COLUMNS), ("hi", i64 * MAX_COLUMNS)]


class _KernelView(C.Structure):
    _native_ = "tr_pairplan_kernel_view"
    _fields_ = [(n, i32) for n in ("kind", "ncols", "policy", "symmetric", "fkind", "degree", "nterms", "tbits")] + [
        ("c", i64), ("h", i64), ("dflt", i64), ("ntable", u64), ("table", vp), ("weight", i64 * mk.MAX_TERMS),
        ("exps", (u8 * MAX_COLUMNS) * mk.MAX_TERMS)]


class _Options(C.Structure):
    _native_ = "tr_pairplan_options"
    _fields_ = [(n, i32) for n in ("devices", "strategies", "force_strategy", "force_device", "force_repr",
                                   "custom_rates")] + [("reserved", i32 * 2),
                                                       ("rates", (C.c_double * len(reg.COST_COMPONENTS)) * 2)]


class _Candidate(C.Structure):
    _native_ = "tr_pairplan_candidate"
    _fields_ = [("reg", reg._Candidate)] + [(n, i32) for n in ("strategy", "device", "repr", "reuse", "exact", "policy",
                                                                "form", "bound_bits")] + [
        (n, u64) for n in ("evaluations", "native_requests", "entries", "largest", "state_bytes", "device_bytes",
                           "work_items", "reserved2")]


class _PlanInfo(C.Structure):
    _native_ = "tr_pairplan_plan_info"
    _fields_ = [(n, i32) for n in ("ncandidates", "chosen", "code", "reason", "forced", "devices", "kind",
                                   "policy")] + [(n, u64) for n in ("requests", "groups", "elements", "bytes")] + [
        ("refusal", _Refusal), ("profile_reused", i32), ("nretained", i32), ("reserved", u64)]


class _Exec(C.Structure):
    _native_ = "tr_pairplan_exec"
    _fields_ = [(n, i32) for n in ("candidate", "strategy", "device", "built")] + [
        (n, u64) for n in ("evaluations", "native_requests", "state_bytes", "requests")] + [("reserved", u64 * 2)]


class _StateInfo(C.Structure):
    _native_ = "tr_pairplan_state_info"
    _fields_ = [(n, i32) for n in ("strategy", "form", "policy", "ncols", "weight", "repr", "mass", "kind")] + [
        ("cols", i32 * MAX_COLUMNS)] + [(n, u64) for n in ("ngroups", "entries", "largest", "elements", "bytes")] + [
        ("reserved", u64 * 4)]


for _s, _n in ((_Spec, 2808), (_Request, 24), (_Result, 400), (_Totals, 80), (_TableView, 216), (_KernelView, 1096),
               (_Options, 224), (_Candidate, 472), (_PlanInfo, 104), (_Exec, 64), (_StateInfo, 136)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

REQUEST_DTYPE = np.dtype([("a", "<u8"), ("b", "<u8"), ("flags", "<i4"), ("reserved", "<i4")])
RESULT_DTYPE = np.dtype([("status", "<i4"), ("empty", "<i4"), ("evaluations", "<u8"), ("num", "<u8", (WORDS,)),
                         ("den", "<u8", (WORDS,)), ("pairs", "<u8", (WORDS,))])
assert REQUEST_DTYPE.itemsize == 24 and RESULT_DTYPE.itemsize == 400

_native = None


def _api():
    global _native
    if _native is None:
        e, R, B = P(C.c_int32), P(_Refusal), P(_Budget)
        sig = {
            "tr_pairplan_default_rates": (None, [vp]),
            "tr_pairplan_retain": (vp, [vp, P(_Spec), i32, i32, B, R, e]),
            "tr_pairplan_state_get_info": (i32, [vp, P(_StateInfo)]),
            "tr_pairplan_state_table_view": (i32, [vp, P(_TableView)]),
            "tr_pairplan_state_free": (None, [vp]),
            "tr_pairplan_make": (i32, [vp, P(_Spec), vp, u64, vp, i32, P(_Options), B, P(vp), R]),
            "tr_pairplan_plan_get_info": (i32, [vp, P(_PlanInfo)]),
            "tr_pairplan_plan_get": (i32, [vp, i32, P(_Candidate)]),
            "tr_pairplan_plan_kernel_view": (i32, [vp, P(_KernelView)]),
            "tr_pairplan_plan_free": (None, [vp]),
            "tr_pairplan_execute": (i32, [vp, i32, B, vp, P(_Exec), R]),
            "tr_pairplan_eval_table": (i32, [P(_TableView), P(_KernelView), vp, u64, vp, B, R]),
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
    lib = math_gpu.lib()
    fn = lib.tr_mgpu_pairplan_upload
    fn.restype, fn.argtypes = vp, [P(_TableView), P(_Budget), P(_Refusal), P(C.c_int)]
    math_gpu.bind("tr_mgpu_pairplan_query", C.c_int, [vp, P(_KernelView), vp, u64, vp, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_pairplan_bytes", u64, [vp])
    math_gpu.bind("tr_mgpu_pairplan_free", None, [vp])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _wide(w):
    """the exact int of a 1024-bit two's complement word array (a _Wide or a numpy row)"""
    words = w.w if hasattr(w, "w") else w
    v = 0
    for i in reversed(range(WORDS)):
        v = (v << 64) | int(words[i])
    return v - (1 << (64 * WORDS)) if v >> (64 * WORDS - 1) else v


def _i128(x):
    return (int(x.hi) << 64) | (int(x.lo) & ((1 << 64) - 1))


def _u256(words):
    v = 0
    for i in reversed(range(4)):
        v = (v << 64) | int(words[i])
    return v


def _i64(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise TypeError(f"{what} must be an exact integer")
    x = int(x)
    if not -(1 << 63) <= x < 1 << 63:
        raise OverflowError(f"{what} does not fit int64")
    return x


def default_rates():
    """the candidates' rates (relative work units per amount, not time): {"cpu": {component: rate}, "cuda": ...}"""
    raw = ((C.c_double * len(reg.COST_COMPONENTS)) * 2)()
    _call("tr_pairplan_default_rates", C.cast(raw, vp))
    return {dev: dict(zip(reg.COST_COMPONENTS, (float(v) for v in raw[d]))) for d, dev in enumerate(("cpu", "cuda"))}


# ------------------------------------------------------------------ kernels
class Kernel:
    """A pairwise kernel k(x, y): abs, mann_whitney, feature (a math_kernel.Kernel), table, band, sqdist, or a name
    without an exact value (rbf, laplace, ...: refused natively when used)."""

    def __init__(self, kind, **params):
        self.kind, self.params = kind, params

    @property
    def symmetric(self):
        if self.kind == "mann_whitney":
            return False
        if self.kind == "table":
            return bool(self.params.get("symmetric", True))
        return True

    def spec(self, scope, columns, weight=None):
        """the native spec over the scope's columns (and weight)"""
        columns = list(columns)
        s = _Spec()
        s.kind = KINDS.get(self.kind, _REFUSED.get(self.kind, 18))
        if not columns:
            raise ValueError("a pairwise statistic needs at least one feature column")
        if len(columns) > MAX_COLUMNS:
            raise UnsupportedError(f"at most {MAX_COLUMNS} feature columns")
        s.ncols = len(columns)
        for j, name in enumerate(columns):
            if name not in scope.names:
                raise ValueError(f"{name!r} is not a scope column")
            s.cols[j] = scope.names.index(name)
        if weight is None:
            s.weight = -1
        elif weight in scope.names:
            s.weight = scope.names.index(weight)
        else:
            raise ValueError(f"{weight!r} is not a scope column")
        keep = None
        if self.kind == "band":
            s.h = _i64(self.params["h"], "the band radius")
        elif self.kind == "table":
            rows = []
            for u, v, k in self.params["pairs"]:
                u = tuple(u) if isinstance(u, (tuple, list)) else (u,)
                v = tuple(v) if isinstance(v, (tuple, list)) else (v,)
                if len(u) != len(columns) or len(v) != len(columns):
                    raise ValueError(f"a table pair has one value per feature column ({len(columns)})")
                rows.append([_i64(x, "a table value") for x in u + v] + [_i64(k, "a table entry")])
            keep = np.ascontiguousarray(np.array(rows, dtype=np.int64).reshape(len(rows), 2 * len(columns) + 1))
            s.ntable, s.table = len(rows), keep.ctypes.data if rows else None
            s.dflt = _i64(self.params.get("default", 0), "the table default")
            s.symmetric = 1 if self.params.get("symmetric", True) else 0
        elif self.kind == "feature":
            s.kernel = self.params["kernel"].spec(scope, columns)
        return s, keep

    def __repr__(self):
        return f"Kernel({self.kind}, {self.params})"


def absdiff():
    """sum_k |x_k - y_k| (|x - y| for one column)"""
    return Kernel("abs")


def mann_whitney():
    """[x > y] + 1/2 [x = y] over one column (ties count half; asymmetric)"""
    return Kernel("mann_whitney")


def features(kern):
    """a KERNEL-001 kernel (math_kernel.polynomial / monomial / delta) over the feature columns"""
    if not isinstance(kern, mk.Kernel):
        raise TypeError("features: a math_kernel.Kernel")
    return Kernel("feature", kernel=kern)


def table(pairs, *, default=0, symmetric=True):
    """a declared similarity: pairs {(u, v): k} or [(u, v, k)] over value tuples (scalars for one column), default
    elsewhere; symmetric: k(v, u) = k(u, v) (give each pair once)"""
    items = [(u, v, k) for (u, v), k in pairs.items()] if isinstance(pairs, dict) else list(pairs)
    return Kernel("table", pairs=items, default=default, symmetric=symmetric)


def band(h):
    """[max_k |x_k - y_k| <= h]: the pairs within L-infinity radius h"""
    return Kernel("band", h=h)


def sqdist():
    """sum_k (x_k - y_k)^2"""
    return Kernel("sqdist")


def kernel(name, **params):
    """a kernel by name; rbf / gaussian / laplace / laplacian or an unknown name is refused natively (no exact value)"""
    makers = {"abs": absdiff, "absdiff": absdiff, "mann_whitney": mann_whitney, "sqdist": sqdist}
    if name in makers:
        return makers[name]()
    if name == "band":
        return band(params["h"])
    if name == "table":
        return table(params["pairs"], default=params.get("default", 0), symmetric=params.get("symmetric", True))
    if name == "feature":
        return features(params["kernel"])
    return Kernel(name, **params)


# ------------------------------------------------------------------ requests and answers
def requests(items):
    """a REQUEST_DTYPE array from dicts {"a", "b" (default a: within one group), "ordered", "self_pairs", "mean"}"""
    items = list(items)
    out = np.zeros(len(items), dtype=REQUEST_DTYPE)
    for i, q in enumerate(items):
        a = int(q["a"])
        b = a if q.get("b") is None else int(q["b"])
        if a < 0 or b < 0:
            raise ValueError("group ids are nonnegative")
        out["a"][i], out["b"][i] = a, b
        out["flags"][i] = (ORDERED if q.get("ordered") else 0) | (SELF if q.get("self_pairs") else 0) | \
                          (MEAN if q.get("mean") else 0)
    return out


def _answer(r, raise_errors=True, where="request"):
    status = int(r["status"])
    if status:
        if raise_errors:
            _check(status, what=where)
        return dict(status=status, value=None, num=None, den=None, pairs=None, empty=False, evaluations=0)
    num, den, pairs = _wide(r["num"]), _wide(r["den"]), _wide(r["pairs"])
    empty = bool(r["empty"])
    value = None
    if not empty:
        f = Fraction(num, den)
        value = f.numerator if f.denominator == 1 else f
    return dict(status=0, value=value, num=num, den=den, pairs=pairs, empty=empty, evaluations=int(r["evaluations"]))


def _ptr(a):
    return a.ctypes.data if len(a) else None


# ------------------------------------------------------------------ retained states
class State:
    """An immutable retained state (native): a support table, a rows table, a PAIR-001 prefix state or a KERNEL-001
    feature state over a scope (kept alive by this object)"""

    def __init__(self, handle, scope, columns, weight, kern, strategy):
        self._handle, self.scope, self.columns, self.weight = handle, scope, list(columns), weight
        self.kernel, self.strategy = kern, strategy
        self._device = None

    def _open(self):
        if not self._handle:
            raise ValueError("the retained state is closed")
        return self._handle

    @property
    def info(self):
        i = _StateInfo()
        _check(_call("tr_pairplan_state_get_info", self._open(), C.byref(i)), what="retained state")
        return dict(strategy=_STRATEGY_NAMES[int(i.strategy)], form=_FORMS.get(int(i.form)),
                    policy=_POLICIES.get(int(i.policy)), columns=list(self.columns), weight=self.weight,
                    repr=_REPRS.get(int(i.repr)), groups=int(i.ngroups), entries=int(i.entries),
                    largest=int(i.largest), elements=int(i.elements), bytes=int(i.bytes))

    def view(self):
        """the native table view (support / rows states; pointers into this state)"""
        v = _TableView()
        _check(_call("tr_pairplan_state_table_view", self._open(), C.byref(v)), what="table view (support or rows)")
        return v

    def totals(self, g):
        """{"rows", "entries", "W", "A", "Q"} of group g (support / rows states)"""
        v = self.view()
        if not 0 <= int(g) < int(v.ngroups):
            raise ValueError("group id out of range")
        t = _Totals.from_address(v.totals + int(g) * C.sizeof(_Totals))
        return dict(rows=int(t.rows), entries=int(t.entries), W=_i128(t.W), A=_i128(t.A), Q=_u256(t.Q))

    def entries(self, g):
        """[(values, count, W, Q)] of group g in table order (values: a tuple, None for a NULL coordinate; a rows table
        stores neither count nor Q: one element each, Q = w^2 by definition)"""
        v = self.view()
        G, n = int(v.ngroups), int(v.ncols)
        if not 0 <= int(g) < G:
            raise ValueError("group id out of range")
        off = np.ctypeslib.as_array((C.c_uint64 * (G + 1)).from_address(v.offset))
        b, e = int(off[g]), int(off[g + 1])
        out = []
        if e == b:
            return out
        N = int(v.nentries)
        vals = np.ctypeslib.as_array((C.c_int64 * (N * n)).from_address(v.value))
        W = np.ctypeslib.as_array((C.c_int64 * (2 * N)).from_address(v.W))
        nulls = np.ctypeslib.as_array((C.c_uint8 * N).from_address(v.nulls)) if v.nulls else None
        Q = np.ctypeslib.as_array((C.c_uint64 * (4 * N)).from_address(v.Q)) if v.Q else None
        cnt = np.ctypeslib.as_array((C.c_uint64 * N).from_address(v.count)) if v.count else None
        for i in range(b, e):
            m = int(nulls[i]) if nulls is not None else 0
            x = tuple(None if (m >> j) & 1 else int(vals[i * n + j]) for j in range(n))
            w = (int(W[2 * i + 1]) << 64) | (int(W[2 * i]) & ((1 << 64) - 1))
            q = _u256(Q[4 * i:4 * i + 4]) if Q is not None else w * w
            out.append((x, int(cnt[i]) if cnt is not None else 1, w, q))
        return out

    def _upload(self, budget=None):
        if self._device is None:
            gpu = _gpu()
            refusal, error = _Refusal(), C.c_int()
            h = gpu.lib().tr_mgpu_pairplan_upload(C.byref(self.view()), C.byref(_budget(budget)), C.byref(refusal),
                                                  C.byref(error))
            gpu.check(error.value, refusal, "table upload")
            if not h:
                raise MemoryError("table upload failed")
            self._device = h
        return self._device

    def device(self, budget=None):
        """the device copy (uploaded once, freed with this object): its bytes"""
        h = self._upload(budget)
        return int(_gpu().lib().tr_mgpu_pairplan_bytes(h))

    def close(self):
        if getattr(self, "_device", None):
            _gpu().lib().tr_mgpu_pairplan_free(self._device)
            self._device = None
        if getattr(self, "_handle", None):
            _call("tr_pairplan_state_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return "State(closed)" if not self._handle else f"State({self.strategy}, columns={self.columns})"


def retain(scope, columns, strategy, *, kernel=None, weight=None, repr="dense", budget=None):
    """a retained state for later plans: "support" (the sorted support with multiplicities), "rows" (the elements,
    DIRECT's table), "prefix" (PAIR-001's DIST-002 prefix state of one column), "features" (KERNEL-001's feature state
    of a features(...) kernel); kernel sets the NULL policy (default absdiff: listwise; a delta feature kernel: NULL a
    category)"""
    if strategy not in _RETAIN:
        raise ValueError(f"strategy is one of {', '.join(_RETAIN)}")
    if scope.source == "table":
        scope.require_fresh("retained pairwise state")
    kern = kernel if kernel is not None else absdiff()
    spec, keep = kern.spec(scope, columns, weight)
    r = 0
    if strategy == "features":
        if repr not in ("dense", "sparse"):
            raise ValueError("repr is 'dense' or 'sparse'")
        r = mt.REPRS[repr]
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_pairplan_retain", scope._handle, C.byref(spec), _RETAIN[strategy], r, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    _check(error.value, refusal, f"retained {strategy} state")
    if not h:
        raise MemoryError("retained state: native allocation failed")
    del keep
    return State(h, scope, columns, weight, kern, strategy)


# ------------------------------------------------------------------ plans
def _force(force):
    """"<strategy>[@cpu|@cuda][:dense|:sparse]" -> (strategy, device, repr)"""
    if force is None:
        return 0, 0, 0
    text, rp, dev = str(force), 0, 0
    if ":" in text:
        text, r = text.split(":", 1)
        if r not in ("dense", "sparse"):
            raise ValueError("a force's representation is :dense or :sparse")
        rp = mt.REPRS[r]
    if "@" in text:
        text, d = text.split("@", 1)
        if d not in ("cpu", "cuda"):
            raise ValueError("a force's device is @cpu or @cuda")
        dev = reg.ON_CPU if d == "cpu" else reg.ON_CUDA
    if text not in STRATEGIES:
        raise ValueError(f"a force names a strategy: {', '.join(STRATEGIES)}")
    return STRATEGIES[text], dev, rp


def plan(scope, columns, kern, items, *, weight=None, devices=("cpu",), strategies=None, retained=(), force=None,
         rates=None, budget=None):
    """every candidate (support, features dense / sparse, prefix, direct; CUDA support and direct with "cuda" in
    devices) with its validity, exact amounts, FND-008 costs, needs, status and reason; the cheapest admissible one
    chosen (or the forced one). A refused plan raises with .plan (the evidence)."""
    if scope.source == "table":
        scope.require_fresh("pairwise plan")
    spec, keep = kern.spec(scope, columns, weight)
    reqs = items if isinstance(items, np.ndarray) else requests(items)
    reqs = np.ascontiguousarray(reqs, dtype=REQUEST_DTYPE)
    o = _Options()
    o.devices = reg._devices(devices)
    o.strategies = 0 if strategies is None else sum(1 << STRATEGIES[s] for s in strategies)
    o.force_strategy, o.force_device, o.force_repr = _force(force)
    if rates is not None:
        o.custom_rates = 1
        base = default_rates()
        for d, dev in enumerate(("cpu", "cuda")):
            row = dict(base[dev], **rates.get(dev, {}))
            for c, comp in enumerate(reg.COST_COMPONENTS):
                o.rates[d][c] = float(row[comp])
    retained = [st for st in retained if st is not None]
    arr = (vp * max(len(retained), 1))(*[st._open() for st in retained])
    handle, refusal = vp(), _Refusal()
    rc = _call("tr_pairplan_make", scope._handle, C.byref(spec), _ptr(reqs), len(reqs), C.cast(arr, vp),
               len(retained), C.byref(o), C.byref(_budget(budget)), C.byref(handle), C.byref(refusal))
    del keep
    if not handle.value:
        _check(rc, refusal, "pairwise plan")
        raise MemoryError("pairwise plan: native allocation failed")
    p = Plan(handle.value, scope, list(columns), weight, kern, reqs, retained, budget)
    if rc:
        try:
            _check(rc, refusal if refusal.resource >= 0 else None, "pairwise plan: " + (p.reason or "refused"))
        except Exception as exc:
            exc.plan = p
            raise
    return p


def query(scope, columns, kern, items, *, weight=None, devices=("cpu",), strategies=None, retained=(), force=None,
          rates=None, budget=None, raw=False):
    """plan(...) and run the chosen candidate"""
    p = plan(scope, columns, kern, items, weight=weight, devices=devices, strategies=strategies, retained=retained,
             force=force, rates=rates, budget=budget)
    return p.run(budget=budget, raw=raw)


class Plan:
    """The evidence of a pairwise plan (native): every candidate, the choice and why; run() executes a candidate."""

    def __init__(self, handle, scope, columns, weight, kern, reqs, retained, budget):
        self._handle, self.scope, self.columns, self.weight, self.kernel = handle, scope, columns, weight, kern
        self.requests, self.retained, self._budget = reqs, list(retained), budget
        i = _PlanInfo()
        _check(_call("tr_pairplan_plan_get_info", handle, C.byref(i)), what="plan")
        self._info = i

    def _open(self):
        if not self._handle:
            raise ValueError("the plan is closed")
        return self._handle

    @property
    def info(self):
        i = self._info
        return dict(candidates=int(i.ncandidates), chosen=None if i.chosen < 0 else int(i.chosen), code=int(i.code),
                    reason=_REASON.get(i.reason), forced=bool(i.forced), requests=int(i.requests),
                    groups=int(i.groups), elements=int(i.elements), bytes=int(i.bytes),
                    kind={v: k for k, v in KINDS.items()}.get(int(i.kind)), policy=_POLICIES.get(int(i.policy)),
                    devices=[d for b, d in ((1, "cpu"), (2, "cuda")) if i.devices & b],
                    profile_reused=bool(i.profile_reused), retained=int(i.nretained),
                    refusal=None if not i.refusal.refused else dict(resource=RESOURCES[i.refusal.resource],
                                                                   need=int(i.refusal.need), limit=int(i.refusal.limit)),
                    cost_unit="relative work units; not time")

    chosen = property(lambda self: self.info["chosen"])
    reason = property(lambda self: self.info["reason"])

    def candidate(self, i):
        c = _Candidate()
        _check(_call("tr_pairplan_plan_get", self._open(), int(i), C.byref(c)), what="plan candidate")
        g = c.reg
        return dict(index=int(i), strategy=_STRATEGY_NAMES[int(c.strategy)], device=_DEVICES[int(c.device)],
                    repr=_REPRS.get(int(c.repr)), name=self._name(c), status=_STATUS.get(g.status),
                    reason=_REASON.get(g.reason), valid=bool(g.valid), code=int(g.code),
                    refusal=reg._refusal_dict(g.refusal) if g.refusal.kind else None,
                    budget=None if not g.budget.refused else dict(resource=RESOURCES[g.budget.resource],
                                                                 need=int(g.budget.need), limit=int(g.budget.limit)),
                    need={n: int(g.need.need[k]) for k, n in enumerate(RESOURCES) if g.need.need[k]},
                    amounts=dict(zip(reg.COST_COMPONENTS, (int(v) for v in g.amount))),
                    costs=dict(zip(reg.COST_COMPONENTS, (float(v) for v in g.cost))), total=float(g.total),
                    reuse=None if c.reuse < 0 else int(c.reuse), exact=bool(c.exact), policy=_POLICIES.get(int(c.policy)),
                    form=_FORMS.get(int(c.form)), bound_bits=int(c.bound_bits), evaluations=int(c.evaluations),
                    native_requests=int(c.native_requests), entries=int(c.entries), largest=int(c.largest),
                    state_bytes=int(c.state_bytes), device_bytes=int(c.device_bytes), work_items=int(c.work_items))

    @staticmethod
    def _name(c):
        s = _STRATEGY_NAMES[int(c.strategy)] + "@" + _DEVICES[int(c.device)]
        return s + (":" + _REPRS[int(c.repr)] if c.repr else "")

    @property
    def candidates(self):
        return [self.candidate(i) for i in range(int(self._info.ncandidates))]

    def kernel_view(self):
        kv = _KernelView()
        _check(_call("tr_pairplan_plan_kernel_view", self._open(), C.byref(kv)), what="kernel view")
        return kv

    def index(self, name):
        """the index of a candidate by name ("support@cpu", "features@cpu:sparse", "direct@cuda", ...)"""
        for c in self.candidates:
            if c["name"] == name:
                return c["index"]
        raise ValueError(f"the plan has no candidate {name!r}")

    def run(self, candidate=None, *, budget=None, raw=False):
        """execute a candidate (None: the chosen one; an index or a name): {"results": [answers] (raw: the
        RESULT_DTYPE array), "candidate", "name", "strategy", "device", "backend", "built", "evaluations",
        "native_requests", "state_bytes"}; a request the native code refuses raises unless raw"""
        if isinstance(candidate, str):
            candidate = self.index(candidate)
        index = self.chosen if candidate is None else int(candidate)
        if index is None:
            _check(int(self._info.code) or -1, self._info.refusal if self._info.refusal.refused else None,
                   "pairwise plan: " + (self.reason or "refused"))
        cand = self.candidate(index)
        budget = budget if budget is not None else self._budget
        n = len(self.requests)
        out = np.zeros(max(n, 1), dtype=RESULT_DTYPE)
        if cand["device"] == "cuda":
            info = self._run_cuda(index, cand, out, budget)
        else:
            ex, refusal = _Exec(), _Refusal()
            _check(_call("tr_pairplan_execute", self._open(), index, C.byref(_budget(budget)), out.ctypes.data,
                         C.byref(ex), C.byref(refusal)), refusal, f"candidate {cand['name']}")
            info = dict(built=bool(ex.built), evaluations=int(ex.evaluations), native_requests=int(ex.native_requests),
                        state_bytes=int(ex.state_bytes), backend="cpu", device_name="native CPU")
        out = out[:n]
        res = dict(candidate=index, name=cand["name"], strategy=cand["strategy"], device=cand["device"], **info)
        res["results"] = out if raw else [_answer(r, where=f"request {i}") for i, r in enumerate(out)]
        return res

    def _run_cuda(self, index, cand, out, budget):
        gpu = _gpu()
        strategy = "support" if cand["strategy"] == "support" else "rows"
        if cand["reuse"] is not None:
            st, built = self.retained[cand["reuse"]], False
        else:
            st, built = retain(self.scope, self.columns, strategy, kernel=self.kernel, weight=self.weight,
                               budget=budget), True
        try:
            dev = st._upload(budget)
            kv, refusal = self.kernel_view(), _Refusal()
            n = len(self.requests)
            gpu.run("tr_mgpu_pairplan_query", dev, C.byref(kv), _ptr(self.requests), n, out.ctypes.data,
                    C.byref(_budget(budget)), C.byref(refusal), refusal=refusal, what=f"candidate {cand['name']}")
            d = gpu.device()
            evals = int(sum(int(r["evaluations"]) for r in out[:n]))
            return dict(built=built, evaluations=evals, native_requests=n, state_bytes=st.info["bytes"],
                        backend="cuda-emulated" if d["emulated"] else "cuda", device_name=d["name"])
        finally:
            if built:
                st.close()

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_pairplan_plan_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Plan(closed)"
        i = self.info
        return f"Plan({i['kind']}, {i['requests']} requests, {i['candidates']} candidates, chosen {i['chosen']})"


def eval_table(state, plan_, *, backend="cpu", budget=None):
    """the raw answers of plan_'s requests over a retained support / rows state, on the CPU (tr_pairplan_eval_table)
    or the device (tr_mgpu_pairplan_query): the two evaluators the table strategies run"""
    reqs = plan_.requests
    n = len(reqs)
    out = np.zeros(max(n, 1), dtype=RESULT_DTYPE)
    kv, refusal = plan_.kernel_view(), _Refusal()
    if backend == "cpu":
        _check(_call("tr_pairplan_eval_table", C.byref(state.view()), C.byref(kv), _ptr(reqs), n, out.ctypes.data,
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "table evaluation")
    elif backend == "cuda":
        gpu = _gpu()
        gpu.run("tr_mgpu_pairplan_query", state._upload(budget), C.byref(kv), _ptr(reqs), n, out.ctypes.data,
                C.byref(_budget(budget)), C.byref(refusal), refusal=refusal, what="table evaluation on the device")
    else:
        raise ValueError("backend is 'cpu' or 'cuda'")
    return out[:n]
