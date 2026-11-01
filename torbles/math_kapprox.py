# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""KERNEL-002: approximate spectral feature kernels, natively on the CPU or with the CUDA kernels
(torbles/math_kapprox.c, math_kapprox_gpu.cu; every contract in math_kapprox.h).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_kapprox as ka, math_kernel as mk
    scope = mbk.Scope.from_arrays({"x": xs, "y": ys, "w": ws}, scales={"x": 1, "y": 1}, groups=g, ngroups=2)
    st = ka.features(scope, ["x", "y"], ka.gaussian(gamma=Fraction(1, 2)), method="taylor", degree=8, weight="w")
    st = ka.features(scope, ["x", "y"], ka.laplace(1), method="rff", features=256, seed=7, failure=Fraction(1, 10**6))
    a = st.mmd2(0, 1)              # Answer: a.value (lo, hi) of the approximant, a.error, a.terms, a.bound, a.label
    st.sum(0, 1); st.mean(0); st.witness(0, [(12, -3)]); st.query([dict(op="mmd2", a=0, b=1, unbiased=False)])
    st.record                      # norm, assumptions, budget, per-pair bound and its parts, provenance (seed, count)
    st.authorize_exact()           # NotAuthorizedError ("approximate"; "statistical" for random features)
    data = st.dumps(); ka.loads(data, scope)           # FND-011 record format: identical answers after a reload
    ka.features(scope, ["x", "y"], mk.polynomial(2, c=1))   # an exact kernel: KERNEL-001's exact feature state

Kernels: gaussian exp(-gamma |x - y|^2) (method "taylor": deterministic certified features on a declared domain, or
"rff"), laplace exp(-gamma |x - y|_1) and cauchy prod 1 / (1 + gamma^2 (x_j - y_j)^2) (method "rff": random Fourier
features through SPEC-001, a probabilistic bound with its failure probability). x is the columns' exact value
(logical / 10^scale); witness points are given as logical integers. Every answer is an exact rational enclosure of the
approximant with its error terms: never presented as the exact kernel value. Nothing is computed in Python. Opt-in:
`import torbles` does not load this module.
"""
import ctypes as C
import json
from fractions import Fraction
from numbers import Rational

import numpy as np

from . import algq
from . import math_bank as mbk
from . import math_carriers as _mc
from . import math_fourier as _mf
from . import math_kernel as _mk
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["Kernel", "ApproxState", "Answer", "gaussian", "laplace", "cauchy", "kernel", "features", "need", "loads",
           "generate", "KERNELS", "METHODS", "MAX_DIMS", "MAX_DEGREE", "MAX_FEATURES", "GENERATOR"]

KERNELS = {"gaussian": 1, "laplace": 2, "cauchy": 3}
_KERNEL_NAMES = {v: k for k, v in KERNELS.items()}
_ALIASES = {"rbf": "gaussian", "gaussian": "gaussian", "laplace": "laplace", "laplacian": "laplace",
            "cauchy": "cauchy"}
METHODS = {"taylor": 1, "rff": 2}
_METHOD_NAMES = {v: k for k, v in METHODS.items()}
OPS = {"sum": 1, "mean": 2, "mmd2": 3, "witness": 4}
SELF, UNORDERED, UNBIASED = 1, 2, 4
LABELS = {5: "certified", 6: "probabilistic"}
TERMS = {1: "remainder", 2: "coefficient", 3: "economization", 4: "quantization", 5: "rounding", 6: "truncation",
         7: "residual", 8: "probabilistic"}
ASSUMPTIONS = ((1, "domain"), (2, "derivative"), (4, "lipschitz"), (8, "range"), (16, "independent"),
               (32, "no_overflow"))
MAX_DIMS, MAX_DEGREE, MAX_FEATURES, KQ_LIMBS = 16, 12, 1 << 20, 6
GENERATOR = ("splitmix64 words w_i = splitmix64(key + i), key = splitmix64(seed ^ 0xD1B54A32D192ED03) (UNC-002's "
             "stream convention); U = (2N + 1) / 2^129 from 128-bit N; gaussian: Box-Muller sqrt(-2 ln U1) (cos, sin)"
             "(2 pi U2) times sqrt(2 gamma); laplace: gamma cot(pi U); cauchy: +-gamma (-ln U); certified fixed-point "
             "transforms rounded to dyadic rationals with fbits fraction bits")
_Q = {"per_pair": 1, "hoeffding": 2, "bias": 3, "radius2": 4, "U": 5, "failure": 6, "budget": 7, "gamma": 8,
      "eta": 9, "pbad": 10, "coef": 11, "eta_num": 12}
P = C.POINTER


class _Spec(C.Structure):
    _native_ = "tr_kapprox_spec"
    _fields_ = [(n, C.c_int32) for n in ("kernel", "method", "ndims", "weight", "degree", "has_radius", "has_budget",
                                         "reserved")] + [
        ("column", C.c_int32 * MAX_DIMS), ("gamma", _mc._Q), ("radius2", _mc._Q), ("failure", _mc._Q),
        ("budget", _mc._Q), ("nfreq", C.c_uint64), ("seed", C.c_uint64)]


class _PlanView(C.Structure):
    _native_ = "tr_kapprox_plan_view"
    _fields_ = [(n, C.c_int32) for n in ("method", "ndims", "weight", "degree", "scale", "shift", "xscale",
                                         "reserved")] + [
        ("ngroups", C.c_uint64), ("nitems", C.c_uint64), ("tmax", C.c_uint64), ("kq", C.c_uint64 * KQ_LIMBS),
        ("column", C.c_int32 * MAX_DIMS), ("exps", C.c_void_p), ("parent", C.c_void_p), ("axis", C.c_void_p),
        ("fourier", _mf._PlanView)]


class _Group(C.Structure):
    _native_ = "tr_kapprox_group"
    _fields_ = [("rows", C.c_uint64), ("valid", C.c_uint64), ("W", _mc._I128), ("A", _mc._I128),
                ("Q", C.c_uint64 * 3), ("reserved", C.c_uint64)]


class _Request(C.Structure):
    _native_ = "tr_kapprox_request"
    _fields_ = [("op", C.c_int32), ("flags", C.c_int32), ("a", C.c_uint64), ("b", C.c_uint64), ("point", C.c_uint64),
                ("reserved", C.c_int64)]


class _Result(C.Structure):
    _native_ = "tr_kapprox_result"
    _fields_ = [(n, C.c_int32) for n in ("status", "empty", "label", "nterms")] + [
        ("term_kind", C.c_int32 * 4), ("lo_d", C.c_double), ("hi_d", C.c_double), ("error_d", C.c_double)] + [
        (n, C.c_void_p) for n in ("lo", "hi", "error", "bound_lo", "bound_hi", "failure", "pairs")] + [
        ("term", C.c_void_p * 4)]


class _Info(C.Structure):
    _native_ = "tr_kapprox_info"
    _fields_ = [(n, C.c_int32) for n in ("kernel", "method", "ndims", "weight", "degree", "scale", "wscale", "xscale",
                                         "shift", "fbits", "label", "provenance", "norm", "has_radius", "has_budget",
                                         "scoped")] + [
        ("assumptions", C.c_uint32), ("generator", C.c_uint32)] + [
        (n, C.c_uint64) for n in ("ngroups", "nitems", "nfreq", "count", "seed", "selected", "rows", "bytes", "id")] + [
        ("maxsq", C.c_uint64 * 3), ("tmax", C.c_uint64)] + [
        (n, C.c_double) for n in ("per_pair_hi", "hoeffding_hi", "bias_hi", "radius2_hi")]


for _s, _n in ((_Spec, 176), (_PlanView, 312), (_Group, 80), (_Request, 40), (_Result, 144), (_Info, 208)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, u64, B, R, E = C.c_void_p, C.c_int32, C.c_uint64, P(_Budget), P(_Refusal), P(C.c_int32)
        sig = {
            "tr_kapprox_plan_create": (vp, [vp, P(_Spec), B, R, E]),
            "tr_kapprox_plan_get_view": (i32, [vp, P(_PlanView)]),
            "tr_kapprox_plan_free": (None, [vp]),
            "tr_kapprox_sums": (i32, [P(_PlanView), vp, vp, vp, vp, vp, vp, vp]),
            "tr_kapprox_from_sums": (vp, [vp, vp, vp, vp, vp, vp, vp, vp, B, R, E]),
            "tr_kapprox_build": (vp, [vp, P(_Spec), B, R, E]),
            "tr_kapprox_need": (i32, [vp, P(_Spec), i32, P(_Need)]),
            "tr_kapprox_generate": (i32, [i32, _mc._Q, i32, u64, u64, vp, P(i32), B, R]),
            "tr_kapprox_get_info": (i32, [vp, P(_Info)]),
            "tr_kapprox_get_spec": (i32, [vp, P(_Spec)]),
            "tr_kapprox_get_group": (i32, [vp, u64, P(_Group)]),
            "tr_kapprox_get_sums": (i32, [vp, u64, vp, u64]),
            "tr_kapprox_get_frequency": (i32, [vp, u64, vp, P(C.c_uint32)]),
            "tr_kapprox_get_q": (vp, [vp, i32, i32]),
            "tr_kapprox_query": (i32, [vp, vp, u64, vp, u64, vp, B, R]),
            "tr_kapprox_result_free": (None, [vp, u64]),
            "tr_kapprox_authorize_exact": (i32, [vp, u64, vp]),
            "tr_kapprox_fresh": (i32, [vp, vp, vp]),
            "tr_kapprox_save": (i32, [vp, vp, u64, vp, u64, P(u64), B, R]),
            "tr_kapprox_load": (vp, [vp, u64, vp, B, R, vp, vp, E]),
            "tr_kapprox_get_metadata": (i32, [vp, vp, u64, P(u64)]),
            "tr_kapprox_free": (None, [vp]),
            "tr_bigq_copy": (vp, [vp, B, R, E]),
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
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_kapprox_sums", C.c_int, [P(_PlanView), C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                                        C.c_void_p, C.c_void_p, C.c_void_p, P(_Budget), P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _budget(budget):
    return C.byref((budget or Budget()).raw)


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _take(handle):
    """the Fraction of an owned big rational (freed by the wrapper); None for NULL"""
    if not handle:
        return None
    return _mc.BigRational._wrap(handle).to_fraction()


def _borrowed(handle):
    """the Fraction of a borrowed big rational (copied first)"""
    if not handle:
        return None
    error = C.c_int32()
    h = _call("tr_bigq_copy", handle, None, None, C.byref(error))
    _check(error.value, what="big rational")
    return _take(h)


def _i128(x):
    return int.from_bytes(bytes(x), "little", signed=True)


def _u192(words):
    return int(words[0]) | int(words[1]) << 64 | int(words[2]) << 128


# ---------------------------------------------------------------------- kernels
class Kernel:
    """A declared kernel without an exact finite feature map: gaussian, laplace or cauchy with an exact gamma > 0."""

    def __init__(self, kind, gamma):
        if kind not in _ALIASES:
            raise ValueError(f"kernel is one of {', '.join(sorted(_ALIASES))}")
        self.kind, self.gamma = _ALIASES[kind], gamma

    def __repr__(self):
        return f"Kernel({self.kind}, gamma={self.gamma})"


def gaussian(gamma=None, *, lengthscale=None):
    """exp(-gamma |x - y|^2); lengthscale l gives gamma = 1 / (2 l^2) exactly (exact rationals only)"""
    if (gamma is None) == (lengthscale is None):
        raise ValueError("give gamma or lengthscale")
    if lengthscale is not None:
        if isinstance(lengthscale, float) or not isinstance(lengthscale, Rational):
            raise TypeError("lengthscale: an exact rational (int or Fraction)")
        gamma = Fraction(1, 2) / (Fraction(lengthscale) ** 2)
    return Kernel("gaussian", gamma)


def laplace(gamma):
    """exp(-gamma |x - y|_1)"""
    return Kernel("laplace", gamma)


def cauchy(gamma):
    """prod_j 1 / (1 + gamma^2 (x_j - y_j)^2)"""
    return Kernel("cauchy", gamma)


def kernel(name, **params):
    """a kernel by name: gaussian / rbf, laplace / laplacian, cauchy (declared approximations), or one of KERNEL-001's
    exact kernels (polynomial, linear, homogeneous, monomial, delta)"""
    if name in _ALIASES:
        if _ALIASES[name] == "gaussian":
            return gaussian(params.get("gamma"), lengthscale=params.get("lengthscale"))
        return Kernel(name, params["gamma"])
    return _mk.kernel(name, **params)


def _spec(scope, columns, kern, method, degree, features, seed, failure, radius2, error_budget, weight):
    if method not in METHODS:
        raise ValueError("method is 'taylor' (deterministic, certified) or 'rff' (random Fourier features)")
    names = list(scope.names)
    s = _Spec()
    s.kernel, s.method, s.ndims = KERNELS[kern.kind], METHODS[method], len(columns)
    if len(columns) <= MAX_DIMS:
        for j, c in enumerate(columns):
            if c not in names:
                raise ValueError(f"{c!r} is not a column of the scope ({names})")
            s.column[j] = names.index(c)
    if weight is None:
        s.weight = -1
    else:
        if weight not in names:
            raise ValueError(f"{weight!r} is not a column of the scope ({names})")
        s.weight = names.index(weight)
    s.gamma.numerator, s.gamma.denominator = _mf._pair(kern.gamma, "gamma")
    if radius2 is not None:
        s.has_radius = 1
        s.radius2.numerator, s.radius2.denominator = _mf._pair(radius2, "radius2")
    if error_budget is not None:
        s.has_budget = 1
        s.budget.numerator, s.budget.denominator = _mf._pair(error_budget, "error_budget")
    if method == "taylor":
        s.degree = int(degree)
    else:
        s.nfreq, s.seed = int(features), int(seed) & ((1 << 64) - 1)
        s.failure.numerator, s.failure.denominator = _mf._pair(failure, "failure")
    return s


def _is_exact(kern):
    return isinstance(kern, _mk.Kernel)


def features(scope, columns, kern, *, method=None, degree=8, features=256, seed=0, failure=Fraction(1, 100),
             radius2=None, error_budget=None, weight=None, backend="cpu", budget=None, **exact):
    """The feature state of a kernel over the scope's columns.

    A KERNEL-001 kernel (math_kernel: polynomial, linear, homogeneous, monomial, delta) is answered exactly by
    KERNEL-001 (its FeatureState; `exact` options: degree, repr). A declared approximate kernel (gaussian, laplace,
    cauchy) needs method "taylor" (gaussian: degree N <= 12; radius2: the declared domain |x|^2 <= radius2, default
    the data's largest |x|^2) or "rff" (features D, seed, failure delta), an optional error_budget on the per-pair
    bound (NotConvergedError when exceeded); backend "cuda" accumulates the features on the device."""
    columns = list(columns)
    if _is_exact(kern):
        if method is not None:
            raise ValueError("an exact KERNEL-001 kernel takes no approximation method")
        return _mk.features(scope, columns, kern, weight=weight, backend=backend, budget=budget, **exact)
    if exact:
        raise TypeError(f"unknown options {sorted(exact)}")
    if not isinstance(kern, Kernel):
        raise TypeError("kern: ka.gaussian / ka.laplace / ka.cauchy, or a KERNEL-001 kernel (math_kernel)")
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    if scope.source == "table":
        scope.require_fresh("approximate kernel features")
    s = _spec(scope, columns, kern, method, degree, features, seed, failure, radius2, error_budget, weight)
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = _call("tr_kapprox_build", scope._handle, C.byref(s), _budget(budget), C.byref(refusal), C.byref(error))
        return ApproxState(_made(h, error, refusal, "approximate kernel features"), scope, columns, weight, "cpu",
                           "native CPU")
    plan = _call("tr_kapprox_plan_create", scope._handle, C.byref(s), _budget(budget), C.byref(refusal),
                 C.byref(error))
    _made(plan, error, refusal, "approximate kernel features")
    try:
        pv = _PlanView()
        _check(_call("tr_kapprox_plan_get_view", plan, C.byref(pv)), what="plan view")
        G, n = int(pv.ngroups), int(pv.nitems)
        groups = (_Group * max(G, 1))()
        maxsq = (C.c_uint64 * 3)()
        sums = np.zeros((max(G * n * 2, 1), 2), dtype=np.int64)
        frows = np.zeros(max(G, 1), dtype=np.uint64)
        fvalid = np.zeros(max(G, 1), dtype=np.uint64)
        fwsum = np.zeros((max(G, 1), 2), dtype=np.int64)
        view = scope.view()
        gpu = _gpu()
        refusal = _Refusal()
        gpu.run("tr_mgpu_kapprox_sums", C.byref(pv), C.byref(view), groups, maxsq, sums.ctypes.data,
                frows.ctypes.data, fvalid.ctypes.data, fwsum.ctypes.data, _budget(budget), C.byref(refusal),
                refusal=refusal, what="approximate kernel features on the device")
        d = gpu.device()
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_kapprox_from_sums", plan, scope._handle, groups, maxsq, sums.ctypes.data, frows.ctypes.data,
                  fvalid.ctypes.data, fwsum.ctypes.data, _budget(budget), C.byref(refusal), C.byref(error))
        return ApproxState(_made(h, error, refusal, "approximate kernel features"), scope, columns, weight,
                           "cuda-emulated" if d["emulated"] else "cuda", d["name"])
    finally:
        _call("tr_kapprox_plan_free", plan)


def need(scope, columns, kern, *, method, degree=8, features=256, seed=0, failure=Fraction(1, 100), radius2=None,
         error_budget=None, weight=None, backend="cpu"):
    """the needs a build checks before allocating ({resource: need}); backend "cuda": its device copies"""
    s = _spec(scope, list(columns), kern, method, degree, features, seed, failure, radius2, error_budget, weight)
    out = _Need()
    _check(_call("tr_kapprox_need", scope._handle, C.byref(s), 1 if backend == "cuda" else 0, C.byref(out)),
           what="approximate kernel needs")
    return {r: int(out.need[i]) for i, r in enumerate(RESOURCES) if out.need[i]}


def generate(kern, ndims, count, seed, budget=None):
    """the generator's count draws of ndims coordinates (before deduplication), natively: (draws, fbits)"""
    if not isinstance(kern, Kernel):
        raise TypeError("a declared kernel (gaussian, laplace, cauchy)")
    q = _mc._Q(*_mf._pair(kern.gamma, "gamma"))
    out = np.zeros((max(int(count) * int(ndims), 1), 2), dtype=np.int64)
    fbits, refusal = C.c_int32(), _Refusal()
    _check(_call("tr_kapprox_generate", KERNELS[kern.kind], q, int(ndims), int(count), int(seed) & ((1 << 64) - 1),
                 out.ctypes.data, C.byref(fbits), _budget(budget), C.byref(refusal)), refusal, "frequency generator")
    d = int(ndims)
    draws = [tuple(Fraction(int(out[f * d + j, 0]), int(out[f * d + j, 1])) for j in range(d)) for f in range(int(count))]
    return draws, int(fbits.value)


def loads(data, scope=None, *, budget=None):
    """A state from its record (tr_kapprox_save's bytes). With scope: the recorded source must be fresh (FND-007
    exact comparison; StaleSourceError) and the plan rebuilt from the recorded spec (the frequencies redrawn from the
    seed) must agree; without: an unscoped state. Refusals as FND-011's (math_persist exceptions)."""
    from . import math_persist as _mp
    from . import math_certificates as _mcert
    data = bytes(data)
    buf = (C.c_uint8 * max(len(data), 1)).from_buffer_copy(data or b"\0")
    st, stale, error, refusal = _mp._Status(), _mcert._Staleness(), C.c_int32(), _Refusal()
    h = _call("tr_kapprox_load", buf, len(data), None if scope is None else scope._handle, _budget(budget),
              C.byref(refusal), C.byref(st), C.byref(stale), C.byref(error))
    if not h:
        if error.value == -13:
            reasons = _mcert._reasons(stale.reasons)
            err = _mp.StaleSourceError(f"approximate kernel state: the scope is not the recorded source "
                                       f"({', '.join(reasons)})")
            err.reasons = reasons
            raise err
        _mp._raise_status(error.value, st, refusal, "approximate kernel state")
        _check(error.value, refusal, "approximate kernel state")
        raise MemoryError("approximate kernel state: native allocation failed")
    length = C.c_uint64()
    _check(_call("tr_kapprox_get_metadata", h, None, 0, C.byref(length)), what="metadata")
    raw = (C.c_uint8 * max(int(length.value), 1))()
    _check(_call("tr_kapprox_get_metadata", h, raw, int(length.value), C.byref(length)), what="metadata")
    meta = json.loads(bytes(raw)[:int(length.value)].decode()) if length.value else {}
    if meta.get("kapprox") != 1:
        _call("tr_kapprox_free", h)
        raise ValueError("not a record written by math_kapprox (its metadata names the columns)")
    state = ApproxState(h, scope, meta["columns"], meta.get("weight"), "loaded", "native CPU")
    state.user_metadata = meta.get("user")
    return state


# ---------------------------------------------------------------------- answers
class Answer:
    """One approximate answer: value (lo, hi) exactly encloses the approximant's value; the exact kernel quantity lies
    in bound (lo - error, hi + error), certified (label "certified") or with probability >= 1 - failure
    ("probabilistic"); terms: [(kind, magnitude)], the error spent term by term. Never exact."""
    exact = False
    norm = "abs"

    def __init__(self, op, status, empty, label, value, error, terms, bound, failure, pairs):
        self.op, self.status, self.empty, self.label = op, status, empty, label
        self.value, self.error, self.terms, self.bound, self.failure, self.pairs = value, error, terms, bound, failure, pairs

    @property
    def certified(self):
        return self.label == "certified"

    def as_dict(self):
        return dict(op=self.op, status=self.status, empty=self.empty, label=self.label, value=self.value,
                    error=self.error, terms=self.terms, bound=self.bound, failure=self.failure, pairs=self.pairs,
                    exact=False, norm=self.norm)

    def __repr__(self):
        if self.empty:
            return f"Answer({self.op}: empty)"
        lo, hi = self.bound
        return f"Answer({self.op}: in [{float(lo):.6g}, {float(hi):.6g}], {self.label}, error {float(self.error):.3g})"


class ApproxState:
    """An immutable approximate kernel feature state (native): the certified feature sums per group, the weight
    statistics, the declared approximation and its error record."""

    def __init__(self, handle, scope, columns, weight, backend, device):
        self._h, self.scope, self.columns, self.weight = handle, scope, list(columns), weight
        self.backend, self.device = backend, device
        self.user_metadata = None

    def _open(self):
        if not self._h:
            raise ValueError("the approximate kernel state is closed")
        return self._h

    def _info(self):
        i = _Info()
        _check(_call("tr_kapprox_get_info", self._open(), C.byref(i)), what="approximate kernel state")
        return i

    def q(self, name, index=0):
        """an exact rational of the state (per_pair, hoeffding, bias, radius2, U, failure, budget, gamma, eta, pbad,
        eta_num, coef[index]) or None"""
        return _borrowed(_call("tr_kapprox_get_q", self._open(), _Q[name], int(index)))

    @property
    def info(self):
        i = self._info()
        return dict(kernel=_KERNEL_NAMES[i.kernel], method=_METHOD_NAMES[i.method], columns=list(self.columns),
                    weight=self.weight, degree=int(i.degree), groups=int(i.ngroups), items=int(i.nitems),
                    frequencies=int(i.nfreq), count=int(i.count), seed=int(i.seed), fbits=int(i.fbits),
                    scale=int(i.scale), wscale=int(i.wscale), xscale=int(i.xscale), shift=int(i.shift),
                    selected=int(i.selected), rows=int(i.rows), bytes=int(i.bytes), id=int(i.id),
                    maxsq=_u192(i.maxsq), tmax=int(i.tmax), scoped=bool(i.scoped), backend=self.backend,
                    device=self.device)

    @property
    def record(self):
        """the approximation record (UNC-002's vocabulary): norm, assumptions, budget, the per-pair bound and its
        parts, label, provenance (deterministic, or randomized with the generator, seed, count, failure)"""
        i = self._info()
        rff = i.method == METHODS["rff"]
        rec = dict(kernel=_KERNEL_NAMES[i.kernel], method=_METHOD_NAMES[i.method], gamma=self.q("gamma"),
                   norm="abs", label=LABELS[i.label], exact=False,
                   provenance="randomized" if rff else "deterministic",
                   assumptions=[n for b, n in ASSUMPTIONS if i.assumptions & b],
                   radius2=self.q("radius2"), radius_declared=bool(i.has_radius),
                   budget=self.q("budget"), per_pair=self.q("per_pair"))
        if rff:
            rec.update(generator=GENERATOR, seed=int(i.seed), count=int(i.count), distinct=int(i.nfreq),
                       fbits=int(i.fbits), failure=self.q("failure"), hoeffding=self.q("hoeffding"),
                       bias=self.q("bias"), eta=self.q("eta"), eta_num=self.q("eta_num"), pbad=self.q("pbad"),
                       per_answer="h R (probabilistic, failure delta) + bias F (deterministic)")
        else:
            rec.update(degree=int(i.degree), U=self.q("U"), per_answer="eps F (certified remainder)")
        return rec

    @property
    def frequencies(self):
        """RFF: [(frequency (d Fractions), multiplicity)] in the canonical (sorted, distinct) order"""
        i = self._info()
        if i.method != METHODS["rff"]:
            return []
        d, out = int(i.ndims), []
        raw = np.zeros((d, 2), dtype=np.int64)
        mult = C.c_uint32()
        for f in range(int(i.nfreq)):
            _check(_call("tr_kapprox_get_frequency", self._h, f, raw.ctypes.data, C.byref(mult)), what="frequency")
            out.append((tuple(Fraction(int(raw[j, 0]), int(raw[j, 1])) for j in range(d)), int(mult.value)))
        return out

    def group(self, g):
        """{"rows", "valid", "weight" (sum w), "abs_weight" (sum |w|), "squares" (sum w^2)}: exact, value units"""
        x, i = _Group(), self._info()
        _check(_call("tr_kapprox_get_group", self._open(), int(g), C.byref(x)), what="group (id out of range)")
        u = Fraction(10) ** int(i.wscale)
        return dict(rows=int(x.rows), valid=int(x.valid), weight=Fraction(_i128(x.W), u),
                    abs_weight=Fraction(_i128(x.A), u), squares=Fraction(_u192(x.Q), u * u))

    def sums(self, g):
        """the raw feature sums of group g: [(floor, ceil)] integers over 2^scale (logical weight units)"""
        n = int(self._info().nitems)
        raw = np.zeros((max(2 * n, 1), 2), dtype=np.int64)
        _check(_call("tr_kapprox_get_sums", self._open(), int(g), raw.ctypes.data, 2 * n), what="sums")
        vals = [(int(raw[k, 0]) & ((1 << 64) - 1)) | (int(raw[k, 1]) << 64) for k in range(2 * n)]
        return [(vals[2 * t], vals[2 * t + 1]) for t in range(n)]

    def _points(self, points):
        if points is None:
            return None, 0
        d = len(self.columns)
        pts = [tuple(p) if isinstance(p, (list, tuple)) else (p,) for p in points]
        arr = np.zeros(max(len(pts) * d, 1), dtype=np.int64)
        for i, p in enumerate(pts):
            if len(p) != d:
                raise ValueError(f"a point has one logical value per column ({d})")
            for j, x in enumerate(p):
                arr[i * d + j] = _mk._i64(x, "a point coordinate")
        return arr, len(pts)

    def query(self, requests, *, points=None, budget=None, raw=False):
        """answers of request dicts (op: sum, mean, mmd2, witness; a, b: groups; self_pairs, ordered, unbiased; point:
        an index into points, logical values): [Answer]; raw=False raises on a request's error"""
        requests = list(requests)
        arr = (_Request * max(len(requests), 1))()
        for i, q in enumerate(requests):
            if q.get("op") not in OPS:
                raise ValueError(f"op is one of {', '.join(OPS)}")
            b = q.get("b")
            r = arr[i]
            r.op, r.a, r.b, r.point = OPS[q["op"]], int(q["a"]), int(q["a"] if b is None else b), int(q.get("point", 0))
            r.flags = ((SELF if q.get("self_pairs") else 0) | (UNORDERED if q.get("ordered", True) is False else 0) |
                       (UNBIASED if q.get("unbiased") else 0))
        pts, npts = self._points(points)
        out = (_Result * max(len(requests), 1))()
        refusal = _Refusal()
        _check(_call("tr_kapprox_query", self._open(), arr, len(requests), None if pts is None else pts.ctypes.data,
                     npts, out, _budget(budget), C.byref(refusal)), refusal, "approximate kernel requests")
        answers, first = [], None
        for i in range(len(requests)):            # take every handle first: nothing leaks when a request failed
            r = out[i]
            vals = {k: _take(getattr(r, k)) for k in ("lo", "hi", "error", "bound_lo", "bound_hi", "failure", "pairs")}
            terms = [_take(r.term[k]) for k in range(4)]
            if r.status:
                first = first if first is not None else (i, int(r.status))
                answers.append(Answer(requests[i]["op"], int(r.status), False, None, None, None, [], None, None, None))
                continue
            answers.append(Answer(requests[i]["op"], 0, bool(r.empty), LABELS.get(r.label),
                                  None if r.empty else (vals["lo"], vals["hi"]), vals["error"],
                                  [(TERMS[int(r.term_kind[k])], terms[k]) for k in range(int(r.nterms))],
                                  None if r.empty else (vals["bound_lo"], vals["bound_hi"]), vals["failure"],
                                  vals["pairs"]))
        if first is not None and not raw:
            _check(first[1], refusal if first[1] == -5 else None, f"request {first[0]}")
        return answers

    def sum(self, a, b=None, *, self_pairs=False, ordered=True, budget=None):
        """sum over the selected pairs of w_i v_l k(x_i, y_l) (KERNEL-001's pair semantics): an Answer (.pairs)"""
        return self.query([dict(op="sum", a=a, b=b, self_pairs=self_pairs, ordered=ordered)], budget=budget)[0]

    def mean(self, a, b=None, *, self_pairs=False, ordered=True, budget=None):
        """the kernel mean embedding inner product <mu_a, mu_b> (empty without pair weight)"""
        return self.query([dict(op="mean", a=a, b=b, self_pairs=self_pairs, ordered=ordered)], budget=budget)[0]

    def mmd2(self, a, b, *, unbiased=True, budget=None):
        return self.query([dict(op="mmd2", a=a, b=b, unbiased=unbiased)], budget=budget)[0]

    def witness(self, a, points, *, budget=None):
        """mu_a(z) at each point (logical values inside the domain)"""
        pts = list(points)
        return self.query([dict(op="witness", a=a, point=i) for i in range(len(pts))], points=pts, budget=budget)

    # ---------------------------------------------------------------- FND-007 / FND-011
    def decide(self, field=0):
        """the native refusal of an exact rewrite as a dict (never authorized)"""
        from . import math_certificates as _mcert
        d = _mcert._Decision()
        rc = _call("tr_kapprox_authorize_exact", self._open(), int(field), C.byref(d))
        return dict(authorized=bool(d.authorized), reason=_mcert.REFUSALS[d.reason], label=LABELS.get(d.kind),
                    field=int(d.field), code=int(rc))

    def authorize_exact(self, field=0):
        """an approximate state never authorizes an exact rewrite: NotAuthorizedError ("approximate", "statistical")"""
        from . import math_certificates as _mcert
        d = self.decide(field)
        err = _mcert.NotAuthorizedError(f"an approximate kernel state ({d['label']}) cannot authorize an exact "
                                        f"rewrite: {d['reason']}")
        err.reason, err.decision = d["reason"], d
        raise err

    def fresh(self, scope):
        """the recorded source against a current scope: {"fresh", "reasons", "first"}"""
        from . import math_certificates as _mcert
        st = _mcert._Staleness()
        rc = _call("tr_kapprox_fresh", self._open(), scope._handle, C.byref(st))
        if rc not in (0, -13):
            _check(rc, what="freshness")
        first = _mcert._reasons(st.first)
        return dict(fresh=rc == 0, reasons=list(_mcert._reasons(st.reasons)), first=first[0] if first else None)

    def dumps(self, metadata=None, *, budget=None):
        """the record (FND-011's format, type tag 0x4b02, exact 0); metadata: a JSON-serializable value kept with it"""
        meta = json.dumps(dict(kapprox=1, columns=self.columns, weight=self.weight, user=metadata),
                          sort_keys=True).encode()
        mbuf = (C.c_uint8 * max(len(meta), 1)).from_buffer_copy(meta)
        size, refusal = C.c_uint64(), _Refusal()
        _check(_call("tr_kapprox_save", self._open(), mbuf, len(meta), None, 0, C.byref(size), _budget(budget),
                     C.byref(refusal)), refusal, "approximate kernel record")
        out = (C.c_uint8 * max(int(size.value), 1))()
        _check(_call("tr_kapprox_save", self._h, mbuf, len(meta), out, int(size.value), C.byref(size),
                     _budget(budget), C.byref(refusal)), refusal, "approximate kernel record")
        return bytes(out)[:int(size.value)]

    def close(self):
        if getattr(self, "_h", None):
            _call("tr_kapprox_free", self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._h:
            return "ApproxState(closed)"
        i = self.info
        return (f"ApproxState({i['kernel']}, {i['method']}, columns={i['columns']}, groups={i['groups']}, "
                f"{self.backend})")
