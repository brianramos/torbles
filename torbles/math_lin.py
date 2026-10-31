# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LIN-001 / LIN-002: grouped exact moment matrices and regression systems, and the exact and numerical operators
that finalize them, natively on the CPU or with the CUDA kernels (torbles/math_lin.c, math_lin_gpu.cu).

    from torbles import math_bank as mbk, math_lin as lin
    scope = mbk.Scope.from_table(db["f"], ["x1", "x2", "y"], group_by=["g"])
    st = lin.moments(scope, ["x1", "x2"], "y", policy="listwise", backend="cpu")    # or backend="cuda"
    st.matrix(g)          # the augmented Gram matrix [1, x1, x2, y] of group g (exact ints, logical units)
    st.fit(g, ridge=Fraction(1, 10), features=["x1"])     # exact coefficients (a solution set), RSS, TSS, R^2
    st.covariance(g)      # exact covariance, squared correlation and its sign
    st.pca(g)             # numerical eigensystem with rigorous residual and eigenvalue bounds (approximate)
    lin.solve(A, b)       # exact solution set: unique / family / none (with a certificate)
    lin.inertia(A)        # exact inertia and class of a symmetric matrix, with certificates

Exact values are Python int / Fraction, marshalled as limbs; every result is computed natively (Python formats it).
Policies: "listwise" (common-listwise: a row counts when every feature and the target are present; a true Gram
matrix) or "pairwise" (each entry over its own complete rows, with per-entry counts; possibly indefinite).
Refusals: AlgebraBudgetError (budgets, before allocation), UnsupportedError (more than 16 features, systems past
64 x 63), OverflowError (width 128 without a proof), DomainError (an operator the policy does not define),
NotConvergedError (Jacobi sweeps), StaleStateError (a table scope whose table changed). Opt-in: `import torbles`
does not load this module.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _GFP
from .math_errors import RESOURCES, NotConvergedError, UnsupportedError, check as _check

__all__ = ["Moments", "Solution", "Inertia", "Fit", "Covariance", "Eigen", "Batch", "moments", "moments_from_bank",
           "solve", "solve_batch", "inertia", "eigen", "primes", "need", "POLICIES", "MAX_FEATURES", "CONTRACT"]

POLICIES = {"listwise": 0, "pairwise": 1}
MAX_FEATURES, MAX_COLS, MAX_SLOTS, MAX_ROWS, MAX_UNKNOWNS, WORDS = 16, 18, 715, 64, 63, 3
UNDEFINED = 2
_NEED = {"cpu": 0, "device": 1, "adopt": 2}
_CLASSES = {1: "positive definite", 2: "positive semidefinite", 3: "negative definite", 4: "negative semidefinite",
            5: "indefinite", 6: "zero"}
_KINDS = {1: "unique", 2: "family", 3: "none"}
_METHODS = {1: "bareiss", 2: "modular", 3: "jacobi"}
_SOURCES = {1: "fused", 2: "bank", 3: "adopted", 4: "combined"}
PART = dict(x=0, null=1, cert=2, neg=3, pos=4, kernel=5, A=6, b=7, stats=8, cov=9, r2=10, sign=11, count=12,
            design=13, values=14, vectors=15, residual=16, bound=17, ortho=18, width=19)
CONTRACT = ("approximate: binary64 cyclic Jacobi on the midpoint of an enclosure of the matrix; certified bounds valid "
            "for every matrix in the enclosure: residual[i] >= ||A v_i - lambda_i v_i||_2, an eigenvalue of A lies "
            "within bound[i] of lambda_i (bound = residual / ||v_i||_2), orthogonality >= max |V^T V - I|; no claim "
            "on eigenvector accuracy (close eigenvalues make eigenvectors ill-conditioned, whatever the exactness of "
            "the input moments)")
_U64 = (1 << 64) - 1
P = C.POINTER


class _Spec(C.Structure):
    _native_ = "tr_lin_spec"
    _fields_ = [("nfeatures", C.c_int32), ("target", C.c_int32), ("feature", C.c_int32 * MAX_FEATURES),
                ("policy", C.c_int32), ("intercept", C.c_int32), ("width", C.c_int32), ("reserved", C.c_int32)]


class _Slot(C.Structure):
    _native_ = "tr_lin_slot"
    _fields_ = [("a", C.c_uint8), ("b", C.c_uint8), ("kind", C.c_uint8), ("reserved", C.c_uint8), ("need", C.c_uint32)]


class _States(C.Structure):
    _native_ = "tr_lin_states"
    _fields_ = [("ngroups", C.c_uint64), ("nslots", C.c_int32), ("words", C.c_int32), ("rows", C.c_void_p),
                ("count", C.c_void_p), ("sum", C.c_void_p)]


class _Info(C.Structure):
    _native_ = "tr_lin_info"
    _fields_ = [(n, C.c_uint64) for n in ("ngroups", "present", "rows", "bytes")] + [("digest", C.c_uint64 * 2)] + [
        (n, C.c_int32) for n in ("k", "nslots", "ngram", "nfeatures", "target", "policy", "intercept", "width",
                                 "marginals", "source", "scoped", "reserved")] + [
        ("column", C.c_int32 * MAX_COLS), ("scale", C.c_int32 * MAX_COLS)]


class _QBuf(C.Structure):
    _native_ = "tr_lin_qbuf"
    _fields_ = [("count", C.c_uint64), ("nlimbs", C.c_uint64), ("sign", C.c_void_p), ("nlen", C.c_void_p),
                ("dlen", C.c_void_p), ("limbs", C.c_void_p)]


class _FitSpec(C.Structure):
    _native_ = "tr_lin_fit_spec"
    _fields_ = [("group", C.c_uint64), ("intercept", C.c_int32), ("masked", C.c_int32), ("features", C.c_uint32),
                ("reserved", C.c_uint32), ("ridge_num", C.c_int64), ("ridge_den", C.c_int64)]


class _ResultInfo(C.Structure):
    _native_ = "tr_lin_result_info"
    _fields_ = [(n, C.c_int32) for n in ("what", "status", "rows", "cols", "rank", "nullity", "positive", "negative",
                                         "zero", "method")] + [
        (n, C.c_uint64) for n in ("pivots", "primes", "iterations", "used")] + [
        ("certified", C.c_int32), ("converged", C.c_int32)]


class _BatchView(C.Structure):
    _native_ = "tr_lin_batch_view"
    _fields_ = [("nsys", C.c_uint64), ("words", C.c_uint64), ("entries", C.c_uint64), ("data", C.c_void_p),
                ("offset", C.c_void_p), ("rows", C.c_void_p), ("cols", C.c_void_p)]


for _s, _n in ((_Spec, 88), (_Slot, 8), (_States, 40), (_Info, 240), (_QBuf, 48), (_FitSpec, 40), (_ResultInfo, 80),
               (_BatchView, 56)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e, B, R = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32), P(_Budget), P(_Refusal)
        sig = {
            "tr_lin_moments_build": (vp, [vp, P(_Spec), B, R, e]),
            "tr_lin_moments_need": (i32, [vp, P(_Spec), i32, P(_Need)]),
            "tr_lin_moments_from_bank": (vp, [vp, vp, P(_Spec), B, R, e]),
            "tr_lin_moments_from_states": (vp, [vp, P(_Spec), i32, P(_States), B, R, e]),
            "tr_lin_moments_combine": (vp, [vp, vp, i32, B, R, e]),
            "tr_lin_moments_get_info": (i32, [vp, P(_Info)]),
            "tr_lin_moments_get_states": (i32, [vp, P(_States)]),
            "tr_lin_moments_get_slots": (i32, [vp, vp, i32]),
            "tr_lin_moments_equal": (i32, [vp, vp]),
            "tr_lin_moments_free": (None, [vp]),
            "tr_lin_solve": (vp, [i32, i32, P(_QBuf), P(_QBuf), B, R, e]),
            "tr_lin_inertia": (vp, [i32, P(_QBuf), B, R, e]),
            "tr_lin_system": (vp, [vp, P(_FitSpec), B, R, e]),
            "tr_lin_fit": (vp, [vp, P(_FitSpec), B, R, e]),
            "tr_lin_covariance": (vp, [vp, u64, i32, B, R, e]),
            "tr_lin_eigen": (vp, [i32, vp, vp, u64, B, R, e]),
            "tr_lin_eigen_exact": (vp, [i32, P(_QBuf), u64, B, R, e]),
            "tr_lin_pca": (vp, [vp, u64, i32, i32, u64, B, R, e]),
            "tr_lin_result_get_info": (i32, [vp, P(_ResultInfo)]),
            "tr_lin_result_export": (i32, [vp, i32, P(_QBuf)]),
            "tr_lin_result_doubles": (i32, [vp, i32, vp, u64, P(u64)]),
            "tr_lin_result_count": (u64, [vp, i32]),
            "tr_lin_result_limbs": (u64, [vp, i32]),
            "tr_lin_result_free": (None, [vp]),
            "tr_lin_batch_systems": (vp, [i32, vp, vp, P(_QBuf), P(_QBuf), B, R, e]),
            "tr_lin_batch_fits": (vp, [vp, i32, P(_FitSpec), B, R, e]),
            "tr_lin_batch_get_view": (i32, [vp, P(_BatchView)]),
            "tr_lin_batch_primes_needed": (u64, [vp]),
            "tr_lin_batch_bytes": (u64, [vp, u64]),
            "tr_lin_primes": (i32, [u64, P(_GFP)]),
            "tr_lin_batch_residues": (i32, [vp, P(_GFP), u64, vp, B, R]),
            "tr_lin_batch_finish": (i32, [vp, P(_GFP), u64, vp, vp, P(u64), B, R]),
            "tr_lin_batch_free": (None, [vp]),
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
    from .math_bank import _View
    math_gpu.bind("tr_mgpu_lin_moments", C.c_int, [P(_View), P(_Spec), C.c_int32, P(_Budget), P(_States), P(_Refusal)])
    math_gpu.bind("tr_mgpu_lin_eliminate", C.c_int, [P(_BatchView), P(_GFP), C.c_uint64, P(_Budget), C.c_void_p,
                                                     P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _backend_name():
    from . import math_gpu
    d = math_gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


# ---------------------------------------------------------------------- packed rationals (marshalling only)
def _exact(v, what="value"):
    if isinstance(v, bool) or not isinstance(v, (Integral, Rational)):
        raise TypeError(f"{what}: exact values are int or Fraction (got {type(v).__name__}); floats are not exact input")
    return Fraction(v)


def _pack(values, what="values"):
    """a tr_lin_qbuf of exact values (int / Fraction) and the arrays that keep it alive"""
    values = [_exact(v, what) for v in values]
    n = len(values)
    sign = np.zeros(max(n, 1), np.int8)
    nlen = np.zeros(max(n, 1), np.uint32)
    dlen = np.zeros(max(n, 1), np.uint32)
    chunks = []
    for i, f in enumerate(values):
        num, den = f.numerator, f.denominator
        sign[i] = (num > 0) - (num < 0)
        a = -num if num < 0 else num
        nb = (a.bit_length() + 63) // 64
        if nb:
            chunks.append(a.to_bytes(8 * nb, "little"))
        nlen[i] = nb
        if den != 1:
            db = (den.bit_length() + 63) // 64
            chunks.append(den.to_bytes(8 * db, "little"))
            dlen[i] = db
    raw = b"".join(chunks)
    limbs = np.frombuffer(raw, dtype=np.uint64).copy() if raw else np.zeros(1, np.uint64)
    q = _QBuf(n, len(raw) // 8, sign.ctypes.data, nlen.ctypes.data, dlen.ctypes.data, limbs.ctypes.data)
    return q, (sign, nlen, dlen, limbs)


def _decode(sign, nlen, dlen, limbs, n):
    raw = limbs.tobytes()
    out, off = [], 0
    for i in range(n):
        s = int(sign[i])
        if s == UNDEFINED:
            out.append(None)
            continue
        a, b = int(nlen[i]), int(dlen[i])
        num = int.from_bytes(raw[8 * off:8 * (off + a)], "little")
        off += a
        den = int.from_bytes(raw[8 * off:8 * (off + b)], "little") if b else 1
        off += b
        out.append(Fraction(-num if s < 0 else num, den))
    return out


class _Result:
    """a native tr_lin_result (owned); parts exported on demand"""

    def __init__(self, handle):
        self._h = handle
        self.info = _ResultInfo()
        _check(_call("tr_lin_result_get_info", handle, C.byref(self.info)), what="result")

    def part(self, name):
        p = PART[name]
        n = int(_call("tr_lin_result_count", self._h, p))
        if not n:
            return None
        nl = int(_call("tr_lin_result_limbs", self._h, p))
        sign, nlen, dlen = np.zeros(n, np.int8), np.zeros(n, np.uint32), np.zeros(n, np.uint32)
        limbs = np.zeros(max(nl, 1), np.uint64)
        q = _QBuf(n, nl, sign.ctypes.data, nlen.ctypes.data, dlen.ctypes.data, limbs.ctypes.data)
        _check(_call("tr_lin_result_export", self._h, p, C.byref(q)), what=f"result part {name}")
        return _decode(sign, nlen, dlen, limbs, n)

    def doubles(self, name):
        p = PART[name]
        n = int(_call("tr_lin_result_count", self._h, p))
        out, cnt = np.zeros(max(n, 1), np.float64), C.c_uint64()
        _check(_call("tr_lin_result_doubles", self._h, p, out.ctypes.data, n, C.byref(cnt)), what=f"result part {name}")
        return out[:n].copy()

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _native is not None:
            _native["tr_lin_result_free"](h)
            self._h = None


def _rows(flat, n):
    return [flat[i * n:(i + 1) * n] for i in range(len(flat) // n)] if flat else []


def _ints(values):
    return None if values is None else [None if v is None else int(v) for v in values]


# ---------------------------------------------------------------------- results
class Solution:
    """The exact solution set of A x = b: kind "unique" (x), "family" (x + span(nullspace)) or "none"
    (certificate y with y^T A = 0, y^T b = 1); rank and pivot columns exact; verified natively (certified)."""

    def __init__(self, r, backend="cpu", device="native CPU"):
        i = r.info
        self.kind, self.rank, self.nullity = _KINDS[i.status], int(i.rank), int(i.nullity)
        self.rows, self.unknowns = int(i.rows), int(i.cols)
        self.pivots = [j for j in range(self.unknowns) if (i.pivots >> j) & 1]
        self.method, self.primes, self.certified = _METHODS[i.method], int(i.primes), bool(i.certified)
        self.backend, self.device = backend, device
        self.x = r.part("x")
        self.nullspace = _rows(r.part("null"), self.unknowns)
        self.certificate = r.part("cert")

    @property
    def unique(self):
        return self.kind == "unique"

    def same(self, other):
        """the same canonical solution set (kind, rank, pivots, particular solution, basis, certificate)"""
        return (self.kind, self.rank, self.pivots, self.x, self.nullspace, self.certificate) == (
            other.kind, other.rank, other.pivots, other.x, other.nullspace, other.certificate)

    def __repr__(self):
        return f"Solution({self.kind}, rank={self.rank}, x={self.x}, nullity={self.nullity}, {self.method})"


class Inertia:
    """Exact inertia (positive, negative, zero) and class of a symmetric matrix, with verified certificates:
    negative_direction (v^T A v < 0), positive_direction (v^T A v > 0), kernel (an exact basis, n0 vectors)."""

    def __init__(self, r):
        i = r.info
        self.n, self.positive, self.negative, self.zero = int(i.rows), int(i.positive), int(i.negative), int(i.zero)
        self.cls, self.rank, self.certified = _CLASSES[i.status], int(i.rank), bool(i.certified)
        self.negative_direction, self.positive_direction = r.part("neg"), r.part("pos")
        self.kernel = _rows(r.part("kernel"), self.n)

    def __repr__(self):
        return f"Inertia({self.cls}: +{self.positive} -{self.negative} 0x{self.zero})"


class Fit(Solution):
    """An exact least-squares fit from a moment state: the solution set of the normal equations (coefficients in
    value units, in the order of `design`), A and b, and (common-listwise) rss, tss, r2, sigma2, yty, n and
    projection = ||X b||^2 (None where undefined: no rows, a constant target, no residual degrees of freedom, a ridge
    penalty for sigma2)."""

    def __init__(self, r, names, backend="cpu", device="native CPU"):
        super().__init__(r, backend, device)
        self.design = [names[int(d)] for d in r.part("design")]
        self.A = _rows(r.part("A"), len(self.design))
        self.b = r.part("b")
        st = r.part("stats") or [None] * 7
        self.rss, self.tss, self.r2, self.sigma2, self.yty = st[:5]
        self.n = None if st[5] is None else int(st[5])
        self.projection = st[6]                       # ||X b||^2: the squared norm of the projection of y

    @property
    def coefficients(self):
        return dict(zip(self.design, self.x)) if self.x is not None else None

    def same(self, other):
        return super().same(other) and (self.design, self.A, self.b, self.rss, self.tss, self.r2, self.sigma2,
                                        self.yty, self.n, self.projection) == (
            other.design, other.A, other.b, other.rss, other.tss, other.r2, other.sigma2, other.yty, other.n,
            other.projection)


class Covariance:
    """Exact covariance (value units, ddof 0 or 1), squared correlation and its sign, and the rows of every entry,
    over the non-constant columns (features, then the target); None where undefined (n <= ddof, a zero variance)."""

    def __init__(self, r, names):
        p = int(r.info.rows)
        self.columns, self.policy = list(names), ("listwise", "pairwise")[int(r.info.status)]
        self.cov, self.r2 = _rows(r.part("cov"), p), _rows(r.part("r2"), p)
        self.sign = [_ints(row) for row in _rows(r.part("sign"), p)]
        self.count = [_ints(row) for row in _rows(r.part("count"), p)]


class Eigen:
    """A numerical symmetric eigensystem (APPROXIMATE, see CONTRACT): values descending, vectors[i] (unit, largest
    component positive), and the certified bounds residual, bound, orthogonality, plus the input width."""
    approximate, certified = True, True
    contract = CONTRACT

    def __init__(self, r):
        n = int(r.info.rows)
        self.n, self.sweeps, self.converged = n, int(r.info.iterations), bool(r.info.converged)
        self.values = r.doubles("values")
        self.vectors = r.doubles("vectors").reshape(n, n)
        self.residual, self.bound = r.doubles("residual"), r.doubles("bound")
        self.orthogonality, self.width = float(r.doubles("ortho")[0]), float(r.doubles("width")[0])
        self.matrix = _rows(r.part("cov"), n) if int(_call("tr_lin_result_count", r._h, PART["cov"])) else None


# ---------------------------------------------------------------------- moment states
def _spec(scope, features, target, policy, intercept, width):
    features = list(features)
    if len(features) > MAX_FEATURES:
        raise UnsupportedError(f"{len(features)} features: at most {MAX_FEATURES}")
    if policy not in POLICIES:
        raise ValueError(f"policy is one of {', '.join(POLICIES)}")
    if width not in ("auto", 128, 192):
        raise ValueError("width is 'auto', 128 or 192")
    names = list(scope.names)
    s = _Spec()
    s.nfeatures = len(features)
    for i, f in enumerate(features):
        if f not in names:
            raise ValueError(f"{f!r} is not a column of the scope")
        s.feature[i] = names.index(f)
    if target is not None and target not in names:
        raise ValueError(f"{target!r} is not a column of the scope")
    s.target = -1 if target is None else names.index(target)
    s.policy, s.intercept, s.width = POLICIES[policy], int(bool(intercept)), 0 if width == "auto" else int(width)
    return s


def _need_raw(scope, spec, target):
    out = _Need()
    _check(_call("tr_lin_moments_need", scope._handle, C.byref(spec), _NEED[target], C.byref(out)), what="moment needs")
    return out


def need(scope, features, target=None, *, policy="listwise", intercept=True, width="auto", backend="cpu"):
    """the needs a build checks before allocating ({resource: need}; backend "cuda": device copies and the adopted
    state, resource by resource the larger)"""
    spec = _spec(scope, features, target, policy, intercept, width)
    raw = _need_raw(scope, spec, "cpu") if backend == "cpu" else _both_needs(scope, spec)
    return {name: int(raw.need[i]) for i, name in enumerate(RESOURCES) if raw.need[i]}


def _both_needs(scope, spec):
    dev, adopt, both = _need_raw(scope, spec, "device"), _need_raw(scope, spec, "adopt"), _Need()
    for i in range(len(RESOURCES)):
        both.need[i] = max(dev.need[i], adopt.need[i])
    return both


def moments(scope, features, target=None, *, policy="listwise", intercept=True, width="auto", backend="cpu",
            budget=None):
    """The grouped exact moment state of a scope (math_bank.Scope): per group the count, sums, X^T X, X^T y, y^T y
    as the Gram matrix of [1, features..., target]; backend "cuda" runs the fused Gram kernels (Moments.backend says
    "cuda" on a device, "cuda-emulated" in the emulation)."""
    if scope.source == "table":
        scope.require_fresh("moments")
    spec = _spec(scope, features, target, policy, intercept, width)
    error, refusal = C.c_int32(), _Refusal()
    if backend == "cpu":
        h = _call("tr_lin_moments_build", scope._handle, C.byref(spec), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Moments._wrap(_made(h, error, refusal, "moments"), scope, features, target, "cpu", "native CPU")
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    gpu = _gpu()
    raw = _both_needs(scope, spec)
    _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw), C.byref(refusal)), refusal,
           "moments on the device")
    k = 1 + len(list(features)) + (target is not None)
    S = k * (k + 1) // 2 + (2 * (k - 1) * (k - 2) if policy == "pairwise" and k > 2 else 0)
    G = scope.groups
    rows = np.zeros(G, np.uint64)
    count = np.zeros(max(G * S, 1), np.uint64)
    sums = np.zeros(max(G * S * WORDS, 1), np.uint64)
    st = _States(G, S, WORDS, rows.ctypes.data, count.ctypes.data, sums.ctypes.data)
    view = scope.view()
    marg = int(policy == "pairwise")
    gpu.run("tr_mgpu_lin_moments", C.byref(view), C.byref(spec), marg, C.byref(_budget(budget)), C.byref(st),
            C.byref(refusal), refusal=refusal, what="moments on the device")
    h = _call("tr_lin_moments_from_states", scope._handle, C.byref(spec), marg, C.byref(st), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Moments._wrap(_made(h, error, refusal, "moment states"), scope, features, target, *_backend_name())


def moments_from_bank(bank, features, target=None, *, intercept=True, budget=None):
    """The moment state assembled from a FND-004 bank of a scope (math_bank.build): its basis must hold every
    monomial of degree <= 2 of the columns; mode "joint" over exactly these columns gives "listwise", mode "sql"
    gives "pairwise" (without pair-complete marginals)."""
    scope = bank.scope
    if scope is None:
        raise ValueError("an unscoped bank has no columns to name")
    if scope.source == "table":
        scope.require_fresh("moments from a bank")
    policy = "listwise" if bank.mode == "joint" else "pairwise"
    spec = _spec(scope, features, target, policy, intercept, "auto")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_lin_moments_from_bank", scope._handle, bank._handle, C.byref(spec), C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Moments._wrap(_made(h, error, refusal, "moments from a bank"), scope, features, target, "bank", "native CPU")


def _fitspec(names, group, intercept, ridge, features, d):
    fs = _FitSpec()
    fs.group = int(group)
    fs.intercept = -1 if intercept is None else int(bool(intercept))
    if features is None:
        fs.masked, fs.features = 0, 0
    else:
        mask = 0
        for f in features:
            if f not in names[1:1 + d]:
                raise ValueError(f"{f!r} is not a feature of the state")
            mask |= 1 << (names.index(f) - 1)
        fs.masked, fs.features = 1, mask
    lam = _exact(ridge, "ridge")
    if lam < 0:
        raise ValueError("the ridge penalty is >= 0")
    if not (-(1 << 63) <= lam.numerator < (1 << 63) and 0 < lam.denominator < (1 << 63)):
        raise OverflowError("the ridge penalty's numerator and denominator must fit int64")
    fs.ridge_num, fs.ridge_den = lam.numerator, lam.denominator
    return fs


class Moments:
    """An immutable grouped exact moment state (native). Columns: "1", the features, the target."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_lin.moments(...) or math_lin.moments_from_bank(...)")

    @classmethod
    def _wrap(cls, handle, scope, features, target, backend, device):
        self = object.__new__(cls)
        self._handle, self.scope, self.backend, self.device = handle, scope, backend, device
        self.features, self.target = tuple(features), target
        self.names = ("1",) + self.features + ((target,) if target is not None else ())
        i = self._info()
        self.policy = ("listwise", "pairwise")[i.policy]
        self.k, self.nslots, self.intercept = int(i.k), int(i.nslots), bool(i.intercept)
        self.width, self.marginals, self.source = int(i.width), bool(i.marginals), _SOURCES[i.source]
        self.scales = tuple(int(i.scale[j]) for j in range(self.k))
        sl = (_Slot * self.nslots)()
        _check(_call("tr_lin_moments_get_slots", handle, sl, self.nslots), what="slots")
        self.slots = tuple((int(s.a), int(s.b), int(s.kind), int(s.need)) for s in sl)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the moment state is closed")

    def _info(self):
        self._open()
        out = _Info()
        _check(_call("tr_lin_moments_get_info", self._handle, C.byref(out)), what="moments")
        return out

    @property
    def groups(self):
        return int(self._info().ngroups)

    @property
    def present(self):
        return int(self._info().present)

    @property
    def rows(self):
        return int(self._info().rows)

    @property
    def bytes(self):
        return int(self._info().bytes)

    def arrays(self):
        """copies of the native states: rows uint64[G], count uint64[G, S], sum uint64[G, S, 3] (192-bit words)"""
        self._open()
        st = _States()
        _check(_call("tr_lin_moments_get_states", self._handle, C.byref(st)), what="states")
        g, s = int(st.ngroups), int(st.nslots)
        rows = np.ctypeslib.as_array((C.c_uint64 * g).from_address(st.rows)).copy()
        count = np.ctypeslib.as_array((C.c_uint64 * (g * s)).from_address(st.count)).reshape(g, s).copy()
        sums = np.ctypeslib.as_array((C.c_uint64 * (g * s * WORDS)).from_address(st.sum)).reshape(g, s, WORDS).copy()
        return rows, count, sums

    @staticmethod
    def _value(words):
        v = int(words[0]) | (int(words[1]) << 64) | (int(words[2]) << 128)
        return v - (1 << 192) if v >> 191 else v

    def state(self, g):
        """{"present", "rows", "count": [per slot], "sum": [per slot]} of group g (exact ints, logical units)"""
        self._open()
        st = _States()
        _check(_call("tr_lin_moments_get_states", self._handle, C.byref(st)), what="states")
        g, s = int(g), int(st.nslots)
        if not 0 <= g < int(st.ngroups):
            raise ValueError("group id out of range")
        rows = int(C.c_uint64.from_address(st.rows + 8 * g).value)
        count = np.ctypeslib.as_array((C.c_uint64 * s).from_address(st.count + 8 * g * s)).tolist()
        sums = np.ctypeslib.as_array((C.c_uint64 * (s * WORDS)).from_address(st.sum + 8 * WORDS * g * s)).reshape(s, WORDS)
        return dict(present=bool(rows), rows=rows, count=[int(c) for c in count], sum=[self._value(w) for w in sums])

    def _gram(self, g, which):
        s = self.state(g)
        k = self.k
        m = [[None] * k for _ in range(k)]
        for j, (a, b, kind, _) in enumerate(self.slots):
            if kind == 0:
                m[a][b] = m[b][a] = s[which][j]
        return m

    def matrix(self, g):
        """the augmented Gram matrix of group g over [1, features..., target], exact ints in logical units
        (value units: entry (a, b) / 10^(scales[a] + scales[b])); pairwise: each entry over its own rows"""
        return self._gram(g, "sum")

    def counts(self, g):
        """the rows of every Gram entry of group g (common-listwise: all equal)"""
        return self._gram(g, "count")

    def pair_marginals(self, g, a, b):
        """pairwise: (count, sum a, sum b, sum a^2, sum b^2) over the rows where both columns are present"""
        if not self.marginals:
            raise ValueError("this state keeps no pair-complete marginals (a listwise state, or one from a bank)")
        ia, ib = self.names.index(a), self.names.index(b)
        if ia == ib or not ia or not ib:
            raise ValueError("two distinct non-constant columns")
        if ia > ib:
            ia, ib = ib, ia
            swap = True
        else:
            swap = False
        s = self.state(g)
        for j, (sa, sb, kind, need) in enumerate(self.slots):
            if kind == 1 and need == (1 << ia) | (1 << ib):
                c, x1, x2, q1, q2 = s["count"][j], s["sum"][j], s["sum"][j + 1], s["sum"][j + 2], s["sum"][j + 3]
                return (c, x2, x1, q2, q1) if swap else (c, x1, x2, q1, q2)
        raise AssertionError("pair not found")

    def equal(self, other):
        self._open(); other._open()
        return bool(_check(_call("tr_lin_moments_equal", self._handle, other._handle), what="moments comparison"))

    def __eq__(self, other):
        return isinstance(other, Moments) and self.equal(other)

    __hash__ = None

    def combine(self, other, sign=1, budget=None):
        """self + other (merge of disjoint selections) or self - other (sign -1: deleting rows that other summarizes)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_lin_moments_combine", self._handle, other._handle, int(sign), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        out = Moments._wrap(_made(h, error, refusal, "combine"), None, self.features, self.target, "cpu", "native CPU")
        return out

    def __add__(self, other):
        return self.combine(other, 1)

    def __sub__(self, other):
        return self.combine(other, -1)

    # ------------------------------------------------------------------ LIN-002 operators
    def _fs(self, group, intercept, ridge, features):
        return _fitspec(list(self.names), group, intercept, ridge, features, len(self.features))

    def system(self, group, *, intercept=None, ridge=0, features=None, budget=None):
        """the normal equations of a fit (value units): {"A", "b", "design", "yty", "n"}"""
        fs = self._fs(group, intercept, ridge, features)
        error, refusal = C.c_int32(), _Refusal()
        r = _Result(_made(_call("tr_lin_system", self._handle, C.byref(fs), C.byref(_budget(budget)), C.byref(refusal),
                                C.byref(error)), error, refusal, "normal equations"))
        design = [self.names[int(d)] for d in r.part("design")]
        st = r.part("stats")
        return dict(A=_rows(r.part("A"), len(design)), b=r.part("b"), design=design, yty=st[4],
                    n=None if st[5] is None else int(st[5]))

    def fit(self, group, *, intercept=None, ridge=0, features=None, budget=None):
        """the exact least-squares fit of group g (Fit); ridge: an exact penalty lambda >= 0 on the features;
        features: the subset kept (deletion requests); intercept: None = the state's declaration"""
        fs = self._fs(group, intercept, ridge, features)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_lin_fit", self._handle, C.byref(fs), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Fit(_Result(_made(h, error, refusal, "fit")), self.names)

    def fits(self, requests, *, backend="cpu", budget=None, max_rounds=8):
        """many fits of this state (requests: dicts of fit's arguments, each with "group"); backend "cuda" solves
        them together by modular elimination on the device (one work item per fit and prime), then Chinese
        remaindering and exact verification natively: the same exact results as the CPU"""
        specs = [self._fs(r["group"], r.get("intercept"), r.get("ridge", 0), r.get("features")) for r in requests]
        if backend == "cpu":
            out = []
            for fs in specs:
                error, refusal = C.c_int32(), _Refusal()
                h = _call("tr_lin_fit", self._handle, C.byref(fs), C.byref(_budget(budget)), C.byref(refusal),
                          C.byref(error))
                out.append(Fit(_Result(_made(h, error, refusal, "fit")), self.names))
            return out
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        arr = (_FitSpec * len(specs))(*specs)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_lin_batch_fits", self._handle, len(specs), arr, C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        batch = Batch(_made(h, error, refusal, "batch of fits"), fits=True)
        results, backend_name, device = batch.solve("cuda", budget=budget, max_rounds=max_rounds)
        return [Fit(r, self.names, backend_name, device) for r in results]

    def covariance(self, group, ddof=1, *, budget=None):
        """exact covariance (value units), squared correlation and its sign of group g (Covariance)"""
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_lin_covariance", self._handle, int(group), int(ddof), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return Covariance(_Result(_made(h, error, refusal, "covariance")), self.names[1:])

    def pca(self, group, kind="covariance", ddof=1, *, sweeps=None, budget=None):
        """the numerical eigensystem of the features' covariance (exact entries, enclosed) or correlation (enclosed
        entries) of group g (Eigen, APPROXIMATE with certified bounds; Eigen.matrix: the exact covariance)"""
        kinds = {"covariance": 0, "correlation": 1}
        if kind not in kinds:
            raise ValueError("kind is 'covariance' or 'correlation'")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_lin_pca", self._handle, int(group), kinds[kind], int(ddof), int(sweeps or 0),
                  C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
        return Eigen(_Result(_made(h, error, refusal, "pca")))

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_lin_moments_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Moments(closed)"
        return (f"Moments({self.policy}, columns={list(self.names)}, groups={self.groups}, present={self.present}, "
                f"width={self.width}, {self.backend})")


# ---------------------------------------------------------------------- exact systems
def _matrix(A, what="A"):
    rows = [list(r) for r in A]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError(f"{what}: a nonempty rectangular matrix")
    return rows


def solve(A, b, *, backend="cpu", budget=None, max_rounds=8):
    """The exact solution set of A x = b (rational entries): Solution. backend "cuda": Gauss-Jordan elimination
    modulo several primes on the device, Chinese remaindering, reconstruction and exact verification natively."""
    rows = _matrix(A)
    b = list(b)
    if len(b) != len(rows):
        raise ValueError("b has one value per row of A")
    m, n = len(rows), len(rows[0])
    if backend == "cuda":
        return solve_batch([(rows, b)], backend="cuda", budget=budget, max_rounds=max_rounds)[0]
    if backend != "cpu":
        raise ValueError("backend is 'cpu' or 'cuda'")
    qa, ka = _pack([v for r in rows for v in r], "A")
    qb, kb = _pack(b, "b")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_lin_solve", m, n, C.byref(qa), C.byref(qb), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Solution(_Result(_made(h, error, refusal, "solve")))


def solve_batch(systems, *, backend="cpu", budget=None, max_rounds=8):
    """[Solution] of many systems [(A, b), ...]; backend "cuda" solves them together on the device (one work item
    per system and prime)"""
    systems = [(_matrix(A), list(b)) for A, b in systems]
    if backend == "cpu":
        return [solve(A, b, budget=budget) for A, b in systems]
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    rows = np.array([len(A) for A, _ in systems], np.int32)
    unk = np.array([len(A[0]) for A, _ in systems], np.int32)
    for (A, b) in systems:
        if len(b) != len(A):
            raise ValueError("b has one value per row of A")
    qa, ka = _pack([v for A, _ in systems for r in A for v in r], "A")
    qb, kb = _pack([v for _, b in systems for v in b], "b")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_lin_batch_systems", len(systems), rows.ctypes.data, unk.ctypes.data, C.byref(qa), C.byref(qb),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    batch = Batch(_made(h, error, refusal, "batch of systems"))
    results, name, device = batch.solve("cuda", budget=budget, max_rounds=max_rounds)
    return [Solution(r, name, device) for r in results]


def primes(count):
    """the first `count` primes of the modular path (the largest below 2^62, descending) as (p, tr_gfp array)"""
    arr = (_GFP * int(count))()
    _check(_call("tr_lin_primes", int(count), arr), what="primes")
    return [int(f.p) for f in arr], arr


class Batch:
    """Systems for the modular path (native): the CUDA kernel's view, the residues per system and prime (on the
    device, or the CPU reference), and the native finish (CRT, reconstruction, exact verification)."""

    def __init__(self, handle, fits=False):
        self._h, self.fits = handle, fits
        self.view = _BatchView()
        _check(_call("tr_lin_batch_get_view", handle, C.byref(self.view)), what="batch view")
        self.primes_needed = int(_call("tr_lin_batch_primes_needed", handle))
        rows = np.ctypeslib.as_array((C.c_int32 * int(self.view.nsys)).from_address(self.view.rows))
        cols = np.ctypeslib.as_array((C.c_int32 * int(self.view.nsys)).from_address(self.view.cols))
        self._blocks = [(int(m), int(c)) for m, c in zip(rows, cols)]

    def words(self, nprimes):
        return sum(nprimes * (4 + m * c) for m, c in self._blocks)

    def residues(self, nprimes, backend="cpu", budget=None):
        """uint64 residues of every system modulo the first nprimes primes (headers and det-scaled reduced rows)"""
        refusal = _Refusal()
        if backend == "cuda":                        # the device copies are budgeted before anything is allocated
            raw = _Need()
            raw.need[RESOURCES.index("bytes")] = int(_call("tr_lin_batch_bytes", self._h, int(nprimes)))
            _check(_call("tr_math_budget_check", C.byref(_budget(budget)), C.byref(raw), C.byref(refusal)), refusal,
                   "modular elimination on the device")
        _, fields = primes(nprimes)
        out = np.zeros(max(self.words(nprimes), 1), np.uint64)
        if backend == "cpu":
            _check(_call("tr_lin_batch_residues", self._h, fields, nprimes, out.ctypes.data, C.byref(_budget(budget)),
                         C.byref(refusal)), refusal, "residues")
        elif backend == "cuda":
            _gpu().run("tr_mgpu_lin_eliminate", C.byref(self.view), fields, nprimes, C.byref(_budget(budget)),
                       out.ctypes.data, C.byref(refusal), refusal=refusal, what="modular elimination on the device")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return out, fields

    def finish(self, nprimes, residues, fields, budget=None):
        """(results, need): the native results, or None and the primes a rerun needs"""
        n = int(self.view.nsys)
        out = (C.c_void_p * n)()
        need, refusal = C.c_uint64(), _Refusal()
        rc = _call("tr_lin_batch_finish", self._h, fields, nprimes, residues.ctypes.data, out, C.byref(need),
                   C.byref(_budget(budget)), C.byref(refusal))
        if rc == -14:
            return None, int(need.value)
        _check(rc, refusal, "batch finish")
        return [_Result(out[i]) for i in range(n)], 0

    def solve(self, backend="cuda", budget=None, max_rounds=8):
        k = self.primes_needed
        for _ in range(max_rounds):
            residues, fields = self.residues(k, backend, budget)
            results, more = self.finish(k, residues, fields, budget)
            if results is not None:
                name, device = _backend_name() if backend == "cuda" else ("cpu", "native CPU")
                return results, name, device
            k = more
        raise NotConvergedError(f"the modular batch did not find enough agreeing primes in {max_rounds} rounds")

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _native is not None:
            _native["tr_lin_batch_free"](h)
            self._h = None


def inertia(A, *, budget=None):
    """the exact inertia and class of a symmetric rational matrix (Inertia)"""
    rows = _matrix(A)
    n = len(rows)
    if len(rows[0]) != n:
        raise ValueError("A must be square")
    q, keep = _pack([v for r in rows for v in r], "A")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_lin_inertia", n, C.byref(q), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Inertia(_Result(_made(h, error, refusal, "inertia")))


def eigen(A=None, *, lo=None, hi=None, sweeps=None, budget=None):
    """The numerical eigensystem (Eigen, APPROXIMATE with certified bounds) of a symmetric matrix: A exact (int /
    Fraction entries, enclosed natively) or the enclosure lo <= A <= hi (exact doubles, symmetric)."""
    error, refusal = C.c_int32(), _Refusal()
    if A is not None:
        rows = _matrix(A)
        n = len(rows)
        if len(rows[0]) != n:
            raise ValueError("A must be square")
        q, keep = _pack([v for r in rows for v in r], "A")
        h = _call("tr_lin_eigen_exact", n, C.byref(q), int(sweeps or 0), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
    else:
        l, u = np.ascontiguousarray(lo, np.float64), np.ascontiguousarray(hi, np.float64)
        if l.ndim != 2 or l.shape[0] != l.shape[1] or l.shape != u.shape:
            raise ValueError("lo and hi: square matrices of one shape")
        h = _call("tr_lin_eigen", l.shape[0], l.ctypes.data, u.ctypes.data, int(sweeps or 0), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
    return Eigen(_Result(_made(h, error, refusal, "eigen")))
