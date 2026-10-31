# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LIN-003: leave-group-out and low-rank model updates on LIN-001 / LIN-002 states, natively on the CPU or with the
CUDA kernel (torbles/math_linupd.c, math_linupd_gpu.cu).

    from fractions import Fraction
    from torbles import math_bank as mbk, math_lin as lin, math_linupd as lu
    st = lin.moments(scope, ["x1", "x2"], "y", policy="listwise")            # LIN-001 grouped moment states
    M = lu.fit(st, group=0, ridge=Fraction(1, 2))                            # an exact solved model with A^-1
    M.x, M.kind, M.rss, M.inverse, M.det                                     # exact Fractions
    M.delete_rows([dict(x1=3, x2=-1, y=7)])                                  # Sherman-Morrison, denominator 1 - h
    M.insert_rows([...]); M.update(U, V, db); M.ridge(Fraction(2))           # low-rank updates, exact
    M.delete_rows(rows, fallback="fail")                                     # SingularError when det(K) = 0
    M.subtract(st_other, group)                                              # minus a compatible group state
    L = lu.leave_one_group_out(st, backend="cuda")                           # every group left out (modular, exact)
    M.float_solve(tol=1e-9)                                                  # binary64 with a certified error bound

Every exact result is a Python int / Fraction computed natively (Python formats it). The update contract when the
denominator det(K) is exactly 0 (the updated system is singular) is declared per call: fallback="rebuild" (an exact
rebuild: LIN-002's canonical solution set) or "fail" (SingularError). The floating variant refuses
(NotConvergedError) when its condition bound fails. Opt-in: `import torbles` does not load this module.
"""
import ctypes as C
from fractions import Fraction
from numbers import Integral, Rational

import numpy as np

from . import algq
from . import math_lin as _lin
from .math_budgets import _Budget, _Refusal, Budget
from .math_errors import NotConvergedError, check as _check

__all__ = ["Model", "LogoPlan", "fit", "system", "leave_one_group_out", "logo_plan", "FALLBACKS", "MAX_RANK",
           "FLOAT_CONTRACT"]

FALLBACKS = {"fail": 0, "rebuild": 1}
MAX_RANK, MAX_COLS = 64, 18
PART = dict(x=0, null=1, cert=2, A=3, A0=4, b=5, inverse=6, stats=7, det=8, denominator=9, design=10)
_METHODS = {1: "built", 2: "woodbury", 3: "rebuilt", 4: "subtracted", 5: "modular"}
_KINDS = {1: "unique", 2: "family", 3: "none"}
_REASONS = {0: "certified", 1: "a zero pivot in binary64 (singular to working precision)",
            2: "alpha = ||I - R A||_inf >= 1 (the condition bound fails)", 3: "the certified error bound exceeds tol"}
FLOAT_CONTRACT = ("approximate, certified: x~ in binary64 from an approximate inverse R of A's midpoint (one "
                  "refinement step); with alpha >= ||I - R A||_inf < 1 over the exact enclosure of A, A is nonsingular "
                  "and ||x - x~||_inf <= bound = ||R (b - A x~)||_inf / (1 - alpha) (outward-rounded intervals); "
                  "refused when alpha >= 1 or bound > tol; kappa >= ||A||_inf ||R||_inf is the condition estimate")
P = C.POINTER


class _Info(C.Structure):
    _native_ = "tr_linupd_info"
    _fields_ = [(n, C.c_int32) for n in ("n", "kind", "rank", "method", "regression", "listwise", "intercept",
                                         "ridged", "inverse", "fallback", "update_rank", "denominator_sign")] + [
        (n, C.c_uint64) for n in ("rows", "group", "pivots", "primes")] + [("design", C.c_int32 * MAX_COLS)]


class _LogoView(C.Structure):
    _native_ = "tr_linupd_logo_view"
    _fields_ = [("ngroups", C.c_uint64), ("nsys", C.c_uint64)] + [
        (n, C.c_int32) for n in ("k", "nslots", "n", "ycol")] + [
        ("design", C.c_int32 * MAX_COLS), ("pen", C.c_int32 * MAX_COLS), ("scale", C.c_int32 * MAX_COLS),
        ("T", C.c_int32), ("reserved", C.c_int32), ("ridge_num", C.c_int64), ("ridge_den", C.c_int64),
        ("sum", C.c_void_p), ("total", C.c_void_p), ("groups", C.c_void_p)]


class _FloatInfo(C.Structure):
    _native_ = "tr_linupd_float_info"
    _fields_ = [(n, C.c_double) for n in ("alpha", "bound", "kappa", "xnorm", "tol")] + [
        ("certified", C.c_int32), ("reason", C.c_int32)]


for _s, _n in ((_Info, 152), (_LogoView, 296), (_FloatInfo, 48)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None
_gpu_bound = False


def _api():
    global _native
    if _native is None:
        vp, i32, i64, u64, dbl, B, R, E = (C.c_void_p, C.c_int32, C.c_int64, C.c_uint64, C.c_double, P(_Budget),
                                          P(_Refusal), P(C.c_int32))
        Q = P(_lin._QBuf)
        sig = {
            "tr_linupd_fit": (vp, [vp, P(_lin._FitSpec), i32, B, R, E]),
            "tr_linupd_system": (vp, [i32, Q, Q, i32, B, R, E]),
            "tr_linupd_update": (vp, [vp, i32, Q, Q, Q, i32, B, R, E]),
            "tr_linupd_rows": (vp, [vp, i32, i32, Q, Q, i32, B, R, E]),
            "tr_linupd_ridge": (vp, [vp, i64, i64, i32, i32, B, R, E]),
            "tr_linupd_subtract": (vp, [vp, vp, u64, i32, i32, B, R, E]),
            "tr_linupd_get_info": (i32, [vp, P(_Info)]),
            "tr_linupd_count": (u64, [vp, i32]),
            "tr_linupd_limbs": (u64, [vp, i32]),
            "tr_linupd_export": (i32, [vp, i32, Q]),
            "tr_linupd_equal": (i32, [vp, vp]),
            "tr_linupd_float_solve": (i32, [vp, dbl, vp, P(_FloatInfo), B, R]),
            "tr_linupd_free": (None, [vp]),
            "tr_linupd_logo_plan": (vp, [vp, P(_lin._FitSpec), u64, vp, B, R, E]),
            "tr_linupd_logo_get_view": (i32, [vp, P(_LogoView)]),
            "tr_linupd_logo_primes_needed": (u64, [vp]),
            "tr_linupd_logo_bytes": (u64, [vp, u64]),
            "tr_linupd_logo_solve": (i32, [vp, i32, vp, B, R]),
            "tr_linupd_logo_residues": (i32, [P(_LogoView), vp, u64, vp, B, R]),
            "tr_linupd_logo_finish": (i32, [vp, vp, u64, vp, vp, P(u64), B, R]),
            "tr_linupd_logo_free": (None, [vp]),
            "tr_math_budget_check": (i32, [B, vp, R]),
        }
        native = {}
        for name, (restype, argtypes) in sig.items():
            fn = getattr(algq.lib, name)
            fn.restype, fn.argtypes = restype, argtypes
            native[name] = fn
        _native = native
    return _native


def _call(name, *args):
    return _api()[name](*args)


def _gpu():
    global _gpu_bound
    from . import math_gpu
    if not _gpu_bound:
        math_gpu.bind("tr_mgpu_linupd_logo", C.c_int, [P(_LogoView), C.c_void_p, C.c_uint64, P(_Budget), C.c_void_p,
                                                       P(_Refusal)])
        _gpu_bound = True
    return math_gpu


def _raw(budget):
    return C.byref((budget or Budget()).raw)


def _made(handle, error, refusal, what):
    _check(error.value, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


def _fallback(name):
    if name not in FALLBACKS:
        raise ValueError(f"fallback is one of {', '.join(FALLBACKS)}")
    return FALLBACKS[name]


def _exact(v, what):
    return _lin._exact(v, what)


def _rows(flat, n):
    return [flat[i * n:(i + 1) * n] for i in range(len(flat) // n)] if flat else []


# ---------------------------------------------------------------------- models
class Model:
    """An immutable exact solved system (native): the canonical solution set of A x = b (LIN-002's), the system, and
    (regression models) A0, y^T y and LIN-002's statistics; A^-1 and det(A) when kept; the last update's denominator
    det(K) (Sherman-Morrison: 1 + v^T A^-1 u) and how the model was obtained."""

    def __init__(self, handle, names=None, backend="cpu", device="native CPU"):
        self._h, self.names, self.backend, self.device = handle, names, backend, device
        i = _Info()
        _check(_call("tr_linupd_get_info", handle, C.byref(i)), what="model")
        self.n, self.kind, self.rank = int(i.n), _KINDS[int(i.kind)], int(i.rank)
        self.nullity = self.n - self.rank if self.kind != "none" else None
        self.method, self.fallback = _METHODS[int(i.method)], bool(i.fallback)
        self.regression, self.listwise = bool(i.regression), bool(i.listwise)
        self.intercept, self.ridged, self.has_inverse = bool(i.intercept), bool(i.ridged), bool(i.inverse)
        self.update_rank, self.denominator_sign = int(i.update_rank), (None if i.denominator_sign == 2 else
                                                                         int(i.denominator_sign))
        self.rows, self.group, self.primes = int(i.rows), int(i.group), int(i.primes)
        self.pivots = [j for j in range(self.n) if (int(i.pivots) >> j) & 1]
        self._design = [int(i.design[u]) for u in range(self.n)] if self.regression else None
        self._cache = {}

    def _open(self):
        if not self._h:
            raise ValueError("the model is closed")
        return self._h

    def part(self, name):
        """an exact part (Fractions; None for an undefined value), or None when the model has none"""
        if name in self._cache:
            return self._cache[name]
        p = PART[name]
        n = int(_call("tr_linupd_count", self._open(), p))
        out = None
        if n:
            nl = int(_call("tr_linupd_limbs", self._h, p))
            sign, nlen, dlen = np.zeros(n, np.int8), np.zeros(n, np.uint32), np.zeros(n, np.uint32)
            limbs = np.zeros(max(nl, 1), np.uint64)
            q = _lin._QBuf(n, nl, sign.ctypes.data, nlen.ctypes.data, dlen.ctypes.data, limbs.ctypes.data)
            _check(_call("tr_linupd_export", self._h, p, C.byref(q)), what=f"model part {name}")
            out = _lin._decode(sign, nlen, dlen, limbs, n)
        self._cache[name] = out
        return out

    # exact parts
    @property
    def x(self):
        return self.part("x")

    @property
    def nullspace(self):
        return _rows(self.part("null"), self.n)

    @property
    def certificate(self):
        return self.part("cert")

    @property
    def A(self):
        return _rows(self.part("A"), self.n)

    @property
    def A0(self):
        return _rows(self.part("A0"), self.n) if self.regression else None

    @property
    def b(self):
        return self.part("b")

    @property
    def inverse(self):
        return _rows(self.part("inverse"), self.n) if self.has_inverse else None

    @property
    def det(self):
        d = self.part("det")
        return d[0] if d else None

    @property
    def denominator(self):
        """the last update's det(K) = det(I + V^T A^-1 U) (rank one: 1 + v^T A^-1 u), exactly; None when unknown"""
        d = self.part("denominator")
        return d[0] if d else None

    @property
    def design(self):
        if not self.regression:
            return None
        return [self.names[d] for d in self._design] if self.names else list(self._design)

    def _stat(self, i):
        st = self.part("stats")
        return None if st is None else st[i]

    rss = property(lambda self: self._stat(0))
    tss = property(lambda self: self._stat(1))
    r2 = property(lambda self: self._stat(2))
    sigma2 = property(lambda self: self._stat(3))
    yty = property(lambda self: self._stat(4))
    projection = property(lambda self: self._stat(6))

    @property
    def coefficients(self):
        return dict(zip(self.design, self.x)) if self.regression and self.x is not None else None

    def same(self, other):
        """the same canonical solution set, system, statistics and (when kept) inverse, decided natively"""
        return bool(_check(_call("tr_linupd_equal", self._open(), other._open()), what="model comparison"))

    # updates
    def _wrap(self, h, error, refusal, what):
        return Model(_made(h, error, refusal, what), self.names, self.backend, self.device)

    def update(self, U, V, db=None, *, fallback="rebuild", budget=None):
        """A + U V^T (U, V: n x r exact matrices), b + db: Sherman-Morrison-Woodbury from the kept inverse with the
        exact capacitance K = I + V^T A^-1 U; det(K) = 0 (a singular update) follows the fallback contract"""
        U, V = [list(r) for r in U], [list(r) for r in V]
        if len(U) != self.n or len(V) != self.n or not U or any(len(r) != len(U[0]) for r in U + V):
            raise ValueError(f"U and V are {self.n} x r matrices of one shape")
        r = len(U[0])
        qu, ku = _lin._pack([v for row in U for v in row], "U")
        qv, kv = _lin._pack([v for row in V for v in row], "V")
        qd = kd = None
        if db is not None:
            db = list(db)
            if len(db) != self.n:
                raise ValueError(f"db has {self.n} values")
            qd, kd = _lin._pack(db, "db")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_linupd_update", self._open(), r, C.byref(qu), C.byref(qv), C.byref(qd) if qd is not None else None,
                  _fallback(fallback), _raw(budget), C.byref(refusal), C.byref(error))
        return self._wrap(h, error, refusal, "low-rank update")

    def _row_values(self, rows):
        if not self.regression:
            from .math_domains import DomainError
            raise DomainError("rows updates need a regression model (a fit of a moment state)")
        feats = [self.names[d] for d in self._design if d != 0]
        target = self.names[-1]
        Z, y = [], []
        for row in rows:
            if isinstance(row, dict):
                missing = [f for f in feats + [target] if f not in row]
                if missing:
                    raise ValueError(f"a row needs every design feature and the target: missing {missing}")
                Z.extend(row[f] for f in feats)
                y.append(row[target])
            else:
                xs, yv = row
                xs = list(xs)
                if len(xs) != len(feats):
                    raise ValueError(f"a row has {len(feats)} feature values ({feats}) and the target")
                Z.extend(xs)
                y.append(yv)
        return Z, y

    def _rows_update(self, rows, sign, fallback, budget):
        rows = list(rows)
        Z, y = self._row_values(rows)
        qz, kz = _lin._pack(Z, "row values")
        qy, ky = _lin._pack(y, "target values")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_linupd_rows", self._open(), sign, len(rows), C.byref(qz), C.byref(qy), _fallback(fallback),
                  _raw(budget), C.byref(refusal), C.byref(error))
        return self._wrap(h, error, refusal, "row update")

    def delete_rows(self, rows, *, fallback="rebuild", budget=None):
        """the model without these complete rows (dicts {feature: value, target: value} in value units, or
        (feature values in design order, y)): rank-r downdate, denominator det(I - Z A^-1 Z^T) (one row: 1 - h)"""
        return self._rows_update(rows, -1, fallback, budget)

    def insert_rows(self, rows, *, fallback="rebuild", budget=None):
        """the model with these rows added (rank-r update, denominator det(I + Z A^-1 Z^T))"""
        return self._rows_update(rows, 1, fallback, budget)

    def ridge(self, lam, *, method="woodbury", fallback="rebuild", budget=None):
        """the model with ridge lambda' (an exact >= 0 penalty on the features): the rank-p Woodbury update
        (lambda' - lambda) E E^T, or method="rebuild" from A0 + lambda' D; both exact and equal"""
        lam = _exact(lam, "ridge")
        if lam < 0:
            raise ValueError("the ridge penalty is >= 0")
        if not (-(1 << 63) <= lam.numerator < (1 << 63) and lam.denominator < (1 << 63)):
            raise OverflowError("the ridge penalty's numerator and denominator must fit int64")
        methods = {"woodbury": 1, "rebuild": 2}
        if method not in methods:
            raise ValueError("method is 'woodbury' or 'rebuild'")
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_linupd_ridge", self._open(), lam.numerator, lam.denominator, methods[method], _fallback(fallback),
                  _raw(budget), C.byref(refusal), C.byref(error))
        return self._wrap(h, error, refusal, "ridge change")

    def subtract(self, st, group, *, sign=-1, inverse=None, budget=None):
        """the normal equations minus (sign -1) or plus (+1) the Gram matrix of group g of a compatible moment state
        (same columns, target, policy and scales), solved exactly again"""
        keep = self.has_inverse if inverse is None else bool(inverse)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_linupd_subtract", self._open(), st._handle, int(group), int(sign), int(keep), _raw(budget),
                  C.byref(refusal), C.byref(error))
        return self._wrap(h, error, refusal, "group subtraction")

    def float_solve(self, tol=1e-9, *, fallback="fail", budget=None):
        """the binary64 solution with a certified bound (FLOAT_CONTRACT): {"x" (floats), "bound", "alpha", "kappa",
        "xnorm", "tol", "certified": True}; NotConvergedError (with .info) when the condition bound fails, unless
        fallback="exact" (then the exact solution, "certified": False, and the failed info)"""
        if fallback not in ("fail", "exact"):
            raise ValueError("fallback is 'fail' or 'exact'")
        out = np.zeros(max(self.n, 1), np.float64)
        info, refusal = _FloatInfo(), _Refusal()
        rc = _call("tr_linupd_float_solve", self._open(), float(tol), out.ctypes.data, C.byref(info), _raw(budget),
                   C.byref(refusal))
        res = dict(bound=info.bound, alpha=info.alpha, kappa=info.kappa, xnorm=info.xnorm, tol=info.tol,
                   reason=_REASONS.get(int(info.reason), "unknown"), contract=FLOAT_CONTRACT)
        if rc == -14:
            if fallback == "exact":
                return dict(res, x=self.x, certified=False, exact=True)
            err = NotConvergedError(f"floating solve refused: {res['reason']} (alpha={info.alpha:.3g}, "
                                    f"bound={info.bound:.3g}, kappa={info.kappa:.3g}, tol={tol:.3g})")
            err.info = res
            raise err
        _check(rc, refusal, "floating solve")
        return dict(res, x=out[:self.n].tolist(), certified=True, exact=False)

    def close(self):
        if getattr(self, "_h", None):
            _call("tr_linupd_free", self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"Model({self.kind}, n={self.n}, rank={self.rank}, method={self.method}, fallback={self.fallback}, "
                f"inverse={self.has_inverse}, {self.backend})")


def fit(st, group=0, *, intercept=None, ridge=0, features=None, inverse=True, budget=None):
    """the exact solved model of a fit of one group of a LIN-001 moment state (math_lin.Moments): LIN-002's normal
    equations (value units, exact ridge on the features, a feature subset), with A^-1 and det(A) kept (inverse=True)
    for low-rank updates"""
    fs = st._fs(group, intercept, ridge, features)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_linupd_fit", st._handle, C.byref(fs), int(bool(inverse)), _raw(budget), C.byref(refusal),
              C.byref(error))
    return Model(_made(h, error, refusal, "model"), st.names)


def system(A, b, *, inverse=True, budget=None):
    """the exact solved model of a square rational system A x = b (int / Fraction entries)"""
    rows = _lin._matrix(A)
    n = len(rows)
    if len(rows[0]) != n:
        raise ValueError("A must be square")
    b = list(b)
    if len(b) != n:
        raise ValueError("b has one value per row of A")
    qa, ka = _lin._pack([v for r in rows for v in r], "A")
    qb, kb = _lin._pack(b, "b")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_linupd_system", n, C.byref(qa), C.byref(qb), int(bool(inverse)), _raw(budget), C.byref(refusal),
              C.byref(error))
    return Model(_made(h, error, refusal, "model"))


class LogoPlan:
    """The leave-one-group-out systems of a moment state (native): the copied group states and their total, the
    exact integer complement systems; solved exactly on the CPU, or modulo primes on the device then Chinese-
    remaindered and verified natively."""

    def __init__(self, handle, names):
        self._h, self.names = handle, names
        self.view = _LogoView()
        _check(_call("tr_linupd_logo_get_view", handle, C.byref(self.view)), what="LOGO plan")
        self.systems, self.n = int(self.view.nsys), int(self.view.n)
        self.primes_needed = int(_call("tr_linupd_logo_primes_needed", handle))
        self.groups = np.ctypeslib.as_array((C.c_uint64 * self.systems).from_address(self.view.groups)).tolist()

    def device_bytes(self, nprimes):
        return int(_call("tr_linupd_logo_bytes", self._h, int(nprimes)))

    def _models(self, arr, backend, device):
        return [Model(arr[i], self.names, backend, device) for i in range(self.systems)]

    def solve(self, backend="cpu", *, inverse=False, budget=None, max_rounds=8):
        """[Model] for every left-out group, in plan order"""
        if backend == "cpu":
            arr = (C.c_void_p * self.systems)()
            refusal = _Refusal()
            _check(_call("tr_linupd_logo_solve", self._h, int(bool(inverse)), arr, _raw(budget), C.byref(refusal)),
                   refusal, "leave-one-group-out")
            return self._models(arr, "cpu", "native CPU")
        if backend != "cuda":
            raise ValueError("backend is 'cpu' or 'cuda'")
        if inverse:
            raise ValueError("the modular path returns the solution sets (inverse=False)")
        k = self.primes_needed
        for _ in range(max_rounds):
            residues, fields = self.residues(k, "cuda", budget)
            models, more = self.finish(k, residues, fields, budget)
            if models is not None:
                name, device = _lin._backend_name()
                for m in models:
                    m.backend, m.device = name, device
                return models
            k = more
        raise NotConvergedError(f"leave-one-group-out: not enough agreeing primes in {max_rounds} rounds")

    def residues(self, nprimes, backend="cpu", budget=None):
        """the residues of every system modulo the first nprimes primes of the modular path (headers and det-scaled
        reduced rows), on the device or the CPU reference"""
        _, fields = _lin.primes(nprimes)
        out = np.zeros(max(self.systems * nprimes * (4 + self.n * (self.n + 1)), 1), np.uint64)
        refusal = _Refusal()
        if backend == "cpu":
            _check(_call("tr_linupd_logo_residues", C.byref(self.view), fields, nprimes, out.ctypes.data, _raw(budget),
                         C.byref(refusal)), refusal, "LOGO residues")
        elif backend == "cuda":
            _gpu().run("tr_mgpu_linupd_logo", C.byref(self.view), fields, nprimes, _raw(budget), out.ctypes.data,
                       C.byref(refusal), refusal=refusal, what="leave-one-group-out on the device")
        else:
            raise ValueError("backend is 'cpu' or 'cuda'")
        return out, fields

    def finish(self, nprimes, residues, fields, budget=None):
        """([Model], 0) or (None, the primes a rerun needs)"""
        arr = (C.c_void_p * self.systems)()
        need, refusal = C.c_uint64(), _Refusal()
        rc = _call("tr_linupd_logo_finish", self._h, fields, nprimes, residues.ctypes.data, arr, C.byref(need),
                   _raw(budget), C.byref(refusal))
        if rc == -14:
            return None, int(need.value)
        _check(rc, refusal, "LOGO finish")
        return self._models(arr, "cpu", "native CPU"), 0

    def close(self):
        if getattr(self, "_h", None):
            _call("tr_linupd_logo_free", self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def logo_plan(st, *, groups=None, intercept=None, ridge=0, features=None, budget=None):
    """the leave-one-group-out plan of a moment state (groups: the ids to leave out, None = every group)"""
    fs = st._fs(0, intercept, ridge, features)
    if groups is None:
        sel, n = None, 0
    else:
        sel = np.ascontiguousarray([int(g) for g in groups], dtype=np.uint64)
        n = len(sel)
        if not n:
            raise ValueError("give at least one group")
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_linupd_logo_plan", st._handle, C.byref(fs), n, sel.ctypes.data if sel is not None else None,
              _raw(budget), C.byref(refusal), C.byref(error))
    return LogoPlan(_made(h, error, refusal, "leave-one-group-out plan"), st.names)


def leave_one_group_out(st, *, groups=None, intercept=None, ridge=0, features=None, inverse=False, backend="cpu",
                        budget=None, max_rounds=8):
    """[Model]: for every group g (or the given ones) the exact fit of the data without g, from the moment state's
    total minus g's state (no rescan); backend "cuda" solves every (group, prime) on the device"""
    plan = logo_plan(st, groups=groups, intercept=intercept, ridge=ridge, features=features, budget=budget)
    try:
        return plan.solve(backend, inverse=inverse, budget=budget, max_rounds=max_rounds)
    finally:
        plan.close()
