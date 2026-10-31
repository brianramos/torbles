# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-007: proof certificates and their evidence scope, natively on the CPU or with the CUDA kernel
(torbles/math_certificates.h/.c, math_certificates_gpu.cu).

    from torbles import math_bank as mbk, math_certificates as mc
    scope = mbk.Scope.from_table(db["f"], ["x", "a", "s", "y"], rows=mask)
    cert = mc.snapshot(scope, mc.affine("a", "x", 3, -7))            # exhaustive; backend="cuda" too
    cert.fresh(scope)                                                 # Freshness(fresh, reasons, first, ...)
    cert.authorize_exact(scope)                                       # Authorization, or a refusal raised
    mc.hypothesis(scope, mc.sum_of("s", "x", "y"), size=100, seed=1)   # a sample: never exact
    mc.law(mc.polynomial({(1, 1): 1, (1, 1): -1}, ["x", "y"]))        # a ring identity, decided natively
    mc.schema(dim_scope, mc.unique("id"), enforcement="load")        # an enforced schema fact
    mc.verify(scope, [claims...], backend="cuda")                     # exhaustive evidence of many claims
    mc.reverify([certs...], scope, backend="cuda")                   # re-certification after an invalidation
    mc.derive(claim, [premises], [cofactors])                         # an ideal-membership certificate

Kinds: "law" (holds by mathematics, the law named), "schema" (a declared constraint verified and enforced:
enforcement "load" or "write"), "snapshot" (verified on every selected row of an exact scope) and "hypothesis"
(checked on a sample). Only laws, enforced schema facts and fresh exact snapshot certificates authorize an exact
rewrite; a sampled or approximate claim never does. Freshness reports every reason that applies: identity, revision,
rows, selection, null_domain, columns, grouping, join_revision, join_key. Nothing is computed in Python. Opt-in:
`import torbles` does not load it.
"""
import ctypes as C
import uuid
from collections import namedtuple

import numpy as np

from . import algq
from . import math_bank as _mbk
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_carriers import _GFP
from .math_domains import DomainError
from .math_errors import RESOURCES, StaleStateError, UnsupportedError, check as _check

__all__ = ["Claim", "Certificate", "Dimension", "Freshness", "Authorization", "ClaimViolated", "NotAuthorizedError",
           "StaleCertificateError", "affine", "sum_of", "product", "parity", "refinement", "coverage", "polynomial",
           "unique", "not_null", "reference", "snapshot", "hypothesis", "schema", "law", "derive", "verify",
           "reverify", "describe", "code_scope", "KINDS", "CLAIMS", "POLICIES", "LAWS", "STALE_REASONS"]

KINDS = {"law": 1, "schema": 2, "snapshot": 3, "hypothesis": 4}
CLAIMS = {"affine": 1, "sum": 2, "product": 3, "parity": 4, "refinement": 5, "coverage": 6, "polynomial": 7,
          "unique": 8, "not_null": 9, "reference": 10}
POLICIES = {"forbid": 1, "skip": 2, "propagate": 3}
LAWS = {"ring_identity": 1, "gfp_function": 2}
ENFORCEMENT = {"load": 1, "write": 2}
SAMPLERS = {"uniform": 1, "stride": 2, "prefix": 3}
STALE_REASONS = ("identity", "revision", "rows", "selection", "null_domain", "columns", "grouping", "join_revision",
                 "join_key")
REFUSALS = {0: None, 1: "statistical", 2: "approximate", 3: "carrier", 4: "stale", 5: "incomplete", 6: "no scope",
            7: "premise"}
UNITS = {1: "rows", 2: "groups"}
CARRIER_I128, CARRIER_GFP = 2, 6
NONE = (1 << 64) - 1
_I64 = (-(1 << 63), (1 << 63) - 1)
_NAMES = {v: k for k, v in CLAIMS.items()}
_KIND_NAMES = {v: k for k, v in KINDS.items()}
_POLICY_NAMES = {v: k for k, v in POLICIES.items()}
_LAW_NAMES = {v: k for k, v in LAWS.items()}
P = C.POINTER


class ClaimViolated(DomainError):
    """The claim does not hold on the scope (no certificate); .evidence has the violations and the witness."""


class NotAuthorizedError(DomainError):
    """The certificate cannot authorize an exact rewrite; .reason ("statistical", "approximate", "carrier",
    "no scope", "incomplete", "premise") and .decision."""


class StaleCertificateError(StaleStateError):
    """The certificate's scope changed; .reasons (every reason that applies), .first, .freshness."""


# ------------------------------------------------------------------ ctypes mirrors (registered with _native_)
class _Bit(C.Structure):
    _native_ = "tr_cert_bit"
    _fields_ = [("col", C.c_int32), ("kind", C.c_int32), ("bit", C.c_int32), ("reserved", C.c_int32),
                ("base", C.c_int64)]


class _Claim(C.Structure):
    _native_ = "tr_cert_claim"
    _fields_ = [("kind", C.c_int32), ("policy", C.c_int32), ("ncols", C.c_int32), ("nbits", C.c_int32),
                ("col", C.c_int32 * 8), ("carrier", C.c_int32), ("constant", C.c_int32), ("field", C.c_uint64),
                ("tolerance", C.c_uint64), ("scale", C.c_int64), ("bias", C.c_int64), ("mask", C.c_uint64),
                ("nterms", C.c_uint64), ("nvalues", C.c_uint64), ("gf", _GFP), ("bits", C.c_void_p),
                ("exponents", C.c_void_p), ("coefficients", C.c_void_p), ("keys", C.c_void_p),
                ("targets", C.c_void_p), ("used", C.c_void_p)]


class _Evidence(C.Structure):
    _native_ = "tr_cert_evidence"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "selected", "checked", "excluded", "violations", "witness",
                                          "overflow")] + [("unit", C.c_int32), ("holds", C.c_int32)]


class _Sample(C.Structure):
    _native_ = "tr_cert_sample"
    _fields_ = [("method", C.c_int32), ("reserved", C.c_int32)] + [
        (n, C.c_uint64) for n in ("size", "seed", "population", "distinct")]


class _Staleness(C.Structure):
    _native_ = "tr_cert_staleness"
    _fields_ = [("reasons", C.c_uint32), ("first", C.c_int32), ("column", C.c_int32), ("dim", C.c_int32),
                ("recorded", C.c_uint64), ("current", C.c_uint64)]


class _Decision(C.Structure):
    _native_ = "tr_cert_decision"
    _fields_ = [(n, C.c_int32) for n in ("authorized", "reason", "kind", "claim", "policy", "premise")] + [
        ("field", C.c_uint64), ("stale", _Staleness)]


class _Info(C.Structure):
    _native_ = "tr_cert_info"
    _fields_ = [(n, C.c_int32) for n in ("kind", "claim", "law", "enforcement", "policy", "carrier", "exact",
                                         "backend")] + [
        ("field", C.c_uint64), ("modulus", C.c_uint64), ("tolerance", C.c_uint64)] + [
        (n, C.c_int32) for n in ("scoped", "npremises", "depth", "reserved")] + [
        ("bytes", C.c_uint64), ("digest", C.c_uint64 * 2), ("evidence", _Evidence), ("sample", _Sample)]


class _ClaimDesc(C.Structure):
    _native_ = "tr_cert_claim_desc"
    _fields_ = [(n, C.c_int32) for n in ("kind", "min_cols", "max_cols", "unit", "policies", "output", "local",
                                         "field")] + [("name", C.c_char * 32)]


class _LawDesc(C.Structure):
    _native_ = "tr_cert_law_desc"
    _fields_ = [(n, C.c_int32) for n in ("law", "claim", "any_ring", "reserved")] + [
        ("name", C.c_char * 48), ("statement", C.c_char * 112)]


class _Column(C.Structure):
    _native_ = "tr_cert_column"
    _fields_ = [(n, C.c_int32) for n in ("index", "col", "valid", "hi", "shift", "scale", "storage", "reserved")] + [
        ("offset", C.c_int64), ("nulls", C.c_uint64), ("null_digest", C.c_uint64 * 2)]


class _Dim(C.Structure):
    _native_ = "tr_cert_dim"
    _fields_ = [("identity", C.c_uint8 * 16), ("revision", C.c_uint64), ("rows", C.c_uint64), ("used", C.c_uint64),
                ("used_digest", C.c_uint64 * 2), ("duplicates", C.c_uint64), ("nulls", C.c_uint64),
                ("key", C.c_int32), ("key_valid", C.c_int32), ("unique", C.c_int32), ("reserved", C.c_int32)]


class _Source(C.Structure):
    _native_ = "tr_cert_source"
    _fields_ = [("identity", C.c_uint8 * 16)] + [(n, C.c_uint64) for n in ("revision", "rows", "selected", "ngroups")] + [
        (n, C.c_uint64 * 2) for n in ("version", "selection", "grouping", "data")] + [
        (n, C.c_int32) for n in ("source", "ncols", "ndims", "flags")] + [
        ("grouping_bytes", C.c_uint64), ("col", _Column * 32), ("dim", _Dim * 8)]


for _s, _n in ((_Bit, 24), (_Claim, 192), (_Evidence, 64), (_Sample, 40), (_Staleness, 32), (_Decision, 64),
               (_Info, 200), (_ClaimDesc, 64), (_LawDesc, 176), (_Column, 64), (_Dim, 88), (_Source, 2888)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, e = C.c_void_p, C.c_int32, C.c_uint64, P(C.c_int32)
        sig = {
            "tr_cert_claim_describe": (i32, [i32, P(_ClaimDesc)]),
            "tr_cert_law_describe": (i32, [i32, P(_LawDesc)]),
            "tr_cert_claim_check": (i32, [P(_Claim), i32, P(_Need)]),
            "tr_cert_poly_canonical": (i32, [i32, u64, vp, vp, u64, i32, vp, vp, P(u64)]),
            "tr_cert_dim_describe": (i32, [vp, vp, vp, i32, i32, P(_Budget), P(_Refusal), P(_Dim)]),
            "tr_cert_record_get": (i32, [vp, P(_Source)]),
            "tr_cert_record_selection": (i32, [vp, vp, u64]),
            "tr_cert_verify": (i32, [vp, P(_Claim), i32, P(_Budget), P(_Evidence), P(_Refusal)]),
            "tr_cert_verify_view": (i32, [P(_mbk._View), P(_Claim), i32, P(_Budget), P(_Evidence), P(_Refusal)]),
            "tr_cert_verify_need": (i32, [P(_mbk._View), P(_Claim), i32, i32, P(_Need)]),
            "tr_cert_law": (vp, [i32, P(_Claim), P(_Budget), P(_Refusal), e]),
            "tr_cert_schema": (vp, [vp, P(_Claim), i32, P(_Dim), i32, P(_Evidence), P(_Budget), P(_Refusal), e]),
            "tr_cert_snapshot": (vp, [vp, P(_Claim), P(_Dim), i32, P(_Evidence), P(_Budget), P(_Refusal), e]),
            "tr_cert_hypothesis": (vp, [vp, P(_Claim), P(_Sample), P(_Dim), i32, P(_Evidence), P(_Budget),
                                        P(_Refusal), e]),
            "tr_cert_adopt": (vp, [vp, P(_Claim), P(_Evidence), i32, P(_Dim), i32, P(_Budget), P(_Refusal), e]),
            "tr_cert_reverify": (vp, [vp, vp, P(_Dim), i32, P(_Evidence), P(_Budget), P(_Refusal), e]),
            "tr_cert_derive": (vp, [P(_Claim), P(vp), P(_Claim), i32, P(_Budget), P(_Refusal), e]),
            "tr_cert_fresh": (i32, [vp, vp, P(_Dim), i32, P(_Staleness)]),
            "tr_cert_authorize_exact": (i32, [vp, u64, vp, P(_Dim), i32, P(_Decision)]),
            "tr_cert_get_info": (i32, [vp, P(_Info)]),
            "tr_cert_get_claim": (i32, [vp, P(_Claim)]),
            "tr_cert_get_record": (vp, [vp]),
            "tr_cert_get_premise": (vp, [vp, i32]),
            "tr_cert_clone": (vp, [vp, e]),
            "tr_cert_equal": (i32, [vp, vp]),
            "tr_cert_serial_size": (u64, [vp]),
            "tr_cert_serialize": (i32, [vp, vp, u64, P(u64)]),
            "tr_cert_deserialize": (vp, [vp, u64, P(u64), P(_Budget), P(_Refusal), e]),
            "tr_cert_free": (None, [vp]),
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
    math_gpu.bind("tr_mgpu_cert_verify", C.c_int, [P(_mbk._View), P(_Claim), C.c_int32, P(_Budget), P(_Evidence),
                                                  P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _i64(value, what):
    v = int(value)
    if not _I64[0] <= v <= _I64[1]:
        raise OverflowError(f"{what} must fit int64")
    return v


def _identity(table):
    """the table wrapper's lifetime token (the one FND-004 scopes use)"""
    token = table.__dict__.get("_math_scope_identity")
    if token is None:
        token = table.__dict__.setdefault("_math_scope_identity", uuid.uuid4().bytes)
    return token


# ------------------------------------------------------------------ claims
class Claim:
    """A typed claim over named columns (scope column names; for a law, variable names). Built by the helpers
    (affine, sum_of, product, parity, refinement, coverage, polynomial, unique, not_null, reference)."""

    def __init__(self, kind, columns, *, nulls="forbid", tolerance=0, modulus=None, field=0, **params):
        if kind not in CLAIMS:
            raise ValueError(f"unknown claim kind {kind!r} (one of {', '.join(CLAIMS)})")
        if nulls not in POLICIES:
            raise ValueError(f"nulls is one of {', '.join(POLICIES)}")
        self.kind, self.columns, self.nulls = kind, tuple(columns), nulls
        self.tolerance, self.modulus, self.field = int(tolerance), modulus, int(field)
        if self.tolerance < 0:
            raise ValueError("tolerance is >= 0")
        self.params = params
        self.dimension = params.pop("dimension", None)      # REFERENCE: (Table, key) reached through

    def __repr__(self):
        extra = "".join(f", {k}={v!r}" for k, v in self.params.items() if k in ("scale", "bias", "constant"))
        return f"Claim({self.kind}, {list(self.columns)}{extra}, nulls={self.nulls})"

    def native(self, names=None):
        """(the tr_cert_claim, the arrays it points into); names: the scope's column names (None: a law, whose
        columns are variables 0..n-1)"""
        c = _Claim()
        c.kind, c.policy = CLAIMS[self.kind], POLICIES[self.nulls]
        cols = []
        for name in self.columns:
            if names is None:
                cols.append(len(cols))
            else:
                if name not in names:
                    raise ValueError(f"{name!r} is not a column of the scope ({', '.join(names)})")
                cols.append(list(names).index(name))
        if not 1 <= len(cols) <= 8:
            raise ValueError("a claim reads 1..8 columns")
        c.ncols = len(cols)
        for i in range(8):
            c.col[i] = cols[i] if i < len(cols) else -1
        c.carrier = CARRIER_GFP if (self.modulus is not None or self.field) else CARRIER_I128
        c.field = int(self.field)
        if self.modulus is not None:
            c.gf.p = int(self.modulus)
        elif self.field:
            c.gf.p = int(self.field)
        c.tolerance = self.tolerance
        keep = []
        p = self.params
        if self.kind == "affine":
            c.scale, c.bias = _i64(p.get("scale", 1), "scale"), _i64(p.get("bias", 0), "bias")
        elif self.kind == "parity":
            bits = p["bits"]
            arr = (_Bit * len(bits))()
            for j, (col, kind, bit, base) in enumerate(bits):
                arr[j] = _Bit(col, kind, bit, 0, _i64(base, "base"))
            c.nbits, c.bits, c.mask, c.constant = len(bits), C.cast(arr, C.c_void_p), int(p["mask"]), int(p["constant"])
            keep.append(arr)
        elif self.kind == "polynomial":
            exps, coefs = p["exponents"], p["coefficients"]
            c.nterms = len(coefs)
            if len(coefs):
                c.exponents, c.coefficients = exps.ctypes.data, coefs.ctypes.data
            keep += [exps, coefs]
        elif self.kind in ("refinement", "coverage"):
            keys = p["keys"]
            c.nvalues = len(keys)
            if len(keys):
                c.keys = keys.ctypes.data
            keep.append(keys)
            if self.kind == "refinement":
                targets = p["targets"]
                if len(keys):
                    c.targets = targets.ctypes.data
                keep.append(targets)
        elif self.kind == "reference":
            used = p["used"]
            c.nvalues = int(p["rows"])
            if c.nvalues:
                c.used = used.ctypes.data
            keep.append(used)
        return c, keep


def affine(out, x, scale=1, bias=0, *, nulls="forbid", tolerance=0, modulus=None):
    """out = scale * x + bias (DISC-001); tolerance > 0: |residual| <= tolerance (approximate); modulus: checked mod p"""
    return Claim("affine", [out, x], nulls=nulls, tolerance=tolerance, modulus=modulus, scale=scale, bias=bias)


def sum_of(out, x, y, *, nulls="forbid", tolerance=0, modulus=None):
    """out = x + y (DISC-001)"""
    return Claim("sum", [out, x, y], nulls=nulls, tolerance=tolerance, modulus=modulus)


def product(out, x, y, *, nulls="forbid", tolerance=0, modulus=None):
    """out = x * y (DISC-001)"""
    return Claim("product", [out, x, y], nulls=nulls, tolerance=tolerance, modulus=modulus)


def parity(variables, constant=0, *, nulls="forbid", offsets=None):
    """XOR of bit variables = constant (FIELD-001). A variable: a column name (logical != 0), (name, "nonzero"),
    (name, "bit", b[, base]) (bit b of (logical - base) mod 2^64), or (name, "code", b) (bit plane b of the stored
    code: base = the codec offset, from offsets={name: offset} or the scope at use)."""
    names, bits = [], []
    pending = []
    for v in variables:
        spec = (v, "nonzero") if isinstance(v, str) else tuple(v)
        name, kind = spec[0], spec[1]
        if name not in names:
            names.append(name)
        col = names.index(name)
        if kind == "nonzero":
            bits.append([col, 1, 0, 0])
        elif kind == "bit":
            b = int(spec[2])
            bits.append([col, 2, b, int(spec[3]) if len(spec) > 3 else 0])
        elif kind == "code":
            bits.append([col, 2, int(spec[2]), None])
            pending.append((len(bits) - 1, name))
        else:
            raise ValueError(f"bit variable {v!r}: kind is nonzero, bit or code")
        if kind != "nonzero" and not 0 <= bits[-1][2] <= 63:
            raise ValueError("bit numbers are 0..63")
    if len(bits) > 64:
        raise UnsupportedError("a parity claim has at most 64 bit variables")
    claim = Claim("parity", names, nulls=nulls, bits=bits, mask=(1 << len(bits)) - 1, constant=int(constant))
    claim._code_bits = pending
    if offsets is not None:
        claim.resolve_offsets(offsets)
    return claim


def _resolve_code_bits(claim, scope):
    if claim.kind != "parity" or not getattr(claim, "_code_bits", None):
        return claim
    claim.resolve_offsets({name: scope.column(name)["offset"] for _, name in claim._code_bits})
    return claim


def _claim_resolve(self, offsets):
    for j, name in self._code_bits:
        self.params["bits"][j][3] = int(offsets[name])
    self._code_bits = []
    return self


Claim.resolve_offsets = _claim_resolve


def refinement(coarse, fine, mapping, *, nulls="forbid"):
    """coarse = mapping[fine] on every selected row: a functional dependency with its map ({fine value: coarse
    value}, logical integers; a fine value outside the map violates) (MULTI-001)"""
    items = sorted((_i64(k, "key"), _i64(v, "value")) for k, v in dict(mapping).items())
    keys = np.ascontiguousarray([k for k, _ in items], dtype=np.int64)
    targets = np.ascontiguousarray([v for _, v in items], dtype=np.int64)
    return Claim("refinement", [coarse, fine], nulls=nulls, keys=keys, targets=targets)


def coverage(category, required):
    """every present group of the scope holds every required category (logical values of the column; NULL
    categories never count; at most 64) (QUANT-002)"""
    keys = np.ascontiguousarray(sorted({_i64(v, "category") for v in required}), dtype=np.int64)
    if len(keys) > 64:
        raise UnsupportedError("a coverage claim has at most 64 required categories")
    return Claim("coverage", [category], nulls="skip", keys=keys)


def polynomial(terms, columns, *, nulls="forbid", tolerance=0, modulus=None, field=0):
    """sum of coefficient * prod column^exponent = 0 (DISC-002): terms {exponent tuple: coefficient} or
    [(coefficient, {column: exponent})]; field p > 0: the claim holds in GF(p) (checked mod p)"""
    columns = list(columns)
    rows, coefs = [], []
    items = terms.items() if isinstance(terms, dict) else [(None, t) for t in terms]
    for key, value in items:
        if key is None:
            coef, powers = value
            e = [0] * len(columns)
            for name, k in dict(powers).items():
                e[columns.index(name)] += int(k)
        else:
            e, coef = [int(k) for k in key], value
            if len(e) != len(columns):
                raise ValueError("one exponent per column")
        if any(not 0 <= k <= 255 for k in e):
            raise ValueError("exponents are 0..255")
        rows.append(e)
        coefs.append(_i64(coef, "coefficient"))
    exps = np.ascontiguousarray(np.array(rows, dtype=np.uint8).reshape(len(rows), len(columns)))
    co = np.ascontiguousarray(coefs, dtype=np.int64)
    if field and modulus is None:
        modulus = field
    return Claim("polynomial", columns, nulls=nulls, tolerance=tolerance, modulus=modulus, field=field,
                 exponents=exps, coefficients=co)


def unique(column, *, nulls="forbid"):
    """no two selected present rows share a value (a key; nulls="forbid": a NULL violates, as a PRIMARY KEY)"""
    return Claim("unique", [column], nulls=nulls)


def not_null(column):
    """no selected row is NULL"""
    return Claim("not_null", [column], nulls="forbid")


def reference(column, dimension, *, nulls="forbid", key=None):
    """the foreign-key code (a dimension row id: FND-004's convention) names a used row of the dimension Table"""
    used = np.ascontiguousarray(np.array(dimension.used_bm, copy=True), dtype=np.uint64)
    n = int(dimension.n)
    words = (n + 63) // 64
    used = np.ascontiguousarray(used[:max(words, 1)])
    if n % 64 and words:
        used[words - 1] &= np.uint64((1 << (n % 64)) - 1)
    key = key if key is not None else getattr(dimension, "pk", None)
    return Claim("reference", [column], nulls=nulls, used=used, rows=n,
                 dimension=(dimension, key) if key is not None else None)


# ------------------------------------------------------------------ dimensions
class Dimension:
    """A dimension reached through, described natively now: revision, rows, used rows, whether the key is unique
    and NULL-free among the used rows (the join multiplicity)."""

    def __init__(self, table, key):
        if key not in table.cols:
            raise ValueError(f"{table.name} has no column {key!r}")
        self.table, self.key = table, key
        valid = getattr(table, "nullable", {}).get(key)
        raw, refusal = _Dim(), _Refusal()
        token = (C.c_uint8 * 16)(*_identity(table))
        used = np.ascontiguousarray(table.used_bm, dtype=np.uint64)
        _check(_call("tr_cert_dim_describe", table.at, token, used.ctypes.data, table.idx(key),
                     table.idx(valid) if valid else -1, C.byref(_budget(None)), C.byref(refusal), C.byref(raw)),
               refusal, "dimension")
        self.raw = raw

    @property
    def info(self):
        r = self.raw
        return dict(table=self.table.name, key=self.key, revision=int(r.revision), rows=int(r.rows),
                    used=int(r.used), duplicates=int(r.duplicates), nulls=int(r.nulls), unique=bool(r.unique))

    def __repr__(self):
        return f"Dimension({self.table.name}.{self.key}, unique={bool(self.raw.unique)})"


def _dims(scope, dims, claim=None):
    """the native dimension records: the scope's lookups (keyed by their primary key), the claim's reference and
    the explicit ones ((Table, key) pairs or Dimension objects)"""
    found = []
    seen = set()

    def add(table, key):
        if key is None or (id(table), key) in seen:
            return
        seen.add((id(table), key))
        found.append(Dimension(table, key))
    for d in getattr(scope, "_dims", ()) or ():
        add(d[0], getattr(d[0], "pk", None))
    if claim is not None and getattr(claim, "dimension", None):
        add(*claim.dimension)
    for d in dims or ():
        if isinstance(d, Dimension):
            if (id(d.table), d.key) not in seen:
                seen.add((id(d.table), d.key))
                found.append(d)
        else:
            add(*d)
    if len(found) > 8:
        raise UnsupportedError("a certificate records at most 8 dimensions")
    arr = (_Dim * max(len(found), 1))()
    for i, d in enumerate(found):
        arr[i] = d.raw
    return arr, len(found)


def code_scope(table, columns, *, rows=None, budget=None):
    """a FND-004 scope over the stored codes of columns (offset 0, scale 0): for foreign-key columns, whose codec has
    no logical value (codes are dimension row ids) and plain uint32 columns"""
    nullable = getattr(table, "nullable", {})
    specs = (_mbk._Column * max(len(columns), 1))()
    for i, name in enumerate(columns):
        if name not in table.cols:
            raise ValueError(f"{table.name} has no column {name!r}")
        if name in getattr(table, "wide", {}):
            raise UnsupportedError(f"{name!r}: a self-extending column has no single stored code")
        if getattr(table, "surr", {}).get(name) is not None:
            raise UnsupportedError(f"{name!r}: references through a derived surrogate are not supported")
        specs[i] = _mbk._Column(table.idx(name), table.idx(nullable[name]) if name in nullable else -1, -1, 0, 0, 0, 0)
    bits, source = _mbk._bitmap(rows, table.n)
    if bits is None:
        bits = table.used_bm
    token = (C.c_uint8 * 16)(*_identity(table))
    error, refusal = C.c_int32(), _Refusal()
    handle = _mbk._call("tr_scope_from_table", table.at, token, bits.ctypes.data, specs, len(columns), None, 0,
                        C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    s = _mbk.Scope._new()
    s._handle = _mbk._made(handle, error, refusal, "scope")
    s.source, s.table, s.names, s.keys = "table", table, tuple(columns), ()
    s._rows, s._dims = source, ()
    return s


# ------------------------------------------------------------------ results
Freshness = namedtuple("Freshness", "fresh reasons first column dimension recorded current")
Authorization = namedtuple("Authorization", "kind claim field nulls")


def _reasons(mask):
    return tuple(n for i, n in enumerate(STALE_REASONS) if mask >> i & 1)


def _freshness(s):
    first = _reasons(s.first)[0] if s.first else None
    return Freshness(not s.reasons, _reasons(s.reasons), first, int(s.column) if s.column >= 0 else None,
                     int(s.dim) if s.dim >= 0 else None, int(s.recorded), int(s.current))


def _evidence(e, backend="cpu", device="native CPU"):
    return dict(rows=int(e.rows), selected=int(e.selected), checked=int(e.checked), excluded=int(e.excluded),
                violations=int(e.violations), witness=None if e.witness == NONE else int(e.witness),
                overflow=int(e.overflow), unit=UNITS.get(e.unit, "rows"), holds=bool(e.holds), backend=backend,
                device=device)


def _made(handle, error, refusal, what, evidence=None):
    code = error.value
    if code == -6 and evidence is not None:
        ev = _evidence(evidence)
        err = ClaimViolated(f"{what}: the claim does not hold ({ev['violations']} violating {ev['unit']}, smallest "
                            f"{ev['witness']})")
        err.evidence = ev
        raise err
    if code == -3 and evidence is not None and evidence.overflow:
        err = OverflowError(f"{what}: {int(evidence.overflow)} rows leave the exact 128-bit evaluation")
        err.evidence = _evidence(evidence)
        raise err
    _check(code, refusal, what)
    if not handle:
        raise MemoryError(f"{what}: native allocation failed")
    return handle


class Certificate:
    """An immutable native certificate (tr_cert). kind, claim, info, evidence, source(); fresh(scope),
    authorize_exact(scope), decide(scope); to_bytes() / Certificate.from_bytes()."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use math_certificates.snapshot / hypothesis / schema / law / derive / Certificate.from_bytes")

    @classmethod
    def _wrap(cls, handle, names=None, dims=(), backend="cpu", device="native CPU", owned=True):
        self = object.__new__(cls)
        self._handle, self._owned = handle, owned
        self.names, self.backend, self.device = tuple(names) if names is not None else None, backend, device
        self._dim_specs = tuple(dims)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the certificate is closed")
        return self._handle

    def _info(self):
        out = _Info()
        _check(_call("tr_cert_get_info", self._open(), C.byref(out)), what="certificate")
        return out

    @property
    def kind(self):
        return _KIND_NAMES[self._info().kind]

    @property
    def info(self):
        i = self._info()
        return dict(kind=_KIND_NAMES[i.kind], claim=_NAMES[i.claim], law=_LAW_NAMES.get(i.law),
                    enforcement={1: "load", 2: "write"}.get(i.enforcement), nulls=_POLICY_NAMES[i.policy],
                    carrier={CARRIER_I128: "i128", CARRIER_GFP: "gfp"}[i.carrier], exact=bool(i.exact),
                    field=int(i.field), modulus=int(i.modulus) or None, tolerance=int(i.tolerance),
                    scoped=bool(i.scoped), premises=int(i.npremises), depth=int(i.depth), bytes=int(i.bytes),
                    verified_by=("cpu", "cuda library")[i.backend], digest=(int(i.digest[0]), int(i.digest[1])),
                    backend=self.backend, device=self.device)

    @property
    def evidence(self):
        return _evidence(self._info().evidence, self.backend, self.device)

    @property
    def sample(self):
        s = self._info().sample
        if not s.method:
            return None
        return dict(method={v: k for k, v in SAMPLERS.items()}[s.method], size=int(s.size), seed=int(s.seed),
                    population=int(s.population), distinct=int(s.distinct))

    @property
    def claim(self):
        """the stored claim: kind, columns (scope column indices; names when known), its parameters"""
        c = _Claim()
        _check(_call("tr_cert_get_claim", self._open(), C.byref(c)), what="certificate claim")
        cols = [int(c.col[i]) for i in range(c.ncols)]
        out = dict(kind=_NAMES[c.kind], nulls=_POLICY_NAMES[c.policy], columns=cols, field=int(c.field),
                   tolerance=int(c.tolerance), modulus=int(c.gf.p) or None)
        if self.names is not None and all(0 <= k < len(self.names) for k in cols):
            out["names"] = [self.names[k] for k in cols]
        if c.kind == CLAIMS["affine"]:
            out.update(scale=int(c.scale), bias=int(c.bias))
        if c.kind == CLAIMS["parity"]:
            bits = (_Bit * c.nbits).from_address(c.bits)
            out.update(constant=int(c.constant), mask=int(c.mask),
                       bits=[(int(b.col), {1: "nonzero", 2: "plane"}[b.kind], int(b.bit), int(b.base)) for b in bits])
        if c.kind == CLAIMS["polynomial"]:
            n, m = int(c.nterms), int(c.ncols)
            exps = np.ctypeslib.as_array((C.c_uint8 * (n * m)).from_address(c.exponents)).reshape(n, m) if n else []
            coefs = np.ctypeslib.as_array((C.c_int64 * n).from_address(c.coefficients)) if n else []
            out["terms"] = {tuple(int(k) for k in exps[t]): int(coefs[t]) for t in range(n)}
        if c.kind in (CLAIMS["refinement"], CLAIMS["coverage"]):
            n = int(c.nvalues)
            keys = list(np.ctypeslib.as_array((C.c_int64 * n).from_address(c.keys))) if n else []
            if c.kind == CLAIMS["refinement"]:
                targets = list(np.ctypeslib.as_array((C.c_int64 * n).from_address(c.targets))) if n else []
                out["mapping"] = {int(k): int(v) for k, v in zip(keys, targets)}
            else:
                out["required"] = [int(k) for k in keys]
        if c.kind == CLAIMS["reference"]:
            out["rows"] = int(c.nvalues)
        return out

    def _native_claim(self):
        c = _Claim()
        _check(_call("tr_cert_get_claim", self._open(), C.byref(c)), what="certificate claim")
        return c

    def source(self):
        """the recorded scope (None for a law or a derived certificate): identity, revision, rows, selection, the
        claim columns' encodings and NULL domains, the dimensions"""
        rec = _call("tr_cert_get_record", self._open())
        if not rec:
            return None
        return _source(rec)

    @property
    def premises(self):
        out = []
        for i in range(self._info().npremises):
            h = _call("tr_cert_get_premise", self._open(), i)
            out.append(Certificate._wrap(h, self.names, owned=False) if h else None)
        for p in out:
            p._parent = self                      # borrowed: keep the owner alive
        return out

    def fresh(self, scope=None, dims=None):
        """Freshness against the current scope (FND-004 Scope; None only for a law) and dimensions (default: the
        scope's lookups and the dimensions recorded at creation, described now)"""
        st = _Staleness()
        if scope is None:
            if self._info().kind != KINDS["law"]:
                raise ValueError("give the current scope")
            return Freshness(True, (), None, None, None, 0, 0)
        arr, n = _dims(scope, list(dims or ()) + list(self._dim_specs))
        scope._open()
        rc = _call("tr_cert_fresh", self._open(), scope._handle, arr, n, C.byref(st))
        if rc not in (0, -13):
            _check(rc, what="certificate freshness")
        return _freshness(st)

    def decide(self, scope=None, dims=None, field=0):
        """the native decision as a dict (never raises for a refusal)"""
        d = _Decision()
        arr, n = _dims(scope, list(dims or ()) + list(self._dim_specs)) if scope is not None else ((_Dim * 1)(), 0)
        rc = _call("tr_cert_authorize_exact", self._open(), int(field), scope._handle if scope is not None else None,
                   arr, n, C.byref(d))
        if rc == -1:
            _check(rc, what="authorization")
        return dict(authorized=bool(d.authorized), reason=REFUSALS[d.reason], kind=_KIND_NAMES[d.kind],
                    claim=_NAMES[d.claim], nulls=_POLICY_NAMES[d.policy], field=int(d.field),
                    premise=int(d.premise) if d.premise >= 0 else None,
                    stale=_freshness(d.stale) if d.stale.reasons else None, code=int(rc))

    def authorize_exact(self, scope=None, dims=None, field=0):
        """Authorization (the rewrite holds exactly on the scope; nulls="skip": on its complete rows only), or
        NotAuthorizedError / StaleCertificateError with the exact reason"""
        d = self.decide(scope, dims, field)
        if d["authorized"]:
            return Authorization(d["kind"], d["claim"], d["field"], d["nulls"])
        if d["reason"] in ("stale",) or (d["reason"] == "premise" and d["stale"] is not None):
            f = d["stale"]
            err = StaleCertificateError(f"the certificate's scope changed: {', '.join(f.reasons)}")
            err.reasons, err.first, err.freshness, err.decision = f.reasons, f.first, f, d
            raise err
        err = NotAuthorizedError(f"a {d['kind']} certificate cannot authorize an exact rewrite: {d['reason']}")
        err.reason, err.decision = d["reason"], d
        raise err

    def reverify(self, scope, dims=None, backend="cpu", budget=None):
        """the same claim re-verified on every selected row of the current scope (a new certificate)"""
        return reverify([self], scope, dims=dims, backend=backend, budget=budget)[0]

    def to_bytes(self):
        """the canonical serialization (no address; FND-011 wraps it into a checksummed record)"""
        n = int(_call("tr_cert_serial_size", self._open()))
        buf = (C.c_uint8 * max(n, 1))()
        written = C.c_uint64()
        _check(_call("tr_cert_serialize", self._handle, buf, n, C.byref(written)), what="serialize")
        return bytes(buf[:written.value])

    @classmethod
    def from_bytes(cls, data, names=None, budget=None):
        data = bytes(data)
        buf = (C.c_uint8 * max(len(data), 1)).from_buffer_copy(data or b"\0")
        error, refusal, used = C.c_int32(), _Refusal(), C.c_uint64()
        h = _call("tr_cert_deserialize", buf, len(data), C.byref(used), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        return cls._wrap(_made(h, error, refusal, "certificate bytes"), names)

    def equal(self, other):
        return bool(_check(_call("tr_cert_equal", self._open(), other._open()), what="certificate comparison"))

    def __eq__(self, other):
        return isinstance(other, Certificate) and self.equal(other)

    __hash__ = None

    def close(self):
        if getattr(self, "_handle", None):
            if self._owned:
                _call("tr_cert_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if not self._handle:
            return "Certificate(closed)"
        i = self.info
        return f"Certificate({i['kind']}, {i['claim']}, nulls={i['nulls']}, exact={i['exact']})"


def _source(rec):
    s = _Source()
    _check(_call("tr_cert_record_get", rec, C.byref(s)), what="record")
    cols = [dict(index=int(c.index), col=int(c.col), valid=int(c.valid), hi=int(c.hi), scale=int(c.scale),
                 offset=int(c.offset), nulls=int(c.nulls)) for c in s.col[:s.ncols]]
    dims = [dict(identity=bytes(d.identity).hex(), revision=int(d.revision), rows=int(d.rows), used=int(d.used),
                 key=int(d.key), unique=bool(d.unique), duplicates=int(d.duplicates), nulls=int(d.nulls))
            for d in s.dim[:s.ndims]]
    return dict(source={1: "table", 2: "arrays"}[s.source], identity=bytes(s.identity).hex(), revision=int(s.revision),
                rows=int(s.rows), selected=int(s.selected), groups=int(s.ngroups),
                selection_digest=(int(s.selection[0]), int(s.selection[1])), exact_selection=bool(s.flags & 1),
                grouping_bound=bool(s.flags & 2), columns=cols, dimensions=dims)


def _scope_claim(scope, claim):
    if isinstance(claim, Certificate):
        return claim._native_claim(), []
    _resolve_code_bits(claim, scope)
    return claim.native(scope.names)


def _require_fresh(scope, what):
    if getattr(scope, "source", None) == "table":
        scope.require_fresh(what)


def snapshot(scope, claim, *, dims=None, backend="cpu", budget=None):
    """an exhaustive snapshot certificate of claim over every selected row of scope, or ClaimViolated (evidence,
    witness) / OverflowError (rows undecided in 128 bits). backend "cuda": the CUDA kernel verifies, the CPU library
    adopts its evidence after consistency checks."""
    _require_fresh(scope, "snapshot")
    c, keep = _scope_claim(scope, claim)
    arr, n = _dims(scope, dims, claim)
    error, refusal, ev = C.c_int32(), _Refusal(), _Evidence()
    if backend == "cpu":
        h = _call("tr_cert_snapshot", scope._handle, C.byref(c), arr, n, C.byref(ev), C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        return Certificate._wrap(_made(h, error, refusal, "snapshot", ev), scope.names, _dim_specs(claim))
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    evs, label, device = _device_verify(scope, [claim], budget)
    ev = evs[0]
    if not ev.holds:
        error.value = -3 if ev.overflow else -6
        _made(None, error, refusal, "snapshot on the device", ev)
    checked, ckeep = _checked_claim(scope, claim)
    h = _call("tr_cert_adopt", scope._handle, C.byref(checked), C.byref(ev), 1, arr, n, C.byref(_budget(budget)),
              C.byref(refusal), C.byref(error))
    return Certificate._wrap(_made(h, error, refusal, "adopted snapshot"), scope.names, _dim_specs(claim), label, device)


def _dim_specs(claim):
    d = getattr(claim, "dimension", None) if not isinstance(claim, Certificate) else None
    return (d,) if d else ()


def hypothesis(scope, claim, size, *, seed=0, method="uniform", dims=None, budget=None):
    """a statistical hypothesis: claim checked on `size` sampled selected rows (method uniform (with replacement,
    seeded), stride or prefix); ClaimViolated when the sample refutes it. Never authorizes an exact rewrite."""
    _require_fresh(scope, "hypothesis")
    if method not in SAMPLERS:
        raise ValueError(f"method is one of {', '.join(SAMPLERS)}")
    c, keep = _scope_claim(scope, claim)
    arr, n = _dims(scope, dims, claim)
    sample = _Sample(SAMPLERS[method], 0, int(size), int(seed) & NONE, 0, 0)
    error, refusal, ev = C.c_int32(), _Refusal(), _Evidence()
    h = _call("tr_cert_hypothesis", scope._handle, C.byref(c), C.byref(sample), arr, n, C.byref(ev),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Certificate._wrap(_made(h, error, refusal, "hypothesis", ev), scope.names, _dim_specs(claim))


def schema(scope, claim, *, enforcement="load", dims=None, budget=None):
    """an enforced schema fact (unique, not_null or reference) verified on every selected row (give the table's used
    rows); enforcement "load" (verified at load, valid for this revision and every sub-selection) or "write" (the
    storage checks every write)"""
    _require_fresh(scope, "schema fact")
    if enforcement not in ENFORCEMENT:
        raise ValueError("enforcement is 'load' or 'write'")
    c, keep = _scope_claim(scope, claim)
    arr, n = _dims(scope, dims, claim)
    error, refusal, ev = C.c_int32(), _Refusal(), _Evidence()
    h = _call("tr_cert_schema", scope._handle, C.byref(c), ENFORCEMENT[enforcement], arr, n, C.byref(ev),
              C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    return Certificate._wrap(_made(h, error, refusal, "schema fact", ev), scope.names, _dim_specs(claim))


def law(claim, law="ring_identity", *, budget=None):
    """a universal law (a polynomial claim over variables): "ring_identity" (canonical form zero in Z[x]) or
    "gfp_function" (vanishes on GF(p)^n: field=p); DomainError when the claim is not an instance"""
    if law not in LAWS:
        raise ValueError(f"law is one of {', '.join(LAWS)}")
    c, keep = claim.native(None)
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_cert_law", LAWS[law], C.byref(c), C.byref(_budget(budget)), C.byref(refusal), C.byref(error))
    if error.value == -6:
        raise DomainError(f"the claim is not an instance of the law {law!r}")
    return Certificate._wrap(_made(h, error, refusal, "law"), claim.columns)


def derive(claim, premises, cofactors, scope, *, budget=None):
    """a polynomial claim derived from premise certificates with cofactor polynomials (claim - sum cofactor_i *
    premise_i = 0 in Z[x], verified natively): kind = the weakest premise's"""
    c, keep = claim.native(scope.names)
    if len(premises) != len(cofactors):
        raise ValueError("one cofactor per premise")
    ptrs = (C.c_void_p * max(len(premises), 1))(*[p._open() for p in premises])
    cof = (_Claim * max(len(cofactors), 1))()
    for i, q in enumerate(cofactors):
        raw, k = q.native(scope.names)
        cof[i] = raw
        keep += k
    error, refusal = C.c_int32(), _Refusal()
    h = _call("tr_cert_derive", C.byref(c), ptrs, cof, len(premises), C.byref(_budget(budget)), C.byref(refusal),
              C.byref(error))
    if error.value == -6:
        raise DomainError("the claim is not the combination of the premises with these cofactors")
    return Certificate._wrap(_made(h, error, refusal, "derivation"), scope.names)


def _checked_claim(scope, claim):
    """a claim checked (and canonical) as the kernel reads it: from a certificate, or built and checked natively"""
    if isinstance(claim, Certificate):
        return claim._native_claim(), [claim]
    c, keep = _scope_claim(scope, claim)
    if c.kind == CLAIMS["polynomial"] and c.nterms:
        e = np.zeros(c.nterms * c.ncols, np.uint8)
        k = np.zeros(c.nterms, np.int64)
        m = C.c_uint64()
        _check(_call("tr_cert_poly_canonical", c.ncols, c.nterms, c.exponents, c.coefficients,
                     int(c.field) if c.carrier == CARRIER_GFP else 0, 0, e.ctypes.data, k.ctypes.data, C.byref(m)),
               what="polynomial")
        c.exponents, c.coefficients, c.nterms = e.ctypes.data, k.ctypes.data, m.value
        keep += [e, k]
    need = _Need()
    _check(_call("tr_cert_claim_check", C.byref(c), len(scope.names), C.byref(need)), what="claim")
    return c, keep


def _device_verify(scope, claims, budget):
    gpu = _gpu()
    view = scope.view()
    raw = (_Claim * max(len(claims), 1))()
    keep = []
    for i, claim in enumerate(claims):
        c, k = _checked_claim(scope, claim)
        raw[i] = c
        keep += k
    out = (_Evidence * max(len(claims), 1))()
    refusal = _Refusal()
    gpu.run("tr_mgpu_cert_verify", C.byref(view), raw, len(claims), C.byref(_budget(budget)), out, C.byref(refusal),
            refusal=refusal, what="certificate verification on the device")
    info = gpu.device()
    return list(out[:len(claims)]), ("cuda-emulated" if info["emulated"] else "cuda"), info["name"]


def verify(scope, claims, *, backend="cpu", budget=None):
    """exhaustive evidence of each claim (or certificate's claim) over every selected row of scope: [dict]"""
    _require_fresh(scope, "verification")
    claims = list(claims)
    if backend == "cuda":
        evs, label, device = _device_verify(scope, claims, budget)
        return [_evidence(e, label, device) for e in evs]
    if backend != "cpu":
        raise ValueError("backend is 'cpu' or 'cuda'")
    raw = (_Claim * max(len(claims), 1))()
    keep = []
    for i, claim in enumerate(claims):
        c, k = _scope_claim(scope, claim)
        raw[i] = c
        keep += k
    out = (_Evidence * max(len(claims), 1))()
    refusal = _Refusal()
    _check(_call("tr_cert_verify", scope._handle, raw, len(claims), C.byref(_budget(budget)), out, C.byref(refusal)),
           refusal, "certificate verification")
    return [_evidence(e) for e in out[:len(claims)]]


def reverify(certificates, scope, *, dims=None, backend="cpu", budget=None):
    """re-certification after an invalidation: each certificate's claim verified on every selected row of the current
    scope (backend "cuda": one launch for all of them); new certificates, or ClaimViolated for the first that fails"""
    _require_fresh(scope, "re-verification")
    certificates = list(certificates)
    out = []
    if backend == "cpu":
        for cert in certificates:
            arr, n = _dims(scope, list(dims or ()) + list(cert._dim_specs))
            error, refusal, ev = C.c_int32(), _Refusal(), _Evidence()
            h = _call("tr_cert_reverify", cert._open(), scope._handle, arr, n, C.byref(ev), C.byref(_budget(budget)),
                      C.byref(refusal), C.byref(error))
            out.append(Certificate._wrap(_made(h, error, refusal, "re-verification", ev), scope.names, cert._dim_specs))
        return out
    if backend != "cuda":
        raise ValueError("backend is 'cpu' or 'cuda'")
    evs, label, device = _device_verify(scope, certificates, budget)
    for cert, ev in zip(certificates, evs):
        error, refusal = C.c_int32(), _Refusal()
        if not ev.holds:
            error.value = -3 if ev.overflow else -6
            _made(None, error, refusal, "re-verification on the device", ev)
        arr, n = _dims(scope, list(dims or ()) + list(cert._dim_specs))
        c = cert._native_claim()
        h = _call("tr_cert_adopt", scope._handle, C.byref(c), C.byref(ev), 1, arr, n, C.byref(_budget(budget)),
                  C.byref(refusal), C.byref(error))
        out.append(Certificate._wrap(_made(h, error, refusal, "adopted re-verification"), scope.names, cert._dim_specs,
                                     label, device))
    return out


def need(scope, claims, *, backend="cpu"):
    """the needs a verification checks before allocating: {resource: need}"""
    view = scope.view()
    raw = (_Claim * max(len(claims), 1))()
    keep = []
    for i, claim in enumerate(claims):
        c, k = _checked_claim(scope, claim)
        raw[i] = c
        keep += k
    out = _Need()
    _check(_call("tr_cert_verify_need", C.byref(view), raw, len(claims), 0 if backend == "cpu" else 1, C.byref(out)),
           what="needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}


def describe():
    """the native constant tables: claim kinds (columns, unit, NULL policies) and laws"""
    claims, laws = {}, {}
    for name, k in CLAIMS.items():
        d = _ClaimDesc()
        _check(_call("tr_cert_claim_describe", k, C.byref(d)), what="claim kind")
        claims[name] = dict(columns=(int(d.min_cols), int(d.max_cols)), unit=UNITS[d.unit], output=bool(d.output),
                            local=bool(d.local), nulls=[n for n, v in POLICIES.items() if d.policies >> v & 1],
                            field=int(d.field), native_name=d.name.decode())
    for name, k in LAWS.items():
        d = _LawDesc()
        _check(_call("tr_cert_law_describe", k, C.byref(d)), what="law")
        laws[name] = dict(any_ring=bool(d.any_ring), name=d.name.decode(), statement=d.statement.decode())
    return dict(claims=claims, laws=laws)
