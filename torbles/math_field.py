# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FIELD-001: packed GF(2) vectors and matrices and exact parity dependencies over table rows, natively on the CPU or
with the CUDA kernels (torbles/math_field.c, math_field_gpu.cu).

    from torbles import math_bank as mbk, math_field as mf
    A = mf.Matrix.from_bits([[1, 0, 1], [0, 1, 1]])          # packed: 64 columns per uint64 word
    A.rank(); R, pivots = A.rref(); N = A.nullspace()          # canonical RREF and nullspace basis
    A.solve([1, 0])          # {"status": "unique" | "none" | "family", "x", "basis", "certificate", "rank"}
    A @ B; A + B; A.T; A.matvec(x); H.syndromes(X, backend="cuda")   # one syndrome per row of X
    mf.parity(v, n), mf.dot(a, b, n)                           # field elements 0 / 1
    mf.weight(v, n), A.column_counts()                         # integers (popcounts), not field elements
    bits = mf.Bits.from_scope(mbk.Scope.from_table(db["t"], ["a", "b", "c", "code"]),
                              ["a", "b", "c", ("code", "code", 0), ("code", "code", 1)])
    bits.verify([mf.relation(["a", "b", "c"], 1)], backend="cuda")    # exhaustive: violations and a witness row
    bits.discover(sample=256, affine=True, backend="cuda")    # proposals from a sample, exact verification

Field laws are kept apart from integer arithmetic: sums of vectors are XORs and inner products parities (field elements
0 / 1); Hamming weights and column counts are integers; they agree only mod 2. Matrices must be canonical (no bit past
cols), else ValueError. Budgets raise AlgebraBudgetError before anything is allocated, a stale scope StaleStateError.
Nothing is computed in Python (packing bits into words is marshalling). Opt-in: `import torbles` does not load it.
"""
import ctypes as C
from numbers import Integral

import numpy as np

from . import algq
from .math_bank import Scope
from .math_budgets import _Budget, _Need, _Refusal, Budget
from .math_errors import RESOURCES, check as _check

__all__ = ["Matrix", "Bits", "relation", "parity", "weight", "dot", "bits_need", "STATUS"]

MAX_DIM, MAX_VARS = 1 << 31, 64
STATUS = {0: "none", 1: "unique", 2: "family"}
NONZERO, PLANE = 1, 2
MODES = {"sql": 0, "listwise": 1}
_NO_ROW = (1 << 64) - 1
P = C.POINTER


class _Matrix(C.Structure):
    _native_ = "tr_field_matrix"
    _fields_ = [("rows", C.c_uint64), ("cols", C.c_uint64), ("stride", C.c_uint64), ("words", C.c_void_p)]


class _BitSpec(C.Structure):
    _native_ = "tr_field_bitspec"
    _fields_ = [("col", C.c_int32), ("kind", C.c_int32), ("bit", C.c_int32), ("reserved", C.c_int32),
                ("base", C.c_int64)]


class _BitsView(C.Structure):
    _native_ = "tr_field_bits_view"
    _fields_ = [("rows", C.c_uint64), ("nw", C.c_uint64), ("selected", C.c_uint64), ("nvars", C.c_int32),
                ("reserved", C.c_int32), ("selection", C.c_void_p), ("value", C.c_void_p), ("valid", C.c_void_p)]


class _Relation(C.Structure):
    _native_ = "tr_field_relation"
    _fields_ = [("vars", C.c_uint64), ("constant", C.c_int32), ("reserved", C.c_int32)]


class _Evidence(C.Structure):
    _native_ = "tr_field_evidence"
    _fields_ = [(n, C.c_uint64) for n in ("checked", "excluded", "violations", "witness")] + [
        ("holds", C.c_int32), ("reserved", C.c_int32)]


class _DiscoverSpec(C.Structure):
    _native_ = "tr_field_discover_spec"
    _fields_ = [("sample", C.c_uint64), ("affine", C.c_int32), ("reserved", C.c_int32)]


class _Discovery(C.Structure):
    _native_ = "tr_field_discovery"
    _fields_ = [(n, C.c_uint64) for n in ("complete", "excluded", "initial_sample", "sample_rows", "rank",
                                          "row_checks")] + [
        (n, C.c_int32) for n in ("rounds", "nrelations", "nrejected", "columns")]


class _BitsInfo(C.Structure):
    _native_ = "tr_field_bits_info"
    _fields_ = [(n, C.c_uint64) for n in ("rows", "nw", "selected", "bytes")] + [
        ("scope_digest", C.c_uint64 * 2), ("nvars", C.c_int32), ("has_valid", C.c_int32)]


for _s, _n in ((_Matrix, 32), (_BitSpec, 24), (_BitsView, 56), (_Relation, 16), (_Evidence, 40),
               (_DiscoverSpec, 16), (_Discovery, 64), (_BitsInfo, 56)):
    assert C.sizeof(_s) == _n, (_s.__name__, C.sizeof(_s))

_native = None


def _api():
    global _native
    if _native is None:
        vp, i32, u64, PM = C.c_void_p, C.c_int32, C.c_uint64, P(_Matrix)
        pu, pi, e = P(C.c_uint64), P(C.c_int32), P(C.c_int32)
        sig = {
            "tr_field_check": (i32, [PM]),
            "tr_field_rank": (i32, [PM, P(_Budget), pu, P(_Refusal)]),
            "tr_field_rref": (i32, [PM, PM, vp, pu, P(_Budget), P(_Refusal)]),
            "tr_field_nullspace": (i32, [PM, PM, pu, P(_Budget), P(_Refusal)]),
            "tr_field_solve": (i32, [PM, vp, vp, PM, vp, pi, pu, P(_Budget), P(_Refusal)]),
            "tr_field_mul": (i32, [PM, PM, PM, P(_Budget), P(_Refusal)]),
            "tr_field_add": (i32, [PM, PM, PM]),
            "tr_field_transpose": (i32, [PM, PM, P(_Budget), P(_Refusal)]),
            "tr_field_matvec": (i32, [PM, vp, vp]),
            "tr_field_syndromes": (i32, [PM, PM, PM, P(_Budget), P(_Refusal)]),
            "tr_field_row_sum": (i32, [PM, vp]),
            "tr_field_column_counts": (i32, [PM, vp]),
            "tr_field_parity": (i32, [vp, u64, pi]),
            "tr_field_dot": (i32, [vp, vp, u64, pi]),
            "tr_field_weight": (i32, [vp, u64, pu]),
            "tr_field_bits_build": (vp, [vp, P(_BitSpec), i32, P(_Budget), P(_Refusal), e]),
            "tr_field_bits_need": (i32, [vp, P(_BitSpec), i32, P(_Need)]),
            "tr_field_bits_get_view": (i32, [vp, P(_BitsView)]),
            "tr_field_bits_get_info": (i32, [vp, P(_BitsInfo)]),
            "tr_field_bits_free": (None, [vp]),
            "tr_field_bits_combine": (i32, [P(_BitsView), P(_Relation), i32, vp, vp]),
            "tr_field_verify": (i32, [P(_BitsView), P(_Relation), i32, i32, P(_Budget), P(_Evidence), P(_Refusal)]),
            "tr_field_discover": (i32, [P(_BitsView), P(_DiscoverSpec), P(_Budget), P(_Relation), P(_Evidence), i32,
                                        P(_Relation), P(_Evidence), i32, P(_Discovery), P(_Refusal)]),
            "tr_field_discover_needs": (i32, [P(_BitsView), P(_DiscoverSpec), i32, P(_Need)]),
            "tr_field_propose": (i32, [P(_BitsView), P(_DiscoverSpec), P(_Budget), P(_Relation), i32, pi,
                                       P(_Discovery), P(_Refusal)]),
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
    PM = P(_Matrix)
    math_gpu.bind("tr_mgpu_field_syndromes", C.c_int, [PM, PM, PM, P(_Budget), P(_Refusal)])
    math_gpu.bind("tr_mgpu_field_verify", C.c_int, [P(_BitsView), P(_Relation), C.c_int32, C.c_int32, P(_Budget),
                                                   P(_Evidence), P(_Refusal)])
    math_gpu.bind("tr_mgpu_field_discover", C.c_int, [P(_BitsView), P(_DiscoverSpec), P(_Budget), P(_Relation),
                                                     P(_Evidence), C.c_int32, P(_Relation), P(_Evidence), C.c_int32,
                                                     P(_Discovery), P(_Refusal)])
    return math_gpu


def _budget(budget):
    return (budget or Budget()).raw


def _backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend is 'cpu' or 'cuda'")
    return backend


def _device_label(gpu):
    d = gpu.device()
    return ("cuda-emulated" if d["emulated"] else "cuda"), d["name"]


# ---------------------------------------------------------------------- vectors
def _nwords(nbits):
    return (int(nbits) + 63) // 64


def _vector(v, nbits):
    """the canonical words of a vector given as a Python int (bit j = entry j), a 0/1 sequence or uint64 words"""
    nbits = int(nbits)
    if not 0 <= nbits <= MAX_DIM:
        raise ValueError(f"a vector has 0..{MAX_DIM} entries")
    W = _nwords(nbits)
    if isinstance(v, np.ndarray) and v.dtype == np.uint64:
        words = np.ascontiguousarray(v.reshape(-1))
        if len(words) != W:
            raise ValueError(f"{nbits} entries are {W} words")
        return words
    if isinstance(v, Integral) and not isinstance(v, bool):
        v = int(v)
        if v < 0 or v >> nbits:
            raise ValueError(f"the vector {v:#x} has bits past entry {nbits - 1} (or is negative)")
        return np.frombuffer(v.to_bytes(8 * W, "little"), dtype=np.uint64).copy() if W else np.zeros(0, np.uint64)
    a = np.asarray(v)
    if a.ndim != 1 or len(a) != nbits or (a.size and not np.isin(a, (0, 1)).all()):
        raise ValueError(f"a vector of {nbits} entries 0 / 1")
    padded = np.zeros(W * 64, dtype=np.uint8)
    padded[:nbits] = a
    return np.packbits(padded, bitorder="little").view(np.uint64).copy() if W else np.zeros(0, np.uint64)


def _to_int(words):
    return int.from_bytes(np.ascontiguousarray(words, dtype=np.uint64).tobytes(), "little")


def _ptr(a):
    return a.ctypes.data if a.size else None


def parity(v, nbits):
    """the GF(2) sum of the entries: a field element 0 / 1 (= weight mod 2)"""
    w, out = _vector(v, nbits), C.c_int32()
    _check(_call("tr_field_parity", _ptr(w), int(nbits), C.byref(out)), what="parity")
    return int(out.value)


def dot(a, b, nbits):
    """the GF(2) inner product <a, b> = parity(popcount(a AND b)): a field element 0 / 1"""
    x, y, out = _vector(a, nbits), _vector(b, nbits), C.c_int32()
    _check(_call("tr_field_dot", _ptr(x), _ptr(y), int(nbits), C.byref(out)), what="inner product")
    return int(out.value)


def weight(v, nbits):
    """the Hamming weight popcount(v): an integer, not a field element"""
    w, out = _vector(v, nbits), C.c_uint64()
    _check(_call("tr_field_weight", _ptr(w), int(nbits), C.byref(out)), what="weight")
    return int(out.value)


# ---------------------------------------------------------------------- matrices
class Matrix:
    """A packed GF(2) matrix: rows x cols bits, row-major, bit j of a row in uint64 word j // 64 (bit j % 64); the
    words past cols are zero (canonical). Immutable by convention: operations return new matrices."""

    def __init__(self, rows, cols, words=None, stride=None):
        rows, cols = int(rows), int(cols)
        if not (0 <= rows <= MAX_DIM and 0 <= cols <= MAX_DIM):
            raise ValueError(f"rows and cols are 0..{MAX_DIM}")
        W = _nwords(cols)
        stride = W if stride is None else int(stride)
        if stride < W:
            raise ValueError("stride < ceil(cols / 64)")
        if words is None:
            words = np.zeros((rows, stride), dtype=np.uint64)
        else:
            words = np.ascontiguousarray(words, dtype=np.uint64)
            if words.size != rows * stride:
                raise ValueError(f"{rows} x {stride} words expected, got {words.size}")
            words = words.reshape(rows, stride)
        self.rows, self.cols, self.stride, self.words = rows, cols, stride, words

    def _raw(self):
        return _Matrix(self.rows, self.cols, self.stride, _ptr(self.words))

    @classmethod
    def from_bits(cls, bits):
        """from a 2-D array of 0 / 1 entries"""
        a = np.asarray(bits)
        if a.ndim != 2 or (a.size and not np.isin(a, (0, 1)).all()):
            raise ValueError("a 2-D array of 0 / 1 entries")
        rows, cols = a.shape
        W = _nwords(cols)
        padded = np.zeros((rows, W * 64), dtype=np.uint8)
        padded[:, :cols] = a
        words = np.packbits(padded, axis=1, bitorder="little").view(np.uint64) if W else np.zeros((rows, 0), np.uint64)
        return cls(rows, cols, words)

    @classmethod
    def from_ints(cls, rows, cols):
        """from Python ints, one per row (bit j = column j)"""
        rows = list(rows)
        return cls(len(rows), cols, np.stack([_vector(r, cols) for r in rows]) if rows else None)

    @classmethod
    def identity(cls, n):
        return cls.from_ints([1 << i for i in range(int(n))], int(n))

    def to_bits(self):
        W = _nwords(self.cols)
        if not self.rows or not W:
            return np.zeros((self.rows, self.cols), dtype=np.uint8)
        raw = np.ascontiguousarray(self.words[:, :W]).view(np.uint8)
        return np.unpackbits(raw, axis=1, bitorder="little")[:, :self.cols]

    def to_ints(self):
        W = _nwords(self.cols)
        return [_to_int(self.words[i, :W]) for i in range(self.rows)]

    def check(self):
        """None, or ValueError when a bit lies past cols (not canonical)"""
        _check(_call("tr_field_check", C.byref(self._raw())), what="GF(2) matrix (canonical: no bit past cols)")

    def __eq__(self, other):
        W = _nwords(self.cols)
        return (isinstance(other, Matrix) and (self.rows, self.cols) == (other.rows, other.cols)
                and np.array_equal(self.words[:, :W], other.words[:, :W]))

    __hash__ = None

    def __repr__(self):
        return f"Matrix({self.rows} x {self.cols} over GF(2))"

    # -------------------------------------------------------------- elimination
    def rank(self, *, budget=None):
        out, refusal = C.c_uint64(), _Refusal()
        _check(_call("tr_field_rank", C.byref(self._raw()), C.byref(_budget(budget)), C.byref(out), C.byref(refusal)),
               refusal, "GF(2) rank")
        return int(out.value)

    def rref(self, *, budget=None):
        """(R, pivots): the reduced row echelon form and the pivot column of each of its first rank rows"""
        out = Matrix(self.rows, self.cols)
        piv = np.zeros(max(min(self.rows, self.cols), 1), dtype=np.uint64)
        rank, refusal = C.c_uint64(), _Refusal()
        _check(_call("tr_field_rref", C.byref(self._raw()), C.byref(out._raw()), piv.ctypes.data, C.byref(rank),
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "GF(2) RREF")
        return out, [int(p) for p in piv[:int(rank.value)]]

    def nullspace(self, *, budget=None):
        """the canonical nullspace basis: a (cols - rank) x cols matrix, one vector per free column"""
        basis = Matrix(self.cols, self.cols)
        n, refusal = C.c_uint64(), _Refusal()
        _check(_call("tr_field_nullspace", C.byref(self._raw()), C.byref(basis._raw()), C.byref(n),
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "GF(2) nullspace")
        return Matrix(int(n.value), self.cols, basis.words[:int(n.value)])

    def solve(self, b, *, basis=True, certificate=True, budget=None):
        """A x = b: {"status": "none" | "unique" | "family", "rank", "x" (Python int, bit j = x_j; None for none),
        "basis" (the nullspace, for unique and family), "certificate" (y with y^T A = 0, y^T b = 1, for none)}"""
        bw = _vector(b, self.rows)
        x = np.zeros(max(_nwords(self.cols), 1), dtype=np.uint64)
        cert = np.zeros(max(_nwords(self.rows), 1), dtype=np.uint64)
        nb = Matrix(self.cols, self.cols) if basis else None
        status, rank, refusal = C.c_int32(), C.c_uint64(), _Refusal()
        _check(_call("tr_field_solve", C.byref(self._raw()), _ptr(bw), x.ctypes.data,
                     C.byref(nb._raw()) if nb is not None else None, cert.ctypes.data if certificate else None,
                     C.byref(status), C.byref(rank), C.byref(_budget(budget)), C.byref(refusal)), refusal,
               "GF(2) solve")
        st, r = STATUS[status.value], int(rank.value)
        out = dict(status=st, rank=r, x=None, basis=None, certificate=None)
        if st == "none":
            if certificate:
                out["certificate"] = _to_int(cert[:_nwords(self.rows)])
        else:
            out["x"] = _to_int(x[:_nwords(self.cols)])
            if nb is not None:
                out["basis"] = Matrix(self.cols - r, self.cols, nb.words[:self.cols - r])
        return out

    # -------------------------------------------------------------- products and sums
    def mul(self, other, *, budget=None):
        out, refusal = Matrix(self.rows, other.cols), _Refusal()
        _check(_call("tr_field_mul", C.byref(self._raw()), C.byref(other._raw()), C.byref(out._raw()),
                     C.byref(_budget(budget)), C.byref(refusal)), refusal, "GF(2) product")
        return out

    __matmul__ = mul

    def add(self, other):
        out = Matrix(self.rows, self.cols)
        _check(_call("tr_field_add", C.byref(self._raw()), C.byref(other._raw()), C.byref(out._raw())),
               what="GF(2) sum (same shapes)")
        return out

    __add__ = add

    def transpose(self, *, budget=None):
        out, refusal = Matrix(self.cols, self.rows), _Refusal()
        _check(_call("tr_field_transpose", C.byref(self._raw()), C.byref(out._raw()), C.byref(_budget(budget)),
                     C.byref(refusal)), refusal, "GF(2) transpose")
        return out

    @property
    def T(self):
        return self.transpose()

    def matvec(self, x):
        """A x as a Python int (bit i = row i)"""
        xw = _vector(x, self.cols)
        y = np.zeros(max(_nwords(self.rows), 1), dtype=np.uint64)
        _check(_call("tr_field_matvec", C.byref(self._raw()), _ptr(xw), y.ctypes.data), what="GF(2) matvec")
        return _to_int(y[:_nwords(self.rows)])

    def syndromes(self, vectors, *, backend="cpu", budget=None):
        """H x_j for every row x_j of `vectors` (a Matrix with this matrix's cols): a Matrix nvec x rows"""
        out, refusal = Matrix(vectors.rows, self.rows), _Refusal()
        args = (C.byref(self._raw()), C.byref(vectors._raw()), C.byref(out._raw()), C.byref(_budget(budget)),
                C.byref(refusal))
        if _backend(backend) == "cpu":
            _check(_call("tr_field_syndromes", *args), refusal, "syndromes")
            out.backend, out.device = "cpu", "native CPU"
            return out
        gpu = _gpu()
        gpu.run("tr_mgpu_field_syndromes", *args, refusal=refusal, what="syndromes on the device")
        out.backend, out.device = _device_label(gpu)
        return out

    def row_sum(self):
        """the GF(2) sum (XOR) of the rows, as a Python int"""
        out = np.zeros(max(_nwords(self.cols), 1), dtype=np.uint64)
        _check(_call("tr_field_row_sum", C.byref(self._raw()), out.ctypes.data), what="row sum")
        return _to_int(out[:_nwords(self.cols)])

    def column_counts(self):
        """the integer number of rows with each column's bit set"""
        out = np.zeros(max(self.cols, 1), dtype=np.uint64)
        _check(_call("tr_field_column_counts", C.byref(self._raw()), out.ctypes.data), what="column counts")
        return [int(c) for c in out[:self.cols]]


# ---------------------------------------------------------------------- parity dependencies over table rows
def relation(variables, constant=0):
    """XOR of the variables (indices or labels of a Bits) = constant (0 or 1)"""
    if constant not in (0, 1, True, False):
        raise ValueError("the constant of a parity relation is 0 or 1")
    return (tuple(variables), int(constant))


def _evidence(e):
    return dict(checked=int(e.checked), excluded=int(e.excluded), violations=int(e.violations),
                witness=None if e.witness == _NO_ROW else int(e.witness), holds=bool(e.holds))


def _specs(scope, specs):
    """native bit specs and labels: "name" / (name, "nonzero"); (name, "bit", b[, base]); (name, "code", b)"""
    specs = [s if isinstance(s, tuple) else (s, "nonzero") for s in specs]
    arr = (_BitSpec * max(len(specs), 1))()
    labels = []
    names = list(scope.names)
    for j, s in enumerate(specs):
        name, kind = s[0], s[1]
        if name not in names:
            raise ValueError(f"the scope has no column {name!r}")
        col = names.index(name)
        if kind == "nonzero" and len(s) == 2:
            arr[j] = _BitSpec(col, NONZERO, 0, 0, 0)
            labels.append(name)
        elif kind in ("bit", "code") and len(s) in (3, 4):
            b = s[2]
            if isinstance(b, bool) or not isinstance(b, Integral) or not 0 <= int(b) <= 63:
                raise ValueError(f"{s!r}: the bit index is 0..63")
            if kind == "code":
                if len(s) != 3:
                    raise ValueError(f"{s!r}: a code plane takes no base")
                base = scope.column(name)["offset"]
            else:
                base = int(s[3]) if len(s) == 4 else 0
            if not -(1 << 63) <= base < (1 << 63):
                raise ValueError(f"{s!r}: base is an int64 logical value")
            arr[j] = _BitSpec(col, PLANE, int(b), 0, base)
            labels.append(f"{name}.{kind}{int(b)}")
        else:
            raise ValueError(f"{s!r}: a column name, (name, 'nonzero'), (name, 'bit', b[, base]) or (name, 'code', b)")
    if len(set(labels)) != len(labels):
        raise ValueError("a bit variable is given twice")
    return arr, labels


class Bits:
    """An immutable native bit matrix over the selected rows of a scope: one bit variable per spec (a Boolean
    column: logical != 0; or a bit plane of a column), with SQL validity."""

    def __init__(self, *args, **kwargs):
        raise TypeError("use Bits.from_scope(scope, specs)")

    @classmethod
    def from_scope(cls, scope, specs, *, budget=None):
        """specs: "name" or (name, "nonzero") (logical != 0); (name, "bit", b[, base]) (bit b of logical - base);
        (name, "code", b) (bit plane b of the stored code: base = the codec offset)"""
        if not isinstance(scope, Scope):
            raise TypeError("scope: a math_bank.Scope")
        if scope.source == "table":
            scope.require_fresh("bit columns")
        arr, labels = _specs(scope, specs)
        error, refusal = C.c_int32(), _Refusal()
        h = _call("tr_field_bits_build", scope._handle, arr, len(labels), C.byref(_budget(budget)), C.byref(refusal),
                  C.byref(error))
        _check(error.value, refusal, "bit columns")
        if not h:
            raise MemoryError("bit columns: native allocation failed")
        self = object.__new__(cls)
        self._handle, self.scope, self.labels = h, scope, tuple(labels)
        return self

    def _open(self):
        if not self._handle:
            raise ValueError("the bit columns are closed")

    def _fresh(self, what):
        self._open()
        if self.scope.source == "table":
            self.scope.require_fresh(what)

    def view(self):
        self._open()
        v = _BitsView()
        _check(_call("tr_field_bits_get_view", self._handle, C.byref(v)), what="bit columns view")
        return v

    @property
    def info(self):
        self._open()
        i = _BitsInfo()
        _check(_call("tr_field_bits_get_info", self._handle, C.byref(i)), what="bit columns")
        return dict(rows=int(i.rows), selected=int(i.selected), variables=int(i.nvars), words=int(i.nw),
                    bytes=int(i.bytes), has_null=bool(i.has_valid), labels=self.labels)

    @property
    def nvars(self):
        return len(self.labels)

    def words(self, j):
        """copies of variable j's value and valid words"""
        v = self.view()
        nw = int(v.nw)
        if not 0 <= int(j) < self.nvars:
            raise ValueError("variable index out of range")
        if not nw:
            return np.zeros(0, np.uint64), np.zeros(0, np.uint64)
        value = np.ctypeslib.as_array((C.c_uint64 * nw).from_address(v.value + 8 * nw * int(j))).copy()
        src = v.valid + 8 * nw * int(j) if v.valid else v.selection
        return value, np.ctypeslib.as_array((C.c_uint64 * nw).from_address(src)).copy()

    def _relations(self, relations):
        rels = list(relations)
        arr = (_Relation * max(len(rels), 1))()
        for i, r in enumerate(rels):
            variables, constant = r if isinstance(r, tuple) and len(r) == 2 and not isinstance(r[1], str) else (r, 0)
            mask = 0
            for v in variables:
                j = self.labels.index(v) if isinstance(v, str) else int(v)
                if not 0 <= j < self.nvars:
                    raise ValueError(f"relation {r!r}: no variable {v!r}")
                if (mask >> j) & 1:
                    raise ValueError(f"relation {r!r}: variable {v!r} twice (x XOR x = 0: drop both)")
                mask |= 1 << j
            if constant not in (0, 1, True, False):
                raise ValueError(f"relation {r!r}: the constant is 0 or 1")
            arr[i] = _Relation(mask, int(constant), 0)
        return arr, len(rels)

    def _format_relation(self, r):
        return dict(vars=tuple(j for j in range(self.nvars) if (int(r.vars) >> j) & 1),
                    labels=tuple(self.labels[j] for j in range(self.nvars) if (int(r.vars) >> j) & 1),
                    constant=int(r.constant))

    def verify(self, relations, *, mode="sql", backend="cpu", budget=None):
        """exhaustive evidence of each relation (variables, constant) over the selected rows: checked, excluded
        (a NULL among its variables, mode "sql", or among all variables, mode "listwise"), violations, the smallest
        violating row (witness), holds"""
        self._fresh("parity verification")
        if mode not in MODES:
            raise ValueError("mode is 'sql' or 'listwise'")
        arr, n = self._relations(relations)
        out, refusal, view = (_Evidence * max(n, 1))(), _Refusal(), self.view()
        args = (C.byref(view), arr, n, MODES[mode], C.byref(_budget(budget)), out, C.byref(refusal))
        if _backend(backend) == "cpu":
            _check(_call("tr_field_verify", *args), refusal, "parity verification")
            label = ("cpu", "native CPU")
        else:
            gpu = _gpu()
            gpu.run("tr_mgpu_field_verify", *args, refusal=refusal, what="parity verification on the device")
            label = _device_label(gpu)
        return [dict(relation=self._format_relation(arr[i]), **_evidence(out[i]), backend=label[0], device=label[1])
                for i in range(n)]

    def combine(self, rel, *, mode="sql"):
        """the substitution of a relation: (value words of XOR of its variables XOR constant on the rows where it is
        checked, valid words of those rows)"""
        self._fresh("parity combination")
        arr, n = self._relations([rel])
        view = self.view()
        nw = int(view.nw)
        value, valid = np.zeros(max(nw, 1), np.uint64), np.zeros(max(nw, 1), np.uint64)
        _check(_call("tr_field_bits_combine", C.byref(view), arr, MODES[mode], value.ctypes.data, valid.ctypes.data),
               what="parity combination")
        return value[:nw], valid[:nw]

    def discover(self, *, sample=256, affine=True, backend="cpu", budget=None):
        """every GF(2) relation among the variables (affine: with a constant) that holds on every complete selected
        row: proposals from a sample of `sample` complete rows (0: all), exhaustive verification, failed proposals
        reported (rejected, with their witness rows) and their witnesses added until every proposal holds"""
        self._fresh("parity discovery")
        if isinstance(sample, bool) or not isinstance(sample, Integral) or sample < 0:
            raise ValueError("sample is a row count >= 0")
        spec = _DiscoverSpec(int(sample), int(bool(affine)), 0)
        view = self.view()
        cols = self.nvars + int(bool(affine))
        fcap, rcap = max(cols, 1), max(cols * (cols + 1) // 2, 1)
        found, fev = (_Relation * fcap)(), (_Evidence * fcap)()
        rej, rev = (_Relation * rcap)(), (_Evidence * rcap)()
        stats, refusal = _Discovery(), _Refusal()
        args = (C.byref(view), C.byref(spec), C.byref(_budget(budget)), found, fev, fcap, rej, rev, rcap,
                C.byref(stats), C.byref(refusal))
        if _backend(backend) == "cpu":
            _check(_call("tr_field_discover", *args), refusal, "parity discovery")
            label = ("cpu", "native CPU")
        else:
            gpu = _gpu()
            gpu.run("tr_mgpu_field_discover", *args, refusal=refusal, what="parity discovery on the device")
            label = _device_label(gpu)
        return dict(relations=[dict(self._format_relation(found[i]), **_evidence(fev[i]))
                               for i in range(int(stats.nrelations))],
                    rejected=[dict(self._format_relation(rej[i]), **_evidence(rev[i]))
                              for i in range(int(stats.nrejected))],
                    **{n: int(getattr(stats, n)) for n, _ in _Discovery._fields_ if n not in ("nrelations", "nrejected")},
                    backend=label[0], device=label[1])

    def propose(self, *, sample=256, affine=True, budget=None):
        """the first-round proposals of discover (candidate generation from the sample, not verified)"""
        self._fresh("parity proposals")
        if isinstance(sample, bool) or not isinstance(sample, Integral) or sample < 0:
            raise ValueError("sample is a row count >= 0")
        spec = _DiscoverSpec(int(sample), int(bool(affine)), 0)
        cap = max(self.nvars + int(bool(affine)), 1)
        out, n, stats, refusal = (_Relation * cap)(), C.c_int32(), _Discovery(), _Refusal()
        _check(_call("tr_field_propose", C.byref(self.view()), C.byref(spec), C.byref(_budget(budget)), out, cap,
                     C.byref(n), C.byref(stats), C.byref(refusal)), refusal, "parity proposals")
        return dict(proposals=[self._format_relation(out[i]) for i in range(int(n.value))],
                    complete=int(stats.complete), excluded=int(stats.excluded), sample_rows=int(stats.initial_sample),
                    sample_rank=int(stats.rank))

    def discover_need(self, *, sample=256, affine=True, backend="cpu"):
        spec = _DiscoverSpec(int(sample), int(bool(affine)), 0)
        out = _Need()
        _check(_call("tr_field_discover_needs", C.byref(self.view()), C.byref(spec), int(_backend(backend) == "cuda"),
                     C.byref(out)), what="discovery needs")
        return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}

    def close(self):
        if getattr(self, "_handle", None):
            _call("tr_field_bits_free", self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"Bits({len(self.labels)} variables, {self.info['selected']} rows)" if self._handle else "Bits(closed)"


def bits_need(scope, specs):
    """the needs Bits.from_scope checks: {resource: need}"""
    arr, labels = _specs(scope, specs)
    out = _Need()
    _check(_call("tr_field_bits_need", scope._handle, arr, len(labels), C.byref(out)), what="bit column needs")
    return {name: int(out.need[i]) for i, name in enumerate(RESOURCES) if out.need[i]}
