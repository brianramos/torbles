# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Executing bound SQL (sqlir) on torbles.

A query block over a star join runs as engine queries (algq), never row by row:

  * WHERE: each AND-term on one table becomes a plane predicate (typed comparisons, BETWEEN, IN, LIKE prefixes), a
    dimension row mask (any predicate on one dimension, evaluated on its distinct values) or a code set (any predicate
    on one fact column, evaluated on the codes the column can hold); the rest is residual.
  * The atoms S: the columns the block needs per group (GROUP BY columns, columns of residual predicates and of
    aggregate arguments that are not polynomials). The engine groups the passing fact rows by S (popcount algebra
    on the planes) and returns, per group, the row count and the sums of the moment polynomials: an aggregate whose
    argument is a polynomial in exact columns (SUM, AVG, VAR_*, STDDEV_*, COVAR_*, CORR, REGR_*, COUNT) is a
    function of a few exact sums (x, x^2, x y), so it never needs S to hold its columns.
  * Group-level evaluation: residual predicates, GROUP BY expressions, the other aggregates, HAVING, window
    functions, ORDER BY are evaluated on the S-groups, each standing for its count of rows.

Blocks that are not star joins (outer, cross and non-key joins, derived tables, VALUES) materialize each FROM item as
a multiset (an engine GROUP BY over the columns needed, with counts) and join those multisets."""
import copy, functools, math
from decimal import Decimal
from fractions import Fraction
import numpy as np
import algq
import sqltypes as T
from sqltypes import SqlError
import sqlir as I
import sqlfuncs as F
from sqlfuncs import Err


class Relation:
    """a query result: column names, SQL types, rows (tuples; built from the columns when first read)"""
    def __init__(self, names, types, rows, cols=None):
        self.names, self.types, self._rows, self._cols = list(names), list(types), rows, cols
    @classmethod
    def of_columns(cls, names, types, cols, n):
        return cls(names, types, None if cols else [()] * n, cols if cols else None)
    @property
    def rows(self):
        if self._rows is None: self._rows = list(zip(*self._cols))
        return self._rows
    @rows.setter
    def rows(self, v): self._rows, self._cols = v, None
    @property
    def columns(self): return self.names
    def __len__(self): return len(self._rows) if self._rows is not None else (len(self._cols[0]) if self._cols else 0)
    def __iter__(self): return iter(self.rows)
    def __repr__(self): return f"Relation({self.names}, {len(self.rows)} rows)"
    def fetchall(self): return list(self.rows)
    def fetchone(self): return self.rows[0] if self.rows else None
    def df(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.names)
    def scalar(self):
        return self.rows[0][0] if self.rows else None


class ColArr:
    """a column as the engine holds it: codes (an integer array whose order and equality are the values'), a validity
    mask (None: no NULL), and the codec that decodes codes into values (None: the codes are the values)"""
    __slots__ = ("codes", "valid", "codec")
    def __init__(self, codes, valid=None, codec=None):
        c = np.asarray(codes)
        if c.dtype != np.int64 and c.dtype != object and c.dtype != np.float64:   # int64 (Python ints past 2^63); float64: computed doubles
            c = c.astype(np.int64) if c.dtype != np.uint64 or not len(c) or int(c.max()) < 1 << 63 else c.astype(object)
        self.codes, self.valid, self.codec = c, (None if valid is None else np.asarray(valid, bool)), codec
    def __len__(self): return len(self.codes)
    def decode(self):
        c = self.codes
        vals = self.codec.decode(c) if self.codec is not None else c.tolist()
        if self.valid is not None and not self.valid.all():
            vals = [v if ok else None for v, ok in zip(vals, self.valid.tolist())]
        return vals
    def take(self, idx):
        idx = np.asarray(idx, np.int64)
        return ColArr(self.codes[idx], None if self.valid is None else self.valid[idx], self.codec)


class Cols(dict):
    """a block's columns {expression key: [value per row]}; a column kept as engine codes (arr) is decoded when it is
    first read as values; distinct: the rows are distinct combinations of the codes (_dict_eval's own rows)"""
    __slots__ = ("arr", "distinct")
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.arr = {}; self.distinct = False
    def __missing__(self, k):
        a = self.arr.get(k)
        if a is None: raise KeyError(k)
        v = a.decode(); dict.__setitem__(self, k, v)
        return v
    def __contains__(self, k): return dict.__contains__(self, k) or k in self.arr
    def get(self, k, default=None):
        if dict.__contains__(self, k): return dict.__getitem__(self, k)
        if k in self.arr: return self[k]
        return default
    def __setitem__(self, k, v):
        self.arr.pop(k, None); dict.__setitem__(self, k, v)
    def keys(self): return list(dict.keys(self)) + [k for k in self.arr if not dict.__contains__(self, k)]
    def __iter__(self): return iter(self.keys())
    def __len__(self): return len(self.keys())
    def items(self): return [(k, self[k]) for k in self.keys()]
    def values(self): return [self[k] for k in self.keys()]
    def array(self, k):
        """the column as codes (ColArr), or None when it only exists as values"""
        return self.arr.get(k)
    def put(self, k, a):
        dict.pop(self, k, None); self.arr[k] = a
    def take(self, idx):
        out = Cols()
        for k, a in self.arr.items(): out.arr[k] = a.take(idx)
        for k, v in dict.items(self):
            if k not in self.arr: dict.__setitem__(out, k, [v[i] for i in idx])
        return out


class Rel:
    """rows of a block while it is evaluated: cols {expression key: [value per row]} (Cols: engine columns stay codes
    until read as values), cnt [rows each row stands for], mom {(variant, moment): [sum per row]}, vcnt {variant: [count
    per row]}"""
    __slots__ = ("n", "cols", "cnt", "mom", "vcnt")
    def __init__(self, n):
        self.n, self.cols, self.cnt, self.mom, self.vcnt = n, Cols(), [1] * n, {}, {}
    def take(self, idx):
        r = Rel(len(idx))
        r.cols = self.cols.take(idx)
        r.cnt = [self.cnt[i] for i in idx]
        r.mom = {k: [v[i] for i in idx] for k, v in self.mom.items()}
        r.vcnt = {k: [v[i] for i in idx] for k, v in self.vcnt.items()}
        return r


class Cx:
    """execution context: the database, the engine, CTE results, the values of enclosing queries (correlation)"""
    def __init__(self, db, engine="alg"):
        self.db, self.engine = db, engine
        self.ctes, self.outers, self.memo, self.stats = {}, [], {}, dict(engine_queries=0, star_blocks=0, general_blocks=0)

    def outer_value(self, e):
        k = e.col.key()
        for env in reversed(self.outers):
            if k in env: return env[k]
        raise SqlError(f"internal: correlated column {e.col.name} has no value")


# ================================================================== expressions
def eval_const(e):
    """the value of a constant expression (in the context queries run in: exact decimals to 38 digits and past)"""
    import decimal
    r = Rel(1)
    with decimal.localcontext(T.CTX): v = ev(e, r, None)[0]
    if isinstance(v, Err): raise SqlError(v.msg)
    return v


def _apply(s, args, n):
    impl = s.impl
    if not args: return [impl()] * n if s.deterministic else [impl() for _ in range(n)]
    out = []
    prop = s.nulls == "propagate"
    for row in zip(*args):
        bad = None
        for x in row:
            if isinstance(x, Err): bad = x; break
        if bad is not None and not (s.name in ("and", "or", "coalesce", "is_null", "is_not_null")):
            out.append(bad); continue
        if prop and None in row:
            out.append(None); continue
        try:
            out.append(impl(*row))
        except SqlError as ex: out.append(Err(str(ex)))
        except (ZeroDivisionError, ValueError, OverflowError, ArithmeticError, IndexError, TypeError) as ex:
            out.append(Err(f"{s.name}: {ex}"))
    return out


def ev(e, rel, cx):
    """the values of expression e on every row of rel"""
    v = rel.cols.get(e.key())
    if v is not None: return v
    n = rel.n
    if isinstance(e, I.Lit): return [e.v] * n
    if isinstance(e, I.Func):
        s = F.SCALARS.get(e.fn)
        if s is None: raise SqlError(f"internal: no implementation of {e.fn}")
        if all(isinstance(a, I.Lit) for a in e.args) and s.deterministic and n > 1:
            one = Rel(1); return [ev(e, one, cx)[0]] * n
        if e.fn == "in" and len(e.args) > 8 and all(isinstance(a, I.Lit) for a in e.args[1:]):
            return _in_set(e, ev(e.args[0], rel, cx))
        if COLUMNAR and VEC_SCALARS and n >= 64 and e.fn in _VEC_FNS and codes_of(e, rel, cx) is not None:
            return rel.cols[e.key()]                      # computed on codes, decoded once
        if COLUMNAR and DICT_EVAL and n >= 64:
            r = _dict_eval(e, rel, cx)
            if r is not None: return r
        argv = [ev(a, rel, cx) for a in e.args]
        if VEC_SCALARS and n >= 64 and e.fn in _VEC_FNS:
            r = _vec_func(e.fn, argv, n)
            if r is not None: return r
        return _apply(s, argv, n)
    if isinstance(e, I.Case):
        if COLUMNAR and DICT_EVAL and n >= 64:
            r = _dict_eval(e, rel, cx)
            if r is not None: return r
        conds = [ev(c, rel, cx) for c, _ in e.whens]
        vals = [ev(x, rel, cx) for _, x in e.whens]
        el = ev(e.else_, rel, cx) if e.else_ is not None else [None] * n
        out = []
        for i in range(n):
            r = el[i]
            for c, vv in zip(conds, vals):
                if isinstance(c[i], Err): r = c[i]; break
                if c[i] is True: r = vv[i]; break
            out.append(r)
        return out
    if isinstance(e, I.Cast):
        if COLUMNAR and VEC_SCALARS and n >= 64 and codes_of(e, rel, cx) is not None: return rel.cols[e.key()]
        if COLUMNAR and DICT_EVAL and n >= 64:
            r = _dict_eval(e, rel, cx)
            if r is not None: return r
        a = ev(e.arg, rel, cx); out = []
        if VEC_SCALARS and n >= 64:
            r = _vec_cast(a, e.arg.t, e.t)
            if r is not None: return r
        for x in a:
            if isinstance(x, Err) or x is None: out.append(x); continue
            try: out.append(T.cast_value(x, e.arg.t, e.t, not e.try_))
            except SqlError as ex: out.append(Err(str(ex)))
        return out
    if isinstance(e, I.Outer): return [cx.outer_value(e)] * n
    if isinstance(e, I.Subq): return eval_subquery(e, rel, cx)
    if isinstance(e, I.Col):
        raise SqlError(f"internal: column {e.b.alias}.{e.name} is not available here")
    if isinstance(e, (I.Agg, I.Win, I.Grouping)):
        raise SqlError(f"internal: {I.render(e)} was not computed")
    raise SqlError(f"internal: cannot evaluate {e!r}")


# ------------------------------------------------------------------ expressions on the distinct codes
DICT_EVAL = True            # an expression of code columns is evaluated once per distinct combination of their codes


def _code_leaves(e, rel, out):
    """the code columns of rel that e reads, into out {key: ColArr}; False when a part of e is not a deterministic
    function of rel's code columns (a column held as values, an aggregate, a subquery, a non-deterministic function)"""
    k = e.key()
    if k in rel.cols:
        a = rel.cols.array(k)
        if a is None: return False
        out[k] = a; return True
    if isinstance(e, I.Lit): return True
    if isinstance(e, I.Func):
        s = F.SCALARS.get(e.fn)
        return s is not None and s.deterministic and all(_code_leaves(a, rel, out) for a in e.args)
    if isinstance(e, (I.Case, I.Cast)): return all(_code_leaves(x, rel, out) for x in e.children())
    return False


def _dict_eval(e, rel, cx):
    """e on every row of rel, evaluated once per distinct combination of the codes it reads: a scalar expression is a
    function of its inputs' values, so on a column of G distinct values it costs G evaluations and a gather, not n
    (None: e reads something else, or the rows are mostly distinct)"""
    if getattr(rel.cols, "distinct", False): return None
    leaves = {}
    if not _code_leaves(e, rel, leaves) or not leaves: return None
    import sqlwin
    dense = []
    for a in leaves.values():
        d = np.unique(a.codes, return_inverse=True)[1].reshape(-1).astype(np.int64)
        dense.append(d if a.valid is None else np.where(a.valid, d + 1, 0))
    _, first, inv = np.unique(sqlwin._combine(dense), return_index=True, return_inverse=True)
    if len(first) > rel.n // 2: return None
    small = Rel(len(first)); small.cols.distinct = True
    for k, a in leaves.items(): small.cols.put(k, a.take(first))
    vals = ev(e, small, cx)
    return [vals[i] for i in inv.reshape(-1).tolist()]


# ------------------------------------------------------------------ scalar operators as arrays
VEC_SCALARS = True          # arithmetic, comparisons and logic on int / double columns as numpy arrays
COLUMNAR = True             # the engine's GROUP BY results and row bitmaps stay codes (ColArr) until read as values
ROWS_BLOCK = True           # a block without aggregates reads its rows from the planes (False: groups them first)
_NoneT = type(None)
_VEC_FNS = {"+", "-", "*", "/", "%", "mod", "//", "divide", "neg", "=", "<>", "<", "<=", ">", ">=", "and", "or", "not", "is_null", "is_not_null"}
_CMPF = {"=": np.equal, "<>": np.not_equal, "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


def _numcol(vals):
    """(kind, array, NULL mask or None, largest magnitude) for a column of ints (not booleans) below 2^62 or of doubles
    without NaN; None for anything else (the row-by-row path)"""
    tys = set(map(type, vals))
    nul = _NoneT in tys; tys.discard(_NoneT)
    if tys == {int}:
        nn = [v for v in vals if v is not None] if nul else vals
        mx = max(max(nn), -min(nn)) if nn else 0
        if mx >= 1 << 62: return None
        arr = np.array([0 if v is None else v for v in vals] if nul else vals, np.int64)
    elif tys == {float}:
        arr = np.array([0.0 if v is None else v for v in vals] if nul else vals, np.float64)
        if np.isnan(arr).any(): return None
        mx = None
    elif not tys: return None
    else: return None
    return ("int" if tys == {int} else "float"), arr, (np.fromiter((v is None for v in vals), bool, len(vals)) if nul else None), mx


def _out(arr, nul):
    if nul is None or not nul.any(): return arr.tolist()
    o = arr.astype(object); o[nul] = None
    return o.tolist()


def _vec_func(fn, argv, n):
    """the operator on whole columns, with the row-by-row results (exact integers, IEEE doubles, three-valued logic);
    None when a column is not plain ints / doubles / booleans"""
    if fn in ("and", "or", "not"):
        for col in argv:
            if not set(map(type, col)) <= {bool, _NoneT}: return None
        ts = [np.fromiter((v is True for v in c), bool, n) for c in argv]
        fs = [np.fromiter((v is False for v in c), bool, n) for c in argv]
        if fn == "not": return _out(fs[0], ~(ts[0] | fs[0]))
        if fn == "and":
            f = np.logical_or.reduce(fs); t = np.logical_and.reduce(ts)
            return _out(t, ~(t | f))
        t = np.logical_or.reduce(ts); f = np.logical_and.reduce(fs)
        return _out(t, ~(t | f))
    if fn in ("is_null", "is_not_null"):
        if any(isinstance(v, Err) for v in argv[0]): return None
        m = np.fromiter((v is None for v in argv[0]), bool, n)
        return (m if fn == "is_null" else ~m).tolist()
    cols = [_numcol(c) for c in argv]
    if any(c is None for c in cols): return None
    nul = None
    for c in cols:
        if c[2] is not None: nul = c[2] if nul is None else (nul | c[2])
    kinds = {c[0] for c in cols}
    if fn == "neg":
        return _out(-cols[0][1], nul)
    a, b = cols[0][1], cols[1][1]
    if fn in _CMPF:
        if kinds == {"int", "float"} and max(c[3] for c in cols if c[0] == "int") >= 1 << 53: return None
        return _out(_CMPF[fn](a, b), nul)
    if fn in ("%", "mod", "//", "divide"):            # integers, no zero divisor: truncating division, as SQL's
        if kinds != {"int"}: return None
        live = b if nul is None else b[~nul]
        if (live == 0).any(): return None
        bb = np.where(b == 0, 1, b)
        if fn in ("%", "mod"): return _out(np.fmod(a, bb), nul)
        q = np.abs(a) // np.abs(bb)
        return _out(np.where((a >= 0) == (bb >= 0), q, -q), nul)
    if fn == "/":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return _out(a.astype(np.float64) / b.astype(np.float64), nul)
    if kinds == {"int"}:
        ma, mb = cols[0][3], cols[1][3]
        if fn == "*" and ma * mb >= 1 << 63: return None
        return _out(a + b if fn == "+" else a - b if fn == "-" else a * b, nul)
    if kinds == {"int", "float"} and max(c[3] for c in cols if c[0] == "int") >= 1 << 53: return None
    a, b = a.astype(np.float64), b.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return _out(a + b if fn == "+" else a - b if fn == "-" else a * b, nul)


_INT_T = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"}


def _vec_cast(vals, src, dst):
    """CAST between integer types and DOUBLE on a whole column (None: row by row)"""
    if src is None or not (src.integer or src.id == "DOUBLE"): return None
    if not (dst.id == "DOUBLE" or dst.id in _INT_T): return None
    c = _numcol(vals)
    if c is None: return None
    kind, arr, nul, mx = c
    if dst.id == "DOUBLE":
        return _out(arr.astype(np.float64), nul) if kind == "int" else None if src.id != "DOUBLE" else vals
    if kind != "int": return None
    lo, hi = T.INT_RANGE[dst.id]
    if lo <= -mx and mx <= hi: return vals                  # every value fits: unchanged
    return None


# ------------------------------------------------------------------ expressions on codes
import coltypes as _ct
_BOOL = _ct.BoolCodec()
_INTC = _ct.IntCodec(0, "BIGINT")


def _num(a):
    """a code column as numbers: ("int", int64 values) | ("dec", int64 scaled values, scale) | ("float", float64) | None"""
    cd, c = a.codec, a.codes
    if c.dtype == object: return None
    if cd is None: return ("float", c, 0) if c.dtype == np.float64 else ("int", c, 0)
    k = getattr(cd, "kind", None)
    if k in ("int", "bool", "decimal"):
        off = int(getattr(cd, "offset", 0))
        if len(c) and abs(off) + int(np.abs(c).max()) >= 1 << 62: return None
        return ("dec", c + off, cd.scale) if k == "decimal" else ("int", c + off, 0)
    if k == "dict" and getattr(cd, "floating", False):
        fv = np.array(cd.values, np.float64)
        return ("float", fv[np.clip(c, 0, max(len(fv) - 1, 0))] if len(fv) else np.zeros(len(c)), 0)
    return None


def _mk(kind, v, sc, valid):
    """a computed column of numbers as a code column"""
    if kind == "float": return ColArr(v, valid, None)
    if kind == "dec": return ColArr(v, valid, _ct.DecimalCodec(sc, 0))
    if kind == "bool": return ColArr(v.astype(np.int64), valid, _BOOL)
    return ColArr(v, valid, _INTC)


def _andv(a, b):
    if a is None: return b
    if b is None: return a
    return a & b


def _to_scale(v, s_from, s_to):
    """exact rescaling of scaled integers up (None when it could overflow int64)"""
    if s_to == s_from: return v
    f = 10 ** (s_to - s_from)
    if len(v) and int(np.abs(v).max()) * f >= 1 << 62: return None
    return v * f


def codes_of(e, rel, cx):
    """e on every row of rel as a code column (ColArr), computed on the codes of the engine's columns without
    decoding them; None when e is not computed that way (row by row instead)"""
    if not (COLUMNAR and VEC_SCALARS): return None
    a = rel.cols.array(e.key())
    if a is not None: return a
    if rel.n < 64: return None
    r = _ev_codes(e, rel, cx)
    if r is not None and len(r) == rel.n: rel.cols.put(e.key(), r); return r
    return None


def _operand(e, rel, cx):
    """(kind, values, scale, valid) of an operand: a code column or a constant"""
    if isinstance(e, I.Lit):
        v = e.v; n = rel.n
        if v is None: return ("null", None, 0, np.zeros(n, bool))
        if isinstance(v, bool): return ("bool", np.full(n, int(v), np.int64), 0, None)
        if isinstance(v, int) and abs(v) < 1 << 62: return ("int", np.full(n, v, np.int64), 0, None)
        if isinstance(v, float): return ("float", np.full(n, v, np.float64), 0, None)
        if isinstance(v, Decimal) and v.is_finite():
            sg, dg, ex = v.as_tuple(); sc = max(0, -ex)
            iv = int(v.scaleb(sc)) if sc <= 18 else None
            if iv is not None and abs(iv) < 1 << 62: return ("dec", np.full(n, iv, np.int64), sc, None)
        return None
    a = codes_of(e, rel, cx)
    if a is None: return None
    if a.codec is _BOOL or getattr(a.codec, "kind", None) == "bool":
        if a.codes.dtype == object: return None
        return ("bool", a.codes, 0, a.valid)
    x = _num(a)
    if x is None: return None
    return (x[0], x[1], x[2], a.valid)


def _as_float(o):
    k, v, sc, valid = o
    if k == "float": return v
    if k in ("int", "bool"):
        if len(v) and int(np.abs(v).max()) >= 1 << 53: return None
        return v.astype(np.float64)
    if k == "dec":
        if len(v) and int(np.abs(v).max()) >= 1 << 53: return None
        return v.astype(np.float64) / float(10 ** sc)
    return None


def _ev_codes(e, rel, cx):
    """arithmetic, comparisons, AND / OR / NOT, IS NULL and numeric casts on code columns (exact integers, exact
    scaled decimals, IEEE doubles with SQL's NaN order, three-valued logic); None for anything else"""
    if isinstance(e, I.Cast):
        o = _operand(e.arg, rel, cx)
        if o is None or o[0] == "null": return None
        k, v, sc, valid = o
        t = e.t
        if t.id == "DOUBLE":
            f = _as_float(o)
            return None if f is None else _mk("float", f, 0, valid)
        if t.id == "DECIMAL" and k in ("int", "dec") and t.s >= sc:
            w = _to_scale(v, sc, t.s)
            if w is None or (len(w) and int(np.abs(w).max()) >= 10 ** t.p): return None
            return _mk("dec", w, t.s, valid)
        if t.integer and k == "int":
            lo, hi = T.INT_RANGE.get(t.id, (None, None))
            if lo is None or (len(v) and (int(v.min()) < lo or int(v.max()) > hi)): return None
            return _mk("int", v, 0, valid)
        return None
    if not isinstance(e, I.Func): return None
    fn = e.fn
    if fn in ("and", "or", "not"):
        ops = [_operand(a, rel, cx) for a in e.args]
        if any(o is None or o[0] not in ("bool", "null") for o in ops): return None
        n = rel.n
        ts = [np.zeros(n, bool) if o[0] == "null" else ((o[1] != 0) if o[3] is None else (o[1] != 0) & o[3]) for o in ops]
        fs = [np.zeros(n, bool) if o[0] == "null" else ((o[1] == 0) if o[3] is None else (o[1] == 0) & o[3]) for o in ops]
        if fn == "not": t, f = fs[0], ts[0]
        elif fn == "and": f = np.logical_or.reduce(fs); t = np.logical_and.reduce(ts)
        else: t = np.logical_or.reduce(ts); f = np.logical_and.reduce(fs)
        return _mk("bool", t, 0, t | f)
    if fn in ("is_null", "is_not_null"):
        a = codes_of(e.args[0], rel, cx)
        if a is None: return None
        m = a.valid if a.valid is not None else np.ones(rel.n, bool)
        return _mk("bool", ~m if fn == "is_null" else m, 0, None)
    if fn not in ("+", "-", "*", "/", "=", "<>", "<", "<=", ">", ">=", "neg", "%", "mod", "//", "divide"): return None
    ops = [_operand(a, rel, cx) for a in e.args]
    if any(o is None for o in ops): return None
    if any(o[0] == "null" for o in ops): return None
    valid = None
    for o in ops: valid = _andv(valid, o[3])
    kinds = {o[0] for o in ops}
    if fn == "neg":
        k, v, sc, _ = ops[0]
        return None if k == "bool" else _mk(k, -v, sc, valid)
    if "bool" in kinds and kinds != {"bool"}: return None
    (ka, va, sa, _), (kb, vb, sb, _) = ops
    if fn in ("%", "mod", "//", "divide"):              # integers, no zero divisor: truncating, as SQL's
        if kinds != {"int"}: return None
        live = vb if valid is None else vb[valid]
        if (live == 0).any(): return None
        bb = np.where(vb == 0, 1, vb)
        if fn in ("%", "mod"): return _mk("int", np.fmod(va, bb), 0, valid)
        qq = np.abs(va) // np.abs(bb)
        return _mk("int", np.where((va >= 0) == (bb >= 0), qq, -qq), 0, valid)
    if fn == "/" or "float" in kinds:                     # double arithmetic (DuckDB: / on numbers is a double division)
        fa, fb = _as_float(ops[0]), _as_float(ops[1])
        if fa is None or fb is None: return None
        if fn in ("+", "-", "*", "/"):
            with np.errstate(all="ignore"):
                r = fa + fb if fn == "+" else fa - fb if fn == "-" else fa * fb if fn == "*" else fa / fb
            return _mk("float", r, 0, valid)
        an, bn = np.isnan(fa), np.isnan(fb)                 # NaN is the largest value and equals itself
        with np.errstate(invalid="ignore"):
            lt = (fa < fb) | (~an & bn); eq = (fa == fb) | (an & bn)
    else:                                                  # exact: integers and scaled decimals
        S = max(sa, sb)
        xa, xb = _to_scale(va, sa, S), _to_scale(vb, sb, S)
        if xa is None or xb is None: return None
        if fn in ("+", "-"):
            r = xa + xb if fn == "+" else xa - xb
            if e.t.id == "DECIMAL" and e.t.s != S: return None
            return _mk("dec" if S else "int", r, S, valid)
        if fn == "*":
            if len(va) and int(np.abs(va).max()) * int(np.abs(vb).max() if len(vb) else 0) >= 1 << 62: return None
            if e.t.id == "DECIMAL" and e.t.s != sa + sb: return None
            return _mk("dec" if sa + sb else "int", va * vb, sa + sb, valid)
        lt, eq = xa < xb, xa == xb
    r = {"=": eq, "<>": ~eq, "<": lt, "<=": lt | eq, ">": ~(lt | eq), ">=": ~lt}[fn]
    return _mk("bool", r, 0, valid)


def rows_where(e, rel, cx):
    """the rows where condition e is TRUE (on codes when it can be computed there)"""
    a = codes_of(e, rel, cx) if e is not None else None
    if a is not None and (a.codec is _BOOL or getattr(a.codec, "kind", None) == "bool") and a.codes.dtype != object:
        m = a.codes != 0
        if a.valid is not None: m &= a.valid
        return np.flatnonzero(m).tolist()
    return truth(ev(e, rel, cx))


def _in_set(e, xs):
    """x IN (many constants): a hash lookup per row (NaN and type-crossing values compare as SQL does)"""
    vals = [a.v for a in e.args[1:]]
    has_null = any(v is None for v in vals)
    st = {nkey(v) for v in vals if v is not None}
    out = []
    for x in xs:
        if x is None or isinstance(x, Err): out.append(x if isinstance(x, Err) else None); continue
        out.append(True if nkey(x) in st else (None if has_null else False))
    return out


def truth(vals):
    """WHERE / HAVING / FILTER: rows whose condition is TRUE (an error fails the query)"""
    out = []
    for i, v in enumerate(vals):
        if isinstance(v, Err): raise SqlError(v.msg)
        if v is True: out.append(i)
    return out


def nkey(v):
    """a hashable key under which values that SQL treats as not distinct are equal (NULLs, NaNs)"""
    if v is None: return ("\0null",)
    if isinstance(v, float):
        if v != v: return ("\0nan",)
        return v + 0.0
    if isinstance(v, Decimal) and v.is_nan(): return ("\0nan",)       # DECFLOAT NaN: one group, as NaN = NaN
    if isinstance(v, list): return ("\0list",) + tuple(nkey(x) for x in v)
    if isinstance(v, tuple):                   # a row whose fields are all NULL is not distinct from the NULL row (ISO 8.15)
        if all(x is None for x in v): return ("\0null",)
        return ("\0row",) + tuple(nkey(x) for x in v)
    return v


def _ckeys(vals, t):
    """values -> their comparison keys under the collation of type t (unchanged without one)"""
    f = T.coll_key(t) if t is not None else None
    if f is None: return vals
    return [f(v) if isinstance(v, str) else v for v in vals]


def _out_type(q, i):
    """the type of output column i of a query"""
    if isinstance(q, I.Select): return q.items[i][0].t
    if isinstance(q, I.SetOp): return q.items[i][1]
    return None


def cmp_key(desc, nulls_first):
    """a cmp function for sorting values in one direction with NULL placement"""
    def c(a, b):
        if a is None or b is None:
            if a is None and b is None: return 0
            return (-1 if nulls_first else 1) if a is None else (1 if nulls_first else -1)
        r = F.cmp_values(a, b)
        return -r if desc else r
    return c


def sort_rows(idx, keys, specs):
    """sort row indexes by key columns: specs [(desc, nulls_first)]; stable"""
    if len(idx) > 48 and keys:                             # each key as dense order codes, one stable lexsort
        import sqlwin
        codes = [sqlwin.order_codes([k[i] for i in idx], d, nf) for k, (d, nf) in zip(keys, specs)]
        return [idx[p] for p in np.lexsort(codes[::-1]).tolist()]
    cmps = [cmp_key(d, nf) for d, nf in specs]
    def c(i, j):
        for k, f in zip(keys, cmps):
            r = f(k[i], k[j])
            if r: return r
        return 0
    return sorted(idx, key=functools.cmp_to_key(c))


# ================================================================== queries
import datetime as _dt
_AS_IS = {type(None), int, float, str, bool, bytes, _dt.date, _dt.datetime, _dt.time}     # returned unchanged


def run(db, q, engine="alg"):
    """execute a bound statement -> Relation (values as returned to the caller)"""
    import decimal, sqlfront
    T.SESSION["zone"] = getattr(db, "_sql_zone", None)
    if isinstance(q, sqlfront.SetSession):
        db._sql_zone = q.zone
        return Relation([], [], [])
    F.CLOCK["now"] = None
    cx = Cx(db, engine)
    with decimal.localcontext(T.CTX):
        r = run_query(q, cx)
        dec = [t.s if t.id == "DECIMAL" else None for t in r.types]
        cols = r._cols if r._cols is not None else (list(zip(*r.rows)) if r.rows and r.types else [])
        out = []
        for col, s, t in zip(cols, dec, r.types):          # column by column: most columns are returned as they are
            tys = set(map(type, col))
            for x in tys:
                if issubclass(x, Err): raise SqlError(next(v for v in col if isinstance(v, Err)).msg)
            if tys <= _AS_IS or (s is None and tys <= _AS_IS | {Decimal, T.DecFloat}):
                out.append(col); continue
            conv = []
            for v in col:
                if s is not None and isinstance(v, Decimal): v = T._dec_quant(v, s)
                conv.append(T.out_value(v, t))
            out.append(conv)
        rows = list(zip(*out)) if out else [()] * len(r)
    res = Relation(r.names, r.types, rows)
    res.stats = cx.stats
    return res


def run_query(q, cx):
    if isinstance(q, I.Select): return run_select(q, cx)
    if isinstance(q, I.TableFn): return run_tablefn(q, cx)
    if isinstance(q, I.SetOp): return run_setop(q, cx)
    if isinstance(q, I.Values): return run_values(q, cx)
    if isinstance(q, I.Recursive): return run_recursive(q, cx)
    raise SqlError(f"internal: cannot run {q!r}")


def run_tablefn(q, cx):
    """UNNEST(a, b, ...) [WITH ORDINALITY]: one row per element, the shorter collections padded with NULLs (ISO 7.6);
    range / generate_series"""
    one = Rel(1)
    vals = []
    for a in q.args:
        v = ev(a, one, cx)[0]
        if isinstance(v, Err): raise SqlError(v.msg)
        vals.append(v)
    if q.name == "match_recognize":
        import sqlmr
        inp = run_query(q.input, cx)
        return Relation(q.names, q.types, sqlmr.run(q.mr, inp.rows, cx))
    if q.name == "json_table":
        import sqljson
        sp = q.spec
        if "cpath" not in sp:                         # compile the paths and the column options once
            def prep(cols):
                for c in cols:
                    if c["kind"] == "nested": c["cpath"] = sqljson.compile_path(c["path"]); prep(c["cols"]); continue
                    if c["kind"] == "ordinality": continue
                    c["opts"]["cpath"] = sqljson.compile_path(c["opts"]["path"]); c["rt"] = T.type_from_name(c["rtype"])
            sp["cpath"] = sqljson.compile_path(sp["path"]); prep(sp["cols"])
        if vals[0] is None: return Relation(q.names, q.types, [])
        np_ = sp["np"]
        sp["defaults"] = vals[1 + np_:]
        rows = sqljson.json_table_rows(sp, vals[0], vals[1:1 + np_], [T.type_from_name(x) for x in sp["ptypes"]])
        out = []
        for r in rows:
            row = []
            for i, t in enumerate(q.types):
                v = r.get(i)
                if v is not None and t.id in ("VARCHAR",) and not isinstance(v, str): v = T.to_text(v, None)
                if isinstance(v, bool) and t.integer: v = int(v)
                row.append(v)
            out.append(tuple(row))
        return Relation(q.names, q.types, out)
    if q.name == "unnest":
        cols = [v if v is not None else [] for v in vals]
        n = max((len(c) for c in cols), default=0)
        rows = [tuple(c[i] if i < len(c) else None for c in cols) for i in range(n)]
    else:
        if any(v is None for v in vals): rows = []
        else:
            if len(vals) == 1: lo, hi, st = 0, vals[0], 1
            elif len(vals) == 2: lo, hi, st = vals[0], vals[1], 1
            else: lo, hi, st = vals
            if st == 0: raise SqlError(f"{q.name}: the step cannot be 0")
            if q.name == "generate_series": hi = hi + (1 if st > 0 else -1)
            rows = [(v,) for v in range(lo, hi, st)]
    if q.ordinality: rows = [r + (i + 1,) for i, r in enumerate(rows)]
    return Relation(q.names, q.types, rows)


def run_values(q, cx):
    rows = []
    one = Rel(1)
    for r in q.rows:
        rows.append(tuple(ev(x, one, cx)[0] for x in r))
    return Relation(q.names, q.types, rows)


def run_recursive(q, cx):
    base = run_query(q.base, cx)
    names = [c for c, _, _ in q.binding.cols]; types = [t for _, t, _ in q.binding.cols]
    result = [tuple(r) for r in base.rows]
    seen = {tuple(nkey(v) for v in r) for r in result} if not q.all else None
    if not q.all:
        uniq = []; s2 = set()
        for r in result:
            k = tuple(nkey(v) for v in r)
            if k not in s2: s2.add(k); uniq.append(r)
        result = uniq
    work = list(result)
    for it in range(100000):
        if not work: break
        cx.ctes[id(q.binding)] = Relation(names, types, work)
        cx.memo.clear()
        step = run_query(q.step, cx)
        new = []
        for r in step.rows:
            if seen is not None:
                k = tuple(nkey(v) for v in r)
                if k in seen: continue
                seen.add(k)
            new.append(tuple(r))
        result += new; work = new
    else: raise SqlError("recursive CTE did not terminate (100000 iterations)")
    cx.ctes.pop(id(q.binding), None)
    return Relation(names, types, result)


def _setop_side(q, cx, types):
    """one operand of a set operation as columns: code columns (ColArr) where its rows were kept as codes, else values"""
    if isinstance(q, I.Select) and COLUMNAR and VEC_SCALARS:
        out = run_select(q, cx, as_rel=True)
        if isinstance(out, Rel):
            cols = []
            for (x, _), t in zip(q.items, types):
                a = codes_of(x, out, cx) if x.t == t and T.coll_key(t) is None else None
                if a is None:
                    v = ev(x, out, cx)
                    a = v if x.t == t else [u if u is None or isinstance(u, Err) else T.cast_value(u, x.t, t) for u in v]
                    for u in a:
                        if isinstance(u, Err): raise SqlError(u.msg)
                cols.append(a)
            return cols, out.n
        rel = out
    else: rel = run_query(q, cx)
    cols = [list(c) for c in zip(*rel.rows)] if rel.rows else [[] for _ in types]
    cols = [c if t0 == t1 else [v if v is None or isinstance(v, Err) else T.cast_value(v, t0, t1) for v in c]
            for c, t0, t1 in zip(cols, rel.types, types)]
    return cols, len(rel.rows)


def _col_values(c, idx):
    """values of a column (code column or values) at the positions idx"""
    if isinstance(c, ColArr): return c.take(idx).decode()
    return [c[i] for i in idx]


def run_setop(q, cx):
    types = [t for _, t in q.items]
    if COLUMNAR and VEC_SCALARS and q.op in ("union", "intersect", "except") and not any(T.coll_key(t) for t in types):
        import sqlwin                                  # the operands as histograms of their row codes
        ca, na = _setop_side(q.left, cx, types)
        cb, nb = _setop_side(q.right, cx, types)
        parts = []
        for a, b in zip(ca, cb):
            ja, jb = sqlwin.joint_codes(a, b)
            ma = sqlwin.nn_mask(a) if isinstance(a, ColArr) else sqlwin.notnull(sqlwin.obj(a))
            mb = sqlwin.nn_mask(b) if isinstance(b, ColArr) else sqlwin.notnull(sqlwin.obj(b))
            ja = np.where(ma, np.asarray(ja), -1); jb = np.where(mb, np.asarray(jb), -1)     # NULL: a value of its own
            parts.append(np.unique(np.concatenate([ja, jb]), return_inverse=True)[1].reshape(-1).astype(np.int64))
        codes = sqlwin._combine(parts) if parts else np.zeros(na + nb, np.int64)
        if q.op == "union":
            keep = np.arange(na + nb) if q.all else sqlwin.first_rows(codes)
        else:
            c1, c2 = codes[:na], codes[na:]
            if q.all:                                  # the k-th copy of a row of A meets the k-th copy in B
                nb_ = np.bincount(c2, minlength=int(codes.max()) + 1) if len(codes) else np.zeros(0, np.int64)
                k = sqlwin.occurrence(c1)
                inb = k < nb_[c1] if len(c1) else np.zeros(0, bool)
                keep = np.flatnonzero(inb if q.op == "intersect" else ~inb)
            else:
                first = sqlwin.first_rows(c1)
                inb = np.isin(c1[first], c2)
                keep = first[inb if q.op == "intersect" else ~inb]
        ka, kb = keep[keep < na], keep[keep >= na] - na
        cols = [_col_values(a, ka) + _col_values(b, kb) for a, b in zip(ca, cb)]
        rows = list(zip(*cols)) if cols else [()] * len(keep)
        names = [n for n, _ in q.items]
        rel = Rel(len(rows))
        for i in range(len(names)): rel.cols[("pos", i)] = cols[i]
        order = [(("pos", o[0][1]), o[1], o[2]) for o in q.order]
        rows = _order_limit(rows, rel, order, q, cx, None)
        return Relation(names, types, rows)
    a, b = run_query(q.left, cx), run_query(q.right, cx)
    def conv(rel):
        cols = list(zip(*rel.rows)) if rel.rows else [()] * len(types)
        out = []
        for col, t0, t1 in zip(cols, rel.types, types):
            if t0 == t1: out.append(col); continue
            out.append([v if v is None or isinstance(v, Err) else T.cast_value(v, t0, t1) for v in col])
        return list(zip(*out)) if out and rel.rows else [()] * len(rel.rows)
    ra, rb = conv(a), conv(b)
    op = q.op
    kf = [T.coll_key(t) for t in types]
    if any(kf):                                       # collated columns: rows equal when their keys are
        return _setop_collated(q, ra, rb, kf, names_types=(q.items, types), cx=cx)
    import sqlwin
    if op not in ("union", "intersect", "except"): raise SqlError(f"set operation {op} is not supported")
    rows = ra + rb
    na = len(ra)
    codes = sqlwin.row_codes([list(c) for c in zip(*rows)] if rows and types else [], len(rows))
    if op == "union":
        if not q.all: rows = [rows[i] for i in sqlwin.first_rows(codes).tolist()]
    else:
        ca, cb = codes[:na], codes[na:]
        if q.all:                                     # the k-th copy of a row of A meets the k-th copy in B
            nb = np.bincount(cb, minlength=int(codes.max()) + 1 if len(codes) else 0) if len(codes) else np.zeros(0, np.int64)
            k = sqlwin.occurrence(ca)
            inb = k < nb[ca] if len(ca) else np.zeros(0, bool)
            keep = np.flatnonzero(inb if op == "intersect" else ~inb)
        else:
            first = sqlwin.first_rows(ca)
            inb = np.isin(ca[first], cb)
            keep = first[inb if op == "intersect" else ~inb]
        rows = [ra[i] for i in keep.tolist()]
    names = [n for n, _ in q.items]
    rel = Rel(len(rows))
    for i in range(len(names)): rel.cols[("pos", i)] = [r[i] for r in rows]
    order = [(("pos", o[0][1]), o[1], o[2]) for o in q.order]
    rows = _order_limit(rows, rel, order, q, cx, None)
    return Relation(names, types, rows)


def _setop_collated(q, ra, rb, kf, names_types, cx):
    def key(r): return tuple(nkey(f(v) if f is not None and isinstance(v, str) else v) for v, f in zip(r, kf))
    if q.op == "union":
        rows = ra + rb
        if not q.all:
            seen, out = set(), []
            for r in rows:
                k = key(r)
                if k not in seen: seen.add(k); out.append(r)
            rows = out
    else:
        cb = {}
        for r in rb: cb[key(r)] = cb.get(key(r), 0) + 1
        rows, seen = [], set()
        for r in ra:
            k = key(r)
            if q.all:
                if (cb.get(k, 0) > 0) == (q.op == "intersect"):
                    rows.append(r)
                if cb.get(k, 0) > 0: cb[k] -= 1
            elif k not in seen and (k in cb) == (q.op == "intersect"):
                seen.add(k); rows.append(r)
    items, types = names_types
    names = [n for n, _ in items]
    rel = Rel(len(rows))
    for i in range(len(names)): rel.cols[("pos", i)] = [r[i] for r in rows]
    order = [(("pos", o[0][1]), o[1], o[2]) for o in q.order]
    rows = _order_limit(rows, rel, order, q, cx, None)
    return Relation(names, types, rows)


def _distinct(rows):
    if len(rows) > 32 and rows and len(rows[0]):
        import sqlwin
        return [rows[i] for i in sqlwin.first_rows(sqlwin.row_codes([list(c) for c in zip(*rows)], len(rows))).tolist()]
    seen, out = set(), []
    for r in rows:
        k = tuple(nkey(v) for v in r)
        if k not in seen: seen.add(k); out.append(r)
    return out


def _order_limit(rows, rel, order, q, cx, extra_rel):
    """ORDER BY, OFFSET, LIMIT / FETCH (PERCENT, WITH TIES) on output rows (rel holds the sort keys)"""
    idx = list(range(len(rows)))
    keys = []
    if order:
        for spec, desc, nf in order:
            if spec[0] == "pos": keys.append(_ckeys([r[spec[1]] for r in rows], _out_type(q, spec[1])))
            else: keys.append(_ckeys(ev(spec[1], rel, cx), spec[1].t))
        for kcol in keys:
            for v in kcol:
                if isinstance(v, Err): raise SqlError(v.msg)
        idx = sort_rows(idx, keys, [(d, nf) for _, d, nf in order])
    off = _int_param(_count_value(q.offset, cx), "OFFSET") or 0
    lim = _count_value(q.limit, cx)
    if lim is not None:
        if q.percent:
            p = float(lim)
            if p < 0: raise SqlError("LIMIT percentage must be non-negative")
            # ISO FETCH FIRST p PERCENT: the ceiling of p% of the rows; DuckDB's LIMIT p%: the floor
            exact = Fraction(Decimal(str(p))) * len(idx) / 100
            lim = math.ceil(exact) if q.percent == "iso" else math.floor(exact)
        else: lim = _int_param(lim, "LIMIT")
    if off: idx = idx[off:]
    if lim is not None:
        if q.with_ties and lim > 0 and len(idx) > lim:
            last = idx[lim - 1]; k = lim
            while k < len(idx) and all(F.cmp_values(kc[idx[k]], kc[last]) == 0 if kc[idx[k]] is not None and kc[last] is not None
                                       else (kc[idx[k]] is None and kc[last] is None) for kc in keys):
                k += 1
            idx = idx[:k]
        else: idx = idx[:lim]
    return [rows[i] for i in idx]


def _order_codes_idx(out, s, cx):
    """the output rows' positions after ORDER BY, OFFSET and LIMIT, sorted on the codes of the keys; None when a key is
    not a code column (or is collated)"""
    import sqlwin
    keys = []
    for spec, desc, nf in s.order:
        e = s.items[spec[1]][0] if spec[0] == "pos" else spec[1]
        if T.coll_key(e.t) is not None: return None
        a = codes_of(e, out, cx)
        if a is None: return None
        keys.append(sqlwin.code_order(a, desc, nf))
    perm = np.lexsort(keys[::-1]) if keys else np.arange(out.n)
    off = _int_param(_count_value(s.offset, cx), "OFFSET") or 0
    lim = _count_value(s.limit, cx)
    if lim is not None: lim = _int_param(lim, "LIMIT")
    perm = perm[off:] if lim is None else perm[off:off + lim]
    return perm.tolist()


def _count_value(v, cx):
    """a FETCH / OFFSET count: a constant, or ("expr", e) evaluated now"""
    if isinstance(v, tuple) and len(v) == 2 and v[0] == "expr":
        x = ev(v[1], Rel(1), cx)[0]
        if isinstance(x, Err): raise SqlError(x.msg)
        if x is None: raise SqlError("FETCH / OFFSET count is NULL")
        return x
    return v


def _int_param(v, what):
    if v is None: return None
    if isinstance(v, float): v = int(v)
    if isinstance(v, Decimal): v = int(v)
    if not isinstance(v, int) or v < 0: raise SqlError(f"{what} must be a non-negative integer")
    return v


# ================================================================== TABLESAMPLE
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix64(x):
    """splitmix64, element-wise on uint64 (wrapping arithmetic)"""
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _sample_seed(sp, cx):
    """REPEATABLE (seed): that seed; else one random seed per statement"""
    if sp.seed is not None: return int(sp.seed) & 0xFFFFFFFFFFFFFFFF
    k = ("sample", id(sp))
    if k not in cx.memo: cx.memo[k] = int.from_bytes(__import__("os").urandom(8), "little")
    return cx.memo[k]


def sample_mask(n, used, sp, cx):
    """the stored rows a TABLESAMPLE keeps: BERNOULLI each row, SYSTEM each 64-row block (one word of every plane),
    with probability p %; RESERVOIR n rows: the n rows of smallest hash. The decision for a row is a hash of the seed
    and the row's position, so REPEATABLE (s) keeps the same rows in every query."""
    seed = _mix64(np.array([_sample_seed(sp, cx)], np.uint64))[0]
    idx = np.arange(n, dtype=np.uint64)
    unit = (idx >> np.uint64(6)) if sp.method == "system" else idx
    h = _mix64(unit ^ seed)
    used = np.asarray(used[:n], bool) if used is not None else np.ones(n, bool)
    if sp.rows is not None:
        cand = np.nonzero(used)[0]
        keep = cand[np.argsort(h[cand], kind="stable")[:sp.rows]]
        mask = np.zeros(n, bool); mask[keep] = True
        return mask
    thr = float(sp.pct) / 100.0 * 2.0 ** 53
    return ((h >> np.uint64(11)).astype(np.float64) < thr) & used


def sample_rel(rel, sp, cx):
    """a TABLESAMPLE of a multiset (rows standing for cnt rows each): each row kept with probability p (a binomial
    number of the cnt rows), or n rows drawn without replacement (a multivariate hypergeometric draw)"""
    rng = np.random.default_rng(_sample_seed(sp, cx))
    cnt = np.asarray(rel.cnt, np.int64)
    if sp.rows is not None:
        k = min(sp.rows, int(cnt.sum()))
        new = rng.multivariate_hypergeometric(cnt, k) if len(cnt) else cnt
    else:
        new = rng.binomial(cnt, float(sp.pct) / 100.0) if len(cnt) else cnt
    keep = [i for i in range(rel.n) if new[i] > 0]
    out = rel.take(keep)
    out.cnt = [int(new[i]) for i in keep]
    return out


# ================================================================== SELECT
def block_bindings(f):
    out = []
    def go(x):
        if isinstance(x, I.Join): go(x.left); go(x.right)
        elif isinstance(x, I.Binding): out.append(x)
    go(f)
    return out


def join_conds(f):
    out = []
    def go(x):
        if isinstance(x, I.Join):
            go(x.left); go(x.right)
            out.append(x)
    go(f)
    return out


def collect(exprs, cls):
    out = {}
    for x in exprs:
        if x is None: continue
        def f(n):
            if isinstance(n, cls): out.setdefault(n.key(), n)
        I.walk(x, f)
    return out


def subqueries_in(exprs):
    return list(collect(exprs, I.Subq).values())


def outer_refs_into(q, bset):
    """columns of the bindings bset that query q (or its subqueries) reads as correlated references"""
    found, seen = {}, set()
    def vx(x):
        if isinstance(x, I.Outer) and x.col.b in bset: found[x.col.key()] = x.col
        if isinstance(x, I.Subq): vq(x.q)
    def vq(q):
        if id(q) in seen: return                       # a recursive CTE reads itself
        seen.add(id(q))
        if isinstance(q, I.Select):
            exprs = [x for x, _ in q.items] + list(q.group) + [q.where, q.having, q.qualify] + \
                    [o[0][1] for o in q.order if o[0][0] == "expr"] + [j.cond for j in join_conds(q.from_)] + (q.distinct_on and [d[1] for d in q.distinct_on if d[0] == "expr"] or [])
            for x in exprs:
                if x is not None: I.walk(x, vx)
            for b in block_bindings(q.from_):
                if b.q is not None and not hasattr(b, "shared"): vq(b.q)
        elif isinstance(q, I.SetOp): vq(q.left); vq(q.right)
        elif isinstance(q, I.Recursive): vq(q.base); vq(q.step)
        elif isinstance(q, I.TableFn):
            for a in q.args: I.walk(a, vx)
            if getattr(q, 'input', None) is not None: vq(q.input)
        elif isinstance(q, I.Values):
            for r in q.rows:
                for x in r: I.walk(x, vx)
    vq(q)
    return list(found.values())


def run_select(s, cx, as_rel=False):
    if type(s.limit) is int and s.limit == 0 and not s.percent:
        # FETCH FIRST 0 ROWS / LIMIT 0: no row is output, so nothing is evaluated (as DuckDB: an error in a value
        # that no output row uses does not fail the query)
        return Relation([n for _, n in s.items], [x.t for x, _ in s.items], [])
    bindings = block_bindings(s.from_)
    bset = set(bindings)
    items = [x for x, _ in s.items]
    order_exprs = [o[0][1] for o in s.order if o[0][0] == "expr"]
    don = [d[1] for d in (s.distinct_on or []) if d[0] == "expr"]
    top = items + [s.having, s.qualify] + order_exprs + don
    aggs = collect(top, I.Agg)
    wins = collect(top, I.Win)
    # columns the block reads after the WHERE clause (per group or per row)
    need = {}
    def addcols(x):
        if x is None: return
        for c in I.cols_of(x): need[c.key()] = c
    for g in s.group: addcols(g)
    if not s.aggregate:
        for x in top: addcols(x)
    else:
        for a in aggs.values():
            pass                                   # decided by the planner (moments or values)
        for w in wins.values():
            for p in [*w.partition, *[o[0] for o in w.order], *w.args]:
                for c in I.find(p, I.Col, stop=(I.Agg,)): need[c.key()] = c
    for sq in subqueries_in(top + [s.where] + [j.cond for j in join_conds(s.from_)]):
        for c in outer_refs_into(sq.q, bset): need[c.key()] = c
    if s.from_ is None:
        rel = Rel(1); aggplan = {}
        if s.where is not None:
            keep = truth(ev(s.where, rel, cx)); rel = rel.take(keep)
    else:
        rel, aggplan = None, {}
        if all(b.table is not None for b in bindings):
            rel, aggplan = plan_star(s, bindings, need, aggs, cx)
        if rel is None:
            rel, aggplan = run_general(s, bindings, need, aggs, cx)
    return finish_select(s, rel, aggs, wins, aggplan, cx, as_rel)


# ------------------------------------------------------------------ group-level evaluation
def finish_select(s, rel, aggs, wins, aggplan, cx, as_rel=False):
    names = [n for _, n in s.items]; types = [x.t for x, _ in s.items]
    if s.aggregate:
        out = aggregate(s, rel, aggs, aggplan, cx)
    else:
        # one output row per underlying row: expand the groups' multiplicities
        if rel.cnt.count(1) != len(rel.cnt):
            idx = [i for i in range(rel.n) for _ in range(rel.cnt[i])]
            rel = rel.take(idx)
        out = rel
    if wins: compute_windows(list(wins.values()), out, cx)
    if s.qualify is not None:
        out = out.take(rows_where(s.qualify, out, cx))
    if as_rel and not s.distinct and not s.order and s.limit is None and s.offset is None:
        return out                                    # the rows before their values are built (a set operation's side)
    if COLUMNAR and s.distinct and not s.distinct_on and not s.order and s.limit is None and s.offset is None and out.n > 32:
        import sqlwin                                 # DISTINCT on codes: the first row of each code tuple, then values
        src = [codes_of(x, out, cx) if T.coll_key(x.t) is None else None for x, _ in s.items]
        if src and all(a is not None for a in src):
            keep = sqlwin.first_rows(sqlwin._combine([sqlwin.code_groups(a) for a in src])).tolist()
            out = out.take(keep)
            cols = [ev(x, out, cx) for x, _ in s.items]
            return Relation.of_columns(names, types, cols, out.n)
    if COLUMNAR and s.order and not s.distinct and not s.with_ties and not s.percent and out.n > 32:
        idx = _order_codes_idx(out, s, cx)              # ORDER BY / OFFSET / LIMIT on codes: only the rows output are decoded
        if idx is not None:
            out = out.take(idx)
            cols = [ev(x, out, cx) for x, _ in s.items]
            return Relation.of_columns(names, types, cols, out.n)
    cols = [ev(x, out, cx) for x, _ in s.items]
    if not s.distinct and not s.order and s.limit is None and s.offset is None:
        return Relation.of_columns(names, types, cols, out.n)        # rows built once, by the caller
    rows = list(zip(*cols)) if cols else [()] * out.n
    for i in range(len(s.items)): out.cols[("pos", i)] = cols[i]
    if s.distinct:
        if s.distinct_on:
            keys = [out.cols[("pos", d[1])] if d[0] == "pos" else ev(d[1], out, cx) for d in s.distinct_on]
            idx = list(range(len(rows)))
            if s.order:
                okeys = [out.cols[("pos", sp[1])] if sp[0] == "pos" else ev(sp[1], out, cx) for sp, _, _ in s.order]
                idx = sort_rows(idx, okeys, [(d, nf) for _, d, nf in s.order])
            seen, keep = set(), []
            for i in idx:
                k = tuple(nkey(kc[i]) for kc in keys)
                if k not in seen: seen.add(k); keep.append(i)
            keep.sort()
            out = out.take(keep); rows = [rows[i] for i in keep]
        else:
            kf = [T.coll_key(t) for t in types]
            if len(rows) > 32 and cols:                    # codes of the rows (collation keys), their first rows
                import sqlwin
                keyed = [c if f is None else [f(v) if isinstance(v, str) else v for v in c] for c, f in zip(cols, kf)]
                keep = sqlwin.first_rows(sqlwin.row_codes(keyed, len(rows))).tolist()
            else:
                seen, keep = set(), []
                for i, r in enumerate(rows):
                    k = tuple(nkey(f(v) if f is not None and isinstance(v, str) else v) for v, f in zip(r, kf))
                    if k not in seen: seen.add(k); keep.append(i)
            out = out.take(keep); rows = [rows[i] for i in keep]
    rows = _order_limit(rows, out, s.order, s, cx, None)
    return Relation(names, types, rows)


def _group_rows(s, rel, sets, gvals, gkeys, collated):
    """the output groups row by row: (Rel of the GROUP BY columns and GROUPING() values, members)"""
    out_groups = []                                  # (set index, key values tuple, member row indexes)
    for si, st in enumerate(sets):
        if not st:                                   # GROUP BY () or no GROUP BY: one group, even with no rows
            out_groups.append((si, (), list(range(rel.n)))); continue
        gm = {}
        order = []
        for i in range(rel.n):
            k = tuple(nkey(gkeys[j][i]) for j in st)
            g = gm.get(k)
            if g is None:
                g = gm[k] = (list(gvals[j][i] for j in st), []); order.append(k)
            else:
                for p_, j in enumerate(st):
                    if collated[j] and gvals[j][i] is not None and F.cmp_values(gvals[j][i], g[0][p_]) < 0: g[0][p_] = gvals[j][i]
            g[1].append(i)
        for k in order:
            vals, members = gm[k]
            out_groups.append((si, tuple(vals), members))
    out = Rel(len(out_groups))
    for j, g in enumerate(s.group):
        out.cols[g.key()] = [vals[sets[si].index(j)] if j in sets[si] else None for si, vals, _ in out_groups]
    _grouping_cols(s, out, sets, [si for si, _, _ in out_groups])
    members = _Members(m for _, _, m in out_groups)
    if VEC_GROUPS:
        sizes = np.array([len(m) for m in members], np.int64)
        members.flat = (np.concatenate([np.asarray(m, np.int64) for m in members]) if len(members) else np.zeros(0, np.int64),
                        np.cumsum(sizes) - sizes, np.cumsum(sizes))
        members.w = _counts(rel.cnt)
    return out, members


def _grouping_cols(s, out, sets, gset):
    """GROUPING(): bit i set when argument i is not grouped in the row's set (first argument most significant); gset:
    the grouping set of each output row"""
    for gx in collect([x for x, _ in s.items] + [s.having, s.qualify] + [o[0][1] for o in s.order if o[0][0] == "expr"], I.Grouping).values():
        pos = []
        for a in gx.args:
            k = a.key(); p = next((j for j, g in enumerate(s.group) if g.key() == k), None)
            if p is None: raise SqlError("GROUPING() arguments must be GROUP BY expressions")
            pos.append(p)
        per_set = []
        for st in sets:
            v = 0
            for p in pos: v = (v << 1) | (0 if p in st else 1)
            per_set.append(v)
        out.cols[gx.key()] = [per_set[si] for si in gset]


def _group_arrays(s, rel, sets, garr, gvals, gkeys, collated):
    """the output groups as arrays: per grouping set one stable sort of the combined group codes (groups numbered in
    order of appearance, members are runs of one row array); the GROUP BY columns stay codes, gathered at each group's
    first row (NULL in the sets that do not group them) -> (Rel, _FlatMembers)"""
    import sqlwin
    codes, pieces, off = {}, [], 0
    for si, st in enumerate(sets):
        if not st:                                   # GROUP BY () or no GROUP BY: one group, even with no rows
            pieces.append((si, st, None, np.arange(rel.n), np.array([0, rel.n]), {})); continue
        for j in st:
            if j not in codes: codes[j] = sqlwin.code_groups(garr[j]) if garr[j] is not None else sqlwin.group_codes(gkeys[j])
        comb = sqlwin._combine([codes[j] for j in st])
        _, first, inv = np.unique(comb, return_index=True, return_inverse=True)
        rank = np.empty(len(first), np.int64); rank[np.argsort(first, kind="stable")] = np.arange(len(first))
        gid = rank[inv.reshape(-1)]
        perm = np.argsort(gid, kind="stable")
        bounds = np.concatenate(([0], np.cumsum(np.bincount(gid, minlength=len(first)))))
        heads = perm[bounds[:-1]]
        least = {}
        for j in st:
            if collated[j]:                          # the least of the group's values (the first of equals)
                col = sqlwin.obj(gvals[j])[perm]
                nn = sqlwin.notnull(col)
                oc = np.full(len(col), np.iinfo(np.int64).max, np.int64)
                if nn.any(): oc[nn] = sqlwin.order_codes(col[nn].tolist()) * len(col) + np.flatnonzero(nn)
                best = np.minimum.reduceat(oc, bounds[:-1]) if len(col) else oc
                rep = [gvals[j][i] for i in heads.tolist()]
                least[j] = [col[b % len(col)] if b != np.iinfo(np.int64).max else r for b, r in zip(best.tolist(), rep)]
        pieces.append((si, st, heads, perm, bounds, least))
    sizes = [len(b) - 1 for _, _, _, _, b, _ in pieces]
    out = Rel(int(sum(sizes)))
    for j, g in enumerate(s.group):
        arr = garr[j] if garr[j] is not None and not collated[j] else None
        if arr is not None:                          # codes: one gather per set, NULL codes where the set does not group j
            cs, vs = [], []
            for (si, st, heads, _, _, _), G in zip(pieces, sizes):
                if j in st:
                    a = arr.take(heads); cs.append(a.codes)
                    vs.append(a.valid if a.valid is not None else np.ones(G, bool))
                else: cs.append(np.zeros(G, arr.codes.dtype)); vs.append(np.zeros(G, bool))
            cc = np.concatenate(cs) if cs else np.zeros(0, arr.codes.dtype); vv = np.concatenate(vs) if vs else np.zeros(0, bool)
            out.cols.put(g.key(), ColArr(cc, None if vv.all() else vv, arr.codec))
            continue
        col = []
        for (si, st, heads, _, _, least), G in zip(pieces, sizes):
            if j not in st: col += [None] * G
            elif j in least: col += least[j]
            else: col += [gvals[j][i] for i in heads.tolist()]
        out.cols[g.key()] = col
    _grouping_cols(s, out, sets, [si for (si, *_), G in zip(pieces, sizes) for _ in range(G)])
    rows = np.concatenate([perm for _, _, _, perm, _, _ in pieces]) if pieces else np.zeros(0, np.int64)
    st_, en_, o = [], [], 0
    for _, _, _, perm, b, _ in pieces:
        st_.append(b[:-1] + o); en_.append(b[1:] + o); o += len(perm)
    starts = np.concatenate(st_) if st_ else np.zeros(0, np.int64); ends = np.concatenate(en_) if en_ else np.zeros(0, np.int64)
    return out, _FlatMembers(rows.astype(np.int64), starts.astype(np.int64), ends.astype(np.int64), _counts(rel.cnt))


class _FlatMembers:
    """the rows of each output group as runs of one array: flat = (rows group after group, run starts, run ends), w the
    rows' counts; iterating or indexing gives each group's rows as a list (made once, on first use)"""
    def __init__(self, rows, starts, ends, w):
        self.flat, self.w, self._lists = (rows, starts, ends), w, None
    def __len__(self): return len(self.flat[1])
    def _ls(self):
        if self._lists is None:
            r = self.flat[0].tolist()
            self._lists = [r[a:b] for a, b in zip(self.flat[1].tolist(), self.flat[2].tolist())]
        return self._lists
    def __iter__(self): return iter(self._ls())
    def __getitem__(self, i): return self._ls()[i]
    def singletons(self):
        """each group's only row, when every group has exactly one (else None)"""
        rows, st, en = self.flat
        return rows[st] if len(st) and bool(((en - st) == 1).all()) else None


def aggregate(s, rel, aggs, aggplan, cx):
    """GROUP BY (with grouping sets) and every aggregate, from the rows of rel (each standing for cnt rows)"""
    vec = VEC_GROUPS and rel.n > 32
    garr = [codes_of(g, rel, cx) if vec and T.coll_key(g.t) is None else None for g in s.group]
    gvals = [None if a is not None else ev(g, rel, cx) for g, a in zip(s.group, garr)]      # engine columns stay codes
    for col in gvals:
        for v in col or ():
            if isinstance(v, Err): raise SqlError(v.msg)
    sets = s.sets if s.sets is not None else [list(range(len(s.group)))]
    gkeys = [None if v is None else _ckeys(v, g.t) for v, g in zip(gvals, s.group)]      # collated: grouped by key, the least value shown
    collated = [gv is not None and gk is not gv for gk, gv in zip(gkeys, gvals)]
    if vec: out, members = _group_arrays(s, rel, sets, garr, gvals, gkeys, collated)
    else: out, members = _group_rows(s, rel, sets, gvals, gkeys, collated)
    for k, a in aggs.items():
        out.cols[k] = agg_values(a, rel, members, aggplan.get(k), cx)
    if s.having is not None:
        out = out.take(rows_where(s.having, out, cx))
    return out


def _xsum(vals):
    s = 0
    for v in vals:
        if isinstance(v, float) or isinstance(s, float): s = float(s) + float(v)
        else: s = s + v
    return s


VEC_GROUPS = True           # GROUP BY and the value aggregates as array algebra (sqlwin); False: group by group (tests)


class _Members(list):
    """the rows of each output group (lists), with .flat = (all rows group after group, run starts, run ends) and .w the
    rows' counts as an array"""
    flat = None; w = None


def _counts(cnt):
    """row counts as an int64 array (object past 2^62)"""
    try:
        a = np.asarray(cnt, np.int64)
        return a if not len(a) or int(a.max()) < 1 << 62 else np.asarray(cnt, object)
    except OverflowError: return np.asarray(cnt, object)


def _agg_arrays(a, rel, members, args, filt, d):
    """the aggregate over every group at once (sqlwin.aggregate on the groups' runs, each row weighted by its count);
    None when it is computed group by group"""
    import sqlwin
    if a.order or a.name.startswith("__iso_h") or (a.name not in sqlwin._AGGS and a.name not in sqlwin._ORDER_AGGS): return None
    if getattr(members, "flat", None) is not None: rows, starts, ends = members.flat
    else:
        sizes = np.array([len(m) for m in members], np.int64)
        rows = np.concatenate([np.asarray(m, np.int64) for m in members]) if members else np.zeros(0, np.int64)
        ends = np.cumsum(sizes); starts = ends - sizes
    sizes = ends - starts
    n = len(rows)
    wall = getattr(members, "w", None)
    if wall is None: wall = _counts(rel.cnt)
    w = wall[rows] if n else np.zeros(0, np.int64)
    cols = [c.take(rows) if isinstance(c, ColArr) else sqlwin.obj(c)[rows] for c in args]
    valid = (w != 0) if n else np.zeros(0, bool)
    if filt is not None:
        fm = np.zeros(rel.n, bool); fm[filt] = True; valid &= fm[rows]
    if a.distinct:                                   # one row per distinct value (tuple) in each group, weight 1
        if d.skip_nulls and a.name != "count_star":
            for c in cols: valid &= sqlwin.nn_mask(c)
        akf = [T.coll_key(x.t) for x in a.args]
        gc = []
        for c, f in zip(cols, akf):
            if isinstance(c, ColArr) and f is None: gc.append(sqlwin.code_groups(c)); continue
            v = sqlwin.values_of(c).tolist()
            gc.append(sqlwin.group_codes(v if f is None else [f(x) if isinstance(x, str) else x for x in v]))
        vc = sqlwin._combine(gc) if gc else np.zeros(n, np.int64)
        grp = np.repeat(np.arange(len(members)), sizes)
        o = np.lexsort((np.arange(n), vc, grp, ~valid))
        firsts = np.zeros(n, bool)
        if n:
            G, V, Va = grp[o], vc[o], valid[o]
            new = np.ones(n, bool); new[1:] = (G[1:] != G[:-1]) | (V[1:] != V[:-1]) | (Va[1:] != Va[:-1])
            firsts[o[new]] = True
        valid &= firsts
        w = np.ones(n, np.int64)
    if w.dtype == object: return None
    try:
        if a.name in sqlwin._ORDER_AGGS:
            return sqlwin.order_aggregate(a.name, a.params, cols[0], valid, w, starts, ends).tolist()
        return sqlwin.aggregate(a.name, cols, valid, [(starts, ends)], n, w).tolist()
    except sqlwin.NoVec: return None


def agg_values(a, rel, members, plan, cx):
    """one aggregate for every output group (members: the rel rows of each group)"""
    if plan is not None and plan.get("kind") == "state":
        k = a.key()
        if a.name in ("min", "max"):                 # the engine's extremes per row
            col = rel.cols[("x", k)]; j = 0 if a.name == "min" else 1
            one = members.singletons() if hasattr(members, "singletons") else None
            if one is not None: return [None if col[i] is None else col[i][j] for i in one.tolist()]
            if VEC_GROUPS and getattr(members, "flat", None) is not None:
                got = _extreme_arrays(col, j, members.flat)
                if got is not None: return got
            out = []
            for mem in members:
                vals = [col[i][j] for i in mem if col[i] is not None]
                if not vals: out.append(None); continue
                best = vals[0]
                for v in vals[1:]:
                    if F.cmp_values(v, best) * (1 if j == 0 else -1) < 0: best = v
                out.append(best)
            return out
        cnt = rel.cols[("n", k)]
        nm = len(MOMENTS[a.name](None, None)) if a.name in MOMENTS else 0
        if VEC_GROUPS and a.name in ("count", "count_star", "sum") and getattr(members, "flat", None) is not None:
            got = _count_sum_arrays(a.name, cnt, rel.cols[("m", k, 0)] if a.name == "sum" else None, members.flat)
            if got is not None: return got
        one = members.singletons() if hasattr(members, "singletons") and a.name in ("count", "count_star", "sum") else None
        if one is not None:                          # one row per group: its count, its sum
            idx = one.tolist()
            if a.name != "sum": return [cnt[i] for i in idx]
            m0 = rel.cols[("m", k, 0)]
            return [None if cnt[i] == 0 else (0 if m0[i] is None else m0[i]) for i in idx]
        if VEC_GROUPS:
            got = _moment_arrays(a.name, cnt, [rel.cols[("m", k, j)] for j in range(nm)], members)
            if got is not None: return got
        out = []
        for mem in members:
            n = sum(cnt[i] for i in mem)
            sums = [_sum_exact(rel.cols[("m", k, j)][i] for i in mem) for j in range(nm)]
            out.append(_moment_final(a.name, n, sums))
        return out
    d = F.AGGS[a.name]
    filt = rows_where(a.filter, rel, cx) if a.filter is not None else None
    fset = set(filt) if filt is not None else None
    if VEC_GROUPS:                                   # engine columns as codes, others as values
        src = [codes_of(x, rel, cx) for x in a.args]
        src = [c if c is not None else ev(x, rel, cx) for c, x in zip(src, a.args)]
        for col in src:
            if not isinstance(col, ColArr):
                for v in col:
                    if isinstance(v, Err): raise SqlError(v.msg)
        got = _agg_arrays(a, rel, members, src, filt, d)
        if got is not None: return got
    args = [ev(x, rel, cx) for x in a.args]
    okeys = [_ckeys(ev(o[0], rel, cx), o[0].t) for o in a.order]
    for col in args + okeys:
        for v in col:
            if isinstance(v, Err): raise SqlError(v.msg)
    out = []
    for mem in members:
        rows = []
        for i in mem:
            if fset is not None and i not in fset: continue
            w = rel.cnt[i]
            if w == 0: continue
            vals = tuple(c[i] for c in args)
            if d.skip_nulls and a.name != "count_star" and any(v is None for v in vals): continue
            rows.append((vals, w, i))
        if a.distinct:
            seen, dd = set(), []
            akf = [T.coll_key(x.t) for x in a.args]
            for r in rows:
                k = tuple(nkey(f(v) if f is not None and isinstance(v, str) else v) for v, f in zip(r[0], akf))
                if k not in seen: seen.add(k); dd.append((r[0], 1, r[2]))
            rows = dd
        if a.order:
            idx = sort_rows(list(range(len(rows))), [[okeys[j][r[2]] for r in rows] for j in range(len(a.order))],
                            [(o[1], o[2]) for o in a.order])
            rows = [rows[k] for k in idx]
        rr = [(r[0], r[1]) for r in rows]
        if a.name.startswith("__iso_h"):
            out.append(_hypothetical(a, rr))
        elif a.name == "count_star": out.append(sum(w for _, w in rr))
        else: out.append(d.final(rr, a.params))
    return out


def _hypothetical(a, rows):
    """RANK / DENSE_RANK / PERCENT_RANK / CUME_DIST (args) WITHIN GROUP (ORDER BY ...): the rank of the hypothetical row"""
    kind = a.name[7:]
    hyp = tuple(a.params)
    specs = [(o[1], o[2]) for o in a.order]
    cmps = [cmp_key(d, nf) for d, nf in specs]
    def c(x):
        for xv, hv, f in zip(x, hyp, cmps):
            r = f(xv, hv)
            if r: return r
        return 0
    n = sum(w for _, w in rows)
    less = sum(w for x, w in rows if c(x) < 0)
    eq = sum(w for x, w in rows if c(x) == 0)
    if kind == "rank": return less + 1
    if kind == "dense_rank": return len({tuple(nkey(v) for v in x) for x, _ in rows if c(x) < 0}) + 1
    if kind == "percent_rank": return 0.0 if n == 0 else less / n
    if kind == "cume_dist": return (less + eq + 1) / (n + 1)
    raise SqlError(kind)


def _sum_exact(vals):
    s = 0
    for v in vals:
        if v is None: continue
        s = s + v
    return s


_MOMENT_VEC = {"avg", "var_pop", "var_samp", "stddev_pop", "stddev_samp", "covar_pop", "covar_samp", "corr", "regr_slope",
               "regr_intercept", "regr_r2", "regr_avgx", "regr_avgy", "regr_sxx", "regr_syy", "regr_sxy"}


def _exact_ints(vals):
    """exact numbers (int, Decimal, Fraction; None as 0) -> (object array of integers, their common denominator)"""
    if not set(map(type, vals)) <= {int, Decimal, Fraction, _NoneT}: return None
    rs = [(0, 1) if v is None else v.as_integer_ratio() for v in vals]
    dens = {d for _, d in rs}
    if dens == {1}: return np.array([p for p, _ in rs], object), 1
    den = math.lcm(*dens)
    return np.array([p if d == den else p * (den // d) for p, d in rs], object), den


def _moment_arrays(name, cnt, cols, members):
    """_moment_final for every group at once: the groups' sums of the engine's exact per-row moments, then the statistic
    as exact integer arithmetic on arrays (the doubles _moment_final gives); None: computed group by group"""
    if name not in _MOMENT_VEC or len(members) < 16: return None
    ex = [_exact_ints(c) for c in cols]
    if any(x is None for x in ex): return None
    if getattr(members, "flat", None) is not None: rows, starts, ends = members.flat
    else:
        sizes = np.array([len(m) for m in members], np.int64)
        rows = np.concatenate([np.asarray(m, np.int64) for m in members]) if len(sizes) and sizes.sum() else np.zeros(0, np.int64)
        ends = np.cumsum(sizes); starts = ends - sizes
    def gsum(arr):                                   # per group: the sum over its member rows (exact)
        pre = np.concatenate([np.zeros(1, object), np.cumsum(arr[rows])]) if len(rows) else np.zeros(1, object)
        return pre[ends] - pre[starts]
    n = gsum(np.array(cnt, object))
    S = [gsum(a) for a, _ in ex]; K = [k for _, k in ex]
    out = np.full(len(members), None, object)
    import sqlwin
    if name == "avg":
        h = np.nonzero(np.asarray(n != 0, bool))[0]; out[h] = sqlwin.fdiv(S[0][h], n[h] * K[0])
    elif name in ("var_pop", "var_samp", "stddev_pop", "stddev_samp"):
        samp = name.endswith("samp")
        h = np.nonzero(np.asarray(n >= (2 if samp else 1), bool))[0]
        N, S1, S2, k1, k2 = n[h], S[0][h], S[1][h], K[0], K[1]
        v = sqlwin.fdiv(N * S2 * (k1 * k1) - S1 * S1 * k2, (k2 * k1 * k1) * N * ((N - 1) if samp else N))
        out[h] = np.sqrt(v) if name.startswith("stddev") else v
    else:                                            # moments y, x, yy, xx, xy over one common denominator L
        L = math.lcm(*K)
        Sy, Sx = S[0] * (L // K[0]), S[1] * (L // K[1])
        Syy, Sxx, Sxy = (S[j] * (L * L // K[j]) for j in (2, 3, 4))
        out = sqlwin.bivariate_arrays(name, n, Sy, Sx, Syy, Sxx, Sxy, L, L)
    return out.tolist()


def _extreme_arrays(col, j, flat):
    """MIN (j = 0) / MAX (j = 1) of every group from the engine's per-row extremes: the values' order codes, one sort by
    (group, code, row), the first row of each group (the first of equal values, as the comparisons keep it)"""
    import sqlwin
    rows, starts, ends = flat
    vals = [None if col[i] is None else col[i][j] for i in rows.tolist()]
    nn = np.fromiter((v is not None for v in vals), bool, len(vals))
    G = len(starts)
    out = [None] * G
    if not nn.any(): return out
    try: code = sqlwin.order_codes([v for v in vals if v is not None])
    except Exception: return None
    grp = np.repeat(np.arange(G), ends - starts)[nn]
    at = np.flatnonzero(nn)
    o = np.lexsort((at, -code if j else code, grp))
    first = np.ones(len(o), bool); first[1:] = grp[o][1:] != grp[o][:-1]
    for g, p in zip(grp[o][first].tolist(), at[o][first].tolist()): out[g] = vals[p]
    return out


def _count_sum_arrays(name, cnt, m0, flat):
    """COUNT / SUM of every group from the engine's per-row states: exact sums over each group's run (SUM of integers
    only; None: group by group)"""
    rows, starts, ends = flat
    def gsum(arr):
        pre = np.concatenate([np.zeros(1, object), np.cumsum(arr[rows])]) if len(rows) else np.zeros(1, object)
        return pre[ends] - pre[starts]
    n = gsum(np.array(cnt, object))
    if name != "sum": return n.tolist()
    if not set(map(type, m0)) <= {int, _NoneT}: return None
    s = gsum(np.array([0 if v is None else v for v in m0], object))
    return [None if c == 0 else v for c, v in zip(n.tolist(), s.tolist())]


def moment_values(a, rel, members, plan):
    """an aggregate from the engine's per-group moments (exact sums of polynomials) and counts"""
    v = plan["variant"]; ms = plan["moments"]
    cnt = rel.vcnt.get(v, rel.cnt) if v != 0 else rel.cnt
    filt = plan.get("rowfilter")
    if VEC_GROUPS:
        mem = members if filt is None else [[i for i in m if i in filt] for m in members]
        got = _moment_arrays(a.name, cnt, [rel.mom[(v, m)] for m in ms], mem)
        if got is not None: return got
    out = []
    for mem in members:
        if filt is not None: mem = [i for i in mem if i in filt]
        n = sum(cnt[i] for i in mem)
        sums = [_sum_exact(rel.mom[(v, m)][i] for i in mem) for m in ms]
        out.append(_moment_final(a.name, n, sums))
    return out


def _moment_final(name, n, s):
    """an aggregate from exact moments (ISO: degenerate regressions and correlations are NULL)"""
    if name in ("count", "count_star"): return n
    if name == "sum": return None if n == 0 else s[0]
    if n == 0 and name not in ("regr_count",): return None
    fr = [F.frac(x) for x in s]
    if name == "avg": return F.ffloat(fr[0] / n)
    if name in ("var_pop", "var_samp", "stddev_pop", "stddev_samp"):
        v = F.var_from(n, fr[0], fr[1], name.endswith("samp"))
        if name.startswith("stddev") and v is not None: v = math.sqrt(v)
        return v
    # bivariate: moments y, x, yy, xx, xy
    return F.bivariate(name, (n, fr[0], fr[1], fr[2], fr[3], fr[4]))


MOMENTS = {"sum": lambda y, x: [y], "avg": lambda y, x: [y], "count": lambda y, x: [],
           "var_pop": lambda y, x: [y, ("*", y, y)], "var_samp": lambda y, x: [y, ("*", y, y)],
           "stddev_pop": lambda y, x: [y, ("*", y, y)], "stddev_samp": lambda y, x: [y, ("*", y, y)]}
for _k in ("covar_pop", "covar_samp", "corr", "regr_slope", "regr_intercept", "regr_count", "regr_r2", "regr_avgx",
           "regr_avgy", "regr_sxx", "regr_syy", "regr_sxy"):
    MOMENTS[_k] = lambda y, x: [y, x, ("*", y, y), ("*", x, x), ("*", x, y)]


# ------------------------------------------------------------------ windows
VEC_WINDOWS = True          # window functions as array algebra (sqlwin); False: row by row on each partition (tests)


def compute_windows(wins, rel, cx):
    import sqlwin
    n = rel.n
    layouts = {}                                           # windows with the same PARTITION BY / ORDER BY share one ordering
    for w in wins:
        if w.key() in rel.cols: continue
        def codes_or(e, keyed=True):
            """an engine column as its codes (ordered like its values, no collation), else its values (collation keys)"""
            a = codes_of(e, rel, cx) if VEC_WINDOWS else None
            if a is not None and (not keyed or T.coll_key(e.t) is None): return a
            v = ev(e, rel, cx)
            return _ckeys(v, e.t) if keyed else v
        parts = [codes_or(p) for p in w.partition]
        okeys = [codes_or(o[0]) for o in w.order]
        main = _win_args(w)
        inputs = [ev(a, rel, cx) for a in w.args[len(main):]]          # the columns its row pattern recognition reads
        if any(_has_nwf(a) for a in main) or (w.filter is not None and _has_nwf(w.filter)) or any(_has_nwf(o[0]) for o in w.agg_order):
            args, filt = None, None
        else:
            args = [codes_or(a, False) for a in main]
            filt = set(rows_where(w.filter, rel, cx)) if w.filter is not None else None
        for col in parts + okeys + (args or []) + inputs:
            if isinstance(col, ColArr): continue
            for v in col:
                if isinstance(v, Err): raise SqlError(v.msg)
        okv = lambda: [_ckeys(ev(o[0], rel, cx), o[0].t) for o in w.order]       # the ORDER BY values (row by row)
        specs = [(o[1], o[2]) for o in w.order]
        lk = (tuple(p.key() for p in w.partition), tuple((o[0].key(), o[1], o[2]) for o in w.order))
        L = layouts.get(lk)
        if L is None: L = layouts[lk] = sqlwin.layout(parts, okeys, specs, n)
        bound_vals = {}
        if w.frame is not None:
            for side, bd in (("a", w.frame[1]), ("b", w.frame[2])):
                if bd[1] is not None: bound_vals[side] = ev(bd[1], rel, cx)
        rpr = None
        if w.rpr is not None:                              # the matches, shared by the window functions of one window
            mkey = ("rpr-match", w.rpr.spec, tuple(p.key() for p in w.partition), tuple((o[0].key(), o[1], o[2]) for o in w.order),
                    I.Win("", [], None, frame=w.frame).key())
            got = rel.cols.get(mkey)
            rpr = (inputs, got if got is not None else [None] * n, got is None)
        aorder = ([_ckeys(ev(o[0], rel, cx), o[0].t) for o in w.agg_order], [(o[1], o[2]) for o in w.agg_order]) if w.agg_order else None
        res = None
        if VEC_WINDOWS and rpr is None and args is not None and aorder is None and n:
            fm = None
            if filt is not None:
                fm = np.zeros(n, bool); fm[list(filt)] = True
            vb = {}
            for side, bd in (("a", w.frame[1]), ("b", w.frame[2])) if w.frame is not None else ():
                if bd[1] is None: continue
                if not I.cols_of(bd[1]) and not I.has(bd[1], I.Subq) and not I.has(bd[1], I.Outer):
                    vb[side] = sqlwin.Const(bound_vals[side][0])          # a constant offset
                else: vb[side] = bound_vals[side]
            ranged = w.frame is not None and w.frame[0] == "range" and any(bd[1] is not None for bd in w.frame[1:])
            try:
                rk = (okeys[0] if isinstance(okeys[0], ColArr) else okv()[0]) if len(okeys) == 1 and ranged else None
                out = sqlwin.evaluate(w, L, args, fm, vb, rk, bool(specs and specs[0][0]))
                full = np.empty(n, object); full[L.perm] = out
                res = full.tolist()
            except sqlwin.NoVec:
                res = None
        if res is None:                                    # row by row on each partition: values
            res = [None] * n
            okeys = okv()
            if args is not None: args = [ev(a, rel, cx) for a in main]
            for idx, peers in L.partitions():
                _window_partition(w, idx, okeys, specs, args, filt, res, rel, cx, rpr, aorder, peers, bound_vals)
        if rpr is not None and rpr[2]: rel.cols[mkey] = rpr[1]
        rel.cols[w.key()] = res


def _win_args(w):
    """a window function's own arguments (without the inputs of its row pattern recognition)"""
    return w.args[:len(w.args) - len(w.rpr.names)] if w.rpr is not None else w.args


def _peer_bounds(idx, okeys):
    """for each position in the ordered partition: (first, last+1) of its peer group"""
    m = len(idx); start = [0] * m; end = [0] * m
    i = 0
    while i < m:
        j = i + 1
        while j < m and all(_same(k[idx[j]], k[idx[i]]) for k in okeys): j += 1
        for t in range(i, j): start[t], end[t] = i, j
        i = j
    return start, end


def _same(a, b):
    if a is None or b is None: return a is None and b is None
    return F.cmp_values(a, b) == 0


def _window_partition(w, idx, okeys, specs, args, filt, res, rel, cx, rpr=None, aorder=None, peers=None, bound_vals=None):
    """one ordered partition, row by row (idx: its rows in order; peers: each position's peer group (start, end))"""
    m = len(idx); name = w.name
    if peers is not None: pstart, pend = peers
    else: pstart, pend = _peer_bounds(idx, okeys) if w.order else ([0] * m, [m] * m)
    if name == "row_number":
        for p, i in enumerate(idx): res[i] = p + 1
        return
    if name == "rank":
        for p, i in enumerate(idx): res[i] = pstart[p] + 1
        return
    if name == "dense_rank":
        d, last = 0, -1
        for p, i in enumerate(idx):
            if pstart[p] != last: d += 1; last = pstart[p]
            res[i] = d
        return
    if name == "percent_rank":
        for p, i in enumerate(idx): res[i] = 0.0 if m <= 1 else pstart[p] / (m - 1)
        return
    if name == "cume_dist":
        for p, i in enumerate(idx): res[i] = pend[p] / m
        return
    if name == "ntile":
        for p, i in enumerate(idx):
            k = args[0][i]
            if k is None: res[i] = None; continue
            if k <= 0: raise SqlError("Argument for ntile must be greater than zero")
            q, r = divmod(m, k)
            # the first r buckets hold q + 1 rows
            res[i] = p // (q + 1) + 1 if p < r * (q + 1) else (p - r * (q + 1)) // q + 1 + r if q else p + 1
        return
    if name in ("lag", "lead"):
        vals = args[0]
        for p, i in enumerate(idx):
            off = args[1][i] if len(args) > 1 else 1
            dflt = args[2][i] if len(args) > 2 else None
            if off is None: res[i] = None; continue
            step = off if name == "lead" else -off
            if w.ignore_nulls:
                q, k, got = p, abs(step), None
                d1 = 1 if step >= 0 else -1
                if step == 0: res[i] = vals[i]; continue
                while k > 0:
                    q += d1
                    if q < 0 or q >= m: break
                    if vals[idx[q]] is not None: k -= 1
                res[i] = vals[idx[q]] if 0 <= q < m and k == 0 else dflt
            else:
                q = p + step
                res[i] = vals[idx[q]] if 0 <= q < m else dflt
        return
    # framed functions: first_value, last_value, nth_value, aggregates
    mode, a, b = w.frame if w.frame is not None else ("range", ("unbounded_preceding", None), ("current", None))
    if not w.order and mode == "range":
        a = ("unbounded_preceding", None) if a[0] in ("current",) else a
        b = ("unbounded_following", None) if b[0] in ("current",) else b
    if bound_vals is None:
        bound_vals = {}
        for side, bd in (("a", a), ("b", b)):
            if bd[1] is not None: bound_vals[side] = ev(bd[1], rel, cx)
    aux = _frame_aux(mode, a, b, idx, okeys, specs, pstart, pend, m)
    ex = w.exclude.upper().replace(" ", "_") if w.exclude else None
    if ex == "NO_OTHERS": ex = None

    def bounds_of(p, i):
        lo = _frame_edge(mode, a, "a", p, i, idx, okeys, specs, pstart, pend, bound_vals, m, aux)
        hi = _frame_edge(mode, b, "b", p, i, idx, okeys, specs, pstart, pend, bound_vals, m, aux)
        lo, hi = min(max(lo, 0), m), min(hi, m)           # a frame past the partition's end is empty
        return lo, max(lo, hi)

    def frame_of(p, lo, hi):
        """the frame positions (a range unless rows are excluded): frames are never all kept, so memory stays O(m)"""
        if ex is None: return range(lo, hi)
        if ex == "CURRENT_ROW": return [q for q in range(lo, hi) if q != p]
        if ex == "GROUP": return [q for q in range(lo, hi) if not (pstart[p] <= q < pend[p])]
        if ex == "TIES": return [q for q in range(lo, hi) if q == p or not (pstart[p] <= q < pend[p])]
        raise SqlError(f"EXCLUDE {w.exclude} is not supported")

    override = None
    if rpr is not None:                                    # row pattern recognition (R020): the frame reduced to the match
        inputs, cache, fill = rpr
        if fill:
            import sqlmr
            got = sqlmr.window_matches(w.rpr, [tuple(c[i] for c in inputs) for i in idx],
                                       [bounds_of(p, i) for p, i in enumerate(idx)], cx)
            for p, i in enumerate(idx): cache[i] = got[p]
        mt = [cache[i] for i in idx]
        if name == "__rpr_measure":
            for p, i in enumerate(idx): res[i] = None if mt[p] is None else mt[p][2][w.rpr.measure]
            return
        override = [(0, 0) if x is None else (x[0], x[1]) for x in mt]
        ex = None
    if args is None:                                       # nested window functions (T619) in the arguments
        frames = []
        for p, i in enumerate(idx):
            lo, hi = override[p] if override else bounds_of(p, i)
            frames.append(list(frame_of(p, lo, hi)))
        _nested_aggregate(w, idx, frames, res, rel, cx, m)
        return
    fast = _fast_window_agg(w, name, idx, args, filt) if ex is None and not w.distinct and aorder is None else None
    last_key = last_val = None
    for p, i in enumerate(idx):
        lo, hi = override[p] if override else bounds_of(p, i)
        key = (lo, hi) if ex is None and name != "nth_value" else None
        if key is not None and key == last_key: res[i] = last_val; continue     # the same frame as the row before
        if fast is not None:
            v = fast(lo, hi)
        elif name in ("first_value", "last_value", "nth_value"):
            vals = args[0]
            cand = [idx[q] for q in frame_of(p, lo, hi)]
            if w.ignore_nulls: cand = [c for c in cand if vals[c] is not None]
            if name == "first_value": v = vals[cand[0]] if cand else None
            elif name == "last_value": v = vals[cand[-1]] if cand else None
            else:
                k = args[1][i]
                if k is None: v = None
                else:
                    if k <= 0: raise SqlError("NTH_VALUE: the position must be positive")
                    if w.from_last: cand = cand[::-1]
                    v = vals[cand[k - 1]] if k <= len(cand) else None
        else:
            d = F.AGGS[name]
            rows = []
            ids = [idx[q] for q in frame_of(p, lo, hi)]
            if aorder is not None: ids = sort_rows(ids, aorder[0], aorder[1])        # the aggregate's own ORDER BY
            for r in ids:
                if filt is not None and r not in filt: continue
                vals = tuple(c[r] for c in args)
                if d.skip_nulls and name != "count_star" and any(x is None for x in vals): continue
                rows.append((vals, 1))
            if w.distinct:
                seen, dd = set(), []
                for r in rows:
                    k = tuple(nkey(x) for x in r[0])
                    if k not in seen: seen.add(k); dd.append(r)
                rows = dd
            v = len(rows) if name == "count_star" else d.final(rows, w.params)
        res[i] = v
        if key is not None: last_key, last_val = key, v


def _frame_aux(mode, a, b, idx, okeys, specs, pstart, pend, m):
    """what the frame edges of a partition need once: the peer groups (GROUPS), the non-NULL ORDER BY keys (RANGE
    with an offset), so that each edge is found in O(log m)"""
    aux = {}
    if mode == "groups" and (a[1] is not None or b[1] is not None):
        gstarts = []
        for q in range(m):
            if pstart[q] == q: gstarts.append(q)
        aux["gstarts"] = gstarts
        aux["gindex"] = {g: k for k, g in enumerate(gstarts)}
    if mode == "range" and (a[1] is not None or b[1] is not None) and len(okeys) == 1:
        key = okeys[0]
        nn = [q for q in range(m) if key[idx[q]] is not None]
        aux["nn_lo"] = nn[0] if nn else (m if specs[0][1] else 0)
        aux["nn_keys"] = [key[idx[q]] for q in nn]
    return aux


def _count_before(keys, tgt, desc, strict):
    """how many leading keys (sorted ascending, or descending when desc) lie before tgt: strictly before (strict) or
    up to and including it"""
    lo, hi = 0, len(keys)
    while lo < hi:
        mid = (lo + hi) // 2
        c = F.cmp_values(keys[mid], tgt)
        if desc: c = -c
        if c < 0 or (c == 0 and not strict): lo = mid + 1
        else: hi = mid
    return lo


def _fast_window_agg(w, name, idx, args, filt):
    """COUNT / SUM / AVG from prefix sums and MIN / MAX from a sparse table over the ordered partition: each frame in
    O(1), with the results the aggregate gives on the frame's rows (exact sums, correctly rounded floats)"""
    if name not in ("count_star", "count", "sum", "avg", "min", "max"): return None
    if name == "count_star": vals = [None] * len(idx)
    else:
        if len(args) != 1: return None
        col = args[0]; vals = [col[r] for r in idx]
    keep = [filt is None or r in filt for r in idx]
    m = len(idx)
    if name == "count_star":
        pc = [0] * (m + 1)
        for q in range(m): pc[q + 1] = pc[q] + (1 if keep[q] else 0)
        return lambda lo, hi: pc[hi] - pc[lo]
    live = [v if k else None for v, k in zip(vals, keep)]
    if name == "count":
        pc = [0] * (m + 1)
        for q in range(m): pc[q + 1] = pc[q] + (live[q] is not None)
        return lambda lo, hi: pc[hi] - pc[lo]
    if name in ("min", "max"):
        if any(isinstance(v, (list, tuple, dict)) for v in live): return None
        better = (lambda x, y: x if F.cmp_values(x, y) <= 0 else y) if name == "min" else (lambda x, y: x if F.cmp_values(x, y) >= 0 else y)
        def best(x, y):
            if x is None: return y
            if y is None: return x
            return better(x, y)
        st = [live]
        k = 1
        while 2 * k <= m:
            prev = st[-1]
            st.append([best(prev[j], prev[j + k]) for j in range(m - 2 * k + 1)])
            k *= 2
        def q(lo, hi):
            if hi <= lo: return None
            lv = (hi - lo).bit_length() - 1
            return best(st[lv][lo], st[lv][hi - (1 << lv)])
        return q
    nn = [v for v in live if v is not None]
    if any(isinstance(v, (bool, T.DecFloat, T.Interval)) or not isinstance(v, (int, float, Decimal)) for v in nn): return None
    kind = "float" if any(isinstance(v, float) for v in nn) else ("dec" if any(isinstance(v, Decimal) for v in nn) else "int")
    if kind == "float" and not all(math.isfinite(v) for v in nn): return None        # NaN / infinities: the aggregate's own sum
    from fractions import Fraction
    ps = [0] * (m + 1); pc = [0] * (m + 1)
    for q in range(m):
        v = live[q]
        ps[q + 1] = ps[q] + (0 if v is None else (v if kind == "int" else Fraction(v)))
        pc[q + 1] = pc[q] + (v is not None)
    def out(lo, hi):
        n = pc[hi] - pc[lo]
        if n == 0: return None
        tot = ps[hi] - ps[lo]
        if name == "sum":
            if kind == "int": return tot
            if kind == "float": return F.ffloat(tot)
            return T.CTX.divide(Decimal(tot.numerator), Decimal(tot.denominator))
        if kind == "float": return F.ffloat(tot) / n
        return F.ffloat(Fraction(tot) / n)
    return out


def _has_nwf(e):
    if isinstance(e, I.Func) and e.fn in ("__nwf_row_number", "__nwf_value_of"): return True
    if isinstance(e, I.Subq): return e.child is not None and _has_nwf(e.child)     # x IN (subquery): its operand
    return any(_has_nwf(c) for c in e.children())


def _marker_row(marker, p, q, frame, m):
    """the position in the ordered partition a row marker denotes, for current row p and frame row q"""
    if marker == "CURRENT_ROW": return p
    if marker == "FRAME_ROW": return q
    if marker == "BEGIN_PARTITION": return 0
    if marker == "END_PARTITION": return m - 1
    if marker == "BEGIN_FRAME": return frame[0] if frame else None
    return frame[-1] if frame else None                    # END_FRAME


def _nested_aggregate(w, idx, frames, res, rel, cx, m):
    """a window aggregate whose argument or FILTER holds ROW_NUMBER(<row marker>) or VALUE_OF(... AT <row marker>)
    (ISO 6.10, T619): the argument is evaluated once per (current row, frame row) pair; a column outside the nested
    functions reads the frame row, VALUE_OF reads its expression at the marked row of the partition (the default
    outside it)"""
    pairs = [(p, q) for p, fr in enumerate(frames) for q in fr]
    prel = Rel(len(pairs))
    def fill(e):
        k = e.key()
        if k in prel.cols: return
        if isinstance(e, I.Func) and e.fn in ("__nwf_row_number", "__nwf_value_of"):
            rows = [_marker_row(e.extra[0], p, q, frames[p], m) for p, q in pairs]
            if e.fn == "__nwf_row_number":
                prel.cols[k] = [None if r is None else r + 1 for r in rows]; return
            vals = ev(e.args[0], rel, cx); d = e.args[1].v
            fill(e.args[2]); dflt = prel.cols[e.args[2].key()]
            out = []
            for j, r in enumerate(rows):
                r = None if r is None or d is None else r + d
                out.append(vals[idx[r]] if r is not None and 0 <= r < m else dflt[j])
            prel.cols[k] = out; return
        if not _has_nwf(e):
            col = ev(e, rel, cx)
            prel.cols[k] = [col[idx[q]] for _, q in pairs]; return
        for c in e.children(): fill(c)
    main = _win_args(w)
    for a in main: fill(a)
    if w.filter is not None: fill(w.filter)
    for o in w.agg_order: fill(o[0])
    okeys = [_ckeys(ev(o[0], prel, cx), o[0].t) for o in w.agg_order]
    ospecs = [(o[1], o[2]) for o in w.agg_order]
    argv = [ev(a, prel, cx) for a in main]
    fv = ev(w.filter, prel, cx) if w.filter is not None else None
    for col in argv + ([fv] if fv is not None else []):
        for v in col:
            if isinstance(v, Err): raise SqlError(v.msg)
    d = F.AGGS[w.name]
    j = 0
    for p, i in enumerate(idx):
        rows = []
        js = list(range(j, j + len(frames[p]))); j += len(frames[p])
        if okeys: js = sort_rows(js, okeys, ospecs)              # the aggregate's own ORDER BY, per frame
        for jj in js:
            if fv is None or fv[jj] is True:
                vals = tuple(c[jj] for c in argv)
                if not (d.skip_nulls and w.name != "count_star" and any(v is None for v in vals)): rows.append((vals, 1))
        if w.distinct:
            seen, dd = set(), []
            for r in rows:
                k = tuple(nkey(v) for v in r[0])
                if k not in seen: seen.add(k); dd.append(r)
            rows = dd
        res[i] = len(rows) if w.name == "count_star" else d.final(rows, w.params)


def _frame_edge(mode, bd, side, p, i, idx, okeys, specs, pstart, pend, bound_vals, m, aux=None):
    """frame start (side a, inclusive) or end (side b, exclusive) as a position in the ordered partition"""
    kind = bd[0]
    if kind == "unbounded_preceding": return 0
    if kind == "unbounded_following": return m
    if kind == "current":
        if mode == "rows": return p if side == "a" else p + 1
        return pstart[p] if side == "a" else pend[p]
    off = bound_vals[side][i]
    if off is None: raise SqlError("frame offset must not be NULL")
    if mode == "rows":
        if off < 0: raise SqlError("frame offset must not be negative")
        q = p - off if kind == "preceding" else p + off
        return q if side == "a" else q + 1
    if mode == "groups":
        if off < 0: raise SqlError("frame offset must not be negative")
        gstarts = aux["gstarts"]; gi = aux["gindex"][pstart[p]]
        tgt = gi - off if kind == "preceding" else gi + off
        if side == "a":
            if tgt < 0: return 0
            if tgt >= len(gstarts): return m
            return gstarts[tgt]
        if tgt < 0: return 0
        if tgt >= len(gstarts): return m
        return pend[gstarts[tgt]]
    # RANGE with an offset: one ORDER BY key, values within off of the current one
    if len(okeys) != 1: raise SqlError("RANGE with an offset needs exactly one ORDER BY column")
    key = okeys[0]; desc = specs[0][0]
    x = key[i]
    if x is None:
        return pstart[p] if side == "a" else pend[p]
    sgn = -1 if kind == "preceding" else 1
    if desc: sgn = -sgn
    tgt = F._add(x, off) if sgn > 0 else F._sub(x, off)
    # the rows with non-NULL keys are one block of the ordered partition, sorted: a binary search there
    nk = aux["nn_keys"]; base = aux["nn_lo"]
    return base + _count_before(nk, tgt, desc, strict=(side == "a"))


# ------------------------------------------------------------------ subqueries
def eval_subquery(e, rel, cx):
    q = e.q
    if not e.correlated:
        k = ("u", id(e))
        if k not in cx.memo: cx.memo[k] = _subq_result(e, run_query(q, cx))
        r = cx.memo[k]
        return [_subq_apply(e, r, x) for x in (ev(e.child, rel, cx) if e.child is not None else [None] * rel.n)]
    # correlated: once per distinct tuple of the outer values it reads
    refs = _outer_cols(e)
    cols = []
    for c in refs:
        v = rel.cols.get(c.key())
        if v is None: v = [cx.outer_value(I.Outer(c, 1))] * rel.n
        cols.append(v)
    child = ev(e.child, rel, cx) if e.child is not None else [None] * rel.n
    out = []
    for i in range(rel.n):
        vals = tuple(c[i] for c in cols)
        k = ("c", id(e), tuple(nkey(v) for v in vals))
        r = cx.memo.get(k)
        if r is None:
            cx.outers.append({c.key(): v for c, v in zip(refs, vals)})
            try: r = _subq_result(e, run_query(q, cx))
            finally: cx.outers.pop()
            cx.memo[k] = r
        out.append(_subq_apply(e, r, child[i]))
    return out


def _outer_cols(e):
    """the enclosing queries' columns a correlated subquery reads"""
    found, seen = {}, set()
    def vx(x):
        if isinstance(x, I.Outer): found[x.col.key()] = x.col
        if isinstance(x, I.Subq): vq(x.q)
    def vq(q):
        if id(q) in seen: return
        seen.add(id(q))
        if isinstance(q, I.Select):
            exprs = [x for x, _ in q.items] + list(q.group) + [q.where, q.having, q.qualify] + \
                    [o[0][1] for o in q.order if o[0][0] == "expr"] + [j.cond for j in join_conds(q.from_)]
            for x in exprs:
                if x is not None: I.walk(x, vx)
            for b in block_bindings(q.from_):
                if b.q is not None: vq(b.q)
        elif isinstance(q, I.SetOp): vq(q.left); vq(q.right)
        elif isinstance(q, I.Recursive): vq(q.base); vq(q.step)
        elif isinstance(q, I.Values):
            for r in q.rows:
                for x in r: I.walk(x, vx)
        elif isinstance(q, I.TableFn):
            for a in q.args: I.walk(a, vx)
            if getattr(q, 'input', None) is not None: vq(q.input)
    vq(e.q)
    return list(found.values())


def _subq_result(e, r):
    if e.kind == "scalar":
        if len(r.rows) > 1: raise SqlError("More than one row returned by a subquery used as an expression")
        return r.rows[0][0] if r.rows else None
    if e.kind == "exists": return bool(r.rows)
    if e.kind == "unique":
        seen = set()
        for row in r.rows:
            if any(v is None for v in row): continue
            k = tuple(nkey(v) for v in row)
            if k in seen: return False
            seen.add(k)
        return True
    if e.kind == "any":
        return [row[0] if len(row) == 1 else list(row) for row in r.rows]
    if e.kind == "match": return list(r.rows)
    if e.kind == "array": return [row[0] for row in r.rows]
    raise SqlError(e.kind)


def _match(e, rows, x):
    """row MATCH [UNIQUE] SIMPLE | PARTIAL | FULL (rows) (ISO 8.13): TRUE or FALSE, never UNKNOWN"""
    kind, uniq = e.op
    nulls = [v is None for v in x]
    if kind == "SIMPLE" and any(nulls): return True
    if kind in ("PARTIAL", "FULL") and all(nulls): return True
    if kind == "FULL" and any(nulls): return False
    hits = 0
    for row in rows:
        if all(v is None or (w is not None and F.cmp_values(v, w) == 0) for v, w in zip(x, row)):
            hits += 1
            if not uniq: return True
            if hits > 1: return False
    return hits == 1 if uniq else False


def _subq_apply(e, r, x):
    if e.kind == "match": return x if isinstance(x, Err) else _match(e, r, x)
    if e.kind != "any": return r
    if isinstance(x, Err): return x
    op = F.SCALARS[e.op].impl
    out = False
    xs = x
    for v in r:
        if isinstance(v, list):                      # row value IN (subquery)
            parts = []
            for a, b in zip(xs if isinstance(xs, (list, tuple)) else [xs], v):
                parts.append(None if a is None or b is None else F.cmp_values(a, b) == 0)
            res = F._and(*parts)
        elif x is None or v is None: res = None
        else: res = op(x, v)
        if res is True: return True
        if res is None: out = None
    return out


def fold_outer(e, cx):
    """replace correlated references by their current values (constants while the subquery runs once)"""
    if not cx.outers: return e
    def f(x):
        if isinstance(x, I.Outer):
            try: v = cx.outer_value(x)
            except SqlError: return None
            return I.Lit(v, x.t)
        return None
    return I.subst(e, f)


def fold_subqueries(e, cx):
    """uncorrelated subqueries -> constants (scalar, EXISTS, UNIQUE) or value lists (x op ANY (...))"""
    def f(x):
        if isinstance(x, I.Subq) and not x.correlated:
            k = ("u", id(x))
            if k not in cx.memo: cx.memo[k] = _subq_result(x, run_query(x.q, cx))
            r = cx.memo[k]
            if x.kind in ("scalar", "exists", "unique"): return I.Lit(r, x.t)
            if x.kind == "any" and not isinstance(x.child, I.Func) and all(not isinstance(v, list) for v in r):
                child = I.subst(x.child, f)
                return any_rewrite(child, x.op, r)
        return None
    return I.subst(e, f)


def any_rewrite(x, op, vals):
    """x op ANY (values) as an ordinary expression with the same three-valued result"""
    nn = [v for v in vals if v is not None]
    has_null = len(nn) < len(vals)
    if not vals: return I.Lit(False, T.BOOLEAN)
    if op == "=":
        lits = [I.Lit(v, x.t) for v in _uniq(nn)] + ([I.Lit(None, x.t)] if has_null else [])
        return I.Func("in", [x] + lits, T.BOOLEAN)
    if not nn: return I.Lit(None, T.BOOLEAN)
    if op in ("<", "<="): pick = max(nn, key=functools.cmp_to_key(F.cmp_values))
    elif op in (">", ">="): pick = min(nn, key=functools.cmp_to_key(F.cmp_values))
    elif op == "<>":
        u = _uniq(nn)
        if len(u) >= 2: base = I.Case([(I.Func("is_not_null", [x], T.BOOLEAN), I.Lit(True, T.BOOLEAN))], I.Lit(None, T.BOOLEAN), T.BOOLEAN)
        else: base = I.Func("<>", [x, I.Lit(u[0], x.t)], T.BOOLEAN)
        return base if not has_null else I.Func("or", [base, I.Lit(None, T.BOOLEAN)], T.BOOLEAN)
    else: raise SqlError(f"{op} ANY is not supported")
    base = I.Func(op, [x, I.Lit(pick, x.t)], T.BOOLEAN)
    return base if not has_null else I.Func("or", [base, I.Lit(None, T.BOOLEAN)], T.BOOLEAN)


def _uniq(vals):
    seen, out = set(), []
    for v in vals:
        k = nkey(v)
        if k not in seen: seen.add(k); out.append(v)
    return out


# ================================================================== the star executor (engine queries)
class NotStar(Exception):
    pass


def _strip_cast(e):
    """a column under value-preserving conversions (widening numeric casts, DATE -> TIMESTAMP)"""
    while isinstance(e, I.Cast) and not e.try_:
        a, t = e.arg.t, e.t
        if a.integer and (t.integer or t.id == "DECIMAL" or t.floating) or (a.id == "DECIMAL" and (t.id == "DECIMAL" and t.s >= a.s or t.floating)) \
                or (a.id == "DATE" and t.id == "TIMESTAMP") or (a.id == "VARCHAR" and t.id == "VARCHAR"):
            e = e.arg
        else: break
    return e


def _fk_info(col):
    """(dimension table name, its key column) when col is a stored foreign key"""
    t = col.b.table
    if t is None: return None
    cd = getattr(t, "codecs", {}).get(col.name)
    if cd is not None and getattr(cd, "kind", "") == "ref": return cd.dim, cd.pk
    rd = getattr(t, "_refdim", {}).get(col.name)
    if rd is not None and col.name in getattr(t, "refbit", {}):
        return rd.name, getattr(rd, "pk", None) or "id"
    return None


def _edges(conj):
    """fact.fk = dim.pk equalities among conjuncts: (conjunct, fact col, dim col)"""
    out = []
    for c in conj:
        if isinstance(c, I.Func) and c.fn == "=" and len(c.args) == 2:
            a, b = _strip_cast(c.args[0]), _strip_cast(c.args[1])
            if isinstance(a, I.Col) and isinstance(b, I.Col) and a.b is not b.b:
                for x, y in ((a, b), (b, a)):
                    fi = _fk_info(x)
                    if fi and y.b.table is not None and fi[0] == y.b.table.name and fi[1] == y.name:
                        out.append((c, x, y))
    return out


def _pk_of(t):
    return getattr(t, "pk", None)


class StarPlan:
    def __init__(self, s, bindings, cx):
        self.s, self.cx = s, cx
        self.bindings = bindings
        self.tables = {}                    # binding -> algq Table (aliases for a table used twice)
        self.fkdim = {}                     # fk name -> dim binding
        self.hidden = []                    # implicit joins for fact FK columns read as values


def plan_star(s, bindings, need, aggs, cx):
    """run the block's star join on the engine; None when the FROM clause is not a star join"""
    joins = join_conds(s.from_)
    left = {}                                         # LEFT JOIN dim: binding -> its ON conjuncts
    for j in joins:
        if getattr(j, "partition", None) is not None: return None, None
        if j.kind in ("inner", "cross"): continue
        if j.kind == "left" and isinstance(j.right, I.Binding) and j.right.table is not None:
            left[j.right] = _conjuncts(j.cond); continue
        return None, None
    conj = []
    for j in joins:
        if j.kind in ("inner", "cross"): conj += _conjuncts(j.cond)
    conj += _conjuncts(s.where)
    edges = _edges(conj + [c for cs in left.values() for c in cs])
    # the fact: every other binding is joined to it through one of its foreign keys
    fact, used = None, None
    cands = sorted(bindings, key=lambda b: -b.table.n)
    for f in cands:
        dims, ok, fks = {}, True, {}
        for b in bindings:
            if b is f: continue
            e = next((e for e in edges if e[1].b is f and e[2].b is b and e[1].name not in fks), None)
            if e is None: ok = False; break
            dims[b] = e; fks[e[1].name] = b
        if ok: fact, used = f, (dims, fks); break
    if fact is None or fact in left: return None, None
    dims, fks = used
    if any(b.table is fact.table for b in dims): return None, None
    consumed = {id(e[0]) for e in dims.values()}
    lefts = {}
    for b, cs in left.items():                        # ON: the key equality, fact-only and dimension-only terms
        e = dims.get(b)
        if e is None or id(e[0]) not in {id(c) for c in cs}: return None, None
        cf, cd = [], []
        for c in cs:
            if c is e[0]: continue
            bs = {x.b for x in I.cols_of(c)}
            if I.has(c, I.Subq) or I.has(c, I.Outer): return None, None
            if bs <= {fact}: cf.append(c)
            elif bs <= {b}: cd.append(c)
            else: return None, None
        lefts[b] = (e[1].name, cf, cd)
    rest = [c for c in conj if id(c) not in consumed]
    if s.sample is not None or any(b.sample is not None for b in lefts): return None, None
    cx.stats["star_blocks"] += 1
    P = _Planner(s, fact, dims, fks, cx)
    P.left = lefts
    for b in dims:                                     # a sampled dimension: its kept rows, a row mask
        if b.sample is not None:
            t = P.tab[b]
            P.samples.append(algq.RowMask(t, sample_mask(t.n, t.used, b.sample, cx), (), sql="TABLESAMPLE"))
    if fact.sample is not None:                       # a sampled fact: its kept rows, ANDed with the passing rows
        P.fact_sample = sample_mask(fact.table.n, getattr(fact.table, "used", None), fact.sample, cx)
    try: return P.run(rest, need, aggs)
    except _NotStarNow:
        cx.stats["star_blocks"] -= 1
        return None, None


def _conjuncts(e):
    if e is None: return []
    if isinstance(e, I.Func) and e.fn == "and": return [x for a in e.args for x in _conjuncts(a)]
    return [e]


class _Planner:
    def __init__(self, s, fact, dims, fks, cx):
        self.s, self.fact, self.cx = s, fact, cx
        self.tab = {fact: fact.table}
        seen = {id(fact.table)}
        for b in dims:
            t = b.table
            if id(t) in seen:
                t = copy.copy(t)
            seen.add(id(t)); self.tab[b] = t
        self.joins = {b: e[1].name for b, e in dims.items()}       # dim binding -> fk name
        self.fkb = dict(fks)                                        # fk name -> dim binding
        self.hidden = {}                                            # fk name -> hidden binding for an implicit join
        self.left, self.nulled = {}, set()                          # LEFT JOIN dims; dims NULL in the current pass
        self.samples, self.fact_sample = [], None                   # TABLESAMPLE: dimension row masks; fact rows

    # columns -------------------------------------------------------------
    def canon(self, col):
        """the column the engine reads for col: a fact FK column is its raw values (when some references have no
        match) or its dimension's key (the stored surrogate is not the value); -> (binding, column name)"""
        if col.b is self.fact:
            fi = _fk_info(col)
            if fi is not None:
                if col.name + "#raw" in self.fact.table.cols: return self.fact, col.name + "#raw"
                b = self.fkb.get(col.name)
                if b is not None and b in self.nulled: b = None
                if b is None:
                    b = self.hidden.get(col.name)
                    if b is not None and b not in self.joins: self.joins[b] = col.name
                    if b is None:
                        orph = getattr(self.fact.table, "orphans", {}).get(col.name, 0)
                        if orph: raise SqlError(f"{self.fact.table.name}.{col.name} has {orph} reference(s) without a match; "
                                                "their values are not stored (join the referenced table instead)")
                        dimt = self.fact.table._refdim[col.name]
                        b = I.Binding(f"__{col.name}", [(fi[1], col.t, False)], table=dimt)
                        t = dimt
                        if any(x is dimt for x in self.tab.values()): t = copy.copy(dimt)
                        self.tab[b] = t; self.joins[b] = col.name; self.hidden[col.name] = b
                return b, fi[1]
        return col.b, col.name

    def algcol(self, col):
        b, name = self.canon(col)
        return algq.Col(self.tab[b], name)

    def binding_of(self, col): return self.canon(col)[0]

    def strip(self, e):
        """a column under conversions that keep its values (an exact column converted to DOUBLE only when every
        value it holds is a double: integers below 2^53, decimals of at most 15 digits)"""
        while isinstance(e, I.Cast) and not e.try_:
            a, t = e.arg.t, e.t
            if t.floating and a.exact:
                base = e.arg
                while isinstance(base, I.Cast): base = base.arg
                if not (isinstance(base, I.Col) and self.double_exact(base)): break
                e = e.arg; continue
            if _strip_cast(I.Cast(e.arg, t, False)) is e.arg or _strip_cast(e) is not e:
                nxt = _strip_cast(e)
                if nxt is e: break
                e = e.arg
                continue
            break
        return e

    def double_exact(self, col):
        t = col.t
        if t.id == "DECIMAL": return t.p <= 15
        if not t.integer: return False
        b, name = self.canon(col)
        tb = self.tab[b]; cd = getattr(tb, "codecs", {}).get(name)
        mx = tb.maxcode(name) if hasattr(tb, "maxcode") else tb.colmax(name)
        off = cd.offset if cd is not None and hasattr(cd, "offset") else 0
        return max(abs(off), abs(off + mx)) < 2 ** 53

    # predicates ------------------------------------------------------------
    def native(self, e):
        """an IR predicate as a plane predicate that holds where e is TRUE (algq Pred), or None"""
        tf = self.native_tf(e)
        return None if tf is None else tf[0]

    def valid(self, col):
        """the plane predicate 'col IS NOT NULL' (None: the column has no NULLs)"""
        b, name = self.canon(col)
        t = self.tab[b]
        vc = getattr(t, "nullable", {}).get(name)
        if vc is None: return None
        return algq.Cmp(algq.Col(t, vc), "EQ", 1)

    def native_tf(self, e):
        """(TRUE, FALSE) plane predicates of a three-valued predicate: where e is TRUE and where it is FALSE (rows where
        it is UNKNOWN satisfy neither); None when e is not a plane predicate"""
        def both(t, f): return (t, f)
        def neg(p): return algq.Not(p)
        def AND(a, b_): return algq.BoolOp("AND", a, b_)
        def OR(a, b_): return algq.BoolOp("OR", a, b_)
        def guard(col, p):                         # p on the codes, where col IS NOT NULL
            v = self.valid(col)
            return (p, neg(p)) if v is None else (AND(v, p), AND(v, neg(p)))
        if isinstance(e, I.Func):
            fn = e.fn
            if fn in ("and", "or"):
                parts = [self.native_tf(a) for a in e.args]
                if any(p is None for p in parts): return None
                T_, F_ = parts[0]
                for t, f in parts[1:]:
                    T_, F_ = (AND(T_, t), OR(F_, f)) if fn == "and" else (OR(T_, t), AND(F_, f))
                return T_, F_
            if fn == "not":
                p = self.native_tf(e.args[0]); return None if p is None else (p[1], p[0])
            if fn in ("is_true", "is_false", "is_not_true", "is_not_false", "is_unknown", "is_not_unknown"):
                p = self.native_tf(e.args[0])
                if p is None: return None
                t, f = p
                return {"is_true": (t, neg(t)), "is_false": (f, neg(f)), "is_not_true": (neg(t), t), "is_not_false": (neg(f), f),
                        "is_unknown": (AND(neg(t), neg(f)), OR(t, f)), "is_not_unknown": (OR(t, f), AND(neg(t), neg(f)))}[fn]
            if fn in ("=", "<>", "<", "<=", ">", ">="):
                a, b = self.strip(e.args[0]), self.strip(e.args[1])
                op = fn
                if isinstance(a, I.Lit) and isinstance(b, I.Col): a, b = b, a; op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)
                if isinstance(a, I.Col) and isinstance(b, I.Lit):
                    if b.v is None: f0 = self._false(a); return (f0, f0)          # comparison with NULL: UNKNOWN
                    v = self._lit_for(a, b)
                    if v is _NO: return None
                    c = self.algcol(a)
                    try:
                        p = {"=": c.__eq__, "<>": c.__ne__, "<": c.__lt__, "<=": c.__le__, ">": c.__gt__, ">=": c.__ge__}[op](v)
                    except (TypeError, ValueError): return None
                    return guard(a, p)
                return None
            if fn in ("between", "between_sym"):
                x, lo, hi = (self.strip(a) for a in e.args)
                if isinstance(x, I.Col) and isinstance(lo, I.Lit) and isinstance(hi, I.Lit):
                    if lo.v is None or hi.v is None: return None
                    a_, b_ = self._lit_for(x, lo), self._lit_for(x, hi)
                    if a_ is _NO or b_ is _NO: return None
                    c = self.algcol(x)
                    try:
                        if fn == "between_sym" and F.cmp_values(a_, b_) > 0: a_, b_ = b_, a_
                        return guard(x, c.between(a_, b_))
                    except (TypeError, ValueError): return None
                return None
            if fn == "in":
                x = self.strip(e.args[0])
                if isinstance(x, I.Col) and all(isinstance(self.strip(a), I.Lit) for a in e.args[1:]) and len(e.args) <= 65:
                    lits = [self.strip(a) for a in e.args[1:]]
                    has_null = any(l.v is None for l in lits)
                    vals = [self._lit_for(x, l) for l in lits if l.v is not None]
                    if any(v is _NO for v in vals): return None
                    if not vals: f0 = self._false(x); return (f0, f0 if has_null else (self.valid(x) or self._true(x)))
                    try: p = self.algcol(x).isin(vals)
                    except (TypeError, ValueError): return None
                    t, f = guard(x, p)
                    return (t, self._false(x) if has_null else f)
                return None
            if fn in ("like", "starts_with", "prefix") and len(e.args) == 2:
                x, p = self.strip(e.args[0]), e.args[1]
                if isinstance(x, I.Col) and isinstance(p, I.Lit) and isinstance(p.v, str) and x.t.id == "VARCHAR":
                    pat = p.v; q = None
                    try:
                        if fn == "like":
                            body = pat[:-1] if pat.endswith("%") else None
                            if body is not None and "%" not in body and "_" not in body: q = self.algcol(x).startswith(body)
                            elif "%" not in pat and "_" not in pat: q = self.algcol(x) == pat
                        else: q = self.algcol(x).startswith(pat)
                    except TypeError: return None
                    return None if q is None else guard(x, q)
                return None
            if fn in ("is_null", "is_not_null"):
                x = e.args[0]
                if isinstance(x, I.Col):
                    v = self.valid(x)
                    if v is None: v = self._true(x)
                    return (neg(v), v) if fn == "is_null" else (v, neg(v))
                return None
            return None
        if isinstance(e, I.Col) and e.t.id == "BOOLEAN":
            try: return guard(e, self.algcol(e) == True)
            except TypeError: return None
        return None

    def _lit_for(self, col, lit):
        """a constant in the column's terms (a string constant against a typed column is converted)"""
        v = lit.v
        if isinstance(v, Err): return _NO
        t = col.t
        if isinstance(v, T.Interval): return _NO
        if t.id == "VARCHAR" and not isinstance(v, str): return _NO
        if t.id != "VARCHAR" and isinstance(v, str):
            try: v = T.cast_value(v, T.VARCHAR, t)
            except SqlError: return _NO
        if t.id == "BOOLEAN" and not isinstance(v, bool): return _NO
        if t.numeric and not isinstance(v, (int, Decimal, float)) or isinstance(v, bool) and t.id != "BOOLEAN": return _NO
        if isinstance(v, float) and (v != v or v in (math.inf, -math.inf)): return _NO
        if t.id == "DATE" and not isinstance(v, (T._dt.date,)): return _NO
        if t.id == "TIMESTAMP" and not isinstance(v, (T._dt.date,)): return _NO
        return v

    def _false(self, col):
        c = self.algcol(col); return algq.Cmp(c, "LT", 0)
    def _true(self, col):
        return algq.Not(self._false(col))

    def tables_of(self, e):
        out = set()
        for c in I.cols_of(e): out.add(self.binding_of(c))
        return out

    def rowmask(self, b, e):
        """a predicate on one dimension: evaluated on the dimension's distinct value tuples -> its passing rows"""
        t = self.tab[b]; cols = I.cols_of(e)
        names = sorted({self.canon(c)[1] for c in cols})
        nb = getattr(t, "nullable", {})
        vcols = [nb[c] for c in names if c in nb]
        vnames = {c: len(names) + i for i, c in enumerate(c for c in names if c in nb)}
        n = t.n
        if not names:
            v = ev(e, Rel(1), self.cx)[0]
            mask = np.full(n, v is True)
        else:
            codes = np.stack([t.decoded(c).astype(np.int64) for c in names + vcols], 1)
            u, inv = np.unique(codes, axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            r = Rel(len(u))
            for j, c in enumerate(names):
                cd = t.codecs.get(c) if getattr(t, "codecs", None) else None
                vals = cd.decode(u[:, j]) if cd is not None else u[:, j].tolist()
                if c in vnames: vals = [None if not ok_ else x for x, ok_ in zip(vals, u[:, vnames[c]].tolist())]
                for col in cols:
                    if self.canon(col)[1] == c: r.cols[col.key()] = vals
            res = ev(e, r, self.cx)
            for x in res:
                if isinstance(x, Err): raise SqlError(x.msg)
            ok = np.array([x is True for x in res], bool)
            mask = ok[inv]
        mask &= t.used[:n] if len(t.used) >= n else mask
        return algq.RowMask(t, mask, names + vcols, sql=I.render(e))     # the selection is a function of these columns

    def codeset(self, col, e):
        """a predicate on one fact column: evaluated on every code the column can hold -> intervals or a code set
        (returns a list of engine predicates, or None)"""
        b, name = self.canon(col)
        t = self.tab[b]
        if name in getattr(t, "wide", {}): return None
        mx = t.maxcode(name) if hasattr(t, "maxcode") else t.colmax(name)
        cd = t.codecs.get(name) if getattr(t, "codecs", None) else None
        if cd is not None and getattr(cd, "kind", "") in ("dict", "rank"):
            mx = min(mx, len(cd.values if cd.kind == "dict" else cd.raw) - 1)
        c = algq.Col(t, name)
        v = self.valid(col)
        # the value the predicate takes on a NULL
        r0 = Rel(1)
        for x in I.cols_of(e): r0.cols[x.key()] = [None]
        at_null = ev(e, r0, self.cx)[0] if v is not None else None
        if isinstance(at_null, Err): at_null = None
        if mx < 0 or t.n == 0:                         # no values stored: only NULLs (or no rows)
            return [algq.Not(v)] if (at_null is True and v is not None) else [algq.Cmp(c, "LT", 0)]
        def present():
            """the codes the column stores (cached): the values the predicate is ever applied to"""
            pc = t.__dict__.setdefault("_present_codes", {})
            if name not in pc:
                cs_ = np.asarray(t.decoded(name), np.int64)
                vc_ = getattr(t, "nullable", {}).get(name)            # NULL rows store code 0: not a value
                if vc_ is not None: cs_ = cs_[np.asarray(t.decoded(vc_)) != 0]
                pc[name] = np.unique(cs_)
            return pc[name]
        def run(codes):
            vals = cd.decode(codes) if cd is not None else codes.tolist()
            r = Rel(len(vals))
            for x in I.cols_of(e): r.cols[x.key()] = vals
            return ev(e, r, self.cx)
        if mx + 1 > 4 * t.n + 1024:
            # a code domain much wider than the rows (a timestamp range, a sparse key): the codes present, each
            # run of them that passes an interval up to the next present code
            pcs = present()
            res = run(pcs)
            for x in res:
                if isinstance(x, Err): raise SqlError(x.msg)
            okp = np.array([x is True for x in res], bool)
            chg = np.flatnonzero(np.diff(okp.astype(np.int8)) != 0) + 1
            bounds = np.concatenate(([0], chg, [len(pcs)]))
            lo_c = np.where(bounds[:-1] == 0, 0, pcs[np.minimum(bounds[:-1], len(pcs) - 1)])
            hi_c = np.where(bounds[1:] == len(pcs), mx, pcs[np.minimum(bounds[1:], len(pcs) - 1)] - 1)
            sel = okp[bounds[:-1]] if len(pcs) else np.zeros(0, bool)
            starts, ends = lo_c[sel], hi_c[sel]
            if len(starts) > 8:
                if mx >= 1 << 22: return None
                ok = np.zeros(mx + 1, bool)
                for a_, b_ in zip(starts.tolist(), ends.tolist()): ok[a_:b_ + 1] = True
        else:
            codes = np.arange(mx + 1, dtype=np.int64)
            res = run(codes)
            bad = [i for i, x in enumerate(res) if isinstance(x, Err)]
            if bad:                                    # an error matters only for a code the column holds
                hit = np.intersect1d(np.array(bad, np.int64), present())
                if len(hit): raise SqlError(res[int(hit[0])].msg)
            ok = np.array([x is True for x in res], bool)
            edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
            starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
        p = None
        if len(starts) <= 8:
            for a_, b_ in zip(starts.tolist(), ends.tolist()):
                q = algq._code_pred(c, (a_, b_))
                p = q if p is None else algq.BoolOp("OR", p, q)
            if p is None: p = algq.Cmp(c, "LT", 0)
        if at_null is True:                           # NULL rows pass too: (NOT valid) OR (the codes)
            if p is None: return None
            out = algq.BoolOp("OR", algq.Not(v), p)
            out.sql = I.render(e); return [out]
        if p is not None:
            if v is not None: p = algq.BoolOp("AND", v, p)
            p.sql = I.render(e)
            return [p]
        out = [algq.CodeSet(c, ok, sql=I.render(e))]
        return out + ([v] if v is not None else [])

    # aggregates ------------------------------------------------------------
    def poly(self, e):
        """an exact polynomial (algq expression) of e, or None (NULLs are excluded by the caller's variant)"""
        e0 = e
        if isinstance(e, I.Cast) and not e.try_:
            a, t = e.arg.t, e.t
            if a.exact and (t.integer and a.integer and True or t.id == "DECIMAL" and (a.integer or a.s <= t.s)):
                return self.poly(e.arg)
            return None
        if isinstance(e, I.Lit):
            if isinstance(e.v, bool) or e.v is None: return None
            if isinstance(e.v, int): return algq.Const(e.v)
            if isinstance(e.v, Decimal): return algq.Const(e.v)
            return None
        if isinstance(e, I.Col):
            if not e.t.exact: return None
            c = self.algcol(e)
            cd = c.codec()
            if cd is not None and getattr(cd, "kind", "") == "ref": return None
            if cd is not None and cd.affine is None and algq._bridge_col(c) != "hide": return None
            return c
        if isinstance(e, I.Func):
            if e.fn in ("+", "-", "*") and len(e.args) == 2:
                if not (e.args[0].t.exact and e.args[1].t.exact): return None
                l, r = self.poly(e.args[0]), self.poly(e.args[1])
                if l is None or r is None: return None
                return algq.Arith(e.fn, l, r)
            if e.fn == "neg":
                a = self.poly(e.args[0]); return None if a is None else algq.Arith("-", algq.Const(0), a)
        return None

    def poly_of_tuple(self, m):
        if isinstance(m, tuple):
            a, b = self.poly_of_tuple(m[1]), self.poly_of_tuple(m[2])
            return algq.Arith("*", a, b)
        return self.poly(m)

    # the run ---------------------------------------------------------------
    def run(self, rest, need, aggs):
        """the block's rows on the engine. A LEFT JOIN runs as disjoint passes: the matched rows, and the rows
        without a match (no ON match on the fact side, a reference without a dimension row, a dimension row failing
        the ON terms), the dimension's columns NULL in the latter. Each aggregate's per-group state (counts,
        moment sums, extremes) is kept per row of the result, so that the passes add up."""
        joins0 = dict(self.joins)
        configs = [dict(joined=dict(joins0), nulled=set(), extra=list(self.samples))]
        for b, (fk, cf, cd) in self.left.items():
            tf = None
            if cf:
                # fact-side ON terms: a foreign key read there is its own value (a hidden join, never the joined dim)
                # (of every LEFT JOIN's dimension: a pass that leaves one of them unmatched still reads its key)
                self.nulled = {b} | set(self.left)
                try: tfs = [self.native_tf(c) for c in cf]
                except SqlError: raise _NotStarNow()            # references without a match: no hidden join
                finally: self.nulled = set()
                if any(x is None for x in tfs): raise _NotStarNow()
                T_ = tfs[0][0]
                for x in tfs[1:]: T_ = algq.BoolOp("AND", T_, x[0])
                tf = T_
            md = self.rowmask(b, _and(cd)) if cd else None
            nd = None
            if md is not None:
                t = self.tab[b]; mask = np.unpackbits(md.bm.view(np.uint8), bitorder="little")[:t.n].astype(bool)
                nd = algq.RowMask(t, ~mask & t.used[:t.n], md.cols, sql="NOT " + (md.sql or ""))
            orphans = getattr(self.fact.table, "orphans", {}).get(fk, 0)
            new = []
            for c in configs:
                base_extra = c["extra"] + ([tf] if tf is not None else [])
                m = dict(joined=dict(c["joined"]), nulled=set(c["nulled"]), extra=base_extra + ([md] if md is not None else []))
                new.append(m)                                        # matched
                j2 = dict(c["joined"]); j2.pop(b, None)
                if tf is not None:
                    new.append(dict(joined=dict(j2), nulled=c["nulled"] | {b}, extra=c["extra"] + [algq.Not(tf)]))
                if orphans:
                    new.append(dict(joined=dict(j2), nulled=c["nulled"] | {b}, extra=base_extra + [algq.RefOk(self.tab[self.fact], fk, False)]))
                if nd is not None:
                    new.append(dict(joined=dict(c["joined"]), nulled=c["nulled"] | {b}, extra=base_extra + [nd]))
            configs = new
        parts, plans = [], []
        for c in configs:
            self.joins = dict(c["joined"]); self.nulled = set(c["nulled"])
            for fk, hb in self.hidden.items(): self.joins[hb] = fk         # implicit joins drop no row (no orphans)
            rel, plan = self.run_pass(rest, need, aggs, c["extra"])
            parts.append(rel); plans.append(plan)
        self.joins, self.nulled = joins0, set()
        if len(parts) == 1: return parts[0], plans[0]
        return _concat(parts), plans[0]

    def nullify(self, e):
        """e with the columns of the dimensions that are NULL in this pass replaced by NULL (and constants folded)"""
        if not self.nulled or e is None: return e
        def f(x):
            if isinstance(x, I.Col) and x.b in self.nulled: return I.Lit(None, x.t)
            return None
        return _fold_consts(I.subst(e, f))

    def run_pass(self, rest, need, aggs, extra):
        s, cx = self.s, self.cx
        first_pass = not getattr(self, "_ran_pass", False); self._ran_pass = True
        preds = list(extra)            # engine predicates (algq), conjunction
        residual = []                  # IR conjuncts evaluated per group
        empty = False
        for c in rest:
            c = self.nullify(fold_subqueries(fold_outer(c, cx), cx))
            if isinstance(c, I.Lit):
                if c.v is True: continue
                empty = True; continue
            p = self.engine_pred(c)
            if p is not None: preds += p
            else: residual.append(c)
        atoms = {k: c for k, c in need.items() if c.b not in self.nulled}
        nulled_atoms = {k: c for k, c in need.items() if c.b in self.nulled}
        for c in residual:
            for col in I.cols_of(c): atoms[col.key()] = col
            for sq in subqueries_in([c]):
                for col in outer_refs_into(sq.q, set(self.tab)): atoms[col.key()] = col
        # aggregates: moments (polynomial arguments) or values (their columns become atoms)
        variants = {(): 0}              # extra engine predicates (FILTER, IS NOT NULL) -> variant id
        vpreds = {0: []}
        moments = {}                    # (variant, moment index) registry: list of algq polynomials per variant
        mlist = {0: []}
        mkeys = {0: {}}
        aggplan = {}
        modes = self.__dict__.setdefault("_mode", {})          # per aggregate, fixed by the first pass: state | value
        for k, a in aggs.items():
            if modes.get(k) != "value":
                a2 = self.nullify_agg(a)
                if a2 is None:                      # its arguments are NULL in this pass: nothing to aggregate here
                    if k not in modes: modes[k] = "value" if self.agg_plan(a, {(): 0}, {0: []}, {0: []}, {0: {}}) is None else "state"
                    if modes[k] == "state": aggplan[k] = dict(kind="zero"); continue
                    plan = None
                else:
                    plan = self.agg_plan(a2, variants, vpreds, mlist, mkeys)
                    if plan is None and modes.get(k) == "state": raise _NotStarNow()
                    modes.setdefault(k, "state" if plan is not None else "value")
            else: plan = None
            if plan is None:
                modes[k] = "value"
                for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                    for col in I.cols_of(x):
                        if col.b in self.nulled: nulled_atoms[col.key()] = col
                        else: atoms[col.key()] = col
                    for sq in subqueries_in([x]):
                        for col in outer_refs_into(sq.q, set(self.tab)): atoms[col.key()] = col
            else:
                aggplan[k] = plan
                if plan.get("rowfilter_expr") is not None:
                    for col in I.cols_of(plan["rowfilter_expr"]): atoms[col.key()] = col
        # engine keys: one per distinct (binding, column)
        keys, kpos, vpos, atom_list = [], {}, {}, []
        for k, col in atoms.items():
            cb = self.canon(col)
            if cb not in kpos:
                kpos[cb] = len(keys); keys.append(self.algcol(col))
                vp = self.valid(col)
                if vp is not None: vpos[cb] = len(keys); keys.append(vp.col)     # NULL: the validity is a key too
            atom_list.append((k, kpos[cb], vpos.get(cb)))
        nkeys = len(keys) + sum(1 for c in {self.canon(c): c for c in atoms.values()}.values()
                                if self.canon(c)[0] is self.fact and self.canon(c)[1] in getattr(self.fact.table, "wide", {}))
        def to_values():
            """every aggregate from its arguments' values in every pass: the passes of a block agree on how each
            aggregate is computed (a later pass cannot switch an aggregate an earlier one computed as a state)"""
            if not first_pass and any(modes.get(k) == "state" for k in aggs): raise _NotStarNow()
            for k in aggs: modes[k] = "value"
        # a block without aggregates returns one row per fact row: read the passing rows' columns from the planes
        # (a row bitmap) instead of grouping them by every column it reads
        rows_block = ROWS_BLOCK and not s.aggregate and not aggs and (bool(s.items) or self.dense_atoms(atoms))
        if nkeys > algq.MAXKEY or self.fact_sample is not None or rows_block:
            # more columns than GROUP BY keys (or a sampled fact): the passing rows themselves (a row bitmap), their
            # columns read from the planes; every aggregate is then computed from its arguments' values
            to_values()
            for k, a in aggs.items():
                for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                    for col in I.cols_of(x):
                        (nulled_atoms if col.b in self.nulled else atoms)[col.key()] = col
            aggplan = {}
            rel = self.rows_rel(atoms, preds, empty)
        else:
            try: rel = self.engine(keys, atom_list, preds, empty, variants, vpreds, mlist)
            except _TooManyGroups:
                # the atoms' code domain passes 2^128: the passing rows themselves
                to_values()
                for k, a in aggs.items():
                    for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                        for col in I.cols_of(x):
                            (nulled_atoms if col.b in self.nulled else atoms)[col.key()] = col
                aggplan = {}
                rel = self.rows_rel(atoms, preds, empty)
        for k, col in nulled_atoms.items():               # the NULL side of a LEFT JOIN pass: NULL codes of the column's codec,
            t = self.tab.get(col.b)                        # so the passes concatenate as codes
            cd = getattr(t, "codecs", {}).get(col.name) if t is not None else None
            if COLUMNAR and cd is not None: rel.cols.put(k, ColArr(np.zeros(rel.n, np.int64), np.zeros(rel.n, bool), cd))
            else: rel.cols[k] = [None] * rel.n
        for k, plan in aggplan.items():
            if plan.get("rowfilter_expr") is not None:
                plan["rowfilter"] = set(truth(ev(plan["rowfilter_expr"], rel, cx)))
        if residual:
            keep = rows_where(_and(residual), rel, cx)
            rel = rel.take(keep)
            for plan in aggplan.values():
                if plan.get("rowfilter") is not None:
                    pos = {old: new for new, old in enumerate(keep)}
                    plan["rowfilter"] = {pos[i] for i in plan["rowfilter"] if i in pos}
        _states(rel, aggs, aggplan)
        return rel, {k: dict(kind="state") for k in aggs if k in aggplan}

    def case_plan(self, a, variants, vpreds, mlist, mkeys):
        """SUM / AVG / COUNT of a CASE: each branch is a variant of the engine queries (the rows where its condition is
        the first TRUE one), its value a moment polynomial or a constant; the result adds the branches up"""
        e = a.args[0]
        eps0 = []
        if a.filter is not None:
            for c in _conjuncts(fold_subqueries(fold_outer(a.filter, self.cx), self.cx)):
                if isinstance(c, I.Lit) and c.v is True: continue
                p = self.engine_pred(c) if not isinstance(c, I.Lit) else None
                if p is None: return None
                eps0 += p
        whens = list(e.whens) + [(None, e.else_)]
        branches, prior = [], []
        for cond, val in whens:
            preds = list(eps0)
            for pc in prior:
                p = self.engine_pred(I.Func("is_not_true", [pc], T.BOOLEAN))
                if p is None: return None
                preds += p
            if cond is not None:
                p = self.engine_pred(cond)
                if p is None:
                    if isinstance(cond, I.Lit): p = [] if cond.v is True else None
                    if p is None: return None
                preds += p
                prior.append(cond)
            if val is None or (isinstance(val, I.Lit) and val.v is None): continue        # NULL: nothing to add
            if isinstance(val, I.Lit):
                if isinstance(val.v, bool) or not isinstance(val.v, (int, Decimal)): return None
                const, poly = val.v, None
            else:
                if a.name != "count" and self.poly(val) is None: return None
                if a.name == "count" and not isinstance(val, I.Col) and self.poly(val) is None: return None
                const, poly = None, val
                for col in I.cols_of(val):
                    vp = self.valid(col)
                    if vp is not None: preds.append(vp)
            sig = ("sig",) + tuple(sorted(_pred_sig(p) for p in preds)) if preds else ()
            if sig not in variants:
                variants[sig] = len(variants); vpreds[variants[sig]] = list(preds); mlist[variants[sig]] = []; mkeys[variants[sig]] = {}
            v = variants[sig] if preds else 0
            mi = None
            if poly is not None and a.name != "count":
                k = poly.key()
                if k not in mkeys[v]: mkeys[v][k] = len(mlist[v]); mlist[v].append(self.poly(poly))
                mi = mkeys[v][k]
            branches.append((v, mi, const))
        return dict(kind="case", branches=branches)

    def nullify_agg(self, a):
        """the aggregate as it reads in this pass (None: every argument is NULL here, so nothing is aggregated)"""
        if not self.nulled: return a
        args = [self.nullify(x) for x in a.args]
        filt = self.nullify(a.filter) if a.filter is not None else None
        if a.name != "count_star" and any(isinstance(x, I.Lit) and x.v is None for x in args) and F.AGGS[a.name].skip_nulls:
            return None
        if isinstance(filt, I.Lit):
            if filt.v is not True: return None
            filt = None
        return I.Agg(a.name, args, a.t, a.distinct, filt, [(self.nullify(o[0]), o[1], o[2]) for o in a.order], a.params)

    def engine_pred(self, c):
        """an AND-term as engine predicates (a list, all must hold): plane predicate, dimension row mask, fact code
        set; None: residual"""
        if I.has(c, I.Subq) or I.has(c, I.Outer) or I.has(c, I.Agg) or I.has(c, I.Win): return None
        if not _deterministic(c): return None
        bs = self.tables_of(c)
        if len(bs) != 1: return None
        b = next(iter(bs))
        p = self.native(c)
        if p is not None:
            if p.sql is None: p.sql = I.render(c)
            return [p]
        if b is self.fact:
            cols = {self.canon(x) for x in I.cols_of(c)}
            if len(cols) == 1:
                return self.codeset(I.cols_of(c)[0], c)
            return None
        return [self.rowmask(b, c)]

    def agg_plan(self, a, variants, vpreds, mlist, mkeys):
        """how an aggregate is computed: from engine moments ({kind: moment, variant, moments}), from the engine's
        MIN / MAX by plane elimination ({kind: ext}), or None (from the values of its arguments per group). A variant
        is a set of extra engine predicates: the aggregate's FILTER and 'IS NOT NULL' of its nullable columns."""
        name = a.name
        if name in ("sum", "avg", "count") and not a.distinct and not a.order and isinstance(a.args[0], I.Case):
            return self.case_plan(a, variants, vpreds, mlist, mkeys)
        ext = name in ("min", "max") and len(a.args) == 1 and isinstance(a.args[0], I.Col) \
            and self.canon(a.args[0])[0] is self.fact and self.canon(a.args[0])[1] not in getattr(self.fact.table, "wide", {})
        if (a.distinct and not ext) or a.order: return None
        if name not in MOMENTS and name != "count_star" and not ext: return None
        eps, rowfilter = [], None
        if a.filter is not None:
            f = fold_subqueries(fold_outer(a.filter, self.cx), self.cx)
            for c in _conjuncts(f):
                if isinstance(c, I.Lit) and c.v is True: continue
                p = self.engine_pred(c) if not isinstance(c, I.Lit) else None
                if p is None: eps = None; break
                eps += p
            if eps is None:
                if I.has(f, I.Subq): return None
                rowfilter, eps = f, []
        if name != "count_star":
            if name == "count" and isinstance(a.args[0], I.Col): pass       # COUNT(column): its validity alone
            elif not ext:
                y = a.args[0]; x = a.args[1] if len(a.args) > 1 else None
                if self.poly(y) is None or (x is not None and self.poly(x) is None): return None
            seen = set()
            for arg in a.args:                         # NULL arguments are skipped: only rows where all are valid
                for col in I.cols_of(arg):
                    cb = self.canon(col)
                    if cb in seen: continue
                    seen.add(cb)
                    vp = self.valid(col)
                    if vp is not None: eps.append(vp)
        key = tuple(sorted(id(p) for p in eps))
        if eps:
            sig = tuple(sorted(_pred_sig(p) for p in eps))
            key = ("sig",) + sig
        if key not in variants:
            variants[key] = len(variants); vpreds[variants[key]] = list(eps); mlist[variants[key]] = []; mkeys[variants[key]] = {}
        v = variants[key] if eps else 0
        if name == "count_star" or name == "count":
            return dict(kind="moment", variant=v, moments=[], rowfilter_expr=rowfilter)
        if ext:
            k = ("ext", self.canon(a.args[0])[1])
            if k not in mkeys[v]: mkeys[v][k] = k; mlist[v].append(k)
            return dict(kind="ext", variant=v, key=k, rowfilter_expr=rowfilter)
        y = a.args[0]; x = a.args[1] if len(a.args) > 1 else None
        ms = []
        for m in MOMENTS[name](y, x):
            k = _mkey(m)
            if k not in mkeys[v]:
                mkeys[v][k] = len(mlist[v]); mlist[v].append(self.poly_of_tuple(m))
            ms.append(mkeys[v][k])
        return dict(kind="moment", variant=v, moments=ms, rowfilter_expr=rowfilter)

    def dense_atoms(self, atoms):
        """would grouping by the atoms leave about as many groups as rows? (their code domains multiplied: the bound
        on the groups; then reading the rows is cheaper than grouping them)"""
        n = max(self.tab[self.fact].n, 1)
        dom = 1
        for col in atoms.values():
            b, name = self.canon(col)
            t = self.tab[b]
            try: m = t.maxcode(name) + 1
            except Exception: return False
            dom *= m * (2 if name in getattr(t, "nullable", {}) else 1)
            if dom * 4 >= n: return True
        return False

    def rows_rel(self, atoms, preds, empty):
        """the passing fact rows (engine row bitmap) with the atoms' values read from the planes"""
        q = algq.Query(self.tab[self.fact]); q.join_mode = "auto"
        for b, fk in self.joins.items(): q.join(self.tab[b], fk)
        for p in preds: q.where(p)
        self.cx.stats["engine_queries"] += 1
        ids = np.zeros(0, np.uint64) if empty else q.rows(self.cx.engine)
        if self.fact_sample is not None: ids = ids[self.fact_sample[np.asarray(ids, np.int64)]]
        rel = Rel(len(ids))
        fact = self.tab[self.fact]
        cache = {}
        def fact_codes(name):
            if name not in cache:
                v = fact.get_rows(name, ids).astype(np.uint64)
                w = getattr(fact, "wide", {}).get(name)
                if w is not None: v = v + (fact.get_rows(w[1], ids).astype(np.uint64) << np.uint64(w[0]))
                cache[name] = v
            return cache[name]
        for k, col in atoms.items():
            b, name = self.canon(col)
            if b is self.fact:
                codes = fact_codes(name); t = fact
            else:
                t = self.tab[b]; fk = self.joins[b]
                f = fact_codes(fk).astype(np.int64)
                sg = getattr(fact, "surr", {}).get(fk)
                rows = (f & sg.mask) if sg is not None else f
                codes = t.decoded(name)[rows]
            cd = getattr(t, "codecs", {}).get(name)
            vc = getattr(t, "nullable", {}).get(name)
            ok = None
            if vc is not None: ok = np.asarray(fact_codes(vc) if b is self.fact else t.decoded(vc)[rows]) != 0
            if COLUMNAR: rel.cols.put(k, ColArr(codes, ok, cd)); continue      # decoded when read as values
            codes = codes.tolist()                       # Python integers: 64-bit codes past 2^63 stay exact
            vals = cd.decode(algq._codes_array(codes)) if cd is not None else codes
            if ok is not None: vals = [x if o else None for x, o in zip(vals, ok.tolist())]
            rel.cols[k] = vals
        return rel

    def engine(self, keys, atom_list, preds, empty, variants, vpreds, mlist):
        """the engine queries: per variant, COUNT and the sum of each moment polynomial, grouped by the atoms"""
        if COLUMNAR and keys and not empty: return self.engine_columns(keys, atom_list, preds, variants, vpreds, mlist)
        cx = self.cx
        def query(extra):
            q = algq.Query(self.tab[self.fact])
            q.join_mode = "auto"
            for b, fk in self.joins.items(): q.join(self.tab[b], fk)
            for p in preds + extra: q.where(p)
            if keys: q.group_by(*keys)
            return q
        def run(q):
            cx.stats["engine_queries"] += 1
            try: return q.run(cx.engine)
            except (TypeError, OverflowError, ValueError) as ex:
                m = str(ex)
                if "group domain too large" in m or "hidden GROUP BY columns" in m or "exceed 128 bits" in m:
                    raise _TooManyGroups()                 # the passing rows themselves, aggregated exactly in Python
                raise SqlError(f"the engine cannot run this block: {ex}")
        def first_poly(vid):
            return next((m for m in mlist[vid] if not (isinstance(m, tuple) and m[0] == "ext")), None)
        def both(extra, m):                      # one engine run: the sums of m and the row counts
            q = query(extra).sum(m); q._both = True
            return run(q)
        pre = {}                                 # variant -> (counts, sums of its first moment)
        if empty:
            base = {} if keys else 0
        else:
            m0 = first_poly(0)
            if m0 is not None:
                r = both([], m0)
                if keys: base = {g: v[1] for g, v in r.items()}; pre[0] = {g: v[0] for g, v in r.items()}
                else: base = r[1]; pre[0] = r[0]
            else: base = run(query([]).count())
        if not keys:
            n = base if not empty else 0
            rel = Rel(1 if n else 0); rel.cnt = [n] if n else []
            groups = [()] if n else []
        else:
            groups = list(base.keys())
            rel = Rel(len(groups)); rel.cnt = [base[g] for g in groups]
        for k, pos, vp in atom_list:
            rel.cols[k] = [g[pos] for g in groups] if vp is None else [g[pos] if g[vp] else None for g in groups]
        for vid in sorted(set(variants.values())):
            extra = vpreds[vid]
            if vid != 0:
                m1 = first_poly(vid)
                if empty or not groups: cnts = {} if keys else 0
                elif m1 is not None:
                    r = both(extra, m1)
                    if keys: cnts = {g: v[1] for g, v in r.items()}; pre[vid] = {g: v[0] for g, v in r.items()}
                    else: cnts = r[1]; pre[vid] = r[0]
                else: cnts = run(query(extra).count())
                rel.vcnt[vid] = [cnts.get(g, 0) for g in groups] if keys else ([cnts] if groups else [])
            for mi, m in enumerate(mlist[vid]):
                ext = isinstance(m, tuple) and m[0] == "ext"
                if empty or not groups: sums = {} if keys else None
                elif ext: sums = run(query(extra).minmax(algq.Col(self.tab[self.fact], m[1])))
                elif vid in pre and m is first_poly(vid): sums = pre[vid]
                else: sums = run(query(extra).sum(m))
                mk = (vid, m) if ext else (vid, mi)
                if keys: rel.mom[mk] = [sums.get(g) for g in groups]
                else: rel.mom[mk] = [sums] if groups else []
        return rel


    def engine_columns(self, keys, atom_list, preds, variants, vpreds, mlist):
        """engine() with the groups as columns (Query.run_columns): the atoms stay codes, every further engine run is
        aligned with the first by its key codes"""
        cx = self.cx
        def query(extra):
            q = algq.Query(self.tab[self.fact]); q.join_mode = "auto"
            for b, fk in self.joins.items(): q.join(self.tab[b], fk)
            for p in preds + extra: q.where(p)
            q.group_by(*keys)
            return q
        def run(q):
            cx.stats["engine_queries"] += 1
            try: return q.run_columns(cx.engine)
            except (TypeError, OverflowError, ValueError) as ex:
                m = str(ex)
                if "group domain too large" in m or "hidden GROUP BY columns" in m or "exceed 128 bits" in m:
                    raise _TooManyGroups()
                raise SqlError(f"the engine cannot run this block: {ex}")
        def first_poly(vid):
            return next((m for m in mlist[vid] if not (isinstance(m, tuple) and m[0] == "ext")), None)
        def rowkey(g):
            """the key columns of a run, comparable across runs: (codes as int64, or uint64 past 2^63; validity or None)"""
            out = []
            for codes, valid, _ in g.keys:
                c = ColArr(codes).codes                           # int64 (object past 2^63)
                if c.dtype == object:
                    try: c = np.asarray(c, np.uint64)
                    except OverflowError: pass
                out.append((c, valid))
            return out
        def rowvals(g):
            """the key tuples of a run as hashable values (runs that gave values and runs that gave codes meet here)"""
            cols = [[nkey(v) for v in (vals if codes is None else ColArr(codes, valid, vals).decode())]
                    for codes, valid, vals in g.keys]
            return list(zip(*cols)) if cols else [()] * len(g)
        pre = {}
        m0 = first_poly(0)
        if m0 is not None:
            q = query([]).sum(m0); q._both = True
            g = run(q); base_cnt = [v[1] for v in g.vals]; pre[0] = [v[0] for v in g.vals]
        else:
            g = run(query([]).count()); base_cnt = list(g.vals)
        G = len(base_cnt)
        rel = Rel(G); rel.cnt = base_cnt
        coded = all(c is not None for c, _, _ in g.keys)
        base_keys = rowkey(g) if coded else None
        for k, pos, vp in atom_list:
            codes, valid, cd = g.keys[pos]
            if codes is None:                            # values (a bridge result): as before
                vals = cd
                if vp is not None: vals = [x if ok else None for x, ok in zip(vals, g.keys[vp][2])]
                rel.cols[k] = vals; continue
            if vp is not None:
                vv = g.keys[vp][0] != 0
                valid = vv if valid is None else (valid & vv)
            rel.cols.put(k, ColArr(codes, valid, cd))
        lookup = {}
        def align(h):
            """the base group of each group of a later run (every later run's groups are among the base's)"""
            if coded and all(c is not None for c, _, _ in h.keys):
                import sqlwin
                cols = []
                for (a, va), (b, vb) in zip(base_keys, rowkey(h)):
                    c = np.concatenate([a, b]) if a.dtype == b.dtype else np.concatenate([a.astype(object), b.astype(object)])
                    if c.dtype == np.int64 and len(c) and int(c.min()) >= 0 and int(c.max()) < 1 << 40: d = c + 1
                    else: d = np.unique(c, return_inverse=True)[1].reshape(-1).astype(np.int64) + 1    # dense, from 1
                    if va is not None or vb is not None:                  # NULL: 0, whatever its code
                        ok = np.concatenate([np.ones(len(a), bool) if va is None else va, np.ones(len(b), bool) if vb is None else vb])
                        d = np.where(ok, d, 0)
                    cols.append(d)
                rc = sqlwin._combine(cols) if cols else np.zeros(G + len(h), np.int64)
                rb, rh = rc[:G], rc[G:]
                o = np.argsort(rb, kind="stable")
                pos = o[np.searchsorted(rb[o], rh)]
                return pos
            if "d" not in lookup: lookup["d"] = {t: i for i, t in enumerate(rowvals(g))}
            return np.array([lookup["d"][t] for t in rowvals(h)], np.int64)
        def spread(h, vals, default):
            out = [default] * G
            for p, v in zip(align(h).tolist(), vals): out[p] = v
            return out
        for vid in sorted(set(variants.values())):
            extra = vpreds[vid]
            if vid != 0:
                m1 = first_poly(vid)
                if not G: rel.vcnt[vid] = []
                elif m1 is not None:
                    q = query(extra).sum(m1); q._both = True
                    h = run(q)
                    rel.vcnt[vid] = spread(h, [v[1] for v in h.vals], 0); pre[vid] = (h, [v[0] for v in h.vals])
                else:
                    h = run(query(extra).count()); rel.vcnt[vid] = spread(h, h.vals, 0)
            for mi, m in enumerate(mlist[vid]):
                ext = isinstance(m, tuple) and m[0] == "ext"
                mk = (vid, m) if ext else (vid, mi)
                if not G: rel.mom[mk] = []; continue
                if ext: h = run(query(extra).minmax(algq.Col(self.tab[self.fact], m[1]))); sums = spread(h, h.vals, None)
                elif vid in pre and m is first_poly(vid):
                    sums = pre[0] if vid == 0 else spread(pre[vid][0], pre[vid][1], None)
                else: h = run(query(extra).sum(m)); sums = spread(h, h.vals, None)
                rel.mom[mk] = sums
        return rel


_NO = object()


class _TooManyGroups(Exception):
    pass


def _states(rel, aggs, aggplan):
    """each engine-computed aggregate's state per row: ("n", k) count, ("m", k, j) moment sums, ("x", k) extremes"""
    for k, plan in aggplan.items():
        kind = plan.get("kind")
        filt = plan.get("rowfilter")
        if kind == "zero":
            rel.cols[("n", k)] = [0] * rel.n; rel.cols[("x", k)] = [None] * rel.n
            for j in range(8): rel.cols[("m", k, j)] = [None] * rel.n
            continue
        if kind == "case":                           # the branches add up: rows with a value, and their sum
            n = [0] * rel.n; sm = [None] * rel.n
            for v, mi, const in plan["branches"]:
                cnt = rel.vcnt.get(v, rel.cnt) if v != 0 else rel.cnt
                for i in range(rel.n):
                    c = cnt[i]
                    if not c: continue
                    n[i] += c
                    add = rel.mom[(v, mi)][i] if mi is not None else (const * c if const is not None else None)
                    if add is not None: sm[i] = add if sm[i] is None else sm[i] + add
            rel.cols[("n", k)] = n; rel.cols[("m", k, 0)] = sm
            for j in range(1, 8): rel.cols[("m", k, j)] = [None] * rel.n
            rel.cols[("x", k)] = [None] * rel.n
            continue
        v = plan["variant"]
        if kind == "moment":
            cnt = rel.vcnt.get(v, rel.cnt) if v != 0 else rel.cnt
            rel.cols[("n", k)] = [c if filt is None or i in filt else 0 for i, c in enumerate(cnt)]
            for j, m in enumerate(plan["moments"]):
                col = rel.mom[(v, m)]
                rel.cols[("m", k, j)] = [x if filt is None or i in filt else None for i, x in enumerate(col)]
            for j in range(len(plan["moments"]), 8): rel.cols[("m", k, j)] = [None] * rel.n
            rel.cols[("x", k)] = [None] * rel.n
        elif kind == "ext":
            col = rel.mom[(v, plan["key"])]
            rel.cols[("x", k)] = [x if filt is None or i in filt else None for i, x in enumerate(col)]
            rel.cols[("n", k)] = [0] * rel.n
            for j in range(8): rel.cols[("m", k, j)] = [None] * rel.n


class _NotStarNow(Exception):
    pass


def _fold_consts(e):
    """evaluate the constant parts of an expression (bottom-up); AND with a FALSE term is FALSE, OR with a TRUE one TRUE"""
    def f(x):
        if isinstance(x, I.Func) and x.fn in ("and", "or"):
            stop = x.fn == "or"
            if any(isinstance(c, I.Lit) and c.v is stop for c in x.args): return I.Lit(stop, T.BOOLEAN)
        if isinstance(x, (I.Func, I.Case, I.Cast)) and all(isinstance(c, I.Lit) for c in x.children()):
            if isinstance(x, I.Func) and not F.SCALARS.get(x.fn, F.SCALARS["+"]).deterministic: return x
            if isinstance(x, I.Func) and x.fn in ("row", "__iso_similar_pat"): return x
            try: return I.Lit(eval_const(x), x.t)
            except SqlError: return x
        return x
    return I.subst_post(e, f)


def _pred_sig(p):
    """a structural signature of an engine predicate (variants with the same predicates share engine queries)"""
    if isinstance(p, algq.Cmp): return ("cmp", id(p.col.table), p.col.name, p.op, p.v)
    if isinstance(p, algq.Between): return ("btw", id(p.col.table), p.col.name, p.a, p.b)
    if isinstance(p, algq.BoolOp): return (p.op, _pred_sig(p.l), _pred_sig(p.r))
    if isinstance(p, algq.Not): return ("not", _pred_sig(p.x))
    return ("obj", id(p))


def _mkey(m):
    if isinstance(m, tuple): return (m[0], _mkey(m[1]), _mkey(m[2]))
    return m.key()


def _and(parts):
    if len(parts) == 1: return parts[0]
    return I.Func("and", parts, T.BOOLEAN)


def _deterministic(e):
    ok = [True]
    def f(x):
        if isinstance(x, I.Func):
            s = F.SCALARS.get(x.fn)
            if s is not None and not s.deterministic: ok[0] = False
    I.walk(e, f)
    return ok[0]


# ================================================================== the general executor (multisets and joins)
def run_general(s, bindings, need, aggs, cx):
    """FROM items materialized as multisets (engine GROUP BY with counts, or query results), joined, filtered"""
    cx.stats["general_blocks"] += 1
    allx = [x for x, _ in s.items] + list(s.group) + [s.where, s.having, s.qualify] + \
        [o[0][1] for o in s.order if o[0][0] == "expr"] + [j.cond for j in join_conds(s.from_)] + \
        [d[1] for d in (s.distinct_on or []) if d[0] == "expr"]
    cols = dict(need)
    for x in allx:
        if x is None: continue
        for c in I.cols_of(x): cols[c.key()] = c
    for sq in subqueries_in(allx):
        for c in outer_refs_into(sq.q, set(bindings)): cols[c.key()] = c
    for b in bindings:                                 # lateral items: the columns of earlier items they read
        if b.q is not None and not hasattr(b, "shared"):
            for c in outer_refs_into(b.q, set(bindings)): cols[c.key()] = c
    per = {}
    for c in cols.values(): per.setdefault(c.b, {})[c.key()] = c
    # WHERE terms on one FROM item that is not on the NULL side of an outer join go down to that item
    where = [fold_subqueries(fold_outer(c, cx), cx) for c in _conjuncts(s.where)]
    pushed, rest = {}, []
    if s.sample is not None: rest, where = where, []          # USING SAMPLE: after the joins, before WHERE
    for c in where:
        bs = {x.b for x in I.cols_of(c)}
        if len(bs) == 1 and not I.has(c, I.Subq) and not I.has(c, I.Outer) and not next(iter(bs)).outer_null:
            pushed.setdefault(next(iter(bs)), []).append(c)
        elif not bs and isinstance(c, I.Lit):
            if c.v is not True: rest.append(c)
        else: rest.append(c)
    rel = _materialize(s.from_, per, pushed, cx)
    if s.sample is not None: rel = sample_rel(rel, s.sample, cx)
    if rest:
        rel = rel.take(rows_where(_and(rest), rel, cx))
    return rel, {}


def _materialize(f, per, pushed, cx):
    if isinstance(f, I.Binding): return _mat_binding(f, per.get(f, {}), pushed.get(f, []), cx)
    L = _materialize(f.left, per, pushed, cx)
    lb = set(block_bindings(f.left))
    refs = {}
    for rb in block_bindings(f.right):
        if rb.q is not None and not hasattr(rb, "shared"):
            for c in outer_refs_into(rb.q, lb): refs[c.key()] = c
    if refs: return _lateral(f, L, list(refs.values()), per, pushed, cx)
    R = _materialize(f.right, per, pushed, cx)
    return _join(f, L, R, cx)


def _lateral(j, L, refs, per, pushed, cx):
    """a lateral join: the right side evaluated once per distinct tuple of the left columns it reads"""
    if j.kind in ("right", "full"): raise SqlError("a LATERAL item cannot be on the left of a RIGHT / FULL join")
    lv = [L.cols[c.key()] for c in refs]
    groups = {}
    for i in range(L.n): groups.setdefault(tuple(nkey(v[i]) for v in lv), []).append(i)
    li, rc, rv = [], [], {}
    for idx in groups.values():
        cx.outers.append({c.key(): lv[x][idx[0]] for x, c in enumerate(refs)})
        try: R = _materialize(j.right, per, pushed, cx)
        finally: cx.outers.pop()
        for k in R.cols: rv.setdefault(k, [None] * len(li))
        for i in idx:
            for r in range(R.n):
                li.append(i); rc.append(R.cnt[r])
                for k in rv: rv[k].append(R.cols[k][r] if k in R.cols else None)
    cols = list(rv)
    C = Rel(len(li))
    for k, v in L.cols.items(): C.cols[k] = [v[i] for i in li]
    for k, v in rv.items(): C.cols[k] = v
    C.cnt = [L.cnt[i] * c for i, c in zip(li, rc)]
    conds_on = _conjuncts(j.cond)
    if conds_on:
        keep = truth(ev(_and(conds_on), C, cx)); C = C.take(keep); li = [li[k] for k in keep]
    if j.kind == "left":
        hit = set(li)
        extra = [i for i in range(L.n) if i not in hit]
        if extra:
            P = L.take(extra)
            for k in cols: P.cols[k] = [None] * P.n
            C = _concat([C, P])
    return C


def _mat_binding(b, cols, conds, cx):
    cols = list(cols.values())
    if b.sample is not None and b.table is None:                # a sampled derived table: a sample of its rows
        sp, b.sample = b.sample, None
        try: rel = _mat_binding(b, {c.key(): c for c in cols}, [], cx)
        finally: b.sample = sp
        rel = sample_rel(rel, sp, cx)
        return rel.take(truth(ev(_and(conds), rel, cx))) if conds else rel
    if b.table is not None:
        s = I.Select(); s.from_ = b
        s.items = []; s.where = _and(conds) if conds else None
        need = {c.key(): c for c in cols}
        rel, _ = plan_star(s, [b], need, {}, cx)
        return rel
    # a derived table, CTE or VALUES: its rows (a CTE once per query)
    shared = getattr(b, "shared", None)
    if shared is not None and id(shared) in cx.ctes: res = cx.ctes[id(shared)]
    elif getattr(b, "rec", None) is not None and id(b.rec.binding) in cx.ctes: res = cx.ctes[id(b.rec.binding)]
    else:
        res = run_query(b.q, cx)
        if shared is not None and not getattr(b, "rec", None): cx.ctes[id(shared)] = res
    names = [c for c, _, _ in b.cols]
    rel = Rel(len(res.rows))
    for c in cols:
        j = names.index(c.name)
        rel.cols[c.key()] = [r[j] for r in res.rows]
    if conds:
        rel = rel.take(truth(ev(_and(conds), rel, cx)))
    return rel


def _join(j, L, R, cx):
    part = getattr(j, "partition", None)
    if part is not None:                               # a partitioned outer join: each partition joined on its own
        side, exprs = part
        P = L if side == "left" else R
        keys = [ev(x, P, cx) for x in exprs]
        groups = {}
        for i in range(P.n): groups.setdefault(tuple(nkey(k[i]) for k in keys), []).append(i)
        parts = []
        for idx in groups.values():
            Pi = P.take(idx)
            C = _join_plain(j, Pi, R, cx) if side == "left" else _join_plain(j, L, Pi, cx)
            vals = {x.key(): Pi.cols[x.key()][0] for x in exprs}
            for k, v in vals.items():                    # the padded rows keep their partition's values
                col = C.cols[k]
                C.cols[k] = [v if c is None else c for c in col]
            parts.append(C)
        if not parts: return _join_plain(j, L, R, cx)
        return _concat(parts)
    return _join_plain(j, L, R, cx)


def _join_plain(j, L, R, cx):
    kind = j.kind
    conds = _conjuncts(j.cond)
    lkeys, rkeys, other, nulleq = [], [], [], []
    lset, rset = set(L.cols), set(R.cols)
    for c in conds:
        if isinstance(c, I.Func) and c.fn in ("=", "is_not_distinct") and len(c.args) == 2:
            a, b = c.args
            ka, kb = {x.key() for x in I.cols_of(a)}, {x.key() for x in I.cols_of(b)}
            if ka and kb and ka <= lset and kb <= rset: lkeys.append(a); rkeys.append(b); nulleq.append(c.fn != "="); continue
            if ka and kb and ka <= rset and kb <= lset: lkeys.append(b); rkeys.append(a); nulleq.append(c.fn != "="); continue
        other.append(c)
    import sqlwin
    if lkeys:                                        # equal keys: the rows' codes, matched by a sort and two searches
        lv = [codes_of(x, L, cx) for x in lkeys]; rv = [codes_of(x, R, cx) for x in rkeys]
        lv = [c if c is not None else ev(x, L, cx) for c, x in zip(lv, lkeys)]
        rv = [c if c is not None else ev(x, R, cx) for c, x in zip(rv, rkeys)]
        lc, rc = [], []
        lok = np.ones(L.n, bool); rok = np.ones(R.n, bool)
        for a, b, ne in zip(lv, rv, nulleq):
            ja, jb = sqlwin.joint_codes(a, b)          # the two sides' values matched on their dictionaries
            ma, mb = sqlwin.nn_mask(a) if isinstance(a, ColArr) else sqlwin.notnull(sqlwin.obj(a)), \
                sqlwin.nn_mask(b) if isinstance(b, ColArr) else sqlwin.notnull(sqlwin.obj(b))
            if ne:                                     # IS NOT DISTINCT FROM: NULL matches NULL (a code of its own)
                ja = np.where(ma, ja, -1); jb = np.where(mb, jb, -1)
            else: lok &= ma; rok &= mb                 # = never matches NULL
            lc.append(np.asarray(ja)); rc.append(np.asarray(jb))
        both = [np.unique(np.concatenate([x, y]), return_inverse=True)[1].reshape(-1).astype(np.int64) for x, y in zip(lc, rc)]
        codes = sqlwin._combine(both)
        cl, cr = codes[:L.n], codes[L.n:]
        rr = np.flatnonzero(rok); o = rr[np.argsort(cr[rr], kind="stable")]; cs = cr[o]
        lr = np.flatnonzero(lok)
        lo = np.searchsorted(cs, cl[lr], "left"); cnt = np.searchsorted(cs, cl[lr], "right") - lo
        li = np.repeat(lr, cnt)
        ri = o[np.repeat(lo - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))] if len(li) else np.zeros(0, np.int64)
    else:
        li = np.repeat(np.arange(L.n), R.n); ri = np.tile(np.arange(R.n), L.n)
    C = _combine(L, R, li, ri)
    if other:
        keep = np.array(truth(ev(_and(other), C, cx)), np.int64)
        C = C.take(keep.tolist()); li, ri = li[keep], ri[keep]
    if kind in ("inner", "cross"): return C
    if kind in ("semi", "anti"):
        hit = np.zeros(L.n, bool); hit[li] = True
        return L.take(np.flatnonzero(hit if kind == "semi" else ~hit).tolist())
    parts = [C]
    if kind in ("left", "full"):
        hit = np.zeros(L.n, bool); hit[li] = True
        extra = np.flatnonzero(~hit).tolist()
        if extra: parts.append(_pad(L.take(extra), R))
    if kind in ("right", "full"):
        hit = np.zeros(R.n, bool); hit[ri] = True
        extra = np.flatnonzero(~hit).tolist()
        if extra: parts.append(_pad(R.take(extra), L))
    return _concat(parts)


def _combine(L, R, li, ri):
    """the pairs (li[k], ri[k]) of rows of L and R as one relation (engine columns stay codes)"""
    import sqlwin
    C = Rel(len(li))
    for side, idx in ((L, li), (R, ri)):
        for k, a in side.cols.arr.items(): C.cols.put(k, a.take(idx))
        for k, v in dict.items(side.cols):
            if k not in side.cols.arr: dict.__setitem__(C.cols, k, sqlwin.obj(v)[idx].tolist() if len(idx) else [])
    a, b = sqlwin.obj(L.cnt)[li], sqlwin.obj(R.cnt)[ri]
    C.cnt = (a * b).tolist() if len(li) else []
    return C


def _pad(A, other):
    for k in other.cols:
        if k not in A.cols: A.cols[k] = [None] * A.n
    return A


def _concat(parts):
    n = sum(p.n for p in parts)
    C = Rel(n)
    keys = []
    for p in parts:
        for k in p.cols:
            if k not in keys: keys.append(k)
    for k in keys:
        arrs = [p.cols.arr.get(k) for p in parts]
        if all(a is not None for a in arrs) and len({id(a.codec) for a in arrs}) == 1:   # codes of one codec: stay codes
            valid = None if all(a.valid is None for a in arrs) else \
                np.concatenate([a.valid if a.valid is not None else np.ones(len(a), bool) for a in arrs])
            C.cols.put(k, ColArr(np.concatenate([a.codes for a in arrs]), valid, arrs[0].codec)); continue
        out = []
        for p in parts: out += p.cols[k] if k in p.cols else [None] * p.n
        C.cols[k] = out
    C.cnt = [c for p in parts for c in p.cnt]
    return C
