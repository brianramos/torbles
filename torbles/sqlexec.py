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
import copy, functools, math, operator
from decimal import Decimal
from fractions import Fraction
import numpy as np
import algq
import sqltypes as T
import sqlwin
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
        if self._rows is None:
            columns = [col.tolist() if isinstance(col, np.ndarray) else col for col in self._cols]
            self._rows = list(zip(*columns))
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
    def fetch_columns(self):
        """Return NumPy columns in SQL projection order, including duplicate names."""
        cols = self._cols
        if cols is None:
            cols = list(zip(*self.rows)) if self.rows else [()] * len(self.names)
        arrays = []
        for col in cols:
            if isinstance(col, np.ndarray):
                arrays.append(col.copy())
            else:
                array = np.empty(len(col), dtype=object)
                array[:] = list(col)
                arrays.append(array)
        return tuple(arrays)
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
        ia = np.asarray(idx, np.int64) if self.arr else None   # the positions as an array once, for every code column
        il = None
        for k, a in self.arr.items(): out.arr[k] = a.take(ia)
        for k, v in dict.items(self):
            if k not in self.arr:
                if il is None: il = _positions(idx)
                dict.__setitem__(out, k, _take_list(v, il))
        return out


class Rel:
    """rows of a block while it is evaluated: cols {expression key: [value per row]} (Cols: engine columns stay codes
    until read as values), cnt [rows each row stands for], mom {(variant, moment): [sum per row]}, vcnt {variant: [count
    per row]}"""
    __slots__ = ("n", "cols", "cnt", "mom", "vcnt", "grouped", "plane_ordered", "native_window", "native_qualified", "native_terminal")
    def __init__(self, n):
        self.n, self.cols, self.cnt, self.mom, self.vcnt = n, Cols(), [1] * n, {}, {}
        self.grouped = False
        self.plane_ordered = self.native_window = self.native_qualified = self.native_terminal = False
    def take(self, idx):
        r = Rel(len(idx))
        r.cols = self.cols.take(idx)
        il = _positions(idx) if (self.mom or self.vcnt or self.cnt.count(1) != len(self.cnt)) else None
        if il is not None and self.cnt.count(1) != len(self.cnt): r.cnt = _take_list(self.cnt, il)   # else Rel's own 1s
        r.mom = {k: _take_list(v, il) for k, v in self.mom.items()}
        r.vcnt = {k: _take_list(v, il) for k, v in self.vcnt.items()}
        return r


def _positions(idx):
    """row positions as a list of Python ints (indexing a list with numpy integers is several times slower)"""
    return idx.tolist() if isinstance(idx, np.ndarray) else idx


def _take_list(v, il):
    """[v[i] for i in il], at C speed"""
    if len(il) > 1: return list(operator.itemgetter(*il)(v))
    return [v[i] for i in il]


class Cx:
    """execution context: the database, the engine, CTE results, the values of enclosing queries (correlation)"""
    def __init__(self, db, engine="alg"):
        self.db, self.engine = db, engine
        self.ctes, self.outers, self.memo, self.stats = {}, [], {}, dict(engine_queries=0, star_blocks=0, general_blocks=0,
                                                                        series_windows=0, snowflake_blocks=0, snowflake_edges=0)
        self.series = {}            # id(block) -> {window key: _Series} while the block runs (windows from their series)

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


# ------------------------------------------------------------------ bounded exact types
# Values are exact Python integers and Decimals (they never wrap), so an integer or DECIMAL operation whose value
# leaves its SQL type is an error where it is computed (DuckDB's overflow errors), on every path that computes it:
# row by row (_apply), on value columns (_vec_func), on code columns (_ev_codes), and in the planner's proofs before
# an expression becomes an engine polynomial or a plane predicate. Like every runtime error it fails the query only
# if the value is used.
_BOUNDED_OPS = {"+", "-", "*", "neg", "abs", "//", "divide", "%", "mod", "<<", "round", "gcd", "lcm", "factorial"}
_DUCK_INT = {"TINYINT": "INT8", "SMALLINT": "INT16", "INTEGER": "INT32", "BIGINT": "INT64", "HUGEINT": "INT128",
             "UTINYINT": "UINT8", "USMALLINT": "UINT16", "UINTEGER": "UINT32", "UBIGINT": "UINT64", "UHUGEINT": "UINT128"}


def type_range(t):
    """(lo, hi): the values of an integer type, or of DECIMAL(p, s) (|v| < 10^(p - s)); None for any other type"""
    if t is None: return None
    r = T.INT_RANGE.get(t.id)
    if r is not None or t.id != "DECIMAL" or t.p is None: return r
    m = Decimal(10 ** t.p - 1).scaleb(-t.s)
    return -m, m


def overflow_error(fn, t, args):
    """DuckDB's message for an operation of type t whose exact value on args is outside t"""
    a = [str(x) for x in args]
    if fn == "neg": return "Out of Range Error: Overflow in negation of numeric value!"
    if fn == "abs": return f"Out of Range Error: Overflow on abs({a[0]})"
    if fn == "<<": return f"Out of Range Error: Overflow in left shift ({a[0]} << {a[1]})"
    if fn == "round": return f"Out of Range Error: Overflow in ROUND of integer ({a[0]})"
    if fn in ("gcd", "lcm", "factorial"): return f"Out of Range Error: {fn} value is out of range for {t}"
    if fn in ("//", "divide", "%", "mod"): return f"Out of Range Error: Overflow in division of {a[0]} / {a[1]}"
    what = {"+": "addition", "-": "subtraction", "*": "multiplication"}[fn]
    if t.id == "DECIMAL": return f"Out of Range Error: Overflow in {what} of DECIMAL({t.p}) ({a[0]} {fn} {a[1]})"
    return f"Out of Range Error: Overflow in {what} of {_DUCK_INT.get(t.id, t.id)} ({a[0]} {fn} {a[1]})!"


def _bounded(fn, t):
    """the range an operation's values must keep (None: unbounded, or not an integer / DECIMAL operation)"""
    return type_range(t) if fn in _BOUNDED_OPS else None


def _apply(s, args, n, t=None):
    """a scalar function row by row; t: its result type (an integer or DECIMAL operation's values are checked)"""
    impl = s.impl
    if not args: return [impl()] * n if s.deterministic else [impl() for _ in range(n)]
    out = []
    prop = s.nulls == "propagate"
    rng = _bounded(s.name, t)
    sdiv = rng is not None and s.name in ("%", "mod") and t.integer and rng[0] < 0
    for row in zip(*args):
        bad = None
        for x in row:
            if isinstance(x, Err): bad = x; break
        if bad is not None and not (s.name in ("and", "or", "coalesce", "is_null", "is_not_null")):
            out.append(bad); continue
        if prop and None in row:
            out.append(None); continue
        try:
            v = impl(*row)
        except SqlError as ex: out.append(Err(str(ex))); continue
        except (ZeroDivisionError, ValueError, OverflowError, ArithmeticError, IndexError, TypeError) as ex:
            out.append(Err(f"{s.name}: {ex}")); continue
        if rng is not None and type(v) in (int, Decimal) and not rng[0] <= v <= rng[1] or \
                sdiv and row[1] == -1 and row[0] == rng[0]:        # DuckDB: the least value % -1 overflows, as // does
            v = Err(overflow_error(s.name, t, row))
        out.append(v)
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
        if s.nulls == "propagate" and any(isinstance(a, I.Lit) and a.v is None for a in e.args):
            return [None] * n           # NULL whatever the other arguments are: they are not computed (DuckDB folds it)
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
        if e.extra == "strict" and any(isinstance(v, Err) for v in argv[0]):      # x IS [NOT] NULL that computes x
            return [v if isinstance(v, Err) else (v is None) == (e.fn == "is_null") for v in argv[0]]   # (sqlfront)
        if VEC_SCALARS and n >= 64 and e.fn in _VEC_FNS:
            r = _vec_func(e.fn, argv, n, e.t)
            if r is not None: return r
        return _apply(s, argv, n, e.t)
    if isinstance(e, I.Case):
        if COLUMNAR and DICT_EVAL and n >= 64:
            r = _dict_eval(e, rel, cx)
            if r is not None: return r
        conds = [ev(c, rel, cx) for c, _ in e.whens]
        # a WHEN is a filter of the rows (DuckDB selects them term by term): an error in one term of its AND / OR
        # counts only for a row that the others leave undecided (_select)
        conds = [_select(c, rel, cx) if any(isinstance(v, Err) for v in cv) else cv for (c, _), cv in zip(e.whens, conds)]
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
RAW_PLANE_ORDER = True       # direct radix partition of stored planes for simple fact ordering
TOPK = True                 # ORDER BY ... LIMIT k on fact columns reads only the rows that can be among the first k
TOPK_MAX = 100_000          # (rank selection on the planes; past k = this, every passing row is read)
# rank selection, top-k bounds and COUNT(DISTINCT) cell bits from this many fact rows on (on fewer, reading the values
# costs less than the extra engine queries)
ALG_MIN_ROWS = int(__import__("os").environ.get("TORBLES_ALG_MIN_ROWS", 1 << 16))
RANK_AGGS = True            # MEDIAN / PERCENTILE_CONT / _DISC of a fact column: rank selection on its planes per group
RANK_DIRECT = 2048          # (8 bits per engine query while groups * this <= fact rows; more groups: the whole code at once)
_RANK_AGGS = {"median", "quantile_cont", "percentile_cont", "quantile_disc", "percentile_disc", "quantile"}
DIST_BITS = True            # COUNT(DISTINCT column): one bit per (group, value) cell in the engine, then a popcount per group
JOIN_PREAGG = True          # aggregates over an inner join of two FROM items: each side grouped first, then the groups joined
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


def _int_fits(r, nul, t, a=None, b=None):
    """do the integer results r of an operation of type t stay inside t (rows NULL in nul aside)? With a, b: the
    operands of a % or //, whose least value % -1 overflows too. (False: the row-by-row path gives each row's error)"""
    rng = T.INT_RANGE.get(t.id) if t is not None else None
    if rng is None: return True
    live = slice(None) if nul is None else ~nul
    v = r[live]
    if len(v) and (int(v.min()) < rng[0] or int(v.max()) > rng[1]): return False
    return a is None or rng[0] >= 0 or not bool(((a[live] == rng[0]) & (b[live] == -1)).any())


def _vec_func(fn, argv, n, t=None):
    """the operator on whole columns, with the row-by-row results (exact integers, IEEE doubles, three-valued logic);
    None when a column is not plain ints / doubles / booleans, or when an integer result leaves its type t (the
    row-by-row path then gives that row its overflow error)"""
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
        r = -cols[0][1]
        return _out(r, nul) if kinds != {"int"} or _int_fits(r, nul, t) else None
    a, b = cols[0][1], cols[1][1]
    if fn in _CMPF:
        if kinds == {"int", "float"} and max(c[3] for c in cols if c[0] == "int") >= 1 << 53: return None
        return _out(_CMPF[fn](a, b), nul)
    if fn in ("%", "mod", "//", "divide"):            # integers, no zero divisor: truncating division, as SQL's
        if kinds != {"int"}: return None
        live = b if nul is None else b[~nul]
        if (live == 0).any(): return None
        bb = np.where(b == 0, 1, b)
        if fn in ("%", "mod"): r = np.fmod(a, bb)
        else:
            q = np.abs(a) // np.abs(bb)
            r = np.where((a >= 0) == (bb >= 0), q, -q)
        return _out(r, nul) if _int_fits(r, nul, t, a, b) else None
    if fn == "/":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return _out(a.astype(np.float64) / b.astype(np.float64), nul)
    if kinds == {"int"}:
        ma, mb = cols[0][3], cols[1][3]
        if fn == "*" and ma * mb >= 1 << 63: return None
        r = a + b if fn == "+" else a - b if fn == "-" else a * b
        return _out(r, nul) if _int_fits(r, nul, t) else None
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
        if len(c) and abs(off) + sqlwin.absmax(c) >= 1 << 62: return None
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
    if sqlwin.absmax(v) * f >= 1 << 62: return None
    return v * f


def codes_of(e, rel, cx):
    """e on every row of rel as a code column (ColArr), computed on the codes of the engine's columns without
    decoding them; None when e is not computed that way (row by row instead)"""
    if not (COLUMNAR and VEC_SCALARS): return None
    a = rel.cols.array(e.key())
    if a is not None: return a
    if e.key() in rel.cols: return None              # held as values: that is the column (not recomputed from its parts,
    if rel.n < 64: return None                       # which grouping sets may have set to NULL)
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
        if sqlwin.absmax(v) >= 1 << 53: return None
        return v.astype(np.float64)
    if k == "dec":
        if sqlwin.absmax(v) >= 1 << 53: return None
        return v.astype(np.float64) / float(10 ** sc)
    return None


def _codes_fit(r, valid, t, a=None, b=None):
    """do the exact results r of an integer / DECIMAL operation of type t (DECIMAL: scaled by 10^t.s) stay inside t,
    on the valid rows? With a, b: the operands of a % or //, whose least value % -1 overflows too"""
    if t.id == "DECIMAL":
        v = r if valid is None else r[valid]
        return sqlwin.absmax(v) < 10 ** t.p
    return _int_fits(r, None if valid is None else ~valid, t, a, b)


def _ev_codes(e, rel, cx):
    """arithmetic, comparisons, AND / OR / NOT, IS NULL and numeric casts on code columns (exact integers, exact
    scaled decimals, IEEE doubles with SQL's NaN order, three-valued logic); None for anything else, and for an
    integer or DECIMAL result outside its type (the row-by-row path gives those rows their overflow errors)"""
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
            if w is None or sqlwin.absmax(w) >= 10 ** t.p: return None
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
        if k == "bool" or k == "int" and not _codes_fit(-v, valid, e.t): return None
        return _mk(k, -v, sc, valid)
    if "bool" in kinds and kinds != {"bool"}: return None
    (ka, va, sa, _), (kb, vb, sb, _) = ops
    if fn in ("%", "mod", "//", "divide"):              # integers, no zero divisor: truncating, as SQL's
        if kinds != {"int"}: return None
        live = vb if valid is None else vb[valid]
        if (live == 0).any(): return None
        bb = np.where(vb == 0, 1, vb)
        if fn in ("%", "mod"): r = np.fmod(va, bb)
        else:
            qq = np.abs(va) // np.abs(bb)
            r = np.where((va >= 0) == (bb >= 0), qq, -qq)
        return _mk("int", r, 0, valid) if _codes_fit(r, valid, e.t, va, vb) else None
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
            if e.t.id == "DECIMAL" and e.t.s != S or not _codes_fit(r, valid, e.t): return None
            return _mk("dec" if S else "int", r, S, valid)
        if fn == "*":
            if sqlwin.absmax(va) * sqlwin.absmax(vb) >= 1 << 62: return None
            if e.t.id == "DECIMAL" and e.t.s != sa + sb or not _codes_fit(va * vb, valid, e.t): return None
            return _mk("dec" if sa + sb else "int", va * vb, sa + sb, valid)
        lt, eq = xa < xb, xa == xb
    r = {"=": eq, "<>": ~eq, "<": lt, "<=": lt | eq, ">": ~(lt | eq), ">=": ~lt}[fn]
    return _mk("bool", r, 0, valid)


def _select(e, rel, cx):
    """condition e as a filter (WHERE, ON, HAVING, a CASE's WHEN) on every row of rel: True (kept), False (dropped:
    FALSE or NULL) or the row's Err. An AND drops a row on any term that is FALSE or NULL, an OR keeps it on any TRUE
    term, whatever the others are: DuckDB evaluates a filter's terms one after the other on the rows still undecided,
    so an error in one counts only for a row the other terms leave undecided (a projection computes AND / OR in full:
    there NULL AND <error> fails)"""
    if isinstance(e, I.Func) and e.fn in ("and", "or"):
        parts = [_select(x, rel, cx) for x in e.args]
        decide = e.fn == "or"                                # the value that decides: TRUE for OR, FALSE for AND
        out = []
        for vs in zip(*parts):
            if decide in vs: out.append(decide); continue
            out.append(next((v for v in vs if isinstance(v, Err)), not decide))
        return out
    return [v if v is True or isinstance(v, Err) else False for v in ev(e, rel, cx)]


def filter_rows(e, rel, cx):
    """condition e as a filter (_select) -> (the rows where it is TRUE, {row: error} of the rows it errs on)"""
    a = codes_of(e, rel, cx)
    if a is not None and (a.codec is _BOOL or getattr(a.codec, "kind", None) == "bool") and a.codes.dtype != object:
        m = a.codes != 0
        if a.valid is not None: m &= a.valid
        return np.flatnonzero(m).tolist(), {}
    vals = ev(e, rel, cx)
    if not any(isinstance(v, Err) for v in vals): return [i for i, v in enumerate(vals) if v is True], {}
    vals = _select(e, rel, cx)                       # (where the full value has no error, the same rows)
    return [i for i, v in enumerate(vals) if v is True], {i: v for i, v in enumerate(vals) if isinstance(v, Err)}


def rows_where(e, rel, cx):
    """the rows where condition e is TRUE (on codes when it can be computed there), as a filter (filter_rows): an error
    on a row that the other terms pass fails the query"""
    rows, errs = filter_rows(e, rel, cx)
    if errs: raise SqlError(errs[min(errs)].msg)
    return rows


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
        if getattr(r, "_primitive_window_output", False):
            r.stats = cx.stats
            return r
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
    if cx.engine == "gpu":
        from torbles.sqlwindowplanes import execute_resident
        from torbles.sqlwindowconsumers import run_setop as window_setop

        result = window_setop(q, cx, execute_resident)
        if result is not None:
            return result
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
        keys.append(sqlwin.sort_keys(a, desc, nf))
    perm = sqlwin.lexsort(keys[::-1], out.n)
    off = _int_param(_count_value(s.offset, cx), "OFFSET") or 0
    lim = _count_value(s.limit, cx)
    if lim is not None: lim = _int_param(lim, "LIMIT")
    return perm[off:] if lim is None else perm[off:off + lim]          # (an array of positions)


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
    if cx.engine == "gpu" and not as_rel:
        from torbles.sqlwindowplanes import execute_resident, group_derived
        from torbles.sqlwindowconsumers import run_select as window_select

        result = window_select(s, cx, execute_resident)
        if result is None and s.group:
            result = group_derived(s, cx)
        if result is not None:
            return result
    if cx.engine == "gpu" and s.aggregate and not as_rel:
        from torbles.sqlwindowplanes import reduce_derived

        reduced = reduce_derived(s, cx)
        if reduced is not None:
            return reduced
    bindings = block_bindings(s.from_)
    bset = set(bindings)
    items = [x for x, _ in s.items]
    order_exprs = [o[0][1] for o in s.order if o[0][0] == "expr"]
    don = [d[1] for d in (s.distinct_on or []) if d[0] == "expr"]
    top = items + [s.having, s.qualify] + order_exprs + don
    aggs = collect(top, I.Agg)
    wins = collect(top, I.Win)
    if wins:                                       # windows read from their aggregated series (before the block's rows:
        cx.series[id(s)] = series_windows(s, wins, cx)          # ORDER BY ... LIMIT may then read only the rows output)
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
            keep = rows_where(s.where, rel, cx); rel = rel.take(keep)
    else:
        rel, aggplan = None, {}
        if not wins and any(hasattr(b, "writable") for b in bindings):
            rel = _writable_aggregate(s, bindings, aggs, cx)
        if rel is None and all(b.table is not None and not hasattr(b, "_dml_rel") for b in bindings):
            rel, aggplan = plan_star(s, bindings, need, aggs, cx)
        if rel is None and JOIN_PREAGG:
            got = join_preagg(s, aggs, cx)
            if got is not None: rel, aggplan = got
        if rel is None:
            rel, aggplan = run_general(s, bindings, need, aggs, cx)
    return finish_select(s, rel, aggs, wins, aggplan, cx, as_rel)


def _writable_bounds(terms, bindings):
    """Lower nonnullable integer comparisons to intersected inclusive uint32 ranges."""
    max_code = (1 << 32) - 1
    bounds = {b: {} for b in bindings}
    empty = False
    for term in terms:
        e = _fold_consts(term)
        if isinstance(e, I.Lit) and (e.v is None or type(e.v) is bool):
            if e.v is True:
                continue
            # Keep checking unsupported terms so the ordinary SQL path keeps their error semantics.
            empty = True
            continue
        if not isinstance(e, I.Func):
            return None
        args = [_strip_cast(x) for x in e.args]
        op = e.fn
        if op == "between" and len(args) == 3:
            col, low, high = args
            if not all(isinstance(v, I.Lit) and type(v.v) is int for v in (low, high)):
                return None
            low, high = low.v, high.v
        elif op in ("=", "<", "<=", ">", ">=") and len(args) == 2:
            col, lit = args
            if isinstance(col, I.Lit):
                col, lit = lit, col
                op = {"=": "=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]
            if not isinstance(lit, I.Lit) or type(lit.v) is not int:
                return None
            value = lit.v
            low = value + (op == ">") if op in ("=", ">", ">=") else 0
            high = value - (op == "<") if op in ("=", "<", "<=") else max_code
        else:
            return None
        if not isinstance(col, I.Col) or col.b not in bounds or not col.t.integer:
            return None
        before_low, before_high = bounds[col.b].get(col.name, (0, max_code))
        bounds[col.b][col.name] = max(low, before_low), min(high, before_high)
    return bounds, empty or any(low > high for columns in bounds.values() for low, high in columns.values())


def _writable_aggregate(s, bindings, aggs, cx):
    """Native fixed-width reductions, using the normal SQL finalizer on the exact results."""
    if (cx.engine not in ("alg", "gpu") or not s.aggregate or s.sets is not None or s.sample is not None
            or any(not hasattr(b, "writable") or b.sample is not None or hasattr(b, "_dml_rel") for b in bindings)
            or any(a.distinct or a.filter is not None or a.order for a in aggs.values())):
        return None
    args = {}
    for key, agg in aggs.items():
        arg = agg.args[0] if len(agg.args) == 1 else None
        if not (agg.name == "count_star" or agg.name in ("sum", "count") and isinstance(arg, I.Col)):
            return None
        args[key] = arg
    terms = _conjuncts(s.where)
    if len(bindings) == 2:
        return _writable_join_sum(s, bindings, aggs, args, terms, cx)
    if len(bindings) != 1 or len(s.group) > 1:
        return None
    binding = bindings[0]
    group = s.group[0] if s.group else None
    if group is not None and (not isinstance(group, I.Col) or group.b is not binding):
        return None
    if any(arg is not None and arg.b is not binding for arg in args.values()):
        return None
    filters = _writable_bounds(terms, bindings)
    if filters is None:
        return None
    bounds, empty = filters
    where, store = bounds[binding], binding.writable.store
    if group is None:
        out = Rel(1)
        for key, agg in aggs.items():
            if agg.name == "sum":
                value = None if empty else store.sum(args[key].name, where, engine=cx.engine)
            else:
                value = 0 if empty else store.count(where, engine=cx.engine)
            out.cols[key] = [value]
    else:
        columns = {}
        for key, agg in aggs.items():
            if empty:
                values = {}
            elif agg.name == "sum":
                values = store.group_sum(group.name, args[key].name, where, engine=cx.engine)
            else:
                values = store.group_count(group.name, where, engine=cx.engine)
            columns[key] = values
        groups = list(next(iter(columns.values()))) if columns else (
            [] if empty else list(store.group_count(group.name, where, engine=cx.engine))
        )
        out = Rel(len(groups))
        out.cols[group.key()] = groups
        for key, values in columns.items():
            out.cols[key] = [values[g] for g in groups]
    cx.stats["writable_native_blocks"] = cx.stats.get("writable_native_blocks", 0) + 1
    out.grouped = True
    return out


def _writable_join_sum(s, bindings, aggs, args, terms, cx):
    """Join a fact SUM grouped on its foreign key with a unique dimension key."""
    if (s.group or not aggs or any(a.name != "sum" for a in aggs.values())
            or not isinstance(s.from_, I.Join) or s.from_.kind not in ("inner", "cross")
            or getattr(s.from_, "partition", None) is not None):
        return None
    terms = terms + _conjuncts(s.from_.cond)
    fact = next(iter(args.values())).b
    if fact not in bindings or any(arg.b is not fact for arg in args.values()):
        return None
    dim = bindings[1] if bindings[0] is fact else bindings[0]
    for i, term in enumerate(terms):
        if not isinstance(term, I.Func) or term.fn != "=" or len(term.args) != 2:
            continue
        fk, pk = map(_strip_cast, term.args)
        if isinstance(fk, I.Col) and fk.b is dim:
            fk, pk = pk, fk
        if (isinstance(fk, I.Col) and isinstance(pk, I.Col) and fk.b is fact and pk.b is dim
                and pk.name == dim.writable.primary_key):
            break
    else:
        return None
    filters = _writable_bounds(terms[:i] + terms[i + 1:], bindings)
    if filters is None:
        return None
    bounds, empty = filters
    keep = set()
    if not empty:
        dim_cols = dim.writable.scan(list(dict.fromkeys([pk.name, *bounds[dim]])))
        ids = dim_cols[pk.name]
        mask = np.ones(len(ids), bool)
        for name, (low, high) in bounds[dim].items():
            mask &= (dim_cols[name] >= low) & (dim_cols[name] <= high)
        keep = set(ids[mask].tolist())
    out = Rel(1)
    for key, arg in args.items():
        matches = []
        if keep:
            grouped = fact.writable.store.group_sum(fk.name, arg.name, bounds[fact], engine=cx.engine)
            matches = [value for code, value in grouped.items() if code in keep]
        out.cols[key] = [sum(matches) if matches else None]
    cx.stats["writable_native_blocks"] = cx.stats.get("writable_native_blocks", 0) + 1
    cx.stats["writable_native_joins"] = cx.stats.get("writable_native_joins", 0) + 1
    out.grouped = True
    return out


# ------------------------------------------------------------------ group-level evaluation
def finish_select(s, rel, aggs, wins, aggplan, cx, as_rel=False):
    names = [n for _, n in s.items]; types = [x.t for x, _ in s.items]
    if rel.plane_ordered:
        if as_rel and not s.order and s.limit is None and s.offset is None:
            return rel
        return Relation.of_columns(names, types, [ev(x, rel, cx) for x, _ in s.items], rel.n)
    if s.aggregate:
        out = aggregate(s, rel, aggs, aggplan, cx)
    else:
        # one output row per underlying row: expand the groups' multiplicities
        if rel.cnt.count(1) != len(rel.cnt):
            idx = [i for i in range(rel.n) for _ in range(rel.cnt[i])]
            rel = rel.take(idx)
        out = rel
    if (out.native_window and not as_rel and (s.qualify is None or out.native_qualified)
            and not s.distinct and (out.native_terminal
            or (not s.order and s.limit is None and s.offset is None))):
        from torbles.coltypes import IntCodec

        cols = [out.cols.array(expr.key()) for expr, _ in s.items]
        arrays = []
        for col in cols:
            if col is None or col.valid is not None or col.codes.dtype != np.int64:
                break
            if col.codec is None:
                arrays.append(col.codes)
                continue
            if type(col.codec) is not IntCodec:
                break
            offset = col.codec.offset
            if offset and len(col) and not (
                -(1 << 63) <= int(col.codes.min()) + offset
                and int(col.codes.max()) + offset < (1 << 63)
            ):
                break
            if not -(1 << 63) <= offset < (1 << 63):
                break
            arrays.append(col.codes + offset if offset else col.codes)
        else:
            if arrays:
                result = Relation.of_columns(names, types, arrays, out.n)
                result._primitive_window_output = True
                return result
    if out.native_terminal:
        return Relation.of_columns(names, types, [ev(expr, out, cx) for expr, _ in s.items], out.n)
    if wins: compute_windows(list(wins.values()), out, cx, cx.series.get(id(s)))
    if s.qualify is not None and not out.native_qualified:
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
            if not (len(idx) == out.n and (out.n == 0 or (idx[0] == 0 and bool((np.diff(idx) == 1).all())))):
                out = out.take(idx)                     # (rows already in that order are kept as they are)
            cols = [ev(x, out, cx) for x, _ in s.items]
            return Relation.of_columns(names, types, cols, out.n)
    cols = [ev(x, out, cx) for x, _ in s.items]
    if not s.distinct and not s.order and s.limit is None and s.offset is None:
        return Relation.of_columns(names, types, cols, out.n)        # rows built once, by the caller
    rows = list(zip(*cols)) if cols else [()] * out.n
    for i in range(len(s.items)): out.cols[("pos", i)] = cols[i]
    if s.distinct:
        if s.distinct_on:
            # the DISTINCT ON keys and the ORDER BY that picks each one's row compare under their collations, as
            # ORDER BY and DISTINCT do (x COLLATE NOCASE: 'a' and 'A' are one key)
            def keyvals(spec):
                if spec[0] == "pos": return _ckeys(out.cols[("pos", spec[1])], _out_type(s, spec[1]))
                return _ckeys(ev(spec[1], out, cx), spec[1].t)
            keys = [keyvals(d) for d in s.distinct_on]
            idx = list(range(len(rows)))
            okeys = [keyvals(sp) for sp, _, _ in s.order]
            for kc in keys + okeys:
                for v in kc:
                    if isinstance(v, Err): raise SqlError(v.msg)
            if s.order: idx = sort_rows(idx, okeys, [(d, nf) for _, d, nf in s.order])
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
        out.cols[gx.key()] = [per_set[0]] * out.n if gset is None else [per_set[si] for si in gset]


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
    if rel.grouped:
        _grouping_cols(s, rel, [list(range(len(s.group)))], None)
        return rel if s.having is None else rel.take(rows_where(s.having, rel, cx))
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
    if plan is not None and plan.get("kind") == "rank": return plan["fn"](members)
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
    # a row whose argument, FILTER or ORDER BY value is an error (an overflow, a failed cast) makes its group's
    # aggregate that error, which fails the query only if the aggregate is used (an output row, HAVING, ORDER BY): a
    # group that a HAVING on its keys drops does not fail it (DuckDB filters those rows before aggregating them; the
    # binder moves such a HAVING term to WHERE when there are no grouping sets, sqlfront.having_pushdown)
    bad = {}
    filt = _rows_where_errs(a.filter, rel, cx, bad) if a.filter is not None else None
    fset = set(filt) if filt is not None else None
    if VEC_GROUPS:                                   # engine columns as codes, others as values
        src = [codes_of(x, rel, cx) for x in a.args]
        src = [c if c is not None else _scrub(ev(x, rel, cx), bad) for c, x in zip(src, a.args)]
        got = _agg_arrays(a, rel, members, src, filt, d)
        if got is not None: return _group_errs(got, members, bad)
    args = [_scrub(ev(x, rel, cx), bad) for x in a.args]
    okeys = [_scrub(_ckeys(ev(o[0], rel, cx), o[0].t), bad) for o in a.order]
    gerr = _group_errs([None] * len(members), members, bad)
    out = []
    for g, mem in enumerate(members):
        if gerr[g] is not None: out.append(gerr[g]); continue
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


def _scrub(vals, bad):
    """a value column with its errors recorded in bad {row: Err} (the first per row) and replaced by NULL"""
    if isinstance(vals, ColArr) or not any(isinstance(v, Err) for v in vals): return vals
    out = list(vals)
    for i, v in enumerate(vals):
        if isinstance(v, Err): bad.setdefault(i, v); out[i] = None
    return out


def _rows_where_errs(e, rel, cx, bad):
    """rows_where, with a row whose condition is an error recorded in bad instead of failing the query"""
    rows, errs = filter_rows(e, rel, cx)
    for i, v in errs.items(): bad.setdefault(i, v)
    return rows


def _group_errs(out, members, bad):
    """the aggregate of a group holding a row recorded in bad is that row's error (the first of the group's rows)"""
    if not bad: return out
    out = list(out)
    for g, mem in enumerate(members):
        for i in mem:
            if i in bad: out[g] = bad[i]; break
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


# ------------------------------------------------------------------ windows from their aggregated series S(t)
# A window function whose value depends on its row only through the row's group of (PARTITION BY values, ORDER BY key)
# is a function of the window's aggregated series S(t) = sum over the groups of (rows, non-NULL counts, sums) t^key
# (docs/ENGINE.md, window analytics). One engine GROUP BY gives the series, on the planes or the GPU, without a row
# being read, sorted or decoded for it. The frame is then a polynomial applied to the series: a running aggregate is
# S(t) / (1 - t), prefix sums over the ordered groups; a frame of GROUPS offsets, or of ROWS offsets when every group is
# one row, is a difference of two such sums; the rank functions are prefix sums of the counts. Each row reads its
# group's value. With ORDER BY x ... LIMIT n on the window's own ORDER BY key x and a frame that ends at the current
# row's peers (causal: no later row counts), the rows output need the series only up to the key B of the n-th row,
# S(t) mod t^(B+1): the GROUP BY runs under x <= B, and the block reads only the rows that can be output (rank selection
# on the planes, as ORDER BY ... LIMIT does without windows).
SERIES_WINDOWS = True
SERIES_STRICT = __import__("os").environ.get("TORBLES_SERIES_STRICT") == "1"   # tests and fuzzing: an unexpected error
                            # in the series path is raised (else the window is computed from its rows)
SERIES_LUT = 1 << 22        # rows look their group up by code tables while the codes' spans fit this, else by values
# without a LIMIT bound, a series is read when its key columns' groups are this much fewer than the rows (each group
# costs some Python: at 1M rows, 62K groups take about 0.1 s, the rows about 0.4 s)
SERIES_ROWS_PER_GROUP = int(__import__("os").environ.get("TORBLES_SERIES_ROWS_PER_GROUP", 16))
_SERIES_RANKS = {"row_number", "rank", "dense_rank", "percent_rank", "cume_dist"}
_SERIES_AGGS = {"count", "sum", "avg", "mean"}
_SERIES_KEY_TYPES = {"VARCHAR", "DATE", "TIMESTAMP", "BOOLEAN"}


class _Series:
    """a window's value per group of its (partition, order key) values: keys (the expressions), gvals (each key's value
    per group), vals (the window's value per group); truncated: only the groups up to an ORDER BY ... LIMIT bound (rows
    past it are never output)"""
    __slots__ = ("keys", "gvals", "vals", "truncated")
    def __init__(self, keys, gvals, vals, truncated): self.keys, self.gvals, self.vals, self.truncated = keys, gvals, vals, truncated

    def values_for(self, rel, cx):
        miss = object()
        out = self._by_codes(rel, cx, miss) if rel.n > 4096 else None
        if isinstance(out, ColArr): return out
        if out is None:
            cols = [ev(e, rel, cx) for e in self.keys]
            for col in cols:
                for v in col:
                    if isinstance(v, Err): raise SqlError(v.msg)
            tab = dict(zip(zip(*self.gvals), self.vals)) if self.keys else {(): v for v in self.vals}
            out = [tab.get(t, miss) for t in zip(*cols)] if cols else [tab.get((), miss)] * rel.n
        bad = [i for i, v in enumerate(out) if v is miss]
        if bad and not self.truncated: raise SqlError("internal: a row's group is missing from its window's series")
        for i in bad: out[i] = None                       # past the ORDER BY ... LIMIT bound: not output
        return out

    def _by_codes(self, rel, cx, miss):
        """the lookup on the rows' codes, without decoding them: each key's group values encoded in the rows' codec, a
        code table per key, the groups' mixed-radix index; None when a key is not a code column of a small span"""
        arrs = [codes_of(e, rel, cx) for e in self.keys]
        if not arrs or any(a is None or a.codes.dtype != np.int64 for a in arrs): return None
        ng = len(self.vals)
        gok, gidx = np.ones(ng, bool), np.zeros(ng, np.int64)
        idx, radix = np.zeros(rel.n, np.int64), 1
        for a, gv in zip(arrs, self.gvals):
            c = a.codes
            lo, hi = (int(c.min()), int(c.max())) if len(c) else (0, 0)
            if hi - lo >= SERIES_LUT: return None
            slot, codes = {}, []
            for v in dict.fromkeys(gv):
                if v is None: continue
                if a.codec is None:
                    if type(v) is not int: return None
                    code = v
                else:
                    iv = a.codec.interval("EQ", v)
                    if iv is None or iv[0] is None or iv[0] != iv[1]: continue      # no row can hold this value
                    code = int(iv[0])
                if lo <= code <= hi: slot[v] = len(codes); codes.append(code)
            m = len(codes) + 2                                # the values, NULL, a value no group has
            lut = np.full(hi - lo + 1, m - 1, np.int64)
            if codes: lut[np.array(codes, np.int64) - lo] = np.arange(len(codes))
            r = lut[c - lo]
            if a.valid is not None: r = np.where(a.valid, r, m - 2)
            gs = np.fromiter((m - 2 if v is None else slot.get(v, -1) for v in gv), np.int64, ng)
            gok &= gs >= 0                                    # a value the rows' codec cannot hold: no row is in the group
            gidx = gidx * m + np.maximum(gs, 0)
            idx = idx * m + r
            radix *= m
            if radix > SERIES_LUT: return None
        tys = set(map(type, self.vals))
        if tys and (tys <= {int} or tys <= {float}) and (tys != {int} or all(-(1 << 63) <= v < 1 << 63 for v in self.vals)):
            dt = np.int64 if tys <= {int} else np.float64      # the values as an array (no object per row)
            tab, hit = np.zeros(radix, dt), np.zeros(radix, bool)
            tab[gidx[gok]] = np.array(self.vals, dt)[gok]; hit[gidx[gok]] = True
            if bool(hit[idx].all()): return ColArr(tab[idx])
        table = np.full(radix, miss, object)
        vals = np.empty(ng, object); vals[:] = self.vals
        table[gidx[gok]] = vals[gok]
        return table[idx].tolist()


def _series_key_ok(e):
    return isinstance(e, I.Col) and T.coll_key(e.t) is None and (e.t.exact or e.t.id in _SERIES_KEY_TYPES)


def _series_spec(w):
    """how window w is computed from its series, or None: (frame mode, start, end, single, causal): the frame's edges
    as (kind, offset) counted in groups; single: every group must be one row (ROWS frames, ROW_NUMBER); causal: the
    frame ends at the current row's peers, so no later row counts"""
    if w.distinct or w.filter is not None or w.rpr is not None or w.agg_order or w.params: return None
    if w.exclude and w.exclude.upper().replace(" ", "_") != "NO_OTHERS": return None
    if w.name in _SERIES_RANKS or w.name == "count_star":
        if w.args: return None
    elif w.name not in _SERIES_AGGS or len(w.args) != 1 or not isinstance(w.args[0], I.Col) or not w.args[0].t.exact:
        return None
    if len(w.order) > 1 or not all(_series_key_ok(p) for p in w.partition) or not all(_series_key_ok(o[0]) for o in w.order):
        return None
    if w.name in _SERIES_RANKS:                            # no frame: the peers that precede, and the partition's size
        return ("range", ("unbounded_preceding", 0), ("current", 0), w.name == "row_number",
                bool(w.order) and w.name not in ("percent_rank", "cume_dist"))
    mode, a, b = w.frame if w.frame is not None else ("range", ("unbounded_preceding", None), ("current", None))
    if not w.order:
        if mode == "rows": return None                     # positions in an unordered partition
        if mode == "range":                                # every row a peer of every other: the whole partition
            a = ("unbounded_preceding", None) if a[0] == "current" else a
            b = ("unbounded_following", None) if b[0] == "current" else b
    edges = []
    for kind, off in (a, b):
        if kind in ("preceding", "following"):             # constant offsets in groups (or single-row groups)
            if mode == "range" or not isinstance(off, I.Lit) or type(off.v) is not int or off.v < 0: return None
            edges.append((kind, off.v))
        else: edges.append((kind, 0))
    return mode, edges[0], edges[1], mode == "rows", bool(w.order) and edges[1][0] in ("current", "preceding")


def _series_top(s):
    """(first ORDER BY expression, desc, nulls first, rows read with the OFFSET) when the block outputs a prefix of
    its first ORDER BY key's order (ORDER BY ... LIMIT), else None"""
    if not s.order or s.limit is None or s.percent or s.distinct or s.distinct_on is not None or s.qualify is not None: return None
    lim = s.limit.v if isinstance(s.limit, I.Lit) else s.limit
    off = 0 if s.offset is None else (s.offset.v if isinstance(s.offset, I.Lit) else s.offset)
    if type(lim) is not int or type(off) is not int or lim <= 0 or off < 0 or lim + off > TOPK_MAX: return None
    e, desc, nf = s.order[0]
    if isinstance(e, tuple): e = s.items[e[1]][0] if e[0] == "pos" else e[1]
    return (e, desc, nf, lim + off) if isinstance(e, I.Col) else None


def _series_domain(keys):
    """an upper bound on the key columns' groups (their stored codes' spans), inf when unknown"""
    dom = 1
    for e in keys:
        t = getattr(e.b, "table", None)
        try: m = int(t.maxcode(e.name)) + 1
        except Exception: return float("inf")
        dom *= m + (1 if e.name in getattr(t, "nullable", {}) else 0)
    return dom


def series_windows(s, wins, cx):
    """{window key: _Series} for the windows of block s computed from their aggregated series (above); the other
    windows are computed from the block's rows"""
    if not SERIES_WINDOWS or not COLUMNAR or s.aggregate or s.from_ is None or s.sample is not None: return {}
    bindings = block_bindings(s.from_)
    if not bindings or any(b.table is None or b.sample is not None for b in bindings): return {}
    conds = [s.where] + [j.cond for j in join_conds(s.from_)]
    if any(c is not None and (not _deterministic(c) or I.has(c, I.Outer)) for c in conds): return {}
    if any(getattr(j, "partition", None) is not None for j in join_conds(s.from_)): return {}
    nrows = max(int(b.table.n) for b in bindings)
    if nrows < ALG_MIN_ROWS: return {}
    sigs = {}
    for k, w in wins.items():
        sp = _series_spec(w)
        if sp is None or any(I.has(x, I.Outer) for x in w.children()): continue
        sig = (tuple(p.key() for p in w.partition), tuple((o[0].key(), bool(o[1]), bool(o[2])) for o in w.order))
        sigs.setdefault(sig, []).append((k, w, sp))
    top = _series_top(s)
    out = {}
    for ws in sigs.values():
        w0 = ws[0][1]
        keys = list(w0.partition) + [o[0] for o in w0.order]
        trunc = None
        if top is not None and w0.order and all(sp[4] for _, _, sp in ws):
            o, desc, nf = w0.order[0]
            if top[0].key() == o.key() and bool(top[1]) == bool(desc) and bool(top[2]) == bool(nf): trunc = (o, desc, nf, top[3])
        if trunc is None and _series_domain(keys) * SERIES_ROWS_PER_GROUP > nrows: continue     # too many groups: no gain
        try: got = _run_series(s, keys, ws, trunc, cx)
        except SqlError:
            if SERIES_STRICT: raise
            continue                                       # the block raises its own errors as it reads its rows
        except Exception:
            if SERIES_STRICT: raise
            continue
        out.update(got)
    cx.stats["series_windows"] = cx.stats.get("series_windows", 0) + len(out)
    return out


def _dec_int(v, sc):
    """an exact DECIMAL (or integer) value times 10^sc, as an integer"""
    if type(v) is int: return v * 10 ** sc
    sg, dg, ex = v.as_tuple()
    if ex + sc < 0: raise SqlError("internal: a DECIMAL sum with more digits than its scale")
    c = int("".join(map(str, dg))) * 10 ** (ex + sc) if dg else 0
    return -c if sg else c


def _run_series(s, keys, ws, trunc, cx):
    """the windows ws (one PARTITION BY and ORDER BY) from one engine GROUP BY of their series -> {key: _Series}"""
    import sqlfront, sqlwin
    where = s.where
    if trunc is not None:                                  # the key of the n-th row output bounds the series
        o, desc, nf, k = trunc
        qb = I.Select(); qb.from_ = s.from_; qb.where = s.where; qb.items = [(o, "b")]
        qb.order = [(("expr", o), desc, nf)]; qb.limit, qb.offset = 1, k - 1
        got = run_select(qb, cx).rows
        p = None
        if got:
            B = got[0][0]
            if B is None: p = sqlfront.make_func("is_null", [o]) if nf else None
            else:
                p = sqlfront.make_func(">=" if desc else "<=", [o, I.Lit(B, o.t)])
                if nf: p = sqlfront.make_func("or", [p, sqlfront.make_func("is_null", [o])])
        if p is None: trunc = None                         # fewer rows than that, or NULLs last: the whole series
        else: where = p if where is None else sqlfront.make_and([where, p])
    q = I.Select(); q.from_ = s.from_; q.where = where; q.aggregate = True; q.group = list(keys)
    q.items = [(e, f"k{i}") for i, e in enumerate(keys)] + [(I.Agg("count_star", [], T.BIGINT), "n")]
    col = {}
    for _, w, _ in ws:
        if w.name in _SERIES_AGGS and w.args[0].key() not in col:
            x = w.args[0]; col[x.key()] = len(q.items)
            q.items += [(I.Agg("count", [x], T.BIGINT), f"c{len(q.items)}"),
                        (I.Agg("sum", [x], F.agg_typer("sum", [x.t])), f"s{len(q.items) + 1}")]
    res = run_select(q, cx)
    cols = res._cols if res._cols is not None else [list(c) for c in zip(*res.rows)] or [[] for _ in q.items]
    ng = len(cols[0])
    if not ng: return {k: _Series(keys, [[] for _ in keys], [], trunc is not None) for k, _, _ in ws}
    nk = len(keys); npart = nk - (1 if ws[0][1].order else 0)
    spec = ws[0][1].order[0] if ws[0][1].order else None
    # the groups partition after partition, each in the window's order (dense codes, one lexsort)
    sk = [sqlwin.order_codes(list(cols[j]), False, False) for j in range(npart)]
    sk_all = ([sqlwin.order_codes(list(cols[npart]), bool(spec[1]), bool(spec[2]))] if spec is not None else []) + sk[::-1]
    perm = np.lexsort(sk_all) if sk_all else np.arange(ng)
    new = np.zeros(ng, bool); new[0] = True
    for c in sk:
        cs_ = c[perm]; new[1:] |= cs_[1:] != cs_[:-1]
    g = np.arange(ng)
    starts = np.flatnonzero(new); ends = np.append(starts[1:], ng)
    pid = np.cumsum(new) - 1
    ps, pe = starts[pid], ends[pid]                        # each group's partition, as [ps, pe) of the sorted groups
    gi, G = g - ps, pe - ps                                # its index in the partition, the partition's groups
    cnt = np.asarray(cols[nk], np.int64)[perm]
    PC = np.concatenate(([0], np.cumsum(cnt)))
    before, N = PC[g] - PC[ps], PC[pe] - PC[ps]           # rows of the partition before the group; the partition's rows
    gvals = [[cols[j][i] for i in perm.tolist()] for j in range(nk)]
    out = {}
    for k, w, (mode, a, b, single, _) in ws:
        if single and not (cnt == 1).all(): continue       # ROWS over peers: positions within the groups
        out[k] = _Series(keys, gvals, _series_values(w, a, b, gi, G, ps, cnt, PC, before, N, cols, col, perm), trunc is not None)
    return out


def _series_values(w, a, b, gi, G, ps, cnt, PC, before, N, cols, col, perm):
    """window w's value at every group (in the sorted order), from the groups' counts and sums: the frame's edges in
    groups (sqlwin._edge on positions), prefix sums over the sorted groups"""
    name = w.name
    if name in ("row_number", "rank"): return (before + 1).tolist()
    if name == "dense_rank": return (gi + 1).tolist()
    if name == "percent_rank": return np.where(N <= 1, 0.0, before / np.maximum(N - 1, 1)).tolist()
    if name == "cume_dist": return ((before + cnt) / np.maximum(N, 1)).tolist()     # (N = 0: no row reads it)
    gmax = int(np.max(G, initial=0))
    def edge(bd, side):
        kind, off = bd
        if kind == "unbounded_preceding": return np.zeros_like(gi)
        if kind == "unbounded_following": return G
        if kind == "current": return gi if side == "a" else gi + 1
        off = min(off, gmax + 1)                           # past every group (an offset up to 2^63 - 1 must not wrap)
        t = gi - off if kind == "preceding" else gi + off
        return np.where(t < 0, 0, np.where(t >= G, G, t if side == "a" else t + 1))
    lo = np.minimum(np.maximum(edge(a, "a"), 0), G)
    hi = np.maximum(lo, np.minimum(edge(b, "b"), G))
    L, H = ps + lo, ps + hi
    if name == "count_star": return (PC[H] - PC[L]).tolist()
    x = w.args[0]; j = col[x.key()]
    dec = x.t.id == "DECIMAL"; sc = x.t.s if dec else 0
    nn = np.asarray(cols[j], np.int64)[perm]
    PN = np.concatenate(([0], np.cumsum(nn)))
    c = PN[H] - PN[L]
    if name == "count": return c.tolist()
    sv = cols[j + 1]
    sums = [0 if v is None else _dec_int(v, sc) for v in (sv[i] for i in perm.tolist())] if dec else \
        [0 if v is None else v for v in (sv[i] for i in perm.tolist())]
    big = max(map(abs, sums)) * (len(sums) + 1) >= 1 << 62
    X = np.array(sums, object if big else np.int64)
    PS = np.concatenate((np.zeros(1, X.dtype), np.cumsum(X)))
    tot = PS[H] - PS[L]
    has = c > 0
    res = np.full(len(c), None, object)
    if name == "sum":
        if dec:
            D = Decimal(10 ** sc)
            res[has] = [T.CTX.divide(Decimal(int(t)), D) for t in tot[has].tolist()]
        else: res[has] = tot[has].astype(object) if not big else tot[has]
    else:                                                  # AVG: the exact quotient, rounded once
        f = 10 ** sc
        if not big and int(np.abs(tot).max(initial=0)) < 1 << 53 and int(c.max(initial=0)) * f < 1 << 53:
            res[has] = (tot[has] / (c[has] * f)).astype(object)
        else: res[has] = [int(t) / (int(d) * f) for t, d in zip(tot[has].tolist(), c[has].tolist())]
    return res.tolist()


# ------------------------------------------------------------------ windows
VEC_WINDOWS = True          # window functions as array algebra (sqlwin); False: row by row on each partition (tests)


def compute_windows(wins, rel, cx, series=None):
    import sqlwin
    n = rel.n
    layouts = {}                                           # windows with the same PARTITION BY / ORDER BY share one ordering
    for w in wins:
        if w.key() in rel.cols: continue
        if rel.cols.array(w.key()) is not None:
            continue                                       # computed from the source planes before gathering payload
        if series and w.key() in series:                   # from the window's aggregated series: each row's group's value
            v = series[w.key()].values_for(rel, cx)
            if isinstance(v, ColArr): rel.cols.put(w.key(), v)
            else: rel.cols[w.key()] = v
            continue
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
        if (VEC_WINDOWS and n and len(wins) == 1 and w.name == "row_number" and not w.args
                and (w.frame is None or all(bound[1] is None for bound in w.frame[1:]))
                and w.rpr is None and w.filter is None and not w.agg_order
                and all(isinstance(col, ColArr) for col in parts + okeys)):
            rel.cols.put(w.key(), ColArr(sqlwin.row_numbers(parts, okeys, specs, n, stats=cx.stats)))
            continue
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
                if isinstance(out, np.ndarray) and out.dtype in (np.int64, np.float64):
                    full = np.empty(n, out.dtype); full[L.perm] = out
                    rel.cols.put(w.key(), ColArr(full))        # kept as an array: decoded only if read as values
                    continue
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


def _frac_range(r):
    return None if r is None else (Fraction(r[0]), Fraction(r[1]))


def _within(r, t):
    """is the interval r (Fractions) inside the values of the integer / DECIMAL type t?"""
    tr = _frac_range(type_range(t))
    return tr is not None and tr[0] <= r[0] and r[1] <= tr[1]


def _exact_cast(a, t):
    """a cast from exact type a to exact type t that keeps every value that fits t (no rounding: an integer, or a
    DECIMAL to a scale at least its own)"""
    if t.integer: return a.integer or a.id == "DECIMAL" and a.s == 0
    return t.id == "DECIMAL" and (a.integer or a.id == "DECIMAL" and a.s <= t.s)


def _widening(a, t):
    """does CAST from type a to type t give back every value of a unchanged (never rounding, never failing)?"""
    if a.exact and t.exact: return _exact_cast(a, t) and _within(_frac_range(type_range(a)), t)
    if a.exact and t.floating:                       # every value an exact double / float
        r = _frac_range(type_range(a))
        return a.integer and max(-r[0], r[1]) <= (2 ** 53 if t.id == "DOUBLE" else 2 ** 24) or \
            a.id == "DECIMAL" and t.id == "DOUBLE" and a.p <= 15
    if a.id == "DATE" and t.id == "TIMESTAMP": return True
    return a.id == "VARCHAR" and t.id == "VARCHAR" and t.p is None      # (a length cuts strings)


def _strip_cast(e):
    """an expression under conversions that keep every value of its type (widening casts, DATE -> TIMESTAMP): a
    narrowing cast stays (the planner's own proofs, on the values a column holds, may remove one: _Planner.strip)"""
    while isinstance(e, I.Cast) and not e.try_ and _widening(e.arg.t, e.t): e = e.arg
    return e


def expr_range(e, col_range):
    """[lo, hi] (Fractions) bounding the values of an exact expression, proved from col_range(column) -> [lo, hi] or
    None, when every cast and integer / DECIMAL operation in it keeps its type on those values (interval arithmetic);
    None when that is not proved. An expression it bounds can be computed exactly (an engine polynomial, a column
    compared without its casts) with the values and errors of its SQL types: none of its operations can fail."""
    if isinstance(e, I.Lit):
        v = e.v
        if type(v) is int or type(v) is Decimal and v.is_finite(): return (Fraction(v), Fraction(v))
        return None
    if isinstance(e, I.Col): return col_range(e) if e.t.exact else None
    if isinstance(e, I.Cast):
        if e.try_ or not (e.arg.t.exact and _exact_cast(e.arg.t, e.t)): return None
        r = expr_range(e.arg, col_range)
        return r if r is not None and _within(r, e.t) else None
    if isinstance(e, I.Func) and (e.fn in ("+", "-", "*", "__xmul") and len(e.args) == 2 or e.fn == "neg" and len(e.args) == 1):
        rs = [expr_range(a, col_range) for a in e.args]
        if any(r is None for r in rs): return None
        if e.fn == "neg": r = (-rs[0][1], -rs[0][0])
        elif e.fn == "+": r = (rs[0][0] + rs[1][0], rs[0][1] + rs[1][1])
        elif e.fn == "-": r = (rs[0][0] - rs[1][1], rs[0][1] - rs[1][0])
        else:
            c = [x * y for x in rs[0] for y in rs[1]]
            r = (min(c), max(c))
        return r if e.fn == "__xmul" or _within(r, e.t) else None            # (__xmul: the planner's exact product)
    return None


def _shifted(e):
    """e without its top x + c of a constant c, of SMALLINT / INTEGER / BIGINT (DuckDB's SUM / AVG rewrites take c out
    of the sum: x + c is not computed)"""
    while isinstance(e, I.Func) and e.fn == "+" and len(e.args) == 2 and e.t.id in ("SMALLINT", "INTEGER", "BIGINT"):
        l, r = e.args
        if F.lit_int(r) and not F.lit_int(l): e = l
        elif F.lit_int(l) and not F.lit_int(r): e = r
        else: break
    return e


def stored_range(t, name):
    """[lo, hi] (Fractions) bounding every value stored column `name` of engine table t holds (NULL rows store code 0,
    a value of the column too): the span of its codes through its codec; None when unknown"""
    try: mx = int(t.maxcode(name)) if hasattr(t, "maxcode") else int(t.colmax(name))
    except Exception: return None
    cd = getattr(t, "codecs", {}).get(name)
    if cd is None: return (Fraction(0), Fraction(mx))             # plain codes: the values themselves
    aff = getattr(cd, "affine", None)
    if aff is not None and getattr(cd, "kind", "") in ("int", "decimal"):
        off, sc = aff
        return (Fraction(off, 10 ** sc), Fraction(off + mx, 10 ** sc))
    kind = getattr(cd, "kind", "")
    if kind == "rank" and getattr(cd.base, "affine", None) is not None and cd.raw:     # wide values: ranks of the sorted ones
        lo, hi = cd.decode([0, min(mx, len(cd.raw) - 1)])
    elif kind == "dict" and getattr(cd, "numeric", False) and not getattr(cd, "floating", False) and cd.values:
        lo, hi = cd.values[0], cd.values[min(mx, len(cd.values) - 1)]            # sorted, as the codes are
    else: return None
    if not all(type(x) is int or type(x) is Decimal and x.is_finite() for x in (lo, hi)): return None
    return (Fraction(lo), Fraction(hi))


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
    if fact is None:
        if left: return None, None
        return _plan_snowflake(s, bindings, joins, conj, edges, need, aggs, cx)
    if fact in left: return None, None
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


def _plan_snowflake(s, bindings, joins, conj, edges, need, aggs, cx):
    """Fold ancestor filters into masks on the fact's direct dimensions.

    A subtree that only supplies filters can keep or remove a row.
    Group keys can also be pulled back through the tree. Other ancestor
    values are provided by the original joins after a fact prefilter.
    """
    refuse = (None, None)
    if len(bindings) < 3 or s.sample is not None or any(b.sample is not None for b in bindings):
        return refuse
    if len({id(b.table) for b in bindings}) != len(bindings):
        return refuse
    incoming = {}
    children = {b: [] for b in bindings}
    for cond, source, target in edges:
        if not all(_join_key_term(arg) and _strip_cast(arg).t.integer for arg in cond.args):
            continue
        table, fk = source.b.table, source.name
        if fk not in table.fks_to(target.b.table):
            continue
        if target.b in incoming or getattr(getattr(table, "surr", {}).get(fk), "stale", None):
            return refuse
        incoming[target.b] = (cond, source, target)
        children[source.b].append(target.b)
    roots = [b for b in bindings if b not in incoming]
    if len(roots) != 1:
        return refuse
    fact = roots[0]
    reachable = [fact]
    for b in reachable:
        reachable.extend(children[b])
    if len(reachable) != len(bindings):
        return refuse
    direct = {b: incoming[b] for b in children[fact]}
    deep = set(incoming) - set(direct)
    if not deep:
        return refuse
    consumed = {id(edge[0]) for edge in incoming.values()}
    if any(id(c) not in consumed for j in joins for c in _conjuncts(j.cond)):
        return refuse

    outputs = [x for x, _ in s.items] + list(s.group) + [s.having, s.qualify]
    outputs += [o[0][1] for o in s.order if o[0][0] == "expr"]
    outputs += [d[1] for d in (s.distinct_on or []) if d[0] == "expr"]
    needs_deep = any(c.b in deep for c in need.values()) or any(
        c.b in deep for expr in outputs if expr is not None for c in I.cols_of(expr))

    predicates, rest = {}, []
    try:
        for c in conj:
            if id(c) in consumed:
                continue
            refs = {col.b for col in I.cols_of(c)}
            if not refs & deep:
                rest.append(c)
                continue
            if len(refs) != 1 or not _deterministic(c):
                return refuse
            b = next(iter(refs))
            local = _Planner(s, b, {}, {}, cx)
            predicate = local.native(c)
            if predicate is None or local.hidden:
                return refuse
            predicates.setdefault(b, []).append(predicate)

        projections = _snowflake_group_keys(s, incoming, direct, deep, need, aggs, outputs, rest, cx.engine) if needs_deep else None
        if needs_deep and not projections and not predicates:
            return refuse
        masks = {}
        for b in reversed(reachable[1:]):
            if not children[b] and b not in predicates:
                continue
            q = algq.Query(b.table, join_mode="auto")
            for child in children[b]:
                q.join(child.table, incoming[child][1].name)
                if child in masks:
                    q.where(masks.pop(child))
            for predicate in predicates.get(b, ()):
                q.where(predicate)
            q._rows = q._rowbits = True
            bitmap = q._run_codes("plain" if cx.engine == "plain" else "alg")
            cx.stats["engine_queries"] += 1
            masks[b] = algq.RowMask._from_bitmap(b.table, bitmap)
    except (ValueError, TypeError, OverflowError, SqlError, _NotStarNow):
        return refuse

    fks = {edge[1].name: b for b, edge in direct.items()}
    planner = _Planner(s, fact, direct, fks, cx)
    planner.samples.extend(masks.values())
    if needs_deep:
        result = _snowflake_grouped(planner, incoming, direct, masks, rest, need, aggs, projections) if projections else None
        if result is None:
            if not predicates:
                return refuse
            block = I.Select()
            block.from_ = fact
            planner = _Planner(block, fact, direct, fks, cx)
            planner.samples.extend(masks.values())
            return run_general(s, bindings, need, aggs, cx, prejoin=(planner, len(deep)))
    else:
        try:
            result = planner.run(rest, need, aggs)
        except (ValueError, SqlError, _NotStarNow):
            return refuse
    cx.stats["star_blocks"] += 1
    cx.stats["snowflake_blocks"] += 1
    cx.stats["snowflake_edges"] += len(deep)
    return result


def _snowflake_group_keys(s, incoming, direct, deep, need, aggs, outputs, rest, engine):
    """The ancestor group keys that can be represented by columns on direct dimensions."""
    if not s.aggregate or not s.group or engine not in ("alg", "gpu"):
        return None
    group_cols = {g.key(): g for g in s.group if isinstance(g, I.Col) and g.b in deep}
    required = {c.key() for c in need.values() if c.b in deep}
    required.update(c.key() for e in outputs if e is not None for c in I.cols_of(e) if c.b in deep)
    if not required or not required <= group_cols.keys() or len(required) > algq.MAXKEY:
        return None
    # Deferred rank or distinct queries could outlive the temporary group dimensions.
    if any(a.distinct or a.name in _RANK_AGGS for a in aggs.values()):
        return None
    if any(c.b in deep for a in aggs.values() for c in I.cols_of(a)):
        return None
    if subqueries_in(outputs + rest):
        return None

    projections = {}
    for key, col in group_cols.items():
        if key not in required:
            continue
        t = col.b.table
        codec = t.codecs.get(col.name)
        if codec is None or codec.kind == "ref" or col.name in getattr(t, "wide", {}):
            return None
        root = col.b
        while root not in direct:
            root = incoming[root][1].b
        projections.setdefault(root, []).append((key, col, codec))
    return projections


def _snowflake_grouped(planner, incoming, direct, masks, rest, need, aggs, projections):
    """Compose ancestor group codes on dimension rows, then aggregate on the fact's planes."""
    # Table.space caches dimensions on its owner. Keep temporary dimensions out of the shared fact's cache.
    fact = planner.fact.table
    fact_view = copy.copy(fact)
    fact_view._spaces = dict(getattr(fact, "_spaces", {}))
    planner.tab[planner.fact] = fact_view
    transient = []
    try:
        for root, cols in projections.items():
            used = np.unpackbits(masks[root].bm.view(np.uint8), bitorder="little")[:root.table.n].astype(bool)
            selected = np.flatnonzero(used)
            rows = {root: selected}

            def pullback(binding):
                if binding not in rows:
                    _, source, _ = incoming[binding]
                    parent = source.b.table
                    fk = parent.decoded(source.name)[pullback(source.b)]
                    mask = parent.kmask(source.name)
                    rows[binding] = fk & np.uint32(mask) if mask else fk
                return rows[binding]

            data, codecs, nullable = {}, {}, {}
            for index, (key, col, codec) in enumerate(cols):
                name = f"p{index}"
                ids = pullback(col.b)
                data[name] = np.zeros(root.table.n, np.uint32)
                data[name][selected] = col.b.table.decoded(col.name)[ids]
                codecs[name] = codec
                validity = getattr(col.b.table, "nullable", {}).get(col.name)
                if validity is not None:
                    nullable[name] = valid_name = f"{name}#v"
                    data[valid_name] = np.zeros(root.table.n, np.uint32)
                    data[valid_name][selected] = col.b.table.decoded(validity)[ids]
            dim = algq.Table(f"{root.table.name}#snowflake", data, used=used)
            transient.append(dim)
            dim.codecs, dim.nullable = codecs, nullable
            binding = I.Binding(dim.name, [], table=dim)
            planner.tab[binding] = dim
            planner.joins[binding] = direct[root][1].name
            for index, (key, _, _) in enumerate(cols):
                planner.snowflake_keys[key] = binding, f"p{index}"
        return planner.run(rest, need, aggs)
    except (ValueError, TypeError, OverflowError, IndexError, SqlError, _NotStarNow):
        return None
    finally:
        for dim in transient:
            dim.close()


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
        self.snowflake_keys = {}

    # columns -------------------------------------------------------------
    def canon(self, col):
        """the column the engine reads for col: a fact FK column is its raw values (when some references have no
        match) or its dimension's key (the stored surrogate is not the value); -> (binding, column name)"""
        if self.snowflake_keys and (projected := self.snowflake_keys.get(col.key())) is not None:
            return projected
        if col.b is self.fact:
            fi = _fk_info(col)
            if fi is not None:
                if col.name + "#raw" in self.fact.table.cols: return self.fact, col.name + "#raw"
                b = self.fkb.get(col.name)
                if b is not None and b in self.nulled: b = None
                if b is None: b = self.hidden_join(col.name, (fi[1], col.t, False))
                return b, fi[1]
            mv = (getattr(self.fact.table, "moved", None) or {}).get(col.name)
            if mv is not None:                          # stored once per value of mv: a column of mv's dimension
                return self.hidden_join(mv, (col.name, col.t, True)), col.name
        return col.b, col.name

    def hidden_join(self, fk, col):
        """the binding of the dimension that fact foreign key fk references, joined implicitly (a hidden binding: the
        query reads a column there that it names as the fact's, col = (name, type, nullable))"""
        b = self.hidden.get(fk)
        if b is not None:
            if b not in self.joins: self.joins[b] = fk
            if all(c != col[0] for c, _, _ in b.cols): b.cols.append(col)
            return b
        orph = getattr(self.fact.table, "orphans", {}).get(fk, 0)
        if orph: raise SqlError(f"{self.fact.table.name}.{fk} has {orph} reference(s) without a match; "
                                "their values are not stored (join the referenced table instead)")
        dimt = self.fact.table._refdim[fk]
        b = I.Binding(f"__{fk}", [col], table=dimt)
        t = dimt
        if any(x is dimt for x in self.tab.values()): t = copy.copy(dimt)
        self.tab[b] = t; self.joins[b] = fk; self.hidden[fk] = b
        return b

    def algcol(self, col):
        b, name = self.canon(col)
        return algq.Col(self.tab[b], name)

    def binding_of(self, col): return self.canon(col)[0]

    def strip(self, e):
        """a column under conversions that keep its values: each cast proved on the values its argument holds (a
        narrowing integer / DECIMAL cast whose source values fit the target; an exact column converted to DOUBLE or
        REAL only when every value it holds is exactly one). A cast that can fail or round stays: the predicate is
        then not a plane predicate, and the cast is evaluated (with its error) on the values"""
        while isinstance(e, I.Cast) and not e.try_ and self.keeps_values(e): e = e.arg
        return e

    def value_range(self, col):
        """[lo, hi] bounding the values the engine reads for column col (its stored codes' span), or None"""
        b, name = self.canon(col)
        return stored_range(self.tab[b], name)

    def exact_range(self, e):
        """[lo, hi] of an exact expression of the block's columns, when its casts and integer / DECIMAL operations
        provably keep their types on the values the columns hold (expr_range); None otherwise"""
        return expr_range(e, self.value_range)

    def keeps_values(self, c):
        """does CAST c give back every value its argument takes here, unchanged and without an error?"""
        a, t = c.arg.t, c.t
        if _widening(a, t): return True
        if not a.exact or not (t.exact or t.floating): return False
        r = self.exact_range(c.arg)
        if r is None: return False
        if t.exact: return _exact_cast(a, t) and _within(r, t)
        m = max(-r[0], r[1])                               # to a binary float: every value one exactly
        if a.integer or a.id == "DECIMAL" and a.s == 0: return m <= (2 ** 53 if t.id == "DOUBLE" else 2 ** 24)
        return t.id == "DOUBLE" and m * 10 ** a.s < 10 ** 15       # decimals of at most 15 digits (they round-trip)

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

    def rowmask(self, b, e, lazy=False):
        """a predicate on one dimension: evaluated on the dimension's distinct value tuples -> its passing rows. An
        error on a row in use fails the query, or with lazy (a WHERE term) gives None: the term then stays residual,
        evaluated on the joined rows that pass the block's other terms (an error matters only where it is used)"""
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
            bad = [i for i, x in enumerate(res) if isinstance(x, Err)]
            if bad:
                inuse = np.asarray(t.used[:n], bool) if len(t.used) >= n else np.ones(n, bool)
                hit = np.intersect1d(np.array(bad, np.int64), inv[inuse])
                if len(hit):
                    if lazy: return None
                    raise SqlError(res[int(hit[0])].msg)
            ok = np.array([x is True for x in res], bool)
            mask = ok[inv]
        mask &= t.used[:n] if len(t.used) >= n else mask
        return algq.RowMask(t, mask, names + vcols, sql=I.render(e))     # the selection is a function of these columns

    def codeset(self, col, e):
        """a predicate on one fact column: evaluated on every code the column can hold -> intervals or a code set
        (returns a list of engine predicates, or None). An error on a value the column holds gives None as well: the
        term stays residual, evaluated on the rows that pass the block's other terms, so that it fails the query only
        if a row needs that value (DuckDB evaluates such a term after the cheaper ones)"""
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
        if isinstance(at_null, Err): return None
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
            if any(isinstance(x, Err) for x in res): return None
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
                if len(hit): return None
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
    def poly(self, e, shift=False):
        """an exact polynomial (algq expression) of e, or None (NULLs are excluded by the caller's variant). The engine
        sums polynomials exactly, without the SQL types of their parts: e is lowered only when its casts and integer /
        DECIMAL operations provably keep their types on the values the columns hold (exact_range), so that no value
        the sums stand for is a conversion or overflow error. Otherwise the aggregate takes its argument's values,
        computed (and checked) per group. shift: e's top x + c (constant c) need not keep its type (an AVG that DuckDB
        computes as SUM(x) / COUNT(*) + c, never computing x + c: _shifted)"""
        if self.exact_range(_shifted(e) if shift else e) is None: return None
        return self._poly(e)

    def _poly(self, e):
        if isinstance(e, I.Cast):                     # proved to keep every value (exact_range)
            return self._poly(e.arg)
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
            if e.fn in ("+", "-", "*", "__xmul") and len(e.args) == 2:
                if not (e.args[0].t.exact and e.args[1].t.exact): return None
                l, r = self._poly(e.args[0]), self._poly(e.args[1])
                if l is None or r is None: return None
                return algq.Arith("*" if e.fn == "__xmul" else e.fn, l, r)
            if e.fn == "neg":
                a = self._poly(e.args[0]); return None if a is None else algq.Arith("-", algq.Const(0), a)
        return None

    def poly_of_tuple(self, m, shift=False):
        if isinstance(m, tuple):
            a, b = self.poly_of_tuple(m[1]), self.poly_of_tuple(m[2])
            return algq.Arith("*", a, b)
        return self.poly(m, shift)

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
            # dimension-side ON terms: an error on a dimension row counts only for the pairs the join's equality
            # matches, so a row mask that errs leaves the block to run_general (ON terms on the matched pairs)
            md = self.rowmask(b, _and(cd), lazy=True) if cd else None
            if cd and md is None: raise _NotStarNow()
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
        if not empty and not residual:
            from torbles import sqlencodedgroups, sqlgroupplans, sqlscalarplanes, sqltextplanes
            grouped = sqlscalarplanes.run(self, preds, aggs)
            if grouped is None:
                grouped = sqlencodedgroups.run(self, preds, aggs)
            if grouped is None:
                grouped = self.raw_plane_group(preds, aggs)
            if grouped is not None:
                return grouped, {}
            for lower in (sqlgroupplans.special_count, sqlgroupplans.late_count, sqltextplanes.project_group):
                grouped = lower(self, preds, aggs)
                if grouped is not None:
                    return grouped, {}
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
            if modes.get(k, "rank") == "rank" and not residual and not empty:
                rc = self.rank_col(a)
                if rc is not None: modes[k] = "rank"; aggplan[k] = dict(kind="rank", col=rc); continue
            if modes.get(k, "dist") == "dist" and not residual and not empty:
                dc = self.dist_col(a)
                if dc is not None: modes[k] = "dist"; aggplan[k] = dict(kind="dist", col=dc); continue
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
        def engine_keys():
            """the engine keys: one per distinct (binding, column), and the validity of a NULL-able one"""
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
            return keys, kpos, vpos, atom_list, nkeys
        keys, kpos, vpos, atom_list, nkeys = engine_keys()
        # rank selection and COUNT(DISTINCT) cells group by the block's keys and the column too: when that would pass
        # the engine's key limits, the aggregate takes its values (and the block, if need be, its rows) as before
        def kbits(k):
            t = k.table; W = getattr(t, "wide", None) or {}
            return int(t.width(k.name)) + (int(t.width(W[k.name][1])) if k.name in W else 0)
        def nullable(k): return isinstance(k, algq.Col) and k.name in (getattr(k.table, "nullable", None) or {})
        nn = sum(1 for x in keys if nullable(x))                 # (the typed engine call adds their validity again)
        drop = [k for k, p in aggplan.items() if p.get("kind") in ("rank", "dist") and
                (nkeys + nn + 1 + (p["kind"] == "dist" and nullable(p["col"][0])) > algq.MAXKEY or sum(kbits(x) for x in keys) +
                 (int(self.tab[self.fact].width(p["col"][0])) if p["kind"] == "rank" else kbits(p["col"][0])) > 120)]
        if drop:
            for k in drop:
                aggplan.pop(k); modes[k] = "value"; a = aggs[k]
                for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                    for col in I.cols_of(x): atoms[col.key()] = col
            keys, kpos, vpos, atom_list, nkeys = engine_keys()
        def to_values():
            """every aggregate from its arguments' values in every pass: the passes of a block agree on how each
            aggregate is computed (a later pass cannot switch an aggregate an earlier one computed as a state)"""
            if not first_pass and any(modes.get(k) == "state" for k in aggs): raise _NotStarNow()
            for k in aggs: modes[k] = "value"
        # a block without aggregates returns one row per fact row: read the passing rows' columns from the planes
        # (a row bitmap) instead of grouping them by every column it reads
        rows_block = ROWS_BLOCK and not s.aggregate and not aggs and (bool(s.items) or self.dense_atoms(atoms))
        # Native groups are already SQL groups when the exact same nonnullable integer keys were grouped
        # and every aggregate has a state for the whole group. Multiple LEFT JOIN passes can repeat keys.
        direct_groups = (
            COLUMNAR and VEC_GROUPS and s.aggregate and bool(s.group) and s.sets is None and not self.left
            and not empty and not residual and nkeys <= algq.MAXKEY and self.fact_sample is None
            and all(isinstance(g, I.Col) and g.t.id in T.INT_RANGE for g in s.group)
            and set(atoms) == {g.key() for g in s.group} and not vpos
            and len(aggplan) == len(aggs)
            and all(p["kind"] == "moment" and p["variant"] == 0 and p.get("rowfilter_expr") is None
                    and aggs[k].name in ("sum", "count", "count_star") for k, p in aggplan.items())
        )
        if nkeys > algq.MAXKEY or self.fact_sample is not None or rows_block:
            # more columns than GROUP BY keys (or a sampled fact): the passing rows themselves (a row bitmap), their
            # columns read from the planes; every aggregate is then computed from its arguments' values
            to_values()
            for k, a in aggs.items():
                for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                    for col in I.cols_of(x):
                        (nulled_atoms if col.b in self.nulled else atoms)[col.key()] = col
            aggplan = {}
            ids = None
            if rows_block and not residual and not empty:
                from torbles.sqlrowplanes import bounded_ids
                from torbles.sqlscalarplanes import point_ids
                bounded = point_ids(self)
                if bounded is None:
                    bounded = bounded_ids(self, preds)
                if bounded is not None:
                    ids, scanned = bounded
                    # Internal join scans have requested atoms but no SELECT items.
                    if s.items:
                        payload = {c.key() for x, _ in s.items for c in I.cols_of(x)}
                        atoms = {k: c for k, c in atoms.items() if k in payload}
                    out = self.rows_rel(atoms, preds, False, ids=ids, fact_order=False)
                    out.plane_ordered = True
                    if s.order and scanned is not None:
                        cx.stats["raw_plane_topk_rows"] = scanned
                    return out, {}
                from torbles.sqlwindowplanes import run as cuda_window

                plane_rel = cuda_window(self, atoms, preds)
                if plane_rel is None:
                    plane_rel = self.raw_plane_rows(atoms, preds)
                if plane_rel is not None:
                    return plane_rel, {}
                ids = self.topk_ids(preds)
                if ids is None: preds = preds + self.topk_preds(preds)
            rel = self.rows_rel(atoms, preds, empty, ids=ids)
        else:
            try: rel = self.engine(keys, atom_list, preds, empty, variants, vpreds, mlist, direct_groups)
            except _TooManyGroups:
                direct_groups = False
                # the atoms' code domain passes 2^128: the passing rows themselves
                to_values()
                for k, a in aggs.items():
                    for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                        for col in I.cols_of(x):
                            (nulled_atoms if col.b in self.nulled else atoms)[col.key()] = col
                aggplan = {}
                rel = self.rows_rel(atoms, preds, empty)
        align = self.__dict__.pop("_align", None)
        for k, plan in list(aggplan.items()):                # rank selection / distinct cells, per output group
            if plan.get("kind") in ("rank", "dist"):
                fn = self.rank_values if plan["kind"] == "rank" else self.dist_values
                aggplan[k] = dict(kind="rank", fn=functools.partial(fn, plan["col"], aggs[k], rel, list(keys), list(preds),
                                                                    dict(self.joins), align))
        for k, col in nulled_atoms.items():               # the NULL side of a LEFT JOIN pass: NULL codes of the column's codec,
            t = self.tab.get(col.b)                        # so the passes concatenate as codes
            cd = getattr(t, "codecs", {}).get(col.name) if t is not None else None
            if COLUMNAR and cd is not None: rel.cols.put(k, ColArr(np.zeros(rel.n, np.int64), np.zeros(rel.n, bool), cd))
            else: rel.cols[k] = [None] * rel.n
        if residual:
            rel = rel.take(rows_where(_and(residual), rel, cx))
        for k, plan in aggplan.items():                      # an aggregate's FILTER, on the rows the WHERE keeps (an
            if plan.get("rowfilter_expr") is not None:       # error on a row it drops is not the query's)
                plan["rowfilter"] = set(rows_where(plan["rowfilter_expr"], rel, cx))
        if direct_groups and all(rel.cols.array(k) is not None and rel.cols.array(k).valid is None for k in atoms):
            # A base native group has a positive count. Its states can become SQL result columns directly.
            out = Rel(rel.n); out.cols = rel.cols; out.grouped = True
            for k, plan in aggplan.items():
                if aggs[k].name == "sum":
                    m = rel.mom[(0, plan["moments"][0])]
                    if isinstance(m, np.ndarray): out.cols.put(k, ColArr(m, codec=_INTC))
                    else: out.cols[k] = [0 if x is None else x for x in m]
                else: out.cols.put(k, ColArr(rel.cnt, codec=_INTC))
            cx.stats["direct_groups"] = cx.stats.get("direct_groups", 0) + 1
            return out, {}
        if isinstance(rel.cnt, np.ndarray): rel.cnt = rel.cnt.tolist()
        _states(rel, aggs, aggplan)
        return rel, {k: (aggplan[k] if aggplan[k].get("kind") == "rank" else dict(kind="state")) for k in aggs if k in aggplan}

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
        from torbles.sqltextplanes import dictionary_predicate
        fast = dictionary_predicate(self, c)
        if fast is not None:
            return fast
        if b is self.fact:
            cols = {self.canon(x) for x in I.cols_of(c)}
            if len(cols) == 1:
                return self.codeset(I.cols_of(c)[0], c)
            return None
        m = self.rowmask(b, c, lazy=True)
        return None if m is None else [m]

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
        # AVG(x + c) of a block without GROUP BY over columns without NULLs: DuckDB computes it as SUM(x) / COUNT(*) + c
        # (its statistics show no NULL), never computing x + c, so an overflow there is no error
        shift = name == "avg" and not self.s.group and self.s.sets is None and not a.distinct and a.filter is None \
            and _shifted(a.args[0]) is not a.args[0] and all(self.valid(c) is None for c in I.cols_of(a.args[0]))
        if name != "count_star":
            if name == "count" and isinstance(a.args[0], I.Col): pass       # COUNT(column): its validity alone
            elif not ext:
                y = a.args[0]; x = a.args[1] if len(a.args) > 1 else None
                if self.poly(y, shift) is None or (x is not None and self.poly(x) is None): return None
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
                mkeys[v][k] = len(mlist[v]); mlist[v].append(self.poly_of_tuple(m, shift))
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

    def topk_spec(self):
        """The bounded rank and leading fact keys usable by either top-k path."""
        s, cx = self.s, self.cx
        if not TOPK or self.left or self.fact_sample is not None or not s.order or s.limit is None or s.percent: return 0, []
        if self.tab[self.fact].n < ALG_MIN_ROWS: return 0, []
        if s.distinct or s.distinct_on is not None or s.qualify is not None: return 0, []
        # windows over every row need them all: those in the SELECT list and those in the ORDER BY (ORDER BY x, LAG(id)
        # OVER (ORDER BY id) LIMIT k read LAG from the k rows); windows read from their series are computed on every row
        wl = collect([x for x, _ in s.items] + [o[0][1] for o in s.order if o[0][0] == "expr"], I.Win)
        if wl and not all(k in cx.series.get(id(s), {}) for k in wl): return 0, []
        lim = s.limit.v if isinstance(s.limit, I.Lit) else s.limit
        off = 0 if s.offset is None else (s.offset.v if isinstance(s.offset, I.Lit) else s.offset)
        if type(lim) is not int or type(off) is not int or lim < 0 or off < 0: return 0, []
        k = lim + off
        if not 0 < k <= TOPK_MAX: return 0, []
        fact = self.tab[self.fact]
        okeys = []
        for e, desc, nf in s.order:
            if isinstance(e, tuple): e = s.items[e[1]][0] if e[0] == "pos" else e[1]
            if not isinstance(e, I.Col) or T.coll_key(e.t) is not None: break
            b, name = self.canon(e)
            if b is not self.fact or name in getattr(fact, "wide", {}) or name not in fact.codecs: break
            vc = getattr(fact, "nullable", {}).get(name)
            if vc is not None and nf: break                     # NULLs first: they are among the first rows
            okeys.append((name, bool(desc), vc))
        return k, okeys

    def topk_ids(self, preds):
        """Exact GPU shortlist for a bounded fact sort, or None for rank predicates."""
        if self.cx.engine != "gpu" or self.joins or self.s.with_ties:
            return None
        k, keys = self.topk_spec()
        if not keys or len(keys) != len(self.s.order) or len(keys) > 2 or k > 1024:
            return None
        if any(validity is not None for _, _, validity in keys):
            return None
        q = algq.Query(self.tab[self.fact], join_mode="auto")
        for pred in preds: q.where(pred)
        ids = q._topk_rows([(name, desc) for name, desc, _ in keys], k)
        if ids is not None:
            self.cx.stats["engine_queries"] += 1
            self.cx.stats["gpu_mask_topk"] = self.cx.stats.get("gpu_mask_topk", 0) + 1
        return ids

    def topk_preds(self, preds):
        """ORDER BY ... LIMIT k (+ OFFSET) of a block without aggregates, ordered by fact columns: rank selection on the
        planes (algq.Query.rank_codes) gives the code of the k-th row on the first ORDER BY column; the block then
        reads only the rows before it or tied with it, and while the ties are many, the next ORDER BY column splits
        them the same way. The final ORDER BY / LIMIT sorts what is read as before (the same rows, in the same order).
        -> [one engine predicate], or [] when the block reads every passing row"""
        k, okeys = self.topk_spec()
        if not okeys: return []
        cx, fact = self.cx, self.tab[self.fact]
        q0 = algq.Query(fact); q0.join_mode = "auto"
        for b, fk in self.joins.items(): q0.join(self.tab[b], fk)
        for p in preds: q0.where(p)
        def code_cmp(name, op, code, vc):
            c = algq.Cmp(algq.Col(fact, name), op, int(code))
            return c if vc is None else algq.BoolOp("AND", algq.Cmp(algq.Col(fact, vc), "EQ", 1), c)
        before, tie, need = [], None, k
        for i, (name, desc, vc) in enumerate(okeys):
            q = algq.copy_query(q0)
            if tie is not None: q.pred = tie if q.pred is None else algq.BoolOp("AND", q.pred, tie)
            cx.stats["engine_queries"] += 1
            rank = -need if desc else need - 1
            got, total, spans = q.rank_codes(algq.Col(fact, name), [rank], engine=cx.engine, with_spans=True)
            if not got:                                         # fewer rows than needed: every row of this tie group
                if i == 0: return []
                break
            t = got[rank]
            below, ne = spans[rank]
            nb = total - below - ne if desc else below
            better = code_cmp(name, "GT" if desc else "LT", t, vc)
            eq = code_cmp(name, "EQ", t, vc)
            if tie is not None: better, eq = algq.BoolOp("AND", tie, better), algq.BoolOp("AND", tie, eq)
            before.append(better); tie = eq; need -= nb
            if ne <= max(4 * k, 1000) or need <= 0: break
        p = tie
        for b in reversed(before): p = algq.BoolOp("OR", b, p)
        return [p]

    def rank_col(self, a):
        """MEDIAN / PERCENTILE_CONT / _DISC of a fact column read from the planes by rank selection (rank_values): the
        column is then not a GROUP BY key of the block. -> (column, its validity column or None), or None"""
        if not RANK_AGGS or not COLUMNAR or self.left or self.fact_sample is not None or self.tab[self.fact].n < ALG_MIN_ROWS: return None
        if a.name not in _RANK_AGGS or a.distinct or a.filter is not None or a.order or len(a.args) != 1: return None
        e = a.args[0]
        if not isinstance(e, I.Col) or T.coll_key(e.t) is not None or e.b in self.nulled: return None
        try: qs = F._q_param(a.params) if a.name != "median" else Fraction(1, 2)
        except Exception: return None
        if not all(isinstance(q, Fraction) and 0 <= q <= 1 for q in (qs if isinstance(qs, list) else [qs])): return None
        b, name = self.canon(e)
        fact = self.tab[self.fact]
        if b is not self.fact or name in getattr(fact, "wide", {}) or name not in fact.codecs: return None
        if not 0 < int(fact.width(name)) <= 62: return None
        return name, getattr(fact, "nullable", {}).get(name)

    def rank_values(self, rc, a, rel, keys, preds, joins, align, members):
        """the aggregate of every output group by rank selection on the column's planes: per group, the code at each
        wanted rank, found 8 bits at a time. Each level is one engine query grouped by the block's keys and the code's
        leading bits, on the code ranges the prefixes found so far leave, so that no (group, value) pair is formed and
        only the codes at the wanted ranks are decoded. With many groups, one query on the whole code instead."""
        name, vc = rc
        cx = self.cx; fact = self.tab[self.fact]; eng = cx.engine
        G, to_groups, _ = _out_groups(rel, members)
        out = [None] * G
        if to_groups is None: return out
        w = int(fact.width(name)); nk = len(keys)
        keyed = any(isinstance(k, algq.Col) and k.table is fact and k.name == name for k in keys)
        nn = [algq.Cmp(algq.Col(fact, vc), "EQ", 1)] if vc is not None else []
        def level(extra, sh):
            """passing rows per (block group, code bits [sh, w)) -> (output group, those bits, rows) per entry"""
            if sh or keyed: key, rev = algq._bits_key(fact, name, sh, w - sh, "plain" if eng == "plain" else eng)
            else: key, rev = algq.Col(fact, name), False
            q = algq.Query(fact); q.join_mode = "auto"
            for b, fk in joins.items(): q.join(self.tab[b], fk)
            for p in preds + nn + extra: q.where(p)
            q.group_by(*keys, key); q.count()
            cx.stats["engine_queries"] += 1
            try: h = q.run_columns(eng)
            except (TypeError, OverflowError, ValueError) as ex: raise SqlError(f"the engine cannot run this block: {ex}")
            if not len(h): return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
            pos = np.asarray(align(algq.Groups(h.keys[:nk], h.vals)), np.int64) if nk else np.zeros(len(h), np.int64)
            c, _, vv = h.keys[nk]
            code = np.asarray(c if c is not None else vv).astype(np.int64)
            if rev: code = _bit_reverse(code, w - sh)
            return to_groups(pos, code, np.asarray(h.vals, np.int64))
        def select(eid, code, cnt, rid, rr):
            """per request (entry id rid, rank rr among that id's rows): the entry code holding the rank, the rank left"""
            o = np.lexsort((code, eid)); ge, ce, ne = eid[o], code[o], cnt[o]
            if len(ge) > 1:                                   # one entry per (id, code): several block groups add up
                new = np.ones(len(ge), bool); new[1:] = (ge[1:] != ge[:-1]) | (ce[1:] != ce[:-1])
                heads = np.flatnonzero(new); ge, ce, ne = ge[heads], ce[heads], np.add.reduceat(ne, heads)
            C = np.cumsum(ne); Cp = C - ne
            base = Cp[np.searchsorted(ge, rid, side="left")]
            i = np.searchsorted(C, base + rr, side="right")
            return ce[i], rr - (Cp[i] - base)
        B = w if rel.n * RANK_DIRECT > fact.n else 8
        nb = min(B, w)
        grp, code, cnt = level([], w - nb)
        N = np.zeros(G, np.int64); np.add.at(N, grp, cnt)
        # the ranks wanted: per fraction, PERCENTILE_CONT's two neighbours or PERCENTILE_DISC's one
        if a.name == "median": qs, single, cont = [Fraction(1, 2)], True, True
        else:
            q = F._q_param(a.params); single = not isinstance(q, list); qs = [q] if single else q
            cont = a.name in ("quantile_cont", "percentile_cont")
        desc = not cont and len(a.params) > 1 and a.params[1] == "desc"
        Nl = N.tolist(); req_g, req_r, slots = [], [], []
        for qi, q in enumerate(qs):
            for g, n in enumerate(Nl):
                if not n: continue
                if cont:
                    p = q * (n - 1); lo, hi = math.floor(p), math.ceil(p)
                    slots.append((qi, g, p, lo, hi, len(req_g))); req_g += [g, g]; req_r += [lo, hi]
                else:
                    k = max(0, math.ceil(q * n) - 1)
                    slots.append((qi, g, None, k, k, len(req_g))); req_g.append(g); req_r.append(n - 1 - k if desc else k)
        if not req_g: return out
        rg = np.array(req_g, np.int64); rr = np.array(req_r, np.int64)
        rp, rr = select(grp, code, cnt, rg, rr)
        done = nb
        while done < w:
            nb = min(B, w - done); span = w - done
            P = np.unique(rp)
            # the prefixes' code ranges (adjacent ones merged; past 16, the narrowest gaps closed too)
            brk = np.flatnonzero(P[1:] != P[:-1] + 1)
            if len(brk) > 15: brk = np.sort(brk[np.argsort(P[brk + 1] - P[brk], kind="stable")[-15:]])
            lo_p = np.concatenate([P[:1], P[brk + 1]]); hi_p = np.concatenate([P[brk], P[-1:]])
            pred = None
            for lo, hi in zip(lo_p.tolist(), hi_p.tolist()):
                x = algq.Between(algq.Col(fact, name), lo << span, ((hi + 1) << span) - 1)
                pred = x if pred is None else algq.BoolOp("OR", pred, x)
            grp, code, cnt = level([pred], span - nb)
            ids = np.unique(np.stack([np.concatenate([rg, grp]), np.concatenate([rp, code >> nb])], 1), axis=0,
                            return_inverse=True)[1].reshape(-1)
            rp, rr = select(ids[len(rg):], code, cnt, ids[:len(rg)], rr)
            done += nb
        vals = ColArr(rp, None, algq.Col(fact, name).codec()).decode()
        if a.name == "median" and vals and not isinstance(vals[0], (int, float, Decimal, bool, _dt.date, F.Interval)):
            cont = False                                           # MEDIAN of a type without a midpoint: the lower one
        res = [[None] * G for _ in qs]
        for qi, g, p, lo, hi, j in slots:
            if cont and p is not None: res[qi][g] = F.qc_mix(vals[j], vals[j + 1], p, lo, hi)
            else: res[qi][g] = vals[j]
        for g in range(G):
            if Nl[g]: out[g] = res[0][g] if single else [r[g] for r in res]
        return out

    def dist_col(self, a):
        """COUNT(DISTINCT column) of a column the engine reads as a key (dist_values): the column is then not a GROUP BY
        key of the block. Codes are one per value for these types (no collation, no doubles' -0 / NaN codes).
        -> (its engine key, its validity predicate or None), or None"""
        if not DIST_BITS or not COLUMNAR or self.left or self.fact_sample is not None or self.tab[self.fact].n < ALG_MIN_ROWS: return None
        if a.name != "count" or not a.distinct or a.filter is not None or a.order or len(a.args) != 1: return None
        e = a.args[0]
        if not isinstance(e, I.Col) or T.coll_key(e.t) is not None or e.b in self.nulled: return None
        if not (e.t.integer or e.t.id in ("DECIMAL", "DATE", "TIMESTAMP", "TIME", "VARCHAR", "BOOLEAN")): return None
        b, name = self.canon(e)
        if b is self.fact and name in getattr(self.fact.table, "wide", {}): return None
        return self.algcol(e), self.valid(e)

    def dist_values(self, dc, a, rel, keys, preds, joins, align, members):
        """COUNT(DISTINCT) of every output group. When each output group is one block group: one engine query grouped
        by the block's keys and the column, a bit per (group, value) cell set by the passing rows, each group's bits
        counted (algq A_DIST); else the cells themselves, each output group's distinct values counted"""
        key, vp = dc
        cx = self.cx; fact = self.tab[self.fact]
        G, to_groups, one = _out_groups(rel, members)
        if to_groups is None: return [0] * G
        nk = len(keys)
        def query():
            q = algq.Query(fact); q.join_mode = "auto"
            for b, fk in joins.items(): q.join(self.tab[b], fk)
            for p in preds + ([vp] if vp is not None else []): q.where(p)
            q.group_by(*keys, key); q.count()
            cx.stats["engine_queries"] += 1
            return q
        if one is not None and cx.engine in ("alg", "gpu") and (not nk or getattr(align, "coded", False)):
            q = query(); q._codes = True; q._raw = True; q._distinct = True
            try: km, _, cn = q._run_codes("alg")
            except ValueError: pass
            else:
                pos = np.asarray(align(_raw_groups(keys, km)), np.int64) if nk else np.zeros(len(cn), np.int64)
                per = np.zeros(rel.n, np.int64); per[pos] = cn
                return per[one].tolist()
        try: h = query().run_columns(cx.engine)          # the cells, as the block's own groups came back
        except (TypeError, OverflowError, ValueError) as ex: raise SqlError(f"the engine cannot run this block: {ex}")
        out = np.zeros(G, np.int64)
        if len(h):
            import sqlwin
            pos = np.asarray(align(algq.Groups(h.keys[:nk], h.vals)), np.int64) if nk else np.zeros(len(h), np.int64)
            if one is not None:                          # one block group per output group: each cell is one value
                per = np.bincount(pos, minlength=rel.n).astype(np.int64)
                return per[one].tolist()
            c, _, vv = h.keys[nk]
            code = np.asarray(c, np.int64) if c is not None else np.asarray(sqlwin.group_codes([nkey(v) for v in vv]), np.int64)
            grp, code, _ = to_groups(pos, code, np.ones(len(h), np.int64))
            if len(grp):
                u = np.unique(np.stack([grp, code], 1), axis=0)
                out = np.bincount(u[:, 0], minlength=G).astype(np.int64)
        return out.tolist()

    def raw_plane_group(self, preds, aggs):
        """Compute a bounded COUNT-ordered result directly from the fact's planes."""
        from torbles import sqlgroupplans

        s, cx, fact = self.s, self.cx, self.tab[self.fact]
        if (algq._plane_group_topk is None or not COLUMNAR or not VEC_GROUPS
                or cx.engine not in ("alg", "gpu") or not sqlgroupplans.single_fact(self)
                or not ALG_MIN_ROWS <= fact.n < 1 << 32 or s.sets is not None or not 1 <= len(s.group) <= 5
                or s.having is not None or s.qualify is not None or s.distinct or s.distinct_on is not None
                or s.percent or s.with_ties or len(s.order) != 1):
            return None
        lim = s.limit.v if isinstance(s.limit, I.Lit) else s.limit
        off = s.offset.v if isinstance(s.offset, I.Lit) else s.offset
        off = 0 if off is None else off
        if type(lim) is not int or type(off) is not int or lim <= 0 or off < 0 or lim + off > 1024:
            return None
        spec, desc, _ = s.order[0]
        order = s.items[spec[1]][0] if spec[0] == "pos" else spec[1]
        if not desc or not isinstance(order, I.Agg) or order.name not in ("count", "count_star"):
            return None
        group_keys = {g.key() for g in s.group}
        if any(x.key() not in group_keys and x.key() not in aggs for x, _ in s.items):
            return None

        layout = sqlgroupplans.layout(self, s.group)
        if layout is None:
            return None
        physical, deferred = layout
        keys = [key for _, key in physical]
        domain = 1
        for low, high, shift, transform, _, _ in keys:
            if transform == 1:                       # EXTRACT(MINUTE)
                domain *= 60
            else:
                width = fact.width(low) if high is None else shift + fact.width(high)
                domain *= 1 << int(width)
            if domain >= 1 << 14:
                break
        else:
            return None                              # small domains already group cheaply by popcount

        measures, plans = [], {}
        for key, agg in aggs.items():
            if agg.distinct or agg.filter is not None or agg.order:
                return None
            if agg.name == "count_star":
                plans[key] = ("count", 0, 0)
                continue
            if agg.name not in ("count", "sum", "avg") or len(agg.args) != 1 or not agg.args[0].t.integer:
                return None
            arg = _strip_cast(agg.args[0])
            if not isinstance(arg, I.Col) or arg.b is not self.fact or arg.nullable:
                return None
            binding, name = self.canon(arg)
            if binding is not self.fact or name not in fact.cols or name in getattr(fact, "nullable", {}):
                return None
            if agg.name == "count":
                plans[key] = ("count", 0, 0)
                continue
            codec = fact.codecs.get(name)
            if (codec is None or codec.kind != "int" or name in getattr(fact, "wide", {}) or name in fact.sidecars):
                return None
            offset = int(codec.offset)
            max_code = (1 << int(fact.width(name))) - 1
            if fact.n * max_code >= 1 << 64:
                return None
            if agg.name == "sum":
                low, high = T.INT_RANGE[agg.t.id]
                if min(0, offset) * fact.n < low or max(0, offset + max_code) * fact.n > high:
                    return None
            if name not in measures:
                if len(measures) == 2:
                    return None
                measures.append(name)
            plans[key] = (agg.name, measures.index(name), offset)

        if cx.engine == "gpu":
            from torbles import sqlgpugroups

            if lim + off > sqlgpugroups.LIMIT:
                return None
        cold_scope = sqlgroupplans.cold_wide_group_scope(self, preds) if cx.engine == "gpu" else None
        bitmap = cold_scope[0] if cold_scope is not None else sqlgroupplans.bitmap(self, preds)
        if bitmap is None:
            return None
        specs = [key[:-1] for key in keys]
        if cx.engine == "gpu":
            result = None
            if cold_scope is not None:
                _, selected, upload_bytes = cold_scope
                if selected is not None:
                    result = fact._plane_group(specs, measures, lim + off, bitmap)
                    if result is not None:
                        cx.stats["cold_wide_group_cpu"] = cx.stats.get("cold_wide_group_cpu", 0) + 1
                        cx.stats["cold_wide_group_rows"] = cx.stats.get("cold_wide_group_rows", 0) + selected
                        cx.stats["cold_wide_group_avoided_plane_bytes"] = (
                            cx.stats.get("cold_wide_group_avoided_plane_bytes", 0) + upload_bytes
                        )
            if result is None:
                result = sqlgpugroups.group(fact, specs, measures, lim + off, bitmap, cx.stats)
        else:
            result = None
            if not measures:
                from torbles import sqlcpuinterchanges
                if sqlcpuinterchanges.enabled():
                    result = sqlcpuinterchanges.count_topk(fact, specs, lim + off, bitmap, cx.stats)
            if result is None:
                result = fact._plane_group(specs, measures, lim + off, bitmap)
        if result is None:
            return None
        codes, counts, sums = result
        counts = counts[off:].tolist()
        rel = Rel(len(counts))
        for i, (expr, key) in enumerate(physical):
            rel.cols[expr.key()] = key[-1](codes[off:, i])
        for expr in deferred:
            rel.cols[expr.key()] = ev(expr, rel, cx)
        for key, (kind, index, offset) in plans.items():
            if kind == "count":
                rel.cols[key] = counts
            else:
                exact = [code + offset * n for code, n in zip(sums[off:, index].tolist(), counts)]
                rel.cols[key] = exact if kind == "sum" else [v / n for v, n in zip(exact, counts)]
        rel.plane_ordered = True
        cx.stats["engine_queries"] += 1
        cx.stats["raw_plane_group_rows"] = cx.stats.get("raw_plane_group_rows", 0) + fact.n
        cx.stats["raw_plane_group_groups"] = cx.stats.get("raw_plane_group_groups", 0) + rel.n
        return rel

    def raw_plane_rows(self, atoms, preds):
        """A full fact sort or one ROW_NUMBER can order IDs before reading payload values."""
        s, cx, fact = self.s, self.cx, self.tab[self.fact]
        if (not RAW_PLANE_ORDER or not COLUMNAR or cx.engine not in ("alg", "gpu") or self.joins
                or self.left or self.fact_sample is not None or fact.parts is not None
                or fact.n < 1 << 17 or s.distinct or s.distinct_on is not None
                or s.percent or s.with_ties or algq._plane_order is None):
            return None
        items = [x for x, _ in s.items]
        order = [(items[spec[1]] if spec[0] == "pos" else spec[1], desc) for spec, desc, _ in s.order]
        output_exprs = items + [s.qualify] + [x for x, _ in order]
        if subqueries_in(output_exprs + [s.where]):
            return None
        windows = list(collect(output_exprs, I.Win).values())
        w, npart = None, 0
        if windows:
            if not VEC_WINDOWS or len(windows) != 1:
                return None
            w = windows[0]
            if (w.name != "row_number" or w.args or w.rpr is not None or w.filter is not None
                    or w.agg_order or (w.frame is not None and any(b[1] is not None for b in w.frame[1:]))):
                return None
            order = [(x, False) for x in w.partition] + [(x, desc) for x, desc, _ in w.order]
            npart = len(w.partition)
        else:
            if s.qualify is not None:
                return None
            lim = s.limit.v if isinstance(s.limit, I.Lit) else s.limit
            off = s.offset.v if isinstance(s.offset, I.Lit) else s.offset
            off = 0 if off is None else off
            if type(off) is not int or off < 0 or (lim is not None and (type(lim) is not int or lim < 0)):
                return None
            if lim is not None and 0 < lim + off <= TOPK_MAX and self.topk_spec()[1]:
                return None                                 # preserve bounded rank selection
            output_exprs = items
        if not 0 < len(order) <= 12:
            return None

        keys, directions = [], []
        for col, desc in order:
            if not isinstance(col, I.Col) or col.b is not self.fact:
                return None
            binding, name = self.canon(col)
            codec = fact.codecs.get(name)
            if (binding is not self.fact or codec is None or codec.kind not in ("int", "bool")
                    or name in getattr(fact, "nullable", {}) or name in getattr(fact, "wide", {})):
                return None
            keys.append(name); directions.append(bool(desc))

        bitmap, count = None, fact.n
        if preds:
            q = algq.Query(fact, join_mode="auto")
            for pred in preds: q.where(pred)
            q._rows = q._rowbits = True
            bitmap = q._run_codes("alg")
            count = int(q.last_plan.total_cnt)
            cx.stats["engine_queries"] += 1
        result = fact._plane_order(keys, directions, npart, bitmap, count, w is not None)
        if result is None:
            return None
        ids, ranks = result
        cx.stats["engine_queries"] += 1
        stat = "raw_plane_row_number_rows" if w is not None else "raw_plane_order_rows"
        cx.stats[stat] = cx.stats.get(stat, 0) + (len(ids) if ids is not None else len(ranks))
        payload = {c.key() for e in output_exprs if e is not None for c in I.find(e, I.Col, stop=(I.Win,))}
        payload_atoms = {k: c for k, c in atoms.items() if k in payload}
        if w is None:
            ids = ids[off:] if lim is None else ids[off:off + lim]
            rel = self.rows_rel(payload_atoms, preds, False, ids=ids, fact_order=False)
            rel.plane_ordered = True
            return rel

        if ids is None:
            source = np.arange(fact.n, dtype=np.uint64)
            by_row = ranks
        else:
            by_row = np.empty(fact.n, np.int64)
            by_row[ids] = ranks
            source = np.flatnonzero(np.unpackbits(bitmap.view(np.uint8), bitorder="little")[:fact.n]).astype(np.uint64)
            by_row = by_row[source]
        rel = self.rows_rel(payload_atoms, preds, False, ids=source)
        rel.cols.put(w.key(), ColArr(by_row))
        return rel

    def rows_rel(self, atoms, preds, empty, ids=None, fact_order=True):
        """Read the selected fact rows' atoms from the planes."""
        if ids is None:
            q = algq.Query(self.tab[self.fact]); q.join_mode = "auto"
            for b, fk in self.joins.items(): q.join(self.tab[b], fk)
            for p in preds: q.where(p)
            self.cx.stats["engine_queries"] += 1
            ids = np.zeros(0, np.uint64) if empty else q.rows(self.cx.engine)
        if self.fact_sample is not None: ids = ids[self.fact_sample[np.asarray(ids, np.int64)]]
        rel = Rel(len(ids))
        fact = self.tab[self.fact]
        cache = {}
        # Dense gathers amortize word transposition and parallel decoding.
        # The temporary arrays belong to this block, not the table's cache.
        dense = fact.n >= 1 << 16 and len(ids) * 4 >= fact.n * 3
        def read_codes(name):
            if not dense: return fact.get_rows(name, ids)
            values = fact._decode_column(name)
            # The bitmap yields unique ascending row ids, so full coverage
            # already has exactly the output's row order.
            return values if fact_order and len(ids) == fact.n else values[ids]
        def fact_codes(name):
            if name not in cache:
                v = read_codes(name)
                w = getattr(fact, "wide", {}).get(name)
                if w is not None: v = v.astype(np.uint64) + (read_codes(w[1]).astype(np.uint64) << np.uint64(w[0]))
                cache[name] = v
            return cache[name]
        def dim_codes(table, name, rows):
            if len(rows) * 64 < table.n:
                return table.get_rows(name, rows)
            return table.decoded(name)[rows]

        for k, col in atoms.items():
            b, name = self.canon(col)
            if b is self.fact:
                codes = fact_codes(name); t = fact
            else:
                t = self.tab[b]; fk = self.joins[b]
                f = fact_codes(fk).astype(np.int64)
                sg = getattr(fact, "surr", {}).get(fk)
                rows = (f & sg.mask) if sg is not None else f
                codes = dim_codes(t, name, rows)
            cd = getattr(t, "codecs", {}).get(name)
            vc = getattr(t, "nullable", {}).get(name)
            ok = None
            if vc is not None: ok = np.asarray(fact_codes(vc) if b is self.fact else dim_codes(t, vc, rows)) != 0
            if COLUMNAR: rel.cols.put(k, ColArr(codes, ok, cd)); continue      # decoded when read as values
            codes = codes.tolist()                       # Python integers: 64-bit codes past 2^63 stay exact
            vals = cd.decode(algq._codes_array(codes)) if cd is not None else codes
            if ok is not None: vals = [x if o else None for x, o in zip(vals, ok.tolist())]
            rel.cols[k] = vals
        return rel

    def engine(self, keys, atom_list, preds, empty, variants, vpreds, mlist, direct_groups=False):
        """the engine queries: per variant, COUNT and the sum of each moment polynomial, grouped by the atoms"""
        if COLUMNAR and keys and not empty:
            return self.engine_columns(keys, atom_list, preds, variants, vpreds, mlist, direct_groups)
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
        first_ext = None
        if empty:
            base = {} if keys else 0
        else:
            m0 = first_poly(0)
            if m0 is not None:
                r = both([], m0)
                if keys: base = {g: v[1] for g, v in r.items()}; pre[0] = {g: v[0] for g, v in r.items()}
                else: base = r[1]; pre[0] = r[0]
            elif (not keys and mlist[0] and isinstance(mlist[0][0], tuple)
                  and mlist[0][0][0] == "ext" and mlist[0][0][1] not in getattr(self.tab[self.fact], "nullable", {})):
                # Scalar extrema already count the matching rows in the same scan.
                first_ext = mlist[0][0]
                ext_value = run(query([]).minmax(algq.Col(self.tab[self.fact], first_ext[1])))
                base = 0 if ext_value is None else ext_value.observations
                if base is None: base = run(query([]).count())
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
                elif ext and vid == 0 and m is first_ext: sums = ext_value
                elif ext: sums = run(query(extra).minmax(algq.Col(self.tab[self.fact], m[1])))
                elif vid in pre and m is first_poly(vid): sums = pre[vid]
                else: sums = run(query(extra).sum(m))
                mk = (vid, m) if ext else (vid, mi)
                if keys: rel.mom[mk] = [sums.get(g) for g in groups]
                else: rel.mom[mk] = [sums] if groups else []
        return rel


    def engine_columns(self, keys, atom_list, preds, variants, vpreds, mlist, direct_groups=False):
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
            q._separate_group_counts = direct_groups
            try: return q.run_columns(cx.engine)
            except (TypeError, OverflowError, ValueError) as ex:
                m = str(ex)
                if "group domain too large" in m or "hidden GROUP BY columns" in m or "exceed 128 bits" in m:
                    raise _TooManyGroups()
                if "dimension columns through lookups" in m: raise _NotStarNow()     # the general path joins them
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
            g = run(q)
            if g.counts is not None: base_cnt, pre[0] = g.counts, g.vals
            else: base_cnt, pre[0] = [v[1] for v in g.vals], [v[0] for v in g.vals]
        else:
            g = run(query([]).count()); base_cnt = g.vals
        G = len(base_cnt)
        rel = Rel(G); rel.cnt = base_cnt
        coded = all(c is not None for c, _, _ in g.keys)
        base_keys = None
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
            nonlocal base_keys
            if coded and all(c is not None for c, _, _ in h.keys):
                if base_keys is None: base_keys = rowkey(g)
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
            values = vals.tolist() if isinstance(vals, np.ndarray) else vals
            for p, v in zip(align(h).tolist(), values): out[p] = v
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
        align.coded = coded                              # (raw key matrices align only with a coded base run)
        self._align = align                              # later runs of the same groups (rank_values, dist_values)
        return rel


_NO = object()


class _TooManyGroups(Exception):
    pass


def _out_groups(rel, members):
    """the output groups of the block's rows (members: each group's rows; a row is in one group, or several under
    GROUPING SETS) -> (number of groups, to_groups(pos, code, cnt) carrying per-row entries to their groups (None when
    no group has a row), each group's only row when every group has exactly one (else None))"""
    flat = getattr(members, "flat", None)
    if flat is None:
        sizes = np.array([len(m) for m in members], np.int64)
        rows = np.concatenate([np.asarray(m, np.int64) for m in members]) if len(members) else np.zeros(0, np.int64)
        ends = np.cumsum(sizes); starts = ends - sizes
    else: rows, starts, ends = flat
    rows = np.asarray(rows, np.int64)
    G = len(starts)
    if not G or not rel.n or not len(rows): return G, None, None
    sizes = np.asarray(ends) - np.asarray(starts)
    one = rows[np.asarray(starts)] if bool((sizes == 1).all()) and len(np.unique(rows)) == len(rows) else None
    pgrp = np.repeat(np.arange(G, dtype=np.int64), sizes)[np.argsort(rows, kind="stable")]
    per = np.bincount(rows, minlength=rel.n).astype(np.int64); first = np.cumsum(per) - per
    def to_groups(pos, code, cnt):
        reps = per[pos]
        idx = np.repeat(np.arange(len(pos)), reps)
        off = np.arange(len(idx)) - np.repeat(np.cumsum(reps) - reps, reps)
        return pgrp[np.repeat(first[pos], reps) + off], code[idx], cnt[idx]
    return G, to_groups, one


def _raw_groups(keys, km):
    """an engine run's raw key matrix as Groups for engine_columns' align (a NULL-able key's validity from its
    validity key among the keys, as the base run has it)"""
    at = {(id(k.table), k.name): j for j, k in enumerate(keys) if isinstance(k, algq.Col)}
    cols = []
    for i, k in enumerate(keys):
        valid = None
        if isinstance(k, algq.Col):
            vc = getattr(k.table, "nullable", {}).get(k.name)
            j = at.get((id(k.table), vc)) if vc is not None else None
            if j is not None: valid = np.asarray(km[:, j]) != 0
        cols.append((np.asarray(km[:, i], np.int64), valid, None))
    return algq.Groups(cols, [0] * len(km))


def _bit_reverse(k, n):
    """the n-bit codes k with their bits in reverse order (a PlaneKey's lowest plane is its key's top bit)"""
    out = np.zeros(len(k), np.int64)
    for b in range(n): out |= ((k >> b) & 1) << (n - 1 - b)
    return out


def _states(rel, aggs, aggplan):
    """each engine-computed aggregate's state per row: ("n", k) count, ("m", k, j) moment sums, ("x", k) extremes"""
    for k, plan in aggplan.items():
        kind = plan.get("kind")
        filt = plan.get("rowfilter")
        if kind == "rank": continue                  # computed per output group (_Planner.rank_values)
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
import decimal as _decimal
_XCTX = _decimal.Context(prec=_decimal.MAX_PREC)


def _xmul(a, b):
    """the exact product of two partial sums or counts (int or Decimal; None counts as 0)"""
    if a is None or b is None: return 0
    if type(a) is int and type(b) is int: return a * b
    return _XCTX.multiply(Decimal(a), Decimal(b))


def _exact_num(t): return t.integer or t.id == "DECIMAL"


def join_preagg(s, aggs, cx):
    """an aggregate block over an inner equi-join of two FROM items (stored tables, derived tables, CTEs), without
    forming the pairs of rows: each side is grouped first, by the columns the block reads outside its aggregates (the
    join keys, GROUP BY, ...), with its partial sums; the groups are joined, and each aggregate is combined from the
    two sides' partial sums: a sum over one side counts once per row of the other side (its sum times the other
    group's count), a sum of a product of the two sides' factors is the product of the sides' sums, MIN / MAX are the
    sides' own. Any other aggregate keeps its arguments' columns among the grouped ones and counts each pair of groups
    as the product of their counts. -> (rel of the joined groups, their aggregate states), or None"""
    import sqlfront
    if not s.aggregate: return None
    f = s.from_
    if not isinstance(f, I.Join) or f.kind not in ("inner", "cross"): return None
    L, R = f.left, f.right
    if not isinstance(L, I.Binding) or not isinstance(R, I.Binding) or L is R: return None
    if s.sample is not None or L.sample is not None or R.sample is not None: return None
    order_x = [o[0][1] for o in s.order if o[0][0] == "expr"]
    don = [d[1] for d in (s.distinct_on or []) if d[0] == "expr"]
    top = [x for x in [x for x, _ in s.items] + [s.having, s.qualify] + order_x + don if x is not None]
    allx = top + list(s.group) + [x for x in (s.where, f.cond) if x is not None]
    for x in allx:
        if I.has(x, I.Subq) or I.has(x, I.Outer) or not _deterministic(x): return None
    for b in (L, R):
        if b.q is not None and not hasattr(b, "shared") and outer_refs_into(b.q, {L, R}): return None
    def sides(e): return {c.b for c in I.cols_of(e)}
    pl, pr, eqs, cross = [], [], [], []
    for c in _conjuncts(s.where) + _conjuncts(f.cond):
        bs = sides(c)
        if bs <= {L}: pl.append(c)
        elif bs <= {R}: pr.append(c)
        elif isinstance(c, I.Func) and c.fn == "=" and len(c.args) == 2 and sides(c.args[0]) | sides(c.args[1]) == {L, R} \
                and len(sides(c.args[0])) == 1 and len(sides(c.args[1])) == 1:
            a, b = c.args if sides(c.args[0]) == {L} else c.args[::-1]
            if a.t.id != b.t.id or T.coll_key(a.t) is not None or T.coll_key(b.t) is not None: cross.append(c)
            else: eqs.append((a, b))
        else: cross.append(c)
    if not eqs: return None
    for X, i in ((L, 0), (R, 1)):                      # a side joined on its primary key: at most one pair per row of
        pk = getattr(X.table, "pk", None) if X.table is not None else None      # the other side, as cheap as it is
        if pk is not None and any(isinstance(ab[i], I.Col) and ab[i].name == pk for ab in eqs): return None
    keep = {L: {}, R: {}}
    def keep_cols(cols):
        for c in cols:
            if c.b not in keep or T.coll_key(c.t) is not None: return False
            keep[c.b][c.key()] = c
        return True
    for x in top:
        if not keep_cols(I.find(x, I.Col, stop=(I.Agg,))): return None
    for x in list(s.group) + [e for ab in eqs for e in ab] + cross:
        if not keep_cols(I.cols_of(x)): return None
    # each aggregate: from the sides' partial sums (moments), the sides' extremes, or the values of kept columns
    def factors(e):
        """e as a product of factors that each read one side (constants: either), or None"""
        if isinstance(e, I.Func) and e.fn == "*" and len(e.args) == 2:
            a, b = factors(e.args[0]), factors(e.args[1])
            return None if a is None or b is None else a + b
        return [e] if len(sides(e)) <= 1 else None
    def mfactors(m):
        if isinstance(m, tuple): return mfactors(m[1]) + mfactors(m[2])
        return factors(m)
    def is_prod(e): return isinstance(e, I.Func) and e.fn == "*" and len(e.args) == 2 and factors(e) is not None
    def bounded_prods(e):
        """does e's product tree hold a product of an integer / DECIMAL type? (factors() splits them: the pairs' products
        are never computed, so each must provably keep its type)"""
        return is_prod(e) and (type_range(e.t) is not None or bounded_prods(e.args[0]) or bounded_prods(e.args[1]))
    def tree_range(e, leaf):
        """the interval of e's product tree from its factors' (leaf(factor) -> [lo, hi], () for a factor NULL on every
        row, or None): True above an approximate product, None when a product of a bounded type is not proved"""
        if is_prod(e):
            a, b = tree_range(e.args[0], leaf), tree_range(e.args[1], leaf)
            if a is None or b is None: return None
            if type_range(e.t) is None: return True
            if a == () or b == (): return ()
            if a is True or b is True: return None
            c = [x * y for x in a for y in b]; r = (min(c), max(c))
            return r if _within(r, e.t) else None
        if type_range(e.t) is None: return True
        return expr_range(e, lambda c: None) if isinstance(e, I.Lit) else leaf(e)
    def side_src(X):
        """the stored binding a side reads and the derived table's columns as its expressions (None: a stored table),
        or (None, None) for any other derived table"""
        q0 = X.q
        if X.table is not None: return X, None
        if isinstance(q0, I.Select) and isinstance(q0.from_, I.Binding) and q0.from_.table is not None and not q0.aggregate \
                and not q0.group and q0.having is None and q0.qualify is None and not q0.distinct and q0.distinct_on is None \
                and not q0.order and q0.limit is None and q0.offset is None and q0.sample is None and q0.from_.sample is None:
            inner = [x for x, _ in q0.items] + ([q0.where] if q0.where is not None else [])
            names = [n for _, n in q0.items]
            if len(set(names)) == len(names) and not any(I.has(x, I.Subq) or I.has(x, I.Outer) or I.has(x, I.Win) or I.has(x, I.Agg)
                                                         or not _deterministic(x) for x in inner):
                return q0.from_, {n: x for x, n in q0.items}
        return None, None
    def static_leaf(y):
        """a factor's range from the codes its side's stored columns hold, or None"""
        X = side_of(y); src, m = side_src(X)
        if src is None: return None
        z = y if m is None else I.subst(y, lambda n: m[n.name] if isinstance(n, I.Col) and n.b is X else None)
        return expr_range(z, lambda c: stored_range(src.table, c.name) if c.b is src else None)
    parts = {L: {}, R: {}}
    def part(X, agg): parts[X].setdefault(agg.key(), agg); return agg.key()
    nkeys = {X: part(X, I.Agg("count_star", [], T.BIGINT)) for X in (L, R)}
    def side_of(e): bs = sides(e); return R if bs == {R} else L
    plans = {}
    checks = []                                     # (argument, {factor: (side, its MIN part, its MAX part)}) to prove
    for k, a in aggs.items():
        plan = None
        if a.filter is None and not a.order and not a.name.startswith("__iso_h"):
            if a.name == "count_star" and not a.distinct: plan = ("count",)
            elif a.name in ("min", "max") and len(a.args) == 1 and len(sides(a.args[0])) == 1:
                X = side_of(a.args[0])
                plan = ("ext", X, part(X, I.Agg("min", [a.args[0]], a.args[0].t)), part(X, I.Agg("max", [a.args[0]], a.args[0].t)))
            elif a.name in MOMENTS and not a.distinct and a.args:
                fs = [factors(x) for x in a.args]
                ok = all(x is not None and not any(isinstance(y, I.Lit) and y.v is None for y in x) for x in fs)
                if ok and a.name != "count":
                    ok = all(_exact_num(x.t) for x in a.args) and all(_exact_num(y.t) for x in fs for y in x)
                if ok:
                    for x in a.args:                 # the split products: proved from the stored columns, or from the
                        if not bounded_prods(x) or tree_range(x, static_leaf) is not None: continue     # sides' extremes
                        lv = {}
                        for y in factors(x):
                            if isinstance(y, I.Lit) or type_range(y.t) is None: continue
                            X = side_of(y)
                            lv[y.key()] = (X, part(X, I.Agg("min", [y], y.t)), part(X, I.Agg("max", [y], y.t)))
                        checks.append((x, lv))
                    val = {L: {}, R: {}}
                    for y in (y for x in fs for y in x if not isinstance(y, I.Lit)): val[side_of(y)][y.key()] = y
                    # each side's rows with every factor of the arguments valid: its count, and the sums of the
                    # moments' factors over them (no FILTER when the sum's own factors are those: SUM skips NULLs)
                    cond = {X: (_and([sqlfront.make_func("is_not_null", [y]) for y in val[X].values()]) if len(val[X]) > 1 else None) for X in (L, R)}
                    cnt = {}
                    for X in (L, R):
                        if not val[X]: cnt[X] = nkeys[X]
                        elif len(val[X]) == 1: cnt[X] = part(X, I.Agg("count", list(val[X].values()), T.BIGINT))
                        else: cnt[X] = part(X, I.Agg("count_star", [], T.BIGINT, filter=cond[X]))
                    ms = []
                    y = a.args[0]; x = a.args[1] if len(a.args) > 1 else None
                    for m in (MOMENTS[a.name](y, x) if a.name != "count" else []):
                        fl = {L: [], R: []}
                        for z in mfactors(m): fl[side_of(z) if not isinstance(z, I.Lit) else L].append(z)
                        mk = []
                        for X in (L, R):
                            if not fl[X]: mk.append(cnt[X]); continue
                            prod = None
                            own = {z.key() for z in fl[X] if not isinstance(z, I.Lit)}
                            for z in fl[X]:
                                if z.t.integer: z = sqlfront.cast_to(z, T.HUGEINT)
                                prod = z if prod is None else sqlfront.make_func("__xmul", [prod, z])
                            mk.append(part(X, I.Agg("sum", [prod], F.AGGS["sum"].typer([prod.t]),
                                                    filter=None if own == set(val[X]) else cond[X])))
                        ms.append(tuple(mk))
                    plan = ("moment", cnt[L], cnt[R], ms)
        if plan is None:
            for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                if not keep_cols(I.cols_of(x)): return None
        plans[k] = plan
    cx.stats["join_preagg"] = cx.stats.get("join_preagg", 0) + 1
    def side_rel(X, preds):
        """the side's groups: the kept columns, the partial aggregates. A derived table or CTE that only filters and
        computes columns of one stored table is read from that table (the engine groups its rows)"""
        src, m = side_src(X); q0 = X.q
        if src is None: src = X
        sub = (lambda e: I.subst(e, lambda n: m[n.name] if isinstance(n, I.Col) and n.b is X else None)) if m is not None else (lambda e: e)
        kc, pa = list(keep[X].values()), list(parts[X].values())
        g = [sub(c) for c in kc]; ps = [sub(a) for a in pa]
        w = [sub(p) for p in preds] + ([q0.where] if m is not None and q0.where is not None else [])
        q = I.Select(); q.from_ = src; q.where = _and(w) if w else None
        q.group = list({e.key(): e for e in g}.values()); q.aggregate = True
        q.items = [(e, f"__key{i}") for i, e in enumerate(q.group)] + [(a, f"__part{i}") for i, a in enumerate(ps)]
        out = run_select(q, cx, as_rel=True)
        if m is not None:                                    # under the derived table's own columns
            for old, new in list(zip(kc, g)) + list(zip(pa, ps)):
                if old.key() == new.key(): continue
                arr = out.cols.array(new.key())
                if arr is not None: out.cols.put(old.key(), arr)
                else: out.cols[old.key()] = out.cols[new.key()]
        return out
    A = side_rel(L, pl); B = side_rel(R, pr)
    # an error among the sides' partial values (a factor's overflow, a failed cast in it): its pairs are formed, row
    # pair by row pair, so that it fails the query only where it is used; so too when a product the moments split may
    # leave its type on the factors' extremes the sides found
    def partial_errs():
        for X, rel_ in ((L, A), (R, B)):
            for key in parts[X]:
                if rel_.cols.array(key) is None and any(isinstance(v, Err) for v in rel_.cols[key]): return True
        for x, lv in checks:
            rng = {}
            for yk, (X, kmin, kmax) in lv.items():
                rel_ = A if X is L else B
                lo = [v for v in rel_.cols[kmin] if v is not None]; hi = [v for v in rel_.cols[kmax] if v is not None]
                rng[yk] = () if not lo else (Fraction(min(lo)), Fraction(max(hi)))
            if tree_range(x, lambda y: rng.get(y.key())) is None: return True
        return False
    if partial_errs():
        cx.stats["join_preagg"] -= 1
        return None
    # the pairs of groups with equal join keys (NULL keys pair with nothing)
    import sqlwin
    ka, kb = [], []
    for a, b in eqs:
        for e, rel_, out in ((a, A, ka), (b, B, kb)):
            vals = ev(e, rel_, cx)
            for v in vals:
                if isinstance(v, Err): raise SqlError(v.msg)
            out.append(vals)
    kc, ok = [], np.ones(A.n + B.n, bool)
    for va, vb in (zip(ka, kb) if A.n and B.n else ()):
        allv = list(va) + list(vb)
        ok &= np.fromiter((v is not None for v in allv), bool, len(allv))
        kc.append(np.asarray(sqlwin.group_codes(allv), np.int64))
    cc = sqlwin._combine(kc) if len(kc) > 1 else kc[0] if kc else np.zeros(A.n + B.n, np.int64)
    if not (A.n and B.n): ok[:] = False
    ca, cb, oka = cc[:A.n], np.where(ok[A.n:], cc[A.n:], -1), ok[:A.n]
    ob = np.argsort(cb, kind="stable"); sb = cb[ob]
    lo = np.searchsorted(sb, ca, "left"); hi = np.searchsorted(sb, ca, "right")
    m = np.where(oka, hi - lo, 0)
    ia = np.repeat(np.arange(A.n), m)
    ib = ob[np.repeat(lo, m) + (np.arange(len(ia)) - np.repeat(np.cumsum(m) - m, m))] if len(ia) else np.zeros(0, np.int64)
    P = Rel(len(ia))
    for X, rel_, idx in ((L, A, ia), (R, B, ib)):
        for key in keep[X]:
            arr = rel_.cols.array(key)
            if arr is not None: P.cols.put(key, arr.take(idx))
            else: col = rel_.cols[key]; P.cols[key] = [col[i] for i in idx.tolist()]
    def pcol(rel_, key, idx):
        col = rel_.cols[key]; return [col[i] for i in idx.tolist()]
    nA, nB = pcol(A, nkeys[L], ia), pcol(B, nkeys[R], ib)
    P.cnt = [x * y for x, y in zip(nA, nB)]
    aggplan = {}
    for k, plan in plans.items():
        if plan is None: continue
        if plan[0] == "count": aggplan[k] = dict(kind="moment", variant=0, moments=[]); continue
        vid = len(aggplan) + 1
        if plan[0] == "ext":
            mn, mx = pcol(A if plan[1] is L else B, plan[2], ia if plan[1] is L else ib), pcol(A if plan[1] is L else B, plan[3], ia if plan[1] is L else ib)
            P.mom[(vid, "x")] = [None if x is None else (x, y) for x, y in zip(mn, mx)]
            aggplan[k] = dict(kind="ext", variant=vid, key="x"); continue
        _, cl, cr, ms = plan
        P.vcnt[vid] = [x * y for x, y in zip(pcol(A, cl, ia), pcol(B, cr, ib))]
        for j, (ml, mr) in enumerate(ms):
            P.mom[(vid, j)] = [_xmul(x, y) for x, y in zip(pcol(A, ml, ia), pcol(B, mr, ib))]
        aggplan[k] = dict(kind="moment", variant=vid, moments=list(range(len(ms))))
    if cross:
        P = P.take(rows_where(_and(cross), P, cx))
    _states(P, aggs, aggplan)
    return P, {k: dict(kind="state") for k in aggplan}


def run_general(s, bindings, need, aggs, cx, prejoin=None):
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
    from_ = s.from_
    if s.sample is None: from_, rest = _equijoins(from_, rest)
    deferred = []                                      # terms that failed on a row of their item: checked after the joins
    rel = _materialize(from_, per, pushed, cx, deferred, prejoin)
    if s.sample is not None: rel = sample_rel(rel, s.sample, cx)
    rest = deferred + rest
    if rest:
        rel = rel.take(rows_where(_and(rest), rel, cx))
    return rel, {}


def _join_key_term(x):
    """a column, or a column widened to an integer type holding all of its values: a join key that cannot fail"""
    if isinstance(x, I.Col): return True
    if isinstance(x, I.Cast) and isinstance(x.arg, I.Col) and not x.try_:
        a, b = T.INT_RANGE.get(x.arg.t.id), T.INT_RANGE.get(x.t.id)
        return a is not None and b is not None and b[0] <= a[0] and a[1] <= b[1]
    return False


def _equijoins(f, rest):
    """WHERE equalities between columns of the two sides of a cross or inner join (one no outer join holds on its NULL
    side) become that join's keys, so that it matches rows instead of forming every pair (FROM a, b WHERE a.k = b.k,
    left to the general path): -> (the FROM tree, the WHERE terms left). Columns only, so a term computed earlier
    fails nothing the WHERE would not."""
    taken = set()
    def go(x):
        if not isinstance(x, I.Join) or x.kind not in ("inner", "cross"): return x
        left, right = go(x.left), go(x.right)
        lb, rb = set(block_bindings(left)), set(block_bindings(right))
        add = []
        for i, c in enumerate(rest):
            if i in taken or not (isinstance(c, I.Func) and c.fn == "=" and len(c.args) == 2): continue
            a, b = c.args
            if not (_join_key_term(a) and _join_key_term(b)): continue
            ba, bb = {y.b for y in I.cols_of(a)}, {y.b for y in I.cols_of(b)}
            if ba <= lb and bb <= rb or ba <= rb and bb <= lb: add.append(c); taken.add(i)
        if not add and left is x.left and right is x.right: return x
        j = copy.copy(x); j.left, j.right = left, right
        if add: j.kind, j.cond = "inner", _and(([x.cond] if x.cond is not None else []) + add)
        return j
    out = go(f)
    return out, [c for i, c in enumerate(rest) if i not in taken]


def _materialize(f, per, pushed, cx, deferred=None, prejoin=None):
    if isinstance(f, I.Binding):
        cols, conds = per.get(f, {}), pushed.get(f, [])
        if prejoin is not None and f is prejoin[0].fact:
            planner, edges = prejoin
            try:
                rel, _ = planner.run([], cols, {})
            except (ValueError, SqlError, _NotStarNow):
                pass
            else:
                cx.stats["snowflake_blocks"] += 1
                cx.stats["snowflake_edges"] += edges
                return _pushed_rows(rel, conds, cx, deferred)
        return _mat_binding(f, cols, conds, cx, deferred)
    L = _materialize(f.left, per, pushed, cx, deferred, prejoin)
    lb = set(block_bindings(f.left))
    refs = {}
    for rb in block_bindings(f.right):
        if rb.q is not None and not hasattr(rb, "shared"):
            for c in outer_refs_into(rb.q, lb): refs[c.key()] = c
    if refs: return _lateral(f, L, list(refs.values()), per, pushed, cx)
    R = _materialize(f.right, per, pushed, cx, deferred, prejoin)
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
        keep = rows_where(_and(conds_on), C, cx); C = C.take(keep); li = [li[k] for k in keep]
    if j.kind == "left":
        hit = set(li)
        extra = [i for i in range(L.n) if i not in hit]
        if extra:
            P = L.take(extra)
            for k in cols:
                if k[0] not in _CACHED: P.cols[k] = [None] * P.n
            C = _concat([C, P])
    return C


def _pushed_rows(rel, conds, cx, deferred):
    """rel's rows that pass the WHERE terms pushed down to its FROM item. A row a term fails on (an overflow, a failed
    cast) is kept, and the terms join the ones checked after the joins (deferred): there an error counts only for a
    row the block's other terms leave undecided, as everywhere (a term on another item may drop the row)"""
    if not conds: return rel
    if deferred is None: return rel.take(rows_where(_and(conds), rel, cx))
    rows, errs = filter_rows(_and(conds), rel, cx)
    if errs: deferred.extend(conds); rows = sorted(set(rows) | set(errs))
    return rel.take(rows)


def _mat_binding(b, cols, conds, cx, deferred=None):
    cols = list(cols.values())
    if b.sample is not None and b.table is None:                # a sampled derived table: a sample of its rows
        sp, b.sample = b.sample, None
        try: rel = _mat_binding(b, {c.key(): c for c in cols}, [], cx)
        finally: b.sample = sp
        return _pushed_rows(sample_rel(rel, sp, cx), conds, cx, deferred)
    source = getattr(b, "_dml_rel", None)
    if source is not None:
        rel = Rel(source.n)
        rel.cnt = source.cnt
        for c in cols:
            key = c.key()
            array = source.cols.array(key)
            if array is None:
                rel.cols[key] = source.cols[key]
            else:
                rel.cols.put(key, array)
        return _pushed_rows(rel, conds, cx, deferred)
    writable = getattr(b, "writable", None)
    if writable is not None:
        names = list(dict.fromkeys(c.name for c in cols))
        data = writable.scan(names) if names else {}
        n = len(data[names[0]]) if names else writable.store.count()
        rel = Rel(n)
        for c in cols:
            rel.cols.put(c.key(), ColArr(data[c.name]))
        cx.stats["writable_scan_rows"] = cx.stats.get("writable_scan_rows", 0) + n
        return _pushed_rows(rel, conds, cx, deferred)
    if b.table is not None:
        s = I.Select(); s.from_ = b
        s.items = []; s.where = _and(conds) if conds else None
        need = {c.key(): c for c in cols}
        try:
            rel, _ = plan_star(s, [b], need, {}, cx)
            return rel
        except SqlError:
            if not conds or deferred is None: raise
        s.where = None                                  # a term failed on a row: the rows, then the terms (_pushed_rows)
        rel, _ = plan_star(s, [b], need, {}, cx)
        return _pushed_rows(rel, conds, cx, deferred)
    # a derived table, CTE or VALUES: its rows (a CTE once per query)
    shared = getattr(b, "shared", None)
    if shared is not None and id(shared) in cx.ctes: res = cx.ctes[id(shared)]
    elif getattr(b, "rec", None) is not None and id(b.rec.binding) in cx.ctes: res = cx.ctes[id(b.rec.binding)]
    else:
        query = b.q
        if cx.engine == "gpu" and conds and shared is None and not getattr(b, "rec", None):
            from torbles.sqlwindowplanes import push_derived_qualify

            query = push_derived_qualify(b, conds)
            if query is not b.q:
                cx.stats["window_derived_qualify_pushdowns"] = cx.stats.get("window_derived_qualify_pushdowns", 0) + 1
        res = run_query(query, cx)
        if shared is not None and not getattr(b, "rec", None): cx.ctes[id(shared)] = res
    names = [c for c, _, _ in b.cols]
    rel = Rel(len(res))
    for c in cols:
        j = names.index(c.name)
        if getattr(res, "_primitive_window_output", False):
            rel.cols.put(c.key(), ColArr(res._cols[j]))
        else:
            rel.cols[c.key()] = [r[j] for r in res.rows]
    return _pushed_rows(rel, conds, cx, deferred)


def _join(j, L, R, cx):
    part = getattr(j, "partition", None)
    if part is not None:                               # a partitioned outer join: each partition joined on its own
        side, exprs = part
        P = L if side == "left" else R
        keys = [ev(x, P, cx) for x in exprs]
        for col in keys:
            for v in col:
                if isinstance(v, Err): raise SqlError(v.msg)
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
        for col in lv + rv:                          # a key that is an error (an overflow, a failed cast) matches as
            if not isinstance(col, ColArr):          # nothing can: every row's key decides its pairs, so it fails the query
                for v in col:
                    if isinstance(v, Err): raise SqlError(v.msg)
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
        keep = np.array(rows_where(_and(other), C, cx), np.int64)
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


# keys of values cached for an expression (codes_of, a join's ON terms on its pairs): they hold for the rows they were
# computed on only, and are computed again from the expression's columns where they are missing
_CACHED = frozenset(("lit", "fn", "case", "cast", "outer", "subq", "grouping"))


def _pad(A, other):
    """A's rows with the other side's columns NULL (an expression of NULLs need not be NULL: those are left out)"""
    for k in other.cols:
        if k not in A.cols and k[0] not in _CACHED: A.cols[k] = [None] * A.n
    return A


def _concat(parts):
    n = sum(p.n for p in parts)
    C = Rel(n)
    keys = []
    for p in parts:
        for k in p.cols:
            if k not in keys: keys.append(k)
    keys = [k for k in keys if k[0] not in _CACHED or all(k in p.cols for p in parts)]   # a cache some parts lack
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
