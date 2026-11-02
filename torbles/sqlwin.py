# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Window functions as array algebra.

A window's rows are ordered once: every PARTITION BY and ORDER BY column becomes dense integer codes (equal codes for
values SQL treats as peers, codes ascending along the sort order with its direction and NULL placement), and one stable
lexsort of the codes orders every partition at once. A partition and a peer group are then runs of equal codes, and
each window function is arithmetic on positions in that order:

  * ROW_NUMBER, RANK, DENSE_RANK, PERCENT_RANK, CUME_DIST, NTILE: the positions of the row, of its peer group and of
    its partition;
  * LAG, LEAD: a shifted position; with IGNORE NULLS the k-th entry of the non-NULL positions, found by a prefix count;
  * frame edges: a position (ROWS), the start or end of a peer group (GROUPS, RANGE), and for RANGE with an offset the
    rank of the target value among the partition's keys (one joint ranking of keys and targets, one search);
    EXCLUDE CURRENT ROW / GROUP / TIES leaves at most three runs of positions;
  * COUNT, SUM, AVG, VAR_*, STDDEV_*, COVAR_*, CORR, REGR_*, EVERY, BOOL_OR, COUNT_IF: differences of exact prefix sums
    of the moments (floats and decimals scaled to integers exactly), rounded once;
  * MIN, MAX, ARG_MIN, ARG_MAX: range minima of (value code, position) over a sparse table built one level at a time;
  * FIRST_VALUE, LAST_VALUE, NTH_VALUE [FROM LAST] [IGNORE NULLS], ANY_VALUE: positions, through the eligible positions.

The results are the ones the row-by-row definitions give (exact sums, correctly rounded doubles, the leftmost of tied
minima). A window function outside these (DISTINCT, an ordered aggregate, nested window functions, row pattern
recognition, other aggregates) raises NoVec and runs on each partition's rows, in the same order."""
import datetime as _dt
import functools
from decimal import Decimal
from fractions import Fraction
import numpy as np
import sqlfuncs as F
import sqltypes as T
from sqltypes import SqlError


class NoVec(Exception):
    """this window function runs on the rows of each partition instead"""


_NONE = type(None)
_NUM = {int, bool, float, Decimal}


def obj(col):
    """a list as a 1-D object array (tuples and lists stay single elements)"""
    return np.fromiter(col, object, len(col))


def notnull(a):
    return np.fromiter((v is not None for v in a), bool, len(a))


def is_codes(c):
    """a column given as engine codes (sqlexec.ColArr: codes, valid, codec) rather than values"""
    return hasattr(c, "codec") and hasattr(c, "codes")


def code_groups(a):
    """group codes of a code column: equal codes are equal values; NULL is a group of its own"""
    c = a.codes
    g = np.unique(c, return_inverse=True)[1].reshape(-1).astype(np.int64) if c.dtype != object else group_codes(c.tolist())
    if a.valid is not None and len(g): g = np.where(a.valid, g, g.max() + 1)
    return g


def code_order(a, desc=False, nulls_first=False, mask=None):
    """order codes of a code column (codes order like the values); with mask: of those rows only, without NULLs"""
    c = a.codes
    if mask is not None:
        sel = c[mask]
        g = np.unique(sel, return_inverse=True)[1].reshape(-1).astype(np.int64) if sel.dtype != object else order_codes(sel.tolist())
        return ((g.max() if len(g) else 0) - g) if desc else g
    valid = a.valid if a.valid is not None else np.ones(len(c), bool)
    sel = c[valid]
    if sel.dtype == object: g = order_codes(sel.tolist()); k = int(g.max()) + 1 if len(g) else 0
    else:
        u, g = np.unique(sel, return_inverse=True); g = g.reshape(-1).astype(np.int64); k = len(u)
    if desc: g = (k - 1) - g
    out = np.full(len(c), -1 if nulls_first else k, np.int64); out[valid] = g
    return out


def _comparable(a):
    """a code column as values comparable across columns: ("int", int64 exact values), ("dec", int64 scaled, scale),
    ("float", float64), ("day" / "us", int64), ("dict", codes, dictionary values); None otherwise"""
    cd, c = a.codec, a.codes
    if c.dtype == object: return None
    if cd is None: return ("float", c) if c.dtype == np.float64 else ("int", c)
    k = getattr(cd, "kind", None)
    off = int(getattr(cd, "offset", 0))
    if k in ("int", "bool", "decimal", "date", "timestamp"):
        if len(c) and abs(off) + int(np.abs(c).max()) >= 1 << 62: return None
        v = c + off
        if k == "decimal": return ("dec", v, cd.scale)
        if k == "date": return ("day", v)
        if k == "timestamp": return ("tz" if cd.zoned else "us", v)
        return ("int", v)
    if k == "dict":
        if getattr(cd, "floating", False):
            fv = np.array(cd.values, np.float64)
            return ("float", fv[np.clip(c, 0, max(len(fv) - 1, 0))] if len(fv) else np.zeros(len(c)))
        return ("dict", c, cd.values)
    return None


def joint_codes(a, b):
    """codes of two columns (code columns or value lists) under which equal values are equal: (codes of a, codes of b),
    NULLs not distinguished (mask them); the code columns' dictionaries or values are matched, not their rows"""
    if is_codes(a) and is_codes(b):
        if a.codec is b.codec and a.codes.dtype != object: return a.codes, b.codes
        x, y = _comparable(a), _comparable(b)
        if x is not None and y is not None:
            kx, ky = x[0], y[0]
            if kx == "dict" and ky == "dict":                # two dictionaries: one joint dictionary of their values
                import sqlexec
                d = {}
                ma = np.array([d.setdefault(sqlexec.nkey(v), len(d)) for v in x[2]] or [0], np.int64)
                mb = np.array([d.setdefault(sqlexec.nkey(v), len(d)) for v in y[2]] or [0], np.int64)
                return ma[np.clip(x[1], 0, len(ma) - 1)], mb[np.clip(y[1], 0, len(mb) - 1)]
            num = {"int", "dec", "float"}
            if kx in num and ky in num:
                if "float" in (kx, ky):
                    def fl(z):
                        if z[0] == "float": return z[1]
                        if len(z[1]) and int(np.abs(z[1]).max()) >= 1 << 53: return None
                        return z[1].astype(np.float64) / (10.0 ** z[2] if z[0] == "dec" else 1.0)
                    fa, fb = fl(x), fl(y)
                    if fa is not None and fb is not None:
                        u = np.unique(np.concatenate([fa, fb]), return_inverse=True)[1].reshape(-1)
                        return u[:len(fa)], u[len(fa):]
                else:
                    sa, sb = (x[2] if kx == "dec" else 0), (y[2] if ky == "dec" else 0)
                    S = max(sa, sb)
                    va, vb = x[1] * 10 ** (S - sa), y[1] * 10 ** (S - sb)
                    return va, vb
            if kx == ky and kx in ("day", "us", "tz"): return x[1], y[1]
            if {kx, ky} == {"day", "us"}:
                return (x[1] * 86_400_000_000 if kx == "day" else x[1]), (y[1] * 86_400_000_000 if ky == "day" else y[1])
    import sqlexec                                        # values (decoded): codes by nkey over both
    va = a.decode() if is_codes(a) else list(a)
    vb = b.decode() if is_codes(b) else list(b)
    g = group_codes(va + vb)
    return g[:len(va)], g[len(va):]


def nn_mask(c):
    """the rows that are not NULL (values or a code column)"""
    if is_codes(c): return c.valid.copy() if c.valid is not None else np.ones(len(c.codes), bool)
    return notnull(c)


def values_of(c):
    """the column as an object array of values (a code column is decoded)"""
    return obj(c.decode()) if is_codes(c) else c


def values_at(c, pos):
    """the values at positions pos (-1: NULL); a code column decodes only those"""
    if not is_codes(c): return _take(c, pos)
    out = np.empty(len(pos), object); ok = pos >= 0
    if ok.any(): out[ok] = obj(c.take(pos[ok]).decode())
    return out


def _prefix(x):
    """[0, x0, x0 + x1, ...] (an int64 count, or exact Python integers for an object array)"""
    if x.dtype == object:
        out = np.empty(len(x) + 1, object); out[0] = 0
        if len(x): out[1:] = np.cumsum(x)
        return out
    return np.concatenate((np.zeros(1, np.int64), np.cumsum(x, dtype=np.int64)))


# ------------------------------------------------------------------ codes
def _python_ordered(nn, tys):
    """do Python's order and equality on these values agree with SQL's (cmp_values)?"""
    if not tys <= {int, bool, float, Decimal, str, bytes, _dt.date, _dt.datetime, _dt.time}: return False
    if tys & {str, bytes, _dt.date, _dt.datetime, _dt.time} and len(tys) > 1: return False
    if tys & {_dt.datetime, _dt.time} and len({v.tzinfo is None for v in nn}) > 1: return False
    if float in tys and any(v != v for v in nn if type(v) is float): return False
    if Decimal in tys and any(v.is_nan() for v in nn if type(v) is Decimal): return False
    return True


def _codes_nonnull(nn, tys):
    """dense ascending codes of non-NULL values (equal codes for values that compare equal) and their number"""
    if not nn: return np.zeros(0, np.int64), 0
    if tys <= {int, bool}:
        try: a = np.array(nn, dtype=np.int64)
        except OverflowError: a = None
        if a is not None:
            u, inv = np.unique(a, return_inverse=True); return inv.astype(np.int64).reshape(-1), len(u)
    if tys <= {float, int, bool} and float in tys and all(-(1 << 53) <= v <= (1 << 53) for v in nn if type(v) is not float):
        a = np.array(nn, dtype=np.float64)            # NaN is the largest value, NaNs are equal, -0.0 = 0.0
        u, inv = np.unique(a, return_inverse=True, equal_nan=True); return inv.astype(np.int64).reshape(-1), len(u)
    if _python_ordered(nn, tys):
        u = sorted(set(nn))
        d = {v: i for i, v in enumerate(u)}
        return np.array(list(map(d.__getitem__, nn)), np.int64), len(u)
    import sqlexec
    reps = {}
    for v in nn: reps.setdefault(sqlexec.nkey(v), v)
    keys = list(reps)
    order = sorted(keys, key=functools.cmp_to_key(lambda x, y: F.cmp_values(reps[x], reps[y])))
    code, k, prev, first = {}, -1, None, True
    for key in order:
        v = reps[key]
        if first or F.cmp_values(prev, v) != 0: k += 1
        code[key] = k; prev, first = v, False
    return np.array([code[sqlexec.nkey(v)] for v in nn], np.int64), k + 1


def order_codes(vals, desc=False, nulls_first=False):
    """dense codes of a column for one sort direction: equal for peers (NULLs together), ascending along the sort order
    (descending values, NULLs first or last)"""
    n = len(vals)
    tys = set(map(type, vals))
    has_null = _NONE in tys
    tys.discard(_NONE)
    if has_null:
        isnull = np.fromiter((v is None for v in vals), bool, n)
        nn = [v for v in vals if v is not None]
    else:
        isnull, nn = None, vals
    c, k = _codes_nonnull(nn, tys)
    if desc: c = (k - 1) - c
    if isnull is None: return c
    out = np.empty(n, np.int64)
    out[~isnull] = c
    out[isnull] = -1 if nulls_first else k
    return out


def group_codes(vals):
    """codes under which values that are not distinct (nkey) are equal"""
    tys = set(map(type, vals))
    if tys <= {int, _NONE} and int in tys and len(vals) > 64:  # integers: numpy's unique, NULL apart
        nn = [v for v in vals if v is not None] if _NONE in tys else vals
        if max(nn) < 1 << 63 and min(nn) >= -(1 << 63):
            a = np.array([0 if v is None else v for v in vals] if _NONE in tys else vals, np.int64)
            c = np.unique(a, return_inverse=True)[1].reshape(-1).astype(np.int64)
            if _NONE in tys: c = np.where(np.fromiter((v is None for v in vals), bool, len(vals)), c.max() + 1, c)
            return c
    d = {}
    if tys <= {int, bool, str, bytes, _dt.date, _NONE}:
        return np.array([d.setdefault(v, len(d)) for v in vals], np.int64)
    import sqlexec
    nk = sqlexec.nkey
    return np.array([d.setdefault(nk(v), len(d)) for v in vals], np.int64)


def row_codes(cols, n):
    """codes of the rows of several columns, equal for rows that are not distinct (every column by nkey)"""
    if not cols: return np.zeros(n, np.int64)
    return _combine([group_codes(c) for c in cols])


def first_rows(codes):
    """the positions of each code's first row, in row order"""
    if not len(codes): return np.zeros(0, np.int64)
    _, first = np.unique(codes, return_index=True)
    return np.sort(first)


def occurrence(codes):
    """for each row, how many earlier rows have its code"""
    n = len(codes)
    if not n: return np.zeros(0, np.int64)
    o = np.argsort(codes, kind="stable")
    c = codes[o]
    start = np.ones(n, bool); start[1:] = c[1:] != c[:-1]
    run0 = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    out = np.empty(n, np.int64); out[o] = np.arange(n) - run0
    return out


def _combine(cols):
    c = cols[0]
    for d in cols[1:]:
        k = int(d.max(initial=-1)) + 1                     # (no rows: no partition)
        if k > len(d) + 1:                                 # wide codes (eq_keys): dense ones, so c * k + d stays exact
            d = np.unique(d, return_inverse=True)[1].reshape(-1).astype(np.int64); k = int(d.max(initial=-1)) + 1
        if (int(c.max(initial=-1)) + 1) * k >= 1 << 62:
            c = np.unique(c, return_inverse=True)[1].reshape(-1).astype(np.int64)
        c = c * k + d
    return c


# ------------------------------------------------------------------ layout
class Layout:
    """a window's ordering: perm (sorted position -> row), per sorted position its partition (pid, start ps, end pe) and
    peer group (gid, start gs, end ge); gstarts / gends per peer group"""
    __slots__ = ("n", "perm", "ar", "pid", "ps", "pe", "gid", "gs", "ge", "gstarts", "gends", "starts", "ends")

    def partitions(self):
        """(row indexes in order, (peer starts, peer ends) as positions within the partition) per partition"""
        perm = self.perm.tolist(); gs = self.gs.tolist(); ge = self.ge.tolist()
        for s, e in zip(self.starts.tolist(), self.ends.tolist()):
            yield perm[s:e], ([g - s for g in gs[s:e]], [g - s for g in ge[s:e]])


def eq_keys(a):
    """a code column's group key for sorting and comparing neighbours: its integer codes themselves, shifted to start
    at 0 (equal codes are equal values), NULL a value of its own; no np.unique (code_groups: dense codes)"""
    c = a.codes
    if c.dtype != np.int64 or not len(c): return code_groups(a)
    lo, hi = int(c.min()), int(c.max())
    if hi - lo >= 1 << 62: return code_groups(a)          # codes this wide: dense ones (the sums below stay in int64)
    g = c - lo if lo else c
    if a.valid is not None: g = np.where(a.valid, g, int(g.max()) + 1)
    return g


def sort_keys(a, desc=False, nulls_first=False):
    """a code column's sort key: its integer codes (they order like the values), turned for DESC, NULLs placed;
    np.lexsort orders by it as by code_order, without np.unique"""
    c = a.codes
    if c.dtype != np.int64 or not len(c): return code_order(a, desc, nulls_first)
    lo, hi = int(c.min()), int(c.max())
    if hi - lo >= 1 << 62: return code_order(a, desc, nulls_first)
    g = (hi - c) if desc else (c - lo if lo else c)
    if a.valid is not None: g = np.where(a.valid, g, -1 if nulls_first else hi - lo + 1)
    return g


def lexsort(keys, n):
    """np.lexsort(keys) (the last key the primary one): when the other key is already in order row by row (an ORDER BY
    on the rows' own order, often), a stable sort on the primary key alone, a radix sort when its codes fit 16 bits"""
    if not keys: return np.arange(n)
    if len(keys) <= 2 and n > 1:
        if len(keys) == 2 and not bool((keys[0][1:] >= keys[0][:-1]).all()): return np.lexsort(keys)
        k = keys[-1]
        if len(keys) == 1 and bool((k[1:] >= k[:-1]).all()): return np.arange(n)       # already in order
        lo, hi = int(k.min()), int(k.max())
        if hi - lo < 1 << 16: k = (k - lo).astype(np.uint16)
        return np.argsort(k, kind="stable")
    return np.lexsort(keys)


def layout(parts, okeys, specs, n):
    """order the rows of a window: parts / okeys are the PARTITION BY / ORDER BY value columns (collation keys applied),
    specs [(desc, nulls_first)]"""
    L = Layout(); L.n = n
    pc = _combine([eq_keys(p) if is_codes(p) else group_codes(p) for p in parts]) if parts else None
    oc = [sort_keys(k, d, nf) if is_codes(k) else order_codes(k, d, nf) for k, (d, nf) in zip(okeys, specs)]
    keys = list(reversed(oc)) + ([pc] if pc is not None else [])
    L.perm = lexsort(keys, n)
    L.ar = np.arange(n)
    newp = np.zeros(n, bool)
    if n: newp[0] = True
    if pc is not None and n:
        P = pc[L.perm]; newp[1:] = P[1:] != P[:-1]
    if oc:
        newg = newp.copy()
        for o in oc:
            O = o[L.perm]; newg[1:] |= O[1:] != O[:-1]
    else:
        newg = newp                                     # no ORDER BY: the partition is one peer group
    L.starts = np.flatnonzero(newp); L.ends = np.append(L.starts[1:], n).astype(np.int64)
    L.pid = np.cumsum(newp) - 1
    L.ps = L.starts[L.pid]; L.pe = L.ends[L.pid]
    L.gstarts = np.flatnonzero(newg); L.gends = np.append(L.gstarts[1:], n).astype(np.int64)
    L.gid = np.cumsum(newg) - 1
    L.gs = L.gstarts[L.gid]; L.ge = L.gends[L.gid]
    return L


# ------------------------------------------------------------------ frames
class Const:
    """a frame offset that is the same on every row (a constant expression)"""
    __slots__ = ("v",)
    def __init__(self, v): self.v = v


def _ints(offs, n):
    """integer offsets per row (an error for NULL or negative ones, as the row-by-row path), capped past the partition"""
    if isinstance(offs, Const):
        o = offs.v
        if o is None: raise SqlError("frame offset must not be NULL")
        if type(o) is not int: raise NoVec
        if o < 0: raise SqlError("frame offset must not be negative")
        return np.full(n, min(o, n + 1), np.int64)
    offs = offs.tolist()
    if any(o is None for o in offs): raise SqlError("frame offset must not be NULL")
    if not all(type(o) is int for o in offs): raise NoVec
    if any(o < 0 for o in offs): raise SqlError("frame offset must not be negative")
    return np.array([min(o, n + 1) for o in offs], np.int64)


def _num_array(xs):
    """the values as an int64 or float64 array when that is exact (NaN stays NaN), else None"""
    tys = set(map(type, xs))
    if tys <= {int}:
        if xs and (max(xs) >= 1 << 62 or min(xs) <= -(1 << 62)): return None
        return np.array(xs, np.int64)
    if tys <= {float}: return np.array(xs, np.float64)
    return None


def _targets(xs, offs, sgn):
    """key + offset (FOLLOWING) or key - offset (PRECEDING) per row: an array for integer or double keys, else values"""
    if isinstance(offs, Const):
        o = offs.v
        a = _num_array(xs)
        if a is not None and type(o) in (int, float) and (type(o) is float or abs(o) < 1 << 62):
            if a.dtype == np.int64 and type(o) is float:
                if not len(a) or np.abs(a).max() < 1 << 53: a = a.astype(np.float64)
                else: a = None
            if a is not None: return a, (a + o if sgn > 0 else a - o)
        offs = [o] * len(xs)
    else: offs = offs.tolist()
    tx = set(map(type, xs)); to = set(map(type, offs))
    if tx <= _NUM and to <= _NUM and bool not in tx:
        tg = [x + o for x, o in zip(xs, offs)] if sgn > 0 else [x - o for x, o in zip(xs, offs)]
    else:
        f = F._add if sgn > 0 else F._sub
        tg = [f(x, o) for x, o in zip(xs, offs)]
    return None, tg


def _code_offset(key, v):
    """a RANGE offset in the key column's code units (codes are values shifted, scaled for decimals), or None"""
    cd, c = key.codec, key.codes
    if c.dtype == object or v is None: return None
    k = getattr(cd, "kind", None)
    o = None
    if k == "int" and type(v) is int: o = v
    elif k == "decimal":
        if type(v) is int: o = v * 10 ** cd.scale
        elif type(v) is Decimal and v.is_finite():
            q = v.scaleb(cd.scale)
            if q == q.to_integral_value(): o = int(q)
    elif k == "date":
        if type(v) is int: o = v
        elif isinstance(v, F.Interval) and v.months == 0 and v.micros == 0: o = v.days
    elif k == "timestamp" and isinstance(v, F.Interval) and v.months == 0:
        o = v.days * 86_400_000_000 + v.micros
    if o is None or (len(c) and abs(o) + int(np.abs(c).max()) >= 1 << 62): return None
    return o


def _range_edge(kind, side, L, offs, key, desc):
    """RANGE <offset> PRECEDING / FOLLOWING: the rows whose key lies within the offset of the current row's key; NULL keys
    frame their peers"""
    out = (L.gs if side == "a" else L.ge).copy()
    nnidx = np.flatnonzero(nn_mask(key))
    if not len(nnidx): return out
    sgn = -1 if kind == "preceding" else 1
    if desc: sgn = -sgn
    oc = _code_offset(key, offs.v) if is_codes(key) and isinstance(offs, Const) else None
    if oc is not None:                                   # the offset in code units: keys and targets stay codes
        x = key.codes[nnidx]
        arr, tg = x, (x + oc if sgn > 0 else x - oc)
        xs = None
    else:
        if is_codes(key): key = values_of(key)
        if not isinstance(offs, Const):
            offs = offs[nnidx]
            if any(v is None for v in offs): raise SqlError("frame offset must not be NULL")
        xs = key[nnidx].tolist()
        arr, tg = _targets(xs, offs, sgn)
    m = len(nnidx)
    if arr is not None:                                  # keys and targets ranked together, as numbers
        both = np.concatenate((arr, tg))
        u, codes = np.unique(both, return_inverse=True, equal_nan=True)
        codes = codes.reshape(-1).astype(np.int64)
        if desc: codes = (len(u) - 1) - codes
    else:
        if any(t is None for t in tg): raise NoVec
        codes = order_codes(xs + tg, desc)
    K = int(codes.max()) + 1
    pid = L.pid[nnidx]
    comp = pid * K + codes[:m]                           # ascending: partitions in order, keys sorted within each
    j = np.searchsorted(comp, pid * K + codes[m:], side="left" if side == "a" else "right")
    f = np.searchsorted(comp, pid * K, side="left")
    out[nnidx] = nnidx[f] + (j - f)
    return out


def _edge(mode, bd, side, L, bvals, key, desc):
    kind = bd[0]
    if kind == "unbounded_preceding": return L.ps
    if kind == "unbounded_following": return L.pe
    if kind == "current":
        if mode == "rows": return L.ar if side == "a" else L.ar + 1
        return L.gs if side == "a" else L.ge
    offs = bvals[side]
    if mode == "rows":
        off = _ints(offs, L.n)
        q = L.ar - off if kind == "preceding" else L.ar + off
        return q if side == "a" else q + 1
    if mode == "groups":
        off = _ints(offs, L.n)
        g0 = L.gid[L.ps]
        gi = L.gid - g0; G = L.gid[L.pe - 1] - g0 + 1
        tgt = gi - off if kind == "preceding" else gi + off
        gx = np.clip(g0 + tgt, 0, len(L.gstarts) - 1)
        val = L.gstarts[gx] if side == "a" else L.gends[gx]
        return np.where(tgt < 0, L.ps, np.where(tgt >= G, L.pe, val))
    if key is None: raise SqlError("RANGE with an offset needs exactly one ORDER BY column")
    if isinstance(offs, Const):
        if offs.v is None: raise SqlError("frame offset must not be NULL")
    elif any(o is None for o in offs): raise SqlError("frame offset must not be NULL")
    return _range_edge(kind, side, L, offs, key, desc)


def frame_bounds(w, L, bvals, key, desc):
    """[lo, hi) per row, as sorted positions within the row's partition"""
    mode, a, b = w.frame if w.frame is not None else ("range", ("unbounded_preceding", None), ("current", None))
    if not w.order and mode == "range":
        a = ("unbounded_preceding", None) if a[0] == "current" else a
        b = ("unbounded_following", None) if b[0] == "current" else b
    lo = _edge(mode, a, "a", L, bvals, key, desc)
    hi = _edge(mode, b, "b", L, bvals, key, desc)
    lo = np.minimum(np.maximum(lo, L.ps), L.pe)
    hi = np.maximum(lo, np.minimum(hi, L.pe))
    return lo, hi


def segments(w, L, lo, hi):
    """the frame without its excluded rows: up to three runs [s, e) in order"""
    ex = w.exclude.upper().replace(" ", "_") if w.exclude else None
    if ex is None or ex == "NO_OTHERS": return [(lo, hi)]
    if ex == "CURRENT_ROW": x0, x1 = L.ar, L.ar + 1
    elif ex in ("GROUP", "TIES"): x0, x1 = L.gs, L.ge
    else: raise SqlError(f"EXCLUDE {w.exclude} is not supported")
    segs = [(lo, np.maximum(lo, np.minimum(hi, x0)))]
    if ex == "TIES":
        inside = (L.ar >= lo) & (L.ar < hi)
        segs.append((L.ar, np.where(inside, L.ar + 1, L.ar)))
    segs.append((np.minimum(hi, np.maximum(lo, x1)), hi))
    return segs


def _over(pref, segs):
    tot = None
    for s, e in segs:
        d = pref[e] - pref[s]
        tot = d if tot is None else tot + d
    return tot


# ------------------------------------------------------------------ positions
def _first(E, segs, n):
    """the first eligible position of each frame (-1 when none); n positions"""
    out = np.full(len(segs[0][0]), -1, np.int64)
    if not n: return out
    nxt = np.empty(n + 1, np.int64); nxt[n] = n
    nxt[:n] = np.minimum.accumulate(np.where(E, np.arange(n), n)[::-1])[::-1]
    for s, e in reversed(segs):
        c = nxt[s]
        out = np.where(c < e, c, out)
    return out


def _last(E, segs, n):
    out = np.full(len(segs[0][0]), -1, np.int64)
    if not n: return out
    prv = np.maximum.accumulate(np.where(E, np.arange(n), -1))
    for s, e in segs:
        c = prv[np.maximum(e - 1, 0)]
        out = np.where((e > s) & (c >= s), c, out)
    return out


def _nth(E, segs, k, from_last, n):
    """the k-th eligible position of each frame, from its start or its end (-1 when there are fewer)"""
    c = _prefix(E.astype(np.int64))
    epos = np.flatnonzero(E)
    out = np.full(len(segs[0][0]), -1, np.int64)
    rem = k.copy()
    for s, e in (reversed(segs) if from_last else segs):
        cnt = c[e] - c[s]
        hit = (out < 0) & (rem >= 1) & (rem <= cnt)
        if hit.any():
            j = c[e] - rem if from_last else c[s] + rem - 1
            out = np.where(hit, epos[np.clip(j, 0, max(len(epos) - 1, 0))] if len(epos) else -1, out)
        rem = np.where(out < 0, rem - cnt, rem)
    return out


def _take(vals, pos):
    out = np.empty(len(pos), object)
    ok = pos >= 0
    out[ok] = vals[pos[ok]]
    return out


def _range_reduce(key, segs, fn, ident):
    """fn (np.minimum / np.maximum) of key over each frame's runs, ident when empty: a sparse table built one level at
    a time, each query answered at its level (memory O(n))"""
    n = len(key)
    out = np.full(len(segs[0][0]), ident, key.dtype)
    qs = []
    for s, e in segs:
        ln = e - s
        lv = np.where(ln > 0, np.frexp(np.maximum(ln, 1).astype(np.float64))[1] - 1, -1)
        qs.append((s, e, lv))
    top = max((int(lv.max()) for _, _, lv in qs if len(lv)), default=-1) if n else -1
    cur = key
    for lv in range(top + 1):
        w = 1 << lv
        for s, e, lvs in qs:
            r = np.flatnonzero(lvs == lv)
            if len(r): out[r] = fn(out[r], fn(cur[s[r]], cur[e[r] - w]))
        if lv < top: cur = fn(cur[:len(cur) - w], cur[w:])
    return out


# ------------------------------------------------------------------ exact sums
def _scaled(vals, valid):
    """the valid values as exact integers over one scale: (kind, X object array (0 where invalid), scale, specials)
    with value = X / scale; specials (pinf, ninf, nan) masks for non-finite floats"""
    if is_codes(vals):
        cd, c = vals.codec, vals.codes
        kind = getattr(cd, "kind", None)
        if kind in ("int", "decimal", "bool") and c.dtype != object:      # affine: value = (code + offset) / 10^scale
            off = int(getattr(cd, "offset", 0))
            if c.size and abs(off) + int(np.abs(c).max()) >= 1 << 62: return _scaled(values_of(vals), valid)
            X = np.zeros(len(c), np.int64); X[valid] = c[valid] + off
            sc = 10 ** int(getattr(cd, "scale", 0)) if kind == "decimal" else 1
            return ("dec" if kind == "decimal" else "int"), X.astype(object), sc, None
        if kind == "dict" and getattr(cd, "floating", False) and c.dtype != object:   # a DOUBLE dictionary: its values
            fv = np.array(cd.values, np.float64)
            a = np.zeros(len(c), np.float64); a[valid] = fv[c[valid]]
            return _scaled_doubles(a, valid)
        return _scaled(values_of(vals), valid)
    n = len(vals)
    sel = vals[valid].tolist()
    tys = set(map(type, sel))
    X = np.zeros(n, object)
    if not tys or tys <= {int, bool}:
        X[valid] = [int(v) for v in sel] if bool in tys else sel
        return "int", X, 1, None
    if tys <= {int, bool, float} and all(-(1 << 53) <= v <= (1 << 53) for v in sel if type(v) is not float):
        a = np.zeros(n, np.float64); a[valid] = np.array(sel, np.float64)     # exact: doubles, and integers below 2^53
        return _scaled_doubles(a, valid)
    if tys <= {int, bool, float}:
        fin, pinf, ninf, nan = [], np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        pos = np.flatnonzero(valid)
        S = 0; rat = []
        for v in sel:
            if type(v) is float and (v != v or v in (float("inf"), float("-inf"))):
                rat.append(None); continue
            a, b = v.as_integer_ratio() if type(v) is float else (int(v), 1)
            rat.append((a, b)); S = max(S, b.bit_length() - 1)
        xs = []
        for i, r in enumerate(rat):
            if r is None:
                v = sel[i]
                (nan if v != v else pinf if v > 0 else ninf)[pos[i]] = True; xs.append(0)
            else: xs.append(r[0] << (S - (r[1].bit_length() - 1)))
        X[valid] = xs
        return "float", X, 1 << S, (pinf, ninf, nan) if (pinf.any() or ninf.any() or nan.any()) else None
    if tys <= {int, bool, Decimal}:
        S = 0
        for v in sel:
            if type(v) is Decimal:
                if not v.is_finite(): raise NoVec
                S = max(S, -v.as_tuple().exponent)
        xs = []
        for v in sel:
            if type(v) is Decimal:
                sg, dg, ex = v.as_tuple()
                c = int("".join(map(str, dg))) if dg else 0
                xs.append((-c if sg else c) * 10 ** (ex + S))
            else: xs.append(int(v) * 10 ** S)
        X[valid] = xs
        return "dec", X, 10 ** S, None
    raise NoVec


def _scaled_doubles(a, valid):
    """doubles (a float64 array; the valid rows count) as exact integers over one power of two"""
    n = len(a)
    X = np.zeros(n, object)
    pinf, ninf, nan = valid & (a == np.inf), valid & (a == -np.inf), valid & np.isnan(a)
    mant, ex = np.frexp(np.where(valid & np.isfinite(a), a, 0.0))
    M = (mant * 9007199254740992.0).astype(np.int64)      # 2^53 times the mantissa: an exact integer
    ex = ex.astype(np.int64) - 53
    nz = M != 0
    E = min(0, int(ex[nz].min())) if nz.any() else 0
    sh = np.where(nz, ex - E, 0)
    X[:] = M.astype(object) << sh.astype(object)
    return "float", X, 1 << -E, (pinf, ninf, nan) if (pinf.any() or ninf.any() or nan.any()) else None


def _div(num, den):
    """num / den correctly rounded (integers), past the double range as the Fraction path does"""
    try: return num / den
    except OverflowError: return float(Decimal(num) / Decimal(den))


def fdiv(num, den):
    """num / den of integer arrays (object or int64), each the double nearest the exact quotient (F.ffloat)"""
    num, den = np.asarray(num, object), np.asarray(den, object)
    if not len(num): return np.zeros(0, np.float64)
    try: return (num / den).astype(np.float64)
    except OverflowError: return np.array([_div(a, b) for a, b in zip(num.tolist(), den.tolist())], np.float64)


def bivariate_arrays(name, n, Sy, Sx, Syy, Sxx, Sxy, ky, kx):
    """covariance, correlation and regression statistics of every group at once (F.bivariate's definitions, each the
    double nearest the exact value) from exact integer moment sums: the values are y = Y / ky and x = X / kx, so
    sum y = Sy / ky, sum yy = Syy / ky^2, sum xy = Sxy / (kx ky), ... With
        Cxy = n Sxy - Sx Sy,  Cxx = n Sxx - Sx^2,  Cyy = n Syy - Sy^2      (exact integers)
    covar_pop = Cxy / (n^2 kx ky), regr_slope = Cxy kx / (Cxx ky), regr_intercept = (Sy Cxx - Cxy Sx) / (Cxx ky n),
    regr_r2 = Cxy^2 / (Cxx Cyy), ... -> object array (None where ISO gives NULL)"""
    n = np.asarray(n).astype(object)
    out = np.full(len(n), None, object)
    if name == "regr_count": out[:] = n; return out
    idx = np.nonzero(np.asarray(n != 0, bool))[0]
    if not len(idx): return out
    N = n[idx]; Y, X = (np.asarray(a, object)[idx] for a in (Sy, Sx))
    YY, XX, XY = (np.asarray(a, object)[idx] for a in (Syy, Sxx, Sxy))
    Cxy = N * XY - X * Y
    if name == "regr_avgx": out[idx] = fdiv(X, N * kx); return out
    if name == "regr_avgy": out[idx] = fdiv(Y, N * ky); return out
    if name == "covar_pop": out[idx] = fdiv(Cxy, N * N * (kx * ky)); return out
    if name == "covar_samp":
        m = np.asarray(N >= 2, bool); out[idx[m]] = fdiv(Cxy[m], N[m] * (N[m] - 1) * (kx * ky)); return out
    if name == "regr_sxy": out[idx] = fdiv(Cxy, N * (kx * ky)); return out
    Cxx = N * XX - X * X
    if name == "regr_sxx": out[idx] = fdiv(Cxx, N * (kx * kx)); return out
    Cyy = N * YY - Y * Y
    if name == "regr_syy": out[idx] = fdiv(Cyy, N * (ky * ky)); return out
    nx = np.asarray(Cxx != 0, bool)
    if name == "corr":
        m = nx & np.asarray(Cyy != 0, bool)
        a = fdiv(Cxy[m], N[m] * (kx * ky)); b = fdiv(Cxx[m], N[m] * (kx * kx)); c = fdiv(Cyy[m], N[m] * (ky * ky))
        out[idx[m]] = a / np.sqrt(b * c); return out
    if name == "regr_slope": out[idx[nx]] = fdiv(Cxy[nx] * kx, Cxx[nx] * ky); return out
    if name == "regr_intercept": out[idx[nx]] = fdiv(Y[nx] * Cxx[nx] - Cxy[nx] * X[nx], Cxx[nx] * ky * N[nx]); return out
    if name == "regr_r2":
        one = nx & ~np.asarray(Cyy != 0, bool); m = nx & np.asarray(Cyy != 0, bool)
        out[idx[one]] = 1.0; out[idx[m]] = fdiv(Cxy[m] * Cxy[m], Cxx[m] * Cyy[m]); return out
    raise NoVec(name)


def _special(sp, segs):
    """per frame: NaN, +inf or -inf when the frame holds non-finite values (IEEE: inf - inf is NaN), else None"""
    if sp is None: return None
    pinf, ninf, nan = (_over(_prefix(m.astype(np.int64)), segs) for m in sp)
    out = np.full(len(pinf), None, object)
    out[ninf > 0] = float("-inf"); out[pinf > 0] = float("inf")
    out[(nan > 0) | ((pinf > 0) & (ninf > 0))] = float("nan")
    return out


# ------------------------------------------------------------------ evaluation
_AGGS = {"count_star", "count", "count_if", "countif", "sum", "fsum", "sumkahan", "kahan_sum", "avg", "mean", "min", "max",
         "arg_min", "min_by", "argmin", "arg_max", "max_by", "argmax", "every", "bool_and", "bool_or", "any", "some",
         "any_value", "arbitrary", "first", "last", "var_pop", "var_samp", "variance", "stddev_pop", "stddev_samp",
         "stddev", "covar_pop", "covar_samp", "corr", "regr_slope", "regr_intercept", "regr_count", "regr_r2", "regr_avgx",
         "regr_avgy", "regr_sxx", "regr_syy", "regr_sxy", "bit_and", "bit_or", "bit_xor"}
_BIV = {"covar_pop", "covar_samp", "corr", "regr_slope", "regr_intercept", "regr_count", "regr_r2", "regr_avgx", "regr_avgy",
        "regr_sxx", "regr_syy", "regr_sxy"}


def evaluate(w, L, args, filt, bvals, key, desc):
    """the window function's values in sorted order (an array), or NoVec. args: the argument columns (lists, row order),
    filt: FILTER row mask (row order) or None, bvals: {side: frame offsets (row order)}, key: the single ORDER BY key
    column (row order) or None"""
    n = L.n; name = w.name; perm = L.perm
    p = L.ar - L.ps; m = L.pe - L.ps
    S = lambda col: col.take(perm) if is_codes(col) else obj(col)[perm]
    V = lambda col: values_of(S(col))
    if name == "row_number": return p + 1
    if name == "rank": return L.gs - L.ps + 1
    if name == "dense_rank": return L.gid - L.gid[L.ps] + 1
    if name == "percent_rank": return np.where(m <= 1, 0.0, (L.gs - L.ps) / np.maximum(m - 1, 1))
    if name == "cume_dist": return (L.ge - L.ps) / m
    if name == "ntile":
        ks = V(args[0]).tolist()
        if not all(type(k) is int for k in ks if k is not None): raise NoVec
        if any(k <= 0 for k in ks if k is not None): raise SqlError("Argument for ntile must be greater than zero")
        kk = np.array([1 if k is None else min(k, n + 1) for k in ks], np.int64)
        q, r = np.divmod(m, kk)
        big = r * (q + 1)
        v = np.where(p < big, p // (q + 1) + 1, np.where(q != 0, (p - big) // np.maximum(q, 1) + 1 + r, p + 1))
        out = np.empty(n, object); out[:] = v
        out[[k is None for k in ks]] = None
        return out
    if name in ("lag", "lead"):
        vals = V(args[0])
        dflt = V(args[2]) if len(args) > 2 else None
        if len(args) > 1:
            offs = V(args[1]).tolist()
            if not all(type(o) is int for o in offs if o is not None): raise NoVec
            offnull = np.array([o is None for o in offs], bool)
            off = np.array([0 if o is None else max(min(o, n + 1), -(n + 1)) for o in offs], np.int64)
        else:
            offnull, off = None, np.ones(n, np.int64)
        step = off if name == "lead" else -off
        if not w.ignore_nulls:
            tgt = L.ar + step
            ok = (tgt >= L.ps) & (tgt < L.pe)
        else:
            nnm = notnull(vals)
            c = _prefix(nnm.astype(np.int64)); nnpos = np.flatnonzero(nnm); K = len(nnpos)
            k = np.abs(step)
            j = np.where(step > 0, c[L.ar + 1] + k - 1, c[L.ar] - k)
            pos = nnpos[np.clip(j, 0, max(K - 1, 0))] if K else np.zeros(n, np.int64)
            ok = (j >= 0) & (j < K) & (pos >= L.ps) & (pos < L.pe)
            tgt = np.where(step == 0, L.ar, pos)
            ok = ok | (step == 0)
        src = vals[np.clip(tgt, 0, max(n - 1, 0))]
        out = np.where(ok, src, dflt if dflt is not None else None).astype(object)
        if offnull is not None: out[offnull] = None
        return out
    # framed functions
    if w.distinct and name not in ("count", "sum", "avg", "mean", "min", "max"): raise NoVec
    if name not in ("first_value", "last_value", "nth_value") and name not in _AGGS and name not in _FRAME_ORDER: raise NoVec
    bv = {s: v if isinstance(v, Const) else S(v) for s, v in bvals.items()}
    lo, hi = frame_bounds(w, L, bv, S(key) if key is not None else None, desc)
    segs = segments(w, L, lo, hi)
    if name in ("first_value", "last_value", "nth_value"):
        vals = S(args[0])
        E = nn_mask(vals) if w.ignore_nulls else np.ones(n, bool)
        if name == "first_value": return values_at(vals, _first(E, segs, n))
        if name == "last_value": return values_at(vals, _last(E, segs, n))
        ks = V(args[1]).tolist()
        if not all(type(k) is int for k in ks if k is not None): raise NoVec
        if any(k <= 0 for k in ks if k is not None): raise SqlError("NTH_VALUE: the position must be positive")
        kk = np.array([1 if k is None else min(k, n + 1) for k in ks], np.int64)
        out = values_at(vals, _nth(E, segs, kk, w.from_last, n))
        out[[k is None for k in ks]] = None
        return out
    valid = np.ones(n, bool) if filt is None else filt[perm]
    if w.distinct and name in ("count", "sum", "avg", "mean"):
        return distinct_frames(name, S(args[0]), valid, segs, n)
    if name in _FRAME_ORDER: return order_frames(name, w.params, S(args[0]), valid, segs, n)
    return aggregate(name, [S(a) for a in args], valid, segs, n)      # MIN / MAX (DISTINCT): the same


_FRAME_ORDER = {"median", "quantile_cont", "percentile_cont", "quantile_disc", "percentile_disc", "quantile"}


def order_frames(name, params, vals, valid, segs, n):
    """MEDIAN and PERCENTILE_CONT / _DISC over every frame (runs of sorted positions, EXCLUDE included): each row gets
    its rank in (value, position) order, the order the row-by-row path sorts a frame in; the frame's row at rank k is
    found by a parallel binary search on that rank, every round counting each frame's rows below the midpoint with
    merge-sort-tree dominance counts (log n rounds of O(log n) searches per frame, for all frames at once)"""
    import math
    valid = valid & nn_mask(vals)
    desc = name in ("quantile_disc", "percentile_disc", "quantile") and len(params) > 1 and params[1] == "desc"
    pos = np.flatnonzero(valid)
    rank = np.full(n, n, np.int64)                       # invalid rows: past every rank searched
    if len(pos):
        code = code_order(vals, desc, mask=valid) if is_codes(vals) else order_codes(vals[pos].tolist(), desc)
        byrank = pos[np.lexsort((pos, code))]
        rank[byrank] = np.arange(len(byrank))
    else: byrank = pos
    q = _dominance(rank, None, n)
    nq = len(segs[0][0])
    def below(t):                                        # per frame: its valid rows of rank < t
        c = np.zeros(nq, np.int64)
        for s_, e_ in segs:
            ne = e_ > s_
            if ne.any(): c += np.where(ne, q(e_, t)[0] - q(s_, t)[0], 0)
        return c
    N = below(np.full(nq, n, np.int64))
    def kth(k):                                          # per frame: the row at 0-based rank k (k < N)
        lo = np.zeros(nq, np.int64); hi = np.full(nq, max(len(byrank) - 1, 0), np.int64)
        while True:
            act = lo < hi
            if not act.any(): break
            mid = (lo + hi) // 2
            left = below(mid + 1) > k
            hi = np.where(act & left, mid, hi); lo = np.where(act & ~left, mid + 1, lo)
        return byrank[np.clip(lo, 0, max(len(byrank) - 1, 0))] if len(byrank) else np.zeros(nq, np.int64)
    Nl = N.tolist()
    if name == "median":
        qs, single = [Fraction(1, 2)], True
        num = values_at(vals, pos[:1]).tolist() if len(pos) else []
        cont = not (bool(num) and not isinstance(num[0], (int, float, Decimal, bool, _dt.date, F.Interval)))
    else:
        qv = F._q_param(params)
        single = not isinstance(qv, list); qs = [qv] if single else qv
        cont = name in ("quantile_cont", "percentile_cont")
    res = []
    for qf in qs:
        if cont:
            pr = [Fraction(qf) * (m - 1) if m else Fraction(0) for m in Nl]
            lo_r = [math.floor(x) for x in pr]; hi_r = [math.ceil(x) for x in pr]
            a = values_at(vals, kth(np.array(lo_r, np.int64))); b = values_at(vals, kth(np.array(hi_r, np.int64)))
            res.append([None if m == 0 else F.qc_mix(x, y, pp, l, h) for x, y, pp, l, h, m in zip(a, b, pr, lo_r, hi_r, Nl)])
        else:
            k = [max(0, math.ceil(Fraction(qf) * m) - 1) if m else 0 for m in Nl]
            res.append(values_at(vals, kth(np.array(k, np.int64))).tolist())
    out = np.full(nq, None, object)
    for g in range(nq):
        if Nl[g] == 0: continue
        out[g] = res[0][g] if single else [r[g] for r in res]
    return out


# ------------------------------------------------------------------ DISTINCT over frames: dominance counts
def _dominance(prev, wts, n):
    """a merge-sort tree over positions: query(e, s) -> for each query, how many positions j < e have prev[j] < s
    (and the sum of their weights). Each level k sorts every aligned block of 2^k positions by prev once."""
    levels = []
    nb = max(1, (n - 1).bit_length())
    big = n + 2
    for k in range(nb + 1):
        key = (np.arange(n) >> k) * big + (prev + 1)
        o = np.argsort(key, kind="stable")
        cw = None
        if wts is not None:
            cw = np.empty(n + 1, object); cw[0] = 0
            if n: cw[1:] = np.cumsum(wts[o])
        levels.append((key[o], cw))
    def query(e, s):
        cnt = np.zeros(len(e), np.int64)
        tot = np.zeros(len(e), object) if wts is not None else None
        for k in range(nb + 1):
            hit = np.flatnonzero((e >> k) & 1)
            if not len(hit): continue
            blk = (e[hit] >> (k + 1)) << 1                  # the aligned block of 2^k positions below e at level k
            start = blk << k
            keys, cw = levels[k]
            idx = np.searchsorted(keys, blk * big + (s[hit] + 1), "left")
            cnt[hit] += idx - start
            if tot is not None: tot[hit] = tot[hit] + (cw[idx] - cw[start])
        return cnt, tot
    return query


def distinct_frames(name, col, valid, segs, n):
    """COUNT / SUM / AVG (DISTINCT x) over each frame: a value counts once, at its first occurrence in the frame, that is
    where its previous occurrence lies before the frame's start. Over one run [s, e): the positions j < e whose
    previous occurrence is before s, less the positions before s (a dominance count); frames of several runs (EXCLUDE
    in the middle of the frame) are counted row by row."""
    valid = valid & nn_mask(col)
    vc = code_groups(col) if is_codes(col) else group_codes(col.tolist())
    pos = np.flatnonzero(valid)
    prev = np.full(n, n, np.int64)                       # invalid rows: never counted
    if len(pos):
        o = pos[np.lexsort((pos, vc[pos]))]
        same = np.zeros(len(o), bool); same[1:] = vc[o[1:]] == vc[o[:-1]]
        pv = np.full(len(o), -1, np.int64); pv[same] = o[:-1][same[1:]]
        prev[o] = pv
    nq = len(segs[0][0])
    ne = sum((e > s).astype(np.int64) for s, e in segs)
    s1 = np.zeros(nq, np.int64); e1 = np.zeros(nq, np.int64)
    for s_, e_ in segs:
        one = e_ > s_
        s1 = np.where(one, s_, s1); e1 = np.where(one, e_, e1)
    want_sum = name != "count"
    if want_sum:
        kind, X, scale, sp = _scaled(col, valid)
        wts = X
    else: wts = None
    q = _dominance(prev, wts, n)
    cnt_e, sum_e = q(e1, s1)
    cv = _prefix(valid.astype(np.int64))
    cnt = cnt_e - cv[s1]
    if want_sum:
        pw = _prefix(X)
        tot = sum_e - pw[s1]
    multi = np.flatnonzero(ne >= 2)
    if len(multi):                                     # several runs: the frame's distinct values, row by row
        for i in multi.tolist():
            seen = {}
            for s_, e_ in segs:
                for j in range(int(s_[i]), int(e_[i])):
                    if valid[j] and vc[j] not in seen: seen[vc[j]] = j
            cnt[i] = len(seen)
            if want_sum: tot[i] = sum(X[j] for j in seen.values())
    if name == "count": return cnt
    out = np.full(nq, None, object)
    has = cnt > 0
    avg = name in ("avg", "mean")
    if kind == "int": out[has] = (tot[has] / cnt[has].astype(object)) if avg else tot[has]
    elif kind == "float":
        try: sv = (tot[has] / scale).astype(np.float64)
        except OverflowError: sv = np.array([_div(t, scale) for t in tot[has].tolist()], np.float64)
        out[has] = sv / cnt[has] if avg else sv
    elif avg: out[has] = tot[has] / (scale * cnt[has].astype(object))
    else:
        D = Decimal(scale)
        out[has] = [T.CTX.divide(Decimal(t), D) for t in tot[has].tolist()]
    spv = _special(sp, segs)
    if spv is not None:
        hit = has & np.array([v is not None for v in spv], bool)
        out[hit] = spv[hit]
    return out


def aggregate(name, cols, valid, segs, n, w=None):
    """an aggregate over frames or groups: cols the argument columns (object arrays, in order), valid the rows that
    take part (FILTER; rows standing for no row excluded), segs the runs [s, e) of each frame / group, w how many rows
    each row stands for (None: one). Raises NoVec for the aggregates computed row by row."""
    if name not in _AGGS: raise NoVec
    d = F.AGGS[name]
    nq = len(segs[0][0])
    valid = valid.copy()
    if name in _BIV:                                       # rows where both are non-NULL (moments2)
        for c in cols: valid &= nn_mask(c)
    elif name != "count_star" and d.skip_nulls:
        for c in cols: valid &= nn_mask(c)
    wv = valid.astype(np.int64) if w is None else np.where(valid, w, 0)
    cnt = _over(_prefix(wv), segs)
    if name in ("count_star", "count"): return cnt
    if name in ("count_if", "countif", "every", "bool_and", "bool_or", "any", "some"):
        vals = cols[0]
        if is_codes(vals) and getattr(vals.codec, "kind", None) == "bool": tru = vals.codes[valid] != 0
        else: tru = np.fromiter((bool(v) for v in values_of(vals)[valid]), bool, int(valid.sum()))
        tm = np.zeros(n, bool); tm[np.flatnonzero(valid)[tru]] = True
        t = _over(_prefix(np.where(tm, wv, 0)), segs)
        out = np.full(nq, None, object)
        has = cnt > 0
        if name in ("count_if", "countif"): out[has] = t[has]
        else: out[has] = (t == cnt)[has] if name in ("every", "bool_and") else (t > 0)[has]
        return out
    if name in ("any_value", "arbitrary", "first"): return values_at(cols[0], _first(valid, segs, n))
    if name == "last": return values_at(cols[0], _last(valid, segs, n))
    if name in ("bit_and", "bit_or", "bit_xor"):           # 64-bit integers: XOR by prefixes, AND / OR as range reductions
        sel = values_of(cols[0])[valid].tolist()
        if not all(type(v) is int for v in sel) or (sel and (max(sel) >= 1 << 63 or min(sel) < -(1 << 63))): raise NoVec
        X = np.zeros(n, np.int64); X[valid] = sel
        if name == "bit_xor":
            if w is not None: X = np.where(w % 2 == 1, X, 0)            # a value taken an even number of times cancels
            P = np.concatenate((np.zeros(1, np.int64), np.bitwise_xor.accumulate(X))) if n else np.zeros(1, np.int64)
            r = None
            for s_, e_ in segs:
                d = P[e_] ^ P[s_]; r = d if r is None else r ^ d
        else:
            ident = -1 if name == "bit_and" else 0
            r = _range_reduce(np.where(valid, X, ident), segs, np.bitwise_and if name == "bit_and" else np.bitwise_or, ident)
        out = np.full(nq, None, object); has = cnt > 0
        out[has] = r[has]
        return out
    ar = np.arange(n)
    if name in ("min", "max", "arg_min", "min_by", "argmin", "arg_max", "max_by", "argmax"):
        cv = cols[-1]
        code = np.zeros(n, np.int64)
        if is_codes(cv): code[valid] = code_order(cv, mask=valid)     # codes order like the values: nothing decoded
        else:
            sel = cv[valid].tolist()
            if any(isinstance(v, dict) for v in sel): raise NoVec
            code[valid] = order_codes(sel)
        big = np.iinfo(np.int64).max
        if name in ("min", "arg_min", "min_by", "argmin"):           # the first of the least values
            k = np.where(valid, code * n + ar, big)
            r = _range_reduce(k, segs, np.minimum, big)
            pos = np.where(r == big, -1, r % max(n, 1))
        else:                                                       # the first of the greatest values
            k = np.where(valid, code * n + (n - 1 - ar), -1)
            r = _range_reduce(k, segs, np.maximum, -1)
            pos = np.where(r < 0, -1, n - 1 - r % max(n, 1))
        return values_at(cols[0], pos)
    W = None if w is None else w.astype(object)
    def wsum(X):                                           # sums of X (weighted) over the runs
        return _over(_prefix(X if W is None else X * W), segs)
    if name in ("sum", "fsum", "sumkahan", "kahan_sum", "avg", "mean"):
        kind, X, scale, sp = _scaled(cols[0], valid)
        avg = name in ("avg", "mean")
        tot = None
        if kind == "int":
            try: Xi = X.astype(np.int64)
            except OverflowError: Xi = None
            mx = (int(np.abs(Xi).max()) if Xi is not None and len(Xi) else 0) if Xi is not None else 1 << 62
            mx *= (1 if w is None else max(int(w.max()), 1) if len(w) else 1)
            if mx * (n + 1) < 1 << 62:
                tot = _over(_prefix(Xi * (1 if w is None else w)), segs)
                if not avg:                                # every frame has a value: the int64 sums as they are
                    return tot if bool((cnt > 0).all()) else np.where(cnt > 0, tot.astype(object), None)
                if mx * (n + 1) < 1 << 53:
                    return np.where(cnt > 0, tot / np.maximum(cnt, 1), None).astype(object)
                tot = tot.astype(object)
        if tot is None: tot = wsum(X)
        out = np.full(nq, None, object)
        has = cnt > 0
        if kind == "int":
            out[has] = (tot[has] / cnt[has].astype(object)) if avg else tot[has]
        elif kind == "float":
            try: sv = (tot[has] / scale).astype(np.float64)
            except OverflowError: sv = np.array([_div(t, scale) for t in tot[has].tolist()], np.float64)
            out[has] = sv / cnt[has] if avg else sv
        elif avg:
            out[has] = tot[has] / (scale * cnt[has].astype(object))
        else:
            D = Decimal(scale)
            out[has] = [T.CTX.divide(Decimal(t), D) for t in tot[has].tolist()]
        spv = _special(sp, segs)
        if spv is not None:
            hit = has & np.array([v is not None for v in spv], bool)
            out[hit] = spv[hit]
        return out
    if name in ("var_pop", "var_samp", "variance", "stddev_pop", "stddev_samp", "stddev"):
        kind, X, scale, sp = _scaled(cols[0], valid)
        s1 = wsum(X); s2 = wsum(X * X)
        samp = name in ("var_samp", "variance", "stddev_samp", "stddev")
        out = np.full(nq, None, object)
        ok = cnt >= (2 if samp else 1)
        if ok.any():
            c = cnt[ok].astype(object)
            num = c * s2[ok] - s1[ok] * s1[ok]                 # n^2 scale^2 times the variance: exact
            den = c * ((c - 1) if samp else c) * (scale * scale)
            try: v = (num / den).astype(np.float64)
            except OverflowError: v = np.array([_div(a, b) for a, b in zip(num.tolist(), den.tolist())], np.float64)
            if name.startswith("stddev"): v = np.sqrt(v)
            out[ok] = v
        spv = _special(sp, segs)
        if spv is not None:                                     # NaN / infinities: NaN, one value: 0 (DuckDB)
            hit = ok & np.array([x is not None for x in spv], bool)
            out[hit] = np.where(cnt[hit] == 1, 0.0, np.nan)
        return out
    # covariance, correlation, regression: exact moments of the pairs, the shared definition
    ky, Y, sy, spy = _scaled(cols[0], valid)
    kx, Xx, sx, spx = _scaled(cols[1], valid)
    pre = [wsum(a).tolist() for a in (Y, Xx, Y * Y, Xx * Xx, Xx * Y)]
    bad = None
    if spy is not None or spx is not None:
        anyb = np.zeros(n, bool)
        for sp in (spy, spx):
            if sp is not None: anyb |= sp[0] | sp[1] | sp[2]
        bad = _over(_prefix(anyb.astype(np.int64)), segs)
    out = bivariate_arrays(name, cnt, *[np.array(p, object) for p in pre], sy, sx)
    if bad is not None: out[np.asarray(bad > 0, bool)] = F.bivariate(name, None)
    return out


# ------------------------------------------------------------------ order statistics of groups
def group_order(vals, valid, w, starts, ends, desc=False):
    """the valid rows of each group (run [s, e)) in the order of their values, ties in row order, with cumulative
    weights: (o positions, C cumulative weights, B weight before each group, N weight of each group)"""
    G = len(starts)
    grp = np.repeat(np.arange(G), ends - starts)
    pos = np.flatnonzero(valid)
    if not len(pos): code = np.zeros(0, np.int64)
    elif is_codes(vals): code = code_order(vals, desc, mask=valid)
    else: code = order_codes(vals[pos].tolist(), desc)
    o = pos[np.lexsort((pos, code, grp[pos]))] if len(pos) else pos
    C = np.cumsum(w[o]) if len(o) else np.zeros(0, np.int64)
    go = grp[o]
    first = np.searchsorted(go, np.arange(G), "left"); last = np.searchsorted(go, np.arange(G), "right")
    B = np.where(first > 0, C[np.maximum(first - 1, 0)] if len(C) else 0, 0)
    N = np.where(last > first, (C[np.maximum(last - 1, 0)] if len(C) else 0) - B, 0)
    return o, C, B, N


def pick(order, k):
    """per group, the position of the row at weighted rank k (0-based; k < N), -1 for an empty group"""
    o, C, B, N = order
    if not len(o): return np.full(len(N), -1, np.int64)
    idx = np.searchsorted(C, B + k, side="right")
    return np.where(N > 0, o[np.clip(idx, 0, len(o) - 1)], -1)


_ORDER_AGGS = {"median", "quantile_cont", "percentile_cont", "quantile_disc", "percentile_disc", "quantile", "mode"}


def order_aggregate(name, params, vals, valid, w, starts, ends):
    """MEDIAN, PERCENTILE_CONT / _DISC (one fraction or a list) and MODE of every group at once: the groups' rows sorted
    by value codes once, each rank found by a search of the cumulative counts"""
    import math
    G = len(starts)
    valid = valid & nn_mask(vals)
    if name == "mode":
        import sqlexec
        pos = np.flatnonzero(valid)
        out = np.full(G, None, object)
        if not len(pos): return out
        grp = np.repeat(np.arange(G), ends - starts)[pos]
        kc = code_groups(vals.take(pos)) if is_codes(vals) else group_codes([sqlexec.nkey(v) for v in vals[pos].tolist()])
        o = np.lexsort((pos, kc, grp))                         # (group, value, first row)
        g2, k2 = grp[o], kc[o]
        new = np.ones(len(o), bool); new[1:] = (g2[1:] != g2[:-1]) | (k2[1:] != k2[:-1])
        heads = np.flatnonzero(new)
        tot = np.add.reduceat(w[pos][o], heads)                # weight of each (group, value)
        hg, hrow = g2[heads], pos[o[heads]]                    # its group, its first row
        desc = bool(params) and params[0] == "desc"
        oc = code_order(vals.take(hrow), desc) if is_codes(vals) else order_codes(vals[hrow].tolist(), desc)
        best = np.lexsort((oc, -tot, hg))                      # per group: most frequent, then first in the order
        gfirst = np.ones(len(best), bool); gfirst[1:] = hg[best][1:] != hg[best][:-1]
        sel = best[gfirst]
        out[hg[sel]] = values_at(vals, hrow[sel])
        return out
    desc = name in ("quantile_disc", "percentile_disc", "quantile") and len(params) > 1 and params[1] == "desc"
    order = group_order(vals, valid, w, starts, ends, desc)
    N = order[3].tolist()
    if name == "median":
        qs, single = [Fraction(1, 2)], True
        fv = np.flatnonzero(valid)[:1]
        num = values_at(vals, fv).tolist() if len(fv) else []
        disc = bool(num) and not isinstance(num[0], (int, float, Decimal, bool, _dt.date, F.Interval))
        cont = not disc
    else:
        q = F._q_param(params)
        single = not isinstance(q, list); qs = [q] if single else q
        cont = name in ("quantile_cont", "percentile_cont")
    res = []
    for q in qs:
        if cont:
            lo_r, hi_r, pos_r = [], [], []
            for n in N:
                p = Fraction(q) * (n - 1) if n else Fraction(0)
                lo_r.append(math.floor(p)); hi_r.append(math.ceil(p)); pos_r.append(p)
            a = values_at(vals, pick(order, np.array(lo_r, np.int64))); b = values_at(vals, pick(order, np.array(hi_r, np.int64)))
            res.append([None if n == 0 else F.qc_mix(x, y, p, l, h) for x, y, p, l, h, n in zip(a, b, pos_r, lo_r, hi_r, N)])
        else:
            k = [max(0, math.ceil(Fraction(q) * n) - 1) if n else 0 for n in N]
            res.append(values_at(vals, pick(order, np.array(k, np.int64))).tolist())
    out = np.full(G, None, object)
    for g in range(G):
        if N[g] == 0: continue
        out[g] = res[0][g] if single else [r[g] for r in res]
    return out
