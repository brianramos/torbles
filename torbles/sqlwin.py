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
        k = int(d.max()) + 1
        if (int(c.max()) + 1) * k >= 1 << 62:
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


def layout(parts, okeys, specs, n):
    """order the rows of a window: parts / okeys are the PARTITION BY / ORDER BY value columns (collation keys applied),
    specs [(desc, nulls_first)]"""
    L = Layout(); L.n = n
    pc = _combine([group_codes(p) for p in parts]) if parts else None
    oc = [order_codes(k, d, nf) for k, (d, nf) in zip(okeys, specs)]
    keys = list(reversed(oc)) + ([pc] if pc is not None else [])
    L.perm = np.lexsort(keys) if keys else np.arange(n)
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


def _range_edge(kind, side, L, offs, key, desc):
    """RANGE <offset> PRECEDING / FOLLOWING: the rows whose key lies within the offset of the current row's key; NULL keys
    frame their peers"""
    out = (L.gs if side == "a" else L.ge).copy()
    nnidx = np.flatnonzero(notnull(key))
    if not len(nnidx): return out
    if not isinstance(offs, Const):
        offs = offs[nnidx]
        if any(v is None for v in offs): raise SqlError("frame offset must not be NULL")
    sgn = -1 if kind == "preceding" else 1
    if desc: sgn = -sgn
    xs = key[nnidx].tolist()
    arr, tg = _targets(xs, offs, sgn)
    m = len(xs)
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
    n = len(vals)
    sel = vals[valid].tolist()
    tys = set(map(type, sel))
    X = np.zeros(n, object)
    if not tys or tys <= {int, bool}:
        X[valid] = [int(v) for v in sel] if bool in tys else sel
        return "int", X, 1, None
    if tys <= {int, bool, float} and all(-(1 << 53) <= v <= (1 << 53) for v in sel if type(v) is not float):
        a = np.array(sel, np.float64)                  # exact: doubles, and integers below 2^53
        pos = np.flatnonzero(valid)
        pinf, ninf, nan = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        pinf[pos[a == np.inf]] = True; ninf[pos[a == -np.inf]] = True; nan[pos[np.isnan(a)]] = True
        mant, ex = np.frexp(np.where(np.isfinite(a), a, 0.0))
        M = (mant * 9007199254740992.0).astype(np.int64)      # 2^53 times the mantissa: an exact integer
        ex = ex.astype(np.int64) - 53
        nz = M != 0
        E = min(0, int(ex[nz].min())) if nz.any() else 0
        sh = np.where(nz, ex - E, 0)
        X[valid] = M.astype(object) << sh.astype(object)
        return "float", X, 1 << -E, (pinf, ninf, nan) if (pinf.any() or ninf.any() or nan.any()) else None
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


def _div(num, den):
    """num / den correctly rounded (integers), past the double range as the Fraction path does"""
    try: return num / den
    except OverflowError: return float(Decimal(num) / Decimal(den))


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
    S = lambda col: obj(col)[perm]
    if name == "row_number": return p + 1
    if name == "rank": return L.gs - L.ps + 1
    if name == "dense_rank": return L.gid - L.gid[L.ps] + 1
    if name == "percent_rank": return np.where(m <= 1, 0.0, (L.gs - L.ps) / np.maximum(m - 1, 1))
    if name == "cume_dist": return (L.ge - L.ps) / m
    if name == "ntile":
        ks = S(args[0]).tolist()
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
        vals = S(args[0])
        dflt = S(args[2]) if len(args) > 2 else None
        if len(args) > 1:
            offs = S(args[1]).tolist()
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
    if w.distinct: raise NoVec
    if name not in ("first_value", "last_value", "nth_value") and name not in _AGGS: raise NoVec
    bv = {s: v if isinstance(v, Const) else S(v) for s, v in bvals.items()}
    lo, hi = frame_bounds(w, L, bv, S(key) if key is not None else None, desc)
    segs = segments(w, L, lo, hi)
    if name in ("first_value", "last_value", "nth_value"):
        vals = S(args[0])
        E = notnull(vals) if w.ignore_nulls else np.ones(n, bool)
        if name == "first_value": return _take(vals, _first(E, segs, n))
        if name == "last_value": return _take(vals, _last(E, segs, n))
        ks = S(args[1]).tolist()
        if not all(type(k) is int for k in ks if k is not None): raise NoVec
        if any(k <= 0 for k in ks if k is not None): raise SqlError("NTH_VALUE: the position must be positive")
        kk = np.array([1 if k is None else min(k, n + 1) for k in ks], np.int64)
        out = _take(vals, _nth(E, segs, kk, w.from_last, n))
        out[[k is None for k in ks]] = None
        return out
    valid = np.ones(n, bool) if filt is None else filt[perm]
    return aggregate(name, [S(a) for a in args], valid, segs, n)


def aggregate(name, cols, valid, segs, n, w=None):
    """an aggregate over frames or groups: cols the argument columns (object arrays, in order), valid the rows that
    take part (FILTER; rows standing for no row excluded), segs the runs [s, e) of each frame / group, w how many rows
    each row stands for (None: one). Raises NoVec for the aggregates computed row by row."""
    if name not in _AGGS: raise NoVec
    d = F.AGGS[name]
    nq = len(segs[0][0])
    valid = valid.copy()
    if name in _BIV:                                       # rows where both are non-NULL (moments2)
        for c in cols: valid &= notnull(c)
    elif name != "count_star" and d.skip_nulls:
        for c in cols: valid &= notnull(c)
    wv = valid.astype(np.int64) if w is None else np.where(valid, w, 0)
    cnt = _over(_prefix(wv), segs)
    if name in ("count_star", "count"): return cnt
    if name in ("count_if", "countif", "every", "bool_and", "bool_or", "any", "some"):
        vals = cols[0]
        tru = np.fromiter((bool(v) for v in vals[valid]), bool, int(valid.sum()))
        tm = np.zeros(n, bool); tm[np.flatnonzero(valid)[tru]] = True
        t = _over(_prefix(np.where(tm, wv, 0)), segs)
        out = np.full(nq, None, object)
        has = cnt > 0
        if name in ("count_if", "countif"): out[has] = t[has]
        else: out[has] = (t == cnt)[has] if name in ("every", "bool_and") else (t > 0)[has]
        return out
    if name in ("any_value", "arbitrary", "first"): return _take(cols[0], _first(valid, segs, n))
    if name == "last": return _take(cols[0], _last(valid, segs, n))
    if name in ("bit_and", "bit_or", "bit_xor"):           # 64-bit integers: XOR by prefixes, AND / OR as range reductions
        sel = cols[0][valid].tolist()
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
        sel = cv[valid].tolist()
        if any(isinstance(v, dict) for v in sel): raise NoVec
        code = np.zeros(n, np.int64)
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
        return _take(cols[0], pos)
    W = None if w is None else w.astype(object)
    def wsum(X):                                           # sums of X (weighted) over the runs
        return _over(_prefix(X if W is None else X * W), segs)
    if name in ("sum", "fsum", "sumkahan", "kahan_sum", "avg", "mean"):
        kind, X, scale, sp = _scaled(cols[0], valid)
        avg = name in ("avg", "mean")
        tot = None
        if kind == "int":
            mx = max((abs(v) for v in X[valid].tolist()), default=0) * (1 if w is None else max(int(w.max()), 1) if len(w) else 1)
            if mx * (n + 1) < 1 << 62:
                tot = _over(_prefix(X.astype(np.int64) * (1 if w is None else w)), segs)
                if not avg: return np.where(cnt > 0, tot.astype(object), None)
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
    cl = cnt.tolist()
    out = np.full(nq, None, object)
    last_k = last_v = None
    for i in range(nq):
        if bad is not None and bad[i] > 0: out[i] = F.bivariate(name, None); continue
        mo = (cl[i], Fraction(pre[0][i], sy), Fraction(pre[1][i], sx), Fraction(pre[2][i], sy * sy),
              Fraction(pre[3][i], sx * sx), Fraction(pre[4][i], sx * sy))
        if mo == last_k: out[i] = last_v; continue
        last_k, last_v = mo, F.bivariate(name, mo)
        out[i] = last_v
    return out


# ------------------------------------------------------------------ order statistics of groups
def group_order(vals, valid, w, starts, ends, desc=False):
    """the valid rows of each group (run [s, e)) in the order of their values, ties in row order, with cumulative
    weights: (o positions, C cumulative weights, B weight before each group, N weight of each group)"""
    G = len(starts)
    grp = np.repeat(np.arange(G), ends - starts)
    pos = np.flatnonzero(valid)
    code = order_codes(vals[pos].tolist(), desc) if len(pos) else np.zeros(0, np.int64)
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
    valid = valid & notnull(vals)
    if name == "mode":
        import sqlexec
        pos = np.flatnonzero(valid)
        out = np.full(G, None, object)
        if not len(pos): return out
        grp = np.repeat(np.arange(G), ends - starts)[pos]
        kc = group_codes([sqlexec.nkey(v) for v in vals[pos].tolist()])
        o = np.lexsort((pos, kc, grp))                         # (group, value, first row)
        g2, k2 = grp[o], kc[o]
        new = np.ones(len(o), bool); new[1:] = (g2[1:] != g2[:-1]) | (k2[1:] != k2[:-1])
        heads = np.flatnonzero(new)
        tot = np.add.reduceat(w[pos][o], heads)                # weight of each (group, value)
        hg, hrow = g2[heads], pos[o[heads]]                    # its group, its first row
        desc = bool(params) and params[0] == "desc"
        oc = order_codes(vals[hrow].tolist(), desc)
        best = np.lexsort((oc, -tot, hg))                      # per group: most frequent, then first in the order
        gfirst = np.ones(len(best), bool); gfirst[1:] = hg[best][1:] != hg[best][:-1]
        sel = best[gfirst]
        out[hg[sel]] = vals[hrow[sel]]
        return out
    desc = name in ("quantile_disc", "percentile_disc", "quantile") and len(params) > 1 and params[1] == "desc"
    order = group_order(vals, valid, w, starts, ends, desc)
    N = order[3].tolist()
    if name == "median":
        qs, single = [Fraction(1, 2)], True
        num = [v for v in vals[valid].tolist()[:1]]
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
            a = _take(vals, pick(order, np.array(lo_r, np.int64))); b = _take(vals, pick(order, np.array(hi_r, np.int64)))
            res.append([None if n == 0 else F.qc_mix(x, y, p, l, h) for x, y, p, l, h, n in zip(a, b, pos_r, lo_r, hi_r, N)])
        else:
            k = [max(0, math.ceil(Fraction(q) * n) - 1) if n else 0 for n in N]
            res.append(_take(vals, pick(order, np.array(k, np.int64))).tolist())
    out = np.full(G, None, object)
    for g in range(G):
        if N[g] == 0: continue
        out[g] = res[0][g] if single else [r[g] for r in res]
    return out
