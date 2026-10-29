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
    """a query result: column names, SQL types, rows (tuples)"""
    def __init__(self, names, types, rows):
        self.names, self.types, self.rows = list(names), list(types), rows
    @property
    def columns(self): return self.names
    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows)
    def __repr__(self): return f"Relation({self.names}, {len(self.rows)} rows)"
    def fetchall(self): return list(self.rows)
    def fetchone(self): return self.rows[0] if self.rows else None
    def df(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.names)
    def scalar(self):
        return self.rows[0][0] if self.rows else None


class Rel:
    """rows of a block while it is evaluated: cols {expression key: [value per row]}, cnt [rows each row stands for],
    mom {(variant, moment): [sum per row]}, vcnt {variant: [count per row]}"""
    __slots__ = ("n", "cols", "cnt", "mom", "vcnt")
    def __init__(self, n):
        self.n, self.cols, self.cnt, self.mom, self.vcnt = n, {}, [1] * n, {}, {}
    def take(self, idx):
        r = Rel(len(idx))
        r.cols = {k: [v[i] for i in idx] for k, v in self.cols.items()}
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
    """the value of a constant expression"""
    r = Rel(1)
    v = ev(e, r, None)[0]
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
        return _apply(s, [ev(a, rel, cx) for a in e.args], n)
    if isinstance(e, I.Case):
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
        a = ev(e.arg, rel, cx); out = []
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
    if isinstance(v, list): return ("\0list",) + tuple(nkey(x) for x in v)
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
    cmps = [cmp_key(d, nf) for d, nf in specs]
    def c(i, j):
        for k, f in zip(keys, cmps):
            r = f(k[i], k[j])
            if r: return r
        return 0
    return sorted(idx, key=functools.cmp_to_key(c))


# ================================================================== queries
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
        rows = []
        dec = [t.s if t.id == "DECIMAL" else None for t in r.types]
        for row in r.rows:
            out = []
            for v, s in zip(row, dec):
                if isinstance(v, Err): raise SqlError(v.msg)
                if s is not None and isinstance(v, Decimal): v = T._dec_quant(v, s)
                out.append(T.out_value(v))
            rows.append(tuple(out))
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


def run_setop(q, cx):
    a, b = run_query(q.left, cx), run_query(q.right, cx)
    types = [t for _, t in q.items]
    def conv(rel):
        out = []
        for r in rel.rows:
            out.append(tuple(v if t0 == t1 or v is None or isinstance(v, Err) else T.cast_value(v, t0, t1) for v, t0, t1 in zip(r, rel.types, types)))
        return out
    ra, rb = conv(a), conv(b)
    op = q.op
    kf = [T.coll_key(t) for t in types]
    if any(kf):                                       # collated columns: rows equal when their keys are
        return _setop_collated(q, ra, rb, kf, names_types=(q.items, types), cx=cx)
    if op == "union":
        rows = ra + rb
        if not q.all: rows = _distinct(rows)
    else:
        cb = {}
        for r in rb:
            k = tuple(nkey(v) for v in r); cb[k] = cb.get(k, 0) + 1
        rows = []
        if op == "intersect":
            if q.all:
                for r in ra:
                    k = tuple(nkey(v) for v in r)
                    if cb.get(k, 0) > 0: cb[k] -= 1; rows.append(r)
            else: rows = [r for r in _distinct(ra) if tuple(nkey(v) for v in r) in cb]
        elif op == "except":
            if q.all:
                for r in ra:
                    k = tuple(nkey(v) for v in r)
                    if cb.get(k, 0) > 0: cb[k] -= 1
                    else: rows.append(r)
            else: rows = [r for r in _distinct(ra) if tuple(nkey(v) for v in r) not in cb]
        else: raise SqlError(f"set operation {op} is not supported")
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
    vq(q)
    return list(found.values())


def run_select(s, cx):
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
    return finish_select(s, rel, aggs, wins, aggplan, cx)


# ------------------------------------------------------------------ group-level evaluation
def finish_select(s, rel, aggs, wins, aggplan, cx):
    names = [n for _, n in s.items]; types = [x.t for x, _ in s.items]
    if s.aggregate:
        out = aggregate(s, rel, aggs, aggplan, cx)
    else:
        # one output row per underlying row: expand the groups' multiplicities
        if any(c != 1 for c in rel.cnt):
            idx = [i for i in range(rel.n) for _ in range(rel.cnt[i])]
            rel = rel.take(idx)
        out = rel
    if wins: compute_windows(list(wins.values()), out, cx)
    if s.qualify is not None:
        out = out.take(truth(ev(s.qualify, out, cx)))
    cols = [ev(x, out, cx) for x, _ in s.items]
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
            seen, keep = set(), []
            kf = [T.coll_key(t) for t in types]
            for i, r in enumerate(rows):
                k = tuple(nkey(f(v) if f is not None and isinstance(v, str) else v) for v, f in zip(r, kf))
                if k not in seen: seen.add(k); keep.append(i)
            out = out.take(keep); rows = [rows[i] for i in keep]
    rows = _order_limit(rows, out, s.order, s, cx, None)
    return Relation(names, types, rows)


def aggregate(s, rel, aggs, aggplan, cx):
    """GROUP BY (with grouping sets) and every aggregate, from the rows of rel (each standing for cnt rows)"""
    gvals = [ev(g, rel, cx) for g in s.group]
    for col in gvals:
        for v in col:
            if isinstance(v, Err): raise SqlError(v.msg)
    sets = s.sets if s.sets is not None else [list(range(len(s.group)))]
    gkeys = [_ckeys(v, g.t) for v, g in zip(gvals, s.group)]      # collated: grouped by key, the least value shown
    collated = [gk is not gv for gk, gv in zip(gkeys, gvals)]
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
        for k in order: gm[k] = (tuple(gm[k][0]), gm[k][1])
        for k in order:
            vals, members = gm[k]
            out_groups.append((si, vals, members))
    n = len(out_groups)
    out = Rel(n)
    for j, g in enumerate(s.group):
        col = []
        for si, vals, members in out_groups:
            st = sets[si]
            col.append(vals[st.index(j)] if j in st else None)
        out.cols[g.key()] = col
    # GROUPING(): bit i set when argument i is not grouped in the row's set (first argument most significant)
    for gx in collect([x for x, _ in s.items] + [s.having, s.qualify] + [o[0][1] for o in s.order if o[0][0] == "expr"], I.Grouping).values():
        pos = []
        for a in gx.args:
            k = a.key(); p = next((j for j, g in enumerate(s.group) if g.key() == k), None)
            if p is None: raise SqlError("GROUPING() arguments must be GROUP BY expressions")
            pos.append(p)
        col = []
        for si, _, _ in out_groups:
            v = 0
            for p in pos: v = (v << 1) | (0 if p in sets[si] else 1)
            col.append(v)
        out.cols[gx.key()] = col
    members = [m for _, _, m in out_groups]
    for k, a in aggs.items():
        out.cols[k] = agg_values(a, rel, members, aggplan.get(k), cx)
    if s.having is not None:
        out = out.take(truth(ev(s.having, out, cx)))
    return out


def _xsum(vals):
    s = 0
    for v in vals:
        if isinstance(v, float) or isinstance(s, float): s = float(s) + float(v)
        else: s = s + v
    return s


def agg_values(a, rel, members, plan, cx):
    """one aggregate for every output group (members: the rel rows of each group)"""
    if plan is not None and plan.get("kind") == "state":
        k = a.key()
        if a.name in ("min", "max"):                 # the engine's extremes per row
            col = rel.cols[("x", k)]; j = 0 if a.name == "min" else 1
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
        out = []
        for mem in members:
            n = sum(cnt[i] for i in mem)
            sums = [_sum_exact(rel.cols[("m", k, j)][i] for i in mem) for j in range(nm)]
            out.append(_moment_final(a.name, n, sums))
        return out
    d = F.AGGS[a.name]
    args = [ev(x, rel, cx) for x in a.args]
    filt = truth(ev(a.filter, rel, cx)) if a.filter is not None else None
    fset = set(filt) if filt is not None else None
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


def moment_values(a, rel, members, plan):
    """an aggregate from the engine's per-group moments (exact sums of polynomials) and counts"""
    v = plan["variant"]; ms = plan["moments"]
    cnt = rel.vcnt.get(v, rel.cnt) if v != 0 else rel.cnt
    filt = plan.get("rowfilter")
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
def compute_windows(wins, rel, cx):
    n = rel.n
    for w in wins:
        if w.key() in rel.cols: continue
        parts = [_ckeys(ev(p, rel, cx), p.t) for p in w.partition]
        okeys = [_ckeys(ev(o[0], rel, cx), o[0].t) for o in w.order]
        args = [ev(a, rel, cx) for a in w.args]
        filt = set(truth(ev(w.filter, rel, cx))) if w.filter is not None else None
        for col in parts + okeys + args:
            for v in col:
                if isinstance(v, Err): raise SqlError(v.msg)
        groups = {}
        for i in range(n):
            groups.setdefault(tuple(nkey(p[i]) for p in parts), []).append(i)
        res = [None] * n
        specs = [(o[1], o[2]) for o in w.order]
        for idx in groups.values():
            if w.order: idx = sort_rows(idx, okeys, specs)
            _window_partition(w, idx, okeys, specs, args, filt, res, rel, cx)
        rel.cols[w.key()] = res


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


def _window_partition(w, idx, okeys, specs, args, filt, res, rel, cx):
    m = len(idx); name = w.name
    pstart, pend = _peer_bounds(idx, okeys) if w.order else ([0] * m, [m] * m)
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
    bound_vals = {}
    for side, bd in (("a", a), ("b", b)):
        if bd[1] is not None: bound_vals[side] = ev(bd[1], rel, cx)
    for p, i in enumerate(idx):
        lo = _frame_edge(mode, a, "a", p, i, idx, okeys, specs, pstart, pend, bound_vals, m)
        hi = _frame_edge(mode, b, "b", p, i, idx, okeys, specs, pstart, pend, bound_vals, m)
        frame = list(range(max(lo, 0), min(hi, m)))
        if w.exclude:
            ex = w.exclude.upper().replace(" ", "_")
            if ex == "CURRENT_ROW": frame = [q for q in frame if q != p]
            elif ex == "GROUP": frame = [q for q in frame if not (pstart[p] <= q < pend[p])]
            elif ex == "TIES": frame = [q for q in frame if q == p or not (pstart[p] <= q < pend[p])]
        if name in ("first_value", "last_value", "nth_value"):
            vals = args[0]
            cand = [idx[q] for q in frame]
            if w.ignore_nulls: cand = [c for c in cand if vals[c] is not None]
            if name == "first_value": res[i] = vals[cand[0]] if cand else None
            elif name == "last_value": res[i] = vals[cand[-1]] if cand else None
            else:
                k = args[1][i]
                if k is None: res[i] = None; continue
                if k <= 0: raise SqlError("NTH_VALUE: the position must be positive")
                if w.from_last: cand = cand[::-1]
                res[i] = vals[cand[k - 1]] if k <= len(cand) else None
            continue
        d = F.AGGS[name]
        rows = []
        for q in frame:
            r = idx[q]
            if filt is not None and r not in filt: continue
            vals = tuple(c[r] for c in args)
            if d.skip_nulls and name != "count_star" and any(v is None for v in vals): continue
            rows.append((vals, 1))
        if w.distinct:
            seen, dd = set(), []
            for r in rows:
                k = tuple(nkey(v) for v in r[0])
                if k not in seen: seen.add(k); dd.append(r)
            rows = dd
        res[i] = len(rows) if name == "count_star" else d.final(rows, w.params)


def _frame_edge(mode, bd, side, p, i, idx, okeys, specs, pstart, pend, bound_vals, m):
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
        gstarts = sorted(set(pstart)); gi = gstarts.index(pstart[p])
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
    if side == "a":
        q = 0
        while q < m and (key[idx[q]] is None or (F.cmp_values(key[idx[q]], tgt) < 0 if not desc else F.cmp_values(key[idx[q]], tgt) > 0)):
            if key[idx[q]] is None and not specs[0][1]: break
            q += 1
        return q
    q = 0
    while q < m and key[idx[q]] is not None and (F.cmp_values(key[idx[q]], tgt) <= 0 if not desc else F.cmp_values(key[idx[q]], tgt) >= 0):
        q += 1
    if specs[0][1]:                                  # NULLS FIRST: the NULL peers precede
        while q < m and key[idx[q]] is None: q += 1
    return q


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
        elif isinstance(q, I.TableFn):
            for a in q.args: I.walk(a, vx)
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
            for a, b in zip(xs if isinstance(xs, list) else [xs], v):
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
        if mx >= 1 << 22: return None
        codes = np.arange(mx + 1, dtype=np.int64)
        vals = cd.decode(codes) if cd is not None else codes.tolist()
        r = Rel(len(vals))
        for x in I.cols_of(e): r.cols[x.key()] = vals
        res = ev(e, r, self.cx)
        for x in res:
            if isinstance(x, Err): raise SqlError(x.msg)
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
                self.nulled = {b}
                tfs = [self.native_tf(c) for c in cf]
                self.nulled = set()
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
        if nkeys > algq.MAXKEY or self.fact_sample is not None:
            # more columns than GROUP BY keys (or a sampled fact): the passing rows themselves (a row bitmap), their
            # columns read from the planes; every aggregate is then computed from its arguments' values
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
                for k, a in aggs.items():
                    for x in a.args + [o[0] for o in a.order] + ([a.filter] if a.filter is not None else []):
                        for col in I.cols_of(x):
                            (nulled_atoms if col.b in self.nulled else atoms)[col.key()] = col
                aggplan = {}
                rel = self.rows_rel(atoms, preds, empty)
        for k, col in nulled_atoms.items(): rel.cols[k] = [None] * rel.n
        for k, plan in aggplan.items():
            if plan.get("rowfilter_expr") is not None:
                plan["rowfilter"] = set(truth(ev(plan["rowfilter_expr"], rel, cx)))
        if residual:
            keep = truth(ev(_and(residual), rel, cx))
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
            codes = codes.tolist()                       # Python integers: 64-bit codes past 2^63 stay exact
            vals = cd.decode(algq._codes_array(codes)) if cd is not None else codes
            vc = getattr(t, "nullable", {}).get(name)
            if vc is not None:
                ok = fact_codes(vc) if b is self.fact else t.decoded(vc)[rows]
                vals = [x if o else None for x, o in zip(vals, ok.tolist())]
            rel.cols[k] = vals
        return rel

    def engine(self, keys, atom_list, preds, empty, variants, vpreds, mlist):
        """the engine queries: per variant, COUNT and the sum of each moment polynomial, grouped by the atoms"""
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
                if "group domain too large" in str(ex): raise _TooManyGroups()
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
        rel = rel.take(truth(ev(_and(rest), rel, cx)))
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
    kind = j.kind
    conds = _conjuncts(j.cond)
    lkeys, rkeys, other = [], [], []
    lset, rset = set(L.cols), set(R.cols)
    for c in conds:
        if isinstance(c, I.Func) and c.fn in ("=", "is_not_distinct") and len(c.args) == 2:
            a, b = c.args
            ka, kb = {x.key() for x in I.cols_of(a)}, {x.key() for x in I.cols_of(b)}
            if ka and kb and ka <= lset and kb <= rset: lkeys.append(a); rkeys.append(b); continue
            if ka and kb and ka <= rset and kb <= lset: lkeys.append(b); rkeys.append(a); continue
        other.append(c)
    pairs = []
    if lkeys:
        lv = [ev(x, L, cx) for x in lkeys]; rv = [ev(x, R, cx) for x in rkeys]
        h = {}
        for i in range(R.n):
            k = tuple(nkey(c[i]) for c in rv)
            if any(c[i] is None for c in rv): continue
            h.setdefault(k, []).append(i)
        for i in range(L.n):
            if any(c[i] is None for c in lv): continue
            for jx in h.get(tuple(nkey(c[i]) for c in lv), ()): pairs.append((i, jx))
    else:
        pairs = [(i, jx) for i in range(L.n) for jx in range(R.n)]
    C = _combine(L, R, pairs)
    if other:
        keep = truth(ev(_and(other), C, cx))
        C = C.take(keep); pairs = [pairs[k] for k in keep]
    if kind in ("inner", "cross"): return C
    if kind in ("semi", "anti"):
        hit = {i for i, _ in pairs}
        idx = [i for i in range(L.n) if (i in hit) == (kind == "semi")]
        return L.take(idx)
    extra_l = [i for i in range(L.n) if i not in {p[0] for p in pairs}] if kind in ("left", "full") else []
    extra_r = [i for i in range(R.n) if i not in {p[1] for p in pairs}] if kind in ("right", "full") else []
    parts = [C]
    if extra_l: parts.append(_pad(L.take(extra_l), R))
    if extra_r: parts.append(_pad(R.take(extra_r), L))
    return _concat(parts)


def _combine(L, R, pairs):
    C = Rel(len(pairs))
    li = [p[0] for p in pairs]; ri = [p[1] for p in pairs]
    for k, v in L.cols.items(): C.cols[k] = [v[i] for i in li]
    for k, v in R.cols.items(): C.cols[k] = [v[i] for i in ri]
    C.cnt = [L.cnt[a] * R.cnt[b] for a, b in pairs]
    return C


def _pad(A, other):
    for k in other.cols:
        if k not in A.cols: A.cols[k] = [None] * A.n
    return A


def _concat(parts):
    n = sum(p.n for p in parts)
    C = Rel(n)
    keys = set()
    for p in parts: keys |= set(p.cols)
    for k in keys:
        out = []
        for p in parts: out += p.cols.get(k, [None] * p.n)
        C.cols[k] = out
    C.cnt = [c for p in parts for c in p.cnt]
    return C
