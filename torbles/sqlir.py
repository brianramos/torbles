# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The bound form of a query, shared by the SQL text front end (sqlfront) and the Python API (sql.py): typed
expression trees and query blocks. Both front ends build these; the executor (sqlexec) plans them onto the engine."""
import sqltypes as T


class Node:
    """an expression; .t is its SQL type, .key() a hashable structural identity (equal trees, equal keys)"""
    __slots__ = ("t", "_k")
    nullable = True
    def key(self):
        try: return self._k
        except AttributeError: pass
        self._k = k = self._key()
        return k
    def children(self): return ()
    def __repr__(self): return render(self)


class Lit(Node):
    """a constant. strlit: written as a string literal (it converts implicitly to the type it meets)"""
    __slots__ = ("v", "strlit")
    def __init__(self, v, t, strlit=False): self.v, self.t, self.strlit = v, t, strlit
    @property
    def nullable(self): return self.v is None
    def _key(self): return ("lit", type(self.v).__name__, self.v if not isinstance(self.v, list) else tuple(self.v), str(self.t))


class Col(Node):
    """a column of a FROM item (binding): a stored column, or a column of a derived table / CTE"""
    __slots__ = ("b", "name", "_nullable")
    def __init__(self, b, name, t, nullable=False): self.b, self.name, self.t, self._nullable = b, name, t, nullable
    @property
    def nullable(self): return self._nullable or getattr(self.b, "outer_null", False)
    def _key(self): return ("col", id(self.b), self.name)


class Outer(Node):
    """a reference to a column of an enclosing query (a correlated subquery); depth = scopes outward"""
    __slots__ = ("col", "depth")
    def __init__(self, col, depth): self.col, self.depth, self.t = col, depth, col.t
    nullable = True
    def _key(self): return ("outer", self.col.key(), self.depth)


class Func(Node):
    """a scalar function or operator; fn: the registry entry name (sqlfuncs.SCALARS)"""
    __slots__ = ("fn", "args", "extra")
    def __init__(self, fn, args, t, extra=None): self.fn, self.args, self.t, self.extra = fn, list(args), t, extra
    def children(self): return self.args
    @property
    def nullable(self):
        import sqlfuncs
        return sqlfuncs.nullable(self)
    def _key(self): return ("fn", self.fn, tuple(a.key() for a in self.args), _hk(self.extra))


class Case(Node):
    __slots__ = ("whens", "else_")
    def __init__(self, whens, else_, t): self.whens, self.else_, self.t = [(c, v) for c, v in whens], else_, t
    def children(self): return [x for w in self.whens for x in w] + ([self.else_] if self.else_ is not None else [])
    @property
    def nullable(self): return self.else_ is None or self.else_.nullable or any(v.nullable for _, v in self.whens)
    def _key(self): return ("case", tuple((c.key(), v.key()) for c, v in self.whens), None if self.else_ is None else self.else_.key())


class Cast(Node):
    __slots__ = ("arg", "try_")
    def __init__(self, arg, t, try_=False): self.arg, self.t, self.try_ = arg, t, try_
    def children(self): return [self.arg]
    @property
    def nullable(self): return self.arg.nullable or self.try_
    def _key(self): return ("cast", self.arg.key(), str(self.t), self.try_)


class Agg(Node):
    """an aggregate function: name (sqlfuncs.AGGS), args, DISTINCT, FILTER (WHERE ...), ORDER BY inside the call
    [(expr, desc, nulls_first)], params: constant arguments (a separator, a fraction)"""
    __slots__ = ("name", "args", "distinct", "filter", "order", "params")
    def __init__(self, name, args, t, distinct=False, filter=None, order=(), params=()):
        self.name, self.args, self.t, self.distinct, self.filter = name, list(args), t, distinct, filter
        self.order, self.params = list(order), tuple(params)
    def children(self): return self.args + ([self.filter] if self.filter is not None else []) + [o[0] for o in self.order]
    def _key(self):
        return ("agg", self.name, tuple(a.key() for a in self.args), self.distinct, None if self.filter is None else self.filter.key(),
                tuple((o[0].key(), o[1], o[2]) for o in self.order), _hk(self.params))


class Win(Node):
    """a window function: name, args, PARTITION BY, ORDER BY, frame (mode ROWS/RANGE/GROUPS, start, end: (kind, expr)),
    exclude, ignore_nulls, filter, from_last (NTH_VALUE FROM LAST), distinct"""
    __slots__ = ("name", "args", "partition", "order", "frame", "exclude", "ignore_nulls", "filter", "from_last", "distinct", "params")
    def __init__(self, name, args, t, partition=(), order=(), frame=None, exclude=None, ignore_nulls=False, filter=None,
                 from_last=False, distinct=False, params=()):
        self.name, self.args, self.t = name, list(args), t
        self.partition, self.order, self.frame, self.exclude = list(partition), list(order), frame, exclude
        self.ignore_nulls, self.filter, self.from_last, self.distinct, self.params = ignore_nulls, filter, from_last, distinct, tuple(params)
    def children(self):
        out = self.args + self.partition + [o[0] for o in self.order] + ([self.filter] if self.filter is not None else [])
        if self.frame:
            for b in (self.frame[1], self.frame[2]):
                if b[1] is not None: out.append(b[1])
        return out
    def _key(self):
        fr = None if self.frame is None else (self.frame[0], (self.frame[1][0], None if self.frame[1][1] is None else self.frame[1][1].key()),
                                               (self.frame[2][0], None if self.frame[2][1] is None else self.frame[2][1].key()))
        return ("win", self.name, tuple(a.key() for a in self.args), tuple(p.key() for p in self.partition),
                tuple((o[0].key(), o[1], o[2]) for o in self.order), fr, self.exclude, self.ignore_nulls,
                None if self.filter is None else self.filter.key(), self.from_last, self.distinct, _hk(self.params))


class Subq(Node):
    """a subquery: kind scalar | exists | any | all | unique; for any / all: child <op> ANY / ALL (query)"""
    __slots__ = ("kind", "q", "child", "op", "correlated")
    def __init__(self, kind, q, t, child=None, op=None, correlated=False):
        self.kind, self.q, self.t, self.child, self.op, self.correlated = kind, q, t, child, op, correlated
    def children(self): return [self.child] if self.child is not None else []
    def _key(self): return ("subq", self.kind, id(self.q), None if self.child is None else self.child.key(), self.op)


class Grouping(Node):
    """GROUPING(a, b, ...): which arguments are rolled up in the current grouping set (bits, first argument highest)"""
    __slots__ = ("args",)
    def __init__(self, args): self.args, self.t = list(args), T.BIGINT
    nullable = False
    def children(self): return self.args
    def _key(self): return ("grouping", tuple(a.key() for a in self.args))


def _hk(x):
    if isinstance(x, (list, tuple)): return tuple(_hk(y) for y in x)
    if isinstance(x, dict): return tuple(sorted((k, _hk(v)) for k, v in x.items()))
    if isinstance(x, Node): return x.key()
    return x


def walk(e, fn):
    """call fn on every node of an expression tree (not inside subqueries)"""
    fn(e)
    for c in e.children(): walk(c, fn)


def cols_of(e):
    """the columns an expression reads (not inside subqueries; outer references excluded)"""
    out = {}
    def f(x):
        if isinstance(x, Col): out[x.key()] = x
    walk(e, f)
    return list(out.values())


def has(e, cls):
    found = []
    def f(x):
        if isinstance(x, cls): found.append(x)
    walk(e, f)
    return bool(found)


def find(e, cls, stop=()):
    """nodes of class cls in e, outermost first, not descending into nodes of classes stop"""
    out = []
    def go(x):
        if isinstance(x, cls): out.append(x); return
        if isinstance(x, stop): return
        for c in x.children(): go(c)
    go(e)
    return out


def subst(e, fn):
    """rebuild e bottom-up: fn(node) returns a replacement or None (keep, with rebuilt children)"""
    r = fn(e)
    if r is not None: return r
    if isinstance(e, Func): return Func(e.fn, [subst(a, fn) for a in e.args], e.t, e.extra)
    if isinstance(e, Case): return Case([(subst(c, fn), subst(v, fn)) for c, v in e.whens], None if e.else_ is None else subst(e.else_, fn), e.t)
    if isinstance(e, Cast): return Cast(subst(e.arg, fn), e.t, e.try_)
    if isinstance(e, Agg):
        return Agg(e.name, [subst(a, fn) for a in e.args], e.t, e.distinct, None if e.filter is None else subst(e.filter, fn),
                   [(subst(o[0], fn), o[1], o[2]) for o in e.order], e.params)
    if isinstance(e, Win):
        fr = e.frame
        if fr is not None:
            fr = (fr[0], (fr[1][0], None if fr[1][1] is None else subst(fr[1][1], fn)), (fr[2][0], None if fr[2][1] is None else subst(fr[2][1], fn)))
        return Win(e.name, [subst(a, fn) for a in e.args], e.t, [subst(p, fn) for p in e.partition],
                   [(subst(o[0], fn), o[1], o[2]) for o in e.order], fr, e.exclude, e.ignore_nulls,
                   None if e.filter is None else subst(e.filter, fn), e.from_last, e.distinct, e.params)
    if isinstance(e, Subq): return Subq(e.kind, e.q, e.t, None if e.child is None else subst(e.child, fn), e.op, e.correlated)
    if isinstance(e, Grouping): return Grouping([subst(a, fn) for a in e.args])
    return e


_OPS = {"+": "+", "-": "-", "*": "*", "/": "/", "//": "//", "%": "%", "||": "||", "=": "=", "<>": "<>", "<": "<", "<=": "<=",
        ">": ">", ">=": ">="}


def render(e):
    """an expression as SQL-like text (result column names, messages)"""
    if isinstance(e, Lit):
        if e.v is None: return "NULL"
        if isinstance(e.v, str): return "'" + e.v.replace("'", "''") + "'"
        return T.to_text(e.v, e.t)
    if isinstance(e, Col): return e.name
    if isinstance(e, Outer): return render(e.col)
    if isinstance(e, Func):
        if e.fn in _OPS and len(e.args) == 2: return f"({render(e.args[0])} {_OPS[e.fn]} {render(e.args[1])})"
        if e.fn == "neg": return f"-({render(e.args[0])})"
        if e.fn in ("and", "or"): return "(" + f" {e.fn.upper()} ".join(render(a) for a in e.args) + ")"
        if e.fn == "not": return f"(NOT {render(e.args[0])})"
        return f"{e.fn}({', '.join(render(a) for a in e.args)})"
    if isinstance(e, Case):
        return "CASE " + " ".join(f"WHEN {render(c)} THEN {render(v)}" for c, v in e.whens) + (f" ELSE {render(e.else_)}" if e.else_ is not None else "") + " END"
    if isinstance(e, Cast): return f"CAST({render(e.arg)} AS {e.t})"
    if isinstance(e, Agg):
        a = ", ".join(render(x) for x in e.args) if e.args else ("*" if e.name == "count_star" else "")
        name = "count" if e.name == "count_star" else e.name
        return f"{name}({'DISTINCT ' if e.distinct else ''}{a})" + (f" FILTER (WHERE {render(e.filter)})" if e.filter is not None else "")
    if isinstance(e, Win): return f"{e.name}({', '.join(render(a) for a in e.args)}) OVER (...)"
    if isinstance(e, Subq): return f"({e.kind} subquery)"
    if isinstance(e, Grouping): return f"GROUPING({', '.join(render(a) for a in e.args)})"
    return "?"


# ------------------------------------------------------------------ query blocks
class Binding:
    """a FROM item: a stored table (table = the torbles Table), or a derived table / CTE / VALUES (q = its query);
    cols: [(name, SqlType, nullable)]; outer_null: its columns can be NULL through an outer join"""
    def __init__(self, alias, cols, table=None, q=None, rows=None):
        self.alias, self.cols, self.table, self.q, self.rows = alias, list(cols), table, q, rows
        self.outer_null = False
    def col(self, name):
        for c, t, nl in self.cols:
            if c == name: return Col(self, c, t, nl)
        return None
    def __repr__(self): return f"Binding({self.alias})"


class Join:
    """a join in the FROM tree: kind inner | left | right | full | cross; cond: the ON condition (USING / NATURAL
    become equalities)"""
    def __init__(self, kind, left, right, cond=None): self.kind, self.left, self.right, self.cond = kind, left, right, cond


class Select:
    """one SELECT block"""
    def __init__(self):
        self.from_ = None                     # a Binding, a Join tree, or None (no FROM)
        self.where = None
        self.group = []                       # GROUP BY expressions (distinct)
        self.sets = None                      # grouping sets: lists of indexes into group (None: one set, all of them)
        self.aggregate = False                # an aggregate query (GROUP BY, an aggregate, or HAVING)
        self.having = None
        self.items = []                       # [(expr, name)]
        self.qualify = None
        self.distinct = False
        self.distinct_on = None
        self.order = []                       # [(expr, desc, nulls_first)]
        self.limit = None; self.offset = None; self.with_ties = False; self.percent = False


class SetOp:
    def __init__(self, op, all_, left, right):
        self.op, self.all, self.left, self.right = op, all_, left, right
        self.order, self.limit, self.offset, self.with_ties, self.percent = [], None, None, False, False
        self.items = None                     # output columns: [(name, type)]


class Values:
    def __init__(self, rows, types, names): self.rows, self.types, self.names = rows, types, names


class Recursive:
    """WITH RECURSIVE: base UNION [ALL] step, step reading the binding itself"""
    def __init__(self, binding, base, step, all_): self.binding, self.base, self.step, self.all = binding, base, step, all_
