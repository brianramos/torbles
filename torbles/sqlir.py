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
    def _key(self): return ("lit", type(self.v).__name__, _vkey(self.v), str(self.t))


def _vkey(v):
    """a hashable key of a value (arrays and rows nest; 1, 1.0 and TRUE stay apart; NaN equals NaN)"""
    if isinstance(v, list): return ("\0list",) + tuple(_vkey(x) for x in v)
    if isinstance(v, tuple): return ("\0row",) + tuple(_vkey(x) for x in v)
    if isinstance(v, float) and v != v: return ("\0nan",)
    if isinstance(v, dict): return ("\0dict",) + tuple(sorted((str(k), _vkey(x)) for k, x in v.items()))
    return (type(v).__name__, v)


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
    __slots__ = ("name", "args", "partition", "order", "frame", "exclude", "ignore_nulls", "filter", "from_last", "distinct", "params", "rpr")
    def __init__(self, name, args, t, partition=(), order=(), frame=None, exclude=None, ignore_nulls=False, filter=None,
                 from_last=False, distinct=False, params=(), rpr=None):
        self.name, self.args, self.t = name, list(args), t
        self.partition, self.order, self.frame, self.exclude = list(partition), list(order), frame, exclude
        self.ignore_nulls, self.filter, self.from_last, self.distinct, self.params = ignore_nulls, filter, from_last, distinct, tuple(params)
        self.rpr = rpr                      # row pattern recognition in the window (R020): sqlexec runs it
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
                None if self.filter is None else self.filter.key(), self.from_last, self.distinct, _hk(self.params),
                None if self.rpr is None else self.rpr.key())


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
                   None if e.filter is None else subst(e.filter, fn), e.from_last, e.distinct, e.params, e.rpr)
    if isinstance(e, Subq): return Subq(e.kind, e.q, e.t, None if e.child is None else subst(e.child, fn), e.op, e.correlated)
    if isinstance(e, Grouping): return Grouping([subst(a, fn) for a in e.args])
    return e


def subst_post(e, fn):
    """rebuild e bottom-up: children first, then fn(rebuilt node) gives the node to use"""
    if isinstance(e, Func): x = Func(e.fn, [subst_post(a, fn) for a in e.args], e.t, e.extra)
    elif isinstance(e, Case): x = Case([(subst_post(c, fn), subst_post(v, fn)) for c, v in e.whens], None if e.else_ is None else subst_post(e.else_, fn), e.t)
    elif isinstance(e, Cast): x = Cast(subst_post(e.arg, fn), e.t, e.try_)
    else: x = e
    return fn(x)


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
        self.sample = None                     # TABLESAMPLE: a Sample
    def col(self, name):
        for c, t, nl in self.cols:
            if c == name: return Col(self, c, t, nl)
        return None
    def __repr__(self): return f"Binding({self.alias})"


class Sample:
    """TABLESAMPLE BERNOULLI | SYSTEM (p) [REPEATABLE (seed)] (ISO 7.6), or a number of rows (DuckDB's USING SAMPLE n
    ROWS / RESERVOIR): method bernoulli | system | reservoir; pct or rows; seed None when not repeatable"""
    def __init__(self, method, pct=None, rows=None, seed=None):
        self.method, self.pct, self.rows, self.seed = method, pct, rows, seed


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
        self.sample = None                    # USING SAMPLE (DuckDB): on the FROM result, before WHERE


class SetOp:
    def __init__(self, op, all_, left, right):
        self.op, self.all, self.left, self.right = op, all_, left, right
        self.order, self.limit, self.offset, self.with_ties, self.percent = [], None, None, False, False
        self.items = None                     # output columns: [(name, type)]


class Values:
    def __init__(self, rows, types, names): self.rows, self.types, self.names = rows, types, names


class TableFn:
    """a table function in FROM: UNNEST(collection, ...) [WITH ORDINALITY], range / generate_series; its arguments
    may read columns of the FROM items before it (a lateral reference)"""
    def __init__(self, name, args, names, types, ordinality=False):
        self.name, self.args, self.names, self.types, self.ordinality = name, args, names, types, ordinality


class Recursive:
    """WITH RECURSIVE: base UNION [ALL] step, step reading the binding itself"""
    def __init__(self, binding, base, step, all_): self.binding, self.base, self.step, self.all = binding, base, step, all_


# ------------------------------------------------------------------ SQL text of a bound query (Python API: to_sql)
def _qi(name): return '"' + str(name).replace('"', '""') + '"'


def _lit_sql(v, t):
    import datetime as dt
    from decimal import Decimal
    if v is None: return "NULL" if t is None or t.id == "NULL" else f"CAST(NULL AS {t})"
    if isinstance(v, bool): return "TRUE" if v else "FALSE"
    if isinstance(v, int): return str(v) if -(2**31) <= v < 2**31 else f"CAST('{v}' AS {t if t is not None and t.integer else 'HUGEINT'})"
    if isinstance(v, Decimal): return format(v, "f") if t is None or t.id != "DECIMAL" else f"CAST('{format(v, 'f')}' AS {t})"
    if isinstance(v, float): return f"CAST('{T.fmt_double(v)}' AS DOUBLE)"
    if isinstance(v, str): return "'" + v.replace("'", "''") + "'"
    if isinstance(v, dt.datetime): return f"TIMESTAMP '{T.to_text(v, T.TIMESTAMP)}'"
    if isinstance(v, dt.date): return f"DATE '{v.isoformat()}'"
    if isinstance(v, dt.time): return f"TIME '{T.to_text(v, T.TIME)}'"
    if isinstance(v, T.Interval): return f"INTERVAL '{v}'"
    if isinstance(v, list): return "[" + ", ".join(_lit_sql(x, t.child if t is not None and t.id == "LIST" else None) for x in v) + "]"
    raise T.SqlError(f"no SQL literal for {v!r}")


_INFIX = {"+": "+", "-": "-", "*": "*", "/": "/", "//": "//", "%": "%", "||": "||", "=": "=", "<>": "<>", "<": "<",
          "<=": "<=", ">": ">", ">=": ">=", "is_distinct": "IS DISTINCT FROM", "is_not_distinct": "IS NOT DISTINCT FROM",
          "&": "&", "|": "|", "<<": "<<", ">>": ">>"}
_POSTFIX = {"is_null": "IS NULL", "is_not_null": "IS NOT NULL", "is_true": "IS TRUE", "is_false": "IS FALSE",
            "is_unknown": "IS UNKNOWN", "is_not_true": "IS NOT TRUE", "is_not_false": "IS NOT FALSE", "is_not_unknown": "IS NOT UNKNOWN"}


def expr_sql(e):
    if isinstance(e, Lit): return _lit_sql(e.v, e.t)
    if isinstance(e, Col): return f"{_qi(e.b.alias)}.{_qi(e.name)}"
    if isinstance(e, Outer): return expr_sql(e.col)
    if isinstance(e, Func):
        a = [expr_sql(x) for x in e.args]; fn = e.fn
        if fn in _INFIX and len(a) == 2: return f"({a[0]} {_INFIX[fn]} {a[1]})"
        if fn in _POSTFIX: return f"({a[0]} {_POSTFIX[fn]})"
        if fn == "neg": return f"(- {a[0]})"
        if fn == "__bitnot": return f"(~ {a[0]})"
        if fn in ("and", "or"): return "(" + f" {fn.upper()} ".join(a) + ")"
        if fn == "not": return f"(NOT {a[0]})"
        if fn == "between": return f"({a[0]} BETWEEN {a[1]} AND {a[2]})"
        if fn == "between_sym": return f"({a[0]} BETWEEN SYMMETRIC {a[1]} AND {a[2]})"
        if fn == "in": return f"({a[0]} IN ({', '.join(a[1:])}))"
        if fn in ("like", "ilike"): return f"({a[0]} {fn.upper()} {a[1]}" + (f" ESCAPE {a[2]}" if len(a) > 2 else "") + ")"
        if fn == "glob": return f"({a[0]} GLOB {a[1]})"
        if fn == "__iso_similar": return f"({a[0]} SIMILAR TO {a[1]}" + (f" ESCAPE {a[2]}" if len(a) > 2 else "") + ")"
        if fn == "__iso_like_regex": return f"({a[0]} LIKE_REGEX {a[1]}" + (f" FLAG {a[2]}" if len(a) > 2 else "") + ")"
        if fn == "__iso_overlaps": return f"(({a[0]}, {a[1]}) OVERLAPS ({a[2]}, {a[3]}))"
        if fn == "__nwf_row_number": return f"ROW_NUMBER({e.extra[0]})"
        if fn == "__nwf_value_of":
            d = e.args[1].v or 0
            return f"VALUE_OF({a[0]} AT {e.extra[0]}{f' + {d}' if d > 0 else f' - {-d}' if d < 0 else ''}, {a[2]})"
        if fn in ("__first", "__grp"): return a[0]
        if fn == "list_value": return "[" + ", ".join(a) + "]"
        if fn == "row": return "(" + ", ".join(a) + ")"
        return f"{fn}({', '.join(a)})"
    if isinstance(e, Case):
        out = "CASE " + " ".join(f"WHEN {expr_sql(c)} THEN {expr_sql(v)}" for c, v in e.whens)
        return out + (f" ELSE {expr_sql(e.else_)}" if e.else_ is not None else "") + " END"
    if isinstance(e, Cast): return f"{'TRY_CAST' if e.try_ else 'CAST'}({expr_sql(e.arg)} AS {e.t})"
    if isinstance(e, Agg): return _agg_sql(e)
    if isinstance(e, Win): return _win_sql(e)
    if isinstance(e, Subq):
        q = to_sql(e.q)
        if e.kind == "scalar": return f"({q})"
        if e.kind == "exists": return f"(EXISTS ({q}))"
        if e.kind == "unique": return f"(UNIQUE ({q}))"
        return f"({expr_sql(e.child)} {e.op} ANY ({q}))"
    if isinstance(e, Grouping): return f"GROUPING({', '.join(expr_sql(x) for x in e.args)})"
    raise T.SqlError(f"cannot render {e!r}")


def _order_sql(order):
    return ", ".join(f"{expr_sql(x) if isinstance(x, Node) else x} {'DESC' if d else 'ASC'} NULLS {'FIRST' if nf else 'LAST'}" for x, d, nf in order)


def _agg_sql(e):
    a = [expr_sql(x) for x in e.args]
    filt = f" FILTER (WHERE {expr_sql(e.filter)})" if e.filter is not None else ""
    dist = "DISTINCT " if e.distinct else ""
    ob = f" ORDER BY {_order_sql(e.order)}" if e.order else ""
    if e.name == "count_star": return f"count(*){filt}"
    if e.name in ("quantile_cont", "quantile_disc"):
        q = e.params[0]; desc = len(e.params) > 1 and e.params[1] == "desc"
        return f"{e.name}({dist}{a[0]}, {_lit_sql(q, None) if not isinstance(q, list) else _lit_sql(q, None)}){filt}" if not desc else \
            f"percentile_disc({_lit_sql(q, None)}) WITHIN GROUP (ORDER BY {a[0]} DESC){filt}"
    if e.name == "mode" and e.params: return f"mode() WITHIN GROUP (ORDER BY {a[0]}{' DESC' if e.params[0] == 'desc' else ''}){filt}"
    if e.name == "string_agg": return f"string_agg({dist}{a[0]}, {_lit_sql(e.params[0], None)}{ob}){filt}"
    if e.name == "__iso_listagg": return f"LISTAGG({dist}{a[0]}, {_lit_sql(e.params[0], None)}) WITHIN GROUP (ORDER BY {_order_sql(e.order)}){filt}"
    if e.name.startswith("__iso_h"):
        return f"{e.name[7:].upper()}({', '.join(_lit_sql(p, None) for p in e.params)}) WITHIN GROUP (ORDER BY {_order_sql(e.order)}){filt}"
    return f"{e.name}({dist}{', '.join(a)}{ob}){filt}"


def _bound_sql(b):
    kind, x = b
    if kind == "unbounded_preceding": return "UNBOUNDED PRECEDING"
    if kind == "unbounded_following": return "UNBOUNDED FOLLOWING"
    if kind == "current": return "CURRENT ROW"
    return f"{expr_sql(x)} {'PRECEDING' if kind == 'preceding' else 'FOLLOWING'}"


def _win_sql(w):
    a = [expr_sql(x) for x in w.args]
    if w.name == "count_star": head = "count(*)"
    elif w.name in ("string_agg",): head = f"string_agg({a[0]}, {_lit_sql(w.params[0], None)})"
    elif w.name in ("quantile_cont", "quantile_disc"): head = f"{w.name}({a[0]}, {_lit_sql(w.params[0], None)})"
    else: head = f"{w.name}({'DISTINCT ' if w.distinct else ''}{', '.join(a)}{' IGNORE NULLS' if w.ignore_nulls else ''})"
    if w.from_last: head = head.replace("nth_value(", "nth_value(", 1) + " FROM LAST"
    if w.filter is not None: head += f" FILTER (WHERE {expr_sql(w.filter)})"
    parts = []
    if w.partition: parts.append("PARTITION BY " + ", ".join(expr_sql(p) for p in w.partition))
    if w.order: parts.append("ORDER BY " + _order_sql(w.order))
    if w.frame is not None and w.name not in ("row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag", "lead"):
        mode, s0, e0 = w.frame
        parts.append(f"{mode.upper()} BETWEEN {_bound_sql(s0)} AND {_bound_sql(e0)}")
        if w.exclude: parts.append("EXCLUDE " + w.exclude.replace("_", " ").replace("CURRENT ROW", "CURRENT ROW"))
    return f"{head} OVER ({' '.join(parts)})"


def _from_sql(f):
    if f is None: return ""
    if isinstance(f, Binding):
        if f.table is not None: return f"{_qi(getattr(f.table, 'sqlname', f.table.name))} AS {_qi(f.alias)}"
        if isinstance(f.q, Values):
            rows = ", ".join("(" + ", ".join(expr_sql(x) for x in r) + ")" for r in f.q.rows)
            return f"(VALUES {rows}) AS {_qi(f.alias)}({', '.join(_qi(c) for c, _, _ in f.cols)})"
        return f"({to_sql(f.q)}) AS {_qi(f.alias)}({', '.join(_qi(c) for c, _, _ in f.cols)})"
    l, r = _from_sql(f.left), _from_sql(f.right)
    if f.kind == "cross": return f"{l} CROSS JOIN {r}"
    kw = {"inner": "JOIN", "left": "LEFT JOIN", "right": "RIGHT JOIN", "full": "FULL JOIN", "semi": "SEMI JOIN", "anti": "ANTI JOIN"}[f.kind]
    return f"{l} {kw} {r} ON {expr_sql(f.cond) if f.cond is not None else 'TRUE'}"


def to_sql(q):
    """a bound query as SQL text (ISO forms where torbles has them; DuckDB runs the rest)"""
    if isinstance(q, Select):
        out = "SELECT " + ("DISTINCT " if q.distinct and not q.distinct_on else "")
        if q.distinct_on: out += "DISTINCT ON (" + ", ".join(expr_sql(d[1]) if d[0] == "expr" else str(d[1] + 1) for d in q.distinct_on) + ") "
        out += ", ".join(f"{expr_sql(x)} AS {_qi(n)}" for x, n in q.items)
        if q.from_ is not None: out += " FROM " + _from_sql(q.from_)
        if q.where is not None: out += " WHERE " + expr_sql(q.where)
        if q.group or q.sets is not None:
            g = [expr_sql(x) for x in q.group]
            if q.sets is None: out += " GROUP BY " + ", ".join(g)
            else: out += " GROUP BY GROUPING SETS (" + ", ".join("(" + ", ".join(g[i] for i in st) + ")" for st in q.sets) + ")"
        if q.having is not None: out += " HAVING " + expr_sql(q.having)
        if q.qualify is not None: out += " QUALIFY " + expr_sql(q.qualify)
    elif isinstance(q, SetOp):
        out = f"({to_sql(q.left)}) {q.op.upper()}{' ALL' if q.all else ''} ({to_sql(q.right)})"
    elif isinstance(q, Values):
        return "VALUES " + ", ".join("(" + ", ".join(expr_sql(x) for x in r) + ")" for r in q.rows)
    else: raise T.SqlError("this query cannot be rendered as SQL text")
    if q.order:
        out += " ORDER BY " + ", ".join(f"{(str(o[0][1] + 1) if o[0][0] == 'pos' else expr_sql(o[0][1]))} {'DESC' if o[1] else 'ASC'} NULLS {'FIRST' if o[2] else 'LAST'}" for o in q.order)
    cnt = lambda v: f"({expr_sql(v[1])})" if isinstance(v, tuple) else str(v)
    if q.limit is not None: out += f" LIMIT {cnt(q.limit)}" + (" PERCENT" if q.percent else "")
    if q.offset is not None: out += f" OFFSET {cnt(q.offset)}"
    return out
