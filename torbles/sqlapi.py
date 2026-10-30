# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The SQL front end as a Python API: the same bound queries (sqlir), the same planner and executor as db.sql().

    from torbles import F
    lo, d = db.table("lineorder"), db.table("date")
    r = (db.select(d.d_year, F.sum(lo.lo_revenue).alias("revenue"))
           .where(d.d_year.between(1993, 1997) & (lo.lo_discount < 3))
           .group_by(d.d_year)
           .order_by(F.desc("revenue"))
           .run())
    r.rows; r.columns; r.df()
    print(q.to_sql())                      # the same query as SQL text (DuckDB runs it too: q.verify())

Joins: .join(table, on, how="inner" | "left" | "right" | "full" | "cross"); without any FROM / JOIN the tables the
query mentions are joined through their declared foreign keys (one fact, its dimensions). Every SQL function is
F.<name>(...): scalar functions, aggregates (F.sum(x, distinct=True, filter=cond, order_by=[...])), window functions
(F.rank().over(partition_by=[...], order_by=[...]); F.sum(x).over(order_by=[...], rows=(-2, 0))), F.case(), F.lit(),
F.exists(query), F.scalar(query); x.isin(values or query), x.like(pattern), x.between(a, b), x.cast("DECIMAL(10,2)"),
x.is_null(), x.alias(name), x.asc() / x.desc(nulls_first=True)."""
import datetime as _dt
from decimal import Decimal
import sqltypes as T
from sqltypes import SqlError
import sqlir as I
import sqlfuncs as SF
import sqlfront as FR


def _node(x):
    """a Python value or expression -> IR node"""
    if isinstance(x, E): return x.n
    if isinstance(x, Query): return I.Subq("scalar", x._bind(), FR.q_columns(x._bind())[0][1])
    if isinstance(x, I.Node): return x
    if isinstance(x, float): return I.Lit(x, T.DOUBLE)
    if isinstance(x, str): return I.Lit(x, T.VARCHAR, True)
    return FR.lit_of(x)


class E:
    """an expression of the Python API (wraps a bound, typed IR node)"""
    __slots__ = ("n", "name")
    def __init__(self, n, name=None): self.n, self.name = n, name
    def __repr__(self): return f"E({I.render(self.n)})"
    def alias(self, name): return E(self.n, name)
    as_ = alias
    @property
    def type(self): return self.n.t

    def _f(self, fn, *others, rev=False):
        args = [self.n] + [_node(o) for o in others]
        if rev: args = args[::-1]
        if fn in ("=", "<>", "<", "<=", ">", ">=", "is_distinct", "is_not_distinct") and any(FR._rowish(a) for a in args):
            return E(FR.Binder(None).cmp(fn, args[0], args[1]))          # rows compare field by field (ISO 8.2)
        return E(FR.make_func(fn, args))
    def __add__(self, o): return self._f("+", o)
    def __radd__(self, o): return self._f("+", o, rev=True)
    def __sub__(self, o): return self._f("-", o)
    def __rsub__(self, o): return self._f("-", o, rev=True)
    def __mul__(self, o): return self._f("*", o)
    def __rmul__(self, o): return self._f("*", o, rev=True)
    def __truediv__(self, o): return self._f("/", o)
    def __rtruediv__(self, o): return self._f("/", o, rev=True)
    def __floordiv__(self, o): return self._f("//", o)
    def __mod__(self, o): return self._f("%", o)
    def __neg__(self): return E(FR.make_func("neg", [self.n]))
    def __pos__(self): return self
    def __eq__(self, o): return self._f("=", o)
    def __ne__(self, o): return self._f("<>", o)
    def __lt__(self, o): return self._f("<", o)
    def __le__(self, o): return self._f("<=", o)
    def __gt__(self, o): return self._f(">", o)
    def __ge__(self, o): return self._f(">=", o)
    def __and__(self, o): return E(FR.make_func("and", [self.n, _node(o)]))
    def __rand__(self, o): return E(FR.make_func("and", [_node(o), self.n]))
    def __or__(self, o): return E(FR.make_func("or", [self.n, _node(o)]))
    def __ror__(self, o): return E(FR.make_func("or", [_node(o), self.n]))
    def __invert__(self): return E(FR.make_func("not", [self.n]))
    __hash__ = object.__hash__
    def concat(self, o): return self._f("||", o)
    def between(self, a, b, symmetric=False):
        x, lo, hi = self.n, _node(a), _node(b)
        if any(FR._rowish(v) for v in (x, lo, hi)):
            B = FR.Binder(None); fwd = FR.make_and([B.cmp(">=", x, lo), B.cmp("<=", x, hi)])
            return E(FR.make_func("or", [fwd, FR.make_and([B.cmp(">=", x, hi), B.cmp("<=", x, lo)])]) if symmetric else fwd)
        return E(FR.make_func("between_sym" if symmetric else "between", [x, lo, hi]))
    def isin(self, *vals):
        if len(vals) == 1 and isinstance(vals[0], Query):
            q = vals[0]._bind(); t = FR.q_columns(q)[0][1]
            return E(I.Subq("any", q, T.BOOLEAN, FR.cast_to(self.n, T.supertype(self.n.t, t)), "=", correlated=FR._correlated(q)))
        if len(vals) == 1 and isinstance(vals[0], (list, tuple, set)): vals = tuple(vals[0])
        return E(FR.make_func("in", [self.n] + [_node(v) for v in vals]))
    def notin(self, *vals): return ~self.isin(*vals)
    def like(self, p, escape=None): return E(FR.make_func("like", [self.n, _node(p)] + ([_node(escape)] if escape is not None else [])))
    def ilike(self, p): return E(FR.make_func("ilike", [self.n, _node(p)]))
    def similar_to(self, p, escape=None): return E(FR.make_func("__iso_similar", [self.n, _node(p)] + ([_node(escape)] if escape else [])))
    def like_regex(self, p, flags=None): return E(FR.make_func("__iso_like_regex", [self.n, _node(p)] + ([_node(flags)] if flags else [])))
    def is_null(self): return E(FR._row_null(self.n, "is_null") if FR._rowish(self.n) else FR.make_func("is_null", [self.n]))
    def is_not_null(self): return E(FR._row_null(self.n, "is_not_null") if FR._rowish(self.n) else FR.make_func("is_not_null", [self.n]))
    def is_distinct_from(self, o): return self._f("is_distinct", o)
    def is_not_distinct_from(self, o): return self._f("is_not_distinct", o)
    def is_true(self): return E(FR.make_func("is_true", [self.n]))
    def is_false(self): return E(FR.make_func("is_false", [self.n]))
    def is_unknown(self): return E(FR.make_func("is_unknown", [self.n]))
    def cast(self, typ, try_=False):
        t = typ if isinstance(typ, T.SqlType) else T.type_from_name(typ)
        return E(FR.cast_to(self.n, t, try_) if isinstance(self.n, I.Lit) else I.Cast(self.n, t, try_))
    def try_cast(self, typ): return self.cast(typ, True)
    def collate(self, name):
        """x COLLATE name (NOCASE, NOACCENT, NFC, UNICODE, ...): comparisons, grouping and sorting by that collation"""
        return E(FR.make_func("__collate", [self.n, I.Lit(str(name).upper(), T.VARCHAR)]), self.name)
    def field(self, name):
        """a field of a row value: x.field("a") is SQL's x.a (T051)"""
        return E(FR.field_ref(self.n, name), name)
    def asc(self, nulls_first=False): return Order(self, False, nulls_first)
    def desc(self, nulls_first=False): return Order(self, True, nulls_first)

    def over(self, partition_by=(), order_by=(), rows=None, range=None, groups=None, exclude=None):
        """this aggregate or window function as a window: frame rows / range / groups = (start, end), each None
        (unbounded), 0 (current row), -k (k preceding) or +k (k following)"""
        n = self.n
        part = [_node(p) for p in _list(partition_by)]
        order = [_order(o) for o in _list(order_by)]
        frame = None
        for mode, fr in (("rows", rows), ("range", range), ("groups", groups)):
            if fr is not None:
                def b(x, start):
                    if x is None: return ("unbounded_preceding" if start else "unbounded_following", None)
                    if isinstance(x, (int, Decimal, float)) and x == 0: return ("current", None)
                    if isinstance(x, (_dt.timedelta, T.Interval)):
                        iv = x if isinstance(x, T.Interval) else T.Interval(0, x.days, x.seconds * 1_000_000 + x.microseconds)
                        neg = iv < T.Interval()
                        return ("preceding" if neg else "following", I.Lit(-iv if neg else iv, T.INTERVAL))
                    return ("preceding" if x < 0 else "following", FR.lit_of(abs(x)))
                frame = (mode, b(fr[0], True), b(fr[1], False))
        if frame is None and order: frame = ("range", ("unbounded_preceding", None), ("current", None))
        for x in part + [o[0] for o in order]: FR.check_nested(x)
        if isinstance(n, I.Agg):
            return E(I.Win(n.name, n.args, n.t, part, order, frame, exclude, False, n.filter, False, n.distinct, n.params), self.name)
        for x in n.args: FR.check_nested(x)
        if isinstance(n, _WinCall):
            return E(I.Win(n.kind, n.args, SF.win_typer(n.kind, n.args), part, order, frame, exclude, n.ignore_nulls, None, n.from_last), self.name)
        raise SqlError("over() applies to an aggregate or a window function")

    def within_group(self, *order_by):
        """an ordered-set aggregate's ORDER BY: F.percentile_cont(0.5).within_group(x)"""
        n = self.n
        if not isinstance(n, _Ordered): raise SqlError("within_group() applies to an ordered-set aggregate")
        return n.finish([_order(o) for o in order_by])


class Order:
    def __init__(self, e, desc=False, nulls_first=False): self.e, self.desc, self.nulls_first = e, desc, nulls_first


def _order(o):
    if isinstance(o, Order): return (_node(o.e) if not isinstance(o.e, str) else o.e, o.desc, o.nulls_first)
    return (_node(o) if not isinstance(o, str) else o, False, False)


def _list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]


class _WinCall(I.Node):
    """a window-only function before over() (row_number, rank, lag, ...)"""
    __slots__ = ("kind", "args", "ignore_nulls", "from_last")
    def __init__(self, kind, args, ignore_nulls=False, from_last=False):
        self.kind, self.args, self.ignore_nulls, self.from_last = kind, args, ignore_nulls, from_last
        self.t = T.BIGINT
    def _key(self): return ("wincall", self.kind, tuple(a.key() for a in self.args))


class _Ordered(I.Node):
    """an ordered-set aggregate waiting for WITHIN GROUP (ORDER BY ...)"""
    __slots__ = ("name", "params")
    def __init__(self, name, params): self.name, self.params, self.t = name, params, T.NULL
    def _key(self): return ("ordered", self.name)
    def finish(self, order):
        name, p = self.name, self.params
        x, desc = order[0][0], order[0][1]
        if name in ("percentile_cont", "percentile_disc"):
            q = p[0]
            agg = "quantile_cont" if name == "percentile_cont" else "quantile_disc"
            if desc and agg == "quantile_cont": q = 1 - Decimal(str(q))
            params = (q, "desc") if desc and agg == "quantile_disc" else (q,)
            return E(I.Agg(agg, [x], SF.agg_typer(agg, [x.t], params), params=params))
        if name == "mode": return E(I.Agg("mode", [x], x.t, params=("desc" if desc else "asc",)))
        if name == "listagg":
            sep = p[0] if p else ""
            return E(I.Agg("__iso_listagg", [FR.cast_to(p[1], T.VARCHAR)] if len(p) > 1 else [], T.VARCHAR, order=order, params=(sep,)))
        if name in ("rank", "dense_rank", "percent_rank", "cume_dist"):
            if len(p) != len(order): raise SqlError("hypothetical set function: one argument per ORDER BY column")
            return E(I.Agg(f"__iso_h{name}", [o[0] for o in order], T.BIGINT if name in ("rank", "dense_rank") else T.DOUBLE,
                           order=order, params=tuple(p)))
        raise SqlError(name)


class _Functions:
    """F.<name>(...): every SQL function of torbles"""
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return lambda *args, **kw: self.call(name, *args, **kw)

    def call(self, name, *args, distinct=False, filter=None, order_by=(), ignore_nulls=False, from_last=False):
        lname = name.lower().rstrip("_")
        if lname == "row_number" and len(args) == 1 and isinstance(args[0], str):        # ROW_NUMBER(<row marker>) (T619)
            if args[0].upper() not in FR._ROW_MARKERS: raise SqlError(f"ROW_NUMBER({args[0]}): a row marker is one of {sorted(FR._ROW_MARKERS)}")
            return E(I.Func("__nwf_row_number", [], T.BIGINT, extra=(args[0].upper(),)))
        nodes = [_node(a) for a in args]
        if lname == "typeof" and len(nodes) == 1:            # the declared type, as db.sql() gives it
            return E(I.Lit(str(nodes[0].t) if nodes[0].t.id != "NULL" else '"NULL"', T.VARCHAR))
        if lname in SF.WINDOW_ONLY or lname in ("row_number", "ntile", "lag", "lead", "first_value", "last_value", "nth_value"):
            if lname in ("rank", "dense_rank", "percent_rank", "cume_dist") and args:
                return E(_Ordered(lname, [a.v if isinstance(a, I.Lit) else a for a in nodes]))
            return E(_WinCall(lname, nodes, ignore_nulls, from_last))
        if lname in ("percentile_cont", "percentile_disc", "mode") or (lname == "listagg" and not order_by and len(args) <= 2 and False):
            if lname == "mode" and nodes: return E(I.Agg("mode", nodes, nodes[0].t))
            return E(_Ordered(lname, [a.v if isinstance(a, I.Lit) else a for a in nodes]))
        aname = FR._AGG_ALIAS.get(lname, lname)
        if lname == "count" and not args: aname = "count_star"
        if aname in SF.AGGS or aname == "count_star":
            params = ()
            if aname in ("string_agg", "listagg"):
                aname = "string_agg"; params = (nodes[1].v if len(nodes) > 1 else ",",); nodes = [FR.cast_to(nodes[0], T.VARCHAR)]
            elif aname in ("quantile_cont", "quantile_disc") and len(nodes) == 2: params = (nodes[1].v,); nodes = nodes[:1]
            order = [_order(o) for o in _list(order_by)]
            filt = _node(filter) if filter is not None else None
            t = T.BIGINT if aname == "count_star" else SF.agg_typer(aname, [n.t for n in nodes], params)
            return E(I.Agg(aname, nodes, t, distinct, filt, order, params))
        fname = FR._FN_ALIAS.get(lname, lname)
        if fname == "extract": return E(FR.make_func("date_part", nodes))
        if fname not in SF.SCALARS: raise SqlError(f"no SQL function {name}")
        return E(FR.make_func(fname, nodes))

    # forms that are not plain calls
    def lit(self, v, typ=None):
        n = _node(v)
        return E(n if typ is None else FR.cast_to(n, T.type_from_name(typ) if isinstance(typ, str) else typ))
    def col(self, ref): raise SqlError("use db.table(name).column")
    def case(self, *whens, else_=None, operand=None):
        """F.case((cond, value), ..., else_=value); with operand: F.case((v1, r1), ..., operand=x) compares x = v1, ..."""
        ws = []
        for c, v in whens:
            cn = _node(c) if operand is None else FR.make_func("=", [_node(operand), _node(c)])
            ws.append((cn, _node(v)))
        el = _node(else_) if else_ is not None else None
        vals = [v for _, v in ws] + ([el] if el is not None else [])
        t = T.supertype_all([v.t for v in vals], [getattr(v, "strlit", False) for v in vals])
        if t.id == "NULL": t = T.INTEGER
        return E(I.Case([(c, FR.cast_to(v, t)) for c, v in ws], FR.cast_to(el, t) if el is not None else None, t))
    def exists(self, q): b = q._bind(); return E(I.Subq("exists", b, T.BOOLEAN, correlated=FR._correlated(b)))
    def unique(self, q): b = q._bind(); return E(I.Subq("unique", b, T.BOOLEAN, correlated=FR._correlated(b)))
    def scalar(self, q):
        b = q._bind(); return E(I.Subq("scalar", b, FR.q_columns(b)[0][1], correlated=FR._correlated(b)))
    def any(self, x, op, q):
        """x <op> ANY (query)"""
        b = q._bind(); return E(I.Subq("any", b, T.BOOLEAN, _node(x), op, correlated=FR._correlated(b)))
    def all(self, x, op, q):
        neg = {"=": "<>", "<>": "=", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[op]
        return ~self.any(x, neg, q)
    def interval(self, n, unit):
        return E(I.Lit(T.parse_interval(f"{n} {unit}"), T.INTERVAL))
    def grouping(self, *xs): return E(I.Grouping([_node(x) for x in xs]))
    def expr(self, text, *tables):
        """any SQL value expression, bound against the given tables: F.expr("JSON_VALUE(d.doc, '$.a' RETURNING INTEGER)", d),
        F.expr("PERIOD (t.s, t.e) CONTAINS DATE '2020-01-05'", t), F.expr("x IS NORMALIZED", t) — every form db.sql()
        accepts in an expression (aggregates and window functions too)"""
        db = tables[0]._db if tables else None
        nodes, ptext = FR.parse("SELECT " + text)
        if len(nodes) != 1 or len(nodes[0].get("select_list") or []) != 1: raise SqlError(f"F.expr takes one value expression, not {text!r}")
        b = FR.Binder(db, ptext, dict(FR.FLAGS))
        return E(b.expr(nodes[0]["select_list"][0], FR.Scope([t._b for t in tables]), FR.Ctx(agg=True, win=True, where="F.expr")))
    def value_of(self, x, at="CURRENT_ROW", offset=0, default=None):
        """VALUE_OF(x AT <row marker> [+ | - offset] [, default]) in the argument of a window aggregate (T619); with
        F.row_number("FRAME_ROW") etc.: F.sum(F.case((F.value_of(t.x) <> t.x, t.v))).over(partition_by=[t.k])"""
        m = at.upper()
        if m not in FR._ROW_MARKERS: raise SqlError(f"VALUE_OF ... AT {at}: a row marker is one of {sorted(FR._ROW_MARKERS)}")
        v = _node(x); d = _node(default) if default is not None else I.Lit(None, T.NULL)
        t = T.supertype(v.t, d.t, getattr(v, "strlit", False), getattr(d, "strlit", False))
        if t.id == "NULL": t = v.t
        return E(I.Func("__nwf_value_of", [FR.cast_to(v, t), I.Lit(int(offset), T.BIGINT), FR.cast_to(d, t)], t, extra=(m,)))
    def asc(self, x, nulls_first=False): return Order(x if not isinstance(x, str) else x, False, nulls_first)
    def desc(self, x, nulls_first=False): return Order(x if not isinstance(x, str) else x, True, nulls_first)
    def overlaps(self, s1, e1, s2, e2): return E(FR.make_func("__iso_overlaps", [_node(s1), _node(e1), _node(s2), _node(e2)]))


F = _Functions()


class TableRef:
    """a table of the database in the Python API: its columns are attributes (E expressions)"""
    def __init__(self, db, name, alias=None):
        self._db = db
        self._b = FR.Binder(db).table_binding(name, alias)
    def __getattr__(self, k):
        if k.startswith("_"): raise AttributeError(k)
        c = FR._find_col(self._b, k)
        if c is None: raise AttributeError(f"{self._b.alias} has no column {k}")
        return E(c, c.name)
    def __getitem__(self, k): return self.__getattr__(k)
    def columns(self): return [c for c, _, _ in self._b.cols]
    def __repr__(self): return f"TableRef({self._b.alias})"


class Query:
    """a SELECT under construction (immutable steps return the same object for chaining)"""
    def __init__(self, db, items):
        self.db = db
        self._items = list(items)
        self._from, self._joins = None, []
        self._where, self._group, self._sets, self._having, self._qualify = [], [], None, [], []
        self._order, self._limit, self._offset, self._distinct = [], None, None, False
        self._setops = []
        self._bound = None

    def from_(self, t): self._from = t; return self
    def join(self, t, on=None, how="inner"):
        self._joins.append((t, on, how.lower())); return self
    def left_join(self, t, on): return self.join(t, on, "left")
    def where(self, *conds): self._where += list(conds); return self
    def group_by(self, *xs): self._group = list(xs); self._sets = None; return self
    def rollup(self, *xs):
        self._group = list(xs); self._sets = [list(range(k)) for k in range(len(xs), -1, -1)]; return self
    def cube(self, *xs):
        n = len(xs); self._group = list(xs)
        self._sets = [[i for i in range(n) if m >> (n - 1 - i) & 1] for m in range(2 ** n - 1, -1, -1)]; return self
    def grouping_sets(self, *sets):
        allx = []
        for st in sets:
            for x in st:
                if not any(x is y for y in allx): allx.append(x)
        self._group = allx
        self._sets = [[next(i for i, y in enumerate(allx) if y is x) for x in st] for st in sets]; return self
    def having(self, *conds): self._having += list(conds); return self
    def qualify(self, *conds): self._qualify += list(conds); return self
    def order_by(self, *specs): self._order = list(specs); return self
    def limit(self, n, offset=None): self._limit, self._offset = n, offset; return self
    def offset(self, n): self._offset = n; return self
    def distinct(self): self._distinct = True; return self
    def union(self, q, all=False): self._setops.append(("union", all, q)); return self
    def intersect(self, q, all=False): self._setops.append(("intersect", all, q)); return self
    def except_(self, q, all=False): self._setops.append(("except", all, q)); return self

    # ------------------------------------------------------------ binding
    def _bind(self):
        if self._bound is not None: return self._bound
        s = I.Select()
        items = [(x if isinstance(x, E) else E(_node(x))) for x in self._items]
        s.items = [(x.n, x.name or I.render(x.n)) for x in items]
        bindings = self._from_tree(s, items)
        s.where = FR.make_and([_node(c) for c in self._where]) if self._where else None
        s.group = [_node(g) for g in self._group]
        s.sets = self._sets
        s.having = FR.make_and([_node(c) for c in self._having]) if self._having else None
        s.qualify = FR.make_and([_node(c) for c in self._qualify]) if self._qualify else None
        s.aggregate = bool(s.group) or s.sets is not None or any(I.has(x, I.Agg) for x, _ in s.items) or s.having is not None
        s.distinct = self._distinct
        names = [n.lower() for _, n in s.items]
        for o in self._order:
            e, desc, nf = _order(o)
            if isinstance(e, str):
                if e.lower() not in names: raise SqlError(f"ORDER BY {e}: no such output column")
                s.order.append((("pos", names.index(e.lower())), desc, nf))
            else:
                k = e.key()
                pos = next((i for i, (x, _) in enumerate(s.items) if x.key() == k), None)
                s.order.append((("pos", pos) if pos is not None else ("expr", e), desc, nf))
        s.limit, s.offset = self._limit, self._offset
        for x in [x for x, _ in s.items] + [c for c in (s.where, s.having, s.qualify) if c is not None] + [o[0][1] for o in s.order if o[0][0] == "expr"] + s.group:
            FR.check_nested(x)
        if s.aggregate: FR._check_grouped(s)
        q = s
        for op, all_, other in self._setops:
            ob = other._bind()
            a, b = FR.q_columns(q), FR.q_columns(ob)
            so = I.SetOp(op, all_, q, ob)
            so.items = [(na, T.supertype(ta, tb)) for (na, ta), (_, tb) in zip(a, b)]
            q = so
        self._bound = q
        return q

    def _from_tree(self, s, items):
        if self._from is not None:
            tree = self._from._b
            for t, on, how in self._joins:
                j = I.Join(how if how != "cross" else "cross", tree, t._b, _node(on) if on is not None else None)
                if how in ("left", "full"): t._b.outer_null = True
                tree = j
            s.from_ = tree
            return
        # no FROM: the tables the query mentions, joined through their foreign keys
        found = {}
        exprs = [x.n for x in items] + [_node(c) for c in self._where + self._having + self._group]
        for x in exprs:
            for c in I.cols_of(x): found.setdefault(id(c.b), c.b)
            for sq in I.find(x, I.Subq): pass
        bs = list(found.values())
        if not bs: s.from_ = None; return
        if len(bs) == 1: s.from_ = bs[0]; return
        for f in bs:
            conds, ok = [], True
            for d in bs:
                if d is f: continue
                fks = [c for c in f.table.fks_to(d.table)] if f.table is not None else []
                if len(fks) != 1: ok = False; break
                pk = getattr(d.table, "pk", None)
                conds.append(FR.make_func("=", [f.col(fks[0]), d.col(pk)]))
            if ok:
                tree = f
                for d, c in zip([d for d in bs if d is not f], conds): tree = I.Join("inner", tree, d, c)
                s.from_ = tree
                return
        raise SqlError("the tables cannot be joined through their foreign keys: give .from_() and .join(..., on=...)")

    # ------------------------------------------------------------ running
    def run(self, engine="alg"):
        import sqlexec
        return sqlexec.run(self.db, self._bind(), engine)

    def to_sql(self):
        return I.to_sql(self._bind())

    def verify(self, engine="alg", tol=1e-9):
        """(equal, torbles rows, source rows): the query on torbles and its SQL text on the source DuckDB database"""
        import sqlfuzz
        got = self.run(engine).rows
        want = self.db.con.execute(self.to_sql()).fetchall()
        ordered = bool(self._order) and self._limit is not None
        return sqlfuzz.same(got, want, ordered, tol), got, want
