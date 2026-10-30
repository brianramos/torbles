# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The SQL function library: scalar functions and operators, aggregate functions, window functions; their result
types and their values. ISO/IEC 9075 functions and the common DuckDB ones (DuckDB's rules where ISO leaves a choice).

Scalar functions run on group-level values: the distinct values of a dimension's columns, the distinct codes of a
fact column, or the rows of an aggregated result, never on fact rows one by one (the engine filters and aggregates
those on the bit planes). A function raising an error yields an Err value that only fails the query if it is used."""
import calendar, datetime as _dt, math, re, unicodedata
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING, ROUND_DOWN
from fractions import Fraction
import sqltypes as T
from sqltypes import SqlError, SqlType, Interval, CTX


class Err:
    """a deferred runtime error (division by zero, a failed cast): CASE may discard it, otherwise it fails the query"""
    __slots__ = ("msg",)
    def __init__(self, msg): self.msg = msg
    def __repr__(self): return f"Err({self.msg})"


def check(vals):
    """raise the first deferred error among result values"""
    for v in vals:
        if isinstance(v, Err): raise SqlError(v.msg)
    return vals


# ================================================================== scalar registry
class Scalar:
    __slots__ = ("name", "typer", "impl", "nulls", "deterministic")
    def __init__(self, name, typer, impl, nulls, deterministic=True):
        self.name, self.typer, self.impl, self.nulls, self.deterministic = name, typer, impl, nulls, deterministic


SCALARS = {}


def scalar(names, typer, nulls="propagate", deterministic=True):
    """register impl(*values) -> value. typer(args) -> (result type, [argument types to cast to, or None])
    nulls: "propagate" (NULL in, NULL out), "custom" (impl sees NULLs)"""
    def deco(fn):
        for n in names.split():
            SCALARS[n] = Scalar(n, typer, fn, nulls, deterministic)
        return fn
    return deco


def nullable(f):
    """can this function call yield NULL?"""
    s = SCALARS.get(f.fn)
    if f.fn in ("is_null", "is_not_null", "is_distinct", "is_not_distinct", "is_true", "is_false", "is_unknown",
                "is_not_true", "is_not_false", "is_not_unknown", "typeof", "concat", "concat_ws"):
        return False
    if f.fn in ("coalesce",): return all(a.nullable for a in f.args)
    if f.fn in ("nullif", "try_cast", "/", "sqrt", "ln", "log", "split_part", "regexp_extract", "__iso_substring_similar",
                "substring_regex", "nth_list", "list_extract", "greatest", "least", "chr", "__row_field"):
        return True
    if s is not None and s.nulls == "propagate": return any(a.nullable for a in f.args)
    return True


def fixed(t):
    return lambda args: (t, None)


def same_as(i=0):
    return lambda args: (args[i].t, None)


def lit_int(a):
    import sqlir
    return isinstance(a, sqlir.Lit) and isinstance(a.v, int) and not isinstance(a.v, bool)


def _strlit(a):
    import sqlir
    return isinstance(a, sqlir.Lit) and a.strlit


def _num_t(args, op):
    """result type of a binary arithmetic operator, and the operand types (a DOUBLE operation converts both)"""
    a, b = args[0].t, args[1].t
    ta, tb = a, b
    if a.id == "NULL": a = b
    if b.id == "NULL": b = a
    if a.id == "NULL": return T.INTEGER, None
    if _strlit(args[0]) and b.numeric: a = T.DOUBLE if b.floating else b
    if _strlit(args[1]) and a.numeric: b = T.DOUBLE if a.floating else a
    if not (a.numeric and b.numeric):
        raise SqlError(f"No function matches {op}({ta}, {tb})")
    if "DECFLOAT" in (a.id, b.id):                      # DECFLOAT arithmetic: rounded to the result's precision
        r = T.supertype(a, b)
        return r, [r, r]
    if a.integer and b.integer:
        if lit_int(args[1]) and _fits(args[1].v, a): b = a
        if lit_int(args[0]) and _fits(args[0].v, b): a = b
        r = T._int_supertype(a, b)
        if op == "/": return T.DOUBLE, [T.DOUBLE, T.DOUBLE]
        return r, None
    if a.floating or b.floating:
        r = T.FLOAT if a.id == b.id == "FLOAT" else T.DOUBLE
        return r, [r, r]
    da, db = T.as_decimal(a), T.as_decimal(b)
    if op in ("+", "-", "%"):
        s = max(da.s, db.s); w = max(da.p - da.s, db.p - db.s)
        return T.DECIMAL(min(38, w + s + (1 if op != "%" else 1)), s), None
    if op == "*":
        s = da.s + db.s; p = da.p + db.p
        if da.p <= 18 and db.p <= 18 and p > 18: p = 18
        if s > 38: return T.DOUBLE, [T.DOUBLE, T.DOUBLE]
        return T.DECIMAL(min(38, max(p, s)), s), None
    return T.DOUBLE, [T.DOUBLE, T.DOUBLE]                   # / and // on decimals: DOUBLE (DuckDB)


def _fits(v, t):
    if not t.integer: return False
    lo, hi = T.INT_RANGE[t.id]; return lo <= v <= hi


def _arith_typer(op):
    def ty(args):
        a, b = args[0].t, args[1].t
        tem = _temporal_arith(op, a, b, args)
        if tem is not None: return tem
        if op == "//" and not (a.integer and b.integer) and a.numeric and b.numeric:
            return T.DOUBLE, [T.DOUBLE, T.DOUBLE]
        return _num_t(args, op)
    return ty


def _temporal_arith(op, a, b, args):
    I, D, TS, TM = "INTERVAL", "DATE", "TIMESTAMP", "TIME"
    if a.zoned or b.zoned:                         # WITH TIME ZONE: the same rules, zoned results
        return _zoned_arith(op, a, b, args)
    if _strlit(args[0]) and b.id in (D, TS, I): a = b if op != "-" or b.id == I else a
    if _strlit(args[1]) and a.id in (D, TS) and op == "-": b = a
    if _strlit(args[1]) and a.id in (D, TS, TM) and op in ("+", "-") and b.id == "VARCHAR": b = T.INTERVAL
    ids = (a.id, b.id)
    if op == "+":
        if ids in ((D, I), (I, D), (TS, I), (I, TS)): return T.TIMESTAMP, [a if a.id != "VARCHAR" else T.INTERVAL, b]
        if ids in ((D, TM), (TM, D)): return T.TIMESTAMP, None
        if a.id == D and b.integer: return T.DATE, None
        if b.id == D and a.integer: return T.DATE, None
        if ids == (I, I): return T.INTERVAL, None
        if ids in ((TM, I), (I, TM)): return T.TIME, None
    if op == "-":
        if ids == (D, D): return T.BIGINT, [a, b]
        if ids in ((TS, TS), (D, TS), (TS, D)): return T.INTERVAL, [T.TIMESTAMP, T.TIMESTAMP]
        if ids in ((D, I), (TS, I)): return T.TIMESTAMP, [a, b]
        if a.id == D and b.integer: return T.DATE, None
        if ids == (I, I): return T.INTERVAL, None
        if ids == (TM, I): return T.TIME, None
        if ids == (TM, TM): return T.INTERVAL, None
    if op == "*":
        if a.id == I and b.numeric: return T.INTERVAL, None
        if b.id == I and a.numeric: return T.INTERVAL, None
    if op == "/" and a.id == I and b.numeric: return T.INTERVAL, None
    return None


def _zoned_arith(op, a, b, args):
    I, Z, ZT = "INTERVAL", "TIMESTAMPTZ", "TIMETZ"
    # a string literal on the other side: an interval, or (subtracted from a timestamp) a timestamp
    if _strlit(args[0]) and op == "+" and b.id in (Z, ZT): a = T.INTERVAL
    if _strlit(args[1]) and a.id in (Z, ZT):
        b = T.TIMESTAMPTZ if op == "-" and a.id == Z and re.match(r"\s*-?\d{1,6}-\d", str(args[1].v)) else T.INTERVAL
    ids = (a.id, b.id)
    if a.id in (Z, "TIMESTAMP", "DATE") and b.id in (Z, "TIMESTAMP", "DATE") and op == "-": return T.INTERVAL, [T.TIMESTAMPTZ, T.TIMESTAMPTZ]
    if op == "+" and ids in ((Z, I), (I, Z)): return T.TIMESTAMPTZ, [a if a.id != "VARCHAR" else T.INTERVAL, b if b.id != "VARCHAR" else T.INTERVAL]
    if op == "-" and ids == (Z, I): return T.TIMESTAMPTZ, None
    if op == "+" and ids in ((ZT, I), (I, ZT)): return T.TIMETZ, None
    if op == "-" and ids == (ZT, I): return T.TIMETZ, None
    if op == "-" and a.id in (ZT, "TIME") and b.id in (ZT, "TIME"): return T.INTERVAL, [T.TIMETZ, T.TIMETZ]
    if op == "+" and ids in (("DATE", ZT), (ZT, "DATE")): return T.TIMESTAMPTZ, None
    return None


def _to_dt(v):
    return v if isinstance(v, _dt.datetime) else _dt.datetime(v.year, v.month, v.day)


def _iv_scale(iv, k):
    k = Fraction(repr(k)) if isinstance(k, float) else Fraction(k)
    months = iv.months * k; mi = int(months)
    days = iv.days * k + (months - mi) * 30; di = int(days)
    micros = iv.micros * k + (days - di) * 86_400_000_000
    return Interval(mi, di, int(micros))


def _add(a, b):
    if isinstance(a, (int, Decimal, float)) and isinstance(b, (int, Decimal, float)) and not isinstance(a, bool):
        return a + b
    if isinstance(a, Interval) and isinstance(b, Interval): return a + b
    if isinstance(b, Interval) and isinstance(a, (_dt.date, _dt.datetime)): return T.add_interval(a, b)
    if isinstance(a, Interval) and isinstance(b, (_dt.date, _dt.datetime)): return T.add_interval(b, a)
    if isinstance(a, _dt.time) and isinstance(b, Interval): return _time_add(a, b, 1)
    if isinstance(b, _dt.time) and isinstance(a, Interval): return _time_add(b, a, 1)
    if isinstance(a, _dt.date) and not isinstance(a, _dt.datetime) and isinstance(b, int): return a + _dt.timedelta(days=b)
    if isinstance(b, _dt.date) and not isinstance(b, _dt.datetime) and isinstance(a, int): return b + _dt.timedelta(days=a)
    if isinstance(a, _dt.date) and isinstance(b, _dt.time): return _dt.datetime.combine(a, b)
    if isinstance(b, _dt.date) and isinstance(a, _dt.time): return _dt.datetime.combine(b, a)
    return a + b


def _time_add(t, iv, sign):
    us = (t.hour * 3600 + t.minute * 60 + t.second) * 1_000_000 + t.microsecond + sign * iv.micros
    us %= 86_400_000_000
    return _dt.time(us // 3_600_000_000, us // 60_000_000 % 60, us // 1_000_000 % 60, us % 1_000_000, tzinfo=t.tzinfo)


def _sub(a, b):
    if isinstance(a, (int, Decimal, float)) and isinstance(b, (int, Decimal, float)): return a - b
    if isinstance(a, Interval) and isinstance(b, Interval): return a - b
    if isinstance(b, Interval) and isinstance(a, (_dt.date, _dt.datetime)): return T.add_interval(a, b, -1)
    if isinstance(a, _dt.time) and isinstance(b, Interval): return _time_add(a, b, -1)
    if isinstance(a, _dt.time) and isinstance(b, _dt.time):
        if a.tzinfo is not None and b.tzinfo is not None:            # TIME WITH TIME ZONE: compare the UTC times
            a, b = T.in_zone(a, _dt.timezone.utc), T.in_zone(b, _dt.timezone.utc)
        return Interval(0, 0, ((a.hour - b.hour) * 3600 + (a.minute - b.minute) * 60 + a.second - b.second) * 1_000_000 + a.microsecond - b.microsecond)
    if isinstance(a, _dt.datetime) or isinstance(b, _dt.datetime):
        d = _to_dt(a) - _to_dt(b)
        return Interval(0, d.days, d.seconds * 1_000_000 + d.microseconds)        # DuckDB: days and time
    if isinstance(a, _dt.date) and isinstance(b, _dt.date): return (a - b).days
    if isinstance(a, _dt.date) and isinstance(b, int): return a - _dt.timedelta(days=b)
    return a - b


def _mul(a, b):
    if isinstance(a, Interval): return _iv_scale(a, b)
    if isinstance(b, Interval): return _iv_scale(b, a)
    return a * b


def _div(a, b):
    if isinstance(a, T.DecFloat) or isinstance(b, T.DecFloat):
        if b == 0: return Err("Division by zero")
        return CTX.divide(Decimal(a), Decimal(b))
    if isinstance(a, Interval):
        if b == 0: return Err("Division by zero")
        return _iv_scale(a, 1 / Fraction(str(b)) if isinstance(b, (float, Decimal)) else Fraction(1, b))
    a, b = float(a), float(b)
    if b == 0:
        if a == 0 or a != a: return math.nan
        return math.copysign(math.inf, a) * (math.copysign(1, b))
    return a / b


def _idiv(a, b):
    if isinstance(a, float) or isinstance(b, float): return _div(a, b)
    if b == 0: return Err("Division by zero in expression (//). Use TRY(...) to return NULL")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if b == 0: return math.nan
        return math.fmod(a, b)
    if b == 0: return Err("Division by zero in expression (%). Use TRY(...) to return NULL")
    if isinstance(a, Decimal) or isinstance(b, Decimal):
        return CTX.remainder(Decimal(a), Decimal(b))
    r = abs(a) % abs(b)
    return r if a >= 0 else -r


scalar("+", _arith_typer("+"))(_add)
scalar("-", _arith_typer("-"))(_sub)
scalar("*", _arith_typer("*"))(_mul)
scalar("/", _arith_typer("/"))(_div)
scalar("// divide", _arith_typer("//"))(_idiv)
scalar("% mod", _arith_typer("%"))(_mod)


@scalar("__decfloat", lambda a: (T.DECFLOAT(a[1].v), None))
def _decfloat(x, p): return T.decfloat(x, p) if isinstance(x, Decimal) or isinstance(x, (int, float)) else x


@scalar("neg", lambda a: (a[0].t if a[0].t.id != "NULL" else T.INTEGER, None))
def _neg(a): return -a


@scalar("pos", same_as())
def _pos(a): return a


# ------------------------------------------------------------------ comparisons and logic
def _cmp_typer(args):
    a, b = args
    t = T.supertype(a.t, b.t, _strlit(a), _strlit(b))
    if t.id == "VARCHAR" and (a.t.id != "VARCHAR" or b.t.id != "VARCHAR") and not (_strlit(a) or _strlit(b)):
        # DuckDB compares a number with a string column as strings only through an explicit cast
        if a.t.numeric or b.t.numeric: t = T.DOUBLE if (a.t.floating or b.t.floating) else (a.t if a.t.numeric else b.t)
    return T.BOOLEAN, [t, t]


def cmp_values(a, b):
    """-1 / 0 / 1 for two non-NULL values of one type (NaN equals NaN and is the largest; a DATE meets a TIMESTAMP
    as its midnight)"""
    if isinstance(a, _dt.date) and isinstance(b, _dt.date) and isinstance(a, _dt.datetime) != isinstance(b, _dt.datetime):
        a, b = _to_dt(a), _to_dt(b)
    if isinstance(a, (_dt.datetime, _dt.time)) and isinstance(b, (_dt.datetime, _dt.time)) and (a.tzinfo is None) != (b.tzinfo is None):
        a = T.localize(a) if a.tzinfo is None else a; b = T.localize(b) if b.tzinfo is None else b
        if isinstance(a, _dt.time):
            a, b = T.in_zone(a, _dt.timezone.utc).replace(tzinfo=None), T.in_zone(b, _dt.timezone.utc).replace(tzinfo=None)
    if isinstance(a, float) or isinstance(b, float) or isinstance(a, T.DecFloat) or isinstance(b, T.DecFloat):
        an = a != a if not isinstance(a, Decimal) else a.is_nan()
        bn = b != b if not isinstance(b, Decimal) else b.is_nan()
        if an or bn: return 0 if an and bn else (1 if an else -1)
    if isinstance(a, (list, tuple)):
        for x, y in zip(a, b):
            if x is None or y is None:
                if x is None and y is None: continue
                return 1 if x is None else -1
            c = cmp_values(x, y)
            if c: return c
        return (len(a) > len(b)) - (len(a) < len(b))
    return (a > b) - (a < b)


for _op, _f in (("=", lambda c: c == 0), ("<>", lambda c: c != 0), ("<", lambda c: c < 0), ("<=", lambda c: c <= 0),
                (">", lambda c: c > 0), (">=", lambda c: c >= 0)):
    scalar(_op, _cmp_typer)((lambda f: lambda a, b: f(cmp_values(a, b)))(_f))


@scalar("is_distinct", _cmp_typer, nulls="custom")
def _isd(a, b):
    if a is None or b is None: return (a is None) != (b is None)
    return cmp_values(a, b) != 0


@scalar("is_not_distinct", _cmp_typer, nulls="custom")
def _isnd(a, b):
    if a is None or b is None: return a is None and b is None
    return cmp_values(a, b) == 0


def _bool_typer(args):
    for a in args:
        if a.t.id not in ("BOOLEAN", "NULL") and not _strlit(a): raise SqlError(f"{a.t} is not BOOLEAN")
    return T.BOOLEAN, [T.BOOLEAN] * len(args)


@scalar("and", _bool_typer, nulls="custom")
def _and(*xs):
    out = True
    for x in xs:
        if x is False: return False
        if isinstance(x, Err): return x
        if x is None: out = None
    return out


@scalar("or", _bool_typer, nulls="custom")
def _or(*xs):
    out = False
    for x in xs:
        if x is True: return True
        if isinstance(x, Err): return x
        if x is None: out = None
    return out


@scalar("not", _bool_typer)
def _not(x): return not x


scalar("is_null", fixed(T.BOOLEAN), nulls="custom")(lambda x: x is None)
scalar("is_not_null", fixed(T.BOOLEAN), nulls="custom")(lambda x: x is not None)
scalar("is_true", _bool_typer, nulls="custom")(lambda x: x is True)
scalar("is_false", _bool_typer, nulls="custom")(lambda x: x is False)
scalar("is_unknown", _bool_typer, nulls="custom")(lambda x: x is None)
scalar("is_not_true", _bool_typer, nulls="custom")(lambda x: x is not True)
scalar("is_not_false", _bool_typer, nulls="custom")(lambda x: x is not False)
scalar("is_not_unknown", _bool_typer, nulls="custom")(lambda x: x is not None)


def _between_typer(args):
    t = T.supertype_all([a.t for a in args], [_strlit(a) for a in args])
    return T.BOOLEAN, [t] * len(args)


@scalar("between", _between_typer, nulls="custom")
def _between(x, lo, hi):
    a = None if x is None or lo is None else cmp_values(x, lo) >= 0
    b = None if x is None or hi is None else cmp_values(x, hi) <= 0
    return _and(a, b)


@scalar("between_sym", _between_typer, nulls="custom")
def _between_sym(x, lo, hi):
    return _or(_between(x, lo, hi), _between(x, hi, lo))


@scalar("in", _between_typer, nulls="custom")
def _in(x, *vals):
    if x is None: return None
    out = False
    for v in vals:
        if v is None: out = None
        elif cmp_values(x, v) == 0: return True
    return out


# ------------------------------------------------------------------ conditional
def _common_typer(args):
    t = T.supertype_all([a.t for a in args], [_strlit(a) for a in args])
    return t, [t] * len(args)


@scalar("coalesce", _common_typer, nulls="custom")
def _coalesce(*xs):
    for x in xs:
        if x is not None: return x
    return None


@scalar("nullif", lambda a: (T.supertype(a[0].t, a[1].t, _strlit(a[0]), _strlit(a[1])) if a[0].t.id == "NULL" else a[0].t,
                             [None, T.supertype(a[0].t, a[1].t, _strlit(a[0]), _strlit(a[1]))]), nulls="custom")
def _nullif(a, b):
    if a is None: return None
    if b is not None and cmp_values(a, b) == 0: return None
    return a


@scalar("greatest", _common_typer, nulls="custom")
def _greatest(*xs):
    vs = [x for x in xs if x is not None]
    if not vs: return None
    best = vs[0]
    for v in vs[1:]:
        if cmp_values(v, best) > 0: best = v
    return best


@scalar("least", _common_typer, nulls="custom")
def _least(*xs):
    vs = [x for x in xs if x is not None]
    if not vs: return None
    best = vs[0]
    for v in vs[1:]:
        if cmp_values(v, best) < 0: best = v
    return best


# ------------------------------------------------------------------ numeric
def _num1(args):
    t = args[0].t
    if t.id == "NULL": return T.DOUBLE, None
    if not t.numeric:
        if _strlit(args[0]): return T.DOUBLE, [T.DOUBLE]
        raise SqlError(f"numeric function on {t}")
    return t, None


def _dbl(n=1):
    def ty(args):
        if len(args) != n and n >= 0: raise SqlError(f"expected {n} argument(s)")
        for a in args:
            if not (a.t.numeric or a.t.id == "NULL" or _strlit(a)): raise SqlError(f"numeric function on {a.t}")
        return T.DOUBLE, [T.DOUBLE] * len(args)
    return ty


def _abs_typer(args):
    if args[0].t.id == "INTERVAL": return T.INTERVAL, None
    return _num1(args)


@scalar("abs", _abs_typer)
def _abs(x):
    if isinstance(x, Interval): return -x if x._key() < 0 else x        # ABS(interval) (ISO 6.30)
    return abs(x)


@scalar("sign", lambda a: (T.TINYINT, None))
def _sign(x):
    if isinstance(x, float) and x != x: return 0
    return (x > 0) - (x < 0)


def _flt(f):
    def g(*xs):
        try: return f(*xs)
        except (ValueError, ZeroDivisionError): return math.nan
        except OverflowError: return math.inf
    return g


scalar("sqrt", _dbl())(_flt(lambda x: math.sqrt(x) if x >= 0 else math.nan))
scalar("cbrt", _dbl())(lambda x: math.copysign(abs(x) ** (1 / 3), x))
scalar("exp", _dbl())(_flt(math.exp))
scalar("ln", _dbl())(lambda x: math.nan if x != x or x < 0 else (-math.inf if x == 0 else math.log(x)))
scalar("log10", _dbl())(lambda x: math.nan if x != x or x < 0 else (-math.inf if x == 0 else math.log10(x)))
scalar("log2", _dbl())(lambda x: math.nan if x != x or x < 0 else (-math.inf if x == 0 else math.log2(x)))


@scalar("log", lambda a: _dbl(len(a))(a))
def _log(*xs):
    if len(xs) == 1:
        x = xs[0]; return math.nan if x != x or x < 0 else (-math.inf if x == 0 else math.log10(x))
    b, x = xs
    try: return math.log(x) / math.log(b)
    except (ValueError, ZeroDivisionError): return math.nan


@scalar("power pow", _dbl(2))
def _power(a, b):
    try:
        r = a ** b
        if isinstance(r, complex): return math.nan
        return float(r)
    except ZeroDivisionError: return math.inf
    except OverflowError: return math.inf


for _n, _f in (("sin", math.sin), ("cos", math.cos), ("tan", math.tan), ("asin", math.asin), ("acos", math.acos), ("atan", math.atan),
               ("sinh", math.sinh), ("cosh", math.cosh), ("tanh", math.tanh), ("asinh", math.asinh), ("acosh", math.acosh),
               ("atanh", math.atanh), ("degrees", math.degrees), ("radians", math.radians)):
    scalar(_n, _dbl())(_flt(_f))
scalar("atan2", _dbl(2))(_flt(math.atan2))
scalar("cot", _dbl())(_flt(lambda x: 1 / math.tan(x)))
scalar("pi", lambda a: (T.DOUBLE, None))(lambda: math.pi)
scalar("isnan", _dbl())(lambda x: x != x)
scalar("isinf", _dbl())(lambda x: x in (math.inf, -math.inf))
scalar("isfinite", _dbl())(lambda x: math.isfinite(x))


def _round_typer(args):
    t = args[0].t
    if t.id == "DECIMAL":
        n = args[1].v if len(args) > 1 and lit_int(args[1]) else 0
        s = max(0, min(t.s, n))
        return T.DECIMAL(t.p, s), None
    return _num1(args[:1])[0], ([None, T.INTEGER] if len(args) > 1 else None)


def _quant(x, n, mode):
    q = Decimal(1).scaleb(-n)
    return x.quantize(q, rounding=mode, context=CTX) if n >= 0 else (x / (Decimal(10) ** -n)).quantize(Decimal(1), rounding=mode, context=CTX) * (Decimal(10) ** -n)


def _round_float(x, n):
    if not math.isfinite(x): return x
    m = 10.0 ** n
    y = x * m
    r = math.floor(abs(y) + 0.5) * math.copysign(1, y)
    return r / m if n else r


@scalar("round", _round_typer)
def _round(x, n=0):
    if isinstance(x, float): return _round_float(x, n)
    if isinstance(x, int):
        if n >= 0: return x
        m = 10 ** -n; q, r = divmod(abs(x), m); q += r * 2 >= m
        return (q * m) if x >= 0 else -(q * m)
    r = _quant(x, n, ROUND_HALF_UP)
    return r


def _floor_typer(args):
    t = args[0].t
    if t.id == "DECIMAL": return T.DECIMAL(t.p, 0), None
    return _num1(args)[0], None


@scalar("floor", _floor_typer)
def _floor(x):
    if isinstance(x, float): return float(math.floor(x)) if math.isfinite(x) else x
    if isinstance(x, int): return x
    return x.to_integral_value(rounding=ROUND_FLOOR)


@scalar("ceil ceiling", _floor_typer)
def _ceil(x):
    if isinstance(x, float): return float(math.ceil(x)) if math.isfinite(x) else x
    if isinstance(x, int): return x
    return x.to_integral_value(rounding=ROUND_CEILING)


@scalar("trunc truncate", _round_typer)
def _trunc(x, n=0):
    if isinstance(x, float):
        if not math.isfinite(x): return x
        m = 10.0 ** n; return math.trunc(x * m) / m if n else float(math.trunc(x))
    if isinstance(x, int):
        if n >= 0: return x
        m = 10 ** -n; return (abs(x) // m * m) * (1 if x >= 0 else -1)
    return _quant(x, n, ROUND_DOWN)


@scalar("even", _dbl())
def _even(x):
    v = math.ceil(abs(x)); v += v % 2
    return math.copysign(v, x)


@scalar("gcd", lambda a: (T.BIGINT, None))
def _gcd(a, b): return math.gcd(a, b)


@scalar("lcm", lambda a: (T.BIGINT, None))
def _lcm(a, b): return abs(a * b) // math.gcd(a, b) if a and b else 0


@scalar("factorial", lambda a: (T.HUGEINT, None))
def _fact(n): return math.factorial(n) if n >= 0 else 1


@scalar("bit_count", lambda a: (T.TINYINT, None))
def _bitcount(x): return bin(x & ((1 << 64) - 1) if x < 0 else x).count("1")


def _int2(args): return T._int_supertype(args[0].t if args[0].t.integer else T.INTEGER, args[1].t if args[1].t.integer else T.INTEGER), None


scalar("&", _int2)(lambda a, b: a & b)
scalar("|", _int2)(lambda a, b: a | b)
scalar("xor", _int2)(lambda a, b: a ^ b)
scalar("<<", _int2)(lambda a, b: a << b)
scalar(">>", _int2)(lambda a, b: a >> b)
scalar("__bitnot", same_as())(lambda a: ~a)


@scalar("width_bucket", lambda a: (T.BIGINT, [T.DOUBLE, T.DOUBLE, T.DOUBLE, T.BIGINT]))
def _width_bucket(x, lo, hi, n):
    """ISO WIDTH_BUCKET: n equal-width buckets over [lo, hi); 0 below, n + 1 above (lo > hi: descending buckets)"""
    if n <= 0: return Err("WIDTH_BUCKET: the number of buckets must be positive")
    if lo == hi: return Err("WIDTH_BUCKET: the bounds must differ")
    if lo < hi:
        if x < lo: return 0
        if x >= hi: return n + 1
        return int((x - lo) * n / (hi - lo)) + 1
    if x > lo: return 0
    if x <= hi: return n + 1
    return int((lo - x) * n / (lo - hi)) + 1


# ------------------------------------------------------------------ strings
def _str(n=None, out=T.VARCHAR, argt=None):
    def ty(args):
        if n is not None and len(args) not in (n if isinstance(n, tuple) else (n,)): raise SqlError(f"wrong number of arguments")
        casts = []
        for i, a in enumerate(args):
            want = (argt[i] if argt and i < len(argt) else T.VARCHAR)
            casts.append(want if a.t != want and (want.id != "VARCHAR" or a.t.id != "VARCHAR") else None)
        return out, casts
    return ty


scalar("length len char_length character_length", _str(1, T.BIGINT))(lambda s: len(s))
scalar("strlen octet_length", _str(1, T.BIGINT))(lambda s: len(s.encode("utf-8")))
scalar("bit_length", _str(1, T.BIGINT))(lambda s: 8 * len(s.encode("utf-8")))
scalar("upper ucase", _str(1))(lambda s: s.replace("ß", "ẞ").upper())
scalar("lower lcase", _str(1))(lambda s: s.lower())
scalar("reverse", _str(1))(lambda s: s[::-1])
scalar("ascii", _str(1, T.INTEGER))(lambda s: ord(s[0]) if s else 0)
scalar("chr", lambda a: (T.VARCHAR, [T.INTEGER]))(lambda n: chr(n) if 0 <= n < 0x110000 else Err(f"chr: invalid code point {n}"))
scalar("initcap", _str(1))(lambda s: s.title())


def _concat_typer(args): return T.VARCHAR, [T.VARCHAR if a.t.id != "VARCHAR" else None for a in args]


@scalar("||", lambda a: (T.VARCHAR if all(x.t.id != "NULL" for x in a) else T.NULL, [T.VARCHAR if x.t.id not in ("VARCHAR", "NULL") else None for x in a]))
def _cat(a, b): return a + b


@scalar("concat", _concat_typer, nulls="custom")
def _concat(*xs): return "".join(x for x in xs if x is not None)


@scalar("concat_ws", _concat_typer, nulls="custom")
def _concat_ws(sep, *xs):
    if sep is None: return None
    return sep.join(x for x in xs if x is not None)


@scalar("substring substr", _str((2, 3), argt=[T.VARCHAR, T.BIGINT, T.BIGINT]))
def _substring(s, start, n=None):
    L = len(s)
    if start < 0: start = L + start + 1
    if n is None: a, b = start, L + 1
    elif n >= 0: a, b = start, start + n
    else: a, b = start + n, start
    a = max(a, 1); b = min(b, L + 1)
    return s[a - 1:b - 1] if b > a else ""


scalar("left", _str(2, argt=[T.VARCHAR, T.BIGINT]))(lambda s, n: s[:n] if n >= 0 else s[:max(0, len(s) + n)])
scalar("right", _str(2, argt=[T.VARCHAR, T.BIGINT]))(lambda s, n: (s[len(s) - n:] if n < len(s) else s) if n >= 0 else s[-n:])
scalar("position strpos instr", lambda a: (T.BIGINT, [T.VARCHAR, T.VARCHAR]))(lambda s, sub: s.find(sub) + 1)


def _trim_chars(chars): return " " if chars is None else chars


scalar("trim btrim", _str((1, 2)))(lambda s, c=None: s.strip(_trim_chars(c)))
scalar("ltrim", _str((1, 2)))(lambda s, c=None: s.lstrip(_trim_chars(c)))
scalar("rtrim", _str((1, 2)))(lambda s, c=None: s.rstrip(_trim_chars(c)))


def _pad(s, n, fill, left):
    if n <= len(s): return s[:max(0, n)]
    if not fill: return Err("Insufficient padding in RPAD/LPAD")
    k = n - len(s); p = (fill * (k // len(fill) + 1))[:k]
    return p + s if left else s + p


scalar("lpad", _str((2, 3), argt=[T.VARCHAR, T.INTEGER, T.VARCHAR]))(lambda s, n, f=" ": _pad(s, n, f, True))
scalar("rpad", _str((2, 3), argt=[T.VARCHAR, T.INTEGER, T.VARCHAR]))(lambda s, n, f=" ": _pad(s, n, f, False))
scalar("repeat", _str(2, argt=[T.VARCHAR, T.BIGINT]))(lambda s, n: s * max(0, n))
scalar("replace", _str(3))(lambda s, a, b: s.replace(a, b) if a else s)
scalar("translate", _str(3))(lambda s, a, b: "".join((b[a.index(ch)] if a.index(ch) < len(b) else "") if ch in a else ch for ch in s))
scalar("contains", _str(2, T.BOOLEAN))(lambda s, t: t in s)
scalar("starts_with prefix", _str(2, T.BOOLEAN))(lambda s, t: s.startswith(t))
scalar("ends_with suffix", _str(2, T.BOOLEAN))(lambda s, t: s.endswith(t))
scalar("split_part", _str(3, argt=[T.VARCHAR, T.VARCHAR, T.BIGINT]))(
    lambda s, d, i: (lambda parts: parts[i - 1] if 0 < i <= len(parts) else (parts[i] if -len(parts) <= i < 0 else ""))(s.split(d) if d else list(s)))


@scalar("overlay", _str((3, 4), argt=[T.VARCHAR, T.VARCHAR, T.BIGINT, T.BIGINT]))
def _overlay(s, r, start, n=None):
    """ISO OVERLAY(s PLACING r FROM start [FOR n]) = SUBSTRING(s FROM 1 FOR start - 1) || r || SUBSTRING(s FROM start + n)"""
    if n is None: n = len(r)
    return _iso_sub(s, 1, start - 1) + r + _iso_sub(s, start + n, None)


def _iso_sub(s, start, n):
    """ISO SUBSTRING(s FROM start [FOR n]): positions start .. start + n - 1, intersected with the string"""
    if n is not None and n < 0: return Err("SUBSTRING: negative length")
    a = max(start, 1); b = len(s) + 1 if n is None else min(start + n, len(s) + 1)
    return s[a - 1:b - 1] if b > a else ""


scalar("__iso_substring", _str((2, 3), argt=[T.VARCHAR, T.BIGINT, T.BIGINT]))(lambda s, a, n=None: _iso_sub(s, a, n))


@scalar("normalize", _str((1, 2)))
def _normalize(s, form="NFC"): return unicodedata.normalize(str(form).upper(), s)


# LIKE / SIMILAR / regular expressions
_like_cache = {}


def like_regex(p, esc=None, ci=False):
    k = (p, esc, ci)
    r = _like_cache.get(k)
    if r is None:
        out, i = [], 0
        while i < len(p):
            ch = p[i]
            if esc is not None and ch == esc:
                i += 1
                if i >= len(p): raise SqlError("LIKE pattern must not end with the escape character")
                out.append(re.escape(p[i]))
            elif ch == "%": out.append(".*")
            elif ch == "_": out.append(".")
            else: out.append(re.escape(ch))
            i += 1
        r = _like_cache[k] = re.compile("".join(out), re.S | (re.I if ci else 0))
    return r


def _esc(e):
    if e is None: return None
    if len(e) > 1: raise SqlError("the LIKE escape must be one character")
    return e or None


scalar("like ~~", _str((2, 3), T.BOOLEAN))(lambda s, p, e=None: like_regex(p, _esc(e)).fullmatch(s) is not None)
scalar("like_escape", _str(3, T.BOOLEAN))(lambda s, p, e: like_regex(p, _esc(e)).fullmatch(s) is not None)
scalar("not_like !~~", _str((2, 3), T.BOOLEAN))(lambda s, p, e=None: like_regex(p, _esc(e)).fullmatch(s) is None)
scalar("ilike ~~*", _str((2, 3), T.BOOLEAN))(lambda s, p, e=None: like_regex(p, _esc(e), True).fullmatch(s) is not None)
scalar("ilike_escape", _str(3, T.BOOLEAN))(lambda s, p, e: like_regex(p, _esc(e), True).fullmatch(s) is not None)
scalar("not_ilike !~~*", _str((2, 3), T.BOOLEAN))(lambda s, p, e=None: like_regex(p, _esc(e), True).fullmatch(s) is None)


def _glob_re(p):
    out, i = [], 0
    while i < len(p):
        ch = p[i]
        if ch == "*": out.append(".*")
        elif ch == "?": out.append(".")
        elif ch == "[":
            j = p.find("]", i + 1)
            if j < 0: out.append(re.escape(ch))
            else:
                body = p[i + 1:j]
                if body.startswith("!"): body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]"); i = j
        elif ch == "\\" and i + 1 < len(p): i += 1; out.append(re.escape(p[i]))
        else: out.append(re.escape(ch))
        i += 1
    return re.compile("".join(out), re.S)


scalar("glob ~~~", _str(2, T.BOOLEAN))(lambda s, p: _glob_re(p).fullmatch(s) is not None)

_re_cache = {}


def _rx(p, flags=""):
    k = (p, flags)
    r = _re_cache.get(k)
    if r is None:
        f = 0
        for c in flags or "":
            f |= {"i": re.I, "s": re.S, "m": re.M, "x": re.X, "c": 0, "g": 0, "n": re.S, "l": 0, "p": 0, "w": 0}.get(c, 0)
        try: r = _re_cache[k] = re.compile(_posix_to_py(p), f)
        except re.error as e: raise SqlError(f"invalid regular expression {p!r}: {e}")
    return r


def _posix_to_py(p):
    return (p.replace("[[:alpha:]]", "[^\\W\\d_]").replace("[[:digit:]]", "\\d").replace("[[:alnum:]]", "[^\\W_]")
             .replace("[[:space:]]", "\\s").replace("[[:upper:]]", "[A-Z]").replace("[[:lower:]]", "[a-z]")
             .replace("[[:punct:]]", "[!-/:-@\\[-`{-~]"))


scalar("regexp_full_match", _str((2, 3), T.BOOLEAN))(lambda s, p, f="": _rx(p, f).fullmatch(s) is not None)
scalar("regexp_matches ~", _str((2, 3), T.BOOLEAN))(lambda s, p, f="": _rx(p, f).search(s) is not None)
scalar("!~", _str(2, T.BOOLEAN))(lambda s, p: _rx(p).search(s) is None)


@scalar("regexp_replace", _str((3, 4)))
def _regexp_replace(s, p, r, f=""):
    rr = re.sub(r"\\(\d)", r"\\g<\1>", r)
    return _rx(p, f).sub(rr, s, count=0 if "g" in (f or "") else 1)


@scalar("regexp_extract", _str((2, 3, 4), argt=[T.VARCHAR, T.VARCHAR, T.INTEGER, T.VARCHAR]))
def _regexp_extract(s, p, g=0, f=""):
    m = _rx(p, f).search(s)
    if not m: return ""
    return m.group(g) or ""


def similar_to_regex(p, esc=None):
    """an ISO SIMILAR TO pattern as a regular expression: % _ | * + ? {m,n} ( ) [ ] with an optional escape"""
    out, i = [], 0
    while i < len(p):
        ch = p[i]
        if esc is not None and ch == esc:
            i += 1
            if i >= len(p): raise SqlError("SIMILAR pattern must not end with the escape character")
            out.append(re.escape(p[i]))
        elif ch == "%": out.append(".*")
        elif ch == "_": out.append(".")
        elif ch in "|*+?(){},": out.append(ch)
        elif ch == "[":
            j = i + 1
            if j < len(p) and p[j] in "^!": j += 1
            if j < len(p) and p[j] == "]": j += 1
            while j < len(p) and p[j] != "]": j += 1
            body = p[i + 1:j]
            if body.startswith("!"): body = "^" + body[1:]
            out.append("[" + body.replace("\\", "\\\\") + "]"); i = j
        else: out.append(re.escape(ch))
        i += 1
    return "".join(out)


@scalar("__iso_similar", _str((2, 3), T.BOOLEAN))
def _iso_similar(s, p, esc=None):
    return re.fullmatch(similar_to_regex(p, _esc(esc)), s, re.S) is not None


@scalar("__iso_substring_similar", _str(3))
def _iso_substring_similar(s, p, esc):
    """ISO SUBSTRING(s SIMILAR p ESCAPE e): p = a <esc>" b <esc>" c; the part matched by b when a b c matches s"""
    if esc is None or len(esc) != 1: return Err("SUBSTRING SIMILAR needs a one-character escape")
    parts = p.split(esc + '"')
    if len(parts) != 3: return Err("SUBSTRING SIMILAR: the pattern needs two escaped double quotes")
    rx = "(?:" + similar_to_regex(parts[0], esc) + ")(" + similar_to_regex(parts[1], esc) + ")(?:" + similar_to_regex(parts[2], esc) + ")"
    m = re.fullmatch(rx, s, re.S)
    return m.group(1) if m else None


def _xq_flags(f): return f or ""


@scalar("__iso_like_regex", _str((2, 3), T.BOOLEAN))
def _iso_like_regex(s, p, f=""): return _rx(p, _xq_flags(f)).search(s) is not None


@scalar("occurrences_regex __iso_occurrences_regex", _str((2, 3, 4, 5), T.BIGINT, argt=[T.VARCHAR, T.VARCHAR, T.VARCHAR, T.BIGINT, T.BOOLEAN]))
def _occ_regex(p, s, f="", start=1, octets=False):
    return sum(1 for _ in _rx(p, _xq_flags(f)).finditer(s, _char_start(s, start, octets)))


def _char_start(s, start, octets):
    """a FROM position in characters (USING OCTETS: in octets of the UTF-8 text)"""
    if not octets: return max(0, start - 1)
    b = s.encode("utf-8")[:max(0, start - 1)]
    return len(b.decode("utf-8", "ignore"))


def _out_pos(s, i, octets):
    return (len(s[:i].encode("utf-8")) if octets else i) + 1


@scalar("position_regex __iso_position_regex", _str((2, 3, 4, 5, 6, 7, 8), T.BIGINT, argt=[T.VARCHAR, T.VARCHAR, T.VARCHAR, T.BIGINT, T.BIGINT, T.BIGINT, T.BOOLEAN, T.BOOLEAN]))
def _pos_regex(p, s, f="", start=1, occ=1, group=0, after=False, octets=False):
    """POSITION_REGEX([START | AFTER] p IN s [FROM n] [USING OCTETS] [OCCURRENCE k] [GROUP g]): 0 when not found"""
    rx = _rx(p, _xq_flags(f))
    if group > rx.groups: return Err(f"POSITION_REGEX: the pattern has no group {group}")
    for i, m in enumerate(rx.finditer(s, _char_start(s, start, octets)), 1):
        if i == occ:
            if m.start(group) < 0: return 0
            return _out_pos(s, m.end(group) if after else m.start(group), octets)
    return 0


@scalar("substring_regex __iso_substring_regex", _str((2, 3, 4, 5, 6, 7), argt=[T.VARCHAR, T.VARCHAR, T.VARCHAR, T.BIGINT, T.BIGINT, T.BIGINT, T.BOOLEAN]))
def _sub_regex(p, s, f="", start=1, occ=1, group=0, octets=False):
    rx = _rx(p, _xq_flags(f))
    if group > rx.groups: return Err(f"SUBSTRING_REGEX: the pattern has no group {group}")
    for i, m in enumerate(rx.finditer(s, _char_start(s, start, octets)), 1):
        if i == occ: return m.group(group)
    return None


@scalar("__iso_translate_regex", _str((2, 3, 4, 5), argt=[T.VARCHAR, T.VARCHAR, T.VARCHAR, T.VARCHAR, T.BIGINT]))
def _tr_regex(p, s, r="", f="", occ=0):
    rx = _rx(p, _xq_flags(f)); rr = re.sub(r"\$(\d)", r"\\g<\1>", r or "")
    if not occ: return rx.sub(rr, s)
    out, n = [], 0
    def rep(m):
        nonlocal n; n += 1
        return m.expand(rr) if n == occ else m.group(0)
    return rx.sub(rep, s)


# ------------------------------------------------------------------ dates and times
def _dpart_typer(args):
    f = args[0].v.lower() if hasattr(args[0], "v") and isinstance(args[0].v, str) else ""
    if f in ("epoch", "julian"): return T.DOUBLE, None
    return T.BIGINT, None


def _week(d): return d.isocalendar()[1]


def _parts(f, d):
    f = f.lower()
    if isinstance(d, Interval):
        if f in ("year", "years", "y"): return d.months // 12
        if f in ("month", "months", "mon"): return d.months % 12
        if f in ("day", "days", "d"): return d.days
        if f in ("hour", "hours", "h"): return d.micros // 3_600_000_000
        if f in ("minute", "minutes", "min", "m"): return d.micros // 60_000_000 % 60
        if f in ("second", "seconds", "sec", "s"): return d.micros // 1_000_000 % 60
        if f == "epoch": return (d.months * 30 * 86400 + d.days * 86400) + d.micros / 1e6
        raise SqlError(f"EXTRACT({f}) from an interval is not supported")
    if f in ("timezone", "timezone_hour", "timezone_minute"):
        # ISO: the displacement of the value (0 for a value without time zone)
        off = d.utcoffset() if isinstance(d, (_dt.datetime, _dt.time)) and d.tzinfo is not None else None
        us = 0 if off is None else off // _dt.timedelta(microseconds=1)
        sg = -1 if us < 0 else 1; us = abs(us)
        if f == "timezone": return sg * us // 1_000_000
        if f == "timezone_hour": return sg * (us // 3_600_000_000)
        return sg * (us // 60_000_000 % 60)
    if isinstance(d, _dt.time):
        names = {"hour": "hour", "hours": "hour", "h": "hour", "hr": "hour", "minute": "minute", "minutes": "minute", "min": "minute",
                 "m": "minute", "second": "second", "seconds": "second", "sec": "second", "s": "second", "millisecond": "millisecond",
                 "milliseconds": "millisecond", "ms": "millisecond", "microsecond": "microsecond", "microseconds": "microsecond",
                 "us": "microsecond", "epoch": "epoch"}
        if f not in names: raise SqlError(f"EXTRACT({f}) from a time is not supported")
        return {"hour": d.hour, "minute": d.minute, "second": d.second, "millisecond": d.second * 1000 + d.microsecond // 1000,
                "microsecond": d.second * 1_000_000 + d.microsecond, "epoch": d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6}[names[f]]
    dd = d.date() if isinstance(d, _dt.datetime) else d
    if f in ("year", "years", "yr", "y"): return d.year
    if f in ("month", "months", "mon"): return d.month
    if f in ("day", "days", "d", "dayofmonth"): return d.day
    if f in ("quarter",): return (d.month - 1) // 3 + 1
    if f in ("dow", "dayofweek", "weekday"): return (dd.weekday() + 1) % 7
    if f in ("isodow",): return dd.isoweekday()
    if f in ("doy", "dayofyear"): return dd.timetuple().tm_yday
    if f in ("week", "weeks", "weekofyear", "w"): return _week(dd)
    if f in ("isoyear",): return dd.isocalendar()[0]
    if f in ("yearweek",): y, w, _ = dd.isocalendar(); return y * 100 + w
    if f in ("decade",): return d.year // 10
    if f in ("century",): return (d.year - 1) // 100 + 1 if d.year > 0 else d.year // 100
    if f in ("millennium",): return (d.year - 1) // 1000 + 1 if d.year > 0 else d.year // 1000
    if f in ("era",): return 1 if d.year > 0 else 0
    if f in ("julian",): return float(dd.toordinal() + 1721425)
    tm = d if isinstance(d, _dt.datetime) else _dt.datetime(d.year, d.month, d.day)
    if f in ("hour", "hours", "h", "hr"): return tm.hour
    if f in ("minute", "minutes", "min", "m"): return tm.minute
    if f in ("second", "seconds", "sec", "s"): return tm.second
    if f in ("millisecond", "milliseconds", "ms", "msec"): return tm.second * 1000 + tm.microsecond // 1000
    if f in ("microsecond", "microseconds", "us", "usec"): return tm.second * 1_000_000 + tm.microsecond
    if f in ("epoch",):
        e = tm - (_dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) if tm.tzinfo is not None else _dt.datetime(1970, 1, 1))
        return e.days * 86400 + e.seconds + e.microseconds / 1e6
    raise SqlError(f"unknown date part {f!r}")


@scalar("date_part datepart", _dpart_typer)
def _date_part(f, d): return _parts(f, d)


def _part_fn(f):
    return lambda d: _parts(f, d)


for _f in ("year", "month", "day", "quarter", "dayofweek", "isodow", "dayofyear", "week", "weekofyear", "hour", "minute", "second",
           "millisecond", "microsecond", "decade", "century", "millennium", "isoyear", "yearweek", "era", "dayofmonth"):
    scalar(_f, fixed(T.BIGINT))(_part_fn(_f))
scalar("epoch", fixed(T.DOUBLE))(_part_fn("epoch"))
scalar("dayname", fixed(T.VARCHAR))(lambda d: calendar.day_name[d.weekday()])
scalar("monthname", fixed(T.VARCHAR))(lambda d: calendar.month_name[d.month])


@scalar("last_day", fixed(T.DATE))
def _last_day(d): return _dt.date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


@scalar("make_date", lambda a: (T.DATE, [T.BIGINT, T.BIGINT, T.BIGINT]))
def _make_date(y, m, d):
    try: return _dt.date(y, m, d)
    except ValueError: return Err(f"Date out of range: {y}-{m}-{d}")


@scalar("make_timestamp", lambda a: (T.TIMESTAMP, [T.BIGINT] * 5 + [T.DOUBLE]))
def _make_ts(y, mo, d, h, mi, s):
    try: return _dt.datetime(y, mo, d, h, mi, int(s), int(round((s - int(s)) * 1e6)))
    except ValueError: return Err("Timestamp out of range")


@scalar("make_time", lambda a: (T.TIME, [T.BIGINT, T.BIGINT, T.DOUBLE]))
def _make_time(h, mi, s): return _dt.time(h, mi, int(s), int(round((s - int(s)) * 1e6)))


def _trunc_dt(f, d):
    if isinstance(d, _dt.datetime) and d.tzinfo is not None:        # zoned: truncate its local time, keep its zone
        return T.fixed(_trunc_dt(f, d.replace(tzinfo=None)).replace(tzinfo=d.tzinfo))
    f = f.lower(); t = _to_dt(d)
    if f in ("millennium",): return _dt.datetime((t.year // 1000) * 1000, 1, 1)
    if f in ("century",): return _dt.datetime((t.year // 100) * 100, 1, 1)
    if f in ("decade",): return _dt.datetime((t.year // 10) * 10, 1, 1)
    if f in ("year", "years", "yr", "y"): return _dt.datetime(t.year, 1, 1)
    if f in ("quarter",): return _dt.datetime(t.year, (t.month - 1) // 3 * 3 + 1, 1)
    if f in ("month", "months", "mon"): return _dt.datetime(t.year, t.month, 1)
    if f in ("week", "weeks", "w"): x = t - _dt.timedelta(days=t.weekday()); return _dt.datetime(x.year, x.month, x.day)
    if f in ("day", "days", "d", "dayofmonth"): return _dt.datetime(t.year, t.month, t.day)
    if f in ("hour", "hours", "h"): return t.replace(minute=0, second=0, microsecond=0)
    if f in ("minute", "minutes", "min", "m"): return t.replace(second=0, microsecond=0)
    if f in ("second", "seconds", "sec", "s"): return t.replace(microsecond=0)
    if f in ("millisecond", "milliseconds", "ms"): return t.replace(microsecond=t.microsecond // 1000 * 1000)
    if f in ("microsecond", "microseconds", "us"): return t
    raise SqlError(f"unknown date part {f!r}")


@scalar("date_trunc datetrunc", lambda a: (T.TIMESTAMPTZ if a[1].t.zoned else T.TIMESTAMP, None))
def _date_trunc(f, d): return _trunc_dt(f, d)


def _months_between(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def _dd_t(a):
    if a[1].t.zoned or a[2].t.zoned: return T.TIMESTAMPTZ
    return T.TIMESTAMP if a[1].t.id == "TIMESTAMP" or a[2].t.id == "TIMESTAMP" else None


@scalar("date_diff datediff", lambda a: (T.BIGINT, [None, _dd_t(a), _dd_t(a)]))
def _date_diff(f, a, b):
    f = f.lower(); A, B = _to_dt(a), _to_dt(b)
    if f in ("year", "years", "yr", "y"): return B.year - A.year
    if f in ("quarter",): return (B.year * 4 + (B.month - 1) // 3) - (A.year * 4 + (A.month - 1) // 3)
    if f in ("month", "months", "mon"): return _months_between(A, B)
    if f in ("day", "days", "d", "dayofmonth", "doy", "dayofyear"): return (B.date() - A.date()).days
    if f in ("week", "weeks", "w"): return (B.date() - A.date()).days // 7 if B >= A else -((A.date() - B.date()).days // 7)
    if f in ("decade",): return B.year // 10 - A.year // 10
    if f in ("century",): return B.year // 100 - A.year // 100
    us = (B - A) // _dt.timedelta(microseconds=1)
    for k, unit in (("hour", 3_600_000_000), ("hours", 3_600_000_000), ("h", 3_600_000_000), ("minute", 60_000_000), ("minutes", 60_000_000),
                    ("min", 60_000_000), ("m", 60_000_000), ("second", 1_000_000), ("seconds", 1_000_000), ("s", 1_000_000),
                    ("sec", 1_000_000), ("millisecond", 1000), ("milliseconds", 1000), ("ms", 1000), ("microsecond", 1), ("microseconds", 1), ("us", 1)):
        if f == k:
            hb = (B.replace(microsecond=0) if unit >= 1_000_000 else B); ha = (A.replace(microsecond=0) if unit >= 1_000_000 else A)
            if unit == 3_600_000_000: hb, ha = B.replace(minute=0, second=0, microsecond=0), A.replace(minute=0, second=0, microsecond=0)
            if unit == 60_000_000: hb, ha = B.replace(second=0, microsecond=0), A.replace(second=0, microsecond=0)
            return ((hb - ha) // _dt.timedelta(microseconds=1)) // unit
    raise SqlError(f"unknown date part {f!r}")


@scalar("date_add", lambda a: (T.TIMESTAMP if a[1].t.id == "INTERVAL" else T.DATE, None))
def _date_add(d, x): return _add(d, x)


@scalar("date_sub", lambda a: (T.BIGINT, None))
def _date_sub(f, a, b):
    f = f.lower(); A, B = _to_dt(a), _to_dt(b)
    if f in ("month", "months", "mon", "year", "years", "quarter", "decade", "century"):
        m = _months_between(A, B)
        if m > 0 and (B.day, B.time()) < (A.day, A.time()): m -= 1
        if m < 0 and (B.day, B.time()) > (A.day, A.time()): m += 1
        return {"year": m // 12, "years": m // 12, "quarter": m // 3, "decade": m // 120, "century": m // 1200}.get(f, m) if m >= 0 else -((-m) // {"year": 12, "years": 12, "quarter": 3, "decade": 120, "century": 1200}.get(f, 1))
    us = (B - A) // _dt.timedelta(microseconds=1)
    unit = {"day": 86_400_000_000, "days": 86_400_000_000, "week": 7 * 86_400_000_000, "hour": 3_600_000_000, "minute": 60_000_000,
            "second": 1_000_000, "millisecond": 1000, "microsecond": 1}.get(f)
    if unit is None: raise SqlError(f"unknown date part {f!r}")
    return int(us / unit)


@scalar("age", lambda a: (T.INTERVAL, [T.TIMESTAMP] * len(a)))
def _age(a, b=None):
    if b is None: b = _dt.datetime.combine(_dt.date.today(), _dt.time())
    sign = 1
    if a < b: a, b, sign = b, a, -1
    months = _months_between(b, a); x = T.add_months(b, months)
    if x > a: months -= 1; x = T.add_months(b, months)
    d = a - x
    return Interval(sign * months, sign * d.days, sign * (d.seconds * 1_000_000 + d.microseconds))


_STRF = {"%Y": lambda d: f"{d.year:04d}", "%m": lambda d: f"{d.month:02d}", "%d": lambda d: f"{d.day:02d}"}


@scalar("strftime", lambda a: (T.VARCHAR, None))
def _strftime(d, fmt): return _to_dt(d).strftime(fmt)


@scalar("strptime", lambda a: (T.TIMESTAMP, None))
def _strptime(s, fmt):
    try: return _dt.datetime.strptime(s, fmt)
    except ValueError as e: return Err(f"Could not parse {s!r} with {fmt!r}: {e}")


def _to_iv(months=0, days=0, micros=0):
    return lambda n: Interval(int(n) * months, int(n) * days if days else 0, int(round(n * micros)) if micros else 0)


for _n, _iv in (("to_years", _to_iv(months=12)), ("to_months", _to_iv(months=1)), ("to_days", _to_iv(days=1)),
                ("to_weeks", _to_iv(days=7)), ("to_hours", _to_iv(micros=3_600_000_000)), ("to_minutes", _to_iv(micros=60_000_000)),
                ("to_seconds", _to_iv(micros=1_000_000)), ("to_milliseconds", _to_iv(micros=1000)), ("to_microseconds", _to_iv(micros=1)),
                ("to_decades", _to_iv(months=120)), ("to_centuries", _to_iv(months=1200)), ("to_millennia", _to_iv(months=12000))):
    scalar(_n, fixed(T.INTERVAL))(_iv)


_DT_FIELDS = ["DAY", "HOUR", "MINUTE", "SECOND"]
_DT_US = {"HOUR": 3_600_000_000, "MINUTE": 60_000_000, "SECOND": 1_000_000}
_DT_MAX = {"HOUR": 24, "MINUTE": 60, "SECOND": 60}


def _iv_text(text, qual):
    """an <unquoted interval string> under an interval qualifier (ISO 5.3): '1-2' YEAR TO MONTH, '3 04:05:06.5'
    DAY TO SECOND, '90' MINUTE, '-1:30' HOUR TO MINUTE, ..."""
    s = str(text).strip(); sign = 1
    if s[:1] in "+-":
        sign = -1 if s[0] == "-" else 1; s = s[1:].strip()
    q = qual.upper().split(); start, end = q[0], q[-1]
    bad = SqlError(f"invalid interval string {text!r} for INTERVAL {qual}")
    try:
        if start in ("YEAR", "MONTH"):
            if start == "YEAR" and end == "MONTH":
                y, m = s.split("-")
                if int(m) >= 12: raise bad
                return Interval(sign * (int(y) * 12 + int(m)))
            if end != start: raise bad
            return Interval(sign * int(s) * (12 if start == "YEAR" else 1))
        fields = _DT_FIELDS[_DT_FIELDS.index(start):_DT_FIELDS.index(end) + 1]
        if not fields: raise bad
        if start == "DAY" and len(fields) > 1:
            d, rest = s.split(None, 1); parts = [d] + rest.split(":")
        else: parts = s.split(":")
        if len(parts) != len(fields): raise bad
        days = us = 0
        for k, (f, p) in enumerate(zip(fields, parts)):
            v = Decimal(p) if f == "SECOND" else int(p)
            if v < 0 or (k > 0 and v >= _DT_MAX[f]): raise bad
            if f == "DAY": days = int(v)
            else: us += int(v * _DT_US[f])
        return Interval(0, sign * days, sign * us)
    except (ValueError, ArithmeticError): raise bad


@scalar("__iso_interval", fixed(T.INTERVAL))
def _iso_interval(text, qual):
    """INTERVAL '<text>' <qualifier> (YEAR TO MONTH, DAY TO SECOND, HOUR(3), ...)"""
    try: return _iv_text(text, qual)
    except SqlError as e: return Err(str(e))


@scalar("__iso_interval_cast", fixed(T.INTERVAL))
def _iso_interval_cast(x, qual):
    """CAST(x AS INTERVAL <qualifier>): a string in that form, or an exact number of the single field"""
    if isinstance(x, Interval): return x
    if isinstance(x, (int, Decimal, float)) and not isinstance(x, bool):
        q = qual.upper().split()
        if len(q) != 1: return Err(f"CAST of a number to INTERVAL {qual}: a single field is needed")
        f = q[0]
        if f in ("YEAR", "MONTH"):
            if x != int(x): return Err(f"CAST({x} AS INTERVAL {f}): not a whole number")
            return Interval(int(x) * (12 if f == "YEAR" else 1))
        if f == "DAY":
            d = Decimal(str(x)); di = int(d)
            return Interval(0, di, int((d - di) * 86_400_000_000))
        return Interval(0, 0, int(Decimal(str(x)) * _DT_US[f]))
    try: return _iv_text(str(x), qual)
    except SqlError as e: return Err(str(e))


@scalar("__iso_dt_diff", lambda a: (T.INTERVAL, [T.TIMESTAMPTZ if (a[0].t.zoned or a[1].t.zoned) else T.TIMESTAMP if (a[0].t.id == "TIMESTAMP" or a[1].t.id == "TIMESTAMP") else None,
                                                T.TIMESTAMPTZ if (a[0].t.zoned or a[1].t.zoned) else T.TIMESTAMP if (a[0].t.id == "TIMESTAMP" or a[1].t.id == "TIMESTAMP") else None, None]))
def _iso_dt_diff(a, b, qual):
    """(a - b) <qualifier>: the difference of two datetimes as an interval of those fields (ISO 6.42), the trailing
    field truncated toward zero"""
    q = qual.upper().split(); start, end = q[0], q[-1]
    if isinstance(a, _dt.time):
        A = _dt.datetime.combine(_dt.date(2000, 1, 3), a.replace(tzinfo=None)) - (a.utcoffset() or _dt.timedelta())
        B = _dt.datetime.combine(_dt.date(2000, 1, 3), b.replace(tzinfo=None)) - (b.utcoffset() or _dt.timedelta())
    else: A, B = _to_dt(a), _to_dt(b)
    if start in ("YEAR", "MONTH"):
        m = _months_between(B, A)
        if m > 0 and (A.day, A.time()) < (B.day, B.time()): m -= 1
        if m < 0 and (A.day, A.time()) > (B.day, B.time()): m += 1
        if end == "YEAR": m = int(m / 12) * 12
        return Interval(m)
    us = (A - B) // _dt.timedelta(microseconds=1)
    unit = {"DAY": 86_400_000_000, "HOUR": 3_600_000_000, "MINUTE": 60_000_000, "SECOND": 1}[end]
    us = int(us / unit) * unit if unit > 1 else us
    if start == "DAY":
        d = int(us / 86_400_000_000)
        return Interval(0, d, us - d * 86_400_000_000)
    return Interval(0, 0, us)


# the clock: one instant per statement (set by the executor), seen in the SQL-session time zone
CLOCK = {"now": None}


def _now():
    """CURRENT_TIMESTAMP: the statement's instant with the session's displacement"""
    if CLOCK["now"] is None: CLOCK["now"] = _dt.datetime.now(_dt.timezone.utc)
    return T.in_zone(CLOCK["now"])


scalar("current_date today", fixed(T.DATE), deterministic=False)(lambda: _now().date())
scalar("current_timestamp now get_current_timestamp transaction_timestamp", fixed(T.TIMESTAMPTZ), deterministic=False)(lambda: _now())
scalar("localtimestamp", fixed(T.TIMESTAMP), deterministic=False)(lambda: _now().replace(tzinfo=None))
scalar("current_time get_current_time", fixed(T.TIMETZ), deterministic=False)(lambda: _now().timetz())
scalar("localtime", fixed(T.TIME), deterministic=False)(lambda: _now().time())


def _at_typer(args):
    """x AT TIME ZONE z / x AT LOCAL (ISO 6.36): the same instant with another displacement, WITH TIME ZONE"""
    x = args[1]
    if x.t.id in ("TIME", "TIMETZ"): return T.TIMETZ, [None, T.TIMETZ]
    if x.t.id in ("TIMESTAMP", "TIMESTAMPTZ", "DATE") or _strlit(x): return T.TIMESTAMPTZ, [None, T.TIMESTAMPTZ]
    raise SqlError(f"AT TIME ZONE needs a datetime, not {x.t}")


@scalar("timezone __iso_at_zone", _at_typer)
def _at_zone(z, x): return T.in_zone(x, z)


@scalar("__frac_trunc", same_as())
def _frac_trunc(v, p):
    """a datetime with p fractional digits of seconds (TIMESTAMP(p), CURRENT_TIMESTAMP(p))"""
    q = 10 ** (6 - p)
    return v.replace(microsecond=v.microsecond // q * q)


def _at_local_typer(args):
    t, casts = _at_typer([None] + list(args))
    return t, casts[1:]                                # x is taken in the session's zone first when it has none


scalar("__iso_at_local", _at_local_typer, deterministic=False)(lambda x: T.in_zone(x))


# SQL-session values (ISO 6.4 <general value specification>)
def _user():
    import getpass
    try: return getpass.getuser()
    except Exception: return "torbles"


scalar("current_user user session_user system_user", fixed(T.VARCHAR), deterministic=False)(_user)
scalar("current_role", fixed(T.VARCHAR), nulls="custom", deterministic=False)(lambda: None)
scalar("current_catalog current_database", fixed(T.VARCHAR), deterministic=False)(lambda: "memory")
scalar("current_schema", fixed(T.VARCHAR), deterministic=False)(lambda: "main")
scalar("current_default_transform_group", fixed(T.VARCHAR), deterministic=False)(lambda: "")   # no user-defined types: no transform group
scalar("current_path", fixed(T.VARCHAR), deterministic=False)(lambda: '"main"')


# ------------------------------------------------------------------ binary strings (ISO 6.33 / 8.5: BLOB)
BINARY = {"||": "__b_concat", "concat": "__b_concat", "octet_length": "__b_octet_length", "length": "__b_octet_length",
          "char_length": "__b_octet_length", "strlen": "__b_octet_length", "substring": "__b_substring", "position": "__b_position",
          "overlay": "__b_overlay", "trim": "__b_trim", "btrim": "__b_trim", "ltrim": "__b_ltrim", "rtrim": "__b_rtrim",
          "lpad": "__b_lpad", "rpad": "__b_rpad",
          "like": "__b_like", "not_like": "__b_not_like", "starts_with": "__b_starts_with", "reverse": "__b_reverse"}


def _bin_args(n, out=T.BLOB, argt=None):
    def ty(args):
        if len(args) not in (n if isinstance(n, tuple) else (n,)): raise SqlError("wrong number of arguments")
        casts = []
        for i, a in enumerate(args):
            want = (argt[i] if argt and i < len(argt) else T.BLOB)
            casts.append(None if a.t == want else want)
        return out, casts
    return ty


scalar("__b_concat", _bin_args((2, 3, 4, 5, 6, 7, 8)))(lambda *xs: b"".join(xs))
scalar("__b_octet_length", _bin_args(1, T.BIGINT))(lambda b: len(b))
scalar("__b_reverse", _bin_args(1))(lambda b: b[::-1])


@scalar("__b_substring", _bin_args((2, 3), T.BLOB, [T.BLOB, T.BIGINT, T.BIGINT]))
def _b_substring(b, start, n=None):
    if n is not None and n < 0: return Err("SUBSTRING: negative length")
    lo = start - 1; hi = len(b) if n is None else lo + n
    return b[max(lo, 0):max(hi, 0)]


scalar("__b_position", _bin_args(2, T.BIGINT))(lambda b, sub: b.find(sub) + 1)


@scalar("__b_overlay", _bin_args((3, 4), T.BLOB, [T.BLOB, T.BLOB, T.BIGINT, T.BIGINT]))
def _b_overlay(b, r, start, n=None):
    if n is None: n = len(r)
    return b[:start - 1] + r + b[start - 1 + n:]


def _trim_octets(c): return b"\x00" if c is None else c


scalar("__b_trim", _bin_args((1, 2)))(lambda b, c=None: b.strip(_trim_octets(c)))
scalar("__b_ltrim", _bin_args((1, 2)))(lambda b, c=None: b.lstrip(_trim_octets(c)))
scalar("__b_rtrim", _bin_args((1, 2)))(lambda b, c=None: b.rstrip(_trim_octets(c)))
scalar("__b_starts_with", _bin_args(2, T.BOOLEAN))(lambda b, p: b.startswith(p))


def _b_pad(b, n, f, left):
    """LPAD / RPAD of a binary string (SQL:2023 T055): to n octets, the fill repeated (default X'00'), or cut"""
    if n < 0: return Err("LPAD / RPAD: a negative length")
    if len(b) >= n: return b[:n]
    if not f: return b
    fill = (f * (n - len(b)))[:n - len(b)]
    return fill + b if left else b + fill


scalar("__b_lpad", _bin_args((2, 3), T.BLOB, [T.BLOB, T.INTEGER, T.BLOB]))(lambda b, n, f=b"\x00": _b_pad(b, n, f, True))
scalar("__b_rpad", _bin_args((2, 3), T.BLOB, [T.BLOB, T.INTEGER, T.BLOB]))(lambda b, n, f=b"\x00": _b_pad(b, n, f, False))


def _b_like(b, p, e=None):
    """<octet like predicate>: X'25' (%) and X'5F' (_) are the wildcards"""
    return like_regex(p.decode("latin-1"), _esc(e.decode("latin-1") if isinstance(e, bytes) else e)).fullmatch(b.decode("latin-1")) is not None


scalar("__b_like", _bin_args((2, 3), T.BOOLEAN))(_b_like)
scalar("__b_not_like", _bin_args((2, 3), T.BOOLEAN))(lambda b, p, e=None: not _b_like(b, p, e))


@scalar("hex to_hex", lambda a: (T.VARCHAR, None))
def _hex(x):
    if isinstance(x, (bytes, bytearray)): return x.hex().upper()
    if isinstance(x, str): return x.encode("utf-8").hex().upper()
    if isinstance(x, int): return format(x & ((1 << 64) - 1) if x < 0 else x, "X")
    return Err(f"hex: unsupported value {x!r}")


@scalar("unhex from_hex", lambda a: (T.BLOB, [T.VARCHAR]))
def _unhex(s):
    try: return bytes.fromhex(s if len(s) % 2 == 0 else "0" + s)
    except ValueError: return Err(f"unhex: invalid hexadecimal {s!r}")


scalar("encode", lambda a: (T.BLOB, [T.VARCHAR]))(lambda s: s.encode("utf-8"))


@scalar("decode", lambda a: (T.VARCHAR, [T.BLOB]))
def _decode(b):
    try: return b.decode("utf-8")
    except UnicodeDecodeError: return Err("decode: the BLOB is not valid UTF-8")


# ------------------------------------------------------------------ ISO string forms
@scalar("__iso_pred_normalized", fixed(T.BOOLEAN))
def _is_normalized(x, form): return unicodedata.is_normalized(form.upper(), x)


scalar("__is_normalized", lambda a: (T.BOOLEAN, [T.VARCHAR, None]))(_is_normalized)


def _octets(s): return s.encode("utf-8")


def _from_octets(b, what):
    try: return b.decode("utf-8")
    except UnicodeDecodeError: raise SqlError(f"{what} USING OCTETS splits a character")


@scalar("__iso_position_octets", lambda a: (T.BIGINT, [T.VARCHAR, T.VARCHAR]))
def _pos_octets(a, b):
    """POSITION(a IN b USING OCTETS): the octet position (1-based, 0 when absent)"""
    return _octets(b).find(_octets(a)) + 1


@scalar("__iso_substring_octets", lambda a: (T.VARCHAR, [T.VARCHAR, T.BIGINT, T.BIGINT]))
def _substr_octets(s, start, n=None):
    b = _octets(s)
    lo = start - 1; hi = len(b) if n is None else lo + n
    if n is not None and n < 0: return Err("SUBSTRING: negative length")
    return _from_octets(b[max(lo, 0):max(hi, 0)], "SUBSTRING")


@scalar("__iso_overlay_octets", lambda a: (T.VARCHAR, [T.VARCHAR, T.VARCHAR, T.BIGINT, T.BIGINT]))
def _overlay_octets(s, r, start, n=None):
    b, rb = _octets(s), _octets(r)
    if n is None: n = len(rb)
    return _from_octets(b[:start - 1] + rb + b[start - 1 + n:], "OVERLAY")


_TRANSLIT = {"UTF8": "utf-8", "UTF-8": "utf-8", "SQL_TEXT": "utf-8", "SQL_CHARACTER": "utf-8", "UTF16": "utf-16", "UTF32": "utf-32",
             "UCS2": "utf-16", "LATIN1": "latin-1", "ISO8859_1": "latin-1", "ASCII": "ascii", "SQL_ASCII": "ascii", "WIN1252": "cp1252"}


@scalar("__iso_transliterate", lambda a: (T.VARCHAR, [T.VARCHAR, None]))
def _transliterate(x, name):
    """TRANSLATE(x USING <character set>): x when every character exists in that character set, else an error"""
    enc = _TRANSLIT.get(name.upper().replace("-", "_").replace("UTF_", "UTF"), _TRANSLIT.get(name.upper()))
    if enc is None: return Err(f"unknown transliteration {name}")
    try: x.encode(enc)
    except UnicodeEncodeError as e: return Err(f"TRANSLATE USING {name}: untranslatable character {x[e.start]!r}")
    return x


@scalar("row", lambda a: (T.ROW_T([("", x) for x in a]), None), nulls="custom")
def _row(*xs): return tuple(xs)


@scalar("__row_field", lambda a: (T.NULL, None))
def _row_field(x, i):
    """a field of a row value (ISO 6.15 <field reference>), by position"""
    return x[i]


# ------------------------------------------------------------------ CAST ... FORMAT '<datetime template>' (ISO T839)
def _fmt_template(v, tpl):
    """a datetime as text by an ISO datetime template (YYYY, MM, DD, HH24, HH12, MI, SS, FFn, A.M., TZH, TZM, ...)"""
    import re as _re
    out, i, T_ = [], 0, tpl.upper()
    d = v if not isinstance(v, _dt.time) else None
    tm = v if isinstance(v, (_dt.datetime, _dt.time)) else None
    toks = ["YYYY", "RRRR", "YYY", "YY", "RR", "Y", "MM", "DDD", "DD", "HH24", "HH12", "HH", "MI", "SSSSS", "SS",
            "FF1", "FF2", "FF3", "FF4", "FF5", "FF6", "FF7", "FF8", "FF9", "A.M.", "P.M.", "TZH", "TZM"]
    while i < len(tpl):
        for k in toks:
            if T_.startswith(k, i):
                if k in ("YYYY", "RRRR"): out.append(f"{d.year:04d}")
                elif k == "YYY": out.append(f"{d.year % 1000:03d}")
                elif k in ("YY", "RR"): out.append(f"{d.year % 100:02d}")
                elif k == "Y": out.append(f"{d.year % 10}")
                elif k == "MM": out.append(f"{d.month:02d}")
                elif k == "DDD": out.append(f"{d.timetuple().tm_yday:03d}")
                elif k == "DD": out.append(f"{d.day:02d}")
                elif k in ("HH24",): out.append(f"{tm.hour:02d}")
                elif k in ("HH12", "HH"): out.append(f"{(tm.hour % 12) or 12:02d}")
                elif k == "MI": out.append(f"{tm.minute:02d}")
                elif k == "SSSSS": out.append(f"{tm.hour * 3600 + tm.minute * 60 + tm.second}")
                elif k == "SS": out.append(f"{tm.second:02d}")
                elif k.startswith("FF"): out.append(f"{tm.microsecond:06d}".ljust(9, "0")[:int(k[2])])
                elif k in ("A.M.", "P.M."): out.append("P.M." if tm.hour >= 12 else "A.M.")
                elif k in ("TZH", "TZM"):
                    off = v.utcoffset() if getattr(v, "tzinfo", None) is not None else _dt.timedelta(0)
                    us = off // _dt.timedelta(microseconds=1)
                    out.append(("-" if us < 0 else "+") + f"{abs(us) // 3_600_000_000:02d}" if k == "TZH" else f"{abs(us) // 60_000_000 % 60:02d}")
                i += len(k); break
        else:
            if tpl[i] == '"':
                j = tpl.index('"', i + 1); out.append(tpl[i + 1:j]); i = j + 1
            else: out.append(tpl[i]); i += 1
    return "".join(out)


@scalar("__iso_cast_format", lambda a: (T.type_from_name(a[1].v), None))
def _cast_format(x, typ, tpl):
    import sqljson
    t = T.type_from_name(typ)
    if t.id == "VARCHAR":
        try: return _fmt_template(x, tpl)
        except (AttributeError, ValueError) as e: return Err(f"CAST ... FORMAT {tpl!r}: {e}")
    try: v = sqljson._template(tpl, x)
    except sqljson.JErr as e: return Err(f"Conversion Error: {e}")
    try: return T.cast_value(v, T.py_type(v), t)
    except SqlError as e: return Err(str(e))


# ------------------------------------------------------------------ collations
def _collate_typer(args):
    x, c = args
    if x.t.id not in ("VARCHAR", "NULL"): raise SqlError(f"COLLATE applies to character strings, not {x.t}")
    T.collation(c.v)                                  # an unknown collation fails here
    return T.with_coll(T.VARCHAR, c.v), None


scalar("__collate", _collate_typer)(lambda x, c: x)                 # the value itself; the type carries the collation


@scalar("__ckey", fixed(T.VARCHAR))
def _ckey(x, c):
    """the comparison key of a string under a collation"""
    f = T.collation(c)
    return f(x) if f is not None else x


# ------------------------------------------------------------------ others
@scalar("typeof", fixed(T.VARCHAR), nulls="custom")
def _typeof(x): return None              # replaced at bind time by the argument's type name


@scalar("__grp", same_as(0), nulls="custom")
def _grp(x): return x                     # a GROUP BY position that repeats an earlier one: a column of its own


@scalar("__first", same_as(0), nulls="custom")
def _first(x, *rest): return x           # typeof(an aggregate): the type name, while the query stays an aggregate


@scalar("__iso_overlaps", lambda a: (T.BOOLEAN, None), nulls="custom")
def _overlaps(s1, e1, s2, e2):
    """(s1, e1) OVERLAPS (s2, e2) by ISO 8.14: an interval second field is a duration; each pair is ordered (a NULL
    start takes the end's place), then (S1 > S2 AND NOT (S1 >= T2 AND T1 >= T2)) OR (S2 > S1 AND NOT (S2 >= T1 AND
    T2 >= T1)) OR (S1 = S2 AND (T1 <> T2 OR T1 = T2)) in three-valued logic"""
    if isinstance(e1, Interval): e1 = None if s1 is None else _add(s1, e1)
    if isinstance(e2, Interval): e2 = None if s2 is None else _add(s2, e2)
    def order(d, e):
        if d is None or (e is not None and cmp_values(e, d) < 0): return e, d
        return d, e
    S1, T1 = order(s1, e1); S2, T2 = order(s2, e2)
    def c(a, b, f): return None if a is None or b is None else f(cmp_values(a, b))
    gt = lambda a, b: c(a, b, lambda x: x > 0)
    ge = lambda a, b: c(a, b, lambda x: x >= 0)
    eq = lambda a, b: c(a, b, lambda x: x == 0)
    ne = lambda a, b: c(a, b, lambda x: x != 0)
    n = lambda x: None if x is None else not x
    return _or(_and(gt(S1, S2), n(_and(ge(S1, T2), ge(T1, T2)))),
               _and(gt(S2, S1), n(_and(ge(S2, T1), ge(T2, T1)))),
               _and(eq(S1, S2), _or(ne(T1, T2), eq(T1, T2))))


@scalar("list_value list_pack", lambda a: (T.LIST(T.supertype_all([x.t for x in a]) if a else T.INTEGER), None), nulls="custom")
def _list_value(*xs): return list(xs)


@scalar("len array_length list_length cardinality", lambda a: (T.BIGINT, None))
def _len(x): return len(x)


@scalar("list_extract array_extract", lambda a: (a[0].t.child if a[0].t.id == "LIST" else T.VARCHAR, None))
def _list_extract(x, i):
    if isinstance(x, str): return x[i - 1] if 0 < i <= len(x) else ""
    return x[i - 1] if 0 < i <= len(x) else (x[i] if -len(x) <= i < 0 else None)


# ------------------------------------------------------------------ arrays (ISO 6.24, 6.47) and multisets (6.43, 6.48)
def _coll(t): return t.child if t.id in ("LIST", "MULTISET") and t.child is not None else T.NULL


@scalar("__iso_array_elem", lambda a: (_coll(a[0].t), [None, T.BIGINT]))
def _array_elem(a, i):
    """a[i]: 1-based; an index outside 1 .. CARDINALITY(a) is an error (ISO: data exception - array element error)"""
    if not 1 <= i <= len(a): return Err(f"data exception: array element error (index {i}, cardinality {len(a)})")
    return a[i - 1]


@scalar("list_slice array_slice", lambda a: (a[0].t, [None, T.BIGINT, T.BIGINT]))
def _list_slice(x, a, b):
    """x[a:b] (DuckDB): 1-based, both ends included, negative positions count from the end"""
    n = len(x)
    lo = a - 1 if a > 0 else (n + a if a < 0 else 0); hi = b if b >= 0 else n + b + 1
    return x[max(lo, 0):max(hi, 0)]


@scalar("trim_array", lambda a: (a[0].t, [None, T.BIGINT]))
def _trim_array(a, n):
    if not 0 <= n <= len(a): return Err(f"data exception: array element error (TRIM_ARRAY by {n} of {len(a)})")
    return a[:len(a) - n]


@scalar("__list_concat", lambda a: (T.supertype_all([x.t for x in a]), None))
def _list_concat(*xs): return [v for x in xs for v in x]


def _ms_typer(args):
    t = args[0].t
    if t.id == "LIST": return T.MULTISET(t.child), None
    if t.id in ("MULTISET", "NULL"): return t if t.id == "MULTISET" else T.MULTISET(T.INTEGER), None
    raise SqlError(f"a multiset is made from a collection, not {t}")


scalar("__ms_from_list", _ms_typer)(lambda x: list(x))


def _mkeys(m):
    """element keys of a multiset (NULLs are not distinct from each other)"""
    from collections import Counter
    import sqlexec
    return Counter(sqlexec.nkey(v) for v in m)


@scalar("set __ms_set", lambda a: (a[0].t, None))
def _ms_set(m):
    import sqlexec
    seen, out = set(), []
    for v in m:
        k = sqlexec.nkey(v)
        if k not in seen: seen.add(k); out.append(v)
    return out


@scalar("element __ms_element", lambda a: (_coll(a[0].t), None))
def _ms_element(m):
    """ELEMENT(m): its only element; NULL when m is empty; an error past one (cardinality violation)"""
    if len(m) == 0: return None
    if len(m) > 1: return Err(f"data exception: cardinality violation (ELEMENT of {len(m)} elements)")
    return m[0]


@scalar("__ms_is_a_set", fixed(T.BOOLEAN))
def _ms_is_a_set(m): return all(c == 1 for c in _mkeys(m).values())


@scalar("__ms_member", lambda a: (T.BOOLEAN, [_coll(a[1].t) if a[1].t.id in ("LIST", "MULTISET") else None, None]), nulls="custom")
def _ms_member(x, m):
    """x MEMBER OF m (ISO 8.16)"""
    if m is None: return None
    if not m: return False
    if x is None: return None
    if any(v is not None and cmp_values(v, x) == 0 for v in m): return True
    return None if any(v is None for v in m) else False


@scalar("__ms_submultiset", fixed(T.BOOLEAN), nulls="custom")
def _ms_submultiset(a, b):
    """a SUBMULTISET OF b (ISO 8.17): each element of a matched to a distinct equal element of b"""
    if a is None or b is None: return None
    if len(a) > len(b): return False
    ka, kb = _mkeys([v for v in a if v is not None]), _mkeys([v for v in b if v is not None])
    if any(kb.get(k, 0) < c for k, c in ka.items()):
        return None if (any(v is None for v in b) and sum(max(0, c - kb.get(k, 0)) for k, c in ka.items()) <= sum(1 for v in b if v is None)) else False
    if any(v is None for v in a): return None
    return True


def _ms_op(op):
    def f(a, b, mode="ALL"):
        import sqlexec
        if op == "UNION":                                  # DISTINCT: SET(a UNION ALL b)
            return _ms_set(list(a) + list(b)) if mode.upper() == "DISTINCT" else list(a) + list(b)
        if mode.upper() == "DISTINCT": a, b = _ms_set(a), _ms_set(b)
        kb = _mkeys(b)
        out = []
        for v in a:
            k = sqlexec.nkey(v)
            if op == "EXCEPT":
                if kb.get(k, 0) > 0: kb[k] -= 1
                else: out.append(v)
            else:
                if kb.get(k, 0) > 0: kb[k] -= 1; out.append(v)
        return out
    return f


for _op in ("UNION", "EXCEPT", "INTERSECT"):
    scalar(f"__ms_{_op.lower()}", lambda a: (T.supertype(a[0].t, a[1].t), None))(_ms_op(_op))


@scalar("__ms_eq", fixed(T.BOOLEAN))
def _ms_eq(a, b):
    """multisets are equal when every element occurs as often in both (ISO 8.2)"""
    if any(v is None for v in a) or any(v is None for v in b):
        return None if len(a) == len(b) else False
    return _mkeys(a) == _mkeys(b)


# ------------------------------------------------------------------ SQL/JSON (sqljson)
_JOPTS = {}


def jopts(text):
    """the clauses of an SQL/JSON call (a JSON string made by the pre-pass and completed by the binder), compiled once"""
    o = _JOPTS.get(text)
    if o is None:
        import json as _json, sqljson
        o = _json.loads(text)
        if o.get("path") is not None: o["cpath"] = sqljson.compile_path(o["path"])
        o["rt"] = T.type_from_name(o["returning"]) if o.get("returning") else None
        o["ptypes_t"] = [T.type_from_name(x) for x in o.get("ptypes", [])]
        o["vtypes_t"] = [T.type_from_name(x) for x in o.get("vtypes", [])]
        _JOPTS[text] = o
    return o


def _jv(ctx, opts, *rest):
    import sqljson
    if ctx is None: return None
    o = jopts(opts); np_ = o.get("np", 0)
    return sqljson.json_value(o, ctx, list(rest[:np_]), o["ptypes_t"], list(rest[np_:]), o["rt"] or T.VARCHAR)


def _jq(ctx, opts, *rest):
    import sqljson
    if ctx is None: return None
    o = jopts(opts)
    return sqljson.json_query(o, ctx, list(rest), o["ptypes_t"])


def _je(ctx, opts, *rest):
    import sqljson
    if ctx is None: return None
    o = jopts(opts)
    return sqljson.json_exists(o, ctx, list(rest), o["ptypes_t"])


def _jobj(opts, *kv):
    import sqljson
    o = jopts(opts)
    return sqljson.json_object(kv[0::2], kv[1::2], o["vtypes_t"], o["fmts"], o.get("absent", False), o.get("unique", False))


def _jarr(opts, *vals):
    import sqljson
    o = jopts(opts)
    return sqljson.json_array(vals, o["vtypes_t"], o["fmts"], o.get("null_on_null", False))


def _jarr_list(vals, opts):
    import sqljson
    if vals is None: return None
    o = jopts(opts)
    return sqljson.json_array(vals, o["vtypes_t"] * len(vals), [False] * len(vals), False)


def _jparse(x, unique=False):
    import sqljson
    if isinstance(x, (bytes, bytearray)): x = x.decode("utf-8")
    try: return sqljson.dumps(sqljson.parse(x, bool(unique)))
    except sqljson.JErr as e: return Err(f"JSON: {e}")


def _jscalar(x, unique=False):
    import sqljson
    try: return sqljson.json_scalar(x, None)
    except SqlError as e: return Err(str(e))


def _jserialize(x, opts):
    o = jopts(opts)
    rt = o["rt"] or T.VARCHAR
    return x.encode("utf-8") if rt.id == "BLOB" else x


for _n, _f in (("__json_value", _jv), ("__json_query", _jq), ("__json_exists", _je), ("__json_object", _jobj),
               ("__json_array", _jarr), ("__json_array_list", _jarr_list), ("__json_parse json", _jparse),
               ("__json_scalar", _jscalar), ("__json_serialize", _jserialize)):
    scalar(_n, lambda a: (T.JSON, None), nulls="custom")(_f)


@scalar("__iso_is_json", fixed(T.BOOLEAN))
def _is_json(x, kind, unique):
    import sqljson
    return sqljson.is_json(x, kind, bool(unique))


# ================================================================== aggregates
class AggDef:
    """an aggregate function. typer(arg types, params) -> result type. final(rows, params) computes it from the
    group's rows: [(argument values tuple, multiplicity)] (FILTER and NULL skipping already applied when skip_nulls).
    moments: None, or the kind of moment evaluation ("sum", "avg", "var", "cov", ...) when the argument is a polynomial"""
    __slots__ = ("name", "typer", "final", "skip_nulls", "moments", "ordered", "nargs")
    def __init__(self, name, typer, final, skip_nulls=True, moments=None, ordered=False, nargs=1):
        self.name, self.typer, self.final, self.skip_nulls, self.moments, self.ordered, self.nargs = name, typer, final, skip_nulls, moments, ordered, nargs


AGGS = {}


def agg(names, typer, skip_nulls=True, moments=None, ordered=False, nargs=1):
    def deco(fn):
        for n in names.split(): AGGS[n] = AggDef(n, typer, fn, skip_nulls, moments, ordered, nargs)
        return fn
    return deco


def _sum_t(ts, p=()):
    t = ts[0]
    if t.integer: return T.HUGEINT if t.id != "UHUGEINT" else T.UHUGEINT
    if t.id == "DECIMAL": return T.DECIMAL(38, t.s)
    if t.id == "DECFLOAT": return t
    if t.id == "INTERVAL": return T.INTERVAL
    if t.id == "BOOLEAN": return T.HUGEINT
    if t.id == "NULL": return T.HUGEINT
    if t.numeric: return T.DOUBLE
    raise SqlError(f"SUM({t}) is not defined")


def _xsum(vals):
    """exact sum of ints / Decimals (any precision), float sum when a float takes part"""
    s = 0
    fl = False
    for v in vals:
        if isinstance(v, float): fl = True; break
    if fl: return math.fsum(float(v) for v in vals)
    for v in vals: s = CTX.add(s, v) if isinstance(s, Decimal) or isinstance(v, Decimal) else s + v
    return s


@agg("count_star", lambda ts, p=(): T.BIGINT, skip_nulls=False, nargs=0)
def _count_star(rows, p=()): return sum(w for _, w in rows)


@agg("count", lambda ts, p=(): T.BIGINT, moments="count")
def _count(rows, p=()): return sum(w for _, w in rows)


@agg("count_if countif", lambda ts, p=(): T.BIGINT)
def _count_if(rows, p=()): return sum(w for (v,), w in rows if v)


@agg("sum fsum sumkahan kahan_sum", _sum_t, moments="sum")
def _sum(rows, p=()):
    if not rows: return None
    if isinstance(rows[0][0][0], Interval):
        out = Interval()
        for (v,), w in rows: out = out + _iv_scale(v, w)
        return out
    if isinstance(rows[0][0][0], bool): return sum(int(v) * w for (v,), w in rows)
    return _xsum([v * w for (v,), w in rows])


@agg("product", lambda ts, p=(): T.DOUBLE)
def _product(rows, p=()):
    if not rows: return None
    out = 1.0
    for (v,), w in rows: out *= float(v) ** w
    return out


def _avg_t(ts, p=()):
    t = ts[0]
    if t.id == "INTERVAL": return T.INTERVAL
    if t.id == "DECFLOAT": return t
    if not (t.numeric or t.id == "NULL" or t.id == "BOOLEAN"): raise SqlError(f"AVG({t}) is not defined")
    return T.DOUBLE


def frac(v):
    if isinstance(v, bool): return Fraction(int(v))
    if isinstance(v, int): return Fraction(v)
    if isinstance(v, Decimal): return Fraction(v)
    return Fraction(v)


def ffloat(fr):
    """a Fraction as the nearest double"""
    try: return fr.numerator / fr.denominator
    except OverflowError: return float(Decimal(fr.numerator) / Decimal(fr.denominator))


@agg("avg mean", _avg_t, moments="avg")
def _avg(rows, p=()):
    n = sum(w for _, w in rows)
    if not n: return None
    if any(isinstance(v, T.DecFloat) for (v,), _ in rows):
        return T.decfloat(CTX.divide(sum((Decimal(v) * w for (v,), w in rows), Decimal(0)), Decimal(n)))
    if any(isinstance(v, float) for (v,), _ in rows):
        return math.fsum(float(v) * w for (v,), w in rows) / n
    if isinstance(rows[0][0][0], Interval):
        s = _sum(rows); return _iv_scale(s, Fraction(1, n))
    return ffloat(sum(frac(v) * w for (v,), w in rows) / n)


def _minmax_t(ts, p=()): return ts[0]


@agg("min", _minmax_t)
def _min(rows, p=()):
    if not rows: return None
    best = rows[0][0][0]
    for (v,), _ in rows:
        if cmp_values(v, best) < 0: best = v
    return best


@agg("max", _minmax_t)
def _max(rows, p=()):
    if not rows: return None
    best = rows[0][0][0]
    for (v,), _ in rows:
        if cmp_values(v, best) > 0: best = v
    return best


@agg("any_value arbitrary first", _minmax_t)
def _any_value(rows, p=()):
    return rows[0][0][0] if rows else None


@agg("last", _minmax_t)
def _last(rows, p=()):
    return rows[-1][0][0] if rows else None


@agg("arg_min min_by argmin", lambda ts, p=(): ts[0], nargs=2)
def _arg_min(rows, p=()):
    rows = [r for r in rows if r[0][1] is not None]
    if not rows: return None
    best = rows[0]
    for r in rows:
        if cmp_values(r[0][1], best[0][1]) < 0: best = r
    return best[0][0]


@agg("arg_max max_by argmax", lambda ts, p=(): ts[0], nargs=2)
def _arg_max(rows, p=()):
    rows = [r for r in rows if r[0][1] is not None]
    if not rows: return None
    best = rows[0]
    for r in rows:
        if cmp_values(r[0][1], best[0][1]) > 0: best = r
    return best[0][0]


@agg("every bool_and", lambda ts, p=(): T.BOOLEAN, moments="every")
def _every(rows, p=()):
    if not rows: return None
    return all(v for (v,), _ in rows)


@agg("bool_or any some", lambda ts, p=(): T.BOOLEAN, moments="any")
def _bool_or(rows, p=()):
    if not rows: return None
    return any(v for (v,), _ in rows)


@agg("bit_and", lambda ts, p=(): ts[0])
def _bit_and(rows, p=()):
    if not rows: return None
    out = -1
    for (v,), _ in rows: out &= v
    return out


@agg("bit_or", lambda ts, p=(): ts[0])
def _bit_or(rows, p=()):
    if not rows: return None
    out = 0
    for (v,), _ in rows: out |= v
    return out


@agg("bit_xor", lambda ts, p=(): ts[0])
def _bit_xor(rows, p=()):
    if not rows: return None
    out = 0
    for (v,), w in rows:
        if w % 2: out ^= v
    return out


# statistics: exact moments (Fractions) for exact inputs, then one rounding to DOUBLE
def moments1(rows):
    n = 0; s1 = s2 = Fraction(0); fl = False
    for (v,), w in rows:
        if isinstance(v, float):
            fl = True
            if not math.isfinite(v): return None, None, None, True, v
        x = frac(v); n += w; s1 += x * w; s2 += x * x * w
    return n, s1, s2, fl, None


def _var(rows, samp):
    n, s1, s2, fl, bad = moments1(rows)
    if bad is not None: return math.nan
    if n is None or n == 0 or (samp and n < 2): return None
    m2 = s2 - s1 * s1 / n
    return ffloat(m2 / (n - 1 if samp else n))


def var_from(n, s1, s2, samp):
    if n == 0 or (samp and n < 2): return None
    m2 = Fraction(s2) - Fraction(s1) * Fraction(s1) / n
    return ffloat(m2 / (n - 1 if samp else n))


fixed_t = lambda ts, p=(): T.DOUBLE
agg("var_pop", fixed_t, moments="var_pop")(lambda rows, p=(): _var(rows, False))
agg("var_samp variance", fixed_t, moments="var_samp")(lambda rows, p=(): _var(rows, True))


def _sqrt_or_none(x): return None if x is None else (math.sqrt(x) if x == x and x >= 0 else math.nan)


agg("stddev_pop", fixed_t, moments="stddev_pop")(lambda rows, p=(): _sqrt_or_none(_var(rows, False)))
agg("stddev_samp stddev", fixed_t, moments="stddev_samp")(lambda rows, p=(): _sqrt_or_none(_var(rows, True)))


def moments2(rows):
    """(n, sum y, sum x, sum yy, sum xx, sum xy) over the rows where both are non-NULL (args: y, x)"""
    n = 0; sy = sx = syy = sxx = sxy = Fraction(0)
    for (y, x), w in rows:
        if y is None or x is None: continue
        if isinstance(y, float) and not math.isfinite(y) or isinstance(x, float) and not math.isfinite(x): return None
        Y, X = frac(y), frac(x)
        n += w; sy += Y * w; sx += X * w; syy += Y * Y * w; sxx += X * X * w; sxy += X * Y * w
    return n, sy, sx, syy, sxx, sxy


def bivariate(kind, m):
    """covariance, correlation and regression statistics from exact moments (args y, x as in ISO)"""
    if m is None: return math.nan
    n, sy, sx, syy, sxx, sxy = m
    if kind == "regr_count": return n
    if n == 0: return None
    cxy = sxy - sx * sy / n; cxx = sxx - sx * sx / n; cyy = syy - sy * sy / n
    if kind == "covar_pop": return ffloat(cxy / n)
    if kind == "covar_samp": return None if n < 2 else ffloat(cxy / (n - 1))
    if kind == "regr_avgx": return ffloat(sx / n)
    if kind == "regr_avgy": return ffloat(sy / n)
    if kind == "regr_sxx": return ffloat(cxx)
    if kind == "regr_syy": return ffloat(cyy)
    if kind == "regr_sxy": return ffloat(cxy)
    if kind == "corr":                               # ISO: NULL when either variance is 0 (DuckDB gives NaN)
        if cxx == 0 or cyy == 0: return None
        return ffloat(cxy) / math.sqrt(ffloat(cxx) * ffloat(cyy))
    if kind == "regr_slope": return None if cxx == 0 else ffloat(cxy / cxx)
    if kind == "regr_intercept": return None if cxx == 0 else ffloat((sy - (cxy / cxx) * sx) / n)
    if kind == "regr_r2":
        if cxx == 0: return None
        if cyy == 0: return 1.0
        return ffloat(cxy * cxy / (cxx * cyy))
    raise SqlError(kind)


for _k in ("covar_pop", "covar_samp", "corr", "regr_slope", "regr_intercept", "regr_count", "regr_r2", "regr_avgx", "regr_avgy",
           "regr_sxx", "regr_syy", "regr_sxy"):
    agg(_k, (lambda k: lambda ts, p=(): T.BIGINT if k == "regr_count" else T.DOUBLE)(_k), skip_nulls=False, moments=_k, nargs=2)(
        (lambda k: lambda rows, p=(): bivariate(k, moments2(rows)))(_k))


@agg("skewness", fixed_t)
def _skew(rows, p=()):
    n = 0; m = [Fraction(0)] * 4
    for (v,), w in rows:
        x = frac(v); n += w
        for i in range(1, 4): m[i] += x ** i * w
    if n < 3: return None
    mean = m[1] / n; m2 = m[2] / n - mean ** 2; m3 = m[3] / n - 3 * mean * m[2] / n + 2 * mean ** 3
    if m2 == 0: return math.nan
    return ffloat(m3) / ffloat(m2) ** 1.5 * math.sqrt(n * (n - 1)) / (n - 2)


@agg("kurtosis", fixed_t)
def _kurt(rows, p=()):
    n = 0; m = [Fraction(0)] * 5
    for (v,), w in rows:
        x = frac(v); n += w
        for i in range(1, 5): m[i] += x ** i * w
    if n < 4: return None
    mean = m[1] / n
    m2 = m[2] / n - mean ** 2
    m4 = m[4] / n - 4 * mean * m[3] / n + 6 * mean ** 2 * m[2] / n - 3 * mean ** 4
    if m2 == 0: return math.nan
    return (n - 1) * ((n + 1) * ffloat(m4) / ffloat(m2) ** 2 - 3 * (n - 1)) / ((n - 2) * (n - 3))


@agg("entropy", fixed_t)
def _entropy(rows, p=()):
    cnt = {}
    for (v,), w in rows: cnt[v] = cnt.get(v, 0) + w
    n = sum(cnt.values())
    if not n: return 0.0
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())


def _sorted_rows(rows, desc=False):
    import functools
    return sorted(rows, key=functools.cmp_to_key(lambda a, b: cmp_values(a[0][0], b[0][0])), reverse=desc)


def _quantile_cont_one(rows, q):
    rs = _sorted_rows(rows)
    n = sum(w for _, w in rs)
    if not n: return None
    pos = Fraction(q) * (n - 1) if not isinstance(q, float) else Fraction(q) * (n - 1)
    lo = math.floor(pos); hi = math.ceil(pos)
    def at(k):
        c = 0
        for (v,), w in rs:
            c += w
            if k < c: return v
        return rs[-1][0][0]
    a, b = at(lo), at(hi)
    if isinstance(a, (_dt.date, _dt.datetime)):
        if lo == hi: return a
        da, db = _to_dt(a), _to_dt(b); return da + (db - da) * float(pos - lo)
    if isinstance(a, Decimal):
        exp = min(a.as_tuple().exponent, b.as_tuple().exponent)
        x = frac(a) + (frac(b) - frac(a)) * (pos - lo)
        q = Decimal(x.numerator) / Decimal(x.denominator) if x.denominator != 1 else Decimal(x.numerator)
        return CTX.quantize(q, Decimal(1).scaleb(exp)) if False else q.quantize(Decimal(1).scaleb(exp), rounding=ROUND_DOWN, context=CTX)
    if not isinstance(a, (int, float, bool)): return a if pos == lo else b
    if lo == hi: return float(a) if not isinstance(a, Interval) else a
    return ffloat(frac(a) + (frac(b) - frac(a)) * (pos - lo))


def _quantile_disc_one(rows, q, desc=False):
    rs = _sorted_rows(rows, desc)
    n = sum(w for _, w in rs)
    if not n: return None
    k = max(0, math.ceil(Fraction(q) * n) - 1) if not isinstance(q, float) else max(0, math.ceil(q * n - 1e-12) - 1)
    c = 0
    for (v,), w in rs:
        c += w
        if k < c: return v
    return rs[-1][0][0]


def _q_param(p):
    q = p[0] if p else Fraction(1, 2)
    if isinstance(q, list): return [Fraction(str(x)) if isinstance(x, float) else frac(x) for x in q]
    return Fraction(str(q)) if isinstance(q, float) else frac(q)


def _qc_t(ts, p=()):
    t = ts[0] if (ts[0].temporal or ts[0].id == "DECIMAL") else T.DOUBLE
    return T.LIST(t) if p and isinstance(p[0], list) else t


@agg("quantile_cont percentile_cont", _qc_t)
def _quantile_cont(rows, p=()):
    q = _q_param(p)
    if isinstance(q, list): return None if not rows else [_quantile_cont_one(rows, x) for x in q]
    return _quantile_cont_one(rows, q)


@agg("quantile_disc percentile_disc quantile", lambda ts, p=(): T.LIST(ts[0]) if p and isinstance(p[0], list) else ts[0])
def _quantile_disc(rows, p=()):
    q = _q_param(p); desc = len(p) > 1 and p[1] == "desc"
    if isinstance(q, list): return None if not rows else [_quantile_disc_one(rows, x, desc) for x in q]
    return _quantile_disc_one(rows, q, desc)


@agg("median", lambda ts, p=(): ts[0] if ts[0].id == "DECIMAL" or ts[0].temporal or not ts[0].numeric else T.DOUBLE)
def _median(rows, p=()):
    if not rows: return None
    if not isinstance(rows[0][0][0], (int, float, Decimal, bool, _dt.date, Interval)):
        return _quantile_disc_one(rows, Fraction(1, 2))
    return _quantile_cont_one(rows, Fraction(1, 2))


@agg("mode", lambda ts, p=(): ts[0])
def _mode(rows, p=()):
    """the most frequent value; ties: the first in the sort order (WITHIN GROUP (ORDER BY x [DESC])), else the smallest"""
    cnt = {}
    for (v,), w in rows: cnt[v] = cnt.get(v, 0) + w
    if not cnt: return None
    top = max(cnt.values())
    cands = [v for v, c in cnt.items() if c == top]
    import functools
    cands.sort(key=functools.cmp_to_key(cmp_values), reverse=bool(p) and p[0] == "desc")
    return cands[0]


@agg("string_agg listagg group_concat", lambda ts, p=(): T.VARCHAR, ordered=True, nargs=1)
def _string_agg(rows, p=()):
    if not rows: return None
    sep = p[0] if p else ","
    if sep is None: sep = ""
    return sep.join(str(v) for (v,), w in rows for _ in range(w))


@agg("collect", lambda ts, p=(): T.MULTISET(ts[0]))
def _collect(rows, p=()):
    """COLLECT(x): the multiset of the group's values (NULLs eliminated); NULL for no rows"""
    if not rows: return None
    return [v for (v,), w in rows for _ in range(w)]


@agg("fusion", lambda ts, p=(): ts[0] if ts[0].id == "MULTISET" else T.MULTISET(_coll(ts[0])))
def _fusion(rows, p=()):
    """FUSION(m): the multiset union (ALL) of the group's multisets"""
    if not rows: return None
    return [x for (v,), w in rows for _ in range(w) for x in v]


@agg("intersection", lambda ts, p=(): ts[0] if ts[0].id == "MULTISET" else T.MULTISET(_coll(ts[0])))
def _intersection(rows, p=()):
    """INTERSECTION(m): the multiset intersection (ALL) of the group's multisets"""
    if not rows: return None
    acc = None
    for (v,), w in rows:
        for _ in range(w): acc = list(v) if acc is None else _ms_op("INTERSECT")(acc, v)
    return acc


def _jagg_t(ts, p=()):
    o = jopts(p[0]) if p else {}
    return o.get("rt") or T.JSON


@agg("__json_objectagg", _jagg_t, skip_nulls=False, nargs=2)
def _json_objectagg(rows, p=()):
    """JSON_OBJECTAGG(k VALUE v): NULL ON NULL by default; an empty group gives NULL"""
    import sqljson
    if not rows: return None
    o = jopts(p[0])
    keys, vals = [], []
    for (k, v), w in rows:
        for _ in range(w): keys.append(k); vals.append(v)
    return sqljson.json_object(keys, vals, o["vtypes_t"] * len(vals), [o["fmts"][0] if o.get("fmts") else False] * len(vals),
                               o.get("absent", False), o.get("unique", False))


@agg("__json_arrayagg", _jagg_t, skip_nulls=False, ordered=True)
def _json_arrayagg(rows, p=()):
    """JSON_ARRAYAGG(v [ORDER BY ...]): ABSENT ON NULL by default; an empty group gives NULL"""
    import sqljson
    if not rows: return None
    o = jopts(p[0])
    vals = [v for (v,), w in rows for _ in range(w)]
    return sqljson.json_array(vals, o["vtypes_t"] * len(vals), [o["fmts"][0] if o.get("fmts") else False] * len(vals),
                              o.get("null_on_null", False))


@agg("array_agg list", lambda ts, p=(): T.LIST(ts[0]), skip_nulls=False, ordered=True)
def _array_agg(rows, p=()):
    if not rows: return None
    return [v for (v,), w in rows for _ in range(w)]


@agg("__iso_listagg", lambda ts, p=(): T.VARCHAR, ordered=True)
def _iso_listagg(rows, p=()):
    """ISO LISTAGG(x, sep [ON OVERFLOW ERROR | TRUNCATE filler WITH|WITHOUT COUNT]) WITHIN GROUP (ORDER BY ...)"""
    if not rows: return None
    sep = p[0] if p and p[0] is not None else ""
    return sep.join(str(v) for (v,), w in rows for _ in range(w))


@agg("approx_count_distinct", lambda ts, p=(): T.BIGINT)
def _approx_cd(rows, p=()): return len({v for (v,), _ in rows})


@agg("histogram", lambda ts, p=(): T.VARCHAR)
def _histogram(rows, p=()):
    cnt = {}
    for (v,), w in rows: cnt[v] = cnt.get(v, 0) + w
    return cnt


# hypothetical set functions: RANK(v) WITHIN GROUP (ORDER BY x): the rank v would have among the group's rows
def _hypo(kind):
    def f(rows, p=(), desc=False, nulls_first=False):
        v = p[0]
        n = sum(w for _, w in rows)
        def before(x):
            if x is None or v is None:
                if x is None and v is None: return False
                return (x is None) == nulls_first
            c = cmp_values(x, v)
            return c > 0 if desc else c < 0
        less = sum(w for (x,), w in rows if before(x))
        equal = sum(w for (x,), w in rows if (x is None and v is None) or (x is not None and v is not None and cmp_values(x, v) == 0))
        if kind == "rank": return less + 1
        if kind == "dense_rank":
            return len({x for (x,), _ in rows if before(x)}) + 1
        if kind == "percent_rank": return 0.0 if n == 0 else less / n
        if kind == "cume_dist": return (less + equal + 1) / (n + 1)
    return f


for _k in ("rank", "dense_rank", "percent_rank", "cume_dist"):
    agg(f"__iso_h{_k}", (lambda k: lambda ts, p=(): T.BIGINT if k in ("rank", "dense_rank") else T.DOUBLE)(_k), skip_nulls=False, ordered=True)(_hypo(_k))


def agg_typer(name, ts, params=()):
    a = AGGS.get(name)
    if a is None: raise SqlError(f"unknown aggregate function {name}")
    return a.typer(ts, params) if a.typer.__code__.co_argcount > 1 else a.typer(ts)


# ================================================================== window functions
WINDOW_ONLY = {"row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag", "lead", "first_value",
               "last_value", "nth_value", "rank_dense"}


def win_typer(name, args):
    if name in ("row_number", "rank", "dense_rank", "rank_dense", "ntile"): return T.BIGINT
    if name in ("percent_rank", "cume_dist"): return T.DOUBLE
    if name in ("lag", "lead"):
        if len(args) > 2: return T.supertype(args[0].t, args[2].t, False, _strlit(args[2]))
        return args[0].t
    if name in ("first_value", "last_value", "nth_value"): return args[0].t
    raise SqlError(f"unknown window function {name}")
