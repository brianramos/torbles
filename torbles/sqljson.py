# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SQL/JSON (ISO/IEC 9075-2:2023): JSON text and the SQL/JSON data model, the SQL/JSON path language (lax and strict
modes, accessors, filters, arithmetic, item methods), and the functions built on them: IS JSON, JSON_OBJECT,
JSON_ARRAY, JSON_OBJECTAGG, JSON_ARRAYAGG, JSON_VALUE, JSON_QUERY, JSON_EXISTS, JSON_TABLE, JSON_SCALAR,
JSON_SERIALIZE, JSON(...).

SQL/JSON items are Python values: None (the JSON null), bool, int / Decimal / float (numbers, exact where the text
is), str, list (arrays), dict (objects), and datetime values made by .datetime() and its relatives. JSON values in
SQL are their text (type JSON), minified. These functions run where every scalar function runs: on the distinct
values the engine's GROUP BY returns, each standing for its count of rows."""
import datetime as _dt, json, math, re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import sqltypes as T
from sqltypes import SqlError

MISSING = object()


class JErr(Exception):
    """an SQL/JSON error (structural or type error, out-of-range subscript in strict mode, ...): becomes the
    function's ON ERROR behavior"""


# ================================================================== JSON text <-> items
def _pairs_unique(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen: raise JErr(f"duplicate key {k!r}")
        seen.add(k)
    return dict(pairs)


def _num(s):
    """a JSON number: an exact integer or decimal; past the decimal range, a double"""
    if re.fullmatch(r"-?\d+", s): return int(s)
    try:
        d = Decimal(s)
        if d.is_finite() and abs(d.adjusted()) < 1000: return d
    except InvalidOperation: pass
    return float(s)


def parse(text, unique=False):
    """JSON text -> an SQL/JSON item (JErr when the text is not JSON; with unique, when an object repeats a key)"""
    if not isinstance(text, str): raise JErr("JSON text must be a character string")
    try:
        return json.loads(text, parse_float=_num, parse_int=int, parse_constant=lambda c: _bad(c),
                          object_pairs_hook=_pairs_unique if unique else dict)
    except JErr: raise
    except (ValueError, RecursionError) as e: raise JErr(f"invalid JSON text: {e}")


def _bad(c): raise JErr(f"invalid JSON text: {c}")


def dumps(v):
    """an SQL/JSON item -> JSON text (minified; numbers exact)"""
    if v is None: return "null"
    if v is True: return "true"
    if v is False: return "false"
    if isinstance(v, int): return str(v)
    if isinstance(v, Decimal):
        if not v.is_finite(): raise JErr("a JSON number must be finite")
        s = format(v, "f")
        return s
    if isinstance(v, float):
        if not math.isfinite(v): raise JErr(f"{v} is not a JSON number")
        return T.fmt_double(v)
    if isinstance(v, str): return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list): return "[" + ",".join(dumps(x) for x in v) + "]"
    if isinstance(v, dict): return "{" + ",".join(json.dumps(str(k), ensure_ascii=False) + ":" + dumps(x) for k, x in v.items()) + "}"
    if isinstance(v, (_dt.date, _dt.time)): return json.dumps(T.to_text(v, None))
    if isinstance(v, (bytes, bytearray)): raise JErr("a binary string has no JSON form")
    if isinstance(v, T.Interval): raise JErr("an interval has no JSON form")
    raise JErr(f"no JSON form for {v!r}")


def from_sql(v, t, fmt_json=False, items=False):
    """an SQL value -> an SQL/JSON item: JSON-typed values (and FORMAT JSON) are parsed; strings are strings;
    numbers, booleans; datetimes as their ISO text in a constructor, as datetime items in PASSING (items=True); NULL
    as the JSON null"""
    if v is None: return None
    if fmt_json or (t is not None and t.id == "JSON"): return parse(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
    if items and isinstance(v, (_dt.date, _dt.time)): return v
    if isinstance(v, bool) or isinstance(v, (int, Decimal, str)): return v
    if isinstance(v, float):
        if not math.isfinite(v): raise JErr(f"{v} has no JSON form")
        return v
    if isinstance(v, list): return [from_sql(x, t.child if t is not None and t.id in ("LIST", "MULTISET") else None) for x in v]
    if isinstance(v, (_dt.date, _dt.time)): return T.to_text(v, t)
    raise JErr(f"{t} has no JSON form")


# ================================================================== the path language: lexer
_PTOK = re.compile(r"""\s*(?:
    (?P<num>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?) |
    (?P<str>"(?:[^"\\]|\\.)*") |
    (?P<var>\$[A-Za-z_][A-Za-z_0-9]*|\$"(?:[^"\\]|\\.)*") |
    (?P<op>==|!=|<>|<=|>=|&&|\|\||\.\*|\[\*\]|[$@.\[\](),?!<>+\-*/%:]) |
    (?P<id>[A-Za-z_][A-Za-z_0-9]*)
)""", re.X)


def _lex(s):
    out, i = [], 0
    while i < len(s):
        if s[i:].strip() == "": break
        m = _PTOK.match(s, i)
        if not m or m.end() == i: raise SqlError(f"syntax error in SQL/JSON path {s!r} at {s[i:i + 15]!r}")
        k = m.lastgroup; x = m.group(k)
        if k == "str": x = json.loads(x)
        elif k == "var": x = json.loads(x[1:]) if x[1:2] == '"' else x[1:]
        out.append((k, x)); i = m.end()
    out.append(("end", None))
    return out


# ================================================================== the path language: parser (AST as tuples)
class _P:
    def __init__(self, text):
        self.text = text; self.t = _lex(text); self.i = 0

    def peek(self, k=0): return self.t[self.i + k]
    def take(self): x = self.t[self.i]; self.i += 1; return x
    def at(self, kind, val=None, k=0):
        x = self.t[self.i + k]
        return x[0] == kind and (val is None or (x[1].lower() if kind == "id" else x[1]) == val)
    def expect(self, kind, val=None):
        if not self.at(kind, val): raise SqlError(f"SQL/JSON path {self.text!r}: expected {val or kind}, found {self.peek()[1]!r}")
        return self.take()

    def path(self):
        mode = "lax"
        if self.at("id", "lax") or self.at("id", "strict"): mode = self.take()[1].lower()
        e = self.wff()
        if not self.at("end"): raise SqlError(f"SQL/JSON path {self.text!r}: unexpected {self.peek()[1]!r}")
        return mode, e

    def wff(self):
        e = self.mul()
        while self.at("op", "+") or self.at("op", "-"):
            op = self.take()[1]; e = ("arith", op, e, self.mul())
        return e

    def mul(self):
        e = self.unary()
        while self.at("op", "*") or self.at("op", "/") or self.at("op", "%"):
            op = self.take()[1]; e = ("arith", op, e, self.unary())
        return e

    def unary(self):
        if self.at("op", "+") or self.at("op", "-"):
            op = self.take()[1]; return ("neg" if op == "-" else "pos", self.unary())
        return self.accessors(self.primary())

    def primary(self):
        k, x = self.peek()
        if k == "op" and x == "$": self.take(); return ("ctx",)
        if k == "op" and x == "@": self.take(); return ("cur",)
        if k == "var": self.take(); return ("var", x)
        if k == "num": self.take(); return ("lit", T_num(x))
        if k == "str": self.take(); return ("lit", x)
        if k == "id" and x.lower() in ("true", "false", "null"):
            self.take(); return ("lit", {"true": True, "false": False, "null": None}[x.lower()])
        if k == "id" and x.lower() == "last": self.take(); return ("last",)
        if k == "op" and x == "(":
            self.take(); e = self.wff(); self.expect("op", ")"); return e
        raise SqlError(f"SQL/JSON path {self.text!r}: unexpected {x!r}")

    def accessors(self, e):
        while True:
            if self.at("op", ".*"): self.take(); e = ("wild", e); continue
            if self.at("op", "[*]"): self.take(); e = ("awild", e); continue
            if self.at("op", "."):
                self.take(); k, x = self.take()
                if k == "str": e = ("member", e, x); continue
                if k != "id": raise SqlError(f"SQL/JSON path {self.text!r}: a member name after '.'")
                if self.at("op", "("):
                    self.take(); args = []
                    while not self.at("op", ")"):
                        a = self.take()
                        if a[0] not in ("num", "str"): raise SqlError(f"SQL/JSON path {self.text!r}: method arguments are literals")
                        args.append(T_num(a[1]) if a[0] == "num" else a[1])
                        if self.at("op", ","): self.take()
                    self.expect("op", ")")
                    e = ("method", e, x.lower(), args); continue
                e = ("member", e, x); continue
            if self.at("op", "["):
                self.take(); subs = []
                while True:
                    a = self.wff(); b = None
                    if self.at("id", "to"): self.take(); b = self.wff()
                    subs.append((a, b))
                    if self.at("op", ","): self.take(); continue
                    break
                self.expect("op", "]")
                e = ("index", e, subs); continue
            if self.at("op", "?"):
                self.take(); self.expect("op", "("); p = self.pred(); self.expect("op", ")")
                e = ("filter", e, p); continue
            return e

    # predicates (filters)
    def pred(self):
        p = self.conj()
        while self.at("op", "||"): self.take(); p = ("or", p, self.conj())
        return p

    def conj(self):
        p = self.neg()
        while self.at("op", "&&"): self.take(); p = ("and", p, self.neg())
        return p

    def neg(self):
        if self.at("op", "!"):
            self.take(); self.expect("op", "("); p = self.pred(); self.expect("op", ")"); return ("not", p)
        return self.ppred()

    def ppred(self):
        if self.at("id", "exists") and self.at("op", "(", 1):
            self.take(); self.take(); e = self.wff(); self.expect("op", ")"); return ("exists", e)
        if self.at("op", "("):                         # ( predicate ) [is unknown], or a parenthesized wff
            save = self.i
            try:
                self.take(); p = self.pred(); self.expect("op", ")")
                if self.at("id", "is") and self.at("id", "unknown", 1): self.take(); self.take(); return ("unknown", p)
                if not (self.at("op") and self.peek()[1] in ("==", "!=", "<>", "<", "<=", ">", ">=", "+", "-", "*", "/", "%")) \
                        and not self.at("id", "like_regex") and not self.at("id", "starts"):
                    return p
            except SqlError: pass
            self.i = save
        a = self.wff()
        if self.at("op") and self.peek()[1] in ("==", "!=", "<>", "<", "<=", ">", ">="):
            op = self.take()[1]; return ("cmp", "!=" if op == "<>" else op, a, self.wff())
        if self.at("id", "like_regex"):
            self.take(); pat = self.expect("str")[1]; fl = ""
            if self.at("id", "flag"): self.take(); fl = self.expect("str")[1]
            return ("like_regex", a, pat, fl)
        if self.at("id", "starts") and self.at("id", "with", 1):
            self.take(); self.take()
            k, x = self.take()
            if k not in ("str", "var"): raise SqlError(f"SQL/JSON path {self.text!r}: starts with a string or a variable")
            return ("starts", a, ("lit", x) if k == "str" else ("var", x))
        raise SqlError(f"SQL/JSON path {self.text!r}: a predicate is needed in a filter")


def T_num(s):
    if isinstance(s, (int, Decimal, float)): return s
    if re.fullmatch(r"\d+", s): return int(s)
    return Decimal(s)


_CACHE = {}


def compile_path(text):
    """an SQL/JSON path -> (mode, ast); syntax errors are reported when the query is bound"""
    if not isinstance(text, str): raise SqlError("an SQL/JSON path must be a character string")
    c = _CACHE.get(text)
    if c is None: c = _CACHE[text] = _P(text).path()
    return c


# ================================================================== the path language: evaluation
def _num_item(x): return isinstance(x, (int, Decimal, float)) and not isinstance(x, bool)


def _type_name(x):
    if x is None: return "null"
    if isinstance(x, bool): return "boolean"
    if _num_item(x): return "number"
    if isinstance(x, str): return "string"
    if isinstance(x, list): return "array"
    if isinstance(x, dict): return "object"
    if isinstance(x, _dt.datetime): return "timestamp with time zone" if x.tzinfo else "timestamp without time zone"
    if isinstance(x, _dt.date): return "date"
    if isinstance(x, _dt.time): return "time with time zone" if x.tzinfo else "time without time zone"
    return "unknown"


class _Ctx:
    def __init__(self, mode, root, vars_):
        self.lax = mode == "lax"; self.root = root; self.vars = vars_; self.ids = None

    def object_id(self, x):
        """.keyvalue()'s id of an object: its position in a preorder walk of the context item (then of each PASSING
        value, 10^10 apart), so equal queries give equal ids; objects made during the evaluation come after"""
        if self.ids is None:
            self.ids = {}
            def walk(v, base, n):
                if isinstance(v, (dict, list)):
                    if isinstance(v, dict): self.ids.setdefault(id(v), base + n[0])
                    n[0] += 1
                    for c in (v.values() if isinstance(v, dict) else v): walk(c, base, n)
            walk(self.root, 0, [0])
            for k, v in enumerate(self.vars.values(), 1): walk(v, k * 10_000_000_000, [0])
            self.next = 10_000_000_000 * (len(self.vars) + 1)
        if id(x) not in self.ids: self.ids[id(x)] = self.next; self.next += 1
        return self.ids[id(x)]


def evaluate(path, item, vars_=None):
    """the sequence (a list) of SQL/JSON items the path selects from item (JErr on an error)"""
    mode, ast = path
    return _ev(ast, _Ctx(mode, item, vars_ or {}), item, None)


def _unwrap(seq, cx):
    """lax mode: arrays in the sequence are replaced by their elements (one level)"""
    if not cx.lax: return seq
    out = []
    for x in seq:
        if isinstance(x, list): out.extend(x)
        else: out.append(x)
    return out


def _ev(e, cx, cur, last):
    k = e[0]
    if k == "ctx": return [cx.root]
    if k == "cur":
        if cur is MISSING: raise JErr("@ outside a filter")
        return [cur]
    if k == "lit": return [e[1]]
    if k == "var":
        if e[1] not in cx.vars: raise JErr(f"no PASSING value named {e[1]}")
        return [cx.vars[e[1]]]
    if k == "last":
        if last is None: raise JErr("last outside an array subscript")
        return [last]
    if k == "member":
        out = []
        for x in _ev(e[1], cx, cur, last):
            if isinstance(x, list) and cx.lax:
                for y in x:
                    if isinstance(y, dict) and e[2] in y: out.append(y[e[2]])
                    elif not cx.lax: raise JErr(f"no member {e[2]}")
                continue
            if isinstance(x, dict):
                if e[2] in x: out.append(x[e[2]])
                elif not cx.lax: raise JErr(f"strict mode: no member {e[2]!r}")
            elif not cx.lax: raise JErr(f"strict mode: .{e[2]} of a {_type_name(x)}")
        return out
    if k == "wild":
        out = []
        for x in _ev(e[1], cx, cur, last):
            if isinstance(x, list) and cx.lax:
                for y in x:
                    if isinstance(y, dict): out.extend(y.values())
                continue
            if isinstance(x, dict): out.extend(x.values())
            elif not cx.lax: raise JErr(f"strict mode: .* of a {_type_name(x)}")
        return out
    if k == "awild":
        out = []
        for x in _ev(e[1], cx, cur, last):
            if isinstance(x, list): out.extend(x)
            elif cx.lax: out.append(x)
            else: raise JErr(f"strict mode: [*] of a {_type_name(x)}")
        return out
    if k == "index":
        out = []
        for x in _ev(e[1], cx, cur, last):
            arr = x if isinstance(x, list) else ([x] if cx.lax else None)
            if arr is None: raise JErr(f"strict mode: a subscript of a {_type_name(x)}")
            n = len(arr)
            for a, b in e[2]:
                lo = _subscript(a, cx, cur, n - 1)
                hi = _subscript(b, cx, cur, n - 1) if b is not None else lo
                if not cx.lax and (lo < 0 or hi >= n or lo > hi): raise JErr(f"strict mode: subscript {lo}{'' if b is None else f' to {hi}'} of {n} elements")
                for i in range(max(lo, 0), min(hi, n - 1) + 1): out.append(arr[i])
        return out
    if k == "filter":
        out = []
        for x in _unwrap(_ev(e[1], cx, cur, last), cx):
            if _pred(e[2], cx, x) is True: out.append(x)
        return out
    if k == "method":
        return _method(e, cx, cur, last)
    if k in ("neg", "pos"):
        out = []
        for x in _unwrap(_ev(e[1], cx, cur, last), cx):
            if not _num_item(x): raise JErr(f"unary {'-' if k == 'neg' else '+'} of a {_type_name(x)}")
            out.append(-x if k == "neg" else x)
        return out
    if k == "arith":
        a = _unwrap(_ev(e[2], cx, cur, last), cx); b = _unwrap(_ev(e[3], cx, cur, last), cx)
        if len(a) != 1 or len(b) != 1 or not _num_item(a[0]) or not _num_item(b[0]):
            raise JErr(f"arithmetic {e[1]} needs one number on each side")
        return [_arith(e[1], a[0], b[0])]
    raise JErr(f"internal: path node {k}")


def _subscript(e, cx, cur, last):
    v = _ev(e, cx, cur, last)
    if len(v) != 1 or not _num_item(v[0]): raise JErr("an array subscript must be one number")
    return int(v[0])


def _arith(op, a, b):
    if isinstance(a, float) or isinstance(b, float): a, b = float(a), float(b)
    elif isinstance(a, Decimal) or isinstance(b, Decimal): a, b = Decimal(a), Decimal(b)
    try:
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/":
            if b == 0: raise JErr("division by zero")
            if isinstance(a, int) and isinstance(b, int):
                q = Decimal(a) / Decimal(b)
                return int(q) if q == int(q) else q
            return a / b
        if op == "%":
            if b == 0: raise JErr("division by zero")
            if isinstance(a, float): return math.fmod(a, b)
            r = abs(a) % abs(b); return r if a >= 0 else -r
    except (ArithmeticError, InvalidOperation) as ex: raise JErr(str(ex))


# ------------------------------------------------------------------ item methods
_DT_TEMPLATE = [("YYYY", r"(?P<Y>\d{4})"), ("MM", r"(?P<M>\d{2})"), ("DD", r"(?P<D>\d{2})"), ("HH24", r"(?P<H>\d{2})"),
                ("HH12", r"(?P<h>\d{2})"), ("HH", r"(?P<h>\d{2})"), ("MI", r"(?P<MI>\d{2})"), ("SSSSS", r"(?P<SSSSS>\d{1,5})"),
                ("SS", r"(?P<S>\d{2})"), ("FF1", r"(?P<F>\d{1})"), ("FF2", r"(?P<F>\d{2})"), ("FF3", r"(?P<F>\d{3})"),
                ("FF4", r"(?P<F>\d{4})"), ("FF5", r"(?P<F>\d{5})"), ("FF6", r"(?P<F>\d{6})"), ("A.M.", r"(?P<AP>[AP]\.M\.)"),
                ("P.M.", r"(?P<AP>[AP]\.M\.)"), ("TZH", r"(?P<TZH>[+-]\d{2})"), ("TZM", r"(?P<TZM>\d{2})"),
                ("RRRR", r"(?P<Y>\d{4})"), ("YYY", r"(?P<y3>\d{3})"), ("YY", r"(?P<y2>\d{2})"), ("Y", r"(?P<y1>\d)")]


def _template(tpl, s):
    """a datetime from s by an ISO datetime template (YYYY-MM-DD HH24:MI:SS.FF TZH:TZM ...)"""
    rx, i, has_t, has_d, has_z = "", 0, False, False, False
    while i < len(tpl):
        for name, pat in _DT_TEMPLATE:
            if tpl.upper().startswith(name, i):
                rx += pat; i += len(name)
                has_t |= name[0] in "HMSFA" and name != "MM"; has_d |= name[0] in "YMDR" and name != "MI"; has_z |= name.startswith("TZ")
                break
        else:
            if tpl[i] == '"':
                j = tpl.index('"', i + 1); rx += re.escape(tpl[i + 1:j]); i = j + 1
            else: rx += re.escape(tpl[i]); i += 1
    m = re.fullmatch(rx, s.strip())
    if not m: raise JErr(f"{s!r} does not match the datetime template {tpl!r}")
    g = m.groupdict()
    y = int(g["Y"]) if g.get("Y") else 2000 + int(g["y2"]) if g.get("y2") else 1970
    mo, d = int(g.get("M") or 1), int(g.get("D") or 1)
    h = int(g.get("H") or 0)
    if g.get("h"):
        h = int(g["h"]) % 12 + (12 if (g.get("AP") or "A").startswith("P") else 0)
    mi, sec = int(g.get("MI") or 0), int(g.get("S") or 0)
    if g.get("SSSSS"): sec = int(g["SSSSS"]); h, mi, sec = sec // 3600, sec // 60 % 60, sec % 60
    us = int((g.get("F") or "").ljust(6, "0")[:6] or 0)
    tz = None
    if g.get("TZH"):
        off = _dt.timedelta(hours=abs(int(g["TZH"])), minutes=int(g.get("TZM") or 0))
        tz = _dt.timezone(-off if g["TZH"].startswith("-") else off)
    try:
        if has_d and not has_t: return _dt.date(y, mo, d)
        if has_t and not has_d: return _dt.time(h, mi, sec, us, tzinfo=tz)
        return _dt.datetime(y, mo, d, h, mi, sec, us, tzinfo=tz)
    except ValueError as ex: raise JErr(f"{s!r}: {ex}")


def _datetime_item(s, tpl=None):
    """.datetime(): a date, time or timestamp (with or without time zone) from its ISO text, or by a template"""
    if not isinstance(s, str): raise JErr(f".datetime() of a {_type_name(s)}")
    if tpl is not None: return _template(tpl, s)
    x = s.strip()
    try:
        if re.fullmatch(r"-?\d{1,6}-\d{1,2}-\d{1,2}", x): return T.parse_date(x)
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?", x): return T.parse_time(x)
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?\s*(Z|[+-]\d{1,2}(:?\d{2})?)", x): return T.parse_timetz(x)
        if T.has_zone(x): return T.parse_timestamptz(x)
        return T.parse_timestamp(x)
    except SqlError as ex: raise JErr(str(ex))


def _to_number(x, what):
    if _num_item(x): return x
    if isinstance(x, str):
        try:
            v = T_num(x.strip()) if re.fullmatch(r"\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*", x) else None
        except InvalidOperation: v = None
        if v is None: raise JErr(f".{what}() of the string {x!r}")
        if isinstance(x, str) and re.search(r"[eE]", x): return float(x)
        return int(x.strip()) if re.fullmatch(r"\s*[+-]?\d+\s*", x) else Decimal(x.strip())
    raise JErr(f".{what}() of a {_type_name(x)}")


def _method(e, cx, cur, last):
    name, args = e[2], e[3]
    seq = _ev(e[1], cx, cur, last)
    if name not in ("size", "type"): seq = _unwrap(seq, cx)
    out = []
    for x in seq:
        if name == "type": out.append(_type_name(x))
        elif name == "size":
            if isinstance(x, list): out.append(len(x))
            elif cx.lax: out.append(1)
            else: raise JErr(f"strict mode: .size() of a {_type_name(x)}")
        elif name in ("double", "number", "decimal", "integer", "bigint"):
            v = _to_number(x, name)
            if name == "double":
                f = float(v)
                if not math.isfinite(f): raise JErr(".double(): not finite")
                out.append(f)
            elif name == "number": out.append(v)
            elif name == "decimal":
                d = Decimal(str(v)) if isinstance(v, float) else Decimal(v)
                if args:
                    p = int(args[0]); s = int(args[1]) if len(args) > 1 else 0
                    d = d.quantize(Decimal(1).scaleb(-s), rounding=ROUND_HALF_UP)
                    if abs(d) >= Decimal(10) ** (p - s): raise JErr(f".decimal({p}, {s}): {v} does not fit")
                out.append(d)
            else:
                iv = int(Decimal(str(v)).to_integral_value(rounding=ROUND_HALF_UP))
                lo, hi = T.INT_RANGE["INTEGER" if name == "integer" else "BIGINT"]
                if not lo <= iv <= hi: raise JErr(f".{name}(): {v} is out of range")
                out.append(iv)
        elif name in ("ceiling", "floor", "abs"):
            if not _num_item(x): raise JErr(f".{name}() of a {_type_name(x)}")
            if name == "abs": out.append(abs(x))
            elif isinstance(x, int): out.append(x)
            elif isinstance(x, float): out.append(float(math.ceil(x) if name == "ceiling" else math.floor(x)))
            else: out.append(int(math.ceil(x) if name == "ceiling" else math.floor(x)))
        elif name == "boolean":
            if isinstance(x, bool): out.append(x)
            elif _num_item(x) and not isinstance(x, float) and x == int(x): out.append(x != 0)
            elif isinstance(x, str) and x.strip().lower() in ("true", "false", "t", "f", "yes", "no", "y", "n", "1", "0", "on", "off"):
                out.append(x.strip().lower() in ("true", "t", "yes", "y", "1", "on"))
            else: raise JErr(f".boolean() of {x!r}")
        elif name == "string":
            if isinstance(x, (list, dict)): raise JErr(f".string() of a {_type_name(x)}")
            out.append(x if isinstance(x, str) else dumps(x) if not isinstance(x, (_dt.date, _dt.time)) else T.to_text(x, None))
        elif name == "datetime":
            out.append(_datetime_item(x, args[0] if args else None))
        elif name in ("date", "time", "time_tz", "timestamp", "timestamp_tz"):
            v = x if isinstance(x, (_dt.date, _dt.time)) else _datetime_item(x)
            p = int(args[0]) if args else None
            try:
                if name == "date":
                    if isinstance(v, _dt.time): raise JErr(".date() of a time")
                    out.append(v.date() if isinstance(v, _dt.datetime) else v); continue
                if name in ("time", "time_tz"):
                    if isinstance(v, _dt.datetime): v = v.timetz()
                    elif not isinstance(v, _dt.time): raise JErr(f".{name}() of a date")
                    v = (v if v.tzinfo is not None else T.localize(v)) if name == "time_tz" else (v.replace(tzinfo=None) if v.tzinfo is None else T.local(v))
                else:
                    if isinstance(v, _dt.time): raise JErr(f".{name}() of a time")
                    if not isinstance(v, _dt.datetime): v = _dt.datetime(v.year, v.month, v.day)
                    v = (v if v.tzinfo is not None else T.localize(v)) if name == "timestamp_tz" else (v if v.tzinfo is None else T.local(v))
            except SqlError as ex: raise JErr(str(ex))
            if p is not None: q = 10 ** (6 - min(p, 6)); v = v.replace(microsecond=v.microsecond // q * q)
            out.append(v)
        elif name == "keyvalue":
            if not isinstance(x, dict): raise JErr(f".keyvalue() of a {_type_name(x)}")
            oid = cx.object_id(x)
            for kk, vv in x.items(): out.append({"name": kk, "value": vv, "id": oid})
        else: raise JErr(f"unknown item method .{name}()")
    return out


# ------------------------------------------------------------------ predicates: True / False / None (unknown)
def _cmpable(a, b):
    if a is None or b is None: return True
    if isinstance(a, bool) or isinstance(b, bool): return isinstance(a, bool) and isinstance(b, bool)
    if _num_item(a) and _num_item(b): return True
    if isinstance(a, str) and isinstance(b, str): return True
    if isinstance(a, _dt.datetime) and isinstance(b, _dt.datetime): return (a.tzinfo is None) == (b.tzinfo is None) or True
    if isinstance(a, _dt.date) and isinstance(b, _dt.date): return True
    if isinstance(a, _dt.time) and isinstance(b, _dt.time): return True
    return False


def _cmp1(op, a, b):
    """one comparison: True / False, or JErr when the items do not compare"""
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)): raise JErr("arrays and objects do not compare")
    if a is None or b is None:
        eq = a is None and b is None
        if op == "==": return eq
        if op == "!=": return not eq
        if op in ("<=", ">="): return eq
        return False
    if not _cmpable(a, b): raise JErr("the items do not compare")
    try:
        if isinstance(a, _dt.date) and isinstance(b, _dt.date) and type(a) is not type(b):
            a, b = (a if isinstance(a, _dt.datetime) else _dt.datetime(a.year, a.month, a.day)), (b if isinstance(b, _dt.datetime) else _dt.datetime(b.year, b.month, b.day))
        if isinstance(a, (_dt.datetime, _dt.time)) and (a.tzinfo is None) != (b.tzinfo is None):
            a = T.localize(a) if a.tzinfo is None else a; b = T.localize(b) if b.tzinfo is None else b
        if _num_item(a) and _num_item(b) and (isinstance(a, float) != isinstance(b, float)): a, b = float(a), float(b)
        c = (a > b) - (a < b)
    except TypeError: raise JErr("the items do not compare")
    return {"==": c == 0, "!=": c != 0, "<": c < 0, "<=": c <= 0, ">": c > 0, ">=": c >= 0}[op]


def _pred(p, cx, cur):
    k = p[0]
    if k == "and":
        a = _pred(p[1], cx, cur)
        if a is False: return False
        b = _pred(p[2], cx, cur)
        if b is False: return False
        return True if a is True and b is True else None
    if k == "or":
        a = _pred(p[1], cx, cur)
        if a is True: return True
        b = _pred(p[2], cx, cur)
        if b is True: return True
        return False if a is False and b is False else None
    if k == "not":
        a = _pred(p[1], cx, cur)
        return None if a is None else not a
    if k == "unknown": return _pred(p[1], cx, cur) is None
    if k == "exists":
        try: return len(_ev(p[1], cx, cur, None)) > 0
        except JErr: return None
    try:
        if k == "cmp":
            a = _unwrap(_ev(p[2], cx, cur, None), cx); b = _unwrap(_ev(p[3], cx, cur, None), cx)
        else: a = _unwrap(_ev(p[1], cx, cur, None), cx)
    except JErr: return None
    found, err = False, False
    if k == "cmp":
        for x in a:
            for y in b:
                try:
                    if _cmp1(p[1], x, y): found = True
                except JErr: err = True
                if found and cx.lax: return True
                if err and not cx.lax: return None
    elif k == "like_regex":
        import sqlfuncs
        rx = sqlfuncs._rx(p[2], sqlfuncs._xq_flags(p[3]))
        for x in a:
            if not isinstance(x, str): err = True; continue
            if rx.search(x): found = True
            if found and cx.lax: return True
            if err and not cx.lax: return None
    elif k == "starts":
        pre = _ev(p[2], cx, cur, None)
        if len(pre) != 1 or not isinstance(pre[0], str): return None
        for x in a:
            if not isinstance(x, str): err = True; continue
            if x.startswith(pre[0]): found = True
            if found and cx.lax: return True
            if err and not cx.lax: return None
    if err: return None
    return found


# ================================================================== the query functions
def _context(text, t):
    """the context item of a query function: JSON text (or a JSON value)"""
    if t is not None and t.id not in ("JSON", "VARCHAR", "NULL", "BLOB"):
        raise SqlError(f"the context item of a JSON query function must be JSON text, not {t}")
    if isinstance(text, (bytes, bytearray)): text = text.decode("utf-8")
    return parse(text)


def _vars(names, vals, types, fmts=None):
    fmts = fmts or [False] * len(names)
    return {n: from_sql(v, t, f, items=True) for n, v, t, f in zip(names, vals, types, fmts)}


def path_vars(path):
    """the names of the variables a compiled path reads"""
    out = set()
    def walk(e):
        if isinstance(e, tuple):
            if len(e) == 2 and e[0] == "var" and isinstance(e[1], str): out.add(e[1])
            for x in e: walk(x)
        elif isinstance(e, list):
            for x in e: walk(x)
    walk(path[1])
    return out


def check_vars(path_text, names):
    """a path variable without a PASSING value is an error of the query, not of the data"""
    miss = sorted(path_vars(compile_path(path_text)) - set(names))
    if miss: raise SqlError(f"SQL/JSON path {path_text!r}: no PASSING value named {', '.join('$' + m for m in miss)}")


class BehaviorError(SqlError):
    """ERROR ON EMPTY / ERROR ON ERROR asked for"""


def _behavior(spec, err, defaults):
    """ON EMPTY / ON ERROR: ('ERROR',) raises; ('NULL',); ('DEFAULT', value)"""
    kind = spec[0]
    if kind == "ERROR": raise BehaviorError(f"SQL/JSON: {err}")
    if kind == "DEFAULT": return ("value", defaults[spec[1]])
    if kind == "EMPTY ARRAY": return ("value", "[]")
    if kind == "EMPTY OBJECT": return ("value", "{}")
    if kind in ("TRUE", "FALSE"): return ("value", kind == "TRUE")
    if kind == "UNKNOWN": return ("value", None)
    return ("value", None)


def json_value(o, ctx, pvals, ptypes, defaults, rt):
    """JSON_VALUE: the one scalar the path selects, as the RETURNING type; NULL ON EMPTY / ON ERROR by default"""
    try: item = _context(ctx, None); vars_ = _vars(o["names"], pvals, ptypes, o.get("pfmt"))
    except JErr as ex: return _err_or(o, ex, defaults, rt)
    return _value_item(o, item, vars_, defaults, rt)


def _value_item(o, item, vars_, defaults, rt):
    try: seq = evaluate(o["cpath"], item, vars_)
    except JErr as ex: return _err_or(o, ex, defaults, rt)
    if not seq: return _empty_or(o, "no SQL/JSON item", defaults, rt)
    if len(seq) > 1: return _err_or(o, f"{len(seq)} SQL/JSON items where one is needed", defaults, rt)
    x = seq[0]
    if isinstance(x, (list, dict)): return _err_or(o, f"an SQL/JSON {_type_name(x)} is not a scalar", defaults, rt)
    if x is None: return None
    try: return _to_sql(x, rt)
    except (SqlError, JErr) as ex: return _err_or(o, ex, defaults, rt)


def _empty_or(o, msg, defaults, rt):
    return _cast_default(_behavior(o["empty"], msg, defaults)[1], rt)


def _err_or(o, msg, defaults, rt=None):
    return _cast_default(_behavior(o["error"], msg, defaults)[1], rt)


def _cast_default(v, rt):
    if v is None or rt is None: return v
    return T.cast_value(v, T.py_type(v), rt)


def _to_sql(x, rt):
    """an SQL/JSON scalar as an SQL value of type rt"""
    if rt is None or rt.id in ("VARCHAR", "JSON"):
        if isinstance(x, str): return x
        if isinstance(x, (_dt.date, _dt.time)): return T.to_text(x, None)
        return dumps(x)
    if isinstance(x, bool):
        if rt.id == "BOOLEAN": return x
        raise JErr(f"a boolean is not {rt}")
    if _num_item(x):
        if not (rt.numeric or rt.id == "BOOLEAN"): raise JErr(f"a number is not {rt}")
        return T.cast_value(x, T.DOUBLE if isinstance(x, float) else T.DECIMAL(38, 10) if isinstance(x, Decimal) else T.HUGEINT, rt)
    if isinstance(x, str): return T.cast_value(x, T.VARCHAR, rt)
    if isinstance(x, (_dt.date, _dt.time)):
        return T.cast_value(x, T.py_type(x), rt)
    raise JErr(f"cannot return {_type_name(x)} as {rt}")


def json_query(o, ctx, pvals, ptypes):
    """JSON_QUERY: the selected items as JSON text, with or without an array wrapper"""
    try: item = _context(ctx, None); vars_ = _vars(o["names"], pvals, ptypes, o.get("pfmt"))
    except JErr as ex: return _behavior(o["error"], ex, [])[1]
    return _query_item(o, item, vars_)


def _query_item(o, item, vars_):
    try: seq = evaluate(o["cpath"], item, vars_)
    except JErr as ex: return _behavior(o["error"], ex, [])[1]
    w = o["wrapper"]
    if not seq: return _behavior(o["empty"], "no SQL/JSON item", [])[1]          # ON EMPTY comes before any wrapper
    # CONDITIONAL: a wrapper only around several items (one item, even a scalar, stays as it is)
    if w == "UNCONDITIONAL" or (w == "CONDITIONAL" and len(seq) != 1):
        out = list(seq)
    else:
        if len(seq) != 1: return _behavior(o["error"], f"{len(seq)} SQL/JSON items without an array wrapper", [])[1]
        out = seq[0]
        if not isinstance(out, (list, dict)):
            if o["quotes"] == "OMIT" and isinstance(out, str): return out
            if not o.get("scalars", True): return _behavior(o["error"], "a scalar without an array wrapper", [])[1]
    try: return dumps(out)
    except JErr as ex: return _behavior(o["error"], ex, [])[1]


def json_exists(o, ctx, pvals, ptypes):
    """JSON_EXISTS: does the path select anything? FALSE ON ERROR by default"""
    try:
        item = _context(ctx, None)
        return len(evaluate(o["cpath"], item, _vars(o["names"], pvals, ptypes, o.get("pfmt")))) > 0
    except JErr as ex: return _behavior(o["error"], ex, [])[1]


def is_json(text, kind="VALUE", unique=False):
    """x IS JSON [VALUE | ARRAY | OBJECT | SCALAR] [WITH UNIQUE KEYS]"""
    if isinstance(text, (bytes, bytearray)):
        try: text = text.decode("utf-8")
        except UnicodeDecodeError: return False
    try: v = parse(text, unique)
    except JErr: return False
    if kind == "ARRAY": return isinstance(v, list)
    if kind == "OBJECT": return isinstance(v, dict)
    if kind == "SCALAR": return not isinstance(v, (list, dict))
    return True


def json_object(keys, vals, types, fmts, absent_on_null=False, unique=False):
    """JSON_OBJECT(k VALUE v, ...): NULL ON NULL by default"""
    out, seen = [], set()                  # members in order; a repeated key stays unless WITH UNIQUE KEYS (ISO 6.39)
    for k, v, t, f in zip(keys, vals, types, fmts):
        if k is None: raise SqlError("data exception: null value not allowed (a JSON object key)")
        if v is None and absent_on_null: continue
        k = k if isinstance(k, str) else T.to_text(k, None)
        if unique and k in seen: raise SqlError(f"data exception: duplicate JSON object key value {k!r}")
        seen.add(k)
        try: out.append(json.dumps(k, ensure_ascii=False) + ":" + dumps(from_sql(v, t, f)))
        except JErr as ex: raise SqlError(f"JSON_OBJECT: {ex}")
    return "{" + ",".join(out) + "}"


def json_array(vals, types, fmts, null_on_null=False):
    """JSON_ARRAY(v, ...): ABSENT ON NULL by default"""
    out = []
    for v, t, f in zip(vals, types, fmts):
        if v is None and not null_on_null: continue
        try: out.append(from_sql(v, t, f))
        except JErr as ex: raise SqlError(f"JSON_ARRAY: {ex}")
    return dumps(out)


def json_scalar(v, t):
    """JSON_SCALAR(v): an SQL scalar as a JSON value"""
    if v is None: return None
    if isinstance(v, float) and not math.isfinite(v): raise SqlError(f"JSON_SCALAR: {v} has no JSON form")
    if isinstance(v, (_dt.date, _dt.time)): return dumps(T.to_text(v, t))
    try: return dumps(from_sql(v, t))
    except JErr as ex: raise SqlError(f"JSON_SCALAR: {ex}")


# ================================================================== JSON_TABLE
def json_table_rows(spec, ctx, pvals, ptypes):
    """the rows of JSON_TABLE (dicts: output position -> value): the row path's items, each with its columns; a
    NESTED PATH's rows joined to their parent row as an outer join, sibling NESTED PATHs as a union (ISO's default
    plan)"""
    try:
        item = parse(ctx) if not isinstance(ctx, (bytes, bytearray)) else parse(ctx.decode("utf-8"))
        vars_ = _vars(spec["names"], pvals, ptypes, spec.get("pfmt"))
        items = evaluate(spec["cpath"], item, vars_)
    except JErr as ex:
        if spec["error"] == "ERROR": raise SqlError(f"JSON_TABLE: {ex}")
        return []
    J = _JT(spec, vars_)
    plan = spec.get("plan")
    if plan is not None:
        if plan[0] == "pc" and plan[1] != spec.get("pname"): raise SqlError(f"JSON_TABLE PLAN: {plan[1]} is not the row path's name")
        if plan[0] not in ("pc", "name"): raise SqlError("JSON_TABLE PLAN: starts with the row path's name")
    top = (plan[2], plan[3]) if plan is not None and plan[0] == "pc" else None
    out = []
    for ordn, it in enumerate(items, 1): out += J.item_rows(spec["cols"], it, ordn, top)
    return out


class _JT:
    """JSON_TABLE's rows: a path's items, each with its columns, joined with its NESTED PATHs' rows as the plan says
    (ISO 7.11: by default OUTER, and sibling paths as a UNION; PLAN DEFAULT (INNER | OUTER, CROSS | UNION) changes
    both, PLAN (p OUTER (a CROSS b)) sets them per path)"""
    def __init__(self, spec, vars_):
        self.spec, self.vars, self.on_error, self.defaults = spec, vars_, spec["error"], spec["defaults"]
        self.pdef = spec.get("plan_default") or ["OUTER", "UNION"]

    def item_rows(self, cols, it, ordn, plan):
        base = {c["pos"]: _jt_col(c, it, self.vars, ordn, self.on_error, self.defaults) for c in cols if c["kind"] != "nested"}
        nested = {(c.get("pname") or f"#{i}"): c for i, c in enumerate(cols) if c["kind"] == "nested"}
        if not nested: return [base]
        if plan is None: plan = (self.pdef[0], ("sib", self.pdef[1], [("name", n) for n in nested]))
        else:
            def names(x): return [x[1]] if x[0] in ("name", "pc") else [n for y in x[2] for n in names(y)]
            got = names(plan[1])
            if sorted(got) != sorted(nested): raise SqlError(f"JSON_TABLE PLAN: the plan names {', '.join(got)}; the NESTED PATHs here are {', '.join(nested)} (each once)")
        join, tree = plan
        sub = self.plan_rows(tree, nested, it)
        if not sub: return [base] if join == "OUTER" else []
        return [{**base, **r} for r in sub]

    def plan_rows(self, tree, nested, it):
        kind = tree[0]
        if kind in ("name", "pc"):
            nl = nested.get(tree[1])
            if nl is None: raise SqlError(f"JSON_TABLE PLAN: {tree[1]} is not a NESTED PATH here")
            try: sub = evaluate(nl["cpath"], it, self.vars)
            except JErr as ex:
                if self.on_error == "ERROR": raise SqlError(f"JSON_TABLE: {ex}")
                sub = []
            plan = (tree[2], tree[3]) if kind == "pc" else None
            if plan is None and self.spec.get("plan") is not None and any(c["kind"] == "nested" for c in nl["cols"]):
                raise SqlError(f"JSON_TABLE PLAN: {tree[1]} has NESTED PATHs, so the plan names them ({tree[1]} OUTER | INNER (...))")
            out = []
            for ordn, s in enumerate(sub, 1): out += self.item_rows(nl["cols"], s, ordn, plan)
            return out
        parts = [self.plan_rows(x, nested, it) for x in tree[2]]
        if tree[1] == "UNION": return [r for p in parts for r in p]
        out = [{}]
        for p in parts: out = [{**a, **b} for a in out for b in p]
        return out


def _jt_col(c, it, vars_, ordn, on_error, defaults):
    kind = c["kind"]
    if kind == "ordinality": return ordn
    o = c["opts"]
    try:
        if kind == "exists":
            try: return len(evaluate(o["cpath"], it, vars_)) > 0
            except JErr as ex: return _behavior(o["error"], ex, defaults)[1]
        if kind == "query": return _query_item(o, it, vars_)
        return _value_item(o, it, vars_, defaults, c["rt"])
    except BehaviorError: raise
    except SqlError:
        if on_error == "ERROR" or o["error"][0] == "ERROR": raise
        return None
