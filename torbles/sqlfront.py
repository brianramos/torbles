# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SQL text -> bound query (sqlir). DuckDB's parser reads the statement (json_serialize_sql: syntax only, no data);
a token pre-pass first rewrites the ISO/IEC 9075 forms DuckDB's grammar lacks (BETWEEN SYMMETRIC, LIKE_REGEX,
LISTAGG ... WITHIN GROUP, hypothetical set functions, the *_REGEX functions, OVERLAPS, UNIQUE, FETCH ... WITH TIES,
SUBSTRING ... SIMILAR, NTH_VALUE ... FROM LAST, INTERVAL ... TO ...) into marker function calls it can parse. The
binder resolves names against the torbles catalog, types every expression (sqlfuncs) and builds sqlir blocks."""
import json, re
from decimal import Decimal
import sqltypes as T
from sqltypes import SqlError
import sqlir as I
import sqlfuncs as F

# ================================================================== ISO pre-pass
_TOK = re.compile(r"""
    (?P<ws>\s+) | (?P<lc>--[^\n]*) | (?P<bc>/\*.*?\*/) |
    (?P<ustr>[uU]&'(?:[^']|'')*') | (?P<uqid>[uU]&"(?:[^"]|"")*") | (?P<bstr>[xX]'(?:[^']|'')*') |
    (?P<str>[eEnN]?'(?:[^']|'')*') | (?P<qid>"(?:[^"]|"")*") |
    (?P<num>0[xX](?:_?[0-9A-Fa-f])+ | 0[oO](?:_?[0-7])+ | 0[bB](?:_?[01])+ |
            (?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)? | \.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?) |
    (?P<id>[A-Za-z_][A-Za-z_0-9$]*) |
    (?P<op>::|<=|>=|<>|!=|\|\||->>|->|\*\*|//|<<|>>|!~~\*|!~~|~~\*|~~~|~~|[-+*/%^=<>!~&|(),.;\[\]{}:?#@$])
""", re.X | re.S)


class Tok:
    __slots__ = ("k", "s", "done")
    def __init__(self, k, s): self.k, self.s, self.done = k, s, False
    def up(self): return self.s.upper() if self.k == "id" else None
    def __repr__(self): return self.s


def _unicode_escapes(body, esc, what):
    """the text of a U&'...' literal or U&"..." identifier: <esc>XXXX and <esc>+XXXXXX are code points, <esc><esc> is
    the escape character itself (ISO 5.3)"""
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c != esc: out.append(c); i += 1; continue
        if body[i + 1:i + 2] == esc: out.append(esc); i += 2; continue
        m = re.match(r"\+([0-9A-Fa-f]{6})|([0-9A-Fa-f]{4})", body[i + 1:])
        if not m: raise SqlError(f"invalid Unicode escape in {what}")
        cp = int(m.group(1) or m.group(2), 16)
        if cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF: raise SqlError(f"invalid Unicode code point {cp:X} in {what}")
        out.append(chr(cp)); i += 1 + len(m.group(0))
    return "".join(out)


def _num_text(s):
    """a numeric literal as DuckDB reads it: underscores dropped, 0x / 0o / 0b integers in decimal (ISO 2023 T661, T662)"""
    s = s.replace("_", "")
    if len(s) > 2 and s[0] == "0" and s[1] in "xXoObB":
        return str(int(s[2:], {"x": 16, "o": 8, "b": 2}[s[1].lower()]))
    return s


def tokenize(sql):
    out, i = [], 0
    while i < len(sql):
        m = _TOK.match(sql, i)
        if not m: raise SqlError(f"syntax error at {sql[i:i + 20]!r}")
        k, x = m.lastgroup, m.group(0)
        i = m.end()
        if k in ("ws", "lc", "bc"): continue
        if k == "num": x = _num_text(x)
        elif k in ("ustr", "uqid"):
            q = x[2]; body = x[3:-1].replace(q + q, q)
            esc = "\\"
            m2 = re.match(r"(\s|--[^\n]*|/\*.*?\*/)*UESCAPE\s*'([^'])'", sql[i:], re.I | re.S)
            if m2:
                esc = m2.group(2); i += m2.end()
                if esc in "0123456789abcdefABCDEF+'\" \t\n": raise SqlError(f"invalid UESCAPE character {esc!r}")
            text = _unicode_escapes(body, esc, "a Unicode literal")
            if k == "ustr": k, x = "str", "'" + text.replace("'", "''") + "'"
            else: k, x = "qid", '"' + text.replace('"', '""') + '"'
        elif k == "str" and x[0] in "nN": x = x[1:]                       # national character literal: one character set
        if k == "str" and x[0] == "'" and out and out[-1].k == "str" and out[-1].s[0] == "'":
            out[-1] = Tok("str", out[-1].s[:-1] + x[1:]); continue        # 'a' 'b': one literal (ISO 5.3 separators)
        if k == "bstr" and out and out[-1].k == "bstr":
            out[-1] = Tok("bstr", out[-1].s[:-1] + x[2:]); continue
        out.append(Tok(k, x))
    return out


_OPEN, _CLOSE = {"(": ")", "[": "]"}, {")": "(", "]": "["}


def _match_paren(t, i):
    """index of the ')' / ']' matching the '(' / '[' at i"""
    d = 0
    for j in range(i, len(t)):
        if t[j].s in _OPEN: d += 1
        elif t[j].s in _CLOSE:
            d -= 1
            if d == 0: return j
    raise SqlError("unbalanced parentheses")


def _match_open(t, j):
    d = 0
    for i in range(j, -1, -1):
        if t[i].s in _CLOSE: d += 1
        elif t[i].s in _OPEN:
            d -= 1
            if d == 0: return i
    raise SqlError("unbalanced parentheses")


def _operand_back(t, j):
    """start index of the primary expression ending at token j"""
    i = j
    if t[i].k == "str" and i > 0 and t[i - 1].up() in ("DATE", "TIME", "TIMESTAMP", "TIMESTAMPTZ", "TIMETZ", "INTERVAL"):
        return i - 1                                    # a typed literal: TIMESTAMP '...'
    while True:
        if t[i].s in _CLOSE:
            i = _match_open(t, i)
            if i > 0 and t[i - 1].k in ("id", "qid") and t[i - 1].up() not in _KW_BREAK: i -= 1
            elif i > 0 and t[i].s == "[" and t[i - 1].s in _CLOSE: i -= 1; continue        # f(x)[2], a[1][2]
        break
    while i >= 2 and t[i - 1].s == "." and t[i - 2].k in ("id", "qid"): i -= 2
    while i >= 2 and t[i - 1].s == "::": i -= 2
    return i


def _operand_fwd(t, i):
    """end index (exclusive) of the primary expression starting at token i"""
    j = i
    if t[j].s in _OPEN: j = _match_paren(t, j) + 1
    elif t[j].k in ("id", "qid"):
        j += 1
        while j + 1 < len(t) and t[j].s == "." : j += 2
        if j < len(t) and t[j].s in _OPEN: j = _match_paren(t, j) + 1
    elif t[j].s in ("-", "+"): return _operand_fwd(t, j + 1)
    else: j += 1
    while j < len(t) and t[j].s == "[": j = _match_paren(t, j) + 1       # subscripts
    while j + 1 < len(t) and t[j].s == "::": j += 2
    return j


_KW_BREAK = {"SELECT", "WHERE", "AND", "OR", "NOT", "ON", "WHEN", "THEN", "ELSE", "HAVING", "BY", "IN", "IS", "LIKE",
             "BETWEEN", "CASE", "FROM", "AS", "EXISTS", "ALL", "ANY", "SOME", "RETURN", "DISTINCT"}
_EXPR_END = {"AND", "OR", "THEN", "WHEN", "ELSE", "END", "FROM", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET",
             "FETCH", "UNION", "EXCEPT", "INTERSECT", "WINDOW", "QUALIFY", "ON", "JOIN", "INNER", "LEFT", "RIGHT", "FULL",
             "CROSS", "USING", "AS", "ASC", "DESC", "NULLS", "ROWS", "RANGE", "GROUPS", "PRECEDING", "FOLLOWING", "IS"}


def _split_commas(t, a, b):
    """top-level comma-separated parts of tokens a..b-1"""
    parts, d, s = [], 0, a
    for j in range(a, b):
        if t[j].s in ("(", "["): d += 1
        elif t[j].s in (")", "]"): d -= 1
        elif t[j].s == "," and d == 0: parts.append((s, j)); s = j + 1
    parts.append((s, b))
    return parts


def _text(t, a, b): return " ".join(x.s for x in t[a:b])


def _depth0(t, a, k):
    """is token k at parenthesis depth 0 relative to a?"""
    d = 0
    for j in range(a, k):
        if t[j].s in ("(", "["): d += 1
        elif t[j].s in (")", "]"): d -= 1
    return d == 0


def _kw(t, i, *words):
    return all(i + k < len(t) and t[i + k].up() == w for k, w in enumerate(words))


def _top_kw(t, a, b, word):
    d = 0
    for j in range(a, b):
        if t[j].s == "(": d += 1
        elif t[j].s == ")": d -= 1
        elif d == 0 and t[j].up() == word: return j
    return -1


def _rx_args(t, a, b, name):
    """the arguments of OCCURRENCES_REGEX / POSITION_REGEX / SUBSTRING_REGEX / TRANSLATE_REGEX(...):
    pattern [FLAG f] IN subject [WITH repl] [FROM start] [USING ...] [OCCURRENCE n | ALL] [GROUP g]"""
    kws = {"FLAG", "IN", "WITH", "FROM", "USING", "OCCURRENCE", "GROUP"}
    parts, cur, d, key = {}, [], 0, "PAT"
    j = a
    if name == "POSITION_REGEX" and t[j].up() in ("START", "AFTER"):
        parts["MODE"] = t[j].up(); j += 1
    while j < b:
        x = t[j]
        if x.s == "(": d += 1
        elif x.s == ")": d -= 1
        if d == 0 and x.up() in kws:
            parts[key] = _text(t, cur[0], cur[-1] + 1) if cur else ""
            key, cur = x.up(), []
            if key == "USING":
                parts["OCTETS"] = t[j + 1].up() == "OCTETS"; j += 2; continue
            j += 1; continue
        cur.append(j); j += 1
    parts[key] = _text(t, cur[0], cur[-1] + 1) if cur else ""
    occ = parts.get("OCCURRENCE", "")
    if occ.upper() == "ALL": occ = "0"
    def a_(k, default): return parts.get(k) or default
    oct_ = "TRUE" if parts.get("OCTETS") else "FALSE"
    if name == "TRANSLATE_REGEX":
        return f"__iso_translate_regex({parts['PAT']}, {parts['IN']}, {a_('WITH', chr(39) * 2)}, {a_('FLAG', chr(39) * 2)}, {occ or '0'})"
    if name == "OCCURRENCES_REGEX":
        return f"__iso_occurrences_regex({parts['PAT']}, {parts['IN']}, {a_('FLAG', chr(39) * 2)}, {a_('FROM', '1')}, {oct_})"
    if name == "POSITION_REGEX":
        after = "TRUE" if parts.get("MODE") == "AFTER" else "FALSE"
        return f"__iso_position_regex({parts['PAT']}, {parts['IN']}, {a_('FLAG', chr(39) * 2)}, {a_('FROM', '1')}, {occ or '1'}, {a_('GROUP', '0')}, {after}, {oct_})"
    return f"__iso_substring_regex({parts['PAT']}, {parts['IN']}, {a_('FLAG', chr(39) * 2)}, {a_('FROM', '1')}, {occ or '1'}, {a_('GROUP', '0')}, {oct_})"


FLAGS = {}

_UNITS = ("YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND")
_CMP_OPS = ("=", "<>", "!=", "<", ">", "<=", ">=")
_PRED_START = ("IS", "NOT", "BETWEEN", "IN", "LIKE", "ILIKE", "SIMILAR", "MEMBER", "SUBMULTISET", "OVERLAPS", "LIKE_REGEX", "GLOB")


def _marker(neg, name, args):
    """a postfix predicate  x <pred>  as  x IS [NOT] DISTINCT FROM __iso_pred_<name>(args): DuckDB parses it with the
    precedence of a predicate, and the binder turns it back into the predicate"""
    return tokenize(f"IS {'' if neg else 'NOT '}DISTINCT FROM __iso_pred_{name}({', '.join(args)})")


def _qualifier(t, j):
    """an interval qualifier at j: <unit> [(p [, s])] [TO <unit> [(p)]] -> (text 'DAY TO SECOND', end index) or None"""
    if j >= len(t) or t[j].up() not in _UNITS: return None
    a = t[j].up(); j += 1
    if j < len(t) and t[j].s == "(": j = _match_paren(t, j) + 1
    b = None
    if j + 1 < len(t) and t[j].up() == "TO" and t[j + 1].up() in _UNITS:
        b = t[j + 1].up(); j += 2
        if j < len(t) and t[j].s == "(": j = _match_paren(t, j) + 1
    return (a if b is None else f"{a} TO {b}"), j


def _case_end(t, i):
    """index of the END closing the CASE at i"""
    d = 0
    for j in range(i, len(t)):
        u = t[j].up()
        if u == "CASE": d += 1
        elif u == "END":
            d -= 1
            if d == 0: return j
    raise SqlError("CASE without END")


def _case_top(t, a, b, words):
    """top-level (outside parentheses and nested CASE) positions of the keywords in words within a..b-1"""
    out, d, c = [], 0, 0
    for j in range(a, b):
        x = t[j]
        if x.s == "(": d += 1
        elif x.s == ")": d -= 1
        elif x.up() == "CASE": c += 1
        elif x.up() == "END": c -= 1
        elif d == 0 and c == 0 and x.up() in words: out.append(j)
    return out


def _extended_case(t, i):
    """CASE x WHEN 1, 2 THEN .. WHEN > 5 THEN .. WHEN IS NULL THEN .. (ISO F262 / F263) -> a searched CASE, or None"""
    e = _case_end(t, i)
    tops = _case_top(t, i + 1, e, ("WHEN", "THEN", "ELSE"))
    if not tops or t[tops[0]].up() != "WHEN" or tops[0] == i + 1: return None
    op = t[i + 1:tops[0]]
    whens = [k for k in tops if t[k].up() == "WHEN"]
    thens = [k for k in tops if t[k].up() == "THEN"]
    if len(whens) != len(thens): return None
    ext = False
    conds = []
    for w, th in zip(whens, thens):
        items = _split_commas(t, w + 1, th)
        if len(items) > 1: ext = True
        parts = []
        for a, b in items:
            first = t[a]
            if first.s in _CMP_OPS or first.up() in _PRED_START:
                ext = True; parts.append(["("] + [x.s for x in op] + [")"] + [x.s for x in t[a:b]])
            else:
                parts.append(["("] + [x.s for x in op] + [")", "=", "("] + [x.s for x in t[a:b]] + [")"])
        conds.append(" OR ".join("(" + " ".join(p) + ")" for p in parts))
    if not ext: return None
    out = [Tok("id", "CASE")]
    k = 0
    pos = whens + [e]
    for idx, (w, th) in enumerate(zip(whens, thens)):
        nxt = next((x for x in tops if x > th and t[x].up() in ("WHEN", "ELSE")), e)
        out += [Tok("id", "WHEN")] + tokenize(conds[idx]) + t[th:nxt]
    els = next((x for x in tops if t[x].up() == "ELSE"), None)
    if els is not None: out += t[els:e]
    out.append(Tok("id", "END"))
    return out, e


# ------------------------------------------------------------------ SQL/JSON syntax
_JCLAUSE = {"PASSING", "RETURNING", "ERROR", "NULL", "DEFAULT", "EMPTY", "WITH", "WITHOUT", "KEEP", "OMIT", "TRUE", "FALSE",
            "UNKNOWN", "COLUMNS", "FORMAT", "ABSENT", "ON", "PLAN", "AS"}


def _jstr(x): return "'" + json.dumps(x).replace("'", "''") + "'"


class _JParse:
    """the clauses of an SQL/JSON function call, from its tokens"""
    def __init__(self, t, a, b, what):
        self.t, self.i, self.b, self.what = t, a, b, what
        self.passing, self.names, self.defaults = [], [], []
        self.opts = dict(empty=["NULL"], error=["NULL"], wrapper="WITHOUT", quotes="KEEP", returning=None, names=[])

    def at(self, *words): return _kw(self.t, self.i, *words) and self.i + len(words) <= self.b
    def done(self): return self.i >= self.b
    def err(self, msg): raise SqlError(f"{self.what}: {msg}")

    def expr_until(self, stop):
        """tokens of an expression up to a top-level stop (a word, or a tuple of words that must follow each other),
        a comma, or the end"""
        stops = [(w,) if isinstance(w, str) else tuple(w) for w in stop]
        j, d = self.i, 0
        while j < self.b:
            x = self.t[j]
            if x.s in ("(", "["): d += 1
            elif x.s in (")", "]"): d -= 1
            elif d == 0 and (x.s == "," or (j > self.i and any(_kw(self.t, j, *w) for w in stops))): break
            j += 1
        out = _text(self.t, self.i, j); self.i = j
        if not out: self.err("an expression is missing")
        return out

    def format_json(self):
        if self.at("FORMAT", "JSON"):
            self.i += 2
            if self.at("ENCODING"): self.i += 2
            return True
        return False

    def path_and_passing(self):
        x = self.t[self.i] if self.i < self.b else None
        if x is None or x.k != "str": self.err("the path must be a character string literal")
        self.opts["path"] = x.s[1:-1].replace("''", "'"); self.i += 1
        if self.at("AS"): self.i += 2                   # a path name (JSON_TABLE)
        if self.at("PASSING"):
            self.i += 1
            while True:
                self.passing.append(self.expr_until([("AS",)]))
                if not self.at("AS"): self.err("PASSING <value> AS <name>")
                self.i += 1; self.names.append(self.t[self.i].s.strip('"')); self.i += 1
                if self.i < self.b and self.t[self.i].s == ",": self.i += 1; continue
                break
        self.opts["names"] = self.names

    def type_until(self):
        j, d = self.i, 0
        while j < self.b:
            u = self.t[j].up()
            if self.t[j].s == "(": d += 1
            elif self.t[j].s == ")": d -= 1
            elif d == 0:
                if u in ("ERROR", "NULL", "DEFAULT", "EMPTY", "KEEP", "OMIT", "TRUE", "FALSE", "UNKNOWN", "FORMAT", "PATH", "EXISTS"): break
                if u in ("WITH", "WITHOUT") and not _kw(self.t, j + 1, "TIME", "ZONE"): break
                if self.t[j].s == ",": break
            j += 1
        out = _text(self.t, self.i, j); self.i = j
        return out

    def clauses(self, allow):
        while not self.done():
            if self.t[self.i].s == ",": return                # the end of a JSON_TABLE column definition
            if self.at("RETURNING") and "returning" in allow:
                self.i += 1; self.opts["returning"] = self.type_until(); self.format_json(); continue
            if (self.at("WITHOUT", "ARRAY", "WRAPPER") or self.at("WITHOUT", "WRAPPER")) and "wrapper" in allow:
                self.i += 3 if self.at("WITHOUT", "ARRAY") else 2; self.opts["wrapper"] = "WITHOUT"; continue
            if self.at("WITH") and "wrapper" in allow and not self.at("WITH", "UNIQUE"):
                self.i += 1; kind = "UNCONDITIONAL"
                if self.at("CONDITIONAL"): kind = "CONDITIONAL"; self.i += 1
                elif self.at("UNCONDITIONAL"): self.i += 1
                if self.at("ARRAY"): self.i += 1
                if not self.at("WRAPPER"): self.err("WITH [CONDITIONAL | UNCONDITIONAL] [ARRAY] WRAPPER")
                self.i += 1; self.opts["wrapper"] = kind; continue
            if (self.at("KEEP") or self.at("OMIT")) and "quotes" in allow:
                self.opts["quotes"] = self.t[self.i].up(); self.i += 1
                if self.at("QUOTES"): self.i += 1
                if self.at("ON", "SCALAR", "STRING"): self.i += 3
                continue
            beh = None
            for words in (("ERROR",), ("NULL",), ("EMPTY", "ARRAY"), ("EMPTY", "OBJECT"), ("EMPTY",), ("TRUE",), ("FALSE",), ("UNKNOWN",)):
                if self.at(*words) and (_kw(self.t, self.i + len(words), "ON") or words == ("EMPTY",)):
                    beh = [" ".join(words) if words != ("EMPTY",) else "EMPTY ARRAY"]; self.i += len(words); break
            if beh is None and self.at("DEFAULT"):
                self.i += 1
                j, d = self.i, 0
                while j < self.b and not (d == 0 and _kw(self.t, j, "ON") and self.t[j + 1].up() in ("EMPTY", "ERROR")):
                    if self.t[j].s == "(": d += 1
                    elif self.t[j].s == ")": d -= 1
                    j += 1
                self.defaults.append(_text(self.t, self.i, j)); self.i = j
                beh = ["DEFAULT", len(self.defaults) - 1]
            if beh is not None:
                if not (self.at("ON", "EMPTY") or self.at("ON", "ERROR")): self.err("<behavior> ON EMPTY | ON ERROR")
                which = self.t[self.i + 1].up().lower(); self.i += 2
                if which not in allow: self.err(f"ON {which.upper()} is not allowed here")
                self.opts[which] = beh; continue
            self.err(f"unexpected {self.t[self.i].s!r}")


def _json_call(t, i):
    """JSON_VALUE / JSON_QUERY / JSON_EXISTS / JSON_OBJECT / JSON_ARRAY / JSON_OBJECTAGG / JSON_ARRAYAGG /
    JSON_SERIALIZE (...) -> a call of a marker function with the clauses as a JSON string; None when the call is
    already plain"""
    u = t[i].up(); c = _match_paren(t, i + 1); a, b = i + 2, c
    P = _JParse(t, a, b, u)
    if u in ("JSON_VALUE", "JSON_QUERY", "JSON_EXISTS"):
        P.i = a
        ctx = P.expr_until([("FORMAT", "JSON")]); P.format_json()
        if P.done() or t[P.i].s != ",": P.err("JSON context item, path")
        P.i += 1
        P.path_and_passing()
        if u == "JSON_EXISTS": P.opts["error"] = ["FALSE"]
        allow = {"JSON_VALUE": ("returning", "empty", "error"), "JSON_QUERY": ("returning", "wrapper", "quotes", "empty", "error"),
                 "JSON_EXISTS": ("error",)}[u]
        P.clauses(allow)
        args = [ctx, _jstr(P.opts)] + P.passing + P.defaults
        P.opts["np"] = len(P.passing)
        args[1] = _jstr(P.opts)
        return tokenize(f"__{u.lower()}({', '.join(args)})"), c
    if u in ("JSON_OBJECT", "JSON_OBJECTAGG"):
        P.i = a; keys, vals, fmts = [], [], []
        absent, unique = (False, False)
        stops = [("VALUE",), ("NULL", "ON", "NULL"), ("ABSENT", "ON", "NULL"), ("WITH", "UNIQUE"), ("WITHOUT", "UNIQUE"), ("RETURNING",), ("FORMAT", "JSON")]
        while not P.done():
            if P.at("NULL", "ON", "NULL"): P.i += 3; absent = False; continue
            if P.at("ABSENT", "ON", "NULL"): P.i += 3; absent = True; continue
            if P.at("WITH", "UNIQUE") or P.at("WITHOUT", "UNIQUE"):
                unique = P.at("WITH"); P.i += 2
                if P.at("KEYS"): P.i += 1
                continue
            if P.at("RETURNING"): P.i += 1; P.opts["returning"] = P.type_until(); P.format_json(); continue
            if P.at("KEY"): P.i += 1
            j, d = P.i, 0
            while j < b:                                   # the key: up to VALUE or ':' (or ',' in DuckDB's json_object(k, v))
                x = t[j]
                if x.s in ("(", "["): d += 1
                elif x.s in (")", "]"): d -= 1
                elif d == 0 and (x.up() == "VALUE" or x.s in (":", ",")): break
                j += 1
            k = _text(t, P.i, j); P.i = j
            if j >= b: P.err("KEY <name> VALUE <value>")
            P.i += 1
            v = P.expr_until(stops); f = P.format_json()
            keys.append(k); vals.append(v); fmts.append(f)
            if not P.done() and t[P.i].s == ",": P.i += 1
        opts = dict(absent=absent, unique=unique, fmts=fmts, returning=P.opts["returning"])
        if u == "JSON_OBJECTAGG":
            if len(keys) != 1: P.err("one KEY ... VALUE ... pair")
            return tokenize(f"__json_objectagg({keys[0]}, {vals[0]}, {_jstr(opts)})"), c
        return tokenize(f"__json_object({', '.join([_jstr(opts)] + [x for kv in zip(keys, vals) for x in kv])})"), c
    if u in ("JSON_ARRAY", "JSON_ARRAYAGG"):
        P.i = a
        if u == "JSON_ARRAY" and P.i < b and t[P.i].up() in ("SELECT", "WITH", "VALUES", "TABLE"):     # JSON_ARRAY(<query>)
            q_end = P.i
            while q_end < b and not (t[q_end].up() in ("RETURNING", "FORMAT") and _depth0(t, P.i, q_end)): q_end += 1
            qtxt = "(" + _text(t, P.i, q_end) + ")"
            P.i = q_end; P.format_json(); P.clauses(("returning",))
            return tokenize(f"__json_array_q({qtxt}, {_jstr(dict(returning=P.opts['returning']))})"), c
        vals, fmts, null_on_null, order = [], [], False, ""
        stops = [("NULL", "ON", "NULL"), ("ABSENT", "ON", "NULL"), ("RETURNING",), ("ORDER", "BY"), ("FORMAT", "JSON")]
        while not P.done():
            if P.at("NULL", "ON", "NULL"): P.i += 3; null_on_null = True; continue
            if P.at("ABSENT", "ON", "NULL"): P.i += 3; null_on_null = False; continue
            if P.at("RETURNING"): P.i += 1; P.opts["returning"] = P.type_until(); P.format_json(); continue
            if P.at("ORDER", "BY") and u == "JSON_ARRAYAGG":
                j = P.i
                while j < b and not (_depth0(t, P.i, j) and (_kw(t, j, "NULL", "ON") or _kw(t, j, "ABSENT", "ON") or t[j].up() == "RETURNING")): j += 1
                order = " " + _text(t, P.i, j); P.i = j; continue
            vals.append(P.expr_until(stops)); fmts.append(P.format_json())
            if not P.done() and t[P.i].s == ",": P.i += 1
        opts = dict(null_on_null=null_on_null, fmts=fmts, returning=P.opts["returning"])
        if u == "JSON_ARRAYAGG":
            if len(vals) != 1: P.err("one value")
            return tokenize(f"__json_arrayagg({vals[0]}, {_jstr(opts)}{order})"), c
        return tokenize(f"__json_array({', '.join([_jstr(opts)] + vals)})"), c
    if u == "JSON_SERIALIZE":
        P.i = a; x = P.expr_until([("FORMAT", "JSON"), ("RETURNING",)]); P.format_json(); P.clauses(("returning",))
        return tokenize(f"__json_serialize({x}, {_jstr(P.opts)})"), c
    if u in ("JSON", "JSON_SCALAR"):
        P.i = a; x = P.expr_until([("FORMAT", "JSON"), ("WITH", "UNIQUE"), ("WITHOUT", "UNIQUE")]); P.format_json(); unique = False
        if P.at("WITH", "UNIQUE") or P.at("WITHOUT", "UNIQUE"):
            unique = P.at("WITH"); P.i += 2
            if P.at("KEYS"): P.i += 1
        if not P.done(): P.err(f"unexpected {t[P.i].s!r}")
        return tokenize(f"__{'json_scalar' if u == 'JSON_SCALAR' else 'json_parse'}({x}, {'TRUE' if unique else 'FALSE'})"), c
    return None


def _json_table(t, i):
    """JSON_TABLE(ctx, 'path' [AS n] [PASSING ...] COLUMNS (...) [PLAN ...] [ERROR | EMPTY ON ERROR]) -> a table
    function call with the column definitions as a JSON string"""
    c = _match_paren(t, i + 1); a, b = i + 2, c
    P = _JParse(t, a, b, "JSON_TABLE")
    ctx = P.expr_until([("FORMAT", "JSON")]); P.format_json()
    if P.done() or t[P.i].s != ",": P.err("JSON context item, path")
    P.i += 1
    P.path_and_passing()
    defaults = P.defaults

    def columns(P):
        if not P.at("COLUMNS") or t[P.i + 1].s != "(": P.err("COLUMNS (...)")
        e = _match_paren(t, P.i + 1); cols = []
        Q = _JParse(t, P.i + 2, e, "JSON_TABLE"); Q.defaults = defaults
        while not Q.done():
            if Q.at("NESTED"):
                Q.i += 1
                if Q.at("PATH"): Q.i += 1
                x = t[Q.i]
                if x.k != "str": Q.err("NESTED [PATH] '<path>'")
                Q.i += 1
                if Q.at("AS"): Q.i += 2
                cols.append(dict(kind="nested", path=x.s[1:-1].replace("''", "'"), cols=columns(Q)))
            else:
                name = t[Q.i].s.strip('"'); Q.i += 1
                if Q.at("FOR", "ORDINALITY"): Q.i += 2; cols.append(dict(name=name, kind="ordinality"))
                else:
                    typ = Q.type_until(); kind = "value"
                    o = dict(empty=["NULL"], error=["NULL"], wrapper="WITHOUT", quotes="KEEP", names=[])
                    Q.opts = o
                    if Q.at("FORMAT", "JSON"): Q.format_json(); kind = "query"
                    if Q.at("EXISTS"): Q.i += 1; kind = "exists"; o["error"] = ["FALSE"]
                    path = "lax $." + json.dumps(name)
                    if Q.at("PATH"):
                        Q.i += 1; x = t[Q.i]
                        if x.k != "str": Q.err("PATH '<path>'")
                        path = x.s[1:-1].replace("''", "'"); Q.i += 1
                    o["path"] = path
                    Q.clauses({"value": ("empty", "error"), "query": ("wrapper", "quotes", "empty", "error"), "exists": ("error",)}[kind])
                    cols.append(dict(name=name, kind=kind, type=typ, opts=o))
            if not Q.done() and t[Q.i].s == ",": Q.i += 1
        P.i = e + 1
        return cols
    cols = columns(P)
    err = "EMPTY"
    while not P.done():
        if P.at("PLAN"): P.i = _match_paren(t, P.i + 1) + 1 if P.i + 1 < b and t[P.i + 1].s == "(" else P.i + 2; continue
        if P.at("ERROR", "ON", "ERROR"): err = "ERROR"; P.i += 3; continue
        if P.at("EMPTY", "ON", "ERROR"): err = "EMPTY"; P.i += 3; continue
        P.err(f"unexpected {t[P.i].s!r}")
    spec = dict(path=P.opts["path"], names=P.names, cols=cols, error=err, np=len(P.passing))
    return tokenize(f"__json_table({', '.join([ctx, _jstr(spec)] + P.passing + defaults)})"), c


def _mr_expr(t, a, b):
    """an expression of MATCH_RECOGNIZE as text: RUNNING / FINAL <navigation or aggregate>(...) -> __mr_running(...)"""
    out, j = [], a
    while j < b:
        if t[j].up() in ("RUNNING", "FINAL") and j + 2 < b and t[j + 1].k == "id" and t[j + 2].s == "(":
            c = _match_paren(t, j + 2)
            out.append(f"__mr_{t[j].up().lower()}({_mr_expr(t, j + 1, c + 1)})"); j = c + 1; continue
        out.append(t[j].s); j += 1
    return " ".join(out)


def _match_recognize(t, i):
    """<table primary> MATCH_RECOGNIZE (...) -> __match_recognize((SELECT * FROM <table primary>), '<clauses>')"""
    a = _operand_back(t, i - 1)
    src = _text(t, a, i)
    if t[a].s != "(": src = f"(SELECT * FROM {src})"
    c = _match_paren(t, i + 1)
    j, b = i + 2, c
    spec = dict(partition=[], order=[], measures=[], rows="one", empty="show", skip=["past", None], pattern=[], subsets={}, define={})
    heads = ("PARTITION", "ORDER", "MEASURES", "ONE", "ALL", "AFTER", "PATTERN", "SUBSET", "DEFINE", "INITIAL", "SEEK")
    def items(j):
        """comma-separated items up to the next clause"""
        out, s0, d = [], j, 0
        while j < b:
            if t[j].s in ("(", "["): d += 1
            elif t[j].s in (")", "]"): d -= 1
            elif d == 0 and t[j].s == ",": out.append((s0, j)); s0 = j + 1
            elif d == 0 and t[j].up() in heads and j > s0: break
            j += 1
        out.append((s0, j))
        return out, j
    while j < b:
        u = t[j].up()
        if _kw(t, j, "PARTITION", "BY"):
            its, j = items(j + 2); spec["partition"] = [_mr_expr(t, x, y) for x, y in its]
        elif _kw(t, j, "ORDER", "BY"):
            its, j = items(j + 2)
            for x, y in its:
                desc, nf = False, None
                if y - x >= 2 and _kw(t, y - 2, "NULLS"): nf = t[y - 1].up() == "FIRST"; y -= 2
                if t[y - 1].up() in ("ASC", "DESC"): desc = t[y - 1].up() == "DESC"; y -= 1
                spec["order"].append([_mr_expr(t, x, y), desc, nf if nf is not None else False])
        elif u == "MEASURES":
            its, j = items(j + 1)
            for x, y in its:
                k = _top_kw(t, x, y, "AS")
                if k < 0: raise SqlError("MEASURES <expression> AS <name>")
                spec["measures"].append([_mr_expr(t, x, k), t[k + 1].s.strip('"')])
        elif _kw(t, j, "ONE", "ROW", "PER", "MATCH"): spec["rows"] = "one"; j += 4
        elif _kw(t, j, "ALL", "ROWS", "PER", "MATCH"):
            spec["rows"] = "all"; j += 4
            if _kw(t, j, "SHOW", "EMPTY", "MATCHES"): j += 3
            elif _kw(t, j, "OMIT", "EMPTY", "MATCHES"): spec["empty"] = "omit"; j += 3
            elif _kw(t, j, "WITH", "UNMATCHED", "ROWS"): spec["empty"] = "unmatched"; j += 3
        elif _kw(t, j, "AFTER", "MATCH", "SKIP"):
            j += 3
            if _kw(t, j, "PAST", "LAST", "ROW"): spec["skip"] = ["past", None]; j += 3
            elif _kw(t, j, "TO", "NEXT", "ROW"): spec["skip"] = ["next", None]; j += 3
            elif _kw(t, j, "TO", "FIRST"): spec["skip"] = ["first", t[j + 2].s.upper()]; j += 3
            elif _kw(t, j, "TO", "LAST"): spec["skip"] = ["last", t[j + 2].s.upper()]; j += 3
            elif _kw(t, j, "TO"): spec["skip"] = ["last", t[j + 1].s.upper()]; j += 2
            else: raise SqlError("AFTER MATCH SKIP PAST LAST ROW | TO NEXT ROW | TO [FIRST | LAST] <variable>")
        elif u in ("INITIAL", "SEEK"):
            if u == "SEEK": raise SqlError("SEEK is not supported in MATCH_RECOGNIZE (INITIAL is the default)")
            j += 1
        elif u == "PATTERN":
            e = _match_paren(t, j + 1); spec["pattern"] = [x.s for x in t[j + 2:e]]; j = e + 1
        elif u == "SUBSET":
            its, j = items(j + 1)
            for x, y in its:
                o = [k for k in range(x, y) if t[k].s == "("][0]; e = _match_paren(t, o)
                spec["subsets"][t[x].s.upper()] = [t[k].s.upper() for k in range(o + 1, e) if t[k].s != ","]
        elif u == "DEFINE":
            its, j = items(j + 1)
            for x, y in its:
                if not _kw(t, x + 1, "AS"): raise SqlError("DEFINE <variable> AS <condition>")
                spec["define"][t[x].s.upper()] = _mr_expr(t, x + 2, y)
        else: raise SqlError(f"MATCH_RECOGNIZE: unexpected {t[j].s!r}")
    if not spec["pattern"]: raise SqlError("MATCH_RECOGNIZE needs a PATTERN")
    return a, c, tokenize(f"__match_recognize({src}, {_jstr(spec)})")


def _multiset_ops(t):
    """a MULTISET INTERSECT b binds tighter than MULTISET UNION / EXCEPT (ISO 6.48); each becomes __ms_<op>(a, b, mode)"""
    for ops in (("INTERSECT",), ("UNION", "EXCEPT")):
        changed = True
        while changed:
            changed = False
            for i in range(1, len(t) - 1):
                if t[i].up() == "MULTISET" and t[i + 1].up() in ops:
                    j = i + 2; mode = "ALL"
                    if j < len(t) and t[j].up() in ("ALL", "DISTINCT"): mode = t[j].up(); j += 1
                    a = _operand_back(t, i - 1); b = _operand_fwd(t, j)
                    t[a:b] = tokenize(f"__ms_{t[i + 1].up().lower()}({_text(t, a, i)}, {_text(t, j, b)}, '{mode}')")
                    changed = True; break
    return t


def iso_prepass(sql):
    """rewrite ISO syntax that DuckDB's grammar does not accept into calls of marker functions"""
    FLAGS.clear()
    t = tokenize(sql)
    t = _multiset_ops(t)
    changed = True
    guard = 0
    while changed:
        changed = False; guard += 1
        if guard > 10000: raise SqlError("pre-pass did not converge")
        for i in range(len(t)):
            u = t[i].up()
            if u is None: continue
            prev = t[i - 1] if i else None
            # BETWEEN SYMMETRIC a AND b  ->  BETWEEN __iso_sym(a) AND b
            if u == "BETWEEN" and i + 1 < len(t) and t[i + 1].up() in ("SYMMETRIC", "ASYMMETRIC"):
                sym = t[i + 1].up() == "SYMMETRIC"
                if not sym: del t[i + 1]; changed = True; break
                j, d = i + 2, 0
                while j < len(t):
                    if t[j].s == "(": d += 1
                    elif t[j].s == ")": d -= 1
                    elif d == 0 and t[j].up() == "AND": break
                    j += 1
                t[i + 1:j] = [Tok("id", "__iso_sym"), Tok("op", "(")] + t[i + 2:j] + [Tok("op", ")")]
                changed = True; break
            # x [NOT] LIKE_REGEX p [FLAG f]
            if u == "LIKE_REGEX":
                neg = prev is not None and prev.up() == "NOT"
                lend = i - 2 if neg else i - 1
                a = _operand_back(t, lend); b = _operand_fwd(t, i + 1)
                args = [_text(t, a, lend + 1), _text(t, i + 1, b)]
                e = b
                if e < len(t) and t[e].up() == "FLAG": f_end = _operand_fwd(t, e + 1); args.append(_text(t, e + 1, f_end)); e = f_end
                rep = ("NOT " if neg else "") + f"__iso_like_regex({', '.join(args)})"
                t[a:e] = tokenize(rep); changed = True; break
            # ANY(...) / SOME(...) as aggregates (not after a comparison operator)
            if u in ("ANY", "SOME") and i + 1 < len(t) and t[i + 1].s == "(" and not (prev is not None and prev.s in ("=", "<>", "!=", "<", ">", "<=", ">=")):
                inner = t[i + 2] if i + 2 < len(t) else None
                if inner is None or inner.up() not in ("SELECT", "WITH", "VALUES", "TABLE"):
                    t[i] = Tok("id", "bool_or"); changed = True; break
            # hypothetical set functions and LISTAGG ... WITHIN GROUP (ORDER BY ...)
            if u in ("RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "LISTAGG") and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                if _kw(t, c + 1, "WITHIN", "GROUP") and c + 3 < len(t) and t[c + 3].s == "(":
                    c2 = _match_paren(t, c + 3)
                    order = t[c + 4:c2]                      # ORDER BY ...
                    args = t[i + 2:c]
                    if u == "LISTAGG":
                        o = _top_kw(t, i + 2, c, "ON")
                        extra = []
                        if o >= 0:
                            ov = [x.s for x in t[o:c]]
                            mode = ov[2].upper() if len(ov) > 2 else "ERROR"
                            filler = next((x for x in ov[3:] if x.startswith("'")), "'...'")
                            wc = "'WITH'" if any(x.upper() == "WITH" for x in ov) else "'WITHOUT'"
                            extra = [Tok("op", ","), Tok("str", f"'{mode}'"), Tok("op", ","), Tok("str", filler), Tok("op", ","), Tok("str", wc)]
                            args = t[i + 2:o]
                        if len(_split_commas(args, 0, len(args))) == 1: args = args + [Tok("op", ","), Tok("str", "''")]
                        name = "__iso_listagg"
                        t[i:c2 + 1] = [Tok("id", name), Tok("op", "(")] + args + extra + order + [Tok("op", ")")]
                    else:
                        t[i:c2 + 1] = [Tok("id", f"__iso_h{u.lower()}"), Tok("op", "(")] + args + order + [Tok("op", ")")]
                    changed = True; break
            # SUBSTRING(s SIMILAR p ESCAPE e)
            if u == "SUBSTRING" and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                s = _top_kw(t, i + 2, c, "SIMILAR")
                if s >= 0:
                    e = _top_kw(t, s, c, "ESCAPE")
                    args = [_text(t, i + 2, s), _text(t, s + 1, e if e >= 0 else c), _text(t, e + 1, c) if e >= 0 else "NULL"]
                    t[i:c + 1] = tokenize(f"__iso_substring_similar({', '.join(args)})"); changed = True; break
            # the *_REGEX functions
            if u in ("OCCURRENCES_REGEX", "POSITION_REGEX", "SUBSTRING_REGEX", "TRANSLATE_REGEX") and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                t[i:c + 1] = tokenize(_rx_args(t, i + 2, c, u)); changed = True; break
            # UNIQUE (subquery)
            if u == "UNIQUE" and i + 2 < len(t) and t[i + 1].s == "(" and t[i + 2].up() in ("SELECT", "WITH", "VALUES", "TABLE"):
                c = _match_paren(t, i + 1)
                t[i:c + 1] = [Tok("id", "__iso_unique"), Tok("op", "(")] + t[i + 1:c + 1] + [Tok("op", ")")]
                changed = True; break
            # (a, b) OVERLAPS (c, d)
            if u == "OVERLAPS" and prev is not None and prev.s == ")" and i + 1 < len(t) and t[i + 1].s == "(":
                a = _match_open(t, i - 1); c = _match_paren(t, i + 1)
                left = [_text(t, x, y) for x, y in _split_commas(t, a + 1, i - 1)]
                right = [_text(t, x, y) for x, y in _split_commas(t, i + 2, c)]
                t[a:c + 1] = tokenize(f"__iso_overlaps({', '.join(left + right)})"); changed = True; break
            # FETCH FIRST n PERCENT ROWS ONLY / FETCH FIRST n ROWS WITH TIES
            if u == "FETCH" and i + 1 < len(t) and t[i + 1].up() in ("FIRST", "NEXT"):
                j = i + 2; n = []
                while j < len(t) and t[j].up() not in ("ROW", "ROWS", "PERCENT"): n.append(t[j]); j += 1
                pct = j < len(t) and t[j].up() == "PERCENT"
                if pct: j += 1
                if j < len(t) and t[j].up() in ("ROW", "ROWS"): j += 1
                ties = _kw(t, j, "WITH", "TIES")
                if pct or ties:
                    end = j + 2 if (ties or _kw(t, j, "ONLY") is False) else j
                    if not ties and _kw(t, j, "ONLY"): end = j + 1
                    nt = n or [Tok("num", "1")]
                    if ties: rep = [Tok("id", "LIMIT"), Tok("id", "__iso_ties"), Tok("op", "(")] + nt + ([Tok("op", ","), Tok("num", "1")] if pct else []) + [Tok("op", ")")]
                    else: rep = [Tok("id", "LIMIT"), Tok("id", "__iso_pct"), Tok("op", "(")] + nt + [Tok("op", ")")]
                    t[i:end] = rep; changed = True; break
            # NTH_VALUE(x, n) FROM FIRST | FROM LAST; f(x) RESPECT NULLS | IGNORE NULLS OVER
            if u in ("NTH_VALUE", "FIRST_VALUE", "LAST_VALUE", "LAG", "LEAD") and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                if _kw(t, c + 1, "FROM", "FIRST"): del t[c + 1:c + 3]; changed = True; break
                if _kw(t, c + 1, "FROM", "LAST"):
                    del t[c + 1:c + 3]; t[i] = Tok("id", "__iso_nth_value_last"); changed = True; break
                if _kw(t, c + 1, "RESPECT", "NULLS"): del t[c + 1:c + 3]; changed = True; break
                if _kw(t, c + 1, "IGNORE", "NULLS"):
                    del t[c + 1:c + 3]
                    args = _split_commas(t, i + 2, c)
                    a0 = args[0][1]
                    t[a0:a0] = [Tok("id", "IGNORE"), Tok("id", "NULLS")]
                    changed = True; break
            # INTERVAL 'x' <unit> TO <unit>
            if u == "INTERVAL" and i + 4 < len(t) and t[i + 1].k == "str" and t[i + 3].up() == "TO":
                q = f"{t[i + 2].up()} TO {t[i + 4].up()}"
                t[i:i + 5] = tokenize(f"__iso_interval({t[i + 1].s}, '{q}')"); changed = True; break
            # x [NOT] SIMILAR TO p [ESCAPE e]  ->  x [NOT] SIMILAR TO __iso_similar_pat(p [, e])
            if u == "SIMILAR" and i + 2 < len(t) and t[i + 1].up() == "TO" and t[i + 2].up() != "__ISO_SIMILAR_PAT":
                b = _operand_fwd(t, i + 2); args = [_text(t, i + 2, b)]; e = b
                if e < len(t) and t[e].up() == "ESCAPE":
                    f_end = _operand_fwd(t, e + 1); args.append(_text(t, e + 1, f_end)); e = f_end
                t[i + 2:e] = tokenize(f"__iso_similar_pat({', '.join(args)})"); changed = True; break
            # TIMESTAMP / TIME [WITH | WITHOUT TIME ZONE] '<literal>': WITH TIME ZONE when said, or when the literal
            # has a time zone (ISO 5.3)
            if u in ("TIMESTAMP", "TIME") and not (prev is not None and prev.up() in ("AS", "WITHOUT", "WITH")):
                j = i + 1; zoned = None
                if _kw(t, j, "WITH", "TIME", "ZONE"): zoned = True; j += 3
                elif _kw(t, j, "WITHOUT", "TIME", "ZONE"): zoned = False; j += 3
                if j < len(t) and t[j].k == "str" and t[j].s[0] == "'":
                    lit = t[j].s[1:-1].replace("''", "'")
                    if zoned is None: zoned = T.has_zone(lit)
                    name = (u + "TZ") if zoned else u
                    if name != u or j != i + 1:
                        t[i:j] = [Tok("id", "TIMESTAMPTZ" if name == "TIMESTAMPTZ" else name)]; changed = True; break
            # <datetime primary> AT LOCAL  ->  __iso_at_local(...)
            if u == "AT" and i + 1 < len(t) and t[i + 1].up() == "LOCAL" and i > 0:
                a = _operand_back(t, i - 1)
                t[a:i + 2] = [Tok("id", "__iso_at_local"), Tok("op", "(")] + t[a:i] + [Tok("op", ")")]; changed = True; break
            # TABLESAMPLE BERNOULLI | SYSTEM (p) [REPEATABLE (s)]: p is a percentage (ISO 7.6)
            if u == "TABLESAMPLE" and i + 2 < len(t) and t[i + 1].up() in ("BERNOULLI", "SYSTEM") and t[i + 2].s == "(":
                c = _match_paren(t, i + 2)
                if not any(x.s == "%" or x.up() in ("PERCENT", "ROWS") for x in t[i + 3:c]):
                    t[c:c] = [Tok("id", "PERCENT")]; changed = True; break
            # multi-word type names DuckDB's grammar lacks: BINARY VARYING, BINARY / CHARACTER LARGE OBJECT, ...
            if u in ("BINARY", "CHARACTER", "CHAR", "NCHAR", "NATIONAL"):
                for words, name in ((("BINARY", "VARYING"), "VARBINARY"), (("BINARY", "LARGE", "OBJECT"), "BLOB"),
                                    (("CHARACTER", "LARGE", "OBJECT"), "CLOB"), (("CHAR", "LARGE", "OBJECT"), "CLOB"),
                                    (("NATIONAL", "CHARACTER", "LARGE", "OBJECT"), "NCLOB"), (("NATIONAL", "CHAR", "LARGE", "OBJECT"), "NCLOB"),
                                    (("NCHAR", "LARGE", "OBJECT"), "NCLOB"), (("NCHAR", "VARYING"), "VARCHAR"),
                                    (("NATIONAL", "CHAR", "VARYING"), "VARCHAR"), (("NATIONAL", "CHAR"), "VARCHAR")):
                    if _kw(t, i, *words):
                        t[i:i + len(words)] = [Tok("id", name)]; changed = True; break
                if changed: break
            # <table primary> PARTITION BY (<columns>) <outer join> (ISO 7.10, F403): kept for the binder by alias
            if u == "PARTITION" and _kw(t, i + 1, "BY") and i + 2 < len(t) and t[i + 2].s == "(" and i > 0 and t[i - 1].k in ("id", "qid"):
                c = _match_paren(t, i + 2)
                nxt = t[c + 1].up() if c + 1 < len(t) else None
                if nxt in ("LEFT", "RIGHT", "FULL", "JOIN", "INNER", "ON", "USING", "NATURAL", "CROSS"):
                    FLAGS.setdefault("pjoin", {})[t[i - 1].s.strip('"').lower()] = [_text(t, a, b) for a, b in _split_commas(t, i + 3, c)]
                    del t[i:c + 1]; changed = True; break
            # <table primary> MATCH_RECOGNIZE (...)
            if u == "MATCH_RECOGNIZE" and i + 1 < len(t) and t[i + 1].s == "(":
                a, c, rep = _match_recognize(t, i)
                t[a:c + 1] = rep; changed = True; break
            # SQL/JSON functions
            if u in ("JSON_VALUE", "JSON_QUERY", "JSON_EXISTS", "JSON_OBJECT", "JSON_ARRAY", "JSON_OBJECTAGG", "JSON_ARRAYAGG",
                     "JSON_SERIALIZE", "JSON_SCALAR", "JSON_TABLE") and i + 1 < len(t) and t[i + 1].s == "(" and not t[i].done:
                r = _json_table(t, i) if u == "JSON_TABLE" else _json_call(t, i)
                if r is None: t[i].done = True
                else:
                    t[i:r[1] + 1] = r[0]; changed = True; break
            if u == "JSON" and i + 1 < len(t) and t[i + 1].s == "(" and not (prev is not None and prev.s in (".", "AS")) and not t[i].done:
                r = _json_call(t, i)
                t[i:r[1] + 1] = r[0]; changed = True; break
            # x IS [NOT] JSON [VALUE | ARRAY | OBJECT | SCALAR] [WITH | WITHOUT UNIQUE [KEYS]]
            if u == "IS" and (_kw(t, i + 1, "JSON") or _kw(t, i + 1, "NOT", "JSON")):
                neg = t[i + 1].up() == "NOT"; j = i + (3 if neg else 2); kind = "VALUE"; uniq = False
                if j < len(t) and t[j].up() in ("VALUE", "ARRAY", "OBJECT", "SCALAR"): kind = t[j].up(); j += 1
                if _kw(t, j, "WITH", "UNIQUE") or _kw(t, j, "WITHOUT", "UNIQUE"):
                    uniq = t[j].up() == "WITH"; j += 2
                    if _kw(t, j, "KEYS"): j += 1
                t[i:j] = _marker(neg, "json", [f"'{kind}'", "TRUE" if uniq else "FALSE"]); changed = True; break
            # MULTISET [...] / MULTISET (<query>) / ARRAY (<query>)
            if u == "MULTISET" and i + 1 < len(t) and t[i + 1].s == "[":
                d = 0
                for c in range(i + 1, len(t)):
                    if t[c].s == "[": d += 1
                    elif t[c].s == "]":
                        d -= 1
                        if d == 0: break
                t[i:c + 1] = [Tok("id", "__ms_from_list"), Tok("op", "(")] + t[i + 1:c + 1] + [Tok("op", ")")]; changed = True; break
            if u in ("MULTISET", "ARRAY") and i + 2 < len(t) and t[i + 1].s == "(" and t[i + 2].up() in ("SELECT", "WITH", "VALUES", "TABLE"):
                c = _match_paren(t, i + 1)
                t[i:c + 1] = [Tok("id", "__iso_ms_q" if u == "MULTISET" else "__iso_array_q"), Tok("op", "(")] + t[i + 1:c + 1] + [Tok("op", ")")]
                changed = True; break
            # x [NOT] MEMBER [OF] m, a [NOT] SUBMULTISET [OF] b, m IS [NOT] A SET (ISO 8.16 - 8.18)
            if u in ("MEMBER", "SUBMULTISET") and i > 0:
                neg = prev.up() == "NOT"
                j = i + 1 + (1 if _kw(t, i + 1, "OF") else 0)
                b = _operand_fwd(t, j)
                start = i - 1 if neg else i
                t[start:b] = _marker(neg, u.lower(), [_text(t, j, b)]); changed = True; break
            if u == "IS" and (_kw(t, i + 1, "A", "SET") or _kw(t, i + 1, "NOT", "A", "SET")):
                neg = t[i + 1].up() == "NOT"
                t[i:i + (4 if neg else 3)] = _marker(neg, "aset", []); changed = True; break
            # CAST(x AS <type> MULTISET)
            if u == "CAST" and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                a = _top_kw(t, i + 2, c, "AS")
                if a >= 0 and t[c - 1].up() == "MULTISET":
                    t[i:c + 1] = tokenize(f"__ms_from_list(CAST({_text(t, i + 2, a)} AS {_text(t, a + 1, c - 1)}[]))"); changed = True; break
            # x IS [NOT] [NFC | NFD | NFKC | NFKD] NORMALIZED
            if u == "IS":
                j = i + 1; neg = False
                if j < len(t) and t[j].up() == "NOT": neg = True; j += 1
                form = "NFC"
                if j < len(t) and t[j].up() in ("NFC", "NFD", "NFKC", "NFKD"): form = t[j].up(); j += 1
                if j < len(t) and t[j].up() == "NORMALIZED":
                    t[i:j + 1] = _marker(neg, "normalized", [f"'{form}'"]); changed = True; break
            # <row> MATCH [UNIQUE] [SIMPLE | PARTIAL | FULL] (<subquery>)  (ISO 8.13)
            if u == "MATCH" and i > 0:
                j = i + 1; uniq = False; kind = "SIMPLE"
                if j < len(t) and t[j].up() == "UNIQUE": uniq = True; j += 1
                if j < len(t) and t[j].up() in ("SIMPLE", "PARTIAL", "FULL"): kind = t[j].up(); j += 1
                if j + 1 < len(t) and t[j].s == "(" and t[j + 1].up() in ("SELECT", "WITH", "VALUES", "TABLE"):
                    c = _match_paren(t, j)
                    t[i:c + 1] = _marker(False, "match", [f"'{kind}'", "TRUE" if uniq else "FALSE"]) [:-1] + [Tok("op", ",")] + t[j:c + 1] + [Tok("op", ")")]
                    changed = True; break
            # PERIOD (s1, e1) <OVERLAPS | EQUALS | CONTAINS | [IMMEDIATELY] PRECEDES | SUCCEEDS> PERIOD (s2, e2) | x
            # (ISO 8.20, periods closed-open [s, e)): comparisons of the bounds, so plane predicates where they apply
            if u == "PERIOD" and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                lp = [_text(t, a, b) for a, b in _split_commas(t, i + 2, c)]
                j = c + 1; imm = False
                if j < len(t) and t[j].up() == "IMMEDIATELY": imm = True; j += 1
                op = t[j].up() if j < len(t) else None
                if len(lp) == 2 and op in ("OVERLAPS", "EQUALS", "CONTAINS", "PRECEDES", "SUCCEEDS"):
                    s1, e1 = f"({lp[0]})", f"({lp[1]})"
                    if _kw(t, j + 1, "PERIOD") and j + 2 < len(t) and t[j + 2].s == "(":
                        c2 = _match_paren(t, j + 2)
                        rp = [_text(t, a, b) for a, b in _split_commas(t, j + 3, c2)]
                        if len(rp) != 2: raise SqlError("PERIOD (start, end) takes two values")
                        s2, e2 = f"({rp[0]})", f"({rp[1]})"; end = c2 + 1
                        cond = {"OVERLAPS": f"{s1} < {e2} AND {s2} < {e1}", "EQUALS": f"{s1} = {s2} AND {e1} = {e2}",
                                "CONTAINS": f"{s1} <= {s2} AND {e1} >= {e2}",
                                "PRECEDES": f"{e1} = {s2}" if imm else f"{e1} <= {s2}",
                                "SUCCEEDS": f"{s1} = {e2}" if imm else f"{s1} >= {e2}"}[op]
                    elif op == "CONTAINS":
                        end = _operand_fwd(t, j + 1); x = f"({_text(t, j + 1, end)})"
                        cond = f"{s1} <= {x} AND {e1} > {x}"
                    else: raise SqlError(f"PERIOD ... {op} needs a PERIOD on its right")
                    if imm and op not in ("PRECEDES", "SUCCEEDS"): raise SqlError("IMMEDIATELY goes with PRECEDES or SUCCEEDS")
                    t[i:end] = tokenize(f"({cond})"); changed = True; break
            # simple CASE with extended WHEN operands (lists, comparisons, predicates)
            if u == "CASE" and i + 1 < len(t) and t[i + 1].up() != "WHEN" and not getattr(t[i], "done", False):
                x = _extended_case(t, i)
                if x is None: t[i].done = True
                else:
                    out, e = x; out[0].done = True
                    t[i:e + 1] = out; changed = True; break
            # GROUP BY DISTINCT | ALL <grouping elements> (ISO T434): duplicate grouping sets removed / kept
            if u == "GROUP" and _kw(t, i + 1, "BY") and i + 2 < len(t) and t[i + 2].up() in ("DISTINCT", "ALL"):
                nxt = t[i + 3] if i + 3 < len(t) else None
                if t[i + 2].up() == "DISTINCT":
                    t[i + 2:i + 3] = [Tok("id", "__iso_gb_distinct"), Tok("op", "("), Tok("op", ")"), Tok("op", ",")]; changed = True; break
                if nxt is not None and nxt.s not in (";", ")") and nxt.up() not in ("HAVING", "ORDER", "LIMIT", "OFFSET", "QUALIFY",
                                                                                  "WINDOW", "UNION", "EXCEPT", "INTERSECT", "FETCH"):
                    del t[i + 2]; changed = True; break               # ISO ALL: the default (DuckDB's GROUP BY ALL has no list)
            # CAST(x AS <datetime or character type> FORMAT '<template>') (ISO T839)
            if u == "CAST" and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                a = _top_kw(t, i + 2, c, "AS"); f = _top_kw(t, i + 2, c, "FORMAT")
                if a >= 0 and f > a and f + 2 == c and t[f + 1].k == "str":
                    t[i:c + 1] = tokenize(f"__iso_cast_format({_text(t, i + 2, a)}, '{_text(t, a + 1, f)}', {t[f + 1].s})"); changed = True; break
            # CAST(x AS INTERVAL <qualifier>)
            if u == "CAST" and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                a = _top_kw(t, i + 2, c, "AS")
                if a >= 0 and a + 1 < c and t[a + 1].up() == "INTERVAL":
                    q = _qualifier(t, a + 2)
                    if q is not None and q[1] == c:
                        t[i:c + 1] = tokenize(f"__iso_interval_cast({_text(t, i + 2, a)}, '{q[0]}')"); changed = True; break
            # INTERVAL [+|-] 'text' <unit> [(p)] [TO <unit> [(p)]]
            if u == "INTERVAL" and i + 1 < len(t):
                j = i + 1; sign = ""
                if t[j].s in ("+", "-") and j + 1 < len(t) and t[j + 1].k == "str": sign = t[j].s; j += 1
                if t[j].k == "str":
                    q = _qualifier(t, j + 1)
                    if q is not None and (sign or "(" in [x.s for x in t[j + 1:q[1]]]):
                        body = t[j].s[1:-1]
                        if sign == "-": body = body[1:] if body.startswith("-") else "-" + body
                        t[i:q[1]] = tokenize(f"__iso_interval('{body}', '{q[0]}')"); changed = True; break
            # (<datetime> - <datetime>) <qualifier>: the difference as an interval of those fields (ISO 6.42)
            if u in _UNITS and prev is not None and prev.s == ")" and not (i >= 2 and t[i - 2].up() == "AS"):
                a = _match_open(t, i - 1)
                if not (a > 0 and (t[a - 1].k in ("id", "qid") and t[a - 1].up() not in _KW_BREAK)):
                    minus = [k for k in range(a + 1, i - 1) if t[k].s == "-" and _depth0(t, a + 1, k)
                             and t[k - 1].s not in _CMP_OPS + ("(", ",", "+", "-", "*", "/") and t[k - 1].up() not in _KW_BREAK]
                    q = _qualifier(t, i)
                    if minus and q is not None:
                        m = minus[-1]
                        t[a:q[1]] = tokenize(f"__iso_dt_diff(({_text(t, a + 1, m)}), ({_text(t, m + 1, i - 1)}), '{q[0]}')")
                        changed = True; break
            # CHAR_LENGTH / POSITION / SUBSTRING / OVERLAY (... USING CHARACTERS | OCTETS)
            if u in ("CHAR_LENGTH", "CHARACTER_LENGTH", "POSITION", "SUBSTRING", "OVERLAY") and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                k = _top_kw(t, i + 2, c, "USING")
                if k >= 0 and k + 1 < c and t[k + 1].up() in ("CHARACTERS", "OCTETS"):
                    octets = t[k + 1].up() == "OCTETS"
                    del t[k:k + 2]; c -= 2
                    if octets:
                        if u in ("CHAR_LENGTH", "CHARACTER_LENGTH"): t[i] = Tok("id", "octet_length")
                        else:
                            inner = t[i + 2:c]
                            if u == "POSITION":
                                m = _top_kw(t, i + 2, c, "IN")
                                args = [_text(t, i + 2, m), _text(t, m + 1, c)]
                            else:
                                kws = [j for j in range(i + 2, c) if t[j].up() in ("FROM", "FOR", "PLACING") and _depth0(t, i + 2, j)]
                                cuts = [i + 2] + kws + [c]
                                args = [_text(t, cuts[x] + (0 if x == 0 else 1), cuts[x + 1]) for x in range(len(cuts) - 1)]
                            t[i:c + 1] = tokenize(f"__iso_{u.lower()}_octets({', '.join(args)})")
                    changed = True; break
            # TRANSLATE (x USING <transliteration>)
            if u in ("TRANSLATE", "CONVERT") and i + 1 < len(t) and t[i + 1].s == "(":
                c = _match_paren(t, i + 1)
                k = _top_kw(t, i + 2, c, "USING")
                if k >= 0 and k + 2 == c:
                    t[i:c + 1] = tokenize(f"__iso_transliterate({_text(t, i + 2, k)}, '{t[k + 1].s}')"); changed = True; break
            # <with list element> ... AS (<query>) SEARCH ... / CYCLE ... (ISO 7.17, T133): removed and kept for the binder
            if u in ("SEARCH", "CYCLE") and prev is not None and prev.s == ")":
                o = _match_open(t, i - 1)
                if o > 0 and t[o - 1].up() == "AS":
                    k = o - 2
                    if t[k].s == ")": k = _match_open(t, k) - 1
                    name = t[k].s.strip('"').lower()
                    spec = FLAGS.setdefault("search_cycle", {}).setdefault(name, {})
                    j = i + 1
                    def cols_until(j, stop):
                        out = []
                        while j < len(t) and t[j].up() != stop:
                            if t[j].s != ",": out.append(t[j].s.strip('"'))
                            j += 1
                        return out, j
                    if u == "SEARCH":
                        if not (j + 2 < len(t) and t[j].up() in ("DEPTH", "BREADTH") and _kw(t, j + 1, "FIRST", "BY")):
                            raise SqlError("SEARCH DEPTH | BREADTH FIRST BY <columns> SET <column>")
                        kind = t[j].up(); cs, j = cols_until(j + 3, "SET")
                        spec["search"] = dict(kind=kind, cols=cs, set=t[j + 1].s.strip('"')); j += 2
                    else:
                        cs, j = cols_until(j, "SET")
                        mark = t[j + 1].s.strip('"'); j += 2; v1, v0 = "TRUE", "FALSE"
                        if _kw(t, j, "TO"):
                            d = _top_kw(t, j + 1, len(t), "DEFAULT")
                            v1 = _text(t, j + 1, d); u2 = _top_kw(t, d + 1, len(t), "USING")
                            v0 = _text(t, d + 1, u2); j = u2
                        if not _kw(t, j, "USING"): raise SqlError("CYCLE <columns> SET <mark> [TO <value> DEFAULT <value>] USING <path>")
                        spec["cycle"] = dict(cols=cs, mark=mark, v1=v1, v0=v0, path=t[j + 1].s.strip('"')); j += 2
                    del t[i:j]; changed = True; break
            # UNION / EXCEPT / INTERSECT [ALL] CORRESPONDING -> BY NAME (the binder keeps the common columns only)
            if u == "CORRESPONDING":
                by = None
                if _kw(t, i + 1, "BY") and i + 2 < len(t) and t[i + 2].s == "(":
                    c = _match_paren(t, i + 2); by = [t[k].s.strip('"') for k in range(i + 3, c) if t[k].s != ","]
                    del t[i + 1:c + 1]
                t[i:i + 1] = [Tok("id", "BY"), Tok("id", "NAME")]; FLAGS["corresponding"] = True
                FLAGS.setdefault("corresponding_by", []).append(by); changed = True; break
    return " ".join(x.s for x in t)


# ================================================================== parsing
_parser = None


def parse(sql):
    """SQL text -> DuckDB's syntax tree of each statement (dicts)"""
    global _parser
    import duckdb
    if _parser is None: _parser = duckdb.connect()
    text = iso_prepass(sql)
    r = json.loads(_parser.execute("SELECT json_serialize_sql(?)", [text]).fetchone()[0])
    if r.get("error"):
        raise SqlError(r.get("error_message", "syntax error") + ("" if text == sql else f"\n(after the ISO pre-pass: {text})"))
    return [s["node"] for s in r["statements"]], text


# ================================================================== binding
_CMP = {"COMPARE_EQUAL": "=", "COMPARE_NOTEQUAL": "<>", "COMPARE_LESSTHAN": "<", "COMPARE_GREATERTHAN": ">",
        "COMPARE_LESSTHANOREQUALTO": "<=", "COMPARE_GREATERTHANOREQUALTO": ">=", "COMPARE_DISTINCT_FROM": "is_distinct",
        "COMPARE_NOT_DISTINCT_FROM": "is_not_distinct"}
_FN_ALIAS = {"**": "power", "^": "power", "~~": "like", "!~~": "not_like", "~~*": "ilike", "!~~*": "not_ilike", "~~~": "glob",
             "not_like_escape": "not_like_escape", "mod": "%", "fmod": "%", "divide": "//", "list_value": "list_value",
             "array_value": "list_value", "ceiling": "ceil", "character_length": "char_length", "strpos": "position",
             "instr": "position", "nullif": "nullif", "len": "length", "ucase": "upper", "lcase": "lower", "substr": "substring",
             "datepart": "date_part", "datediff": "date_diff", "datesub": "date_sub", "datetrunc": "date_trunc",
             "get_current_timestamp": "current_timestamp", "now": "current_timestamp", "transaction_timestamp": "current_timestamp",
             "today": "current_date", "pow": "power",
             "truncate": "trunc"}
_AGG_ALIAS = {"mean": "avg", "variance": "var_samp", "stddev": "stddev_samp", "string_agg": "string_agg", "group_concat": "string_agg",
              "listagg": "string_agg", "list": "array_agg", "arbitrary": "any_value", "bool_and": "every", "any": "bool_or",
              "some": "bool_or", "min_by": "arg_min", "max_by": "arg_max", "argmin": "arg_min", "argmax": "arg_max",
              "percentile_cont": "quantile_cont", "percentile_disc": "quantile_disc", "quantile": "quantile_disc",
              "countif": "count_if", "fsum": "sum", "sumkahan": "sum", "kahan_sum": "sum", "first": "any_value"}
_WIN_KIND = {"WINDOW_ROW_NUMBER": "row_number", "WINDOW_RANK": "rank", "WINDOW_RANK_DENSE": "dense_rank",
             "WINDOW_PERCENT_RANK": "percent_rank", "WINDOW_CUME_DIST": "cume_dist", "WINDOW_NTILE": "ntile", "WINDOW_LAG": "lag",
             "WINDOW_LEAD": "lead", "WINDOW_FIRST_VALUE": "first_value", "WINDOW_LAST_VALUE": "last_value", "WINDOW_NTH_VALUE": "nth_value"}


_NILADIC = {"CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP", "LOCALTIME", "LOCALTIMESTAMP", "CURRENT_USER", "USER",
            "SESSION_USER", "SYSTEM_USER", "CURRENT_ROLE", "CURRENT_CATALOG", "CURRENT_SCHEMA", "CURRENT_PATH"}


def lit_of(v, strlit=False):
    """a constant node for a Python value"""
    if v is None: return I.Lit(None, T.NULL)
    return I.Lit(v, T.py_type(v) if not strlit else T.VARCHAR, strlit)


def cast_to(e, t, try_=False):
    """e converted to type t (constants are converted now; NULL stays NULL; a constant that does not convert fails
    the query only where it is evaluated, as a runtime error)"""
    if t is None or e.t == t: return e
    if isinstance(e, I.Lit):
        if e.v is None or isinstance(e.v, F.Err): return I.Lit(e.v, t)
        try: return I.Lit(T.cast_value(e.v, e.t, t, not try_), t)
        except SqlError as ex: return I.Lit(F.Err(str(ex)), t)
    return I.Cast(e, t, try_)


_COLLATED = {"=", "<>", "<", "<=", ">", ">=", "is_distinct", "is_not_distinct", "between", "between_sym", "in", "like",
             "position", "starts_with", "ends_with", "contains", "replace", "split_part", "strpos"}


def _collated_args(name, args):
    """comparisons of strings under an explicit collation (ISO 9.15): every string argument compared by its key"""
    colls = {a.t.coll for a in args if getattr(a.t, "coll", None)}
    if not colls: return args
    if len(colls) > 1: raise SqlError(f"collation conflict: {' and '.join(sorted(colls))}")
    c = colls.pop()
    if T.collation(c) is None: return [I.Lit(a.v, T.VARCHAR, getattr(a, "strlit", False)) if isinstance(a, I.Lit) else a for a in args]
    out = []
    for a in args:
        if a.t.id in ("VARCHAR", "NULL") or (isinstance(a, I.Lit) and a.strlit):
            a = make_func("__ckey", [a, I.Lit(c, T.VARCHAR)])
        out.append(a)
    return out


def make_func(name, args, extra=None):
    """a typed scalar call: the registry's typer gives the result type and the argument conversions"""
    s = F.SCALARS.get(name)
    if s is None: raise SqlError(f"Scalar Function with name {name} does not exist")
    if name in _COLLATED: args = _collated_args(name, args)
    if name in F.BINARY and any(a.t.id == "BLOB" for a in args):         # binary strings: their own functions
        name = F.BINARY[name]; s = F.SCALARS[name]
    if name in ("||", "concat") and any(a.t.id == "LIST" for a in args):    # array concatenation
        name = "__list_concat"; s = F.SCALARS[name]
    if name in ("=", "<>", "is_not_distinct", "is_distinct") and any(a.t.id == "MULTISET" for a in args):
        eq = make_func("__ms_eq", args)                                        # multiset (in)equality
        if name in ("is_not_distinct", "is_distinct"):
            nn = make_and([make_func("is_not_null", [a]) for a in args])
            both = make_and([make_func("is_null", [a]) for a in args])
            eq = make_func("or", [both, make_and([nn, make_func("coalesce", [eq, I.Lit(False, T.BOOLEAN)])])])
        return eq if name in ("=", "is_not_distinct") else make_func("not", [eq])
    if name in ("<", "<=", ">", ">=") and any(a.t.id == "MULTISET" for a in args):
        raise SqlError("multisets compare only for equality")
    rt, casts = s.typer(args)
    if casts:
        args = [cast_to(a, c) if c is not None else a for a, c in zip(args, list(casts) + [None] * (len(args) - len(casts)))]
    f = I.Func(name, args, rt, extra)
    if s.deterministic and all(isinstance(a, I.Lit) for a in args) and name not in ("typeof",):
        import sqlexec
        try: return I.Lit(sqlexec.eval_const(f), rt)            # constant folding
        except SqlError: return f                               # a runtime error: raised only if evaluated
    return f


def make_and(parts):
    parts = [p for p in parts if p is not None]
    if not parts: return None
    if len(parts) == 1: return parts[0]
    flat = []
    for p in parts:
        if isinstance(p, I.Func) and p.fn == "and": flat += p.args
        else: flat.append(p)
    return I.Func("and", flat, T.BOOLEAN)


def conjuncts(e):
    if e is None: return []
    if isinstance(e, I.Func) and e.fn == "and": return [x for a in e.args for x in conjuncts(a)]
    return [e]


class Scope:
    """the FROM items visible at one query level; parent: the enclosing query's scope (correlation)"""
    def __init__(self, bindings, parent=None, using=None):
        self.bindings, self.parent, self.using = bindings, parent, using or {}
        self.correlated = False

    def lookup(self, names):
        """-> Col of this scope, or None"""
        names = [n for n in names]
        if len(names) >= 2:
            tname, cname = names[-2], names[-1]
            hits = [b for b in self.bindings if _ieq(b.alias, tname)]
            if not hits: return None
            if len(hits) > 1: raise SqlError(f"table name {tname!r} is ambiguous")
            c = _find_col(hits[0], cname)
            if c is None: raise SqlError(f"Table {tname!r} does not have a column named {cname!r}")
            return c
        cname = names[0]
        u = self.using.get(cname.lower())
        if u is not None: return u
        hits = []
        for b in self.bindings:
            c = _find_col(b, cname)
            if c is not None: hits.append(c)
        if len(hits) > 1: raise SqlError(f"column reference {cname!r} is ambiguous")
        return hits[0] if hits else None


def _ieq(a, b): return a == b or a.lower() == b.lower()


def _shown(b, c):
    """the name a FROM item shows for its column c (a table's columns renamed by AS x(a, b, ...))"""
    d = getattr(b, "display", None)
    return d.get(c, c) if d else c


def _find_col(b, name):
    if getattr(b, "display", None):
        for c, t, nl in b.cols:
            if _shown(b, c) == name: return I.Col(b, c, t, nl)
        for c, t, nl in b.cols:
            if _shown(b, c).lower() == name.lower(): return I.Col(b, c, t, nl)
        return None
    for c, t, nl in b.cols:
        if c == name: return I.Col(b, c, t, nl)
    for c, t, nl in b.cols:
        if c.lower() == name.lower(): return I.Col(b, c, t, nl)
    return None


class _MRScope(Scope):
    """the names of MATCH_RECOGNIZE: <variable>.<column> (a pattern variable or a SUBSET name), and unqualified
    columns (the universal row pattern variable)"""
    def __init__(self, univ, varb, parent):
        super().__init__([], parent); self.univ, self.varb = univ, varb

    def lookup(self, names):
        if len(names) == 2:
            b = next((b for v, b in self.varb.items() if v.lower() == names[0].lower()), None)
            if b is None: return None
            c = _find_col(b, names[1])
            if c is None: raise SqlError(f"pattern variable {names[0]} has no column {names[1]}")
            return c
        if len(names) == 1: return _find_col(self.univ, names[0])
        return None


class Ctx:
    """what an expression may contain where it is bound"""
    def __init__(self, agg=False, win=False, aliases=None, where="expression"):
        self.agg, self.win, self.aliases, self.where = agg, win, aliases or {}, where
        self.in_agg = False


class Binder:
    def __init__(self, db, text="", flags=None):
        self.db, self.text, self.flags = db, text, dict(flags or {})
        self.ctes = []                 # stack of {name: CTE}
        self.subqueries = []

    # ------------------------------------------------------------ catalog
    def table_binding(self, name, alias):
        for frame in reversed(self.ctes):
            for k, cte in frame.items():
                if _ieq(k, name): return self.cte_binding(cte, alias or k)
        t = None
        for k, tb in self.db.tables.items():
            if k == name: t = tb; break
        if t is None:
            for k, tb in self.db.tables.items():
                if k.lower() == name.lower(): t = tb; break
        if t is None: raise SqlError(f"Table with name {name} does not exist")
        meta = getattr(self.db, "meta", {}).get(t.name, {}).get("columns", {})
        cols = []
        for c in _visible_cols(t):
            typ = None
            if c in meta and meta[c].get("type"):
                try: typ = T.type_from_name(meta[c]["type"])
                except SqlError: typ = None
            if typ is None: typ = T.from_codec(t.codecs.get(c))
            nl = c in getattr(t, "nullable", ()) or (c + "#raw") in getattr(t, "nullable", ()) or getattr(t, "fk_nulls", {}).get(c, 0) > 0
            cols.append((c, typ, nl))
        return I.Binding(alias or t.name, cols, table=t)

    def cte_binding(self, cte, alias):
        if cte.get("recursive_binding") is not None and cte.get("binding_in_progress"):
            b = cte["recursive_binding"]
            nb = I.Binding(alias, b.cols, q=b.q); nb.rec = b.rec; return nb
        if cte.get("q") is None:
            cte["binding_in_progress"] = True
            q = self.bind_cte(cte)
            cte["q"] = q
            cte["binding_in_progress"] = False
        q = cte["q"]
        qc = [n for n, _ in q_columns(q)]
        names = list(cte["aliases"] or []) + qc[len(cte["aliases"] or []):]        # SEARCH / CYCLE add columns
        cols = [(n, t, True) for n, (_, t) in zip(names, q_columns(q))]
        b = I.Binding(alias, cols, q=q)
        b.shared = cte                     # materialized once per query
        return b

    def bind_cte(self, cte):
        node = cte["node"]
        if node["type"] == "RECURSIVE_CTE_NODE":
            base = self.bind_query(node["left"], None)
            names = cte["aliases"] or [n for n, _ in q_columns(base)]
            sc = (self.flags.get("search_cycle") or {}).get(cte["name"].lower())
            cols = [(n, t, True) for n, (_, t) in zip(names, q_columns(base))]
            if sc: base, cols = _search_cycle_base(base, names, cols, sc, self)
            b = I.Binding(cte["name"], cols)
            rec = I.Recursive(b, base, None, bool(node.get("union_all")))
            b.rec = rec; b.q = rec
            cte["recursive_binding"] = b
            rec.step = self.bind_query(node["right"], None)
            st = q_columns(rec.step)
            if len(st) != len(names): raise SqlError("the recursive part of a CTE must have as many columns as its base")
            if sc: rec.step = _search_cycle_step(rec, names, sc, self)
            return rec
        return self.bind_query(node, None)

    def _references(self, node, name):
        s = json.dumps(node)
        return f'"table_name": "{name}"' in s or f'"table_name": "{name.lower()}"' in s

    # ------------------------------------------------------------ statements
    def bind_statement(self, node):
        return self.bind_query(node, None)

    def push_ctes(self, node, recursive_hint=False):
        m = (node.get("cte_map") or {}).get("map") or []
        if not m: return False
        frame = {}
        for kv in m:
            v = kv["value"]
            frame[kv["key"]] = dict(name=kv["key"], node=v["query_node"], aliases=v.get("aliases") or [], q=None,
                                    recursive=True)
        self.ctes.append(frame)
        return True

    def bind_query(self, node, parent):
        pushed = self.push_ctes(node)
        try:
            t = node["type"]
            if t == "SELECT_NODE": q = self.bind_select(node, parent)
            elif t == "SET_OPERATION_NODE": q = self.bind_setop(node, parent)
            elif t == "RECURSIVE_CTE_NODE":
                q = self.bind_setop(dict(node, type="SET_OPERATION_NODE", setop_type="UNION",
                                         children=[node["left"], node["right"]]), parent)
            else: raise SqlError(f"{t} statements are not supported (queries only)")
        finally:
            if pushed: self.ctes.pop()
        return q

    def bind_setop(self, node, parent):
        kids = node.get("children") or [node["left"], node["right"]]
        op = node["setop_type"]; all_ = bool(node.get("setop_all"))
        if op == "UNION_BY_NAME":
            qs = [self.bind_query(k, parent) for k in kids]
            names = []
            for q in qs:
                for n, _ in q_columns(q):
                    if n.lower() not in [x.lower() for x in names]: names.append(n)
            if self.flags.get("corresponding"):            # ISO: the columns every operand has, in the first's order
                names = [n for n in names if all(n.lower() in [m.lower() for m, _ in q_columns(q)] for q in qs)]
                if not names: raise SqlError("CORRESPONDING: the operands have no column name in common")
                byl = self.flags.get("corresponding_by") or []
                by = byl.pop(0) if byl else None
                if by:                                     # CORRESPONDING BY (columns): those, in that order
                    for c in by:
                        if c.lower() not in [n.lower() for n in names]: raise SqlError(f"CORRESPONDING BY: {c} is not a column of every operand")
                    names = [next(n for n in names if n.lower() == c.lower()) for c in by]
            qs = [_project_by_name(q, names) for q in qs]
            op = "UNION"
        else:
            qs = [self.bind_query(k, parent) for k in kids]
        cur = qs[0]
        for q in qs[1:]:
            a, b = q_columns(cur), q_columns(q)
            if len(a) != len(b): raise SqlError(f"Set operations can only apply to expressions with the same number of result columns")
            s = I.SetOp(op.lower(), all_, cur, q)
            s.items = [(na, T.supertype(ta, tb)) for (na, ta), (_, tb) in zip(a, b)]
            cur = s
        if not isinstance(cur, I.SetOp):
            return cur
        self.bind_modifiers(node, cur, None, setop=True)
        return cur

    # ------------------------------------------------------------ FROM
    def bind_from(self, ft, parent, out):
        """-> FROM tree (Binding or Join); out collects the bindings in order; also returns USING columns"""
        typ = ft["type"]
        if typ == "EMPTY": return None
        if typ == "BASE_TABLE":
            b = self.table_binding(ft["table_name"], ft.get("alias") or None)
            if ft.get("column_name_alias"): b = _rename(b, ft["column_name_alias"])
            b.sample = _sample_spec(ft.get("sample"))
            out.append(b); return b
        if typ == "SUBQUERY":
            q = self.bind_query(ft["subquery"]["node"], parent)
            names = list(ft.get("column_name_alias") or []) + [n for n, _ in q_columns(q)][len(ft.get("column_name_alias") or []):]
            b = I.Binding(ft.get("alias") or f"subquery{len(out)}", [(n, t, True) for n, (_, t) in zip(names, q_columns(q))], q=q)
            b.sample = _sample_spec(ft.get("sample"))
            out.append(b); return b
        if typ == "EXPRESSION_LIST":
            rows = []
            sc = Scope([], parent)
            for r in ft["values"]:
                rows.append([self.expr(x, sc, Ctx()) for x in r])
            n = len(rows[0])
            types = [T.supertype_all([r[i].t for r in rows], [getattr(r[i], "strlit", False) for r in rows]) for i in range(n)]
            rows = [[cast_to(r[i], types[i]) for i in range(n)] for r in rows]
            names = list(ft.get("column_name_alias") or []) + [f"col{i}" for i in range(n)][len(ft.get("column_name_alias") or []):]
            q = I.Values(rows, types, names)
            b = I.Binding(ft.get("alias") or "valueslist", [(nm, t, True) for nm, t in zip(names, types)], q=q)
            out.append(b); return b
        if typ == "JOIN":
            left = self.bind_from(ft["left"], parent, out)
            nleft = len(out)
            # the right item may read the columns of the items before it (LATERAL, ISO T491; also UNNEST(t.a))
            right = self.bind_from(ft["right"], Scope(list(out), parent), out)
            kind = {"INNER": "inner", "LEFT": "left", "RIGHT": "right", "OUTER": "full", "FULL": "full",
                    "SEMI": "semi", "ANTI": "anti"}.get(ft.get("join_type", "INNER"), None)
            if kind is None: raise SqlError(f"join type {ft.get('join_type')} is not supported")
            ref = ft.get("ref_type", "REGULAR")
            if ref == "CROSS": kind = "cross" if kind == "inner" else kind
            if ref not in ("REGULAR", "CROSS", "NATURAL"): raise SqlError(f"{ref} joins are not supported")
            lb, rb = out[:nleft], out[nleft:]
            using = list(ft.get("using_columns") or [])
            if ref == "NATURAL":
                lnames = {_shown(b, c).lower() for b in lb for c, _, _ in b.cols}
                using = [_shown(b, c) for b in rb for c, _, _ in b.cols if _shown(b, c).lower() in lnames]
                if not using: kind = "cross" if kind == "inner" else kind
            j = I.Join(kind, left, right)
            sc = Scope(out, parent)
            pj = self.flags.get("pjoin") or {}
            for side, part in (("left", left), ("right", right)):
                if isinstance(part, I.Binding) and part.alias.lower() in pj:
                    if kind not in ("left", "right", "full"): raise SqlError("PARTITION BY in a join goes with an outer join")
                    exprs = [self.expr(_parse_expr(x), Scope([part], parent), Ctx(where="PARTITION BY")) for x in pj[part.alias.lower()]]
                    for x in exprs:
                        if not isinstance(x, I.Col): raise SqlError("the PARTITION BY of a partitioned join takes columns")
                    j.partition = (side, exprs)
            if using:
                conds = []
                j.using = []
                for cname in using:
                    lc = Scope(lb).lookup([cname]); rc = Scope(rb).lookup([cname])
                    if lc is None or rc is None: raise SqlError(f"USING column {cname!r} is not in both sides")
                    conds.append(self.cmp("=", lc, rc)); j.using.append((cname, lc, rc))
                j.cond = make_and(conds)
            elif ft.get("condition") is not None:
                j.cond = self.expr(ft["condition"], sc, Ctx(where="ON"))
            if kind in ("left", "full"):
                for b in rb: b.outer_null = True
            if kind in ("right", "full"):
                for b in lb: b.outer_null = True
            return j
        if typ == "TABLE_FUNCTION":
            return self.table_function(ft, parent, out)
        raise SqlError(f"FROM item {typ} is not supported")

    def table_function(self, ft, parent, out):
        fn = ft["function"]
        name = fn["function_name"].lower()
        if name == "__match_recognize": return self.match_recognize(ft, fn["arguments"], parent, out)
        args = [self.expr(a["expression"], Scope([], parent), Ctx()) for a in fn["arguments"]]
        alias = list(ft.get("column_name_alias") or [])
        ordinality = ft.get("with_ordinality") in ("WITH_ORDINALITY", True)
        if name in ("range", "generate_series"):
            if not 1 <= len(args) <= 3: raise SqlError(f"{name} takes 1 to 3 arguments")
            args = [cast_to(a, T.BIGINT) if a.t.integer or a.t.id == "NULL" else a for a in args]
            if any(not a.t.integer for a in args): raise SqlError(f"{name} of {', '.join(str(a.t) for a in args)} is not supported")
            names, types = [name], [T.BIGINT]
        elif name == "__json_table":                 # JSON_TABLE (ISO 7.11)
            return self.json_table(ft, args, alias, out)
        elif name == "unnest":                       # UNNEST(collection, ...) [WITH ORDINALITY] (ISO 7.6)
            if not args: raise SqlError("UNNEST needs a collection")
            for a in args:
                if a.t.id not in ("LIST", "MULTISET", "NULL"): raise SqlError(f"UNNEST of {a.t}: an array or a multiset is needed")
            names = ["unnest"] if len(args) == 1 else [f"unnest{k + 1}" for k in range(len(args))]
            types = [a.t.child if a.t.id != "NULL" else T.INTEGER for a in args]
        else: raise SqlError(f"table function {name} is not supported")
        if ordinality: names, types = names + ["ordinality"], types + [T.BIGINT]
        names = alias + names[len(alias):]
        if len(names) != len(types): raise SqlError(f"{name}: {len(types)} column(s), {len(alias)} name(s) given")
        q = I.TableFn(name, args, names, types, ordinality)
        b = I.Binding(ft.get("alias") or name, [(nm, t, True) for nm, t in zip(names, types)], q=q)
        out.append(b); return b

    def match_recognize(self, ft, raw, parent, out):
        import sqlmr
        q = self.bind_query(raw[0]["expression"]["subquery"]["node"], parent)
        spec = json.loads(raw[1]["expression"]["literal"]["text"])
        cols = q_columns(q)
        names = [n for n, _ in cols]
        pattern = sqlmr.parse_pattern(spec["pattern"])
        pvars = sqlmr.pattern_vars(pattern)
        subsets = {k: v for k, v in spec["subsets"].items()}
        for k, vs in subsets.items():
            if k in pvars: raise SqlError(f"SUBSET {k}: the name of a pattern variable")
            for v in vs:
                if v not in pvars: raise SqlError(f"SUBSET {k}: {v} is not a pattern variable")
        for v in spec["define"]:
            if v not in pvars: raise SqlError(f"DEFINE {v}: not a variable of the PATTERN")
        univ = I.Binding("__mr", [(n, t, True) for n, t in cols])
        varb = {v: I.Binding(v, [(n, t, True) for n, t in cols]) for v in pvars + list(subsets)}
        sc = _MRScope(univ, varb, parent)
        prev, self.mr_ctx = getattr(self, "mr_ctx", False), True
        try:
            bind = lambda text, agg=True: self.expr(_parse_expr(text), sc, Ctx(agg=agg, where="MATCH_RECOGNIZE"))
            partition = [bind(x, False) for x in spec["partition"]]
            order = [(bind(x, False), d, nf) for x, d, nf in spec["order"]]
            define = {v: self.boolean(bind(x), "DEFINE") for v, x in spec["define"].items()}
            measures = [(bind(x), n) for x, n in spec["measures"]]
        finally: self.mr_ctx = prev
        pnames = []
        for p in partition:
            if not (isinstance(p, I.Col) and p.b is univ): raise SqlError("PARTITION BY in MATCH_RECOGNIZE takes columns")
            pnames.append(p.name)
        mnames = [n for _, n in measures]
        if spec["rows"] == "one":
            onames = pnames + mnames
            otypes = [p.t for p in partition] + [m.t for m, _ in measures]
            outs = None
        else:
            onames, otypes, outs = [], [], []
            for p in partition: onames.append(p.name); otypes.append(p.t); outs.append(("in", p.name))
            for o in order:
                if isinstance(o[0], I.Col) and o[0].b is univ and o[0].name not in onames:
                    onames.append(o[0].name); otypes.append(o[0].t); outs.append(("in", o[0].name))
            for k, (m, n) in enumerate(measures): onames.append(n); otypes.append(m.t); outs.append(("m", k))
            for n, t in cols:
                if n not in onames: onames.append(n); otypes.append(t); outs.append(("in", n))
        if len({n.lower() for n in onames}) != len(onames): raise SqlError("MATCH_RECOGNIZE: a measure has the name of a column")
        mr = sqlmr.MR(pattern, univ, varb, subsets, define, measures, partition, order, spec["rows"], spec["empty"], tuple(spec["skip"]), outs, names)
        alias = list(ft.get("column_name_alias") or [])
        onames = alias + onames[len(alias):]
        tq = I.TableFn("match_recognize", [], onames, otypes); tq.mr = mr; tq.input = q
        b = I.Binding(ft.get("alias") or "match_recognize", [(n, t, True) for n, t in zip(onames, otypes)], q=tq)
        out.append(b); return b

    def json_table(self, ft, args, alias, out):
        import sqljson
        spec = json.loads(args[1].v)
        np_ = spec["np"]; passing, defaults = args[2:2 + np_], args[2 + np_:]
        spec["ptypes"] = [str(a.t) for a in passing]
        names, types = [], []
        def level(cols):
            for c in cols:
                if c["kind"] == "nested":
                    sqljson.compile_path(c["path"]); level(c["cols"]); continue
                c["pos"] = len(names); names.append(c["name"])
                if c["kind"] == "ordinality": types.append(T.BIGINT); continue
                t = T.type_from_name(c["type"]) if c.get("type") else (T.JSON if c["kind"] == "query" else T.VARCHAR)
                if c["kind"] == "exists" and t.id not in ("BOOLEAN", "VARCHAR") and not t.integer: raise SqlError(f"EXISTS column {c['name']}: BOOLEAN, an integer or a string type")
                types.append(t); c["rtype"] = str(t)
                sqljson.compile_path(c["opts"]["path"])
        sqljson.compile_path(spec["path"])
        level(spec["cols"])
        if len(set(n.lower() for n in names)) != len(names): raise SqlError("JSON_TABLE: column names must be distinct")
        names = alias + names[len(alias):]
        x = args[0]
        if x.t.id not in ("VARCHAR", "JSON", "NULL", "BLOB") and not getattr(x, "strlit", False):
            raise SqlError(f"JSON_TABLE needs JSON text, not {x.t}")
        q = I.TableFn("json_table", [x] + passing + defaults, names, types)
        q.spec = spec
        b = I.Binding(ft.get("alias") or "json_table", [(nm, t, True) for nm, t in zip(names, types)], q=q)
        out.append(b); return b

    # ------------------------------------------------------------ SELECT
    def bind_select(self, node, parent):
        s = I.Select()
        bindings = []
        s.from_ = self.bind_from(node.get("from_table") or {"type": "EMPTY"}, parent, bindings)
        s.sample = _sample_spec(node.get("sample"))
        using = {}
        def collect(j):
            if isinstance(j, I.Join):
                collect(j.left); collect(j.right)
                for cname, lc, rc in getattr(j, "using", []):
                    if j.kind == "right": using[cname.lower()] = rc
                    elif j.kind == "full": using[cname.lower()] = self.func("coalesce", [lc, rc])
                    else: using[cname.lower()] = lc
        collect(s.from_)
        sc = Scope(bindings, parent, using)
        s.scope = sc
        # SELECT list (stars expanded)
        items = []
        for e in node["select_list"]:
            if e["class"] == "STAR": items += self.expand_star(e, sc, s.from_)
            else:
                x = self.expr(e, sc, Ctx(agg=True, win=True))
                items.append((x, e.get("alias") or self.name_of(e, x)))
        s.items = items
        aliases = {}
        for i, (x, n) in enumerate(items):
            aliases.setdefault(n.lower(), x)
        if node.get("where_clause") is not None:
            s.where = self.boolean(self.expr(node["where_clause"], sc, Ctx(where="WHERE")), "WHERE")
        # GROUP BY
        groups = []
        gb_distinct, idxmap = False, []
        for g in node.get("group_expressions") or []:
            if g["class"] == "FUNCTION" and g["function_name"] == "__iso_gb_distinct":
                gb_distinct = True; idxmap.append(None); continue
            idxmap.append(len(groups))
            if g["class"] == "CONSTANT" and g["literal"]["kind"] == "INTEGER":
                k = int(g["literal"]["text"])
                if not 1 <= k <= len(items): raise SqlError(f"GROUP BY term out of range: {k}")
                x = items[k - 1][0]
                if any(x.key() == y.key() for y in groups) and s.items[k - 1][0] is x and any(
                        s.items[j][0].key() == x.key() and j != k - 1 for j in range(len(items))):
                    # GROUP BY 1, 2 where items 1 and 2 are the same expression: two grouping columns (DuckDB)
                    x = I.Func("__grp", [x], x.t, extra=("pos", k - 1))
                    s.items[k - 1] = (x, s.items[k - 1][1]); items = s.items
            elif g["class"] == "COLUMN_REF" and len(g["column_names"]) == 1 and sc.lookup(g["column_names"]) is None \
                    and g["column_names"][0].lower() in aliases:
                x = aliases[g["column_names"][0].lower()]
            else:
                x = self.expr(g, sc, Ctx(where="GROUP BY"))
            if I.has(x, I.Agg): raise SqlError("GROUP BY clause cannot contain aggregates")
            if I.has(x, I.Win): raise SqlError("GROUP BY clause cannot contain window functions")
            groups.append(x)
        # distinct group expressions, sets as index lists
        uniq, pos = [], []
        for g in groups:
            k = g.key()
            for i, u in enumerate(uniq):
                if u.key() == k: pos.append(i); break
            else: pos.append(len(uniq)); uniq.append(g)
        s.group = uniq
        sets = node.get("group_sets") or []
        if sets:
            s.sets = [sorted({pos[idxmap[i]] for i in st if idxmap[i] is not None}) for st in sets]
            if gb_distinct:                              # GROUP BY DISTINCT: each grouping set once
                seen = []
                for st in s.sets:
                    if st not in seen: seen.append(st)
                s.sets = seen
            if s.sets == [list(range(len(uniq)))] and len(sets) == 1: s.sets = None
        if node.get("aggregate_handling") == "FORCE_AGGREGATES":          # GROUP BY ALL
            s.group = [x for x, _ in items if not I.has(x, I.Agg) and not I.has(x, I.Win) and I.cols_of(x)]
            s.sets = None
        if node.get("having") is not None:
            s.having = self.boolean(self.expr(node["having"], sc, Ctx(agg=True, aliases=aliases, where="HAVING")), "HAVING")
        if node.get("qualify") is not None:
            s.qualify = self.boolean(self.expr(node["qualify"], sc, Ctx(agg=True, win=True, aliases=aliases, where="QUALIFY")), "QUALIFY")
        s.aggregate = bool(s.group) or s.sets is not None or any(I.has(x, I.Agg) for x, _ in items) or s.having is not None \
            or (s.qualify is not None and I.has(s.qualify, I.Agg))
        self.bind_modifiers(node, s, sc, aliases=aliases)
        if s.aggregate:
            _functional_dependencies(s)
            _check_grouped(s)
        return s

    def bind_modifiers(self, node, s, sc, setop=False, aliases=None):
        cols = q_columns(s)
        for m in node.get("modifiers") or []:
            t = m["type"]
            if t == "ORDER_MODIFIER":
                for o in m["orders"]:
                    desc = o["type"] == "DESCENDING"
                    nf = {"NULLS_FIRST": True, "NULLS FIRST": True, "NULLS_LAST": False, "NULLS LAST": False}.get(o.get("null_order"), False)
                    s.order.append((self.order_expr(o["expression"], s, sc, setop, aliases), desc, nf))
            elif t == "LIMIT_MODIFIER" or t == "LIMIT_PERCENT_MODIFIER":
                lim = m.get("limit"); off = m.get("offset")
                if lim is not None:
                    le = self.expr(lim, Scope([], sc), Ctx(where="LIMIT"))
                    if isinstance(le, I.Func) and le.fn == "__iso_ties":
                        s.with_ties = True
                        if len(le.args) > 1: s.percent = "iso"
                        le = le.args[0]
                    if isinstance(le, I.Func) and le.fn == "__iso_pct":           # ISO FETCH ... PERCENT: ceiling
                        s.percent = "iso"; le = le.args[0]
                    if (t == "LIMIT_PERCENT_MODIFIER" or m.get("limit_type") == "PERCENTAGE") and not s.percent: s.percent = True
                    s.limit = _count_param(le, "FETCH FIRST / LIMIT")
                if off is not None: s.offset = _count_param(self.expr(off, Scope([], sc), Ctx(where="OFFSET")), "OFFSET")
            elif t == "DISTINCT_MODIFIER":
                if setop: continue
                s.distinct = True
                if m.get("distinct_on_targets"):
                    s.distinct_on = [self.order_expr(x, s, sc, setop, aliases) for x in m["distinct_on_targets"]]
            else:
                raise SqlError(f"{t} is not supported")
        if s.with_ties and not s.order: raise SqlError("FETCH ... WITH TIES needs ORDER BY")

    def order_expr(self, e, s, sc, setop, aliases):
        """ORDER BY term: a position, an output column name, or (in a SELECT) any expression"""
        cols = q_columns(s)
        if e["class"] == "CONSTANT" and e["literal"]["kind"] == "INTEGER":
            k = int(e["literal"]["text"])
            if not 1 <= k <= len(cols): raise SqlError(f"ORDER term out of range: {k}")
            return ("pos", k - 1)
        if e["class"] == "COLUMN_REF" and len(e["column_names"]) == 1:
            n = e["column_names"][0]
            for i, (cn, _) in enumerate(cols):
                if cn == n or cn.lower() == n.lower():
                    if setop or sc is None or sc.lookup([n]) is None or _is_plain_col(s.items[i][0], n): return ("pos", i)
        if setop: raise SqlError("ORDER BY of a set operation must name an output column")
        x = self.expr(e, sc, Ctx(agg=True, win=True, aliases=aliases, where="ORDER BY"))
        for i, (it, _) in enumerate(s.items):
            if it.key() == x.key(): return ("pos", i)
        return ("expr", x)

    def expand_star(self, e, sc, from_):
        rel = e.get("relation_name") or ""
        excl = {x.lower() for x in (e.get("exclude_list") or [])}
        for q in e.get("qualified_exclude_list") or []:
            excl.add((q.get("column_name") or q.get("name") or "").lower())
        repl = {}
        for r in e.get("replace_list") or []:
            repl[r["name"].lower() if "name" in r else r["column"].lower()] = r
        if e.get("columns") or e.get("expr") is not None: raise SqlError("COLUMNS(...) is not supported")
        out, seen_using = [], set()
        using_cols = {}
        def walk(j):
            if isinstance(j, I.Join):
                walk(j.left); walk(j.right)
                for cname, lc, rc in getattr(j, "using", []): using_cols[id(rc.b), rc.name.lower()] = cname
        walk(from_)
        for b in sc.bindings:
            if rel and not _ieq(b.alias, rel): continue
            for c0, t, nl in b.cols:
                c = _shown(b, c0)
                if c.lower() in excl: continue
                if not rel and (id(b), c0.lower()) in using_cols: continue
                col = I.Col(b, c0, t, nl)
                if not rel and c.lower() in sc.using and sc.using[c.lower()].key() != col.key() and c.lower() in seen_using: continue
                if not rel and c.lower() in sc.using:
                    seen_using.add(c.lower()); col = sc.using[c.lower()]
                out.append((col, c))
        if rel and not out and not any(_ieq(b.alias, rel) for b in sc.bindings): raise SqlError(f"Referenced table {rel!r} not found")
        if not out and sc.bindings: raise SqlError("SELECT * with no columns")
        return out

    def name_of(self, e, x):
        if isinstance(x, I.Col) and e["class"] == "COLUMN_REF": return _shown(x.b, x.name)
        loc, ln = e.get("query_location"), e.get("query_location_length")
        if isinstance(loc, int) and isinstance(ln, int) and 0 < ln < 1000 and loc < len(self.text):
            return self.text[loc:loc + ln]
        return I.render(x)

    def boolean(self, x, where):
        if x.t.id not in ("BOOLEAN", "NULL"):
            if isinstance(x, I.Lit) and x.strlit: return cast_to(x, T.BOOLEAN)
            if x.t.integer or x.t.id == "VARCHAR": return cast_to(x, T.BOOLEAN)
            raise SqlError(f"{where} clause must be a BOOLEAN, not {x.t}")
        return x

    # ------------------------------------------------------------ expressions
    def func(self, name, args, extra=None): return make_func(name, args, extra)

    def cmp(self, op, a, b):
        if _is_row(a) or _is_row(b): return self.row_cmp(op, a, b)
        return make_func(op, [a, b])

    def row_cmp(self, op, a, b):
        la = a.args if _is_row(a) else [a]; lb = b.args if _is_row(b) else [b]
        if len(la) != len(lb): raise SqlError("row values of different sizes")
        if op in ("=", "is_not_distinct"): return make_and([make_func(op, [x, y]) for x, y in zip(la, lb)])
        if op in ("<>", "is_distinct"):
            return make_func("not", [self.row_cmp("=" if op == "<>" else "is_not_distinct", a, b)]) if op == "<>" else \
                make_func("not", [self.row_cmp("is_not_distinct", a, b)])
        strict = op in ("<", ">"); base = "<" if op in ("<", "<=") else ">"
        # (a1, a2) < (b1, b2)  =  a1 < b1 OR (a1 = b1 AND a2 < b2) ... (lexicographic, ISO 8.2)
        out = None
        for k in range(len(la) - 1, -1, -1):
            last = k == len(la) - 1
            here = make_func(base if (strict or not last) else base + "=", [la[k], lb[k]])
            if out is None: out = here
            else: out = make_func("or", [make_func(base, [la[k], lb[k]]), make_and([make_func("=", [la[k], lb[k]]), out])])
        return out

    def expr(self, e, sc, ctx):
        c = e["class"]
        if c == "CONSTANT": return self.constant(e["literal"])
        if c == "COLUMN_REF": return self.column(e["column_names"], sc, ctx)
        if c == "COMPARISON":
            r = e["right"]
            if r.get("class") == "FUNCTION" and r["function_name"].startswith("__iso_pred_") \
                    and e["type"] in ("COMPARE_DISTINCT_FROM", "COMPARE_NOT_DISTINCT_FROM"):
                x = self.iso_pred(r["function_name"][11:], e["left"], r.get("arguments") or [], sc, ctx)
                return self.func("not", [x]) if e["type"] == "COMPARE_DISTINCT_FROM" else x
            return self.cmp(_CMP[e["type"]], self.expr(e["left"], sc, ctx), self.expr(r, sc, ctx))
        if c == "CONJUNCTION":
            kids = [self.boolean(self.expr(x, sc, ctx), "AND/OR") for x in e["children"]]
            fn = "and" if e["type"] == "CONJUNCTION_AND" else "or"
            return self.func(fn, kids) if not all(isinstance(k, I.Lit) for k in kids) else self.func(fn, kids)
        if c == "OPERATOR": return self.operator(e, sc, ctx)
        if c == "BETWEEN":
            x = self.expr(e["input"], sc, ctx); lo = e["lower"]
            if lo["class"] == "FUNCTION" and lo["function_name"] == "__iso_sym":
                return self.func("between_sym", [x, self.expr(lo["arguments"][0]["expression"], sc, ctx), self.expr(e["upper"], sc, ctx)])
            return self.func("between", [x, self.expr(lo, sc, ctx), self.expr(e["upper"], sc, ctx)])
        if c == "CASE":
            whens = [(self.boolean(self.expr(k["when_expr"], sc, ctx), "WHEN"), self.expr(k["then_expr"], sc, ctx)) for k in e["case_checks"]]
            el = self.expr(e["else_expr"], sc, ctx) if e.get("else_expr") is not None else None
            if isinstance(el, I.Lit) and el.v is None and el.t.id == "NULL": el = None
            vals = [v for _, v in whens] + ([el] if el is not None else [])
            t = T.supertype_all([v.t for v in vals], [getattr(v, "strlit", False) for v in vals])
            if t.id == "NULL": t = T.INTEGER
            whens = [(w, cast_to(v, t)) for w, v in whens]
            return I.Case(whens, cast_to(el, t) if el is not None else None, t)
        if c == "CAST":
            x = self.expr(e["child"], sc, ctx); t = self.type_of(e["type_expr"])
            if isinstance(x, I.Lit):
                if x.v is None or isinstance(x.v, F.Err): return I.Lit(x.v, t)
                try: return I.Lit(T.cast_value(x.v, x.t, t, not e.get("try_cast")), t)
                except SqlError as ex:
                    if e.get("try_cast"): return I.Lit(None, t)
                    return I.Lit(F.Err(str(ex)), t)
            return I.Cast(x, t, bool(e.get("try_cast")))
        if c == "FUNCTION": return self.function(e, sc, ctx)
        if c == "WINDOW": return self.window(e, sc, ctx)
        if c == "SUBQUERY": return self.subquery(e, sc, ctx)
        if c == "COLLATE":
            x = self.expr(e["child"], sc, ctx)
            if isinstance(x, I.Lit) and x.t.id == "VARCHAR": x = I.Lit(x.v, T.VARCHAR)
            return self.func("__collate", [x, I.Lit(str(e["collation"]).upper(), T.VARCHAR)])
        if c == "POSITIONAL_REFERENCE": raise SqlError("positional references (#n) are not supported")
        if c == "STAR": raise SqlError("* is only allowed in the SELECT list or COUNT(*)")
        raise SqlError(f"expression {c} is not supported")

    def sql_json(self, name, raw, sc, ctx):
        """the SQL/JSON functions the pre-pass wrote as __json_*(...): typed calls whose clauses travel as a constant"""
        import sqljson
        if name == "__json_array_q":                              # JSON_ARRAY(<query>)
            q = self.bind_query(raw[0]["expression"]["subquery"]["node"], sc)
            cols = q_columns(q)
            if len(cols) != 1: raise SqlError("JSON_ARRAY(<query>): the query must have one column")
            o = json.loads(self.expr(raw[1]["expression"], sc, ctx).v); o["vtypes"] = [str(cols[0][1])]
            sub = I.Subq("array", q, T.LIST(cols[0][1]), correlated=_correlated(q))
            rt = T.type_from_name(o["returning"]) if o.get("returning") else T.JSON
            return I.Func("__json_array_list", [sub, I.Lit(json.dumps(o), T.VARCHAR)], rt)
        args = self.args_of({"arguments": raw}, sc, ctx)
        lit = lambda o: I.Lit(json.dumps(o), T.VARCHAR)
        def text_arg(x, what):
            if x.t.id not in ("VARCHAR", "JSON", "NULL", "BLOB") and not getattr(x, "strlit", False):
                raise SqlError(f"{what} needs JSON text, not {x.t}")
            return x
        if name in ("__json_value", "__json_query", "__json_exists"):
            o = json.loads(args[1].v)
            sqljson.compile_path(o["path"])                     # a syntax error in the path fails the query now
            np_ = o.get("np", 0); passing, defaults = args[2:2 + np_], args[2 + np_:]
            o["ptypes"] = [str(a.t) for a in passing]
            what = name[2:].upper()
            if name == "__json_value":
                rt = T.type_from_name(o["returning"]) if o.get("returning") else T.VARCHAR
                defaults = [cast_to(d, rt) for d in defaults]
            elif name == "__json_query": rt = T.type_from_name(o["returning"]) if o.get("returning") else T.JSON
            else: rt = T.BOOLEAN
            return I.Func(name, [text_arg(args[0], what), lit(o)] + passing + defaults, rt)
        if name in ("__json_object", "__json_array"):
            o = json.loads(args[0].v); rest = args[1:]
            if name == "__json_object":
                keys = [cast_to(k, T.VARCHAR) if k.t.id != "VARCHAR" else k for k in rest[0::2]]
                vals = rest[1::2]; rest = [x for kv in zip(keys, vals) for x in kv]
            else: vals = rest
            o["vtypes"] = [str(v.t) for v in vals]
            rt = T.type_from_name(o["returning"]) if o.get("returning") else T.JSON
            return I.Func(name, [lit(o)] + rest, rt)
        if name == "__json_serialize":
            o = json.loads(args[1].v)
            rt = T.type_from_name(o["returning"]) if o.get("returning") else T.VARCHAR
            if rt.id not in ("VARCHAR", "BLOB", "JSON"): raise SqlError(f"JSON_SERIALIZE RETURNING {rt}: a character or binary string type")
            return I.Func(name, [text_arg(args[0], "JSON_SERIALIZE"), args[1]], rt)
        if name in ("__json_parse", "__json_scalar"):
            if name == "__json_parse": text_arg(args[0], "JSON")
            return I.Func(name, args[:1] + [args[1] if len(args) > 1 else I.Lit(False, T.BOOLEAN)], T.JSON)
        raise SqlError(f"internal: {name}")

    def mr_function(self, name, e, sc, ctx):
        """PREV / NEXT / FIRST / LAST, RUNNING / FINAL, CLASSIFIER(), MATCH_NUMBER(), COUNT(A.*) in MATCH_RECOGNIZE"""
        raw = e.get("arguments") or []
        if name in ("prev", "next", "first", "last"):
            if not 1 <= len(raw) <= 2: raise SqlError(f"{name.upper()}(<expression> [, <offset>])")
            x = self.expr(raw[0]["expression"], sc, ctx)
            args = [x] + ([self.expr(raw[1]["expression"], sc, ctx)] if len(raw) > 1 else [])
            if len(args) > 1 and not isinstance(args[1], I.Lit): raise SqlError(f"{name.upper()}: the offset must be a constant")
            return I.Func("__mr_nav", args, x.t, extra=(name,))
        if name in ("__mr_running", "__mr_final"):
            x = self.expr(raw[0]["expression"], sc, ctx)
            return I.Func("__mr_sem", [x], x.t, extra=(name[5:],))
        if name == "classifier": return I.Func("__mr_classifier", [], T.VARCHAR)
        if name == "match_number": return I.Func("__mr_match_number", [], T.BIGINT)
        if name == "count" and len(raw) == 1 and raw[0]["expression"].get("class") == "STAR":
            rn = (raw[0]["expression"].get("relation_name") or "").upper()
            return I.Agg("count_star", [], T.BIGINT, params=("mrvar", rn) if rn else ())
        return None

    def iso_pred(self, name, left, args, sc, ctx):
        """the postfix predicates the pre-pass wrote as  x IS NOT DISTINCT FROM __iso_pred_<name>(...)"""
        arg = lambda k: args[k]["expression"] if "expression" in args[k] else args[k]
        if name == "normalized":
            x = self.expr(left, sc, ctx)
            if x.t.id not in ("VARCHAR", "NULL") and not getattr(x, "strlit", False): raise SqlError(f"IS NORMALIZED applies to character strings, not {x.t}")
            return self.func("__is_normalized", [x, self.expr(arg(0), sc, ctx)])
        if name == "match":
            x = self.expr(left, sc, ctx)
            if not _is_row(x): x = I.Func("row", [x], T.SqlType("ROW"))
            kind = self.expr(arg(0), sc, ctx).v; uniq = self.expr(arg(1), sc, ctx).v
            sq = arg(2)
            if sq.get("class") != "SUBQUERY": raise SqlError("MATCH needs a subquery")
            q = self.bind_query(sq["subquery"]["node"], sc)
            if len(q_columns(q)) != len(x.args): raise SqlError("MATCH: the row and the subquery have different degrees")
            return I.Subq("match", q, T.BOOLEAN, x, (kind, bool(uniq)), correlated=_correlated(q))
        if name == "member":
            return self.func("__ms_member", [self.expr(left, sc, ctx), self.expr(arg(0), sc, ctx)])
        if name == "submultiset":
            return self.func("__ms_submultiset", [self.expr(left, sc, ctx), self.expr(arg(0), sc, ctx)])
        if name == "aset":
            return self.func("__ms_is_a_set", [self.expr(left, sc, ctx)])
        if name == "json":
            x = self.expr(left, sc, ctx)
            if x.t.id not in ("VARCHAR", "JSON", "NULL", "BLOB") and not getattr(x, "strlit", False):
                raise SqlError(f"IS JSON applies to character and binary strings, not {x.t}")
            return self.func("__iso_is_json", [x, self.expr(arg(0), sc, ctx), self.expr(arg(1), sc, ctx)])
        raise SqlError(f"internal: unknown ISO predicate {name}")

    def type_of(self, te):
        name = te["type_name"]
        if name.lower() == "list":
            return T.LIST(self.type_of(te["children"][0]))
        if name.lower() == "array":                            # <type> ARRAY [n]: an array of at most n elements
            ch = te["children"]
            n = int(ch[1]["literal"]["text"]) if len(ch) > 1 and ch[1].get("class") == "CONSTANT" else None
            return T.LIST(self.type_of(ch[0]), n if n and n > 0 else None)
        args = []
        for ch in te.get("children") or []:
            if ch.get("class") == "CONSTANT": args.append(int(ch["literal"]["text"]))
        if name.upper().startswith("DECIMAL") or name.upper() in ("NUMERIC", "DEC"):
            return T.type_from_name("DECIMAL", args)
        return T.type_from_name(name)

    def constant(self, lit):
        k, txt = lit["kind"], lit.get("text")
        if k == "NULL_LITERAL" or k == "NULL": return I.Lit(None, T.NULL)
        if k == "INTEGER" or k in ("BIGINT", "HUGEINT", "UBIGINT"):
            v = int(txt)
            for name in ("INTEGER", "BIGINT", "HUGEINT", "UHUGEINT"):
                lo, hi = T.INT_RANGE[name]
                if lo <= v <= hi: return I.Lit(v, T.SqlType(name))
            raise SqlError(f"integer literal {txt} is too large")
        if k == "NUMERIC" or k == "DECIMAL" or k == "DOUBLE":
            if "e" in txt.lower(): return I.Lit(float(txt), T.DOUBLE)
            d = Decimal(txt)
            tup = d.as_tuple(); s = max(0, -tup.exponent)
            digits = len(tup.digits)
            p = max(digits, s, 1)
            if p > 38: return I.Lit(float(d), T.DOUBLE)
            return I.Lit(d, T.DECIMAL(p, s))
        if k == "STRING" or k == "VARCHAR": return I.Lit(txt, T.VARCHAR, True)
        if k == "BOOLEAN": return I.Lit(txt.lower() == "true", T.BOOLEAN)
        if k in ("HEX", "BLOB"):                                         # X'...': a binary string literal
            h = re.sub(r"\s+", "", txt)
            try: return I.Lit(bytes.fromhex(h), T.BLOB)
            except ValueError: raise SqlError(f"invalid binary string literal X'{txt}'")
        raise SqlError(f"literal of kind {k} is not supported")

    def column(self, names, sc, ctx):
        names = list(names)
        depth, s = 0, sc
        while s is not None:
            c = s.lookup(names)
            if c is not None:
                if depth == 0: return c
                sc.correlated = True
                return I.Outer(c, depth)
            s = s.parent; depth += 1
        if len(names) == 1 and ctx.aliases and names[0].lower() in ctx.aliases:      # an output column alias (DuckDB)
            return ctx.aliases[names[0].lower()]
        if len(names) == 1 and names[0].upper() in _NILADIC:                       # CURRENT_DATE, CURRENT_USER, ...
            return self.func(names[0].lower(), [])
        for k in (2, 1):                              # j.a.b on a JSON column: the simplified accessor (ISO 2023, T860)
            if len(names) > k:
                try: base = self.column(names[:k], sc, ctx)
                except SqlError: continue
                if base.t.id == "JSON":
                    return _json_accessor(base, "".join("." + json.dumps(n) for n in names[k:]))
        raise SqlError(f'Referenced column "{".".join(names)}" not found in FROM clause')

    def operator(self, e, sc, ctx):
        t = e["type"]; kids = e.get("children") or []
        if t == "OPERATOR_NOT":
            return self.func("not", [self.boolean(self.expr(kids[0], sc, ctx), "NOT")])
        if t in ("OPERATOR_IS_NULL", "OPERATOR_IS_NOT_NULL"):
            x = self.expr(kids[0], sc, ctx); fn = "is_null" if t == "OPERATOR_IS_NULL" else "is_not_null"
            if _is_row(x): return make_and([self.func(fn, [a]) for a in x.args])     # every field NULL / not NULL (ISO 8.8)
            return self.func(fn, [x])
        if t in ("COMPARE_IN", "COMPARE_NOT_IN"):
            x = self.expr(kids[0], sc, ctx)
            vals = [self.expr(k, sc, ctx) for k in kids[1:]]
            if _is_row(x):
                r = make_func("or", [self.row_cmp("=", x, v) for v in vals]) if len(vals) > 1 else self.row_cmp("=", x, vals[0])
            else: r = self.func("in", [x] + vals)
            return self.func("not", [r]) if t == "COMPARE_NOT_IN" else r
        if t == "OPERATOR_COALESCE": return self.func("coalesce", [self.expr(k, sc, ctx) for k in kids])
        if t == "OPERATOR_NULLIF": return self.func("nullif", [self.expr(k, sc, ctx) for k in kids])
        if t == "ARRAY_EXTRACT":
            a, i = (self.expr(k, sc, ctx) for k in kids)
            if a.t.id == "JSON":                         # j[i] on JSON (ISO 2023 simplified accessor)
                if isinstance(i, I.Lit) and isinstance(i.v, int): return _json_accessor(a, f"[{i.v}]")
                if isinstance(i, I.Lit) and isinstance(i.v, str): return _json_accessor(a, "." + json.dumps(i.v))
                raise SqlError("a JSON subscript must be a constant")
            if a.t.id in ("LIST", "MULTISET"): return self.func("__iso_array_elem", [a, i])     # ISO: an index past the end is an error
            return self.func("list_extract", [a, i])
        if t == "ARRAY_CONSTRUCTOR": return self.func("list_value", [self.expr(k, sc, ctx) for k in kids])
        if t == "ARRAY_SLICE": return self.func("list_slice", [self.expr(k, sc, ctx) for k in kids])
        if t == "GROUPING_FUNCTION":
            if not ctx.agg: raise SqlError("GROUPING() is only allowed in the SELECT list, HAVING and ORDER BY of a grouped query")
            return I.Grouping([self.expr(k, sc, ctx) for k in kids])
        raise SqlError(f"operator {t} is not supported")

    def args_of(self, e, sc, ctx):
        return [self.expr(a["expression"] if "expression" in a else a, sc, ctx) for a in e.get("arguments") or []]

    def function(self, e, sc, ctx):
        name = e["function_name"].lower()
        if getattr(self, "mr_ctx", False):
            r = self.mr_function(name, e, sc, ctx)
            if r is not None: return r
        if name in F.AGGS or name in _AGG_ALIAS or name == "count_star" or name.startswith("__iso_h") or name == "__iso_listagg":
            if not (name in F.SCALARS and name not in F.AGGS and name not in _AGG_ALIAS):
                return self.aggregate(name, e, sc, ctx)
        if e.get("distinct") or e.get("filter") is not None:
            raise SqlError(f"{name} is not an aggregate function: DISTINCT / FILTER not allowed")
        raw = e.get("arguments") or []
        if name == "__iso_unique":
            q = self.bind_query(raw[0]["expression"]["subquery"]["node"], sc)
            return I.Subq("unique", q, T.BOOLEAN, correlated=_correlated(q))
        if name.startswith("__json_") and name not in ("__json_objectagg", "__json_arrayagg"):
            return self.sql_json(name, raw, sc, ctx)
        if name == "__iso_cast_format":
            args = self.args_of(e, sc, ctx)
            dt_ = T.type_from_name(args[1].v)
            src = args[0]
            if dt_.temporal:
                if src.t.id not in ("VARCHAR", "NULL") and not getattr(src, "strlit", False): raise SqlError(f"CAST ... FORMAT: a string to {dt_}, not {src.t}")
            elif dt_.id == "VARCHAR":
                if not src.t.temporal: raise SqlError(f"CAST ... FORMAT: a datetime to a string, not {src.t}")
            else: raise SqlError(f"CAST ... FORMAT: to a datetime or a character string, not {dt_}")
            return I.Func("__iso_cast_format", [src, I.Lit(str(dt_), T.VARCHAR), args[2]], dt_)
        if name in ("__iso_array_q", "__iso_ms_q"):                    # ARRAY (<query>) / MULTISET (<query>)
            q = self.bind_query(raw[0]["expression"]["subquery"]["node"], sc)
            cols = q_columns(q)
            if len(cols) != 1: raise SqlError("ARRAY / MULTISET (<query>): the query must have one column")
            t = T.LIST(cols[0][1]) if name == "__iso_array_q" else T.MULTISET(cols[0][1])
            return I.Subq("array", q, t, correlated=_correlated(q))
        if name == "array_max_cardinality":
            a = self.expr(raw[0]["expression"], sc, ctx)
            if a.t.id != "LIST": raise SqlError(f"ARRAY_MAX_CARDINALITY needs an array, not {a.t}")
            return I.Lit(a.t.p, T.BIGINT)
        if name == "__iso_sym": raise SqlError("BETWEEN SYMMETRIC misplaced")
        if name in ("__iso_ties", "__iso_pct"): return I.Func(name, self.args_of(e, sc, ctx), T.BIGINT)
        if name == "grouping" or name == "grouping_id":
            if not ctx.agg: raise SqlError("GROUPING() is only allowed in the SELECT list, HAVING and ORDER BY of a grouped query")
            return I.Grouping(self.args_of(e, sc, ctx))
        if name == "grouping_function": return I.Grouping(self.args_of(e, sc, ctx))
        if name == "normalize" and len(raw) in (2, 3) and raw[1]["expression"]["class"] == "COLUMN_REF" \
                and raw[1]["expression"]["column_names"][-1].upper() in ("NFC", "NFD", "NFKC", "NFKD"):    # (, length): a type bound
            return self.func("normalize", [self.expr(raw[0]["expression"], sc, ctx), I.Lit(raw[1]["expression"]["column_names"][-1].upper(), T.VARCHAR, True)])
        args = self.args_of(e, sc, ctx)
        if name == "typeof":
            lit = I.Lit(str(args[0].t) if args[0].t.id != "NULL" else '"NULL"', T.VARCHAR)
            return I.Func("__first", [lit, args[0]], T.VARCHAR) if I.has(args[0], I.Agg) or I.has(args[0], I.Win) else lit
        if name == "-" and len(args) == 1: return self.func("neg", args)
        if name == "+" and len(args) == 1: return args[0]
        if name == "~" and len(args) == 1: return self.func("__bitnot", args)
        if name == "~" and len(args) == 2: return self.func("regexp_matches", args)
        if name == "row": return I.Func("row", args, T.NULL)
        if name == "regexp_full_match" and len(args) == 2 and isinstance(args[1], I.Func) and args[1].fn == "__iso_similar_pat":
            return self.func("__iso_similar", [args[0]] + args[1].args)
        if name == "__iso_similar_pat": return I.Func("__iso_similar_pat", args, T.VARCHAR)
        if name in ("like_escape", "not_like_escape", "ilike_escape", "not_ilike_escape"):
            r = self.func({"like_escape": "like", "not_like_escape": "like", "ilike_escape": "ilike", "not_ilike_escape": "ilike"}[name], args)
            return self.func("not", [r]) if name.startswith("not_") else r
        if name in ("not_like", "not_ilike"):
            return self.func("not", [self.func(name[4:], args)])
        if name in ("!~~", "!~~*"): return self.func("not", [self.func("like" if name == "!~~" else "ilike", args)])
        if name == "!~": return self.func("not", [self.func("regexp_matches", args)])
        if name in ("date_part", "datepart", "extract") and args and isinstance(args[0], I.Lit):
            return self.func("date_part", args)
        if name == "__iso_nth_value_last": raise SqlError("NTH_VALUE ... FROM LAST is only a window function")
        name = _FN_ALIAS.get(name, name)
        if name.upper() in ("CURRENT_TIME", "CURRENT_TIMESTAMP", "LOCALTIME", "LOCALTIMESTAMP") and len(args) == 1:
            p = _const(args[0], f"the precision of {name.upper()}")      # CURRENT_TIMESTAMP(p): p fractional digits
            if not isinstance(p, int) or not 0 <= p <= 6: raise SqlError(f"invalid precision {p} for {name.upper()}")
            return self.func("__frac_trunc", [self.func(name, []), I.Lit(p, T.INTEGER)])
        if name not in F.SCALARS: raise SqlError(f"Scalar Function with name {e['function_name']} does not exist")
        return self.func(name, args)

    def aggregate(self, name, e, sc, ctx):
        if not ctx.agg: raise SqlError(f"aggregate functions are not allowed in {ctx.where}")
        if ctx.in_agg: raise SqlError("aggregate function calls cannot be nested")
        name = _AGG_ALIAS.get(name, name)
        ictx = Ctx(where="an aggregate"); ictx.in_agg = True; ictx.agg = False
        ictx2 = Ctx(agg=False, where="an aggregate"); ictx2.in_agg = True
        raw = e.get("arguments") or []
        args = [self.expr(a["expression"] if "expression" in a else a, sc, ictx) for a in raw]
        orders = []
        for o in ((e.get("order_bys") or {}).get("orders") or []):
            desc = o["type"] == "DESCENDING"
            nf = {"NULLS_FIRST": True, "NULLS FIRST": True}.get(o.get("null_order"), False)
            orders.append((self.expr(o["expression"], sc, ictx), desc, nf))
        filt = None
        if e.get("filter") is not None:
            filt = self.boolean(self.expr(e["filter"], sc, ictx), "FILTER")
        distinct = bool(e.get("distinct"))
        params = ()
        if name == "count_star": args = []
        if name in ("quantile_cont", "quantile_disc", "median", "mode") and orders and (name != "median"):
            # PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY x [DESC]) / MODE() WITHIN GROUP (ORDER BY x)
            if name != "mode":
                q = _const(args[0], name) if args else None
                if orders[0][1] and name == "quantile_cont":
                    q = [1 - Decimal(str(x)) for x in q] if isinstance(q, list) else 1 - Decimal(str(q))
                params = (q, "desc") if orders[0][1] and name == "quantile_disc" else (q,)
            else: params = ("desc" if orders[0][1] else "asc",)
            args = [orders[0][0]]; orders = []
        elif name in ("quantile_cont", "quantile_disc") and len(args) == 2:
            params = (_const(args[1], name),); args = args[:1]
        elif name == "string_agg":
            if len(args) == 2: params = (_const(args[1], "string_agg separator"),); args = args[:1]
            else: params = (",",)
        elif name == "__iso_listagg":
            params = tuple(_const(a, "LISTAGG") for a in args[1:]); args = args[:1]
        elif name in ("__json_objectagg", "__json_arrayagg"):
            nv = 2 if name == "__json_objectagg" else 1
            o = json.loads(_const(args[nv], name)); vals = args[:nv]
            if name == "__json_objectagg": vals[0] = cast_to(vals[0], T.VARCHAR)
            o["vtypes"] = [str(vals[-1].t)]
            params = (json.dumps(o),); args = vals
        elif name.startswith("__iso_h"):
            params = tuple(_const(a, name[7:]) for a in args)
            if len(params) != len(orders): raise SqlError("hypothetical set function: one argument per ORDER BY column")
            args = [o[0] for o in orders]
        if name in ("min", "max") and len(args) == 1 and T.coll_key(args[0].t) is not None:
            # MIN / MAX under a collation: the value of least / greatest key
            args = [args[0], self.func("__ckey", [args[0], I.Lit(args[0].t.coll, T.VARCHAR)])]
            name = "arg_min" if name == "min" else "arg_max"
        a = F.AGGS.get(name)
        if a is None: raise SqlError(f"unknown aggregate function {name}")
        if name in ("string_agg", "__iso_listagg") or a.name == "string_agg":
            args = [cast_to(x, T.VARCHAR) for x in args]
        if len(args) != a.nargs and not (name == "count" and len(args) == 0) and not name.startswith("__iso_h"):
            if not (name in ("array_agg", "string_agg") and len(args) == 1):
                raise SqlError(f"{name} takes {a.nargs} argument(s), got {len(args)}")
        if name == "count" and not args: name = "count_star"
        ts = [x.t for x in args]
        rt = F.agg_typer(name, ts, params) if name != "count_star" else T.BIGINT
        return I.Agg(name, args, rt, distinct, filt, orders, params)

    def window(self, e, sc, ctx):
        if not ctx.win: raise SqlError(f"window functions are not allowed in {ctx.where}")
        name = e.get("function_name", "").lower()
        kind = _WIN_KIND.get(e["type"])
        ictx = Ctx(agg=ctx.agg, aliases=ctx.aliases, where="a window")
        args = [self.expr(a["expression"] if "expression" in a else a, sc, ictx) for a in (e.get("arguments") or e.get("children") or [])]
        if e.get("offset_expr") is not None and len(args) < 2: args.append(self.expr(e["offset_expr"], sc, ictx))
        if e.get("default_expr") is not None and len(args) < 3:
            if len(args) < 2: args.append(I.Lit(1, T.BIGINT))
            args.append(self.expr(e["default_expr"], sc, ictx))
        part = [self.expr(p, sc, ictx) for p in e.get("partitions") or []]
        order = []
        for o in e.get("orders") or []:
            nf = {"NULLS_FIRST": True, "NULLS FIRST": True}.get(o.get("null_order"), False)
            order.append((self.expr(o["expression"], sc, ictx), o["type"] == "DESCENDING", nf))
        frame = _frame(e, lambda x: self.expr(x, sc, ictx))
        filt = self.boolean(self.expr(e["filter_expr"], sc, ictx), "FILTER") if e.get("filter_expr") is not None else None
        excl = e.get("exclude_clause") or None
        if excl in ("NO_OTHER", "NO OTHERS", "NONE"): excl = None
        ign = bool(e.get("ignore_nulls"))
        from_last = False
        if name == "__iso_nth_value_last": name, kind, from_last = "nth_value", "nth_value", True
        if (kind is None or e["type"] == "WINDOW_AGGREGATE") and not from_last:
            aname = _AGG_ALIAS.get(name, name)
            if aname not in F.AGGS: raise SqlError(f"unknown window function {name}")
            params = ()
            if aname in ("quantile_cont", "quantile_disc") and len(args) == 2: params = (_const(args[1], name),); args = args[:1]
            if aname == "string_agg":
                params = (_const(args[1], "separator"),) if len(args) > 1 else (",",); args = [cast_to(args[0], T.VARCHAR)]
            if aname in ("__json_objectagg", "__json_arrayagg"):
                nv = 2 if aname == "__json_objectagg" else 1
                o = json.loads(_const(args[nv], aname)); o["vtypes"] = [str(args[nv - 1].t)]
                params = (json.dumps(o),); args = args[:nv]
            if aname == "count" and not args: aname = "count_star"
            rt = F.agg_typer(aname, [a.t for a in args], params) if aname != "count_star" else T.BIGINT
            return I.Win(aname, args, rt, part, order, frame, excl, ign, filt, False, bool(e.get("distinct")), params)
        rt = F.win_typer(kind, args)
        if kind in ("lag", "lead") and len(args) > 2: args[2] = cast_to(args[2], rt)
        return I.Win(kind, args, rt, part, order, frame, excl, ign, filt, from_last)

    def subquery(self, e, sc, ctx):
        kind = e["subquery_type"]
        q = self.bind_query(e["subquery"]["node"], sc)
        corr = _correlated(q)
        cols = q_columns(q)
        if kind == "SCALAR":
            if len(cols) != 1: raise SqlError("Subquery returns more than 1 column")
            return I.Subq("scalar", q, cols[0][1], correlated=corr)
        if kind == "EXISTS": return I.Subq("exists", q, T.BOOLEAN, correlated=corr)
        if kind == "NOT_EXISTS": return self.func("not", [I.Subq("exists", q, T.BOOLEAN, correlated=corr)])
        if kind == "ANY":
            child = self.expr(e["child"], sc, ctx)
            op = _CMP.get(e.get("comparison_type"), "=")
            if len(cols) != 1 and not _is_row(child): raise SqlError("Subquery returns more than 1 column")
            if not _is_row(child):
                t = T.supertype(child.t, cols[0][1], getattr(child, "strlit", False))
                child = cast_to(child, t)
            return I.Subq("any", q, T.BOOLEAN, child, op, correlated=corr)
        raise SqlError(f"subquery {kind} is not supported")


def _frame(e, bind):
    s, en = e.get("start", "UNBOUNDED_PRECEDING"), e.get("end", "CURRENT_ROW_RANGE")
    def kind(k):
        if k == "UNBOUNDED_PRECEDING": return ("unbounded_preceding", None)
        if k == "UNBOUNDED_FOLLOWING": return ("unbounded_following", None)
        if k.startswith("CURRENT_ROW"): return ("current", None)
        if k.startswith("EXPR_PRECEDING"): return ("preceding", None)
        if k.startswith("EXPR_FOLLOWING"): return ("following", None)
        raise SqlError(f"frame bound {k} is not supported")
    mode = "range"
    for k in (s, en):
        if k.endswith("_ROWS"): mode = "rows"
        elif k.endswith("_GROUPS"): mode = "groups"
    a, b = kind(s), kind(en)
    if a[0] in ("preceding", "following") and e.get("start_expr") is not None: a = (a[0], bind(e["start_expr"]))
    if b[0] in ("preceding", "following") and e.get("end_expr") is not None: b = (b[0], bind(e["end_expr"]))
    return (mode, a, b)


def _rowf(args): return I.Func("row", list(args), T.SqlType("ROW"))


def _search_cycle_base(base, names, cols, sc, binder):
    """the base of a recursive CTE with SEARCH / CYCLE: its rows with the sequence column (DEPTH: the path of the
    search columns; BREADTH: depth 0 and the search columns), the cycle mark (not a cycle) and the cycle path"""
    lower = [n.lower() for n in names]
    def pos(c, what):
        if c.lower() not in lower: raise SqlError(f"{what} column {c} is not a column of the query")
        return lower.index(c.lower())
    bb = I.Binding("__base", list(cols), q=base)
    s = I.Select(); s.from_ = bb
    vals = [I.Col(bb, n, t, True) for n, t, _ in cols]
    s.items = [(v, n) for v, (n, _, _) in zip(vals, cols)]
    extra = []
    if "search" in sc:
        p = sc["search"]; sv = [vals[pos(c, "SEARCH")] for c in p["cols"]]
        e = make_func("list_value", [_rowf(sv)]) if p["kind"] == "DEPTH" else make_func("list_value", [I.Lit(0, T.BIGINT)] + sv) \
            if False else (I.Func("list_value", [_rowf(sv)], T.LIST(T.SqlType("ROW"))) if p["kind"] == "DEPTH" else _rowf([I.Lit(0, T.BIGINT)] + sv))
        extra.append((e, p["set"]))
    if "cycle" in sc:
        p = sc["cycle"]; cv = [vals[pos(c, "CYCLE")] for c in p["cols"]]
        v1 = binder.expr(_parse_expr(p["v1"]), Scope([]), Ctx()); v0 = binder.expr(_parse_expr(p["v0"]), Scope([]), Ctx())
        mt = T.supertype(v1.t, v0.t, getattr(v1, "strlit", False), getattr(v0, "strlit", False))
        p["v1x"], p["v0x"], p["mt"] = cast_to(v1, mt), cast_to(v0, mt), mt
        extra.append((p["v0x"], p["mark"]))
        extra.append((I.Func("list_value", [_rowf(cv)], T.LIST(T.SqlType("ROW"))), p["path"]))
    s.items += extra
    return s, list(cols) + [(n, x.t, True) for x, n in extra]


def _search_cycle_step(rec, names, sc, binder):
    """the recursive part with SEARCH / CYCLE: the new rows extend their parent's sequence value and cycle path; a
    row whose cycle columns are already on its path is marked and not expanded"""
    step = rec.step
    if not isinstance(step, I.Select): raise SqlError("SEARCH / CYCLE: the recursive part must be one query specification")
    import sqlexec
    refs = [b for b in sqlexec.block_bindings(step.from_) if getattr(b, "rec", None) is rec]
    if len(refs) != 1: raise SqlError("SEARCH / CYCLE: the recursive part must read the CTE exactly once")
    rb = refs[0]
    col = lambda n: next(I.Col(rb, c, t, nl) for c, t, nl in rb.cols if c == n)
    lower = [n.lower() for n in names]
    items = list(step.items[:len(names)])
    new = lambda c: items[lower.index(c.lower())][0]
    extra = []
    if "search" in sc:
        p = sc["search"]; sv = [new(c) for c in p["cols"]]
        if p["kind"] == "DEPTH":
            e = I.Func("__list_concat", [col(p["set"]), I.Func("list_value", [_rowf(sv)], T.LIST(T.SqlType("ROW")))], T.LIST(T.SqlType("ROW")))
        else:
            depth = make_func("+", [I.Func("__iso_array_elem", [col(p["set"]), I.Lit(1, T.BIGINT)], T.BIGINT), I.Lit(1, T.BIGINT)])
            e = _rowf([depth] + sv)
        extra.append((e, p["set"]))
    if "cycle" in sc:
        p = sc["cycle"]; cv = _rowf([new(c) for c in p["cols"]])
        seen = I.Func("__ms_member", [cv, col(p["path"])], T.BOOLEAN)
        extra.append((I.Case([(seen, p["v1x"])], p["v0x"], p["mt"]), p["mark"]))
        extra.append((I.Func("__list_concat", [col(p["path"]), I.Func("list_value", [cv], T.LIST(T.SqlType("ROW")))], T.LIST(T.SqlType("ROW"))), p["path"]))
        stop = make_func("<>", [col(p["mark"]), p["v1x"]])                 # marked rows are not expanded
        step.where = make_and([step.where, stop])
    step.items = items + extra
    return step


def _parse_expr(text):
    """the parse tree of one value expression"""
    nodes, _ = parse("SELECT " + text)
    return nodes[0]["select_list"][0]


def _json_accessor(base, steps):
    """JSON_QUERY(base, 'lax $<steps>' WITH CONDITIONAL ARRAY WRAPPER NULL ON EMPTY NULL ON ERROR)"""
    if isinstance(base, I.Func) and base.fn == "__json_query":          # j.a.b: one path
        o = json.loads(base.args[1].v)
        if o.get("accessor"):
            o["path"] += steps
            return I.Func("__json_query", [base.args[0], I.Lit(json.dumps(o), T.VARCHAR)], T.JSON)
    o = dict(path="lax $" + steps, names=[], ptypes=[], np=0, empty=["NULL"], error=["NULL"], wrapper="CONDITIONAL", quotes="KEEP",
             returning=None, accessor=True)
    return I.Func("__json_query", [base, I.Lit(json.dumps(o), T.VARCHAR)], T.JSON)


def _sample_spec(sm):
    """DuckDB's parse of TABLESAMPLE / USING SAMPLE -> I.Sample"""
    if not sm: return None
    size = sm.get("sample_size") or {}
    v = size.get("value")
    if size.get("is_null") or v is None: raise SqlError("TABLESAMPLE needs a sample size")
    method = (sm.get("method") or "system").lower()
    seed = sm.get("seed") if sm.get("repeatable") or (sm.get("seed") is not None and sm.get("seed") >= 0) else None
    if sm.get("is_percentage"):
        pct = Decimal(str(v))
        if not 0 <= pct <= 100: raise SqlError(f"TABLESAMPLE percentage {v} is outside 0 .. 100")
        return I.Sample("system" if method == "system" else "bernoulli" if method == "bernoulli" else method, pct=pct, seed=seed)
    n = int(v)
    if n < 0: raise SqlError("TABLESAMPLE: a negative number of rows")
    return I.Sample("reservoir", rows=n, seed=seed)


def _count_param(e, where):
    """a row count of FETCH FIRST / OFFSET: a constant, or an expression without columns (a scalar subquery, a
    parameter-free expression: ISO F860 / F865), evaluated when the query runs"""
    try: return _const(e, where)
    except SqlError:
        if I.cols_of(e) or I.has(e, I.Outer) or I.has(e, I.Agg) or I.has(e, I.Win):
            raise SqlError(f"{where} must not reference columns")
        return ("expr", e)


def _const(e, where):
    if isinstance(e, I.Lit): return e.v
    if isinstance(e, I.Func) and e.fn == "list_value" and all(isinstance(a, I.Lit) for a in e.args): return [a.v for a in e.args]
    if isinstance(e, I.Func) and e.fn == "neg" and isinstance(e.args[0], I.Lit): return -e.args[0].v
    raise SqlError(f"{where} must be a constant")


def _is_row(x): return isinstance(x, I.Func) and x.fn == "row"


def _is_plain_col(x, n): return isinstance(x, I.Col) and x.name.lower() == n.lower()


def _rename(b, names):
    """a table under AS x(a, b, ...): its first columns shown (and referenced) by the new names; a stored table keeps
    its column names underneath (the planes are read by them)"""
    if len(names) > len(b.cols): raise SqlError(f"table {b.alias} has {len(b.cols)} columns, {len(names)} names given")
    if b.table is not None:
        nb = I.Binding(b.alias, b.cols, table=b.table, q=b.q)
        nb.display = {c: names[i] for i, (c, _, _) in enumerate(b.cols) if i < len(names)}
        return nb
    cols = [(names[i] if i < len(names) else c, t, nl) for i, (c, t, nl) in enumerate(b.cols)]
    return I.Binding(b.alias, cols, table=b.table, q=b.q)


def _visible_cols(t):
    order = getattr(t, "col_order", None)
    cs = [c for c in t.cols if not c.startswith("_") and "#" not in c]
    if order: cs = [c for c in order if c in cs] + [c for c in cs if c not in order]
    return cs


def q_columns(q):
    """[(name, type)] of a bound query's output"""
    if isinstance(q, I.Select): return [(n, x.t) for x, n in q.items]
    if isinstance(q, I.SetOp): return q.items
    if isinstance(q, I.Values): return list(zip(q.names, q.types))
    if isinstance(q, I.Recursive): return [(c, t) for c, t, _ in q.binding.cols]
    if isinstance(q, I.TableFn): return list(zip(q.names, q.types))
    raise SqlError(f"not a query: {q!r}")


def _project_by_name(q, names):
    cols = q_columns(q)
    lower = {n.lower(): i for i, (n, _) in enumerate(cols)}
    b = I.Binding("__byname", [(n, t, True) for n, t in cols], q=q)
    s = I.Select(); s.from_ = b
    s.items = [((b.col(cols[lower[n.lower()]][0]) if n.lower() in lower else I.Lit(None, T.NULL)), n) for n in names]
    return s


def _correlated(q):
    """does a bound query read columns of an enclosing query?"""
    found = []
    def visit_expr(x):
        if isinstance(x, I.Outer): found.append(x)
        if isinstance(x, I.Subq) and x.correlated: found.append(x)
    def visit_q(q):
        if isinstance(q, I.Select):
            for x in _block_exprs(q): I.walk(x, visit_expr)
            fr = [q.from_]
            while fr:
                f = fr.pop()
                if isinstance(f, I.Join):
                    fr += [f.left, f.right]
                    if f.cond is not None: I.walk(f.cond, visit_expr)
                elif isinstance(f, I.Binding) and f.q is not None and not hasattr(f, "shared"): visit_q(f.q)
        elif isinstance(q, I.SetOp): visit_q(q.left); visit_q(q.right)
        elif isinstance(q, I.TableFn):
            for a in q.args: I.walk(a, visit_expr)
            if getattr(q, 'input', None) is not None: visit_q(q.input)
    visit_q(q)
    return bool(found)


def _block_exprs(s):
    out = [x for x, _ in s.items] + list(s.group)
    for x in (s.where, s.having, s.qualify):
        if x is not None: out.append(x)
    for o in s.order:
        if o[0][0] == "expr": out.append(o[0][1])
    return out


def _functional_dependencies(s):
    """ISO T301: a column functionally dependent on the GROUP BY columns may be used outside aggregates. Known
    dependencies: a table's columns on its primary key, and equal columns (= conjuncts of WHERE and of inner-join ON
    conditions) on each other. The dependent columns join the GROUP BY list (which changes no group)."""
    if s.sets is not None or not s.group: return
    arrows = []                                       # (x, y): x determines y
    def coleq(c):
        return isinstance(c, I.Func) and c.fn == "=" and len(c.args) == 2 and all(isinstance(a, I.Col) for a in c.args)
    def sides(f):
        out = set()
        def go(x):
            if isinstance(x, I.Join): go(x.left); go(x.right)
            elif isinstance(x, I.Binding): out.add(x)
        go(f); return out
    def conds(f):
        if isinstance(f, I.Join):
            conds(f.left); conds(f.right)
            for c in conjuncts(f.cond):
                if not coleq(c): continue
                a, b = c.args
                if f.kind in ("inner", "cross"): arrows.extend([(a, b), (b, a)])
                elif f.kind in ("left", "right"):     # the preserved side determines the other (NULL when unmatched)
                    other = sides(f.right if f.kind == "left" else f.left)
                    if a.b not in other and b.b in other: arrows.append((a, b))
                    elif b.b not in other and a.b in other: arrows.append((b, a))
    conds(s.from_)
    for c in conjuncts(s.where):
        if coleq(c): arrows.extend([(c.args[0], c.args[1]), (c.args[1], c.args[0])])
    have = {g.key(): g for g in s.group if isinstance(g, I.Col)}
    grew = True
    while grew:
        grew = False
        for x, y in arrows:
            if x.key() in have and y.key() not in have: have[y.key()] = y; grew = True
    keyed = {c.b for c in have.values() if c.b.table is not None and getattr(c.b.table, "pk", None) == c.name}
    if not keyed: return
    gk = {g.key() for g in s.group}
    exprs = [x for x, _ in s.items] + [s.having, s.qualify] + [o[0][1] for o in s.order if o[0][0] == "expr"]
    for x in exprs:
        if x is None: continue
        for c in I.find(x, I.Col, stop=(I.Agg,)):
            if c.key() not in gk and (c.b in keyed or c.key() in have):
                s.group.append(c); gk.add(c.key())


def _check_grouped(s):
    """every column outside an aggregate must be a GROUP BY expression (or inside one)"""
    gk = {g.key() for g in s.group}
    def ok(x):
        if x.key() in gk: return True
        if isinstance(x, (I.Agg, I.Grouping)): return True
        if isinstance(x, I.Col): return False
        if isinstance(x, I.Win):
            return all(ok(c) for c in x.children())
        if isinstance(x, I.Subq): return x.child is None or ok(x.child)
        return all(ok(c) for c in x.children())
    for x, n in s.items:
        if not ok(x): raise SqlError(f'column "{_first_col(x, gk)}" must appear in the GROUP BY clause or must be part of an aggregate function')
    for x in (s.having, s.qualify):
        if x is not None and not ok(x): raise SqlError(f'column "{_first_col(x, gk)}" must appear in the GROUP BY clause or be used in an aggregate function')
    for o in s.order:
        if o[0][0] == "expr" and not ok(o[0][1]):
            raise SqlError(f'ORDER BY column "{_first_col(o[0][1], gk)}" must appear in the GROUP BY clause or be used in an aggregate function')


def _first_col(x, gk):
    for c in I.find(x, I.Col, stop=(I.Agg,)):
        if c.key() not in gk: return c.name
    return "?"


class SetSession:
    """SET TIME ZONE ... (ISO 19.4) / SET TimeZone = ... : changes the SQL-session time zone of the database"""
    def __init__(self, zone): self.zone = zone          # a tzinfo, or None: the default (LOCAL)


def split_statements(sql):
    """the statements of a script (top-level semicolons; strings, identifiers and comments respected)"""
    out, start = [], 0
    for m in _TOK.finditer(sql):
        if m.lastgroup == "op" and m.group(0) == ";":
            out.append(sql[start:m.start()]); start = m.end()
    out.append(sql[start:])
    return [x for x in out if tokenize(x)]


def _set_statement(db, text):
    """SET TIME ZONE LOCAL | <interval or zone name>, SET TimeZone = 'zone' / SET timezone TO 'zone' (None: not a SET)"""
    t = tokenize(text)
    if not t or t[0].up() != "SET": return None
    if _kw(t, 1, "TIME", "ZONE"):
        rest = t[3:]
        if len(rest) == 1 and rest[0].up() == "LOCAL": return SetSession(None)
        expr = _text(rest, 0, len(rest))
    elif len(t) >= 3 and t[1].up() in ("TIMEZONE", "TIME_ZONE") and (t[2].s == "=" or t[2].up() == "TO"):
        expr = _text(t, 3, len(t))
        if len(t) == 4 and t[3].up() in ("LOCAL", "DEFAULT"): return SetSession(None)
    else:
        raise SqlError(f"SET {t[1].s if len(t) > 1 else ''}: only SET TIME ZONE is supported")
    q = bind(db, "SELECT " + expr)[0]
    import sqlexec
    v = sqlexec.run(db, q).rows[0][0]
    if isinstance(v, __import__("datetime").timedelta): v = T.Interval(0, 0, v // __import__("datetime").timedelta(microseconds=1))
    return SetSession(T.zone_of(v))


def bind(db, sql):
    """SQL text -> [bound statement]"""
    T.SESSION["zone"] = getattr(db, "_sql_zone", None)
    out = []
    for text in split_statements(sql):
        st = _set_statement(db, text)
        if st is not None:
            out.append(st); db._sql_zone = st.zone; T.SESSION["zone"] = st.zone; continue
        nodes, ptext = parse(text)
        flags = dict(FLAGS)
        for n in nodes:
            b = Binder(db, ptext, flags)
            out.append(b.bind_statement(n))
    return out
