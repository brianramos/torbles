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
    (?P<str>[eE]?'(?:[^']|'')*') | (?P<qid>"(?:[^"]|"")*") |
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?) |
    (?P<id>[A-Za-z_][A-Za-z_0-9$]*) |
    (?P<op>::|<=|>=|<>|!=|\|\||->>|->|\*\*|//|<<|>>|!~~\*|!~~|~~\*|~~~|~~|[-+*/%^=<>!~&|(),.;\[\]{}:?#@$])
""", re.X | re.S)


class Tok:
    __slots__ = ("k", "s")
    def __init__(self, k, s): self.k, self.s = k, s
    def up(self): return self.s.upper() if self.k == "id" else None
    def __repr__(self): return self.s


def tokenize(sql):
    out, i = [], 0
    while i < len(sql):
        m = _TOK.match(sql, i)
        if not m: raise SqlError(f"syntax error at {sql[i:i + 20]!r}")
        k = m.lastgroup
        if k not in ("ws", "lc", "bc"): out.append(Tok(k, m.group(0)))
        i = m.end()
    return out


def _match_paren(t, i):
    """index of the ')' matching the '(' at i"""
    d = 0
    for j in range(i, len(t)):
        if t[j].s == "(": d += 1
        elif t[j].s == ")":
            d -= 1
            if d == 0: return j
    raise SqlError("unbalanced parentheses")


def _match_open(t, j):
    d = 0
    for i in range(j, -1, -1):
        if t[i].s == ")": d += 1
        elif t[i].s == "(":
            d -= 1
            if d == 0: return i
    raise SqlError("unbalanced parentheses")


def _operand_back(t, j):
    """start index of the primary expression ending at token j"""
    i = j
    if t[i].s == ")":
        i = _match_open(t, i)
        if i > 0 and t[i - 1].k in ("id", "qid") and t[i - 1].up() not in _KW_BREAK: i -= 1
    while i >= 2 and t[i - 1].s == "." and t[i - 2].k in ("id", "qid"): i -= 2
    while i >= 2 and t[i - 1].s == "::": i -= 2
    return i


def _operand_fwd(t, i):
    """end index (exclusive) of the primary expression starting at token i"""
    j = i
    if t[j].s == "(": j = _match_paren(t, j) + 1
    elif t[j].k in ("id", "qid"):
        j += 1
        while j + 1 < len(t) and t[j].s == "." : j += 2
        if j < len(t) and t[j].s == "(": j = _match_paren(t, j) + 1
    elif t[j].s in ("-", "+"): return _operand_fwd(t, j + 1)
    else: j += 1
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
                j += 2; continue                         # USING CHARACTERS | OCTETS: characters either way
            j += 1; continue
        cur.append(j); j += 1
    parts[key] = _text(t, cur[0], cur[-1] + 1) if cur else ""
    occ = parts.get("OCCURRENCE", "")
    if occ.upper() == "ALL": occ = "0"
    def a_(k, default): return parts.get(k) or default
    if name == "TRANSLATE_REGEX":
        return f"__iso_translate_regex({parts['PAT']}, {parts['IN']}, {a_('WITH', chr(39) * 2)}, {a_('FLAG', chr(39) * 2)}, {occ or '0'})"
    if name == "OCCURRENCES_REGEX":
        return f"__iso_occurrences_regex({parts['PAT']}, {parts['IN']}, {a_('FLAG', chr(39) * 2)}, {a_('FROM', '1')})"
    if name == "POSITION_REGEX":
        return f"__iso_position_regex({parts['PAT']}, {parts['IN']}, {a_('FLAG', chr(39) * 2)}, {a_('FROM', '1')}, {occ or '1'})"
    return f"__iso_substring_regex({parts['PAT']}, {parts['IN']}, {a_('FLAG', chr(39) * 2)}, {a_('FROM', '1')}, {occ or '1'})"


def iso_prepass(sql):
    """rewrite ISO syntax that DuckDB's grammar does not accept into calls of marker functions"""
    t = tokenize(sql)
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
            # UNION / EXCEPT / INTERSECT [ALL] CORRESPONDING -> BY NAME
            if u == "CORRESPONDING":
                t[i:i + 1] = [Tok("id", "BY"), Tok("id", "NAME")]; changed = True; break
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
             "localtimestamp": "current_timestamp", "localtime": "current_time", "today": "current_date", "pow": "power",
             "truncate": "trunc"}
_AGG_ALIAS = {"mean": "avg", "variance": "var_samp", "stddev": "stddev_samp", "string_agg": "string_agg", "group_concat": "string_agg",
              "listagg": "string_agg", "list": "array_agg", "arbitrary": "any_value", "bool_and": "every", "any": "bool_or",
              "some": "bool_or", "min_by": "arg_min", "max_by": "arg_max", "argmin": "arg_min", "argmax": "arg_max",
              "percentile_cont": "quantile_cont", "percentile_disc": "quantile_disc", "quantile": "quantile_disc",
              "countif": "count_if", "fsum": "sum", "sumkahan": "sum", "kahan_sum": "sum", "first": "any_value"}
_WIN_KIND = {"WINDOW_ROW_NUMBER": "row_number", "WINDOW_RANK": "rank", "WINDOW_RANK_DENSE": "dense_rank",
             "WINDOW_PERCENT_RANK": "percent_rank", "WINDOW_CUME_DIST": "cume_dist", "WINDOW_NTILE": "ntile", "WINDOW_LAG": "lag",
             "WINDOW_LEAD": "lead", "WINDOW_FIRST_VALUE": "first_value", "WINDOW_LAST_VALUE": "last_value", "WINDOW_NTH_VALUE": "nth_value"}


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


def make_func(name, args, extra=None):
    """a typed scalar call: the registry's typer gives the result type and the argument conversions"""
    s = F.SCALARS.get(name)
    if s is None: raise SqlError(f"Scalar Function with name {name} does not exist")
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


def _find_col(b, name):
    for c, t, nl in b.cols:
        if c == name: return I.Col(b, c, t, nl)
    for c, t, nl in b.cols:
        if c.lower() == name.lower(): return I.Col(b, c, t, nl)
    return None


class Ctx:
    """what an expression may contain where it is bound"""
    def __init__(self, agg=False, win=False, aliases=None, where="expression"):
        self.agg, self.win, self.aliases, self.where = agg, win, aliases or {}, where
        self.in_agg = False


class Binder:
    def __init__(self, db, text=""):
        self.db, self.text = db, text
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
        names = cte["aliases"] or [n for n, _ in q_columns(q)]
        cols = [(n, t, True) for n, (_, t) in zip(names, q_columns(q))]
        b = I.Binding(alias, cols, q=q)
        b.shared = cte                     # materialized once per query
        return b

    def bind_cte(self, cte):
        node = cte["node"]
        if node["type"] == "RECURSIVE_CTE_NODE":
            base = self.bind_query(node["left"], None)
            names = cte["aliases"] or [n for n, _ in q_columns(base)]
            b = I.Binding(cte["name"], [(n, t, True) for n, (_, t) in zip(names, q_columns(base))])
            rec = I.Recursive(b, base, None, bool(node.get("union_all")))
            b.rec = rec; b.q = rec
            cte["recursive_binding"] = b
            rec.step = self.bind_query(node["right"], None)
            st = q_columns(rec.step)
            if len(st) != len(names): raise SqlError("the recursive part of a CTE must have as many columns as its base")
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
            out.append(b); return b
        if typ == "SUBQUERY":
            q = self.bind_query(ft["subquery"]["node"], parent)
            names = list(ft.get("column_name_alias") or []) + [n for n, _ in q_columns(q)][len(ft.get("column_name_alias") or []):]
            b = I.Binding(ft.get("alias") or f"subquery{len(out)}", [(n, t, True) for n, (_, t) in zip(names, q_columns(q))], q=q)
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
            if ft.get("ref_type") == "DEPENDENT" or ft.get("is_lateral"):
                raise SqlError("LATERAL joins are not supported")
            right = self.bind_from(ft["right"], parent, out)
            kind = {"INNER": "inner", "LEFT": "left", "RIGHT": "right", "OUTER": "full", "FULL": "full",
                    "SEMI": "semi", "ANTI": "anti"}.get(ft.get("join_type", "INNER"), None)
            if kind is None: raise SqlError(f"join type {ft.get('join_type')} is not supported")
            ref = ft.get("ref_type", "REGULAR")
            if ref == "CROSS": kind = "cross" if kind == "inner" else kind
            if ref not in ("REGULAR", "CROSS", "NATURAL"): raise SqlError(f"{ref} joins are not supported")
            lb, rb = out[:nleft], out[nleft:]
            using = list(ft.get("using_columns") or [])
            if ref == "NATURAL":
                lnames = {c.lower() for b in lb for c, _, _ in b.cols}
                using = [c for b in rb for c, _, _ in b.cols if c.lower() in lnames]
                if not using: kind = "cross" if kind == "inner" else kind
            j = I.Join(kind, left, right)
            sc = Scope(out, parent)
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
        args = [self.expr(a["expression"], Scope([], parent), Ctx()) for a in fn["arguments"]]
        if name in ("range", "generate_series") and all(isinstance(a, I.Lit) for a in args):
            vals = [a.v for a in args]
            if len(vals) == 1: lo, hi, st = 0, vals[0], 1
            elif len(vals) == 2: lo, hi, st = vals[0], vals[1], 1
            else: lo, hi, st = vals
            if name == "generate_series": hi = hi + (1 if st > 0 else -1)
            rows = [[I.Lit(v, T.BIGINT)] for v in range(lo, hi, st)]
            nm = (ft.get("column_name_alias") or [name])[0]
            q = I.Values(rows, [T.BIGINT], [nm])
            b = I.Binding(ft.get("alias") or name, [(nm, T.BIGINT, False)], q=q)
            out.append(b); return b
        raise SqlError(f"table function {name} is not supported")

    # ------------------------------------------------------------ SELECT
    def bind_select(self, node, parent):
        s = I.Select()
        bindings = []
        s.from_ = self.bind_from(node.get("from_table") or {"type": "EMPTY"}, parent, bindings)
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
        for g in node.get("group_expressions") or []:
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
            s.sets = [sorted({pos[i] for i in st}) for st in sets]
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
        if s.aggregate: _check_grouped(s)
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
                    s.limit = _const(le, "LIMIT")
                if off is not None: s.offset = _const(self.expr(off, Scope([], sc), Ctx(where="OFFSET")), "OFFSET")
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
            for c, t, nl in b.cols:
                if c.lower() in excl: continue
                if not rel and (id(b), c.lower()) in using_cols: continue
                col = I.Col(b, c, t, nl)
                if not rel and c.lower() in sc.using and sc.using[c.lower()].key() != col.key() and c.lower() in seen_using: continue
                if not rel and c.lower() in sc.using:
                    seen_using.add(c.lower()); col = sc.using[c.lower()]
                out.append((col, c))
        if rel and not out and not any(_ieq(b.alias, rel) for b in sc.bindings): raise SqlError(f"Referenced table {rel!r} not found")
        if not out and sc.bindings: raise SqlError("SELECT * with no columns")
        return out

    def name_of(self, e, x):
        if isinstance(x, I.Col) and e["class"] == "COLUMN_REF": return x.name
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
            return self.cmp(_CMP[e["type"]], self.expr(e["left"], sc, ctx), self.expr(e["right"], sc, ctx))
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
        if c == "COLLATE": return self.expr(e["child"], sc, ctx)
        if c == "POSITIONAL_REFERENCE": raise SqlError("positional references (#n) are not supported")
        if c == "STAR": raise SqlError("* is only allowed in the SELECT list or COUNT(*)")
        raise SqlError(f"expression {c} is not supported")

    def type_of(self, te):
        name = te["type_name"]
        if name.lower() == "list":
            return T.LIST(self.type_of(te["children"][0]))
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
        raise SqlError(f'Referenced column "{".".join(names)}" not found in FROM clause')

    def operator(self, e, sc, ctx):
        t = e["type"]; kids = e.get("children") or []
        if t == "OPERATOR_NOT":
            return self.func("not", [self.boolean(self.expr(kids[0], sc, ctx), "NOT")])
        if t == "OPERATOR_IS_NULL": return self.func("is_null", [self.expr(kids[0], sc, ctx)])
        if t == "OPERATOR_IS_NOT_NULL": return self.func("is_not_null", [self.expr(kids[0], sc, ctx)])
        if t in ("COMPARE_IN", "COMPARE_NOT_IN"):
            x = self.expr(kids[0], sc, ctx)
            vals = [self.expr(k, sc, ctx) for k in kids[1:]]
            if _is_row(x):
                r = make_func("or", [self.row_cmp("=", x, v) for v in vals]) if len(vals) > 1 else self.row_cmp("=", x, vals[0])
            else: r = self.func("in", [x] + vals)
            return self.func("not", [r]) if t == "COMPARE_NOT_IN" else r
        if t == "OPERATOR_COALESCE": return self.func("coalesce", [self.expr(k, sc, ctx) for k in kids])
        if t == "OPERATOR_NULLIF": return self.func("nullif", [self.expr(k, sc, ctx) for k in kids])
        if t in ("ARRAY_EXTRACT",): return self.func("list_extract", [self.expr(k, sc, ctx) for k in kids])
        if t == "GROUPING_FUNCTION":
            if not ctx.agg: raise SqlError("GROUPING() is only allowed in the SELECT list, HAVING and ORDER BY of a grouped query")
            return I.Grouping([self.expr(k, sc, ctx) for k in kids])
        raise SqlError(f"operator {t} is not supported")

    def args_of(self, e, sc, ctx):
        return [self.expr(a["expression"] if "expression" in a else a, sc, ctx) for a in e.get("arguments") or []]

    def function(self, e, sc, ctx):
        name = e["function_name"].lower()
        if name in F.AGGS or name in _AGG_ALIAS or name == "count_star" or name.startswith("__iso_h") or name == "__iso_listagg":
            if not (name in F.SCALARS and name not in F.AGGS and name not in _AGG_ALIAS):
                return self.aggregate(name, e, sc, ctx)
        if e.get("distinct") or e.get("filter") is not None:
            raise SqlError(f"{name} is not an aggregate function: DISTINCT / FILTER not allowed")
        raw = e.get("arguments") or []
        if name == "__iso_unique":
            q = self.bind_query(raw[0]["expression"]["subquery"]["node"], sc)
            return I.Subq("unique", q, T.BOOLEAN, correlated=_correlated(q))
        if name == "__iso_sym": raise SqlError("BETWEEN SYMMETRIC misplaced")
        if name in ("__iso_ties", "__iso_pct"): return I.Func(name, self.args_of(e, sc, ctx), T.BIGINT)
        if name == "grouping" or name == "grouping_id":
            if not ctx.agg: raise SqlError("GROUPING() is only allowed in the SELECT list, HAVING and ORDER BY of a grouped query")
            return I.Grouping(self.args_of(e, sc, ctx))
        if name == "grouping_function": return I.Grouping(self.args_of(e, sc, ctx))
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
        elif name.startswith("__iso_h"):
            params = tuple(_const(a, name[7:]) for a in args)
            if len(params) != len(orders): raise SqlError("hypothetical set function: one argument per ORDER BY column")
            args = [o[0] for o in orders]
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
        if kind is None or e["type"] == "WINDOW_AGGREGATE":
            aname = _AGG_ALIAS.get(name, name)
            if aname not in F.AGGS: raise SqlError(f"unknown window function {name}")
            params = ()
            if aname in ("quantile_cont", "quantile_disc") and len(args) == 2: params = (_const(args[1], name),); args = args[:1]
            if aname == "string_agg":
                params = (_const(args[1], "separator"),) if len(args) > 1 else (",",); args = [cast_to(args[0], T.VARCHAR)]
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


def _const(e, where):
    if isinstance(e, I.Lit): return e.v
    if isinstance(e, I.Func) and e.fn == "list_value" and all(isinstance(a, I.Lit) for a in e.args): return [a.v for a in e.args]
    if isinstance(e, I.Func) and e.fn == "neg" and isinstance(e.args[0], I.Lit): return -e.args[0].v
    raise SqlError(f"{where} must be a constant")


def _is_row(x): return isinstance(x, I.Func) and x.fn == "row"


def _is_plain_col(x, n): return isinstance(x, I.Col) and x.name.lower() == n.lower()


def _rename(b, names):
    cols = [(names[i] if i < len(names) else c, t, nl) for i, (c, t, nl) in enumerate(b.cols)]
    nb = I.Binding(b.alias, cols, table=b.table, q=b.q)
    nb.renamed = {names[i]: c for i, (c, _, _) in enumerate(b.cols) if i < len(names)}
    return nb


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
    visit_q(q)
    return bool(found)


def _block_exprs(s):
    out = [x for x, _ in s.items] + list(s.group)
    for x in (s.where, s.having, s.qualify):
        if x is not None: out.append(x)
    for o in s.order:
        if o[0][0] == "expr": out.append(o[0][1])
    return out


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


def bind(db, sql):
    """SQL text -> [bound statement]"""
    nodes, text = parse(sql)
    out = []
    for n in nodes:
        b = Binder(db, text)
        out.append(b.bind_statement(n))
    return out
