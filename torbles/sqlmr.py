# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Row pattern recognition in FROM (ISO/IEC 9075-2:2016 7.6, feature R010): MATCH_RECOGNIZE, and in a window
(7.11, feature R020).

    FROM t MATCH_RECOGNIZE (
        PARTITION BY ... ORDER BY ...
        MEASURES <expr> AS <name>, ...
        ONE ROW PER MATCH | ALL ROWS PER MATCH [SHOW EMPTY MATCHES | OMIT EMPTY MATCHES | WITH UNMATCHED ROWS]
        AFTER MATCH SKIP PAST LAST ROW | TO NEXT ROW | TO FIRST v | TO LAST v | TO v
        PATTERN ( <regular expression over row pattern variables> )
        SUBSET u = (a, b), ...
        DEFINE v AS <condition>, ...)

The pattern is matched in the ordered rows of each partition by backtracking, in the order of preference ISO gives
(greedy quantifiers try more rows first, reluctant ones fewer, alternatives left to right, PERMUTE in the listed
order); a row is mapped to a variable only when that variable's DEFINE condition is TRUE with the row mapped. In DEFINE
and MEASURES, a column qualified by a variable is that variable's last mapped row, PREV / NEXT move by physical rows,
FIRST / LAST pick mapped rows, aggregates run over a variable's mapped rows (RUNNING up to the current row, FINAL over
the whole match), CLASSIFIER() and MATCH_NUMBER() say where a row stands. The input rows come from the engine (the
table's GROUP BY with counts, expanded to rows); the matching itself runs over those rows."""
import itertools
from sqltypes import SqlError
import sqlir as I


# ================================================================== the pattern
def parse_pattern(toks):
    """pattern tokens -> AST: ('var', name), ('seq', [...]), ('alt', [...]), ('quant', node, lo, hi, greedy),
    ('start',), ('end',), ('excl', node)"""
    p = _PP(toks)
    n = p.alt()
    if p.i != len(p.t): raise SqlError(f"row pattern: unexpected {p.t[p.i]!r}")
    return n


class _PP:
    def __init__(self, t): self.t, self.i = [x.upper() if x.isidentifier() else x for x in t], 0
    def peek(self, k=0): return self.t[self.i + k] if self.i + k < len(self.t) else None

    def alt(self):
        items = [self.seq()]
        while self.peek() == "|": self.i += 1; items.append(self.seq())
        return items[0] if len(items) == 1 else ("alt", items)

    def seq(self):
        items = []
        while self.peek() is not None and self.peek() not in ("|", ")", ",") and not (self.peek() == "-" and self.peek(1) == "}"):
            items.append(self.term())
        return items[0] if len(items) == 1 else ("seq", items)

    def term(self):
        n = self.primary()
        while True:
            x = self.peek()
            if x in ("*", "+", "?"):
                self.i += 1; lo, hi = {"*": (0, None), "+": (1, None), "?": (0, 1)}[x]
            elif x == "{" and self.peek(1) != "-":
                self.i += 1; lo, hi = 0, None
                if self.peek() not in (",", "}"): lo = int(self.peek()); self.i += 1; hi = lo
                if self.peek() == ",":
                    self.i += 1; hi = None
                    if self.peek() != "}": hi = int(self.peek()); self.i += 1
                if self.peek() != "}": raise SqlError("row pattern: a quantifier {n, m}")
                self.i += 1
            else: return n
            greedy = True
            if self.peek() == "?": self.i += 1; greedy = False
            n = ("quant", n, lo, hi, greedy)

    def primary(self):
        x = self.peek()
        if x == "(":
            self.i += 1
            if self.peek() == ")": self.i += 1; return ("seq", [])
            n = self.alt()
            if self.peek() != ")": raise SqlError("row pattern: ')' expected")
            self.i += 1; return n
        if x == "^": self.i += 1; return ("start",)
        if x == "$": self.i += 1; return ("end",)
        if x == "{" and self.peek(1) == "-":
            self.i += 2; n = self.alt()
            if not (self.peek() == "-" and self.peek(1) == "}"): raise SqlError("row pattern: an exclusion {- ... -}")
            self.i += 2; return ("excl", n)
        if x == "PERMUTE" and self.peek(1) == "(":
            self.i += 2; items = [self.alt()]
            while self.peek() == ",": self.i += 1; items.append(self.alt())
            if self.peek() != ")": raise SqlError("row pattern: PERMUTE(...)")
            self.i += 1
            return ("alt", [("seq", list(p)) for p in itertools.permutations(items)])
        if x is not None and x.replace("_", "a").isalnum() and not x[0].isdigit():
            self.i += 1; return ("var", x)
        raise SqlError(f"row pattern: unexpected {x!r}")


def pattern_vars(n, out=None):
    out = [] if out is None else out
    k = n[0]
    if k == "var":
        if n[1] not in out: out.append(n[1])
    elif k in ("seq", "alt"):
        for x in n[1]: pattern_vars(x, out)
    elif k in ("quant", "excl"): pattern_vars(n[1], out)
    return out


# ================================================================== matching
class Matcher:
    """the preferred match of the pattern from a start row: a list of (row position, variable, excluded)"""
    def __init__(self, mr, part, cx):
        self.mr, self.part, self.cx, self.n = mr, part, cx, part.n

    def match(self, start, number):
        self.number = number
        for end, assign in self._m(self.mr.pattern, start, (), False):
            return assign, end
        return None, None

    def _m(self, node, pos, assign, excl):
        k = node[0]
        if k == "var":
            if pos < self.n:
                a2 = assign + ((pos, node[1], excl),)
                if self.mr.define_ok(node[1], self.part, a2, pos, self.number, self.cx):
                    yield pos + 1, a2
            return
        if k == "seq":
            yield from self._seq(node[1], 0, pos, assign, excl)
            return
        if k == "alt":
            for x in node[1]: yield from self._m(x, pos, assign, excl)
            return
        if k == "quant":
            yield from self._quant(node, 0, pos, assign, excl)
            return
        if k == "start":
            if pos == 0: yield pos, assign
            return
        if k == "end":
            if pos == self.n: yield pos, assign
            return
        if k == "excl":
            yield from self._m(node[1], pos, assign, True)
            return
        raise SqlError(f"internal: pattern node {k}")

    def _seq(self, items, i, pos, assign, excl):
        if i == len(items): yield pos, assign; return
        for p2, a2 in self._m(items[i], pos, assign, excl):
            yield from self._seq(items, i + 1, p2, a2, excl)

    def _quant(self, node, count, pos, assign, excl):
        _, sub, lo, hi, greedy = node
        can_more = hi is None or count < hi
        if not greedy and count >= lo: yield pos, assign
        if can_more:
            for p2, a2 in self._m(sub, pos, assign, excl):
                if p2 == pos and count >= lo: continue          # an empty iteration adds nothing: stop there
                yield from self._quant(node, count + 1, p2, a2, excl)
        if greedy and count >= lo: yield pos, assign


# ================================================================== evaluation of DEFINE and MEASURES
class MR:
    """a bound MATCH_RECOGNIZE: the pattern, the variables' bindings, DEFINE conditions and MEASURES (IR)"""
    def __init__(self, pattern, univ, varb, subsets, define, measures, partition, order, rows_mode, empty_mode, skip,
                 out, input_names):
        self.pattern, self.univ, self.varb, self.subsets = pattern, univ, varb, subsets
        self.define, self.measures, self.partition, self.order = define, measures, partition, order
        self.rows_mode, self.empty_mode, self.skip = rows_mode, empty_mode, skip
        self.out, self.input_names = out, input_names            # ALL ROWS: [("in", column) | ("m", measure index)]
        self.bvar = {id(b): v for v, b in varb.items()}

    def vars_of(self, b):
        """the pattern variables a qualifier stands for: one variable, a subset's, or None for the universal one"""
        if b is self.univ: return None
        v = self.bvar[id(b)]
        return self.subsets.get(v, [v])

    def define_ok(self, var, part, assign, pos, number, cx):
        cond = self.define.get(var)
        if cond is None: return True
        v = self.eval(cond, part, assign, pos, number, cx, final=False, classifier=var)
        return v is True

    # an expression at a point of a match
    def eval(self, e, part, assign, cur, number, cx, final, classifier=None):
        import sqlexec
        rel = sqlexec.Rel(1)
        self._leaves(e, part, assign, cur, number, cx, final, classifier, rel)
        v = sqlexec.ev(e, rel, cx)[0]
        if isinstance(v, sqlexec.Err): raise SqlError(v.msg)
        return v

    def _mapped(self, assign, vars_, cur, final):
        """positions mapped to the variables (all of them for None), up to cur unless final"""
        out = []
        for p, v, _ in assign:
            if not final and p > cur: break
            if vars_ is None or v in vars_: out.append(p)
        return out

    def _row_of(self, e, assign, cur, final):
        """the row a column reference denotes: the universal variable's current row, or a variable's last mapped row"""
        vs = self.vars_of(e.b)
        if vs is None: return cur
        rows = self._mapped(assign, vs, cur, final)
        return rows[-1] if rows else None

    def _leaves(self, e, part, assign, cur, number, cx, final, classifier, rel):
        k = e.key()
        if k in rel.cols: return
        if isinstance(e, I.Col):
            r = self._row_of(e, assign, cur, final)
            rel.cols[k] = [part.value(e.name, r)]; return
        if isinstance(e, I.Func) and e.fn == "__mr_sem":
            self._leaves(e.args[0], part, assign, cur, number, cx, e.extra[0] == "final", classifier, rel)
            rel.cols[k] = rel.cols[e.args[0].key()]; return
        if isinstance(e, I.Func) and e.fn == "__mr_nav":
            r = self._nav_row(e, assign, cur, final, part)
            rel.cols[k] = [self.value_at(self._nav_arg(e), part, r, cx, assign)]; return
        if isinstance(e, I.Func) and e.fn == "__mr_classifier":     # the variable the current row is mapped to
            rel.cols[k] = [next((v for p, v, _ in assign if p == cur), None)]; return
        if isinstance(e, I.Func) and e.fn == "__mr_match_number":
            rel.cols[k] = [number]; return
        if isinstance(e, I.Agg):
            rel.cols[k] = [self._agg(e, part, assign, cur, cx, final)]; return
        for c in e.children(): self._leaves(c, part, assign, cur, number, cx, final, classifier, rel)

    def _nav_arg(self, e):
        a = e.args[0]
        while isinstance(a, I.Func) and a.fn == "__mr_nav": a = a.args[0]
        return a

    def _nav_row(self, e, assign, cur, final, part):
        """the row PREV / NEXT / FIRST / LAST (expr [, n]) reads"""
        kind = e.extra[0]; n = e.args[1].v if len(e.args) > 1 else (1 if kind in ("prev", "next") else 0)
        if n is None or n < 0: raise SqlError(f"{kind.upper()}: the offset must be a non-negative constant")
        inner = e.args[0]
        if isinstance(inner, I.Func) and inner.fn == "__mr_nav":     # PREV(LAST(A.x), 1): from the row LAST picks
            base = self._nav_row(inner, assign, cur, final, part)
        else:
            cols = list(I.find(inner, I.Col))
            bs = {id(c.b) for c in cols}
            if len(bs) > 1: raise SqlError(f"{kind.upper()}: the expression must reference one pattern variable")
            b = cols[0].b if cols else self.univ
            vs = self.vars_of(b)
            if kind in ("first", "last"):
                rows = self._mapped(assign, vs, cur, final) if vs is not None else self._mapped(assign, None, cur, final)
                if n >= len(rows): return None
                return rows[n] if kind == "first" else rows[-1 - n]
            base = cur if vs is None else (self._mapped(assign, vs, cur, final) or [None])[-1]
        if base is None: return None
        if kind == "prev": r = base - n
        elif kind == "next": r = base + n
        else: r = base
        return r if 0 <= r < part.n else None

    def value_at(self, e, part, r, cx, assign=()):
        """e with every column read at row r (and CLASSIFIER() the variable row r is mapped to)"""
        import sqlexec
        rel = sqlexec.Rel(1)
        for c in I.find(e, I.Col): rel.cols[c.key()] = [part.value(c.name, r)]
        def cls(x):
            if isinstance(x, I.Func) and x.fn == "__mr_classifier":
                rel.cols[x.key()] = [next((v for p, v, _ in assign if p == r), None)]
        I.walk(e, cls)
        v = sqlexec.ev(e, rel, cx)[0]
        if isinstance(v, sqlexec.Err): raise SqlError(v.msg)
        return v

    def _agg(self, a, part, assign, cur, cx, final):
        import sqlfuncs as F
        if a.name == "count_star" and a.params and a.params[0] == "mrvar":
            vs = self.vars_of(self.varb[a.params[1]]) if a.params[1] in self.varb else None
            return len(self._mapped(assign, vs, cur, final))
        cols = [c for x in a.args for c in I.find(x, I.Col)]
        bs = {id(c.b): c.b for c in cols}
        if len(bs) > 1: raise SqlError("an aggregate in MATCH_RECOGNIZE must reference one pattern variable")
        vs = self.vars_of(next(iter(bs.values()))) if bs else None
        rows = self._mapped(assign, vs, cur, final)
        if a.name == "count_star": return len(rows)
        d = F.AGGS[a.name]
        vals = []
        for r in rows:
            v = tuple(self.value_at(x, part, r, cx, assign) for x in a.args)
            if d.skip_nulls and any(x is None for x in v): continue
            vals.append((v, 1))
        if a.distinct:
            import sqlexec
            seen, dd = set(), []
            for v in vals:
                key = tuple(sqlexec.nkey(x) for x in v[0])
                if key not in seen: seen.add(key); dd.append(v)
            vals = dd
        return d.final(vals, a.params)


class Part:
    """one partition's rows, in order: the input columns by name"""
    def __init__(self, names, rows):
        self.names = {n: i for i, n in enumerate(names)}; self.rows = rows; self.n = len(rows)
    def value(self, name, r):
        if r is None or not 0 <= r < self.n: return None
        return self.rows[r][self.names[name]]


def run(mr, rows, cx):
    """the rows of a MATCH_RECOGNIZE over its input rows"""
    import sqlexec
    n = len(rows)
    rel = sqlexec.Rel(n)
    for i, name in enumerate(mr.input_names):
        rel.cols[I.Col(mr.univ, name, None, True).key()] = [r[i] for r in rows]
    pkeys = [sqlexec.ev(p, rel, cx) for p in mr.partition]
    okeys = [sqlexec.ev(o[0], rel, cx) for o in mr.order]
    groups = {}
    for i in range(n): groups.setdefault(tuple(sqlexec.nkey(k[i]) for k in pkeys), []).append(i)
    out = []
    for idx in groups.values():
        if mr.order: idx = sqlexec.sort_rows(idx, okeys, [(d, nf) for _, d, nf in mr.order])
        part = Part(mr.input_names, [rows[i] for i in idx])
        out += _partition(mr, part, cx)
    return out


def _partition(mr, part, cx):
    M = Matcher(mr, part, cx)
    out, start, number = [], 0, 0
    unmatched = set(range(part.n))
    while start < part.n:
        assign, end = M.match(start, number + 1)
        if assign is None:                                 # a row of an earlier match is not an unmatched row
            if mr.empty_mode == "unmatched" and start in unmatched: out.append(_row_all(mr, part, start, (), None, cx, matched=False))
            start += 1; continue
        number += 1
        if end == start:                                       # an empty match
            if mr.rows_mode == "one": out.append(_row_one(mr, part, (), start, number, cx))
            elif mr.empty_mode in ("show", "unmatched"): out.append(_row_all(mr, part, start, (), number, cx, matched=False, empty=True))
            start += 1; continue
        for p, _, _ in assign: unmatched.discard(p)
        if mr.rows_mode == "one": out.append(_row_one(mr, part, assign, end - 1, number, cx))
        else:
            for p, v, ex in assign:
                if not ex: out.append(_row_all(mr, part, p, assign, number, cx))
        start = _skip(mr, assign, start, end)
    return out


def _skip(mr, assign, start, end):
    kind, var = mr.skip
    if kind == "past": return end
    if kind == "next": return start + 1
    vs = mr.subsets.get(var, [var])
    rows = [p for p, v, _ in assign if v in vs]
    if not rows: raise SqlError(f"AFTER MATCH SKIP TO {var}: the variable has no row in the match")
    r = rows[0] if kind == "first" else rows[-1]
    if r == start: raise SqlError(f"AFTER MATCH SKIP TO {kind.upper()} {var}: the first row of the match again")
    return r


def _row_one(mr, part, assign, last, number, cx):
    """ONE ROW PER MATCH: the partition columns and the measures (FINAL)"""
    r0 = assign[0][0] if assign else last
    out = [mr.eval(p, part, assign, r0, number, cx, final=True) for p in mr.partition]
    out += [mr.eval(m, part, assign, last, number, cx, final=True) for m, _ in mr.measures]
    return tuple(out)


def _row_all(mr, part, pos, assign, number, cx, matched=True, empty=False):
    """ALL ROWS PER MATCH: the partition columns, the order columns, the measures (RUNNING), the other columns"""
    if empty: meas = [mr.eval(m, part, (), pos, number, cx, final=True) for m, _ in mr.measures]
    elif matched: meas = [mr.eval(m, part, assign, pos, number, cx, final=False) for m, _ in mr.measures]
    else: meas = [None] * len(mr.measures)
    return tuple(meas[x] if k == "m" else part.value(x, pos) for k, x in mr.out)


# ================================================================== in a window (R020)
class WindowRPR:
    """a window's row pattern recognition: the bound MR, the names of the columns it reads and their expressions
    (extra arguments of the window function), SEEK, and for <measure> OVER w the measure's index"""
    def __init__(self, spec, mr, names, inputs, seek):
        self.spec, self.mr, self.names, self.inputs, self.seek, self.measure = spec, mr, names, inputs, seek, None
    def key(self): return ("rpr", self.spec, self.measure)


def window_matches(rp, rows, bounds, cx):
    """for each row of an ordered partition: None (no match, or a row an earlier match skipped) or (first, last + 1
    of the reduced frame, the measures). The match of a row is sought from that row (INITIAL), or from it and the rows
    after it in its frame (SEEK); it can neither take rows nor read values outside the frame. The measures are
    evaluated at the last row of the match."""
    m = len(rows)
    out = [None] * m
    skip_until, number = 0, 0
    mr = rp.mr
    for p in range(m):
        if p < skip_until: continue
        hi = max(bounds[p][1], p + 1)                     # the frame: from the current row to its end
        part = Part(rp.names, rows[p:hi])
        M = Matcher(mr, part, cx)
        found = None
        for s0 in (range(part.n) if rp.seek else range(1)):
            assign, end = M.match(s0, number + 1)
            if assign is not None: found = (s0, assign, end); break
        if found is None: continue
        s0, assign, end = found
        number += 1
        if end == s0:                                          # an empty match: an empty frame, the measures of no rows
            out[p] = (p + s0, p + s0, tuple(mr.eval(x, part, (), s0, number, cx, final=True) for x, _ in mr.measures))
            skip_until = p + 1; continue
        out[p] = (p + s0, p + end, tuple(mr.eval(x, part, assign, end - 1, number, cx, final=True) for x, _ in mr.measures))
        kind, var = mr.skip
        if kind == "past": skip_until = p + end
        elif kind == "next": skip_until = p + s0 + 1
        else:
            vs = mr.subsets.get(var, [var])
            rs = [q for q, v, _ in assign if v in vs]
            if not rs: raise SqlError(f"AFTER MATCH SKIP TO {var}: the variable has no row in the match")
            r = rs[0] if kind == "first" else rs[-1]
            if r == s0: raise SqlError(f"AFTER MATCH SKIP TO {kind.upper()} {var}: the first row of the match again")
            skip_until = p + r
    return out
