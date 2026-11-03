# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Parse the supported DML structure, retaining SQL expressions for binding."""

from dataclasses import dataclass

from . import sqlfront
from .sqltypes import SqlError


@dataclass(frozen=True)
class Source:
    table: str | None
    alias: str | None
    query: str | None
    sql: str


@dataclass(frozen=True)
class DML:
    kind: str
    target: str
    alias: str | None = None
    source: Source | None = None
    columns: tuple[str, ...] | None = None
    values: tuple[tuple[str, ...], ...] | None = None
    select: str | None = None
    assignments: tuple[tuple[str, str], ...] = ()
    where: str | None = None
    on: str | None = None
    matched_delete: bool = False


_DML = {"INSERT", "UPDATE", "DELETE", "MERGE"}
_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = frozenset(_PAIRS.values())
_UNSUPPORTED = {"RETURNING", "ORDER", "GROUP", "HAVING", "LIMIT", "OFFSET", "FETCH", "QUALIFY", "WINDOW"}


def is_dml(sql):
    """Check the first statement token without parsing a SELECT in full."""
    pos = 0
    while match := sqlfront._TOK.match(sql, pos):
        kind, text = match.lastgroup, match.group(0)
        if kind not in ("ws", "lc", "bc") and text != ";":
            return kind == "id" and text.upper() in _DML
        pos = match.end()
    return False


def _top(tokens, start=0, end=None):
    """Tokens outside nested expressions and CASE bodies."""
    brackets, cases = [], 0
    for i in range(start, len(tokens) if end is None else end):
        token = tokens[i]
        if token.s in _PAIRS:
            brackets.append(_PAIRS[token.s])
        elif token.s in _CLOSERS:
            if not brackets or brackets.pop() != token.s:
                raise SqlError("unbalanced delimiters in DML")
        elif not brackets:
            if token.up() == "CASE":
                cases += 1
            elif token.up() == "END" and cases:
                cases -= 1
            elif not cases:
                yield i
    if brackets or cases:
        raise SqlError("unbalanced expression in DML")


def _parts(tokens, start, end):
    for i in _top(tokens, start, end):
        if tokens[i].s == ",":
            yield start, i
            start = i + 1
    yield start, end


def _identifier(token):
    if token.k == "id":
        return token.s
    if token.k == "qid" and len(token.s) > 2:
        return token.s[1:-1].replace('""', '"')
    raise SqlError(f"expected a DML identifier, got {token.s!r}")


def _unique(names, what):
    if len({name.casefold() for name in names}) != len(names):
        raise SqlError(f"duplicate {what} in DML")


class _Parser:
    def __init__(self, tokens):
        self.tokens, self.i = tokens, 0

    def word(self):
        return self.tokens[self.i].up() if self.i < len(self.tokens) else None

    def symbol(self, value):
        return self.i < len(self.tokens) and self.tokens[self.i].s == value

    def accept(self, word):
        if self.word() != word:
            return False
        self.i += 1
        return True

    def require(self, word):
        if not self.accept(word):
            raise SqlError(f"expected {word} in DML")

    def name(self):
        if self.i == len(self.tokens):
            raise SqlError("expected a DML identifier at the end of the statement")
        name = _identifier(self.tokens[self.i])
        self.i += 1
        return name

    def alias(self, following):
        if self.accept("AS"):
            return self.name()
        if (self.i < len(self.tokens) and self.tokens[self.i].k in ("id", "qid")
                and self.word() not in following | _UNSUPPORTED):
            return self.name()
        return None

    def text(self, start, end):
        if start == end:
            raise SqlError("expected an expression or query in DML")
        return sqlfront._text(self.tokens, start, end)

    def boundary(self, keywords):
        for j in _top(self.tokens, self.i):
            word = self.tokens[j].up()
            if word in keywords:
                if word == "FROM" and j > self.i and self.tokens[j - 1].up() == "DISTINCT":
                    continue
                return j
        return len(self.tokens)

    def expression(self, stops=frozenset()):
        end = self.boundary(stops | _UNSUPPORTED)
        out = self.text(self.i, end)
        self.i = end
        return out

    def parens(self):
        if not self.symbol("("):
            raise SqlError("expected parentheses in DML")
        start = self.i + 1
        closes = [")"]
        for j in range(start, len(self.tokens)):
            symbol = self.tokens[j].s
            if symbol in _PAIRS:
                closes.append(_PAIRS[symbol])
            elif symbol in _CLOSERS:
                if closes.pop() != symbol:
                    raise SqlError("unbalanced delimiters in DML")
                if not closes:
                    self.i = j + 1
                    return start, j
        raise SqlError("unclosed parentheses in DML")

    def columns(self):
        start, end = self.parens()
        columns = []
        for lo, hi in _parts(self.tokens, start, end):
            if hi - lo != 1:
                raise SqlError("expected a comma-separated DML column list")
            columns.append(_identifier(self.tokens[lo]))
        _unique(columns, "target column")
        return tuple(columns)

    def row(self):
        start, end = self.parens()
        return tuple(self.text(lo, hi) for lo, hi in _parts(self.tokens, start, end))

    def source(self, following):
        start = self.i
        if self.symbol("("):
            lo, hi = self.parens()
            if lo == hi or self.tokens[lo].up() not in ("SELECT", "WITH", "VALUES"):
                raise SqlError("DML source must be a table or a parenthesized query")
            query = self.text(lo, hi)
            alias = self.alias(following)
            if alias is None:
                raise SqlError("a derived DML source requires an alias")
            if self.symbol("("):
                self.columns()  # Validate the alias column list; retain it in source SQL.
            table = None
        else:
            table = self.name()
            if self.symbol("."):
                raise SqlError("schema-qualified DML sources are not supported")
            query, alias = None, self.alias(following)
        return Source(table, alias, query, self.text(start, self.i))

    def assignments(self, target, alias, stops):
        end = self.boundary(stops | _UNSUPPORTED)
        assignments = []
        for lo, hi in _parts(self.tokens, self.i, end):
            if lo == hi:
                raise SqlError("expected an assignment after SET")
            name = _identifier(self.tokens[lo])
            pos = lo + 1
            if pos + 1 < hi and self.tokens[pos].s == ".":
                if name.casefold() != (alias or target).casefold():
                    raise SqlError("SET can assign only columns of its target")
                name, pos = _identifier(self.tokens[pos + 1]), pos + 2
            if pos >= hi or self.tokens[pos].s != "=":
                raise SqlError("expected column = expression in SET")
            assignments.append((name, self.text(pos + 1, hi)))
        _unique([name for name, _ in assignments], "assignment column")
        self.i = end
        return tuple(assignments)

    def finish(self, statement):
        if self.i != len(self.tokens):
            raise SqlError(f"unsupported or unexpected DML clause near {self.tokens[self.i].s!r}")
        return statement

    def insert(self):
        self.require("INTO")
        target = self.name()
        columns = self.columns() if self.symbol("(") else None
        if self.accept("VALUES"):
            rows = [self.row()]
            while self.symbol(","):
                self.i += 1
                rows.append(self.row())
            width = len(columns) if columns is not None else len(rows[0])
            if any(len(row) != width for row in rows):
                raise SqlError("INSERT VALUES rows must match the target column count")
            return self.finish(DML("insert", target, columns=columns, values=tuple(rows)))
        if self.word() not in ("SELECT", "WITH"):
            raise SqlError("INSERT requires VALUES or SELECT")
        for j in _top(self.tokens, self.i):
            if sqlfront._kw(self.tokens, j, "ON", "CONFLICT"):
                raise SqlError("ON CONFLICT is not supported in INSERT")
        return DML("insert", target, columns=columns, select=self.text(self.i, len(self.tokens)))

    def update(self):
        target = self.name()
        alias = self.alias({"SET"})
        self.require("SET")
        assignments = self.assignments(target, alias, {"FROM", "WHERE"})
        source = self.source({"WHERE"}) if self.accept("FROM") else None
        where = self.expression() if self.accept("WHERE") else None
        return self.finish(DML("update", target, alias, source, assignments=assignments, where=where))

    def delete(self):
        self.require("FROM")
        target = self.name()
        alias = self.alias({"USING", "WHERE"})
        source = self.source({"WHERE"}) if self.accept("USING") else None
        where = self.expression() if self.accept("WHERE") else None
        return self.finish(DML("delete", target, alias, source, where=where))

    def merge(self):
        self.require("INTO")
        target = self.name()
        alias = self.alias({"USING"})
        self.require("USING")
        source = self.source({"ON"})
        self.require("ON")
        on = self.expression({"WHEN"})
        assignments, columns, values, delete = (), None, None, False
        while self.accept("WHEN"):
            insert = self.accept("NOT")
            self.require("MATCHED")
            if not self.accept("THEN"):
                raise SqlError("conditional or BY SOURCE MERGE branches are not supported")
            if insert:
                if values is not None:
                    raise SqlError("MERGE supports one WHEN NOT MATCHED branch")
                self.require("INSERT")
                columns = self.columns()
                self.require("VALUES")
                row = self.row()
                if len(row) != len(columns):
                    raise SqlError("MERGE INSERT values must match the target column count")
                values = (row,)
            else:
                if assignments or delete:
                    raise SqlError("MERGE supports one WHEN MATCHED branch")
                if self.accept("UPDATE"):
                    self.require("SET")
                    assignments = self.assignments(target, alias, {"WHEN"})
                elif self.accept("DELETE"):
                    delete = True
                else:
                    raise SqlError("WHEN MATCHED supports UPDATE SET or DELETE")
        if not (assignments or delete or values):
            raise SqlError("MERGE requires a WHEN branch")
        return self.finish(DML("merge", target, alias, source, columns, values,
                               assignments=assignments, on=on, matched_delete=delete))


def parse_dml(sql):
    """Parse one INSERT, UPDATE, DELETE or MERGE statement."""
    tokens = sqlfront.tokenize(sql)
    start, end = 0, len(tokens)
    while start < end and tokens[start].s == ";":
        start += 1
    while end > start and tokens[end - 1].s == ";":
        end -= 1
    tokens = tokens[start:end]
    kind = tokens[0].up() if tokens else None
    if kind not in _DML:
        raise SqlError("expected INSERT, UPDATE, DELETE or MERGE")
    for j in _top(tokens):
        if tokens[j].s == ";":
            raise SqlError("execute one DML statement at a time")
        if tokens[j].up() == "RETURNING":
            raise SqlError("RETURNING is not supported in DML")
    parser = _Parser(tokens)
    parser.i = 1
    return getattr(parser, kind.lower())()
