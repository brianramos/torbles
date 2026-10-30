# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Differential fuzzing of the SQL front end: random typed star schemas (the generator of fuzz.py), random SQL text
(filters, expressions, every kind of aggregate, GROUP BY with grouping sets, HAVING, window functions, subqueries,
CTEs, set operations, DISTINCT, ORDER BY / LIMIT), each statement run by torbles (db.sql) and by DuckDB and the
results compared: as multisets, or in order when ORDER BY fixes it; DOUBLE values with a relative tolerance.

    python -m torbles.sqlfuzz --minutes 10
    python -m torbles.sqlfuzz --seed 7 --case 3 --query 12 -v        # replay one statement

Only syntax both engines accept is generated (ISO forms DuckDB lacks are tested in tests/check_sql.py)."""
import argparse, re, datetime as dt, math, os, random, sys, time, traceback
from decimal import Decimal

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path: sys.path.insert(0, _here)
import algq                                            # noqa: E402
import fuzz                                            # noqa: E402
from sqltypes import SqlError                          # noqa: E402

REFUSAL_TEXT = ("is not supported", "references without a match", "reference(s) without a match",
                "more than 12 GROUP BY", "group domain too large", "LATERAL")


def lit(v, typ):
    """a SQL literal for a Python value of a column type"""
    if v is None: return "NULL"
    if isinstance(v, bool): return "TRUE" if v else "FALSE"
    if isinstance(v, Decimal): return format(v, "f") if v == v.to_integral_value() or abs(v.as_tuple().exponent) < 30 else str(float(v))
    if isinstance(v, float):
        if v != v or v in (math.inf, -math.inf): return f"CAST('{v}' AS DOUBLE)"
        return f"CAST({v!r} AS DOUBLE)"
    if isinstance(v, int): return str(v)
    if isinstance(v, dt.datetime): return f"TIMESTAMP '{v.isoformat(sep=' ')}'"
    if isinstance(v, dt.date): return f"DATE '{v.isoformat()}'"
    if isinstance(v, str): return "'" + v.replace("'", "''") + "'"
    return str(v)


NUM = ("INTEGER", "SMALLINT", "BIGINT", "UBIGINT", "DECIMAL", "DOUBLE")


def is_num(t): return t.startswith(NUM)


def moderate(c):
    """a numeric column whose values DuckDB's DOUBLE statistics handle without losing digits (|x| < 2^20)"""
    if not (c[2] in ("INTEGER", "SMALLINT", "BIGINT") or c[2].startswith("DECIMAL")): return False
    vals = [v for v in c[3] if v is not None]
    return all(abs(v) < 2 ** 20 for v in vals)


def q(name): return '"' + name + '"'


class Gen:
    def __init__(self, r, tables):
        self.r, self.tables = r, tables
        self.fact = tables["fact"]
        self.dims = [n for n in tables if n != "fact"]

    # ---------------------------------------------------------------- FROM
    def pick_from(self):
        """(FROM text, [(alias, column, type, values)])"""
        r = self.r
        fcols, _, fks = self.fact
        cols = [("f", c.name, c.typ, c.values) for c in fcols if c.name not in fks or r.random() < 0.3]
        parts = ["fact f"]
        joined = []
        for fk, d in fks.items():
            if r.random() < 0.6 and d not in [x[1] for x in joined]:
                alias = f"j{len(joined)}"
                dcols, pk, _ = self.tables[d]
                joined.append((alias, d, fk, pk))
                cols += [(alias, c.name, c.typ, c.values) for c in dcols]
        style = r.random()
        if style < 0.5:
            parts = ["fact f"]
            for a, d, fk, pk in joined:
                if r.random() < 0.3:                         # LEFT JOIN, sometimes with more ON terms
                    extra = ""
                    if r.random() < 0.4:
                        dc = [c for c in cols if c[0] == a]
                        if dc: extra = f" AND {self.pred(dc, 2)}"
                    elif r.random() < 0.3:
                        fc = [c for c in cols if c[0] == "f"]
                        if fc: extra = f" AND {self.pred(fc, 2)}"
                    parts.append(f"LEFT JOIN {d} {a} ON f.{q(fk)} = {a}.{q(pk)}{extra}")
                else: parts.append(f"JOIN {d} {a} ON f.{q(fk)} = {a}.{q(pk)}")
            return " ".join(parts), cols, []
        frm = ", ".join(["fact f"] + [f"{d} {a}" for a, d, fk, pk in joined])
        conds = [f"f.{q(fk)} = {a}.{q(pk)}" for a, d, fk, pk in joined]
        return frm, cols, conds

    # ---------------------------------------------------------------- expressions
    def const(self, c):
        return lit(fuzz._const_for(self.r, fuzz.Col(c[1], c[2], c[3])), c[2])

    def col(self, cols, pred=None):
        cs = [c for c in cols if pred is None or pred(c[2])]
        if not cs: return None
        return self.r.choice(cs)

    def ref(self, c): return f"{c[0]}.{q(c[1])}"

    def scalar(self, cols, depth=0):
        """(expression text, type)"""
        r = self.r
        c = self.col(cols)
        x = r.random()
        if depth >= 2 or x < 0.45: return self.ref(c), c[2]
        if is_num(c[2]) and c[2] != "UBIGINT":
            k = r.choice([1, 2, 3, 7, 10, -1])
            f = r.choice(["+", "-", "*", "abs", "mod", "round", "neg", "case", "floor", "sign", "cast", "greatest"])
            e = self.ref(c)
            if f in ("+", "-", "*"):
                o = self.col(cols, lambda t: is_num(t) and t != "UBIGINT")
                rhs = self.ref(o) if o and r.random() < 0.5 else str(k)
                return f"({e} {f} {rhs})", "DOUBLE" if "DOUBLE" in (c[2], o[2] if o else "") else c[2]
            if f == "abs": return f"abs({e})", c[2]
            if f == "mod" and not c[2].startswith(("DECIMAL", "DOUBLE")): return f"({e} % {abs(k) or 2})", c[2]
            if f == "round" and c[2].startswith(("DECIMAL", "DOUBLE")): return f"round({e}, {r.choice([0, 1, 2])})", c[2]
            if f == "neg": return f"(-{e})", c[2]
            if f == "floor" and c[2].startswith(("DECIMAL", "DOUBLE")): return f"floor({e})", c[2]
            if f == "sign": return f"sign({e})", "TINYINT"
            if f == "cast" and not c[2].startswith("DOUBLE"): return f"CAST({e} AS DOUBLE)", "DOUBLE"
            if f == "greatest": return f"greatest({e}, {self.const(c)})", c[2]
            return f"CASE WHEN {e} > {self.const(c)} THEN {e} ELSE {self.const(c)} END", c[2]
        if c[2] == "VARCHAR":
            e = self.ref(c)
            f = r.choice(["upper", "lower", "substr", "concat", "length", "trim", "replace", "left", "case", "coalesce"])
            if f in ("upper", "lower", "trim", "reverse"): return f"{f}({e})", "VARCHAR"
            if f == "substr": return f"substring({e}, {r.randint(1, 3)}, {r.randint(0, 3)})", "VARCHAR"
            if f == "concat": return f"({e} || '-')", "VARCHAR"
            if f == "length": return f"length({e})", "BIGINT"
            if f == "replace": return f"replace({e}, 'a', 'X')", "VARCHAR"
            if f == "left": return f"left({e}, {r.randint(0, 2)})", "VARCHAR"
            if f == "coalesce": return f"coalesce(nullif({e}, {self.const(c)}), 'n/a')", "VARCHAR"
            return f"CASE WHEN {e} < {self.const(c)} THEN 'lo' ELSE 'hi' END", "VARCHAR"
        if c[2] in ("DATE", "TIMESTAMP"):
            e = self.ref(c)
            f = r.choice(["year", "month", "trunc", "plus", "dow", "quarter"])
            if f == "year": return f"extract(year from {e})", "BIGINT"
            if f == "month": return f"extract(month from {e})", "BIGINT"
            if f == "dow": return f"extract(dow from {e})", "BIGINT"
            if f == "quarter": return f"extract(quarter from {e})", "BIGINT"
            if f == "trunc": return f"date_trunc('{r.choice(['year', 'month', 'day'])}', {e})", "TIMESTAMP"
            return f"({e} + INTERVAL {r.randint(1, 40)} DAY)", "TIMESTAMP"
        if c[2] == "BOOLEAN":
            return f"(NOT {self.ref(c)})", "BOOLEAN"
        return self.ref(c), c[2]

    def pred(self, cols, depth=0):
        r = self.r
        x = r.random()
        if depth < 2 and x < 0.1: return f"(NOT {self.pred(cols, depth + 1)})"
        if depth < 2 and x < 0.22: return f"({self.pred(cols, depth + 1)} OR {self.pred(cols, depth + 1)})"
        if depth < 2 and x < 0.28: return f"({self.pred(cols, depth + 1)} AND {self.pred(cols, depth + 1)})"
        c = self.col(cols)
        e = self.ref(c)
        y = r.random()
        if y < 0.1:
            a, b = self.const(c), self.const(c)
            return f"{e} {'NOT ' if r.random() < 0.2 else ''}BETWEEN {a} AND {b}"
        if y < 0.2: return f"{e} {'NOT ' if r.random() < 0.2 else ''}IN ({', '.join(self.const(c) for _ in range(r.randint(1, 4)))})"
        if y < 0.27 and c[2] == "VARCHAR":
            vals = [v for v in c[3] if v is not None] or ["a"]
            v = r.choice(vals)
            pat = r.choice([v[:2] + "%", "%" + v[-1:], "_" + v[1:2] + "%", v, "%a%"]).replace("'", "''")   # quoted after slicing
            return f"{e} {'NOT ' if r.random() < 0.2 else ''}LIKE '{pat}'"
        if y < 0.3: return f"{e} IS {'NOT ' if r.random() < 0.5 else ''}NULL"
        if y < 0.42:                                    # two columns of one type, or an expression
            o = self.col(cols, lambda t: t == c[2] or (is_num(t) and is_num(c[2]) and "UBIGINT" not in (t, c[2])))
            if o is not None and o is not c:
                return f"{e} {r.choice(['=', '<>', '<', '<=', '>', '>='])} {self.ref(o)}"
        if y < 0.55:
            s, t = self.scalar(cols, 1)
            if is_num(t) or t == "VARCHAR":
                cc = self.col(cols, lambda tt: tt == t) if t != "BIGINT" else None
                rhs = self.const(cc) if cc is not None else (str(r.randint(-5, 30)) if is_num(t) else "'b'")
                return f"{s} {r.choice(['=', '<>', '<', '>='])} {rhs}"
        op = r.choice(["=", "<>", "<", "<=", ">", ">="])
        return f"{e} {op} {self.const(c)}"

    def agg(self, cols):
        """(aggregate text, type)"""
        r = self.r
        num = lambda t: is_num(t) and t != "UBIGINT"
        x = r.random()
        filt = f" FILTER (WHERE {self.pred(cols, 1)})" if r.random() < 0.15 else ""
        if x < 0.15: return f"count(*){filt}", "BIGINT"
        c = self.col(cols, num) if x < 0.55 else self.col(cols)
        if c is None: return f"count(*){filt}", "BIGINT"
        e = self.ref(c) if r.random() < 0.7 else self.scalar([c], 1)[0]
        if x < 0.55:
            f = r.choice(["sum", "sum", "avg", "min", "max", "count", "stddev_pop", "var_samp", "sum_distinct", "median", "covar_pop", "corr", "regr_slope"])
            if f == "sum_distinct": return f"sum(DISTINCT {e}){filt}", "X"
            if f == "median" and not moderate(c): return f"sum({e}){filt}", "X"
            if f in ("covar_pop", "corr", "regr_slope", "stddev_pop", "var_samp"):
                # statistics on moderate values only: DuckDB evaluates them in DOUBLE, torbles exactly
                mod = [cc for cc in cols if moderate(cc)]
                if not mod: return f"count(*){filt}", "BIGINT"
                y, o = r.choice(mod), r.choice(mod)
                if f in ("stddev_pop", "var_samp"): return f"{f}({self.ref(y)}){filt}", "DOUBLE"
                return f"{f}({self.ref(y)}, {self.ref(o)}){filt}", "DOUBLE"
            return f"{f}({e}){filt}", "X"
        f = r.choice(["min", "max", "count", "count_distinct", "string_agg", "mode", "bool_or", "any"])
        if f == "count_distinct": return f"count(DISTINCT {e}){filt}", "BIGINT"
        if f == "string_agg" and c[2] == "VARCHAR":
            return f"string_agg({e}, ',' ORDER BY {e}){filt}", "VARCHAR"
        if f == "bool_or": return f"bool_or({self.pred(cols, 1)}){filt}", "BOOLEAN"
        if f == "any": return f"bool_and({self.pred(cols, 1)}){filt}", "BOOLEAN"
        if f == "mode" and c[2] not in ("DOUBLE",): return f"count(DISTINCT {e}){filt}", "BIGINT"
        return f"{'min' if f not in ('max', 'count') else f}({e}){filt}", "X"

    # ---------------------------------------------------------------- statements
    def select(self):
        r = self.r
        frm, cols, conds = self.pick_from()
        where = conds + [self.pred(cols) for _ in range(r.choice([0, 1, 1, 2, 3]))]
        kind = r.random()
        if kind < 0.65:                                    # aggregate query
            nk = r.choice([0, 1, 1, 2, 2, 3])
            keys = []
            for _ in range(nk):                            # distinct key expressions (DuckDB binds repeats by position)
                k = self.scalar(cols)
                if k[0] not in [x[0] for x in keys]: keys.append(k)
            aggs = [self.agg(cols) for _ in range(r.randint(1, 3))]
            sel = [k for k, _ in keys] + [a for a, _ in aggs]
            gb = ""
            if keys:
                pos = ", ".join(str(i + 1) for i in range(len(keys)))
                z = r.random()
                if z < 0.12: gb = f" GROUP BY ROLLUP ({pos})"
                elif z < 0.18 and len(keys) <= 3: gb = f" GROUP BY CUBE ({pos})"
                elif z < 0.22: gb = f" GROUP BY GROUPING SETS (({pos}), ())"
                else: gb = f" GROUP BY {pos}"
            having = ""
            if r.random() < 0.2:
                # (CORR / REGR_* of constant columns: ISO NULL, DuckDB NaN, so no IS NOT NULL test on them)
                having = f" HAVING {aggs[0][0]} IS NOT NULL" if r.random() < 0.3 and not aggs[0][0].startswith(("corr", "regr")) \
                    else f" HAVING count(*) > {r.randint(0, 3)}"
            win = ""
            if keys and r.random() < 0.15:
                sel.append(f"rank() OVER (ORDER BY {aggs[0][0]} NULLS LAST, {', '.join(str(i + 1) for i in range(len(sel)))})" if False else
                           f"sum(count(*)) OVER (PARTITION BY {keys[0][0]})")
            # DISTINCT over inexact DOUBLE statistics would merge or split rows by DuckDB's rounding: not compared
            dist = "DISTINCT " if r.random() < 0.05 and not any(t == "DOUBLE" or a.startswith("avg") for a, t in aggs) else ""
            sql = f"SELECT {dist}{', '.join(sel)} FROM {frm}" + (f" WHERE {' AND '.join(where)}" if where else "") + gb + having
            if r.random() < 0.3:
                sql += f" ORDER BY {', '.join(str(i + 1) + ' NULLS FIRST' for i in range(len(sel)))}"; self.feats.add("ORDER BY")
                if r.random() < 0.5: sql += f" LIMIT {r.randint(0, 5)}"; self.feats.add("LIMIT")
            self.kind = ("aggregate, no GROUP BY" if not keys else "aggregate, ROLLUP / CUBE / GROUPING SETS" if gb and "(" in gb
                         else "aggregate, GROUP BY")
            self.feats |= {f for f, on in (("HAVING", having), ("window over groups", win), ("DISTINCT", dist)) if on}
            return sql
        if kind < 0.75:                                    # window functions over the rows (deterministic frames only)
            p, _ = self.scalar(cols); o, _ = self.scalar(cols)
            part = f"PARTITION BY {p} " if r.random() < 0.8 else ""
            items = []
            for _ in range(r.randint(1, 3)):
                z = r.random()
                a, _ = self.agg(cols)
                if z < 0.3: items.append(f"{a} OVER ({part.strip()})")
                elif z < 0.5: items.append(f"{a} OVER ({part}ORDER BY {o} NULLS FIRST)")               # RANGE: peers together
                elif z < 0.6: items.append(f"{a} OVER ({part}ORDER BY {o} DESC NULLS LAST GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW)")
                elif z < 0.7: items.append(f"{a} OVER ({part}ORDER BY {o} NULLS FIRST RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING EXCLUDE GROUP)")
                else: items.append(f"{r.choice(['rank', 'dense_rank', 'percent_rank', 'cume_dist'])}() OVER ({part}ORDER BY {o} NULLS FIRST)")
            sql = f"SELECT {p}, {o}, {', '.join(items)} FROM {frm}" + (f" WHERE {' AND '.join(where)}" if where else "")
            self.kind = "window functions over rows"
            return sql
        if kind < 0.9:                                     # row query
            exprs = [self.scalar(cols)[0] for _ in range(r.randint(1, 4))]
            sql = f"SELECT {'DISTINCT ' if r.random() < 0.3 else ''}{', '.join(exprs)} FROM {frm}" + (f" WHERE {' AND '.join(where)}" if where else "")
            sql += f" ORDER BY {', '.join(str(i + 1) + ' NULLS FIRST' for i in range(len(exprs)))} LIMIT {r.choice([1, 5, 20, 100])}"
            self.kind = "rows, ORDER BY + LIMIT"
            return sql
        # subquery forms
        c = self.col(cols, lambda t: t in ("INTEGER", "VARCHAR", "DATE", "BIGINT", "SMALLINT"))
        if c is None:
            self.kind = "aggregate, no GROUP BY"
            return f"SELECT count(*) FROM {frm}" + (f" WHERE {' AND '.join(where)}" if where else "")
        z = r.random()
        if z < 0.4:
            sub = f"SELECT {self.ref(c).replace(c[0] + '.', 's.')} FROM {self._table_of(c)} s WHERE {self.pred([('s',) + c[1:]], 1)}"
            w = where + [f"{self.ref(c)} {'NOT ' if r.random() < 0.3 else ''}IN ({sub})"]
            self.kind = "IN (subquery)"
            return f"SELECT count(*), {self.agg(cols)[0]} FROM {frm} WHERE {' AND '.join(w)}"
        if z < 0.7:
            sub = f"(SELECT max({q(c[1])}) FROM {self._table_of(c)})"
            w = where + [f"{self.ref(c)} {r.choice(['<', '>=', '='])} {sub}"]
            self.kind = "scalar subquery"
            return f"SELECT count(*) FROM {frm} WHERE {' AND '.join(w)}"
        k, _ = self.scalar(cols)
        a, _ = self.agg(cols)
        inner = f"SELECT {k} AS k, {a} AS v FROM {frm}" + (f" WHERE {' AND '.join(where)}" if where else "") + " GROUP BY 1"
        cond = "WHERE v IS NOT NULL " if not a.startswith(("corr", "regr")) else ""
        self.kind = "CTE + window"
        return f"WITH t AS ({inner}) SELECT k, v, count(*) OVER () FROM t {cond}ORDER BY 1 NULLS FIRST, 2 NULLS FIRST"

    def _table_of(self, c):
        if c[0] == "f": return "fact"
        for n in self.dims:
            if any(cc.name == c[1] for cc in self.tables[n][0]): return n
        return "fact"

    def statement(self):
        """a random statement; self.kind names its form and self.feats lists what else it has (for the timings)"""
        r = self.r
        self.kind, self.feats = None, set()
        if r.random() < 0.06:
            a, b = self.set_side(), self.set_side()
            self.kind = "set operation"
            return f"({a}) {r.choice(['UNION', 'UNION ALL', 'INTERSECT', 'EXCEPT', 'INTERSECT ALL', 'EXCEPT ALL'])} ({b})"
        return self.select()

    def set_side(self):
        r = self.r
        frm, cols, conds = self.pick_from()
        c = self.col(cols, lambda t: t in ("INTEGER", "VARCHAR"))
        if c is None: return "SELECT 1"
        typ = c[2]
        e = self.ref(c) if typ == "VARCHAR" else f"CAST({self.ref(c)} AS VARCHAR)"
        where = conds + [self.pred(cols)]
        return f"SELECT {e} FROM {frm} WHERE {' AND '.join(where)}"


def make_case(cs):
    """fuzz.py's random star schema, with NULLs in some value columns"""
    tables = fuzz.make_case(cs)
    r = random.Random(cs + 31)
    for name, (cols, pk, fks) in tables.items():
        for c in cols:
            if c.name == pk or c.name in fks or r.random() > 0.4: continue
            frac = r.choice([0.02, 0.1, 0.5, 1.0])
            c.values = [None if r.random() < frac else v for v in c.values]
    return tables


# ---------------------------------------------------------------------- comparison
def _close(a, b, tol):
    if a is None or b is None:
        # ISO: CORR / REGR_* of a constant column are NULL; DuckDB returns NaN
        x = b if a is None else a
        return (a is None and b is None) or (isinstance(x, float) and x != x)
    if isinstance(a, float) or isinstance(b, float):
        try: x, y = float(a), float(b)
        except (TypeError, ValueError): return False
        if x != x or y != y: return x != x and y != y
        if x == y: return True
        return abs(x - y) <= tol * max(1.0, abs(x), abs(y))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def _sk(v):
    if v is None: return (0, "")
    if isinstance(v, float): return (1, "%.6e" % v if v == v else "nan")
    if isinstance(v, (int, Decimal)): return (1, "%.6e" % float(v))
    return (2, repr(v))


def same(got, want, ordered, tol=1e-6):
    if len(got) != len(want): return False
    if not ordered:
        got = sorted(got, key=lambda r: [_sk(x) for x in r]); want = sorted(want, key=lambda r: [_sk(x) for x in r])
    if all(all(_close(a, b, tol) for a, b in zip(x, y)) for x, y in zip(got, want)): return True
    if not ordered:                                   # float keys may sort differently: match greedily
        rest = list(want)
        for x in got:
            for i, y in enumerate(rest):
                if all(_close(a, b, tol) for a, b in zip(x, y)): del rest[i]; break
            else: return False
        return True
    return False


def run_one(db, con, sql, times=None):
    """compare one statement; times (a dict) receives the seconds each side took, end to end with its result rows"""
    t0 = time.perf_counter()
    try: want = con.execute(sql).fetchall()
    except Exception as e: return "duckdb-error", f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
    if times is not None: times["duckdb"] = time.perf_counter() - t0; times["rows"] = len(want)
    t0 = time.perf_counter()
    try:
        got = db.sql(sql).rows
        if times is not None: times["torbles"] = time.perf_counter() - t0
    except SqlError as e:
        msg = str(e)
        if any(t in msg for t in REFUSAL_TEXT): return "refused", msg[:200]
        return "ERROR", f"SqlError: {msg}\n{traceback.format_exc(limit=6)}"
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=8)}"
    low = sql.lower()
    ordered = " order by " in low and " limit " in low and "over" not in low
    if same(got, want, ordered): return "ok", ""
    m = re.search(r"\blimit (\d+)$", low.strip())
    if m and len(want) > int(m.group(1)) and len(got) <= int(m.group(1)):
        return "duckdb-error", f"DuckDB returned {len(want)} rows past its LIMIT {m.group(1)} (a DuckDB bug)"
    if _wrapped(got, want):
        return "duckdb-error", "DuckDB's 64-bit integer sum wrapped around (a DuckDB bug; torbles' exact sum differs by k * 2^64)"
    return "MISMATCH", f"torbles ({len(got)} rows) {got[:6]}\nduckdb  ({len(want)} rows) {want[:6]}"


def _wrapped(got, want):
    """do the results differ only by integers that are equal modulo 2^64 (DuckDB wrapping a sum)?"""
    if len(got) != len(want) or not got: return False
    norm = lambda r: tuple((x % (1 << 64)) if isinstance(x, int) and not isinstance(x, bool) else x for x in r)
    a = sorted((norm(r) for r in got), key=lambda r: [_sk(x) for x in r])
    b = sorted((norm(r) for r in want), key=lambda r: [_sk(x) for x in r])
    return same(a, b, True) and not same(got, want, False)


def run(seed, cases, minutes, queries, verbose, only_case=None, only_query=None, stop_on_fail=True, log=print, records=None):
    import duckdb
    t_end = time.time() + 60 * minutes if minutes else None
    tot = dict(ok=0, refused=0, **{"duckdb-error": 0}, MISMATCH=0, ERROR=0)
    reasons, failures = {}, []
    case = 0 if only_case is None else only_case
    try:
        while True:
            if only_case is None:
                if cases is not None and case >= cases: break
                if t_end is not None and time.time() > t_end: break
            cs = seed * 100003 + case
            tables = make_case(cs)
            con = duckdb.connect()
            try: db, _ = fuzz.build_db(tables, cs, con)
            except Exception:
                tot["ERROR"] += 1; failures.append((case, None, "ERROR", traceback.format_exc(limit=6)))
                log(f"case {case}: ERROR building\n{failures[-1][3]}")
                if stop_on_fail or only_case is not None: break
                case += 1; continue
            r = random.Random(cs + 7)
            g = Gen(r, tables)
            nfact = len(tables["fact"][0][0].values)
            for qi in range(queries):
                sql = g.statement()
                if only_query is not None and qi != only_query: continue
                tm = {}
                st, detail = run_one(db, con, sql, tm)
                tot[st] += 1
                if records is not None:
                    feats = set(g.feats)
                    if "LEFT JOIN" in sql: feats.add("LEFT JOIN")
                    records.append(dict(seed=seed, case=case, query=qi, kind=g.kind, feats=sorted(feats), status=st,
                                        torbles=tm.get("torbles"), duckdb=tm.get("duckdb"), rows=tm.get("rows"), fact_rows=nfact))
                if st in ("refused", "duckdb-error"):
                    k = ("(duckdb) " if st == "duckdb-error" else "") + detail[:80]
                    reasons[k] = reasons.get(k, 0) + 1
                if verbose and (only_query is not None or st != "ok"):
                    log(f"case {case} query {qi}: {st}\n  {sql}\n  {detail}")
                if st in ("MISMATCH", "ERROR"):
                    failures.append((case, qi, st, sql, detail))
                    log(f"\n=== {st}: seed {seed} case {case} query {qi}\n{sql}\n{detail}\n"
                        f"replay: python -m torbles.sqlfuzz --seed {seed} --case {case} --query {qi} -v\n")
                    if stop_on_fail: raise StopIteration
            con.close()
            if verbose or case % 10 == 0: log(f"case {case}: totals {tot}")
            if only_case is not None: break
            case += 1
    except StopIteration:
        pass
    return tot, reasons, failures, case


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def timing_table(records):
    """statements by kind, and the time each side took on the ones both answered (ms: mean, median, 95th percentile)"""
    kinds = {}
    for rec in records: kinds.setdefault(rec["kind"], []).append(rec)
    n = len(records)
    head = f"{'statements by kind':42s} {'count':>6s} {'share':>6s}   {'torbles ms mean / median / p95':>32s}   {'DuckDB ms mean / median / p95':>31s}"
    lines = [head]
    def row(name, recs):
        ok = [x for x in recs if x["torbles"] is not None and x["duckdb"] is not None]
        tt = [x["torbles"] * 1e3 for x in ok]; td = [x["duckdb"] * 1e3 for x in ok]
        f = lambda v: f"{sum(v) / len(v):8.1f} {_pct(v, 0.5):8.1f} {_pct(v, 0.95):8.1f}" if v else f"{'':26s}"
        return f"  {name:40s} {len(recs):6d} {100 * len(recs) / n:5.1f}%   {f(tt):>32s}   {f(td):>31s}"
    for k, recs in sorted(kinds.items(), key=lambda x: -len(x[1])): lines.append(row(str(k), recs))
    lines.append(row("all", records))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m torbles.sqlfuzz", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cases", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--queries", type=int, default=30, help="statements per case")
    ap.add_argument("--case", type=int, default=None); ap.add_argument("--query", type=int, default=None)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--stats", default=None, help="write one JSON line per statement (kind, outcome, seconds) to this file")
    a = ap.parse_args(argv)
    seed = a.seed if a.seed is not None else random.randrange(1, 10**6)
    if a.cases is None and a.minutes is None and a.case is None: a.cases = 30
    print(f"torbles {algq.__version__} SQL differential fuzzing against DuckDB, seed {seed}", flush=True)
    t0 = time.time()
    records = []
    tot, reasons, failures, ncase = run(seed, a.cases, a.minutes, a.queries, a.verbose, a.case, a.query,
                                        stop_on_fail=not a.keep_going, log=lambda s: print(s, flush=True), records=records)
    print(f"\n{sum(tot.values())} statements in {time.time() - t0:.0f} s: {tot}")
    if records:
        print(timing_table(records))
        if a.stats:
            import json
            with open(a.stats, "w") as f:
                for rec in records: f.write(json.dumps(rec) + "\n")
    if reasons:
        print("refusals (and statements DuckDB rejected):")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:25]: print(f"  {v:5d}  {k}")
    print("NO MISMATCHES" if not failures else f"{len(failures)} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
