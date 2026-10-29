# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Differential fuzzing: random typed star schemas, random queries, every result compared with DuckDB.

    python -m torbles.fuzz --minutes 10                 # run for 10 minutes from a random seed
    python -m torbles.fuzz --seed 42 --cases 200        # reproducible
    python -m torbles.fuzz --seed 42 --case 17 --query 5 -v     # replay one query

Each case builds random tables in DuckDB (1 to 3 dimensions with hierarchies, a fact table with foreign keys, typed
columns: signed / 64-bit / unsigned integers, decimals, dates, timestamps, strings with quotes and Unicode, booleans,
doubles; edge data: orphaned and NULL references, extreme and constant values, empty tables, sizes around 64-row word
and 4096-row tile boundaries), loads them with the automatic key design, and runs random queries (filters with = <> <
<= > >= BETWEEN IN prefix, NOT and OR, constants inside, outside and between the stored values; GROUP BY on fact and
dimension columns; COUNT or SUM of a random polynomial with decimal constants) on randomly chosen engines and planner
settings: CPU or GPU kernels, every join form, both GROUP BY modes, 1 to 4 threads, cached plans. Every few cases the
database is saved, reloaded (memory-mapped, faces validated), damaged by one flipped bit and reloaded (repaired), and
queried again. A result that differs from DuckDB's is a failure; so is any exception other than a documented refusal
(a sum that could overflow 64 bits, a sum over a column that cannot be summed, a group domain too large)."""
import argparse, datetime as dt, json, os, random, shutil, sys, tempfile, time, traceback
from decimal import Decimal
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path: sys.path.insert(0, _here)
import algq                                                       # noqa: E402
from algq import Query                                            # noqa: E402
import database                                                   # noqa: E402

REFUSALS = (OverflowError,)
REFUSAL_TEXT = ("cannot be summed", "group domain too large", "filter program too long", "more than 18 decimal places",
                "expected a")
ALPHABET = list("abcAB z'_%-") + ["é", "ß", "日", "🙂", "Z", "0", "9"]


# ---------------------------------------------------------------------- random data
class Col:
    def __init__(self, name, typ, values, sqlvals=None):
        self.name, self.typ, self.values = name, typ, values          # python values (None = NULL)
        self.sqlvals = sqlvals                                        # strings for DECIMAL (exact)


def _rand_string(r, maxlen=6):
    return "".join(r.choice(ALPHABET) for _ in range(r.randint(0, maxlen)))


def _rand_column(r, name, n, kind=None, card=None):
    kind = kind or r.choice(["int", "int", "bigint", "ubig", "dec", "dec", "date", "ts", "str", "str", "bool", "double", "const", "smallint"])
    card = card or r.choice([1, 2, 3, 7, 50, 1000, n or 1])
    if kind == "int":
        lo = r.choice([0, -50, -(2**31), 1000, 2**31 - 1000]); span = r.choice([1, 10, 255, 256, 2**16, 2**31 - 1])
        pool = [min(2**31 - 1, max(-(2**31), lo + r.randint(0, span))) for _ in range(card)]
        return Col(name, "INTEGER", [r.choice(pool) for _ in range(n)])
    if kind == "smallint":
        pool = [r.randint(-32768, 32767) for _ in range(card)]
        return Col(name, "SMALLINT", [r.choice(pool) for _ in range(n)])
    if kind == "bigint":
        lo = r.choice([-(2**63), -(2**40), 0, 2**62]); span = r.choice([5, 2**33, 2**62, 2**63 - 1])
        pool = [min(2**63 - 1, max(-(2**63), lo + r.randint(0, span))) for _ in range(card)]
        return Col(name, "BIGINT", [r.choice(pool) for _ in range(n)])
    if kind == "ubig":
        pool = [r.choice([0, 1, 2**64 - 1, 2**63, r.randint(0, 2**64 - 1), r.randint(0, 1000)]) for _ in range(card)]
        return Col(name, "UBIGINT", [r.choice(pool) for _ in range(n)])
    if kind == "dec":
        p, s = r.choice([(4, 2), (9, 2), (9, 4), (18, 2), (18, 0), (12, 6), (38, 3)])
        lim = 10 ** (min(p, 18)) - 1
        pool = [r.randint(-lim, lim) if r.random() < 0.7 else r.randint(0, 99) for _ in range(card)]
        vals = [r.choice(pool) for _ in range(n)]
        return Col(name, f"DECIMAL({p},{s})", [Decimal(v).scaleb(-s) for v in vals])
    if kind == "date":
        base = dt.date(r.choice([1900, 1970, 1992, 2024]), 1, 1); span = r.choice([0, 30, 365 * 7, 365 * 150])
        pool = [base + dt.timedelta(days=r.randint(0, span)) for _ in range(card)]
        return Col(name, "DATE", [r.choice(pool) for _ in range(n)])
    if kind == "ts":
        base = dt.datetime(r.choice([1950, 1970, 2020]), 1, 1); span = r.choice([10**6, 10**12, 10**15])
        pool = [base + dt.timedelta(microseconds=r.randint(0, span)) for _ in range(card)]
        return Col(name, "TIMESTAMP", [r.choice(pool) for _ in range(n)])
    if kind == "str":
        pool = list({_rand_string(r) for _ in range(card)}) or [""]
        return Col(name, "VARCHAR", [r.choice(pool) for _ in range(n)])
    if kind == "bool":
        return Col(name, "BOOLEAN", [r.random() < 0.3 for _ in range(n)])
    if kind == "double":
        pool = [r.choice([0.0, -0.0, 1.5, -2.25, 1e-300, 1e300, round(r.uniform(-1e6, 1e6), r.randint(0, 6))]) for _ in range(card)]
        return Col(name, "DOUBLE", [r.choice(pool) for _ in range(n)])
    if kind == "const":
        v = r.randint(-5, 5); return Col(name, "INTEGER", [v] * n)
    raise ValueError(kind)


def _hierarchy(r, prefix, n):
    """1 to 3 levels, each a function of the next finer one (city -> nation -> region), random types and fan-outs"""
    levels = r.randint(1, 3); fan = [r.choice([2, 3, 5, 8]) for _ in range(levels)]
    top = r.choice([1, 2, 3, 5])
    paths = [(i,) for i in range(top)]
    for f in fan[1:]: paths = [p + (j,) for p in paths for j in range(r.randint(1, f))]
    leaf = [r.choice(paths) for _ in range(n)]
    cols = []
    for lv in range(levels):
        kind = r.choice(["str", "int", "str"])
        def enc(p):
            if kind == "str": return "L" + str(lv) + "_" + "_".join(map(str, p)) + r.choice(["", ""])
            return int(sum(x * 37 ** (lv - i) for i, x in enumerate(p))) - 3
        vals = {}
        cols.append(Col(f"{prefix}_h{lv}", "VARCHAR" if kind == "str" else "INTEGER",
                        [vals.setdefault(p[:lv + 1], enc(p[:lv + 1])) for p in leaf]))
    return cols


def make_case(seed):
    """random tables: {name: (columns, primary key or None, {fk column: dimension})}"""
    r = random.Random(seed)
    ndims = r.choice([0, 1, 1, 2, 2, 3])
    tables = {}
    for d in range(ndims):
        n = r.choice([1, 2, 7, 64, 65, 300, 2000, 5000])
        pk_kind = r.choice(["dense", "sparse", "neg", "big", "str", "date"])
        if pk_kind == "dense": keys, typ = list(range(1, n + 1)), "INTEGER"
        elif pk_kind == "sparse": keys, typ = sorted(r.sample(range(0, 50 * n + 10), n)), "INTEGER"
        elif pk_kind == "neg": keys, typ = [k - n // 2 for k in r.sample(range(0, 3 * n + 3), n)], "INTEGER"
        elif pk_kind == "big": keys, typ = sorted({r.randint(-(2**62), 2**62) for _ in range(n)}), "BIGINT"
        elif pk_kind == "str": keys, typ = sorted({_rand_string(r, 8) + str(i) for i in range(n)}), "VARCHAR"   # sorted: no hash order
        else:
            step = r.choice([1, 3]); keys, typ = [dt.date(2000, 1, 1) + dt.timedelta(days=i * step) for i in range(n)], "DATE"
        r.shuffle(keys); n = len(keys)
        cols = [Col(f"d{d}_key", typ, keys)] + _hierarchy(r, f"d{d}", n)
        for a in range(r.randint(0, 3)): cols.append(_rand_column(r, f"d{d}_a{a}", n))
        tables[f"dim{d}"] = (cols, f"d{d}_key", {})
    m = r.choice([0, 1, 63, 64, 65, 1000, 4096, 4097, 9000, 30000])
    cols, fks = [], {}
    for d in range(ndims):
        dcols, pk, _ = tables[f"dim{d}"]; keys = dcols[0].values
        for role in range(r.choice([1, 1, 1, 2])):
            orphan = r.choice([0, 0, 0.01, 0.3]); null = r.choice([0, 0, 0.05])
            skew = r.random() < 0.3
            vals = []
            for _ in range(m):
                x = r.random()
                if x < null: vals.append(None)
                elif x < null + orphan:
                    vals.append(_orphan_value(r, dcols[0].typ, keys))
                else:
                    vals.append(keys[min(len(keys) - 1, int(r.expovariate(1.0) * 3))] if skew else r.choice(keys))
            name = f"f_d{d}" + ("" if role == 0 else f"_r{role}")
            cols.append(Col(name, dcols[0].typ, vals)); fks[name] = f"dim{d}"
    for a in range(r.randint(1, 6)):
        cols.append(_rand_column(r, f"f_m{a}", m))
    if r.random() < 0.3 and m > 1:                                  # a sorted column (physical order hint)
        c = _rand_column(r, "f_sorted", m, kind="int"); c.values = sorted(c.values); cols.append(c)
    tables["fact"] = (cols, None, fks)
    return tables


def _orphan_value(r, typ, keys):
    ks = set(keys)
    for _ in range(100):
        if typ in ("INTEGER",): v = r.randint(-(2**31), 2**31 - 1)
        elif typ == "BIGINT": v = r.randint(-(2**63), 2**63 - 1)
        elif typ == "VARCHAR": v = _rand_string(r, 9) + "#"
        else: v = dt.date(1990, 1, 1) + dt.timedelta(days=r.randint(0, 40000))
        if v not in ks: return v
    return None


def load_duckdb(tables, con, declare):
    """create the tables in DuckDB (types exact: decimals through strings); declare: PRIMARY / FOREIGN KEY in DDL"""
    import pandas as pd
    for name in sorted(tables, key=lambda x: x != "fact" and 0 or 1):
        cols, pk, fks = tables[name]
        df = pd.DataFrame({c.name: pd.Series([None if v is None else (str(v) if c.typ.startswith(("DECIMAL", "UBIGINT", "BIGINT")) else v) for v in c.values], dtype=object)
                           for c in cols})
        if not len(df.columns): continue
        ddl = []
        for c in cols:
            d = f'"{c.name}" {c.typ}'
            if declare and c.name == pk: d += " PRIMARY KEY"
            if declare and c.name in fks: d += f' REFERENCES {fks[c.name]}("{tables[fks[c.name]][1]}")'
            ddl.append(d)
        con.execute(f"CREATE TABLE {name} ({', '.join(ddl)})")
        if len(df):
            con.register("__df", df)
            sel = ", ".join(f'CAST("{c.name}" AS {c.typ})' for c in cols)
            con.execute(f"INSERT INTO {name} SELECT {sel} FROM __df")
            con.unregister("__df")


# ---------------------------------------------------------------------- random queries (as data: replayable)
def _const_for(r, col):
    vals = [v for v in col.values if v is not None]
    t = col.typ
    if vals and r.random() < 0.65:
        v = r.choice(vals)
        if r.random() < 0.3:                                # just beside an existing value
            if isinstance(v, bool): return not v
            if isinstance(v, int): return v + r.choice([-1, 1])
            if isinstance(v, Decimal): return v + Decimal(r.choice(["0.005", "-0.01", "1", "-0.5"]))
            if isinstance(v, dt.datetime): return v + dt.timedelta(microseconds=r.choice([-1, 1, 500000]))
            if isinstance(v, dt.date): return v + dt.timedelta(days=r.choice([-1, 1]))
            if isinstance(v, float): return v + r.choice([-0.5, 0.25])
            if isinstance(v, str): return v + r.choice(["", "a", "é"])[:1] if r.random() < 0.5 else v[:-1]
        return v
    if t in ("INTEGER", "SMALLINT"): return r.choice([-(2**31), -1, 0, 1, 2**31 - 1, r.randint(-10**6, 10**6), 2**40])
    if t == "BIGINT": return r.choice([-(2**63), 0, 2**63 - 1, r.randint(-(2**63), 2**63 - 1)])
    if t == "UBIGINT": return r.choice([0, 2**64 - 1, 2**63, -1])
    if t.startswith("DECIMAL"): return r.choice([Decimal("0"), Decimal("-0.001"), Decimal("12.345"), Decimal(10**20), 2.5, 7])
    if t == "DATE": return r.choice([dt.date(1, 1, 1), dt.date(1999, 12, 31), dt.date(9999, 12, 31)])
    if t == "TIMESTAMP": return r.choice([dt.datetime(1960, 5, 5, 1, 2, 3), dt.datetime(2020, 1, 1)])
    if t == "VARCHAR": return _rand_string(r, 5)
    if t == "BOOLEAN": return r.random() < 0.5
    if t == "DOUBLE": return r.choice([0.0, -1e301, 1.5, 3.0, 1e-301])
    return 0


def _pred(r, tname, cols, depth=0):
    c = r.choice(cols)
    x = r.random()
    if depth < 2 and x < 0.12: return ("not", _pred(r, tname, cols, depth + 1))
    if depth < 2 and x < 0.25: return ("or", _pred(r, tname, cols, depth + 1), _pred(r, tname, cols, depth + 1))
    if depth < 2 and x < 0.32: return ("and", _pred(r, tname, cols, depth + 1), _pred(r, tname, cols, depth + 1))
    y = r.random()
    if y < 0.12:
        a, b = _const_for(r, c), _const_for(r, c)
        try:
            if a > b and r.random() < 0.8: a, b = b, a
        except TypeError: pass
        return ("between", tname, c.name, a, b)
    if y < 0.22: return ("in", tname, c.name, [_const_for(r, c) for _ in range(r.randint(1, 4))])
    if y < 0.28 and c.typ == "VARCHAR":
        v = r.choice([v for v in c.values if v is not None] or [""])
        return ("prefix", tname, c.name, v[:r.randint(0, len(v))])
    return ("cmp", tname, c.name, r.choice(["EQ", "NE", "LT", "LE", "GT", "GE"]), _const_for(r, c))


SUMMABLE = ("INTEGER", "SMALLINT", "BIGINT", "UBIGINT", "DECIMAL")


def _measure(r, cols, depth=0):
    num = [c for c in cols if c[1].typ.startswith(SUMMABLE)]
    x = r.random()
    if depth >= 2 or x < 0.4 or not num:
        if num and r.random() < 0.85:
            t, c = r.choice(num); return ("col", t, c.name)
        return ("const", r.choice([0, 1, -3, 100, Decimal("0.5"), Decimal("1.25"), 7, Decimal("-0.001")]))
    op = r.choice(["+", "-", "*", "*"])
    return (op, _measure(r, cols, depth + 1), _measure(r, cols, depth + 1))


def make_queries(tables, seed, count):
    r = random.Random(seed * 7919 + 1)
    fcols, _, fks = tables["fact"]
    out = []
    for _ in range(count):
        joins = {}
        for fk, d in fks.items():
            if r.random() < 0.55 and d not in joins: joins[d] = fk
        avail = [("fact", c) for c in fcols if c.name not in fks]
        for d in joins: avail += [(d, c) for c in tables[d][0]]
        preds = []
        for _ in range(r.choice([0, 1, 1, 2, 3])):
            if not avail: break
            t = r.choice(sorted({a for a, _ in avail}))
            preds.append(_pred(r, t, [c for a, c in avail if a == t]))
        keys = []
        for _ in range(r.choice([0, 0, 1, 1, 2, 3])):
            if avail:
                k = r.choice(avail)
                if k not in keys: keys.append(k)
        keys = [(t, c.name) for t, c in keys]
        meas = None if r.random() < 0.3 else _measure(r, avail)
        opts = dict(engine=r.choice(["alg", "alg", "alg", "gpu"]), join_mode=r.choice(["auto", "auto", "factored", "expanded", "lookup"]),
                    groupby=r.choice(["mm", "mm", "split"]), threads=r.choice([1, 2, 4]), repeat=r.random() < 0.2)
        out.append(dict(joins=joins, preds=preds, keys=keys, measure=meas, opts=opts))
    return out


def materialize(spec, db):
    T = db.tables
    def pr(p):
        k = p[0]
        if k == "not": return ~pr(p[1])
        if k == "or": return pr(p[1]) | pr(p[2])
        if k == "and": return pr(p[1]) & pr(p[2])
        col = getattr(T[p[1]], p[2])
        if k == "cmp": return {"EQ": col.__eq__, "NE": col.__ne__, "LT": col.__lt__, "LE": col.__le__, "GT": col.__gt__, "GE": col.__ge__}[p[3]](p[4])
        if k == "between": return col.between(p[3], p[4])
        if k == "in": return col.isin(p[3])
        if k == "prefix": return col.startswith(p[3])
        raise ValueError(k)
    def me(m):
        if m[0] == "col": return getattr(T[m[1]], m[2])
        if m[0] == "const": return algq.Const(m[1])
        a, b = me(m[1]), me(m[2])
        return a + b if m[0] == "+" else a - b if m[0] == "-" else a * b
    q = Query(T["fact"])
    for d, fk in spec["joins"].items(): q.join(T[d], getattr(T["fact"], fk))
    for p in spec["preds"]: q.where(pr(p))
    if spec["keys"]: q.group_by(*[getattr(T[t], c) for t, c in spec["keys"]])
    if spec["measure"] is not None: q.sum(me(spec["measure"]))
    q.join_mode = spec["opts"]["join_mode"]
    return q


def _norm(x):
    """results compared exactly; -0.0 and 0.0 are the same group"""
    if isinstance(x, dict): return {tuple((0.0 if isinstance(v, float) and v == 0 else v) for v in k): val for k, val in x.items()}
    return x


def run_query(db, spec):
    """-> (status, detail): ok | refused | duckdb-error | MISMATCH | ERROR"""
    try: q = materialize(spec, db)
    except Exception as e:
        if isinstance(e, REFUSALS) or any(t in str(e) for t in REFUSAL_TEXT): return "refused", f"{type(e).__name__}: {e}"
        return "ERROR", "building the query: " + traceback.format_exc(limit=6)
    try: sql = q.sql()
    except Exception: return "ERROR", "rendering SQL: " + traceback.format_exc(limit=6)
    try: want = _norm(db.sql_result(q))
    except Exception as e: return "duckdb-error", f"{e}"
    o = spec["opts"]
    algq.set_groupby(o["groupby"]); algq.set_threads(o["threads"])
    try:
        got = _norm(q.run(o["engine"]))
        if o["repeat"]: got2 = _norm(q.run(o["engine"]))
        else: got2 = got
    except Exception as e:
        if isinstance(e, REFUSALS) or any(t in str(e) for t in REFUSAL_TEXT): return "refused", f"{type(e).__name__}: {e}"
        return "ERROR", f"{sql}\n" + traceback.format_exc(limit=8)
    finally:
        algq.set_groupby("mm"); algq.set_threads(1)
    if got != want or got2 != want:
        return "MISMATCH", _diff(sql, got if got != want else got2, want, o)
    return "ok", ""


def _diff(sql, got, want, opts):
    lines = [f"opts {opts}", sql]
    if isinstance(got, dict) and isinstance(want, dict):
        ks = sorted(set(got) | set(want), key=repr)
        bad = [k for k in ks if got.get(k, "<missing>") != want.get(k, "<missing>")]
        lines.append(f"{len(bad)} of {len(ks)} groups differ; first ones:")
        for k in bad[:6]: lines.append(f"   {k}: torbles {got.get(k, '<missing>')!r}  duckdb {want.get(k, '<missing>')!r}")
    else: lines.append(f"torbles {got!r}\nduckdb  {want!r}")
    return "\n".join(lines)


def build_db(tables, seed, con):
    r = random.Random(seed + 99)
    declare = all(all(v is not None and v in set(tables[d][0][0].values) for v in c.values)
                  for c in tables["fact"][0] for fk, d in [(c.name, tables["fact"][2].get(c.name))] if d) and r.random() < 0.5
    load_duckdb(tables, con, declare)
    pks = {n: t[1] for n, t in tables.items() if t[1]}
    fks = {f"fact.{c}": d for c, d in tables["fact"][2].items()}
    names = list(tables)
    if declare: db = database.from_duckdb(con, tables=names)
    else: db = database.from_duckdb(con, tables=names, primary_keys=pks, foreign_keys=fks, keep_plain=r.random() < 0.3)
    return db, declare


def run(seed, cases, minutes, queries, verbose, only_case=None, only_query=None, save_every=4, stop_on_fail=True, log=print):
    import duckdb
    t_end = time.time() + 60 * minutes if minutes else None
    tot = dict(ok=0, refused=0, **{"duckdb-error": 0}, MISMATCH=0, ERROR=0)
    reasons, failures = {}, []
    tmp = tempfile.mkdtemp(prefix="torbles-fuzz-")
    case = 0 if only_case is None else only_case
    try:
        while True:
            if only_case is None:
                if cases is not None and case >= cases: break
                if t_end is not None and time.time() > t_end: break
            cs = seed * 100003 + case
            tables = make_case(cs)
            con = duckdb.connect()
            try:
                db, declared = build_db(tables, cs, con)
            except Exception as e:
                failures.append((case, None, "ERROR", "building the database: " + traceback.format_exc(limit=8)))
                tot["ERROR"] += 1; log(f"case {case}: ERROR building\n{failures[-1][3]}")
                if stop_on_fail or only_case is not None: break
                case += 1; continue
            specs = make_queries(tables, cs, queries)
            dbs = [("built", db)]
            if save_every and case % save_every == 0:
                p = os.path.join(tmp, f"c{case}")
                db.save(p)
                dbs.append(("loaded", database.load(p, con=con)))
                if case % (2 * save_every) == 0:              # one flipped bit in a stored plane: repaired on load
                    p2 = p + "-flipped"; shutil.copytree(p, p2)   # a copy: the open database maps the original
                    f = os.path.join(p2, "fact.tbl")
                    if os.path.exists(f) and os.path.getsize(f) > 8192:
                        b = bytearray(open(f, "rb").read()); pos = 4096 + random.Random(cs).randrange(len(b) - 4096)
                        b[pos] ^= 1 << random.Random(cs + 1).randrange(8); open(f, "wb").write(b)
                        dl = database.load(p2, con=con)
                        dbs.append(("bit-flipped+repaired", dl))
            for tag, d in dbs:
                for qi, spec in enumerate(specs):
                    if only_query is not None and qi != only_query: continue
                    st, detail = run_query(d, spec)
                    tot[st] += 1
                    if st == "duckdb-error": reasons["(duckdb) " + detail[:90]] = reasons.get("(duckdb) " + detail[:90], 0) + 1
                    if st == "refused": reasons[detail.split(":")[0] + ": " + detail.split(":", 1)[1][:60]] = reasons.get(detail.split(":")[0] + ": " + detail.split(":", 1)[1][:60], 0) + 1
                    if verbose and (only_query is not None or st not in ("ok",)):
                        log(f"case {case} query {qi} [{tag}]: {st} {detail}")
                    if st in ("MISMATCH", "ERROR"):
                        failures.append((case, qi, st, f"[{tag}] {detail}"))
                        log(f"\n=== {st}: seed {seed} case {case} query {qi} [{tag}]\n{detail}\n"
                            f"replay: python -m torbles.fuzz --seed {seed} --case {case} --query {qi} -v\n")
                        if stop_on_fail: raise StopIteration
            con.close()
            if verbose or case % 10 == 0:
                log(f"case {case}: {len(tables['fact'][0][0].values) if tables['fact'][0] else 0} fact rows, "
                    f"{len(tables) - 1} dims, keys {'declared' if declared else 'given'}; totals {tot}")
            if only_case is not None: break
            case += 1
    except StopIteration:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return tot, reasons, failures, case


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m torbles.fuzz", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cases", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--queries", type=int, default=25, help="queries per case")
    ap.add_argument("--case", type=int, default=None); ap.add_argument("--query", type=int, default=None)
    ap.add_argument("--keep-going", action="store_true", help="do not stop at the first failure")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    seed = a.seed if a.seed is not None else random.randrange(1, 10**6)
    if a.cases is None and a.minutes is None and a.case is None: a.cases = 50
    print(f"torbles {algq.__version__} differential fuzzing against DuckDB, seed {seed}", flush=True)
    t0 = time.time()
    tot, reasons, failures, ncase = run(seed, a.cases, a.minutes, a.queries, a.verbose, a.case, a.query,
                                        stop_on_fail=not a.keep_going, log=lambda s: print(s, flush=True))
    print(f"\n{ncase + (0 if failures else 1) if a.case is None else 1} case(s), {sum(tot.values())} queries in {time.time() - t0:.0f} s: {tot}")
    if reasons:
        print("refusals (and queries DuckDB rejected):"); [print(f"  {v:5d}  {k}") for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:20]]
    # a refusal is a query DuckDB answers and torbles declines (queries DuckDB rejects are not asked of torbles): the
    # goal is fewer refusals than DuckDB errors
    print(f"refused by torbles {tot['refused']}, rejected by DuckDB {tot['duckdb-error']}: "
          f"{'refused <= DuckDB errors' if tot['refused'] <= tot['duckdb-error'] else 'MORE REFUSALS THAN DuckDB ERRORS'}")
    print("NO MISMATCHES" if not failures else f"{len(failures)} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
