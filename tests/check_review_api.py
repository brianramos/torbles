# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Release review R04 and R07: the engine-level query API (db.query / algq.Query) on nullable typed columns, and its
SQL rendering.
  R04  SUM, MIN / MAX, COUNT(e) and (sum, count) results read only the rows where every column of their argument is not
       NULL (no NULL placeholder code is summed, compared or counted), while the groups are those of the selected rows:
       a group whose argument is NULL on every row is present with SUM / MIN / MAX None and COUNT(e) 0.
  R07  Query.sql() renders every aggregate the API has (MIN / MAX were rendered as COUNT(*)), db.sql_result() decodes
       it into run()'s result form, and the filters on nullable columns render the engine's two-valued semantics.
Every query is compared with an exact Python oracle on the source rows and, through its SQL rendering, with DuckDB;
on the algebraic CPU engine, the plain columnar baseline and the GPU executor (CUDA, or its emulation), every GROUP BY
mode (matrix product, mask split, the row kernel off and for every GROUP BY, the Python planner without plan reuse,
without the key factoring), join form, 1 and 2 threads, and in child processes the C engine's settings run_all.sh
uses (TORBLES_MM_ROWS=2, TORBLES_MM_ROWS=0 with TORBLES_FACTOR=0, TORBLES_POPCNT=scalar).
    python3 tests/check_review_api.py"""
import datetime as dt, os, shutil, subprocess, sys, tempfile
from decimal import Decimal
import numpy as np, pandas as pd, duckdb
import torbles, algq
from algq import Query, Table, MinMax

CHILD = "--child" in sys.argv          # the query matrix once, under the C engine settings of the environment
fails = 0
def check(ok, what, detail=""):
    global fails
    if not ok: fails += 1; print(f"  FAIL {what}" + (f"\n       {detail}" if detail else ""))
    return ok
def attempt(fn, *args):
    """fn(*args), or its exception as text (a failed check, not the end of the run)"""
    try: return fn(*args)
    except Exception as e: return f"{type(e).__name__}: {e}"

# ---------------------------------------------------------------------------------------------- data
rng = np.random.default_rng(2026)
N, ND = 4500, 12                       # past one 4096-row tile; NULLs in every 64-row word
POOL = ["", "a", "b", "b'c", "é", "zz"]
def null(p): return rng.random() < p
F = {c: [] for c in ("g", "x", "y", "h", "z", "dd", "dt", "s", "w", "e", "flag", "k")}
for i in range(N):
    g = int(rng.integers(0, 6)); F["g"].append(g)
    # each group's own range, so that a NULL placeholder (the code of the column's minimum) would show in its MIN / SUM;
    # x is NULL on every row of group 5
    F["x"].append(None if g == 5 or null(0.3) else 10 * g - 15 + int(rng.integers(0, 20)))
    F["y"].append(None if null(0.25) else 100_000 * g - 300_000 + int(rng.integers(0, 100_000)))
    F["h"].append(None if null(0.15) else int(rng.integers(0, 4)))
    F["z"].append(int(rng.integers(1, 10)))
    F["dd"].append(None if null(0.2) else Decimal(1000 * g + int(rng.integers(-500, 999)) - 2000).scaleb(-2))
    F["dt"].append(None if null(0.2) else dt.date(1995, 1, 1) + dt.timedelta(days=365 * g + int(rng.integers(0, 300))))
    F["s"].append(None if null(0.2) else POOL[int(rng.integers(0 if g == 0 else 1, len(POOL)))])
    F["w"].append(None if null(0.25) else int(rng.integers(-2**62, 2**62)))    # a 64-bit range: self-extending
    F["e"].append(None)                                                         # NULL on every row
    F["flag"].append(None if null(0.2) else bool(rng.random() < 0.4))
    F["k"].append(None if null(0.07) else 99 if null(0.03) else int(rng.integers(0, ND)))   # NULL and orphaned references
D = dict(id=list(range(ND)), a=[None if i in (1, 4, 9) else 3 * i - 5 for i in range(ND)],
         b=[None if i in (2, 7) else (i - 6) * 2**40 + i for i in range(ND)],    # past 32 bits: rank-coded (a dimension)
         r=["pqr"[i % 3] for i in range(ND)], q=[None if i % 4 == 0 else "uv"[i % 2] for i in range(ND)])
def frame(cols):                       # text, cast to the declared types by the loader: exact
    return pd.DataFrame({c: pd.Series([None if v is None else (str(v).lower() if isinstance(v, bool) else str(v)) for v in vs],
                                      dtype=object) for c, vs in cols.items()})
DDL = """CREATE TABLE dim (id INTEGER PRIMARY KEY, a INTEGER, b BIGINT, r VARCHAR, q VARCHAR);
CREATE TABLE f (g INTEGER, x INTEGER, y BIGINT, h INTEGER, z INTEGER, dd DECIMAL(9,2), dt DATE, s VARCHAR, w BIGINT,
                e INTEGER, flag BOOLEAN, k INTEGER REFERENCES dim(id));"""
db = torbles.from_ddl(DDL, {"dim": frame(D), "f": frame(F)}, keep_plain=True)
T = {"f": db["f"], "d": db["dim"]}
DIM = {i: {("d", c): D[c][i] for c in D} for i in range(ND)}
ROWS = [{("f", c): F[c][i] for c in F} for i in range(N)]
assert set(T["f"].nullable) >= {"x", "y", "h", "dd", "dt", "s", "w", "e", "flag"} and "w" in T["f"].wide, T["f"].nullable
assert set(T["d"].nullable) == {"a", "b", "q"} and T["d"].codecs["b"].kind == "rank"

# ---------------------------------------------------------------------------------------------- queries and oracle
# expressions: ("c", table, column) | ("k", constant) | (op, a, b);  predicates: ("cmp", t, c, op, v) | ("in", t, c, vals)
# | ("btw", t, c, lo, hi) | ("valid", t, c, 0 or 1) (the validity plane) | ("not", p) | ("and", p, q) | ("or", p, q)
def C(t, c): return ("c", t, c)
X, Y, Z, H, G, DD, W, E, FL = (C("f", c) for c in ("x", "y", "z", "h", "g", "dd", "w", "e", "flag"))
A, B = C("d", "a"), C("d", "b")
OPS = {"EQ": lambda a, b: a == b, "NE": lambda a, b: a != b, "LT": lambda a, b: a < b, "LE": lambda a, b: a <= b,
       "GT": lambda a, b: a > b, "GE": lambda a, b: a >= b}
def ex(e, TT=None):                    # (TT: the tables, by default those built above)
    TT = TT or T
    if e[0] == "c": return TT[e[1]][e[2]]
    if e[0] == "k": return algq.Const(e[1])
    a, b = ex(e[1], TT), ex(e[2], TT)
    return a + b if e[0] == "+" else a - b if e[0] == "-" else a * b
def ev(e, row):                        # SQL arithmetic: NULL when any operand is
    if e[0] == "c": return row[(e[1], e[2])]
    if e[0] == "k": return e[1]
    a, b = ev(e[1], row), ev(e[2], row)
    if a is None or b is None: return None
    return a + b if e[0] == "+" else a - b if e[0] == "-" else a * b
def pr(p, TT=None):
    TT = TT or T
    k = p[0]
    if k == "not": return ~pr(p[1], TT)
    if k == "and": return pr(p[1], TT) & pr(p[2], TT)
    if k == "or": return pr(p[1], TT) | pr(p[2], TT)
    t = TT[p[1]]; col = t[p[2]]
    if k == "cmp": return {"EQ": col.__eq__, "NE": col.__ne__, "LT": col.__lt__, "LE": col.__le__, "GT": col.__gt__, "GE": col.__ge__}[p[3]](p[4])
    if k == "in": return col.isin(p[3])
    if k == "btw": return col.between(p[3], p[4])
    if k == "valid": return t[t.nullable[p[2]]] == p[3]
    raise ValueError(k)
def pv(p, row):                        # the engine API's logic: a comparison with a NULL is FALSE (NOT of it TRUE)
    k = p[0]
    if k == "not": return not pv(p[1], row)
    if k == "and": return pv(p[1], row) and pv(p[2], row)
    if k == "or": return pv(p[1], row) or pv(p[2], row)
    v = row[(p[1], p[2])]
    if k == "valid": return (v is not None) == bool(p[3])
    if v is None: return False
    if k == "cmp": return OPS[p[3]](v, p[4])
    if k == "in": return v in p[3]
    if k == "btw": return p[3] <= v <= p[4]
    raise ValueError(k)

def build(s, TT=None):
    TT = TT or T
    q = Query(TT["f"])
    if s.get("join"): q.join(TT["d"])
    if s.get("where") is not None: q.where(pr(s["where"], TT))
    if s.get("keys"): q.group_by(*[TT[t][c] for _, t, c in s["keys"]])
    kind, arg = s["agg"]
    if kind == "sum": q.sum(ex(arg, TT))
    elif kind == "both": q.sum(ex(arg, TT)); q._both = True
    elif kind == "minmax": q.minmax(ex(arg, TT))
    elif arg is None: q.count()
    else: q.count(ex(arg, TT))
    return q

def oracle(s):
    groups = {}
    for fr in ROWS:
        row = fr
        if s.get("join"):
            dr = DIM.get(fr[("f", "k")])
            if dr is None: continue                   # an inner join: a NULL or orphaned reference never joins
            row = {**fr, **dr}
        if s.get("where") is not None and not pv(s["where"], row): continue
        groups.setdefault(tuple(row[(t, c)] for _, t, c in s.get("keys", [])), []).append(row)
    kind, arg = s["agg"]
    def agg(rows):
        if kind == "count" and arg is None: return len(rows)
        vals = [v for v in (ev(arg, r) for r in rows) if v is not None]     # the observations
        if kind == "count": return len(vals)
        if kind == "minmax": return MinMax((min(vals), max(vals))) if vals else None
        sm = sum(vals[1:], vals[0]) if vals else None
        return (sm, len(vals)) if kind == "both" else sm
    if s.get("keys"): return {k: agg(rs) for k, rs in groups.items()}
    return agg(groups.get((), []))

G5, NONE = ("cmp", "f", "g", "EQ", 5), ("cmp", "f", "g", "EQ", 9)
NOTX = ("not", ("cmp", "f", "x", "GT", 0))
SPECS = {
    "SUM(x)": dict(agg=("sum", X)),
    "SUM(x) by g (group 5 all NULL)": dict(keys=[G], agg=("sum", X)),
    "SUM(x) by nullable h": dict(keys=[H], agg=("sum", X)),
    "SUM(x) by g, h": dict(keys=[G, H], agg=("sum", X)),
    "SUM(x) where g = 5 (only NULLs)": dict(where=G5, agg=("sum", X)),
    "SUM(x) where g = 5 by h": dict(where=G5, keys=[H], agg=("sum", X)),
    "SUM(x), empty selection": dict(where=NONE, agg=("sum", X)),
    "SUM(x) by g, empty selection": dict(where=NONE, keys=[G], agg=("sum", X)),
    "SUM(x * x) by g (sum of squares)": dict(keys=[G], agg=("sum", ("*", X, X))),
    "SUM(x * y) by g (two NULL patterns)": dict(keys=[G], agg=("sum", ("*", X, Y))),
    "SUM(x * y) where g = 5": dict(where=G5, agg=("sum", ("*", X, Y))),
    "SUM(x + y)": dict(agg=("sum", ("+", X, Y))),
    "SUM(x * z) by h (one nullable)": dict(keys=[H], agg=("sum", ("*", X, Z))),
    "SUM(2x - 3) by h": dict(keys=[H], agg=("sum", ("-", ("*", ("k", 2), X), ("k", 3)))),
    "SUM(x * 0.5) by g (decimal constant)": dict(keys=[G], agg=("sum", ("*", X, ("k", Decimal("0.5"))))),
    "SUM(z) by g (no NULLs)": dict(keys=[G], agg=("sum", Z)),
    "SUM(dd) by g (decimal)": dict(keys=[G], agg=("sum", DD)),
    "SUM(dd * x) by h": dict(keys=[H], agg=("sum", ("*", DD, X))),
    "SUM(w) by g (64-bit, self-extending)": dict(keys=[G], agg=("sum", W)),
    "SUM(w + x) by h (128-bit sums)": dict(keys=[H], agg=("sum", ("+", W, X))),
    "SUM(w * w * x) (past 128 bits: bridge)": dict(agg=("sum", ("*", ("*", W, W), X)), duck=False),
    "SUM(w * w * x) by g (bridge)": dict(keys=[G], agg=("sum", ("*", ("*", W, W), X)), duck=False),
    "SUM(e) (NULL on every row)": dict(agg=("sum", E)),
    "SUM(e) by g": dict(keys=[G], agg=("sum", E)),
    "SUM(x) by nullable boolean flag": dict(keys=[FL], agg=("sum", X)),
    "SUM(x * g) by g (factored through the key)": dict(keys=[G], agg=("sum", ("*", X, G))),
    "SUM(x * h + 1) by h (nullable key factor)": dict(keys=[H], agg=("sum", ("+", ("*", X, H), ("k", 1)))),
    "SUM(x) where x > 10 (validity in the filter)": dict(where=("cmp", "f", "x", "GT", 10), keys=[G], agg=("sum", X)),
    "SUM(x) where x < -10 or x > 40 by g": dict(where=("or", ("cmp", "f", "x", "LT", -10), ("cmp", "f", "x", "GT", 40)), keys=[G], agg=("sum", X)),
    "SUM(x) where not x > 0 by g": dict(where=NOTX, keys=[G], agg=("sum", X)),
    "SUM(y) where h in (1, 2) by g": dict(where=("in", "f", "h", [1, 2]), keys=[G], agg=("sum", Y)),
    "SUM(x * dim.a) by dim.r": dict(join=True, keys=[C("d", "r")], agg=("sum", ("*", X, A))),
    "SUM(dim.a) by g": dict(join=True, keys=[G], agg=("sum", A)),
    "SUM(dim.b) by g (rank-coded: bridge)": dict(join=True, keys=[G], agg=("sum", B)),
    "SUM(dim.b)": dict(join=True, agg=("sum", B)),
    "SUM(x) by nullable dim.q": dict(join=True, keys=[C("d", "q")], agg=("sum", X)),
    "SUM(x) where dim.r = 'p' by g": dict(join=True, where=("cmp", "d", "r", "EQ", "p"), keys=[G], agg=("sum", X)),
    "SUM(y) where dim.a > 0 by h": dict(join=True, where=("cmp", "d", "a", "GT", 0), keys=[H], agg=("sum", Y)),
    "MIN/MAX(x)": dict(agg=("minmax", X)),
    "MIN/MAX(x) by g": dict(keys=[G], agg=("minmax", X)),
    "MIN/MAX(x) by h": dict(keys=[H], agg=("minmax", X)),
    "MIN/MAX(x) where g = 5": dict(where=G5, agg=("minmax", X)),
    "MIN/MAX(x) where g = 5 by h": dict(where=G5, keys=[H], agg=("minmax", X)),
    "MIN/MAX(x), empty selection": dict(where=NONE, agg=("minmax", X)),
    "MIN/MAX(y) by g": dict(keys=[G], agg=("minmax", Y)),
    "MIN/MAX(dd) by g (decimal)": dict(keys=[G], agg=("minmax", DD)),
    "MIN/MAX(dt) by g (date)": dict(keys=[G], agg=("minmax", C("f", "dt"))),
    "MIN/MAX(s) by g (string)": dict(keys=[G], agg=("minmax", C("f", "s"))),
    "MIN/MAX(z) by h (no NULLs)": dict(keys=[H], agg=("minmax", Z)),
    "MIN/MAX(e) by g (NULL on every row)": dict(keys=[G], agg=("minmax", E)),
    "MIN/MAX(x) where not x > 0 by g": dict(where=NOTX, keys=[G], agg=("minmax", X)),
    "MIN/MAX(x) by dim.r": dict(join=True, keys=[C("d", "r")], agg=("minmax", X)),
    "COUNT(*) by g": dict(keys=[G], agg=("count", None)),
    "COUNT(x) by g": dict(keys=[G], agg=("count", X)),
    "COUNT(x) by h": dict(keys=[H], agg=("count", X)),
    "COUNT(x)": dict(agg=("count", X)),
    "COUNT(x) where g = 5": dict(where=G5, agg=("count", X)),
    "COUNT(x), empty selection": dict(where=NONE, agg=("count", X)),
    "COUNT(x * y) by g": dict(keys=[G], agg=("count", ("*", X, Y))),
    "COUNT(e) by g": dict(keys=[G], agg=("count", E)),
    "COUNT(flag) by h": dict(keys=[H], agg=("count", FL)),
    "COUNT(dim.a) by g": dict(join=True, keys=[G], agg=("count", A)),
    "COUNT(*) where not x > 0 (NULL rows pass)": dict(where=NOTX, agg=("count", None)),
    "COUNT(*) where not x in (5, 6, 7)": dict(where=("not", ("in", "f", "x", [5, 6, 7])), agg=("count", None)),
    "COUNT(*) where not x between 0 and 9": dict(where=("not", ("btw", "f", "x", 0, 9)), keys=[G], agg=("count", None)),
    "COUNT(*) where x <> 5": dict(where=("cmp", "f", "x", "NE", 5), keys=[H], agg=("count", None)),
    "COUNT(*) where x is NULL (validity plane 0)": dict(where=("valid", "f", "x", 0), keys=[G], agg=("count", None)),
    "COUNT(*) where y is not NULL or x > 30": dict(where=("or", ("valid", "f", "y", 1), ("cmp", "f", "x", "GT", 30)), agg=("count", None)),
    "COUNT(*) where not s = 'b' by g": dict(where=("not", ("cmp", "f", "s", "EQ", "b")), keys=[G], agg=("count", None)),
    "COUNT(*) where not dim.a < 4 by dim.r": dict(join=True, where=("not", ("cmp", "d", "a", "LT", 4)), keys=[C("d", "r")], agg=("count", None)),
    "(SUM, COUNT)(x) by g": dict(keys=[G], agg=("both", X)),
    "(SUM, COUNT)(x)": dict(agg=("both", X)),
    "(SUM, COUNT)(x) where g = 5": dict(where=G5, agg=("both", X)),
    "(SUM, COUNT)(dd) by h": dict(keys=[H], agg=("both", DD)),
}
if CHILD: SPECS = {n: s for i, (n, s) in enumerate(SPECS.items()) if i % 2 == 0 or s.get("join")}

# ---------------------------------------------------------------------------------------------- the engine matrix
DEFAULTS = dict(GROUPBY=algq.GROUPBY, ROWPATH_GROUPS=algq.ROWPATH_GROUPS, NATIVE_PLAN=algq.NATIVE_PLAN,
                PLAN_CACHE=algq.PLAN_CACHE, FACTOR=algq.FACTOR)
CONFIGS = [("default", {}), ("mask split", dict(GROUPBY="split")), ("no row kernel", dict(ROWPATH_GROUPS=0)),
           ("row kernel for 2+ groups", dict(ROWPATH_GROUPS=2)), ("Python planner, no plan reuse", dict(NATIVE_PLAN=False, PLAN_CACHE=False)),
           ("no key factoring", dict(FACTOR=False))]
if CHILD: CONFIGS = CONFIGS[:1]
def configure(over):
    for k, v in {**DEFAULTS, **over}.items():
        if k == "GROUPBY": algq.set_groupby(v)
        else: setattr(algq, k, v)
def groups_dict(gc):
    """Query.run_columns()'s Groups as run()'s dict: key columns as codes (decoded, NULL where not valid) or values"""
    cols = []
    for codes, valid, cd in gc.keys:
        if codes is None: cols.append(list(cd)); continue
        vals = cd.decode(algq._codes_array(codes)) if cd is not None else np.asarray(codes).tolist()
        if valid is not None: vals = [v if ok else None for v, ok in zip(vals, np.asarray(valid).tolist())]
        cols.append(vals)
    return dict(zip(zip(*cols), gc.vals)) if len(gc) else {}

print(f"== R04: nullable aggregates, {len(SPECS)} queries vs exact Python oracle and DuckDB"
      + (f"  [child: {' '.join(f'{k}={v}' for k, v in os.environ.items() if k in ('TORBLES_MM_ROWS', 'TORBLES_FACTOR', 'TORBLES_POPCNT'))}]" if CHILD else ""))
for name, s in SPECS.items():
    want = oracle(s); bad = []
    q = attempt(build, s)
    if isinstance(q, str): check(False, name, q); continue
    if s.get("duck", True):
        src = attempt(db.sql_result, q)
        if src != want: bad.append(f"DuckDB on the SQL rendering: {str(src)[:150]}")
        v = attempt(db.verify, q)
        if not (isinstance(v, tuple) and v[0] is True): bad.append(f"db.verify: {str(v)[:150]}")
    runs = 0
    for cname, over in CONFIGS:
        configure(over)
        q = build(s)                                     # fresh: plans compiled under this configuration, then reused
        for th in (1, 2):
            algq.set_threads(th)
            for mode in (("auto", "factored", "expanded", "lookup") if s.get("join") else ("auto",)):
                q.join_mode = mode
                for eng, gmm in [("alg", None), ("plain", None)] + [("gpu", m) for m in (("word", "popc", "mma") if cname == "default" else ("word",))]:
                    if gmm: algq.gpu_mm(gmm)
                    got = attempt(q.run, eng); runs += 1
                    if got != want and len(bad) < 4: bad.append(f"{cname}/{eng}{'/' + gmm if gmm else ''}/{mode}/{th}T: {str(got)[:150]}")
        if s.get("keys") and not CHILD:                   # the groups as columns
            got = attempt(lambda: groups_dict(q.run_columns("alg")))
            if got != want and len(bad) < 4: bad.append(f"{cname}/run_columns: {str(got)[:150]}")
    configure({}); algq.set_threads(1); algq.gpu_mm("word")
    check(not bad, f"{name} ({runs} runs)", "\n       ".join(bad + [f"want {str(want)[:200]}"]))
    if not bad and not CHILD: print(f"  ok   {name:46s} {runs} runs")

if not CHILD:
    print("== R04: the SQL front end agrees (its own validity handling)"); f0 = fails
    for sql in ("SELECT g, SUM(x), COUNT(x), MIN(x), MAX(x), COUNT(*) FROM f GROUP BY g",
                "SELECT h, SUM(x * y), MIN(dd), MAX(dd) FROM f GROUP BY h"):
        for eng in ("alg", "plain", "gpu"):
            got = attempt(lambda: sorted(db.sql(sql, engine=eng).rows, key=repr)); want = sorted(db.con.execute(sql).fetchall(), key=repr)
            check(got == want, f"db.sql {sql[:60]} [{eng}]", f"{str(got)[:200]} != {str(want)[:200]}")
    if fails == f0: print("  ok   db.sql grouped SUM / COUNT / MIN / MAX over nullable columns")

    print("== R04: the lowering itself"); f0 = fails
    f = T["f"]
    q = db.query(f).group_by(f.g).sum(f.x)
    check(q._null_terms() == [(f, "x#v")], "an unrestricted nullable measure reads its observations apart from its groups")
    for label, qq in [("x > 10", db.query(f).where(f.x > 10).group_by(f.g).sum(f.x)),
                      ("x#v = 1", db.query(f).where(f[f.nullable["x"]] == 1).sum(f.x * f.z)),
                      ("x < 0 or x > 9", db.query(f).where((f.x < 0) | (f.x > 9)).minmax(f.x)),
                      ("db.sql's IS NOT NULL", db.query(f).where(algq.Cmp(algq.Col(f, "x#v"), "EQ", 1)).count(f.x))]:
        check(qq._null_terms() == [], f"a filter requiring x not NULL ({label}) leaves the query as it is")
    check(db.query(f).where(~(f.x > 10)).sum(f.x)._null_terms() == [(f, "x#v")], "NOT of a comparison does not require x")
    check(db.query(f).sum(f.z).count()._null_terms() == [] and db.query(f).count(f.z)._null_terms() == [], "non-nullable aggregates unchanged")
    for eng in ("alg", "plain"):                       # COUNT(e) of a dimension that is not joined: refused, as SUM is
        got = attempt(db.query(f).group_by(f.g).count(T["d"].r).run, eng)
        check(isinstance(got, str) and got.startswith("ValueError"), f"COUNT(dim.r) without the join is refused [{eng}]", str(got)[:120])
    # (a table without self-extending columns: their lowering builds the measure anew on every run, so plans there are
    # compiled again every time, nullable or not)
    db2 = torbles.from_ddl("CREATE TABLE u (g INTEGER, x INTEGER)", {"u": frame(dict(g=F["g"], x=F["x"]))})
    u = db2["u"]; q2 = db2.query(u).group_by(u.g).sum(u.x)
    first = attempt(q2.run); second = attempt(q2.run)
    check(first == second == oracle(dict(keys=[G], agg=("sum", X))) and q2.plan_cached is True,
          "a repeated nullable GROUP BY reuses its compiled plans", f"plan_cached={getattr(q2, 'plan_cached', None)}")
    db2.close()
    q.where(f.h == 2)                                    # the query changes: new observation queries
    check(attempt(q.run) == oracle(dict(keys=[G], where=("cmp", "f", "h", "EQ", 2), agg=("sum", X))), "a changed query after a run")
    q.group_by(f.h)
    check(attempt(q.run) == oracle(dict(keys=[H], where=("cmp", "f", "h", "EQ", 2), agg=("sum", X))), "new GROUP BY keys after a run")
    q.minmax(f.x)
    check(attempt(q.run) == oracle(dict(keys=[H], where=("cmp", "f", "h", "EQ", 2), agg=("minmax", X))), "sum() then minmax() on one query")
    if fails == f0: print("  ok   observations only where needed; plans reused; queries changed between runs")

    print("== R04: a saved and loaded database (memory-mapped planes, validity from the manifest)"); f0 = fails
    tmp = tempfile.mkdtemp(prefix="torbles-review-api-")
    try:
        path = os.path.join(tmp, "db.torbles"); db.save(path)
        ldb = torbles.load(path, con=db.con)
        LT = {"f": ldb["f"], "d": ldb["dim"]}
        check(LT["f"].nullable == T["f"].nullable and LT["d"].nullable == T["d"].nullable, "validity planes survive save / load")
        nl = 0
        for name in ("SUM(x) by g (group 5 all NULL)", "SUM(x * y) by g (two NULL patterns)", "SUM(w) by g (64-bit, self-extending)",
                     "SUM(w * w * x) by g (bridge)", "SUM(dim.b) by g (rank-coded: bridge)", "MIN/MAX(x) where g = 5 by h",
                     "MIN/MAX(s) by g (string)", "COUNT(x) by h", "COUNT(dim.a) by g", "(SUM, COUNT)(x) by g",
                     "COUNT(*) where not x > 0 (NULL rows pass)"):
            s = SPECS[name]; want = oracle(s); q = build(s, LT)
            for eng in ("alg", "gpu"):
                got = attempt(q.run, eng); nl += 1
                check(got == want, f"loaded: {name} [{eng}]", f"{str(got)[:150]} != {str(want)[:150]}")
            if s.get("duck", True):
                v = attempt(ldb.verify, q)
                check(isinstance(v, tuple) and v[0] is True, f"loaded: db.verify {name}", str(v)[:200])
        if fails == f0: print(f"  ok   {nl} runs on the loaded database equal the oracle; db.verify agrees")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- R07
if not CHILD:
    print("== R07: the review's cases"); f0 = fails
    t3 = torbles.from_ddl("CREATE TABLE t (x INTEGER, g INTEGER)", {"t": pd.DataFrame({"x": [-3, 8, 1], "g": [0, 0, 1]})})
    t = t3["t"]; q = t3.query(t).minmax(t.x)
    check(q.sql() == "SELECT MIN(t.x), MAX(t.x) FROM t", "minmax().sql() renders MIN and MAX", q.sql())
    for eng in ("alg", "gpu"):
        v = attempt(t3.verify, q, eng)
        check(v == (True, (-3, 8), (-3, 8)), f"db.verify(minmax) [{eng}]", str(v))
    q = t3.query(t).group_by(t.g).minmax(t.x); v = attempt(t3.verify, q)
    check(v == (True, {(0,): (-3, 8), (1,): (1, 1)}, {(0,): (-3, 8), (1,): (1, 1)}), "db.verify(grouped minmax)", str(v))
    t3.close()
    t3 = torbles.from_ddl("CREATE TABLE t (x INTEGER, g INTEGER)",
                          {"t": pd.DataFrame({"x": pd.Series([3, None, 8], dtype="Int32"), "g": [0, 1, 0]})}, keep_plain=True)
    t = t3["t"]
    for eng in ("alg", "plain", "gpu"):
        got = [attempt(qq.run, eng) for qq in (t3.query(t).sum(t.x), t3.query(t).minmax(t.x), t3.query(t).group_by(t.g).sum(t.x),
                                               t3.query(t).group_by(t.g).minmax(t.x), t3.query(t).where(t.g == 1).sum(t.x),
                                               t3.query(t).where(t.g == 1).minmax(t.x))]
        want = [11, (3, 8), {(0,): 11, (1,): None}, {(0,): (3, 8), (1,): None}, None, None]
        check(got == want, f"R04 reproducer x = [3, NULL, 8] [{eng}]", f"{got} != {want}")
    t3.close()
    if fails == f0: print("  ok   minmax renders MIN / MAX and verifies; x = [3, NULL, 8]: SUM 11, all-NULL group None")

    print("== R07: every aggregate kind renders, and db.verify() agrees, on every engine"); f0 = fails
    f = T["f"]
    KINDS = [("COUNT(*)", lambda q, c: q.count(), lambda c: ["COUNT(*)"]),
             ("COUNT(c)", lambda q, c: q.count(c), lambda c: [f"COUNT({c})"]),
             ("COUNT(1)", lambda q, c: q.count(1), lambda c: ["COUNT(1)"]),
             ("SUM(c)", lambda q, c: q.sum(c), lambda c: [f"SUM({c})"]),
             ("SUM(c * c)", lambda q, c: q.sum(c * c), lambda c: [f"SUM(({c} * {c}))"]),
             ("MIN/MAX(c)", lambda q, c: q.minmax(c), lambda c: [f"MIN({c})", f"MAX({c})"]),
             ("(SUM, COUNT)(c)", lambda q, c: setattr(q.sum(c), "_both", True), lambda c: [f"SUM({c})", f"COUNT({c})"])]
    nk = 0
    for kname, setk, sqlk in KINDS:
        for cname in ("x", "z"):                         # nullable, not
            for keys in ([], [f.g], [f.h]):
                for wlabel, where in (("", None), (" where g = 5", f.g == 5), (" where g = 9", f.g == 9)):
                    q = db.query(f)
                    if where is not None: q.where(where)
                    if keys: q.group_by(*keys)
                    setk(q, f[cname])
                    exp = sqlk(f"f.{cname}")
                    check(q._sql_aggs() == exp, f"{kname} of {cname} renders {exp}", str(q._sql_aggs()))
                    for eng in ("alg", "plain", "gpu"):
                        v = attempt(db.verify, q, eng); nk += 1
                        check(isinstance(v, tuple) and v[0] is True, f"db.verify {kname} of {cname} by {[k.name for k in keys]}{wlabel} [{eng}]", str(v)[:250])
    if fails == f0: print(f"  ok   {len(KINDS)} aggregate kinds x nullable / not x 3 groupings x 3 filters x 3 engines: {nk} verifications")

    print("== R07: range and inequality joins (the plain uint32 API): sql(), sql_preagg() and the engine agree"); f0 = fails
    r2 = np.random.default_rng(5); n2 = 3000
    fact = Table("fact", {"x": r2.integers(0, 5000, n2).astype(np.uint32), "v": r2.integers(0, 1000, n2).astype(np.uint32),
                          "c": r2.integers(0, 4, n2).astype(np.uint32)})
    lo_ = r2.integers(0, 4000, 16).astype(np.uint32)
    band = Table("band", {"lo": lo_, "hi": (lo_ + r2.integers(0, 1500, 16)).astype(np.uint32), "t": (np.arange(16) % 3).astype(np.uint32)})
    con = duckdb.connect()
    for tb in (fact, band): con.register(tb.name, pd.DataFrame({c: v.astype(np.int64) for c, v in tb.data.items()}))
    def rows_of(sql, q, na):
        rows = con.execute(sql).fetchall()
        val = (lambda a: tuple(a) if na > 1 else a[0])
        return {tuple(r[:-na]): val(r[-na:]) for r in rows} if q.keys else val(rows[0][-na:])
    nr = 0
    for kname, setk, _ in KINDS:
        for keys in ([], [band.t], [fact.c, band.t]):
            for path in ("groups", "slices"):
                q = Query(fact).join_range(band, fact.x, band.lo, band.hi)
                if keys: q.group_by(*keys)
                setk(q, fact.v); q.range_mode = path
                na = len(q._sql_aggs())
                plain_sql, pre = attempt(lambda: rows_of(q.sql(), q, na)), attempt(lambda: rows_of(q.sql_preagg(), q, na))
                if kname.startswith("MIN"):
                    check(plain_sql == pre, f"range join {kname} by {[k.name for k in keys]}: sql() = sql_preagg()", f"{str(plain_sql)[:100]} != {str(pre)[:100]}")
                    got = attempt(q.run)
                    check(isinstance(got, str) and got.startswith("ValueError"), f"range join {kname} by {[k.name for k in keys]}: refused (it ran COUNT)", str(got)[:100])
                    continue
                check(plain_sql == pre, f"range join {kname} by {[k.name for k in keys]}: sql() = sql_preagg()", f"{str(plain_sql)[:100]} != {str(pre)[:100]}")
                for eng in ("alg", "plain", "gpu"):
                    got = attempt(q.run, eng); nr += 1
                    check(got == plain_sql, f"range join {kname} by {[k.name for k in keys]} [{path}, {eng}]", f"{str(got)[:100]} != {str(plain_sql)[:100]}")
    if fails == f0: print(f"  ok   {nr} range-join runs equal DuckDB on sql() and sql_preagg(); MIN / MAX over them refused")

# ---------------------------------------------------------------------------------------------- the C engine's settings
if not CHILD:
    print("== the C engine's settings (child processes)")
    for env in ({"TORBLES_MM_ROWS": "2"}, {"TORBLES_MM_ROWS": "0", "TORBLES_FACTOR": "0"}, {"TORBLES_POPCNT": "scalar"}):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env={**os.environ, **env},
                           capture_output=True, text=True, timeout=3600)
        last = (p.stdout.strip().splitlines() or [""])[-1]
        tag = " ".join(f"{k}={v}" for k, v in env.items())
        if check(p.returncode == 0 and last == "ALL MATCH", f"child {tag}", (p.stdout + p.stderr)[-1500:]): print(f"  ok   {tag}")
print("ALL MATCH" if not fails else f"{fails} FAILURES")
