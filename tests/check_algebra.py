# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The algebraic paths that avoid materializing rows, pairs or (group, value) tuples, each run with its switch on and
off (the answers must be identical) and against DuckDB:
  - ORDER BY ... LIMIT k by rank selection on the planes (sqlexec.TOPK)
  - MEDIAN / PERCENTILE_CONT / _DISC per group by rank selection (sqlexec.RANK_AGGS), 8 bits per engine query or the
    whole code at once (sqlexec.RANK_DIRECT)
  - COUNT(DISTINCT column) from one bit per (group, value) cell in the engine (sqlexec.DIST_BITS)
  - aggregates over an inner join of two FROM items from the sides' partial sums, without the row pairs
    (sqlexec.JOIN_PREAGG; doubles on and off within 1e-9: their sums are added in another order)
  - window functions from their aggregated series: one engine GROUP BY of (partition, order key), the frame as prefix
    sums, truncated under ORDER BY ... LIMIT on the window's key (sqlexec.SERIES_WINDOWS)
on a table large enough for the paths to matter (ties, NULLs, a dimension, filters).
    python tests/check_algebra.py [queries per part, default 150] [seed, default 7]"""
import os, random, sys
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import duckdb, numpy as np, pandas as pd
import torbles
import sqlexec, sqlfuzz

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
ENGINE = os.environ.get("TORBLES_SQL_ENGINE", "alg")
rng = np.random.default_rng(SEED); r = random.Random(SEED)
N = 60_000
fact = pd.DataFrame(dict(
    id=np.arange(N), k=rng.integers(0, 40, N), q=rng.integers(1, 51, N), d=rng.integers(0, 11, N),
    v=rng.integers(-2_000_000, 2_000_000, N), w=rng.integers(0, 300, N),
    y=pd.array(np.where(rng.random(N) < 0.15, None, rng.integers(0, 5000, N)), dtype="Int64"),
    p=rng.choice([0.5, 1.25, 2.0, -3.5, 7.75], N), s=rng.choice(["a", "b", "c", "dd", "e"], N)))
rng_e = np.random.default_rng(SEED + 1000)
fact["e"] = [None if x < 0.1 else Decimal(int(c)).scaleb(-2) for x, c in zip(rng_e.random(N), rng_e.integers(-99999, 99999, N))]
dim = pd.DataFrame(dict(k=np.arange(40), region=[f"r{i % 5}" for i in range(40)], cat=[i % 3 for i in range(40)]))
con = duckdb.connect()
con.execute("CREATE TABLE dim (k INTEGER PRIMARY KEY, region VARCHAR, cat INTEGER); CREATE TABLE fact (id INTEGER PRIMARY KEY, "
            "k INTEGER REFERENCES dim(k), q INTEGER, d INTEGER, v INTEGER, w INTEGER, y INTEGER, p DOUBLE, s VARCHAR, e DECIMAL(9,2))")
con.register("fa", fact); con.register("di", dim); con.execute("INSERT INTO dim SELECT * FROM di"); con.execute("INSERT INTO fact SELECT * FROM fa")
con.unregister("fa"); con.unregister("di")
db = torbles.from_duckdb(con, keep_plain=ENGINE == "plain")

sqlexec.ALG_MIN_ROWS = 0            # the paths on this table too (by default from 2^16 fact rows on)
fails = 0; duck_ok = 0; used = {}


def run(sql, flag, on):
    setattr(sqlexec, flag, on)
    try: return db.sql(sql, ENGINE).rows, None
    except Exception as x: return None, f"{type(x).__name__}: {x}"
    finally: setattr(sqlexec, flag, True)


def compare(sql, flag, ordered, duck=True, tol=0.0):
    """the query with the path on and off (identical answers), then against DuckDB (duck: when the answer is
    determined: a LIMIT under an order with ties may keep other tied rows there)"""
    global fails, duck_ok
    a, ea = run(sql, flag, True); b, eb = run(sql, flag, False)
    if ea or eb:
        if ea and eb and ea.split(":")[0] == eb.split(":")[0]: return
        fails += 1; print(f"FAIL {sql}\n   on:  {ea or 'ok'}\n   off: {eb or 'ok'}"); return
    if (not sqlfuzz.same(a, b, ordered, tol)) if tol else ((a != b) if ordered else (sorted(map(repr, a)) != sorted(map(repr, b)))):
        fails += 1; print(f"FAIL (on != off) {sql}\n   on  {a[:4]}\n   off {b[:4]}"); return
    if not duck: return
    try: want = con.execute(sql).fetchall()
    except duckdb.Error as x: fails += 1; print(f"FAIL (DuckDB error {x}) {sql}"); return
    if sqlfuzz.same(a, want, ordered, 1e-9): duck_ok += 1
    else: fails += 1; print(f"FAIL (vs DuckDB) {sql}\n   torbles {a[:4]}\n   duckdb  {want[:4]}")


# ------------------------------------------------------------------ ORDER BY ... LIMIT k: rank selection on the planes
print("== ORDER BY ... LIMIT k by rank selection")
orig_topk = sqlexec._Planner.topk_preds
def counting(self, preds):
    out = orig_topk(self, preds)
    if out: used["topk"] = used.get("topk", 0) + 1
    return out
sqlexec._Planner.topk_preds = counting
COLS = ["q", "d", "v", "w", "y", "p", "s", "id", "k"]
# windows in the ORDER BY after its first key, and windows in the SELECT list (read from their series or not): rank
# selection reads only the rows before the k-th, so it must stay off unless every window is computed on every row
WIN_ORDER = ["lag(v) OVER (ORDER BY id)", "lead(q, 2) OVER (ORDER BY id)", "count(*) OVER (PARTITION BY d)",
             "min(id) OVER (PARTITION BY d ORDER BY id DESC ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING)",
             "sum(v) OVER (PARTITION BY d ORDER BY q)", "rank() OVER (ORDER BY w)"]
WIN_SEL = ["count(*) OVER (PARTITION BY d) AS wc", "rank() OVER (ORDER BY q) AS wr", "sum(w) OVER (PARTITION BY d ORDER BY q) AS ws"]
for qi in range(NQ):
    nkeys = r.choice([1, 1, 2, 2, 3])
    keys = r.sample(COLS, nkeys) + (["id"] if r.random() < 0.7 else [])          # id last: a total order
    order = []
    for c in dict.fromkeys(keys):
        direction = r.choice(["", " DESC", " ASC"])
        nulls = r.choice(["", "", " NULLS LAST", " NULLS FIRST"]) if c == "y" else ""
        order.append(f"{c}{direction}{nulls}")
    where = r.choice(["", "", " WHERE d < 6", " WHERE v > 0 AND q BETWEEN 10 AND 30", " WHERE y IS NOT NULL", " WHERE s <> 'b'"])
    frm = "fact"
    if r.random() < 0.25: frm = "fact JOIN dim ON fact.k = dim.k"; where += (" AND" if where else " WHERE") + " dim.region IN ('r1', 'r3')"
    if r.random() < 0.08: frm = "fact LEFT JOIN dim ON fact.k = dim.k AND dim.cat = 1"                # several passes: not pushed down
    sel = ", ".join(r.sample(["id", "q", "v", "y", "p", "s", "w"], 3))
    lim = r.choice([1, 3, 10, 25, 100, 2000])
    off = r.choice(["", "", f" OFFSET {r.choice([1, 7, 50])}"])
    distinct = "DISTINCT " if r.random() < 0.05 else ""
    if not distinct and r.random() < 0.2:
        order.insert(r.randint(1, max(1, len(order) - 1)), r.choice(WIN_ORDER) + r.choice(["", " DESC"]))
    if not distinct and r.random() < 0.2: sel += ", " + r.choice(WIN_SEL)
    sql = f"SELECT {distinct}{sel}, {', '.join(dict.fromkeys(keys))} FROM {frm}{where} ORDER BY {', '.join(order)} LIMIT {lim}{off}"
    compare(sql, "TOPK", ordered=True, duck=keys[-1] == "id")
print(f"   rank selection used on {used.get('topk', 0)} of {NQ} queries")

# ------------------------------------------------------------------ MEDIAN / PERCENTILE per group: rank selection
print("== MEDIAN / PERCENTILE_CONT / _DISC by rank selection")
orig_rank = sqlexec._Planner.rank_values
def counting_rank(self, *args):
    used["rank"] = used.get("rank", 0) + 1
    return orig_rank(self, *args)
sqlexec._Planner.rank_values = counting_rank
VALS = ["v", "q", "w", "y", "p", "d", "id"]
GROUPS = [[], ["k"], ["d"], ["k", "d"], ["dim.region"], ["dim.region", "d"], ["d % 3"], ["s"], ["y IS NULL"]]
for qi in range(NQ):
    aggs = []
    for _ in range(r.choice([1, 1, 2, 3])):
        c = r.choice(VALS); f = r.random()
        if f < 0.35: aggs.append(f"median({c})")
        elif f < 0.6: aggs.append(f"percentile_cont({r.choice(['0.5', '0.25', '0.9', '0', '1', '0.333'])}) WITHIN GROUP (ORDER BY {c}{r.choice(['', ' DESC'])})")
        elif f < 0.8: aggs.append(f"percentile_disc({r.choice(['0.5', '0.1', '0.75', '0', '1'])}) WITHIN GROUP (ORDER BY {c}{r.choice(['', ' DESC'])})")
        elif f < 0.88: aggs.append(f"quantile_cont({c}, [0.1, 0.5, 0.95])")
        elif f < 0.93: aggs.append(f"quantile_disc({c}, [0.25, 0.75])")
        else: aggs.append(r.choice([f"avg({c})", "count(*)", f"count(DISTINCT {c})", f"min({c})"]))
    if r.random() < 0.08: aggs.append("median(s)")
    g = r.choice(GROUPS)
    frm = "fact"
    if any(x.startswith("dim.") for x in g) or r.random() < 0.15: frm = "fact JOIN dim ON fact.k = dim.k"
    if r.random() < 0.06: frm = "fact LEFT JOIN dim ON fact.k = dim.k AND dim.cat = 1"         # several passes: values
    where = r.choice(["", "", " WHERE d < 6", " WHERE v > 0 AND q BETWEEN 10 AND 30", " WHERE y IS NOT NULL", " WHERE s <> 'b'", " WHERE id < 0"])
    if "dim" in frm and r.random() < 0.5: where += (" AND" if where else " WHERE") + " dim.cat <> 2"
    gb = ""
    if g:
        kind = r.random()
        gb = f" GROUP BY {', '.join(g)}" if kind < 0.85 or len(g) < 2 else r.choice([f" GROUP BY ROLLUP ({', '.join(g)})", f" GROUP BY CUBE ({', '.join(g)})"])
    having = f" HAVING {aggs[0]} IS NOT NULL" if g and r.random() < 0.1 else ""
    sql = f"SELECT {', '.join(g + aggs)} FROM {frm}{where}{gb}{having}"
    sqlexec.RANK_DIRECT = r.choice([1, 10 ** 9])            # 8 bits per query / the whole code at once
    compare(sql, "RANK_AGGS", ordered=False)
sqlexec.RANK_DIRECT = 2048
print(f"   rank selection used for {used.get('rank', 0)} aggregates")

# ------------------------------------------------------------------ COUNT(DISTINCT): a bit per (group, value) cell
print("== COUNT(DISTINCT) from cell bits")
orig_dv = sqlexec._Planner.dist_values
def counting_dv(self, *args):
    used["dist"] = used.get("dist", 0) + 1
    return orig_dv(self, *args)
sqlexec._Planner.dist_values = counting_dv
DCOLS = ["k", "q", "d", "v", "w", "y", "s", "id", "p", "dim.region", "dim.cat"]
for qi in range(NQ):
    aggs = [f"count(DISTINCT {c})" for c in r.sample(DCOLS, r.choice([1, 1, 2]))]
    if r.random() < 0.4: aggs.append(r.choice(["count(*)", "sum(v)", "median(w)", "avg(q)", "count(DISTINCT q) FILTER (WHERE d > 4)", "sum(DISTINCT d)"]))
    g = r.choice(GROUPS + [["dim.cat"], ["k", "s"]])
    frm = "fact JOIN dim ON fact.k = dim.k" if any("dim." in x for x in g + aggs) or r.random() < 0.15 else "fact"
    where = r.choice(["", "", " WHERE d < 6", " WHERE v > 0 AND q BETWEEN 10 AND 30", " WHERE y IS NOT NULL", " WHERE s <> 'b'", " WHERE id < 0"])
    gb = ""
    if g: gb = f" GROUP BY {', '.join(g)}" if r.random() < 0.85 or len(g) < 2 else r.choice([f" GROUP BY ROLLUP ({', '.join(g)})", f" GROUP BY CUBE ({', '.join(g)})"])
    sql = f"SELECT {', '.join(g + aggs)} FROM {frm}{where}{gb}"
    compare(sql, "DIST_BITS", ordered=False)
print(f"   cell bits used for {used.get('dist', 0)} aggregates")

# ------------------------------------------------------------------ aggregates over a join: the sides' partial sums
print("== aggregates over joins from partial sums")
orig_jp = sqlexec.join_preagg
def counting_jp(s, aggs, cx):
    got = orig_jp(s, aggs, cx)
    if got is not None: used["join"] = used.get("join", 0) + 1
    return got
sqlexec.join_preagg = counting_jp
SIDE = ["k", "q", "d", "v", "w", "y", "p", "s", "id"]
def side(name):
    lim = r.choice([1500, 3000, 6000])
    f = r.choice(["", " AND d <> 3", " AND s <> 'a'", " AND y IS NOT NULL", " AND v > 0"])
    form = r.random()
    if form < 0.45: return f"{name} AS (SELECT {', '.join(SIDE)} FROM fact WHERE id < {lim}{f})", name
    if form < 0.6: return f"{name} AS (SELECT k, d, s, y, q % 7 AS q, v + 1 AS v, w, p, id FROM fact WHERE id >= {lim}{f} AND id < {2 * lim})", name
    if form < 0.75: return f"{name} AS (SELECT k, d, s, count(*) AS q, sum(v) AS v, max(w) AS w, min(y) AS y, avg(p) AS p, min(id) AS id FROM fact WHERE id < {4 * lim}{f} GROUP BY k, d, s)", name
    return None, f"(SELECT * FROM fact WHERE id < {lim}{f}) {name}"
JOINS = ["a.k = b.k", "a.k = b.k", "a.d = b.d AND a.s = b.s", "a.y = b.y", "a.s = b.s", "a.q = b.d", "a.k = b.k AND a.v < b.v",
         "a.d = b.d AND a.k + b.k > 40", "a.k = b.k AND a.d = b.d"]
JAGGS = ["count(*)", "count(a.y)", "count(b.y)", "sum(a.v)", "sum(b.w)", "avg(a.q)", "avg(b.y)", "min(a.v)", "max(b.s)",
         "var_pop(a.q)", "stddev_samp(b.v)", "covar_pop(a.q, b.w)", "corr(a.v, b.y)", "regr_slope(a.y, a.w)", "sum(a.q * b.d)",
         "avg(a.v * b.w)", "sum(2 * a.q * b.y)", "count(a.q * b.y)", "count(DISTINCT b.d)", "median(a.v)", "sum(a.p)",
         "sum(DISTINCT a.d)", "sum(a.v) FILTER (WHERE b.q > 25)", "bool_and(a.q > 3)", "max(a.v + b.v)", "sum(a.v + b.w)"]
JGROUPS = [[], ["a.k"], ["b.d"], ["a.s"], ["a.k % 5"], ["a.d", "b.s"], ["a.k + b.d"], ["b.y IS NULL"]]
for qi in range(NQ):
    ca, fa = side("a"); cb, fb = side("b")
    ctes = [c for c in (ca, cb) if c]
    on = r.choice(JOINS)
    kind = r.random()
    frm = f"{fa} JOIN {fb} ON {on}" if kind < 0.7 else f"{fa}, {fb}"
    where = "" if kind < 0.7 else f" WHERE {on}"
    if r.random() < 0.3: where += (" AND" if where else " WHERE") + r.choice([" a.w < 150", " b.q > 10", " a.s <> b.s", " a.d = 2"])
    g = r.choice(JGROUPS)
    aggs = r.sample(JAGGS, r.choice([1, 2, 3, 4]))
    gb = ""
    if g: gb = f" GROUP BY {', '.join(g)}" if r.random() < 0.85 or len(g) < 2 else f" GROUP BY ROLLUP ({', '.join(g)})"
    having = " HAVING count(*) > 3" if g and r.random() < 0.15 else ""
    sql = (f"WITH {', '.join(ctes)} " if ctes else "") + f"SELECT {', '.join(g + aggs)} FROM {frm}{where}{gb}{having}"
    compare(sql, "JOIN_PREAGG", ordered=False, tol=1e-9)
print(f"   partial sums used on {used.get('join', 0)} of {NQ} queries")

# ------------------------------------------------------------------ windows from their aggregated series S(t)
print("== window functions from their aggregated series")
sqlexec.SERIES_STRICT = True                  # an unexpected error in the series path fails the test, not the path
sqlexec.SERIES_ROWS_PER_GROUP = 1             # the series also when its groups are about as many as the rows
orig_series = sqlexec.series_windows
def counting_series(s, wins, cx):
    got = orig_series(s, wins, cx)
    used["series"] = used.get("series", 0) + len(got)
    used["series_trunc"] = used.get("series_trunc", 0) + sum(1 for x in got.values() if x.truncated)
    return got
sqlexec.series_windows = counting_series
WRANK = ["row_number()", "rank()", "dense_rank()", "percent_rank()", "cume_dist()"]
WAGG = ["count(*)", "count(f.y)", "count(f.e)", "sum(f.v)", "sum(f.y)", "sum(f.e)", "avg(f.q)", "avg(f.y)", "avg(f.e)",
        "sum(f.w)", "mean(f.v)"]
WPART = [[], [], ["f.k"], ["f.d"], ["dim.region"], ["f.k", "f.d"], ["f.s"], ["f.y"], ["dim.cat", "f.s"]]
WORDER = [[], ["f.q"], ["f.d"], ["f.w"], ["f.y"], ["f.id"], ["f.id"], ["dim.cat"], ["f.s"], ["f.e"]]
WFRAME = ["", "", "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", "RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
          "RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING", "GROUPS BETWEEN 2 PRECEDING AND CURRENT ROW",
          "GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING", "ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING",
          "RANGE BETWEEN CURRENT ROW AND CURRENT ROW", "GROUPS BETWEEN 1 FOLLOWING AND 3 FOLLOWING",
          "ROWS BETWEEN 2 FOLLOWING AND 1 PRECEDING", "GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING",
          # offsets past every group, up to the largest BIGINT (they must not wrap in 64 bits)
          "GROUPS BETWEEN CURRENT ROW AND 9223372036854775807 FOLLOWING", "GROUPS BETWEEN 9223372036854775807 PRECEDING AND 1 PRECEDING",
          "GROUPS BETWEEN 9223372036854775807 FOLLOWING AND UNBOUNDED FOLLOWING", "ROWS BETWEEN 9223372036854775807 PRECEDING AND CURRENT ROW",
          "GROUPS BETWEEN 1 PRECEDING AND 4611686018427387904 FOLLOWING"]
WWHERE = ["", "", " WHERE f.q > 10", " WHERE dim.region <> 'r1'", " WHERE f.y IS NOT NULL", " WHERE f.w < 100 AND f.d <> 3",
          " WHERE f.q < 0"]                           # no row
WOTHER = ["lag(f.v)", "first_value(f.q)", "min(f.w)", "max(f.y)", "ntile(4)"]
FROMS = ["fact f JOIN dim ON f.k = dim.k", "fact f JOIN dim ON f.k = dim.k",
         "fact f LEFT JOIN dim ON f.k = dim.k AND dim.cat = 1"]
for qi in range(NQ):
    wins, determined = [], True
    part = r.choice(WPART); order = r.choice(WORDER)
    spec = r.choice(["", " DESC"]) + r.choice(["", "", " NULLS FIRST", " NULLS LAST"])
    for wi in range(r.choice([1, 1, 2, 3])):
        p_, o_, sp_ = (part, order, spec) if wi == 0 or r.random() < 0.6 else (r.choice(WPART), r.choice(WORDER), r.choice(["", " DESC"]))
        fn = r.choice(WRANK + WAGG + WAGG + (WOTHER if r.random() < 0.15 else []))
        frame = r.choice(WFRAME) if fn in WAGG and o_ else ""
        over = " ".join(x for x in (f"PARTITION BY {', '.join(p_)}" if p_ else "",
                                     f"ORDER BY {o_[0]}{sp_}" if o_ else "", frame) if x)
        wins.append(f"{fn} OVER ({over})")
        if (fn in ("row_number()", "lag(f.v)", "first_value(f.q)", "ntile(4)") or frame.startswith("ROWS")) and o_ != ["f.id"]:
            determined = False                                # positions among peers: DuckDB may order them otherwise
    frm = r.choice(FROMS); where = r.choice(WWHERE)
    if "LEFT" in frm and "dim." in where: where = ""
    head, tail, ordered = "SELECT f.id, ", "", False
    roll = r.random()
    if order and roll < 0.4:                      # ORDER BY the window's own key, LIMIT: the truncated series
        tail = f" ORDER BY {order[0]}{spec}, f.id LIMIT {r.choice([1, 7, 100, 1000])}" + r.choice(["", "", " OFFSET 5"])
        ordered = True
    elif roll < 0.65:
        tail = " ORDER BY f.id" + r.choice(["", " DESC"]) + r.choice(["", f" LIMIT {r.choice([10, 500])}"])
        ordered = True
    elif roll < 0.68:                             # a window in the ORDER BY
        tail = f" ORDER BY {wins[0]}, f.id LIMIT 50"
        ordered = True
    elif roll < 0.74:                             # a window in the ORDER BY after a fact column (rank selection)
        w2 = r.choice([wins[0], wins[0], "lag(f.v) OVER (ORDER BY f.id)", "count(*) OVER (PARTITION BY f.d)",
                       "min(f.id) OVER (PARTITION BY f.k ORDER BY f.id ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING)"])
        tail = f" ORDER BY f.q{r.choice(['', ' DESC'])}, {w2}{r.choice(['', ' DESC'])}, f.id LIMIT {r.choice([5, 50])}"
        ordered = True
    elif roll < 0.8:                              # QUALIFY on a window
        tail = f" QUALIFY {wins[0]} <= 3"
    elif roll < 0.86:                             # DISTINCT over the windows' values
        head = "SELECT DISTINCT f.k, "
    sql = f"{head}{', '.join(wins)} FROM {frm}{where}{tail}"
    compare(sql, "SERIES_WINDOWS", ordered, duck=determined, tol=1e-9 if not ordered else 0.0)
sqlexec.series_windows = orig_series
print(f"   series used for {used.get('series', 0)} windows, {used.get('series_trunc', 0)} of them truncated by the LIMIT")

print(f"{duck_ok} answers equal to DuckDB")
print("ALL MATCH" if not fails else f"{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
