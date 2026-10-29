# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Official Star Schema Benchmark (O'Neil, O'Neil, Chen, Revilak; SSB spec rev. 3, 2009): data from the official
ssb-dbgen, the 13 official queries verbatim (classic comma-join form), and a DuckDB database built from the .tbl files.
Usage as a module; `python3 ssb_common.py <SF>` generates the data and builds the DuckDB file."""
import os, sys, subprocess, time, duckdb

ROOT = os.environ.get("SSB_DATA", os.path.expanduser("~/torbles_data/ssb"))
DBGEN = os.environ.get("SSB_DBGEN", os.path.expanduser("~/torbles_data/ssb-dbgen"))

QUERIES = {
"Q1.1": """select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date
where lo_orderdate = d_datekey and d_year = 1993 and lo_discount between 1 and 3 and lo_quantity < 25""",
"Q1.2": """select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date
where lo_orderdate = d_datekey and d_yearmonthnum = 199401 and lo_discount between 4 and 6 and lo_quantity between 26 and 35""",
"Q1.3": """select sum(lo_extendedprice*lo_discount) as revenue
from lineorder, date
where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year = 1994 and lo_discount between 5 and 7 and lo_quantity between 26 and 35""",
"Q2.1": """select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA'
group by d_year, p_brand1 order by d_year, p_brand1""",
"Q2.2": """select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand1 between 'MFGR#2221' and 'MFGR#2228' and s_region = 'ASIA'
group by d_year, p_brand1 order by d_year, p_brand1""",
"Q2.3": """select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand1 = 'MFGR#2221' and s_region = 'EUROPE'
group by d_year, p_brand1 order by d_year, p_brand1""",
"Q3.1": """select c_nation, s_nation, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_region = 'ASIA' and s_region = 'ASIA' and d_year >= 1992 and d_year <= 1997
group by c_nation, s_nation, d_year order by d_year asc, revenue desc""",
"Q3.2": """select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_nation = 'UNITED STATES' and s_nation = 'UNITED STATES' and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year order by d_year asc, revenue desc""",
"Q3.3": """select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year order by d_year asc, revenue desc""",
"Q3.4": """select c_city, s_city, d_year, sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_yearmonth = 'Dec1997'
group by c_city, s_city, d_year order by d_year asc, revenue desc""",
"Q4.1": """select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit
from date, customer, supplier, part, lineorder
where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
group by d_year, c_nation order by d_year, c_nation""",
"Q4.2": """select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as profit
from date, customer, supplier, part, lineorder
where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_region = 'AMERICA' and (d_year = 1997 or d_year = 1998) and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
group by d_year, s_nation, p_category order by d_year, s_nation, p_category""",
"Q4.3": """select d_year, s_city, p_brand1, sum(lo_revenue - lo_supplycost) as profit
from date, customer, supplier, part, lineorder
where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_nation = 'UNITED STATES' and (d_year = 1997 or d_year = 1998) and p_category = 'MFGR#14'
group by d_year, s_city, p_brand1 order by d_year, s_city, p_brand1""",
}
# Not part of SSB: predicates on the official keys themselves (a key range, a date-key range on the fact column, a key
# point lookup), to show which key designs keep them algebraic.
EXTRA = {
"K1": """select c_region, sum(lo_revenue) as revenue
from lineorder, customer
where lo_custkey = c_custkey and c_custkey between 10000 and 15000
group by c_region order by c_region""",
"K2": """select sum(lo_revenue) as revenue
from lineorder
where lo_orderdate between 19950101 and 19950331""",
"K3": """select d_year, sum(lo_revenue) as revenue
from lineorder, supplier, date
where lo_suppkey = s_suppkey and lo_orderdate = d_datekey and s_suppkey = 77
group by d_year order by d_year""",
# M1-M3 (not SSB): unfiltered GROUP BY over every fact row, where a group holds many rows of every tile (dense M)
"M1": """select d_year, c_region, sum(lo_revenue) as revenue
from lineorder, date, customer
where lo_orderdate = d_datekey and lo_custkey = c_custkey
group by d_year, c_region order by d_year, c_region""",
"M2": """select c_nation, s_region, sum(lo_revenue - lo_supplycost) as profit
from lineorder, customer, supplier
where lo_custkey = c_custkey and lo_suppkey = s_suppkey
group by c_nation, s_region order by c_nation, s_region""",
"M3": """select d_year, lo_discount, sum(lo_extendedprice * lo_discount) as revenue
from lineorder, date
where lo_orderdate = d_datekey
group by d_year, lo_discount order by d_year, lo_discount""",
}
# the GROUP BY columns of each query (the result is compared as {group tuple: aggregate})
GROUPS = {"Q1.1": [], "Q1.2": [], "Q1.3": [],
          "Q2.1": ["d_year", "p_brand1"], "Q2.2": ["d_year", "p_brand1"], "Q2.3": ["d_year", "p_brand1"],
          "Q3.1": ["c_nation", "s_nation", "d_year"], "Q3.2": ["c_city", "s_city", "d_year"],
          "Q3.3": ["c_city", "s_city", "d_year"], "Q3.4": ["c_city", "s_city", "d_year"],
          "Q4.1": ["d_year", "c_nation"], "Q4.2": ["d_year", "s_nation", "p_category"], "Q4.3": ["d_year", "s_city", "p_brand1"],
          "K1": ["c_region"], "K2": [], "K3": ["d_year"],
          "M1": ["d_year", "c_region"], "M2": ["c_nation", "s_region"], "M3": ["d_year", "lo_discount"]}

SCHEMA = {
 "customer": [("c_custkey", "INTEGER"), ("c_name", "VARCHAR"), ("c_address", "VARCHAR"), ("c_city", "VARCHAR"), ("c_nation", "VARCHAR"),
              ("c_region", "VARCHAR"), ("c_phone", "VARCHAR"), ("c_mktsegment", "VARCHAR")],
 "part": [("p_partkey", "INTEGER"), ("p_name", "VARCHAR"), ("p_mfgr", "VARCHAR"), ("p_category", "VARCHAR"), ("p_brand1", "VARCHAR"),
          ("p_color", "VARCHAR"), ("p_type", "VARCHAR"), ("p_size", "INTEGER"), ("p_container", "VARCHAR")],
 "supplier": [("s_suppkey", "INTEGER"), ("s_name", "VARCHAR"), ("s_address", "VARCHAR"), ("s_city", "VARCHAR"), ("s_nation", "VARCHAR"),
              ("s_region", "VARCHAR"), ("s_phone", "VARCHAR")],
 "date": [("d_datekey", "INTEGER"), ("d_date", "VARCHAR"), ("d_dayofweek", "VARCHAR"), ("d_month", "VARCHAR"), ("d_year", "INTEGER"),
          ("d_yearmonthnum", "INTEGER"), ("d_yearmonth", "VARCHAR"), ("d_daynuminweek", "INTEGER"), ("d_daynuminmonth", "INTEGER"),
          ("d_daynuminyear", "INTEGER"), ("d_monthnuminyear", "INTEGER"), ("d_weeknuminyear", "INTEGER"), ("d_sellingseason", "VARCHAR"),
          ("d_lastdayinweekfl", "INTEGER"), ("d_lastdayinmonthfl", "INTEGER"), ("d_holidayfl", "INTEGER"), ("d_weekdayfl", "INTEGER")],
 "lineorder": [("lo_orderkey", "INTEGER"), ("lo_linenumber", "INTEGER"), ("lo_custkey", "INTEGER"), ("lo_partkey", "INTEGER"),
               ("lo_suppkey", "INTEGER"), ("lo_orderdate", "INTEGER"), ("lo_orderpriority", "VARCHAR"), ("lo_shippriority", "INTEGER"),
               ("lo_quantity", "INTEGER"), ("lo_extendedprice", "INTEGER"), ("lo_ordtotalprice", "INTEGER"), ("lo_discount", "INTEGER"),
               ("lo_revenue", "INTEGER"), ("lo_supplycost", "INTEGER"), ("lo_tax", "INTEGER"), ("lo_commitdate", "INTEGER"),
               ("lo_shipmode", "VARCHAR")],
}
FACT_COLS = ["lo_custkey", "lo_partkey", "lo_suppkey", "lo_orderdate", "lo_quantity", "lo_extendedprice", "lo_discount",
             "lo_revenue", "lo_supplycost"]          # the columns the 13 queries read

def datadir(sf): return os.path.join(ROOT, f"sf{sf}")
def dbfile(sf): return os.path.join(datadir(sf), "ssb.duckdb")

TABLES = ("customer", "part", "supplier", "date", "lineorder")
def complete(sf):
    """True if the DuckDB database exists and holds every SSB table (a leftover of an interrupted run does not)"""
    if not os.path.exists(dbfile(sf)): return False
    try:
        con = duckdb.connect(dbfile(sf), read_only=True)
        have = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}; con.close()
        return set(SCHEMA) <= have
    except Exception: return False

def generate(sf):
    d = datadir(sf); os.makedirs(d, exist_ok=True)
    if complete(sf): return 0.0                                    # a complete database exists: nothing to generate
    for f in (dbfile(sf), dbfile(sf) + ".wal"):                    # an incomplete one is removed and rebuilt
        if os.path.exists(f): os.remove(f)
    for f in ("dbgen", "dists.dss"): subprocess.run(["cp", os.path.join(DBGEN, f), d], check=True)
    # dbgen can create its files with an undefined mode (it reads an uninitialised stat when a file is new), so the
    # files are created beforehand with normal permissions, and set again afterwards
    for t in TABLES:
        p = os.path.join(d, f"{t}.tbl")
        if os.path.exists(p): os.chmod(p, 0o644); os.remove(p)
        open(p, "w").close(); os.chmod(p, 0o644)
    t0 = time.perf_counter()
    subprocess.run(["./dbgen", "-s", str(sf), "-T", "a", "-f"], cwd=d, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for t in TABLES: os.chmod(os.path.join(d, f"{t}.tbl"), 0o644)
    return time.perf_counter() - t0

def build_duckdb(sf):
    """the official schema (all columns, strings as VARCHAR) loaded from the dbgen .tbl files"""
    path = dbfile(sf)
    if os.path.exists(path): return 0.0
    tmp = path + ".building"                                       # renamed only when complete
    for f in (tmp, tmp + ".wal"):
        if os.path.exists(f): os.remove(f)
    con = duckdb.connect(tmp); t0 = time.perf_counter()
    for t, cols in SCHEMA.items():
        decl = ", ".join(f"'{c}': '{ty}'" for c, ty in cols) + ", 'trailing': 'VARCHAR'"
        names = ", ".join(c for c, _ in cols)
        con.execute(f"CREATE TABLE {t} AS SELECT {names} FROM read_csv('{os.path.join(datadir(sf), t + '.tbl')}', "
                    f"delim='|', header=false, quote='', columns={{{decl}}})")
    con.close(); os.replace(tmp, path)
    return time.perf_counter() - t0

if __name__ == "__main__":
    sf = int(sys.argv[1])
    tg = generate(sf); tl = build_duckdb(sf)
    con = duckdb.connect(dbfile(sf), read_only=True)
    print(f"SF{sf}: generated in {tg:.0f}s, DuckDB load {tl:.0f}s;",
          {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in SCHEMA})
