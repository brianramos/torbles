# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Tables wider than one torble (0.4.0). A torble holds 31 columns; a wider table keeps its columns in row-aligned
parts, each a torble, and a plan reads the columns it needs through a view of them (the parts' own planes, read at the
same 64-row word); a filter on more columns than a view holds is evaluated in groups, as row bitmaps ANDed word for
word. The loader first moves the columns that one column determines into a dimension of their own, stored once per
value of it, read through the join substitution. Every answer is compared with numpy (the engine API on plain tables)
or DuckDB (typed tables, SQL and the Python API), on the CPU engine, the GPU executor (or its emulation) and the plain
baseline, before and after save / load; and the detection of the determined columns on its own."""
import os, shutil, sys, tempfile
import numpy as np, pandas as pd
import torbles, algq
from algq import Table, Query, Const
from database import Database, fd_groups, load

fails = 0
def check(ok, what, detail=""):
    global fails
    print(f"  {what:<78} {'ok' if ok else 'FAIL'}")
    if not ok:
        fails += 1
        if detail: print("      " + str(detail)[:600])

ENGINES = ("alg", "gpu", "plain")
rng = np.random.default_rng(2026)
tmp = tempfile.mkdtemp(prefix="torbles-many-")

def groupsum(keys, vals, mask):
    out = {}
    for k, v in zip(zip(*[a[mask] for a in keys]), vals[mask]):
        k = tuple(int(x) for x in k); out[k] = out.get(k, 0) + int(v)
    return out

def norm(rows):
    out = []
    for r in rows:
        rr = []
        for v in r:
            if hasattr(v, "item") and not isinstance(v, (str, bytes)):
                try: v = v.item()
                except Exception: pass
            if type(v).__name__ == "Decimal": v = float(v)
            if isinstance(v, float): v = round(v, 6)
            rr.append(v)
        out.append(tuple(rr))
    return out

def sql_cases(db, title, queries, engines=ENGINES):
    bad = []
    for q in queries:
        want = norm(db.con.execute(q).fetchall())
        for e in engines:
            try: got = norm(db.sql(q, engine=e).rows)
            except Exception as ex: got = f"{type(ex).__name__}: {ex}"
            if got != want: bad.append(f"[{e}] {q}\n        torbles {str(got)[:200]}\n        DuckDB  {str(want)[:200]}")
    check(not bad, f"{title}: {len(queries)} queries x {len(engines)} engines", "\n      ".join(bad[:3]))

try:
    # ------------------------------------------------------------------ the engine API, plain tables
    print("== the engine API: a table of 75 columns in parts, a dimension of 45")
    n, K = 6000, 75
    cols = {f"c{i}": rng.integers(0, 3 + i % 11, n).astype(np.uint32) for i in range(K)}
    cols["fk"] = rng.integers(0, 50, n).astype(np.uint32)
    big = rng.integers(0, 1 << 40, n).astype(np.uint64); big[::7] = rng.integers(0, 50, len(big[::7]))
    cols["big"] = big
    dn = 50
    dcols = {"id": np.arange(dn, dtype=np.uint32)}
    dcols.update({f"a{i}": rng.integers(0, 2 + i % 5, dn).astype(np.uint32) for i in range(44)})
    D = Table("d", dcols)
    T = Table("t", cols, refs={"fk": D}, extend={"big": 32})
    check(T.parts is not None and len(T.parts) == 3 and D.parts is not None and len(D.parts) == 2,
          f"wider than a torble: {len(T.cols)} physical columns in {len(T.parts)} parts, the dimension in {len(D.parts)}")
    check(T._part_of("_refok") is T.parts[0] and T._part_of("fk") is T.parts[0] and
          T._part_of("big#hi") is T._part_of("big") is T._part_of("big#x"),
          "the references' bits and keys in the first part, a column with its hidden ones")
    try: T.at; ok = False
    except TypeError: ok = True
    check(ok, "a table in parts has no single torble: its columns are read through views")
    for eng in ENGINES:
        m = (cols["c3"] == 1) & (cols["c40"] >= 2) & (cols["c70"] < 5)
        r1 = Query(T).where((T.c3 == 1) & (T.c40 >= 2) & (T.c70 < 5)).count().run(eng)
        m2 = cols["c50"] != 0
        r2 = Query(T).where(T.c50 != 0).group_by(T.c1, T.c45).sum(T.c62).run(eng)
        r3 = Query(T).sum(T.c2 * 3 + T.c33 - T.c66).run(eng)
        p = None
        for i in range(64):
            t_ = getattr(T, f"c{i}") < 9
            p = t_ if p is None else (p & t_)
        m4 = np.ones(n, bool)
        for i in range(64): m4 &= cols[f"c{i}"] < 9
        r4 = Query(T).where(p).group_by(T.c0).sum(T.c74).run(eng)
        dsel = (dcols["a3"] == 1) | (dcols["a40"] == 2)
        m5 = dsel[cols["fk"]] & (cols["c60"] > 2)
        r5 = Query(T).join(D).where(((D.a3 == 1) | (D.a40 == 2)) & (T.c60 > 2)).group_by(D.a43).sum(T.c72).run(eng)
        m6 = cols["c10"] < 4
        r6 = Query(T).where(T.c10 < 4).sum(T.big).run(eng)
        check(r1 == int(m.sum()) and {k: int(v) for k, v in r2.items()} == groupsum([cols["c1"], cols["c45"]], cols["c62"], m2)
              and r3 == int((cols["c2"].astype(np.int64) * 3 + cols["c33"] - cols["c66"].astype(np.int64)).sum())
              and {(k if isinstance(k, tuple) else (k,)): int(v) for k, v in r4.items()} == groupsum([cols["c0"]], cols["c74"], m4)
              and {(k if isinstance(k, tuple) else (k,)): int(v) for k, v in r5.items()} == groupsum([dcols["a43"][cols["fk"]]], cols["c72"], m5)
              and r6 == int(big[m6].astype(object).sum()),
              f"[{eng}] filters, GROUP BY, sums across parts; 64 filter columns; a wide dimension; 64-bit values",
              (r1, int(m.sum()), r3, r6))
    hier = Table("h", {"id": np.arange(64, dtype=np.uint32), "lvl": (np.arange(64) // 8).astype(np.uint32),
                       "odd": (np.arange(64) * 37 % 5).astype(np.uint32)})
    F = Table("f", {"k": cols["fk"] + 7, "x": cols["c4"], "y": cols["c9"]}, msb=("k",), refs={"k": hier})
    k_ = cols["fk"] + 7
    want = int(cols["c9"][((k_ // 8) == 3) | (cols["c4"] == 1) | (((k_ * 37 % 5) == 2) & (cols["c4"] > 2))].astype(np.int64).sum())
    got = {e: Query(F).join(hier).where((hier.lvl == 3) | (F.x == 1) | ((hier.odd == 2) & (F.x > 2))).sum(F.y).run(e) for e in ENGINES}
    check(all(v == want for v in got.values()),
          "an OR across a fact and its dimension: the dimension's part as its polynomial on the key's planes", (got, want))
    q = Query(T).where(T.c5 == 2).group_by(T.c66).sum(T.c35)
    a = q.run(); nv = len(T._views); b = q.run()
    check(a == b and len(T._views) == nv, "a query run again: the same view")
    check(0 < nv <= algq.VIEW_CACHE and all(len(v.cols) <= algq.MAXCOL for v in T._views.values()),
          f"views of at most {algq.MAXCOL} columns, cached ({nv})")
    check(T.nplanes() == sum(p.nplanes() for p in T.parts) and T.torus_bytes() == sum(p.torus_bytes() for p in T.parts),
          "planes and bytes: the parts'")
    T.flip(T.parts[0].nplanes() + 5, 17)
    nbad, rows = T.validate()
    fixed = T.repair()
    check(nbad >= 1 and 17 in rows and fixed >= 1 and T.validate()[0] == 0, "a flipped bit in a side part: found and repaired")
    T.close()
    try: q.run(); ok = False
    except algq.TableClosedError: ok = True
    check(ok and T.closed and all(p.closed for p in T.parts), "closed: the parts and views released, a later run refused")

    # ------------------------------------------------------------------ the detection of determined columns
    print("== columns that one column determines")
    k = np.repeat(np.arange(100, dtype=np.uint32), 40); rng.shuffle(k)
    f = {"k": k, "a": (k * 7 % 13).astype(np.uint32), "b": (k // 3).astype(np.uint32), "c": rng.integers(0, 5, len(k)).astype(np.uint32),
         "v": (k % 4).astype(np.uint32), "v#v": (k % 5 != 0).astype(np.uint32), "near": (k % 9).astype(np.uint32), "u": np.arange(len(k), dtype=np.uint32)}
    f["near"][123] = (f["near"][123] + 1) % 9
    g = fd_groups(f, {"v": "v#v"}, ["k", "a", "b", "c", "v", "near", "u"], len(k))
    check("k" in g and sorted(g["k"][0]) == ["a", "b", "v"] and g["k"][1] == 100,
          "k determines a, b and a nullable column; not a column differing on one row", g)
    check("u" not in g and "c" not in g, "no determinant with more values than half the rows, or with fewer than two columns", g)
    g2 = fd_groups(f, {"v": "v#v", "k": "v#v"}, ["k", "a", "b"], len(k))
    check("k" not in g2, "a column with NULLs determines nothing", g2)

    # ------------------------------------------------------------------ SQL: row-aligned parts
    print("== SQL: a fact of 62 columns in parts, a dimension of 40")
    n = 3000
    dim = pd.DataFrame({"d_id": np.arange(40), "d_name": [f"name{i % 7}" for i in range(40)], "d_grp": np.arange(40) % 5})
    for i in range(37): dim[f"d_x{i}"] = rng.integers(0, 3 + i, 40)
    f = {"f_id": np.arange(n), "f_d": rng.integers(0, 40, n)}
    for i in range(26): f[f"i{i}"] = rng.integers(-50, 50 + i, n)
    for i in range(10): f[f"s{i}"] = [f"v{x}" for x in rng.integers(0, 5 + i, n)]
    for i in range(8):
        v = rng.integers(0, 100, n).astype(float); v[rng.random(n) < 0.2] = np.nan
        f[f"n{i}"] = pd.array(np.where(np.isnan(v), None, v), dtype="Int64")
    f["big"] = rng.integers(-(1 << 60), 1 << 60, n)
    f["dt"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D")
    f["x"] = rng.random(n)
    db = Database()
    db.add_table("dim", dim, primary_key="d_id")
    db.add_table("fact", pd.DataFrame(f), primary_key="f_id", foreign_keys={"f_d": "dim"})
    db.build(keep_plain=True)
    t = db["fact"]
    check(t.parts is not None and len(t.parts) == 2 and db["dim"].parts is not None and not t.moved,
          f"{len(t.cols)} physical columns of the fact in {len(t.parts)} parts; the dimension in {len(db['dim'].parts)}")
    Q = [
        "SELECT count(*) FROM fact WHERE i3 > 10 AND s7 = 'v3' AND n6 IS NOT NULL",
        "SELECT s9, sum(i25), count(n7), avg(n0) FROM fact WHERE dt >= DATE '2021-01-01' GROUP BY s9 ORDER BY s9",
        "SELECT d_name, sum(i20 * 2 + i1), max(n5) FROM fact JOIN dim ON f_d = d_id WHERE s8 IN ('v1', 'v2') GROUP BY d_name ORDER BY d_name",
        "SELECT d_x36, d_x2, count(*) FROM fact JOIN dim ON f_d = d_id WHERE d_x30 > 3 AND d_x1 < 2 GROUP BY 1, 2 ORDER BY 1, 2",
        "SELECT sum(big), min(big), max(big) FROM fact WHERE i24 < 0",
        "SELECT f_id, s5, n3, big FROM fact WHERE i10 = 5 AND s1 <> 'v0' ORDER BY f_id LIMIT 12",
        "SELECT count(*) FROM fact WHERE " + " AND ".join(f"i{i} > -48" for i in range(26)) + " AND "
        + " AND ".join(f"s{i} <> 'zz'" for i in range(10)),
        "SELECT d_grp, count(DISTINCT s6), sum(i13) FROM fact, dim WHERE f_d = d_id AND n1 > 20 GROUP BY d_grp ORDER BY d_grp",
        "SELECT * FROM fact WHERE f_id IN (7, 1999) ORDER BY f_id",
        "SELECT s2, row_number() OVER (PARTITION BY s2 ORDER BY f_id) AS r, i9 FROM fact WHERE f_id < 30 ORDER BY f_id",
    ]
    sql_cases(db, "parts", Q)
    path = os.path.join(tmp, "parts"); db.save(path)
    db2 = load(path, con=db.con)
    check([p.cols for p in db2["fact"].parts] == [p.cols for p in t.parts] and db2.integrity["fact"]["rows_with_faults"] == 0,
          "saved and loaded: one file per part, every face checked")
    sql_cases(db2, "parts, loaded", Q, ("alg", "gpu"))
    db2.close()

    # ------------------------------------------------------------------ SQL: groups moved to dimensions of their own
    print("== SQL: a denormalized fact, its determined columns in dimensions of their own")
    n, NC, NP = 4000, 300, 120
    cid = rng.integers(0, NC, n); pid = rng.integers(0, NP, n)
    city = rng.integers(0, 40, NC); nation = city // 4; region = nation // 3
    f = {"o_id": np.arange(n), "cust": cid, "c_name": [f"Customer#{c:04d}" for c in cid], "c_city": [f"city{city[c]}" for c in cid],
         "c_nation": [f"nation{nation[c]}" for c in cid], "c_region": [f"region{region[c]}" for c in cid],
         "c_seg": [["AUTO", "BUILD", None][c % 3] for c in cid], "c_bal": np.round((cid * 37 % 1000) / 7, 2),
         "prod": pid, "p_brand": [f"Brand#{p % 25}" for p in pid], "p_cat": [f"cat{p % 25 // 5}" for p in pid],
         "p_size": pid % 50, "p_price": (pid * 13) % 2000 + 100, "p_near": pid % 7}
    for i in range(40): f[f"m{i}"] = rng.integers(0, 100 + i, n)
    f["qty"] = rng.integers(1, 50, n)
    f["p_near"][5] = (f["p_near"][5] + 1) % 7                  # not a function of prod on one row: stays in the table
    db = Database()
    db.add_table("sales", pd.DataFrame(f), primary_key="o_id")
    db.build(keep_plain=True)
    t = db["sales"]
    det = all(db.con.execute(f'SELECT count(DISTINCT ("{d}", "{c}")) = count(DISTINCT "{d}") FROM sales').fetchone()[0]
              for c, d in t.moved.items())
    check(len(t.groups) >= 2 and {"c_region", "p_cat", "c_seg"} <= set(t.moved) and det and set(t.moved.values()) == set(t.groups)
          and "p_near" not in t.moved and "p_near" in t.cols and t.parts is not None,
          f"{len(t.groups)} groups moved ({len(t.moved)} columns, each determined), a near-dependency kept, the rest in {len(t.parts)} parts",
          (t.groups, t.moved))
    D = t.moved["c_seg"]; G = db[t.groups[D]]
    sizes = [db[g].n for g in t.groups.values()]                 # in the order they were made
    check(G.n == len(set(f[D])) and getattr(G, "generated", None) == ("sales", D) and "c_seg#v" in G.cols and sizes == sorted(sizes),
          f"a group: one row per value of {D} ({G.n}), its NULLs' validity plane with it; the smallest groups first {sizes}")
    Q = [
        "SELECT c_region, sum(qty), count(*) FROM sales GROUP BY c_region ORDER BY c_region",
        "SELECT c_nation, p_cat, sum(m3 * qty) FROM sales WHERE c_region = 'region1' AND p_size < 20 GROUP BY c_nation, p_cat ORDER BY 1, 2",
        "SELECT cust, c_name, count(*) FROM sales WHERE c_city IN ('city3', 'city7') GROUP BY cust, c_name ORDER BY cust",
        "SELECT count(*), sum(c_bal), avg(p_price) FROM sales WHERE c_seg IS NULL",
        "SELECT c_seg, count(*) FROM sales GROUP BY c_seg ORDER BY c_seg NULLS FIRST",
        "SELECT * FROM sales WHERE o_id IN (3, 77, 3999) ORDER BY o_id",
        "SELECT max(c_name), min(p_brand), sum(p_size), sum(m39), sum(p_near) FROM sales WHERE m0 > 50",
        "SELECT p_brand, sum(p_price * qty) AS rev FROM sales GROUP BY p_brand ORDER BY rev DESC, p_brand LIMIT 5",
        "SELECT cust, prod FROM sales WHERE cust < 3 AND prod < 10 ORDER BY o_id",
        "SELECT c_region, rank() OVER (ORDER BY sum(qty) DESC) FROM sales GROUP BY c_region ORDER BY c_region",
        "SELECT count(DISTINCT c_city), count(DISTINCT cust) FROM sales WHERE p_cat = 'cat2' AND m30 < 60",
        "SELECT a.c_city, count(*) FROM sales a JOIN sales b ON a.o_id = b.o_id + 1 WHERE a.c_nation = b.c_nation GROUP BY a.c_city ORDER BY 1",
        "SELECT c_nation, p_near, sum(m35) FROM sales WHERE p_brand LIKE 'Brand#1%' GROUP BY ALL ORDER BY 1, 2",
    ]
    sql_cases(db, "groups and parts", Q)
    q = db.query("sales").where(t.c_region == "region2").group_by(t.c_nation, t.p_cat).sum(t.qty * t.p_price)
    want = {(a, b): v for a, b, v in db.con.execute("SELECT c_nation, p_cat, sum(qty * p_price) FROM sales "
                                                    "WHERE c_region = 'region2' GROUP BY 1, 2").fetchall()}
    got = {e: q.run(e) for e in ENGINES}
    check(all(g_ == want for g_ in got.values()) and "sales#" not in q.sql() and db.verify(q)[0],
          "the Python API reads moved columns as the table's (their dimensions joined by themselves)", (got, want))
    path = os.path.join(tmp, "groups"); db.save(path)
    db2 = load(path, con=db.con)
    check(db2["sales"].moved == t.moved and db2["sales"].groups == t.groups, "saved and loaded with their groups")
    sql_cases(db2, "groups and parts, loaded", Q, ("alg", "gpu"))
    db2.close()
    db3 = Database(db.con); db3.add_table("sales"); db3.build(groups=False)
    check(not db3["sales"].moved and db3["sales"].parts is not None and len(db3["sales"].parts) >= 2, "groups=False: parts only")
    sql_cases(db3, "parts only", Q[:6], ("alg",))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("ALL MATCH" if not fails else f"{fails} FAILURES")
