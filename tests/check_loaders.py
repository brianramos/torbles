# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The ways in: a DuckDB database with declared PRIMARY KEY / FOREIGN KEY constraints (read through
INFORMATION_SCHEMA), a DDL script with CSV and Parquet files, a Frictionless Data Package, data frames with keys given
as arguments, and opt-in key inference. Each gives the same answers as DuckDB. And the refusals: NULLs (unless
filled), a primary key with duplicates, unsupported column types (skipped with a note), too many columns."""
import json, os, shutil, tempfile, numpy as np, pandas as pd, duckdb
import torbles
rng = np.random.default_rng(3)
fails = 0
def check(cond, what):
    global fails
    fails += not cond; print(f"  {what:66s} {'ok' if cond else 'FAIL'}")
n, m = 400, 20_000
reg = rng.integers(0, 4, n)
store = pd.DataFrame(dict(store_id=np.arange(1, n + 1), region=[f"R{x}" for x in reg], city=[f"R{x}-C{y}" for x, y in zip(reg, rng.integers(0, 6, n))]))
sales = pd.DataFrame(dict(sale_id=np.arange(m), store_id=rng.integers(1, n + 1, m), amount=np.round(rng.uniform(0, 500, m), 2), units=rng.integers(1, 9, m)))
def answer(db):
    s, st = db["sales"], db["store"]
    q = db.query(s).join(st).where(st.region == "R2").group_by(st.city).sum(s.amount * s.units)
    return q, q.run()
tmp = tempfile.mkdtemp(prefix="torbles-load-")
try:
    # 1. a DuckDB database that declares its keys
    path = os.path.join(tmp, "shop.duckdb"); con = duckdb.connect(path)
    con.execute("CREATE TABLE store (store_id INTEGER PRIMARY KEY, region VARCHAR, city VARCHAR)")
    con.execute("CREATE TABLE sales (sale_id BIGINT PRIMARY KEY, store_id INTEGER REFERENCES store(store_id), amount DECIMAL(8,2), units INTEGER)")
    con.register("s1", store); con.execute("INSERT INTO store SELECT * FROM s1")
    con.register("s2", sales); con.execute("INSERT INTO sales SELECT * FROM s2"); con.close()
    db1 = torbles.from_duckdb(path)
    q, r1 = answer(db1); want = db1.sql_result(q)
    check(r1 == want, "DuckDB file, keys declared in the database (INFORMATION_SCHEMA)")
    check(db1.meta["sales"]["foreign_keys"] == {"store_id": "store.store_id"}, "the declared foreign key was read")
    # 2. DDL + CSV and Parquet files
    store.to_csv(os.path.join(tmp, "store.csv"), index=False)
    c = duckdb.connect(); c.register("s", sales); c.execute(f"COPY (SELECT * FROM s) TO '{os.path.join(tmp, 'sales.parquet')}' (FORMAT PARQUET)"); c.close()
    ddl = os.path.join(tmp, "schema.sql")
    open(ddl, "w").write("CREATE TABLE store (store_id INTEGER PRIMARY KEY, region VARCHAR, city VARCHAR);\n"
                         "CREATE TABLE sales (sale_id BIGINT, store_id INTEGER REFERENCES store (store_id), amount DECIMAL(8,2), units INTEGER);\n")
    db2 = torbles.from_ddl(ddl, {"store": os.path.join(tmp, "store.csv"), "sales": os.path.join(tmp, "sales.parquet")})
    check(answer(db2)[1] == want, "DDL script with a CSV and a Parquet file")
    check(db2.meta["sales"]["columns"]["amount"]["type"].startswith("DECIMAL(8,2)"), "columns cast to the declared types")
    # 3. a Frictionless Data Package
    pkg = dict(name="shop", resources=[
        dict(name="store", path="store.csv", schema=dict(fields=[dict(name="store_id", type="integer"), dict(name="region", type="string"),
                                                                 dict(name="city", type="string")], primaryKey="store_id")),
        dict(name="sales", path="sales.parquet", schema=dict(fields=[dict(name="sale_id", type="integer"), dict(name="store_id", type="integer"),
                                                                     dict(name="amount", type="number"), dict(name="units", type="integer")],
                                                             foreignKeys=[dict(fields="store_id", reference=dict(resource="store", fields="store_id"))]))])
    json.dump(pkg, open(os.path.join(tmp, "datapackage.json"), "w"))
    db3 = torbles.from_frictionless(os.path.join(tmp, "datapackage.json"), types={"sales.amount": "DECIMAL(8,2)"})
    check(answer(db3)[1] == want, "Frictionless Data Package (Table Schema keys), amount loaded as DECIMAL")
    db3b = torbles.from_frictionless(os.path.join(tmp, "datapackage.json"))
    try: answer(db3b); check(False, "sum over a DOUBLE refused")
    except TypeError as e: check("DECIMAL" in str(e), "a sum over a DOUBLE column is refused, with the types= hint")
    # 4. data frames, keys as arguments; and key inference (opt-in, reported)
    db4 = torbles.from_frames({"store": store, "sales": sales}, primary_keys={"store": "store_id"}, foreign_keys={"sales.store_id": "store"},
                              types={"sales.amount": "DECIMAL(8,2)"})
    check(answer(db4)[1] == want, "data frames with keys as arguments (a float column loaded as DECIMAL)")
    db5 = torbles.Database(); db5.add_table("store", store); db5.add_table("sales", sales); db5.build(infer_keys=True, types={"sales.amount": "DECIMAL(8,2)"})
    check(db5.meta["sales"]["foreign_keys"] == {"store_id": "store.store_id"} and any("inferred" in x for x in db5.notes),
          f"inferred keys are reported: {[x for x in db5.notes if 'inferred' in x]}")
    db6 = torbles.Database(); db6.add_table("store", store); db6.add_table("sales", sales); db6.build()
    check(not db6.meta["sales"]["foreign_keys"], "without declarations or infer_keys, no keys are guessed")
    # 5. refusals and notes
    nul = sales.copy().astype({"units": "float"}); nul.loc[5, "units"] = None
    try: torbles.from_frames({"store": store, "sales": nul}, primary_keys={"store": "store_id"}, foreign_keys={"sales.store_id": "store"}); check(False, "NULLs refused")
    except torbles.LoadError as e: check("NULL" in str(e) and "fill_nulls" in str(e), "NULL values refused, with the fill_nulls hint")
    db7 = torbles.from_frames({"store": store, "sales": nul}, primary_keys={"store": "store_id"}, foreign_keys={"sales.store_id": "store"}, fill_nulls={"sales.units": 0},
                              types={"sales.units": "INTEGER", "sales.amount": "DECIMAL(8,2)"})
    q7 = db7.query(db7["sales"]).sum(db7["sales"].units)
    check(q7.run() == int(sales.units.sum()) - int(sales.units[5]), "fill_nulls replaces them")
    dup = pd.concat([store, store.iloc[:3]])
    try: torbles.from_frames({"store": dup, "sales": sales}, primary_keys={"store": "store_id"}, foreign_keys={"sales.store_id": "store"}); check(False, "duplicate key refused")
    except torbles.LoadError as e: check("duplicate" in str(e), "a primary key with duplicates is refused")
    odd = store.copy(); odd["blob"] = [b"x"] * n
    db8 = torbles.from_frames({"store": odd, "sales": sales}, primary_keys={"store": "store_id"}, foreign_keys={"sales.store_id": "store"})
    check(any("blob" in x for x in db8.notes) and "blob" not in db8["store"].cols, "an unsupported column type is skipped with a note")
    wide = pd.DataFrame({f"c{i}": rng.integers(0, 9, 10) for i in range(40)})
    try: torbles.from_frames({"w": wide}); check(False, "too many columns refused")
    except torbles.LoadError as e: check("columns=" in str(e), "more columns than a torble holds: refused, with the columns= hint")
    dbw = torbles.Database(); dbw.add_table("w", wide, columns=["c1", "c7"]); dbw.build()
    check(dbw["w"].cols == ["c1", "c7"], "columns= loads a subset")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("ALL MATCH" if not fails else f"{fails} FAILURES")
