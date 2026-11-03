# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""MIN / MAX by plane elimination (Query.minmax) and passing-row bitmaps (Query.rows), on every engine (planes, plain
columns, GPU kernels), 1 and 2 threads, against numpy: no GROUP BY, GROUP BY a fact column, a dimension attribute read
by lookup, a key prefix; filters on the fact and on the dimension; codes that span 32 bits."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torbles, algq
from torbles import Table, Query

rng = np.random.default_rng(5)
nd, nf = 300, 70_000
dim = Table("dim", dict(id=np.arange(nd, dtype=np.uint32), grp=rng.integers(0, 9, nd).astype(np.uint32),
                        reg=(np.arange(nd) // 40).astype(np.uint32)), msb=())
fk = rng.integers(0, nd, nf).astype(np.uint32)
cols = dict(fk=fk, x=rng.integers(0, 2**32 - 1, nf, dtype=np.uint64).astype(np.uint32), y=rng.integers(0, 1000, nf).astype(np.uint32),
            g=rng.integers(0, 6, nf).astype(np.uint32), s=np.sort(rng.integers(0, 50, nf)).astype(np.uint32))
fact = Table("fact", cols, msb=("fk",), refs={"fk": dim}, cluster="s")
fails = 0
def want_mm(mask, key=None):
    if key is None:
        v = cols["x"][mask]; return (int(v.min()), int(v.max())) if len(v) else None
    out = {}
    for k in np.unique(key[mask]):
        v = cols["x"][mask & (key == k)]; out[(int(k),)] = (int(v.min()), int(v.max()))
    return out
# The only selected values can equal the CUDA reduction's sentinel or lie in a partial word.
edge = Table("edge", {"x": np.array([0] * 64 + [2**32 - 1], np.uint32),
                      "last": np.array([0] * 64 + [1], np.uint32)}, msb=("x",))
for predicate, expected in ((edge.last == 1, (2**32 - 1, 2**32 - 1)),
                            (edge.last == 0, (0, 0)),
                            ((edge.last == 0) & (edge.x > 0), None)):
    q = Query(edge).where(predicate).minmax(edge.x)
    assert q.run("gpu") == expected and q.device == "gpu", (expected, q.device)
edge.close()

cases = [
    ("no GROUP BY", lambda: Query(fact).minmax(fact.x), np.ones(nf, bool), None),
    ("filter", lambda: Query(fact).where(fact.y < 100).minmax(fact.x), cols["y"] < 100, None),
    ("GROUP BY fact column", lambda: Query(fact).group_by(fact.g).minmax(fact.x), np.ones(nf, bool), cols["g"]),
    ("GROUP BY clustered column", lambda: Query(fact).where(fact.y > 500).group_by(fact.s).minmax(fact.x), cols["y"] > 500, cols["s"]),
    ("GROUP BY dimension attribute", lambda: Query(fact).join(dim).group_by(dim.grp).minmax(fact.x), np.ones(nf, bool), dim.data["grp"][fk]),
    ("dimension filter, GROUP BY region", lambda: Query(fact).join(dim).where(dim.grp < 4).group_by(dim.reg).minmax(fact.x),
     dim.data["grp"][fk] < 4, dim.data["reg"][fk]),
    ("nothing passes", lambda: Query(fact).where(fact.y > 5000).minmax(fact.x), np.zeros(nf, bool), None),
]
for eng in ("alg", "plain", "gpu"):
    for th in (1, 2):
        algq.set_threads(th)
        for name, mk, mask, key in cases:
            q = mk(); got = q.run(eng); want = want_mm(mask, key)
            if eng == "gpu" and key is None: assert q.device == "gpu", (name, q.device)
            if got is not None and not isinstance(got, dict): got = tuple(got)
            if isinstance(got, dict): got = {k: tuple(v) for k, v in got.items()}
            ok = got == want
            fails += not ok
            if not ok or th == 1 and eng == "alg": print(f"  {'ok  ' if ok else 'FAIL'} MIN/MAX {name} [{eng}, {th} thread(s)]" + ("" if ok else f": {got} vs {want}"))
        ids = Query(fact).join(dim).where((fact.y < 300) & (dim.grp == 2)).rows(eng)
        want_ids = np.nonzero((cols["y"] < 300) & (dim.data["grp"][fk] == 2))[0]
        ok = np.array_equal(np.asarray(ids, np.int64), want_ids)
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} passing rows [{eng}, {th} thread(s)]: {len(ids)} rows")
# SQL can reuse the scalar extrema scan's population for COUNT(*) and other base counts.
import duckdb
with duckdb.connect() as con:
    con.execute("CREATE TABLE t AS SELECT i::INTEGER id, CASE WHEN i % 2 = 0 THEN i END v FROM range(131) r(i)")
    db = torbles.from_duckdb(con, keep_plain=True)
    for eng in ("alg", "plain", "gpu"):
        for where in ("", " WHERE id > 130", " WHERE id BETWEEN 62 AND 68"):
            sql = "SELECT MIN(id), MAX(id), COUNT(*) FROM t" + where
            result = db.sql(sql, eng)
            assert result.rows == con.execute(sql).fetchall() and result.stats["engine_queries"] == 1
        sql = "SELECT MIN(id), MAX(id), MIN(v), COUNT(*), COUNT(v) FROM t"
        assert db.sql(sql, eng).rows == con.execute(sql).fetchall()
    db.close()
algq.set_threads(1)
print("ALL MATCH" if not fails else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
