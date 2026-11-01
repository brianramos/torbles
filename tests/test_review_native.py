# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""Release review R05 (native lifetimes) and R08 (empty tables), engine and Python bindings.

R08: Table.nplanes() divided the stored bytes by the word count, which is zero for an empty table (SIGFPE), and a
filtered join on an empty dimension divided by its row count. Every Table method and every Query kind runs on empty
tables (an empty fact, an empty dimension, both) on the alg, plain and gpu engines, against exact answers, next to the
same queries on non-empty tables against Python oracles; typed empty tables through db.sql against DuckDB.

R05: a Table's native storage (its planes or a loaded file's mapping, the sidecars, the plain baseline's table, the GPU
copy) was never freed. The reviewer's mallinfo probe (live native bytes return to the baseline once the tables are
discarded or closed) for built, self-extending, GPU-uploaded and loaded tables; the file mappings of loaded tables;
borrowers (queries and their compiled plans, relationship catalogs, the math modules' scopes, columns and dimension
records) that outlive the user's last reference to their table; close() followed by any use (TableClosedError, not a
crash), double close, close of a table a live query, scope or catalog holds, copies of a table, a failed load, the
release at interpreter exit.

The GPU executor's device copies of host arrays (a looked-up attribute's values, a delta plane), which a handle keeps
for later queries, were keyed by address and length alone: once tables (and their arrays) are freed, a new array at a
freed one's address and length read the old one's copy. They are keyed by the array's tag too, and evicted when the
array is collected or its table released; tested on the emulated GPU, whose kernels read the copies as a GPU's do.

Under AddressSanitizer (bash tests/sanitize_math.sh test_review_native.py) a use of freed memory or a double free
fails the run."""
import copy
import ctypes
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref

import duckdb
import numpy as np
import pandas as pd

import torbles
import algq
from algq import Table, Query, PlaneKey, Surrogate, TableClosedError
from torbles import math_bank as mbk
from torbles import math_certificates as mcert
from torbles import math_registry as reg
from torbles import relationships as rel

ENGINES = ("alg", "plain", "gpu")
E = np.empty(0, np.uint32)
ASAN = "asan" in os.environ.get("LD_PRELOAD", "")


def u32(v): return np.asarray(v, np.uint32)


class _MallInfo(ctypes.Structure):
    _fields_ = [(k, ctypes.c_size_t) for k in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
                                                "uordblks", "fordblks", "keepcost")]


_mallinfo2 = getattr(ctypes.CDLL(None), "mallinfo2", None)              # glibc 2.33 on
if _mallinfo2 is not None: _mallinfo2.restype = _MallInfo
NO_MALLINFO = "mallinfo counts glibc's arenas, not the sanitizer's allocator" if ASAN else \
    "no glibc mallinfo2" if _mallinfo2 is None else None


def live_native():
    """bytes of live malloc allocations: in the arenas and mapped (glibc mallinfo2)"""
    gc.collect()
    m = _mallinfo2()
    return int(m.uordblks) + int(m.hblkhd)


def mapped(directory):
    """the mappings of files under directory in this process"""
    with open("/proc/self/maps") as f:
        return sum(1 for line in f if directory in line)


# ---------------------------------------------------------------------- the star used by the query tests
def star(n, nd=6, seed=0):
    """a fact (x, y, k and foreign keys fk, fl into one dimension: fk stored msb first, so its joins are polynomials on
    its planes, fl joined by lookup) and the dimension (a, r), n fact rows, nd dimension rows; fks past nd are orphans"""
    rng = np.random.default_rng(seed)
    d = dict(a=u32(rng.integers(0, 9, nd)), r=u32(rng.integers(0, 3, nd)))
    dim = Table("d", d)
    f = dict(x=u32(rng.integers(0, 40, n)), y=u32(rng.integers(0, 7, n)), k=u32(rng.integers(0, 4, n)),
             fk=u32(rng.integers(0, nd + 2, n)), fl=u32(rng.integers(0, nd + 2, n)))
    fact = Table("f", f, msb=("fk",), refs={"fk": dim, "fl": dim})
    rr = dict(lo=u32([0, 3, 30]), hi=u32([9, 20, 35]), g=u32([0, 1, 1]))
    ranges = Table("rg", rr)
    return fact, dim, ranges, {k: v.astype(np.int64) for k, v in f.items()}, {k: v.astype(np.int64) for k, v in d.items()}, \
        {k: v.astype(np.int64) for k, v in rr.items()}


def queries(f, d, rg):
    """(name, query) for every Query kind"""
    q = [("count", Query(f).count()), ("sum", Query(f).sum(f.x)), ("affine sum", Query(f).sum(2 * f.x + 3)),
         ("bilinear sum", Query(f).sum(f.x * f.y)), ("cubic sum", Query(f).sum(f.x * f.x * f.y + 1)),
         ("min/max", Query(f).minmax(f.x)),
         ("filter or/not", Query(f).where((f.x > 25) | ~(f.y == 1)).count()),
         ("isin, between", Query(f).where(f.x.isin([3, 4, 5]) | f.y.between(2, 4)).sum(f.x)),
         ("group by", Query(f).group_by(f.k).sum(f.x)), ("group by two, count", Query(f).group_by(f.k, f.y).count()),
         ("group by, min/max", Query(f).group_by(f.k).minmax(f.x)),
         ("range join", Query(f).join_range(rg, f.x, rg.lo, rg.hi).group_by(rg.g).sum(f.y)),
         ("range join total", Query(f).join_range(rg, f.x, rg.lo, rg.hi).count()),
         ("inequality join", Query(f).join_ineq(rg, f.x, "<", rg.lo).sum(f.y))]
    for mode in ("prefix", "factored", "expanded", "lookup"):
        q += [(f"join {mode}", Query(f, join_mode=mode).join(d, f.fk).count()),
              (f"join {mode}, dim filter", Query(f, join_mode=mode).join(d, f.fk).where(d.a > 3).sum(f.x)),
              (f"join {mode}, group by dim", Query(f, join_mode=mode).join(d, f.fk).where(d.a < 7).group_by(d.r).count())]
    q += [("lookup fk join, dim filter", Query(f).join(d, f.fl).where(d.r == 1).sum(f.y)),
          ("join, group by fact key", Query(f).join(d, f.fk).where(d.r > 0).group_by(f.k).sum(f.x))]
    return q


def oracle(name, F, D, R):
    """the exact answer of queries()[name] on the arrays"""
    n, nd = len(F["x"]), len(D["a"])
    x, y, k = F["x"], F["y"], F["k"]
    def total(m, v): return int(v[m].sum()) if m.any() else None
    def groups(keys, m, v=None, ext=False):
        out = {}
        for i in np.flatnonzero(m):
            key = tuple(int(c[i]) for c in keys)
            if ext: lo, hi = out.get(key, (int(v[i]), int(v[i]))); out[key] = (min(lo, int(v[i])), max(hi, int(v[i])))
            else: out[key] = out.get(key, 0) + (1 if v is None else int(v[i]))
        return out
    every = np.ones(n, bool)
    def joined(fk, extra=lambda a, r: np.ones(len(a), bool)):
        ok = fk < nd
        m = ok.copy(); m[ok] = extra(D["a"][fk[ok]], D["r"][fk[ok]])
        return m
    def rjoin(lo_of, hi_of):            # (fact row, range row) pairs
        pairs = [(i, j) for i in range(n) for j in range(len(R["lo"])) if lo_of(j) <= x[i] <= hi_of(j)]
        return pairs
    if name == "count": return n
    if name == "sum": return total(every, x)
    if name == "affine sum": return total(every, 2 * x + 3)
    if name == "bilinear sum": return total(every, x * y)
    if name == "cubic sum": return total(every, x * x * y + 1)
    if name == "min/max": return (int(x.min()), int(x.max())) if n else None
    if name == "filter or/not": return int(((x > 25) | (y != 1)).sum())
    if name == "isin, between": return total(np.isin(x, [3, 4, 5]) | ((y >= 2) & (y <= 4)), x)
    if name == "group by": return groups([k], every, x)
    if name == "group by two, count": return groups([k, y], every)
    if name == "group by, min/max": return groups([k], every, x, ext=True)
    if name in ("range join", "range join total"):
        pairs = rjoin(lambda j: R["lo"][j], lambda j: R["hi"][j])
        if name == "range join total": return len(pairs)
        out = {}
        for i, j in pairs: out[(int(R["g"][j]),)] = out.get((int(R["g"][j]),), 0) + int(y[i])
        return out
    if name == "inequality join":
        pairs = [(i, j) for i in range(n) for j in range(len(R["lo"])) if x[i] < R["lo"][j]]
        return sum(int(y[i]) for i, _ in pairs) if pairs else None
    if name.startswith("join "):
        if name.endswith("dim filter"): return total(joined(F["fk"], lambda a, r: a > 3), x)
        if name.endswith("group by dim"):
            m = joined(F["fk"], lambda a, r: a < 7)
            return groups([np.where(F["fk"] < nd, D["r"][np.minimum(F["fk"], max(nd - 1, 0))] if nd else 0, 0)], m)
        return int(joined(F["fk"]).sum())
    if name == "lookup fk join, dim filter": return total(joined(F["fl"], lambda a, r: r == 1), y)
    if name == "join, group by fact key": return groups([k], joined(F["fk"], lambda a, r: r > 0), x)
    raise KeyError(name)


def run(q, engine):
    r = q.run(engine)
    return tuple(r) if isinstance(r, algq.MinMax) else ({k: tuple(v) if isinstance(v, algq.MinMax) else v
                                                         for k, v in r.items()} if isinstance(r, dict) else r)


# ---------------------------------------------------------------------- the GPU executor's device copies
def cache_entries(t):
    """the device copies of host arrays that table t's GPU handle holds: sorted (tag, bytes)"""
    g, h = algq.gpu_lib(), t.gpu_handle()
    n = g.gpu_cache_entries(h, None, None, 0)
    tags, sizes = (ctypes.c_uint64 * max(n, 1))(), (ctypes.c_uint64 * max(n, 1))()
    g.gpu_cache_entries(h, tags, sizes, n)
    return sorted((int(tags[i]), int(sizes[i])) for i in range(n))


def dimension(nd, seed, values=4):
    """a dimension of nd rows: a takes each of `values` values (a permutation), r in 0..2"""
    rng = np.random.default_rng(seed)
    return Table(f"d{seed}", dict(a=u32(rng.permutation(np.arange(nd) % values)), r=u32(rng.integers(0, 3, nd))))


def lookup_star(nd=7, n=500, seed=8):
    """a fact joined by lookup to a dimension (its fk is not stored msb first): GROUP BY a dimension attribute reads the
    attribute's ranked column, whose device copy the fact's handle caches; fk = nd are orphans"""
    rng = np.random.default_rng(seed)
    dim = dimension(nd, seed)
    return Table("f", dict(x=u32(rng.integers(0, 50, n)), fk=u32(rng.integers(0, nd + 1, n))), refs={"fk": dim}), dim


def by_attribute(fact, dim, col="a"):
    """rows per value of dim.col over the fact rows that join: the query, its exact answer"""
    fk = fact.data["fk"].astype(np.int64); vals = dim.data[col].astype(np.int64)
    want = {}
    for v in vals[fk[fk < dim.n]].tolist(): want[(v,)] = want.get((v,), 0) + 1
    return Query(fact, join_mode="lookup").join(dim, fact.fk).group_by(getattr(dim, col)).count(), want


class EmptyTables(unittest.TestCase):
    """R08"""

    def test_nplanes_empty_child_process(self):
        """the reviewer's reproducer (review_native/probe_empty.py nplanes) in a child: it exited with SIGFPE"""
        code = ("import numpy as np; from torbles import Table, Query; t = Table('empty', {'x': np.empty(0, np.uint32)});"
                "print(t.nplanes()); print([Query(t).count().run(e) for e in ('alg', 'plain', 'gpu')])")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        planes, counts = p.stdout.split("\n")[:2]
        self.assertEqual(int(planes), Table("one", {"x": u32([1])}).nplanes())    # the same widths, the same planes
        self.assertEqual(counts, "[0, 0, 0]")                                       # the review's passing control

    def test_nplanes_counts_the_stored_planes(self):
        """the count no longer divides; on non-empty tables it equals the stored bytes / (8 x words), as before"""
        rng = np.random.default_rng(5)
        for n in (1, 63, 64, 65, 1000, 4097):
            for k in (1, 2, 5):
                cols = {f"c{i}": u32(rng.integers(0, 1 << int(rng.integers(1, 32)), n)) for i in range(k)}
                t = Table("t", cols, msb=("c0",) if k > 1 else ())
                nw = (n + 63) // 64
                self.assertEqual(t.nplanes(), t.torus_bytes() // (8 * nw))
                widths = [t.width(c) for c in cols]
                empty = Table("e", {c: E for c in cols})
                self.assertEqual(empty.torus_bytes(), 0)
                # an empty column has width 1 (a stored value 0 or 1): its planes are those of such a table
                self.assertEqual(empty.nplanes(), Table("w", {c: u32([1]) for c in cols}).nplanes())
                self.assertTrue(all(w >= 1 for w in widths))

    def test_table_methods_on_empty_tables(self):
        dim = Table("d", dict(a=u32([1, 4]), r=u32([0, 1])))
        t = Table("e", dict(x=E, fk=E), msb=("fk",), refs={"fk": dim}, cluster="x")
        self.assertEqual((t.n, t.nrows(), list(t.used_ids)), (0, 0, []))
        self.assertEqual(t.nplanes(), Table("w", dict(x=u32([1]), fk=u32([1])), msb=("fk",), refs={"fk": dim}).nplanes())
        self.assertEqual((t.width("x"), t.colmax("x"), t.maxcode("fk"), t.torus_bytes(), t.sidecar_bytes(), t.plain_bytes()),
                         (1, 0, 0, 0, 0, 0))
        self.assertEqual(len(t.decoded("x")), 0)
        self.assertEqual(len(t.get_rows("x", [])), 0)
        with self.assertRaises(IndexError): t.get_rows("x", [0])            # no row is stored: no read past the planes
        with self.assertRaises(IndexError): t.flip(0, 0)                    # nor a write
        with self.assertRaises(IndexError): Table("one", {"x": u32([3])}).flip(0, 1)
        with self.assertRaises(IndexError): Table("one", {"x": u32([3])}).flip(99, 0)
        self.assertEqual(t.validate(), (0, []))
        self.assertEqual(t.repair(), 0)
        u, ranks = t.ranked("x"); self.assertEqual((len(u), len(ranks)), (0, 0))
        self.assertEqual((t.fks_to(dim), t.kmask("fk"), t["x"].name, t.x.name, t.idx("fk")), (["fk"], 0, "x", "x", 1))
        self.assertIsNotNone(t.gpu_handle()); self.assertIsNotNone(t.plain_handle())
        self.assertEqual(t.space("fk", dim).n, 2)
        sg_dim = Table("s", dict(a=u32([1, 4])))
        sg = Surrogate(1, [0, 1]); sg.move([1], [3])
        moved = Table("m", dict(fk=E), msb=("fk",), refs={"fk": (sg_dim, sg)})
        out, cnt = moved.delta("fk", sg_dim); self.assertEqual((len(out), cnt), (0, 0))
        self.assertEqual(Query(moved).join(sg_dim, moved.fk).where(sg_dim.a == 4).count().run(), 0)
        t.drop_plain()
        with self.assertRaises(RuntimeError): t.plain_handle()
        empty_dim = Table("ed", dict(a=E))
        self.assertEqual((empty_dim.space("a", empty_dim).n, len(empty_dim.ranked("a")[0])), (0, 0))
        t.close(); self.assertTrue(t.closed)

    def test_every_query_kind_on_empty_and_full_tables(self):
        """an empty fact, an empty dimension, both empty and the non-empty controls, on every engine"""
        cases = {"full": (60, 6), "empty fact": (0, 6), "empty dimension": (60, 0), "both empty": (0, 0)}
        for label, (n, nd) in cases.items():
            fact, dim, rg, F, D, R = star(n, nd, seed=n + nd)
            if label == "full": self.assertTrue(len(F["x"]) and len(D["a"]))
            for name, q in queries(fact, dim, rg):
                want = oracle(name, F, D, R)
                if n == 0: self.assertIn(want, (0, None, {}), (label, name))
                for engine in ENGINES:
                    with self.subTest(case=label, query=name, engine=engine):
                        self.assertEqual(run(q, engine), want)
                        if label != "full": self.assertEqual(run(q, engine), want)     # the cached plan
            for engine in ENGINES:
                with self.subTest(case=label, query="group by planes", engine=engine):
                    key, _ = algq._bits_key(fact, "k", 0, 1, engine)          # code bit 0 of k, as each engine numbers planes
                    self.assertIsInstance(key, PlaneKey)
                    want = {}
                    for v in (F["k"] & 1).tolist(): want[(v,)] = want.get((v,), 0) + 1
                    self.assertEqual(Query(fact).group_by(key).count().run(engine), want)
            for engine in ("alg", "plain"):
                with self.subTest(case=label, query="rows / rank selection / columns", engine=engine):
                    rows = Query(fact).where(fact.x > 10).rows(engine)
                    self.assertEqual(rows.tolist(), np.flatnonzero(F["x"] > 10).tolist())
                    got, total = Query(fact).rank_codes(fact.x, [0, -1], engine=engine)
                    self.assertEqual(total, n)
                    self.assertEqual(got, {0: int(F["x"].min()), -1: int(F["x"].max())} if n else {})
                    g = Query(fact).group_by(fact.k).sum(fact.x).run_columns(engine)
                    self.assertEqual(len(g), len(oracle("group by", F, D, R)))
            q = Query(fact).group_by(fact.k, fact.y); q._codes = True; q._raw = True; q._distinct = True
            km, _, cn = q._run_codes("alg")
            self.assertEqual(dict(zip(km[:, 0].tolist(), cn.tolist())),
                             {int(g): len(set(F["y"][F["k"] == g].tolist())) for g in set(F["k"].tolist())})

    def test_self_extending_and_virtual_columns_empty(self):
        for n in (0, 50):
            rng = np.random.default_rng(n)
            v = (rng.integers(0, 1 << 40, n)).astype(np.uint64); k = u32(rng.integers(0, 3, n))
            t = Table("w", dict(v=v, k=k), extend={"v": 32})
            self.assertIsInstance(t.nplanes(), int)
            for engine in ENGINES:
                with self.subTest(n=n, engine=engine):
                    self.assertEqual(Query(t).count().run(engine), n)
                    self.assertEqual(Query(t).sum(t.v).run(engine), int(v.astype(object).sum()) if n else None)
                    self.assertEqual(Query(t).where(t.v > (1 << 39)).count().run(engine), int((v > (1 << 39)).sum()))
                    want = {}
                    for a, b in zip(k.tolist(), v.tolist()): want[(a,)] = want.get((a,), 0) + b
                    self.assertEqual(Query(t).group_by(t.k).sum(t.v).run(engine), want)
            flag = u32(rng.integers(0, 2, n)); vals = u32(rng.integers(0, 1000, n)) * flag
            tv = Table("v", dict(f=flag, val=vals), virtual={"val": ("f", False)})
            self.assertEqual(tv.get_rows("val", np.arange(n)).tolist(), vals.tolist())
            self.assertEqual(Query(tv).sum(tv.val).run(), int(vals.sum()) if n else None)

    def test_typed_empty_tables_sql_against_duckdb(self):
        ddl = ("CREATE TABLE d (id INTEGER PRIMARY KEY, r VARCHAR, w DECIMAL(9,2));"
               "CREATE TABLE f (x BIGINT, g INTEGER, s VARCHAR, v BIGINT, big BIGINT, d INTEGER REFERENCES d(id))")
        dd = pd.DataFrame({"id": pd.Series([1, 2, 3], dtype="int32"), "r": pd.Series(["a", "b", "a"]),
                           "w": pd.Series(["1.50", "2.25", "-3.00"]).map(__import__("decimal").Decimal)})
        n = 9
        ff = pd.DataFrame({"x": pd.Series(np.arange(n) * 7 - 20, dtype="int64"), "g": pd.Series(np.arange(n) % 3, dtype="int32"),
                           "s": pd.Series([f"s{i % 3}" for i in range(n)]),
                           "v": pd.Series([None if i % 4 == 0 else i for i in range(n)], dtype="Int64"),
                           "big": pd.Series(np.arange(n) * (1 << 37), dtype="int64"), "d": pd.Series([1, 2, 3, 9] * 2 + [1], dtype="int32")})
        sql = ["SELECT count(*), sum(x), min(x), max(x), avg(x) FROM f", "SELECT g, count(*), sum(v), min(s) FROM f GROUP BY g ORDER BY g",
               "SELECT count(*) FROM f WHERE s = 's1'", "SELECT sum(big), max(big) FROM f", "SELECT x FROM f ORDER BY x LIMIT 3",
               "SELECT count(DISTINCT g), median(x) FROM f", "SELECT count(*), sum(f.x) FROM f JOIN d ON f.d = d.id",
               "SELECT d.r, sum(f.x), sum(d.w) FROM f JOIN d ON f.d = d.id GROUP BY d.r ORDER BY d.r",
               "SELECT count(*), sum(f.x) FROM f JOIN d ON f.d = d.id WHERE d.r = 'a'", "SELECT count(*) FROM f LEFT JOIN d ON f.d = d.id",
               "SELECT count(*), sum(w) FROM d"]
        for label, fd, fdim in (("full", ff, dd), ("empty fact", ff.iloc[:0], dd), ("empty dimension", ff, dd.iloc[:0]),
                                ("both empty", ff.iloc[:0], dd.iloc[:0])):
            db = torbles.from_ddl(ddl, {"d": fdim, "f": fd}, keep_plain=True)
            self.addCleanup(db.close)
            con = duckdb.connect(); con.execute(ddl.replace(" REFERENCES d(id)", "").replace(" PRIMARY KEY", ""))
            con.register("fd", fd); con.register("dd", fdim)
            con.execute("INSERT INTO f SELECT * FROM fd"); con.execute("INSERT INTO d SELECT * FROM dd")
            for text in sql:
                want = [tuple(r) for r in con.execute(text).fetchall()]
                for engine in ENGINES:
                    with self.subTest(case=label, sql=text, engine=engine):
                        self.assertEqual([tuple(r) for r in db.sql(text, engine=engine).rows], want)
            f = db["f"]
            for engine in ENGINES:
                with self.subTest(case=label, query="engine API", engine=engine):
                    self.assertEqual(db.query(f).sum(f.x).run(engine), int(fd["x"].sum()) if len(fd) else None)
                    self.assertEqual(db.query(f).group_by(f.g).count().run(engine),
                                     {(int(g),): int(c) for g, c in fd.groupby("g").size().items()})
            for t in db.tables.values(): self.assertGreater(t.nplanes(), 0)
            folder = tempfile.mkdtemp(prefix="review-native-"); self.addCleanup(shutil.rmtree, folder, True)
            db.save(folder)
            back = torbles.load(folder); self.addCleanup(back.close)
            for name, t in back.tables.items():
                self.assertEqual((t.nplanes(), t.n), (db[name].nplanes(), db[name].n))
                self.assertEqual(t.validate(), (0, []))
            self.assertEqual([tuple(r) for r in back.sql(sql[0]).rows], [tuple(r) for r in con.execute(sql[0]).fetchall()])


class Lifetimes(unittest.TestCase):
    """R05"""

    def small(self, n=300, seed=1):
        rng = np.random.default_rng(seed)
        x, k = u32(rng.integers(0, 1000, n)), u32(rng.integers(0, 5, n))
        want = {}
        for a, b in zip(k.tolist(), x.tolist()):
            if b > 100: want[(a,)] = want.get((a,), 0) + b
        return Table("t", dict(x=x, k=k, z=u32(rng.integers(0, 9, n)))), want

    # ------------------------------------------------------------------ the reviewer's probe
    HEADER = 64 << 10      # what a closed table kept alive still holds natively: its torble's header (55 KB), no planes

    def probe(self, make, count=20, close=False):
        """live native bytes after making and discarding tables, or after closing them (kept alive, then discarded):
        (bytes live at the end, bytes live while the closed tables were kept, the storage of one table). A closed table
        kept alive also holds its Python row masks (used, used_bm): they are not counted."""
        before = live_native(); kept, refs, held, masks = [], [], 0, 0
        for _ in range(count):
            t = make()
            held = t.torus_bytes() + t.sidecar_bytes()
            refs.append(weakref.ref(t))
            if close: t.close(); kept.append(t); masks += t.used.nbytes + t.used_bm.nbytes
            del t
        while_kept = live_native() - before - masks
        del kept
        self.assertEqual(sum(r() is not None for r in refs), 0)     # all collected (the review's control)
        return live_native() - before, while_kept, held

    def assert_released(self, label, make, count):
        for close in (False, True):
            delta, while_kept, per = self.probe(make, count, close)
            self.assertLess(delta, per // 4, f"{label}, close={close}: {delta} bytes still live")
            if close:
                self.assertGreater(per, 2 * self.HEADER)
                self.assertLess(while_kept, count * self.HEADER, f"{label}: {while_kept} bytes held by {count} closed tables")

    @unittest.skipIf(NO_MALLINFO, NO_MALLINFO)
    def test_reviewer_probe_discarded_and_closed_tables(self):
        values = np.arange(50_000, dtype=np.uint32) * np.uint32(997)
        self.assertEqual(Table("t", {"x": values}).torus_bytes(), 494224)     # as in the review's receipt
        self.assert_released("the review's tables", lambda: Table("t", {"x": values}), 20)

    @unittest.skipIf(NO_MALLINFO, NO_MALLINFO)
    def test_probe_sidecars_virtual_plain_and_gpu_copies(self):
        rng = np.random.default_rng(2)
        v = rng.integers(0, 1 << 44, 50_000).astype(np.uint64); x = u32(rng.integers(0, 1 << 20, 50_000))
        flag = u32(rng.integers(0, 2, 50_000))
        def gpu():
            t = Table("g", {"x": x}); t.gpu_handle(); return t
        gpu().close()                  # first use of the GPU library (a CUDA context) before anything is measured
        def dropped():
            t = Table("p", {"x": x, "f": flag}); t.drop_plain(); return t
        for label, make in (("self-extending", lambda: Table("w", {"v": v, "x": x}, extend={"v": 32})),
                            ("virtual", lambda: Table("v", {"f": flag, "x": x * flag}, virtual={"x": ("f", False)})),
                            ("GPU copy", gpu), ("plain dropped", dropped)):
            self.assert_released(label, make, 8)

    @unittest.skipIf(NO_MALLINFO, NO_MALLINFO)
    def test_probe_loaded_tables(self):
        folder = self.saved()
        before = live_native()
        for _ in range(10):
            db = torbles.load(folder); self.assertGreater(mapped(folder), 0)
            del db
        self.assertLess(live_native() - before, 64 << 10)
        self.assertEqual(mapped(folder), 0)

    def saved(self, extra=()):
        folder = tempfile.mkdtemp(prefix="review-native-"); self.addCleanup(shutil.rmtree, folder, True)
        n = 5000; rng = np.random.default_rng(3)
        frames = {"m": pd.DataFrame({"x": pd.Series(rng.integers(0, 1 << 40, n), dtype="int64"),
                                     "y": pd.Series(rng.integers(-5, 5, n), dtype="int64")})}
        for name in extra: frames[name] = pd.DataFrame({"z": pd.Series(np.arange(10), dtype="int64")})
        db = torbles.from_frames(frames); db.save(folder); db.close()
        return folder

    # ------------------------------------------------------------------ loaded tables: mappings
    def test_loaded_tables_are_unmapped_by_close_and_by_the_finalizer(self):
        folder = self.saved()
        db = torbles.load(folder)
        self.assertTrue(db["m"].sidecars)                      # the 40-bit column has a sidecar file, mapped too
        self.assertGreaterEqual(mapped(folder), 2)
        want = db.sql("SELECT sum(x), count(*) FROM m").rows
        db.close(); self.assertEqual(mapped(folder), 0)
        db.close()                                             # twice: nothing
        db = torbles.load(folder); self.assertEqual(db.sql("SELECT sum(x), count(*) FROM m").rows, want)
        q = db.query(db["m"]).sum(db["m"].x)
        del db; gc.collect()
        self.assertGreater(mapped(folder), 0)                  # the query holds the table
        self.assertEqual(q.run(), want[0][0])
        del q; gc.collect(); self.assertEqual(mapped(folder), 0)

    def test_a_failed_load_releases_what_it_mapped(self):
        folder = self.saved(extra=("z1",))
        import json
        with open(os.path.join(folder, "manifest.json")) as f: man = json.load(f)
        last = man["tables"][man["order"][-1]]["file"]
        self.assertEqual(man["order"][0], "m")                 # mapped (with its sidecar) before the failure
        with open(os.path.join(folder, last), "r+b") as f: f.truncate(100)
        with self.assertRaises(torbles.LoadError): torbles.load(folder)
        self.assertEqual(mapped(folder), 0)

    # ------------------------------------------------------------------ borrowers outlive the user's reference
    def test_queries_and_cached_plans_outlive_the_table_reference(self):
        t, want = self.small()
        qs = {e: Query(t).where(t.x > 100).group_by(t.k).sum(t.x) for e in ENGINES}
        for e, q in qs.items(): self.assertEqual(q.run(e), want)
        ref = weakref.ref(t); fin = t._native.release
        del t; gc.collect()
        self.assertIsNotNone(ref()); self.assertTrue(fin.alive)
        for e, q in qs.items():
            self.assertEqual(q.run(e), want)                   # the cached plans, on the planes the queries keep
            self.assertTrue(q.plan_cached)
        del qs, q; gc.collect()
        self.assertIsNone(ref()); self.assertFalse(fin.alive)  # freed once no borrower is left

    def test_joins_outlive_both_tables(self):
        fact, dim, rg, F, D, R = star(200, 6, seed=4)
        qs = [(name, q) for name, q in queries(fact, dim, rg) if "join" in name]
        refs = [weakref.ref(fact), weakref.ref(dim), weakref.ref(rg)]
        del fact, dim, rg; gc.collect()
        self.assertTrue(all(r() is not None for r in refs))
        for name, q in qs:
            for e in ENGINES: self.assertEqual(run(q, e), oracle(name, F, D, R), (name, e))
        del qs, q; gc.collect()
        self.assertTrue(all(r() is None for r in refs))

    def test_math_objects_outlive_the_table_reference(self):
        t, _ = self.small(400, 6)
        x, k = t.data["x"].astype(np.int64), t.data["k"].astype(np.int64)
        scope = mbk.Scope.from_table(t, ["x"], group_by=["k"])
        cscope = mcert.code_scope(t, ["x", "k"])
        column = reg.Column(t, "x")
        dimension = mcert.Dimension(t, "k")
        ref = weakref.ref(t); del t; gc.collect()
        self.assertIsNotNone(ref())
        self.assertTrue(scope.fresh() and cscope.fresh())
        bank = mbk.build(scope, ["1", "x"])
        for g in range(5):
            st = bank.state(g)
            self.assertEqual((st["rows"], st["sum"]), (int((k == g).sum()), [int((k == g).sum()), int(x[k == g].sum())]))
        res = reg.execute(reg.Request(column, [[0, 1]]), "scan")
        self.assertEqual(res["values"][0], int(x.sum()))
        self.assertEqual((dimension.info["rows"], dimension.info["unique"]), (400, False))
        del scope, cscope, column, dimension, bank, res, st; gc.collect()
        self.assertIsNone(ref())

    def typed(self):
        i = np.arange(300, dtype=np.int64)
        frame = pd.DataFrame(dict(x=i % 17 + 1, y=(i * 7) % 11 + 2, z=(i % 17 + 1) * ((i * 7) % 11 + 2), k=i % 4))
        db = torbles.from_frames({"m": frame})
        return db, frame

    def test_relationship_catalog_outlives_the_table_reference(self):
        from torbles import math_disc
        db, frame = self.typed()
        cat = rel.discover(db, "m", ["x", "y", "z"])
        t = db["m"]; q = db.query(t).sum(t.z)
        ref = weakref.ref(t); del t, db; gc.collect()
        self.assertIsNotNone(ref())
        self.assertEqual(cat.probe("product", "z", "x", "y")["violations"], 0)
        self.assertTrue(any(r["expression"] == "z = x * y" for r in cat.rules))
        self.assertEqual(math_disc.verify(cat, "product", "z", "x", "y")["violations"], 0)
        self.assertEqual(q.run(), int(frame["z"].sum()))
        del q, cat; gc.collect()
        self.assertIsNone(ref())

    # ------------------------------------------------------------------ explicit close
    def test_close_then_use_is_a_clear_error(self):
        t, want = self.small()
        q = Query(t).where(t.x > 100).group_by(t.k).sum(t.x)
        for e in ENGINES: self.assertEqual(q.run(e), want)
        t.gpu_handle()
        scope = mbk.Scope.from_table(t, ["x"], group_by=["k"])
        column = reg.Column(t, "x")
        storage, fin = t._native.s, t._native.release
        t.close()
        self.assertTrue(t.closed)
        self.assertEqual((storage.at, storage.pt, storage.gpu, storage.sides), (None, None, None, []))
        self.assertIsNotNone(storage.header)                              # the emptied header, until the finalizer
        uses = {"at": lambda: t.at, "nplanes": t.nplanes, "width": lambda: t.width("x"), "decoded": lambda: t.decoded("x"),
                "get_rows": lambda: t.get_rows("x", [0]), "validate": t.validate, "repair": t.repair,
                "flip": lambda: t.flip(0, 0), "torus_bytes": t.torus_bytes, "sidecar_bytes": t.sidecar_bytes,
                "gpu_handle": t.gpu_handle, "plain_handle": t.plain_handle, "colmax": lambda: t.colmax("z"),
                "drop_plain": t.drop_plain, "ranked": lambda: t.ranked("x"),
                "new query": lambda: Query(t).count().run(), "new range join": lambda: Query(t).join_ineq(t, t.x, "<", t.k).count().run(),
                "math scope": lambda: mbk.Scope.from_table(t, ["x"]), "registry column": lambda: reg.Column(t, "x"),
                "scope freshness": scope.fresh, "bank build": lambda: mbk.build(scope, ["x"]),
                "registry request": lambda: reg.Request(column, [[0, 1]])}
        uses.update({f"live query, {e}": (lambda e=e: q.run(e)) for e in ENGINES})
        for name, use in uses.items():
            with self.subTest(use=name):
                with self.assertRaises(TableClosedError): use()
        self.assertEqual((scope.rows, column.bytes > 0), (300, True))   # snapshots: their own copies stay readable
        t.close(); t.close()                                             # double close: nothing
        scope.close(); column.close()
        del t, q, scope, column, uses, use; gc.collect()
        self.assertFalse(fin.alive); self.assertIsNone(storage.header)

    def test_close_and_the_relationship_catalogs(self):
        """the catalog attached to the table is closed with it; one kept apart, which decodes through its own pointer to
        the torble, finds the emptied header of a new revision (a stale table), never freed memory"""
        from torbles import math_disc
        db, frame = self.typed()
        attached = rel.discover(db, "m", ["x", "y", "z"])
        detached = rel.Catalog(db["m"], ["x", "y", "z"], attach=False)
        self.assertEqual(detached.probe("sum", "z", "x", "y")["violations"] > 0, True)
        t = db["m"]
        self.assertIs(t._relationships, attached)
        db.close()
        self.assertTrue(t.closed and not hasattr(t, "_relationships"))
        for what, use in (("rules", lambda c: c.rules), ("stats", lambda c: c.stats),
                          ("probe", lambda c: c.probe("product", "z", "x", "y"))):
            with self.subTest(use=what):
                with self.assertRaisesRegex(ValueError, "closed"): use(attached)
                with self.assertRaisesRegex(ValueError, "stale"): use(detached)
        with self.assertRaises(TableClosedError): math_disc.verify(detached, "product", "z", "x", "y")
        db.close(); attached.close(); detached.close()
        with self.assertRaises(TableClosedError): db.sql("SELECT sum(x) FROM m")
        with self.assertRaises(TableClosedError): db.query(t).count().run()

    def test_copies_share_the_storage(self):
        t, want = self.small()
        c = copy.copy(t)                                     # sqlexec binds a table twice this way
        self.assertIsNot(c, t); self.assertEqual(c.at, t.at)
        self.assertIs(copy.deepcopy(t), t)                   # a deep copy of a query keeps its tables
        q = Query(t).where(t.x > 100); q2 = copy.deepcopy(q)
        self.assertIs(q2.fact, t)
        ref = weakref.ref(t); del t, q, q2; gc.collect()
        self.assertIsNone(ref())                              # the original is gone, the copy keeps the storage
        self.assertEqual(Query(c).where(c.x > 100).group_by(c.k).sum(c.x).run(), want)
        c2 = copy.copy(c); c2.close()
        self.assertTrue(c.closed)
        with self.assertRaises(TableClosedError): Query(c).count().run()
        c.close()

    def test_a_dimension_bound_twice_in_sql(self):
        """sqlexec binds one table twice through copy.copy: the copies share its storage, which outlives them"""
        d = pd.DataFrame({"id": pd.Series(np.arange(5), dtype="int64"), "r": pd.Series(np.arange(5) % 2, dtype="int64")})
        f = pd.DataFrame({"a": pd.Series([0, 1, 2, 3, 4, 4], dtype="int64"), "b": pd.Series([1, 1, 2, 0, 3, 9], dtype="int64"),
                          "x": pd.Series(np.arange(6), dtype="int64")})
        db = torbles.from_ddl("CREATE TABLE d (id BIGINT PRIMARY KEY, r BIGINT); CREATE TABLE f (a BIGINT REFERENCES d(id), "
                              "b BIGINT REFERENCES d(id), x BIGINT)", {"d": d, "f": f}, keep_plain=True)
        self.addCleanup(db.close)
        sql = "SELECT d1.r, d2.r, sum(f.x) FROM f JOIN d d1 ON f.a = d1.id JOIN d d2 ON f.b = d2.id GROUP BY d1.r, d2.r ORDER BY 1, 2"
        want = [tuple(r) for r in db.con.execute(sql).fetchall()]
        for engine in ENGINES * 2:
            self.assertEqual([tuple(r) for r in db.sql(sql, engine=engine).rows], want, engine)
            gc.collect()
        self.assertFalse(db["d"].closed)

    def test_release_at_interpreter_exit(self):
        """the finalizer also runs at exit: an atexit hook registered before any finalizer runs after them"""
        code = ("import atexit, weakref\n"
                "assert not weakref.finalize._registered_with_atexit\n"
                "state = {}\n"
                "atexit.register(lambda: print('released at exit:', [f.alive for f in state['fins']], [(s.at, s.header) for s in state['st']]))\n"
                "import numpy as np, torbles\n"
                "ts = [torbles.Table('t', {'x': np.arange(1000, dtype=np.uint32)}) for _ in range(3)]\n"
                "ts[0].gpu_handle(); ts[1].close()\n"
                "state['fins'] = [t._native.release for t in ts]; state['st'] = [t._native.s for t in ts]\n"
                "print('alive:', [f.alive for f in state['fins']])\n")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertIn("alive: [True, True, True]", p.stdout)
        self.assertIn("released at exit: [False, False, False] [(None, None), (None, None), (None, None)]", p.stdout)

    def test_close_of_a_joined_table(self):
        """a query's compiled plan holds its dimension's arrays, not the dimension: once the dimension is closed, the
        query raises TableClosedError on every engine (it answered from the arrays its plan kept)"""
        fact, dim, rg, F, D, R = star(200, 6, seed=4)
        qs = {e: Query(fact).join(dim, fact.fk).where(dim.a > 3).sum(fact.x) for e in ENGINES}
        lookups = {e: Query(fact, join_mode="lookup").join(dim, fact.fk).group_by(dim.r).count() for e in ENGINES}
        by_r = {}
        for v in D["r"][F["fk"][F["fk"] < len(D["r"])]].tolist(): by_r[(v,)] = by_r.get((v,), 0) + 1
        for e in ENGINES:
            self.assertEqual(qs[e].run(e), oracle("join prefix, dim filter", F, D, R))
            self.assertEqual(lookups[e].run(e), by_r)
        dim.close()
        for e in ENGINES:
            for q in (qs[e], lookups[e]):
                with self.subTest(engine=e):
                    with self.assertRaises(TableClosedError): q.run(e)
        self.assertEqual(Query(fact).sum(fact.x).run("gpu"), int(F["x"].sum()))     # the fact is open


class GpuDeviceCopies(unittest.TestCase):
    """The GPU executor's device copies of a query's host arrays (a looked-up attribute's ranked column, a delta plane),
    on the emulated GPU, whose kernels read the copies as a GPU's do: served by address, length and the array's tag,
    never across two arrays at one address; evicted when the array is collected or its table released. What only a
    device shows: the copies' cudaMalloc / cudaMemcpy / cudaFree, and the kernels reading device memory."""

    def setUp(self): gc.collect()

    def assert_live_copies_only(self):
        """every copy held by a live GPU handle is of a live array (one with a tag and a referent)"""
        gc.collect()
        live = {tag for ref, tag in algq._GPU_TAGS.values() if ref() is not None}
        for s in list(algq._GPU_LIVE):
            n = s.glib.gpu_cache_entries(s.gpu, None, None, 0)
            tags = (ctypes.c_uint64 * max(n, 1))(); s.glib.gpu_cache_entries(s.gpu, tags, None, n)
            self.assertLessEqual({int(tags[i]) for i in range(n)}, live)

    def test_copies_are_served_by_address_length_and_tag(self):
        """the cache itself, through gpu_query_tagged: a compiled lookup plan whose attribute array is ours"""
        fact, dim = lookup_star()
        q, want = by_attribute(fact, dim)
        self.assertEqual(q.run("gpu"), want); self.assertEqual(q.device, "gpu")
        plan = algq.QPlan()
        ctypes.memmove(ctypes.addressof(plan), ctypes.addressof(q.last_plan), ctypes.sizeof(plan))
        self.assertEqual((plan.nattr, plan.nkey), (1, 1))
        cap = int(plan.ngroups)                                    # the key's radix: ranks are below it
        outs = [np.zeros(cap, t) for t in (np.uint64, np.int64, np.uint64, np.int64, np.uint64)]
        mm = plan.agg == algq.A_MM                                 # sparse output (TORBLES_GROUPBY=split: dense)
        if mm: plan.out_key, plan.out_sum, plan.out_cnt, plan.out_sum_hi, plan.out_key_hi = (o.ctypes.data for o in outs)
        else: plan.out_sum, plan.out_cnt = outs[1].ctypes.data, outs[2].ctypes.data
        plan.out_cap = cap
        ranks = np.zeros(dim.n, np.uint32)                         # in place of the dimension's ranked column
        plan.attr_vals[0] = ranks.ctypes.data
        fk = fact.data["fk"].astype(np.int64); fk = fk[fk < dim.n]
        g, h = algq.gpu_lib(), fact.gpu_handle()

        def groups(tag, tagged=True):
            tags = (ctypes.c_uint64 * 9)(); tags[0] = tag; plan.out_n = 0
            for o in outs: o[:] = 0
            with algq._GPU_LOCK:
                rc = g.gpu_query_tagged(h, ctypes.byref(plan), tags, None) if tagged else g.gpu_query(h, ctypes.byref(plan), None)
            self.assertEqual(rc, 0)
            if mm: return {int(k): int(c) for k, c in zip(outs[0][:plan.out_n], outs[2][:plan.out_n])}
            return {k: int(c) for k, c in enumerate(outs[2]) if c}

        def expect(r):
            out = {}
            for v in r[fk].tolist(): out[v] = out.get(v, 0) + 1
            return out

        def evict(*tags): return g.gpu_cache_evict(h, (ctypes.c_uint64 * len(tags))(*tags), len(tags))
        base = cache_entries(fact)                                 # the query's own: the dimension's ranked column
        def ours(): return [e for e in cache_entries(fact) if e not in base]
        b = ranks.nbytes
        t1, t2 = next(algq._GPU_SERIAL), next(algq._GPU_SERIAL)
        r1 = (np.arange(dim.n) % cap).astype(np.uint32); r2 = ((r1 + 1) % cap).astype(np.uint32)
        self.assertNotEqual(expect(r1), expect(r2))
        ranks[:] = r1
        self.assertEqual(groups(t1), expect(r1)); self.assertEqual(ours(), [(t1, b)])
        ranks[:] = r2                                              # other values, at the same address, same length:
        self.assertEqual(groups(t1), expect(r1))                   # one key, one copy (what the old key served a new array)
        self.assertEqual(groups(t2), expect(r2)); self.assertEqual(ours(), sorted([(t1, b), (t2, b)]))
        self.assertEqual(evict(t1), 1); self.assertEqual(ours(), [(t2, b)])
        self.assertEqual(groups(t1), expect(r2))                   # evicted: uploaded again, as it is now
        self.assertEqual(evict(t1, t2), 2); self.assertEqual(ours(), [])
        self.assertEqual(groups(0), expect(r2))                    # no tag: a copy for this query only
        self.assertEqual(groups(0, tagged=False), expect(r2))      # gpu_query: no tags
        self.assertEqual(ours(), [])
        filled = []                                                # a full cache (gpu.cu's NCACHE): copies for their query only
        while len(cache_entries(fact)) < 64 and len(filled) < 100:
            filled.append(next(algq._GPU_SERIAL)); self.assertEqual(groups(filled[-1]), expect(r2))
        self.assertEqual(len(cache_entries(fact)), 64)
        ranks[:] = r1; late = next(algq._GPU_SERIAL)
        self.assertEqual(groups(late), expect(r1)); self.assertEqual(len(cache_entries(fact)), 64)
        self.assertNotIn(late, [t for t, _ in cache_entries(fact)])
        self.assertEqual(evict(*filled), len(filled)); self.assertEqual(cache_entries(fact), base)
        self.assertEqual(q.run("gpu"), want)                       # the query's own copy, untouched

    def test_release_evicts_the_copies_of_a_tables_arrays(self):
        """close, and the finalizer, evict the copies of a dimension's columns from the fact's handle, while the
        columns themselves are still referenced (so their collection is not what evicts them)"""
        fact, dim = lookup_star()
        q, want = by_attribute(fact, dim)
        self.assertEqual(q.run("gpu"), want)
        (tag,) = dim._native.s.gtags                               # its ranked column: copied to the fact's handle
        self.assertEqual(cache_entries(fact), [(tag, dim.n * 4)])
        self.assertEqual(q.run("gpu"), want); self.assertEqual(cache_entries(fact), [(tag, dim.n * 4)])   # reused
        held = dim.ranked("a")[1]           # the column stays referenced: only the release can evict its copy
        dim.close()
        self.assertEqual(cache_entries(fact), []); self.assertEqual(dim._native.s.gtags, set())
        with self.assertRaises(TableClosedError): q.run("gpu")
        dim2 = dimension(dim.n, seed=21)                           # the finalizer: the last reference goes
        q2, want2 = by_attribute(fact, dim2)
        self.assertEqual(q2.run("gpu"), want2); self.assertEqual(len(cache_entries(fact)), 1)
        held2, fin = dim2.ranked("a")[1], dim2._native.release     # (referenced, as held)
        del q2, dim2; gc.collect()
        self.assertFalse(fin.alive); self.assertEqual(cache_entries(fact), [])
        self.assertIs(algq._GPU_TAGS[id(held2)][0](), held2)       # the column lives; its copy went with its table
        fact.close()                                               # the fact's handle: its copies go with it
        self.assertNotIn(fact._native.s, algq._GPU_LIVE)

    def test_a_new_array_where_a_freed_one_was_reads_its_own_values(self):
        """two array objects at one address, with one length: the ranked column of each of two dimensions joined to one
        fact, the second made where the first was (a buffer both use). The second query reads its own values, with
        the first one's copy still cached, and after it was evicted."""
        fact, d1 = lookup_star(seed=8)
        d2 = dimension(d1.n, seed=9)
        buf = (ctypes.c_uint32 * d1.n)()
        u1, r1 = d1.ranked("a")
        a1 = np.frombuffer(buf, np.uint32); a1[:] = r1; d1._ranked[("a", "alg")] = (u1, a1)
        q1, want1 = by_attribute(fact, d1)
        self.assertEqual(q1.run("gpu"), want1)
        (t1,) = d1._native.s.gtags; self.assertEqual(cache_entries(fact), [(t1, a1.nbytes)])
        u2, r2 = d2.ranked("a")
        a2 = np.frombuffer(buf, np.uint32); a2[:] = r2; d2._ranked[("a", "alg")] = (u2, a2)
        self.assertEqual((a2.ctypes.data, a2.nbytes), (a1.ctypes.data, a1.nbytes))
        q2, want2 = by_attribute(fact, d2)
        fk = fact.data["fk"].astype(np.int64); fk = fk[fk < d1.n]
        stale = {}                                                 # the answer from the first column's copy
        for v in u2[r1[fk]].tolist(): stale[(v,)] = stale.get((v,), 0) + 1
        self.assertNotEqual(want2, stale)
        self.assertEqual(q2.run("gpu"), want2)
        (t2,) = d2._native.s.gtags
        self.assertNotEqual(t1, t2); self.assertEqual(cache_entries(fact), sorted([(t1, a1.nbytes), (t2, a2.nbytes)]))
        d1.close(); self.assertEqual(cache_entries(fact), [(t2, a2.nbytes)])
        self.assertEqual(q2.run("gpu"), want2)

    def test_the_columns_of_a_table_bound_twice(self):
        """sqlexec binds a table that a query joins twice as a copy (copy.copy): the ranked columns the copy makes die
        with it while the table lives, and the next copy's (of another column, other values) are made where they were.
        Their copies go with them; each query reads its own."""
        fact, dim = lookup_star(seed=3)
        for col in ("a", "r") * 3:
            alias = copy.copy(dim)
            q, want = by_attribute(fact, alias, col)
            self.assertEqual(q.run("gpu"), want, col)
            self.assertEqual(len(cache_entries(fact)), 1)
            del q, alias; self.assert_live_copies_only()
            self.assertEqual(cache_entries(fact), [])
        self.assertIsNone(dim.__dict__.get("_ranked"))             # the columns were the copies' own

    def test_delta_planes(self):
        """a query over moved dimension rows (derived surrogates): its two passes read its delta plane, one copy for
        both, dropped when the query ends; successive moves (each query a new delta plane) answer exactly"""
        rng = np.random.default_rng(12)
        n, N = 300, 4000; k = np.arange(1, n + 1); w = int(n).bit_length()
        reg = rng.integers(0, 4, n)
        def surrogate(r, keys): return (np.asarray(r, np.int64) << w) | np.asarray(keys, np.int64)
        s_of_row = np.zeros(n + 1, np.int64); s_of_row[k] = surrogate(reg, k)
        sg = Surrogate(w, s_of_row)
        pick = rng.integers(0, n, N)
        def dimension_of(region):
            cols = {"region": np.zeros(n + 1, np.uint32), "key": np.zeros(n + 1, np.uint32)}
            cols["region"][k] = region; cols["key"][k] = k
            used = np.zeros(n + 1, bool); used[k] = True
            return Table("dima", cols, used=used)
        dim = dimension_of(reg)
        fact = Table("fact", dict(fka=sg.s_of_row[k[pick]].astype(np.uint32), m=u32(rng.integers(0, 100, N))),
                     msb=("fka",), refs=dict(fka=(dim, sg)))
        xbytes = (N + 63) // 64 * 8
        passes = []
        loop = algq.Query._exec_loop
        def spy(q, dev, table, plan):
            ok = loop(q, dev, table, plan)
            if dev == "gpu" and plan.xmode: passes.append((plan.xmode, ok, [e for e in cache_entries(fact) if e[1] == xbytes]))
            return ok
        algq.Query._exec_loop = spy; self.addCleanup(setattr, algq.Query, "_exec_loop", loop)
        for step in range(4):
            mv = rng.choice(n, 15, replace=False); reg = reg.copy(); reg[mv] = (reg[mv] + 1 + rng.integers(0, 3, 15)) % 4
            sg.move(k[mv], surrogate(reg[mv], k[mv]))
            dim = dimension_of(reg)                                # the dimension now
            q = Query(fact).join(dim, fact.fka).group_by(dim.region).sum(fact.m)
            want = {}
            for r_, m_ in zip(reg[pick].tolist(), fact.data["m"].tolist()): want[(r_,)] = want.get((r_,), 0) + m_
            passes.clear()
            self.assertEqual(q.run("gpu"), want, step); self.assertEqual(q.device, "gpu")
            self.assertEqual([p[:2] for p in passes], [(2, True), (1, True)])       # both passes on the GPU
            (a,), (b,) = passes[0][2], passes[1][2]
            self.assertEqual(a, b)                                 # one copy of the delta plane, for both passes
            del q; gc.collect()
            self.assertNotIn(a, cache_entries(fact))               # dropped with the plane
            self.assert_live_copies_only()
            self.assertEqual(Query(fact).join(dim, fact.fka).group_by(dim.region).sum(fact.m).run("alg"), want)


if __name__ == "__main__":
    unittest.main()
