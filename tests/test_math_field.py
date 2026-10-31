# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FIELD-001: packed GF(2) vectors and matrices, and exact parity dependencies over table rows (torbles/math_field.c,
math_field_gpu.cu, math_field.py).

Oracles, independent of the implementation: brute force over every small binary system written here (row spaces and
kernels enumerated, every right side solved by enumerating all x), a Gauss-Jordan elimination on Python ints (rows as
bit sets) for larger matrices and for the exact relation space of table rows, NumPy integer products mod 2 for
products and syndromes, Python booleans and integers on the source values, and DuckDB SQL (XOR as `<>` with SQL NULLs,
MIN(rowid) for witnesses). Every result with a CUDA path (syndromes, verification, discovery) is also computed there
(emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and compared for equality. Seeds are fixed.
"""
import itertools
import random
import unittest

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_field as mf
from torbles import math_handles as mh
from torbles.math_budgets import Budget
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError


# ---------------------------------------------------------------------- oracles (Python ints as bit sets)
def span(rows):
    """every XOR of a subset of rows"""
    out = {0}
    for r in rows:
        out |= {x ^ r for x in out}
    return out


def apply(rows, x):
    """A x as a bit set (bit i = row i)"""
    return sum(((bin(r & x).count("1") & 1) << i) for i, r in enumerate(rows))


def py_rref(rows, cols):
    rows, r, piv = list(rows), 0, []
    for c in range(cols):
        p = next((i for i in range(r, len(rows)) if (rows[i] >> c) & 1), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i] >> c) & 1:
                rows[i] ^= rows[r]
        piv.append(c)
        r += 1
    return rows, piv


def py_nullspace(rows, cols):
    R, piv = py_rref(rows, cols)
    out = []
    for f in range(cols):
        if f in piv:
            continue
        v = 1 << f
        for i, p in enumerate(piv):
            if (R[i] >> f) & 1:
                v |= 1 << p
        out.append(v)
    return out


def random_matrix(rng, rows, cols, density=0.5):
    return [sum(1 << j for j in range(cols) if rng.random() < density) for _ in range(rows)]


def obj(values):
    return pd.Series(values, dtype=object)


class Field(unittest.TestCase):
    # ------------------------------------------------------------------ exhaustive small systems
    def test_field001_rank_rref_nullspace_match_exhaustive_small_systems(self):
        for r, c in ((1, 1), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3), (2, 5), (4, 3), (3, 4)):
            for code in range(1 << (r * c)):
                rows = [(code >> (i * c)) & ((1 << c) - 1) for i in range(r)]
                A = mf.Matrix.from_ints(rows, c)
                space = span(rows)
                rank = A.rank()
                self.assertEqual(1 << rank, len(space), rows)
                R, piv = A.rref()
                Rint = R.to_ints()
                self.assertEqual(span(Rint), space)                       # same row space
                self.assertEqual(Rint[rank:], [0] * (r - rank))
                for i, p in enumerate(piv):                               # reduced echelon form
                    self.assertEqual(Rint[i] & ((1 << (p + 1)) - 1), 1 << p)
                    self.assertTrue(all(not (Rint[j] >> p) & 1 for j in range(r) if j != i))
                self.assertEqual(piv, sorted(piv))
                kernel = {x for x in range(1 << c) if apply(rows, x) == 0}
                N = A.nullspace().to_ints()
                self.assertEqual(len(N), c - rank)
                self.assertEqual(span(N), kernel)                         # a basis of the kernel
                free = [f for f in range(c) if f not in piv]
                for f, v in zip(free, N):                                 # canonical: e_f + pivot entries only
                    self.assertEqual(v & sum(1 << g for g in free), 1 << f)

    def test_field001_solutions_match_exhaustive_small_systems(self):
        for r, c in ((1, 2), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4)):
            for code in range(1 << (r * c)):
                rows = [(code >> (i * c)) & ((1 << c) - 1) for i in range(r)]
                A = mf.Matrix.from_ints(rows, c)
                images = {}
                for x in range(1 << c):
                    images.setdefault(apply(rows, x), set()).add(x)
                for b in range(1 << r):
                    out = A.solve(b)
                    sols = images.get(b, set())
                    if not sols:
                        self.assertEqual(out["status"], "none", (rows, b))
                        y = out["certificate"]
                        self.assertEqual(apply([sum(((rows[i] >> j) & 1) << i for i in range(r)) for j in range(c)], y), 0)
                        self.assertEqual(bin(y & b).count("1") & 1, 1)     # y^T A = 0, y^T b = 1
                        continue
                    self.assertEqual(out["status"], "unique" if len(sols) == 1 else "family", (rows, b))
                    self.assertIn(out["x"], sols)
                    self.assertEqual({out["x"] ^ v for v in span(out["basis"].to_ints())}, sols)
                    self.assertEqual(out["x"] & ~sum(1 << p for p in A.rref()[1]), 0)   # free variables 0

    # ------------------------------------------------------------------ larger matrices
    def test_field001_known_ranks_densities_and_products(self):
        rng = random.Random(7001)
        nrng = np.random.default_rng(7001)
        for n, ranks in ((32, (0, 1, 16, 32)), (128, (8, 64, 127, 128)), (200, (1, 100, 199))):
            for k in ranks:
                for density in (0.5, 0.05):
                    # L (n x k, full column rank) times U (k x n, full row rank): rank exactly k, rows and cols permuted
                    L = [(1 << i) if i < k else sum(1 << j for j in range(k) if rng.random() < density)
                         for i in range(n)]
                    U = [(1 << i) | (sum(1 << j for j in range(k, n) if rng.random() < density)) for i in range(k)]
                    Lm, Um = mf.Matrix.from_ints(L, k), mf.Matrix.from_ints(U, n)
                    A = Lm @ Um
                    perm = list(range(n)); rng.shuffle(perm)
                    A = mf.Matrix.from_ints([A.to_ints()[p] for p in perm], n)
                    self.assertEqual(A.rank(), k, (n, k, density))
                    R, piv = A.rref()
                    want, wpiv = py_rref(A.to_ints(), n)
                    self.assertEqual((R.to_ints(), piv), (want, wpiv))
                    N = A.nullspace()
                    self.assertEqual(N.to_ints(), py_nullspace(A.to_ints(), n))
                    self.assertTrue(all(A.matvec(v) == 0 for v in N.to_ints()))
        for r, n, c in ((5, 64, 7), (65, 129, 63), (128, 128, 128), (1, 200, 1)):
            a = (nrng.random((r, n)) < 0.3).astype(np.int64)
            b = (nrng.random((n, c)) < 0.3).astype(np.int64)
            A, B = mf.Matrix.from_bits(a), mf.Matrix.from_bits(b)
            self.assertTrue(np.array_equal((A @ B).to_bits(), (a @ b) % 2))
            self.assertTrue(np.array_equal((A @ B).T.to_bits(), ((a @ b) % 2).T))
            self.assertTrue(np.array_equal((A @ B).add(A @ B).to_bits(), np.zeros((r, c), np.uint8)))
            X = (nrng.random((300, n)) < 0.5).astype(np.int64)
            Xm = mf.Matrix.from_bits(X)
            want = (X @ a.T) % 2
            cpu, gpu = A.syndromes(Xm), A.syndromes(Xm, backend="cuda")
            self.assertTrue(np.array_equal(cpu.to_bits(), want))
            self.assertEqual(cpu, gpu)
            self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
            for j in (0, 17, 299):
                self.assertEqual(A.matvec(Xm.to_ints()[j]), cpu.to_ints()[j])

    # ------------------------------------------------------------------ boundaries and encoding
    def test_field001_dimension_and_encoding_boundaries(self):
        rng = random.Random(7101)
        for rows in (0, 1, 2, 63, 64, 65, 130):
            for cols in (0, 1, 63, 64, 65, 127, 128, 129):
                ints = random_matrix(rng, rows, cols, 0.4)
                A = mf.Matrix.from_ints(ints, cols)
                A.check()
                self.assertEqual(A.rank(), len(py_rref(ints, cols)[1]), (rows, cols))
                self.assertEqual(A.rref()[0].to_ints(), py_rref(ints, cols)[0])
                self.assertEqual(A.nullspace().to_ints(), py_nullspace(ints, cols))
                self.assertEqual(A.T.T, A)
                self.assertEqual(A @ mf.Matrix.identity(cols), A)
                self.assertEqual(mf.Matrix.identity(rows) @ A, A)
                X = mf.Matrix.from_ints(random_matrix(rng, 9, cols), cols)
                self.assertEqual(A.syndromes(X), A.syndromes(X, backend="cuda"))
                self.assertEqual(A.syndromes(X).to_ints(), [A.matvec(x) for x in X.to_ints()])
                wide = mf.Matrix(rows, cols, np.hstack([A.words, np.zeros((rows, 2), np.uint64)]), stride=A.stride + 2)
                self.assertEqual((wide.rank(), wide.rref()[0], wide.nullspace()), (A.rank(), A.rref()[0], A.nullspace()))
                if rows and cols % 64:                                  # a stray bit past cols is refused
                    bad = mf.Matrix(rows, cols, A.words.copy())
                    bad.words[rows - 1, -1] |= np.uint64(1) << np.uint64(63)
                    for call in (bad.check, bad.rank, bad.rref, bad.nullspace, lambda: bad @ mf.Matrix.identity(cols),
                                 lambda: bad.syndromes(X), lambda: bad.syndromes(X, backend="cuda"), lambda: bad.T):
                        with self.assertRaises(ValueError):
                            call()
                if rows:                                                # stride padding must be zero too
                    pad = mf.Matrix(rows, cols, np.hstack([A.words, np.ones((rows, 1), np.uint64)]), stride=A.stride + 1)
                    with self.assertRaises(ValueError):
                        pad.rank()
        with self.assertRaises(ValueError):
            mf.parity(1 << 70, 70)                                   # a bit past the vector's length
        with self.assertRaises(ValueError):
            mf.Matrix.from_ints([3], 3).solve(0b10, basis=False)       # b has rows bits
        self.assertEqual(mf.Matrix(0, 0).rank(), 0)
        self.assertEqual(mf.Matrix(0, 5).solve(0)["status"], "family")
        self.assertEqual(mf.Matrix(3, 0).solve(0)["status"], "unique")
        self.assertEqual(mf.Matrix(3, 0).solve(0b100)["status"], "none")

    def test_field001_parity_is_not_popcount(self):
        rng = random.Random(7201)
        for n in (1, 63, 64, 65, 200):
            for _ in range(40):
                a, b = rng.getrandbits(n), rng.getrandbits(n)
                wa, wb, wab = mf.weight(a, n), mf.weight(b, n), mf.weight(a & b, n)
                self.assertEqual((wa, wb), (bin(a).count("1"), bin(b).count("1")))
                self.assertEqual(mf.parity(a, n), wa % 2)                          # parity = weight mod 2
                self.assertEqual(mf.weight(a ^ b, n), wa + wb - 2 * wab)           # not wa + wb
                self.assertEqual(mf.dot(a, b, n), wab % 2)                          # a field element, not wab
        ones = (1 << 70) - 1
        three = mf.Matrix.from_ints([ones] * 3, 70)
        self.assertEqual((three.row_sum(), three.column_counts()), (ones, [3] * 70))   # XOR sum vs integer counts
        four = mf.Matrix.from_ints([ones] * 4, 70)
        self.assertEqual((four.row_sum(), four.column_counts()), (0, [4] * 70))
        self.assertEqual(mf.parity(ones, 70), 0)
        self.assertEqual(mf.weight(ones, 70), 70)
        M = mf.Matrix.from_ints(random_matrix(rng, 33, 130), 130)
        counts, total = M.column_counts(), M.row_sum()
        self.assertEqual(sum((c & 1) << j for j, c in enumerate(counts)), total)
        self.assertEqual(counts, [sum((r >> j) & 1 for r in M.to_ints()) for j in range(130)])
        # over table rows: a parity relation holds where the integer sum is even, not where it is 0
        frame = pd.DataFrame({"x": [True, True, False, True], "y": [True, False, False, True], "z": [False, True, False, True]})
        db = torbles.from_ddl("CREATE TABLE t (x BOOLEAN, y BOOLEAN, z BOOLEAN)", {"t": frame})
        bits = mf.Bits.from_scope(mbk.Scope.from_table(db["t"], ["x", "y", "z"]), ["x", "y", "z"])
        ev = bits.verify([(["x", "y", "z"], 0)])[0]
        sums = [int(a) + int(b) + int(c) for a, b, c in zip(frame.x, frame.y, frame.z)]   # 2, 2, 0, 3
        self.assertEqual((ev["violations"], ev["witness"]), (sum(s % 2 for s in sums), 3))
        self.assertEqual(sum(s == 0 for s in sums), 1)               # "sum = 0" would reject rows 0 and 1

    # ------------------------------------------------------------------ parity dependencies over table rows
    def planted(self, rng, n, near_row, nulls=False):
        a, b, c, g = (rng.integers(0, 2, n).astype(bool) for _ in range(4))
        a[0] = b[0] = c[0] = False                               # row 0 holds the smallest code, a multiple of 4,
        g[0] = True                                              # so the codec offset keeps the two low bit planes
        cols = dict(a=a, b=b, c=c, g=g, d=a ^ b, e=~(b ^ c), f=a ^ c ^ g)
        cols["f"] = cols["f"].copy()
        cols["f"][near_row] = ~cols["f"][near_row]                         # a one-row near miss
        q = rng.integers(-39, 40, n)
        q[0] = -40
        code = q * 4 + (a ^ c).astype(int) + 2 * (b ^ g ^ True).astype(int)
        values = {k: [bool(x) for x in v] for k, v in cols.items()}
        values["code"] = [int(x) for x in code]
        if nulls:
            for i in rng.choice(n, size=n // 25, replace=False):
                values["g"][int(i)] = None
        frame = pd.DataFrame({k: obj(v) for k, v in values.items()})
        ddl = "CREATE TABLE t (a BOOLEAN, b BOOLEAN, c BOOLEAN, g BOOLEAN, d BOOLEAN, e BOOLEAN, f BOOLEAN, code BIGINT)"
        return torbles.from_ddl(ddl, {"t": frame}), values

    SPECS = ["a", "b", "c", "g", "d", "e", "f", ("code", "code", 0), ("code", "code", 1), ("code", "bit", 63)]

    def oracle_bits(self, values, offset):
        """per variable of SPECS: a list of True / False / None per row (from the source values)"""
        out = [values[k] for k in ("a", "b", "c", "g", "d", "e", "f")]
        code = values["code"]
        out.append([None if v is None else bool(((v - offset) >> 0) & 1) for v in code])
        out.append([None if v is None else bool(((v - offset) >> 1) & 1) for v in code])
        out.append([None if v is None else bool((v % (1 << 64)) >> 63) for v in code])
        return out

    def test_field001_planted_parity_dependencies_and_one_row_near_misses(self):
        rng = np.random.default_rng(7301)
        n, sample = 4000, 100
        stride = n // sample
        near = 7 * stride + 3                                    # not one of the sampled rows
        db, values = self.planted(rng, n, near)
        scope = mbk.Scope.from_table(db["t"], ["a", "b", "c", "g", "d", "e", "f", "code"])
        bits = mf.Bits.from_scope(scope, self.SPECS)
        offset = scope.column("code")["offset"]
        ob = self.oracle_bits(values, offset)
        for j in range(len(self.SPECS)):                          # the bit columns themselves
            value, valid = bits.words(j)
            got = np.unpackbits(value.view(np.uint8), bitorder="little")[:n].astype(bool)
            self.assertEqual(got.tolist(), [bool(x) for x in ob[j]], self.SPECS[j])
        cpu = bits.discover(sample=sample)
        gpu = bits.discover(sample=sample, backend="cuda")
        strip = lambda r: {k: v for k, v in r.items() if k not in ("backend", "device")}
        self.assertEqual(strip(cpu), strip(gpu))
        # the exact relation space of the rows: Python elimination over every row ([1, x0, .., x9])
        vecs = [1 | sum(int(ob[j][r]) << (j + 1) for j in range(10)) for r in range(n)]
        want = py_nullspace(vecs, 11)
        got = [int(rel["constant"]) | sum(1 << (j + 1) for j in rel["vars"]) for rel in cpu["relations"]]
        self.assertEqual(got, want)
        self.assertEqual((cpu["rank"], cpu["complete"], cpu["excluded"]), (11 - len(want), n, 0))
        labels = {(rel["labels"], rel["constant"]) for rel in cpu["relations"]}
        self.assertIn((("a", "b", "d"), 0), labels)
        self.assertIn((("b", "c", "e"), 1), labels)
        self.assertIn((("a", "c", "code.code0"), 0), labels)
        self.assertIn((("b", "g", "code.code1"), 1), labels)
        for rel in cpu["relations"]:
            self.assertEqual((rel["violations"], rel["checked"], rel["holds"]), (0, n, True))
        self.assertEqual([(r["labels"], r["constant"], r["violations"], r["witness"]) for r in cpu["rejected"]],
                         [(("a", "c", "g", "f"), 0, 1, near)])
        self.assertEqual((cpu["rounds"], cpu["initial_sample"], cpu["sample_rows"]), (2, sample, sample + 1))
        prop = bits.propose(sample=sample)                       # candidate generation alone: the first round
        self.assertEqual(sorted((p["vars"], p["constant"]) for p in prop["proposals"]),
                         sorted((r["vars"], r["constant"]) for r in cpu["relations"] + cpu["rejected"]))
        self.assertEqual((prop["sample_rows"], prop["sample_rank"], prop["complete"]), (sample, cpu["rank"] - 1, n))
        # verification against DuckDB (XOR as <>), SQL NULL semantics, smallest violating rowid
        con = db.con
        for rel, sql in ((["a", "c", "g", "f"], '((("a" <> "c") <> "g") <> "f")'), (["a", "b", "d"], '(("a" <> "b") <> "d")'),
                         (["b", "e"], '("b" <> "e")')):
            want = con.execute(f"SELECT COUNT(*) FILTER (WHERE {sql} IS NOT NULL), COUNT(*) FILTER (WHERE {sql}), "
                               f"MIN(rowid) FILTER (WHERE {sql}) FROM t").fetchall()[0]
            for backend in ("cpu", "cuda"):
                ev = bits.verify([(rel, 0)], backend=backend)[0]
                self.assertEqual((ev["checked"], ev["violations"], ev["witness"]),
                                 (int(want[0]), int(want[1]), None if want[2] is None else int(want[2])))
        # substitution: d is a XOR b on every row
        value, valid = bits.combine((["a", "b"], 0))
        self.assertTrue(np.array_equal(value, bits.words(4)[0]))
        value, valid = bits.combine((["b", "c"], 1))
        self.assertTrue(np.array_equal(value, bits.words(5)[0]))

    def test_field001_nulls_listwise_discovery_and_sql_verification_match_duckdb(self):
        rng = np.random.default_rng(7401)
        n = 2500
        db, values = self.planted(rng, n, near_row=11, nulls=True)
        sel = rng.random(n) < 0.8
        scope = mbk.Scope.from_table(db["t"], ["a", "b", "c", "g", "d", "e", "f", "code"], rows=sel)
        bits = mf.Bits.from_scope(scope, self.SPECS)
        self.assertTrue(bits.info["has_null"])
        con = db.con
        con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
        where = "rowid IN (SELECT rid FROM sel_rows)"
        for rel, sql in ((["a", "c", "g", "f"], '((("a" <> "c") <> "g") <> "f")'),
                         (["c", "g"], '("c" <> "g")'), (["a", "b", "d"], '(("a" <> "b") <> "d")')):
            want = con.execute(f"SELECT COUNT(*) FILTER (WHERE {sql} IS NOT NULL), COUNT(*) FILTER (WHERE {sql} IS NULL), "
                               f"COUNT(*) FILTER (WHERE {sql}), MIN(rowid) FILTER (WHERE {sql}) FROM t WHERE {where}"
                               ).fetchall()[0]
            for backend in ("cpu", "cuda"):
                ev = bits.verify([(rel, 0)], mode="sql", backend=backend)[0]
                self.assertEqual((ev["checked"], ev["excluded"], ev["violations"], ev["witness"]),
                                 (int(want[0]), int(want[1]), int(want[2]), None if want[3] is None else int(want[3])))
                lw = bits.verify([(rel, 0)], mode="listwise", backend=backend)[0]
                nulls = int(con.execute(f'SELECT COUNT(*) FILTER (WHERE "g" IS NULL) FROM t WHERE {where}').fetchall()[0][0])
                self.assertEqual((lw["checked"], lw["excluded"]), (int(sel.sum()) - nulls, nulls))
        cpu, gpu = bits.discover(sample=0), bits.discover(sample=0, backend="cuda")
        self.assertEqual({k: v for k, v in cpu.items() if k not in ("backend", "device")},
                         {k: v for k, v in gpu.items() if k not in ("backend", "device")})
        ob = self.oracle_bits(values, scope.column("code")["offset"])
        complete = [r for r in np.flatnonzero(sel) if all(ob[j][r] is not None for j in range(10))]
        vecs = [1 | sum(int(ob[j][r]) << (j + 1) for j in range(10)) for r in complete]
        got = [int(rel["constant"]) | sum(1 << (j + 1) for j in rel["vars"]) for rel in cpu["relations"]]
        self.assertEqual(got, py_nullspace(vecs, 11))
        self.assertEqual((cpu["complete"], cpu["excluded"]), (len(complete), int(sel.sum()) - len(complete)))
        for rel in cpu["relations"]:                              # exact on the complete rows, not on the selection
            self.assertEqual((rel["violations"], rel["holds"]), (0, False))
        self.assertEqual(cpu["rounds"], 1)                        # sample = every complete row: nothing to reject
        empty = mf.Bits.from_scope(mbk.Scope.from_table(db["t"], ["a", "b"], rows=np.zeros(n, bool)), ["a", "b"])
        r = empty.discover(sample=0)                              # no complete row: every vector holds vacuously
        self.assertEqual([(x["vars"], x["constant"]) for x in r["relations"]], [((), 1), ((0,), 0), ((1,), 0)])

    def test_field001_bit_planes_of_signed_codes(self):
        xs = [-(1 << 62), -5, -1, 0, 1, 6, (1 << 62) + 3, None]
        frame = pd.DataFrame({"x": obj(xs)})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x"])
        offset = scope.column("x")["offset"]
        specs = [("x", "bit", 0), ("x", "bit", 63), ("x", "bit", 2, -9), ("x", "code", 62), ("x", "code", 0),
                 ("x", "nonzero")]
        bits = mf.Bits.from_scope(scope, specs)
        for j, s in enumerate(specs):
            value, valid = bits.words(j)
            got = [bool((int(value[0]) >> r) & 1) if (int(valid[0]) >> r) & 1 else None for r in range(len(xs))]
            base = offset if s[1] == "code" else (s[3] if len(s) == 4 else 0)
            want = [None if v is None else (v != 0 if s[1] == "nonzero" else bool((((v - base) % (1 << 64)) >> s[2]) & 1))
                    for v in xs]
            self.assertEqual(got, want, s)
        with self.assertRaises(ValueError):
            mf.Bits.from_scope(scope, [("x", "bit", 64)])
        with self.assertRaises(ValueError):
            mf.Bits.from_scope(scope, ["x", ("x", "nonzero")])

    # ------------------------------------------------------------------ budgets, staleness, the ABI
    def test_field001_budgets_at_the_limit_and_one_past(self):
        rng = random.Random(7501)
        A = mf.Matrix.from_ints(random_matrix(rng, 40, 50), 50)
        X = mf.Matrix.from_ints(random_matrix(rng, 30, 50), 50)

        def edge(fn, resource, need):
            fn(Budget(**{resource: need}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                fn(Budget(**{resource: need - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need))
        edge(lambda b: A.rank(budget=b), "tensor", 2000)
        edge(lambda b: A.rank(budget=b), "bytes", 40 * 1 * 8)
        edge(lambda b: A.rref(budget=b), "output", 2000)
        nullity = 50 - A.rank()
        edge(lambda b: A.nullspace(budget=b), "output", nullity * 50)
        edge(lambda b: A.solve(0, budget=b), "tensor", 40 * (50 + 1 + 40))
        edge(lambda b: A.mul(mf.Matrix.identity(50), budget=b), "tensor", 2000 + 2500 + 2000)
        edge(lambda b: A.T.mul(A, budget=b), "output", 2500)
        edge(lambda b: A.syndromes(X, budget=b), "output", 30 * 40)
        edge(lambda b: A.syndromes(X, backend="cuda", budget=b), "bytes", 40 * 8 + 30 * 8 + 30 * 8)
        rng2 = np.random.default_rng(7502)
        db, values = self.planted(rng2, 3000, near_row=5)
        scope = mbk.Scope.from_table(db["t"], ["a", "b", "c", "g", "d", "e", "f", "code"])
        nb = mf.bits_need(scope, self.SPECS)["bytes"]
        edge(lambda b: mf.Bits.from_scope(scope, self.SPECS, budget=b), "bytes", nb)
        bits = mf.Bits.from_scope(scope, self.SPECS)
        rel = [(["a", "b", "d"], 0)] * 3
        edge(lambda b: bits.verify(rel, budget=b), "output", 3)
        nw = (3000 + 63) // 64
        edge(lambda b: bits.verify(rel, backend="cuda", budget=b), "bytes", nw * 8 + 10 * nw * 8 + 3 * 48)
        for backend in ("cpu", "cuda"):
            need = bits.discover_need(sample=64, backend=backend)
            self.assertEqual((need["variables"], need["iterations"], need["support"], need["output"]),
                             (11, 12, 64, 11 + 66))
            for resource in ("variables", "iterations", "support", "output", "bytes"):
                edge(lambda b: bits.discover(sample=64, backend=backend, budget=b), resource, need[resource])
        big = mf._Matrix((1 << 31) + 1, 1, 1, None)                # past the representation limit, natively
        self.assertEqual(mf._call("tr_field_check", mf.C.byref(big)), -12)
        with self.assertRaises(ValueError):
            mf.Matrix((1 << 31) + 1, 1)

    def test_field001_stale_scope_is_refused(self):
        rng = np.random.default_rng(7601)
        db, values = self.planted(rng, 500, near_row=3)
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["a", "b", "d"])
        bits = mf.Bits.from_scope(scope, ["a", "b", "d"])
        self.assertTrue(bits.verify([(["a", "b", "d"], 0)])[0]["holds"])
        t.flip(0, 0)                                              # a mutation: new revision
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                bits.verify([(["a", "b", "d"], 0)], backend=backend)
            with self.assertRaises(StaleStateError):
                bits.discover(backend=backend)
        with self.assertRaises(StaleStateError):
            mf.Bits.from_scope(scope, ["a"])

    def test_field001_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_field.")}
        self.assertEqual(mine, {"tr_field_matrix", "tr_field_bitspec", "tr_field_bits_view", "tr_field_relation",
                                "tr_field_evidence", "tr_field_discover_spec", "tr_field_discovery",
                                "tr_field_bits_info"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_field.")}
        self.assertEqual(cuda, {"tr_field_matrix", "tr_field_bits_view", "tr_field_relation", "tr_field_evidence",
                                "tr_field_discover_spec", "tr_field_discovery"})


if __name__ == "__main__":
    unittest.main()
