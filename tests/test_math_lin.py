# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LIN-001 (grouped moment matrices and regression systems) and LIN-002 (solves, projections, covariance and
eigensystems): torbles/math_lin.c, math_lin_gpu.cu, math_lin.py.

Oracles, independent of the implementation: Python int / Fraction arithmetic written here (explicit matrix products
over the selected rows, Gauss-Jordan elimination to the reduced row echelon form, symmetric elimination for the
inertia, least squares from the raw rows), DuckDB SQL (SUM of products in HUGEINT with GROUP BY and SQL NULLs), and
NumPy (eigvalsh) only for the numerical eigensystems' declared tolerances. Every native result is also computed on
the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and compared for exact equality.
Seeds are fixed.
"""
import math
import unittest
from fractions import Fraction as F

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_lin as lin
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import NotConvergedError, StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

EPS = 2.0 ** -52


def obj(values):
    return pd.Series(values, dtype=object)


# ---------------------------------------------------------------------- independent exact oracles
def rref(M):
    """Gauss-Jordan with Fractions: (R rows, pivot columns)"""
    M = [[F(v) for v in row] for row in M]
    rows, cols = len(M), len(M[0])
    r, piv = 0, []
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = 1 / M[r][c]
        M[r] = [v * inv for v in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return M[:r], piv


def oracle_solve(A, b):
    """(kind, rank, pivots, x0, nullspace) of A x = b by Fraction elimination of [A | b]"""
    n = len(A[0])
    R, piv = rref([list(row) + [bb] for row, bb in zip(A, b)])
    if piv and piv[-1] == n:
        return "none", len(piv) - 1, piv[:-1], None, None
    x = [F(0)] * n
    for t, p in enumerate(piv):
        x[p] = R[t][n]
    null = []
    for f in range(n):
        if f in piv:
            continue
        v = [F(0)] * n
        v[f] = F(1)
        for t, p in enumerate(piv):
            v[p] = -R[t][f]
        null.append(v)
    return ("unique" if len(piv) == n else "family"), len(piv), piv, x, null


def oracle_inertia(A):
    """(n+, n-, n0) of a symmetric Fraction matrix by symmetric elimination (a zero diagonal with a nonzero
    off-diagonal entry is resolved by the congruence e_i += e_j)"""
    S = [[F(v) for v in row] for row in A]
    idx, pos, neg = list(range(len(S))), 0, 0
    while idx:
        i = next((i for i in idx if S[i][i] != 0), None)
        if i is None:
            pair = next(((i, j) for a, i in enumerate(idx) for j in idx[a + 1:] if S[i][j] != 0), None)
            if pair is None:
                break
            i, j = pair
            for c in idx:
                S[i][c] += S[j][c]
            for c in idx:
                S[c][i] += S[c][j]
            continue
        p = S[i][i]
        pos, neg = pos + (p > 0), neg + (p < 0)
        idx.remove(i)
        for a in idx:
            for c in idx:
                S[a][c] -= S[a][i] * S[i][c] / p
    return pos, neg, len(A) - pos - neg


def matvec(A, x):
    return [sum(F(a) * v for a, v in zip(row, x)) for row in A]


def quad(A, v):
    return sum(F(v[i]) * F(A[i][j]) * F(v[j]) for i in range(len(v)) for j in range(len(v)))


# ---------------------------------------------------------------------- data
DDL = "CREATE TABLE f (k BIGINT, x BIGINT, d DECIMAL(12,2), y BIGINT, w BIGINT)"


def cents(c):
    return None if c is None else (f"{c // 100}.{c % 100:02d}" if c >= 0 else f"-{(-c) // 100}.{(-c) % 100:02d}")


def dataset(rng, n, nulls=0.12):
    """signed integers and decimals with NULLs, duplicated rows, an all-NULL group (k = 6) and an absent key (k = 5)"""
    k = [int(v) for v in rng.integers(0, 5, n)]
    x = [int(v) for v in rng.integers(-10**6, 10**6, n)]
    d = [int(v) for v in rng.integers(-10**7, 10**7, n)]               # cents
    y = [int(v) for v in rng.integers(-10**5, 10**5, n)]
    w = [int(v) for v in rng.integers(-1000, 1000, n)]
    for col in (x, d, y, w):
        for i in np.flatnonzero(rng.random(n) < nulls):
            col[int(i)] = None
    for i in rng.choice(n, 6, replace=False):                           # k = 6: every value NULL
        k[int(i)], x[int(i)], d[int(i)], y[int(i)], w[int(i)] = 6, None, None, None, None
    dup = [int(i) for i in rng.choice(n, n // 5, replace=False)]        # duplicated rows (bag semantics)
    for col in (k, x, d, y, w):
        col.extend([col[i] for i in dup])
    frame = pd.DataFrame({"k": obj(k), "x": obj(x), "d": obj([cents(c) for c in d]), "y": obj(y), "w": obj(w)})
    db = torbles.from_ddl(DDL, {"f": frame})
    return db, dict(k=k, x=x, d=d, y=y, w=w)


def explicit_gram(raw, names, keys, rows, policy):
    """{key: (rows, gram, counts)} by explicit products over the selected rows (Python ints)"""
    out = {}
    cols = [None] + list(names)
    for r in range(len(raw["k"])):
        if rows is not None and not rows[r]:
            continue
        key = tuple(raw[c][r] for c in keys)
        z = [1] + [raw[c][r] for c in names]
        e = out.setdefault(key, [0, [[0] * len(cols) for _ in cols], [[0] * len(cols) for _ in cols]])
        e[0] += 1
        if policy == "listwise" and any(v is None for v in z):
            continue
        for a in range(len(cols)):
            for b in range(len(cols)):
                if z[a] is None or z[b] is None:
                    continue
                e[1][a][b] += z[a] * z[b]
                e[2][a][b] += 1
    return out


def logical(name):
    return {"x": 'CAST(f."x" AS HUGEINT)', "d": 'CAST(f."d" * 100 AS HUGEINT)', "y": 'CAST(f."y" AS HUGEINT)',
            "w": 'CAST(f."w" AS HUGEINT)'}[name]


def duck_gram(db, names, rows, policy):
    """{(k,): (rows, gram, counts)} by DuckDB: SUM of HUGEINT products per group (SQL NULLs; FILTER for listwise)"""
    con = db.con
    where = "TRUE"
    if rows is not None:
        con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(np.asarray(rows)).astype(np.int64)}))
        where = "f.rowid IN (SELECT rid FROM sel_rows)"
    z = ["CAST(1 AS HUGEINT)"] + [logical(n) for n in names]
    complete = " AND ".join(f'f."{n}" IS NOT NULL' for n in names)
    parts = []
    for a in range(len(z)):
        for b in range(len(z)):
            expr = f"({z[a]} * {z[b]})"
            if policy == "listwise":
                parts += [f"SUM({expr}) FILTER (WHERE {complete})", f"COUNT(*) FILTER (WHERE {complete})"]
            else:
                parts += [f"SUM({expr})", f"COUNT({expr})"]
    sql = f'SELECT f."k", COUNT(*), {", ".join(parts)} FROM f WHERE {where} GROUP BY f."k"'
    out = {}
    m = len(z)
    for r in con.execute(sql).fetchall():
        vals = r[2:]
        gram = [[0 if vals[2 * (a * m + b)] is None else int(vals[2 * (a * m + b)]) for b in range(m)] for a in range(m)]
        cnt = [[int(vals[2 * (a * m + b) + 1]) for b in range(m)] for a in range(m)]
        out[(r[0],)] = (int(r[1]), gram, cnt)
    return out


def state_by_key(st):
    out = {}
    for g in range(st.groups):
        s = st.state(g)
        if s["present"]:
            out[st.scope.key_values(g)] = (s["rows"], st.matrix(g), st.counts(g))
    return out


class Lin(unittest.TestCase):
    def both(self, scope, features, target=None, **kw):
        """the CPU state, checked equal (natively and word for word) to the emulated, scrambled CUDA state"""
        cpu = lin.moments(scope, features, target, **kw)
        gpu = lin.moments(scope, features, target, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu), kw)
        for a, b in zip(cpu.arrays(), gpu.arrays()):
            self.assertTrue(np.array_equal(a, b))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    # ================================================================== LIN-001
    def test_lin001_match_explicit_products_signed_duplicated_selectively_grouped(self):
        rng = np.random.default_rng(1001)
        db, raw = dataset(rng, 1500)
        t = db["f"]
        names = ["x", "d", "y"]
        for rows in (None, rng.random(t.n) < 0.65):
            scope = mbk.Scope.from_table(t, names, group_by=["k"], rows=rows)
            for policy in ("listwise", "pairwise"):
                st = self.both(scope, ["x", "d"], "y", policy=policy)
                got = state_by_key(st)
                expect = explicit_gram(raw, names, ["k"], rows, policy)
                self.assertEqual(got, {key: (e[0], e[1], e[2]) for key, e in expect.items()}, policy)
                self.assertEqual(got, duck_gram(db, names, rows, policy), policy)
                self.assertEqual(st.rows, scope.selected)
                self.assertEqual(st.present, len(expect))
                self.assertEqual(st.scales, (0, 0, 2, 0))
                for g in range(st.groups):                    # absent groups: no rows, zero states
                    s = st.state(g)
                    if not s["present"]:
                        self.assertEqual((s["rows"], set(s["count"]), set(s["sum"])), (0, {0}, {0}))
                key6 = [g for g in range(st.groups) if scope.key_values(g) == (6,)][0]
                s6 = st.state(key6)                           # all NULL: present, count 0 except the constant
                self.assertTrue(s6["present"] and s6["rows"] > 0)
                self.assertEqual(set(s6["sum"][1:]), {0})
                if policy == "listwise":
                    self.assertEqual(st.matrix(key6)[0][0], 0)
                else:
                    self.assertEqual(st.matrix(key6)[0][0], s6["rows"])
                self.assertFalse(st.state([g for g in range(st.groups) if scope.key_values(g) == (5,)][0])["present"])
        # duplicated rows count twice: a scope of one row and of its duplicate pair
        frame = pd.DataFrame({"k": [1, 1], "x": [-7, -7], "d": ["2.50", "2.50"], "y": [3, 3], "w": [0, 0]})
        db2 = torbles.from_ddl(DDL, {"f": frame})
        st = self.both(mbk.Scope.from_table(db2["f"], ["x", "d", "y"]), ["x", "d"], "y")
        self.assertEqual(st.matrix(0), [[2, -14, 500, 6], [-14, 98, -3500, -42], [500, -3500, 125000, 1500],
                                        [6, -42, 1500, 18]])

    def test_lin001_many_groups_arrays_and_the_bank_path(self):
        rng = np.random.default_rng(1002)
        n, G = 2500, 40                                         # > 16 groups: the flush-on-change kernel path
        x1, x2 = rng.integers(-2**31, 2**31, n), rng.integers(-500, 500, n)
        y = rng.integers(-2**20, 2**20, n)
        v1, v2 = rng.random(n) < 0.85, rng.random(n) < 0.9
        groups = rng.integers(0, G, n)
        groups[groups == 7] = 8                                 # group 7 absent
        rows = rng.random(n) < 0.8
        scope = mbk.Scope.from_arrays({"x1": x1, "x2": x2, "y": y}, valid={"x1": v1, "x2": v2}, groups=groups,
                                      ngroups=G, rows=rows)
        for policy in ("listwise", "pairwise"):
            st = self.both(scope, ["x1", "x2"], "y", policy=policy)
            for g in range(G):
                sel = rows & (groups == g)
                z = [np.ones(n, dtype=object), x1.astype(object), x2.astype(object), y.astype(object)]
                ok = [np.ones(n, bool), v1, v2, np.ones(n, bool)]
                full = sel & v1 & v2
                gram = [[int(sum(z[a][full] * z[b][full])) if policy == "listwise" else
                         int(sum(z[a][sel & ok[a] & ok[b]] * z[b][sel & ok[a] & ok[b]])) for b in range(4)] for a in range(4)]
                self.assertEqual(st.matrix(g), gram, (policy, g))
                self.assertEqual(st.state(g)["rows"], int(sel.sum()))
            # the bank path: monomials of degree <= 2, joint (listwise) or sql (pairwise) mode, same Gram slots
            bank = mbk.build(scope, mbk.monomials(["x1", "x2", "y"], 2), "joint" if policy == "listwise" else "sql")
            fb = lin.moments_from_bank(bank, ["x1", "x2"], "y")
            self.assertEqual(fb.policy, policy)
            self.assertEqual(fb.source, "bank")
            self.assertFalse(fb.marginals)
            for g in range(G):
                self.assertEqual((fb.matrix(g), fb.counts(g), fb.state(g)["rows"]),
                                 (st.matrix(g), st.counts(g), st.state(g)["rows"]))
            if policy == "listwise":
                self.assertTrue(fb.equal(lin.moments(scope, ["x1", "x2"], "y")))
        # a joint bank over more columns than the spec is another listwise domain
        bank = mbk.build(scope, mbk.monomials(["x1", "x2", "y"], 2), "joint")
        with self.assertRaises(DomainError):
            lin.moments_from_bank(bank, ["x1"], "y")
        with self.assertRaises(ValueError):                    # the basis lacks the Gram monomials
            lin.moments_from_bank(mbk.build(scope, ["1", "x1", "y"], "sql"), ["x1"], "y")

    def test_lin001_wide_sums_use_192_bit_words_exactly(self):
        rng = np.random.default_rng(1003)
        n = 600
        x = rng.integers(-(1 << 62), 1 << 62, n)
        y = rng.integers(-(1 << 61), 1 << 61, n)
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, groups=rng.integers(0, 3, n), ngroups=3)
        st = self.both(scope, ["x"], "y")
        self.assertEqual(st.width, 192)                          # 600 * 2^62 * 2^62 > 2^127: not proven for 128
        for g in range(3):
            sel = np.flatnonzero(scope_groups(scope, n) == g)
            z = [[1] * len(sel), [int(x[i]) for i in sel], [int(y[i]) for i in sel]]
            self.assertEqual(st.matrix(g), [[sum(p * q for p, q in zip(z[a], z[b])) for b in range(3)] for a in range(3)])
        self.assertGreater(abs(st.matrix(0)[1][1]), 1 << 127)    # past HUGEINT
        with self.assertRaises(OverflowError):
            lin.moments(scope, ["x"], "y", width=128)
        with self.assertRaises(OverflowError):
            lin.moments(scope, ["x"], "y", width=128, backend="cuda")
        small = mbk.Scope.from_arrays({"x": x % 1000, "y": y % 1000})
        a, b = self.both(small, ["x"], "y"), self.both(small, ["x"], "y", width=192)
        self.assertEqual((a.width, b.width), (128, 192))
        self.assertEqual(a.matrix(0), b.matrix(0))

    def test_lin001_pairwise_and_common_listwise_are_distinct_declared_policies(self):
        # pair-complete correlations +1 (x, y), +1 (y, z), -1 (x, z): pairwise covariance is indefinite
        x = [1, 2, None, None, 1, 2, 0, 1]
        y = [1, 2, 1, 2, None, None, 0, 1]
        z = [None, None, 1, 2, -1, -2, 0, 1]
        frame = pd.DataFrame({"k": [0] * 8, "x": obj(x), "d": obj(["1.00"] * 8), "y": obj(y), "w": obj(z)})
        db = torbles.from_ddl(DDL, {"f": frame})
        scope = mbk.Scope.from_table(db["f"], ["x", "y", "w"])
        lw = self.both(scope, ["x", "y"], "w", policy="listwise")
        pw = self.both(scope, ["x", "y"], "w", policy="pairwise")
        self.assertEqual((lw.policy, pw.policy), ("listwise", "pairwise"))
        self.assertFalse(lw.equal(pw))
        self.assertEqual({c for row in lw.counts(0) for c in row}, {2})          # rows 6, 7 are complete
        self.assertEqual(pw.counts(0)[1][2], 4)                                 # x and y together
        self.assertEqual(pw.counts(0)[1][1], 6)                                 # x alone
        self.assertEqual(pw.counts(0)[0][0], 8)
        # either raw Gram matrix is positive semidefinite (pairwise: the Gram matrix of the data with NULL = 0)
        self.assertEqual(lin.inertia(lw.matrix(0)).negative, 0)
        self.assertEqual(lin.inertia(pw.matrix(0)).negative, 0)
        zero_filled = [[1, a or 0, b or 0, c or 0] for a, b, c in zip(x, y, z)]
        self.assertEqual(pw.matrix(0), [[sum(r[i] * r[j] for r in zero_filled) for j in range(4)] for i in range(4)])
        # covariance from pair-complete means: indefinite pairwise, a true covariance listwise
        pc, lc = pw.covariance(0).cov, lw.covariance(0).cov
        self.assertEqual(pc, [[F(17, 30), F(2, 3), F(-2, 3)], [F(2, 3), F(17, 30), F(2, 3)], [F(-2, 3), F(2, 3), F(13, 6)]])
        self.assertEqual(lin.inertia(pc).cls, "indefinite")
        self.assertEqual(lin.inertia(lc).negative, 0)
        # pair-complete marginals (SQL REGR_* rows) against explicit sums
        c, sx, sy, sxx, syy = pw.pair_marginals(0, "x", "y")
        pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        self.assertEqual((c, sx, sy, sxx, syy), (len(pairs), sum(a for a, _ in pairs), sum(b for _, b in pairs),
                                                 sum(a * a for a, _ in pairs), sum(b * b for _, b in pairs)))
        self.assertEqual(pw.pair_marginals(0, "y", "x"), (c, sy, sx, syy, sxx))
        with self.assertRaises(ValueError):
            lw.pair_marginals(0, "x", "y")

    def test_lin001_dimension_and_output_budgets_are_enforced(self):
        rng = np.random.default_rng(1004)
        n = 300
        cols = {f"x{i}": rng.integers(-50, 50, n) for i in range(18)}
        groups = rng.integers(0, 5, n)
        scope = mbk.Scope.from_arrays(cols, groups=groups, ngroups=5)
        feats = [f"x{i}" for i in range(16)]
        st = self.both(scope, feats, "x16")                     # 16 features: the declared limit
        self.assertEqual((st.k, st.nslots), (18, 171))
        with self.assertRaises(UnsupportedError):
            lin.moments(scope, feats + ["x17"], "x16")
        E, S = 171, 171
        need = lin.need(scope, feats, "x16")
        self.assertEqual((need["tensor"], need["states"], need["output"], need["variables"]), (5 * E, 5 * S, 5, 17))
        for resource, value in (("tensor", 5 * E), ("states", 5 * S), ("output", 5), ("variables", 17),
                                ("bytes", need["bytes"])):
            for backend in ("cpu", "cuda"):
                lim = value if backend == "cpu" or resource != "bytes" else lin.need(scope, feats, "x16", backend="cuda")["bytes"]
                lin.moments(scope, feats, "x16", backend=backend, budget=Budget(**{resource: lim}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    lin.moments(scope, feats, "x16", backend=backend, budget=Budget(**{resource: lim - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), (resource, lim, lim - 1))
        # pairwise states keep the pair-complete marginals: more slots, budgeted the same way
        pn = lin.need(scope, feats[:3], "x16", policy="pairwise")
        self.assertEqual(pn["support"], 15 + 2 * 4 * 3)
        with self.assertRaises(ValueError):
            lin.moments(scope, ["x0", "x0"], "x1")              # distinct columns
        with self.assertRaises(ValueError):
            lin.moments(scope, ["x0"], "x0")

    def test_lin001_adopted_device_states_are_checked_for_consistency(self):
        import ctypes as C
        rng = np.random.default_rng(1006)
        n = 200
        scope = mbk.Scope.from_arrays({"x": rng.integers(-100, 100, n), "y": rng.integers(-100, 100, n)},
                                      valid={"x": rng.random(n) < 0.8}, groups=rng.integers(0, 2, n), ngroups=2)
        for policy in ("listwise", "pairwise"):
            spec = lin._spec(scope, ["x"], "y", policy, True, "auto")
            S = 6 if policy == "listwise" else 10                 # pairwise: 4 pair-complete marginals of (x, y)
            rows, count, sums = np.zeros(2, np.uint64), np.zeros(2 * S, np.uint64), np.zeros(2 * S * 3, np.uint64)
            st = lin._States(2, S, 3, rows.ctypes.data, count.ctypes.data, sums.ctypes.data)
            ref = lin._Refusal()
            lin._gpu().run("tr_mgpu_lin_moments", C.byref(scope.view()), C.byref(spec), 1, C.byref(lin._budget(None)),
                           C.byref(st), C.byref(ref), refusal=ref)

            def adopt(r, c, s):
                t = lin._States(2, S, 3, r.ctypes.data, c.ctypes.data, s.ctypes.data)
                err, ref2 = C.c_int32(), lin._Refusal()
                h = lin._call("tr_lin_moments_from_states", scope._handle, C.byref(spec), 1, C.byref(t),
                              C.byref(lin._budget(None)), C.byref(ref2), C.byref(err))
                return lin._made(h, err, ref2, "adopt")
            ok = lin.Moments._wrap(adopt(rows, count, sums), scope, ["x"], "y", "cuda", "test")
            self.assertTrue(ok.equal(lin.moments(scope, ["x"], "y", policy=policy)))
            for tamper in ("count above rows", "total rows", "constant slot", "bound", "zero count with a sum"):
                r, c, s = rows.copy(), count.copy(), sums.copy()
                if tamper == "count above rows":
                    c[1] = r[0] + 1
                elif tamper == "total rows":
                    r[1] += 1
                elif tamper == "constant slot":
                    s[0] += 1
                elif tamper == "bound":
                    s[3 * 3 + 1] = 1                              # slot (1, 1) = x^2 far past count * 100^2
                else:
                    c[2] = 0
                with self.assertRaises(ValueError, msg=(policy, tamper)):
                    adopt(r, c, s)

    def test_lin001_merges_and_deletions_of_disjoint_selections_are_exact(self):
        rng = np.random.default_rng(1005)
        db, raw = dataset(rng, 900)
        t = db["f"]
        part = rng.random(t.n) < 0.4
        used = np.asarray(t.used, dtype=bool) if hasattr(t, "used") else np.ones(t.n, bool)
        whole = mbk.Scope.from_table(t, ["x", "d", "y"], group_by=["k"])
        a = mbk.Scope.from_table(t, ["x", "d", "y"], group_by=["k"], rows=part & used)
        b = mbk.Scope.from_table(t, ["x", "d", "y"], group_by=["k"], rows=~part & used)
        for policy in ("listwise", "pairwise"):
            sw, sa, sb = (self.both(s, ["x", "d"], "y", policy=policy) for s in (whole, a, b))
            self.assertTrue((sa + sb).equal(sw))
            self.assertTrue((sw - sa).equal(sb))
            self.assertEqual((sw - sa).source, "combined")
            with self.assertRaises(ValueError):                 # deleting more rows than there are
                sa - sw
        with self.assertRaises(DomainError):
            lin.moments(whole, ["x"], "y") + lin.moments(whole, ["d"], "y")
        # a table scope refuses to build once the table changed
        scope = mbk.Scope.from_table(t, ["x", "y"])
        lin.moments(scope, ["x"], "y")
        t.flip(t.idx("x"), 0)
        with self.assertRaises(StaleStateError):
            lin.moments(scope, ["x"], "y")

    # ================================================================== LIN-002
    def test_lin002_exact_rational_solves_match_fraction_elimination(self):
        rng = np.random.default_rng(2001)

        def rnd(m, n, rank=None):
            if rank is None:
                return [[F(int(rng.integers(-9, 10)), int(rng.integers(1, 5))) for _ in range(n)] for _ in range(m)]
            B = [[F(int(rng.integers(-5, 6))) for _ in range(rank)] for _ in range(m)]
            Cm = [[F(int(rng.integers(-5, 6)), int(rng.integers(1, 4))) for _ in range(n)] for _ in range(rank)]
            return [[sum(B[i][k] * Cm[k][j] for k in range(rank)) for j in range(n)] for i in range(m)]
        cases = []
        for _ in range(12):
            m, n = int(rng.integers(1, 7)), int(rng.integers(1, 7))
            A = rnd(m, n)
            cases.append((A, [F(int(rng.integers(-20, 21)), int(rng.integers(1, 6))) for _ in range(m)]))
        for _ in range(12):                                     # rank deficient: consistent and inconsistent
            m, n = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            r = int(rng.integers(1, min(m, n)))
            A = rnd(m, n, r)
            xs = [F(int(rng.integers(-4, 5))) for _ in range(n)]
            cases.append((A, matvec(A, xs)))
            cases.append((A, [v + (i == 0) for i, v in enumerate(matvec(A, xs))]))
        cases.append(([[0, 0], [0, 0]], [0, 0]))
        cases.append(([[0, 0], [0, 0]], [0, 1]))
        cases.append(([[10**30, 1], [1, 10**-0]], [F(1, 3), F(10**25)]))
        kinds = set()
        cpu = [lin.solve(A, b) for A, b in cases]
        gpu = lin.solve_batch(cases, backend="cuda")
        for (A, b), s, g in zip(cases, cpu, gpu):
            kind, rank, piv, x, null = oracle_solve(A, b)
            kinds.add(kind)
            self.assertEqual((s.kind, s.rank, s.pivots), (kind, rank, piv), A)
            self.assertTrue(s.certified)
            if kind == "none":
                self.assertIsNone(s.x)
                y = s.certificate                                # y^T A = 0, y^T b = 1
                self.assertEqual([sum(y[i] * F(A[i][j]) for i in range(len(A))) for j in range(len(A[0]))], [0] * len(A[0]))
                self.assertEqual(sum(yi * F(bi) for yi, bi in zip(y, b)), 1)
            else:
                self.assertEqual((s.x, s.nullspace), (x, null))
                self.assertEqual(matvec(A, s.x), [F(v) for v in b])
                for v in s.nullspace:
                    self.assertEqual(matvec(A, v), [0] * len(A))
            self.assertTrue(s.same(g), A)
            self.assertEqual((s.method, g.method), ("bareiss", "modular"))
            self.assertIn(g.backend, ("cuda", "cuda-emulated"))
        self.assertEqual(kinds, {"unique", "family", "none"})

    def test_lin002_rank_deficient_and_indefinite_cases_are_explicit(self):
        rng = np.random.default_rng(2002)
        classes = set()
        mats = [[[2, 1], [1, 2]], [[1, 1], [1, 1]], [[0, 1], [1, 0]], [[-2, 1], [1, -2]], [[-1, -1], [-1, -1]],
                [[0, 0], [0, 0]], [[0, 0, 1], [0, 0, 1], [1, 1, 0]], [[0, 2, 0], [2, 0, 0], [0, 0, 0]],
                [[F(1, 3), F(1, 2)], [F(1, 2), F(3, 4)]]]
        for _ in range(25):
            n, r = int(rng.integers(1, 6)), int(rng.integers(0, 6))
            B = [[F(int(rng.integers(-4, 5)), int(rng.integers(1, 3))) for _ in range(n)] for _ in range(max(r, 1))]
            signs = [int(rng.choice([-1, 1, 1])) for _ in range(len(B))]
            mats.append([[sum(signs[k] * B[k][i] * B[k][j] for k in range(len(B))) if r else 0 for j in range(n)]
                         for i in range(n)])
        for A in mats:
            ine = lin.inertia(A)
            self.assertEqual((ine.positive, ine.negative, ine.zero), oracle_inertia(A), A)
            classes.add(ine.cls)
            self.assertTrue(ine.certified)
            if ine.negative:
                self.assertLess(quad(A, ine.negative_direction), 0)
            else:
                self.assertIsNone(ine.negative_direction)
            if ine.positive:
                self.assertGreater(quad(A, ine.positive_direction), 0)
            self.assertEqual(len(ine.kernel), ine.zero)
            for v in ine.kernel:
                self.assertEqual(matvec(A, v), [0] * len(A))
            # a singular system never divides by zero: the solve says family or none
            s = lin.solve(A, [1] * len(A))
            self.assertEqual(s.rank, len(A) - ine.zero)
            self.assertEqual(s.kind, "unique" if not ine.zero else s.kind)
            if ine.zero:
                self.assertIn(s.kind, ("family", "none"))
        self.assertEqual(classes, {"positive definite", "positive semidefinite", "negative definite",
                                   "negative semidefinite", "indefinite", "zero"})
        with self.assertRaises(ValueError):
            lin.inertia([[1, 2], [3, 4]])                       # not symmetric

    def test_lin002_fits_projections_and_residuals_match_fraction_least_squares(self):
        rng = np.random.default_rng(2003)
        db, raw = dataset(rng, 700, nulls=0.08)
        t = db["f"]
        mask = rng.random(t.n) < 0.85
        scope = mbk.Scope.from_table(t, ["x", "d", "w", "y"], group_by=["k"], rows=mask)
        st = self.both(scope, ["x", "d", "w"], "y")
        groups = [g for g in range(st.groups) if st.state(g)["present"] and st.matrix(g)[0][0] > 4]
        reqs = []
        for g in groups:
            key = scope.key_values(g)
            rows = [r for r in range(len(raw["k"])) if mask[r] and (raw["k"][r],) == key and
                    all(raw[c][r] is not None for c in ("x", "d", "w", "y"))]
            Xv = {"1": [F(1)] * len(rows), "x": [F(raw["x"][r]) for r in rows],
                  "d": [F(raw["d"][r], 100) for r in rows], "w": [F(raw["w"][r]) for r in rows]}
            yv = [F(raw["y"][r]) for r in rows]
            for features, ridge, intercept in ((None, 0, None), (["x", "w"], 0, None), (["d"], F(5, 2), True),
                                               (None, F(1, 7), False)):
                cols = (["1"] if intercept is not False else []) + (features or ["x", "d", "w"])
                lam = F(ridge)
                A = [[sum(p * q for p, q in zip(Xv[a], Xv[b])) + (lam if a == b and a != "1" else 0) for b in cols]
                     for a in cols]
                bb = [sum(p * q for p, q in zip(Xv[a], yv)) for a in cols]
                kind, rank, piv, beta, null = oracle_solve(A, bb)
                fit = st.fit(g, intercept=intercept, ridge=ridge, features=features)
                self.assertEqual((fit.kind, fit.design, fit.x, fit.A, fit.b), (kind, cols, beta, A, bb))
                res = [yy - sum(beta[i] * Xv[c][r] for i, c in enumerate(cols)) for r, yy in enumerate(yv)]
                rss = sum(e * e for e in res)
                ybar = sum(yv) / len(yv)
                tss = sum((yy - ybar) ** 2 for yy in yv) if "1" in cols else sum(yy * yy for yy in yv)
                self.assertEqual((fit.rss, fit.tss, fit.n, fit.yty), (rss, tss, len(rows), sum(yy * yy for yy in yv)))
                fitted = [sum(beta[i] * Xv[c][r] for i, c in enumerate(cols)) for r in range(len(yv))]
                self.assertEqual(fit.projection, sum(v * v for v in fitted))      # ||X b||^2
                self.assertEqual(fit.r2, 1 - rss / tss)
                self.assertEqual(fit.sigma2, rss / (len(rows) - rank) if not lam and len(rows) > rank else None)
                reqs.append(dict(group=g, intercept=intercept, ridge=ridge, features=features))
                sysd = st.system(g, intercept=intercept, ridge=ridge, features=features)
                self.assertEqual((sysd["A"], sysd["b"], sysd["design"]), (A, bb, cols))
        # the same fits by batched modular elimination on the device
        cpu = st.fits(reqs)
        gpu = st.fits(reqs, backend="cuda")
        for a, b in zip(cpu, gpu):
            self.assertTrue(a.same(b))
            self.assertEqual(b.method, "modular")
        # a rank-deficient design (a feature twice): a family of coefficients with one projection and one RSS
        n = 50
        x = rng.integers(-30, 30, n)
        y = 2 * x + rng.integers(-3, 4, n)
        sc = mbk.Scope.from_arrays({"x": x, "x2": 3 * x, "y": y})
        s2 = self.both(sc, ["x", "x2"], "y")
        fam = s2.fit(0)
        self.assertEqual((fam.kind, fam.rank, fam.nullity), ("family", 2, 1))
        one = s2.fit(0, features=["x"])
        self.assertEqual(fam.rss, one.rss)
        self.assertEqual(matvec(fam.A, fam.nullspace[0]), [0, 0, 0])
        self.assertTrue(st.fits([dict(group=groups[0])], backend="cuda")[0].same(st.fit(groups[0])))
        with self.assertRaises(ValueError):
            lin.moments(sc, ["x"]).fit(0)                        # no target
        pw = lin.moments(sc, ["x", "x2"], "y", policy="pairwise")
        self.assertIsNone(pw.fit(0).rss)                         # statistics need a common-listwise state

    def test_lin002_covariance_and_squared_correlation_are_exact_rationals(self):
        rng = np.random.default_rng(2004)
        db, raw = dataset(rng, 800, nulls=0.15)
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["x", "d", "y"], group_by=["k"])
        for policy in ("listwise", "pairwise"):
            st = self.both(scope, ["x", "d"], "y", policy=policy)
            for g in range(st.groups):
                if not st.state(g)["present"]:
                    continue
                key = scope.key_values(g)
                idx = [r for r in range(len(raw["k"])) if (raw["k"][r],) == key]
                vals = {"x": [None if raw["x"][r] is None else F(raw["x"][r]) for r in idx],
                        "d": [None if raw["d"][r] is None else F(raw["d"][r], 100) for r in idx],
                        "y": [None if raw["y"][r] is None else F(raw["y"][r]) for r in idx]}
                names = ["x", "d", "y"]
                for ddof in (0, 1):
                    cv = st.covariance(g, ddof)
                    self.assertEqual(cv.columns, names)
                    for i, a in enumerate(names):
                        for j, b in enumerate(names):
                            if policy == "listwise":
                                rows = [r for r in range(len(idx)) if all(vals[c][r] is not None for c in names)]
                            else:
                                rows = [r for r in range(len(idx)) if vals[a][r] is not None and vals[b][r] is not None]
                            xa, xb = [vals[a][r] for r in rows], [vals[b][r] for r in rows]
                            nn = len(rows)
                            self.assertEqual(cv.count[i][j], nn)
                            if nn <= ddof:
                                self.assertIsNone(cv.cov[i][j])
                                continue
                            ma, mb = sum(xa) / nn, sum(xb) / nn
                            sab = sum((p - ma) * (q - mb) for p, q in zip(xa, xb))
                            self.assertEqual(cv.cov[i][j], sab / (nn - ddof), (policy, a, b))
                            saa = sum((p - ma) ** 2 for p in xa)
                            sbb = sum((q - mb) ** 2 for q in xb)
                            self.assertEqual(cv.r2[i][j], sab * sab / (saa * sbb) if saa and sbb else None)
                            self.assertEqual(cv.sign[i][j], (sab > 0) - (sab < 0))
        # against DuckDB's CORR / COVAR_SAMP (floating) on one group, pairwise rows
        st = lin.moments(scope, ["x", "d"], "y", policy="pairwise")
        g = [g for g in range(st.groups) if scope.key_values(g) == (2,)][0]
        cs, cr = db.con.execute('SELECT COVAR_SAMP(f."x", f."y"), CORR(f."x", f."y") FROM f WHERE f."k" = 2').fetchone()
        cv = st.covariance(g, 1)
        self.assertAlmostEqual(float(cv.cov[0][2]) / cs, 1.0, places=9)
        self.assertAlmostEqual(cv.sign[0][2] * math.sqrt(cv.r2[0][2]) / cr, 1.0, places=9)
        # a bank-built pairwise state has no pair-complete marginals: covariance is not defined from it
        bank = mbk.build(scope, mbk.monomials(["x", "d", "y"], 2), "sql")
        with self.assertRaises(DomainError):
            lin.moments_from_bank(bank, ["x", "d"], "y").covariance(0)

    def test_lin002_numerical_pca_and_projection_residuals_meet_declared_tolerances(self):
        rng = np.random.default_rng(2005)
        for n in (1, 2, 3, 5, 8, 12):
            # exact eigenvalues from a rational Householder reflection: A = H D H, H = I - 2 v v^T / v^T v
            v = [F(int(rng.integers(-5, 6))) for _ in range(n)]
            v[0] += 7
            vv = sum(c * c for c in v)
            H = [[(1 if i == j else 0) - 2 * v[i] * v[j] / vv for j in range(n)] for i in range(n)]
            D = [F(int(rng.integers(-40, 41)), int(rng.integers(1, 4))) for _ in range(n)]
            A = [[sum(H[i][k] * D[k] * H[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            e = lin.eigen(A)
            self.assertTrue(e.approximate and e.certified and e.converged)
            norm = math.sqrt(sum(float(a) ** 2 for row in A for a in row)) or 1.0
            tol = 64 * n * EPS * norm                            # the declared tolerance (docs/math/lin.md)
            self.assertLessEqual(e.orthogonality, 64 * n * EPS)
            for i in range(n):
                self.assertLessEqual(e.residual[i], tol)
                self.assertLessEqual(e.bound[i], tol)
                # an exact eigenvalue lies within the bound (compared exactly with Fractions)
                lam = F(e.values[i])
                self.assertTrue(any(abs(F(dk) - lam) <= F(e.bound[i]) for dk in D), (n, i))
            self.assertTrue(np.allclose(np.sort(e.values), np.sort(np.linalg.eigvalsh(np.array(A, float))),
                                        atol=tol + 1e-12))
            self.assertTrue(np.allclose(e.vectors @ e.vectors.T, np.eye(n), atol=64 * n * EPS))
        # an enclosure of an inexact matrix: the bounds hold for every matrix inside it
        M = rng.normal(size=(6, 6))
        M = (M + M.T) / 2
        lo, hi = np.nextafter(M, -np.inf), np.nextafter(M, np.inf)
        e = lin.eigen(lo=lo, hi=hi)
        self.assertGreater(e.width, 0)
        w = np.linalg.eigvalsh(M)
        for i in range(6):
            self.assertLessEqual(min(abs(w - e.values[i])), e.bound[i] + 1e-15)
        # PCA of a moment state: the covariance's eigensystem against NumPy on the exact covariance
        n = 400
        x1 = rng.integers(-100, 100, n)
        x2 = 2 * x1 + rng.integers(-20, 20, n)
        x3 = rng.integers(-5, 5, n)
        scope = mbk.Scope.from_arrays({"a": x1, "b": x2, "c": x3})
        st = self.both(scope, ["a", "b", "c"])
        for kind in ("covariance", "correlation"):
            pca = st.pca(0, kind)
            cov = st.covariance(0).cov
            M = np.array([[float(v) for v in row] for row in cov])
            if kind == "correlation":
                s = np.sqrt(np.diag(M))
                M = M / np.outer(s, s)
            w = np.sort(np.linalg.eigvalsh(M))[::-1]
            norm = np.linalg.norm(M)
            self.assertTrue(np.allclose(pca.values, w, atol=64 * 3 * EPS * norm + 1e-9))
            self.assertTrue(np.all(pca.residual <= 64 * 3 * EPS * norm + 4 * pca.width * norm))
            self.assertEqual(pca.matrix, cov)
        with self.assertRaises(NotConvergedError):              # a sweep budget too small to certify
            lin.eigen([[2, 1, 1], [1, 3, 1], [1, 1, 4]], sweeps=1)

    def test_lin002_no_stability_claim_follows_from_exact_input_moments(self):
        # exact moments whose covariance has two nearly equal eigenvalues: the eigenvalues are certified within
        # their bounds, the eigenvectors of the cluster are not claimed (they are ill-conditioned)
        n = 2000
        rng = np.random.default_rng(2006)
        a = rng.integers(-10**6, 10**6, n)
        b = rng.integers(-10**6, 10**6, n)
        c = a + b + rng.integers(-1, 2, n)                       # nearly dependent: a tiny third eigenvalue
        scope = mbk.Scope.from_arrays({"a": a, "b": b, "c": c})
        st = self.both(scope, ["a", "b", "c"])
        e = st.pca(0)
        self.assertTrue(e.approximate)
        self.assertIn("no claim on eigenvector accuracy", e.contract)
        cov = [[float(v) for v in row] for row in st.covariance(0).cov]
        norm = np.linalg.norm(np.array(cov))
        w = np.sort(np.linalg.eigvalsh(np.array(cov)))[::-1]
        for i in range(3):
            self.assertLessEqual(abs(w[i] - e.values[i]), e.bound[i] + 64 * 3 * EPS * norm)
        self.assertLess(e.values[2], 1e-6 * e.values[0])         # nearly singular, still within its bound
        self.assertFalse(hasattr(e, "exact"))
        ine = lin.inertia(st.covariance(0).cov)                  # the exact answer to "singular?" is exact
        self.assertEqual(ine.cls, "positive definite")

    def test_lin002_modular_residues_equal_the_cpu_reference_and_unlucky_primes_are_rejected(self):
        rng = np.random.default_rng(2007)
        systems = [([[int(rng.integers(-10**12, 10**12)) for _ in range(5)] for _ in range(5)],
                    [int(rng.integers(-10**6, 10**6)) for _ in range(5)]) for _ in range(6)]
        rows = np.array([5] * 6, np.int32)
        qa, ka = lin._pack([v for A, _ in systems for r in A for v in r])
        qb, kb = lin._pack([v for _, b in systems for v in b])
        import ctypes as C
        err, ref = C.c_int32(), lin._Refusal()
        h = lin._call("tr_lin_batch_systems", 6, rows.ctypes.data, rows.ctypes.data, C.byref(qa), C.byref(qb),
                      C.byref(lin._budget(None)), C.byref(ref), C.byref(err))
        batch = lin.Batch(lin._made(h, err, ref, "batch"))
        k = batch.primes_needed
        cpu, fields = batch.residues(k, "cpu")
        gpu, _ = batch.residues(k, "cuda")
        self.assertTrue(np.array_equal(cpu, gpu))
        results, _ = batch.finish(k, gpu, fields)
        for (A, b), r in zip(systems, results):
            self.assertTrue(lin.Solution(r).same(lin.solve(A, b)))
        # a determinant divisible by the first three primes: they disagree, a rerun with more primes is exact
        ps, _ = lin.primes(3)
        P3 = ps[0] * ps[1] * ps[2]
        A, b = [[P3, 0], [0, 1]], [1, 1]
        s = lin.solve(A, b, backend="cuda")
        self.assertEqual((s.kind, s.x), ("unique", [F(1, P3), F(1)]))
        self.assertGreater(s.primes, 0)
        self.assertTrue(s.same(lin.solve(A, b)))

    def test_lin002_budgets_at_the_limit_and_one_past(self):
        A = [[1, 2, 3], [4, 5, 6]]
        lin.solve(A, [1, 2], budget=Budget(tensor=8))            # m x (n + 1) = 8
        with self.assertRaises(AlgebraBudgetError) as ctx:
            lin.solve(A, [1, 2], budget=Budget(tensor=7))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("tensor", 8))
        lin.solve(A, [1, 2], budget=Budget(variables=3))
        with self.assertRaises(AlgebraBudgetError):
            lin.solve(A, [1, 2], budget=Budget(variables=2))
        with self.assertRaises(UnsupportedError):
            lin.solve([[1] * 64 for _ in range(2)], [1, 1])      # 64 unknowns
        with self.assertRaises(UnsupportedError):
            lin.solve([[1] for _ in range(65)], [1] * 65)        # 65 equations
        lin.eigen([[2, 1], [1, 2]], sweeps=8, budget=Budget(iterations=8))
        with self.assertRaises(AlgebraBudgetError):
            lin.eigen([[2, 1], [1, 2]], sweeps=8, budget=Budget(iterations=7))
        rng = np.random.default_rng(2008)
        scope = mbk.Scope.from_arrays({"x": rng.integers(0, 9, 50), "y": rng.integers(0, 9, 50)})
        st = lin.moments(scope, ["x"], "y")
        st.fit(0, budget=Budget(tensor=3 * 4))                   # the k x (k + 1) normal system bound
        with self.assertRaises(AlgebraBudgetError):
            st.fit(0, budget=Budget(tensor=3 * 4 - 1))
        # the device copies of a modular batch, at the limit and one past
        import ctypes as C
        rows = np.array([2], np.int32)
        unk = np.array([3], np.int32)
        qa, ka = lin._pack([v for r in A for v in r])
        qb, kb = lin._pack([1, 2])
        err, ref = C.c_int32(), lin._Refusal()
        batch = lin.Batch(lin._made(lin._call("tr_lin_batch_systems", 1, rows.ctypes.data, unk.ctypes.data, C.byref(qa),
                                              C.byref(qb), C.byref(lin._budget(None)), C.byref(ref), C.byref(err)),
                                    err, ref, "batch"))
        k = batch.primes_needed
        need = int(lin._call("tr_lin_batch_bytes", batch._h, k))
        batch.residues(k, "cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            batch.residues(k, "cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
        with self.assertRaises(ValueError):
            lin.solve(A, [1, 2, 3])
        with self.assertRaises(TypeError):
            lin.solve([[1.5]], [1])                              # floats are not exact input


def scope_groups(scope, n):
    """the group id of every row of an array scope (from its native view)"""
    import ctypes as C
    v = scope.view()
    return np.ctypeslib.as_array((C.c_uint32 * n).from_address(v.groups)).copy()


if __name__ == "__main__":
    unittest.main()
