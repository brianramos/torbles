# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEN-002 (convolution and bounded formal series) (torbles/math_conv.c, math_conv_gpu.cu, math_conv.py).

Oracles, independent of the implementation: enumerated Cartesian products of the rows (itertools.product over the
bags, Python ints), Python dict polynomials, DuckDB SQL (CROSS JOIN, key equijoins, per-row sums, NULL counts),
enumerated walks (itertools.product over the steps of each length) and FND-009 truncated series (math_carriers.Series:
products, the reciprocal). Every CUDA result (batched convolutions, per-key conditional products, walk tables;
emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) is compared word for word with its CPU reference.
Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_carriers as mcar, math_conv as mc, math_dist as md
from torbles.math_budgets import Budget
from torbles.math_domains import CarrierError, DomainError
from torbles.math_errors import NoSolutionError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

def cuda_label():
    """the backend label of a CUDA result: "cuda" on a device, "cuda-emulated" when the kernels run in the emulation"""
    from torbles import math_gpu
    return "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def bag(values):
    out = {}
    for v in values:
        if v is not None:
            out[v] = out.get(v, 0) + 1
    return out


def enum_sum(*bags_of_rows):
    """the distribution of the sum of one row from each list (rows with a None value are skipped)"""
    out = {}
    for combo in itertools.product(*bags_of_rows):
        if any(v is None for v in combo):
            continue
        s = sum(combo)
        out[s] = out.get(s, 0) + 1
    return out


def pmul(a, b):
    out = {}
    for u, x in a.items():
        for v, y in b.items():
            k = tuple(p + q for p, q in zip(u, v)) if isinstance(u, tuple) else u + v
            out[k] = out.get(k, 0) + x * y
    return {k: c for k, c in out.items() if c}


def arrays_scope(cols, groups=None, ngroups=None):
    vals = {n: [0 if v is None else v for v in vs] for n, vs in cols.items()}
    valid = {n: [v is not None for v in vs] for n, vs in cols.items()}
    return mbk.Scope.from_arrays(vals, valid=valid, groups=groups, ngroups=ngroups)


def frame(cols, k=None):
    data = {n: pd.Series([v for v in vs], dtype="Int64") for n, vs in cols.items()}
    if k is not None:
        data["k"] = pd.Series(k, dtype="Int64")
    return pd.DataFrame(data)


def sqldict(rows):
    return {int(a): int(b) for a, b in rows if a is not None}


class Convolution(unittest.TestCase):
    # ------------------------------------------------------------------ acceptance 1: enumerated Cartesian products
    def test_gen002_convolution_coefficients_match_enumerated_cartesian_products(self):
        rng = np.random.default_rng(2002)
        xs = [int(v) for v in rng.integers(-4, 6, 9)] + [None]
        ys = [int(v) for v in rng.integers(0, 4, 7)]
        zs = [int(v) for v in rng.integers(-2, 3, 5)] + [None, 1]
        # three different array scopes: different sources, so no declaration is needed
        X = mc.factor(md.histogram(arrays_scope({"x": xs}), ["x"]))
        Y = mc.factor(md.histogram(arrays_scope({"y": ys}), ["y"]))
        Z = mc.factor(md.histogram(arrays_scope({"z": zs}), ["z"]))
        self.assertEqual(mc.product([X, Y]).terms(), enum_sum(xs, ys))
        self.assertEqual(mc.product([X, Y, Z]).terms(), enum_sum(xs, ys, zs))
        r = mc.product([X, Y, Z])
        self.assertEqual((r.info["tuples"], r.info["nulls"], r.total()),
                         (len(xs) * len(ys) * len(zs), len(xs) * len(ys) * len(zs) - 9 * 7 * 6, 9 * 7 * 6))
        self.assertFalse(r.info["declared"])                          # nothing linked: nothing assumed
        # outer: the joint of (X, Y) as a two-variable state, against the enumerated pairs
        pairs = {}
        for a, b in itertools.product(xs, ys):
            if a is not None:
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
        o = mc.product([X, Y], combine="outer")
        self.assertEqual((o.nvars, o.terms()), (2, pairs))
        # multiplicities (sums of k draws) need a declaration for data factors; a free factor (explicit terms) not
        hx = md.histogram(arrays_scope({"x": xs}), ["x"])
        X2 = mc.factor(hx)
        d2 = mc.independent(hx.scope)
        self.assertEqual(mc.product([X2, Y], mult=[2, 1], declaration=d2).terms(), enum_sum(xs, xs, ys))
        self.assertEqual(mc.power(X2, 3, declaration=d2).terms(), enum_sum(xs, xs, xs))
        self.assertTrue(mc.power(X2, 3, declaration=d2).info["declared"])
        coin = mc.from_terms({0: 1, 1: 1})
        self.assertEqual(mc.power(coin, 6).terms(), {k: c for k, c in enumerate([1, 6, 15, 20, 15, 6, 1])})
        self.assertEqual(mc.power(coin, 0).terms(), {0: 1})
        # truncated: only the exponents e < order of P (G = z^shift P), as enumerated
        shift = X2.gen.shift[0] * 2 + Y.gen.shift[0]
        full = enum_sum(xs, xs, ys)
        for order in (1, 4, 7, 50):
            t = mc.product([X2, Y], mult=[2, 1], order=order, declaration=d2)
            self.assertEqual(t.terms(), {v: c for v, c in full.items() if v - shift < order}, order)
        # weights: coefficients are products of the weight sums (enumerated over weighted rows)
        wx, ww = [1, 1, 3, -2], [5, -1, 2, 7]
        W = mc.factor(md.histogram(mbk.Scope.from_arrays({"x": wx, "w": ww}), ["x"], weight="w"), mass="weight")
        V = mc.factor(md.histogram(mbk.Scope.from_arrays({"y": [0, 2, 2], "v": [3, -4, 1]}), ["y"], weight="v"),
                      mass="weight")
        want = {}
        for (a, wa), (b, wb) in itertools.product(zip(wx, ww), zip([0, 2, 2], [3, -4, 1])):
            want[a + b] = want.get(a + b, 0) + wa * wb
        self.assertEqual(mc.product([W, V]).terms(), {k: c for k, c in want.items() if c})
        self.assertIsNone(mc.product([W, V]).info["tuples"])          # weights are not counts
        # two-variable joint states: (x1 + x2, y1 + y2) over the pairs of rows of two tables
        r1 = [(1, 2), (1, 2), (0, 5), (3, None)]
        r2 = [(2, 1), (4, 0)]
        J1 = mc.factor(md.histogram(arrays_scope({"a": [a for a, _ in r1], "b": [b for _, b in r1]}), ["a", "b"]))
        J2 = mc.factor(md.histogram(arrays_scope({"a": [a for a, _ in r2], "b": [b for _, b in r2]}), ["a", "b"]))
        want2 = {}
        for (a1, b1), (a2, b2) in itertools.product(r1, r2):
            if b1 is not None:
                want2[(a1 + a2, b1 + b2)] = want2.get((a1 + a2, b1 + b2), 0) + 1
        self.assertEqual(mc.product([J1, J2]).terms(), want2)
        # big coefficients stay exact (GEN-001's big integers)
        big = mc.power(coin, 200)
        self.assertEqual(big.coefficient(100), 90548514656103281165404177077484163874504589675413336841320)
        self.assertEqual(big.total(), 2 ** 200)
        # a declared product of one keyed table scope equals DuckDB's CROSS JOIN
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, k BIGINT)",
                              {"t": pd.DataFrame({"x": [1, 2, 2, 5, 3, 1], "y": [0, 3, 1, 1, 2, 4],
                                                  "k": [0, 0, 1, 1, 2, 2]})})
        s = mbk.Scope.from_table(db["t"], ["x", "y"], group_by=["k"])
        Xt, Yt = mc.factor(md.histogram(s, ["x"])), mc.factor(md.histogram(s, ["y"]))
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"x": [1, 2, 2, 5, 3, 1], "y": [0, 3, 1, 1, 2, 4], "k": [0, 0, 1, 1, 2, 2]}))
        cross = sqldict(con.execute("SELECT a.x + b.y, COUNT(*) FROM t a CROSS JOIN t b GROUP BY 1").fetchall())
        self.assertEqual(mc.product([Xt, Yt], declaration=mc.independent(s)).terms(), cross)
        con.close()

    def test_gen002_cuda_batched_and_conditional_convolutions_equal_cpu(self):
        rng = np.random.default_rng(2003)
        top = (1 << 63) - 1
        for trial in range(5):
            n = int(rng.integers(1, 7))
            A = [[int(v) for v in rng.integers(-50, 50, int(rng.integers(0, 9)))] for _ in range(n)]
            B = [[int(v) for v in rng.integers(-50, 50, int(rng.integers(0, 9)))] for _ in range(n)]
            if trial == 4:                                            # extreme int64: 192-bit sums past int128
                A = [[top, top, -top - 1, top] * 4]
                B = [[top, top, top, -top - 1] * 4]
            for order in (None, 3):
                cpu, gpu = mc.convolve(A, B, order=order, raw=True), mc.convolve(A, B, order=order, backend="cuda",
                                                                                    raw=True)
                self.assertTrue(np.array_equal(cpu, gpu), (trial, order))
                got = mc.convolve(A, B, order=order)
                for a, b, g in zip(A, B, got):
                    want = [sum(a[i] * b[k - i] for i in range(len(a)) if 0 <= k - i < len(b))
                            for k in range(len(a) + len(b) - 1)] if a and b else []
                    self.assertEqual(g, want[:order] if order else want)
        self.assertGreater(max(abs(v) for v in mc.convolve([[top] * 16], [[top] * 16])[0]), 1 << 127)
        self.assertEqual((mc.convolve([[1]], [[2]]).backend, mc.convolve([[1]], [[2]], backend="cuda").backend),
                         ("cpu", cuda_label()))
        # per-key conditional products in parallel over keys: the raw words and the adopted state equal the CPU's
        for seed in (1, 2):
            r = np.random.default_rng(seed)
            K, n = 40, 3000
            k = r.integers(0, K, n)
            x = r.integers(-20, 60, n)
            y = r.integers(0, 15, n)
            s = mbk.Scope.from_arrays({"x": x, "y": y}, groups=k, ngroups=K)
            fa, fb = mc.family(md.histogram(s, ["x"])), mc.family(md.histogram(s, ["y"]))
            cpu, gpu = mc.conditional(fa, fb), mc.conditional(fa, fb, backend="cuda")
            self.assertTrue(cpu.equal(gpu))
            self.assertEqual(gpu.backend, cuda_label())
            va, vb = fa._view(), fb._view()
            ooff = np.zeros(K + 1, dtype=np.uint64)
            total, pairs = mc.C.c_uint64(), mc.C.c_uint64()
            mc._call("tr_conv_conditional_plan", fa._open(), fb._open(), ooff.ctypes.data, mc.C.byref(total),
                     mc.C.byref(pairs))
            ref = np.zeros(3 * int(total.value), dtype=np.int64)
            bt = mc._Batch(npairs=K, order=0, total=total.value, pairs=pairs.value, aoff=va.offset, boff=vb.offset,
                           ooff=ooff.ctypes.data, a=va.coef, b=vb.coef)
            self.assertEqual(mc._call("tr_conv_batch_run", mc.C.byref(bt), ref.ctypes.data, None, None), 0)
            self.assertTrue(np.array_equal(ref, gpu._raw_out))
            join = {}
            for g in range(K):
                bx, by = bag(x[k == g].tolist()), bag(y[k == g].tolist())
                for u, cu in bx.items():
                    for v, cv in by.items():
                        join[u + v] = join.get(u + v, 0) + cu * cv
            self.assertEqual(cpu.terms(), join)
        # families of dense histograms and of weight masses (the self-equijoin of a weighted family)
        r = np.random.default_rng(7)
        k, x, w = r.integers(0, 5, 200), r.integers(-3, 9, 200), r.integers(-4, 9, 200)
        s = mbk.Scope.from_arrays({"x": x, "w": w}, groups=k, ngroups=5)
        sparse, dense = mc.family(md.histogram(s, ["x"])), mc.family(md.histogram(s, ["x"], repr="dense"))
        self.assertTrue(mc.conditional(sparse, sparse).equal(mc.conditional(dense, dense)))
        fw = mc.family(md.histogram(s, ["x"], weight="w"), mass="weight")
        want = {}
        for g in range(5):
            idx = np.nonzero(k == g)[0]
            for i in idx:
                for j in idx:
                    want[int(x[i] + x[j])] = want.get(int(x[i] + x[j]), 0) + int(w[i]) * int(w[j])
        want = {v: c for v, c in want.items() if c}
        self.assertEqual(mc.conditional(fw, fw).terms(), want)
        self.assertEqual(mc.conditional(fw, fw, backend="cuda").terms(), want)
        self.assertIsNone(mc.conditional(fw, fw).info["tuples"])      # weights are not counts

    # ------------------------------------------------------------------ acceptance 2: keys and dependent events
    def test_gen002_key_conditioned_joins_and_dependent_events_cannot_use_unconditional_products(self):
        rng = np.random.default_rng(2004)
        n = 60
        k = rng.integers(0, 5, n)
        x = rng.integers(0, 6, n)
        y = x + rng.integers(0, 2, n)                                  # y depends on x: dependent events
        xs = [None if i % 11 == 0 else int(v) for i, v in enumerate(x)]
        ys = [None if i % 13 == 3 else int(v) for i, v in enumerate(y)]
        s = arrays_scope({"x": xs, "y": ys}, groups=k, ngroups=5)
        hx, hy, hxy = md.histogram(s, ["x"]), md.histogram(s, ["y"]), md.histogram(s, ["x", "y"])
        X, Y = mc.factor(hx), mc.factor(hy)
        con = duckdb.connect()
        con.register("t", frame({"x": xs, "y": ys}, k=[int(v) for v in k]))
        # the unconditional product of two marginals of one keyed scope is refused, naming the missing declaration
        for args in (dict(factors=[X, Y]), dict(factors=[X, Y], combine="outer"), dict(factors=[X], mult=[2]),
                     dict(factors=[mc.factor(hx, 1), mc.factor(hy, 1)])):
            with self.assertRaises(DomainError) as ctx:
                mc.product(**args)
            e = ctx.exception
            self.assertEqual((e.reason, e.keyed), ("needs_declaration", True))
            self.assertIn("independence declaration", str(e))
            self.assertIsNotNone(mc.check(args["factors"], mult=args.get("mult")))
        self.assertEqual(ctx.exception.factors, (0, 1))
        with self.assertRaises(DomainError) as ctx:
            mc.power(X, 2)
        self.assertEqual(ctx.exception.factors, (0, 0))               # two copies of one data factor
        # a declaration over another scope does not cover this source
        other = arrays_scope({"x": [1, 2]})
        with self.assertRaises(DomainError) as ctx:
            mc.product([X, Y], declaration=mc.independent(other))
        self.assertEqual(ctx.exception.reason, "not_covered")
        # the three exact answers, all different here: independent draws (declared), the key-equijoin (conditional)
        # and the per-row sum (the joint state)
        d = mc.independent(s)
        self.assertTrue(d.covers(s) and not d.covers(other))
        ind = mc.product([X, Y], declaration=d)
        cross = sqldict(con.execute("SELECT a.x + b.y, COUNT(*) FROM t a CROSS JOIN t b WHERE a.x IS NOT NULL AND "
                                    "b.y IS NOT NULL GROUP BY 1").fetchall())
        self.assertEqual(ind.terms(), cross)
        self.assertTrue(ind.info["declared"])
        cond = mc.conditional(mc.family(hx), mc.family(hy))
        join = sqldict(con.execute("SELECT a.x + b.y, COUNT(*) FROM t a JOIN t b ON a.k = b.k WHERE a.x IS NOT NULL "
                                   "AND b.y IS NOT NULL GROUP BY 1").fetchall())
        self.assertEqual(cond.terms(), join)
        self.assertEqual(mc.conditional(mc.family(hx), mc.family(hy), backend="cuda").terms(), join)
        allpairs, nullpairs = con.execute("SELECT COUNT(*), COUNT(*) FILTER (WHERE a.x IS NULL OR b.y IS NULL) "
                                          "FROM t a JOIN t b ON a.k = b.k").fetchone()
        self.assertEqual((cond.info["tuples"], cond.info["nulls"]), (allpairs, nullpairs))   # NULLs counted apart
        joint = mc.joint_sum(mc.factor(hxy))
        rowwise = sqldict(con.execute("SELECT x + y, COUNT(*) FROM t WHERE x IS NOT NULL AND y IS NOT NULL "
                                      "GROUP BY 1").fetchall())
        self.assertEqual(joint.terms(), rowwise)
        self.assertEqual(joint.info["mode"], "joint")
        self.assertEqual(len({str(sorted(t.items())) for t in (cross, join, rowwise)}), 3)
        # the conditional OUTER form: the joint (a.x, b.y) over the equijoin
        co = mc.conditional(mc.family(hx), mc.family(hy), combine="outer")
        pj = {(int(a), int(b)): int(c) for a, b, c in con.execute(
            "SELECT a.x, b.y, COUNT(*) FROM t a JOIN t b ON a.k = b.k WHERE a.x IS NOT NULL AND b.y IS NOT NULL "
            "GROUP BY 1, 2").fetchall()}
        self.assertEqual(co.terms(), pj)
        # marginals of the joint state are linked to each other too
        mx, my = mc.marginal(mc.factor(hxy), 0), mc.marginal(mc.factor(hxy), 1)
        with self.assertRaises(DomainError):
            mc.product([mx, my])
        self.assertEqual(mc.product([mx, my], declaration=d).total(), mx.total() * my.total())
        # two tables keyed by k: the conditional form is their join; their marginals (different sources) multiply
        # without a declaration into the CROSS JOIN, which is not the join
        ka, kb = rng.integers(0, 4, 30), rng.integers(0, 4, 25)
        ax, by = rng.integers(0, 9, 30), rng.integers(-3, 3, 25)
        A = mbk.Scope.from_arrays({"x": ax}, groups=ka, ngroups=4)
        Bs = mbk.Scope.from_arrays({"y": by}, groups=kb, ngroups=4)
        con.register("A", pd.DataFrame({"x": ax, "k": ka}))
        con.register("B", pd.DataFrame({"y": by, "k": kb}))
        fa, fb = mc.family(md.histogram(A, ["x"])), mc.family(md.histogram(Bs, ["y"]))
        jab = sqldict(con.execute("SELECT a.x + b.y, COUNT(*) FROM A a JOIN B b USING (k) GROUP BY 1").fetchall())
        self.assertEqual(mc.conditional(fa, fb).terms(), jab)
        self.assertEqual(mc.conditional(fa, fb, backend="cuda").terms(), jab)
        cab = sqldict(con.execute("SELECT a.x + b.y, COUNT(*) FROM A a CROSS JOIN B b GROUP BY 1").fetchall())
        self.assertEqual(mc.product([fa.marginal(), fb.marginal()]).terms(), cab)
        self.assertNotEqual(cab, jab)
        # key domains: another number of keys, another grouping of the same source
        with self.assertRaises(DomainError) as ctx:
            mc.conditional(fa, mc.family(md.histogram(mbk.Scope.from_arrays({"y": by}, groups=kb, ngroups=5), ["y"])))
        self.assertEqual(ctx.exception.reason, "keys_differ")
        regroup = arrays_scope({"x": xs, "y": ys}, groups=(k + 1) % 5, ngroups=5)
        with self.assertRaises(DomainError) as ctx:
            mc.conditional(mc.family(hx), mc.family(md.histogram(regroup, ["y"])))
        self.assertEqual(ctx.exception.reason, "grouping_differs")
        with self.assertRaises(DomainError):
            mc.conditional(mc.family(hx), mc.family(md.histogram(regroup, ["y"])), backend="cuda")
        # one engine table: every scope over it (other columns, another selection) is the same source
        frame_t = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 2, 5, 1], "k": [0, 1, 0, 1]})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, k BIGINT)", {"t": frame_t})
        s1 = mbk.Scope.from_table(db["t"], ["x"], group_by=["k"])
        s2 = mbk.Scope.from_table(db["t"], ["y"], rows=np.array([True, True, False, True]))
        F1, F2 = mc.factor(md.histogram(s1, ["x"])), mc.factor(md.histogram(s2, ["y"]))
        with self.assertRaises(DomainError):
            mc.product([F1, F2])
        self.assertEqual(mc.product([F1, F2], declaration=mc.independent(s2)).terms(),
                         enum_sum([1, 2, 3, 4], [2, 2, 1]))
        self.assertTrue(mc.independent(s1).covers(s2))
        # free factors and products: the result keeps its sources, so it stays linked
        noise = mc.from_terms({-1: 1, 1: 1})
        xn = mc.product([X, noise])
        self.assertEqual((xn.info["sources"], xn.info["kind"]), (1, "derived"))
        with self.assertRaises(DomainError):
            mc.product([xn, Y])
        # series made from a data factor keep its source: products and walks need the declaration as well
        sx = mc.series(X, 8, bound=10 ** 9)
        with self.assertRaises(DomainError):
            sx * sx
        with self.assertRaises(DomainError):
            sx.walks(3)
        self.assertEqual(sx.mul(sx, declaration=d).coefficients[:3], mc.power(X, 2, declaration=d).series(3).coefficients)
        con.close()

    # ------------------------------------------------------------------ bounded series, walks, declared bounds
    def test_gen002_bounded_series_walks_and_declared_bounds(self):
        steps = {1: 1, 2: 1, 3: 2}                                  # two kinds of 3-steps
        s = mc.Series([0, 1, 1, 2], order=12, bound=10 ** 6)
        for L in (0, 1, 3, 6):
            w = s.walks(L)
            want_rows = [[0] * 12 for _ in range(L + 1)]
            for l in range(L + 1):
                for seq in itertools.product(*([[(v, c) for v, c in steps.items()]] * l)):
                    tot = sum(v for v, _ in seq)
                    if tot < 12:
                        mult = 1
                        for _, c in seq:
                            mult *= c
                        want_rows[l][tot] += mult
            self.assertEqual(w["rows"], want_rows, L)
            self.assertEqual(w["total"].coefficients, [sum(r[e] for r in want_rows) for e in range(12)])
            cw = s.walks(L, backend="cuda")
            self.assertEqual((cw["rows"], cw["total"].coefficients, cw["maxabs"]),
                             (w["rows"], w["total"].coefficients, w["maxabs"]))
        # any length: 1 / (1 - s) mod z^N (s has no constant term), against FND-009's exact reciprocal
        w = s.walks(None)
        recip = mcar.Series([1, -1, -1, -2], order=12).reciprocal()
        self.assertEqual(w["total"].coefficients, [int(c) for c in recip.coefficients])
        self.assertEqual(w["lengths"], 12)
        self.assertEqual(s.walks(10 ** 6, rows=False)["total"].coefficients, w["total"].coefficients)
        with self.assertRaises(NoSolutionError):
            mc.Series([1, 1], order=6).walks(None)                   # zero-length steps: infinitely many walks
        self.assertEqual(mc.Series([1, 1], order=4, bound=100).walks(3)["total"].coefficients, [4, 6, 4, 1])
        # declared bounds: every coefficient checked, at the bound and one below it
        top = s.walks(6)["maxabs"]
        self.assertEqual(s.walks(6, bound=top)["maxabs"], top)
        with self.assertRaises(OverflowError) as ctx:
            s.walks(6, bound=top - 1)
        self.assertIn("exceeds the declared bound", str(ctx.exception))
        with self.assertRaises(OverflowError):
            s.walks(6, bound=top - 1, backend="cuda")
        ref, dev = np.zeros(7 * 12, np.int64), np.zeros(7 * 12, np.int64)
        tr, td = np.zeros(12, np.int64), np.zeros(12, np.int64)
        ir, idv = mc._Walks(), mc._Walks()
        st = np.array(s.coefficients, np.int64)
        self.assertEqual(mc._call("tr_conv_walks_run", st.ctypes.data, 12, 7, top - 1, ref.ctypes.data, tr.ctypes.data,
                                  mc.C.byref(ir), None, None), -3)
        mc._gpu()
        self.assertEqual(mc._gpu().lib().tr_mgpu_conv_walks(st.ctypes.data, 12, 7, top - 1, dev.ctypes.data,
                                                            td.ctypes.data, mc.C.byref(idv), None, None), -3)
        self.assertEqual(ir.violation, idv.violation)                  # the same first violation (row, exponent)
        with self.assertRaises(OverflowError):
            mc.Series([5, 2], order=3, bound=4)                        # an input past its own bound
        a, b = mc.Series([1, 2, 3], order=3, bound=10), mc.Series([3, -1, 4], order=3, bound=10)
        self.assertEqual(a.mul(b, bound=11).coefficients, [3, 5, 11])   # c_2 = 4 - 2 + 9: at the bound
        with self.assertRaises(OverflowError) as ctx:
            a * b                                                      # the default bound 10: one past it
        self.assertIn("coefficient past the bound: 2", str(ctx.exception))
        self.assertEqual((a + b).coefficients, [4, 1, 7])
        self.assertEqual(a.pow(3, bound=10 ** 4).coefficients, [1, 6, 21])
        with self.assertRaises(CarrierError):
            a * mc.Series([1], order=4)                               # different orders do not combine
        # sums capped at an order: truncated powers of a factor equal FND-009's series products
        die = mc.from_terms({v: 1 for v in range(1, 7)})
        for order in (4, 16, 64):
            p = mc.power(die, 5, order=order)
            fs = die.series(order)
            acc = fs
            for _ in range(4):
                acc = acc * fs
            want = {e + 5: int(c) for e, c in enumerate(acc.coefficients) if c}
            self.assertEqual(p.terms(), want, order)
            bs = mc.series(die, order, bound=10 ** 6).pow(5)
            self.assertEqual({e + 5: c for e, c in enumerate(bs.coefficients) if c}, want)
            self.assertEqual(bs.info["shift"], 5)
            self.assertTrue(bs.to_factor().equal(p))

    # ------------------------------------------------------------------ budgets
    def test_gen002_budgets_support_growth_pairs_and_bytes_refused_at_the_limit(self):
        rng = np.random.default_rng(2005)
        x = rng.integers(0, 40, 500)
        hx = md.histogram(mbk.Scope.from_arrays({"x": x}), ["x"])
        h2 = md.histogram(mbk.Scope.from_arrays({"y": rng.integers(-5, 30, 400)}), ["y"])
        X, Y = mc.factor(hx), mc.factor(h2)
        d = mc.independent(hx.scope, h2.scope)
        # two factors: GEN-001's exact needs (pairs, support growth, bytes)
        need = mc.need([X, Y])
        for resource in ("tensor", "support", "bytes"):
            ok = mc.product([X, Y], budget=Budget(**{resource: need[resource]}))
            self.assertEqual(len(ok.terms()), need["terms"])
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.product([X, Y], budget=Budget(**{resource: need[resource] - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need[resource]))
        # a power: the whole chain before its first product (support growth k (span - 1) + 1)
        pn = mc.need([X], mult=[16])
        self.assertEqual(pn["terms"], 16 * 39 + 1)
        for resource in ("tensor", "support", "bytes"):
            mc.power(X, 16, declaration=d, budget=Budget(**{resource: pn[resource]}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.power(X, 16, declaration=d, budget=Budget(**{resource: pn[resource] - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, pn[resource]))
        # a chain of several factors: support growth and pairs checked before the first product
        cn = mc.need([X, Y, X], mult=[1, 2, 1])
        for resource in ("tensor", "support"):
            mc.product([X, Y, X], mult=[1, 2, 1], declaration=d, budget=Budget(**{resource: cn[resource]}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.product([X, Y, X], mult=[1, 2, 1], declaration=d, budget=Budget(**{resource: cn[resource] - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, cn[resource]))
        # the conditional form: keys (states), pairs, per-key outputs, bytes
        K = 8
        s = mbk.Scope.from_arrays({"x": x, "y": x % 7}, groups=x % K, ngroups=K)
        fa, fb = mc.family(md.histogram(s, ["x"])), mc.family(md.histogram(s, ["y"]))
        for resource in ("states", "tensor", "support", "bytes"):
            for backend in ("cpu", "cuda"):
                cn = mc.conditional_need(fa, fb, backend=backend)
                mc.conditional(fa, fb, backend=backend, budget=Budget(**{resource: cn[resource]}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mc.conditional(fa, fb, backend=backend, budget=Budget(**{resource: cn[resource] - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, cn[resource]))
        # a family: its densest key's span (support) and keys (states)
        span = fa.info["max_span"]
        mc.family(md.histogram(s, ["x"]), budget=Budget(support=span))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mc.family(md.histogram(s, ["x"]), budget=Budget(support=span - 1))
        self.assertEqual(ctx.exception.need, span)
        with self.assertRaises(AlgebraBudgetError):
            mc.family(md.histogram(mbk.Scope.from_arrays({"x": [0, 10 ** 12]}), ["x"]))   # a wide dense span
        # batches: output and pairs on both paths, the device copies' bytes
        A, B = [[1, 2, 3], [4]], [[1, 1], [2, 3, 5]]
        for backend in ("cpu", "cuda"):
            for resource, value in (("output", 7), ("tensor", 9)):
                mc.convolve(A, B, backend=backend, budget=Budget(**{resource: value}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mc.convolve(A, B, backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        dev_bytes = 8 * 9 + 24 * 3 + 24 * 7
        mc.convolve(A, B, backend="cuda", budget=Budget(bytes=dev_bytes))
        with self.assertRaises(AlgebraBudgetError):
            mc.convolve(A, B, backend="cuda", budget=Budget(bytes=dev_bytes - 1))
        # series: order (degree = order - 1 <= 64 by default), walk lengths (iterations)
        mc.Series([1], order=65)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mc.Series([1], order=66)
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("degree", 65))
        st = mc.Series([1, 1], order=8, bound=10 ** 9)
        wn = st.walks_need(20)
        self.assertEqual(wn["iterations"], 20)
        st.walks(20, budget=Budget(iterations=20))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            st.walks(20, budget=Budget(iterations=19))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", 20))
        with self.assertRaises(AlgebraBudgetError):
            st.walks(20, backend="cuda", budget=Budget(iterations=19))
        self.assertEqual(st.walks(20, rows=False)["total"].coefficients[0], 21)

    def test_gen002_invalid_inputs_and_opt_in_import(self):
        one, two = mc.from_terms({1: 1}), mc.from_terms({(1, 1): 1})
        with self.assertRaises(ValueError):
            mc.product([one, two])                                     # different numbers of variables
        with self.assertRaises(UnsupportedError):
            mc.product([two, two], combine="outer")                    # outer takes two one-variable factors
        with self.assertRaises(UnsupportedError):
            mc.product([two, two], order=3)
        with self.assertRaises(ValueError):
            mc.joint_sum(one)
        dec = mbk.Scope.from_arrays({"d": [125, 250]}, scales={"d": 2})
        with self.assertRaises(DomainError) as ctx:
            mc.product([mc.factor(md.histogram(dec, ["d"])), one])     # values of different scales do not add
        self.assertEqual(ctx.exception.reason, "scales_differ")
        self.assertEqual(mc.factor(md.histogram(dec, ["d"])).info["scales"], (2,))
        s3 = mbk.Scope.from_arrays({"a": [1], "b": [2], "c": [3]})
        with self.assertRaises(UnsupportedError):
            mc.factor(md.histogram(s3, ["a", "b", "c"]))
        with self.assertRaises(UnsupportedError):
            mc.family(md.histogram(s3, ["a", "b"]))
        with self.assertRaises(TypeError):
            mc.factor(md.histogram(s3, ["a"]).prefix(degree=1))
        with self.assertRaises(OverflowError):
            mc.convolve([[1 << 63]], [[1]])
        with self.assertRaises(TypeError):
            mc.product([one], declaration="independent")
        code = "import sys, torbles; print('torbles.math_conv' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
