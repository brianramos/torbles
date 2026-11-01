# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""TENSOR-001 (sparse, dense and factored moment and count tensors) and TENSOR-002 (costed contraction planning)
(torbles/math_tensor.c, math_tensor_gpu.cu, math_tensor.py).

Oracles, independent of the implementation: tensor entries enumerated here row by row with Python integers (every
multi-index of every contributing row), DuckDB GROUP BY over the typed source table (NULL a group of its own), NumPy
indexing / transposes / sums of those enumerated object arrays, numpy.einsum on object arrays (exact Python ints) as
the direct contraction oracle, and the documented byte and shape formulas recomputed here. Every build is also run
on the CUDA path (emulated here; TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared for equality with
the CPU state, and every CUDA contraction candidate against the oracle. Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_tensor as mt
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

I128 = (1 << 127) - 1


def obj(values):
    return pd.Series(values, dtype=object)


def arrays_scope(rng, n=60, ngroups=4, nulls=0.15, absent=2):
    """small integer columns with NULLs, a weight with NULLs and zeros, group `absent` without rows"""
    vals = {"x": rng.integers(-3, 4, n), "y": rng.integers(-2, 3, n), "z": rng.integers(0, 3, n),
            "c": rng.integers(0, 4, n), "w": rng.integers(-2, 4, n)}
    valid = {k: rng.random(n) >= nulls for k in ("x", "y", "c", "w")}
    groups = rng.integers(0, ngroups, n)
    groups[groups == absent] = 0
    rows = rng.random(n) < 0.85
    scope = mbk.Scope.from_arrays(vals, valid=valid, groups=groups, ngroups=ngroups, rows=rows)
    py = {k: [None if (k in valid and not valid[k][r]) else int(vals[k][r]) for r in range(n)] for k in vals}
    return scope, py, [int(g) for g in groups], [bool(r) for r in rows]


def moment_oracle(py, groups, rows, ngroups, axes, policy, weight):
    """per group: (array of exact sums, array of counts or None, rows, used) by enumerating every multi-index"""
    dims = [len(a) for a in axes]
    ref = {c for a in axes for c in a if c != "1"}
    out = []
    for g in range(ngroups):
        val, cnt = np.zeros(dims, dtype=object), np.zeros(dims, dtype=object)
        nrows = used = 0
        for r in range(len(groups)):
            if not rows[r] or groups[r] != g:
                continue
            nrows += 1
            w = 1 if weight is None else py[weight][r]
            if w is None:
                continue
            if policy == "listwise" and any(py[c][r] is None for c in ref):
                continue
            used += 1
            for idx in itertools.product(*[range(d) for d in dims]):
                f = [1 if axes[k][i] == "1" else py[axes[k][i]][r] for k, i in enumerate(idx)]
                if any(v is None for v in f):
                    continue
                p = w
                for v in f:
                    p *= v
                val[idx] += p
                cnt[idx] += 1
        out.append((val, cnt if policy == "sql" else None, nrows, used))
    return out


def count_oracle(py, groups, rows, ngroups, cols, weight, domain):
    """per group: the dense array of counts (or weight sums) over the (lo, span, null) domain of each axis"""
    dims = [span + int(null) for lo, span, null in domain]
    out = []
    for g in range(ngroups):
        a = np.zeros(dims, dtype=object)
        for r in range(len(groups)):
            if not rows[r] or groups[r] != g:
                continue
            w = 1 if weight is None else py[weight][r]
            if w is None:
                continue
            idx = tuple(span if py[c][r] is None else py[c][r] - lo for c, (lo, span, null) in zip(cols, domain))
            a[idx] += w
        out.append(a)
    return out


def scope_domain(py, rows, cols):
    out = []
    for c in cols:
        present = [py[c][r] for r in range(len(rows)) if rows[r] and py[c][r] is not None]
        null = any(rows[r] and py[c][r] is None for r in range(len(rows)))
        lo = min(present) if present else 0
        out.append((lo, (max(present) - lo + 1) if present else 0, null))
    return out


def sparse_of(a):
    """[(coords, value)] of the nonzero entries in increasing multi-index order"""
    return [(tuple(int(i) for i in idx), a[idx]) for idx in np.ndindex(a.shape) if a[idx] != 0]


def einsum(spec, *arrays):
    """an exact einsum over Python ints (object arrays): NumPy's from 1.25, which first contracts object arrays;
    before it, the same sum formed here over every index assignment"""
    arrays = [np.asarray(a, dtype=object) for a in arrays]
    if np.lib.NumpyVersion(np.__version__) >= "1.25.0":
        return np.einsum(spec, *arrays)
    ins, out = spec.split("->")
    ins = ins.split(",")
    size = {ch: n for lab, a in zip(ins, arrays) for ch, n in zip(lab, a.shape)}
    labels = sorted(size)
    res = np.zeros([size[ch] for ch in out], dtype=object) if out else 0
    for vals in itertools.product(*[range(size[ch]) for ch in labels]):
        at = dict(zip(labels, vals))
        term = 1
        for lab, a in zip(ins, arrays):
            term = term * a[tuple(at[ch] for ch in lab)]
        if out:
            res[tuple(at[ch] for ch in out)] += term
        else:
            res += term
    return res


class Base(unittest.TestCase):
    def assert_cuda_equal(self, cpu, scope, columns, **kw):
        builder = mt.moments if kw.pop("kind", "moment") == "moment" else mt.counts
        gpu = builder(scope, columns, backend="cuda", **kw)
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        self.assertTrue(cpu.equal(gpu) or cpu.same(gpu), (columns, kw))
        self.assertTrue(cpu.same(gpu))
        return gpu

    def assert_sorted(self, t):
        for g in range(t.groups):
            keys = [e[0] for e in t.entries(g)]
            self.assertEqual(keys, sorted(set(keys)))


# ====================================================================== TENSOR-001
class Tensors(Base):
    # ------------------------------------------------------------------ acceptance 1: enumerated entries
    def test_tensor001_moments_match_enumerated_entries(self):
        rng = np.random.default_rng(2001)
        scope, py, groups, rows = arrays_scope(rng)
        G = scope.groups
        cases = [(["x"], 0), (["x", "y"], 1), (["x", "y", "z"], 2), (["x", "y", "z"], 3), (["x", "y"], 4),
                 ([["1", "x"], ["x", "y", "z"]], None), ([["y"], ["1", "z", "x"], ["x", "x"]], None),
                 (["1", "x", "y"], 3)]
        for columns, order in cases:
            axes = [list(c) for c in columns] if order is None else [list(columns)] * order
            for policy in ("listwise", "sql"):
                for weight in (None, "w"):
                    want = moment_oracle(py, groups, rows, G, axes, policy, weight)
                    reprs = ["dense", "sparse"] + (["factored"] if policy == "listwise" else [])
                    for repr_ in reprs:
                        t = mt.moments(scope, columns, order, weight=weight, policy=policy, repr=repr_)
                        info = t.info
                        self.assertEqual((info["kind"], info["repr"], info["dims"]),
                                         ("moment", repr_, tuple(len(a) for a in axes)))
                        self.assertEqual(info["counts"], policy == "sql")
                        for g in range(G):
                            val, cnt, nrows, used = want[g]
                            self.assertEqual(t.group(g), dict(present=nrows > 0, rows=nrows, used=used), (g, repr_))
                            if policy == "sql":
                                v, c = t.array(g)
                                self.assertTrue((v == val).all() and (c == cnt).all(), (columns, weight, repr_, g))
                            else:
                                self.assertTrue((t.array(g) == val).all(), (columns, order, weight, repr_, g))
                            for idx in itertools.islice(np.ndindex(val.shape), 0, None, 7):
                                e = t.entry(g, idx)
                                self.assertEqual(e, (val[idx], cnt[idx]) if policy == "sql" else val[idx])
                        if repr_ == "sparse":
                            self.assert_sorted(t)
                        if repr_ != "factored":
                            self.assert_cuda_equal(t, scope, columns, order=order, weight=weight, policy=policy,
                                                   repr=repr_)
                        if repr_ == "factored" and order is not None and order >= 2:
                            self.assertTrue(info["symmetric"])
        # the sparse strategies (dense scratch, sort) give the same state
        for strategy in ("auto", "count", "sort"):
            for policy in ("listwise", "sql"):
                a = mt.moments(scope, ["x", "y", "1"], 3, policy=policy, repr="sparse", strategy=strategy)
                b = mt.moments(scope, ["x", "y", "1"], 3, policy=policy, repr="sparse")
                self.assertTrue(a.same(b) and a.equal(b))
                self.assertTrue(a.to_dense().equal(mt.moments(scope, ["x", "y", "1"], 3, policy=policy)))

    def test_tensor001_counts_match_duckdb_and_enumeration(self):
        rng = np.random.default_rng(2002)
        n = 400
        k = [int(v) for v in rng.integers(0, 4, n)]
        a = [int(v) for v in rng.integers(-2, 3, n)]
        b = [int(v) for v in rng.integers(0, 5, n)]
        d = [f"{int(v) / 100:.2f}" for v in rng.choice([-125, 0, 250, 375], n)]
        w = [int(v) for v in rng.integers(-3, 9, n)]
        for col in (a, b, d, w):
            for i in np.flatnonzero(rng.random(n) < 0.12):
                col[int(i)] = None
        frame = pd.DataFrame({"k": obj(k), "a": obj(a), "b": obj(b), "d": obj(d), "w": obj(w)})
        db = torbles.from_ddl("CREATE TABLE f (k BIGINT, a BIGINT, b BIGINT, d DECIMAL(10,2), w BIGINT)", {"f": frame})
        t = db["f"]
        mask = rng.random(t.n) < 0.8
        scope = mbk.Scope.from_table(t, ["a", "b", "d", "w"], group_by=["k"], rows=mask)
        con = db.con
        con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(mask).astype(np.int64)}))
        logical = {"a": 'f."a"', "b": 'f."b"', "d": 'CAST(f."d" * 100 AS BIGINT)'}
        for cols in (["a"], ["a", "b"], ["d", "a", "b"], ["b", "d", "a", "b"]):
            for weight in (None, "w"):
                agg = "COUNT(*)" if weight is None else 'SUM(f."w")'
                sql = (f'SELECT f."k", {", ".join(logical[c] for c in cols)}, {agg} FROM f '
                       f'WHERE f.rowid IN (SELECT rid FROM sel_rows) {"AND f.w IS NOT NULL" if weight else ""} '
                       f'GROUP BY {", ".join(str(i + 1) for i in range(len(cols) + 1))}')
                duck = {}
                for row in con.execute(sql).fetchall():
                    duck.setdefault(row[0], {})[tuple(row[1:-1])] = int(row[-1])
                for repr_ in ("sparse", "dense"):
                    c = mt.counts(scope, cols, weight=weight, repr=repr_)
                    axes = c.axes
                    for g in range(scope.groups):
                        key = scope.key_values(g)[0]
                        got = {}
                        for coords, value in c.entries(g):
                            got[tuple(None if ax["null"] == i else ax["lo"] + i for ax, i in zip(axes, coords))] = value
                        want = {kk: v for kk, v in duck.get(key, {}).items() if v != 0}
                        self.assertEqual(got, want, (cols, weight, repr_, g))
                    self.assert_cuda_equal(c, scope, cols, kind="count", weight=weight, repr=repr_)
                    if repr_ == "sparse":
                        self.assert_sorted(c)
        # enumeration over array scopes, with explicit (wider) domains, the NULL slot, and refusals
        scope2, py, groups, rows = arrays_scope(np.random.default_rng(2003), n=80, ngroups=3)
        for cols in (["c"], ["x", "c"], ["c", "x", "y"], ["y", "x", "c", "z"]):
            dom = scope_domain(py, rows, cols)
            for weight in (None, "w"):
                want = count_oracle(py, groups, rows, 3, cols, weight, dom)
                for repr_ in ("dense", "sparse"):
                    c = mt.counts(scope2, cols, weight=weight, repr=repr_)
                    self.assertEqual(c.info["dims"], tuple(s + int(nl) for _, s, nl in dom))
                    for g in range(3):
                        self.assertTrue((c.array(g) == want[g]).all(), (cols, weight, repr_, g))
                    self.assert_cuda_equal(c, scope2, cols, kind="count", weight=weight, repr=repr_)
            wide = [(lo - 1, lo + span + 1, True) for lo, span, _ in dom]
            cw = mt.counts(scope2, cols, domain=wide, repr="dense")
            wdom = [(lo, hi - lo + 1, True) for lo, hi, _ in wide]
            for g in range(3):
                self.assertTrue((cw.array(g) == count_oracle(py, groups, rows, 3, cols, None, wdom)[g]).all())
            self.assert_cuda_equal(cw, scope2, cols, kind="count", domain=wide, repr="dense")
        with self.assertRaises(DomainError):                       # a domain that misses a selected value
            mt.counts(scope2, ["x"], domain=[(0, 3, True)])
        with self.assertRaises(DomainError):                       # NULL without a NULL slot
            mt.counts(scope2, ["x"], domain=[(-3, 3, False)])

    # ------------------------------------------------------------------ acceptance 1: marginalizations
    def test_tensor001_marginals_match_enumeration_and_direct_builds(self):
        rng = np.random.default_rng(2004)
        scope, py, groups, rows = arrays_scope(rng, n=70, ngroups=3)
        cols = ["c", "x", "y"]
        dom = scope_domain(py, rows, cols)
        for weight in (None, "w"):
            want = count_oracle(py, groups, rows, 3, cols, weight, dom)
            for repr_ in ("dense", "sparse", "factored"):
                c = mt.counts(scope, cols, weight=weight, repr="sparse").convert(repr_)
                for keep in ([0], [1], [2], [0, 1], [0, 2], [1, 2], [], [0, 1, 2]):
                    m = c.marginal(keep)
                    summed = tuple(k for k in range(3) if k not in keep)
                    for g in range(3):
                        self.assertTrue((m.array(g) == want[g].sum(axis=summed)).all(), (weight, repr_, keep, g))
                    if keep:                                       # the count tensor of the kept columns
                        direct = mt.counts(scope, [cols[k] for k in keep], weight=weight, repr=m.repr if
                                           m.repr != "factored" else "sparse")
                        self.assertTrue(m.same(direct), (weight, repr_, keep))
                    self.assertEqual(m.kind, "count")
        # moments: sums over axes of the enumerated arrays, counts summed alike (SQL), every representation
        axes = [["x", "y", "1"]] * 3
        for policy in ("listwise", "sql"):
            want = moment_oracle(py, groups, rows, 3, axes, policy, "w")
            reprs = ("dense", "sparse") + (("factored",) if policy == "listwise" else ())
            for repr_ in reprs:
                t = mt.moments(scope, ["x", "y", "1"], 3, weight="w", policy=policy, repr=repr_)
                for keep in ([0], [2], [0, 1], [1, 2], []):
                    m = t.marginal(keep)
                    summed = tuple(k for k in range(3) if k not in keep)
                    self.assertEqual(m.kind, "generic")
                    for g in range(3):
                        val, cnt, _, _ = want[g]
                        if policy == "sql":
                            v, cc = m.array(g)
                            self.assertTrue((v == val.sum(axis=summed)).all() and (cc == cnt.sum(axis=summed)).all())
                        else:
                            self.assertTrue((m.array(g) == val.sum(axis=summed)).all(), (repr_, keep, g))
        # a marginal is the contraction with all-ones vectors: the network agrees on every CPU and CUDA candidate
        t = mt.moments(scope, ["x", "y", "z"], 3, weight="w").slice([None, None, None])
        one = mt.dense([[1, 1, 1]], groups=True)
        for g_label in ("g",):
            net = mt.network([(t, g_label + "ijk"), (mt.dense([1, 1, 1]), "j")], g_label + "ik")
            plan = net.plan(devices=("cpu", "cuda"))
            m = t.marginal([0, 2])
            for i in range(plan.info["candidates"]):
                out, _ = net.contract(plan, i)
                for g in range(3):
                    self.assertTrue((out.array(0)[g] == m.array(g)).all(), i)
        del one

    def test_tensor001_slices_and_permutations_match_numpy(self):
        rng = np.random.default_rng(2005)
        scope, py, groups, rows = arrays_scope(rng, n=50, ngroups=2)
        cols = ["c", "x", "y"]
        dom = scope_domain(py, rows, cols)
        want = count_oracle(py, groups, rows, 2, cols, "w", dom)
        base = mt.counts(scope, cols, weight="w", repr="dense")
        dims = base.dims
        for repr_ in ("dense", "sparse", "factored"):
            c = base.convert(repr_)
            for spec in ([None, None, None], [1, None, (1, 3)], [(1, 3), 0, None], [0, 0, 0], [(0, 0), None, None],
                         [None, (dims[1] - 1, dims[1]), None]):
                s = c.slice(spec)
                index = tuple(x if isinstance(x, int) else (slice(None) if x is None else slice(*x)) for x in spec)
                for g in range(2):
                    self.assertTrue((s.array(g) == want[g][index]).all(), (repr_, spec))
                fixed = any(isinstance(x, int) for x in spec)
                self.assertEqual(s.kind, "generic" if fixed else "count")
            for perm in ([0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]):
                p = c.permute(perm)
                for g in range(2):
                    self.assertTrue((p.array(g) == np.transpose(want[g], perm)).all(), (repr_, perm))
                self.assertEqual([a["column"] for a in p.axes], [cols[k] for k in perm])
                if repr_ == "sparse":
                    self.assert_sorted(p)
        # the axis metadata of a category range: the NULL slot only when the range holds it
        ax = base.slice([None, None, None]).axes[1]
        lo, span, null = dom[1]
        cut = base.slice([None, (span - 1, span + int(null)), None]).axes[1]
        self.assertEqual((cut["lo"], cut["span"], cut["null"]), (lo + span - 1, 1, 1 if null else None))
        self.assertEqual(ax["column"], "x")
        feat = mt.moments(scope, ["x", "y", "1"], 2).slice([(1, 3), None]).axes[0]
        self.assertEqual(feat["columns"], ["y", "1"])

    # ------------------------------------------------------------------ acceptance 2: conversions
    def test_tensor001_conversions_preserve_order_dims_validity_and_coefficients(self):
        rng = np.random.default_rng(2006)
        scope, py, groups, rows = arrays_scope(rng, n=60, ngroups=4)
        big = I128 - 5
        tensors = [
            mt.moments(scope, ["x", "y", "1"], 3, policy="sql", repr="sparse"),        # counts: validity per entry
            mt.moments(scope, ["x", "y"], 2, weight="w"),
            mt.counts(scope, ["x", "c", "y"], repr="sparse"),                          # NULL slots
            mt.sparse((3, 5, 2), [((0, 4, 1), big), ((2, 0, 0), -big), ((1, 1, 1), 3), ((1, 1, 1), 4), ((0, 0, 0), 0)]),
            mt.sparse((2, 3), [(1, (0, 2), 7), (0, (1, 1), -9), (2, (1, 0), 1)], groups=3),
            mt.factors((3, 2), [(5, [[1, 0, -2], [3, 1]]), (-1, [[0, 4, 0], [1, 1]]), (2, 1, [[1, 1, 1], [0, 2]])],
                       groups=3),
            mt.dense(7), mt.dense([[0, 0], [0, 0]]), mt.dense(np.zeros((0, 3), dtype=object)),
        ]
        for t in tensors:
            i0 = t.info
            images = {"dense": t.to_dense(), "sparse": t.to_sparse()}
            if not i0["counts"]:
                images["factored"] = t.to_factored()
            else:
                with self.assertRaises(UnsupportedError):          # per-entry counts have no factored form
                    t.to_factored()
            for name, u in images.items():
                self.assertTrue(u.same(t) and t.same(u), (i0, name))
                info = u.info
                self.assertEqual((info["dims"], info["groups"], info["counts"], info["kind"], info["policy"]),
                                 (i0["dims"], i0["groups"], i0["counts"], i0["kind"], i0["policy"]))
                self.assertEqual(u.axes, t.axes)
                for g in range(t.groups):
                    self.assertEqual(u.group(g), t.group(g))
                    self.assertEqual(u.entries(g), t.entries(g))   # same entries, increasing multi-index order
                    if i0["counts"]:
                        a1, c1 = u.array(g); a2, c2 = t.array(g)
                        self.assertTrue((a1 == a2).all() and (c1 == c2).all())
                    else:
                        self.assertTrue((u.array(g) == t.array(g)).all())
                for other in images.values():                      # every pair of forms, both ways
                    self.assertTrue(u.convert(other.repr).same(other))
                self.assert_sorted(u.to_sparse())
            self.assertTrue(images["dense"].to_sparse().equal(images["sparse"]))
            self.assertTrue(images["sparse"].to_dense().equal(images["dense"]))
            if "factored" in images:
                self.assertTrue(images["factored"].to_dense().equal(images["dense"]))
        # exact coefficients at the 128-bit limit, and SQL validity: an entry kept for its count with value 0
        s = tensors[3]
        self.assertEqual(s.entries(0), [((0, 0, 0), 0), ((0, 4, 1), big), ((1, 1, 1), 7), ((2, 0, 0), -big)][1:])
        sql = tensors[0]
        tiny = mbk.Scope.from_arrays({"x": [0, 0, 5, -5], "y": [1, 7, 2, 3]}, valid={"y": [True, False, True, True]})
        for repr_ in ("sparse", "dense"):
            v = mt.moments(tiny, ["x", "y"], 1, policy="sql", repr=repr_)
            self.assertEqual(v.entries(0), [((0,), 0, 4), ((1,), 6, 3)])         # x: kept for its count 4
            self.assertEqual(v.values().entries(0), [((1,), 6)])                # values only: zeros leave
            self.assertTrue(v.to_sparse().to_dense().equal(v.to_dense()))
        self.assertEqual(sql.values().info["counts"], False)
        self.assertTrue(all(e[1] != 0 for g in range(sql.groups) for e in sql.values().entries(g)))
        # outer products in each representation, and ungrouping
        a, b = mt.dense([[1, -2], [0, 3]], groups=True), mt.dense([[2, 0, 5], [1, 1, 0]], groups=True)
        for repr_ in ("dense", "sparse", "factored"):
            o = mt.outer(a, b, repr_)
            for g in range(2):
                self.assertTrue((o.array(g) == np.multiply.outer(a.array(g), b.array(g))).all(), repr_)
        u = tensors[4].ungroup()
        self.assertEqual(u.dims, (3, 2, 3))
        self.assertEqual(u.axes[0]["kind"], "group")
        self.assertTrue((u.array(0) == np.stack([tensors[4].array(g) for g in range(3)])).all())
        for repr_ in ("dense", "factored"):
            self.assertTrue(tensors[4].convert(repr_).ungroup().same(u))
        # refusals: per-entry counts do not enter outer products; overflow is refused, never wrapped
        with self.assertRaises(DomainError):
            mt.outer(sql, sql, "dense")
        with self.assertRaises(OverflowError):
            mt.outer(mt.dense([big]), mt.dense([2]), "dense")
        with self.assertRaises(OverflowError):
            mt.sparse((2,), [((0,), big), ((1,), big)]).marginal([])
        with self.assertRaises(OverflowError):
            mt.dense([I128 + 1])
        with self.assertRaises(OverflowError):
            mt.factors((1,), [(big, [[1]]), (big, [[1]])])

    def test_tensor001_exactness_proof_before_reading(self):
        # N x max|w| x prod max|x| <= 2^127 - 1 is proven from the scope's extremes before any row is read
        x = np.array([2 ** 31 - 1, -(2 ** 31 - 1), 5], dtype=np.int64)
        scope = mbk.Scope.from_arrays({"x": x})
        t = mt.moments(scope, ["x"], 4)
        self.assertEqual(t.entry(0, (0, 0, 0, 0)), sum(int(v) ** 4 for v in x))
        self.assertTrue(t.same(mt.moments(scope, ["x"], 4, backend="cuda")))
        wide = mbk.Scope.from_arrays({"x": np.array([2 ** 40, 3], dtype=np.int64)})
        for repr_ in ("dense", "sparse", "factored"):
            mt.moments(wide, ["x"], 3, repr=repr_)                      # 2 x 2^120 fits
            with self.assertRaises(OverflowError):
                mt.moments(wide, ["x"], 4, repr=repr_)                  # 2 x 2^160 does not
        with self.assertRaises(OverflowError):
            mt.moments(wide, ["x"], 4, backend="cuda")
        # count tensors: int64 weights summed in 128 bits never overflow (rows < 2^64); a joint domain past 2^32
        # values per axis is outside the scope (UnsupportedError), never truncated
        big = mbk.Scope.from_arrays({"x": np.array([2 ** 62, 2 ** 62], dtype=np.int64),
                                     "w": np.array([2 ** 62 + 1, 2 ** 62 + 3], dtype=np.int64)})
        for backend in ("cpu", "cuda"):
            c = mt.counts(big, ["x"], weight="w", repr="dense", backend=backend)
            self.assertEqual(c.entry(0, (0,)), 2 ** 63 + 4)
        with self.assertRaises(UnsupportedError):
            mt.counts(big, ["x"], weight="w", domain=[(0, 2 ** 62, False)])

    def test_tensor001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(2007)
        scope, py, groups, rows = arrays_scope(rng, n=40, ngroups=3)

        def limit_of(fn, resource):
            try:
                fn(Budget(**{resource: 0}))
            except AlgebraBudgetError as e:
                self.assertEqual(e.resource, resource)
                return e.need
            self.fail(f"no {resource} need")

        for resource, fn in (
                ("tensor", lambda b: mt.moments(scope, ["x", "y", "1"], 3, budget=b)),
                ("bytes", lambda b: mt.moments(scope, ["x", "y", "1"], 3, budget=b)),
                ("states", lambda b: mt.moments(scope, ["x", "y"], 2, repr="sparse", strategy="sort", budget=b)),
                ("support", lambda b: mt.moments(scope, ["x", "y"], 2, repr="factored", budget=b)),
                ("degree", lambda b: mt.counts(scope, ["x", "c"], budget=b)),
                ("variables", lambda b: mt.counts(scope, ["x", "c", "y"], weight="w", budget=b)),
                ("tensor", lambda b: mt.moments(scope, ["x", "y"], 2, backend="cuda", budget=b)),
                ("bytes", lambda b: mt.counts(scope, ["x", "c"], repr="sparse", backend="cuda", budget=b)),
                ("tensor", lambda b: mt.counts(scope, ["x", "c"], repr="sparse").to_dense(budget=b)),
                ("support", lambda b: mt.counts(scope, ["x", "c"], repr="sparse").to_factored(budget=b)),
                ("bytes", lambda b: mt.counts(scope, ["x", "c"], repr="dense").marginal([1], budget=b))):
            need = limit_of(fn, resource)
            fn(Budget(**{resource: need}))                          # at the limit: admitted
            with self.assertRaises(AlgebraBudgetError) as cm:      # one past it: refused before allocation
                fn(Budget(**{resource: need - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), (resource, need, need - 1))
        needs = mt.need(scope, ["x", "y"], 2, backend="cuda")
        self.assertEqual(needs["tensor"], scope.groups * 4)

    def test_tensor001_chunked_builders_equal_one_shot_and_device(self):
        rng = np.random.default_rng(2008)
        scope, py, groups, rows = arrays_scope(rng, n=300, ngroups=5)
        n = scope.rows
        for kw in (dict(kind="moment", columns=["x", "y", "1"], order=3, repr="dense"),
                   dict(kind="moment", columns=["x", "y"], order=2, repr="sparse", policy="sql", strategy="sort"),
                   dict(kind="moment", columns=[["x"], ["y", "1"]], repr="factored", weight="w"),
                   dict(kind="count", columns=["x", "c"], repr="sparse", weight="w"),
                   dict(kind="count", columns=["c", "y"], repr="dense")):
            kw = dict(kw)
            kind, columns, order = kw.pop("kind"), kw.pop("columns"), kw.pop("order", None)
            one = (mt.moments(scope, columns, order, **kw) if kind == "moment" else mt.counts(scope, columns, **kw))
            a = mt.Builder(scope, columns, order, kind=kind, **kw)
            b = mt.Builder(scope, columns, order, kind=kind, **kw)
            a.accumulate(0, 70).accumulate(200, n)
            b.accumulate(70, 130).accumulate(130, 200)
            with self.assertRaises(ValueError):                     # a row is counted once
                b.accumulate(100, 101)
            got = a.merge(b).finish()
            self.assertTrue(got.equal(one), kw)
            with self.assertRaises(ValueError):                     # not every row covered
                mt.Builder(scope, columns, order, kind=kind, **kw).accumulate(0, 10).finish()
        # a larger scrambled device run of every device build
        scope, py, groups, rows = arrays_scope(np.random.default_rng(2009), n=3000, ngroups=7)
        for columns, order, repr_, policy in ((["x", "y", "z", "1"], 3, "dense", "listwise"),
                                              (["x", "y"], 4, "dense", "sql"),
                                              ([["x", "z"], ["y"], ["1", "x"]], None, "sparse", "listwise"),
                                              (["x", "y", "1"], 2, "sparse", "sql")):
            cpu = mt.moments(scope, columns, order, weight="w", policy=policy, repr=repr_)
            self.assert_cuda_equal(cpu, scope, columns, order=order, weight="w", policy=policy, repr=repr_)
        for cols in (["x", "y", "c", "z"], ["c"]):
            for repr_ in ("dense", "sparse"):
                cpu = mt.counts(scope, cols, weight="w", repr=repr_)
                self.assert_cuda_equal(cpu, scope, cols, kind="count", weight="w", repr=repr_)

    def test_tensor001_staleness_invalid_inputs_and_opt_in_import(self):
        frame = pd.DataFrame({"a": obj([1, 2, 3, None]), "b": obj([0, 1, 1, 2])})
        db = torbles.from_ddl("CREATE TABLE f (a BIGINT, b BIGINT)", {"f": frame})
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["a", "b"])
        built = mt.counts(scope, ["a", "b"])
        t.flip(0, 0)
        for fn in (lambda: mt.counts(scope, ["a"]), lambda: mt.moments(scope, ["a"], 2, backend="cuda"),
                   lambda: mt.Builder(scope, ["a"], 1)):
            with self.assertRaises(StaleStateError):
                fn()
        self.assertEqual(built.entries(0), built.entries(0))         # a built tensor stays readable
        s = mbk.Scope.from_arrays({"x": [1, 2], "y": [3, 4]})
        for exc, fn in ((ValueError, lambda: mt.moments(s, ["q"], 1)), (ValueError, lambda: mt.moments(s, ["x"])),
                        (UnsupportedError, lambda: mt.moments(s, ["x"], 9)),
                        (UnsupportedError, lambda: mt.moments(s, ["x"], 2, policy="sql", repr="factored")),
                        (ValueError, lambda: mt.moments(s, ["x"], 1, policy="pairs")),
                        (ValueError, lambda: mt.counts(s, ["1"])), (ValueError, lambda: mt.counts(s, ["x"], repr="tree")),
                        (TypeError, lambda: mt.dense([1.5])), (ValueError, lambda: mt.dense([[1, 2], [3]])),
                        (ValueError, lambda: mt.sparse((2,), [((2,), 1)])),
                        (ValueError, lambda: mt.dense([1, 2]).slice([(1, 3)])),
                        (ValueError, lambda: mt.dense([1, 2]).permute([1])),
                        (ValueError, lambda: mt.dense([1, 2]).entry(0, (2,)))):
            with self.assertRaises(exc):
                fn()
        out = subprocess.run([sys.executable, "-c", "import sys, torbles; print('torbles.math_tensor' in sys.modules)"],
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "False")


# ====================================================================== TENSOR-002
def random_tensor(rng, dims, density=0.6, scale=5, repr_="dense"):
    a = rng.integers(-scale, scale + 1, dims).astype(object)
    a[rng.random(dims) >= density] = 0
    t = mt.dense(a)
    return (t if repr_ == "dense" else t.convert(repr_)), a


NETWORKS = [   # (operand labels, output labels, dims per label)
    (["ij", "jk"], "ik", dict(i=3, j=4, k=2)),
    (["ij", "jk", "kl"], "il", dict(i=2, j=3, k=4, l=2)),
    (["ij", "jk", "kl", "lm"], "im", dict(i=2, j=3, k=2, l=3, m=2)),
    (["ia", "ib", "ic"], "abc", dict(i=3, a=2, b=2, c=2)),                 # a hyperedge shared by three tensors
    (["bij", "bjk"], "bik", dict(b=2, i=2, j=3, k=2)),                    # a batch label kept in the output
    (["ij", "ij"], "", dict(i=3, j=4)),                                   # a full contraction: a scalar
    (["i", "j"], "ji", dict(i=3, j=2)),                                   # an outer product
    (["ijk"], "ki", dict(i=2, j=3, k=4)),                                 # one tensor: a reduction and a permutation
    (["ijk", "kl", "lj"], "i", dict(i=3, j=2, k=3, l=2)),                 # a cycle
    (["ij", "jk", "ki", "il"], "l", dict(i=2, j=3, k=2, l=3)),
]


class Contractions(Base):
    # ------------------------------------------------------------------ acceptance 1: every order against einsum
    def test_tensor002_every_candidate_order_agrees_with_direct_contraction(self):
        rng = np.random.default_rng(2101)
        for labels, output, dims in NETWORKS:
            for reprs in (("dense",) * len(labels), tuple(rng.choice(["dense", "sparse", "factored"], len(labels)))):
                ops, arrays = [], []
                for lab, rp in zip(labels, reprs):
                    t, a = random_tensor(rng, [dims[c] for c in lab], repr_=rp)
                    ops.append((t, lab)); arrays.append(a)
                want = einsum(",".join(labels) + "->" + output, *arrays)
                net = mt.network(ops, output)
                plan = net.plan(devices=("cpu", "cuda"))
                info = plan.info
                n = len(labels)
                total = 1
                for k in range(2, n + 1):
                    total *= k * (k - 1) // 2
                self.assertEqual((info["complete"], info["orders"], info["total_orders"]), (True, total, total))
                self.assertEqual(info["candidates"], 3 * total)
                for c in plan.candidates:
                    out, run = net.contract(plan, c["index"])
                    got = out.array(0)
                    self.assertTrue(np.array_equal(np.asarray(got, dtype=object).reshape(np.shape(want)),
                                                   np.asarray(want, dtype=object)), (labels, c["index"]))
                    self.assertEqual(out.dims, tuple(dims[x] for x in output))
                    if c["device"] == "cuda":
                        self.assertIn(out.backend, ("cuda", "cuda-emulated"))
        # grouped tensors enter with their group label first: per-group contractions of moment tensors
        scope, py, groups, rows = arrays_scope(np.random.default_rng(2102), n=60, ngroups=3)
        T = mt.moments(scope, ["x", "y", "z"], 2, weight="w")
        v = mt.dense([1, -2, 3])
        net = mt.network([(T, "gij"), (v, "i"), (v, "j")], "g")
        plan = net.plan(devices=("cpu", "cuda"))
        want = [einsum("ij,i,j->", T.array(g), [1, -2, 3], [1, -2, 3]) for g in range(3)]
        for c in plan.candidates:
            out, _ = net.contract(plan, c["index"])
            self.assertEqual(list(out.array(0)), want, c["index"])

    def test_tensor002_large_entries_are_exact_or_refused(self):
        # products of 128-bit entries are proven per step: every admissible order is exact, unprovable ones refused
        big = 2 ** 60
        A = mt.dense([[big, 1], [2, -big]])
        B = mt.dense([[big, 3], [-1, big]])
        C = mt.dense([[1, 2], [3, 4]])
        net = mt.network([(A, "ij"), (B, "jk"), (C, "kl")], "il")
        plan = net.plan()
        want = einsum("ij,jk,kl->il", A.array(), B.array(), C.array())
        for c in plan.candidates:
            if c["status"] == "rejected":
                self.assertEqual((c["code"], c["refusal"]["kind"]), (-3, "bound"))
                with self.assertRaises(OverflowError):
                    net.contract(plan, c["index"])
            else:
                out, _ = net.contract(plan, c["index"])
                self.assertTrue((out.array() == want).all())
        huge = mt.dense([[2 ** 100]])
        net = mt.network([(huge, "ij"), (huge, "jk")], "ik")
        with self.assertRaises(OverflowError) as cm:
            net.plan(devices=("cpu", "cuda"))
        self.assertTrue(all(c["refusal"]["kind"] == "bound" for c in cm.exception.plan.candidates))

    # ------------------------------------------------------------------ acceptance 2: costs, workspace, budgets
    def test_tensor002_workspace_and_output_costs_are_complete(self):
        rng = np.random.default_rng(2103)
        H = mt.header_bytes()

        def dense_bytes(cells, order):
            return H + 16 + 16 * cells

        def sparse_bytes(nnz, order):
            return H + 16 + 16 + nnz * (4 * order + 16)

        for labels, output, dims in NETWORKS:
            ops = []
            for lab in labels:
                rp = "sparse" if len(lab) % 2 else "dense"
                ops.append((random_tensor(rng, [dims[c] for c in lab], density=0.5, repr_=rp)[0], lab))
            net = mt.network(ops, output)
            plan = net.plan(devices=("cpu", "cuda"))
            for c in plan.candidates:
                live = 0
                slots = [dict(labels=lab, nnz=len(t.entries(0)) if c["kernel"] == "sparse" else t.info["cells"],
                              bytes=0) for t, lab in ops]
                for k, (t, lab) in enumerate(ops):                 # converted inputs are workspace from the start
                    if t.repr != c["kernel"]:
                        slots[k]["bytes"] = (dense_bytes(t.info["cells"], t.order) if c["kernel"] == "dense"
                                             else sparse_bytes(slots[k]["nnz"], t.order))
                        live += slots[k]["bytes"]
                self.assertEqual(c["input_bytes"], live)
                peak = live
                for s in c["steps"]:
                    a = slots[s["a"]]
                    b = slots[s["b"]] if s["b"] is not None else None
                    cells = int(np.prod([dims[x] for x in s["out"]], dtype=object))
                    terms = int(np.prod([dims[x] for x in s["summed"]], dtype=object))
                    both = set(a["labels"]) | (set(b["labels"]) if b else set())
                    self.assertEqual(set(s["out"]) | set(s["summed"]), both)
                    self.assertEqual((s["cells"], s["terms"]), (cells, terms))
                    if c["kernel"] == "dense":
                        nnz, pairs, size, scratch = cells, cells * terms, dense_bytes(cells, len(s["out"])), 0
                    else:
                        if b:
                            shared = set(a["labels"]) & set(b["labels"])
                            pa = int(np.prod([dims[x] for x in b["labels"] if x not in shared], dtype=object))
                            pb = int(np.prod([dims[x] for x in a["labels"] if x not in shared], dtype=object))
                            pairs = min(a["nnz"] * pa, b["nnz"] * pb, a["nnz"] * b["nnz"])
                        else:
                            pairs = a["nnz"]
                        nnz = min(pairs, cells)
                        size = sparse_bytes(nnz, len(s["out"]))
                        scratch = mt.PAIR_BYTES * pairs + mt.KEY_BYTES * (a["nnz"] + (b["nnz"] if b else 0))
                    self.assertEqual((s["nnz"], s["pairs"], s["bytes"], s["scratch"]), (nnz, pairs, size, scratch))
                    live += size + scratch
                    self.assertEqual(s["live"], live)
                    peak = max(peak, live)
                    live -= scratch
                    for slot, freed in ((a, s["free_a"]), (b, s["free_b"])):
                        if slot is not None:
                            self.assertEqual(freed, slot["bytes"] > 0)
                            if freed:
                                live -= slot["bytes"]
                    slots.append(dict(labels=s["out"], nnz=nnz, bytes=size))
                last = c["steps"][-1]
                self.assertEqual(last["out"], list(output))
                self.assertEqual((c["output_bytes"], c["output_cells"]), (last["bytes"], last["cells"]))
                if c["device"] == "cpu":
                    self.assertEqual((c["peak_bytes"], c["need"]["bytes"]), (peak, peak))
                    out, run = net.contract(plan, c["index"])      # the actual sizes: dense exact, sparse within
                    for s, actual in zip(c["steps"], run["steps"]):
                        if c["kernel"] == "dense":
                            self.assertEqual((actual["nnz"], actual["bytes"]), (s["nnz"], s["bytes"]))
                        else:
                            self.assertLessEqual(actual["nnz"], s["nnz"])
                            self.assertLessEqual(actual["pairs"], s["pairs"])
                    self.assertLessEqual(run["peak_bytes"], c["peak_bytes"])
                    if c["kernel"] == "dense":
                        self.assertEqual(run["peak_bytes"], c["peak_bytes"])
                    self.assertEqual(run["output_bytes"], out.bytes)
                amounts = c["amounts"]
                self.assertEqual(amounts["materialize"], c["output_bytes"])
                self.assertEqual(amounts["fixed"], len(c["steps"]) + (1 if c["device"] == "cuda" else 0))
                if c["kernel"] == "dense":
                    self.assertGreaterEqual(amounts["chain"], sum(s["cells"] * s["terms"] for s in c["steps"]))
                rates = mt.default_rates()[c["device"]]
                self.assertAlmostEqual(c["total"], sum(amounts[k] * rates[k] for k in amounts), delta=1e-6 * c["total"] + 1e-9)

    def test_tensor002_over_budget_intermediates_are_refused_and_the_cheapest_order_changes(self):
        rng = np.random.default_rng(2104)

        def chain(i, j, k, l):
            A, a = random_tensor(rng, [i, j], density=1)
            B, b = random_tensor(rng, [j, k], density=1)
            Cc, cc = random_tensor(rng, [k, l], density=1)
            return mt.network([(A, "ij"), (B, "jk"), (Cc, "kl")], "il"), einsum("ij,jk,kl->il", a, b, cc)

        # (AB)C is cheaper for a long first matrix, A(BC) for a long last one: the chosen order changes
        firsts = []
        for shape in ((2, 30, 2, 30), (30, 2, 30, 2)):
            net, want = chain(*shape)
            plan = net.plan(kernels=("dense",))
            ch = plan.candidate(plan.chosen)
            firsts.append(tuple(sorted((ch["steps"][0]["a"], ch["steps"][0]["b"]))))
            out, _ = net.contract(plan)
            self.assertTrue((out.array() == want).all())
        self.assertEqual(firsts, [(0, 1), (1, 2)])
        # a tensor budget between the intermediates: the large one is refused, at its limit admitted, one past not
        net, want = chain(3, 40, 3, 40)
        plan = net.plan(kernels=("dense",))
        sizes = sorted({max(s["cells"] for s in c["steps"]) for c in plan.candidates})
        self.assertGreater(len(sizes), 1)
        small, large = sizes[0], sizes[-1]
        plan = net.plan(kernels=("dense",), budget=Budget(tensor=large - 1))
        for c in plan.candidates:
            biggest = max(max(s["cells"] for s in c["steps"]), 3 * 40, 40 * 3)
            if biggest > large - 1:
                self.assertEqual((c["status"], c["reason"], c["budget"]["resource"], c["budget"]["need"],
                                  c["budget"]["limit"]), ("rejected", "over budget", "tensor", biggest, large - 1))
                with self.assertRaises(AlgebraBudgetError):
                    net.contract(plan, c["index"])
            else:
                self.assertIn(c["status"], ("chosen", "admissible"))
        out, _ = net.contract(plan)
        self.assertTrue((out.array() == want).all())
        admitted = net.plan(kernels=("dense",), budget=Budget(tensor=large))
        self.assertTrue(all(c["status"] != "rejected" for c in admitted.candidates))
        # nothing fits: refused with the evidence attached; the bytes budget likewise
        with self.assertRaises(AlgebraBudgetError) as cm:
            net.plan(kernels=("dense",), budget=Budget(tensor=small - 1))
        evidence = cm.exception.plan
        self.assertEqual((evidence.chosen, evidence.reason), (None, "no admissible candidate"))
        self.assertTrue(all(c["status"] == "rejected" for c in evidence.candidates))
        # the bytes budget bounds each candidate's peak workspace (and the plan object itself)
        wide, _ = chain(3, 400, 3, 400)
        wplan = wide.plan(kernels=("dense",))
        peaks = sorted(c["peak_bytes"] for c in wplan.candidates)
        self.assertGreater(peaks[0], wplan.info["bytes"])
        p2 = wide.plan(kernels=("dense",), budget=Budget(bytes=peaks[0]))
        self.assertEqual(p2.candidate(p2.chosen)["peak_bytes"], peaks[0])
        self.assertTrue(all(c["budget"]["resource"] == "bytes" for c in p2.candidates if c["status"] == "rejected"))
        with self.assertRaises(AlgebraBudgetError) as cm:
            wide.plan(kernels=("dense",), budget=Budget(bytes=peaks[0] - 1))
        self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), ("bytes", peaks[0], peaks[0] - 1))
        # a force runs exactly its candidate, or refuses: it is never replaced
        costly = max(plan.candidates, key=lambda c: c["total"] if c["status"] != "rejected" else -1)
        forced = net.plan(kernels=("dense",), force=costly["index"])
        self.assertEqual((forced.chosen, forced.reason), (costly["index"], "forced"))
        self.assertTrue(all(c["reason"] in ("not forced", "forced") for c in forced.candidates if c["status"] != "rejected"))
        out, run = net.contract(forced)
        self.assertEqual(run["candidate"], costly["index"])
        rejected = next(c for c in plan.candidates if c["status"] == "rejected")
        with self.assertRaises(AlgebraBudgetError) as cm:
            net.plan(kernels=("dense",), force=rejected["index"], budget=Budget(tensor=large - 1))
        self.assertEqual(cm.exception.plan.reason, "forced candidate refused")
        # sparse states on identical tensors: the sparse kernel is chosen when the operands are very sparse
        A, a = random_tensor(rng, [60, 60], density=0.01, repr_="sparse")
        B, b = random_tensor(rng, [60, 60], density=0.01, repr_="sparse")
        net = mt.network([(A, "ij"), (B, "jk")], "ik")
        plan = net.plan()
        self.assertEqual(plan.candidate(plan.chosen)["kernel"], "sparse")
        out, _ = net.contract(plan)
        self.assertTrue((out.array() == einsum("ij,jk->ik", a, b)).all())
        Ad, Bd = A.to_dense(), B.to_dense()
        dense_net = mt.network([(Ad, "ij"), (Bd, "jk")], "ik")
        dense_plan = dense_net.plan(kernels=("dense",))
        self.assertTrue(dense_net.contract(dense_plan)[0].same(out.to_dense()))

    def test_tensor002_bounded_search_and_iterations_budget(self):
        rng = np.random.default_rng(2105)
        labels = ["ab", "bc", "cd", "de", "ef", "fa"]
        dims = dict(a=2, b=3, c=2, d=3, e=2, f=2)
        ops, arrays = [], []
        for lab in labels:
            t, a = random_tensor(rng, [dims[x] for x in lab])
            ops.append((t, lab)); arrays.append(a)
        net = mt.network(ops, "")
        want = einsum(",".join(labels) + "->", *arrays)
        plan = net.plan(max_orders=10, devices=("cpu", "cuda"))
        info = plan.info
        self.assertEqual((info["complete"], info["orders"], info["total_orders"]), (False, 10, 2700))
        for c in plan.candidates[::4]:
            out, _ = net.contract(plan, c["index"])
            self.assertEqual(out.entry(0, ()), want)
        orders = {tuple((s["a"], s["b"]) for s in c["steps"]) for c in plan.candidates}
        self.assertEqual(len(orders), 10)
        with self.assertRaises(AlgebraBudgetError) as cm:
            net.plan(budget=Budget(iterations=0))
        self.assertEqual(cm.exception.resource, "iterations")
        small = net.plan(budget=Budget(iterations=3))
        self.assertEqual(small.info["orders"], 3)
        five = mt.network(ops[:5], "af")
        p5 = five.plan()
        self.assertEqual((p5.info["complete"], p5.info["orders"]), (False, 64))
        out, _ = five.contract(p5)
        self.assertTrue((out.array() == einsum(",".join(labels[:5]) + "->af", *arrays[:5])).all())

    def test_tensor002_invalid_networks_are_refused(self):
        a, b = mt.dense([[1, 2], [3, 4]]), mt.dense([1, 2, 3])
        grouped = mt.dense([[1, 2], [3, 4]], groups=True)
        scope = mbk.Scope.from_arrays({"x": [1, 2, 3]}, valid={"x": [True, False, True]})
        counted = mt.moments(scope, ["x"], 1, policy="sql")
        for exc, fn in ((UnsupportedError, lambda: mt.network([(a, "ii")], "i")),       # a trace
                        (ValueError, lambda: mt.network([(a, "ij"), (b, "j")], "i")),  # inconsistent dims
                        (ValueError, lambda: mt.network([(a, "ij")], "k")),           # an output label nowhere
                        (ValueError, lambda: mt.network([(a, "ij")], "ii")),          # a repeated output label
                        (ValueError, lambda: mt.network([(grouped, "i")], "i")),      # a group label is missing
                        (DomainError, lambda: mt.network([(counted, "i")], "i")),     # counts do not contract
                        (UnsupportedError, lambda: mt.network([(b, "i")] * 9, "i"))):
            with self.assertRaises(exc):
                fn()
        net = mt.network([(counted.values(), "i")], "")
        self.assertEqual(net.contract()[0].entry(0, ()), 4)
        net = mt.network([(a, "ij"), (a, "jk")], "ik")
        with self.assertRaises(ValueError):
            net.plan(force=99)


if __name__ == "__main__":
    unittest.main()
