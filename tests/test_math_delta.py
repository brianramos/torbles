# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DELTA-001: maintained FND-004 bank states under signed linear updates (torbles/math_delta.c, math_delta_gpu.cu,
math_delta.py).

Oracles: complete reconstruction after every update, twice and independently: the FND-004 builder over the updated
rows (each row repeated by its multiplicity) and DuckDB SQL over the updated data (GROUP BY with SUM(weight) for the
multiplicities and SUM(weight * term) in HUGEINT, SQL NULL semantics, FILTER for the listwise mode); Python integers
for signed measures. Engine-table updates are checked against DuckDB over the table's current logical values (decoded
natively after each mutation). Every update is also applied on the CUDA path (emulated here, TORBLES_EMU_SCRAMBLE=1
for scrambled thread order) and compared with the CPU state for equality, refusals included. Seeds are fixed.
"""
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_delta as md
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError
from torbles.mathematics import AlgebraBudgetError

BASIS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0)]


def duck_weighted(rows, names, terms, ngroups, mode="sql"):
    """[(rows, counts, sums)] per group by DuckDB over a weighted bag {id: (values, valid, group, weight)}"""
    import duckdb
    con = duckdb.connect()
    recs = [dict(g=g, w=w, **{n: (int(v) if ok else None) for n, v, ok in zip(names, vals, valid)})
            for vals, valid, g, w in rows.values() if w]
    frame = pd.DataFrame(recs, columns=["g", "w"] + list(names)).astype(object)
    con.register("b", frame)
    used = sorted({n for t in terms for n, e in zip(names, t) if e})
    joint = " AND ".join(f"{n} IS NOT NULL" for n in used) or "TRUE"
    parts = []
    for t in terms:
        factors = [f"CAST({n} AS HUGEINT)" for n, e in zip(names, t) for _ in range(e)]
        expr = " * ".join(factors) or "1"
        ok = " AND ".join(f"{n} IS NOT NULL" for n, e in zip(names, t) if e) or "TRUE"
        cond = ok if mode == "sql" else joint
        parts += [f"SUM(CAST(w AS HUGEINT)) FILTER (WHERE {cond})", f"SUM(CAST(w AS HUGEINT) * {expr}) FILTER (WHERE {cond})"]
    out = [[0, [0] * len(terms), [0] * len(terms)] for _ in range(ngroups)]
    if recs:
        for r in con.execute(f"SELECT g, SUM(CAST(w AS HUGEINT)), {', '.join(parts)} FROM b GROUP BY g").fetchall():
            vals = r[2:]
            out[int(r[0])] = [int(r[1]), [0 if v is None else int(v) for v in vals[0::2]],
                              [0 if v is None else int(v) for v in vals[1::2]]]
    return out


def materialize(rows, names, ngroups):
    """the FND-004 array scope of a bag (each row repeated by its multiplicity)"""
    vals, valid, groups = {n: [] for n in names}, {n: [] for n in names}, []
    for v, ok, g, w in rows.values():
        for _ in range(w):
            for n, x, f in zip(names, v, ok):
                vals[n].append(int(x)); valid[n].append(bool(f))
            groups.append(g)
    cols = {n: np.array(vals[n], dtype=np.int64) for n in names}
    return mbk.Scope.from_arrays(cols, valid={n: np.array(valid[n], dtype=bool) for n in names},
                                 groups=np.array(groups, dtype=np.int64), ngroups=ngroups)


def state_cells(st):
    return [[s["rows"], s["count"], s["sum"]] for s in st.states()]


def decoded(t, name):
    return mbk._decode(t, name).astype(np.int64)


def logical_frame(t, names):
    """the current logical values of a table (decoded natively now: after flips), None for NULL"""
    out = {}
    for n in names:
        codes = decoded(t, n) + t.codecs[n].affine[0]
        ok = decoded(t, t.nullable[n]) != 0 if n in t.nullable else np.ones(t.n, bool)
        out[n] = pd.Series([int(v) if f else None for v, f in zip(codes.tolist(), ok.tolist())], dtype=object)
    return pd.DataFrame(out)


def find_plane(t, column, row, keep=lambda before, after: True):
    """a stored plane whose flip changes only `column` at `row` (tried and flipped back)"""
    base = {c: decoded(t, c) for c in t.cols}
    for p in range(t.nplanes()):
        t.flip(p, row)
        after = {c: decoded(t, c) for c in t.cols}
        t.flip(p, row)
        changed = [c for c in t.cols if not np.array_equal(after[c], base[c])]
        if changed == [column] and keep(int(base[column][row]), int(after[column][row])):
            return p
    raise AssertionError(f"no plane changes {column} alone at row {row}")


class Delta(unittest.TestCase):
    def both(self, st, delta, **kw):
        """the CPU update, checked equal to the (emulated, scrambled) CUDA update; or the refusal both raise"""
        try:
            cpu = st.apply(delta, **kw)
        except Exception as e:
            with self.assertRaises(type(e)) as ctx:
                st.apply(delta, backend="cuda", **kw)
            for attr in ("group", "term", "reason", "have", "change", "need"):
                self.assertEqual(getattr(ctx.exception, attr, None), getattr(e, attr, None), attr)
            raise
        gpu = st.apply(delta, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu))
        self.assertEqual(cpu.identity, gpu.identity)
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    def check(self, st, rows, names, terms, ngroups, mode):
        """complete reconstruction twice: the FND-004 builder over the materialized rows and DuckDB"""
        rebuilt = mbk.build(materialize(rows, names, ngroups), terms, mode)
        self.assertTrue(st.equal_bank(rebuilt))
        self.assertTrue(st.bank().equal(rebuilt))
        self.assertEqual(state_cells(st), duck_weighted(rows, names, terms, ngroups, mode))

    # ------------------------------------------------------------------ acceptance 1: random signed sequences
    def test_delta001_acceptance_random_signed_sequences_match_reconstruction(self):
        rng = np.random.default_rng(2001)
        names, G = ["x", "y"], 13
        for mode in ("sql", "joint"):
            bag, next_id = {}, 0

            def fresh_row():
                x, y = int(rng.integers(-10**6, 10**6)), int(rng.integers(-500, 501))
                return ([x, y], [bool(rng.random() < 0.85), bool(rng.random() < 0.9)], int(rng.integers(0, G)))
            for _ in range(60):
                bag[next_id] = (*fresh_row(), 1); next_id += 1
            st = md.State.from_bank(mbk.build(materialize(bag, names, G), BASIS, mode))
            base, seen = st.identity, {st.identity}
            for step in range(25):
                ins, dels, wts, moves = (int(v) for v in rng.integers(0, 6, 4))
                vals, valid, groups, weights = [], [], [], []

                def emit(v, ok, g, w):
                    vals.append(v); valid.append(ok); groups.append(g); weights.append(w)
                for _ in range(ins):                                  # inserts: new rows, weight 1..3
                    v, ok, g = fresh_row()
                    w = int(rng.integers(1, 4))
                    bag[next_id] = (v, ok, g, w); next_id += 1
                    emit(v, ok, g, w)
                live = [k for k, r in bag.items() if r[3] > 0]
                for k in rng.choice(live, min(dels, len(live)), replace=False):   # deletes: all copies
                    v, ok, g, w = bag.pop(int(k))
                    emit(v, ok, g, -w)
                live = [k for k, r in bag.items() if r[3] > 0]
                for k in rng.choice(live, min(wts, len(live)), replace=False):    # weight changes, staying >= 0
                    v, ok, g, w = bag[int(k)]
                    nw = int(rng.integers(0, 5))
                    if nw != w:
                        bag[int(k)] = (v, ok, g, nw)
                        emit(v, ok, g, nw - w)
                live = [k for k, r in bag.items() if r[3] > 0]
                for k in rng.choice(live, min(moves, len(live)), replace=False):  # group moves: delete + insert
                    v, ok, g, w = bag[int(k)]
                    ng = int((g + 1 + rng.integers(0, G - 1)) % G)
                    bag[int(k)] = (v, ok, ng, w)
                    emit(v, ok, g, -w); emit(v, ok, ng, w)
                if not weights:
                    continue
                order = rng.permutation(len(weights))                 # the rows of a batch in any order
                d = md.Delta.from_arrays({n: np.array([vals[i][c] for i in order], dtype=np.int64) for c, n in enumerate(names)},
                                         valid={n: np.array([valid[i][c] for i in order]) for c, n in enumerate(names)},
                                         groups=np.array([groups[i] for i in order]), ngroups=G,
                                         weights=np.array([weights[i] for i in order]))
                new = self.both(st, d)
                self.assertEqual((new.step, new.info["base"]), (st.step + 1, base))
                self.assertNotIn(new.identity, seen)
                seen.add(new.identity)
                st = new
                self.check(st, bag, names, BASIS, G, mode)            # after every step
            self.assertFalse(st.bound)                                # array deltas: no scope describes the state

    def test_delta001_acceptance_engine_table_updates_match_reconstruction(self):
        rng = np.random.default_rng(2002)
        n = 1500
        frame = pd.DataFrame({"k": pd.Series([int(v) if rng.random() > 0.05 else None for v in rng.integers(0, 9, n)], dtype=object),
                              "x": pd.Series([int(v) if rng.random() > 0.1 else None for v in rng.integers(-3000, 3001, n)], dtype=object),
                              "d": pd.Series([f"{int(v) / 100:.2f}" if rng.random() > 0.1 else None
                                              for v in rng.integers(-99999, 100000, n)], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE f (k BIGINT, x BIGINT, d DECIMAL(10,2))", {"f": frame})
        t = db["f"]
        names, basis = ["x", "d"], mbk.monomials(["x", "d"], 2) + [(1, 2)]
        sel = rng.random(n) < 0.6
        scope = mbk.Scope.from_table(t, names, group_by=["k"], rows=sel)
        cache = mbk.Cache(capacity=8, max_bytes=64 << 20)
        bank0 = mbk.build(scope, basis)
        cache.put(bank0)
        st = md.State.from_bank(bank0)
        self.assertTrue(st.bound and st.bound_to(scope))
        for step in range(8):
            new_sel = sel.copy()
            if step % 2 == 0:                                         # a selection change: inserts and deletes
                flip = rng.choice(n, 40, replace=False)
                new_sel[flip] = ~new_sel[flip]
            else:                                                     # mutations: value changes and group moves
                for r in rng.choice(np.flatnonzero(sel), 3, replace=False):
                    t.flip(find_plane(t, "x", int(r)), int(r))
                r = int(rng.choice(np.flatnonzero(sel & (decoded(t, "k#v") != 0))))
                t.flip(find_plane(t, "k#v", r), r)                    # a move into the NULL group
                for r in rng.permutation(np.flatnonzero(sel & (decoded(t, "k#v") != 0))):   # a move to another key
                    try:
                        p = find_plane(t, "k", int(r), keep=lambda a, b: b <= t.maxcode("k"))
                    except AssertionError:
                        continue                                      # every flip of this code leaves the radix
                    t.flip(p, int(r))
                    break
                with self.assertRaises(StaleStateError):
                    md.Delta.between(scope, scope)                    # the old scope describes another revision
            new_scope = mbk.Scope.from_table(t, names, group_by=["k"], rows=new_sel)
            d = md.Delta.between(scope, new_scope)
            self.assertEqual(d.info["kind"], "diff")
            self.assertEqual(d.info["revision"], new_scope.revision)
            nxt = self.both(st, d)
            complete = mbk.build(new_scope, basis)                    # complete reconstruction
            self.assertTrue(nxt.equal_bank(complete), step)
            self.assertTrue(nxt.bound_to(new_scope) and not nxt.bound_to(scope))
            scoped = nxt.bank(new_scope)
            self.assertTrue(scoped.equal(complete) and scoped.info["scoped"])
            cur = logical_frame(t, ["k", "x", "d"])
            cur["d"] = [None if v is None else v for v in cur["d"]]
            db.con.register("cur", cur.iloc[np.flatnonzero(new_sel)])
            expect = {}
            exprs = [" * ".join(f"CAST({c} AS HUGEINT)" for c, k in zip(names, tm) for _ in range(k)) or "1" for tm in basis]
            for r in db.con.execute("SELECT k, COUNT(*), " + ", ".join(f"COUNT({e}), SUM({e})" for e in exprs)
                                    + " FROM cur GROUP BY k").fetchall():
                key = None if r[0] is None else int(r[0])
                expect[(key,)] = (int(r[1]), [int(v) for v in r[2::2]], [0 if v is None else int(v) for v in r[3::2]])
            got = {row["key"]: (row["rows"], row["count"], row["sum"]) for row in scoped.table()}
            self.assertEqual(got, expect, step)
            nxt.publish(cache, new_scope)                             # the old versions' entries are dropped
            self.assertTrue(cache.get(new_scope, basis).equal(complete))
            if step % 2 == 1:
                self.assertGreaterEqual(cache.stats["invalidations"], 1)
            with self.assertRaises(DomainError):
                st.bank(new_scope)                                    # the old state is not the new scope's
            st, scope, sel = nxt, new_scope, new_sel
        # a diff starting from another version is refused; a scope delta of the same table gives an unbound state
        older = mbk.Scope.from_table(t, names, group_by=["k"], rows=~sel)
        with self.assertRaises(DomainError):
            st.apply(md.Delta.between(older, scope))
        extra = sel & (np.arange(n) % 7 == 0)
        out = self.both(st, md.Delta.from_scope(mbk.Scope.from_table(t, names, group_by=["k"], rows=extra), sign=-1))
        self.assertFalse(out.bound)
        with self.assertRaises(DomainError):
            out.bank(scope)
        self.assertTrue(out.equal_bank(mbk.build(mbk.Scope.from_table(t, names, group_by=["k"], rows=sel & ~extra), basis)))

    # ------------------------------------------------------------------ acceptance 2: negative multiplicities
    def test_delta001_acceptance_rejects_negative_multiplicities_with_group_and_need(self):
        names, G = ["x"], 6
        bag = {0: ([5], [True], 1, 3), 1: ([7], [True], 2, 1), 2: ([2], [False], 3, 1), 3: ([4], [True], 3, 1),
               4: ([9], [True], 4, 1)}
        terms = [(0,), (1,), (2,)]
        st = md.State.from_bank(mbk.build(materialize(bag, names, G), terms))
        before = state_cells(st)
        cases = [  # (rows, expected group, term, reason, have, change)
            ([([5], [True], 0, -1)], 0, None, "negative rows", 0, -1),                 # group 0 has no rows
            ([([5], [True], 1, -5)], 1, None, "negative rows", 3, -5),                 # 5 copies, 3 present
            ([([7], [False], 2, -1)], 2, 1, "count above rows", 1, 0),                 # a NULL row that is not there
            ([([4], [True], 4, -1)], 4, 1, "sum without rows", 9, -4),                 # 4 is not there, 9 is
            ([([30], [True], 1, -1)], 1, 2, "negative even sum", 75, -900),            # x^2 would go negative
            ([([5], [True], 5, 1), ([5], [True], 0, -2)], 0, None, "negative rows", 0, -2),   # the first group reported
            ([([4], [True], 3, -1), ([2], [True], 3, -1)], 3, 1, "negative count", 1, -2),
        ]
        for rows, g, term, reason, have, change in cases:
            d = md.Delta.from_arrays({"x": [r[0][0] for r in rows]}, valid={"x": [r[1][0] for r in rows]},
                                     groups=[r[2] for r in rows], ngroups=G, weights=[r[3] for r in rows])
            with self.assertRaises(md.NegativeMultiplicityError) as ctx:
                self.both(st, d)
            e = ctx.exception
            self.assertIsInstance(e, DomainError)
            self.assertEqual((e.group, e.term, e.reason, e.have, e.change), (g, term, reason, have, change), rows)
            if change < 0 and term is None:
                self.assertEqual(e.need, -change)
            self.assertEqual(state_cells(st), before)                 # nothing exposed, the state unchanged
            signed = md.State.from_bank(mbk.build(materialize(bag, names, G), terms), promise="signed")
            out = self.both(signed, d)                                # a signed measure accepts it, exactly
            expect = [[r, list(c), list(s)] for r, c, s in before]
            for v, ok, gg, w in rows:
                expect[gg][0] += w
                for j, (e_,) in enumerate(terms):
                    if e_ == 0 or ok[0]:
                        expect[gg][1][j] += w
                        expect[gg][2][j] += w * v[0] ** e_
            self.assertEqual(state_cells(out), expect)
            if any(r[0] < 0 or min(r[1]) < 0 for r in expect):
                with self.assertRaises(DomainError):
                    out.bank()                                        # not a FND-004 bank (counts < 0)
        ok = md.Delta.from_arrays({"x": [5, 5]}, groups=[1, 1], ngroups=G, weights=[-2, -1])   # exactly all three
        empty = self.both(st, ok)
        self.assertEqual((empty.state(1)["rows"], empty.state(1)["present"]), (0, False))
        self.assertEqual(empty.info["present"], 3)

    # ------------------------------------------------------------------ exactness
    def test_delta001_overflow_proof_refuses_before_wrapping(self):
        big = 1 << 62
        scope = mbk.Scope.from_arrays({"x": [big] * 6 + [-big]}, groups=[0] * 7, ngroups=2)
        st = md.State.from_bank(mbk.build(scope, ["x", "x^2"]))      # sum of x^2 = 7 * 2^124 < 2^127
        self.assertEqual(st.state(0)["sum"], [5 * big, 7 * big * big])
        one = md.Delta.from_arrays({"x": [big]}, groups=[0], ngroups=2)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):                    # 8 * 2^124 = 2^127: does not fit
                st.apply(one, backend=backend)
        swap = md.Delta.from_arrays({"x": [big, big]}, groups=[0, 0], ngroups=2, weights=[1, -1])
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):                    # net zero, but the proof is conservative
                st.apply(swap, backend=backend)
        small = md.Delta.from_arrays({"x": [3, -(1 << 40)]}, groups=[0, 1], ngroups=2, weights=[-1, 5])
        out = self.both(st, small)                                    # |x| small: the proof holds
        self.assertEqual(out.state(0)["sum"], [5 * big - 3, 7 * big * big - 9])
        self.assertEqual(out.state(1)["sum"], [5 * -(1 << 40), 5 * (1 << 80)])
        heavy = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"x": [1]}), ["1"]), promise="signed")
        w = md.Delta.from_arrays({"x": [1]}, weights=[1 << 62])
        once = self.both(heavy, w)
        self.assertEqual(once.state(0)["rows"], (1 << 62) + 1)
        with self.assertRaises(OverflowError):
            once.apply(w)                                             # 2^63 + 1 rows: past int64
        with self.assertRaises(OverflowError):
            heavy.apply(md.Delta.from_arrays({"x": [1, 1]}, weights=[1 << 62, 1 << 62]))

    def test_delta001_group_moves_and_weight_changes_exact(self):
        rng = np.random.default_rng(2005)
        names, G, n = ["x", "y"], 40, 2000
        x, y = rng.integers(-(1 << 30), 1 << 30, n), rng.integers(-1000, 1000, n)
        vx, vy = rng.random(n) < 0.9, rng.random(n) < 0.8
        g = rng.integers(0, G, n)
        bag = {i: ([int(x[i]), int(y[i])], [bool(vx[i]), bool(vy[i])], int(g[i]), 1) for i in range(n)}
        for mode in ("sql", "joint"):
            st = md.State.from_bank(mbk.build(materialize(bag, names, G), BASIS, mode))
            work = dict(bag)
            moved = rng.choice(n, 300, replace=False)
            rows = []
            for i in moved:                                           # moves, some with a changed value too
                v, ok, gg, w = work[int(i)]
                ng = int(rng.integers(0, G))
                nv = [v[0] + int(rng.integers(-5, 6)), v[1]] if rng.random() < 0.5 else v
                work[int(i)] = (nv, ok, ng, w)
                rows += [(v, ok, gg, -w), (nv, ok, ng, w)]
            for i in rng.choice(n, 100, replace=False):               # weight changes 1 -> 0..6
                v, ok, gg, w = work[int(i)]
                nw = int(rng.integers(0, 7))
                work[int(i)] = (v, ok, gg, nw)
                rows.append((v, ok, gg, nw - w))
            d = md.Delta.from_arrays({nm: np.array([r[0][c] for r in rows]) for c, nm in enumerate(names)},
                                     valid={nm: np.array([r[1][c] for r in rows]) for c, nm in enumerate(names)},
                                     groups=np.array([r[2] for r in rows]), ngroups=G, weights=np.array([r[3] for r in rows]))
            self.assertEqual(d.rows, len(rows))
            out = self.both(st, d)
            self.check(out, work, names, BASIS, G, mode)
            self.assertEqual(d.info["touched"], len({r[2] for r in rows}))

    def test_delta001_cuda_equals_cpu_on_large_batches_scrambled(self):
        rng = np.random.default_rng(2006)
        n, G = 20000, 3000
        cols = {c: rng.integers(-(1 << 20), 1 << 20, n) for c in ("a", "b", "c")}
        valid = {"a": rng.random(n) < 0.9, "b": rng.random(n) < 0.95}
        basis = mbk.monomials(["a", "b", "c"], 2) + ["a^4", "a*b*c"]
        groups = rng.integers(0, G, n)
        for mode in ("sql", "joint"):
            scope = mbk.Scope.from_arrays(cols, valid=valid, groups=groups, ngroups=G)
            st = md.State.from_bank(mbk.build(scope, basis, mode), promise="signed")
            k = 8000
            d = md.Delta.from_arrays({c: rng.integers(-(1 << 20), 1 << 20, k) for c in ("a", "b", "c")},
                                     valid={"a": rng.random(k) < 0.9, "b": rng.random(k) < 0.95},
                                     groups=rng.integers(0, G, k), ngroups=G,
                                     weights=rng.integers(-3, 4, k))
            out = self.both(st, d)
            self.assertEqual(out.info["rows"], int(n + d.info["weight_net"]))

    # ------------------------------------------------------------------ identity and compatibility
    def test_delta001_identity_lineage_and_incompatible_deltas(self):
        rng = np.random.default_rng(2007)
        n, G = 300, 5
        x = rng.integers(-100, 100, n)
        scope = mbk.Scope.from_arrays({"x": x}, groups=rng.integers(0, G, n), ngroups=G)
        st = md.State.from_bank(mbk.build(scope, ["1", "x"]))
        d1 = md.Delta.from_arrays({"x": [1, 2]}, groups=[0, 1], ngroups=G)
        d2 = md.Delta.from_arrays({"x": [1, 2]}, groups=[0, 1], ngroups=G)
        d3 = md.Delta.from_arrays({"x": [1, 3]}, groups=[0, 1], ngroups=G)
        self.assertEqual(d1.identity, d2.identity)                    # the exact content
        self.assertNotEqual(d1.identity, d3.identity)
        a, b, c = st.apply(d1), st.apply(d2), st.apply(d3)
        self.assertEqual(a.identity, b.identity)                      # deterministic
        self.assertNotEqual(a.identity, c.identity)
        self.assertNotEqual(a.identity, st.identity)
        self.assertEqual((a.step, a.info["base"], st.info["base"]), (1, st.identity, st.identity))
        aa = a.apply(d1)
        self.assertEqual((aa.step, aa.info["base"]), (2, st.identity))
        self.assertNotEqual(aa.identity, a.identity)
        self.assertTrue(aa.equal(st.apply(md.Delta.from_arrays({"x": [1, 2, 1, 2]}, groups=[0, 1, 0, 1], ngroups=G))))
        self.assertNotEqual(aa.identity, st.apply(md.Delta.from_arrays({"x": [1, 2, 1, 2]}, groups=[0, 1, 0, 1],
                                                                           ngroups=G)).identity)   # other lineage
        for bad in (md.Delta.from_arrays({"x": [1]}, groups=[0], ngroups=G + 1),        # other groups
                    md.Delta.from_arrays({"x": [1], "y": [2]}, groups=[0], ngroups=G),  # other columns
                    md.Delta.from_arrays({"x": [1]}, scales={"x": 2}, groups=[0], ngroups=G)):   # another scale
            with self.assertRaises(DomainError):
                st.apply(bad)
        other = mbk.Scope.from_arrays({"x": x}, groups=rng.integers(0, G, n), ngroups=G)
        with self.assertRaises(DomainError):
            st.apply(md.Delta.between(other, scope))                  # a diff from another source
        same = md.Delta.between(scope, scope)
        self.assertEqual(same.rows, 0)
        nothing = self.both(st, same)
        self.assertTrue(nothing.equal(st) and nothing.bound and nothing.step == 1)
        foreign = mbk.build(scope, ["1"])
        foreign.scope = other
        with self.assertRaises(DomainError):                          # a bank of another scope
            md.State.from_bank(foreign)
        db = torbles.from_ddl("CREATE TABLE f (k BIGINT, x BIGINT)", {"f": pd.DataFrame({"k": [1, 2, 1], "x": [4, 5, 6]})})
        db2 = torbles.from_ddl("CREATE TABLE f (k BIGINT, x BIGINT)", {"f": pd.DataFrame({"k": [1, 2, 1], "x": [4, 5, 6]})})
        ts = mbk.Scope.from_table(db["f"], ["x"], group_by=["k"])
        tst = md.State.from_bank(mbk.build(ts, ["1", "x"]))
        with self.assertRaises(DomainError):                          # rows of another table: another schema
            tst.apply(md.Delta.from_scope(mbk.Scope.from_table(db2["f"], ["x"], group_by=["k"])))
        with self.assertRaises(ValueError):
            md.Delta.from_scope(ts, sign=2)
        with self.assertRaises(ValueError):
            md.Delta.from_arrays({"x": [1]}, groups=[7], ngroups=3)

    def test_delta001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(2008)
        n, G = 2000, 50
        xs, gs = rng.integers(-99, 99, n), rng.integers(0, G, n)
        scope = mbk.Scope.from_arrays({"x": xs}, groups=gs, ngroups=G)
        st = md.State.from_bank(mbk.build(scope, ["1", "x", "x^2"]))
        cols = {"x": rng.integers(-99, 99, 300)}
        valid = {"x": rng.random(300) < 0.9}
        grp = rng.integers(0, G, 300)
        needs = md.Delta.need(columns=cols, valid=valid)
        self.assertEqual((needs["output"], needs["states"]), (300, 300))
        for resource, value in needs.items():                         # the delta: output, states, bytes
            md.Delta.from_arrays(cols, valid=valid, groups=grp, ngroups=G, budget=Budget(**{resource: value}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                md.Delta.from_arrays(cols, valid=valid, groups=grp, ngroups=G, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), resource)
        d = md.Delta.from_arrays(cols, valid=valid, groups=grp, ngroups=G)
        ref = st.apply(d)
        for backend in ("cpu", "cuda"):                               # the apply: every resource
            needs = st.need(d, backend)
            self.assertEqual(needs["states"], G * 3)
            for resource, value in needs.items():
                self.assertTrue(st.apply(d, backend=backend, budget=Budget(**{resource: value})).equal(ref))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    st.apply(d, backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        self.assertNotEqual(st.need(d)["bytes"], st.need(d, "cuda")["bytes"])
        diff_need = md.Delta.need(old=scope, new=scope)
        self.assertEqual(diff_need.get("output", 0), 0)
        sel = np.ones(n, bool); sel[:10] = False
        other = mbk.Scope.from_arrays({"x": xs}, groups=gs, ngroups=G, rows=sel)      # 10 rows deleted
        self.assertEqual(md.Delta.need(old=scope, new=other)["output"], 10)
        with self.assertRaises(AlgebraBudgetError):
            md.Delta.between(scope, other, budget=Budget(output=9))
        self.assertEqual(md.Delta.between(scope, other, budget=Budget(output=10)).rows, 10)

    def test_delta001_opt_in_import(self):
        code = "import sys, torbles; print('torbles.math_delta' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
