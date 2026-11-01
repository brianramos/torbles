# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""MULTI-002: adaptive refinement over MULTI-001 hierarchies (threshold, ratio and top-k requests answered from the
coarsest level down with sound per-region bounds, only undecided regions refined, unexpanded regions bounded or
explicitly unresolved) and change localization (DELTA-001 deltas of the leaves rolled up along the affected paths
only, with a complete rebuild when the local path is not compatible) (torbles/math_refine.c, math_refine_gpu.cu,
math_refine.py).

Oracles: complete fine-grain recomputation in Python integers and Fractions written here (per level and group: rows,
per term count and sum, present values and extremes, from the raw rows and the ancestor ids; every request decided
group by group; top-k by sorting with ties by the smaller id), the native complete fine-grain answer
(tr_refine_fine), DuckDB SQL over typed source tables (GROUP BY each level's key with HAVING, ORDER BY / LIMIT), and
the complete reconstruction (a new hierarchy and state built from the rows after the update). Every region bound and
every localized rollup is also computed on the CUDA path (emulated here, TORBLES_EMU_SCRAMBLE=1 for scrambled thread
order) and compared bit for bit with the CPU result. Seeds are fixed.
"""
import math
import subprocess
import sys
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_delta as md
from torbles import math_multi as mm
from torbles import math_refine as mr
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

BASIS = ["1", "x", "d", "x^2", "x*d"]
EXPS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]
NAMES = ("x", "d")


# ------------------------------------------------------------------ data and the fine-grain oracle
def tree(rng, n, sizes, skew=False):
    """leaf ids of n rows and the ids of their ancestors (level k: sizes[k] groups); balanced: parent = id * size_k+1
    // size_k; skewed: three quarters of every level under parent 0 and leaves drawn with a heavy head"""
    if skew:
        p = 1.0 / np.arange(1, sizes[0] + 1) ** 1.2
        leaf = rng.choice(sizes[0], n, p=p / p.sum())
    else:
        leaf = rng.integers(0, sizes[0], n)
    ids, parents = [leaf], []
    for k in range(1, len(sizes)):
        lo, hi = sizes[k - 1], sizes[k]
        par = np.where(np.arange(lo) < lo * 3 // 4, 0, rng.integers(0, hi, lo)) if skew else np.arange(lo) * hi // lo
        parents.append(par.astype(np.int64))
        ids.append(parents[-1][ids[-1]])
    return ids, parents


def dataset(rng, n, sizes, skew=False, heavy=True):
    ids, parents = tree(rng, n, sizes, skew)
    x = rng.integers(-50, 400, n)
    if heavy:                                     # a few heavy leaves: what adaptive refinement can find early
        hot = rng.choice(sizes[0], max(1, sizes[0] // 16), replace=False)
        x = np.where(np.isin(ids[0], hot), x + 5000, x)
    d = rng.integers(0, 80, n)
    vx, vd = rng.random(n) > 0.1, rng.random(n) > 0.05
    return ids, parents, {"x": x, "d": d}, {"x": vx, "d": vd}


def scopes(ids, sizes, cols, valid, rows=None):
    return [mbk.Scope.from_arrays(cols, valid=valid, groups=ids[k], ngroups=sizes[k], rows=rows)
            for k in range(len(sizes))]


def cells(ids, sizes, cols, valid, rows, mode="sql", mult=None):
    """per level, per group: [rows, [count per term], [sum per term], [present per column], [(min, max) per column]]"""
    n = len(ids[0])
    mult = np.ones(n, dtype=np.int64) if mult is None else mult
    out = []
    for k, G in enumerate(sizes):
        lv = [[0, [0] * len(EXPS), [0] * len(EXPS), [0, 0], [None, None]] for _ in range(G)]
        for i in range(n):
            if not rows[i] or not mult[i]:
                continue
            g, w = int(ids[k][i]), int(mult[i])
            c = lv[g]
            c[0] += w
            for j, e in enumerate(EXPS):
                need = [nm for nm, p in zip(NAMES, e) if p] if mode == "sql" else list(NAMES)
                if all(valid[nm][i] for nm in need):
                    c[1][j] += w
                    c[2][j] += w * math.prod(int(cols[nm][i]) ** p for nm, p in zip(NAMES, e))
            for j, nm in enumerate(NAMES):
                if valid[nm][i]:
                    c[3][j] += w
                    v = int(cols[nm][i])
                    lo, hi = c[4][j] or (v, v)
                    c[4][j] = (min(lo, v), max(hi, v))
        out.append(lv)
    return out


def atom_value(cell, atom):
    kind, j, coef = atom
    return coef * (cell[0] if kind == "rows" else cell[1][j] if kind == "count" else cell[2][j])


def agg(cell, atoms):
    return sum(atom_value(cell, a) for a in atoms)


def spec(atoms):
    """the request spec of oracle atoms [(kind, term index, coef)]"""
    out = []
    for kind, j, coef in atoms:
        name = "rows" if kind == "rows" else f"{kind}:{BASIS[j]}"
        out.append((coef, name))
    return out


OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}


def oracle_answer(level_cells, kind, atoms, op=None, T=None, datoms=None, k=None, descending=True):
    present = [g for g, c in enumerate(level_cells) if c[0]]
    if kind == "threshold":
        return {g for g in present if OPS[op](agg(level_cells[g], atoms), T)}
    if kind == "ratio":
        out = set()
        for g in present:
            D = agg(level_cells[g], datoms)
            if D and OPS[op](Fraction(agg(level_cells[g], atoms), D), T):
                out.add(g)
        return out
    key = (lambda g: (-agg(level_cells[g], atoms), g)) if descending else (lambda g: (agg(level_cells[g], atoms), g))
    return set(sorted(present, key=key)[:k])


def request_cases(lv_cells):
    """threshold / ratio / top-k requests with thresholds at quantiles of the target level's values"""
    sums = sorted(agg(c, [("sum", 1, 1)]) for c in lv_cells if c[0])
    q = [sums[int(len(sums) * f)] for f in (0.5, 0.9, 0.99)] if sums else [0]
    out = []
    for T in q:
        out.append(("threshold", [("sum", 1, 1)], dict(op=">", T=Fraction(T))))
        out.append(("threshold", [("sum", 1, 1), ("rows", 0, -3)], dict(op="<=", T=Fraction(T))))
    out.append(("threshold", [("count", 1, 1)], dict(op=">=", T=Fraction(sorted(c[1][1] for c in lv_cells if c[0])[-1]))))
    out.append(("threshold", [("sum", 4, 1), ("sum", 2, -7)], dict(op=">", T=Fraction(q[-1]) * 40)))
    out.append(("threshold", [("sum", 1, 1), ("sum", 1, -1)], dict(op="<", T=Fraction(1))))   # exact cancellation
    out.append(("ratio", [("sum", 1, 1)], dict(datoms=[("count", 1, 1)], op=">=", T=Fraction(q[1], 9))))
    out.append(("ratio", [("sum", 2, 3)], dict(datoms=[("rows", 0, 1)], op="<", T=Fraction(120, 7))))
    out.append(("ratio", [("sum", 1, 1), ("sum", 2, 2), ("count", 1, -1), ("rows", 0, 5)],      # 4 + 4 atoms: the
                dict(datoms=[("count", 1, 1), ("count", 2, 1), ("rows", 0, -1), ("sum", 2, 1)],  # largest plan
                     op=">", T=Fraction(-7, 3))))
    for k in (1, 3, 10):
        out.append(("topk", [("sum", 1, 1)], dict(k=k)))
        out.append(("topk", [("sum", 3, 1)], dict(k=k, descending=False)))
    return out


def kw_for(case):
    kind, atoms, kw = case
    out = {}
    if kind in ("threshold", "ratio"):
        out.update(op=kw["op"], threshold=kw["T"])
    if kind == "ratio":
        out["d"] = spec(kw["datoms"])
    if kind == "topk":
        out.update(k=kw["k"], descending=kw.get("descending", True))
    return out


def oracle_for(level_cells, case):
    kind, atoms, kw = case
    return oracle_answer(level_cells, kind, atoms, kw.get("op"), kw.get("T"), kw.get("datoms"), kw.get("k"),
                         kw.get("descending", True))


class Refine(unittest.TestCase):
    def build(self, rng, n, sizes, skew=False, mode="sql", rows=None, heavy=True):
        ids, parents, cols, valid = dataset(rng, n, sizes, skew, heavy)
        sel = np.ones(n, dtype=bool) if rows is None else rows
        lv = scopes(ids, sizes, cols, valid, sel)
        h = mm.Hierarchy(lv, BASIS, mode=mode)
        st = mr.State(h)
        return st, ids, parents, cols, valid, sel, cells(ids, sizes, cols, valid, sel, mode)

    def assert_entries_sound(self, st, r, case, oc, parents, target, truth):
        """every entry: a sound bound of its present descendants and a decision that holds for each of them; the
        entries partition the present target groups"""
        kind, atoms, kw = case
        present = {g for g, c in enumerate(oc[target]) if c[0]}
        covered = set()
        T = kw.get("T")
        for e in r.entries:
            lvl, g = e["level"], e["group"]
            if lvl == target:
                desc = {g}
            else:
                anc = np.arange(len(oc[target]))
                for k in range(target, lvl):
                    anc = parents[k][anc]
                desc = {t for t in present if anc[t] == g}
            self.assertTrue(desc, e)
            self.assertFalse(desc & covered, e)
            covered |= desc
            for t in desc:
                A = agg(oc[target][t], atoms)
                if kind == "ratio":
                    D = agg(oc[target][t], kw["datoms"])
                    Cv = T.denominator * A - T.numerator * D
                    self.assertTrue((e["lo"] is None or e["lo"] <= Cv) and (e["hi"] is None or Cv <= e["hi"]), (e, t, Cv))
                    self.assertTrue((e["dlo"] is None or e["dlo"] <= D) and (e["dhi"] is None or D <= e["dhi"]), (e, t, D))
                else:
                    self.assertTrue((e["lo"] is None or e["lo"] <= A) and (e["hi"] is None or A <= e["hi"]), (e, t, A))
                dec = e["decision"]
                if dec in ("all", "true", "in"):
                    self.assertIn(t, truth, (case, e))
                elif dec in ("none", "false", "out", "undefined"):
                    self.assertNotIn(t, truth, (case, e))
                else:
                    self.assertEqual(dec, "unresolved")
                    self.assertIn(t, r.unresolved)
            if e["exact"]:
                self.assertEqual(lvl, target)
                self.assertEqual(e["lo"], e["hi"])
        self.assertEqual(covered, present)
        self.assertTrue(r.answer <= truth, case)
        self.assertTrue(truth <= r.answer | r.unresolved, case)

    # ------------------------------------------------------------------ acceptance 1: adaptive == complete fine grain
    def test_multi002_acceptance_adaptive_answers_equal_complete_fine_grain_states(self):
        rng = np.random.default_rng(2002)
        trees = [([24, 4], False), ([64, 16, 4], False), ([64, 16, 4], True), ([96, 24, 6, 2], True)]
        for sizes, skew in trees:
            for mode in ("sql", "joint"):
                sel = rng.random(1800) < 0.9
                st, ids, parents, cols, valid, sel, oc = self.build(rng, 1800, sizes, skew, mode, sel)
                for target in range(len(sizes)):
                    for case in request_cases(oc[target]):
                        kind, atoms, kw = case
                        truth = oracle_for(oc[target], case)
                        args = kw_for(case)
                        r = st.query(kind, target, spec(atoms), **args)
                        self.assertTrue(r.complete)
                        self.assertEqual(r.answer, truth, (sizes, skew, mode, target, case))
                        self.assertEqual(r.unresolved, frozenset())
                        f = st.fine(kind, target, spec(atoms), **args)
                        self.assertEqual(f.answer, truth)
                        for g, v in f.values.items():         # the exact fine-grain values
                            A = agg(oc[target][g], atoms)
                            if kind == "ratio":
                                D = agg(oc[target][g], kw["datoms"])
                                self.assertEqual(v, (kw["T"].denominator * A - kw["T"].numerator * D, D))
                            else:
                                self.assertEqual(v, A)
                        g = st.query(kind, target, spec(atoms), backend="cuda", **args)
                        self.assertIn(g.backend, ("cuda", "cuda-emulated"))
                        self.assertTrue(g.same(r), (sizes, mode, target, case))
                        self.assert_entries_sound(st, r, case, oc, parents, target, truth)
                        self.assertEqual(r.stats["fine_work"], f.stats["fine_work"])

    def test_multi002_acceptance_answers_equal_duckdb_on_typed_tables(self):
        rng = np.random.default_rng(77)
        n = 2500
        sizes = [80, 20, 5]
        ids, parents, cols, valid = dataset(rng, n, sizes, skew=True)
        leaf_key = ids[0] * 7 + 100                            # keys: not the dense ids
        mid_key = ids[1] * 11 + 1000
        top_key = ids[2] * 13 + 5000
        d = [f"{int(v) / 100:.2f}" if valid["d"][i] else None for i, v in enumerate(cols["d"])]
        x = [int(v) if valid["x"][i] else None for i, v in enumerate(cols["x"])]
        df = pd.DataFrame({"leaf": leaf_key, "mid": mid_key, "top": top_key, "x": pd.Series(x, dtype=object),
                           "d": pd.Series(d, dtype=object)})
        db = torbles.from_ddl("CREATE TABLE f (leaf BIGINT, mid BIGINT, top BIGINT, x BIGINT, d DECIMAL(6,2));",
                              {"f": df})
        t = db["f"]
        levels = [mbk.Scope.from_table(t, ["x", "d"], group_by=[k]) for k in ("leaf", "mid", "top")]
        h = mm.Hierarchy(levels, BASIS)
        st = mr.State(h)
        con = db.con
        for target, key in enumerate(("leaf", "mid", "top")):
            sums = sorted(r[0] for r in con.execute(f'SELECT SUM(x) FROM f GROUP BY "{key}"').fetchall()
                          if r[0] is not None)
            for T in (sums[len(sums) // 2], sums[int(len(sums) * 0.9)]):
                duck = {(r[0],) for r in con.execute(
                    f'SELECT "{key}" FROM f GROUP BY "{key}" HAVING SUM(x) > {T}').fetchall()}
                for backend in ("cpu", "cuda"):
                    r = st.threshold(target, "sum:x", ">", T, backend=backend)
                    self.assertEqual({levels[target].key_values(g) for g in r.answer}, duck, (key, T))
            # AVG(d) >= 0.40 (logical: 40): NULL-only groups are undefined, as in SQL
            duck = {(r[0],) for r in con.execute(
                f'SELECT "{key}" FROM f GROUP BY "{key}" HAVING AVG(CAST(d * 100 AS HUGEINT)) >= 40').fetchall()}
            r = st.ratio(target, "sum:d", "count:d", ">=", 40)
            self.assertEqual({levels[target].key_values(g) for g in r.answer}, duck, key)
            for k in (1, 5):
                rows = con.execute(f'SELECT "{key}", SUM(x) s FROM f GROUP BY "{key}" ORDER BY s DESC NULLS LAST').fetchall()
                vals = [0 if v is None else v for _, v in rows]
                if len(vals) > k and vals[k - 1] == vals[k]:
                    continue                                   # a tie at the boundary: ids decide (tested above)
                r = st.topk(target, "sum:x", k)
                self.assertEqual({levels[target].key_values(g) for g in r.answer}, {(kk,) for kk, _ in rows[:k]})

    # ------------------------------------------------------------------ acceptance 2: bounded or unresolved
    def test_multi002_acceptance_every_unexpanded_region_is_bounded_or_unresolved(self):
        rng = np.random.default_rng(303)
        for sizes, skew in (([64, 16, 4], False), ([128, 32, 8, 2], True)):
            st, ids, parents, cols, valid, sel, oc = self.build(rng, 2400, sizes, skew)
            L = len(sizes)
            for target in range(L - 1):
                for case in request_cases(oc[target])[::2]:
                    kind, atoms, kw = case
                    truth = oracle_for(oc[target], case)
                    for limit in (1, 2, 5, 20, 0):
                        args = kw_for(case)
                        r = st.query(kind, target, spec(atoms), expand_limit=limit, **args)
                        self.assert_entries_sound(st, r, case, oc, parents, target, truth)
                        if limit:
                            self.assertLessEqual(sum(r.stats["expanded"]), limit)
                        if not r.complete:
                            self.assertTrue(any(e["decision"] == "unresolved" for e in r.entries))
                            self.assertEqual(r.stats["unresolved_groups"], len(r.unresolved))
                        g = st.query(kind, target, spec(atoms), expand_limit=limit, backend="cuda", **args)
                        self.assertTrue(g.same(r))
            # every region bound of every level contains every present descendant (CPU == CUDA)
            for target in range(L):
                for case in request_cases(oc[target])[:6]:
                    kind, atoms, kw = case
                    for at in range(target, L):
                        cpu, raw = st.bounds(kind, target, spec(atoms), at, **kw_for(case))
                        gpu, rawg = st.bounds(kind, target, spec(atoms), at, backend="cuda", **kw_for(case))
                        self.assertEqual(raw, rawg)
                        anc = np.arange(sizes[target])
                        for k in range(target, at):
                            anc = parents[k][anc]
                        for b in cpu:
                            for t in range(sizes[target]):
                                if anc[t] != b["group"] or not oc[target][t][0]:
                                    continue
                                A = agg(oc[target][t], atoms)
                                if kind == "ratio":
                                    A = kw["T"].denominator * A - kw["T"].numerator * agg(oc[target][t], kw["datoms"])
                                self.assertTrue((b["lo"] is None or b["lo"] <= A) and (b["hi"] is None or A <= b["hi"]))
                            if at == target:
                                self.assertTrue(b["exact"])

    def test_multi002_correlated_terms_merge_into_one_class(self):
        rng = np.random.default_rng(5)
        n = 600
        sizes = [12, 3]
        ids, parents = tree(rng, n, sizes)
        x = rng.integers(10, 20, n)
        every = {"x": np.ones(n, bool), "d": np.ones(n, bool)}
        st = mr.State(mm.Hierarchy(scopes(ids, sizes, {"x": x, "d": x * 2}, every), BASIS))
        # SUM(x) - SUM(x): one validity class whose row expression x - x cancels exactly (UNC-001 affine forms), so
        # every region bounds every leaf by exactly [0, 0] and the request is decided at the coarsest level
        b, _ = st.bounds("threshold", 0, [(1, "sum:x"), (-1, "sum:x")], 1)
        for e in b:
            self.assertEqual((e["lo"], e["hi"]), (0, 0))
            self.assertTrue(e["merged"])
        r = st.threshold(0, [(1, "sum:x"), (-1, "sum:x")], ">", 0)
        self.assertEqual((r.answer, r.stats["evaluated"][0]), (frozenset(), 0))
        # SUM(d) - 2 SUM(x) is 0 too, but d and x are separate columns for UNC-001: sound, not exact
        b2, _ = st.bounds("threshold", 0, [(1, "sum:d"), (-2, "sum:x")], 1)
        for e in b2:
            self.assertTrue(e["lo"] <= 0 <= e["hi"])

    # ------------------------------------------------------------------ change localization
    def updated(self, rng, ids, parents, sizes, cols, valid, mult, frac):
        """a delta touching `frac` of the present leaves (inserts with weights 1 and 2, deletes of one copy of present
        rows) of the rows ids / cols / valid with multiplicities mult, and the rows after it: (delta, new ids, cols,
        valid, multiplicities)"""
        present = sorted({int(g) for g in ids[0][mult > 0]})
        k = max(1, int(round(len(present) * frac)))
        touched = rng.choice(present, k, replace=False)
        ins_leaf = rng.choice(touched, 3 * k)
        ins_x, ins_d = rng.integers(-60, 6000, 3 * k), rng.integers(0, 90, 3 * k)
        ins_vx, ins_vd = rng.random(3 * k) > 0.2, rng.random(3 * k) > 0.1
        ins_w = rng.choice([1, 2], 3 * k)
        cand = np.flatnonzero((mult > 0) & np.isin(ids[0], touched))
        dels = rng.choice(cand, min(len(cand), k), replace=False) if len(cand) else np.zeros(0, dtype=np.int64)
        dl = np.concatenate([ins_leaf, ids[0][dels]])
        dx = np.concatenate([ins_x, cols["x"][dels]])
        dd = np.concatenate([ins_d, cols["d"][dels]])
        dvx = np.concatenate([ins_vx, valid["x"][dels]])
        dvd = np.concatenate([ins_vd, valid["d"][dels]])
        w = np.concatenate([ins_w, -np.ones(len(dels), dtype=np.int64)]).astype(np.int64)
        delta = md.Delta.from_arrays({"x": dx, "d": dd}, valid={"x": dvx, "d": dvd}, groups=dl, ngroups=sizes[0],
                                     weights=w)
        after = mult.astype(np.int64).copy()
        after[dels] -= 1
        after = np.concatenate([after, ins_w.astype(np.int64)])
        leaf2 = np.concatenate([ids[0], ins_leaf])
        ids2 = [leaf2]
        for p in parents:
            ids2.append(p[ids2[-1]])
        cols2 = {"x": np.concatenate([cols["x"], ins_x]), "d": np.concatenate([cols["d"], ins_d])}
        valid2 = {"x": np.concatenate([valid["x"], ins_vx]), "d": np.concatenate([valid["d"], ins_vd])}
        return delta, ids2, cols2, valid2, after

    def expand(self, ids2, cols2, valid2, mult):
        """the rows after the update with every multiplicity spelled out (a rebuild's input)"""
        rep = np.repeat(np.arange(len(mult)), mult)
        return [i[rep] for i in ids2], {k: v[rep] for k, v in cols2.items()}, {k: v[rep] for k, v in valid2.items()}

    def assert_state_matches(self, loc, oc2, sizes, rebuilt=None):
        for l in range(len(sizes)):
            c = loc.cells(l)
            inputs = loc.inputs(l)
            for g in range(sizes[l]):
                want = oc2[l][g]
                self.assertEqual(int(c["rows"][g]), want[0], (l, g))
                self.assertEqual([int(v) for v in c["count"][g]], want[1], (l, g))
                self.assertEqual(c["sum"][g], want[2], (l, g))
                for j in range(2):
                    rows_, present, lo, hi, mod = inputs[g][j]
                    self.assertEqual((rows_, present), (want[0], want[3][j]), (l, g, j))
                    if present:
                        tlo, thi = want[4][j]
                        self.assertTrue(lo <= tlo and hi >= thi, (l, g, j))
                        if mod:
                            self.assertEqual((tlo - lo) % mod, 0)
                            self.assertEqual((thi - lo) % mod, 0)
                        else:
                            self.assertEqual((lo, hi), (tlo, thi))
            if rebuilt is not None:
                rb = rebuilt.inputs(l)
                for g in range(sizes[l]):
                    if loc.tight(l, g):
                        self.assertEqual(inputs[g], rb[g], (l, g))

    def test_multi002_localized_updates_equal_complete_reconstruction(self):
        rng = np.random.default_rng(404)
        for sizes, skew in (([64, 16, 4], False), ([128, 32, 8, 2], True)):
            n = 2000
            st, ids, parents, cols, valid, sel, oc = self.build(rng, n, sizes, skew)
            for frac in (0.01, 0.1, 1.0):
                delta, ids2, cols2, valid2, mult = self.updated(rng, ids, parents, sizes, cols, valid,
                                                                sel.astype(np.int64), frac)
                oc2 = cells(ids2, sizes, cols2, valid2, np.ones(len(mult), bool), "sql", mult)
                loc = st.update(delta)
                gpu = st.update(delta, backend="cuda")
                one = st.update_native(delta)
                info = loc.info
                self.assertEqual((info["strategy"], info["status"], info["step"]), ("local", "carried", 1))
                ei, ec, ev = self.expand(ids2, cols2, valid2, mult)
                rebuilt = mr.State(mm.Hierarchy(scopes(ei, sizes, ec, ev), BASIS))
                self.assert_state_matches(loc, oc2, sizes, rebuilt)
                for other in (gpu, one):                        # the CUDA rollup and the one-call path: identical
                    self.assertEqual(other.changes, loc.changes)
                    for l in range(len(sizes)):
                        a, b = loc.cells(l), other.cells(l)
                        self.assertTrue(np.array_equal(a["rows"], b["rows"]) and np.array_equal(a["count"], b["count"]))
                        self.assertEqual(a["sum"], b["sum"])
                        self.assertEqual(loc.inputs(l), other.inputs(l))
                        self.assertEqual([loc.tight(l, g) for g in range(sizes[l])],
                                         [other.tight(l, g) for g in range(sizes[l])])
                self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
                # the changes: exactly the regions whose state changed, by exactly the difference
                want = {}
                for l in range(len(sizes)):
                    for g in range(sizes[l]):
                        a, b = oc[l][g], oc2[l][g]
                        if a[0] != b[0] or a[1] != b[1] or a[2] != b[2] or a[3] != b[3]:
                            want[(l, g)] = (b[0] - a[0], [y - x for x, y in zip(a[1], b[1])],
                                            [y - x for x, y in zip(a[2], b[2])])
                got = {(c["level"], c["group"]): (c["rows"], c["count"], c["sum"]) for c in loc.changes}
                self.assertEqual(got, want, (sizes, frac))
                self.assertEqual(info["changed"], [sum(1 for (l, _) in want if l == k) for k in range(len(sizes))])
                if frac < 1:
                    self.assertLess(info["work"], info["rebuild_work"])
                # requests after the update: adaptive == fine == the rebuilt state's
                for case in request_cases(oc2[0])[:8]:
                    kind, atoms, kw = case
                    truth = oracle_for(oc2[0], case)
                    r = loc.query(kind, 0, spec(atoms), **kw_for(case))
                    self.assertEqual(r.answer, truth)
                    self.assertEqual(rebuilt.query(kind, 0, spec(atoms), **kw_for(case)).answer, truth)
                    self.assert_entries_sound(loc, r, case, oc2, parents, 0, truth)
                # a chained update (on the CPU and with the CUDA rollup) keeps exact cells
                delta2, ids3, cols3, valid3, mult3 = self.updated(rng, ids2, parents, sizes, cols2, valid2, mult, 0.05)
                oc3 = cells(ids3, sizes, cols3, valid3, np.ones(len(mult3), bool), "sql", mult3)
                for backend in ("cpu", "cuda"):
                    chained = loc.update(delta2, backend=backend)
                    self.assertEqual(chained.info["step"], 2)
                    self.assertNotEqual(chained.info["identity"], loc.info["identity"])
                    self.assert_state_matches(chained, oc3, sizes)

    def test_multi002_incompatible_updates_rebuild_or_refuse(self):
        rng = np.random.default_rng(505)
        sizes = [32, 8, 2]
        n = 800
        ids, parents, cols, valid = dataset(rng, n, sizes)
        ids[0] = np.where(ids[0] == 9, 10, ids[0])               # leaf 9 absent
        for k in range(1, len(sizes)):
            ids[k] = parents[k - 1][ids[k - 1]]
        sel = np.ones(n, bool)
        st = mr.State(mm.Hierarchy(scopes(ids, sizes, cols, valid, sel), BASIS))
        one = md.Delta.from_arrays({"x": np.array([700]), "d": np.array([3])}, groups=np.array([9]), ngroups=sizes[0])
        with self.assertRaises(mr.RebuildRequired) as e:          # a leaf becomes present without a parent
            st.update(one)
        self.assertEqual(e.exception.reason, "new group")
        ids2 = [np.concatenate([ids[0], [9]])]
        for p in parents:
            ids2.append(p[ids2[-1]])
        cols2 = {"x": np.concatenate([cols["x"], [700]]), "d": np.concatenate([cols["d"], [3]])}
        valid2 = {"x": np.concatenate([valid["x"], [True]]), "d": np.concatenate([valid["d"], [True]])}
        levels2 = scopes(ids2, sizes, cols2, valid2)
        for backend in ("cpu", "cuda"):
            rb = st.update(one, levels=levels2, backend=backend)
            info = rb.info
            self.assertEqual((info["strategy"], info["reason"], info["status"]), ("rebuild", "new group", "proven"))
            oc2 = cells(ids2, sizes, cols2, valid2, np.ones(n + 1, bool))
            self.assert_state_matches(rb, oc2, sizes)
            self.assertEqual({(c["level"], c["group"]) for c in rb.changes},
                             {(0, 9), (1, int(parents[0][9])), (2, int(parents[1][parents[0][9]]))})
        with self.assertRaises(DomainError):                      # levels that are not the data after the delta
            st.update(one, levels=scopes(ids, sizes, cols, valid, sel))
        # another schema (a scale the state does not have): rebuilt from the levels, or refused without them
        other = md.Delta.from_arrays({"x": np.array([5]), "d": np.array([1])}, scales={"d": 2}, groups=np.array([1]),
                                     ngroups=sizes[0])
        with self.assertRaises(mr.RebuildRequired) as e:
            st.update(other)
        self.assertEqual(e.exception.reason, "schema")
        # deleting rows that are not there: refused with the violation, never rebuilt around
        bad = md.Delta.from_arrays({"x": np.array([1]), "d": np.array([1])}, groups=np.array([9]), ngroups=sizes[0],
                                   weights=np.array([-1]))
        with self.assertRaises(md.NegativeMultiplicityError) as e:
            st.update(bad, levels=levels2)
        self.assertEqual(e.exception.group, 9)
        with self.assertRaises(UnsupportedError):                 # the bounds count rows: no signed measure
            mr.State(mm.Hierarchy(scopes(ids, sizes, cols, valid, sel), BASIS), promise="signed")

    # ------------------------------------------------------------------ budgets
    def test_multi002_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(606)
        sizes = [48, 12, 3]
        ids, parents, cols, valid = dataset(rng, 900, sizes)
        h = mm.Hierarchy(scopes(ids, sizes, cols, valid), BASIS)
        need = mr.State.need(h)
        mr.State(h, budget=Budget(**need))
        for resource, v in need.items():
            with self.assertRaises(AlgebraBudgetError) as e:
                mr.State(h, budget=Budget(**{resource: v - 1}))
            self.assertEqual(e.exception.resource, resource)
        st = mr.State(h)
        for expand in (0, 3):
            q = st.request_need("threshold", 0, "sum:x", op=">", threshold=900, expand_limit=expand)
            st.threshold(0, "sum:x", ">", 900, expand_limit=expand, budget=Budget(**q))
            for resource, v in q.items():
                with self.assertRaises(AlgebraBudgetError) as e:
                    st.threshold(0, "sum:x", ">", 900, expand_limit=expand, budget=Budget(**{resource: v - 1}))
                self.assertEqual(e.exception.resource, resource)
        # the device bounds of a level's regions
        q = st.request_need("topk", 0, "sum:x", k=3)
        dev = st.bounds_need("topk", 0, "sum:x", 1, 12, k=3)
        both = {r: max(q.get(r, 0), dev.get(r, 0)) for r in set(q) | set(dev)}
        st.bounds("topk", 0, "sum:x", 1, backend="cuda", k=3, budget=Budget(**both))
        for resource in ("output", "bytes"):
            with self.assertRaises(AlgebraBudgetError) as e:
                st.bounds("topk", 0, "sum:x", 1, backend="cuda", k=3, budget=Budget(**{**both, resource: dev[resource] - 1}))
            self.assertEqual(e.exception.resource, resource)
        # a localized update on the CPU and with the CUDA rollup
        delta = md.Delta.from_arrays({"x": np.array([5, 6]), "d": np.array([1, 2])}, groups=np.array([0, 40]),
                                     ngroups=sizes[0])
        for backend in ("cpu", "cuda"):
            u = st.update_need(delta, backend)
            st.update(delta, backend=backend, budget=Budget(**u))
            for resource, v in u.items():
                with self.assertRaises(AlgebraBudgetError) as e:
                    st.update(delta, backend=backend, budget=Budget(**{resource: v - 1}))
                self.assertEqual(e.exception.resource, resource)

    def test_multi002_requests_are_validated(self):
        rng = np.random.default_rng(707)
        sizes = [16, 4]
        ids, parents, cols, valid = dataset(rng, 300, sizes)
        st = mr.State(mm.Hierarchy(scopes(ids, sizes, cols, valid), BASIS))
        with self.assertRaises(ValueError):
            st.threshold(0, "sum:y", ">", 1)                      # not a column
        with self.assertRaises(ValueError):
            st.threshold(0, "sum:d^2", ">", 1)                    # not in the basis
        with self.assertRaises(UnsupportedError):
            st.threshold(0, ["sum:x", "sum:d", "rows", "count:x", "count:d"], ">", 1)
        with self.assertRaises(TypeError):
            st.threshold(0, "sum:x", ">", 1.5)                    # exact thresholds only
        with self.assertRaises(UnsupportedError):                 # q A - p D leaves int64
            st.ratio(0, [((1 << 62), "sum:x")], "count:x", ">", Fraction(3, 7))
        with self.assertRaises(ValueError):
            st.topk(0, "sum:x", 0)
        with self.assertRaises(ValueError):
            st.threshold(2, "sum:x", ">", 1)                      # no level 2
        r = st.threshold(1, "rows", ">=", 1)                      # every present coarse group
        self.assertEqual(r.answer, frozenset(int(g) for g in np.unique(ids[1])))
        ln0, ln1 = st.links(0), st.links(1)                       # the links: parents and children lists
        present = np.unique(ids[0])
        self.assertTrue(np.array_equal(ln0["parent"][present], parents[0][present]))
        self.assertIsNone(ln0["offsets"])
        self.assertIsNone(ln1["parent"])
        off, kids = ln1["offsets"], ln1["children"]
        for p in range(sizes[1]):
            self.assertEqual(kids[off[p]:off[p + 1]].tolist(), [int(g) for g in present if parents[0][g] == p])

    def test_multi002_module_is_opt_in(self):
        code = "import sys, torbles; print('torbles.math_refine' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "False")


if __name__ == "__main__":
    unittest.main()
