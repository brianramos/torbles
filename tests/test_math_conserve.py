# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-003: conservation and stability of invariants across revisions.

Oracles (independent of the implementation): the Gram moments summed row by row with Python integers at every
revision, and the complete native recomputation of the Gram (math_ident.gram on the new scope) after every update; each
identity's evidence (checked, excluded, violating rows) counted row by row with Python integers and its violating rows
counted by DuckDB; the incremental evidence of a delta (weights by sign, witnesses) recomputed in Python; compositions
q(T(x)) - q(x) expanded with Python dict polynomials written here, and counterexample values evaluated with Python
integers. Every native update also runs on the CUDA path (emulated here, threads in a scrambled order) and the
trackers are compared for equality. Seeds are fixed.
"""
import itertools
import unittest

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles import math_conserve as mcons
from torbles import math_delta as md
from torbles import math_ident as mi
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 2003

def cuda_label():
    """the backend label of a CUDA result: "cuda" on a device, "cuda-emulated" when the kernels run in the emulation"""
    from torbles import math_gpu
    return "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def obj(values):
    return pd.Series(values, dtype=object)


def make(values, name="t"):
    frame = pd.DataFrame({c: obj(v) for c, v in values.items()})
    ddl = f"CREATE TABLE {name} (" + ", ".join(f'"{c}" BIGINT' for c in values) + ")"
    return torbles.from_ddl(ddl, {name: frame})


def arrays(cols, nulls=None):
    """an array scope of int columns, None = NULL"""
    valid = {}
    data = {}
    for c, v in cols.items():
        if any(x is None for x in v):
            valid[c] = np.array([x is not None for x in v])
        data[c] = np.array([0 if x is None else int(x) for x in v], dtype=np.int64)
    return mbk.Scope.from_arrays(data, valid=valid or None)


def duck_violations(cols, where):
    con = duckdb.connect()
    con.register("f", pd.DataFrame({c: pd.array(v, dtype="Int64") for c, v in cols.items()}))
    return con.execute(f"SELECT count(*) FROM f WHERE {where}").fetchone()[0]


# ------------------------------------------------------------------ oracles
def basis_oracle(k, d):
    return sorted((e for e in itertools.product(range(d + 1), repeat=k) if sum(e) <= d), key=lambda e: (sum(e), e))


def mono(e, point):
    v = 1
    for x, p in zip(point, e):
        v *= x ** p
    return v


def moments_oracle(cols, variables, degree, mask=None):
    """{moment exponent: exact sum} over the selected rows where every variable is present"""
    basis = basis_oracle(len(variables), degree)
    mom = {tuple(a + b for a, b in zip(basis[i], basis[j])) for i in range(len(basis)) for j in range(i, len(basis))}
    out = {m: 0 for m in mom}
    n = len(cols[variables[0]])
    domain = 0
    for r in range(n):
        if mask is not None and not mask[r]:
            continue
        pt = [cols[v][r] for v in variables]
        if any(x is None for x in pt):
            continue
        domain += 1
        for m in out:
            out[m] += mono(m, pt)
    return out, domain


def evidence_oracle(cols, columns, terms, mask=None):
    """checked, excluded, violations and the smallest violating row of sum c prod col^e = 0 (a NULL in a column the
    polynomial reads: excluded)"""
    n = len(cols[columns[0]])
    support = [q for q in range(len(columns)) if any(e[q] for e in terms)] or list(range(len(columns)))
    checked = excluded = violations = 0
    witness = None
    for r in range(n):
        if mask is not None and not mask[r]:
            continue
        pt = [cols[c][r] for c in columns]
        if any(pt[q] is None for q in support):
            excluded += 1
            continue
        pt = [0 if x is None else x for x in pt]
        checked += 1
        if sum(c * mono(e, pt) for e, c in terms.items()) != 0:
            violations += 1
            witness = r if witness is None else min(witness, r)
    return dict(checked=checked, excluded=excluded, violations=violations, witness=witness)


def p_mul(a, b):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c}


def compose(q, images, n):
    """q(T(x)) - q(x) with Python dict polynomials (images: {variable index: polynomial}; the rest identity maps)"""
    one = tuple([0] * n)
    T = [images.get(v, {tuple(1 if i == v else 0 for i in range(n)): 1}) for v in range(n)]
    out = {}
    for e, c in q.items():
        term = {one: c}
        for v, k in enumerate(e):
            for _ in range(k):
                term = p_mul(term, T[v])
        for f, d in term.items():
            out[f] = out.get(f, 0) + d
    for e, c in q.items():
        out[e] = out.get(e, 0) - c
    return {e: c for e, c in out.items() if c}


def evaluate(p, point):
    return sum(c * mono(e, point) for e, c in p.items())


def strip(ev):
    return {k: v for k, v in ev.items() if k not in ("backend", "device")}


def same_tracker(test, a, b):
    """two trackers hold the same state: records, steps, moments"""
    test.assertEqual(a.moments(), b.moments())
    test.assertEqual([strip_rec(r) for r in a.records()], [strip_rec(r) for r in b.records()])
    test.assertEqual(a.steps(), b.steps())


def strip_rec(r):
    r = dict(r)
    r["evidence"] = strip(r["evidence"])
    return r


SUM = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): -1}          # x + y - s over (x, y, s)
SQUARE = {(2, 0, 0): 1, (0, 1, 0): -1}                      # x^2 - y over (x, z, ...) by position


class Disc003(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)

    def planted(self, n, lo=-40, hi=41):
        x = [int(v) for v in self.rng.integers(lo, hi, n)]
        y = [int(v) for v in self.rng.integers(lo, hi, n)]
        t = [int(v) for v in self.rng.integers(-6, 7, n)]
        return {"x": x, "y": y, "s": [a + b for a, b in zip(x, y)], "t": t, "q": [a * a for a in x]}

    def ids(self, scope, polys=None):
        polys = polys or [SUM, {(2, 0, 0): 1, (0, 0, 1): -1}, {(1, 0, 0): 1, (0, 1, 0): -1}]
        return mi.candidates(scope, ["x", "y", "s"], polys)

    # ---------------------------------------------------------- build
    def test_disc003_build_matches_python_gram_evidence_and_duckdb(self):
        v = self.planted(240)
        v["s"][37] = None
        v["x"][11] = None
        db = make(v)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "s", "t"])
        ids = self.ids(scope)
        tr = mcons.track(scope, ["x", "y", "s"], degree=2, identities=ids)
        want, domain = moments_oracle(v, ["x", "y", "s"], 2)
        self.assertEqual(tr.moments(), want)
        self.assertEqual(tr.info["domain"], domain)
        recs = tr.records()
        for i, poly in enumerate([SUM, {(2, 0, 0): 1, (0, 0, 1): -1}, {(1, 0, 0): 1, (0, 1, 0): -1}]):
            ev = evidence_oracle(v, ["x", "y", "s"], poly)
            got = recs[i]["evidence"]
            self.assertEqual((got["checked"], got["excluded"], got["violations"]),
                             (ev["checked"], ev["excluded"], ev["violations"]), i)
            self.assertEqual(recs[i]["status"], "certified" if not ev["violations"] else "failed")
            if ev["violations"]:
                f = recs[i]["first_failure"]
                self.assertEqual((f["step"], f["source"], f["row"], f["violations"]), (0, "scope", ev["witness"], ev["violations"]))
                self.assertEqual(f["values"], {c: v[c][ev["witness"]] for c in tr.identity(i)["columns"]})
        self.assertEqual(duck_violations(v, "x + y - s <> 0"), 0)
        self.assertEqual(duck_violations(v, "x * x - s <> 0"), recs[1]["evidence"]["violations"])
        cert = tr.certificate(0)
        self.assertEqual(cert.kind, "snapshot")
        self.assertTrue(cert.fresh(scope).fresh)
        self.assertIsNone(tr.certificate(1))
        st = tr.steps()
        self.assertEqual((st[0]["kind"], st[0]["bound"], st[0]["rows"], st[0]["certified"], st[0]["failed"]),
                         ("build", True, 240, 1, 2))

    # ---------------------------------------------------------- incremental updates
    def test_disc003_incremental_updates_match_complete_recomputation(self):
        n = 160
        v = self.planted(n)
        cols = {c: list(v[c]) for c in ("x", "y", "s", "t")}
        mask = [bool(b) for b in self.rng.integers(0, 2, n)]
        polys = [SUM, {(1, 0, 0): 1, (0, 1, 0): -1}]

        def scope_of(cols, mask):
            sc = arrays(cols)
            return mbk.Scope.from_arrays({c: np.array([0 if x is None else x for x in cols[c]], np.int64)
                                          for c in cols},
                                         valid={c: np.array([x is not None for x in cols[c]]) for c in cols
                                                if any(x is None for x in cols[c])} or None,
                                         rows=np.array(mask))
        s = scope_of(cols, mask)
        tr = mcons.track(s, ["x", "y", "s"], degree=2, identities=mi.candidates(s, ["x", "y", "s"], polys))
        trc = mcons.track(s, ["x", "y", "s"], degree=2, identities=mi.candidates(s, ["x", "y", "s"], polys))
        for step in range(1, 7):
            new = {c: list(cols[c]) for c in cols}
            newmask = list(mask)
            for r in self.rng.choice(n, 12, replace=False):          # selection changes (inserts and deletes)
                newmask[int(r)] = not newmask[int(r)]
            for r in self.rng.choice(n, 8, replace=False):           # value changes (some break x + y = s)
                r = int(r)
                tt = int(self.rng.integers(-5, 6))
                new["x"][r] = (new["x"][r] or 0) + tt
                new["y"][r] = (new["y"][r] or 0) - tt if step % 2 else (new["y"][r] or 0) + tt
            if step == 3:
                new["s"][5] = None                                   # a NULL appears
            s2 = scope_of(new, newmask)
            d = md.Delta.between(s, s2)
            ev_cpu, ev_dev = tr.evaluate(d, backend="cpu"), tr.evaluate(d, backend="cuda")
            self.assertEqual(strip(ev_cpu), strip(ev_dev), step)
            self.assertEqual(ev_dev["backend"], cuda_label())
            tr = tr.update(d, scope=s2)
            trc = trc.update(md.Delta.between(s, s2), scope=s2, backend="cuda")
            same_tracker(self, tr, trc)
            # complete recomputation of the Gram: Python and the native full pass
            want, domain = moments_oracle(new, ["x", "y", "s"], 2, newmask)
            self.assertEqual(tr.moments(), want, step)
            self.assertEqual(tr.info["domain"], domain)
            full = mi.gram(s2, mi.plan(s2, ["x", "y", "s"], degree=2))
            self.assertTrue(tr.gram(s2) == full, step)
            for i, poly in enumerate(polys):
                ev = evidence_oracle(new, ["x", "y", "s"], poly, newmask)
                got = tr.record(i)["evidence"]
                self.assertEqual((got["checked"], got["excluded"], got["violations"]),
                                 (ev["checked"], ev["excluded"], ev["violations"]), (step, i))
                self.assertEqual(tr.record(i)["status"], "certified" if not ev["violations"] else "failed")
                if not ev["violations"]:
                    self.assertTrue(tr.certificate(i).fresh(s2).fresh)
                    self.assertTrue(tr.authorize(i, s2)["authorized"])
            cols, mask, s = new, newmask, s2
        st = tr.steps()
        self.assertEqual([x["kind"] for x in st], ["build"] + ["diff"] * 6)
        self.assertTrue(all(x["bound"] for x in st))

    def test_disc003_incremental_evidence_of_a_delta_matches_python(self):
        n = 90
        v = self.planted(n)
        s = arrays({c: v[c] for c in ("x", "y", "s", "t")})
        polys = [SUM, {(1, 0, 0): 1, (0, 1, 0): -1}, {(2, 0, 0): 1, (0, 1, 0): -2, (0, 0, 0): 3}]
        tr = mcons.track(s, ["x", "y", "s"], degree=1, identities=mi.candidates(s, ["x", "y", "s"], polys))
        k = 40
        dx = [int(a) for a in self.rng.integers(-30, 31, k)]
        dy = [int(a) for a in self.rng.integers(-30, 31, k)]
        ds = [a + b if i % 5 else a + b + 1 for i, (a, b) in enumerate(zip(dx, dy))]
        dv = {"x": [None if i == 7 else a for i, a in enumerate(dx)], "y": dy, "s": ds, "t": [0] * k}
        w = [int(a) for a in self.rng.choice([1, 1, 1, 2, -1, 3], k)]
        valid = {"x": np.array([a is not None for a in dv["x"]])}
        d = md.Delta.from_arrays({c: np.array([0 if a is None else a for a in dv[c]], np.int64) for c in dv},
                                 valid=valid, weights=w)
        got = tr.evaluate(d, backend="cpu")
        self.assertEqual(strip(got), strip(tr.evaluate(d, backend="cuda")))
        for i, poly in enumerate(polys):
            want = dict(checked_ins=0, checked_del=0, excluded_ins=0, excluded_del=0, violations_ins=0,
                        violations_del=0, overflow_ins=0, overflow_del=0, witness_ins=None, witness_del=None)
            for r in range(k):
                side = "ins" if w[r] > 0 else "del"
                pt = [dv[c][r] for c in ("x", "y", "s")]
                if any(p is None for p in pt):
                    want["excluded_" + side] += abs(w[r])
                    continue
                want["checked_" + side] += abs(w[r])
                if evaluate(poly, pt):
                    want["violations_" + side] += abs(w[r])
                    key = "witness_" + side
                    want[key] = r if want[key] is None else min(want[key], r)
            self.assertEqual(got["evidence"][i], want, i)
        mom, _ = moments_oracle({"x": [a for a in dx], "y": dy, "s": ds}, ["x", "y", "s"], 1)
        plan_moments = list(tr.moments())
        want_inc = []
        for m in plan_moments:
            want_inc.append(sum(w[r] * mono(m, [dv["x"][r], dy[r], ds[r]]) for r in range(k) if dv["x"][r] is not None))
        self.assertEqual(got["moments"], want_inc)
        self.assertEqual((got["gram_ins"], got["gram_del"]),
                         (sum(a for r, a in enumerate(w) if a > 0 and dv["x"][r] is not None),
                          sum(-a for r, a in enumerate(w) if a < 0 and dv["x"][r] is not None)))

    # ---------------------------------------------------------- acceptance: conservation and witnesses
    def test_disc003_known_conserved_transformations_preserve_certified_quantities(self):
        # the symbolic check: x' = x + t, y' = y - t conserves q = x + y (and x + y + 2t' ... ) universally
        q = {(1, 0, 0): 1, (0, 1, 0): 1}
        T = {"x": {(1, 0, 0): 1, (0, 0, 1): 1}, "y": {(0, 1, 0): 1, (0, 0, 1): -1}}
        law = mcons.conserve(q, T, ["x", "y", "t"])
        self.assertEqual(compose(q, {0: T["x"], 1: T["y"]}, 3), {})
        self.assertTrue(law.conserved)
        self.assertEqual(law.residual, {})
        cert = law.certificate
        self.assertEqual((cert.kind, cert.info["law"]), ("law", "ring_identity"))
        self.assertEqual(cert.authorize_exact().kind, "law")             # universal: no scope needed
        self.assertIsNone(law.counterexample)
        # a quadratic invariant of a rotation-like integer map: x' = y, y' = -x conserves x^2 + y^2
        q2 = {(2, 0, 0): 1, (0, 2, 0): 1}
        T2 = {"x": {(0, 1, 0): 1}, "y": {(1, 0, 0): -1}}
        law2 = mcons.conserve(q2, T2, ["x", "y", "t"])
        self.assertTrue(law2.conserved and law2.certificate is not None)
        self.assertEqual(compose(q2, {0: T2["x"], 1: T2["y"]}, 3), {})
        # on the data: the certified quantity s = x + y survives the transformation applied to the rows
        n = 200
        v = self.planted(n)
        s0 = arrays({c: v[c] for c in ("x", "y", "s", "t")})
        tr = mcons.track(s0, ["x", "y", "s"], degree=1, identities=mi.candidates(s0, ["x", "y", "s"], [SUM]))
        self.assertEqual(tr.record(0)["status"], "certified")
        moved = {"x": [a + b for a, b in zip(v["x"], v["t"])], "y": [a - b for a, b in zip(v["y"], v["t"])],
                 "s": v["s"], "t": v["t"]}
        s1 = arrays(moved)
        for backend in ("cpu", "cuda"):
            t1 = tr.update(md.Delta.between(s0, s1), scope=s1, backend=backend)
            r = t1.record(0)
            self.assertEqual((r["status"], r["certified"], r["runs"], r["streak"]), ("certified", 2, [(0, 1)], 2))
            c1 = t1.certificate(0)
            self.assertTrue(c1.fresh(s1).fresh)
            self.assertEqual(c1.authorize_exact(s1).kind, "snapshot")
            self.assertEqual(duck_violations(moved, "x + y - s <> 0"), 0)
        # the conservation check on before/after columns of the same rows: a snapshot certificate
        pair = arrays({"x": v["x"], "y": v["y"], "x2": moved["x"], "y2": moved["y"]})
        obs = mcons.conserve({(1, 0, 0, 0): 1, (0, 1, 0, 0): 1}, {"x": {(0, 0, 1, 0): 1}, "y": {(0, 0, 0, 1): 1}},
                             ["x", "y", "x2", "y2"])
        self.assertFalse(obs.conserved)                                    # not universal: it depends on the data
        self.assertEqual(obs.residual, compose({(1, 0, 0, 0): 1, (0, 1, 0, 0): 1},
                                               {0: {(0, 0, 1, 0): 1}, 1: {(0, 0, 0, 1): 1}}, 4))
        res = obs.check(pair, {"x": "x", "y": "y", "x2": "x2", "y2": "y2"}, backend="cuda")
        self.assertTrue(res["holds"])
        self.assertEqual(res["certificate"].kind, "snapshot")
        self.assertEqual((res["witnesses"], res["label"]), ([], "snapshot"))
        self.assertTrue(res["certificate"].fresh(pair).fresh)

    def test_disc003_perturbations_create_witnesses(self):
        n = 180
        v = self.planted(n)
        s0 = arrays({c: v[c] for c in ("x", "y", "s", "t")})
        tr = mcons.track(s0, ["x", "y", "s"], degree=1, identities=mi.candidates(s0, ["x", "y", "s"], [SUM]))
        moved = {"x": [a + b for a, b in zip(v["x"], v["t"])], "y": [a - b for a, b in zip(v["y"], v["t"])],
                 "s": v["s"], "t": v["t"]}
        s1 = arrays(moved)
        tr = tr.update(md.Delta.between(s0, s1), scope=s1)
        bad = 53
        pert = {c: list(moved[c]) for c in moved}
        pert["x"][bad] += 1                                                  # one perturbed row
        s2 = arrays(pert)
        for backend in ("cpu", "cuda"):
            t2 = tr.update(md.Delta.between(s1, s2), scope=s2, backend=backend)
            r = t2.record(0)
            self.assertEqual(r["status"], "failed")
            f = r["first_failure"]
            self.assertEqual((f["step"], f["source"], f["violations"]), (2, "inserted", 1))
            self.assertEqual(f["values"], {"x": pert["x"][bad], "y": pert["y"][bad], "s": pert["s"][bad]})
            self.assertEqual(r["runs"], [(0, 1)])                            # the record keeps what held
            self.assertIsNone(t2.certificate(0))
            self.assertEqual(t2.locate(0, s2)["witness"], bad)               # the physical row, by DISC-002
            self.assertEqual(t2.authorize(0, s2)["reason"], "failed")
        self.assertEqual(duck_violations(pert, "x + y - s <> 0"), 1)
        # the before/after check names the perturbed row; DuckDB and Python agree
        pair = arrays({"x": v["x"], "y": v["y"], "x2": pert["x"], "y2": pert["y"]})
        obs = mcons.conserve({(1, 0, 0, 0): 1, (0, 1, 0, 0): 1}, {"x": {(0, 0, 1, 0): 1}, "y": {(0, 0, 0, 1): 1}},
                             ["x", "y", "x2", "y2"])
        res = obs.check(pair, {"x": "x", "y": "y", "x2": "x2", "y2": "y2"})
        self.assertFalse(res["holds"])
        self.assertEqual((res["witnesses"], res["evidence"]["witness"], res["certificate"]), ([bad], bad, None))
        # a transformation that does not conserve q: x' = x + t, y' = y + t (residual 2t), counterexample and rows
        q = {(1, 0, 0): 1, (0, 1, 0): 1}
        T = {"x": {(1, 0, 0): 1, (0, 0, 1): 1}, "y": {(0, 1, 0): 1, (0, 0, 1): 1}}
        law = mcons.conserve(q, T, ["x", "y", "t"])
        want = compose(q, {0: T["x"], 1: T["y"]}, 3)
        self.assertEqual(law.residual, want)
        self.assertEqual(want, {(0, 0, 1): 2})
        ce = law.counterexample
        pt = [ce["point"][k] for k in ("x", "y", "t")]
        self.assertEqual(ce["value"], evaluate(want, pt))
        self.assertNotEqual(ce["value"], 0)
        res = law.check(s0, {"x": "x", "y": "y", "t": "t"}, witnesses=1000)
        self.assertEqual(res["witnesses"], [r for r in range(n) if v["t"][r] != 0])
        self.assertEqual(res["violations"], duck_violations({"t": v["t"]}, "2 * t <> 0"))
        # nonlinear: q = x y under x' = x + t, y' = y - t: residual t (y - x - t); the counterexample is exact
        q2 = {(1, 1, 0): 1}
        law2 = mcons.conserve(q2, {"x": T["x"], "y": {(0, 1, 0): 1, (0, 0, 1): -1}}, ["x", "y", "t"])
        want2 = compose(q2, {0: T["x"], 1: {(0, 1, 0): 1, (0, 0, 1): -1}}, 3)
        self.assertEqual(law2.residual, want2)
        ce2 = law2.counterexample
        self.assertEqual(ce2["value"], evaluate(want2, [ce2["point"][k] for k in ("x", "y", "t")]))
        self.assertNotEqual(ce2["value"], 0)
        # a higher-degree residual: the grid search still finds an exact non-root
        q3 = {(3, 0, 0): 1, (0, 2, 0): 2}
        T3 = {"x": {(1, 0, 0): 1, (0, 0, 2): 1}, "y": {(0, 1, 0): 1, (1, 0, 0): 1}}
        law3 = mcons.conserve(q3, T3, ["x", "y", "t"])
        want3 = compose(q3, {0: T3["x"], 1: T3["y"]}, 3)
        self.assertEqual(law3.residual, want3)
        ce3 = law3.counterexample
        self.assertEqual(ce3["value"], evaluate(want3, [ce3["point"][k] for k in ("x", "y", "t")]))
        self.assertNotEqual(ce3["value"], 0)

    # ---------------------------------------------------------- acceptance: separate evidence classes
    def test_disc003_schema_stability_and_hypotheses_remain_separate_classes(self):
        n = 150
        v = self.planted(n)
        v["q"][9] += 1                                                       # x^2 = q fails at one row
        db = make({c: v[c] for c in ("x", "y", "s", "q")})
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "y", "s", "q"])
        polys = [SUM, {(2, 0, 0, 0): 1, (0, 0, 0, 1): -1}]
        ids = mi.candidates(scope, ["x", "y", "s", "q"], [{(1, 0, 0, 0): 1, (0, 1, 0, 0): 1, (0, 0, 1, 0): -1},
                                                          polys[1]])
        tr = mcons.track(scope, ["x", "y", "s"], degree=1, identities=ids, declared=[False, True])
        # a declaration (schema guarantee) is not verified evidence: the declared identity failed and is not authorized
        c1 = tr.classes(1)
        self.assertEqual((c1["declared"], c1["snapshot"], c1["stability"], c1["status"]), (True, False, False, "failed"))
        self.assertEqual(tr.authorize(1, scope)["reason"], "failed")
        # the stable identity: snapshot at step 0; after an unbound revision only the record remains
        c0 = tr.classes(0)
        self.assertEqual((c0["declared"], c0["snapshot"], c0["stability"]), (False, True, True))
        self.assertTrue(tr.authorize(0, scope)["authorized"])
        d = md.Delta.from_arrays({"x": [2], "y": [3], "s": [5], "q": [4]})
        t1 = tr.update(d)
        r = t1.record(0)
        self.assertEqual((r["status"], r["certified"], r["held"], r["runs"]), ("held", 1, 1, [(0, 0)]))
        a = t1.authorize(0, scope)
        self.assertEqual((a["authorized"], a["reason"], a["cls"]), (False, "stability", "stability"))
        with self.assertRaises(DomainError):
            t1.gram(scope)                                                   # the unbound revision has no scope
        # a sampled hypothesis (DISC-002) is a hypothesis: counted apart, never a certified revision, never authorizing
        g = mi.gram(scope, mi.plan(scope, ["x", "y", "s", "q"], degree=1))
        hyp = mi.hypothesis(scope, g, ids, 0, tolerance=1e-9, size=40, seed=5)
        self.assertEqual(hyp.kind, "hypothesis")
        ch = t1.classes(0, given=[hyp])
        self.assertEqual((ch["hypothesis"], ch["hypothesis_size"], ch["certified"], ch["snapshot"]), (1, 40, 1, False))
        self.assertEqual(t1.record(0)["certified"], 1)
        self.assertEqual(t1.authorize(0, scope, given=[hyp])["reason"], "stability")
        bare = mcons.track(scope, ["x", "y", "s"], degree=1, identities=ids).update(
            md.Delta.from_arrays({"x": [1], "y": [1], "s": [3], "q": [1]}))      # makes identity 0 fail
        self.assertEqual(bare.authorize(0, scope, given=[hyp])["reason"], "failed")
        # an FND-007 schema fact about one of the identity's columns joins DECLARED only
        nn = mc.schema(scope, mc.not_null("x"), enforcement="load")
        cs = t1.classes(0, given=[nn, hyp])
        self.assertEqual((cs["declared"], cs["schema"], cs["hypothesis"], cs["stability"]), (True, 1, 1, True))
        self.assertEqual(t1.authorize(0, scope, given=[nn])["authorized"], False)
        # never converted: a foreign snapshot, a law, or a hypothesis of another claim is refused
        foreign = mc.snapshot(scope, ids.claim(0))
        with self.assertRaises(DomainError):
            t1.classes(0, given=[foreign])
        with self.assertRaises(DomainError):
            t1.classes(1, given=[hyp])
        with self.assertRaises(DomainError):
            t1.classes(0, given=[mc.law(mc.polynomial([(1, {"u": 1, "v": 1}), (-1, {"u": 1, "v": 1})], ["u", "v"]))])
        with self.assertRaises(DomainError):
            t1.classes(0, given=[mc.schema(scope, mc.not_null("q"), enforcement="load")])
        # observed stability is a record, not a proof about the next revision: a mutation makes the snapshot stale
        stable = tr
        t.flip(0, 0)
        a = stable.authorize(0, mbk.Scope.from_table(t, ["x", "y", "s", "q"]))
        self.assertEqual((a["authorized"], a["reason"]), (False, "snapshot"))
        self.assertIn("revision", a["snapshot"]["stale"].reasons)
        self.assertEqual(stable.record(0)["runs"], [(0, 0)])
        with self.assertRaises(StaleStateError):
            stable.gram(scope)

    # ---------------------------------------------------------- tables, groups, unbound revisions, rebind
    def test_disc003_table_mutations_groups_and_rebind(self):
        n = 120
        v = self.planted(n)
        v["k"] = [int(a) for a in self.rng.integers(0, 3, n)]
        db = make({c: v[c] for c in ("x", "y", "s", "k")})
        t = db["t"]
        names = ["x", "y", "s"]
        scope = mbk.Scope.from_table(t, names, group_by=["k"])
        ids = mi.candidates(scope, names, [SUM])
        tr = mcons.track(scope, names, degree=2, identities=ids)
        mask = np.ones(n, bool)
        mask[::7] = False
        s1 = mbk.Scope.from_table(t, names, group_by=["k"], rows=mask)       # the same version, another selection
        t1 = tr.update(md.Delta.between(scope, s1), scope=s1, backend="cuda")
        want, dom = moments_oracle(v, names, 2, mask)
        self.assertEqual((t1.moments(), t1.info["domain"]), (want, dom))
        self.assertEqual(t1.record(0)["status"], "certified")
        t.flip(0, 4)                                                          # a mutation: a new table revision
        s2 = mbk.Scope.from_table(t, names, group_by=["k"], rows=mask)
        d2 = md.Delta.between(s1, s2)
        self.assertGreater(d2.info["rows"], 0)
        t2 = t1.update(d2, scope=s2)
        self.assertTrue(t2.gram(s2) == mi.gram(s2, mi.plan(s2, names, degree=2)))
        self.assertEqual(t2.steps()[-1]["revision"], s2.revision)
        with self.assertRaises(DomainError):                                 # a DIFF from another version
            t2.update(md.Delta.between(s1, s2), scope=s2)
        located = t2.locate(0, s2)                                            # DISC-002's exhaustive verification
        r2 = t2.record(0)["evidence"]
        self.assertEqual((r2["checked"], r2["excluded"], r2["violations"]),
                         (located["checked"], located["excluded"], located["violations"]))
        self.assertEqual(t2.record(0)["status"], "certified" if located["holds"] else "failed")
        # unbound revisions: inserts, a weight change, a delete; then a rebind recomputes and binds again
        d3 = md.Delta.from_arrays({"x": [1, 2], "y": [2, 2], "s": [3, 4]}, groups=[0, 1], ngroups=s2.groups,
                                  weights=[2, 1])
        t3 = t2.update(d3, backend="cuda")
        same_tracker(self, t3, t2.update(d3))
        self.assertEqual(t3.record(0)["status"], "held" if located["holds"] else "failed")
        self.assertFalse(t3.bound)
        rb = t3.rebind(s2)
        self.assertEqual(rb.steps()[-1]["kind"], "rebind")
        self.assertTrue(rb.gram(s2) == mi.gram(s2, mi.plan(s2, names, degree=2)))
        self.assertEqual(rb.record(0)["status"], "certified" if located["holds"] else "failed")
        self.assertEqual(rb.record(0)["evidence"]["violations"], located["violations"])

    def test_disc003_deletes_of_absent_rows_and_incompatible_deltas_are_refused(self):
        v = self.planted(60)
        s0 = arrays({c: v[c] for c in ("x", "y", "s", "t")})
        tr = mcons.track(s0, ["x", "y", "s"], degree=1, identities=mi.candidates(s0, ["x", "y", "s"], [SUM]))
        with self.assertRaises(md.NegativeMultiplicityError):                # more deletes than rows (companion)
            tr.update(md.Delta.from_arrays({"x": [1] * 61, "y": [1] * 61, "s": [2] * 61, "t": [0] * 61},
                                           weights=[-1] * 61))
        with self.assertRaises(DomainError):                                 # another column layout
            tr.update(md.Delta.from_arrays({"x": [1], "y": [1], "s": [2]}))
        with self.assertRaises(DomainError) as cm:                           # deleting a violating row not there:
            tr.update(md.Delta.from_arrays({"x": [3], "y": [4], "s": [9], "t": [0]}, weights=[-1]))
        self.assertNotIsInstance(cm.exception, md.NegativeMultiplicityError)  # the evidence sees it, not the aggregates
        other = arrays({c: [a + 1 for a in v[c]] for c in ("x", "y", "s", "t")})
        s1 = arrays({c: v[c][:59] + [0] for c in ("x", "y", "s", "t")})
        with self.assertRaises(DomainError):                                 # a diff starting at another scope
            tr.update(md.Delta.between(other, s1), scope=s1)
        with self.assertRaises(ValueError):                                  # a diff needs the new scope
            tr.update(md.Delta.between(s0, s1))
        with self.assertRaises(ValueError):                                  # arrays deltas take none
            tr.update(md.Delta.from_arrays({"x": [1], "y": [1], "s": [2], "t": [0]}), scope=s1)
        # the update's proof: huge values in a degree-2 Gram leave 128 bits
        big = mcons.track(s0, ["x", "y", "s"], degree=2)
        with self.assertRaises(OverflowError):
            big.update(md.Delta.from_arrays({"x": [2 ** 62], "y": [2 ** 62], "s": [1], "t": [0]}))
        # an unproven plan (values near 2^62, degree 2) is refused at the build: the Gram must be exact
        huge = arrays({"x": [2 ** 61, 3, -2 ** 62], "y": [1, 2, 3], "s": [0, 0, 0], "t": [0, 0, 0]})
        with self.assertRaises(OverflowError):
            mcons.track(huge, ["x", "y"], degree=2)
        with self.assertRaises(UnsupportedError):
            mcons.track(arrays({f"c{i}": [1, 2] for i in range(9)}), [f"c{i}" for i in range(9)], degree=1)
        # undecided rows: an identity whose exact evaluation needs more than 8 words is never certified
        wide = arrays({"x": [2 ** 60, 3], "y": [1, 2], "s": [0, 0], "t": [0, 0]})
        tw = mcons.track(wide, ["y", "s"], degree=1, identities=mi.candidates(wide, ["x", "y"], [{(9, 0): 1, (0, 1): -1}]))
        self.assertEqual(tw.record(0)["status"], "undecided")
        self.assertEqual(tw.update(md.Delta.from_arrays({"x": [1], "y": [1], "s": [0], "t": [0]})).record(0)["status"],
                         "undecided")

    def test_disc003_device_results_are_checked_before_adoption(self):
        v = self.planted(70)
        s0 = arrays({c: v[c] for c in ("x", "y", "s", "t")})
        tr = mcons.track(s0, ["x", "y", "s"], degree=1, identities=mi.candidates(s0, ["x", "y", "s"], [SUM]))
        row = {c: v[c][0] for c in ("x", "y", "s", "t")}                    # a present row is deleted
        d = md.Delta.from_arrays({"x": [1, 2, row["x"]], "y": [1, 2, row["y"]], "s": [2, 4, row["s"]],
                                  "t": [0, 0, row["t"]]}, weights=[1, 1, -1])
        import ctypes as C
        job = tr._job(d, None)
        try:
            out, keep, label, device = tr._run(job, "cuda", None)
            keep[1][0].checked_ins += 1                                     # a tampered device counter
            err, ref, viol = C.c_int32(), mcons._Refusal(), md._Violation()
            h = mcons._call("tr_cons_tracker_update", tr._handle, job, C.byref(out), 1, None,
                            C.byref(mcons._budget(None)), C.byref(ref), C.byref(viol), C.byref(err))
            self.assertFalse(h)
            self.assertEqual(err.value, -1)
            keep[1][0].checked_ins -= 1
            h = mcons._call("tr_cons_tracker_update", tr._handle, job, C.byref(out), 1, None,
                            C.byref(mcons._budget(None)), C.byref(ref), C.byref(viol), C.byref(err))
            self.assertTrue(h)
            mcons._call("tr_cons_tracker_free", h)
        finally:
            mcons._call("tr_cons_job_free", job)

    # ---------------------------------------------------------- rediscovery on the maintained Gram
    def test_disc003_the_maintained_gram_proposes_the_new_revisions_identities(self):
        n = 150
        v = self.planted(n)
        s0 = arrays({c: v[c] for c in ("x", "y", "s", "q")})
        tr = mcons.track(s0, ["x", "y", "s", "q"], degree=1)
        changed = {c: list(v[c]) for c in ("x", "y", "s", "q")}
        changed["q"] = [3 * a - 2 * b for a, b in zip(v["x"], v["y"])]        # a new affine relation appears
        s1 = arrays(changed)
        t1 = tr.update(md.Delta.between(s0, s1), scope=s1, backend="cuda")
        g = t1.gram(s1)
        k = g.kernel()
        full = mi.gram(s1, mi.plan(s1, ["x", "y", "s", "q"], degree=1)).kernel()
        self.assertEqual((k.rank, k.free, k.vectors), (full.rank, full.free, full.vectors))
        texts = sorted(i["text"] for i in mi.identities(s1, g, k).certify())
        self.assertEqual(len(texts), 2)
        self.assertEqual(sorted(i["text"] for i in mi.discover(s1, ["x", "y", "s", "q"], degree=1).identities), texts)

    # ---------------------------------------------------------- budgets
    def test_disc003_budgets_at_the_limit_and_one_past(self):
        v = self.planted(50)
        s0 = arrays({c: v[c] for c in ("x", "y", "s", "t")})
        ids = mi.candidates(s0, ["x", "y", "s"], [SUM, {(1, 0, 0): 1, (0, 1, 0): -1}])

        def need(fn):
            """the smallest bytes limit that admits fn: raised refusal by refusal (each names its own need)"""
            limit = 1
            while True:
                try:
                    fn(Budget(bytes=limit))
                    return limit
                except AlgebraBudgetError as e:
                    self.assertEqual(e.resource, "bytes")
                    self.assertGreater(e.need, limit)
                    limit = e.need
        build = lambda b: mcons.track(s0, ["x", "y", "s"], degree=2, identities=ids, budget=b)
        nb = need(build)
        # the tracker's own bytes need is the first refusal; then DISC-002's and FND-004's own needs are smaller
        ok = build(Budget(bytes=nb))
        self.assertEqual(ok.record(0)["status"], "certified")
        with self.assertRaises(AlgebraBudgetError) as cm:
            build(Budget(bytes=nb - 1))
        self.assertEqual(cm.exception.resource, "bytes")
        states = ok.info["moments"] + len(ok)                                 # the moments (35) and the identities (2)
        self.assertEqual(states, 37)
        mcons.track(s0, ["x", "y", "s"], degree=2, identities=ids, budget=Budget(states=states))
        with self.assertRaises(AlgebraBudgetError) as cm:
            mcons.track(s0, ["x", "y", "s"], degree=2, identities=ids, budget=Budget(states=states - 1))
        self.assertEqual(cm.exception.resource, "states")
        tr = ok
        d = md.Delta.from_arrays({"x": [1], "y": [2], "s": [3], "t": [0]})
        need_job = tr.need(d)
        self.assertEqual(need_job["states"], states)
        tr.update(d, budget=Budget(states=states))
        with self.assertRaises(AlgebraBudgetError):
            tr.update(d, budget=Budget(states=states - 1))
        dev = tr.need(d, "cuda")
        tr.update(d, backend="cuda", budget=Budget(bytes=max(dev["bytes"], 1 << 20)))
        with self.assertRaises(AlgebraBudgetError) as cm:
            tr.evaluate(d, backend="cuda", budget=Budget(bytes=dev["bytes"] - 1))
        self.assertEqual(cm.exception.resource, "bytes")
        # conservation: the support budget of the expansion and the iterations of the counterexample grid
        q = {(2, 0): 1, (0, 2): 1}
        T = {"x": {(1, 0): 1, (0, 1): 1}, "y": {(1, 0): 1, (0, 1): -1}}      # (x+y)^2 + (x-y)^2 = 2x^2 + 2y^2
        law = mcons.conserve(q, T, ["x", "y"])
        self.assertEqual(law.residual, compose(q, {0: T["x"], 1: T["y"]}, 2))
        self.assertEqual(law.residual, {(2, 0): 1, (0, 2): 1})
        # support: the products (2 x 2 terms) and the accumulated expansion before merging: 3 + 3 terms
        accumulated = sum(len(p_mul(T["x"], T["x"])) for _ in range(1)) + len(p_mul(T["y"], T["y"]))
        self.assertEqual(accumulated, 6)
        mcons.conserve(q, T, ["x", "y"], budget=Budget(support=accumulated))
        with self.assertRaises(AlgebraBudgetError) as cm:
            mcons.conserve(q, T, ["x", "y"], budget=Budget(support=accumulated - 1))
        self.assertEqual((cm.exception.resource, cm.exception.need), ("support", accumulated))
        points = law.info["points"]
        self.assertGreater(points, 0)
        grid = 9                                                              # (2 + 1) x (2 + 1) points
        self.assertTrue(mcons.conserve(q, T, ["x", "y"], budget=Budget(iterations=grid)).info["searched"])
        self.assertFalse(mcons.conserve(q, T, ["x", "y"], budget=Budget(iterations=grid - 1)).info["searched"])


if __name__ == "__main__":
    unittest.main()
