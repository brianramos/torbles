# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SYM-001 / SYM-002 benchmark: basis actions on retained moment states against direct transformed-row evaluation,
and certified orbit reductions against exhaustive evaluation.

    python bench/benchmark_math_sym.py --capability SYM-001 --out FILE [--device] [--quick]

  basis actions    polynomial translations, declared 2D rotations (exact (3/5, 4/5) and a certified 30-degree
                   interval) and feature permutations on bases of 3 / 6 / 15 / 32 states (degree 1 over 2
                   coordinates, 2 over 2, 2 over 4, 1 over 31) of 4 groups, 20,000 rows (quick 3,000): the retained
                   state (an FND-004 bank, "build") and each action (M_T, "build") apart from its use ("run": every
                   transformed moment of every group, first and warm), against the direct transformed-row evaluation
                   (NumPy: the rows transformed in scaled exact int64, then their moments), checked equal first.
  requests         1 / 100 / 10,000 requests (transformed moments of one term of one group under one of 100
                   distinct actions) on the 15-state basis, CPU (and CUDA with --device) against the direct
                   evaluation of each request over the rows of its group (NumPy), checked equal first.
  orbits           certified orbit sizes 2 / 4 / 8 / 16 (a reflection, a rotation by 90 degrees, the dihedral group
                   D4, D4 x a reflection of a third coordinate) over 40,000 rows (quick 4,000) built as full orbits:
                   the support ("build"), the certification (CPU, and CUDA with --device), the orbit summary, and
                   invariant queries on the representatives ("run", first and warm) against the exhaustive native
                   evaluation over every row and DuckDB SQL, checked equal first; plus one broken-multiplicity
                   example (one row removed) that must refuse the certificate (the witness row recorded) and so any
                   orbit reuse, and the original certificate's staleness on the changed selection.
  thresholds       the action's tensor budget: the largest measured admitted and the smallest refused basis.
"""
from fractions import Fraction

import duckdb
import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_sym as ms
from torbles.math_budgets import Budget
from torbles.math_certificates import StaleCertificateError

F = Fraction


def bases(bench, rng):
    rows = 3000 if bench.quick else 20000
    G = 4
    for size, (n, d) in ((3, (2, 1)), (6, (2, 2)), (15, (4, 2)), (32, (31, 1))):
        names = [f"x{i}" for i in range(n)]
        X = rng.integers(-50, 51, (rows, n)).astype(np.int64)
        g = rng.integers(0, G, rows)
        scope = mbk.Scope.from_arrays({names[i]: X[:, i] for i in range(n)}, groups=g, ngroups=G)
        geo = ms.geometry(scope, numeric={k: "m" for k in names})
        B = ms.basis(geo, degree=d)
        assert len(B) == size
        terms = np.array(B.terms, dtype=np.int64)
        p = dict(states=size, coordinates=n, degree=d, rows=rows, groups=G)
        bench.measure("state (FND-004 bank)", "cpu", lambda: mbk.build(scope, B.terms_for(scope), "joint"), repeats=3,
                      phase="build", **p)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        cases = [("translation", ms.translate({names[0]: 3, names[-1]: -7}), None),
                 ("rotation (3/5, 4/5)", ms.rotate(names[0], names[1], cos=F(3, 5), sin=F(4, 5)), 5),
                 ("permutation", ms.permute({names[0]: names[1], names[1]: names[0]}), None),
                 ("rotation 30 degrees (interval)", ms.rotate(names[0], names[1], degrees=30), None)]
        for label, tr, den in cases:
            A = bench.measure("action M_T", "cpu", lambda: ms.action(B, tr, geo), repeats=3, phase="build",
                              transform=label, **p)
            bench.records[-1]["result"] = dict(carrier=A.info["carrier"], nonzeros=A.info["nonzeros"],
                                               permutation=A.info["permutation"])
            reqs = [dict(action=0, group=gg, term=j) for gg in range(G) for j in range(size)]
            for backend in bench.backends():
                bench.measure("transformed moments", backend, lambda: st.query(reqs, actions=[A], backend=backend),
                              repeats=5, phase="run", transform=label, requests=len(reqs), **p)

            def direct():                               # the rows transformed (scaled exact int64), then moments
                Y = X.copy()
                if label == "translation":
                    Y[:, 0] += 3
                    Y[:, -1] -= 7
                elif label.startswith("rotation (3/5"):
                    Y[:, 0], Y[:, 1] = 3 * X[:, 0] - 4 * X[:, 1], 4 * X[:, 0] + 3 * X[:, 1]
                elif label == "permutation":
                    Y[:, 0], Y[:, 1] = X[:, 1], X[:, 0]
                else:
                    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
                    Y = Y.astype(np.float64)
                    Y[:, 0], Y[:, 1] = c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]
                out = np.zeros((G, len(terms)), dtype=Y.dtype)
                for j, t in enumerate(terms):               # term by term over its nonzero exponents only
                    v = np.ones(len(Y), dtype=Y.dtype)
                    for i in np.nonzero(t)[0]:
                        v = v * Y[:, i] ** int(t[i])
                    for gg in range(G):
                        out[gg, j] = v[g == gg].sum()
                return out
            ref = direct()
            got = st.query(reqs, actions=[A])
            if A.info["carrier"] == "exact":
                deg = terms[:, 0] + terms[:, 1]             # the scaled (rotated) coordinates' degrees
                for r, q in zip(got, reqs):
                    want = Fraction(int(ref[q["group"], q["term"]]), (den or 1) ** int(deg[q["term"]]))
                    assert Fraction(r["value"]) == want, (label, q)
            else:
                for r, q in zip(got, reqs):
                    lo, hi = r["value"]
                    v = ref[q["group"], q["term"]]
                    assert lo - 1e-6 * max(1, abs(v)) <= v <= hi + 1e-6 * max(1, abs(v)), (label, q)
            bench.measure("direct transformed rows", "numpy", direct, repeats=3, phase="run", transform=label,
                          exact=A.info["carrier"] == "exact", **p)


def requests(bench, rng):
    rows = 3000 if bench.quick else 20000
    G, n = 4, 4
    names = [f"x{i}" for i in range(n)]
    X = rng.integers(-50, 51, (rows, n)).astype(np.int64)
    g = rng.integers(0, G, rows)
    scope = mbk.Scope.from_arrays({names[i]: X[:, i] for i in range(n)}, groups=g, ngroups=G)
    geo = ms.geometry(scope, numeric={k: "m" for k in names})
    B = ms.basis(geo, degree=2)
    st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
    shifts = [(int(a), int(b)) for a, b in rng.integers(-20, 21, (100, 2))]
    acts = bench.measure("100 actions (translations)", "cpu",
                         lambda: [ms.action(B, ms.translate({names[0]: a, names[2]: b}), geo) for a, b in shifts],
                         repeats=1, phase="build", actions=100, states=len(B))
    terms = np.array(B.terms, dtype=np.int64)
    groups = [X[g == gg] for gg in range(G)]
    for count in ((1, 100) if bench.quick else (1, 100, 10000)):
        reqs = [dict(action=i % 100, group=(i // 100) % G, term=(i * 7) % len(B)) for i in range(count)]
        p = dict(requests=count, states=len(B), rows=rows, actions=100)

        def direct():
            out = []
            for q in reqs:
                a, b = shifts[q["action"]]
                Y = groups[q["group"]].copy()
                Y[:, 0] += a
                Y[:, 2] += b
                out.append(int(np.prod(Y ** terms[q["term"]], axis=1).sum()))
            return out
        ref = direct()
        assert [r["value"] for r in st.query(reqs, actions=acts)] == ref
        for backend in bench.backends():
            bench.measure("requests", backend, lambda: st.query(reqs, actions=acts, backend=backend), repeats=3,
                          phase="run", **p)
        bench.measure("direct transformed rows", "numpy", direct, repeats=3 if count < 10000 else 1, phase="run", **p)


def d4(p):
    x, y = p[0], p[1]
    rest = tuple(p[2:])
    return [(u, v) + rest for u, v in ((x, y), (-y, x), (-x, -y), (y, -x), (-x, y), (y, x), (x, -y), (-y, -x))]


GROUPS = {
    2: (lambda p: [p, (-p[0], p[1], p[2])], lambda: [ms.reflect(x=0)]),
    4: (lambda p: [(u, v, p[2]) for u, v in ((p[0], p[1]), (-p[1], p[0]), (-p[0], -p[1]), (p[1], -p[0]))],
        lambda: [ms.rotate("x", "y", degrees=90)]),
    8: (d4, lambda: [ms.rotate("x", "y", degrees=90), ms.reflect(x=0)]),
    16: (lambda p: [q[:2] + (z,) for q in d4(p) for z in (p[2], 7 - p[2])],
         lambda: [ms.rotate("x", "y", degrees=90), ms.reflect(x=0), ms.reflect(z=F(7, 2))]),
}


def orbits(bench, rng):
    target = 4000 if bench.quick else 40000
    con = duckdb.connect()
    for size, (images, gens) in GROUPS.items():
        rows = []
        while len(rows) < target:                        # generic representatives: every orbit has the full size
            p = (int(rng.integers(1, 400)), int(rng.integers(1, 400)), int(rng.integers(-50, 50)))
            if p[0] == p[1] or 2 * p[2] == 7:
                continue
            orb = sorted(set(images(p)))
            if len(orb) != size:
                continue
            rows += orb * int(rng.integers(1, 4))
        rng.shuffle(rows)
        X = np.array(rows, dtype=np.int64)
        scope = mbk.Scope.from_arrays({"x": X[:, 0], "y": X[:, 1], "z": X[:, 2]})
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        G = ms.group(geo, gens())
        p = dict(orbit_size=size, rows=len(rows), generators=len(gens()))
        sp = bench.measure("support (sorted tuples)", "cpu", lambda: ms.support(scope, geo), repeats=3, phase="build",
                           **p)
        bench.records[-1]["result"] = dict(distinct=sp.info["distinct"])
        cert = None
        for backend in bench.backends():
            cert = bench.measure("certification", backend,
                                 lambda: ms.certify(scope, geo, G, backend=backend, support_=sp), repeats=3,
                                 phase="build", **p)
        orb = bench.measure("orbit summary", "cpu", lambda: cert.orbits(scope), repeats=3, phase="build", **p)
        bench.records[-1]["result"] = dict(orbits=orb.info["orbits"], largest=orb.info["largest"])
        r2 = {"x^2": 1, "y^2": 1}
        qs = [ms.total(r2), ms.count(r2, "<=", 200 ** 2), ms.total({"x^2*y^2": 1})]
        if size == 16:
            qs += [ms.total({"z^2": 1, "z": -7})]
        reduced = orb.query(qs)
        full = ms.evaluate(scope, geo, qs)
        assert [a["value"] for a in reduced] == [b["value"] for b in full]
        con.register("p", {"x": X[:, 0], "y": X[:, 1], "z": X[:, 2]})
        sql = ("SELECT SUM(CAST(x AS HUGEINT) * x + CAST(y AS HUGEINT) * y), "
               "COUNT(*) FILTER (WHERE CAST(x AS HUGEINT) * x + CAST(y AS HUGEINT) * y <= 40000), "
               "SUM(CAST(x AS HUGEINT) * x * y * y)" + (", SUM(CAST(z AS HUGEINT) * z - 7 * z)" if size == 16 else "")
               + " FROM p")
        duck = [int(v) for v in con.execute(sql).fetchall()[0]]
        assert duck == [a["value"] for a in reduced], (duck, reduced)
        q = dict(queries=len(qs), **p)
        bench.measure("invariant queries on representatives", "cpu", lambda: orb.query(qs), repeats=5, phase="run",
                      evaluations=reduced[0]["evaluations"], **q)
        bench.measure("exhaustive evaluation", "cpu", lambda: ms.evaluate(scope, geo, qs), repeats=5, phase="run",
                      evaluations=full[0]["evaluations"], **q)
        bench.measure("duckdb", "duckdb", lambda: con.execute(sql).fetchall(), repeats=5, phase="run", **q)
        con.unregister("p")
        if size == 8:                                    # one broken multiplicity must invalidate orbit reuse
            mask = np.ones(len(rows), bool)
            mask[0] = False
            broken = mbk.Scope.from_arrays({"x": X[:, 0], "y": X[:, 1], "z": X[:, 2]}, rows=mask)
            bgeo = ms.geometry(broken, numeric={"x": "m", "y": "m", "z": "m"})

            def attempt(backend):
                try:
                    ms.certify(broken, bgeo, gens(), backend=backend)
                except ms.SymmetryViolated as e:
                    return dict(refused=True, witness=e.witness, kind=e.kind, generator=e.generator,
                                violations=sum(x["violations"] for x in e.evidence))
                return dict(refused=False)
            for backend in bench.backends():
                r = bench.measure("broken multiplicity: certification", backend, lambda: attempt(backend), repeats=3,
                                  phase="refusal", removed_row=0, **p)
                assert r["refused"], r
            f = cert.fresh(broken)
            try:
                cert.orbits(broken)
                reuse = "allowed"
            except StaleCertificateError as e:
                reuse = f"refused ({', '.join(e.reasons)})"
            assert not f.fresh and reuse.startswith("refused")
            bench.note(f"orbit size 8: one row removed: the certificate is refused (witness row {r['witness']}, "
                       f"{r['kind']} under generator {r['generator']}); the original certificate on the changed "
                       f"selection is stale ({', '.join(f.reasons)}) and orbit reuse is {reuse}")


def thresholds(bench, rng):
    scope = mbk.Scope.from_arrays({f"x{i}": rng.integers(-5, 6, 50) for i in range(31)})
    geo = ms.geometry(scope, numeric={f"x{i}": "m" for i in range(31)})
    small = Budget(tensor=1024)
    b32 = ms.basis(geo, degree=1)                                      # 32 states: 1024 entries
    b36 = ms.basis(geo, degree=2, columns=[f"x{i}" for i in range(7)])  # 36 states: 1296 entries
    tr = ms.translate(x0=1)
    bench.measure("action within tensor=1024", "cpu", lambda: ms.action(b32, tr, geo, budget=small), repeats=1,
                  phase="build", states=32)
    rec = bench.refusal("action within tensor=1024", "cpu", lambda: ms.action(b36, tr, geo, budget=small), states=36)
    bench.threshold("action tensor budget 1024", admitted=dict(states=32, entries=1024),
                    refused=dict(states=36, entries=rec.get("need")), resource=rec.get("resource"))


def main():
    bench = Bench.from_args("SYM-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    bases(bench, rng)
    requests(bench, rng)
    orbits(bench, rng)
    thresholds(bench, rng)
    bench.note("exact answers are rationals (512-bit numerators over the action's denominator); the 30-degree rotation "
               "is a certified interval (binary64 endpoints), its NumPy baseline floating point; every baseline is "
               "checked equal (or enclosed) before it is timed")
    bench.write()


if __name__ == "__main__":
    main()
