# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEOM-001 benchmark: certified sparse ideal reduction, its certificates and the choice of representation.

    python bench/benchmark_math_geom.py --capability GEOM-001 --out FILE [--device] [--quick]

Relation components with 2 / 4 / 8 variables (1 / 2 / 3 free columns, the others defined by polynomials of degree
1 / 2 / 3), sparse (three terms) or dense (every monomial up to the degree) definitions. Reported separately:
discovery (DISC-002: the Gram discovery when its basis fits 63 monomials, else the exhaustive verification and
certification of the declared definitions), the ideal (Buchberger's completion), the reduction of a target, and the
certificate checks (FND-007 tr_cert_derive, the batched dense check on the CPU, CUDA with --device). Then direct
expressions against their elimination form (lex, defined columns eliminated) and their generator form (deglex, free
monomials rewritten through the definitions): terms, columns scanned, bound and actual bit widths (measured on every
row), the representation tr_geom_choose picks, and the scan itself (DuckDB evaluating each form over the rows, the
baseline). Term explosion ((x + 1)^n, (x + y + 1)^n) and cost regressions (a reduction that is no cheaper) are kept;
refusals of the completion and of the support budget are recorded as thresholds, never extrapolated.
"""
import itertools

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_geom as mg
from torbles import math_ident as mi
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError

FREE = {2: ["x"], 4: ["x", "y"], 8: ["x", "y", "z"]}


def monomials(nv, degree):
    return [e for e in itertools.product(range(degree + 1), repeat=nv) if sum(e) <= degree]


def definitions(rng, k, degree, dense):
    """the defining polynomials of the dependent columns over the free ones: {name: {exponents: coefficient}}"""
    free = FREE[k]
    out = {}
    for j in range(k - len(free)):
        if dense:
            terms = {e: int(rng.choice([-3, -2, -1, 1, 2, 3])) for e in monomials(len(free), degree)}
        else:
            top = [e for e in monomials(len(free), degree) if sum(e) == degree]
            low = [e for e in monomials(len(free), degree) if 0 < sum(e) < degree] or [tuple(int(i == 0) for i in range(len(free)))]
            terms = {top[int(rng.integers(len(top)))]: 1, low[int(rng.integers(len(low)))]: int(rng.integers(-3, 4)) or 1,
                     (0,) * len(free): int(rng.integers(-9, 10))}
        out[f"u{j + 1}"] = terms
    return free, out


def evaluate(terms, cols):
    """the defining polynomial on the rows (int64 arrays; the data keeps every value far inside int64)"""
    v = np.zeros(len(cols[0]), dtype=np.int64)
    for e, c in terms.items():
        t = np.full(len(cols[0]), int(c), dtype=np.int64)
        for col, p in zip(cols, e):
            if p:
                t = t * col ** p
        v = v + t
    return v


def component(rng, k, degree, dense, rows):
    free, defs = definitions(rng, k, degree, dense)
    data = {f: rng.integers(-20, 21, size=rows).astype(np.int64) for f in free}
    for name, terms in defs.items():
        data[name] = evaluate(terms, [data[f] for f in free]).astype(np.int64)
    frame = pd.DataFrame(data)
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in data) + ")"
    db = torbles.from_ddl(ddl, {"t": frame}, keep_plain=False)
    return db, frame, free, defs


def relation(name, terms, free, names):
    """u - p(free) over names, as exponent tuples"""
    out = {tuple(int(c == name) for c in names): 1}
    for e, c in terms.items():
        key = tuple(dict(zip(free, e)).get(c2, 0) for c2 in names)
        out[key] = out.get(key, 0) - c
    return {e: c for e, c in out.items() if c}


def sql_of(p):
    """a Polynomial as a DuckDB expression over HUGEINT columns (integer form: D p)"""
    D = 1
    for c in p.terms.values():
        D = D * c.denominator // np.gcd(D, c.denominator)
    parts = []
    for e, c in p.terms.items():
        mono = " * ".join(f"CAST({n} AS HUGEINT)" for n, k in zip(p.columns, e) for _ in range(k)) or "1"
        parts.append(f"({int(c * D)}) * {mono}")
    return " + ".join(parts) or "0"


def duck_scan(frame, p):
    con = duckdb.connect()
    con.register("frame", frame)
    return int(con.execute(f"SELECT sum(abs({sql_of(p)})) FROM frame").fetchone()[0] or 0)


def components(bench):
    rng = np.random.default_rng(bench.seed)
    rows = 2000 if bench.quick else 20_000
    for k in (2, 4, 8):
        for degree in (1, 2, 3):
            for dense in (False, True):
                db, frame, free, defs = component(rng, k, degree, dense, rows)
                names = list(frame)
                scope = mbk.Scope.from_table(db["t"], names)
                p = dict(variables=k, degree=degree, certificate="dense" if dense else "sparse", rows=rows)
                rels = [relation(u, t, free, names) for u, t in defs.items()]
                basis = len(monomials(k, degree))

                def discover():
                    if basis <= mi.MAX_BASIS:
                        found = mi.discover(scope, names, degree=degree)
                        return dict(method="gram", discoveries=len(found.discoveries()), basis=basis)
                    ids = mi.candidates(scope, names, rels).certify()
                    return dict(method="declared (basis past 63)", verified=sum(i["status"] == "verified" for i in ids),
                                basis=basis)
                bench.measure("discovery", "cpu", discover, repeats=1, phase="first", **p)
                ids = mi.candidates(scope, names, rels).certify()
                certs = [i["certificate"] for i in ids]
                order = dict(order="lex", variables=list(defs) + free)               # eliminate the defined columns
                I = bench.measure("ideal (completion)", "cpu", lambda: mg.ideal(certs, **order), repeats=3,
                                  phase="build", **p)
                info = I.info
                target = {}
                us = list(defs)
                for a in range(len(us)):
                    for b in range(a, len(us)):
                        target[tuple(int(c == us[a]) + int(c == us[b]) for c in names)] = a + b + 1
                target[tuple(int(c == us[-1]) for c in names)] = -7
                f = mg.poly(target, names)
                red = bench.measure("reduction", "cpu", lambda: I.reduce(f), repeats=3, phase="first",
                                    basis_elements=info["basis"], pairs=info["pairs"], **p)
                ri = red.info
                P, cof = red.claims()
                sizes = dict(P_terms=len(P.params["coefficients"]), cofactor_terms=sum(len(q.params["coefficients"]) for q in cof),
                             g_terms=ri["g_terms"])
                bench.measure("certificate check (FND-007 tr_cert_derive)", "cpu", lambda: red.certify().kind, repeats=3,
                              phase="warm", **dict(p, **sizes))
                for backend in bench.backends():
                    bench.measure("certificate check (batched dense)", backend,
                                  lambda: mg.check([red], backend=backend)[0]["holds"], repeats=3, phase="warm",
                                  **dict(p, **sizes))
                for backend in bench.backends():
                    bench.measure("agreement on every row", backend,
                                  lambda: mg.agree(scope, [(f, red.g)], backend=backend)[0]["disagreements"], repeats=3,
                                  phase="warm", **p)
                diff = mg.poly(_difference(f, red.g, names), names)
                bench.measure("agreement on every row", "duckdb", lambda: duck_scan(frame, diff), repeats=3,
                              phase="warm", note="sum |D (f - g)| over the rows (0 when they agree)", **p)


def _difference(f, g, names):
    a, b = f.aligned(names), g.aligned(names)
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, 0) - c
    return {e: c for e, c in out.items() if c}


def representations(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 5000 if bench.quick else 100_000
    x = rng.integers(-3000, 3001, size=rows)
    y = rng.integers(-3000, 3001, size=rows)
    data = {"x": x, "y": y, "s": x + y, "z": x * y, "q": x * x - y, "w": x + 1}
    frame = pd.DataFrame(data)
    names = list(data)
    db = torbles.from_ddl("CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in names) + ")", {"t": frame},
                          keep_plain=False)
    scope = mbk.Scope.from_table(db["t"], names)
    defs = {"s": {(1, 0): 1, (0, 1): 1}, "z": {(1, 1): 1}, "q": {(2, 0): 1, (0, 1): -1}, "w": {(1, 0): 1, (0, 0): 1}}
    rels = [relation(u, t, ["x", "y"], names) for u, t in defs.items()]
    certs = [i["certificate"] for i in mi.candidates(scope, names, rels).certify()]
    elim = mg.ideal(certs, order="lex", variables=["s", "z", "q", "w", "x", "y"])          # into x, y
    gen = mg.ideal(certs, order="deglex", variables=["x", "y", "s", "z", "q", "w"])         # into the generators
    cases = {
        "s^2 - 2z (direct in generators)": mg.poly({(0, 0, 2, 0, 0, 0): 1, (0, 0, 0, 1, 0, 0): -2}, names),
        "x^2 + y^2 (direct in free columns)": mg.poly({(2, 0, 0, 0, 0, 0): 1, (0, 2, 0, 0, 0, 0): 1}, names),
        "z q - x^3 y (cancels)": mg.poly({(0, 0, 0, 1, 1, 0): 1, (3, 1, 0, 0, 0, 0): -1, (1, 2, 0, 0, 0, 0): -1}, names),
        "w^6 (term explosion)": mg.poly({(0, 0, 0, 0, 0, 6): 1}, names),
        "s^4 (term explosion)": mg.poly({(0, 0, 4, 0, 0, 0): 1}, names),
        "q w + s z (cost regression)": mg.poly({(0, 0, 0, 0, 1, 1): 1, (0, 0, 1, 1, 0, 0): 1}, names),
    }
    for label, f in cases.items():
        forms = {"direct": f, "elimination": elim.reduce(f).g, "generators": gen.reduce(f).g}
        cands = list(forms.values())
        for priority in ("columns", "bits", "terms"):
            idx, costs = bench.measure("choose", "cpu", lambda: mg.choose(scope, cands, priority=priority, measure=True),
                                       repeats=1, phase="first", case_name=label, priority=priority)
            bench.note(f"{label} [{priority}]: chose {list(forms)[idx]}; " + "; ".join(
                f"{n}: {c['terms']} terms, {c['columns']} columns, bound {c['bound_bits']} bits, actual "
                f"{c['actual_bits']} bits, agrees {c['agrees']}" for n, c in zip(forms, costs)))
        for name, rep in forms.items():
            c = mg.cost(scope, rep)
            bench.measure("scan (DuckDB evaluation of the form)", "duckdb", lambda: duck_scan(frame, rep), repeats=3,
                          phase="warm", case_name=label, form=name, terms=c["terms"], columns=c["columns"],
                          bound_bits=c["bound_bits"], rows=rows)
    # term explosion against the support budget: the largest admitted and the smallest refused exponent
    lim = Budget(support=64)
    admitted = None
    for n in range(2, 40):
        f = mg.poly({(0, 0, n, 0, 0, 0): 1}, names)                                        # s^n -> (x + y)^n
        try:
            g = elim.reduce(f, budget=lim).g
            admitted = dict(exponent=n, terms=len(g))
        except AlgebraBudgetError as e:
            bench.refusal("term explosion (support 64)", "cpu", lambda: elim.reduce(f, budget=lim), exponent=n)
            bench.threshold("term explosion (x + y)^n under support 64", admitted=admitted,
                            refused=dict(exponent=n, need=e.need, resource=e.resource))
            break


def completion_budget(bench):
    rng = np.random.default_rng(bench.seed + 2)
    rows = 1000
    x = rng.integers(-30, 31, size=rows)
    data = {"x": x, "y": x * x, "z": x ** 3, "v": x ** 4}
    frame = pd.DataFrame(data)
    names = list(data)
    db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, z BIGINT, v BIGINT)", {"t": frame}, keep_plain=False)
    scope = mbk.Scope.from_table(db["t"], names)
    rels = [relation("y", {(2,): 1}, ["x"], names), relation("z", {(3,): 1}, ["x"], names),
            relation("v", {(4,): 1}, ["x"], names)]
    certs = [i["certificate"] for i in mi.candidates(scope, names, rels).certify()]
    spec = dict(order="degrevlex", variables=["x", "y", "z", "v"])
    I = bench.measure("completion (rational normal curve)", "cpu", lambda: mg.ideal(certs, **spec), repeats=3,
                      phase="build")
    info = I.info
    need = info["pairs"] + info["reductions"]
    bench.measure("completion at its iteration need", "cpu", lambda: mg.ideal(certs, budget=Budget(iterations=need),
                                                                              **spec).info["basis"],
                  repeats=1, phase="build", iterations=need)
    rec = bench.refusal("completion one iteration short", "cpu",
                        lambda: mg.ideal(certs, budget=Budget(iterations=need - 1), **spec), iterations=need - 1)
    bench.threshold("Buchberger iterations (S-pairs + division steps)", admitted=dict(iterations=need, basis=info["basis"]),
                    refused=dict(iterations=need - 1, need=rec.get("need"), resource=rec.get("resource")))


def main():
    bench = Bench.from_args("GEOM-001", __doc__)
    components(bench)
    representations(bench)
    completion_budget(bench)
    bench.note("The relations are DISC-002 certificates (exhaustive verification, FND-007 snapshots); every reduction "
               "certificate is checked by FND-007 (tr_cert_derive) and by the batched dense check (exact 128-bit slots "
               "under a proven bound). Coefficients are exact int64 rationals: an overflow is refused, never rounded.")
    bench.note("CUDA (tr_mgpu_geom_agree, tr_mgpu_geom_check) is timed only with --device; emulated runs are checked "
               "for equality in tests/test_math_geom.py, never timed as CUDA.")
    bench.write()


if __name__ == "__main__":
    main()
