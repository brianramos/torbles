# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEOM-002 benchmark: bounded elimination, generator selection and local elimination forms.

    python bench/benchmark_math_elim.py --capability GEOM-002 --out FILE [--device] [--quick]

Case 1, relation components with 2 / 4 / 8 variables (1 / 2 / 3 free columns, the others defined by polynomials of
degree 1 / 2 / 3), sparse (three terms) or dense (every monomial up to the degree) definitions, each phase timed on its
own: discovery (DISC-002: the Gram discovery when its basis fits 63 monomials, else the exhaustive certification of
the declared definitions), certificate checking (every relation re-verified by FND-007 on the rows; every
reconstruction's derived certificate D (x - f) = sum D h_k r_k built and checked by tr_cert_derive; the batched row
check of every reconstruction against its stored column: values, NULL mask, range; CUDA only with --device) and
reduction (the generator selection: one elimination order with the defined columns first, then the bounded search by
size under a fixed degree / support budget with its solves, undecided sets and minimality; the local elimination
forms of a query over the chosen generators).

Case 2, direct expressions against their local elimination forms and the normal form over the generators x, y
(s = x + y, z = x y, q = x^2 - y, w = x + 1): terms, columns, bound and actual bit widths (measured on every row), the
cost model's scan / work / total, the candidate tr_elim_forms_make keeps under each priority, and the scan itself
(DuckDB evaluating every form over the rows, the baseline). The term-explosion cases (w^6, s^4) and the cost
regressions (q w + s z, where no expansion is cheaper) are kept and every losing alternative is listed. Thresholds,
measured (never extrapolated): the expansion of s^n under a support budget (admitted, then rejected with the direct
form kept), the solve budget of a search (admitted at its need, refused one below) and the support at which a search
decides every set.
"""
import itertools

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles import math_elim as me
from torbles import math_geom as mg
from torbles import math_ident as mi
from torbles.math_budgets import Budget

FREE = {2: ["x"], 4: ["x", "y"], 8: ["x", "y", "z"]}
SEARCH_BUDGET = dict(degree=12, support=512)         # bounds every completion of the search (undecided past it)


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
            low = [e for e in monomials(len(free), degree) if 0 < sum(e) < degree] or [
                tuple(int(i == 0) for i in range(len(free)))]
            terms = {top[int(rng.integers(len(top)))]: 1, low[int(rng.integers(len(low)))]: int(rng.integers(-3, 4)) or 1,
                     (0,) * len(free): int(rng.integers(-9, 10))}
        out[f"u{j + 1}"] = terms
    return free, out


def evaluate(terms, cols):
    v = np.zeros(len(cols[0]), dtype=np.int64)
    for e, c in terms.items():
        t = np.full(len(cols[0]), int(c), dtype=np.int64)
        for col, p in zip(cols, e):
            if p:
                t = t * col ** p
        v = v + t
    return v


def table(data):
    frame = pd.DataFrame(data)
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in data) + ")"
    db = torbles.from_ddl(ddl, {"t": frame}, keep_plain=False)
    return db, frame


def relation(name, terms, free, names):
    """name - p(free) over names, as exponent tuples"""
    out = {tuple(int(c == name) for c in names): 1}
    for e, c in terms.items():
        key = tuple(dict(zip(free, e)).get(c2, 0) for c2 in names)
        out[key] = out.get(key, 0) - c
    return {e: c for e, c in out.items() if c}


def sql_of(p):
    """a Polynomial as a DuckDB expression over HUGEINT columns (its integer form D p)"""
    D = 1
    for c in p.terms.values():
        D = D * c.denominator // int(np.gcd(D, c.denominator))
    parts = []
    for e, c in p.terms.items():
        mono = " * ".join(f"CAST({n} AS HUGEINT)" for n, k in zip(p.columns, e) for _ in range(k)) or "1"
        parts.append(f"({int(c * D)}) * {mono}")
    return " + ".join(parts) or "0"


def duck_scan(con, p):
    return int(con.execute(f"SELECT sum(abs({sql_of(p)})) FROM frame").fetchone()[0] or 0)


# ------------------------------------------------------------------ case 1: components
def components(bench):
    rng = np.random.default_rng(bench.seed)
    rows = 2000 if bench.quick else 20_000
    grid = [(k, d, dense) for k in (2, 4, 8) for d in (1, 2, 3) for dense in (False, True)]
    for k, degree, dense in grid:
        free, defs = definitions(rng, k, degree, dense)
        data = {f: rng.integers(-20, 21, size=rows).astype(np.int64) for f in free}
        for name, terms in defs.items():
            data[name] = evaluate(terms, [data[f] for f in free])
        db, frame = table(data)
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
        certs = [i["certificate"] for i in mi.candidates(scope, names, rels).certify()]
        for backend in bench.backends():
            bench.measure("certificate check (relations re-verified on the rows)", backend,
                          lambda: len(mc.reverify(certs, scope, backend=backend)), repeats=3, phase="warm", **p)
        order = bench.measure("reduction (one elimination order)", "cpu",
                              lambda: me.select(scope, certs, columns=names, mode="order", order=list(defs) + free),
                              repeats=3, phase="build", **p)
        search = bench.measure("reduction (bounded search)", "cpu",
                               lambda: me.select(scope, certs, columns=names, budget=Budget(**SEARCH_BUDGET)),
                               repeats=1, phase="build", budget=SEARCH_BUDGET, **p)
        si = search.info
        bench.note(f"search {k} variables, degree {degree}, {p['certificate']}: generators {si['generators']} "
                   f"(order: {order.generators}), {si['solves']} solves, {si['undecided']} undecided "
                   f"(smallest size {si['undecided_size']}), minimal {si['minimal']}, largest degree {si['max_degree']}, "
                   f"terms {si['max_terms']}")
        elim = order.info["eliminated"]
        bench.measure("certificate check (reconstructions, FND-007 tr_cert_derive)", "cpu",
                      lambda: [order.certificate(x).kind for x in elim], repeats=3, phase="warm",
                      reconstructions=len(elim), **p)
        for backend in bench.backends():
            res = bench.measure("certificate check (reconstructions on every row)", backend,
                                lambda: order.check(backend=backend), repeats=3, phase="warm",
                                reconstructions=len(elim), **p)
            if not all(c["holds"] for c in res.values()):
                bench.note(f"UNEXPECTED: a reconstruction check failed on {k}/{degree}/{p['certificate']}")
        # a query over the dependent columns: the sum of their pairwise products and a linear term
        target = {}
        for a in range(len(elim)):
            for b in range(a, len(elim)):
                target[tuple(int(c == elim[a]) + int(c == elim[b]) for c in names)] = a + b + 1
        target[tuple(int(c == elim[-1]) for c in names)] = -7
        q = mg.poly(target, names)
        F = bench.measure("reduction (local elimination forms of a query)", "cpu",
                          lambda: me.forms(scope, order, q, max_eliminate=2), repeats=1, phase="build",
                          query_terms=len(target), **p)
        best = F.best
        bench.note(f"forms {k}/{degree}/{p['certificate']}: {F.info['candidates']} candidates, kept {best['kind']} "
                   f"{best['eliminated']} (total {best['cost']['total']}, direct "
                   f"{F.candidates[0]['cost']['total']})")


# ------------------------------------------------------------------ case 2: direct against elimination forms
def representations(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 5000 if bench.quick else 100_000
    x = rng.integers(-3000, 3001, size=rows).astype(np.int64)
    y = rng.integers(-3000, 3001, size=rows).astype(np.int64)
    data = {"x": x, "y": y, "s": x + y, "z": x * y, "q": x * x - y, "w": x + 1}
    db, frame = table(data)
    names = list(data)
    scope = mbk.Scope.from_table(db["t"], names)
    defs = {"s": {(1, 0): 1, (0, 1): 1}, "z": {(1, 1): 1}, "q": {(2, 0): 1, (0, 1): -1}, "w": {(1, 0): 1, (0, 0): 1}}
    rels = [relation(u, t, ["x", "y"], names) for u, t in defs.items()]
    certs = [i["certificate"] for i in mi.candidates(scope, names, rels).certify()]
    sel = bench.measure("generator selection (bounded search)", "cpu", lambda: me.select(scope, certs, columns=names),
                        repeats=3, phase="build", rows=rows)
    info = sel.info
    admissible = sorted((c for c in sel.candidates if c["status"] in ("chosen", "admissible")), key=lambda c: c["total"])
    bench.note(f"generators {info['generators']} of {names} ({info['solves']} solves, minimal {info['minimal']}); "
               f"storing every column costs {info['direct_storage']}; the admissible sets by total: " + "; ".join(
                   f"{c['generators']}: storage {c['storage']} + reconstruction {c['recon_scan'] + c['recon_work']} = "
                   f"{c['total']} ({c['status']})" for c in admissible[:6]))
    for col in sel.columns:
        if col["role"] == "eliminated":
            bench.note(f"column {col['name']}: scanning it stored costs {col['direct']}, reconstructing it "
                       f"{col['reconstruction']['total']} -> read {col['read']}")
    for backend in bench.backends():
        bench.measure("reconstruction check (every eliminated column, every row)", backend,
                      lambda: sel.check(backend=backend), repeats=3, phase="warm", rows=rows)
    P = lambda t: mg.poly(t, names)
    cases = {
        "s^2 - 2z (direct in generators)": P({(0, 0, 2, 0, 0, 0): 1, (0, 0, 0, 1, 0, 0): -2}),
        "x^2 + y^2 (direct in free columns)": P({(2, 0, 0, 0, 0, 0): 1, (0, 2, 0, 0, 0, 0): 1}),
        "z q - x^3 y (cancels)": P({(0, 0, 0, 1, 1, 0): 1, (3, 1, 0, 0, 0, 0): -1, (1, 2, 0, 0, 0, 0): -1}),
        "w^6 (term explosion)": P({(0, 0, 0, 0, 0, 6): 1}),
        "s^4 (term explosion)": P({(0, 0, 4, 0, 0, 0): 1}),
        "q w + s z (cost regression)": P({(0, 0, 0, 0, 1, 1): 1, (0, 0, 1, 1, 0, 0): 1}),
        "x y - z + 3 w (local beats normal)": P({(1, 1, 0, 0, 0, 0): 1, (0, 0, 0, 1, 0, 0): -1, (0, 0, 0, 0, 0, 1): 3}),
    }
    con = duckdb.connect()
    con.register("frame", frame)
    losing = []
    for label, f in cases.items():
        F = bench.measure("forms (build, certify, check every row, cost)", "cpu", lambda: me.forms(scope, sel, f),
                          repeats=1, phase="build", case_name=label, rows=rows)
        cands = F.candidates
        for priority in ("total", "bits", "scan", "work"):
            G = me.forms(scope, sel, f, priority=priority) if priority != "total" else F
            b = G.best
            bench.note(f"{label} [{priority}]: kept {b['kind']} {b['eliminated']} of {len(cands)} candidates")
        for c in cands:
            if c["polynomial"] is None or c["cost"] is None:
                bench.note(f"{label}: candidate {c['index']} ({c['kind']} {c['eliminated']}) {c['status']}: "
                           f"{c['reason']} {c['refusal'] or ''}")
                continue
            k = c["cost"]
            bench.measure("scan (DuckDB evaluation of the form)", "duckdb", lambda: duck_scan(con, c["polynomial"]),
                          repeats=3, phase="warm", case_name=label, form=f"{c['kind']} {c['eliminated']}",
                          terms=k["terms"], columns=k["columns"], bound_bits=k["bound_bits"],
                          actual_bits=k["actual_bits"], words=k["words"], scan=k["scan"], work=k["work"],
                          total=k["total"], status=c["status"], reason=c["reason"], rows=rows)
            if c["index"] and c["status"] != "chosen":
                losing.append(f"{label}: {c['kind']} {c['eliminated']} total {k['total']} (bits {k['actual_bits']}, "
                              f"scan {k['scan']}, work {k['work']}) against the kept {F.best['kind']} "
                              f"{F.best['eliminated']} total {F.best['cost']['total']}")
        if F.best["kind"] == "direct":
            bench.note(f"{label}: expansion loses, the direct form is kept (total {F.best['cost']['total']})")
    bench.note("losing alternatives (retained): " + " | ".join(losing))
    # the expansion of s^n under a support budget: admitted, then rejected (the direct form kept), measured
    small_rows = 2000
    sdb, _ = table({k: v[:small_rows] for k, v in data.items()})
    small = mbk.Scope.from_table(sdb["t"], names)
    scerts = [i["certificate"] for i in mi.candidates(small, names, rels).certify()]
    ssel = me.select(small, scerts, columns=names)
    lim = Budget(support=16)
    admitted = None
    for n in range(2, 40):
        F = me.forms(small, ssel, P({(0, 0, n, 0, 0, 0): 1}), budget=lim)
        local = [c for c in F.candidates if c["kind"] == "local"]
        rejected = [c for c in local if c["reason"] == "budget"]
        if rejected:
            r = rejected[0]["refusal"]
            bench.threshold("local elimination of s^n -> (x + y)^n under support 16", admitted=admitted,
                            refused=dict(exponent=n, resource=r["resource"], need=r["need"], limit=r["limit"],
                                         kept=F.best["kind"]), rows=small_rows)
            break
        admitted = dict(exponent=n, terms=len(local[0]["polynomial"]), kept=F.best["kind"])
    # the solve budget of a search: admitted at its need, refused one below
    need = info["solves"]
    bench.measure("search at its solve need", "cpu",
                  lambda: me.select(small, scerts, columns=names, budget=Budget(iterations=need)).generators,
                  repeats=1, phase="build", iterations=need)
    rec = bench.refusal("search one solve short", "cpu",
                        lambda: me.select(small, scerts, columns=names, budget=Budget(iterations=need - 1)),
                        iterations=need - 1)
    bench.threshold("search solves (one completion per generator set)", admitted=dict(iterations=need),
                    refused=dict(iterations=need - 1, need=rec.get("need"), resource=rec.get("resource")))
    # the support at which the search decides every set (below it some sets are undecided, never guessed)
    limit, last = 1, None
    while True:
        s = me.select(small, scerts, columns=names, budget=Budget(support=limit))
        if not s.info["undecided"]:
            break
        last = dict(support=limit, undecided=s.info["undecided"], generators=s.generators, minimal=s.info["minimal"])
        limit += 1
    bench.threshold("search support deciding every generator set", admitted=dict(support=limit, generators=s.generators,
                                                                                 minimal=s.info["minimal"]),
                    refused=last)


def main():
    bench = Bench.from_args("GEOM-002", __doc__)
    components(bench)
    representations(bench)
    bench.note("Relations are DISC-002 certificates (exhaustive verification on the rows, FND-007 snapshots). Each "
               "reconstruction is the normal form under a lex order with the eliminated columns first (GEOM-001), its "
               "certificate derived and checked by FND-007, and it is kept only when its row check (values, NULL mask, "
               "range) holds. Costs are relative work units of the bit-plane engine from the actual bit widths "
               "measured on every row, not time; DuckDB times the same forms as the baseline.")
    bench.note("CUDA (tr_mgpu_elim_check) is timed only with --device; emulated runs are compared with the CPU in "
               "tests/test_math_elim.py, never timed as CUDA.")
    bench.write()


if __name__ == "__main__":
    main()
