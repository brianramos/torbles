# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-004 benchmark: data scopes and grouped exact moment banks, their cache (warm reuse, refresh, rejection of
incompatible requests), state bytes and eviction at bounded capacities.

    python bench/benchmark_math_bank.py --capability FND-004 --out FILE [--device] [--quick]

Build: scope construction (native decode of the table) and the bank (native CPU; CUDA with --device) over 100k / 1M
rows, 1 / 64 / 4096 groups and 6 / 15 terms (all monomials of degree <= 2 over 2 / 4 columns), against a DuckDB
GROUP BY baseline computing the same COUNT(*), COUNT(term) and SUM(term) in HUGEINT over the source table; each grid
point is checked equal to DuckDB once. Cache: a miss (build + put), warm hits (an independent copy), a warm hit with
the scope rebuilt, an explicit refresh (rebuild + replace), and requests that must miss (another predicate, NULL
policy, revision or group grain), each checked to miss. State bytes against groups x terms, with the states budget's
refusal threshold as measured. Eviction: cyclic access over 8 domains at capacities 1..16 and at byte limits.
"""
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles.math_budgets import Budget


def table(rng, rows):
    """BIGINT values with 5% NULLs and three group keys (1, 64 and 4096 distinct values)"""
    def nullable(v):
        s = pd.Series(v, dtype=object)
        s[rng.random(rows) < 0.05] = None
        return s
    frame = pd.DataFrame({"g1": np.zeros(rows, dtype=np.int64), "g64": rng.integers(0, 64, rows),
                          "g4096": rng.integers(0, 4096, rows),
                          "x": nullable(rng.integers(-10**6, 10**6, rows)), "y": nullable(rng.integers(-10**5, 10**5, rows)),
                          "z": nullable(rng.integers(-1000, 1000, rows)), "w": nullable(rng.integers(0, 10**6, rows))})
    ddl = "CREATE TABLE f (g1 BIGINT, g64 BIGINT, g4096 BIGINT, x BIGINT, y BIGINT, z BIGINT, w BIGINT)"
    return torbles.from_ddl(ddl, {"f": frame})


def duck_sql(names, terms, key):
    parts = ["COUNT(*)"]
    for t in terms:
        factors = [f'CAST("{n}" AS HUGEINT)' for n, e in zip(names, t) for _ in range(e)]
        expr = "(" + " * ".join(factors) + ")" if factors else "1"
        parts += [f"COUNT({expr})", f"SUM({expr})"]
    return f'SELECT "{key}", {", ".join(parts)} FROM f GROUP BY "{key}"'


def check_against_duckdb(db, bank, scope, names, terms, key):
    expect = {}
    for r in db.con.execute(duck_sql(names, terms, key)).fetchall():
        expect[(r[0],)] = (int(r[1]), [int(v) for v in r[2::2]], [0 if v is None else int(v) for v in r[3::2]])
    got = {row["key"]: (row["rows"], row["count"], row["sum"]) for row in bank.table()}
    if got != expect:
        raise SystemExit(f"FND-004 bank differs from DuckDB ({key}, {len(terms)} terms)")


def build_grid(bench, rng):
    for rows in ((100_000,) if bench.quick else (100_000, 1_000_000)):
        db = table(rng, rows)
        t = db["f"]
        for key in ("g1", "g64", "g4096"):
            for names in (["x", "y"], ["x", "y", "z", "w"]):
                terms = mbk.monomials(names, 2)
                p = dict(rows=rows, groups=int(key[1:]), terms=len(terms), columns=len(names))
                scope = bench.measure("scope", "cpu", lambda: mbk.Scope.from_table(t, names, group_by=[key]),
                                      repeats=3, phase="build", **p)
                bank = None
                for backend in bench.backends():
                    bank = bench.measure("bank", backend, lambda: mbk.build(scope, terms, backend=backend),
                                         repeats=3, phase="run", **p)
                bench.measure("bank", "duckdb", lambda: db.con.execute(duck_sql(names, terms, key)).fetchall(),
                              repeats=3, phase="run", **p)
                check_against_duckdb(db, mbk.build(scope, terms), scope, names, terms, key)
                bench.records.append(dict(case="state bytes", backend="cpu", phase="measure", params=p,
                                          bank_bytes=bank.bytes, scope_bytes=scope.bytes,
                                          state_bytes=scope.groups * (8 + 24 * len(terms))))
    return db


def cache_cases(bench, rng):
    rows = 100_000 if bench.quick else 1_000_000
    db = table(rng, rows)
    t = db["f"]
    names, terms = ["x", "y", "z", "w"], mbk.monomials(["x", "y", "z", "w"], 2)
    pred = np.asarray(t.decoded("g64")) < 32                     # a selection: half the groups
    p = dict(rows=rows, groups=64, terms=len(terms))
    cache = mbk.Cache(capacity=16, max_bytes=64 << 20)
    scope = mbk.Scope.from_table(t, names, group_by=["g64"], rows=pred)

    def miss_then_put():
        cache.clear()
        return cache.bank(scope, terms).cache_hit
    bench.measure("cache miss (build + put)", "cpu", miss_then_put, repeats=3, phase="first", **p)
    bench.measure("cache warm hit", "cpu", lambda: cache.get(scope, terms).cache_hit, repeats=20, phase="warm", **p)
    bench.measure("cache warm hit, scope rebuilt", "cpu",
                  lambda: cache.get(mbk.Scope.from_table(t, names, group_by=["g64"], rows=pred), terms).cache_hit,
                  repeats=5, phase="warm", **p)
    bench.measure("cache refresh (rebuild + replace)", "cpu", lambda: cache.refresh(scope, terms).rows,
                  repeats=5, phase="run", **p)
    bench.measure("direct build (no cache)", "cpu", lambda: mbk.build(scope, terms).rows, repeats=5, phase="run", **p)
    bench.measure("scope build", "cpu", lambda: mbk.Scope.from_table(t, names, group_by=["g64"], rows=pred).selected,
                  repeats=5, phase="build", **p)
    # incompatible requests: each must miss while the base entry is live
    other = pred.copy(); other[int(np.flatnonzero(pred)[0])] = False
    requests = {
        "predicate (one row differs)": (mbk.Scope.from_table(t, names, group_by=["g64"], rows=other), terms, "sql"),
        "NULL policy (listwise)": (scope, terms, "joint"),
        "group grain (4096 groups)": (mbk.Scope.from_table(t, names, group_by=["g4096"], rows=pred), terms, "sql"),
        "basis (a term less)": (scope, terms[:-1], "sql"),
    }
    for name, (sc, basis, mode) in requests.items():
        def reject():
            got = cache.get(sc, basis, mode)
            if got is not None:
                raise SystemExit(f"FND-004: an incompatible request hit the cache ({name})")
            return "miss"
        bench.measure("incompatible request", "cpu", reject, repeats=5, phase="warm", request=name, **p)
    t.flip(0, 0); t.flip(0, 0)                                   # a new revision (the data flipped back)
    newer = mbk.Scope.from_table(t, names, group_by=["g64"], rows=pred)

    def revision():
        if cache.get(newer, terms) is not None:
            raise SystemExit("FND-004: a new revision hit the cache")
        return "miss"
    bench.measure("incompatible request", "cpu", revision, repeats=5, phase="warm", request="revision", **p)

    def stale():
        try:
            cache.get(scope, terms)
        except mbk.StaleStateError:
            return "refused: stale"
        raise SystemExit("FND-004: a stale scope was answered")
    bench.measure("stale scope", "cpu", stale, repeats=5, phase="warm", **p)
    bench.measure("cache refresh after a mutation", "cpu", lambda: cache.refresh(newer, terms).rows, repeats=3,
                  phase="run", **p)
    bench.records.append(dict(case="cache stats", backend="cpu", phase="measure", params=p, stats=cache.stats))
    if bench.device:
        bench.measure("cache miss (build + put)", "cuda", lambda: (cache.clear(), cache.bank(newer, terms, backend="cuda"))[1].rows,
                      repeats=3, phase="first", **p)


def state_bytes(bench, rng):
    n = 20_000 if bench.quick else 100_000
    budget = Budget()
    last_ok = None
    for names in (["a"], ["a", "b"], ["a", "b", "c", "d"], [f"v{i}" for i in range(8)], [f"v{i}" for i in range(32)]):
        terms = mbk.monomials(names, 2)
        cols = {nm: rng.integers(-1000, 1000, n) for nm in names}
        for groups in (1, 16, 256, 4096, 65536):
            ids = rng.integers(0, groups, n) if groups > 1 else None
            scope = mbk.Scope.from_arrays(cols, groups=ids, ngroups=groups if groups > 1 else None)
            p = dict(rows=n, groups=groups, terms=len(terms), columns=len(names))
            states = groups * len(terms)
            if states > budget.limits["states"]:
                bench.refusal("state bytes", "cpu", lambda: mbk.build(scope, terms, budget=budget), states=states, **p)
                bench.threshold("bank states (groups x terms)", last_ok, dict(p, states=states),
                                budget_states=budget.limits["states"])
                continue
            bank = bench.measure("state bytes", "cpu", lambda: mbk.build(scope, terms, budget=budget), repeats=1,
                                 phase="build", **p)
            bench.records[-1].update(bank_bytes=bank.bytes, state_bytes=groups * (8 + 24 * len(terms)), states=states)
            last_ok = dict(p, states=states)


def eviction(bench, rng):
    n = 20_000 if bench.quick else 100_000
    cols = {"x": rng.integers(-1000, 1000, n), "y": rng.integers(-1000, 1000, n)}
    scopes = [mbk.Scope.from_arrays(cols, groups=rng.integers(0, 64, n), ngroups=64, rows=rng.random(n) < 0.5)
              for _ in range(8)]
    terms = mbk.monomials(["x", "y"], 2)
    entry = None
    for capacity in (1, 2, 4, 8, 16):
        cache = mbk.Cache(capacity=capacity, max_bytes=256 << 20)

        def cycle():
            for _ in range(3):
                for s in scopes:
                    cache.bank(s, terms)
            return cache.stats
        stats = bench.measure("eviction, cyclic over 8 domains", "cpu", cycle, repeats=1, phase="run",
                              capacity=capacity)
        bench.records[-1]["stats"] = stats
        if entry is None:
            entry = (stats["bytes"] - mbk.Cache(capacity=capacity).stats["bytes"]) // max(stats["entries"], 1)
    fixed = mbk.Cache(capacity=16).stats["bytes"]
    for k in (1, 2, 4, 8):                                      # a byte budget for k entries, capacity 16
        cache = mbk.Cache(capacity=16, max_bytes=fixed + k * entry)

        def cycle():
            for _ in range(3):
                for s in scopes:
                    cache.bank(s, terms)
            return cache.stats
        stats = bench.measure("eviction, byte budget", "cpu", cycle, repeats=1, phase="run", entries_budget=k,
                              max_bytes=fixed + k * entry)
        bench.records[-1]["stats"] = stats
    bench.note(f"one cache entry of 64 groups x {len(terms)} terms holds {entry} bytes (key and bank)")


def main():
    bench = Bench.from_args("FND-004", __doc__)
    rng = np.random.default_rng(bench.seed)
    build_grid(bench, rng)
    cache_cases(bench, rng)
    state_bytes(bench, rng)
    eviction(bench, rng)
    bench.note("Backends: cpu = math_bank.c (single-threaded), cuda = math_bank_gpu.cu (device runs only), duckdb = "
               "the same GROUP BY with COUNT and SUM in HUGEINT over the source table (DuckDB's own threads).")
    bench.note("Scope construction (native decode) is measured apart from the bank; a cache hit returns a copy of the "
               "bank, so its cost grows with groups x terms, not with rows.")
    bench.write()


if __name__ == "__main__":
    main()
