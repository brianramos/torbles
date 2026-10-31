# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-005 benchmark: registry dispatch and the bounded candidate search, measured apart from execution; then the
direct scan, sparse power sums (sparse polynomial), code histogram and bit-plane (factored state) representations
on identical requests, against NumPy, DuckDB, the engine's own plane-algebra SUM and the first slice's moment bank.

    python bench/benchmark_math_registry.py --capability FND-005 --out FILE [--device] [--quick]

Requests: sum of f(x) over the rows selected by an engine predicate (s < k) of a table with a code column x,
for code domains 2^4 .. 2^16, selectivities 1% / 50% and degrees 2 / 6. Phases: "build" (state construction),
"first" (a fresh execution: scan, or build + contract), "warm" (a cached state contracted again; the scan repeats).
Every representation's result is checked against DuckDB before it is timed. State bytes and the explanation's backend
path are recorded. Thresholds: the histogram's states budget, the bit-plane circuit budget and the 128-bit bound.
"""
import ctypes as C

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_registry as reg
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError, MomentBasis


def dispatch(bench):
    api = reg._api()
    m = reg._Module()
    n = 2000

    def lookups():
        for _ in range(n):
            for i in range(9):
                api["module_get"](i, C.byref(m))
        return n * 9
    bench.measure("registry dispatch: descriptor lookups", "cpu", lookups, repeats=5, calls=n * 9)
    chains = []
    for name in ("scan", "histogram", "power", "bitplane", "empty"):
        for agg in ("sum", "max"):
            ch = reg.chain(name, agg)
            chains.append((reg._steps(ch["steps"]), len(ch["steps"]), reg._carrier_id(ch["input"])))
    refusal, out = reg._Refusal(), C.c_int32()

    def checks():
        for _ in range(n):
            for steps, length, inp in chains:
                api["check_chain"](steps, length, inp, 0, 1, C.byref(out), C.byref(refusal))
        return n * len(chains)
    bench.measure("registry dispatch: composition checks", "cpu", checks, repeats=5, calls=n * len(chains))
    codes = np.arange(10_000, dtype=np.uint32) % 4096
    for devices in (("cpu",), ("cpu", "cuda")):
        req = reg.Request(codes, [[1, 2, 3, 4]], rows=np.arange(10_000) % 2 == 0)
        cands = (reg._Candidate * reg.MAX_CANDIDATES)()
        cnt, br = C.c_int32(), reg._BudgetRefusal()
        b = Budget()
        mask = reg._devices(devices)

        def search():
            for _ in range(n):
                api["candidates"](C.byref(req.raw), None, mask, 5000, 10_000, 0, C.byref(b.raw), cands,
                                  reg.MAX_CANDIDATES, C.byref(cnt), C.byref(br))
            return int(cnt.value)
        bench.measure("bounded candidate search", "cpu", search, repeats=5, calls=n, devices="+".join(devices),
                      candidates=len(reg.candidates(req, devices=devices)))
    bench.note("Dispatch and search samples cover `calls` native calls each (per-call time = sample / calls); they "
               "touch no row: a request's execution is timed separately below.")
    req = reg.Request(np.arange(64, dtype=np.uint32), [[1, 1]])
    bench.measure("python wrapper overhead (64-row scan through reg.execute)", "cpu",
                  lambda: reg.execute(req, "scan")["values"][0], repeats=20, rows=64)


def native(req, name, backend, finalize=None, state=None):
    """the executor's native call alone (no Python formatting): the representation's entry point"""
    res, ex, b = reg._Result(), reg._Explain(), Budget()
    rid = reg.REPRESENTATIONS[name]
    _, fn = reg.entry(name, backend)
    handle = state._open() if state is not None else None
    final = reg._final(finalize, name)

    def run():
        rc = (fn(C.byref(req.raw), rid, handle, C.byref(b.raw), C.byref(res), C.byref(ex)) if backend == "cpu" else
              fn(C.byref(req.raw), rid, final, C.byref(b.raw), C.byref(res), C.byref(ex)))
        if rc:
            raise SystemExit(f"{name} {backend} refused while timed: {rc}")
        return reg._i128(res.value[0])
    return run


def native_build(req, name):
    api, ex, err, b = reg._api(), reg._Explain(), C.c_int32(), Budget()
    rid = reg.REPRESENTATIONS[name]

    def run():
        handle = api["state_build"](C.byref(req.raw), rid, 0, 0, C.byref(b.raw), C.byref(ex), C.byref(err))
        if err.value:
            raise SystemExit(f"{name} state refused while timed: {err.value}")
        api["state_free"](handle)
        return int(ex.state_bytes)
    return run


def make_table(rng, rows, ncodes):
    x = rng.integers(0, ncodes, size=rows, dtype=np.int64)
    s = rng.integers(0, 1000, size=rows, dtype=np.int64)
    frame = pd.DataFrame({"x": x, "s": s})
    db = torbles.from_ddl("CREATE TABLE t (x BIGINT, s BIGINT)", {"t": frame}, keep_plain=False)
    return db, frame


def duck_sum(con, poly, k):
    expr = " + ".join(f"CAST({c} AS HUGEINT)" + " * CAST(x AS HUGEINT)" * j for j, c in enumerate(poly))
    return int(con.execute(f"SELECT COALESCE(SUM({expr}), 0) FROM f WHERE s < {k}").fetchone()[0])


def representations(bench):
    rng = np.random.default_rng(bench.seed)
    grid_rows = (20_000, 100_000) if bench.quick else (100_000, 1_000_000)
    domains = (16, 1024, 65536)
    polys = {2: [3, -2, 1], 6: [1, 0, -1, 0, 2, 0, 1]}
    for rows in grid_rows:
        for ncodes in domains:
            db, frame = make_table(rng, rows, ncodes)
            t = db["t"]
            con = duckdb.connect()
            con.register("f", frame)
            col = reg.Column(t, "x")
            for selectivity in (0.01, 0.5):
                k = int(selectivity * 1000)
                ids = db.query(t).where(t.s < k).rows()
                for degree, poly in polys.items():
                    p = dict(rows=rows, ncodes=ncodes, selectivity=selectivity, degree=degree)
                    want = duck_sum(con, poly, k)
                    req = reg.Request(col, [poly], row_ids=ids)
                    for name in ("scan", "histogram", "power", "bitplane"):
                        for backend in bench.backends():
                            kw = {"finalize": "cpu"} if backend == "cuda" and name in ("power", "bitplane") else {}
                            try:
                                got = reg.execute(req, name, backend=backend, **kw)
                            except (AlgebraBudgetError, OverflowError) as e:
                                bench.records.append(dict(case="identical request", backend=backend, phase="refusal",
                                                          params=dict(p, representation=name), refused=True,
                                                          resource=getattr(e, "resource", None), message=str(e)))
                                continue
                            if got["values"] != [want]:
                                raise SystemExit(f"{name} {backend} differs from DuckDB at {p}: {got['values']} != {want}")
                            ex = got["explanation"]
                            path = dict(p, representation=name, path=ex["device"], finalize=ex["finalize"],
                                        state_bytes=ex["state_bytes"], state_elements=ex["state_elements"])
                            run = native(req, name, backend, kw.get("finalize"))
                            bench.measure("identical request", backend, run, repeats=3, phase="first", **path)
                            if name != "scan" and backend == "cpu":
                                bench.measure("identical request", backend, native_build(req, name), repeats=3,
                                              phase="build", **path)
                                st = reg.build_state(req, name)
                                bench.measure("identical request", backend, native(req, name, "cpu", None, st),
                                              repeats=5, phase="warm", **path)
                                st.close()
                    bench.measure("identical request", "duckdb", lambda: duck_sum(con, poly, k), repeats=3, phase="first", **p)
                    bound = rows * sum(abs(c) * (ncodes ** j) for j, c in enumerate(poly))
                    if bound < 2 ** 63:                     # exact in int64: the NumPy baseline is valid
                        xs = frame["x"].to_numpy()
                        sel = frame["s"].to_numpy() < k
                        bench.measure("identical request", "numpy",
                                      lambda: int(np.polynomial.polynomial.polyval(xs[sel], np.array(poly, dtype=np.int64)).sum()),
                                      repeats=3, phase="first", **p)
                    if degree == 2:                           # the engine's plane algebra and the moment bank
                        x = t.x
                        meas = poly[0] + poly[1] * x + poly[2] * x * x
                        got = db.query(t).where(t.s < k).sum(meas).run()
                        if got != want:
                            raise SystemExit(f"engine SUM differs at {p}")
                        bench.measure("identical request", "engine-planes",
                                      lambda: db.query(t).where(t.s < k).sum(meas).run(), repeats=3, phase="first", **p)
                        basis = bench.measure("identical request", "moments-builtin",
                                              lambda: MomentBasis(db.query(t).where(t.s < k), ["x"]), repeats=3,
                                              phase="build", **p)
                        xv = basis.variables["x"]
                        fn = basis.reduce(poly[0] + poly[1] * xv + poly[2] * xv ** 2)
                        if fn.evaluate() != want:
                            raise SystemExit(f"moment bank differs at {p}")
                        bench.measure("identical request", "moments-builtin",
                                      lambda: basis.reduce(poly[0] + poly[1] * xv + poly[2] * xv ** 2).evaluate(),
                                      repeats=5, phase="warm", **p)
                        basis.close()
            col.close()
            con.close()
    bench.note("Representation timings are the executors' native calls (the entry point the descriptor names); "
               "'build' includes freeing the state; 'warm' contracts a cached state.")
    bench.note("Backends: cpu (math_registry.c), cuda (math_registry_gpu.cu, --device only; POWER and BITPLANE "
               "finalize on the CPU), duckdb (SUM over HUGEINT casts), numpy (int64 polyval where the bound fits "
               "int64), engine-planes (torbles' own SUM of a degree-2 measure by plane popcounts), moments-builtin "
               "(the first slice's degree-two moment bank: build = its aggregate queries, warm = reduce + evaluate).")


def thresholds(bench):
    codes = np.arange(50_000, dtype=np.uint32)
    last = None
    for ncodes in (1 << 14, 1 << 15, 1 << 16, (1 << 16) + 1):
        req = reg.Request(codes % ncodes, [[1, 1, 1]], ncodes=ncodes)
        p = dict(ncodes=ncodes, rows=50_000, budget_states=1 << 16)
        rec = bench.refusal("histogram states", "cpu",
                            lambda: reg.execute(req, "histogram", budget=Budget(states=1 << 16)), **p)
        if rec["refused"]:
            bench.threshold("histogram cells", last, p)
        else:
            last = p
    last = None
    req16 = reg.Request(codes % (1 << 16), [[1, 1, 1]], ncodes=1 << 16)
    for D in range(1, 9):
        r = req16.with_polys([[1] * (D + 1)])
        p = dict(width=16, degree=D, budget_circuit=1 << 20)
        rec = bench.refusal("bit-plane circuit", "cpu", lambda: reg.build_state(r, "bitplane"), **p)
        if rec["refused"]:
            bench.threshold("bit-plane degree at width 16", last, p)
            break
        last = p
    last = None
    for k in range(1, 9):                                # the 128-bit bound of POWER: 2^62 x^k over 50,000 rows
        r = req16.with_polys([[0] * k + [1 << 62]])
        p = dict(rows=50_000, ncodes=1 << 16, coefficient=1 << 62, exponent=k)
        try:
            reg.execute(r, "power")
            last = p
        except OverflowError as exc:
            bench.records.append(dict(case="power bound", backend="cpu", phase="refusal", params=p, refused=True,
                                      message=str(exc)))
            bench.threshold("power 128-bit bound", last, p, bound="rows * |a| * M^k < 2^127")
            break


def main():
    bench = Bench.from_args("FND-005", __doc__)
    dispatch(bench)
    representations(bench)
    thresholds(bench)
    bench.note("Costs in the registry are relative work units; this benchmark measures time. The planner's choice "
               "among these candidates, on held-out shapes, is FND-008's benchmark.")
    bench.write()


if __name__ == "__main__":
    main()
