# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""TENSOR-001 / TENSOR-002 benchmark: moment and count tensors in their three forms, and costed contraction plans.

    python bench/benchmark_math_tensor.py --capability TENSOR-001|TENSOR-002 --out FILE [--device] [--quick]

Both capabilities share the two cases (the receipt is written under the capability given):
  states        tensor order 2 / 3 / 4, dimensions 2 / 4 / 8 per axis, support sparsity 1% / 10% / dense: count tensors
                over categorical columns whose rows occupy that fraction of the joint cells (40,000 rows; quick
                5,000), built dense and sparse (CPU; CUDA with --device), converted to every other form, and
                marginalized; moment tensors (order 2 / 3 / 4 over 2 / 4 / 8 columns) built dense (symmetric fast
                path), sparse and factored. Every build records the actual cells, stored entries, terms and bytes.
                Baselines: NumPy (bincount over linear cell ids; einsum of the rows for moments) and DuckDB GROUP BY,
                checked equal to the native states.
  contractions  chains and stars on identical tensors in dense and sparse states (densities 1% / 10% / 100%):
                the plan (every order, kernel and device, FND-008 costs), then every admissible candidate executed
                (first and warm samples) with its planned and actual intermediate shapes and bytes, against
                numpy.einsum (object dtype: exact; int64 with optimize=False / "greedy"). Shapes where the cheapest
                order changes, and tensor / bytes budgets that refuse the large intermediates (recorded at the limit
                and one past), are included.
State construction ("build"), planning ("plan"), first execution and warm reuse ("run": first sample and warm
samples) are kept apart; CUDA is timed only with --device on an actual device.
"""
import time

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_tensor as mt
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError


def count_data(rng, rows, order, dim, sparsity):
    """rows over `order` categorical columns of `dim` values whose joint cells occupy about `sparsity` of dim^order"""
    cells = dim ** order
    k = max(1, int(round(sparsity * cells)))
    chosen = rng.choice(cells, size=k, replace=False)
    ids = chosen[rng.integers(0, k, rows)]
    cols = {}
    rest = ids.copy()
    for a in reversed(range(order)):
        cols[f"c{a}"] = (rest % dim).astype(np.int64)
        rest //= dim
    return {f"c{a}": cols[f"c{a}"] for a in range(order)}, ids, k


def record_state(bench, case, t, **p):
    i = t.info
    bench.note(f"{case} {p}: repr {i['repr']}, dims {i['dims']}, cells {i['cells']}, stored {i['nnz']}, "
               f"terms {i['nterms']}, bytes {i['bytes']}")
    return dict(dims=str(i["dims"]), cells=i["cells"], nnz=i["nnz"], nterms=i["nterms"], bytes=i["bytes"])


def states(bench, rng):
    rows = 5000 if bench.quick else 40000
    orders, dims_grid, sparsities = (2, 3, 4), (2, 4, 8), (0.01, 0.10, 1.0)
    for order in orders:
        for dim in dims_grid:
            for sp in sparsities:
                if dim ** order * sp < 1 and sp < 1:
                    continue
                cols, ids, k = count_data(rng, rows, order, dim, sp)
                scope = mbk.Scope.from_arrays(cols)
                names = list(cols)
                p = dict(order=order, dim=dim, sparsity=sp, rows=rows, occupied=k)
                for repr_ in ("dense", "sparse"):
                    for backend in bench.backends():
                        t = bench.measure("count tensor", backend, lambda: mt.counts(scope, names, repr=repr_,
                                                                                     backend=backend),
                                          repeats=3, phase="build", repr=repr_, **p)
                    shape = record_state(bench, "count tensor", t, repr=repr_, **p)
                    bench.records[-1]["result"] = shape
                    for target in ("dense", "sparse", "factored"):
                        if target == repr_:
                            continue
                        u = bench.measure("conversion", "cpu", lambda: t.convert(target), repeats=3, phase="run",
                                          source=repr_, target=target, **p)
                        bench.records[-1]["result"] = record_state(bench, "conversion", u, source=repr_, **p)
                    m = bench.measure("marginal", "cpu", lambda: t.marginal(list(range(order - 1))), repeats=3,
                                      phase="run", repr=repr_, **p)
                    bench.records[-1]["result"] = record_state(bench, "marginal", m, repr=repr_, **p)
                # baselines: NumPy bincount over linear ids, DuckDB GROUP BY (checked equal)
                dense = mt.counts(scope, names, repr="dense")
                counts = bench.measure("count tensor", "numpy", lambda: np.bincount(ids, minlength=dim ** order),
                                       repeats=3, phase="build", **p)
                got = np.array(dense.array(0).ravel().tolist(), dtype=np.int64)
                lo = [scope.column(c)["min"] for c in names]
                span = [scope.column(c)["max"] - scope.column(c)["min"] + 1 for c in names]
                ref = np.zeros(span, dtype=np.int64)
                np.add.at(ref, tuple(cols[c] - l for c, l in zip(names, lo)), 1)
                assert np.array_equal(got, ref.ravel()), "dense counts differ from NumPy"
                del counts
                con = duckdb.connect()
                con.register("t", pd.DataFrame(cols))
                sql = f"SELECT {', '.join(names)}, COUNT(*) FROM t GROUP BY ALL ORDER BY {', '.join(names)}"
                duck = bench.measure("count tensor", "duckdb", lambda: con.execute(sql).fetchall(), repeats=3,
                                     phase="build", **p)
                assert len(duck) == len(dense.to_sparse().entries(0)), "sparse counts differ from DuckDB"
                con.close()
    # moment tensors (dense data): the symmetric dense build, sparse and factored forms, against NumPy einsum
    mrows = 2000 if bench.quick else 20000
    for order in orders:
        for dim in dims_grid:
            x = rng.integers(-20, 21, (mrows, dim)).astype(np.int64)
            scope = mbk.Scope.from_arrays({f"x{j}": x[:, j] for j in range(dim)})
            names = [f"x{j}" for j in range(dim)]
            p = dict(order=order, dim=dim, rows=mrows)
            for repr_ in ("dense", "sparse", "factored"):
                backends = bench.backends() if repr_ != "factored" else ("cpu",)
                for backend in backends:
                    t = bench.measure("moment tensor", backend, lambda: mt.moments(scope, names, order, repr=repr_,
                                                                                   backend=backend),
                                      repeats=3, phase="build", repr=repr_, **p)
                    bench.records[-1]["result"] = record_state(bench, "moment tensor", t, repr=repr_, **p)
            letters = "abcd"[:order]
            spec = ",".join("r" + c for c in letters) + "->" + letters
            ref = bench.measure("moment tensor", "numpy", lambda: np.einsum(spec, *([x] * order)), repeats=3,
                                phase="build", **p)
            dense = mt.moments(scope, names, order)
            assert np.array_equal(np.array(dense.array(0).ravel().tolist(), dtype=np.int64), ref.ravel())


def chain(rng, dims, density):
    labels = "ijklmn"
    ops, arrays = [], []
    for a in range(len(dims) - 1):
        shape = (dims[a], dims[a + 1])
        arr = rng.integers(-3, 4, shape).astype(np.int64)
        arr[rng.random(shape) >= density] = 0
        ops.append((labels[a] + labels[a + 1], arr))
        arrays.append(arr)
    return ops, labels[0] + labels[len(dims) - 1]


def contractions(bench, rng):
    shapes = [((2, 60, 2, 60), "chain, (AB)C cheaper"), ((60, 2, 60, 2), "chain, A(BC) cheaper"),
              ((8, 8, 8, 8, 8), "chain of four"), ((30, 30, 30), "square chain")]
    if bench.quick:
        shapes = shapes[:3]
    densities = (0.01, 0.10, 1.0)
    for dims, label in shapes:
        for density in densities:
            ops, output = chain(rng, dims, density)
            for state in ("dense", "sparse"):
                tensors = [(mt.dense(a) if state == "dense" else mt.dense(a).to_sparse(), lab) for lab, a in ops]
                net = mt.network(tensors, output)
                p = dict(network=label, dims=str(dims), density=density, state=state)
                devices = ("cpu", "cuda") if bench.device else ("cpu",)
                plan = bench.measure("plan", "cpu", lambda: net.plan(devices=devices), repeats=3, phase="plan", **p)
                info = plan.info
                bench.records[-1]["result"] = dict(candidates=info["candidates"], orders=info["orders"],
                                                   complete=info["complete"], chosen=info["chosen"])
                want = np.einsum(",".join(lab for lab, _ in ops) + "->" + output, *[a.astype(object) for _, a in ops])
                for c in plan.candidates:
                    if c["status"] == "rejected":
                        continue
                    backend = c["device"]
                    q = dict(p, candidate=c["index"], kernel=c["kernel"], order=c["order"], status=c["status"],
                             planned_total=c["total"], planned_peak=c["peak_bytes"],
                             planned_steps=str([(s["out"], s["cells"], s["nnz"], s["bytes"]) for s in c["steps"]]))
                    out = bench.measure("contraction", backend, lambda: net.contract(plan, c["index"]), repeats=3,
                                        phase="run", **q)
                    t, run = out
                    assert np.array_equal(np.asarray(t.array(0), dtype=object), want)
                    bench.records[-1]["result"] = dict(actual_peak=run["peak_bytes"],
                                                       actual_steps=str([(s["nnz"], s["bytes"]) for s in run["steps"]]),
                                                       output_bytes=run["output_bytes"])
            exact = [a.astype(object) for _, a in ops]
            spec = ",".join(lab for lab, _ in ops) + "->" + output
            bench.measure("contraction", "numpy-object", lambda: np.einsum(spec, *exact), repeats=3, phase="run",
                          network=label, dims=str(dims), density=density)
            ints = [a for _, a in ops]
            bench.measure("contraction", "numpy-int64", lambda: np.einsum(spec, *ints), repeats=3, phase="run",
                          network=label, dims=str(dims), density=density, optimize="none")
            bench.measure("contraction", "numpy-int64", lambda: np.einsum(spec, *ints, optimize="greedy"), repeats=3,
                          phase="run", network=label, dims=str(dims), density=density, optimize="greedy")
    # the cheapest order changes with the shape (recorded), and over-budget intermediates are refused
    for dims, label in shapes[:2]:
        ops, output = chain(rng, dims, 1.0)
        net = mt.network([(mt.dense(a), lab) for lab, a in ops], output)
        plan = net.plan(kernels=("dense",))
        ch = plan.candidate(plan.chosen)
        bench.note(f"{label} {dims}: chosen order {ch['order']} first step {ch['steps'][0]['a']},"
                   f"{ch['steps'][0]['b']} -> {ch['steps'][0]['out']} ({ch['steps'][0]['cells']} cells)")
        largest = max(max(s["cells"] for s in c["steps"]) for c in plan.candidates)
        smallest = min(max(s["cells"] for s in c["steps"]) for c in plan.candidates)
        refused = net.plan(kernels=("dense",), budget=Budget(tensor=largest - 1))
        rejected = [c["index"] for c in refused.candidates if c["status"] == "rejected"]
        bench.note(f"{label}: tensor budget {largest - 1} rejects candidates {rejected} (largest intermediate "
                   f"{largest}); the chosen one runs")
        bench.threshold(f"{label}: plan tensor budget", admitted=dict(tensor=smallest),
                        refused=dict(tensor=smallest - 1))
        bench.measure("plan", "cpu", lambda: net.plan(kernels=("dense",), budget=Budget(tensor=smallest)), repeats=1,
                      phase="plan", network=label, budget_tensor=smallest)
        bench.refusal("plan", "cpu", lambda: net.plan(kernels=("dense",), budget=Budget(tensor=smallest - 1)),
                      network=label, budget_tensor=smallest - 1)
        # bytes: the plan object and the smallest candidate peak must both fit (measured at both points)
        limit = max(min(c["peak_bytes"] for c in plan.candidates), plan.info["bytes"])
        admitted = bench.measure("plan", "cpu", lambda: net.plan(kernels=("dense",), budget=Budget(bytes=limit)),
                                 repeats=1, phase="plan", network=label, budget_bytes=limit)
        bench.records[-1]["result"] = dict(chosen=admitted.chosen)
        rec = bench.refusal("plan", "cpu", lambda: net.plan(kernels=("dense",), budget=Budget(bytes=limit - 1)),
                            network=label, budget_bytes=limit - 1)
        if rec["refused"]:
            bench.threshold(f"{label}: plan bytes budget", admitted=dict(bytes=limit), refused=dict(bytes=limit - 1))


def main():
    bench = Bench.from_args("TENSOR-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    t0 = time.perf_counter()
    states(bench, rng)
    bench.note(f"states case: {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    contractions(bench, rng)
    bench.note(f"contractions case: {time.perf_counter() - t0:.1f} s")
    bench.note("costs are the FND-008 planner's relative work units (amount x rate), not times; the measured times "
               "are recorded beside them")
    bench.write()


if __name__ == "__main__":
    main()
