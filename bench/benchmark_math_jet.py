# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""JET-001 / JET-002 benchmark: exact jets of shared polynomial circuits (torbles/math_jet.c, math_jet_gpu.cu).

    python bench/benchmark_math_jet.py --capability JET-001|JET-002 --out FILE [--device] [--quick]

Shared circuits over 6 parameters t0..t5 and 2 data columns x, y (integer constants and center, so that the
independently transformed circuits below have integer coefficients):
  sparse   sum_i t_i (x + i) y               every parameter in its own term: each node's jet lives in one direction
                                             (sparse gradients)
  dense    (1 + sum_i t_i x_(i mod 2))^3     every interaction present: the dense worst-supported case
  random   a fixed-seed random DAG of total degree <= 5
Case 1, seeds and orders: m = 1, 2, 4, 8 random integer seed directions x orders d = 1, 2, 3; plan construction
("build"), the aggregate jet of sum_r f over N rows on the CPU (CUDA with --device), first execution and warm reuse;
state bytes (states 16 S + 16 per group, the plan, the per-point workspace, the device copies) and the static
arithmetic counts (multiplications and additions per row, times the rows).
Case 2, joint versus independent: per-point jets (directions = the parameters, d = 1, 2, 3) of every coefficient at N
points: the joint value/derivative pass against independently transformed circuits, i.e. every partial derivative
d^alpha f differentiated symbolically by the first slice's native polynomial kernel (tr_math_poly_derivative), the
center substituted (tr_math_poly_substitute), then evaluated at the N points by the native batched integer
polynomial evaluator (math_handles: CPU; CUDA with --device). Transformation (build) and evaluation are timed
separately; the two answers are checked equal (alpha! J[alpha] == d^alpha f) on sampled points.
Refusal thresholds, measured (never extrapolated): the support budget between orders 3 and 4 at 8 directions, and the
largest data magnitude the dense order-3 jet admits before a numerator leaves int64.
"""
import math

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk
from torbles import math_jet as mj
from torbles.math_budgets import Budget
from torbles.math_handles import HandleTable
from torbles.mathematics import AlgebraBudgetError, Context

P = 6
PARAMS = [f"t{i}" for i in range(P)]


def circuits(rng):
    b = mj.Builder(params=PARAMS, data=["x", "y"])
    t, (x, y) = b.params, b.data
    sparse = sum(((x + i) * y * t[i] for i in range(1, P)), x * y * t[0])
    lin = sum((t[i] * (x if i % 2 else y) for i in range(1, P)), t[0] * y)
    dense = (lin + 1) ** 3
    out = {"sparse": b.build(sparse), "dense": b.build(dense)}
    # a random DAG: integer constants, total degree <= 5
    r = mj.Builder(params=PARAMS, data=["x", "y"])
    vals, deg = list(r.params) + list(r.data), [1] * (P + 2)
    while len(vals) < 30:
        op, i, j = int(rng.integers(0, 5)), int(rng.integers(0, len(vals))), int(rng.integers(0, len(vals)))
        if op == 0:
            vals.append(vals[i] + vals[j]); deg.append(max(deg[i], deg[j]))
        elif op == 1 and deg[i] + deg[j] <= 5:
            vals.append(vals[i] * vals[j]); deg.append(deg[i] + deg[j])
        elif op == 2:
            vals.append(vals[i] - int(rng.integers(-3, 4))); deg.append(deg[i])
        elif op == 3 and 2 * deg[i] <= 5:
            vals.append(vals[i] ** 2); deg.append(2 * deg[i])
        elif op == 4:
            vals.append(int(rng.integers(-3, 4)) * vals[i]); deg.append(deg[i])
    out["random"] = r.build(sum(vals[-6:], vals[-7]))
    return out, (b, sparse, dense)


def nodes_of(circuit):
    return [(op, a, b, index, value) for op, a, b, index, value in circuit.nodes()]


def first_slice_expr(circuit):
    """the circuit rebuilt with the first slice's native polynomials (8 variables: t0..t5, x, y)"""
    ctx = Context()
    names = PARAMS + ["x", "y"]
    syms = ctx.symbols(*names)
    vals = []
    for op, a, b, index, value in nodes_of(circuit):
        if op == "const":
            vals.append(ctx.constant(value))
        elif op == "param":
            vals.append(syms[index])
        elif op == "data":
            vals.append(syms[P + index])
        elif op == "add":
            vals.append(vals[a] + vals[b])
        elif op == "sub":
            vals.append(vals[a] - vals[b])
        elif op == "mul":
            vals.append(vals[a] * vals[b])
        elif op == "pow":
            vals.append(vals[a] ** index)
        elif op == "scale":
            vals.append(vals[a] * ctx.constant(value))
        else:
            vals.append(-vals[a])
    return vals[-1], names


def independent_polynomials(circuit, monomials, center, table):
    """every d^alpha f with the center substituted, as native integer polynomials in (x, y) (math_handles)"""
    f, names = first_slice_expr(circuit)
    out = []
    for alpha in monomials:
        e = f
        for p, k in enumerate(alpha):
            for _ in range(k):
                e = e.diff(names[p])
        for p in range(P):
            e = e.substitute(names[p], center[p])
        terms = {}
        for exps, coef in e.terms:
            if coef.denominator != 1:
                raise SystemExit("JET bench: a transformed circuit has a non-integer coefficient")
            terms[(exps[P], exps[P + 1])] = int(coef)
        out.append(table.polynomial(terms or {(0, 0): 0}, nvars=2))
    return out


def joint_vs_independent(bench, rng, cs, n):
    xs, ys = rng.integers(-20, 21, n), rng.integers(-20, 21, n)
    pts = np.stack([xs, ys], axis=1).astype(np.int64)
    center = [int(v) for v in rng.integers(-2, 3, P)]
    table = HandleTable(capacity=4096)
    for name in ("sparse", "dense"):
        c = cs[name]
        for d in (1, 2, 3):
            plan = c.plan(d, dict(zip(PARAMS, center)))
            S, info = plan.support, plan.info
            p = dict(circuit=name, order=d, directions=P, points=n, support=S)
            scope = bench.measure("points scope", "cpu", lambda: plan.points_scope({"x": xs, "y": ys}), repeats=2,
                                  phase="build", **p)
            for backend in bench.backends():
                bench.measure("joint value/derivative pass", backend, lambda: plan.aggregate(scope, backend=backend),
                              repeats=3, phase="run", multiplications_per_point=info["multiplications"],
                              additions_per_point=info["additions"], **p)
            joint = plan.aggregate(scope)
            polys = bench.measure("independent transforms (first-slice derivative + substitution)", "reconstruction",
                                  lambda: independent_polynomials(c, plan.monomials, center, table), repeats=2,
                                  phase="build", **p)
            terms = sum(q.term_count for q in polys)
            for backend in bench.backends():
                bench.measure("independent evaluations (batched integer polynomials)",
                              "reconstruction" if backend == "cpu" else "reconstruction-cuda",
                              lambda: [q.evaluate_raw(pts, backend=backend) for q in polys], repeats=3, phase="run",
                              polynomials=len(polys), terms=terms, **p)
            # the two answers agree: alpha! J[alpha] == d^alpha f at sampled points
            sample = sorted(set(range(min(n, 32))) | {int(v) for v in rng.integers(0, n, 32)})
            vals = [q.evaluate(pts[sample])["values"] for q in polys]
            for i, g in enumerate(sample):
                deriv = joint.jet(g).derivatives
                for k in range(S):
                    if deriv[k] != vals[k][i]:
                        raise SystemExit(f"JET bench: joint and independent differ ({name}, d={d}, point {g}, {k})")
            bench.records.append(dict(case="joint vs independent sizes", backend="cpu", phase="measure", params=p,
                                      joint_state_bytes_per_point=16 * S + 16, joint_plan_bytes=info["bytes"],
                                      joint_multiplications_per_point=info["multiplications"],
                                      independent_polynomials=len(polys), independent_terms=terms,
                                      checked_points=len(sample)))
    table.close()


def seeds_and_orders(bench, rng, cs, n):
    xs, ys = rng.integers(-20, 21, n), rng.integers(-20, 21, n)
    scope = mbk.Scope.from_arrays({"x": xs, "y": ys})
    center = {k: int(v) for k, v in zip(PARAMS, rng.integers(-2, 3, P))}
    for name, c in cs.items():
        for m in (1, 2, 4, 8):
            dirs = [[int(v) for v in rng.integers(-2, 3, P)] for _ in range(m)]
            for d in (1, 2, 3):
                p = dict(circuit=name, directions=m, order=d, rows=n)
                plan = bench.measure("plan", "cpu", lambda: c.plan(d, center, dirs), repeats=3, phase="build", **p)
                info = plan.info
                agg = None
                for backend in bench.backends():
                    agg = bench.measure("aggregate jet", backend, lambda: plan.aggregate(scope, backend=backend),
                                        repeats=3, phase="run", **p)
                S = info["support"]
                bench.records.append(dict(
                    case="state bytes and arithmetic", backend="cpu", phase="measure", params=p, support=S,
                    pairs=info["pairs"], state_bytes_per_group=16 * S + 16, aggregate_bytes=agg.info["bytes"],
                    plan_bytes=info["bytes"], workspace_bytes_per_point=8 * info["workspace_words"],
                    device_bytes=plan.need(scope, "cuda")["bytes"], multiplications_per_row=info["multiplications"],
                    additions_per_row=info["additions"], multiplications=info["multiplications"] * n,
                    additions=info["additions"] * n, jet_steps=info["jets"], steps=info["steps"],
                    output_directions=len(info["output_directions"])))


def thresholds(bench, cs):
    dense = cs["dense"]
    dirs = [[int(i == j % P) for i in range(P)] for j in range(8)]
    budget = Budget(support=165)
    ok = bench.refusal("support budget 165: order 3, 8 directions", "cpu",
                       lambda: dense.plan(3, None, dirs, budget=budget), support=165, order=3)
    no = bench.refusal("support budget 165: order 4, 8 directions", "cpu",
                       lambda: dense.plan(4, None, dirs, budget=budget), support=495, order=4)
    bench.threshold("support budget (8 directions)", admitted=dict(order=3, support=165, refused=ok["refused"]),
                    refused=dict(order=4, support=495, refused=no["refused"], resource=no.get("resource")))
    # the largest |x| (powers of two) the dense order-3 jet admits at center 1, by actual runs
    plan = dense.plan(3, {k: 1 for k in PARAMS}, dirs)
    admitted, refused = None, None
    for e in range(1, 63):
        v = 1 << e
        try:
            plan.aggregate(mbk.Scope.from_arrays({"x": [v], "y": [v]}))
            admitted = v
        except OverflowError:
            refused = v
            break
    bench.threshold("int64 numerators of the dense order-3 jet (x = y = 2^e, center 1)",
                    admitted=dict(x=admitted), refused=dict(x=refused))
    try:
        dense.plan(1, None, [[1] * P] * 65, budget=Budget(variables=1 << 20))
    except Exception as exc:          # a representation limit, not a budget: recorded as measured
        bench.note(f"65 directions with a budget that admits them: {type(exc).__name__} ({exc})")


def main():
    bench = Bench.from_args("JET-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    cs, _ = circuits(rng)
    n_rows = 2000 if bench.quick else 100_000
    n_pts = 400 if bench.quick else 20_000
    for name, c in cs.items():
        i = c.info
        bench.records.append(dict(case="circuit", backend="cpu", phase="measure", params=dict(circuit=name),
                                  nodes=i["nodes"], degree=i["degree"], param_degree=i["param_degree"],
                                  bytes=i["bytes"]))
    seeds_and_orders(bench, rng, cs, n_rows)
    joint_vs_independent(bench, rng, cs, n_pts)
    thresholds(bench, cs)
    bench.note("Arithmetic counts are the plan's static counts of 64x64-bit products and 128-bit additions per row "
               "(the evaluation order of math_jet.h); they do not depend on the data.")
    bench.note("The independent baseline evaluates S separately transformed circuits; its build phase is the "
               "first slice's symbolic differentiation and substitution, its run phase the native batched integer "
               "polynomial evaluator. Both answers are checked equal on sampled points.")
    if not bench.device:
        bench.note("CUDA was not timed: only the emulation is available here (an emulated run is not a CUDA timing).")
    bench.write()


if __name__ == "__main__":
    main()
