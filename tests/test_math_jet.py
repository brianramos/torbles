# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""JET-001 (dual-number directional derivatives) and JET-002 (higher-order and multivariate jets) of bounded
polynomial circuits (torbles/math_jet.c, math_jet_gpu.cu, math_jet.py).

Oracles, all independent of the implementation under test:
  * exact symbolic polynomials written here with Python Fractions (dicts of exponent tuples): the circuit expanded
    into f(theta, x), differentiated symbolically along the seed directions (D_u = sum_p u_p d/dtheta_p) and evaluated
    at the center (the derivative convention), and the circuit evaluated over Q[t] (theta = c + U^T t) whose
    coefficients are the Taylor convention;
  * the first slice's native rational polynomial kernel (torbles.mathematics: tr_math_poly_derivative / _evaluate),
    a second, separate implementation of exact differentiation;
  * DuckDB SQL (SUM ... FILTER over the source table) for aggregates over engine tables.
Every native result is also computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, i.e. scrambled
thread order) and compared for exact equality with the CPU states. Seeds are fixed (numpy default_rng).
"""
import ctypes as C
import itertools
import math
import subprocess
import sys
import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_jet as mj
from torbles import math_bank as mbk
from torbles.math_budgets import Budget
from torbles.math_domains import CarrierError, DomainError
from torbles.math_errors import NoSolutionError, StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context

P61 = (1 << 61) - 1


# ---------------------------------------------------------------------------------------------- the polynomial oracle
def padd(p, q, sign=1):
    r = dict(p)
    for k, v in q.items():
        s = r.get(k, 0) + sign * v
        if s:
            r[k] = s
        else:
            r.pop(k, None)
    return r


def pscale(p, c):
    return {k: v * c for k, v in p.items()} if c else {}


def pmul(p, q, maxdeg=None):
    r = {}
    for k1, v1 in p.items():
        d1 = sum(k1)
        for k2, v2 in q.items():
            if maxdeg is not None and d1 + sum(k2) > maxdeg:
                continue
            k = tuple(a + b for a, b in zip(k1, k2))
            s = r.get(k, 0) + v1 * v2
            if s:
                r[k] = s
            else:
                r.pop(k, None)
    return r


def ppow(p, n, nvars, maxdeg=None):
    r = {(0,) * nvars: Fraction(1)}
    for _ in range(n):
        r = pmul(r, p, maxdeg)
    return r


def pderiv(p, i):
    r = {}
    for k, v in p.items():
        if k[i]:
            e = list(k)
            e[i] -= 1
            r[tuple(e)] = r.get(tuple(e), 0) + v * k[i]
    return {k: v for k, v in r.items() if v}


def peval(p, point):
    total = Fraction(0)
    for k, v in p.items():
        t = v
        for x, e in zip(point, k):
            if e:
                t *= Fraction(x) ** e
        total += t
    return total


def interpret(nodes, const, param, data, nvars, maxdeg=None):
    """the circuit's nodes as polynomials in nvars variables (op names as in math_jet.OPS)"""
    vals = []
    one = {(0,) * nvars: Fraction(1)}
    for op, a, b, index, value in nodes:
        value = Fraction(value[0], value[1]) if isinstance(value, tuple) else Fraction(value)
        if op == "const":
            vals.append(pscale(one, value))
        elif op == "param":
            vals.append(param(index))
        elif op == "data":
            vals.append(data(index))
        elif op == "add":
            vals.append(padd(vals[a], vals[b]))
        elif op == "sub":
            vals.append(padd(vals[a], vals[b], -1))
        elif op == "mul":
            vals.append(pmul(vals[a], vals[b], maxdeg))
        elif op == "pow":
            vals.append(ppow(vals[a], index, nvars, maxdeg))
        elif op == "scale":
            vals.append(pscale(vals[a], value))
        elif op == "neg":
            vals.append(pscale(vals[a], -1))
        else:
            raise AssertionError(op)
    return vals[-1]


def expand(nodes, P, D):
    """f(theta, x) as a polynomial in P + D variables (parameters first, then the data columns)"""
    V = P + D
    unit = lambda i: {tuple(1 if j == i else 0 for j in range(V)): Fraction(1)}
    return interpret(nodes, None, unit, lambda c: unit(P + c), V)


def taylor_oracle(nodes, P, D, center, U, x, d):
    """g(t) = f(c + U^T t, x) evaluated over Q[t] (exact, truncated at degree d): {alpha: Taylor coefficient}"""
    m = len(U)
    zero = (0,) * m

    def param(p):
        poly = {zero: Fraction(center[p])} if center[p] else {}
        for j in range(m):
            if U[j][p]:
                poly[tuple(1 if i == j else 0 for i in range(m))] = Fraction(U[j][p])
        return poly
    data = lambda c: ({zero: Fraction(x[c])} if x[c] else {})
    return interpret(nodes, None, param, data, m, d)


def monomials(m, d):
    """the documented order: graded, then the sorted index tuples in lexicographic order"""
    out = []
    for k in range(d + 1):
        for comb in itertools.combinations_with_replacement(range(m), k):
            e = [0] * m
            for i in comb:
                e[i] += 1
            out.append(tuple(e))
    return out


def factorial(alpha):
    return math.prod(math.factorial(a) for a in alpha)


def derivative_oracle(f, P, D, center, U, x, d):
    """{alpha: d^alpha g(0)}: f differentiated symbolically along the directions, then evaluated at (c, x)"""
    m = len(U)
    point = list(center) + list(x)
    memo = {(0,) * m: f}
    out = {}
    for alpha in monomials(m, d):
        if alpha not in memo:
            j = max(i for i in range(m) if alpha[i])
            prev = memo[tuple(a - (i == j) for i, a in enumerate(alpha))]
            q = {}
            for p in range(P):
                if U[j][p]:
                    q = padd(q, pscale(pderiv(prev, p), Fraction(U[j][p])))
            memo[alpha] = q
        out[alpha] = peval(memo[alpha], point)
    return out


def first_slice(nodes, P, D):
    """the circuit built with the first slice's native polynomials (torbles.mathematics), and its variable names"""
    names = [f"p{i}" for i in range(P)] + [f"x{i}" for i in range(D)]
    ctx = Context()
    syms = ctx.symbols(*names)
    vals = []
    for op, a, b, index, value in nodes:
        value = Fraction(value[0], value[1]) if isinstance(value, tuple) else Fraction(value)
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
        elif op == "neg":
            vals.append(-vals[a])
    return vals[-1], names


def first_slice_partial(expr, names, alpha, bindings):
    """d^alpha f / d theta^alpha at the bindings with the first slice's native derivative (identity directions)"""
    e = expr
    for p, k in enumerate(alpha):
        for _ in range(k):
            e = e.diff(names[p])
    used = set(e.support)
    return Fraction(e.evaluate({k: v for k, v in bindings.items() if k in used}))


def random_circuit(rng, P, D, size=12, maxdeg=5, maxpow=3, cmax=3):
    """a random DAG over P parameters and D data columns with total degree <= maxdeg"""
    nodes, deg = [], []
    for p in range(P):
        nodes.append(("param", -1, -1, p, (0, 1))); deg.append(1)
    for c in range(D):
        nodes.append(("data", -1, -1, c, (0, 1))); deg.append(1)

    def frac():
        return (int(rng.integers(-cmax, cmax + 1)), int(rng.integers(1, 4)))
    ops = ["add", "sub", "mul", "pow", "scale", "neg", "const", "mul", "add"]
    while len(nodes) < size:
        op = ops[int(rng.integers(0, len(ops)))]
        n = len(nodes)
        a = int(rng.integers(max(0, n - 6), n)) if rng.random() < 0.7 else int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if op == "const":
            nodes.append(("const", -1, -1, 0, frac())); deg.append(0)
        elif op in ("add", "sub"):
            nodes.append((op, a, b, 0, (0, 1))); deg.append(max(deg[a], deg[b]))
        elif op == "mul":
            if deg[a] + deg[b] > maxdeg:
                continue
            nodes.append((op, a, b, 0, (0, 1))); deg.append(deg[a] + deg[b])
        elif op == "pow":
            k = int(rng.integers(0, maxpow + 1))
            if deg[a] * k > maxdeg:
                continue
            nodes.append((op, a, -1, k, (0, 1))); deg.append(deg[a] * k)
        elif op == "scale":
            nodes.append((op, a, -1, 0, frac())); deg.append(deg[a])
        else:
            nodes.append((op, a, -1, 0, (0, 1))); deg.append(deg[a])
    # the output: a sum of a few nodes, so that most of the DAG is reachable
    picks = sorted({int(v) for v in rng.integers(P + D, len(nodes), 3)}) if len(nodes) > P + D else [0]
    acc = picks[0]
    for q in picks[1:]:
        nodes.append(("add", acc, q, 0, (0, 1))); deg.append(max(deg[acc], deg[q]))
        acc = len(nodes) - 1
    return nodes


def rand_q(rng, lo=-3, hi=3, den=4):
    return Fraction(int(rng.integers(lo, hi + 1)), int(rng.integers(1, den + 1)))


def jet_dict(jet, convention="taylor"):
    return {a: v for a, v in jet.coefficients(convention).items()}


def obj(values):
    return pd.Series(values, dtype=object)


class Jets(unittest.TestCase):
    maxDiff = None

    def both(self, plan, scope, **kw):
        """the CPU jet set, checked equal to the (emulated, scrambled) CUDA one: natively and word for word"""
        cpu = plan.aggregate(scope, **kw)
        gpu = plan.aggregate(scope, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu))
        for x, y in zip(cpu.arrays(), gpu.arrays()):
            self.assertTrue(np.array_equal(x, y))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    def points(self, plan, xs):
        """per-point jets on the CPU and CUDA paths (one group per point)"""
        scope = plan.points_scope({name: [x[i] for x in xs] for i, name in enumerate(plan.data_names)}, n=len(xs))
        return self.both(plan, scope).jets()

    def check_against_oracles(self, nodes, P, D, center, U, xs, d, scales=None, first=None, identity=False):
        """every jet of the plan at the points xs (logical values) against the Taylor and derivative oracles"""
        scales = scales or [0] * D
        circuit = mj.Circuit.from_nodes(nodes, [f"p{i}" for i in range(P)], [f"x{i}" for i in range(D)])
        plan = circuit.plan(d, center, None if identity else U, scales)
        f = expand(nodes, P, D)
        jets = self.points(plan, xs)
        self.assertEqual(list(plan.monomials), monomials(len(U), d))
        compared = 0
        for x, jet in zip(xs, jets):
            vals = [Fraction(v, 10 ** s) for v, s in zip(x, scales)]
            tay = taylor_oracle(nodes, P, D, center, U, vals, d)
            der = derivative_oracle(f, P, D, center, U, vals, d)
            for alpha in monomials(len(U), d):
                self.assertEqual(jet.coefficient(alpha), tay.get(alpha, 0), (alpha, x))
                self.assertEqual(jet.derivative(alpha), der[alpha], (alpha, x))
                self.assertEqual(jet.derivative(alpha), factorial(alpha) * jet.coefficient(alpha))
            if first is not None:
                expr, names = first
                bindings = dict(zip(names, list(center) + vals))
                for alpha in monomials(len(U), d):
                    self.assertEqual(jet.derivative(alpha), first_slice_partial(expr, names, alpha, bindings), alpha)
                    compared += 1
        return plan, jets, compared

    # ============================================================================== JET-001
    def test_jet001_duals_match_symbolic_directional_derivatives(self):
        rng = np.random.default_rng(1001)
        first_slice_checks = 0
        for trial in range(10):
            P, D = int(rng.integers(1, 5)), int(rng.integers(0, 3))
            nodes = random_circuit(rng, P, D, size=int(rng.integers(8, 16)), maxdeg=5)
            try:
                first = first_slice(nodes, P, D)
            except (AlgebraBudgetError, OverflowError):
                first = None                                    # beyond the first slice's own limits
            for k in (1, 2, 3, 5, 8):                            # one to eight seed directions
                U = [[rand_q(rng, -2, 2, 3) for _ in range(P)] for _ in range(k)]
                U[0][0] = Fraction(1)                            # a nonzero direction
                center = [rand_q(rng) for _ in range(P)]
                xs = [[int(rng.integers(-5, 6)) for _ in range(D)] for _ in range(3)]
                plan, jets, _ = self.check_against_oracles(nodes, P, D, center, U, xs, 1)
                for jet in jets:                                 # the dual layout: [value, D_u1 f, ..., D_uk f]
                    self.assertEqual(jet.directional(), [jet.taylor[1 + j] for j in range(k)])
                    self.assertEqual(len(jet.taylor), k + 1)
                if first is not None and k == P:                 # the first slice's derivative, directions U
                    expr, names = first
                    for x, jet in zip(xs, jets):
                        b = dict(zip(names, center + x))
                        grads = [first_slice_partial(expr, names, tuple(int(i == p) for i in range(P)), b) for p in range(P)]
                        for j in range(k):
                            self.assertEqual(jet.directional()[j], sum(U[j][p] * grads[p] for p in range(P)))
                            first_slice_checks += 1
            if first is not None:                                # identity directions: the gradient
                center = [rand_q(rng) for _ in range(P)]
                xs = [[int(rng.integers(-4, 5)) for _ in range(D)] for _ in range(2)]
                U = [[Fraction(int(i == j)) for i in range(P)] for j in range(P)]
                _, _, n = self.check_against_oracles(nodes, P, D, center, U, xs, 1, first=first, identity=True)
                first_slice_checks += n
        self.assertGreater(first_slice_checks, 50)

    def test_jet001_sum_leibniz_and_chain_laws(self):
        rng = np.random.default_rng(1002)
        b = mj.Builder(params=["a", "c", "e"], data=["x", "y"])
        a, c, e = b.params
        x, y = b.data
        f = a * x + c * c - Fraction(2, 3) * e * y
        g = (a - e) * (c + y) + 5
        n = 4
        exprs = {"f": f, "g": g, "f+g": f + g, "f-g": f - g, "f*g": f * g, "f^n": f ** n, "-f": -f, "3f": 3 * f}
        for trial in range(4):
            center = {k: rand_q(rng) for k in ("a", "c", "e")}
            dirs = [[rand_q(rng, -2, 2, 2) for _ in range(3)] for _ in range(int(rng.integers(1, 9)))]
            point = {"x": int(rng.integers(-6, 7)), "y": int(rng.integers(-6, 7))}
            J = {}
            for name, ex in exprs.items():
                plan = b.build(ex).plan(1, center, dirs)
                J[name] = plan.at(point).taylor
                self.assertEqual(plan.at(point, backend="cuda").taylor, J[name])
            F, G, k = J["f"], J["g"], len(dirs)
            self.assertEqual(J["f+g"], [u + v for u, v in zip(F, G)])                       # the sum law
            self.assertEqual(J["f-g"], [u - v for u, v in zip(F, G)])
            self.assertEqual(J["f*g"], [F[0] * G[0]] + [F[0] * G[j] + F[j] * G[0] for j in range(1, k + 1)])   # Leibniz
            self.assertEqual(J["f^n"], [F[0] ** n] + [n * F[0] ** (n - 1) * F[j] for j in range(1, k + 1)])      # chain
            self.assertEqual(J["-f"], [-u for u in F])
            self.assertEqual(J["3f"], [3 * u for u in F])
        # data columns are fixed observations: never differentiated, whatever the directions
        circ = b.build(x * y + 7)
        jet = circ.plan(2, {"a": 1}, [[1, 2, 3], [0, 1, 0]]).at({"x": 3, "y": 4})
        self.assertEqual(jet.taylor, [19] + [0] * (len(jet.taylor) - 1))
        # a parameter without a seeded direction is a constant at its center (a scalar of the program)
        circ = b.build(a * c + e)
        plan = circ.plan(1, {"a": 2, "c": 5, "e": 1}, [[0, 1, 0]])
        self.assertEqual(plan.at({"x": 0, "y": 0}).taylor, [11, 2])
        steps = plan.program()
        self.assertEqual([s["op"] for s in steps if s["jet"] & 4], ["param", "mul", "add"])
        # 0^0 = 1, and a^1 = a
        circ = b.build((a * 0) ** 0 + a ** 1)
        self.assertEqual(circ.plan(1, {"a": 0}, [[1, 0, 0]]).at({"x": 1, "y": 1}).taylor, [1, 1])

    def test_jet001_layout_and_plan_metadata(self):
        b = mj.Builder(params=[f"t{i}" for i in range(8)], data=["x"])
        t, (x,) = b.params, b.data
        sparse = sum((t[i] * x for i in range(1, 8)), t[0] * x)          # every parameter in one term
        dense = (sum(t[1:], t[0]) * x + 1) ** 3                          # every interaction present
        for d in (1, 2, 3):
            ps = b.build(sparse).plan(d)
            pd_ = b.build(dense).plan(d)
            self.assertEqual(list(ps.monomials), monomials(8, d))
            self.assertEqual(ps.info["support"], math.comb(8 + d, d))
            self.assertEqual(ps.info["pairs"], math.comb(16 + d, d))
            self.assertLess(ps.info["multiplications"], pd_.info["multiplications"])
            jet = pd_.at({"x": 2})
            for alpha in monomials(8, d):                               # (2 sum t + 1)^3 at t = 0
                k = sum(alpha)
                self.assertEqual(jet.coefficient(alpha), math.comb(3, k) * 2 ** k * math.factorial(k) // factorial(alpha))
        plan = b.build(sparse).plan(1)
        self.assertEqual(plan.monomials[:9], tuple(tuple(int(i == j) for i in range(8)) for j in range(-1, 8)))
        self.assertEqual(plan.info["denominator"], 1)
        q = mj.Builder(params=["a"], data=["d"])
        a, dcol = q.params[0], q.data[0]
        plan = q.build(a * Fraction(1, 3) + dcol * a).plan(1, {"a": Fraction(1, 2)}, scales=[2])
        dens = plan.denominators()
        self.assertEqual(plan.info["denominator"], 600)                 # lcm(3 * 2, 100 * 2)
        self.assertEqual(max(dens), 600)
        self.assertEqual(plan.at({"d": 150}).taylor, [Fraction(1, 6) + Fraction(3, 4), Fraction(1, 3) + Fraction(3, 2)])

    def test_jet001_aggregates_over_tables_match_duckdb(self):
        rng = np.random.default_rng(1004)
        n = 900
        k = [int(v) for v in rng.integers(0, 6, n)]
        x = [int(v) for v in rng.integers(-40, 41, n)]
        d = [f"{int(v) / 100:.2f}" for v in rng.integers(-5000, 5001, n)]
        for col, p in ((k, 0.05), (x, 0.1), (d, 0.1)):
            for i in np.flatnonzero(rng.random(n) < p):
                col[int(i)] = None
        for i in range(5):                                             # group 7: present, every value NULL
            k[i], x[i] = 7, None
        frame = pd.DataFrame({"k": obj(k), "x": obj(x), "d": obj(d), "z": obj(list(range(n)))})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT, d DECIMAL(10,2), z BIGINT)", {"t": frame})
        t = db["t"]
        b = mj.Builder(params=["a", "c"], data=["x", "d"])
        a, c = b.params
        xx, dd = b.data
        circuit = b.build((a * xx + c * dd) * (a * xx + c * dd) + 3 * a - dd)
        center = {"a": 2, "c": Fraction(-1, 2)}
        sel = rng.random(n) < 0.8
        db.con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
        valid = 'x IS NOT NULL AND d IS NOT NULL'
        e = "(2 * x - 0.5 * d)"
        sql = (f"SELECT k, COUNT(*), COUNT(*) FILTER (WHERE {valid}), "
               f"SUM({e} * {e} + 6 - d) FILTER (WHERE {valid}), "
               f"SUM(2 * {e} * x + 3) FILTER (WHERE {valid}), SUM(2 * {e} * d) FILTER (WHERE {valid}) "
               "FROM t WHERE rowid IN (SELECT rid FROM sel_rows) GROUP BY k")
        expect = {}
        for r in db.con.execute(sql).fetchall():
            expect[(r[0],)] = (r[1], r[2], [Fraction(Decimal(v)) if v is not None else Fraction(0) for v in r[3:]])
        for backend in ("cpu", "cuda"):
            agg = circuit.aggregate(t, center, 1, rows=sel, group_by=["k"], backend=backend)
            scope = agg.scope
            self.assertEqual(scope.column("d")["scale"], 2)
            got = {}
            for g, jet in enumerate(agg.jets()):
                if jet.present:
                    got[scope.key_values(g)] = (jet.rows, jet.count, jet.taylor)
            self.assertEqual(got, expect, backend)
            self.assertEqual(agg.info["rows"], int(sel.sum()))
            s7 = got[(7,)]
            self.assertEqual((s7[1], s7[2]), (0, [0, 0, 0]))           # present, no valid row: count 0, zero jet
        cpu = circuit.aggregate(t, center, 1, rows=sel, group_by=["k"])
        gpu = circuit.aggregate(t, center, 1, rows=sel, group_by=["k"], backend="cuda")
        self.assertTrue(cpu.equal(gpu))
        absent = [g for g in range(cpu.groups) if not cpu.jet(g).present]
        self.assertTrue(absent)                                        # groups without selected rows: rows 0
        for g in absent:
            self.assertEqual((cpu.jet(g).rows, cpu.jet(g).count, set(cpu.jet(g).taylor)), (0, 0, {0}))

    def test_jet001_merged_aggregate_states_keep_every_component(self):
        rng = np.random.default_rng(1005)
        P, D, n, G = 3, 2, 400, 7
        nodes = random_circuit(rng, P, D, size=14, maxdeg=4)
        names, dnames = [f"p{i}" for i in range(P)], [f"x{i}" for i in range(D)]
        circuit = mj.Circuit.from_nodes(nodes, names, dnames)
        f = expand(nodes, P, D)
        xs = [rng.integers(-6, 7, n) for _ in range(D)]
        valid = {dnames[0]: rng.random(n) < 0.85}
        groups = rng.integers(0, G, n)
        rows = rng.random(n) < 0.9
        scope = mbk.Scope.from_arrays(dict(zip(dnames, xs)), valid=valid, groups=groups, ngroups=G, rows=rows)
        center = [rand_q(rng) for _ in range(P)]
        checked = dnames[0] in circuit.info["reads_columns"]           # validity matters only for a column f reads
        for k, order in ((3, 1), (8, 1), (2, 2)):
            U = [[rand_q(rng, -2, 2, 2) for _ in range(P)] for _ in range(k)]
            plan = circuit.plan(order, center, U)
            one = self.both(plan, scope)
            # the oracle: sum over the selected valid rows of each group of the exact row jets
            oracle = [{a: Fraction(0) for a in monomials(k, order)} for _ in range(G)]
            for r in range(n):
                if not rows[r] or (checked and not valid[dnames[0]][r]):
                    continue
                jet = derivative_oracle(f, P, D, center, U, [int(v[r]) for v in xs], order)
                for a in jet:
                    oracle[int(groups[r])][a] += jet[a] / factorial(a)
            for g in range(G):
                self.assertEqual(jet_dict(one.jet(g)), oracle[g], g)
            # chunked accumulation over random disjoint ranges, then merged builders: bit for bit the one-shot states
            cuts = sorted({0, n, *[int(v) for v in rng.integers(0, n, 5)]})
            left, right = plan.builder(scope), plan.builder(scope)
            for i, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
                (left if i % 2 else right).accumulate(lo, hi)
            merged = left.merge(right).finish()
            self.assertTrue(merged.equal(one))
            with self.assertRaises(ValueError):                         # a row is never counted twice
                plan.builder(scope).accumulate(0, 10).accumulate(5, 20)
            # partitions of the rows (disjoint selections, same groups) added: the aggregate of the union
            part = rng.integers(0, 3, n)
            aggs = [plan.aggregate(mbk.Scope.from_arrays(dict(zip(dnames, xs)), valid=valid, groups=groups, ngroups=G,
                                                         rows=rows & (part == i))) for i in range(3)]
            total = aggs[0] + aggs[1] + aggs[2]
            self.assertEqual(total.coefficients(), one.coefficients())
            self.assertEqual(total.arrays()[0].tolist(), one.arrays()[0].tolist())
            self.assertEqual(total.arrays()[1].tolist(), one.arrays()[1].tolist())
            self.assertFalse(total.info["scoped"])
        # components that live in one part only survive the merge (no sparse dropping)
        b = mj.Builder(params=["u", "v"], data=["x", "y"])
        u, v = b.params
        x, y = b.data
        plan = b.build(u * x + v * y).plan(1)
        sa = mbk.Scope.from_arrays({"x": [3, 4, 5], "y": [0, 0, 0]})          # only d/du is nonzero here
        sb = mbk.Scope.from_arrays({"x": [0, 0], "y": [-7, 2]})               # only d/dv here
        ja, jb = plan.aggregate(sa), plan.aggregate(sb, backend="cuda")
        self.assertEqual((ja.jet(0).taylor, jb.jet(0).taylor), ([0, 12, 0], [0, 0, -5]))
        both = ja + jb
        self.assertEqual((both.jet(0).taylor, both.jet(0).rows, both.jet(0).count), ([0, 12, -5], 5, 5))
        # aggregates of other plans or group counts never merge
        other = b.build(u * x + v * y).plan(1, {"u": 1})
        with self.assertRaises(DomainError):
            ja + other.aggregate(sa)
        with self.assertRaises(DomainError):
            ja + plan.aggregate(mbk.Scope.from_arrays({"x": [1], "y": [1]}, groups=[1], ngroups=2))
        with self.assertRaises(DomainError):
            plan.builder(sa).merge(plan.builder(sb))                    # builders of another scope

    def test_jet001_carrier_conversions_keep_every_component_or_refuse(self):
        rng = np.random.default_rng(1006)
        b = mj.Builder(params=["a", "c"], data=["x"])
        a, c = b.params
        x = b.data[0]
        circuit = b.build(Fraction(1, 7) * (a * x + c) ** 3 + Fraction(5, 3) * a * c)
        plan = circuit.plan(2, {"a": Fraction(1, 2), "c": -1}, [[1, 1], [Fraction(1, 5), 0], [0, 3]])
        xs = rng.integers(-30, 31, 300)
        groups = rng.integers(0, 4, 300)
        scope = mbk.Scope.from_arrays({"x": xs}, groups=groups, ngroups=4)
        agg = self.both(plan, scope)
        exact = agg.coefficients()
        exact_d = agg.coefficients("derivative")
        S = agg.support
        q64 = agg.values("q64")
        big = agg.values("bigq")
        self.assertEqual((q64.carrier, big.carrier, q64.exact, big.exact), ("q64", "bigq", True, True))
        self.assertEqual(q64.components(), exact)
        self.assertEqual(big.components(), exact)
        self.assertEqual(len(q64.components()[0]), S)                   # every component of every jet
        self.assertTrue(q64.convert("bigq").equal(big))                  # lossless
        self.assertTrue(big.convert("q64").equal(q64))                   # narrowing that fits: back unchanged
        self.assertEqual(big.convert("bigq").components(), exact)
        dv = agg.values("q64", convention="derivative")
        self.assertEqual(dv.components(), exact_d)
        self.assertEqual(exact_d, [[factorial(al) * v for al, v in zip(agg.monomials, row)] for row in exact])
        # GF(p): every component reduced (n / d mod p), also through q64 / bigq
        for p in (P61, 1000003):
            red = agg.values("gfp", p=p)
            want = [[(v.numerator * pow(v.denominator, -1, p)) % p for v in row] for row in exact]
            self.assertEqual((red.components(), red.p), (want, p))
            self.assertTrue(q64.convert("gfp", p=p).equal(red))
            self.assertTrue(big.convert("gfp", p=p).equal(red))
        with self.assertRaises(NoSolutionError):                          # p divides the denominator (7 * ...)
            agg.values("gfp", p=7)
        with self.assertRaises(ValueError):
            agg.values("gfp", p=91)                                       # not a prime
        with self.assertRaises(UnsupportedError):
            agg.values("gfp", p=(1 << 62) + 135)                          # a prime >= 2^62
        # intervals: tight enclosures, approximate and certified
        iv = agg.values("interval")
        self.assertTrue(iv.approximate)
        for row, ex in zip(iv.components(), exact):
            for (lo, hi), v in zip(row, ex):
                self.assertTrue(Fraction(lo) <= v <= Fraction(hi))
                self.assertTrue(lo == hi or math.nextafter(lo, math.inf) == hi)
        self.assertTrue(big.convert("interval").equal(iv))
        # forbidden conversions: nothing exact comes back from an approximate or modular carrier
        red = agg.values("gfp", p=P61)
        for src, dst in ((red, "q64"), (red, "bigq"), (red, "interval"), (iv, "q64"), (iv, "bigq"), (iv, "gfp")):
            with self.assertRaises(CarrierError):
                src.convert(dst, p=P61)
        with self.assertRaises(CarrierError):
            red.convert("gfp", p=1000003)                                 # GF(p) -> GF(q)
        # the sum law in each carrier commutes with the merge of aggregates
        part = rng.integers(0, 2, 300)
        parts = [plan.aggregate(mbk.Scope.from_arrays({"x": xs}, groups=groups, ngroups=4, rows=part == i))
                 for i in range(2)]
        whole = parts[0] + parts[1]
        self.assertTrue(whole.equal(agg))
        for carrier, p in (("q64", None), ("bigq", None), ("gfp", P61)):
            summed = parts[0].values(carrier, p=p) + parts[1].values(carrier, p=p)
            self.assertEqual(summed.components(), agg.values(carrier, p=p).components(), carrier)
        isum = parts[0].values("interval") + parts[1].values("interval")
        for row, ex in zip(isum.components(), exact):
            for (lo, hi), v in zip(row, ex):
                self.assertTrue(Fraction(lo) <= v <= Fraction(hi))
        with self.assertRaises(CarrierError):
            parts[0].values("q64") + parts[1].values("bigq")
        with self.assertRaises(DomainError):
            parts[0].values("q64") + parts[1].values("q64", convention="derivative")
        # wide aggregates: q64 narrowing refuses (nothing is returned), bigq keeps the exact value
        wb = mj.Builder(params=["a"], data=["x"])
        wa, wx = wb.params[0], wb.data[0]
        wide = wb.build(wa * wx * wx).plan(1, {"a": 1})
        big_rows = np.full(64, (1 << 31) - 1, dtype=np.int64)
        wagg = wide.aggregate(mbk.Scope.from_arrays({"x": big_rows}), backend="cuda")
        want = 64 * ((1 << 31) - 1) ** 2
        self.assertEqual(wagg.jet(0).taylor, [want, want])               # 64 (2^31 - 1)^2 > 2^63
        with self.assertRaises(OverflowError):
            wagg.values("q64")
        self.assertEqual(wagg.values("bigq").components(), [[want, want]])
        self.assertEqual(wagg.values("bigq", convention="derivative").components(), [[want, want]])
        self.assertEqual(wagg.values("gfp", p=P61).components(), [[want % P61, want % P61]])
        self.assertTrue(all(Fraction(lo) <= v <= Fraction(hi) for (lo, hi), v in
                            zip(wagg.values("interval").components()[0], [want, want])))

    def test_jet001_overflow_is_refused_never_wrapped(self):
        b = mj.Builder(params=["a"], data=["x"])
        a, x = b.params[0], b.data[0]
        circuit = b.build((a * x) ** 3)
        plan = circuit.plan(1, {"a": 1})
        ok = mbk.Scope.from_arrays({"x": [1_000_000]})                    # J = [x^3, 3 x^3], 3e18 < 2^63
        self.assertEqual(self.both(plan, ok).jet(0).taylor, [10 ** 18, 3 * 10 ** 18])
        for xs in ([1_500_000], [5, -3, 1_500_000, 7]):                   # 3 x^3 = 1.01e19 > 2^63 - 1: refused
            scope = mbk.Scope.from_arrays({"x": xs})
            for backend in ("cpu", "cuda"):
                with self.assertRaises(OverflowError):
                    plan.aggregate(scope, backend=backend)
        # a NULL or unselected row never decides a refusal
        scope = mbk.Scope.from_arrays({"x": [5, 1_500_000]}, valid={"x": [True, False]})
        self.assertEqual(self.both(plan, scope).jet(0).taylor, [125, 375])
        scope = mbk.Scope.from_arrays({"x": [5, 1_500_000]}, rows=np.array([True, False]))
        self.assertEqual(self.both(plan, scope).jet(0).taylor, [125, 375])
        # the plan's fixed denominators: refused when they leave int64
        q = mj.Builder(params=["a"], data=[])
        with self.assertRaises(OverflowError):
            q.build((Fraction(1, 3) * q.params[0]) ** 40).plan(1)          # 3^40 > 2^63
        self.assertEqual(q.build((Fraction(1, 3) * q.params[0]) ** 37).plan(1, {"a": 3}).at().taylor,
                         [1, Fraction(37, 3)])                           # D = 3^37, J1 numerator 37 * 3^36
        # the chunked builder that refused is spent
        bld = plan.builder(mbk.Scope.from_arrays({"x": [1, 1_500_000, 3]}))
        with self.assertRaises(OverflowError):
            bld.accumulate(0, 3)
        with self.assertRaises(ValueError):
            bld.finish()
        # inputs that do not fit the carriers are refused before anything runs
        with self.assertRaises(OverflowError):
            circuit.plan(1, {"a": 1 << 63})
        with self.assertRaises(TypeError):
            circuit.plan(1, {"a": 0.5})

    def test_jet001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(1008)
        P, D = 3, 2
        nodes = random_circuit(rng, P, D, size=16, maxdeg=4)
        names, dnames = [f"p{i}" for i in range(P)], [f"x{i}" for i in range(D)]
        # the circuit: variables = params + columns, circuit = nodes, bytes
        needs = mj.Circuit.need(nodes, P, D)
        self.assertEqual((needs["variables"], needs["circuit"]), (P + D, len(nodes)))
        for resource, value in needs.items():
            mj.Circuit.from_nodes(nodes, names, dnames, budget=Budget(**{resource: value}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mj.Circuit.from_nodes(nodes, names, dnames, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        circuit = mj.Circuit.from_nodes(nodes, names, dnames)
        # the plan: variables = directions, degree = order, support, tensor, circuit, bytes
        U = [[rand_q(rng) for _ in range(P)] for _ in range(4)]
        needs = circuit.plan_need(1, 4)
        self.assertEqual((needs["variables"], needs["degree"], needs["support"], needs["tensor"]),
                         (4, 1, 5, math.comb(9, 1)))
        ref = circuit.plan(1, None, U).at({"x0": 1, "x1": 2}).taylor
        for resource, value in needs.items():
            self.assertEqual(circuit.plan(1, None, U, budget=Budget(**{resource: value})).at({"x0": 1, "x1": 2}).taylor, ref)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                circuit.plan(1, None, U, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        # the aggregate: support, states = groups x S, bytes; on the CPU and with the device copies
        plan = circuit.plan(1, None, U)
        scope = mbk.Scope.from_arrays({"x0": rng.integers(-3, 4, 200), "x1": rng.integers(-3, 4, 200)},
                                      groups=rng.integers(0, 9, 200), ngroups=9)
        refj = plan.aggregate(scope)
        for backend in ("cpu", "cuda"):
            needs = plan.need(scope, backend)
            self.assertEqual((needs["states"], needs["support"]), (9 * 5, 5))
            for resource, value in needs.items():
                self.assertTrue(plan.aggregate(scope, backend=backend, budget=Budget(**{resource: value})).equal(refj))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    plan.aggregate(scope, backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        self.assertGreater(plan.need(scope, "cuda")["bytes"], plan.need(scope, "cpu")["bytes"])
        # the device copies alone, at the native entry point
        dev = plan._need(scope, 1)
        dbytes = int(dev.need[8])
        gpu = mj._gpu()
        for limit, code in ((dbytes, 0), (dbytes - 1, -5)):
            G, S = scope.groups, plan.support
            rows, cnt, sums = np.zeros(G, np.uint64), np.zeros(G, np.uint64), np.zeros(2 * G * S, np.int64)
            st = mj._States(G, S, rows.ctypes.data, cnt.ctypes.data, sums.ctypes.data)
            refusal, view, sview = mj._Refusal(), plan.view(), scope.view()
            rc = gpu.lib().tr_mgpu_jet_aggregate(C.byref(view), C.byref(sview), C.byref(Budget(bytes=limit).raw),
                                                 C.byref(st), C.byref(refusal))
            self.assertEqual(rc, code)
            if code:
                self.assertEqual((refusal.resource, refusal.need), (8, dbytes))
        # the values: output = jets x support
        with self.assertRaises(AlgebraBudgetError) as ctx:
            refj.values("bigq", budget=Budget(output=9 * 5 - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 45))
        self.assertEqual(len(refj.values("q64", budget=Budget(output=45)).components()), 9)
        # representation limits (a budget that admits them): UnsupportedError
        big = Budget(variables=1 << 20, degree=1 << 20, support=1 << 40, tensor=1 << 40, circuit=1 << 30)
        many = [("param", -1, -1, i, (0, 1)) for i in range(65)] + [("add", 0, 1, 0, (0, 1))]
        with self.assertRaises(UnsupportedError):
            mj.Circuit.from_nodes(many, [f"p{i}" for i in range(65)], budget=big)
        with self.assertRaises(UnsupportedError):
            mj.Circuit.from_nodes([("param", -1, -1, 0, (0, 1)), ("pow", 0, -1, 256, (0, 1))], ["a"])
        with self.assertRaises(ValueError):
            b0 = mj.Builder(params=["a"]); b0.params[0] ** 256

    def test_jet001_null_validity_presence_and_staleness(self):
        b = mj.Builder(params=["a"], data=["x", "y"])
        a = b.params[0]
        x, y = b.data
        plan = b.build(a * x + y ** 0).plan(1, {"a": 2})                  # y only under a^0: still decides validity
        scope = mbk.Scope.from_arrays({"x": [1, 2, 3, 4], "y": [9, 9, 9, 9]},
                                      valid={"x": [True, True, False, True], "y": [True, False, True, True]},
                                      groups=[0, 0, 1, 2], ngroups=4)
        agg = self.both(plan, scope)
        rows, count = agg.arrays()[:2]
        self.assertEqual((rows.tolist(), count.tolist()), ([2, 1, 1, 0], [1, 0, 1, 0]))
        self.assertEqual([agg.jet(g).taylor for g in range(4)], [[3, 1], [0, 0], [9, 4], [0, 0]])
        self.assertEqual([agg.jet(g).present for g in range(4)], [True, True, True, False])
        empty = self.both(plan, mbk.Scope.from_arrays({"x": [1, 2], "y": [1, 1]}, rows=np.zeros(2, bool)))
        self.assertEqual((empty.info["rows"], empty.jet(0).taylor, empty.jet(0).present), (0, [0, 0], False))
        # a table scope goes stale when the table changes; scopes of other columns or scales are refused
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y DECIMAL(6,1))",
                              {"t": pd.DataFrame({"x": obj([1, 2, 3]), "y": obj(["1.5", "2.0", None])})})
        t = db["t"]
        circ = b.build(a * x * y)
        agg = circ.aggregate(t, {"a": 1}, 1)
        self.assertEqual(agg.jet(0).taylor, [Fraction(11, 2), Fraction(11, 2)])
        tscope = mbk.Scope.from_table(t, ["x", "y"])
        circ.plan(1, {"a": 1}, scales=[0, 1]).aggregate(tscope)
        with self.assertRaises(DomainError):
            circ.plan(1, {"a": 1}).aggregate(tscope)                       # scale 0 for a DECIMAL(6,1) column
        with self.assertRaises(DomainError):
            circ.plan(1).aggregate(mbk.Scope.from_arrays({"x": [1]}))      # one column for two
        t.flip(0, 0)
        with self.assertRaises(StaleStateError):
            circ.plan(1, {"a": 1}, scales=[0, 1]).aggregate(tscope)
        # malformed input
        for bad in ([("add", 0, 1, 0, (0, 1))],                           # an operand that is not earlier
                    [("param", -1, -1, 3, (0, 1))],                         # no parameter 3
                    [("const", -1, -1, 0, (1, 0))],                         # a zero denominator
                    [("param", -1, -1, 0, (0, 1)), ("neg", 0, 0, 0, (0, 1))],   # b set on a unary node
                    [("const", 5, -1, 0, (1, 1))]):                         # an operand on a leaf
            with self.assertRaises(ValueError):
                mj.Circuit.from_nodes(bad, ["a"])
        with self.assertRaises(ValueError):
            mj.Circuit.from_nodes([], ["a"])
        with self.assertRaises(ValueError):
            plan.at({"x": 1})                                              # a missing column
        with self.assertRaises(ValueError):
            b.build(a).plan(1, {"zz": 1})                                   # an unknown parameter
        code = "import sys, torbles; print('torbles.math_jet' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])

    def test_jet001_cuda_splits_long_scans_exactly(self):
        # more rows than one launch takes (TR_JET_LAUNCH_ROWS = 2^20): several launches, the same exact states
        rng = np.random.default_rng(1010)
        n, G = (1 << 20) + 3000, 3
        xs = rng.integers(-1000, 1001, n)
        groups = rng.integers(0, G, n).astype(np.uint32)
        b = mj.Builder(params=["a", "c"], data=["x"])
        a, c = b.params
        x = b.data[0]
        plan = b.build(a * x + c * x * x).plan(1, {"a": 1, "c": 2})
        scope = mbk.Scope.from_arrays({"x": xs}, groups=groups, ngroups=G)
        agg = self.both(plan, scope)
        x64 = xs.astype(object)
        for g in range(G):
            sel = x64[groups == g]
            s1, s2 = int(sum(sel)), int(sum(v * v for v in sel))
            self.assertEqual(agg.jet(g).taylor, [s1 + 2 * s2, s1, s2])
            self.assertEqual((agg.jet(g).rows, agg.jet(g).count), (len(sel), len(sel)))
        # a wide plan (20 directions, order 3: S = 1771) leaves few work items under the workspace cap, so launches
        # take at most 64 rows per item: several launches over 350k rows
        q = mj.Builder(params=[f"t{i}" for i in range(20)], data=["x"])
        wide = q.build(sum(q.params[1:], q.params[0]) * q.data[0]).plan(3, {f"t{i}": i for i in range(20)})
        info = wide.info
        per = 8 * (info["workspace_words"] + 3 * info["support"])
        n2 = 350_000
        items = min((n2 + 15) // 16, 1 << 16, (128 << 20) // per)
        self.assertLess(items * 64 * 3, n2)                              # at least three launches
        xs2 = rng.integers(-100, 101, n2)
        agg2 = self.both(wide, mbk.Scope.from_arrays({"x": xs2}))
        sx = int(xs2.astype(object).sum())
        self.assertEqual(agg2.jet(0).taylor[:21], [sum(range(20)) * sx] + [sx] * 20)
        self.assertEqual(set(agg2.jet(0).taylor[21:]), {0})

    def test_jet001_abi_registered_structs_match(self):
        from torbles import math_handles as mh
        report = mh.abi_check()
        checked = {(r["native"], r["library"]) for r in report["checked"]}
        for name in ("tr_jet_node", "tr_jet_circuit_info", "tr_jet_step", "tr_jet_view", "tr_jet_plan_info",
                     "tr_jet_states", "tr_jet_agg_info", "tr_jet_values_info"):
            self.assertIn((name, "libalg"), checked)
        for name in ("tr_jet_view", "tr_jet_step", "tr_jet_states"):
            self.assertIn((name, "cuda"), checked)

    # ============================================================================== JET-002
    def test_jet002_second_and_third_order_match_exact_derivatives(self):
        rng = np.random.default_rng(2001)
        first_checks = 0
        for trial in range(8):
            P, D = int(rng.integers(1, 4)), int(rng.integers(0, 3))
            nodes = random_circuit(rng, P, D, size=int(rng.integers(8, 15)), maxdeg=5)
            try:
                first = first_slice(nodes, P, D)
            except (AlgebraBudgetError, OverflowError):
                first = None
            for order in (2, 3):
                center = [rand_q(rng) for _ in range(P)]
                xs = [[int(rng.integers(-4, 5)) for _ in range(D)] for _ in range(2)]
                I = [[Fraction(int(i == j)) for i in range(P)] for j in range(P)]
                plan, jets, n = self.check_against_oracles(nodes, P, D, center, I, xs, order, first=first, identity=True)
                first_checks += n
                for jet in jets:                                    # Hessians are symmetric; diagonal = 2 J[2 e_i]
                    H = jet.hessian()
                    for i in range(P):
                        for j in range(P):
                            self.assertEqual(H[i][j], H[j][i])
                        self.assertEqual(H[i][i], 2 * jet.coefficient(tuple(2 * (q == i) for q in range(P))))
                    if order == 3 and P >= 2:
                        self.assertEqual(jet.third(0, 0, 1), 2 * jet.coefficient((2, 1) + (0,) * (P - 2)))
                        self.assertEqual(jet.third(0, 1, 0), jet.third(1, 0, 0))
                k = int(rng.integers(1, 5))                          # random rational directions
                U = [[rand_q(rng, -2, 2, 3) for _ in range(P)] for _ in range(k)]
                self.check_against_oracles(nodes, P, D, center, U, xs, order)
        self.assertGreater(first_checks, 100)

    def test_jet002_hessian_vector_products_and_mixed_partials(self):
        rng = np.random.default_rng(2002)
        for trial in range(5):
            P, D = int(rng.integers(2, 5)), 1
            nodes = random_circuit(rng, P, D, size=14, maxdeg=4)
            circuit = mj.Circuit.from_nodes(nodes, [f"p{i}" for i in range(P)], ["x0"])
            f = expand(nodes, P, D)
            center = [rand_q(rng) for _ in range(P)]
            v = [rand_q(rng, -3, 3, 2) for _ in range(P)]
            for x in (-2, 3):
                point = center + [x]
                H = [[peval(pderiv(pderiv(f, i), j), point) for j in range(P)] for i in range(P)]
                Hv = [sum(H[i][j] * v[j] for j in range(P)) for i in range(P)]
                for backend in ("cpu", "cuda"):
                    self.assertEqual(circuit.hvp(v, {"x0": x}, center, backend=backend), Hv)
                # the same jet holds v^T H v = 2 J[2 e_v] and the gradient along v
                dirs = [[int(i == j) for i in range(P)] for j in range(P)] + [v]
                jet = circuit.plan(2, center, dirs).at({"x0": x})
                vHv = sum(v[i] * Hv[i] for i in range(P))
                self.assertEqual(jet.derivative(tuple([0] * P + [2])), vHv)
                self.assertEqual(jet.coefficient(tuple([0] * P + [2])), vHv / 2)
                grad = [peval(pderiv(f, i), point) for i in range(P)]
                self.assertEqual(jet.gradient()[P], sum(gi * vi for gi, vi in zip(grad, v)))
                # mixed partials carry no factorial; the first slice's double derivative agrees
                try:
                    expr, names = first_slice(nodes, P, D)
                except (AlgebraBudgetError, OverflowError):
                    continue
                b = dict(zip(names, point))
                for i in range(P):
                    for j in range(i + 1, P):
                        e = tuple(int(q == i) + int(q == j) for q in range(P))
                        mixed = first_slice_partial(expr, names, e, b)
                        self.assertEqual(jet.coefficient(e + (0,)), mixed)
                        self.assertEqual(jet.derivative(e + (0,)), mixed)

    def test_jet002_factorial_conventions_at_zero_and_nonzero_centers(self):
        rng = np.random.default_rng(2003)
        b = mj.Builder(params=["a"], data=[])
        a = b.params[0]
        cube = b.build(a ** 3)
        for c in (0, 2, Fraction(-1, 3)):
            jet = cube.plan(3, {"a": c}).at()
            self.assertEqual(jet.taylor, [c ** 3, 3 * c ** 2, 3 * c, 1])          # Taylor (Hasse): C(3, k) c^(3-k)
            self.assertEqual(jet.derivatives, [c ** 3, 3 * c ** 2, 6 * c, 6])     # derivatives: k! times
            self.assertEqual(cube.plan(3, {"a": c}).at(backend="cuda").taylor, jet.taylor)
        # a random polynomial: at the zero center (U = I) the Taylor coefficients are f's coefficients in theta
        for trial in range(6):
            P, D = int(rng.integers(1, 4)), int(rng.integers(0, 2))
            nodes = random_circuit(rng, P, D, size=12, maxdeg=5)
            circuit = mj.Circuit.from_nodes(nodes, [f"p{i}" for i in range(P)], [f"x{i}" for i in range(D)])
            f = expand(nodes, P, D)
            x = [int(rng.integers(-3, 4)) for _ in range(D)]
            coeffs = {}                                               # f with the data substituted, by theta exponent
            for e, v in f.items():
                key = e[:P]
                coeffs[key] = coeffs.get(key, 0) + v * math.prod(Fraction(xx) ** k for xx, k in zip(x, e[P:]))
            for d in (2, 3):
                plan = circuit.plan(d)
                jet = self.points(plan, [x])[0]
                for alpha in monomials(P, d):
                    self.assertEqual(jet.coefficient(alpha), coeffs.get(alpha, 0), alpha)
                    self.assertEqual(jet.derivative(alpha), factorial(alpha) * coeffs.get(alpha, 0), alpha)
                # at a nonzero center the coefficients are the binomially shifted ones: sum_beta f_beta C(beta, alpha)
                # c^(beta - alpha), which is the Taylor convention of g(t) = f(c + t)
                center = [rand_q(rng, 1, 3, 2) for _ in range(P)]
                jet = self.points(circuit.plan(d, center), [x])[0]
                for alpha in monomials(P, d):
                    shifted = sum((v * math.prod(math.comb(bb, aa) * Fraction(cc) ** (bb - aa)
                                                 for bb, aa, cc in zip(beta, alpha, center))
                                   for beta, v in coeffs.items() if all(bb >= aa for bb, aa in zip(beta, alpha))),
                                  Fraction(0))
                    self.assertEqual(jet.coefficient(alpha), shifted, alpha)
                    self.assertEqual(jet.derivative(alpha), factorial(alpha) * shifted, alpha)
                agg = circuit.plan(d, center).evaluate({f"x{i}": [x[i]] for i in range(D)}, n=1)
                self.assertEqual(agg.values("bigq", convention="derivative").components(), [jet.derivatives])
                self.assertEqual(agg.values("bigq").components(), [jet.taylor])

    def test_jet002_dimension_and_order_budgets(self):
        b = mj.Builder(params=[f"t{i}" for i in range(6)], data=["x"])
        t = b.params
        circuit = b.build((sum(t[1:], t[0]) + b.data[0]) ** 4)
        for m, d in ((6, 3), (2, 4), (6, 0), (0, 2)):
            dirs = [[int(i == j) for i in range(6)] for j in range(m)]
            needs = circuit.plan_need(d, m)
            self.assertEqual((needs.get("variables", 0), needs.get("degree", 0), needs["support"], needs["tensor"]),
                             (m, d, math.comb(m + d, d), math.comb(2 * m + d, d)))
            ref = circuit.plan(d, None, dirs).at({"x": 1}).taylor
            self.assertEqual(len(ref), math.comb(m + d, d))
            for resource, value in needs.items():
                self.assertEqual(circuit.plan(d, None, dirs, budget=Budget(**{resource: value})).at({"x": 1}).taylor, ref)
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    circuit.plan(d, None, dirs, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (m, d, resource))
        # the dimension (directions) and the order refused one past a budget, in the fixed resource order
        dirs = [[int(i == j) for i in range(6)] for j in range(6)]
        with self.assertRaises(AlgebraBudgetError) as ctx:
            circuit.plan(3, None, dirs, budget=Budget(variables=5, degree=2))
        self.assertEqual(ctx.exception.resource, "variables")
        with self.assertRaises(AlgebraBudgetError) as ctx:
            circuit.plan(3, None, dirs, budget=Budget(degree=2))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("degree", 3, 2))
        # the representation limits, with budgets that admit them
        big = Budget(variables=1 << 20, degree=1 << 20, support=1 << 40, tensor=1 << 40, bytes=1 << 40)
        sixty_five = [[int(i == j % 6) for i in range(6)] for j in range(65)]
        for d, dirs_ in ((1, sixty_five), (17, dirs[:1]), (4, sixty_five[:64])):    # 65 directions, order 17, S > 65535
            with self.assertRaises(UnsupportedError):
                circuit.plan(d, None, dirs_, budget=big)
        self.assertEqual(circuit.plan(16, None, dirs[:1]).support, 17)              # order 16 in one direction
        jet = circuit.plan(16, None, dirs[:1]).at({"x": 1})
        self.assertEqual(jet.taylor, [math.comb(4, k) for k in range(5)] + [0] * 12)   # (t + 1)^4

    def test_jet002_aggregate_hessians_over_scopes(self):
        rng = np.random.default_rng(2005)
        P, D, n, G = 3, 2, 250, 5
        nodes = random_circuit(rng, P, D, size=14, maxdeg=4)
        circuit = mj.Circuit.from_nodes(nodes, [f"p{i}" for i in range(P)], ["x0", "x1"])
        f = expand(nodes, P, D)
        x0, x1 = rng.integers(-5, 6, n), rng.integers(-500, 501, n)
        v1 = rng.random(n) < 0.8
        groups = rng.integers(0, G, n)
        scope = mbk.Scope.from_arrays({"x0": x0, "x1": x1}, valid={"x1": v1}, scales={"x1": 2}, groups=groups, ngroups=G)
        center = [rand_q(rng) for _ in range(P)]
        I = [[Fraction(int(i == j)) for i in range(P)] for j in range(P)]
        checked = "x1" in circuit.info["reads_columns"]
        for order in (2, 3):
            plan = circuit.plan(order, center, scales=[0, 2])
            agg = self.both(plan, scope)
            oracle = [{a: Fraction(0) for a in monomials(P, order)} for _ in range(G)]
            for r in range(n):
                if checked and not v1[r]:
                    continue
                der = derivative_oracle(f, P, D, center, I, [int(x0[r]), Fraction(int(x1[r]), 100)], order)
                for a, val in der.items():
                    oracle[int(groups[r])][a] += val
            for g in range(G):
                self.assertEqual(jet_dict(agg.jet(g), "derivative"), oracle[g], g)
            cuts = sorted({0, n, *[int(v) for v in rng.integers(0, n, 4)]})
            bl = plan.builder(scope)
            for lo, hi in zip(cuts, cuts[1:]):
                bl.accumulate(lo, hi)
            self.assertTrue(bl.finish().equal(agg))

    def test_jet002_wide_directions_and_high_orders(self):
        rng = np.random.default_rng(2006)
        P, D = 12, 1
        nodes = random_circuit(rng, P, D, size=24, maxdeg=4)
        I = [[Fraction(int(i == j)) for i in range(P)] for j in range(P)]
        center = [rand_q(rng) for _ in range(P)]
        self.check_against_oracles(nodes, P, D, center, I, [[2], [-1]], 2, identity=True)
        nodes = random_circuit(rng, 2, 1, size=12, maxdeg=7, maxpow=3)
        U = [[Fraction(1), Fraction(-1, 2)]]
        for order in (4, 6):
            self.check_against_oracles(nodes, 2, 1, [rand_q(rng), rand_q(rng)], U, [[3]], order)


if __name__ == "__main__":
    unittest.main()
