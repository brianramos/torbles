# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-005: the native module and operator registry (torbles/math_registry.c, math_registry_gpu.cu).

Oracles, independent of the implementation: the law table checked by brute force on sample elements of Python models
of each carrier (int, Fraction, bool, GF(2) by xor/and, naturals, residues mod 2^128, outward-rounded intervals by
math.nextafter, floats, min, frozensets, big ints and Fractions, GF(p) residues, complex rectangles of tightly rounded
intervals, truncated series, Kleene's F < U < T with max / min, exact-endpoint intervals); polynomial aggregates
recomputed row by row with Python integers and with DuckDB SQL (HUGEINT); histograms by numpy.bincount; plane counts
and power sums by Python loops; zeta, Moebius and Walsh transforms by their defining sums; the first slice's own
registry (math_domains.modules) and its unchanged test suite; the registry against the code: the exported symbols of
libalg.so and of the CUDA library, the CUDA sources' entry points and docs/algebra_inventory.json. Every executor also
runs on the CUDA path (emulated here, scrambled thread order) and must equal the CPU bit for bit. Seeds are fixed.
"""
import ctypes as C
import itertools
import json
import math
import re
import unittest
from fractions import Fraction
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import algq, math_gpu, math_registry as reg
from torbles.math_budgets import Budget
from torbles.math_domains import CarrierError, OperatorError, modules as first_slice_modules
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

M128 = 1 << 128
P61 = (1 << 61) - 1
ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ independent models of the carriers
def _down(x):
    return -math.inf if x == -math.inf else math.nextafter(x, -math.inf)


def _up(x):
    return math.inf if x == math.inf else math.nextafter(x, math.inf)


class Model:
    """a carrier as Python operations: plus (the fold), times, zero, one, sample elements, the carrier's own
    comparison (None when it has none), negation (an additive inverse candidate), reciprocal candidates and the
    characteristic when it is a known prime (0 otherwise)"""
    def __init__(self, elements, plus, times, zero, one, leq=None, neg=None, recip=None, eq=None, exact=True, char=0):
        self.elements, self.plus, self.times, self.zero, self.one = elements, plus, times, zero, one
        self.leq, self.neg, self.recip, self.exact, self.char = leq, neg, recip, exact, char
        self.eq = eq or (lambda a, b: a == b)


def _iv_add(a, b):
    return (_down(a[0] + b[0]), _up(a[1] + b[1]))


def _iv_mul(a, b):
    p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return (_down(min(p)), _up(max(p)))


def _feq(a, b):
    return a == b or (a != a and b != b)


# tight rounding of exact values (FND-009's intervals): the largest double <= q, the smallest double >= q
def _rd(q):
    x = float(q)
    return x if Fraction(x) <= q else math.nextafter(x, -math.inf)


def _ru(q):
    x = float(q)
    return x if Fraction(x) >= q else math.nextafter(x, math.inf)


def _tadd(a, b):
    return (_rd(Fraction(a[0]) + Fraction(b[0])), _ru(Fraction(a[1]) + Fraction(b[1])))


def _tneg(a):
    return (-a[1], -a[0])


def _tmul(a, b):
    p = [Fraction(x) * Fraction(y) for x in a for y in b]
    return (_rd(min(p)), _ru(max(p)))


def _tdiv(a, b):                                           # b does not contain 0
    q = [Fraction(x) / Fraction(y) for x in a for y in b]
    return (_rd(min(q)), _ru(max(q)))


def _cadd(z, w):                                           # complex rectangles (re, im)
    return (_tadd(z[0], w[0]), _tadd(z[1], w[1]))


def _cmul(z, w):
    return (_tadd(_tmul(z[0], w[0]), _tneg(_tmul(z[1], w[1]))), _tadd(_tmul(z[0], w[1]), _tmul(z[1], w[0])))


def _crecip(z):                                            # conj(z) / |z|^2, as tr_carrier_cinterval_op
    den = _tadd(_tmul(z[0], z[0]), _tmul(z[1], z[1]))
    return [(_tdiv(z[0], den), _tdiv(_tneg(z[1]), den))] if den[0] > 0 else []


def _qmul(a, b):                                           # exact-endpoint intervals
    p = [x * y for x in a for y in b]
    return (min(p), max(p))


def _smul(a, b):                                           # series truncated at order 3
    return tuple(sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(3))


def _srecip(a):
    if a[0] == 0:
        return []
    r0 = 1 / a[0]
    r1 = -a[1] * r0 / a[0]
    r2 = -(a[1] * r1 + a[2] * r0) / a[0]
    return [(r0, r1, r2)]


SETS = [frozenset(s) for r in range(3) for s in itertools.combinations((0, 1), r)]
LEQ = lambda a, b: a <= b                                  # noqa: E731
MODELS = {
    "integer": Model(list(range(-6, 7)), lambda a, b: a + b, lambda a, b: a * b, 0, 1, LEQ, lambda a: -a),
    "rational": Model([Fraction(n, d) for n in range(-3, 4) for d in (1, 2, 3)], lambda a, b: a + b,
                      lambda a, b: a * b, Fraction(0), Fraction(1), LEQ, lambda a: -a, lambda a: [1 / a]),
    "boolean": Model([False, True], lambda a, b: a or b, lambda a, b: a and b, False, True, LEQ),
    "gf2": Model([0, 1], lambda a, b: a ^ b, lambda a, b: a & b, 0, 1, None, lambda a: a),
    "count": Model(list(range(0, 9)), lambda a, b: a + b, lambda a, b: a * b, 0, 1, LEQ),
    "int128": Model([-(1 << 100), -5, -1, 0, 1, 7, 1 << 90], lambda a, b: a + b, lambda a, b: a * b, 0, 1, LEQ,
                    lambda a: -a),
    "mod128": Model([0, 1, 2, M128 - 1, M128 - 2, 1 << 127, 12345], lambda a, b: (a + b) % M128,
                    lambda a, b: (a * b) % M128, 0, 1, LEQ, lambda a: (M128 - a) % M128),
    "interval": Model([(1.0, 1.0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (-0.7, 2.5), (0.0, 0.0), (1e16, 1e16)],
                      _iv_add, _iv_mul, (0.0, 0.0), (1.0, 1.0), lambda a, b: a[1] <= b[0], lambda a: (-a[1], -a[0]),
                      lambda a: [(1 / a[1], 1 / a[0])] if a[0] > 0 or a[1] < 0 else [], exact=False),
    "float64": Model([0.1, 0.2, 0.3, 1.0, -0.5, 0.0, 1e16, float("nan")], lambda a, b: a + b, lambda a, b: a * b,
                     0.0, 1.0, LEQ, lambda a: -a, lambda a: [1 / a] if a == a and a else [], eq=_feq, exact=False),
    "minmax": Model(list(range(-4, 5)) + [math.inf], min, lambda a, b: a * b, math.inf, 1, LEQ),
    "rowset": Model(SETS, lambda a, b: a | b, lambda a, b: a & b, frozenset(), frozenset((0, 1)), LEQ),
    "bigint": Model([-(1 << 200), -3, -1, 0, 1, 2, 1 << 150], lambda a, b: a + b, lambda a, b: a * b, 0, 1, LEQ,
                    lambda a: -a),
    "bigq": Model([Fraction(-(1 << 100), 7), Fraction(-1, 2), Fraction(0), Fraction(1), Fraction(1, 3),
                   Fraction(1 << 90, 3)], lambda a, b: a + b, lambda a, b: a * b, Fraction(0), Fraction(1), LEQ,
                  lambda a: -a, lambda a: [1 / a]),
    "gfp": Model([0, 1, 2, P61 - 1, 12345, (P61 + 1) // 2], lambda a, b: (a + b) % P61, lambda a, b: (a * b) % P61,
                 0, 1, None, lambda a: (P61 - a) % P61, lambda a: [pow(a, P61 - 2, P61)], char=P61),
    "cinterval": Model([((1.0, 1.0), (0.0, 0.0)), ((0.1, 0.1), (0.0, 0.0)), ((0.2, 0.2), (0.1, 0.1)),
                        ((0.0, 0.0), (1.0, 1.0)), ((-0.7, 2.5), (0.5, 1.0)), ((0.0, 0.0), (0.0, 0.0)),
                        ((1e16, 1e16), (0.0, 0.0))], _cadd, _cmul, ((0.0, 0.0), (0.0, 0.0)), ((1.0, 1.0), (0.0, 0.0)),
                       None, lambda z: (_tneg(z[0]), _tneg(z[1])), _crecip, exact=False),
    "series": Model([(Fraction(0),) * 3, (Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)),
                     (Fraction(2), Fraction(-1), Fraction(1, 2)), (Fraction(1, 3), Fraction(0), Fraction(1)),
                     (Fraction(-1), Fraction(0), Fraction(0))],
                    lambda a, b: tuple(x + y for x, y in zip(a, b)), _smul, (Fraction(0),) * 3,
                    (Fraction(1), Fraction(0), Fraction(0)), None, lambda a: tuple(-x for x in a), _srecip),
    "kleene": Model([0, 1, 2], max, min, 0, 2, LEQ),       # FALSE < UNKNOWN < TRUE; OR = max, AND = min
    "qinterval": Model([(Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)), (Fraction(1, 3), Fraction(1, 3)),
                        (Fraction(1), Fraction(2)), (Fraction(-1), Fraction(3)), (Fraction(1, 2), Fraction(1))],
                       lambda a, b: (a[0] + b[0], a[1] + b[1]), _qmul, (Fraction(0), Fraction(0)),
                       (Fraction(1), Fraction(1)), None, lambda a: (-a[1], -a[0]),
                       lambda a: [(1 / a[1], 1 / a[0])] if a[0] > 0 or a[1] < 0 else [], exact=False),
}


def _nan(*xs):
    return any(isinstance(x, float) and x != x for x in xs)


def _holds(model, law):
    """the law on every sample (True), or a counterexample among them (False)"""
    E, P, T, eq = model.elements, model.plus, model.times, model.eq
    if law == "associative":
        return all(eq(P(P(a, b), c), P(a, P(b, c))) for a in E for b in E for c in E)
    if law == "commutative":
        return all(eq(P(a, b), P(b, a)) for a in E for b in E)
    if law == "identity":
        return all(eq(P(model.zero, a), a) for a in E)
    if law == "inverse":                                   # some b (a sample or the negation) with a + b = 0
        return all(any(eq(P(a, b), model.zero) for b in E + ([model.neg(a)] if model.neg else [])) for a in E)
    if law == "distributive":
        return all(eq(T(a, P(b, c)), P(T(a, b), T(a, c))) for a in E for b in E for c in E if not _nan(a, b, c))
    if law == "idempotent":
        return all(eq(P(a, a), a) for a in E)
    if law == "mul_idempotent":
        return all(eq(T(a, a), a) for a in E)
    if law == "ordered":                                   # the carrier's comparison: total and compatible with +
        if model.leq is None:
            return False
        total = all(model.leq(a, b) or model.leq(b, a) for a in E for b in E)
        return total and all(model.leq(P(a, c), P(b, c)) for a in E for b in E for c in E if model.leq(a, b))
    if law == "division":                                  # every element but 0 has a reciprocal
        nonzero = [a for a in E if not eq(a, model.zero)]
        return all(any(eq(T(a, b), model.one) for b in E + (model.recip(a) if model.recip else [])) for a in nonzero)
    if law == "complement":                                # some b with a + b = 1 and a * b = 0, for every a
        return all(any(eq(P(a, b), model.one) and eq(T(a, b), model.zero) for b in E) for a in E)
    if law == "char2":                                     # a + a = 0
        return all(eq(P(a, a), model.zero) for a in E)
    if law == "char0":                                     # 0, 1, 1+1, ... pairwise distinct
        if model.char:                                     # the p-th multiple of 1 is 0
            return False
        seen, acc = [], model.zero
        for _ in range(70):
            if any(eq(acc, s) for s in seen):
                return False
            seen.append(acc)
            acc = P(acc, model.one)
        if model is MODELS["mod128"]:
            return (M128 * 1) % M128 != 0                  # the 2^128-th multiple of 1 is 0
        if model is MODELS["float64"]:
            big = float(1 << 53)
            return big + 1.0 != big                        # 2^53 + 1 rounds to 2^53
        return True
    raise KeyError(law)


def make(columns, ddl_types=None):
    frame = pd.DataFrame({name: pd.Series(v, dtype=object) for name, v in columns.items()})
    types = ddl_types or {}
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" {types.get(c, "BIGINT")}' for c in columns) + ")"
    return torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)


def poly_value(p, x):
    return sum(c * x ** k for k, c in enumerate(p))


def oracle(values, mask, polys, aggregate):
    """Python integers row by row: (values per polynomial, selected, valid, nulls)"""
    sel = [v for v, m in zip(values, mask) if m]
    valid = [v for v in sel if v is not None]
    out = []
    for p in polys:
        vals = [poly_value(p, x) for x in valid]
        out.append(sum(vals) if aggregate == "sum" else (min(vals) if aggregate == "min" else max(vals)) if vals else None)
    return out, len(sel), len(valid), len(sel) - len(valid)


def duck(values, mask, polys, aggregate):
    con = duckdb.connect()
    xs = [v for v, m in zip(values, mask) if m]
    con.execute("CREATE TABLE s (x BIGINT)")
    if xs:
        con.executemany("INSERT INTO s VALUES (?)", [(v,) for v in xs])
    out = []
    for p in polys:
        expr = " + ".join(f"CAST({c} AS HUGEINT)" + "".join(" * CAST(x AS HUGEINT)" for _ in range(k))
                          for k, c in enumerate(p))
        # SUM over no valid row: SQL says NULL, the registry the identity 0 (with valid = 0, as the first slice)
        sql = {"sum": f"COALESCE(SUM({expr}), 0)", "min": f"MIN({expr})", "max": f"MAX({expr})"}[aggregate]
        v = con.execute(f"SELECT {sql} FROM s WHERE x IS NOT NULL").fetchone()[0]
        out.append(None if v is None else int(v))
    con.close()
    return out


def plane_sets(width, D):
    return [m for s in range(min(D, width) + 1) for m in sorted(sum(1 << j for j in c)
                                                                 for c in itertools.combinations(range(width), s))]


def popcount(x):
    return bin(x).count("1")


def transform_oracle(op, direction, v):
    """zeta / Moebius over supersets ("up") or subsets ("down") and the Walsh transform, by their defining sums"""
    n = len(v)
    if op == "walsh":
        return [sum(v[x] * (-1) ** popcount(S & x) for x in range(n)) for S in range(n)]
    out = []
    for S in range(n):
        terms = [A for A in range(n) if (A & S == S if direction == "up" else A & ~S == 0)]
        out.append(sum(v[A] if op == "zeta" else (-1) ** popcount(A ^ S) * v[A] for A in terms))
    return out


def cuda_entry_points(text):
    """the tr_mgpu_* functions a CUDA source defines (a declaration ends with ';' before any '{')"""
    names = []
    for match in re.finditer(r"^[A-Za-z_][\w \*]*?\b(tr_mgpu_\w+)\s*\(", text, re.M):
        rest = text[match.end():]
        brace, semi = rest.find("{"), rest.find(";")
        if brace != -1 and (semi == -1 or brace < semi):
            names.append(match.group(1))
    return names


class Registry(unittest.TestCase):
    def table(self, seed, n, nulls=0.1, lo=-40, hi=200):
        rng = np.random.default_rng(seed)
        x = [int(v) for v in rng.integers(lo, hi + 1, size=n)]
        for i in rng.choice(n, size=int(n * nulls), replace=False) if n else []:
            x[int(i)] = None
        y = [int(v) for v in rng.integers(0, 10, size=n)]
        db = make({"x": x, "y": y})
        self.addCleanup(db.close)
        return db, x, y

    def column(self, db, name="x"):
        col = reg.Column(db["t"], name)
        self.addCleanup(col.close)
        return col

    def both(self, req, representation, **kw):
        """the CPU and the CUDA result, equal bit for bit"""
        cpu = reg.execute(req, representation, backend="cpu", **{k: v for k, v in kw.items() if k != "finalize"})
        cuda = reg.execute(req, representation, backend="cuda", **kw)
        for k in ("values", "selected", "valid", "nulls"):
            self.assertEqual(cpu[k], cuda[k], (representation, k))
        return cpu, cuda

    # -------------------------------------------------------------- descriptors
    def test_fnd005_registry_describes_every_module_and_the_first_slice_builtins(self):
        mods = {m["id"]: m for m in reg.modules()}
        self.assertTrue(set(range(1, 10)) <= set(mods))     # the representations; the families follow (ids >= 10)
        self.assertEqual({i for i in mods if mods[i]["representation"] is None}, set(mods) - set(range(1, 10)))
        self.assertEqual({i: mods[i]["capability"] for i in range(1, 10)},
                         {1: "FND-002", 2: "FND-003", 3: "FND-010", 4: "DISC-001", 5: "FND-005", 6: "FND-005",
                          7: "FND-005", 8: "FND-005", 9: "FND-008"})
        builtin = {r["id"]: r for r in first_slice_modules()}
        self.assertEqual(set(builtin), {1, 2})
        for i in (1, 2):                                # the same operators as the validated first-slice table
            self.assertEqual(mods[i]["operator_ids"], builtin[i]["operator_ids"])
            self.assertIn(builtin[i]["carrier"], mods[i]["carriers"])
            self.assertEqual(mods[i]["devices"], ["cpu"])
        self.assertEqual(mods[1]["operators"], ["poly_add", "poly_sub", "poly_mul", "poly_pow", "poly_derivative",
                                                "poly_integrate", "poly_substitute", "poly_evaluate"])
        self.assertEqual(mods[2]["operators"], ["bank_reduce"])
        gpu = math_gpu.lib()
        laws_of = {c["name"]: set(c["laws"]) for c in reg.carriers()}
        for m in mods.values():
            self.assertTrue(m["cpu_entries"], m["name"])
            for e in m["cpu_entries"]:
                self.assertTrue(hasattr(algq.lib, e), e)
            for e in m["cuda_entries"]:
                self.assertTrue(hasattr(gpu, e), e)
            self.assertEqual(bool(m["cuda_entries"]), "cuda" in m["devices"], m["name"])
            for op, carrier in m["implements"]:            # no module implements what the laws forbid
                contract = "enclosure" if "enclosure" in laws_of[carrier] else "exact"
                self.assertTrue(reg.explain_chain([(op, carrier)], input=carrier, contract=contract)["accepted"],
                                (m["name"], op, carrier))
                self.assertIn(op, m["operators"] if m["id"] > 2 else m["operators"] + ["convert", "powers", "fold"])
            self.assertTrue(set(m["costs"]["cpu"]) == set(reg.COST_COMPONENTS))
        self.assertEqual({k: C.sizeof(getattr(reg, n)) for k, n in (
            ("refusal", "_Refusal"), ("chain", "_Chain"), ("request", "_Request"), ("result", "_Result"),
            ("explanation", "_Explain"), ("state_info", "_StateInfo"), ("candidate", "_Candidate"),
            ("module", "_Module"))}, reg.SIZES)
        # every representation's chains are checked like any composition
        for name in ("scan", "histogram", "power", "bitplane", "empty"):
            ch = reg.chain(name, "sum")
            self.assertEqual(reg.check_chain(ch["steps"], input=ch["input"], implemented=True), "int128", name)
        for name in ("scan", "histogram", "empty"):
            ch = reg.chain(name, "max")
            self.assertEqual(reg.check_chain(ch["steps"], input=ch["input"], implemented=True), "minmax", name)
        aux = reg.chain("histogram", multiplicities=True)
        self.assertEqual(reg.check_chain(aux["steps"], input=aux["input"]), "count")
        for name, out in (("formal", "rational"), ("moments", "rational"), ("cells", "count"), ("verify", "count")):
            ch = reg.chain(name)
            self.assertEqual(reg.check_chain(ch["steps"], input=ch["input"], implemented=True), out, name)

    def test_fnd005_first_slice_dispatch_is_unchanged(self):
        suite = unittest.TestLoader().loadTestsFromName("test_math_modules")
        result = unittest.TestResult()
        suite.run(result)
        self.assertTrue(result.wasSuccessful(), [str(f[1])[-400:] for f in result.failures + result.errors])
        self.assertGreaterEqual(result.testsRun, 6)

    def test_fnd005_carrier_laws_match_brute_force_models(self):
        table = {c["name"]: set(c["laws"]) for c in reg.carriers()}
        algebraic = ("associative", "commutative", "identity", "inverse", "distributive", "idempotent",
                     "mul_idempotent", "ordered", "division", "char0", "complement", "char2")
        for name, model in MODELS.items():
            for law in algebraic:
                self.assertEqual(_holds(model, law), law in table[name], (name, law))
            self.assertEqual("exact" in table[name], model.exact, name)
        # rounding: float and interval sums are not exact; the interval encloses the exact sum
        self.assertNotEqual(Fraction(0.1) + Fraction(0.2), Fraction(0.1 + 0.2))
        a, b = (0.1, 0.1), (0.2, 0.2)
        s = _iv_add(a, b)
        self.assertTrue(Fraction(s[0]) <= Fraction(0.1) + Fraction(0.2) <= Fraction(s[1]))
        self.assertIn("enclosure", table["interval"])
        self.assertEqual(table["code"], {"ordered", "exact"})
        self.assertEqual(set(MODELS) | {"code"}, set(table))   # every carrier has a model (codes: an order only)
        for name in ("interval", "cinterval", "qinterval"):    # enclosures: never exact
            self.assertEqual(("enclosure" in table[name], "exact" in table[name]), (True, False), name)
        # GF(p) for every admitted prime: p = 2 has the laws of GF(2) only (char 2, a * a = a), so they are not GF(p)'s
        self.assertTrue({"char2", "mul_idempotent"} <= table["gf2"] and not {"char2", "mul_idempotent"} & table["gfp"])
        two = Model([0, 1], lambda a, b: (a + b) % 2, lambda a, b: a * b % 2, 0, 1, char=2)
        self.assertTrue(_holds(two, "char2") and not _holds(MODELS["gfp"], "char2"))
        # Kleene is not Boolean: U has no complement (U OR NOT U = U), and no AND-inverse
        self.assertEqual(table["boolean"] - table["kleene"], {"complement", "division"})
        self.assertEqual([max(x, 2 - x) for x in (0, 1, 2)], [2, 1, 2])   # x OR NOT x (NOT x = 2 - x): U at U
        codes = list(range(50))                        # an affine codec preserves the order of codes
        for offset in (-1000, 0, 77):
            vals = [c + offset for c in codes]
            self.assertEqual(sorted(range(50), key=lambda i: vals[i]), codes)

    # -------------------------------------------------------------- acceptance: compositions
    def test_fnd005_composition_refused_when_a_required_law_is_absent(self):
        cases = [  # steps, input, contract, (step, operator, carrier, missing law)
            ([("divide", "integer")], "integer", "exact", (0, "divide", "integer", "division")),
            ([("poly_integrate", "integer")], "integer", "exact", (0, "poly_integrate", "integer", "division")),
            ([("poly_integrate", "gf2")], "gf2", "exact", (0, "poly_integrate", "gf2", "char0")),
            ([("fold", "float64")], "float64", "exact", (0, "fold", "float64", "associative")),
            ([("merge", "interval")], "interval", "enclosure", (0, "merge", "interval", "associative")),
            ([("convert", "int128"), ("distinct_fold", "int128")], "integer", "exact",
             (1, "distinct_fold", "int128", "idempotent")),
            ([("subtract", "minmax")], "minmax", "exact", (0, "subtract", "minmax", "inverse")),
            ([("order", "gf2")], "gf2", "exact", (0, "order", "gf2", "ordered")),
            ([("scale_count", "minmax")], "minmax", "exact", (0, "scale_count", "minmax", "distributive")),
            ([("convert", "mod128"), ("powers", "mod128"), ("order", "mod128")], "integer", "exact",
             (2, "order", "mod128", "ordered")),
            ([("and", "rowset"), ("distinct_fold", "rowset"), ("subtract", "rowset")], "rowset", "exact",
             (2, "subtract", "rowset", "inverse")),
            ([("convert", "integer"), ("eval", "integer"), ("convert", "minmax"), ("powers", "minmax")], "code",
             "exact", (3, "powers", "minmax", "distributive")),
        ]
        for steps, inp, contract, (step, op, carrier, law) in cases:
            with self.assertRaises(CarrierError, msg=steps) as ctx:
                reg.check_chain(steps, input=inp, contract=contract)
            r = ctx.exception.refusal
            self.assertEqual((r["kind"], r["step"], r["operator"], r["carrier"], r["law"]),
                             ("law", step, op, carrier, law), steps)
            self.assertIn(law, str(ctx.exception))
            self.assertFalse(_holds(MODELS[carrier], law), (carrier, law))     # the model has a counterexample
        accepted = [([("convert", "rational"), ("divide", "rational")], "integer", "rational"),
                    ([("poly_integrate", "rational")], "rational", "rational"),
                    ([("convert", "minmax"), ("distinct_fold", "minmax")], "integer", "minmax"),
                    ([("convert", "int128"), ("subtract", "int128")], "integer", "int128"),
                    ([("convert", "mod128"), ("powers", "mod128"), ("contract", "mod128"), ("convert", "int128")],
                     "integer", "int128"),
                    ([("fold", "interval")], "interval", None)]
        for steps, inp, out in accepted[:-1]:
            self.assertEqual(reg.check_chain(steps, input=inp), out, steps)
        with self.assertRaises(CarrierError):          # intervals fold only as an enclosure, never exactly
            reg.check_chain([("convert", "interval")], input="integer")
        # the registry's own representations: POWER cannot answer MIN/MAX (min/max is not distributive)
        db, x, y = self.table(1, 600)
        req = reg.Request(self.column(db), [[3, -1, 2]], aggregate="max", row_ids=db.query(db["t"]).where(db["t"].y < 5).rows())
        for backend in ("cpu", "cuda"):
            with self.assertRaises(CarrierError) as ctx:
                reg.execute(req, "power", backend=backend, **({} if backend == "cpu" else {"finalize": "cpu"}))
            ex = ctx.exception.explanation
            self.assertEqual((ex["status"], ex["refusal"]["kind"], ex["refusal"]["law"], ex["refusal"]["carrier"],
                              ex["refusal"]["step"], ex["rows_scanned"]), ("refused", "law", "distributive", "minmax", 1, 0))
        mask = [v < 5 for v in y]
        expect = oracle(x, mask, [[3, -1, 2]], "max")[0]
        for name in ("scan", "histogram"):
            cpu, cuda = self.both(req, name)
            self.assertEqual(cpu["values"], expect, name)

    def test_fnd005_composition_refused_when_a_carrier_conversion_is_absent(self):
        cases = [([("convert", "integer")], "rational", ("rational", "integer")),
                 ([("fold", "minmax")], "count", ("count", "minmax")),
                 ([("convert", "integer")], "float64", ("float64", "integer")),
                 ([("convert", "integer")], "interval", ("interval", "integer")),
                 ([("convert", "minmax")], "mod128", ("mod128", "minmax")),
                 ([("convert", "interval")], "integer", ("integer", "interval")),      # enclosing: not lossless
                 ([("convert", "rational"), ("convert", "integer")], "integer", ("rational", "integer"))]
        for steps, inp, conv in cases:
            with self.assertRaises(CarrierError, msg=steps) as ctx:
                reg.check_chain(steps, input=inp)
            r = ctx.exception.refusal
            self.assertEqual((r["kind"], r["conversion"]), ("conversion", conv), steps)
            self.assertIn(f"{conv[0]} -> {conv[1]}", str(ctx.exception))
        # under the enclosure contract the enclosing conversion is admitted
        self.assertEqual(reg.check_chain([("convert", "interval")], input="integer", contract="enclosure"), "interval")
        # checked conversions are lossless when admitted
        for steps, inp, out in (([("convert", "integer")], "int128", "integer"), ([("convert", "integer")], "code", "integer"),
                                ([("convert", "int128")], "mod128", "int128")):
            self.assertEqual(reg.check_chain(steps, input=inp), out)
        kinds = {(c["from"], c["to"]): c["kind"] for c in reg.conversions()}
        self.assertEqual(kinds[("integer", "interval")], "enclosing")
        self.assertNotIn(("rational", "integer"), kinds)
        # an oracle for the declarations: the declared exact conversions are injective, the absent ones lose values
        self.assertEqual(len({Fraction(n) for n in range(-50, 50)}), 100)                         # integer -> rational
        self.assertEqual(len({c - 7 for c in range(100)}), 100)                                   # code -> integer
        self.assertEqual(int(Fraction(1, 2)), int(Fraction(0)))                                   # rational -> integer
        self.assertEqual(float(2 ** 53 + 1), float(2 ** 53))                                      # integer -> float64
        # unknown operator, wrong fixed carrier, no implementation
        with self.assertRaises(OperatorError):
            reg.check_chain([(99, "integer")], input="integer")
        with self.assertRaises(CarrierError) as ctx:
            reg.check_chain([("histogram", "integer")], input="integer")
        self.assertEqual(ctx.exception.refusal["kind"], "carrier")
        self.assertEqual(reg.check_chain([("fold", "rational")], input="rational"), "rational")    # the laws hold
        with self.assertRaises(CarrierError) as ctx:                                               # nothing implements it
            reg.check_chain([("fold", "rational")], input="rational", implemented=True)
        self.assertEqual(ctx.exception.refusal["kind"], "implementation")
        # the registry's own representation: BITPLANE multiplicities do not convert to a min/max
        db, x, y = self.table(2, 500)
        req = reg.Request(self.column(db), [[1, 1]], aggregate="min")
        for backend in ("cpu", "cuda"):
            with self.assertRaises(CarrierError) as ctx:
                reg.execute(req, "bitplane", backend=backend, **({} if backend == "cpu" else {"finalize": "cpu"}))
            r = ctx.exception.explanation["refusal"]
            self.assertEqual((r["kind"], r["conversion"], r["step"]), ("conversion", ("count", "minmax"), 4))

    # -------------------------------------------------------------- acceptance: explanations
    def test_fnd005_explanations_name_module_representation_device_and_refusal(self):
        db, x, y = self.table(3, 900)
        col = self.column(db)
        ids = db.query(db["t"]).where(db["t"].y >= 3).rows()
        req = reg.Request(col, [[5, -2, 1], [0, 0, 0, 1]], row_ids=ids)
        mask = [v >= 3 for v in y]
        expect, selected, valid, nulls = oracle(x, mask, req.polys, "sum")
        mods = {m["representation"]: m for m in reg.modules()}
        emulated = math_gpu.device()["emulated"]
        width = max(col_code for col_code in [int(c) for c in db["t"].decoded("x")]).bit_length()
        elements = {"histogram": int(max(db["t"].decoded("x"))) + 1, "power": 4,
                    "bitplane": len(plane_sets(width, 3)), "scan": 0}
        for name in ("scan", "histogram", "power", "bitplane"):
            for backend, finalize in (("cpu", None), ("cuda", None), ("cuda", "cpu")):
                if finalize == "cpu" and name != "histogram":
                    continue
                out = reg.execute(req, name, backend=backend, finalize=finalize)
                ex = out["explanation"]
                self.assertEqual(out["values"], expect, (name, backend))
                self.assertEqual((out["selected"], out["valid"], out["nulls"]), (selected, valid, nulls))
                self.assertEqual((ex["status"], ex["module"], ex["module_name"], ex["capability"], ex["representation"]),
                                 ("ran", mods[name]["id"], mods[name]["name"], "FND-005", name))
                self.assertEqual(ex["device"], "cpu" if backend == "cpu" else ("cuda-emulated" if emulated else "cuda"))
                self.assertEqual(ex["device_name"], "native CPU" if backend == "cpu" else math_gpu.device()["name"])
                self.assertEqual(ex["finalize"], "cpu" if backend == "cpu" or finalize == "cpu" or
                                 name in ("power", "bitplane") else "gpu")
                self.assertEqual((ex["rows_scanned"], ex["state_elements"], ex["refusal"], ex["budget"], ex["reused"]),
                                 (selected, elements[name], None, None, False))
                self.assertEqual(ex["device_bytes"] > 0, backend == "cuda")
        # refusals: budget, overflow, law, stale, proof, bound, invalid input, each named; the ones decided before the
        # scan read no row
        def refused(fn, exc):
            with self.assertRaises(exc) as ctx:
                fn()
            ex = ctx.exception.explanation
            self.assertEqual(ex["status"], "refused")
            if ex["refusal"]["kind"] in ("budget", "law", "conversion", "proof", "stale", "bound"):
                self.assertEqual(ex["rows_scanned"], 0, ex)
            return ex
        ncodes = elements["histogram"]
        for backend in ("cpu", "cuda"):
            ex = refused(lambda: reg.execute(req, "histogram", backend=backend, budget=Budget(states=ncodes - 1)),
                         AlgebraBudgetError)
            self.assertEqual((ex["refusal"]["kind"], ex["budget"]), ("budget", dict(resource="states", need=ncodes,
                                                                                   limit=ncodes - 1)))
            big = req.with_polys([[0, 0, 0, 0, 0, 0, 0, 0, (1 << 62)]])
            ex = refused(lambda: reg.execute(big, "scan", backend=backend), OverflowError)
            self.assertEqual(ex["refusal"]["kind"], "overflow")
            ex = refused(lambda: reg.execute(big, "power", backend=backend, finalize="cpu"), OverflowError)
            self.assertEqual((ex["refusal"]["kind"], ex["refusal"]["conversion"]), ("bound", ("mod128", "int128")))
            ex = refused(lambda: reg.execute(req.with_polys([[1, 1]], "max"), "power", backend=backend,
                                             finalize="cpu"), CarrierError)
            self.assertEqual((ex["refusal"]["kind"], ex["refusal"]["law"]), ("law", "distributive"))
        ex = refused(lambda: reg.execute(req, "empty"), UnsupportedError)
        self.assertEqual(ex["refusal"]["kind"], "proof")
        with self.assertRaises(UnsupportedError):
            reg.execute(req, "empty", backend="cuda")                       # no CUDA entry point is declared
        state = reg.build_state(req, "histogram")
        self.addCleanup(state.close)
        other = reg.Request(col, req.polys, rows=np.ones(len(x), dtype=bool))
        ex = refused(lambda: reg.execute(other, "histogram", state=state), StaleStateError)
        self.assertEqual(ex["refusal"]["kind"], "stale")
        codes = np.array([0, 5, 9], dtype=np.uint32)
        bad = reg.Request(codes, [[1, 1]], ncodes=10)
        bad.raw.ncodes, bad.raw.width = 6, 3                                # a stored code outside the domain
        for backend in ("cpu", "cuda"):
            with self.assertRaises(ValueError) as ctx:
                reg.execute(bad, "scan", backend=backend)
            self.assertEqual(ctx.exception.explanation["refusal"]["kind"], "invalid")

    # -------------------------------------------------------------- executions against Python and DuckDB
    def test_fnd005_executors_equal_python_integers_and_duckdb(self):
        rng = np.random.default_rng(5)
        for seed, n in ((11, 0), (12, 1), (13, 70), (14, 1500), (15, 6000)):
            db, x, y = self.table(seed, n)
            col = self.column(db)
            for trial in range(3):
                polys = [[int(c) for c in rng.integers(-9, 10, size=int(rng.integers(1, 7)))]
                         for _ in range(int(rng.integers(1, 5)))]
                k = int(rng.integers(0, 10))
                ids = db.query(db["t"]).where(db["t"].y < k).rows() if n else np.zeros(0, dtype=np.uint64)
                mask = [v < k for v in y]
                for aggregate in ("sum", "min", "max"):
                    req = reg.Request(col, polys, aggregate=aggregate, row_ids=ids)
                    expect, selected, valid, nulls = oracle(x, mask, polys, aggregate)
                    if n and trial == 0:
                        self.assertEqual(duck(x, mask, polys, aggregate), expect)
                    names = ("scan", "histogram", "power", "bitplane") if aggregate == "sum" else ("scan", "histogram")
                    for name in names:
                        cpu, cuda = self.both(req, name)
                        self.assertEqual((cpu["values"], cpu["selected"], cpu["valid"], cpu["nulls"]),
                                         (expect, selected, valid, nulls), (seed, trial, aggregate, name))
                    if aggregate == "sum":
                        h = reg.execute(req, "histogram", backend="cuda", finalize="cpu")
                        self.assertEqual(h["values"], expect)
        # a DECIMAL column: f applies to the scaled integer (code + offset)
        vals = ["1.25", "-2.50", None, "3.00", "0.01", "-0.07"]
        db = make({"d": vals}, {"d": "DECIMAL(10,2)"})
        self.addCleanup(db.close)
        col = self.column(db, "d")
        scaled = [None if v is None else int(Fraction(v) * 100) for v in vals]
        req = reg.Request(col, [[1, 2, 3]])
        expect = oracle(scaled, [True] * len(vals), req.polys, "sum")
        for name in ("scan", "histogram", "power", "bitplane"):
            cpu, _ = self.both(req, name)
            self.assertEqual((cpu["values"], cpu["selected"], cpu["valid"], cpu["nulls"]), (expect[0], 6, 5, 1))
        self.assertEqual(col.scale, 2)
        # a self-extending column stores only its low bits as codes: refused rather than read partially
        wide = algq.Table("w", {"v": np.array([1, 2 ** 33, 5], dtype=np.uint64)}, extend={"v": 32})
        with self.assertRaises(UnsupportedError):
            reg.Column(wide, "v")

    def test_fnd005_wide_sums_and_extreme_values_are_exact(self):
        n = 3000
        codes = np.arange(n, dtype=np.uint32) % 7
        req = reg.Request(codes, [[(1 << 63) - 1], [-(1 << 63)], [0, (1 << 61)]], offset=-3)
        xs = [int(c) - 3 for c in codes]
        expect = [n * ((1 << 63) - 1), -n * (1 << 63), sum((1 << 61) * v for v in xs)]
        for name in ("scan", "histogram"):
            cpu, _ = self.both(req, name)
            self.assertEqual(cpu["values"], expect, name)
        for name in ("power", "bitplane"):                # the 128-bit bound: 3000 * 2^63 < 2^127 is proven
            cpu, _ = self.both(req, name, finalize="cpu")
            self.assertEqual(cpu["values"], expect, name)
        mx = reg.Request(codes, [[(1 << 63) - 1], [0, -(1 << 60)]], aggregate="max", offset=-3)
        for name in ("scan", "histogram"):
            cpu, _ = self.both(mx, name)
            self.assertEqual(cpu["values"], [(1 << 63) - 1, 3 * (1 << 60)], name)
        # the logical domain must fit int64 before anything runs
        near = reg.Request(np.array([1, 19], dtype=np.uint32), [[0, 1]], offset=(1 << 63) - 10)
        for name in ("scan", "histogram", "power", "bitplane"):
            for backend in ("cpu", "cuda"):
                with self.assertRaises(OverflowError):
                    reg.execute(near, name, backend=backend, **({"finalize": "cpu"} if name in ("power", "bitplane")
                                                                and backend == "cuda" else {}))
        # an evaluation leaving int64 at a Horner step refuses SCAN and HISTOGRAM identically; POWER and BITPLANE
        # compute modulo 2^128 under the proven bound and give the exact sum
        codes = np.array([0, 1, 2, 3, 1000], dtype=np.uint32)
        sel = np.array([True, True, True, True, False])
        req = reg.Request(codes, [[0, 0, 0, 0, 0, 0, 0, 1]], rows=sel, offset=600)      # x^7 with x up to 603
        self.assertGreater(603 ** 7, (1 << 63) - 1)
        for name in ("scan", "histogram"):
            for backend in ("cpu", "cuda"):
                with self.assertRaises(OverflowError):
                    reg.execute(req, name, backend=backend)
        for name in ("power", "bitplane"):
            cpu, _ = self.both(req, name, finalize="cpu")
            self.assertEqual(cpu["values"], [sum((600 + c) ** 7 for c in range(4))])
        unsel = reg.Request(codes, [[0, 0, 0, 0, 0, 0, 0, 1]], rows=np.array([True, False, False, False, False]),
                            offset=0)
        for name in ("scan", "histogram"):                 # only selected valid rows are evaluated
            cpu, _ = self.both(unsel, name)
            self.assertEqual(cpu["values"], [0])

    # -------------------------------------------------------------- states
    def test_fnd005_states_reuse_and_device_states_equal_cpu(self):
        db, x, y = self.table(21, 2500, lo=-30, hi=120)
        col = self.column(db)
        ids = db.query(db["t"]).where(db["t"].y < 6).rows()
        mask = [v < 6 for v in y]
        req = reg.Request(col, [[1, 2, 0, -1]], row_ids=ids)
        offset = col.offset
        codes = db["t"].decoded("x")
        valid = np.array([m and v is not None for m, v in zip(mask, x)])
        hist = np.bincount(codes[valid], minlength=int(codes.max()) + 1).astype(np.uint64)
        width = int(codes.max()).bit_length()
        for name, param in (("histogram", {}), ("power", {"support": [0, 1, 3, 5]}), ("bitplane", {"degree": 4})):
            cpu = reg.build_state(req, name, **param)
            cuda = reg.build_state(req, name, backend="cuda", **param)
            self.addCleanup(cpu.close); self.addCleanup(cuda.close)
            self.assertTrue(np.array_equal(cpu.raw(), cuda.raw()), name)
            ci, gi = cpu.info, cuda.info
            for k in ("representation", "elements", "selected", "valid", "nulls", "support", "degree"):
                self.assertEqual(ci[k], gi[k], (name, k))
            raw = cpu.raw()
            xs = [int(c) + offset for c, ok in zip(codes, valid) if ok]
            if name == "histogram":
                self.assertTrue(np.array_equal(raw, hist))
            elif name == "power":
                for idx, k in enumerate([0, 1, 3, 5]):
                    s = sum(v ** k for v in xs) % M128
                    self.assertEqual(int(raw[2 * idx]) + (int(raw[2 * idx + 1]) << 64), s, k)
            else:
                sets = plane_sets(width, 4)
                vc = [int(c) for c, ok in zip(codes, valid) if ok]
                self.assertEqual([int(v) for v in raw], [sum(1 for c in vc if c & S == S) for S in sets])
            # reuse: another polynomial over the same scope, contracted without a scan, equals a fresh execution
            other = req.with_polys([[7, 0, 3], [0, 1, 0, 1]] if name != "bitplane" else [[7, 0, 3], [0, 1, 0, 1, 1]])
            if name == "power":
                other = req.with_polys([[7, 1, 0, 3], [0, 0, 0, 0, 0, 1]])
            fresh = reg.execute(other, name)
            for st in (cpu, cuda):
                again = reg.execute(other, name, state=st)
                self.assertEqual(again["values"], fresh["values"], name)
                self.assertEqual(again["values"], oracle(x, mask, other.polys, "sum")[0], name)
                self.assertEqual((again["explanation"]["reused"], again["explanation"]["rows_scanned"]), (True, 0))
                self.assertTrue(st.matches(other))
            if name == "power":                            # an exponent outside the state's support
                with self.assertRaises(UnsupportedError):
                    reg.execute(req.with_polys([[0, 0, 1]]), "power", state=cpu)
            if name == "bitplane":                         # a degree above the state's
                with self.assertRaises(UnsupportedError):
                    reg.execute(req.with_polys([[0, 0, 0, 0, 0, 1]]), "bitplane", state=cpu)
        # staleness: a mutation of the table makes the column (and its states) stale
        state = reg.build_state(req, "histogram")
        self.addCleanup(state.close)
        db["t"].flip(0, 0)
        with self.assertRaises(StaleStateError):
            reg.Request(col, [[1]])
        fresh_col = self.column(db)
        moved = reg.Request(fresh_col, [[1, 2, 0, -1]], row_ids=ids)
        self.assertFalse(state.matches(moved))
        with self.assertRaises(StaleStateError):
            reg.execute(moved, "histogram", state=state)

    # -------------------------------------------------------------- budgets
    def test_fnd005_budgets_at_the_limit_and_one_past(self):
        codes = (np.arange(4000, dtype=np.uint32) * 7) % 1500
        req = reg.Request(codes, [[1, 1, 1]], rows=(np.arange(4000) % 3 == 0))
        ncodes = int(codes.max()) + 1
        for backend in ("cpu", "cuda"):
            reg.execute(req, "histogram", backend=backend, budget=Budget(states=ncodes))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                reg.execute(req, "histogram", backend=backend, budget=Budget(states=ncodes - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", ncodes))
        reg.execute(req, "histogram", budget=Budget(bytes=8 * ncodes))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            reg.execute(req, "histogram", budget=Budget(bytes=8 * ncodes - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", 8 * ncodes))
        words = (4000 + 63) // 64
        device = 4000 * 4 + words * 8 + 160 + 8 * ncodes                     # inputs, accumulators, device counts
        reg.execute(req, "histogram", backend="cuda", budget=Budget(bytes=device))
        with self.assertRaises(AlgebraBudgetError):
            reg.execute(req, "histogram", backend="cuda", budget=Budget(bytes=device - 1))
        with self.assertRaises(AlgebraBudgetError):                           # the host copy of a CPU finalization
            reg.execute(req, "histogram", backend="cuda", finalize="cpu", budget=Budget(bytes=device))
        reg.execute(req, "histogram", backend="cuda", finalize="cpu", budget=Budget(bytes=device + 8 * ncodes))
        width = (ncodes - 1).bit_length()
        sets = len(plane_sets(width, 2))
        terms = sum(math.comb(width, s) * 2 ** s for s in range(3))
        for backend in ("cpu", "cuda"):
            kw = {"finalize": "cpu"} if backend == "cuda" else {}
            reg.execute(req, "bitplane", backend=backend, budget=Budget(states=sets, circuit=terms), **kw)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                reg.execute(req, "bitplane", backend=backend, budget=Budget(states=sets - 1), **kw)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", sets))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                reg.execute(req, "bitplane", backend=backend, budget=Budget(circuit=terms - 1), **kw)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("circuit", terms))
        small = reg.Request(codes % 100, [[1] * 9])                            # x^8 fits int64 for x < 100
        self.assertEqual(reg.execute(small, "scan")["values"], [sum(sum(int(c) ** k for k in range(9))
                                                                    for c in codes % 100)])
        for backend in ("cpu", "cuda"):
            with self.assertRaises(AlgebraBudgetError) as ctx:                # the kernel's fixed degree limit
                reg.execute(small.with_polys([[1] * 10]), "scan", backend=backend)
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("degree", 9, 8))
        reg.execute(req.with_polys([[1] * 4]), "scan", budget=Budget(degree=3))
        with self.assertRaises(AlgebraBudgetError):
            reg.execute(req.with_polys([[1] * 5]), "scan", budget=Budget(degree=3))
        reg.execute(req.with_polys([[1]] * 4), "scan", budget=Budget(output=4))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            reg.execute(req.with_polys([[1]] * 4), "scan", budget=Budget(output=3))
        self.assertEqual(ctx.exception.resource, "output")
        # an engine column's decoded codes and validity flags (4 + 4 bytes per row) are budgeted before allocation
        db, x, y = self.table(31, 1000)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            reg.Column(db["t"], "x", budget=Budget(bytes=1000 * 8))
        need = ctx.exception.need
        self.assertGreater(need, 1000 * 8)
        col = reg.Column(db["t"], "x", budget=Budget(bytes=need))
        self.assertEqual(col.bytes, need)
        col.close()
        with self.assertRaises(AlgebraBudgetError) as ctx:
            reg.Column(db["t"], "x", budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
        # the bounded candidate search: iterations = the candidates searched
        n_cpu = len(reg.candidates(req))
        self.assertEqual(n_cpu, 5)
        reg.candidates(req, devices=("cpu", "cuda"), budget=Budget(iterations=10))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            reg.candidates(req, devices=("cpu", "cuda"), budget=Budget(iterations=9))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", 10))

    # -------------------------------------------------------------- candidates and dispatch
    def test_fnd005_bounded_candidate_search_and_dispatch_by_entry_name(self):
        db, x, y = self.table(41, 1200)
        col = self.column(db)
        req = reg.Request(col, [[1, 2, 3]], row_ids=db.query(db["t"]).where(db["t"].y < 4).rows())
        cands = reg.candidates(req, devices=("cpu", "cuda"))
        self.assertEqual([(c["representation"], c["device"], c["finalize"]) for c in cands],
                         [("empty", "cpu", "cpu"), ("scan", "cpu", "cpu"), ("histogram", "cpu", "cpu"),
                          ("power", "cpu", "cpu"), ("bitplane", "cpu", "cpu"), ("scan", "cuda", "gpu"),
                          ("histogram", "cuda", "gpu"), ("histogram", "cuda", "cpu"), ("power", "cuda", "cpu"),
                          ("bitplane", "cuda", "cpu")])
        self.assertEqual([c["valid"] for c in cands], [False] + [True] * 9)
        self.assertEqual(cands[0]["refusal"]["kind"], "proof")
        self.assertTrue(reg.candidates(req, empty_proven=True)[0]["valid"])
        for c in cands:
            self.assertEqual(set(c["amounts"]), set(reg.COST_COMPONENTS))
        ncodes = int(max(db["t"].decoded("x"))) + 1
        self.assertEqual(cands[7]["amounts"]["transfer"] - cands[6]["amounts"]["transfer"],
                         (8 * ncodes + 24) - (16 + 24 + 16))       # the counts copied back, versus the sums
        mx = reg.candidates(req.with_polys([[1, 2]], "max"))
        self.assertEqual({c["representation"]: (c["valid"], c["refusal"] and c["refusal"]["kind"]) for c in mx},
                         {"empty": (False, "proof"), "scan": (True, None), "histogram": (True, None),
                          "power": (False, "law"), "bitplane": (False, "conversion")})
        huge = reg.candidates(req.with_polys([[0, 0, 0, 0, 0, 0, 0, 0, (1 << 62)]]))
        self.assertEqual({c["representation"]: c["refusal"] and c["refusal"]["kind"] for c in huge}["power"], "bound")
        state = reg.build_state(req, "power")
        self.addCleanup(state.close)
        with_state = reg.candidates(req, state=state)
        self.assertEqual((with_state[5]["representation"], with_state[5]["reuse"], with_state[5]["valid"]),
                         ("power", True, True))
        stale = reg.candidates(reg.Request(col, [[1, 2, 3]]), state=state)
        self.assertEqual((stale[5]["valid"], stale[5]["refusal"]["kind"]), (False, "stale"))
        self.assertEqual(reg.entry("scan", "cpu")[0], "tr_reg_execute")
        self.assertEqual(reg.entry("histogram", "cuda")[0], "tr_mgpu_registry_execute")
        self.assertEqual(reg.entry("verify", "cuda")[0], "tr_mgpu_registry_verify")
        with self.assertRaises(UnsupportedError):
            reg.entry("empty", "cuda")

    def test_fnd005_disc_verification_through_the_registry(self):
        from torbles import math_disc
        from torbles.relationships import discover
        rng = np.random.default_rng(51)
        xs = [int(v) for v in rng.integers(-50, 50, size=800)]
        ys = [int(v) for v in rng.integers(-20, 90, size=800)]
        s = [a + b for a, b in zip(xs, ys)]
        s[17] += 1
        db = make({"x": xs, "y": ys, "s": s})
        self.addCleanup(db.close)
        cat = discover(db, "t", ["x", "y", "s"], modules=(), max_candidates=0)
        self.addCleanup(cat.close)
        for backend in ("cpu", "cuda"):
            out = reg.verify(cat, "sum", "s", "x", "y", backend=backend)
            direct = math_disc.verify(cat, "sum", "s", "x", "y", backend=backend)
            ev = out["evidence"]
            self.assertEqual({k: ev[k] for k in ("checked", "excluded", "violations", "witness", "usable")},
                             dict(checked=800, excluded=0, violations=1, witness=17, usable=False))
            self.assertEqual({k: ev[k] for k in ("checked", "violations", "witness")},
                             {k: direct[k] for k in ("checked", "violations", "witness")})
            ex = out["explanation"]
            self.assertEqual((ex["module"], ex["capability"], ex["representation"], ex["rows_scanned"]),
                             (4, "DISC-001", "verify", 800))
        need = math_disc.device_bytes(cat, "sum", "s", "x", "y")
        with self.assertRaises(AlgebraBudgetError) as ctx:
            reg.verify(cat, "sum", "s", "x", "y", backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual(ctx.exception.explanation["budget"], dict(resource="bytes", need=need, limit=need - 1))


    # -------------------------------------------------------------- the registry against the code
    def test_fnd005_registry_agrees_with_the_code(self):
        mods = reg.modules()
        gpu = math_gpu.lib()
        inventory = {c["id"] for c in json.loads((ROOT / "docs" / "algebra_inventory.json").read_text())["capabilities"]}
        ids = [m["id"] for m in mods]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(len({m["key"] for m in mods}), len(mods))
        described_cuda = set()
        for m in mods:
            self.assertIn(m["capability"], inventory, m["key"])               # a capability of the backlog
            for e in m["cpu_entries"]:
                self.assertTrue(hasattr(algq.lib, e), (m["key"], e))          # exported by libalg.so
            for e in m["cuda_entries"]:
                self.assertTrue(hasattr(gpu, e), (m["key"], e))               # exported by the CUDA library
            described_cuda |= set(m["cuda_entries"])
            source = ROOT / "torbles" / ("mathematics.c" if m["family"] == "mathematics" else f"math_{m['family']}.c")
            self.assertTrue(source.exists(), (m["key"], source))
            self.assertEqual("cuda" in m["devices"], (ROOT / "torbles" / f"math_{m['family']}_gpu.cu").exists() and
                             bool(m["cuda_entries"]), m["key"])
            if m["id"] < 10:                                                  # the planned representations
                self.assertEqual((reg.REPRESENTATIONS[m["key"]], m["representation"], m["cost_basis"]),
                                 (m["id"], m["key"], "estimate"))
            else:                                                             # families: own APIs, nothing calibrated
                self.assertEqual((m["representation"], m["requests"], m["cost_basis"]), (None, ["family"], "none"))
                self.assertTrue(all(v == 0 for dev in m["costs"].values() for v in dev.values()), m["key"])
        # every family with CUDA kernels is described, and every entry point a kernel source defines is named by a
        # descriptor; the library core (FND-012: device identity and the accumulator self-test) is not a module
        families = {m["family"] for m in mods}
        defined = {path.name[len("math_"):-len("_gpu.cu")]: cuda_entry_points(path.read_text())
                   for path in sorted((ROOT / "torbles").glob("math_*_gpu.cu"))}
        self.assertIn("tr_mgpu_registry_transform", defined["registry"])
        self.assertNotIn("tr_mgpu_disc_verify", defined["registry"])          # declared there, defined in disc
        self.assertEqual(set(defined.pop("core")), {"tr_mgpu_device_name", "tr_mgpu_emulated", "tr_mgpu_core_selftest"})
        for fam, names in defined.items():
            self.assertIn(fam, families, f"torbles/math_{fam}_gpu.cu has no descriptor")
            self.assertTrue(names, fam)
            for name in names:
                self.assertTrue(hasattr(gpu, name), name)
                self.assertIn(name, described_cuda, f"{name} ({fam}) is named by no descriptor")
        for fam in ("scope", "bank", "carriers", "handles", "lin", "opt", "jet", "inv", "calc", "multi", "delta", "bool",
                    "field", "dist", "resp", "quant", "lattice", "spec", "influence"):
            self.assertIn(fam, families)
        # the implementation rows: one table, within the descriptors' devices and carriers
        rows = reg.implementations()
        ops = {o["name"]: o for o in reg.operators()}
        self.assertEqual({r["module"] for r in rows}, set(ids))
        for m in mods:
            mine = [r for r in rows if r["module"] == m["id"]]
            self.assertLessEqual(len(mine), reg.IMPLS, m["key"])
            self.assertEqual([(r["operator"], r["carrier"]) for r in mine], m["implements"], m["key"])
            for r in mine:
                self.assertTrue(r["devices"] and set(r["devices"]) <= set(m["devices"]), (m["key"], r))
                fixed = {c for c in (ops[r["operator"]]["input"], ops[r["operator"]]["output"]) if c}
                self.assertTrue({r["carrier"]} | fixed <= set(m["carriers"]), (m["key"], r))
            if "cuda" in m["devices"]:
                self.assertTrue(any("cuda" in r["devices"] for r in mine), m["key"])
            if m["id"] >= 10:                     # a family states every law its implemented operators require
                need = set().union(*(set(ops[r["operator"]]["requires"]) for r in mine)) - {"exact", "enclosure"}
                self.assertTrue(need <= set(m["laws_relied"]), (m["key"], need - set(m["laws_relied"])))
        # the Python constants and the native tables agree; the tables are complete and consistent
        self.assertEqual(reg.REFUSALS, reg.names("refusals"))
        self.assertEqual([o["id"] for o in reg.operators()], list(range(1, len(ops) + 1)))
        conv = reg.conversions()
        self.assertEqual(len({(c["from"], c["to"]) for c in conv}), len(conv))
        for c in conv:
            self.assertNotIn("any", (c["from"], c["to"]))
            self.assertNotEqual(c["from"], c["to"])
            self.assertEqual(c["via"] is not None, c["kind"] in ("lift", "reduction", "encoding"), c)
        self.assertEqual({c["kind"]: c["via"] for c in conv if c["via"]},
                         {"lift": "lift", "reduction": "residue", "encoding": "null_encode"})
        # lookups by key and by a capability one module holds
        self.assertEqual(reg.module("LIN-002")["key"], "lin.solve")
        self.assertEqual(reg.module("lattice.transform")["capability"], "LATTICE-002")
        with self.assertRaises(ValueError):
            reg.module("FND-004")                                              # the scope and the bank

    # -------------------------------------------------------------- compositions across the new carriers
    def test_fnd005_compositions_across_new_carriers_name_the_missing_law_or_conversion(self):
        laws = [  # steps, input, contract, (step, operator, carrier, missing law)
            ([("poly_integrate", "gfp")], "gfp", "exact", (0, "poly_integrate", "gfp", "char0")),
            ([("order", "gfp")], "gfp", "exact", (0, "order", "gfp", "ordered")),
            ([("euclid", "gfp")], "gfp", "exact", (0, "euclid", "gfp", "ordered")),
            ([("residue", "gfp"), ("roots", "gfp")], "integer", "exact", (1, "roots", "gfp", "ordered")),
            ([("walsh", "gf2")], "gf2", "exact", (0, "walsh", "gf2", "char0")),           # characteristic 2
            ([("logic", "gf2")], "gf2", "exact", (0, "logic", "gf2", "idempotent")),     # XOR is not OR
            ([("divide", "bigint")], "bigint", "exact", (0, "divide", "bigint", "division")),
            ([("solve", "integer")], "integer", "exact", (0, "solve", "integer", "division")),
            ([("divide", "series")], "series", "exact", (0, "divide", "series", "division")),  # x has no inverse
            ([("order", "series")], "series", "exact", (0, "order", "series", "ordered")),
            ([("subtract", "qinterval")], "qinterval", "enclosure", (0, "subtract", "qinterval", "inverse")),
            ([("multiply", "qinterval")], "qinterval", "enclosure", (0, "multiply", "qinterval", "distributive")),
            ([("fold", "qinterval")], "qinterval", "exact", (0, "fold", "qinterval", "exact")),
            ([("fold", "cinterval")], "cinterval", "enclosure", (0, "fold", "cinterval", "associative")),
            ([("occupancy", "kleene")], "kleene", "exact", (0, "occupancy", "kleene", "complement")),
            ([("difference", "kleene")], "kleene", "exact", (0, "difference", "kleene", "inverse")),
            ([("logic", "integer")], "integer", "exact", (0, "logic", "integer", "idempotent")),
            ([("lift", "integer"), ("occupancy", "integer")], "gf2", "exact", (1, "occupancy", "integer", "complement")),
            ([("mobius", "count")], "count", "exact", (0, "mobius", "count", "inverse")),
            ([("walsh", "count")], "count", "exact", (0, "walsh", "count", "inverse")),
        ]
        for steps, inp, contract, (step, op, carrier, law) in laws:
            with self.assertRaises(CarrierError, msg=steps) as ctx:
                reg.check_chain(steps, input=inp, contract=contract)
            r = ctx.exception.refusal
            self.assertEqual((r["kind"], r["step"], r["operator"], r["carrier"], r["law"]),
                             ("law", step, op, carrier, law), steps)
            self.assertIn(law, str(ctx.exception))
            model = MODELS[carrier]                                   # the model has a counterexample
            self.assertFalse(model.exact if law == "exact" else _holds(model, law), (carrier, law))
        conversions = [  # steps, input, contract, the missing conversion, the operator that alone declares it
            ([("fold", "gf2"), ("fold", "integer")], "gf2", "exact", ("gf2", "integer"), "lift"),
            ([("fold", "gf2"), ("convert", "integer"), ("fold", "integer")], "gf2", "exact", ("gf2", "integer"), "lift"),
            ([("fold", "gf2"), ("fold", "count")], "gf2", "exact", ("gf2", "count"), "lift"),
            ([("multiply", "gfp"), ("fold", "integer")], "gfp", "exact", ("gfp", "integer"), "lift"),
            ([("convert", "gfp")], "integer", "exact", ("integer", "gfp"), "residue"),
            ([("fold", "count"), ("fold", "gf2")], "count", "exact", ("count", "gf2"), "residue"),
            ([("enclose", "interval"), ("divide", "rational")], "interval", "enclosure", ("interval", "rational"), None),
            ([("convert", "bigq")], "interval", "enclosure", ("interval", "bigq"), None),
            ([("convert", "qinterval"), ("convert", "bigq")], "interval", "enclosure", ("qinterval", "bigq"), None),
            ([("enclose", "cinterval"), ("convert", "rational")], "cinterval", "enclosure", ("cinterval", "rational"),
             None),
            ([("convert", "interval")], "bigq", "exact", ("bigq", "interval"), None),        # enclosing: not lossless
            ([("logic", "kleene"), ("occupancy", "boolean")], "kleene", "exact", ("kleene", "boolean"), "null_encode"),
            ([("logic", "kleene"), ("convert", "boolean")], "kleene", "exact", ("kleene", "boolean"), "null_encode"),
            ([("logic", "kleene"), ("histogram", "code")], "kleene", "exact", ("kleene", "code"), "null_encode"),
            ([("logic", "kleene"), ("fold", "count")], "kleene", "exact", ("kleene", "count"), None),
            ([("multiply", "series"), ("convert", "bigq")], "series", "exact", ("series", "bigq"), None),
        ]
        for steps, inp, contract, conv, via in conversions:
            with self.assertRaises(CarrierError, msg=steps) as ctx:
                reg.check_chain(steps, input=inp, contract=contract)
            r = ctx.exception.refusal
            self.assertEqual((r["kind"], r["conversion"], r["via"]), ("conversion", conv, via), steps)
            self.assertIn(f"{conv[0]} -> {conv[1]}", str(ctx.exception))
            if via:
                self.assertIn(f"'{via}'", str(ctx.exception))
        # the oracles of these declarations: GF(2) and GF(p) sums are not integer sums (they agree mod p only), an
        # interval encloses a value without being one, Kleene's UNKNOWN has no Boolean value
        bits = [1, 1, 1, 0, 1]
        parity = 0
        for b in bits:
            parity ^= b
        self.assertEqual((parity, sum(bits), sum(bits) % 2), (0, 4, parity))
        self.assertNotEqual((P61 - 1 + 5) % P61, P61 - 1 + 5)
        lo, hi = _tadd((0.1, 0.1), (0.2, 0.2))
        self.assertTrue(lo < hi and Fraction(lo) < Fraction(0.1) + Fraction(0.2) < Fraction(hi))
        accepted = [  # steps, input, contract, output, every step implemented by a module
            ([("fold", "gf2"), ("lift", "integer"), ("fold", "integer")], "gf2", "exact", "integer", True),
            ([("fold", "count"), ("residue", "gf2"), ("fold", "gf2")], "count", "exact", "gf2", False),
            ([("residue", "gfp"), ("solve", "gfp"), ("crt", "gfp"), ("convert", "bigq")], "integer", "exact", "bigq",
             True),
            ([("residue", "gfp"), ("ratrecon", "gfp")], "bigq", "exact", "bigq", True),
            ([("convert", "bigq"), ("solve", "bigq"), ("convert", "rational")], "bigint", "exact", "rational", True),
            ([("eval", "series"), ("coefficients", "series"), ("convert", "rational")], "series", "exact", "rational",
             True),
            ([("predicate", "integer"), ("logic", "kleene"), ("null_encode", "boolean"), ("occupancy", "boolean")],
             "integer", "exact", "count", True),
            ([("predicate", "integer"), ("logic", "kleene"), ("null_encode", "code"), ("histogram", "code"),
              ("quantify", "count")], "integer", "exact", "kleene", True),
            ([("roots", "bigq")], "bigq", "exact", "qinterval", True),
            ([("enclose", "qinterval"), ("fold", "qinterval"), ("convert", "interval")], "qinterval", "enclosure",
             "interval", True),
            ([("convert", "cinterval"), ("enclose", "cinterval")], "interval", "enclosure", "cinterval", True),
            ([("convert", "int128"), ("mobius", "int128")], "count", "exact", "int128", True),
            ([("bits", "code"), ("convert", "gf2"), ("solve", "gf2")], "code", "exact", "gf2", True),
            ([("convert", "interval")], "bigq", "enclosure", "interval", True),
        ]
        for steps, inp, contract, out, implemented in accepted:
            self.assertEqual(reg.check_chain(steps, input=inp, contract=contract), out, steps)
            res = reg.explain_chain(steps, input=inp, contract=contract, implemented=True)
            self.assertEqual(res["accepted"], implemented, steps)
            if implemented:                               # each step names the module implementing it
                self.assertEqual(len(res["modules"]), len(steps))
                for step, mid in zip(steps, res["modules"]):
                    self.assertIn(step, reg.module(mid)["implements"], steps)
            else:
                self.assertEqual((res["refusal"]["kind"], res["refusal"]["step"]), ("implementation", 1))

    # -------------------------------------------------------------- explanations of two families' executions
    def test_fnd005_explanations_of_lattice_and_walsh_transforms(self):
        from torbles import math_lattice as ml, math_spec as ms
        rng = np.random.default_rng(61)
        lat, wal = reg.module("lattice.transform"), reg.module("spec.walsh")
        self.assertEqual((lat["capability"], wal["capability"]), ("LATTICE-002", "SPEC-002"))
        where = "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"
        entry = {("cpu", "lattice"): "tr_lattice_transform", ("cuda", "lattice"): "tr_mgpu_lattice_transform",
                 ("cpu", "spec"): "tr_spec_walsh", ("cuda", "spec"): "tr_mgpu_spec_walsh"}
        for k in (0, 1, 3, 7):
            n = 1 << k
            counts = [int(x) for x in rng.integers(0, 1 << 40, size=n)]
            signed = [int(x) for x in rng.integers(-(1 << 50), 1 << 50, size=n)]
            cases = [("zeta", "up", "count", "count", counts), ("zeta", "down", "int128", "count", counts),
                     ("mobius", "up", "int128", "count", counts), ("mobius", "down", "integer", "integer", signed),
                     ("zeta", "up", "integer", "integer", signed), ("walsh", "up", "int128", "count", counts),
                     ("walsh", "up", "int128", "int128", signed)]
            for op, direction, carrier, inp, vals in cases:
                expect = transform_oracle(op, direction, vals)
                mod = wal if op == "walsh" else lat
                for backend in ("cpu", "cuda"):
                    out = reg.transform(vals, op, carrier=carrier, input=inp, direction=direction, backend=backend)
                    self.assertEqual(out["values"], expect, (k, op, direction, carrier, backend))
                    ex = out["explanation"]
                    self.assertEqual((ex["status"], ex["module"], ex["module_name"], ex["capability"], ex["family"],
                                      ex["representation"], ex["refusal"], ex["budget"]),
                                     ("ran", mod["id"], mod["name"], mod["capability"], mod["family"], None, None, None))
                    self.assertEqual(ex["entry"], entry[(backend, mod["family"])])
                    self.assertIn(ex["entry"], mod["cpu_entries"] if backend == "cpu" else mod["cuda_entries"])
                    self.assertEqual(ex["device"], "cpu" if backend == "cpu" else where)
                    self.assertEqual((ex["state_elements"], ex["rows_scanned"]), (n, 0))
                    self.assertEqual(ex["device_bytes"], 0 if backend == "cpu" else 32 * n + 8)
                if op == "walsh":                         # the family's own API agrees
                    self.assertEqual([int(v) for v in ms.walsh(vals)["spectrum"]], expect)
                else:
                    self.assertEqual([int(v) for v in ml.transform(vals, f"{op}_{direction}")["values"]], expect)
        v = [int(x) for x in rng.integers(0, 100, size=16)]
        z = reg.transform(v, "zeta", carrier="count")["values"]
        self.assertEqual(reg.transform(z, "mobius", carrier="int128", input="count", backend="cuda")["values"], v)

        def refused(fn, exc, before=True):
            with self.assertRaises(exc) as ctx:
                fn()
            ex = ctx.exception.explanation
            self.assertEqual(ex["status"], "refused")
            if before:                                    # decided before the module ran: no copy, no launch
                self.assertEqual((ex["device_bytes"], ex["native_calls"]), (0, 0), ex)
            return ex
        for backend in ("cpu", "cuda"):
            ex = refused(lambda: reg.transform(v, "mobius", carrier="count", backend=backend), CarrierError)
            self.assertEqual((ex["refusal"]["kind"], ex["refusal"]["law"], ex["refusal"]["carrier"], ex["module"]),
                             ("law", "inverse", "count", 0))
            ex = refused(lambda: reg.transform([1, 0, 1, 1], "walsh", carrier="gf2", backend=backend), CarrierError)
            self.assertEqual((ex["refusal"]["kind"], ex["refusal"]["law"]), ("law", "char0"))
            ex = refused(lambda: reg.transform(v, "zeta", carrier="int128", input="kleene", backend=backend), CarrierError)
            self.assertEqual((ex["refusal"]["kind"], ex["refusal"]["conversion"]), ("conversion", ("kleene", "int128")))
            ex = refused(lambda: reg.transform([1, 1], "zeta", carrier="integer", input="gf2", backend=backend),
                         CarrierError)
            self.assertEqual((ex["refusal"]["conversion"], ex["refusal"]["via"]), (("gf2", "integer"), "lift"))
            ex = refused(lambda: reg.transform(v, "walsh", carrier="integer", backend=backend), CarrierError)
            self.assertEqual((ex["refusal"]["kind"], ex["refusal"]["operator"], ex["refusal"]["carrier"]),
                             ("implementation", "walsh", "integer"))
            big = [1 << 63, 1 << 63]
            ex = refused(lambda: reg.transform(big, "zeta", carrier="count", backend=backend), OverflowError)
            self.assertEqual((ex["refusal"]["kind"], ex["module"]), ("bound", lat["id"]))      # total 2^64 > uint64
            self.assertEqual(reg.transform(big, "zeta", carrier="int128", input="count", backend=backend)["values"],
                             [1 << 64, 1 << 63])
            ex = refused(lambda: reg.transform([1 << 62, 1 << 62], "zeta", carrier="integer", backend=backend),
                         OverflowError, before=False)
            self.assertEqual(ex["refusal"]["kind"], "overflow")              # 2^63 leaves int64: the module refuses
            refused(lambda: reg.transform([-1, 2], "zeta", carrier="count", backend=backend), ValueError)
            reg.transform(v, "walsh", backend=backend, budget=Budget(tensor=16))
            ex = refused(lambda: reg.transform(v, "walsh", backend=backend, budget=Budget(tensor=15)), AlgebraBudgetError)
            self.assertEqual(ex["budget"], dict(resource="tensor", need=16, limit=15))
            reg.transform(v, "zeta", backend=backend, budget=Budget(iterations=4))
            ex = refused(lambda: reg.transform(v, "zeta", backend=backend, budget=Budget(iterations=3)),
                         AlgebraBudgetError)
            self.assertEqual((ex["budget"], ex["module"]), (dict(resource="iterations", need=4, limit=3), lat["id"]))
        ex = refused(lambda: reg.transform(v, "zeta", carrier="bigint", backend="cuda"), CarrierError)
        self.assertEqual((ex["refusal"]["kind"], ex["refusal"]["carrier"], ex["module"]), ("implementation", "bigint", 0))
        ex = refused(lambda: reg.transform(v, "zeta", carrier="bigint"), UnsupportedError)   # FND-009 handles
        self.assertEqual((ex["module"], ex["refusal"]["kind"]), (lat["id"], "unsupported"))
        with self.assertRaises(ValueError):
            reg.transform([1, 2, 3], "zeta")                                    # not 2^k values

if __name__ == "__main__":
    unittest.main()
