# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-005: the native module and operator registry (torbles/math_registry.c, math_registry_gpu.cu).

Oracles, independent of the implementation: the law table checked by brute force on sample elements of Python models
of each carrier (int, Fraction, bool, GF(2) by xor/and, naturals, residues mod 2^128, outward-rounded intervals by
math.nextafter, floats, min, frozensets); polynomial aggregates recomputed row by row with Python integers and with
DuckDB SQL (HUGEINT); histograms by numpy.bincount; plane counts and power sums by Python loops; the first slice's own
registry (math_domains.modules) and its unchanged test suite. Every executor also runs on the CUDA path (emulated
here, scrambled thread order) and must equal the CPU bit for bit. Seeds are fixed.
"""
import ctypes as C
import itertools
import math
import unittest
from fractions import Fraction

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


# ------------------------------------------------------------------ independent models of the carriers
def _down(x):
    return -math.inf if x == -math.inf else math.nextafter(x, -math.inf)


def _up(x):
    return math.inf if x == math.inf else math.nextafter(x, math.inf)


class Model:
    """a carrier as Python operations: plus (the fold), times, zero, one, sample elements, the carrier's own
    comparison (None when it has none), negation (an additive inverse candidate) and reciprocal candidates"""
    def __init__(self, elements, plus, times, zero, one, leq=None, neg=None, recip=None, eq=None, exact=True):
        self.elements, self.plus, self.times, self.zero, self.one = elements, plus, times, zero, one
        self.leq, self.neg, self.recip, self.exact = leq, neg, recip, exact
        self.eq = eq or (lambda a, b: a == b)


def _iv_add(a, b):
    return (_down(a[0] + b[0]), _up(a[1] + b[1]))


def _iv_mul(a, b):
    p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return (_down(min(p)), _up(max(p)))


def _feq(a, b):
    return a == b or (a != a and b != b)


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
    if law == "char0":                                     # 0, 1, 1+1, ... pairwise distinct
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
        self.assertEqual(set(mods), set(range(1, 10)))
        self.assertEqual({i: mods[i]["capability"] for i in mods},
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
        for m in mods.values():
            self.assertTrue(m["cpu_entries"], m["name"])
            for e in m["cpu_entries"]:
                self.assertTrue(hasattr(algq.lib, e), e)
            for e in m["cuda_entries"]:
                self.assertTrue(hasattr(gpu, e), e)
            self.assertEqual(bool(m["cuda_entries"]), "cuda" in m["devices"], m["name"])
            for op, carrier in m["implements"]:            # no module implements what the laws forbid
                self.assertTrue(reg.explain_chain([(op, carrier)], input=carrier)["accepted"], (m["name"], op, carrier))
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
                     "mul_idempotent", "ordered", "division", "char0")
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


if __name__ == "__main__":
    unittest.main()
