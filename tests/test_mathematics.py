"""Exact function calculus and immutable query-derived sufficient states."""
import ctypes as C
from fractions import Fraction
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
import torbles
from torbles import algq
from torbles.mathematics import (AlgebraBudgetError, Context, Function, MomentBasis,
                                 _api, _Rational, _Term)


class NativeFunctionCalculus(unittest.TestCase):
    def test_higher_derivatives_of_exact_parameter_function(self):
        context = Context()
        t = context.parameter("t")
        derivative = Fraction(1, 5)*t**5 - 2*t**3 + 7*t - 9
        for expected in (-1, 8, 36, 48, 24, 0):
            derivative = derivative.diff(t)
            self.assertEqual(derivative.evaluate(t=2), expected)

    def test_exact_rational_normalization_and_cancellation(self):
        context = Context()
        x = context.symbol("x")
        expression = Fraction(1, 3)*x + Fraction(1, 6)*x
        self.assertEqual(expression.evaluate(x=3), Fraction(3, 2))
        self.assertEqual(expression, Fraction(1, 2)*x)
        zero = expression - expression
        self.assertEqual(zero.term_count, 0)
        self.assertEqual(zero.evaluate(), 0)
        self.assertEqual(expression.integral(x).diff(x), expression)

    def test_multivariate_gradient_and_hessian(self):
        context = Context()
        x, y = context.symbols("x", "y")
        expression = Fraction(1, 3)*(x+y)**3 + x*y
        self.assertEqual(expression(x=2, y=3), Fraction(143, 3))
        self.assertEqual(expression.diff(x)(x=2, y=3), 28)
        self.assertEqual(expression.diff("y")(x=2, y=3), 27)
        self.assertEqual(expression.diff(x).diff(x)(x=2, y=3), 10)
        self.assertEqual(expression.diff(x).diff(y)(x=2, y=3), 11)
        self.assertEqual(expression.diff(x).diff(y), expression.diff(y).diff(x))

    def test_definite_integral_preserves_other_parameters_and_input(self):
        context = Context()
        x, y = context.symbols("x", "y")
        expression = x*x + y*x
        integrated = expression.integral(x, 0, 3)
        self.assertEqual(integrated.support, ("y",))
        self.assertEqual(integrated(y=2), 18)
        self.assertEqual(expression(x=2, y=3), 10)
        self.assertEqual(x.integral(x, 0, Fraction(1, 2))(), Fraction(1, 8))
        self.assertEqual(x.integral(x, Fraction(1, 2), 0)(), Fraction(-1, 8))

    def test_binding_and_domain_errors_are_explicit(self):
        context = Context()
        x, y = context.symbols("x", "y")
        with self.assertRaisesRegex(ValueError, "missing"):
            (x+y)(x=1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            x(z=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            x.evaluate({"x": 1}, x=2)
        with self.assertRaises(TypeError):
            x(x=0.5)
        with self.assertRaises(ValueError):
            x + Context().symbol("x")
        with self.assertRaises(ValueError):
            x.integral(x, 0)

    def test_degree_term_and_variable_budgets_do_not_change_inputs(self):
        context = Context()
        symbols = context.symbols(*(f"x{index}" for index in range(8)))
        x = symbols[0]
        with self.assertRaises(AlgebraBudgetError):
            x**9
        expression = context.constant(1)
        with self.assertRaises(AlgebraBudgetError):
            for symbol in symbols:
                expression = expression*(1+symbol)
        with self.assertRaises(AlgebraBudgetError):
            context.symbol("ninth")
        self.assertEqual(x(x0=7), 7)
        self.assertLessEqual(expression.term_count, 128)

    def test_overflow_never_wraps_and_failed_operators_preserve_inputs(self):
        context = Context()
        x = context.symbol("x")
        maximum = (1 << 63)-1
        constant = context.constant(maximum)
        with self.assertRaises(OverflowError):
            constant + 1
        with self.assertRaises(OverflowError):
            (maximum*x)(x=2)
        expression = maximum*x*x
        with self.assertRaises(OverflowError):
            expression.diff(x)
        self.assertEqual(expression(x=1), maximum)
        self.assertEqual(constant(), maximum)
        with self.assertRaises(OverflowError):
            context.constant(Fraction(1, 1 << 63))

    def test_individual_ownership_and_idempotent_close(self):
        context = Context()
        x = context.symbol("x")
        expression = x*x + 1
        x.close()
        x.close()
        self.assertEqual(expression(x=4), 17)
        with self.assertRaises(RuntimeError):
            x.evaluate(x=4)
        with context.constant(7) as value:
            self.assertEqual(value(), 7)
        with self.assertRaises(RuntimeError):
            value()

    def test_ctypes_term_layout_and_native_explanation(self):
        self.assertEqual(C.sizeof(_Term), 24)
        self.assertEqual(_Term.coefficient.offset, 8)
        context = Context()
        x = context.symbol("x")
        terms = (Fraction(2, 3)*x*x - Fraction(1, 2)).terms
        self.assertEqual(dict(terms)[(2, 0, 0, 0, 0, 0, 0, 0)], Fraction(2, 3))
        self.assertEqual(dict(terms)[(0,)*8], Fraction(-1, 2))


class QueryMomentSnapshots(unittest.TestCase):
    def make(self, values=None):
        if values is None:
            values = dict(x=[-3, -3, 1, 4, 7, 1], y=[5, 5, -2, 8, 0, -2], flag=[1, 1, 1, 0, 1, 1])
        frame = pd.DataFrame({name: pd.Series(items, dtype=object) for name, items in values.items()})
        ddl = "CREATE TABLE t (" + ",".join('"'+name+'" BIGINT' for name in values) + ")"
        db = torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)
        self.addCleanup(db.close)
        return db, db["t"], values

    def basis(self, query, columns, **kwargs):
        basis = MomentBasis(query, columns, **kwargs)
        self.addCleanup(basis.close)
        return basis

    def test_signed_offsets_bags_filters_and_query_ownership(self):
        db, table, values = self.make()
        query = db.query(table).where(table.flag == 1).sum(table.y)
        original_measure, original_pred = query.measure, query.pred
        original_keys, original_joins = query.keys, query.joins
        basis = self.basis(query, [table.x, "y"])
        rows = [(x, y) for x, y, keep in zip(values["x"], values["y"], values["flag"]) if keep]
        x, y = basis.variables.values()
        self.assertEqual(basis.row_count, len(rows))
        self.assertEqual(basis.moment((1, 0)), sum(x for x, _ in rows))
        self.assertEqual(basis.moment((1, 1)), sum(x*y for x, y in rows))
        self.assertEqual(basis.reduce(x*x + 2*x*y + y*y)(), sum((x+y)**2 for x, y in rows))
        self.assertIs(query.measure, original_measure)
        self.assertIs(query.pred, original_pred)
        self.assertIs(query.keys, original_keys)
        self.assertIs(query.joins, original_joins)
        self.assertEqual(query.run(), sum(y for _, y in rows))
        self.assertEqual(basis.metadata["native_query_calls"], 6)
        with self.assertRaises(TypeError):
            basis.snapshot["row_count"] = 123

    def test_loss_curve_gradient_hessian_and_duckdb_reference(self):
        db, table, values = self.make()
        basis = self.basis(db.query(table).where(table.flag == 1), ["x", "y"])
        x, y = basis.variables.values()
        a, b = basis.parameter("a"), basis.parameter("b")
        function = basis.reduce((y-a*x-b)**2)
        self.assertIsInstance(function, Function)
        rows = [(x, y) for x, y, keep in zip(values["x"], values["y"], values["flag"]) if keep]
        aa, bb = Fraction(2, 3), Fraction(-1, 2)
        expected = sum((y-aa*x-bb)**2 for x, y in rows)
        self.assertEqual(function(a=aa, b=bb), expected)
        self.assertEqual(function.diff(a)(a=aa, b=bb), sum(-2*x*(y-aa*x-bb) for x, y in rows))
        self.assertEqual(function.diff(b)(a=aa, b=bb), sum(-2*(y-aa*x-bb) for x, y in rows))
        self.assertEqual(function.diff(a).diff(a)(), 2*sum(x*x for x, _ in rows))
        self.assertEqual(function.diff(a).diff(b)(), 2*sum(x for x, _ in rows))
        self.assertEqual(function.diff(b).diff(b)(), 2*len(rows))
        expected_sql = db.con.execute("SELECT sum((y-2*x+1)*(y-2*x+1)) FROM t WHERE flag=1").fetchone()[0]
        self.assertEqual(function(a=2, b=-1), expected_sql)

    def test_after_materialization_calculus_and_reduction_read_no_rows(self):
        db, table, values = self.make()
        basis = self.basis(db.query(table), ["x", "y"])
        x, y = basis.variables.values()
        a, b = basis.parameter("a"), basis.parameter("b")
        rows = list(zip(values["x"], values["y"]))
        with patch.object(algq.Query, "run", side_effect=AssertionError("unexpected source query")):
            function = basis.reduce((y-a*x-b)**2)
            derivative = function.diff(a)
            integrated = function.integral(a, 0, 1)
            self.assertEqual(integrated.support, ("b",))
            for value in range(-10, 11):
                aa, bb = Fraction(value, 3), Fraction(1, 2)
                self.assertEqual(function(a=aa, b=bb), sum((y-aa*x-bb)**2 for x, y in rows))
                self.assertEqual(derivative(a=aa, b=bb), sum(-2*x*(y-aa*x-bb) for x, y in rows))
            self.assertEqual(integrated(b=0), sum(y*y-x*y+Fraction(x*x, 3) for x, y in rows))

    def test_empty_domain_is_zero_linear_functional(self):
        db, table, _ = self.make()
        query = db.query(table).where(table.flag < 0)
        self.assertIsNone(algq.copy_query(query).sum(table.x).run())
        basis = self.basis(query, ["x", "y"])
        x, y = basis.variables.values()
        a = basis.parameter("a")
        function = basis.reduce((y-a*x)**2 + 7)
        self.assertEqual(basis.row_count, 0)
        self.assertEqual(basis.moment((2, 0)), 0)
        self.assertEqual(function(), 0)
        self.assertEqual(function(a=13), 0)
        self.assertEqual(function.diff(a)(), 0)

    def test_snapshot_lifetime_and_source_changes_are_explicit(self):
        db, table, values = self.make()
        basis = self.basis(db.query(table), ["x"])
        x = basis.variables["x"]
        a = basis.parameter("a")
        function = basis.reduce((x-a)**2)
        expected = sum((x-2)**2 for x in values["x"])
        table.flip(0, 0)
        table.repair()
        self.assertEqual(function(a=2), expected)
        metadata = function.metadata
        basis.close()
        basis.close()
        self.assertEqual(function(a=2), expected)
        self.assertEqual(metadata["lifetime"], "immutable snapshot; later source mutations do not refresh it")
        with self.assertRaises(RuntimeError):
            basis.reduce(x)

    def test_snapshot_construction_backend_parity_and_device_evidence(self):
        db, table, values = self.make()
        expected = sum(x*y for x, y in zip(values["x"], values["y"]))
        for engine in ("alg", "plain", "gpu"):
            with self.subTest(engine=engine):
                basis = self.basis(db.query(table), ["x", "y"], engine=engine)
                x, y = basis.variables.values()
                self.assertEqual(basis.reduce(x*y)(), expected)
                self.assertEqual(basis.metadata["engine_requested"], engine)
                self.assertEqual(len(basis.metadata["actual_devices"]), 6)
                self.assertTrue(all(device != "unknown" for device in basis.metadata["actual_devices"]))
                self.assertEqual(basis.metadata["coefficient_executor"], "native CPU")
                if engine == "gpu":
                    self.assertEqual(basis.metadata["gpu_executor"], algq.gpu_name())
                    self.assertEqual(basis.metadata["gpu_emulated"], "emulat" in algq.gpu_name().lower())

    def test_mutation_between_passes_is_detected_even_if_repaired(self):
        db, table, _ = self.make()
        original_run = algq.Query.run
        calls = []
        def mutate(query, engine="alg"):
            result = original_run(query, engine)
            calls.append(1)
            if len(calls) == 2:
                table.flip(0, 0)
                table.repair()
            return result
        with patch.object(algq.Query, "run", mutate), self.assertRaisesRegex(RuntimeError, "source snapshot changed"):
            MomentBasis(db.query(table), ["x", "y"])

    def test_equijoin_selection_and_dimension_revision_guard(self):
        db = torbles.from_ddl(
            "CREATE TABLE d (id INTEGER PRIMARY KEY, keep INTEGER); "
            "CREATE TABLE t (x BIGINT, fk INTEGER REFERENCES d(id))",
            {"d": pd.DataFrame(dict(id=[1, 2], keep=[1, 0])),
             "t": pd.DataFrame(dict(x=[-2, 3, 4, 9], fk=[1, 2, 1, 99]))}, keep_plain=True)
        self.addCleanup(db.close)
        table, dim = db["t"], db["d"]
        query = db.query(table).join(dim).where(dim.keep == 1)
        basis = self.basis(query, ["x"])
        self.assertEqual(basis.row_count, 2)
        self.assertEqual(basis.moment((1,)), 2)
        original_run = algq.Query.run
        calls = []
        def mutate(current, engine="alg"):
            result = original_run(current, engine)
            calls.append(1)
            if len(calls) == 2:
                dim.flip(0, 0)
                dim.repair()
            return result
        with patch.object(algq.Query, "run", mutate), self.assertRaisesRegex(RuntimeError, "source snapshot changed"):
            MomentBasis(query, ["x"])

    def test_mutable_used_domain_change_is_detected(self):
        db, table, _ = self.make()
        original_run = algq.Query.run
        original_bitmap = table.used_bm.copy()
        calls = []
        def mutate(current, engine="alg"):
            result = original_run(current, engine)
            calls.append(1)
            if len(calls) == 2:
                table.used_bm[0] ^= np.uint64(1)
            return result
        try:
            with patch.object(algq.Query, "run", mutate), self.assertRaisesRegex(RuntimeError, "source snapshot changed"):
                MomentBasis(db.query(table), ["x"])
        finally:
            table.used_bm[:] = original_bitmap

    def test_predicate_graph_and_join_scope_are_copied(self):
        db, table, values = self.make()
        predicate = table.flag == 1
        query = db.query(table).where(predicate)
        original_run = algq.Query.run
        calls = []
        def mutate_original(current, engine="alg"):
            result = original_run(current, engine)
            calls.append(1)
            if len(calls) == 1:
                predicate.v = 0
                query.joins.clear()
            return result
        with patch.object(algq.Query, "run", mutate_original):
            basis = self.basis(query, ["x"])
        expected = [x for x, flag in zip(values["x"], values["flag"]) if flag == 1]
        self.assertEqual(basis.row_count, len(expected))
        self.assertEqual(basis.moment((1,)), sum(expected))
        self.assertEqual(predicate.v, 0)

    def test_unsupported_coordinates_and_scope_fail_before_queries(self):
        db, table, _ = self.make(dict(x=[1, None, 2], y=[1, 2, 3], flag=[0, 1, 1]))
        with patch.object(algq.Query, "run", side_effect=AssertionError("must refuse before execution")):
            for query, columns in ((db.query(table), ["x"]), (db.query(table), []),
                                   (db.query(table), ["y", "y"]), (db.query(table).group_by(table.flag), ["y"])):
                with self.subTest(columns=columns), self.assertRaises(ValueError):
                    MomentBasis(query, columns)
        db2 = torbles.from_ddl("CREATE TABLE t (x DOUBLE, y DECIMAL(8,2))",
                              {"t": pd.DataFrame(dict(x=[1.5, 2.5], y=[1, 2]))})
        self.addCleanup(db2.close)
        for column in ("x", "y"):
            with self.assertRaises(ValueError):
                MomentBasis(db2.query("t"), [column])

    def test_native_missing_moment_and_aggregate_type_boundaries(self):
        db, table, _ = self.make()
        basis = self.basis(db.query(table), ["x"])
        x = basis.variables["x"]
        a = basis.parameter("a")
        with self.assertRaises(AlgebraBudgetError):
            basis.reduce(x**3)
        function = basis.reduce((x-a)**2)
        with self.assertRaises(ValueError):
            basis.reduce(function)
        with self.assertRaises(ValueError):
            function + x
        with self.assertRaises(ValueError):
            function.diff(x)
        with self.assertRaises(ValueError):
            function.substitute(x, 3)
        with self.assertRaises(ValueError):
            function(x=2, a=1)
        with self.assertRaises(ValueError):
            basis.parameter("x")

    def test_moment_construction_refuses_unrepresentable_result(self):
        db, table, _ = self.make(dict(x=[1 << 40, (1 << 40)+1]))
        with self.assertRaises(OverflowError):
            MomentBasis(db.query(table), ["x"])

    def test_native_bank_copies_inputs_and_rejects_missing_basis(self):
        native = _api()
        exponents = (C.c_uint8*3)(0, 1, 2)
        values = (C.c_int64*3)(2, 3, 5)
        error = C.c_int32()
        handle = native["bank_create"](1, 3, exponents, values, C.byref(error))
        self.assertTrue(handle)
        self.assertEqual(error.value, 0)
        try:
            values[1] = 999
            output = C.c_int64()
            key = (C.c_uint8*1)(1)
            self.assertEqual(native["bank_moment"](handle, key, C.byref(output)), 0)
            self.assertEqual(output.value, 3)
        finally:
            native["bank_free"](handle)
        self.assertFalse(native["bank_create"](1, 2, exponents, values, C.byref(error)))
        self.assertEqual(error.value, -1)
        values = (C.c_int64*3)(0, 1, 0)
        self.assertFalse(native["bank_create"](1, 3, exponents, values, C.byref(error)))
        self.assertEqual(error.value, -1)


if __name__ == "__main__":
    unittest.main()
