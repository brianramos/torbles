"""Hybrid regions must change native execution while preserving query semantics."""
import ctypes
import unittest

import pandas as pd
import torbles
from torbles.algq import A_GENERAL, Col, M, MInstr, QPlan, lib
from torbles.relationships import discover


class RelationshipDAG(unittest.TestCase):
    def make(self, n=513, *, nullable=False, unique_groups=False):
        values = dict(x=[None if nullable and i % 7 == 0 else i % 8 for i in range(n)],
                      y=[None if nullable and i % 11 == 0 else (i * 3) % 7 for i in range(n)],
                      z=[i % 5 for i in range(n)], flag=[i % 3 for i in range(n)],
                      # Force a sparse domain (>2^22) without allocating millions
                      # of dense output slots for the degree-three measure.
                      bucket=[i * 257 for i in range(n)] if unique_groups else [i % 4 for i in range(n)])
        frame = pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in values.items()})
        db = torbles.from_ddl(
            "CREATE TABLE t (x BIGINT, y BIGINT, z BIGINT, flag BIGINT, bucket BIGINT)",
            {"t": frame}, keep_plain=True)
        cat = discover(db, "t", ["x", "y", "z"], modules=(), system_strategy="hybrid")
        self.addCleanup(cat.close)
        return db, db["t"], cat, values

    @staticmethod
    def expression(t):
        x, y, z = Col(t, "x"), Col(t, "y"), Col(t, "z")
        u = (x + y) * (x - y) + y * y
        return u * z + u * z

    @staticmethod
    def expected(values, *, keep=lambda i: True, grouped=False):
        result = {} if grouped else 0
        for i, (x, y, z, bucket) in enumerate(zip(values["x"], values["y"], values["z"], values["bucket"])):
            if not keep(i) or x is None or y is None:
                continue
            value = 2 * x * x * z
            if grouped:
                key = (bucket,)
                result[key] = result.get(key, 0) + value
            else:
                result += value
        return result

    @staticmethod
    def instructions(q):
        return [(m.op, m.arg, m.c) for m in q.last_plan.meas[:q.last_plan.nmeas]]

    @classmethod
    def arithmetic_count(cls, q):
        return sum(op in (M["ADD"], M["SUB"], M["MUL"]) for op, _, _ in cls.instructions(q))

    def assert_hybrid(self, q):
        self.assertEqual(q.system_plan["choice"], "hybrid")
        self.assertEqual(q.last_plan.agg, A_GENERAL)
        self.assertGreater(q.system_plan["dag_nodes"], 0)
        self.assertLessEqual(q.last_plan.nmeas, 48)

    def test_local_combination_and_sharing_execute_on_cubic_root(self):
        db, t, _, values = self.make()
        q = db.query(t).sum(self.expression(t))
        self.assertEqual(q.run(), self.expected(values))
        self.assert_hybrid(q)
        self.assertEqual(q.system_plan["chain_ops"], 13)
        self.assertEqual(q.system_plan["hybrid_ops"], 3)
        self.assertEqual(self.arithmetic_count(q), 3)
        self.assertGreater(q.system_plan["hybrid_regions"], 0)
        self.assertGreater(q.system_plan["hybrid_registers"], 0)
        ops = [op for op, _, _ in self.instructions(q)]
        self.assertIn(M["SAVE"], ops)
        self.assertIn(M["LOAD"], ops)
        self.assertEqual(q.system_plan["combine_cost"], -1)
        self.assertFalse(q.last_plan.wide)

    def test_auto_selects_reduced_local_regions(self):
        db, t, cat, values = self.make()
        cat.system_strategy = "auto"
        q = db.query(t).sum(self.expression(t))
        self.assertEqual(q.run(), self.expected(values))
        self.assert_hybrid(q)
        self.assertLess(q.system_plan["hybrid_cost"], q.system_plan["chain_cost"])

    def test_negative_expansion_keeps_factored_region(self):
        db, t, _, values = self.make()
        x, y, z = Col(t, "x"), Col(t, "y"), Col(t, "z")
        q = db.query(t).sum((x + y) * (x + y) * z)
        self.assertEqual(q.run(), sum((x+y)*(x+y)*z for x, y, z in zip(values["x"], values["y"], values["z"])))
        self.assert_hybrid(q)
        self.assertEqual(q.system_plan["hybrid_ops"], 3)
        self.assertEqual(self.arithmetic_count(q), 3)
        self.assertEqual(q.system_plan["hybrid_regions"], 0)
        self.assertGreater(q.system_plan["hybrid_registers"], 0)

    def test_commutative_equivalents_share_one_native_result(self):
        db, t, _, values = self.make()
        x, y, z = Col(t, "x"), Col(t, "y"), Col(t, "z")
        q = db.query(t).sum((x+y)*z + (y+x)*z)
        self.assertEqual(q.run(), sum(2*(x+y)*z for x, y, z in zip(values["x"], values["y"], values["z"])))
        self.assert_hybrid(q)
        self.assertEqual(self.arithmetic_count(q), 3)
        self.assertGreater(q.system_plan["hybrid_registers"], 0)

    def test_mode_switches_restore_original_source_and_reuse_its_cache(self):
        db, t, cat, values = self.make()
        q = db.query(t).sum(self.expression(t))
        want = self.expected(values)
        for mode, choice, count in (("hybrid", "hybrid", 3), ("chain", "chain", 13),
                                    ("combine", "preserve", 13), ("auto", "hybrid", 3),
                                    ("chain", "chain", 13), ("hybrid", "hybrid", 3)):
            with self.subTest(mode=mode):
                cat.system_strategy = mode
                self.assertEqual(q.run(), want)
                self.assertEqual(q.system_plan["choice"], choice)
                self.assertEqual(self.arithmetic_count(q), count)
                if choice != "hybrid":
                    self.assertNotIn(M["LOAD"], [m[0] for m in self.instructions(q)])
        self.assertEqual(q.system_plan["cache_hit"], 1)
        self.assertFalse(q.system_plan["profile_ran"])

    def test_warm_execution_keeps_source_trace_and_skips_profile(self):
        db, t, _, values = self.make()
        q = db.query(t).sum(self.expression(t))
        self.assertEqual(q.run(), self.expected(values))
        original = self.instructions(q)
        first = dict(q.system_plan)
        self.assertEqual(first["cache_hit"], 0)
        self.assertTrue(first["profile_ran"])
        self.assertEqual(q.run(), self.expected(values))
        self.assertEqual(self.instructions(q), original)
        self.assertEqual(q.system_plan["chain_ops"], 13)
        self.assertEqual(q.system_plan["cache_hit"], 1)
        self.assertFalse(q.system_plan["profile_ran"])

    def test_group_filter_and_partial_final_word_keep_the_same_domain(self):
        db, t, _, values = self.make(513)
        q = db.query(t).where(t.flag == 1).group_by(t.bucket).sum(self.expression(t))
        self.assertEqual(q.run(), self.expected(values, keep=lambda i: values["flag"][i] == 1, grouped=True))
        self.assert_hybrid(q)

    def test_nullable_operands_keep_explicit_validity_mask(self):
        db, t, _, values = self.make(nullable=True)
        valid = (Col(t, t.nullable["x"]) == 1) & (Col(t, t.nullable["y"]) == 1)
        q = db.query(t).where(valid & (t.flag == 1)).group_by(t.bucket).sum(self.expression(t))
        want = self.expected(values, keep=lambda i: values["flag"][i] == 1, grouped=True)
        self.assertEqual(q.run(), want)
        self.assert_hybrid(q)

    def test_sparse_output_growth_survives_restoring_original_measure(self):
        db, t, cat, values = self.make(20001, unique_groups=True)
        q = db.query(t).group_by(t.bucket).sum(self.expression(t))
        want = self.expected(values, grouped=True)
        for mode in ("hybrid", "chain", "hybrid"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), want)
            self.assertEqual(q.last_plan.out_n, 20001)
            self.assertGreaterEqual(q.last_plan.out_cap, 20001)

    def test_table_revision_invalidates_hybrid_cache(self):
        db, t, _, values = self.make()
        q = db.query(t).sum(self.expression(t))
        want = self.expected(values)
        self.assertEqual(q.run(), want)
        self.assertEqual(q.run(), want)
        self.assertEqual(q.system_plan["cache_hit"], 1)
        t.flip(0, 0)
        t.repair()
        self.assertEqual(q.run(), want)
        self.assert_hybrid(q)
        self.assertEqual(q.system_plan["cache_hit"], 0)

    def test_plain_backend_uses_original_program(self):
        db, t, _, values = self.make()
        q = db.query(t).sum(self.expression(t))
        self.assertEqual(q.run(), self.expected(values))
        self.assert_hybrid(q)
        self.assertEqual(q.run("plain"), self.expected(values))
        self.assertEqual(q.system_plan["choice"], "preserve")
        self.assertEqual(q.device, "plain")
        self.assertEqual(self.arithmetic_count(q), 13)

    def test_gpu_cubic_fallback_is_reported_as_cpu(self):
        db, t, _, values = self.make()
        q = db.query(t).sum(self.expression(t))
        self.assertEqual(q.run("gpu"), self.expected(values))
        self.assert_hybrid(q)
        self.assertTrue(q.device.startswith("cpu"), q.device)
        self.assertIn("gpu", q.device.lower())

    def test_native_executor_rejects_uninitialized_or_out_of_bounds_registers(self):
        db, t, cat, _ = self.make()
        cat.system_strategy = "chain"
        q = db.query(t).sum(self.expression(t))
        q.run()
        plan = q.last_plan
        invalid = [
            [MInstr(M["LOAD"], 0, 0)],
            [MInstr(M["SAVE"], 0, 0)],
            [MInstr(M["CONST"], 0, 1), MInstr(M["SAVE"], 48, 0)],
            [MInstr(M["CONST"], 0, 1), MInstr(M["LOAD"], -1, 0)],
            [MInstr(M["CONST"], 0, 1), MInstr(M["CONST"], 0, 2)],
        ]
        for program in invalid:
            with self.subTest(program=[(m.op, m.arg) for m in program]):
                plan.nmeas = len(program)
                for i, instruction in enumerate(program):
                    plan.meas[i] = instruction
                self.assertEqual(lib.at_query(t.at, ctypes.byref(plan)), -1)
        plan.nmeas = 49
        self.assertEqual(lib.at_query(t.at, ctypes.byref(plan)), -1)

    def test_coefficient_overflow_keeps_exact_modular_chain_region(self):
        # The direct C executor has modular 64-bit arithmetic. A polynomial
        # coefficient that cannot fit int64 must make that algebraic alternative
        # unavailable, while a shared original operation remains valid.
        db, t, cat, values = self.make()
        cat.system_strategy = "chain"
        q = db.query(t).sum(self.expression(t))
        q.run()
        plan = QPlan.from_buffer_copy(bytes(q.last_plan))
        coefficient = (1 << 63) - 1
        product = [MInstr(M["FCOL"], t.idx("x"), 0),
                   MInstr(M["CONST"], 0, coefficient), MInstr(M["MUL"], 0, 0)]
        program = product + product + [MInstr(M["ADD"], 0, 0)]
        plan.agg, plan.nmeas = A_GENERAL, len(program)
        for i, instruction in enumerate(program):
            plan.meas[i] = instruction
        cat.system_strategy = "hybrid"
        info = cat._plan_system(plan, "cpu")
        self.assertEqual(info["choice"], "hybrid")
        self.assertEqual(info["combine_cost"], -1)
        self.assertEqual(lib.at_query(t.at, ctypes.byref(plan)), 0)
        want = sum(2 * coefficient * x for x in values["x"]) & ((1 << 64) - 1)
        if want >= 1 << 63:
            want -= 1 << 64
        self.assertEqual(plan.total_sum, want)


if __name__ == "__main__":
    unittest.main()
