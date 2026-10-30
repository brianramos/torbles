"""Physical-decision reuse must preserve the live query's scope and buffers."""
import ctypes
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torbles
from torbles.algq import A_BIL, A_GENERAL, M, MInstr, QPlan, lib
from torbles.relationships import discover


class PhysicalDecisionCache(unittest.TestCase):
    def make(self, n=513, *, unique_groups=False):
        values = dict(x=[i % 8 for i in range(n)], y=[(i * 3) % 7 for i in range(n)],
                      bucket=list(range(n)) if unique_groups else [i % 4 for i in range(n)])
        db = torbles.from_ddl(
            "CREATE TABLE t (x BIGINT, y BIGINT, bucket BIGINT)",
            {"t": pd.DataFrame(values)}, keep_plain=True)
        t = db["t"]
        cat = discover(db, "t", ["x", "y"], modules=())
        self.addCleanup(cat.close)
        return db, t, cat, values

    @staticmethod
    def query(db, t, *, predicate=None, grouped=False):
        q = db.query(t)
        if predicate is not None: q = q.where(predicate)
        if grouped: q = q.group_by(t.bucket)
        return q.sum((t.x + t.y) * (t.x - t.y))

    @staticmethod
    def expected(values, *, predicate=lambda x, y: True, grouped=False):
        result = {} if grouped else 0
        for x, y, bucket in zip(values["x"], values["y"], values["bucket"]):
            if not predicate(x, y): continue
            value = (x + y) * (x - y)
            if grouped: result[(bucket,)] = result.get((bucket,), 0) + value
            else: result += value
        return result

    def test_warm_hit_retains_profile_evidence_without_resampling(self):
        db, t, _, values = self.make()
        q = self.query(db, t)
        expected = self.expected(values)
        self.assertEqual(q.run(), expected)
        first = dict(q.system_plan)
        self.assertEqual(first["cache_hit"], 0)
        self.assertTrue(first["profile_ran"])
        self.assertGreater(first["sample_words"], 0)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 1)
        self.assertFalse(q.system_plan["profile_ran"])
        self.assertEqual(q.system_plan["sample_words"], first["sample_words"])
        self.assertEqual(q.system_plan["passing_rows"], first["passing_rows"])

    def test_empty_sample_decision_is_reused_without_changing_live_representation(self):
        # The profile samples 32 equispaced words out of 64; word 1 is omitted.
        # A missed match is a reason to preserve the incoming executor, not an
        # empty-result proof. Its physical decision is still reusable.
        n = 4096
        values = dict(x=[i % 8 for i in range(n)], y=[(i * 3) % 7 for i in range(n)],
                      marker=[int(i == 65) for i in range(n)])
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, marker BIGINT)",
                             {"t": pd.DataFrame(values)}, keep_plain=True)
        t = db["t"]
        cat = discover(db, "t", ["x", "y"], modules=())
        self.addCleanup(cat.close)
        q = db.query(t).where(t.marker == 1).sum(t.x * t.y)
        self.assertEqual(q.run(), 6)
        self.assertEqual(q.last_plan.agg, A_BIL)
        self.assertEqual(q.system_plan["choice"], "preserve")
        self.assertEqual(q.system_plan["reason"], "empty sample")
        self.assertEqual(q.system_plan["cache_hit"], 0)
        self.assertGreater(q.system_plan["sample_words"], 0)
        self.assertEqual(q.system_plan["passing_rows"], 0)
        self.assertTrue(q.system_plan["profile_ran"])
        self.assertEqual(q.run(), 6)
        self.assertEqual(q.system_plan["cache_hit"], 1)
        self.assertFalse(q.system_plan["profile_ran"])
        live = q.last_plan
        live.agg, live.nmeas = A_GENERAL, 3
        live.meas[0] = MInstr(M["FCOL"], t.idx("x"), 0)
        live.meas[1] = MInstr(M["FCOL"], t.idx("y"), 0)
        live.meas[2] = MInstr(M["MUL"], 0, 0)
        live.bil_a = live.bil_b = t.idx("marker")
        # Exercise direct mutable-plan use. Query.run intentionally restores its
        # immutable original source when changing physical strategies.
        info = cat._plan_system(live, "cpu")
        self.assertEqual(lib.at_query(t.at, ctypes.byref(live)), 0)
        self.assertEqual(live.total_sum, 6)
        self.assertEqual(info["cache_hit"], 1)
        self.assertEqual(info["original_agg"], A_GENERAL)
        self.assertEqual(info["selected_agg"], A_GENERAL)
        self.assertEqual(live.agg, A_GENERAL)
        self.assertFalse(info["profile_ran"])

    def test_filters_measure_keys_and_strategy_have_separate_decisions(self):
        db, t, cat, values = self.make()
        cases = [
            (self.query(db, t), self.expected(values)),
            (self.query(db, t, predicate=t.x < 3),
             self.expected(values, predicate=lambda x, y: x < 3)),
            (self.query(db, t, grouped=True), self.expected(values, grouped=True)),
            (db.query(t).sum(t.x + 2 * t.y), sum(x + 2*y for x, y in zip(values["x"], values["y"]))),
        ]
        for q, expected in cases:
            self.assertEqual(q.run(), expected)
            self.assertEqual(q.system_plan["cache_hit"], 0)
            self.assertEqual(q.run(), expected)
            self.assertEqual(q.system_plan["cache_hit"], 1)
        q, expected = cases[0]
        cat.system_strategy = "chain"
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["choice"], "chain")
        self.assertEqual(q.system_plan["cache_hit"], 0)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 1)
        cat.system_strategy = "auto"
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 1)

    def test_table_revision_invalidates_the_decision(self):
        db, t, _, values = self.make()
        q = self.query(db, t)
        expected = self.expected(values)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 1)
        t.flip(0, 0)
        t.repair()
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 0)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 1)

    def test_matrix_policy_environment_invalidates_the_decision(self):
        db, t, _, values = self.make()
        q = self.query(db, t, grouped=True)
        expected = self.expected(values, grouped=True)
        for policy in ("0", "1"):
            with patch.dict(os.environ, {"TORBLES_MM_ROWS": policy}):
                self.assertEqual(q.run(), expected)
                self.assertEqual(q.system_plan["cache_hit"], 0)
                self.assertEqual(q.run(), expected)
                self.assertEqual(q.system_plan["cache_hit"], 1)

    def test_hit_only_patches_measure_and_preserves_live_buffers(self):
        db, t, cat, _ = self.make()
        q = self.query(db, t, grouped=True)
        q.run()
        live = QPlan.from_buffer_copy(bytes(q.last_plan))
        # The planner never dereferences these output pointers. Different live
        # storage must survive applying a decision made for a previous plan.
        fields = dict(out_key=0x1230, out_sum=0x2340, out_cnt=0x3450,
                      out_sum_hi=0x4560, out_key_hi=0x5670, out_rows=0x6780,
                      out_cap=87654, out_n=1357, total_sum=-987, total_cnt=2468)
        for name, value in fields.items(): setattr(live, name, value)
        trace = cat._plan_system(live, "cpu")
        self.assertEqual(trace["cache_hit"], 1)
        for name, value in fields.items(): self.assertEqual(getattr(live, name), value, name)

    def test_shared_decision_survives_independent_sparse_buffer_growth(self):
        # sparse group buffers: the matrix product's output (a GROUP BY with this many groups otherwise takes
        # the row kernel's dense arrays, algq.ROWPATH_GROUPS)
        from torbles import algq as _algq
        saved = _algq.ROWPATH_GROUPS; _algq.ROWPATH_GROUPS = 0
        self.addCleanup(setattr, _algq, "ROWPATH_GROUPS", saved)
        db, t, _, values = self.make(17001, unique_groups=True)
        expected = self.expected(values, grouped=True)
        q1, q2 = self.query(db, t, grouped=True), self.query(db, t, grouped=True)
        self.assertEqual(q1.run(), expected)
        self.assertGreaterEqual(q1.last_plan.out_cap, len(expected))
        self.assertEqual(q2.run(), expected)
        self.assertEqual(q2.system_plan["cache_hit"], 1)
        self.assertNotEqual(q2.last_plan.out_key, q1.last_plan.out_key)
        self.assertGreaterEqual(q2.last_plan.out_cap, len(expected))
        self.assertEqual(q1.run(), expected)
        self.assertEqual(q1.system_plan["cache_hit"], 1)

    def test_mutable_external_scopes_bypass_cache_and_observe_new_contents(self):
        db, t, cat, values = self.make()
        cat.system_strategy = "chain"
        q = self.query(db, t, grouped=True)
        q.run()
        q.run()
        self.assertEqual(q.system_plan["cache_hit"], 1)
        nw = (len(values["x"]) + 63) // 64
        scopes = []
        xrows = (ctypes.c_uint64 * nw)(*([(1 << 64) - 1] * nw))
        live = QPlan.from_buffer_copy(bytes(q.last_plan))
        live.xmode, live.xrows = 1, ctypes.addressof(xrows)
        scopes.append((live, xrows, None))
        semibm = (ctypes.c_uint64 * 1)(15)
        live = QPlan.from_buffer_copy(bytes(q.last_plan))
        live.nsemi, live.semi_col[0] = 1, t.idx("bucket")
        live.semi_bm[0], live.semi_n[0] = ctypes.addressof(semibm), 4
        scopes.append((live, semibm, None))
        ranges = (ctypes.c_uint64 * 2)(0, nw)
        live = QPlan.from_buffer_copy(bytes(q.last_plan))
        live.nranges, live.ranges = 1, ctypes.addressof(ranges)
        scopes.append((live, ranges, 1))
        for live, external, only_index in scopes:
            with self.subTest(xmode=live.xmode, nsemi=live.nsemi, nranges=live.nranges):
                for _ in range(2):
                    trace = cat._plan_system(live, "cpu")
                    self.assertEqual(trace["cache_hit"], 0)
                    self.assertEqual(trace["passing_rows"], len(values["x"]))
                for i in range(len(external)):
                    if only_index is None or i == only_index: external[i] = 0
                trace = cat._plan_system(live, "cpu")
                self.assertEqual(trace["cache_hit"], 0)
                self.assertEqual(trace["passing_rows"], 0)
        live = QPlan.from_buffer_copy(bytes(q.last_plan))
        live.nattr = 1
        self.assertEqual(cat._plan_system(live, "cpu")["cache_hit"], 0)

    def test_live_external_bitmap_mutation_changes_executed_results(self):
        db, t, cat, values = self.make()
        cat.system_strategy = "chain"
        q = self.query(db, t, grouped=True)
        expected = self.expected(values, grouped=True)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 1)
        mask = np.full((len(values["x"]) + 63) // 64, (1 << 64) - 1, dtype=np.uint64)
        q.last_plan.xmode, q.last_plan.xrows = 1, mask.ctypes.data
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["cache_hit"], 0)
        self.assertTrue(q.system_plan["profile_ran"])
        mask.fill(0)
        self.assertEqual(q.run(), {})
        self.assertEqual(q.system_plan["cache_hit"], 0)
        self.assertTrue(q.system_plan["profile_ran"])
        self.assertEqual(q.system_plan["passing_rows"], 0)

    def test_gpu_never_samples_to_compare_an_unavailable_chain(self):
        db, t, cat, values = self.make()
        q = self.query(db, t, grouped=True)
        expected = self.expected(values, grouped=True)
        for mode in ("auto", "combine", "chain"):
            cat.system_strategy = mode
            self.assertEqual(q.run("gpu"), expected)
            self.assertEqual(q.device, "gpu")
            self.assertEqual(q.system_plan["sample_words"], 0)
            self.assertEqual(q.system_plan["sampled_rows"], 0)
            self.assertEqual(q.system_plan["gpu_profile_skipped"], 1)
            self.assertFalse(q.system_plan["profile_ran"])
            self.assertEqual(q.run("gpu"), expected)
            self.assertEqual(q.system_plan["cache_hit"], 1)


if __name__ == "__main__":
    unittest.main()
