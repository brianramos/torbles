"""Physical operation chaining versus joint algebra, through actual query execution.

These tests force both native realizations before checking automatic decisions.
They deliberately use no learned equations where testing execution choice, so a
rewrite trace cannot masquerade as a physical chain/combined-system decision.
"""
import unittest

import pandas as pd
import torbles
from torbles.algq import A_GENERAL, A_MM, A_POLYG, Col
from torbles.relationships import discover


class RelationshipSystems(unittest.TestCase):
    def make(self, values, types=None):
        types = types or {}
        frame = pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in values.items()})
        ddl = "CREATE TABLE t (" + ", ".join(
            '"' + k.replace('"', '""') + '" ' + types.get(k, "BIGINT") for k in values) + ")"
        return torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)

    def catalog(self, db, columns=("x", "y"), **options):
        cat = discover(db, "t", list(columns), modules=(), **options)
        self.addCleanup(cat.close)
        return cat

    @staticmethod
    def expr(t):
        x, y = Col(t, "x"), Col(t, "y")
        return (x + y) * (x - y)

    def narrow(self, n=4096):
        values = dict(x=[i % 8 for i in range(n)],
                      y=[(i * 3) % 7 for i in range(n)],
                      bucket=[i % 4 for i in range(n)])
        return self.make(values), values

    def wide_sparse(self, n=4096):
        values = dict(x=[(i * 324983) % (1 << 20) for i in range(n)],
                      y=[(i * 50777) % (1 << 20) for i in range(n)],
                      flag=[int(i % 64 == 0) for i in range(n)],
                      bucket=[i % 4 for i in range(n)])
        return self.make(values), values

    @staticmethod
    def expected(values, keep=None, group=None):
        total = {} if group else 0
        for i, (x, y) in enumerate(zip(values["x"], values["y"])):
            if keep is not None and not keep(i): continue
            value = (int(x) + int(y)) * (int(x) - int(y))
            if group:
                key = (values[group][i],)
                total[key] = total.get(key, 0) + value
            else: total += value
        return total

    def assert_real_choice(self, q, wanted):
        self.assertEqual(q.system_plan["choice"], wanted)
        if wanted == "chain": self.assertEqual(q.last_plan.agg, A_GENERAL)
        elif wanted == "combine": self.assertIn(q.last_plan.agg, (A_POLYG, A_MM))

    def test_force_both_realizations_preserves_grouped_answers(self):
        db, values = self.narrow()
        t = db["t"]
        cat = self.catalog(db)
        q = db.query(t).group_by(t.bucket).sum(self.expr(t))
        expected = self.expected(values, group="bucket")
        for mode in ("chain", "combine"):
            with self.subTest(mode=mode):
                cat.system_strategy = mode
                self.assertEqual(q.run(), expected)
                self.assert_real_choice(q, mode)

    def test_auto_combines_narrow_dense_system(self):
        db, values = self.narrow()
        t = db["t"]
        self.catalog(db, system_strategy="auto")
        q = db.query(t).sum(self.expr(t))
        self.assertEqual(q.run(), self.expected(values))
        self.assert_real_choice(q, "combine")
        self.assertLess(q.system_plan["combine_cost"], q.system_plan["chain_cost"])
        self.assertGreater(q.system_plan["sample_words"], 0)

    def test_auto_chains_wide_selective_system(self):
        db, values = self.wide_sparse()
        t = db["t"]
        cat = self.catalog(db)
        q = db.query(t).where(t.flag == 1).group_by(t.bucket).sum(self.expr(t))
        expected = self.expected(values, keep=lambda i: values["flag"][i] == 1, group="bucket")
        for mode in ("combine", "chain", "auto"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), expected)
            self.assert_real_choice(q, "chain" if mode == "auto" else mode)
        self.assertLess(q.system_plan["chain_cost"], q.system_plan["combine_cost"])
        self.assertGreater(q.system_plan["passing_rows"], 0)
        # Every physical word has exactly one matching row.
        self.assertEqual(q.system_plan["passing_rows"], q.system_plan["sample_words"])

    def test_cached_query_replans_when_strategy_changes(self):
        db, values = self.narrow()
        t = db["t"]
        cat = self.catalog(db)
        q = db.query(t).group_by(t.bucket).sum(self.expr(t))
        for mode in ("chain", "combine", "chain", "auto", "combine"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), self.expected(values, group="bucket"))
            self.assert_real_choice(q, "combine" if mode == "auto" else mode)

    def test_forced_choices_preserve_nullable_filter_domain(self):
        n = 513  # includes a partial final bit-plane word
        values = dict(x=[i % 9 for i in range(n)], y=[(i * 3) % 7 for i in range(n)],
                      flag=[None if i % 5 == 0 else i % 2 for i in range(n)])
        db = self.make(values)
        t = db["t"]
        cat = self.catalog(db)
        q = db.query(t).where(t.flag == 1).sum(self.expr(t))
        expected = self.expected(values, keep=lambda i: values["flag"][i] == 1)
        for mode in ("chain", "combine"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), expected)
            self.assert_real_choice(q, mode)

    def test_null_operands_with_explicit_validity_mask(self):
        values = dict(x=[None if i % 7 == 0 else i % 8 for i in range(513)],
                      y=[None if i % 11 == 0 else i % 7 for i in range(513)])
        db = self.make(values)
        t = db["t"]
        cat = self.catalog(db)
        valid = (Col(t, t.nullable["x"]) == 1) & (Col(t, t.nullable["y"]) == 1)
        q = db.query(t).where(valid).sum(self.expr(t))
        expected = self.expected(values, keep=lambda i: values["x"][i] is not None and values["y"][i] is not None)
        for mode in ("chain", "combine"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), expected)
            self.assert_real_choice(q, mode)

    def test_semijoin_selectivity_preserves_native_join_domain(self):
        n = 4096
        values = dict(x=[(i * 324983) % (1 << 20) for i in range(n)],
                      y=[(i * 50777) % (1 << 20) for i in range(n)],
                      fk=[i % 64 for i in range(n)], bucket=[i % 4 for i in range(n)])
        db = torbles.from_ddl(
            "CREATE TABLE d (id INTEGER PRIMARY KEY, keep INTEGER); "
            "CREATE TABLE t (x BIGINT, y BIGINT, fk INTEGER REFERENCES d(id), bucket INTEGER)",
            {"d": pd.DataFrame(dict(id=list(range(64)), keep=[int(i == 0) for i in range(64)])),
             "t": pd.DataFrame(values)}, keep_plain=True)
        t, d = db["t"], db["d"]
        cat = self.catalog(db)
        q = db.query(t).join(d).where(d.keep == 1).group_by(t.bucket).sum(self.expr(t))
        q.join_mode = "lookup"  # the profile must honor the native semijoin bitmap
        expected = self.expected(values, keep=lambda i: values["fk"][i] == 0, group="bucket")
        for mode in ("combine", "chain", "auto"):
            cat.system_strategy = mode
            self.assertEqual(q.run(), expected)
            self.assert_real_choice(q, "chain" if mode == "auto" else mode)
        self.assertEqual(q.system_plan["passing_rows"], q.system_plan["sample_words"])

    def test_wide_sum_is_not_mislabeled_as_operation_chain(self):
        values = dict(x=[(1 << 40) + i for i in range(64)],
                      y=[(1 << 20) + i % 7 for i in range(64)])
        db = self.make(values)
        t = db["t"]
        self.catalog(db, system_strategy="chain")
        q = db.query(t).sum(t.x * t.y)
        self.assertEqual(q.run(), sum(x * y for x, y in zip(values["x"], values["y"])))
        self.assertTrue(q.last_plan.wide)
        self.assertEqual(q.system_plan["choice"], "preserve")
        self.assertIn("wide", q.system_plan["reason"].lower())

    def test_plain_backend_retains_its_baseline_execution(self):
        db, values = self.narrow()
        t = db["t"]
        self.catalog(db, system_strategy="combine")
        q = db.query(t).group_by(t.bucket).sum(self.expr(t))
        self.assertEqual(q.run("plain"), self.expected(values, group="bucket"))
        self.assertEqual(q.system_plan["choice"], "preserve")
        self.assertIn("baseline", q.system_plan["reason"].lower())

    def test_gpu_does_not_attempt_unsupported_general_chain(self):
        db, values = self.narrow()
        t = db["t"]
        self.catalog(db, system_strategy="chain")
        q = db.query(t).group_by(t.bucket).sum(self.expr(t))
        self.assertEqual(q.run("gpu"), self.expected(values, group="bucket"))
        self.assertNotEqual(q.last_plan.agg, A_GENERAL)
        self.assertIn(q.system_plan["choice"], ("preserve", "combine"))
        self.assertIn("gpu", q.system_plan["reason"].lower())
        self.assertEqual(q.device, "gpu")

    def test_forced_gpu_combination_preserves_results(self):
        db, values = self.narrow()
        t = db["t"]
        self.catalog(db, system_strategy="combine")
        q = db.query(t).group_by(t.bucket).sum(self.expr(t))
        self.assertEqual(q.run("gpu"), self.expected(values, group="bucket"))
        self.assert_real_choice(q, "combine")
        self.assertEqual(q.device, "gpu")

    def test_sparse_group_buffers_survive_strategy_and_backend_switches(self):
        n = 20001  # exceeds the initial sparse result capacity of 16384
        values = dict(x=[i % 8 for i in range(n)], y=[(i * 3) % 7 for i in range(n)],
                      bucket=list(range(n)))
        db = self.make(values)
        t = db["t"]
        cat = self.catalog(db)
        q = db.query(t).group_by(t.bucket).sum(self.expr(t))
        expected = self.expected(values, group="bucket")
        for mode, engine in (("chain", "alg"), ("combine", "alg"),
                             ("combine", "gpu"), ("chain", "alg")):
            with self.subTest(mode=mode, engine=engine):
                cat.system_strategy = mode
                self.assertEqual(q.run(engine), expected)
                self.assert_real_choice(q, mode)
                self.assertTrue(q.last_plan.out_key)
                self.assertEqual(q.last_plan.out_n, n)
                self.assertGreaterEqual(q.last_plan.out_cap, n)

    def test_degree_growth_keeps_chain_when_combination_is_unavailable(self):
        db, values = self.narrow(513)
        t = db["t"]
        cat = self.catalog(db)
        q = db.query(t).sum(t.x * t.y * (t.x + t.y))
        expected = sum(x*y*(x+y) for x, y in zip(values["x"], values["y"]))
        cat.system_strategy = "combine"
        self.assertEqual(q.run(), expected)
        self.assertEqual(q.system_plan["choice"], "preserve")
        self.assertEqual(q.last_plan.agg, A_GENERAL)
        self.assertEqual(q.system_plan["reason"], "unsupported")
        cat.system_strategy = "auto"
        self.assertEqual(q.run(), expected)
        self.assert_real_choice(q, "chain")

    def test_empty_word_ranges_preserve_sum_of_no_rows(self):
        n = 256
        db = torbles.from_ddl(
            "CREATE TABLE d (id INTEGER PRIMARY KEY); "
            "CREATE TABLE t (x INTEGER, y INTEGER, fk INTEGER REFERENCES d(id))",
            {"d": pd.DataFrame(dict(id=list(range(64)))),
             "t": pd.DataFrame(dict(x=[i % 8 for i in range(n)], y=[i % 7 for i in range(n)],
                                    fk=[i // 4 for i in range(n)]))}, keep_plain=True)
        t, d = db["t"], db["d"]
        cat = self.catalog(db)
        q = db.query(t).join(d).where(Col(d, "id") < 0).sum(self.expr(t))
        q.join_mode = "lookup"
        for mode in ("chain", "combine", "auto"):
            cat.system_strategy = mode
            self.assertIsNone(q.run())
            self.assertEqual(q.last_plan.nranges, 1)
            self.assertEqual(q.system_plan["sample_words"], 0)
        self.assertEqual(q.system_plan["choice"], "preserve")
        self.assertIn("empty", q.system_plan["reason"].lower())

    def test_stale_relation_proofs_do_not_disable_safe_physical_choice(self):
        db, values = self.narrow()
        t = db["t"]
        self.catalog(db)
        t.flip(0, 0)
        t.repair()
        q = db.query(t).sum(self.expr(t))
        self.assertEqual(q.run(), self.expected(values))
        self.assertEqual(q.relationship_plan["status"], "stale")
        self.assert_real_choice(q, "combine")

    def test_bad_strategy_is_rejected_at_api_boundary(self):
        db, _ = self.narrow(64)
        with self.assertRaises(ValueError): self.catalog(db, system_strategy="fuse_everything")
        cat = self.catalog(db)
        with self.assertRaises(ValueError): cat.system_strategy = "sometimes"


if __name__ == "__main__":
    unittest.main()
