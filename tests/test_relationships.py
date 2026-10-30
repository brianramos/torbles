"""Native discovery proofs and actual query rewrites (no remote data required).

Run: python -m unittest discover -s tests -p test_relationships.py -v
"""
import tempfile
import unittest

import pandas as pd
import torbles
from torbles.algq import Col
from torbles.relationships import discover


class NativeRelationships(unittest.TestCase):
    def make(self, values, types=None):
        types = types or {}
        frame = pd.DataFrame({name: pd.Series(v, dtype=object) for name, v in values.items()})
        ddl = "CREATE TABLE t (" + ", ".join('"' + c.replace('"', '""') + '" ' + types.get(c, "BIGINT") for c in values) + ")"
        return torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)

    def catalog(self, db, columns=None, **options):
        cat = discover(db, "t", columns, **options)
        self.addCleanup(cat.close)
        return cat

    @staticmethod
    def col(table, name):
        return Col(table, name)

    def test_affine_discovery_uses_original_signed_values(self):
        xs = [-17, -11, -3, 0, 8, 19]
        db = self.make({"x": xs, "y": [2 * x + 3 for x in xs]})
        cat = self.catalog(db, modules=("affine",))
        rule = next(r for r in cat.rules if r["expression"] == "y = 2*x + 3")
        self.assertTrue(rule["usable"])
        self.assertEqual((rule["checked"], rule["excluded"], rule["violations"]), (6, 0, 0))

    def test_sum_and_product_discovered_with_offsets(self):
        xs = [-3, -3, -1, -1, 4, 4]
        ys = [2, 5, 2, 5, 2, 5]
        db = self.make({"x": xs, "y": ys, "sum_xy": [x+y for x, y in zip(xs, ys)],
                        "prod_xy": [x*y for x, y in zip(xs, ys)]})
        cat = self.catalog(db, modules=("sum", "product"), max_candidates=1000)
        expressions = {r["expression"] for r in cat.rules}
        self.assertIn("sum_xy = x + y", expressions)
        self.assertIn("prod_xy = x * y", expressions)

    def test_square_module(self):
        xs = [-3, -2, -1, 1, 2, 4]
        db = self.make({"x": xs, "y": [x*x for x in xs]})
        cat = self.catalog(db, modules=("product",))
        self.assertIn("y = x * x", {r["expression"] for r in cat.rules})

    def test_failed_proof_cannot_cancel_residuals(self):
        db = self.make({"x": [1, 2, 3, 4], "y": [6, 6, 9, 11]})
        cat = self.catalog(db, modules=())
        proof = cat.probe("affine", "y", "x", scale=2, bias=3)
        self.assertFalse(proof["usable"])
        self.assertEqual(proof["violations"], 2)
        self.assertEqual(proof["checked"], 4)
        self.assertEqual(proof["witness"]["stored_row"], 0)
        self.assertEqual(proof["witness"]["values"], {"y": 6, "x": 1})
        self.assertEqual(cat.rules, [])

    def test_nullable_proof_is_evidence_not_rewrite(self):
        db = self.make({"x": [1, 2, None, 4], "y": [5, 7, 9, 11]})
        cat = self.catalog(db, modules=())
        proof = cat.probe("affine", "y", "x", scale=2, bias=3)
        self.assertEqual((proof["checked"], proof["excluded"], proof["violations"]), (3, 1, 0))
        self.assertFalse(proof["usable"])
        self.assertEqual(cat.rules, [])

    def test_empty_constant_and_all_null_are_not_proofs(self):
        for values in ({"x": [], "y": []}, {"x": [1]*4, "y": [5]*4},
                       {"x": [None]*4, "y": [None]*4}):
            with self.subTest(values=values):
                db = self.make(values)
                cat = self.catalog(db, modules=())
                proof = cat.probe("affine", "y", "x", scale=2, bias=3)
                self.assertFalse(proof["usable"])
                self.assertEqual(cat.rules, [])

    def test_native_candidate_budget_and_explicit_registration(self):
        db = self.make({"x": [1, 2, 3], "y": [5, 7, 9]})
        cat = self.catalog(db, max_candidates=0)
        self.assertEqual(cat.rules, [])
        proof = cat.probe("affine", "y", "x", scale=2, bias=3)
        self.assertTrue(proof["usable"])
        self.assertEqual(len(cat.rules), 1)
        cat.probe("affine", "y", "x", scale=2, bias=3)
        self.assertEqual(len(cat.rules), 1, "duplicate proof must not duplicate rules")

    def test_native_stats_report_budget_and_decoded_memory(self):
        db = self.make({"x": [1, 2, 3], "y": [5, 7, 9], "z": [1, 4, 9]})
        cat = self.catalog(db, modules=("affine",), max_candidates=1)
        stats = cat.stats
        self.assertEqual(stats["candidates"], 1)
        self.assertEqual(stats["truncated"], 1)
        self.assertEqual(stats["refused"], 0)
        self.assertEqual(stats["rows"], 3)
        self.assertEqual(stats["used_rows"], 3)
        self.assertEqual(stats["decoded_bytes"], 3*3*4)
        self.assertEqual(stats["rules"], len(cat.rules))

    def test_native_refuses_overflowing_logical_codec_metadata(self):
        # Defense at the native ABI boundary: an invalid offset must never wrap
        # into a purported relationship proof or an optimizer rule.
        db = self.make({"x": [1, 2, 3], "y": [2, 4, 6], "z": [2, 8, 18]})
        db["t"].codecs["x"].offset = (1 << 63)-1
        cat = self.catalog(db, modules=("product",), max_candidates=100)
        self.assertGreater(cat.stats["refused"], 0)
        self.assertEqual(cat.rules, [])
        with self.assertRaisesRegex(ValueError, "exact native limits"):
            cat.probe("product", "z", "x", "y")

    def test_physical_column_ids_are_independent_of_catalog_order_and_names(self):
        xs, ys = [2, 2, 3, 3], [4, 5, 4, 5]
        db = self.make({"junk": [100, 101, 102, 103], "x input": xs,
                        'y"input': ys, "z output": [x*y for x, y in zip(xs, ys)]})
        t = db["t"]
        q = db.query(t).sum(self.col(t, "x input")*self.col(t, 'y"input'))
        baseline = q.run()
        cat = self.catalog(db, ["z output", 'y"input', "x input"], modules=("product",))
        proof = cat.probe("product", "z output", "x input", 'y"input')
        self.assertTrue(proof["usable"])
        self.assertEqual(q.run(), baseline)

    def chain_db(self):
        pairs = [(i, j) for i in range(1, 7) for j in range(1, 8)]
        xs, ys = [256*i+1 for i, j in pairs], [256*j+7 for i, j in pairs]
        zs = [x*y for x, y in zip(xs, ys)]
        return self.make({"x": xs, "y": ys, "z": zs, "w": [(z-7)//256 for z in zs]})

    def test_optimizer_composes_modules_on_query_path(self):
        db = self.chain_db()
        t = db["t"]
        x, y = self.col(t, "x"), self.col(t, "y")
        query = db.query(t).sum(x*y)
        baseline = query.run()
        cat = self.catalog(db, modules=("product", "affine"), max_candidates=1000)
        self.assertEqual(query.run(), baseline)
        trace = query.relationship_plan
        self.assertEqual(trace["status"], "rewritten")
        self.assertGreaterEqual(len(trace["steps"]), 2)
        self.assertLess(trace["after_cost"], trace["before_cost"])
        self.assertTrue(any(r["expression"] == "z = x * y" for r in cat.rules))
        self.assertTrue(any(r["expression"] == "z = 256*w + 7" for r in cat.rules))

    def test_filtered_and_grouped_queries_equal_baseline(self):
        db = self.chain_db()
        t = db["t"]
        x, y = self.col(t, "x"), self.col(t, "y")
        queries = [db.query(t).where(x > 700).sum(x*y),
                   db.query(t).group_by(y).sum(x*y),
                   db.query(t).where(x > 700).group_by(y).sum(x*y),
                   db.query(t).where(x > 999999).sum(x*y)]
        baseline = [q.run() for q in queries]
        self.catalog(db, modules=("product", "affine"), max_candidates=1000)
        self.assertEqual([q.run() for q in queries], baseline)

    def test_sql_aggregates_with_nullable_unrelated_filter_and_group(self):
        pairs = [(i, j) for i in range(1, 7) for j in range(1, 8)]
        xs, ys = [256*i+1 for i, j in pairs], [256*j+7 for i, j in pairs]
        zs = [x*y for x, y in zip(xs, ys)]
        groups = [None if i % 3 == 0 else ("A" if i % 3 == 1 else "B") for i in range(len(xs))]
        flags = [None if i % 4 == 0 else i % 2 for i in range(len(xs))]
        db = self.make({"x": xs, "y": ys, "z": zs, "w": [(z-7)//256 for z in zs],
                        "grp": groups, "flag": flags}, {"grp": "VARCHAR"})
        expected_groups = {}
        for x, y, group, flag in zip(xs, ys, groups, flags):
            if flag is None or flag == 1:
                total, count = expected_groups.get(group, (0, 0))
                expected_groups[group] = (total+x*y, count+1)
        expected = [(group, *expected_groups[group])
                    for group in sorted(expected_groups, key=lambda g: (g is not None, g or ""))]
        sql = """SELECT grp, SUM(x*y) AS total, COUNT(*) AS n FROM t
                 WHERE flag IS NULL OR flag=1
                 GROUP BY grp ORDER BY grp NULLS FIRST"""
        self.assertEqual(db.sql(sql).rows, expected)
        self.catalog(db, ["x", "y", "z", "w"], max_candidates=1000)
        for engine in ("alg", "plain", "gpu"):
            with self.subTest(engine=engine):
                self.assertEqual(db.sql(sql, engine).rows, expected)

    def test_attach_false_and_close_do_not_change_normal_queries(self):
        db = self.chain_db()
        t = db["t"]
        q = db.query(t).sum(self.col(t, "x")*self.col(t, "y"))
        baseline = q.run()
        cat = self.catalog(db, attach=False)
        self.assertFalse(hasattr(t, "_relationships"))
        self.assertEqual(q.run(), baseline)
        cat.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            cat.probe("affine", "z", "w", scale=256, bias=7)

    def test_query_trace_does_not_retain_old_rewrite_after_close(self):
        db = self.chain_db()
        t = db["t"]
        q = db.query(t).sum(self.col(t, "x")*self.col(t, "y"))
        cat = self.catalog(db, max_candidates=1000)
        baseline = q.run()
        self.assertEqual(q.relationship_plan["status"], "rewritten")
        cat.close()
        self.assertEqual(q.run(), baseline)
        self.assertNotEqual(getattr(q, "relationship_plan", {}).get("status"), "rewritten")

    def test_revision_mutation_invalidates_catalog_and_falls_back(self):
        db = self.chain_db()
        t = db["t"]
        q = db.query(t).sum(self.col(t, "x")*self.col(t, "y"))
        cat = self.catalog(db, max_candidates=1000)
        q.run()
        t.flip(0, 0)
        after = q.run()
        self.assertEqual(q.relationship_plan["status"], "stale")
        with self.assertRaisesRegex(ValueError, "stale"):
            cat.probe("affine", "z", "w", scale=256, bias=7)
        with self.assertRaisesRegex(ValueError, "stale"):
            _ = cat.rules
        with self.assertRaisesRegex(ValueError, "stale"):
            _ = cat.stats
        cat.close()
        self.assertEqual(q.run(), after)

    def test_catalog_scope_rejects_different_table(self):
        from torbles.relationships import Polynomial, Trace, _lib
        import ctypes as C
        a, b = self.chain_db(), self.chain_db()
        cat = self.catalog(a)
        source, out, trace = Polynomial(), Polynomial(), Trace()
        rc = _lib.tr_optimize(cat._handle, b["t"].at, b["t"].used_bm.ctypes.data,
                              C.byref(source), C.byref(out), C.byref(trace))
        self.assertEqual(rc, -4)

    def test_used_bitmap_change_invalidates_catalog(self):
        db = self.chain_db()
        t = db["t"]
        cat = self.catalog(db)
        t.used_bm[0] ^= __import__("numpy").uint64(1)
        q = db.query(t).sum(self.col(t, "x")*self.col(t, "y"))
        q.run()
        self.assertEqual(q.relationship_plan["status"], "stale")

    def test_partial_used_domain_cannot_authorize_physical_scan_rewrite(self):
        # The equation holds only on the first three rows. Some engine paths
        # scan all physical rows, so subset evidence cannot authorize rewriting.
        db = self.make({"x": [1, 2, 3, 4], "y": [5, 7, 9, 12]})
        t = db["t"]
        t.used_bm[0] = __import__("numpy").uint64(7)
        q = db.query(t).sum(self.col(t, "y"))
        baseline = q.run()
        cat = self.catalog(db, modules=())
        proof = cat.probe("affine", "y", "x", scale=2, bias=3)
        self.assertEqual((proof["checked"], proof["violations"]), (3, 0))
        self.assertFalse(proof["usable"])
        self.assertEqual(cat.rules, [])
        self.assertEqual(q.run(), baseline)
        self.assertEqual(q.relationship_plan["status"], "unchanged")

    def test_optimized_query_agrees_across_engine_backends(self):
        db = self.chain_db()
        t = db["t"]
        q = db.query(t).where(self.col(t, "x") > 700).group_by(self.col(t, "y")).sum(
            self.col(t, "x")*self.col(t, "y"))
        baseline = q.run("alg")
        self.catalog(db, modules=("affine", "product"), max_candidates=1000)
        for engine in ("alg", "plain", "gpu"):
            with self.subTest(engine=engine):
                self.assertEqual(q.run(engine), baseline)

    def test_randomized_polynomials_preserve_exact_answers(self):
        import random
        for seed in range(10):
            rng = random.Random(731 + seed)
            xs = [rng.randrange(-9, 10) for _ in range(30)]
            ys = [rng.randrange(-7, 8) for _ in range(30)]
            zs = [x*y for x, y in zip(xs, ys)]
            db = self.make({"x": xs, "y": ys, "z": zs,
                            "w": [5*z-11 for z in zs], "s": [x+y for x, y in zip(xs, ys)]})
            t = db["t"]
            x, y, z, w = (self.col(t, c) for c in ("x", "y", "z", "w"))
            expressions = [x*y, 3*x*y-2*z+7, z*z+w, z*x-y, w-5*z]
            queries = [db.query(t).sum(e) for e in expressions]
            baseline = [q.run() for q in queries]
            expected = [sum(xs[i]*ys[i] for i in range(30)),
                        sum(3*xs[i]*ys[i]-2*zs[i]+7 for i in range(30)),
                        sum(zs[i]*zs[i]+5*zs[i]-11 for i in range(30)),
                        sum(zs[i]*xs[i]-ys[i] for i in range(30)), -11*30]
            self.assertEqual(baseline, expected)
            self.catalog(db, max_candidates=1000)
            with self.subTest(seed=seed):
                self.assertEqual([q.run() for q in queries], expected)

    def test_float_and_wide_columns_are_rejected(self):
        db = self.make({"x": [1, 2, 3], "f": [1.2, 2.3, 3.4], "wide": [0, 1 << 40, 2 << 40]},
                        {"f": "DOUBLE"})
        for col in ("f", "wide"):
            with self.subTest(column=col), self.assertRaisesRegex(ValueError, "32-bit stored codes"):
                self.catalog(db, ["x", col])

    def test_saved_database_needs_no_source_connection(self):
        db = self.chain_db()
        with tempfile.TemporaryDirectory(prefix="torbles-relationships-") as path:
            db.save(path)
            loaded = torbles.load(path)
            self.assertIsNone(loaded.con)
            cat = self.catalog(loaded, max_candidates=1000)
            self.assertTrue(any(r["expression"] == "z = x * y" for r in cat.rules))
            t = loaded["t"]
            q = loaded.query(t).sum(self.col(t, "x")*self.col(t, "y"))
            expected = sum((256*i+1)*(256*j+7) for i in range(1, 7) for j in range(1, 8))
            self.assertEqual(q.run(), expected)
            self.assertEqual(q.relationship_plan["status"], "rewritten")

    def test_exact_int64_boundary_offsets(self):
        base = -(1 << 63)
        db = self.make({"x": [base+i for i in range(5)], "y": [base+i+7 for i in range(5)]})
        cat = self.catalog(db, modules=())
        proof = cat.probe("affine", "y", "x", scale=1, bias=7)
        self.assertTrue(proof["usable"])
        self.assertEqual(proof["violations"], 0)

    def test_out_of_range_hypothesis_parameters_are_rejected(self):
        db = self.make({"x": [1, 2], "y": [5, 7]})
        cat = self.catalog(db, modules=())
        with self.assertRaisesRegex(ValueError, "fit int64"):
            cat.probe("affine", "y", "x", scale=1 << 64)


if __name__ == "__main__":
    unittest.main()
