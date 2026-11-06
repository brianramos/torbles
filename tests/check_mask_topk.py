# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Exact GPU bitplane top-k, stable ties and ordinary SQL fallback."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import duckdb
import torbles
from torbles import algq
from algq import Query, gpu_lib


class MaskTopKChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = duckdb.connect()
        cls.con.execute("""CREATE TABLE t AS SELECT i::INTEGER id, ((i * 17) % 7 - 3)::INTEGER v,
                          (i % 19 <> 0) keep, CASE WHEN i % 2 = 0 THEN i::INTEGER END nullable
                          FROM range(131075) r(i)""")
        cls.db = torbles.from_duckdb(cls.con)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        cls.con.close()

    def test_filtered_lexicographic_limits(self):
        for where in ("", "WHERE keep", "WHERE id < 7", "WHERE id > 200000"):
            for order in ("v DESC, id", "v, id DESC", "id DESC"):
                sql = f"SELECT id, v FROM t {where} ORDER BY {order} LIMIT 19 OFFSET 3"
                with self.subTest(where=where, order=order):
                    result = self.db.sql(sql, "gpu")
                    self.assertEqual(result.rows, self.con.execute(sql).fetchall())
                    self.assertEqual(result.stats.get("gpu_mask_topk"), 1)
                    self.assertEqual(result.stats["engine_queries"], 1)

    def test_stable_boundary_ties(self):
        table = self.db["t"]
        ids = Query(table)._topk_rows([("v", True)], 25)
        want = self.con.execute("SELECT id FROM t WHERE v = 3 ORDER BY id LIMIT 25").fetchall()
        self.assertEqual(ids.tolist(), [row[0] for row in want])

    def test_fallback(self):
        for sql in (
            "SELECT id FROM t ORDER BY nullable DESC NULLS LAST, id LIMIT 15",
            "SELECT v FROM t ORDER BY v DESC FETCH FIRST 1 ROW WITH TIES",
            "SELECT id FROM t ORDER BY v, id LIMIT 1025",
        ):
            with self.subTest(sql=sql):
                result = self.db.sql(sql, "gpu")
                self.assertEqual(result.rows, self.db.sql(sql, "alg").rows)
                self.assertNotIn("gpu_mask_topk", result.stats)
        sql = "SELECT id FROM t WHERE keep ORDER BY v DESC, id LIMIT 7"
        with patch.object(gpu_lib(), "gpu_topk", return_value=-1):
            result = self.db.sql(sql, "gpu")
        self.assertEqual(result.rows, self.con.execute(sql).fetchall())
        self.assertNotIn("gpu_mask_topk", result.stats)


class SparsePlaneReads(unittest.TestCase):
    rows = 131_075
    matches = 10

    @classmethod
    def setUpClass(cls):
        cls.con = duckdb.connect()
        cls.addClassCleanup(cls.con.close)
        cls.con.execute("""
            CREATE TABLE sparse AS
            SELECT i::INTEGER AS id, (i % 3 - 1)::INTEGER AS v,
                   (i IN (0, 1, 7, 63, 64, 127, 128, 1024, 131073, 131074))::INTEGER AS marker,
                   CASE WHEN i % 2 = 0 THEN 17179869184 + i % 5
                        ELSE -8589934592 + i % 5 END::BIGINT AS wide,
                   CASE WHEN i % 3 <> 0 THEN (i % 5 - 2)::INTEGER END AS nullable
            FROM range(?) r(i)
        """, [cls.rows])
        cls.db = torbles.from_duckdb(cls.con, groups=False)
        cls.addClassCleanup(cls.db.close)

    def setUp(self):
        self.enterContext(patch.object(algq, "_SPARSE_GATHER", True))

    def test_unordered_sparse_bitmap_and_residual(self):
        fact = self.db["sparse"]
        self.assertIsNone(fact.pk)
        self.assertIn("wide", fact.wide)
        self.assertIn("nullable", fact.nullable)
        for suffix in ("", " AND length(CAST(id + v AS VARCHAR)) > 2"):
            sql = "SELECT wide, nullable, v FROM sparse WHERE marker = 1" + suffix
            with self.subTest(residual=bool(suffix)), patch.object(
                algq.Table, "_decode_column", side_effect=AssertionError("decoded a full payload column")
            ):
                result = self.db.sql(sql)
                expected = self.con.execute(sql).fetchall()
                self.assertCountEqual(result.rows, expected)
                self.assertEqual(result.stats.get("raw_sparse_bitmap_rows"), self.matches, result.stats)
                self.assertEqual(result.stats.get("raw_sparse_gather_rows"), self.matches, result.stats)
                if not suffix:
                    self.assertLess(len(set(expected)), self.matches)
                    self.assertTrue(any(row[1] is None for row in expected))

    def test_sparse_ordered_limit(self):
        sql = ("SELECT id, wide, nullable FROM sparse WHERE marker = 1 "
               "ORDER BY v DESC, id LIMIT 4 OFFSET 2")
        result = self.db.sql(sql)
        self.assertEqual(result.rows, self.con.execute(sql).fetchall())
        self.assertEqual(result.stats.get("raw_sparse_topk_matches"), self.matches, result.stats)
        self.assertEqual(result.stats.get("raw_sparse_gather_rows"), 4, result.stats)

    def test_dense_fallback_and_disabled_route(self):
        sql = "SELECT v FROM sparse WHERE marker = 0"
        result = self.db.sql(sql)
        self.assertCountEqual(result.rows, self.con.execute(sql).fetchall())
        self.assertEqual(len(result.rows), self.rows - self.matches)
        self.assertNotIn("raw_sparse_bitmap_rows", result.stats)
        self.assertNotIn("raw_sparse_gather_rows", result.stats)

        sql = "SELECT wide, nullable FROM sparse WHERE marker = 1"
        with patch.object(algq, "_SPARSE_GATHER", False):
            result = self.db.sql(sql)
        self.assertCountEqual(result.rows, self.con.execute(sql).fetchall())
        self.assertNotIn("raw_sparse_bitmap_rows", result.stats)
        self.assertNotIn("raw_sparse_gather_rows", result.stats)

    def test_packed_sparse_projection_stays_compact(self):
        if algq._at_save_cycles is None or algq._at_has_dense is None:
            self.skipTest("native topology storage unavailable")
        sql = "SELECT wide, nullable FROM sparse WHERE marker = 1"
        expected = self.con.execute(sql).fetchall()
        with TemporaryDirectory() as directory, patch.object(algq, "_COMPRESSED_MASKS", True):
            snapshot = Path(directory) / "encoded"
            self.db.save(snapshot, compression="topology")
            reopened = torbles.load(snapshot)
            try:
                fact = reopened["sparse"]
                self.assertTrue(fact._storage().encoded)
                self.assertFalse(algq._at_has_dense(fact._raw_at))
                result = reopened.sql(sql)
                self.assertCountEqual(result.rows, expected)
                self.assertEqual(result.stats.get("raw_sparse_bitmap_rows"), self.matches, result.stats)
                self.assertEqual(result.stats.get("raw_sparse_gather_rows"), self.matches, result.stats)
                self.assertGreater(result.stats.get("raw_sparse_encoded_columns", 0), 0, result.stats)
                self.assertFalse(algq._at_has_dense(fact._raw_at))
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
