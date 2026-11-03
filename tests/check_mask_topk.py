# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Exact GPU bitplane top-k, stable ties and ordinary SQL fallback."""
import unittest
from unittest.mock import patch

import duckdb
import torbles
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


if __name__ == "__main__":
    unittest.main()