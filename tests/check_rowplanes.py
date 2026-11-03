# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Bounded SQL order on split timestamp and dictionary planes."""
from pathlib import Path
import sys
import unittest

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import sqlrowplanes


ROWS = 131_101


@unittest.skipIf(sqlrowplanes._topk is None, "native bounded row selection library not built")
class RawPlaneTopKChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = duckdb.connect()
        cls.con.execute("SET threads=2")
        payload = ", ".join(f"CAST((i + {j}) % 127 AS INTEGER) AS c{j}" for j in range(36))
        cls.con.execute(
            "CREATE TABLE hits AS SELECT i AS id, "
            "make_timestamp(-30000000000 + ((i*7919)%61)*1000000 "
            "+ (i%29)*1099511627776) AS stamp, "
            "(['z','apple','é','猫',''])[1 + i%5] AS phrase, "
            "CASE WHEN i%17=0 THEN NULL ELSE CAST(i%41 AS INTEGER) END AS optional, "
            f"{payload} FROM range({ROWS}) AS rows(i)"
        )
        cls.db = torbles.from_duckdb(cls.con, tables=["hits"], groups=False, threads=2)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        cls.con.close()

    def check_query(self, statement):
        expected = self.con.execute(statement).fetchall()
        result = self.db.sql(statement)
        self.assertEqual(result.fetchall(), expected)
        return result

    def test_wide_timestamp_dictionary_and_payload_across_parts(self):
        fact = self.db["hits"]
        self.assertIsNotNone(fact.parts)
        self.assertIn("stamp", fact.wide)
        self.assertIn(fact.wide["stamp"][1], fact.sidecars)
        result = self.check_query(
            "SELECT * FROM hits WHERE c33 BETWEEN 20 AND 85 AND phrase <> '' "
            "ORDER BY stamp DESC, phrase ASC, id DESC LIMIT 21 OFFSET 17"
        )
        self.assertGreater(result.stats.get("raw_plane_topk_rows", 0), 21)
        if hasattr(torbles.algq.gpu_lib(), "gpu_topk_bitmap"):
            gpu = self.db.sql(
                "SELECT * FROM hits WHERE c33 BETWEEN 20 AND 85 AND phrase <> '' "
                "ORDER BY stamp DESC, phrase ASC, id DESC LIMIT 21 OFFSET 17", engine="gpu"
            )
            self.assertIn("cuda_raw_topk_rows", gpu.stats)
            self.assertEqual(gpu.rows, result.rows)

    def test_mixed_directions_and_repeated_keys(self):
        result = self.check_query(
            "SELECT id, phrase, stamp FROM hits ORDER BY phrase DESC, stamp ASC, "
            "stamp DESC, id ASC LIMIT 64 OFFSET 3"
        )
        self.assertEqual(result.stats.get("raw_plane_topk_rows"), ROWS)

    def test_fewer_matches_than_limit_and_empty_after_offset(self):
        for offset, size in ((1, 2), (7, 0)):
            with self.subTest(offset=offset):
                result = self.check_query(
                    f"SELECT id FROM hits WHERE id BETWEEN 2 AND 4 ORDER BY stamp, id LIMIT 16 OFFSET {offset}"
                )
                self.assertEqual(len(result.rows), size)
                self.assertIn("raw_plane_topk_rows", result.stats)

    def test_nullable_order_key_keeps_sql_null_order(self):
        result = self.check_query(
            "SELECT optional, id FROM hits ORDER BY optional ASC NULLS FIRST, id DESC LIMIT 19"
        )
        self.assertNotIn("raw_plane_topk_rows", result.stats)


if __name__ == "__main__":
    unittest.main()
