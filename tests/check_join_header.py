# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Verified hierarchy paths from a snapshot header."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import duckdb
import torbles
from torbles import algq, joinheader


class JoinHeaderChecks(unittest.TestCase):
    def test_ancestor_filter_uses_saved_cells_and_preserves_orphans(self):
        source = duckdb.connect()
        self.addCleanup(source.close)
        source.execute("""
            CREATE TABLE region(id INTEGER PRIMARY KEY, label VARCHAR, tag INTEGER);
            INSERT INTO region VALUES (0, 'North', NULL), (1, 'South', 3),
                                      (2, 'East', 9), (3, 'West', 5);
            CREATE TABLE city(id INTEGER PRIMARY KEY, region_id INTEGER REFERENCES region(id), area INTEGER);
            INSERT INTO city SELECT i, i % 4, i % 4 FROM range(2048) t(i);
            CREATE TABLE facts AS
                SELECT CASE i % 2048 WHEN 0 THEN 9000 WHEN 1 THEN NULL ELSE i % 2048 END::INTEGER city_id,
                       CASE WHEN i % 16 = 0 THEN NULL ELSE i % 16 - 7 END::INTEGER val
                FROM range(131083) t(i);
        """)
        original = torbles.from_duckdb(source, foreign_keys={"facts.city_id": "city.id"}, groups=False)
        self.addCleanup(original.close)
        joins = " FROM facts f JOIN city c ON f.city_id=c.id JOIN region r ON c.region_id=r.id "
        sql = ("SELECT COUNT(*), COUNT(f.val), SUM(f.val)" + joins +
               "WHERE r.label <> 'East' AND (r.tag IS NULL OR r.tag >= 5)")
        expected = source.execute(sql).fetchall()
        grouped = "SELECT r.label, COUNT(*)" + joins + "GROUP BY r.label ORDER BY r.label"
        grouped_expected = source.execute(grouped).fetchall()

        with TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            original.save(snapshot, compression="topology")
            manifest_path = snapshot / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["join_hierarchy"]["version"], 1)
            self.assertIn(["city_id", "region_id"],
                          [entry["path"] for entry in manifest["join_hierarchy"]["records"]])
            with patch.dict("os.environ", {"TORBLES_JOIN_HEADER": "1"}):
                reopened = torbles.load(snapshot)
                try:
                    result = reopened.sql(sql)
                    self.assertEqual(result.fetchall(), expected)
                    self.assertEqual(result.stats["join_header_paths"], 2)
                    self.assertTrue(result.stats["periodic_encoded"])
                    self.assertEqual(reopened["city"]._decoded, {})
                    self.assertEqual(reopened["region"]._decoded, {})
                    self.assertFalse(algq._at_has_dense(reopened["city"]._raw_at))
                    result = reopened.sql(grouped)
                    self.assertEqual(result.fetchall(), grouped_expected)
                    self.assertNotIn("join_header_paths", result.stats)
                    reopened["city"].used[0] = False
                    self.assertIsNone(joinheader.record(reopened, "facts", ("city_id", "region_id")))
                finally:
                    reopened.close()

            manifest["join_hierarchy"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            reopened = torbles.load(snapshot)
            try:
                result = reopened.sql(sql)
                self.assertEqual(result.fetchall(), expected)
                self.assertNotIn("join_header_paths", result.stats)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
