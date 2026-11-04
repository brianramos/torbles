# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Exact learned SQL moments, set composition and source identity."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import sqlalgebra


FULL = "SELECT COUNT(*), SUM(x), SUM(y) FROM t"
LOW = FULL + " WHERE k < 0"
HIGH = FULL + " WHERE k > 0"
DERIVED = "SELECT COUNT(*) AS n, SUM(3*x - 2*y) AS s, AVG(5*y - 2*x - 1) AS a FROM t"


class SQLAlgebraChecks(unittest.TestCase):
    def database(self, shift=0, *, overflow=False):
        x = [4, -3, 8, 12, 1, 0] if not overflow else [4, -3, 8, 12, 1, 2**31 - 1]
        frame = pd.DataFrame({
            "k": [-4, -1, 0, 2, 2, 5],
            "x": [v + shift for v in x],
            "y": [10 + shift, 1, -7, 2, 15, -1],
            "maybe": pd.array([None, 3, 6, None, 9, None], dtype="Int32"),
        })
        db = torbles.from_ddl("CREATE TABLE t(k INTEGER, x INTEGER, y INTEGER, maybe INTEGER)", {"t": frame})
        self.addCleanup(db.close)
        return db

    def check_derivation(self, catalog, db, sql, reason):
        expected = db.sql(sql)
        actual = catalog.query(db, sql)
        self.assertEqual(actual.rows, expected.rows)
        self.assertEqual(actual.columns, expected.columns)
        self.assertEqual(actual.types, expected.types)
        self.assertEqual(actual.stats.get("algebra_derivation"), reason)
        self.assertEqual(actual.stats.get("engine_queries"), 0)
        return actual.rows

    def test_exact_new_affine_and_scope_difference(self):
        db = self.database()
        catalog = sqlalgebra.Catalog().learn(db, FULL).learn(db, LOW)
        self.check_derivation(catalog, db, DERIVED, "basis")
        self.check_derivation(catalog, db, DERIVED + " WHERE k >= 0", "scope_difference")
        empty = self.check_derivation(catalog, db, DERIVED + " WHERE k < 0 AND k >= 0", "empty_scope")
        self.assertEqual(empty, [(0, None, None)])

    def test_disjoint_scope_union(self):
        db = self.database()
        catalog = sqlalgebra.Catalog().learn(db, LOW).learn(db, HIGH)
        self.check_derivation(catalog, db, DERIVED + " WHERE k < 0 OR k > 0", "scope_union")

    def test_overlapping_scopes_span_full_table(self):
        db = self.database()
        catalog = sqlalgebra.Catalog()
        catalog.learn(db, FULL + " WHERE k <= 2")
        catalog.learn(db, FULL + " WHERE k >= 0")
        catalog.learn(db, FULL + " WHERE k >= 0 AND k <= 2")
        with patch.object(sqlalgebra, "_read_state", side_effect=AssertionError("span reopened base rows")):
            self.check_derivation(catalog, db, DERIVED, "scope_span")

    def test_saved_database_reopens_with_learned_basis(self):
        db = self.database()
        catalog = sqlalgebra.Catalog().learn(db, FULL).learn(db, LOW)
        with TemporaryDirectory() as tmp:
            path, record = Path(tmp) / "source", Path(tmp) / "algebra.json"
            db.save(path)
            catalog.save(record)
            db.close()
            loaded = torbles.load(path)
            try:
                restored = sqlalgebra.Catalog.load(record)
                with patch.object(sqlalgebra, "_read_state", side_effect=AssertionError("relearned saved basis")):
                    self.check_derivation(restored, loaded, DERIVED, "basis")
                    self.check_derivation(restored, loaded, DERIVED + " WHERE k >= 0", "scope_difference")
            finally:
                loaded.close()

    def test_replaced_table_refreshes_automatically_and_explicitly(self):
        db = self.database()
        catalog = sqlalgebra.Catalog().learn(db, FULL).learn(db, LOW)
        original = self.check_derivation(catalog, db, DERIVED, "basis")
        replacement = self.database(10)
        db.tables["t"].close()
        db.tables["t"] = replacement.tables.pop("t")
        with patch.object(sqlalgebra, "_read_state", wraps=sqlalgebra._read_state) as read:
            updated = self.check_derivation(catalog, db, DERIVED, "basis")
            self.assertGreaterEqual(read.call_count, 1)
        self.assertNotEqual(updated, original)

        replacement = self.database(50)
        db.tables["t"].close()
        db.tables["t"] = replacement.tables.pop("t")
        self.assertIs(catalog.refresh(db), catalog)
        with patch.object(sqlalgebra, "_read_state", side_effect=AssertionError("refreshed twice")):
            self.check_derivation(catalog, db, DERIVED + " WHERE k >= 0", "scope_difference")

    def test_nullable_and_overflow_paths_keep_sql_semantics(self):
        db = self.database(overflow=True)
        catalog = sqlalgebra.Catalog().learn(db, FULL)
        query = "SELECT COUNT(*), SUM(maybe), AVG(maybe) FROM t"
        got = catalog.query(db, query)
        self.assertEqual(got.rows, [(6, 18, 6.0)])
        self.assertNotIn("algebra_derivation", got.stats)

        query = "SELECT SUM(x * 2) FROM t"
        with self.assertRaises(torbles.SqlError) as normal:
            db.sql(query)
        with self.assertRaises(torbles.SqlError) as cached:
            catalog.query(db, query)
        self.assertEqual(str(cached.exception), str(normal.exception))

    def test_corrupt_catalog_rejected(self):
        db = self.database()
        catalog = sqlalgebra.Catalog().learn(db, FULL)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "algebra.json"
            catalog.save(path)
            record = json.loads(path.read_bytes())
            record["catalog"]["states"][0][3] += 1
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid algebra catalog"):
                sqlalgebra.Catalog.load(path)

    def test_writable_update_refresh_and_scalar_finalizers(self):
        import numpy as np

        db = torbles.Database()
        self.addCleanup(db.close)
        db.add_writable_table("w", {"id": np.arange(6), "x": np.arange(6) + 10})
        catalog = sqlalgebra.Catalog().learn(db, "SELECT COUNT(*), SUM(x) FROM w")
        sql = "SELECT SUM(CAST(x AS BIGINT) + 3), AVG(x) FROM w"
        self.assertEqual(catalog.query(db, sql).rows, [(93, 12.5)])
        self.assertIn("algebra_derivation", catalog.query(db, sql).stats)
        db.sql("UPDATE w SET x = 19 WHERE id = 0")
        stale = catalog.query(db, sql, refresh=False)
        self.assertEqual(stale.rows, [(102, 14.0)])
        self.assertNotIn("algebra_derivation", stale.stats)
        refreshed = catalog.query(db, sql)
        self.assertEqual(refreshed.rows, stale.rows)
        self.assertIn("algebra_derivation", refreshed.stats)


if __name__ == "__main__":
    unittest.main()

