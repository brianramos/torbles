# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Exact SQL reductions on a certified periodic table."""

import unittest
from unittest.mock import patch

import duckdb
import numpy as np

import torbles
from torbles import algq


class PeriodicSQLChecks(unittest.TestCase):
    rows = 131_077

    def setUp(self):
        self.oracle = duckdb.connect()
        self.addCleanup(self.oracle.close)
        self.oracle.execute("""
            CREATE TABLE t AS SELECT
                CASE WHEN (i % 192) % 13 = 0 THEN NULL
                    ELSE (i % 192) % 31 - 15 END::INTEGER AS x,
                CASE WHEN (i % 320) % 17 = 0 THEN NULL
                    ELSE (i % 320) % 41 - 20 END::INTEGER AS y,
                CASE WHEN (i % 448) // 64 = 3 THEN NULL
                    ELSE (i % 448) // 64 - 3 END::INTEGER AS g,
                i::INTEGER AS noise
            FROM range(?) r(i)
        """, [self.rows])
        self.db = torbles.from_duckdb(self.oracle, groups=False)
        self.addCleanup(self.db.close)
        self.periods = self.db.prepare_periodic("t")

    def exact(self, sql, periodic=True, engine="alg"):
        expected = self.oracle.execute(sql).fetchall()
        result = self.db.sql(sql, engine=engine)
        self.assertEqual(result.fetchall(), expected)
        self.assertEqual("periodic_representatives" in result.stats, periodic, result.stats)
        if periodic:
            self.assertEqual(result.stats["periodic_rows"], self.rows)
            self.assertLess(result.stats["periodic_representatives"], self.rows)
            self.assertEqual(result.stats["periodic_native_device"], "cpu" if engine == "alg" else "cuda")
        return result.stats

    def test_weighted_cross_column_group_and_tail(self):
        self.assertEqual([self.periods[k] for k in ("x", "y", "g")], [3, 5, 7])
        self.assertFalse(self.periods.get("noise", 0))
        filt = "WHERE x BETWEEN -12 AND 3 OR (x IS NULL AND y > 7)"
        self.exact(f"SELECT COUNT(*), COUNT(y), SUM(y), AVG(y), MIN(y), MAX(y) FROM t {filt}")
        self.exact(f"SELECT g, COUNT(*), COUNT(y), SUM(y), AVG(y) FROM t {filt} "
                   "GROUP BY g HAVING COUNT(*) > 100 ORDER BY g NULLS LAST")
        self.exact("SELECT g, SUM(y) FILTER (WHERE x < 0), COUNT(*) FROM t "
                   "GROUP BY g ORDER BY g NULLS LAST", periodic=False)
        self.exact("SELECT COUNT(*), SUM(y), AVG(y), MIN(y) FROM t WHERE x > 1000")

    def test_cuda_reuses_only_the_uploaded_cycle(self):
        try:
            name = algq.gpu_name().lower()
            cuda = hasattr(algq.gpu_lib(), "gpu_periodic_open") and "emulat" not in name and name != "no cuda device"
        except (OSError, RuntimeError):
            cuda = False
        if not cuda:
            self.skipTest("CUDA periodic kernels unavailable")
        sql = ("SELECT g, COUNT(*), COUNT(x), SUM(x), AVG(y), MIN(y), MAX(x) FROM t "
               "WHERE x BETWEEN -12 AND 3 OR (x IS NULL AND y > 7) "
               "GROUP BY g HAVING COUNT(*) > 100 ORDER BY g NULLS LAST")
        cold = self.exact(sql, engine="gpu")
        self.assertGreater(cold["periodic_uploaded_bytes"], 0)
        self.assertLess(cold["periodic_uploaded_bytes"], self.rows // 8)
        warm = self.exact(sql, engine="gpu")
        self.assertEqual(warm["periodic_uploaded_bytes"], 0)
        self.assertEqual(warm["periodic_gpu_plane_bytes"], cold["periodic_uploaded_bytes"])
        self.assertGreater(warm["periodic_kernel_ms"], 0)
        self.exact("SELECT x, g, COUNT(*), MIN(y), MAX(y), SUM(y) FROM t "
                   "WHERE y IS NOT NULL GROUP BY x, g ORDER BY x NULLS LAST, g NULLS LAST", engine="gpu")

    def test_unproven_column_and_nonaggregate_fall_back(self):
        self.exact("SELECT COUNT(*) FROM t WHERE noise > 90000", periodic=False)
        self.exact("SELECT x FROM t WHERE noise < 4 ORDER BY noise", periodic=False)

    def test_native_and_visibility_changes_invalidate_use(self):
        fact = self.db["t"]
        # Flipping twice restores the data but still changes native revision.
        fact.flip(0, 64)
        fact.flip(0, 64)
        sql = "SELECT COUNT(*) FROM t WHERE x IS NOT NULL"
        self.exact(sql, periodic=False)
        self.db.prepare_periodic("t")
        self.exact(sql)
        fact.used[1] = False
        fact.used_bm[0] &= np.uint64((1 << 64) - 3)
        with patch.object(fact, "_periodic", None):
            original = self.db.sql(sql).fetchall()
        fallback = self.db.sql(sql)
        self.assertEqual(fallback.fetchall(), original)
        self.assertNotIn("periodic_representatives", fallback.stats)


if __name__ == "__main__":
    unittest.main()
