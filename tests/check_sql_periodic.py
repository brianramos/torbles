# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Exact SQL reductions on a certified periodic table."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import duckdb
import numpy as np

import torbles
from torbles import algq
from torbles import sqlperiodic


def has_cuda():
    try:
        name = algq.gpu_name().lower()
        return hasattr(algq.gpu_lib(), "gpu_periodic_open_sectors") and "emulat" not in name and name != "no cuda device"
    except (OSError, RuntimeError):
        return False


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
                ((i % 704) // 64)::INTEGER AS h,
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
        if not has_cuda():
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
        self.check_composed_cycle(engine="gpu")

    def check_composed_cycle(self, engine):
        stats = self.exact("SELECT h, COUNT(*), SUM(y) FROM t WHERE x < 2 GROUP BY h ORDER BY h", engine=engine)
        self.assertEqual(stats["periodic_representatives"], 3 * 5 * 11 * 64)

    def test_cpu_composes_independent_column_periods(self):
        self.check_composed_cycle(engine="alg")

    def test_unproven_column_and_nonaggregate_fall_back(self):
        self.exact("SELECT COUNT(*) FROM t WHERE noise > 90000", periodic=False)
        self.exact("SELECT x FROM t WHERE noise < 4 ORDER BY noise", periodic=False)

    def test_native_and_visibility_changes_invalidate_use(self):
        fact = self.db["t"]
        # Flipping twice restores the data but still changes native revision.
        fact.flip(0, 64)
        fact.flip(0, 64)
        sql = "SELECT COUNT(*) FROM t WHERE x IS NOT NULL"
        with patch.object(sqlperiodic, "_discover", None):
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


class AutomaticSectorChecks(unittest.TestCase):
    def test_ingestion_combines_independent_vertical_edges(self):
        if algq._at_save_cycles is None:
            self.skipTest("native topology writer unavailable")
        oracle = duckdb.connect()
        self.addCleanup(oracle.close)
        oracle.execute("""
            CREATE TABLE edges AS
            WITH phases AS (
                SELECT i, (i // 64 % 11 = 0)::INTEGER AS a, (i // 64 % 13 = 0)::INTEGER AS b
                FROM range(524325) t(i)
            )
            SELECT a + 2 * xor(a, b) AS x,
                   CASE WHEN i // 262144 % 2 = 0 THEN b ELSE 1 - b END AS shifted
            FROM phases
        """)
        db = torbles.from_duckdb(oracle, groups=False)
        self.addCleanup(db.close)
        sql = "SELECT COUNT(*), SUM(x), MIN(x), MAX(x) FROM edges"
        expected = oracle.execute(sql).fetchall()
        before = db.sql(sql)
        self.assertEqual(before.fetchall(), expected)
        self.assertNotIn("periodic_native_device", before.stats)
        with TemporaryDirectory() as directory:
            snapshot = Path(directory) / "edges"
            db.save(snapshot, compression="topology")
            reopened = torbles.load(snapshot)
            try:
                after = reopened.sql(sql)
                self.assertEqual(after.fetchall(), expected)
                self.assertEqual(after.stats["periodic_native_device"], "cpu")
                self.assertEqual(after.stats["periodic_scope"], "table")
                self.assertEqual(after.stats["periodic_representatives"], 11 * 13 * 64)
                self.assertGreater(after.stats["periodic_ingested_proof_uses"], 0)
                local_sql = "SELECT COUNT(*), SUM(shifted) FROM edges"
                local = reopened.sql(local_sql)
                self.assertEqual(local.fetchall(), oracle.execute(local_sql).fetchall())
                self.assertEqual(local.stats["periodic_scope"], "sectors")
                self.assertEqual(local.stats["periodic_certified_sectors"], 2)
            finally:
                reopened.close()

    def test_automatic_local_cycles_across_physical_parts(self):
        if sqlperiodic._discover is None:
            self.skipTest("native sector planner unavailable")
        rows_per_sector = sqlperiodic._SECTOR_WORDS * 64
        rows = rows_per_sector * 4 + 32_777
        filler = ", ".join(f"0::INTEGER AS filler_{i}" for i in range(32))
        oracle = duckdb.connect()
        self.addCleanup(oracle.close)
        oracle.execute(f"""
            CREATE TABLE sectors AS
            WITH position AS (
                SELECT i // {rows_per_sector} AS s, i % {rows_per_sector} AS lane
                FROM range({rows}) t(i)
            ), phase AS (
                SELECT *, lane % (64 * (s + 3)) AS p FROM position
            )
            SELECT CASE WHEN s = 1 THEN lane % 104729
                        WHEN p % 13 = 0 THEN NULL ELSE p % 97 - 48 + s END::INTEGER AS x,
                   CASE WHEN p // 64 = 2 THEN NULL ELSE p // 64 - 2 END::INTEGER AS g,
                   {filler},
                   CASE WHEN p % 17 = 0 THEN NULL ELSE p % 41 - 20 - s END::INTEGER AS y
            FROM phase
        """)
        db = torbles.from_duckdb(oracle, groups=False)
        self.addCleanup(db.close)
        table = db["sectors"]
        self.assertIsNot(table._part_of("x"), table._part_of("y"))
        template = ("SELECT g, COUNT(*), COUNT(y), SUM(y), AVG(y), MIN(y), MAX(y) FROM sectors "
                    "WHERE x BETWEEN -12 AND {} OR (x IS NULL AND y < 4) "
                    "GROUP BY g ORDER BY g NULLS LAST")

        def check(bound, engine):
            sql = template.format(bound)
            result = db.sql(sql, engine=engine)
            self.assertEqual(result.fetchall(), oracle.execute(sql).fetchall())
            stats = result.stats
            self.assertEqual(stats.get("periodic_scope"), "sectors", stats)
            self.assertEqual(stats["periodic_certified_sectors"], 4)
            self.assertEqual(stats["periodic_ordinary_sectors"], 1)
            self.assertLess(stats["periodic_representatives"], rows // 4)
            self.assertEqual(stats["periodic_native_device"], "cpu" if engine == "alg" else "cuda")
            return stats

        self.assertEqual(check(20, "alg")["periodic_probes"], 2)
        self.assertEqual(check(21, "alg")["periodic_proof_cache_hits"], 1)
        if has_cuda():
            cold = check(22, "gpu")
            self.assertEqual(cold["periodic_proof_cache_hits"], 1)
            self.assertGreater(cold["periodic_uploaded_bytes"], 0)
            self.assertEqual(check(23, "gpu")["periodic_uploaded_bytes"], 0)
        part = table._part_of("y")
        part.flip(0, 64)
        part.flip(0, 64)
        self.assertEqual(check(24, "alg")["periodic_probes"], 2)

        if algq._at_save_cycles is None:
            return
        with TemporaryDirectory() as directory:
            snapshot = Path(directory) / "ingested"
            db.save(snapshot, compression="topology")
            with self.assertRaises(FileExistsError):
                db.save(snapshot, compression="topology")

            for engine in (("alg", "gpu") if has_cuda() else ("alg",)):
                db = torbles.load(snapshot)
                try:
                    self.assertEqual(db.integrity["sectors"]["rows_with_faults"], 0)
                    cold = check(25, engine)
                    self.assertGreater(cold["periodic_ingested_proof_uses"], 0)
                    self.assertNotIn("periodic_probes", cold)
                    warm = check(26, engine)
                    self.assertEqual(warm["periodic_proof_cache_hits"], 1)
                    if engine == "gpu":
                        self.assertEqual(warm["periodic_uploaded_bytes"], 0)
                    else:
                        changed_part = db["sectors"]._part_of("y")
                        changed_part.flip(0, 64)
                        changed_part.flip(0, 64)
                        stale = check(27, engine)
                        self.assertLess(stale["periodic_ingested_proof_uses"],
                                        cold["periodic_ingested_proof_uses"])
                finally:
                    db.close()

            plane_file = snapshot / "sectors.part0.tbl"
            original = plane_file.read_bytes()
            self.assertEqual(int.from_bytes(original[8:12], "little"), 2)
            for invalid in (original[:-1], original[:-1] + bytes([original[-1] ^ 1])):
                plane_file.write_bytes(invalid)
                with self.assertRaises(torbles.LoadError):
                    torbles.load(snapshot, validate=False)
            plane_file.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
