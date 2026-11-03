# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""SQL correctness and dispatch for CUDA ordered windows."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import algq


class CUDAOrderedWindowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(algq.gpu_lib(), "gpu_plane_window"):
            raise unittest.SkipTest("native CUDA ordered-window library unavailable")
        # Non-aligned row count exercises the bitmap tail and a tile boundary.
        row = np.arange(65_601, dtype=np.int32)
        cls.frame = pd.DataFrame(dict(
            id=row, part=(row * 7) % 23 - 12, score=(row * 11) % 19 - 8,
            amount=(row * 17) % 113 - 65, keep=(row % 73 == 0) | (row % 127 == 0),
        ))
        cls.db = torbles.from_ddl(
            "CREATE TABLE t(id INTEGER, part INTEGER, score INTEGER, amount INTEGER, keep BOOLEAN)",
            {"t": cls.frame}, keep_plain=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def check(self, query, route, expected=None):
        if expected is None:
            expected = self.db.con.execute(query).fetchall()
        with patch.dict("os.environ", {"TORBLES_CUDA_WINDOWS": "1"}):
            actual = self.db.sql(query, engine="gpu")
        self.assertEqual(actual.rows, expected)
        if route:
            self.assertEqual(actual.stats.get(route), len(expected), actual.stats)
            self.assertEqual(actual.stats.get("cuda_ordered_window_rows"), len(expected), actual.stats)
        else:
            self.assertNotIn("cuda_ordered_window_rows", actual.stats)

    def test_partitioned_running_sum_signed_values(self):
        self.check(
            "SELECT id, SUM(amount) OVER (PARTITION BY part ORDER BY score DESC, id "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t ORDER BY id",
            "cuda_running_sum",
        )

    def test_sparse_descending_rank_preserves_input_ties(self):
        # SQL permits any tie order. CUDA's stable physical row order is a
        # stronger contract, so DuckDB's arbitrary tie order is not the oracle.
        part, score = (self.frame[name].to_numpy() for name in ("part", "score"))
        ids = np.flatnonzero(self.frame.keep.to_numpy())
        ordered = ids[np.lexsort((-score[ids], part[ids]))]
        ranks = {}
        previous = None
        for row in ordered:
            key = int(part[row])
            rank = 1 if key != previous else rank + 1
            previous = key
            ranks[int(row)] = rank
        expected = [(int(row), ranks[int(row)]) for row in ids]
        self.check(
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY part ORDER BY score DESC) "
            "FROM t WHERE keep ORDER BY id", "cuda_row_number", expected,
        )

    def test_no_partition(self):
        cases = (
            ("ROW_NUMBER() OVER (ORDER BY score DESC, id DESC)", "cuda_row_number"),
            ("SUM(amount) OVER (ORDER BY id DESC "
             "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)", "cuda_running_sum"),
        )
        for expression, route in cases:
            with self.subTest(route=route):
                self.check(f"SELECT id, {expression} FROM t ORDER BY id", route)

    def test_unsupported_frames_fall_back(self):
        for frame in ("ROWS BETWEEN 2 PRECEDING AND CURRENT ROW", "RANGE UNBOUNDED PRECEDING"):
            with self.subTest(frame=frame):
                order = "score, id" if frame.startswith("ROWS") else "score"
                self.check(
                    f"SELECT id, SUM(amount) OVER (PARTITION BY part ORDER BY {order} {frame}) "
                    "FROM t ORDER BY id", None,
                )


    def test_qualify_compacts_rank_and_signed_sum(self):
        if not hasattr(algq.gpu_lib(), "gpu_plane_window_qualify"):
            self.skipTest("CUDA QUALIFY unavailable")
        queries = (
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY part ORDER BY score DESC, id) AS rn "
            "FROM t QUALIFY 3 >= rn ORDER BY id",
            "SELECT id, SUM(amount) OVER (PARTITION BY part ORDER BY id "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total "
            "FROM t WHERE keep QUALIFY total < -200 ORDER BY id",
            "SELECT id FROM t QUALIFY ROW_NUMBER() OVER (ORDER BY id) < 0",
        )
        for query in queries:
            with self.subTest(query=query):
                expected = self.db.con.execute(query).fetchall()
                actual = self.db.sql(query, engine="gpu")
                self.assertEqual(actual.rows, expected)
                self.assertEqual(actual.stats.get("cuda_window_qualify_rows"), len(expected), actual.stats)

    def test_columnar_result_preserves_sql_values_and_row_api(self):
        query = "SELECT id, ROW_NUMBER() OVER (PARTITION BY part ORDER BY id) AS rn FROM t"
        actual = self.db.sql(query, engine="gpu")
        columns = actual.fetch_columns()
        self.assertEqual(len(columns), 2)
        self.assertTrue(all(column.dtype == np.int64 for column in columns))
        expected = sorted(self.db.con.execute(query).fetchall())
        self.assertEqual(list(zip(*(column.tolist() for column in columns))), expected)
        columns[0][0] = -1
        self.assertEqual(actual.fetchone(), expected[0])
        self.assertEqual(actual.fetchall(), expected)


    def test_window_terminal_order_and_physical_slice(self):
        if not hasattr(algq.gpu_lib(), "gpu_plane_window_terminal"):
            self.skipTest("CUDA terminal window unavailable")
        ordered = (
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM t "
            "QUALIFY rn > 10 ORDER BY rn DESC LIMIT 17 OFFSET 4"
        )
        actual = self.db.sql(ordered, engine="gpu")
        self.assertEqual(actual.fetchall(), self.db.con.execute(ordered).fetchall())
        self.assertEqual(actual.stats.get("cuda_window_terminal_rows"), 17, actual.stats)
        physical = "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM t LIMIT 7 OFFSET 9"
        actual = self.db.sql(physical, engine="gpu")
        self.assertEqual(actual.fetchall(), [(i, len(self.frame) - i) for i in range(9, 16)])
        self.assertEqual(actual.stats.get("cuda_window_terminal_rows"), 7, actual.stats)

    def test_derived_window_filter_moves_before_materialization(self):
        query = (
            "SELECT id, rn FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY part ORDER BY id) AS rn "
            "FROM t) AS ranked WHERE rn <= 2 ORDER BY id"
        )
        actual = self.db.sql(query, engine="gpu")
        self.assertEqual(actual.fetchall(), self.db.con.execute(query).fetchall())
        self.assertEqual(actual.stats.get("window_derived_qualify_pushdowns"), 1, actual.stats)
        self.assertEqual(actual.stats.get("cuda_window_qualify_rows"), 46, actual.stats)


    def test_packed_temporary_planes_reuse_signed_values(self):
        if not hasattr(algq.gpu_lib(), "gpu_plane_window_torus_terminal"):
            self.skipTest("CUDA temporary value planes unavailable")
        query = (
            "SELECT id, SUM(amount) OVER (PARTITION BY part ORDER BY id "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total FROM t "
            "WHERE keep QUALIFY -500 <= total AND total < -200 ORDER BY id"
        )
        actual = self.db.sql(query, engine="gpu")
        expected = self.db.con.execute(query).fetchall()
        self.assertEqual(actual.fetchall(), expected)
        self.assertGreater(actual.stats.get("cuda_window_torus_planes", 0), 0, actual.stats)
        transferred = actual.stats.get("cuda_window_download_bytes", 0) + actual.stats.get("cuda_window_payload_bytes", 0)
        self.assertEqual(transferred, 12 * len(expected), actual.stats)

    def test_derived_scalar_reductions_keep_window_on_device(self):
        if not hasattr(algq.gpu_lib(), "gpu_plane_window_aggregate"):
            self.skipTest("CUDA temporary window reduction unavailable")
        inner = ("SELECT SUM(amount) OVER (PARTITION BY part ORDER BY id "
                 "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total FROM t WHERE keep")
        for aggregate, predicate in (
            ("COUNT(*)", "total < -200"),
            ("SUM(total)", "total >= -1000 AND total < -200"),
            ("MIN(total)", "total < 0"),
            ("MAX(total)", "total < 0"),
            ("SUM(total)", "total > 1000000000"),
        ):
            with self.subTest(aggregate=aggregate, predicate=predicate):
                query = f"SELECT {aggregate} FROM ({inner}) AS w WHERE {predicate}"
                result = self.db.sql(query, engine="gpu")
                self.assertEqual(result.fetchall(), self.db.con.execute(query).fetchall())
                self.assertEqual(result.stats.get("cuda_window_aggregate"), 1, result.stats)


    def test_payload_gather_stays_on_device_until_slice(self):
        if not hasattr(algq.gpu_lib(), "gpu_plane_window_payload"):
            self.skipTest("CUDA window payload unavailable")
        query = ("SELECT id, score, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM t "
                 "QUALIFY rn <= 50 ORDER BY rn LIMIT 9 OFFSET 2")
        expected = self.db.con.execute(query).fetchall()
        with patch.object(self.db["t"], "_decode_column", side_effect=AssertionError("host payload decode")):
            actual = self.db.sql(query, engine="gpu")
        self.assertEqual(actual.fetchall(), expected)
        self.assertEqual(actual.stats.get("cuda_window_payload_columns"), 2, actual.stats)
        self.assertEqual(actual.stats.get("cuda_window_payload_bytes"), 72, actual.stats)
        self.assertEqual(actual.stats.get("cuda_window_download_bytes"), 72, actual.stats)

    def test_resident_group_and_shared_relations(self):
        if not hasattr(algq.gpu_lib(), "gpu_plane_window_relation"):
            self.skipTest("CUDA resident relation unavailable")
        shared = "WITH ranked AS (SELECT ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t) "
        queries = (
            ("SELECT part, COUNT(*), SUM(rn) FROM (SELECT part, ROW_NUMBER() OVER "
             "(PARTITION BY part ORDER BY id) AS rn FROM t) w GROUP BY part ORDER BY part", "group"),
            (shared + "SELECT COUNT(*) FROM ranked a JOIN ranked b ON a.rn=b.rn "
             "WHERE a.rn<=64 AND b.rn<=128", "join_count"),
            (shared + "SELECT rn FROM ranked WHERE rn<=128 INTERSECT SELECT rn FROM ranked WHERE rn<=64",
             "intersection"),
            (shared + "SELECT rn FROM ranked WHERE rn<=128 EXCEPT SELECT rn FROM ranked WHERE rn<=64",
             "difference"),
            (shared + "SELECT rn FROM ranked WHERE rn<=64 UNION SELECT rn FROM ranked WHERE rn>=65590",
             "union"),
            (shared + "SELECT COUNT(*) FROM ranked WHERE rn<=64 UNION ALL "
             "SELECT COUNT(*) FROM ranked WHERE rn<=128", "counts"),
        )
        for query, kind in queries:
            with self.subTest(kind=kind):
                expected = self.db.con.execute(query).fetchall()
                actual = self.db.sql(query, engine="gpu")
                self.assertEqual(sorted(actual.fetchall()), sorted(expected))
                self.assertEqual(actual.stats.get("cuda_window_resident_builds"), 1, actual.stats)
                self.assertIn("cuda_window_resident_" + kind, actual.stats)


if __name__ == "__main__":
    unittest.main()
