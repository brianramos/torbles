# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Lexicographic rank boundaries and partitioned ROW_NUMBER against DuckDB."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import algq
import sqlwin


class EncodedOrderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 145_537
        i = np.arange(n, dtype=np.int64)
        a = pd.array(np.where(i % 13 < 10, 7, 2), dtype="Int32")
        b = pd.array((i * 37) % 10_000 - 5_000, dtype="Int32")
        part = pd.array(i % 23 - 10, dtype="Int32")
        a[i % 109 == 0] = None
        b[i % 17 == 0] = None
        part[i % 29 == 0] = None
        frame = pd.DataFrame(dict(id=i, a=a, b=b, p=part, keep=(i % 19 != 0), v=(i % 2) * (1 << 42) + i))
        cls.db = torbles.from_ddl("CREATE TABLE t(id BIGINT PRIMARY KEY, a INTEGER, b INTEGER, p INTEGER, "
                                  "keep BOOLEAN, v BIGINT)", {"t": frame})

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_packed_sort_signed_spans(self):
        rng = np.random.default_rng(47)
        n = 1 << 15
        a = rng.integers(-128, 128, size=n, dtype=np.int8)
        b = rng.integers(-2048, 2048, size=n, dtype=np.int16)
        for keys in ([a], [a, b], [a, b % 7]):
            self.assertTrue(np.array_equal(sqlwin.lexsort(keys, n), np.lexsort(keys)))

    def test_secondary_rank_reuses_histogram_counts(self):
        common = "SELECT a, b FROM t WHERE keep "
        queries = (
            (common + "ORDER BY a DESC NULLS LAST, b ASC NULLS LAST LIMIT 17 OFFSET 5", None),
            (common + "ORDER BY a ASC NULLS LAST, b DESC NULLS LAST FETCH FIRST 17 ROWS WITH TIES",
             common + "QUALIFY rank() OVER (ORDER BY a ASC NULLS LAST, b DESC NULLS LAST) <= 17 "
             "ORDER BY a ASC NULLS LAST, b DESC NULLS LAST"),
        )
        for sql, oracle in queries:
            want = self.db.con.execute(oracle or sql).fetchall()
            for engine in ("alg", "gpu"):
                with self.subTest(sql=sql, engine=engine):
                    result = self.db.sql(sql, engine)
                    self.assertEqual(result.rows, want)
                    # One rank search per ORDER BY key, plus the final bitmap.
                    self.assertEqual(result.stats["engine_queries"], 3, result.stats)

    def test_partitioned_row_number_with_dense_filtered_output(self):
        for tie_breaker in ("id", "id DESC"):
            sql = (f"SELECT id, v, row_number() OVER (PARTITION BY p, a ORDER BY b DESC NULLS FIRST, {tie_breaker}) "
                   "FROM t WHERE keep ORDER BY id")
            want = self.db.con.execute(sql).fetchall()
            for engine in ("alg", "gpu"):
                with self.subTest(engine=engine, tie_breaker=tie_breaker), \
                        patch.object(sqlwin, "layout", side_effect=AssertionError("peer layout")):
                    result = self.db.sql(sql, engine)
                    self.assertEqual(result.rows, want)
                    if algq._order_codes is not None:
                        self.assertEqual(result.stats.get("native_row_number_rows"), len(want))

    @unittest.skipIf(algq._order_codes is None, "native order library not built")
    def test_stable_native_order_and_partition_starts_across_chunks(self):
        n = 131_117
        i = np.arange(n, dtype=np.uint32)
        codes = ((i & 1) << 11) | ((i * 37) % 1024)
        perm = np.argsort(codes, kind="stable")
        split = (n + 1) // 2
        expected = np.empty(n, dtype=np.int64)
        expected[perm[:split]] = np.arange(1, split + 1)
        expected[perm[split:]] = np.arange(1, n - split + 1)
        original = algq.lib.at_get_threads()
        try:
            for threads in (1, 4):
                torbles.set_threads(threads)
                with self.subTest(threads=threads):
                    np.testing.assert_array_equal(sqlwin._native_packed_order(codes, 12), perm)
                    np.testing.assert_array_equal(sqlwin._native_packed_order(codes, 12, 11), expected)
        finally:
            torbles.set_threads(original)


if __name__ == "__main__":
    unittest.main()
