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


if __name__ == "__main__":
    unittest.main()
