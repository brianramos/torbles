# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Weighted text reductions from native code histograms."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import sqltexthist


@unittest.skipIf(sqltexthist._code_histogram is None, "native code histogram unavailable")
class TextHistogramChecks(unittest.TestCase):
    @staticmethod
    def dataset(generated):
        row = np.arange(70_019)
        token = row % 47
        refs = np.array([""] + [
            f"https://www.{'café' if i % 3 else 'other'}.test/{i}/é" for i in range(1, 47)
        ], dtype=object)
        fields = dict(ref=refs[token], g=(row // 47) % 5)
        if generated:
            fields.update({f"attr{i}": token * (i + 1) for i in range(32)})
        return torbles.from_frames({"t": pd.DataFrame(fields)})

    def test_weighted_regex_average_count_min_and_having(self):
        sql = (r"SELECT REGEXP_REPLACE(ref, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, "
               "COUNT(*) AS n, AVG(STRLEN(ref)) AS l, MIN(ref) FROM t WHERE ref <> '' AND g > 0 "
               "GROUP BY k HAVING COUNT(*) > 100 ORDER BY l DESC, k")
        for generated in (False, True):
            with self.subTest(generated=generated):
                db = self.dataset(generated)
                try:
                    self.assertEqual(db["t"].codecs["ref"].kind, "ref" if generated else "dict")
                    result = db.sql(sql)
                    self.assertEqual(result.rows, db.con.execute(sql).fetchall())
                    self.assertEqual(result.stats.get("dictionary_histogram_codes"), 46)
                    self.assertGreater(result.stats["dictionary_projection_rows"], 40_000)
                    with patch.object(sqltexthist, "project_group", return_value=None):
                        fallback = db.sql(sql)
                    self.assertEqual(result.rows, fallback.rows)
                    self.assertNotIn("dictionary_histogram_codes", fallback.stats)
                finally:
                    db.close()

    def test_histogram_honors_sparse_fact_used_bitmap(self):
        codes = np.array([4, 3, 4, 4, 3, 2, 0], np.uint32)
        used = np.array([True, False, True, False, True, True, False])
        table = torbles.Table("sparse", {"v": codes}, used=used)
        self.addCleanup(table.close)
        counts = sqltexthist.code_counts(table, "v", table.used_bm)
        self.assertEqual(counts.tolist(), [0, 0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()

