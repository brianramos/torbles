# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""SQL exact distinct states and bounded unordered native grouping."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles.sqlencodednative import _group


@unittest.skipIf(_group is None, "native encoded grouping library not built")
class EncodedGroupsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        row = np.arange(70_013, dtype=np.int64)
        optional = pd.array(row % 127, dtype="Int32")
        optional[row % 11 == 0] = None
        cls.db = torbles.from_ddl(
            "CREATE TABLE t(uid BIGINT, region INTEGER, phrase VARCHAR, score INTEGER, opt INTEGER)",
            {"t": pd.DataFrame(dict(
                uid=((row % 307) - 129) * (1 << 49) + row % 7,
                region=row % 13 - 6, phrase=np.array(["", "alpha", "beta", "delta"])[row % 4],
                score=row % 37 - 21, opt=optional,
            ))},
        )

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_scalar_sparse_and_dense_distinct(self):
        for column in ("uid", "phrase"):
            for where in ("", " WHERE score > 13"):
                query = f"SELECT COUNT(DISTINCT {column}) FROM t{where}"
                result = self.db.sql(query)
                self.assertEqual(result.rows, self.db.con.execute(query).fetchall())
                self.assertEqual(result.stats.get("encoded_group_groups"), 1)

    def test_grouped_distinct_mixed_states(self):
        select = ("SELECT region, phrase, COUNT(DISTINCT uid) AS u, COUNT(*) AS c, SUM(score), AVG(score) "
                  "FROM t WHERE phrase <> '' AND score > -15 GROUP BY region, phrase")
        oracle = {r[:2]: r for r in self.db.con.execute(select).fetchall()}
        for order, index in (("u", 2), ("c", 3)):
            result = self.db.sql(select + f" ORDER BY {order} DESC LIMIT 7 OFFSET 2")
            self.assertEqual(result.stats.get("encoded_group_groups"), 7)
            self.assertEqual([r[index] for r in result.rows],
                             sorted((r[index] for r in oracle.values()), reverse=True)[2:9])
            self.assertEqual(len({r[:2] for r in result.rows}), 7)
            for row in result.rows:
                self.assertEqual(row, oracle[row[:2]])

    def test_moved_group_key_projects_generated_dimension(self):
        row = np.arange(70_013, dtype=np.int64)
        ip = row % 47
        frame = pd.DataFrame({
            "ip": ip * 31 + 100,
            "region": ip % 7,
            "uid": (row % 83 - 41) * (1 << 48),
            **{f"attr{i}": ip * (i + 1) for i in range(32)},
        })
        db = torbles.from_frames({"t": frame})
        self.addCleanup(db.close)
        self.assertEqual(db["t"].moved["region"], "ip")
        select = "SELECT region, COUNT(DISTINCT uid) AS c FROM t GROUP BY region"
        oracle = dict(db.con.execute(select).fetchall())
        actual = db.sql(select + " ORDER BY c DESC LIMIT 4")
        self.assertEqual(actual.stats.get("encoded_group_groups"), 4)
        self.assertEqual([n for _, n in actual.rows], sorted(oracle.values(), reverse=True)[:4])
        self.assertTrue(all(oracle[k] == n for k, n in actual.rows))
        if hasattr(torbles.algq.gpu_lib(), "gpu_group_distinct_planes"):
            gpu = db.sql(select + " ORDER BY c DESC LIMIT 4", engine="gpu")
            self.assertIn("cuda_encoded_group_distinct", gpu.stats)
            self.assertEqual([n for _, n in gpu.rows], sorted(oracle.values(), reverse=True)[:4])
            self.assertTrue(all(oracle[k] == n for k, n in gpu.rows))

    def test_unordered_limit_and_nullable_fallback(self):
        select = "SELECT uid, phrase, COUNT(*) FROM t GROUP BY uid, phrase"
        result = self.db.sql(select + " LIMIT 9")
        oracle = {r[:2]: r[2] for r in self.db.con.execute(select).fetchall()}
        self.assertEqual(result.stats.get("encoded_group_groups"), 9)
        self.assertEqual(len({r[:2] for r in result.rows}), 9)
        self.assertTrue(all(oracle[r[:2]] == r[2] for r in result.rows))
        query = "SELECT COUNT(DISTINCT opt) FROM t"
        result = self.db.sql(query)
        self.assertEqual(result.rows, self.db.con.execute(query).fetchall())
        self.assertNotIn("encoded_group_groups", result.stats)


if __name__ == "__main__":
    unittest.main()

