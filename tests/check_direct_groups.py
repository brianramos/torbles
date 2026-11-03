# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Native group states pass directly to SQL finalization for matching GROUP BY keys."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
import sqlexec


class DirectGroupChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        i = list(range(520))
        frame = pd.DataFrame({"g": [j % 130 - 65 for j in i], "h": [j % 3 for j in i],
                              "v": [j * 7 % 51 - 26 for j in i], "wide_sum": [2**62 + j % 3 for j in i],
                              "nullable_g": pd.array([None if j % 5 == 0 else j % 130 for j in i], dtype="Int32"),
                              "nullable_v": pd.array([None if j % 5 == 0 else j for j in i], dtype="Int32")})
        cls.db = torbles.from_ddl("CREATE TABLE f(g INTEGER, h INTEGER, v INTEGER, wide_sum BIGINT, "
                                  "nullable_g INTEGER, nullable_v INTEGER)",
                                  {"f": frame}, keep_plain=True)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def check(self, sql, direct):
        for engine in ("alg", "plain"):
            with self.subTest(sql=sql, engine=engine):
                result = self.db.sql(sql, engine=engine)
                with patch.object(sqlexec, "VEC_GROUPS", False):
                    reference = self.db.sql(sql, engine=engine)
                self.assertEqual(result.rows, reference.rows)
                self.assertEqual(result.rows, self.db.con.execute(sql).fetchall())
                self.assertEqual(bool(result.stats.get("direct_groups")), direct)

    def test_exact_group_keys_and_states(self):
        self.check("SELECT g, SUM(v), COUNT(*) FROM f GROUP BY g ORDER BY g", True)
        self.check("SELECT g, h, SUM(v) AS s, SUM(v * v) FROM f GROUP BY g,h "
                   "HAVING SUM(v) < 0 ORDER BY s, g, h LIMIT 20 OFFSET 3", True)
        self.check("SELECT g, COUNT(*) FROM f WHERE v > 100 GROUP BY g ORDER BY g", True)

    def test_wide_exact_sum_order_and_having(self):
        self.check("SELECT g, SUM(wide_sum) AS s FROM f GROUP BY g "
                   "HAVING SUM(wide_sum) > 18446744073709551619 ORDER BY s, g LIMIT 9", True)

    def test_aligned_moments_feed_join_preaggregation(self):
        sql = ("SELECT b.h, VAR_POP(a.v) FROM f a JOIN f b ON a.g = b.g "
               "GROUP BY b.h ORDER BY b.h")
        for engine in ("alg", "plain"):
            with self.subTest(engine=engine):
                result = self.db.sql(sql, engine=engine)
                with patch.object(sqlexec, "JOIN_PREAGG", False):
                    reference = self.db.sql(sql, engine=engine)
                self.assertEqual(result.rows, reference.rows)
                self.assertEqual(result.stats.get("join_preagg"), 1)
                self.assertGreater(result.stats.get("direct_groups", 0), 0)

    def test_fallbacks_preserve_sql_grouping(self):
        self.check("SELECT g % 7, SUM(v) FROM f GROUP BY g % 7 ORDER BY 1", False)
        self.check("SELECT nullable_g, SUM(v) FROM f GROUP BY nullable_g ORDER BY 1", False)
        self.check("SELECT g, SUM(nullable_v) FROM f GROUP BY g ORDER BY g", False)
        self.check("SELECT g, SUM(v) FROM f GROUP BY GROUPING SETS ((g), ()) ORDER BY g", False)


if __name__ == "__main__":
    unittest.main()
