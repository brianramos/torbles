# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Dictionary LIKE and grouped text projections over encoded bitplanes."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import sqltexthist, sqltextplanes


class TextPlaneChecks(unittest.TestCase):
    @staticmethod
    def database(n=70_013):
        ids = np.arange(n)
        urls = np.array(["", "https://www.google.com/a", "https://www.café.com/é",
                         "https://www.café.com/zz", "line\nGoogle", "escape_%Google"], dtype=object)
        titles = np.array([None, "Google", "google", "not Google", "other"], dtype=object)
        data = pd.DataFrame(dict(id=ids, g=ids % 7, url=urls[ids % len(urls)],
                                 title=titles[ids % len(titles)]))
        return torbles.from_ddl(
            "CREATE TABLE t(id INTEGER, g INTEGER, url VARCHAR, title VARCHAR)", {"t": data}
        )

    def check_query(self, db, sql, *, predicates=0, projection=False, duck=True):
        result = db.sql(sql)
        with patch.object(sqltextplanes, "dictionary_predicate", return_value=None), \
                patch.object(sqltextplanes, "project_group", return_value=None):
            self.assertEqual(result.rows, db.sql(sql).rows)
        if duck:
            self.assertEqual(result.rows, db.con.execute(sql).fetchall())
        self.assertEqual(result.stats.get("dictionary_text_predicates", 0), predicates)
        self.assertEqual("dictionary_plane_groups" in result.stats
                         or bool(result.stats.get("dictionary_projection_rows")), projection)
        return result

    def test_like_and_not_like_keep_null_unknown(self):
        db = self.database()
        self.addCleanup(db.close)
        self.check_query(db, "SELECT count(*) FROM t WHERE title LIKE '%Google%'", predicates=1)
        self.check_query(db, "SELECT count(*) FROM t WHERE title NOT LIKE '%Google%'", predicates=1)
        self.check_query(db, "SELECT count(*) FROM t WHERE title ILIKE '%GOOGLE%' "
                         "AND url NOT LIKE '%.google.%'", predicates=2)
        self.check_query(db, "SELECT count(*) FROM t WHERE url LIKE '%!_!%Google' ESCAPE '!'", predicates=1)

    def test_utf8_lengths_count_gate_and_regex_minimum(self):
        db = self.database()
        self.addCleanup(db.close)
        self.check_query(db, "SELECT g, AVG(STRLEN(url)) AS l, COUNT(*) AS n FROM t WHERE url <> '' "
                         "GROUP BY g HAVING COUNT(*) > 5000 ORDER BY l DESC, g", projection=True)
        self.check_query(db, "SELECT g, AVG(CHAR_LENGTH(url)) AS l, COUNT(*) AS n FROM t WHERE url <> '' "
                         "GROUP BY g HAVING COUNT(*) > 5000 ORDER BY l DESC, g", projection=True)
        regex = self.check_query(db, r"SELECT REGEXP_REPLACE(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, "
                         "AVG(STRLEN(url)) AS l, COUNT(*) AS n, MIN(url) FROM t WHERE url <> '' "
                         "GROUP BY k HAVING COUNT(*) > 5000 ORDER BY l DESC, k", projection=True)
        if sqltexthist._code_histogram is not None:
            self.assertEqual(regex.stats.get("dictionary_histogram_codes"), 5)
        self.check_query(db, "SELECT g, AVG(STRLEN(title)) AS l, COUNT(*) AS n FROM t "
                         "GROUP BY g HAVING COUNT(*) > 5000 ORDER BY l DESC, g", projection=True)
        self.assertEqual(db.sql("SELECT g, AVG(STRLEN(url)) FROM t GROUP BY g "
                                "HAVING COUNT(*) > 100000 ORDER BY g").rows, [])

    def test_generated_dictionary_key_and_moved_nullable_value(self):
        row = np.arange(70_013)
        det = row % 17
        attrs = {f"attr{i}": det * (i + 1) for i in range(32)}
        data = pd.DataFrame(dict(url=np.array([f"https://google.com/é/{i}" for i in range(17)])[det],
                                 g=(row // 17) % 3, **attrs))
        db = torbles.from_frames({"t": data})
        self.addCleanup(db.close)
        self.assertEqual(db["t"].codecs["url"].kind, "ref")
        res = self.check_query(db, "SELECT g, COUNT(*), AVG(STRLEN(url)), MIN(url) FROM t "
                               "WHERE url LIKE '%google.com%' GROUP BY g ORDER BY g",
                               predicates=1, projection=True)
        self.assertEqual(res.stats.get("dictionary_dimension_predicates"), 1)
        self.assertEqual(res.stats.get("dictionary_dimension_projection_rows"), len(row))

        labels = np.array([None, "https://café.com/é", "other"], dtype=object)
        data = pd.DataFrame(dict(det=det, g=(row // 17) % 3, url=labels[det % 3], **attrs))
        db = torbles.from_frames({"t": data})
        self.addCleanup(db.close)
        self.assertEqual(db["t"].moved["url"], "det")
        res = self.check_query(db, "SELECT g, COUNT(*), AVG(STRLEN(url)), MIN(url) FROM t "
                               "GROUP BY g ORDER BY g", projection=True)
        self.assertEqual(res.stats.get("dictionary_dimension_projection_rows"), len(row))
        res = self.check_query(db, "SELECT g, COUNT(*), AVG(STRLEN(url)), MIN(url) FROM t "
                               "WHERE det = 0 GROUP BY g ORDER BY g", projection=True)
        self.assertTrue(all(r[2:] == (None, None) for r in res.rows))
        res = self.check_query(db, "SELECT count(*) FROM t WHERE url NOT LIKE '%café.com%'",
                               predicates=1)
        self.assertEqual(res.stats.get("dictionary_dimension_predicates"), 1)

    @unittest.skipIf(sqltextplanes._lookup_sums is None, "native plane lookup unavailable")
    def test_length_contracts_only_groups_that_can_pass_having(self):
        rows = np.arange(70_013)
        ids = rows % 17
        data = pd.DataFrame(dict(
            url=np.array([f"https://café.com/{i}" for i in range(17)])[ids],
            g=np.where(rows < 15000, 0, rows % 37 + 1),
            **{f"attr{i}": ids * (i + 1) for i in range(32)},
        ))
        db = torbles.from_frames({"t": data})
        self.addCleanup(db.close)
        self.assertEqual(db["t"].codecs["url"].kind, "ref")
        query = ("SELECT g, COUNT(*), AVG(STRLEN(url)) FROM t WHERE url LIKE '%café%' "
                 "GROUP BY g HAVING COUNT(*) >= 10000 ORDER BY g")
        result = self.check_query(db, query, predicates=1, projection=True)
        self.assertEqual(result.stats.get("dictionary_plane_groups"), 1)
        self.assertEqual(result.stats.get("dictionary_plane_selected_rows"), 15000)

    def test_text_caches_belong_to_open_table(self):
        first = self.database()
        self.addCleanup(first.close)
        table = first["t"]
        query = "SELECT g, AVG(STRLEN(url)), COUNT(*) FROM t WHERE url LIKE '%com%' GROUP BY g ORDER BY g"
        expected = self.check_query(first, query, predicates=1, projection=True).rows
        self.assertTrue(table._text_predicates)
        self.assertTrue(table._text_lengths)
        first.close()
        self.assertNotIn("_text_predicates", table.__dict__)
        self.assertNotIn("_text_lengths", table.__dict__)

        second = self.database(70_017)
        self.addCleanup(second.close)
        answer = self.check_query(second, query, predicates=1, projection=True).rows
        self.assertNotEqual(answer, expected)


if __name__ == "__main__":
    unittest.main()

