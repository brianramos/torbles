# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Dictionary LIKE and grouped text projections over encoded bitplanes."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import coltypes, sqltexthist, sqltextplanes


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

    def test_packed_varchar_open_and_native_literal_mask(self):
        db = self.database(1031)
        self.addCleanup(db.close)
        with TemporaryDirectory() as archive:
            db.save(archive)
            packed = torbles.load(archive, packed_text=True)
            self.addCleanup(packed.close)
            values = packed["t"].codecs["url"].values
            self.assertIsInstance(values, coltypes.PackedStrings)
            self.assertIsInstance(values.blob, np.memmap)
            self.assertEqual(list(values), db["t"].codecs["url"].values)
            if sqltextplanes._packed_contains is None:
                self.skipTest("native packed substring support unavailable")
            queries = (
                "SELECT COUNT(*) FROM t WHERE url LIKE '%café%'",
                "SELECT COUNT(*) FROM t WHERE url NOT LIKE '%café%'",
                "SELECT COUNT(*) FROM t WHERE title NOT LIKE '%Google%'",
                "SELECT url, COUNT(*) FROM t GROUP BY url ORDER BY url",
            )
            for sql in queries:
                with self.subTest(sql=sql):
                    self.assertEqual(packed.sql(sql).rows, db.sql(sql).rows)
            packed.close()

    def test_cuda_referrer_projection_matches_torbles_regex(self):
        try:
            if not hasattr(torbles.algq.gpu_lib(), "gpu_referrer_prepare"):
                self.skipTest("native CUDA referrer projection unavailable")
            if "emulat" in torbles.algq.gpu_name().lower():
                self.skipTest("CUDA device required")
        except (OSError, RuntimeError):
            self.skipTest("CUDA device unavailable")
        strings = [
            "", "abc", "https://www.abc/", "http://abc/x", "http://www./", "http://www.abc",
            "http://café/é", "café", "http://a\nb/x", "http://h/p\n", "h\n", "http://h/p\nx",
            "http://h/p\n\n", "http://a\x00b/", "https:///x", "HTTPS://abc/", "www.",
        ]
        db = torbles.from_ddl(
            "CREATE TABLE r(url VARCHAR NOT NULL)",
            {"r": pd.DataFrame({"url": strings * 4096})},
        )
        self.addCleanup(db.close)
        with TemporaryDirectory() as archive:
            db.save(archive)
            packed = torbles.load(archive, cuda_referrers=(("r", "url"),))
            self.addCleanup(packed.close)
            receipt = packed.text_preparation["r", "url"]
            self.assertIsNotNone(receipt)
            self.assertFalse(receipt["cached"])
            self.assertTrue(sqltexthist.prepare_referrer(packed["r"], "url")["cached"])
            for length in ("OCTET_LENGTH", "CHAR_LENGTH"):
                query = (r"SELECT REGEXP_REPLACE(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, "
                         f"COUNT(*), AVG({length}(url)), MIN(url) FROM r WHERE url <> '' "
                         "GROUP BY k HAVING COUNT(*) >= 2 ORDER BY k")
                expected = db.sql(query).rows
                with patch.object(sqltextplanes, "_regex_group", side_effect=AssertionError("Python projection")):
                    actual = packed.sql(query, engine="gpu")
                self.assertEqual(actual.stats.get("gpu_referrer_moments"), 1)
                self.assertEqual(actual.rows, expected)
            packed.close()


    def test_persisted_referrer_uses_counts_without_fact_scan(self):
        from torbles import referrer_index, sqlgroupplans

        strings = ["", "abc", "https://www.abc/", "http://abc/x", "http://www./",
                   "http://café/é", "café", "http://h/p\n", "h\n", "http://h/p\nx"]
        db = torbles.from_ddl("CREATE TABLE r(url VARCHAR NOT NULL)",
                             {"r": pd.DataFrame({"url": strings * 7000})})
        self.addCleanup(db.close)
        query = (r"SELECT REGEXP_REPLACE(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, "
                 "COUNT(*), AVG(OCTET_LENGTH(url)), MIN(url) FROM r {where} "
                 "GROUP BY k HAVING COUNT(*) > 1 ORDER BY k")
        queries = [query.format(where=where) for where in ("", "WHERE url <> ''")]
        expected = [db.sql(sql).rows for sql in queries]
        with TemporaryDirectory() as archive:
            receipt = db.save(archive, referrers=(("r", "url"),))
            self.assertEqual(receipt["records"][0]["codes"], len(strings))
            packed = torbles.load(archive)
            self.addCleanup(packed.close)
            self.assertIsNotNone(referrer_index.index_for(packed["r"], "url"))
            backends = [("alg", "cpu")]
            try:
                if hasattr(torbles.algq.gpu_lib(), "gpu_referrer_counted_prepare"):
                    backends.append(("gpu", "gpu"))
            except (OSError, RuntimeError):
                pass
            with patch.object(sqlgroupplans, "bitmap", side_effect=AssertionError("fact scan")), \
                    patch.object(sqltextplanes, "_regex_group", side_effect=AssertionError("regex rebuild")):
                for engine, stat in backends:
                    for sql, wanted in zip(queries, expected, strict=True):
                        result = packed.sql(sql, engine=engine)
                        self.assertEqual(result.rows, wanted)
                        self.assertEqual(result.stats.get(f"{stat}_referrer_counted_moments"), 1)
                        self.assertEqual(result.stats.get("referrer_fact_rows_scanned"), 0)
            other = query.format(where="WHERE url = 'abc'")
            result = packed.sql(other)
            self.assertEqual(result.rows, db.sql(other).rows)
            self.assertNotIn("cpu_referrer_counted_moments", result.stats)
            packed["r"].flip(0, 0)
            self.assertIsNone(referrer_index.index_for(packed["r"], "url"))
            packed.close()

            data = next((Path(archive) / referrer_index.DEFAULT_DIR).glob("*-counts.npy"))
            with data.open("ab") as stream:
                stream.write(b"changed")
            stale = torbles.load(archive)
            self.addCleanup(stale.close)
            self.assertIsNone(referrer_index.index_for(stale["r"], "url"))
            stale.close()


    def test_referrer_index_keeps_topology_archive_compact(self):
        from torbles import algq, referrer_index

        if algq._at_save_cycles is None or algq._at_has_dense is None:
            self.skipTest("native topology compression unavailable")
        urls = ["http://a/x", "http://b/xyz"] * 65536
        db = torbles.from_ddl("CREATE TABLE r(url VARCHAR NOT NULL)",
                             {"r": pd.DataFrame({"url": urls})})
        self.addCleanup(db.close)
        query = (r"SELECT REGEXP_REPLACE(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, "
                 "COUNT(*), AVG(OCTET_LENGTH(url)), MIN(url) FROM r GROUP BY k ORDER BY k")
        expected = db.sql(query).rows
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "encoded"
            db.save(archive, compression="topology", referrers=(("r", "url"),))
            loaded = torbles.load(archive)
            try:
                part = loaded["r"]._part_of("url")
                self.assertTrue(part._storage().encoded)
                self.assertIsNotNone(referrer_index.index_for(loaded["r"], "url"))
                result = loaded.sql(query, engine="alg")
                self.assertEqual(result.rows, expected)
                self.assertEqual(result.stats.get("cpu_referrer_counted_moments"), 1)
                self.assertFalse(algq._at_has_dense(part._raw_at))
            finally:
                loaded.close()

    def test_referrer_index_is_skipped_after_load_repairs_fact(self):
        from torbles import referrer_index

        rows = ["http://a/x", "http://b/xyz"] * 4096
        db = torbles.from_ddl("CREATE TABLE r(url VARCHAR NOT NULL)",
                             {"r": pd.DataFrame({"url": rows})})
        self.addCleanup(db.close)
        query = (r"SELECT REGEXP_REPLACE(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, "
                 "COUNT(*), AVG(OCTET_LENGTH(url)), MIN(url) FROM r GROUP BY k ORDER BY k")
        expected = db.sql(query).rows
        db["r"].flip(0, 0)  # Flip a value bit to the other valid dictionary code.
        with TemporaryDirectory() as archive:
            db.save(archive, referrers=(("r", "url"),))
            unchecked = torbles.load(archive, validate=False)
            try:
                self.assertIsNotNone(referrer_index.index_for(unchecked["r"], "url"))
                self.assertNotEqual(unchecked.sql(query).rows, expected)
            finally:
                unchecked.close()
            repaired = torbles.load(archive)
            try:
                self.assertEqual(repaired.integrity["r"]["repaired"], 1)
                self.assertIsNone(referrer_index.index_for(repaired["r"], "url"))
                self.assertEqual(repaired.sql(query).rows, expected)
            finally:
                repaired.close()

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

