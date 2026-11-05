# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Fused raw-plane COUNT, SUM and AVG against the existing SQL executor."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import algq


@unittest.skipIf(algq._plane_group_topk is None, "native fused grouping library not built")
class RawPlaneGroupChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        row = np.arange(70_013, dtype=np.int64)
        watch = ((row % 19) - 9) * (1 << 49) + row % 7
        ip = ((row // 5) % 17) - 11
        flag = (row % 23) - 14
        width = (row % 79) + 100
        optional = pd.array(width, dtype="Int32")
        optional[row % 43 == 0] = None
        frame = pd.DataFrame(dict(watch=watch, ip=ip, flag=flag, width=width, optional=optional))
        cls.db = torbles.from_ddl(
            "CREATE TABLE t(watch BIGINT, ip INTEGER, flag INTEGER, width INTEGER, optional INTEGER)",
            {"t": frame}, keep_plain=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_fused_wide_keys_signed_sum_average_filter_and_offset(self):
        select = ("SELECT watch, ip, COUNT(*) AS c, SUM(flag), AVG(width), COUNT(width) "
                  "FROM t WHERE flag > -12 AND width <= 170 GROUP BY watch, ip")
        query = select + " ORDER BY c DESC LIMIT 10 OFFSET 3"
        expected = self.db.con.execute(select + " ORDER BY c DESC, watch, ip LIMIT 10 OFFSET 3").fetchall()
        result = self.db.sql(query)
        self.assertEqual(result.rows, expected)
        self.assertEqual(result.stats.get("raw_plane_group_rows"), 70_013)
        self.assertEqual(result.stats.get("raw_plane_group_groups"), 10)

    def test_reused_measure_and_stable_count_ties(self):
        query = "SELECT watch, COUNT(*) AS c, SUM(flag), AVG(flag) FROM t GROUP BY watch"
        expected = self.db.con.execute(query + " ORDER BY c DESC, watch LIMIT 14").fetchall()
        result = self.db.sql(query + " ORDER BY COUNT(*) DESC LIMIT 14")
        self.assertEqual(result.rows, expected)
        self.assertEqual(result.stats.get("raw_plane_group_groups"), 14)

    def test_minute_groups_from_signed_wide_timestamp_and_dictionary(self):
        row = np.arange(70_013, dtype=np.int64)
        seconds = (row % 181 - 92) * 91
        stamp = seconds.astype("datetime64[s]").astype("datetime64[us]")
        user = (row % 37 - 18) * (1 << 49)
        phrase = np.array(["", "alpha", "caf\u00e9"])[row % 3]
        frame = pd.DataFrame(dict(uid=user, event_time=stamp, phrase=phrase))
        db = torbles.from_ddl(
            "CREATE TABLE t(uid BIGINT, event_time TIMESTAMP, phrase VARCHAR)", {"t": frame}
        )
        self.addCleanup(db.close)
        self.assertIn("event_time", db["t"].wide)
        select = ("SELECT uid, EXTRACT(minute FROM event_time) AS m, phrase, COUNT(*) AS c "
                      "FROM t GROUP BY uid, m, phrase")
        actual = db.sql(select + " ORDER BY c DESC LIMIT 10")
        expected = db.con.execute(select + " ORDER BY c DESC, uid, m, phrase LIMIT 10").fetchall()
        self.assertEqual(actual.rows, expected)
        self.assertEqual(actual.stats.get("raw_plane_group_groups"), 10)
        bucket = ("SELECT DATE_TRUNC('minute', event_time) AS bucket, COUNT(*) "
                  "FROM t GROUP BY bucket ORDER BY bucket LIMIT 10 OFFSET 4")
        actual = db.sql(bucket)
        self.assertEqual(actual.rows, db.con.execute(bucket).fetchall())
        self.assertEqual(actual.stats.get("plane_group_ordered_bucket"), 10)

    def test_generated_dimension_key_decoded_only_for_output_groups(self):
        row = np.arange(70_013, dtype=np.int64)
        ip = (row % 13) * 13 + 100
        frame = pd.DataFrame({"ip": ip, "watch": (row % 31) * (1 << 48),
                              **{f"attr{i}": ip * (i + 1) for i in range(32)}})
        db = torbles.from_frames({"t": frame})
        self.addCleanup(db.close)
        self.assertEqual(db["t"].codecs["ip"].kind, "ref")
        select = "SELECT watch, ip, COUNT(*) AS c FROM t GROUP BY watch, ip"
        actual = db.sql(select + " ORDER BY c DESC LIMIT 10")
        oracle = dict((tuple(row[:2]), row[2]) for row in db.con.execute(select).fetchall())
        counts = sorted(oracle.values(), reverse=True)[:10]
        self.assertEqual([r[2] for r in actual.rows], counts)
        self.assertEqual(len({tuple(r[:2]) for r in actual.rows}), 10)
        self.assertTrue(all(oracle[tuple(r[:2])] == r[2] for r in actual.rows))
        self.assertEqual(actual.stats.get("raw_plane_group_groups"), 10)

    def test_affine_and_constant_group_keys(self):
        query = ("SELECT 1, watch, watch-2, COUNT(*) AS c FROM t "
                 "GROUP BY 1, watch, watch-2 ORDER BY c DESC LIMIT 10")
        result = self.db.sql(query)
        self.assertEqual(result.stats.get("raw_plane_group_groups"), 10)
        oracle = self.db.con.execute(query.replace("ORDER BY c DESC", "ORDER BY c DESC, watch")).fetchall()
        self.assertEqual(result.rows, oracle)

    def test_scalar_moments_and_small_wide_lookup(self):
        query = "SELECT COUNT(*), SUM(flag), AVG(watch) FROM t WHERE ip > 2"
        result = self.db.sql(query)
        self.assertEqual(result.stats.get("scalar_plane_moments"), 2)
        oracle = self.db.con.execute("SELECT COUNT(*), SUM(flag), SUM(watch) FROM t WHERE ip > 2").fetchone()
        self.assertEqual(result.rows, [(oracle[0], oracle[1], oracle[2] / oracle[0])])
        # A selective point lookup returns exact rows; a frequent match
        # that exceeds its bounded ID buffer uses the ordinary route.
        query = "SELECT watch, width FROM t WHERE watch = 123456789012345"
        result = self.db.sql(query)
        self.assertEqual(result.stats.get("raw_plane_point_lookup"), 0)
        self.assertEqual(result.rows, self.db.con.execute(query).fetchall())
        query = "SELECT watch, width FROM t WHERE watch = -5066549580791808"
        self.assertEqual(self.db.sql(query).rows, self.db.con.execute(query).fetchall())

    def test_floating_scalar_measures_keep_floating_semantics(self):
        query = ("SELECT SUM(CAST(flag AS DOUBLE)), AVG(CAST(flag AS DOUBLE)) "
                 "FROM t WHERE ip > 2")
        result = self.db.sql(query)
        self.assertNotIn("scalar_plane_moments", result.stats)
        self.assertEqual(result.rows, self.db.con.execute(query).fetchall())

    def test_case_branch_groups_and_empty_offset(self):
        query = ("SELECT watch, ip, flag, width, CASE WHEN ip > 2 THEN phrase ELSE '' END AS k, COUNT(*) c "
                 "FROM labels GROUP BY watch, ip, flag, width, k")
        frame = self.db.con.execute("SELECT *, CASE WHEN width % 2 = 0 THEN 'even' ELSE 'odd' END phrase FROM t").df()
        db = torbles.from_frames({"labels": frame})
        self.addCleanup(db.close)
        result = db.sql(query + " ORDER BY c DESC LIMIT 10 OFFSET 3")
        self.assertEqual(result.stats.get("plane_group_case_branches"), 10)
        rows = db.con.execute(query).fetchall()
        counts = sorted((r[-1] for r in rows), reverse=True)[3:13]
        self.assertEqual([r[-1] for r in result.rows], counts)
        self.assertTrue(set(result.rows) <= set(rows))
        result = self.db.sql("SELECT ip, COUNT(*) c FROM t WHERE flag = -14 GROUP BY ip ORDER BY c DESC LIMIT 10 OFFSET 10000")
        self.assertEqual(result.rows, [])
        self.assertIn("plane_group_empty_offset", result.stats)

    def test_count_rank_gathers_only_winning_groups(self):
        query = ("SELECT ip, COUNT(*) c, MIN(width), COUNT(DISTINCT optional) "
                 "FROM t WHERE width < 150 GROUP BY ip ORDER BY c DESC LIMIT 4 OFFSET 2")
        result = self.db.sql(query)
        self.assertIn("plane_group_late_states", result.stats)
        oracle = self.db.con.execute(query.split("ORDER BY")[0]).fetchall()
        counts = sorted((row[1] for row in oracle), reverse=True)[2:6]
        self.assertEqual([row[1] for row in result.rows], counts)
        self.assertTrue(set(result.rows) <= set(oracle))

    def test_high_offset_on_selective_groups(self):
        query = ("SELECT ip, width, COUNT(*) c FROM t WHERE flag = -14 "
                 "GROUP BY ip, width ORDER BY c DESC LIMIT 10 OFFSET 2000")
        result = self.db.sql(query)
        self.assertIn("plane_group_packed_offset", result.stats)
        self.assertEqual(result.rows, self.db.con.execute(query).fetchall())

    def test_cold_cuda_wide_filter_uses_a_bounded_cpu_scope(self):
        from torbles import sqlgpugroups

        row = np.arange(70_013, dtype=np.int64)
        sparse, dense = -((1 << 52) + 19), (1 << 53) + 25
        ref = (row % 127 - 61) * (1 << 42) + row % 23
        key = (row % 97 - 47) * (1 << 47) + row % 7
        day, value = 1 + row % 3, 1 + row % 11
        groups = np.repeat(np.arange(9), [2, 7, 4, 7, 3, 6, 4, 1, 8])
        ref[-len(groups):], ref[:3_000] = sparse, dense
        key[-len(groups):] = (groups - 4) * (1 << 48)
        day[-len(groups):] = 1 + groups % 3
        value[-1] = 0
        nullable = pd.array(ref, dtype="Int64")
        nullable[100] = None
        frame = pd.DataFrame(dict(ref=nullable, k=key, d=day, v=value))
        db = torbles.from_ddl(
            "CREATE TABLE t(ref BIGINT, k BIGINT, d INTEGER, v INTEGER)", {"t": frame}, groups=False,
        )
        self.addCleanup(db.close)
        self.assertIn("ref", db["t"].wide)
        self.assertIn("k", db["t"].wide)

        def query(literal, offset=2):
            return (f"SELECT k, d, COUNT(*) c, SUM(v), AVG(v) FROM t "
                    f"WHERE ref = {literal} AND d BETWEEN 1 AND 3 AND v <> 0 "
                    f"GROUP BY k,d ORDER BY c DESC LIMIT 4 OFFSET {offset}")

        def oracle(sql):
            return db.con.execute(sql.replace("ORDER BY c DESC", "ORDER BY c DESC, k, d")).fetchall()

        with patch.object(algq.Table, "gpu_handle", side_effect=AssertionError("cold part uploaded")):
            for offset in (2, 20):
                with self.subTest(offset=offset):
                    sql = query(sparse, offset)
                    result = db.sql(sql, engine="gpu")
                    self.assertEqual(result.rows, oracle(sql))
                    self.assertEqual(result.stats.get("cold_wide_group_cpu"), 1)
                    self.assertEqual(result.stats.get("cold_wide_group_rows"), len(groups) - 1)
                    self.assertGreater(result.stats.get("cold_wide_group_avoided_plane_bytes", 0), 0)
                    self.assertNotIn("cuda_scope_bitmaps", result.stats)

        with patch.object(algq.Table, "gpu_handle", side_effect=AssertionError("cold part uploaded")):
            sql = (f"SELECT d, COUNT(*) c FROM t WHERE ref = {sparse} "
                   "GROUP BY d ORDER BY c DESC LIMIT 4 OFFSET 2000")
            result = db.sql(sql, engine="gpu")
        self.assertEqual(result.rows, db.con.execute(sql).fetchall())
        self.assertIn("plane_group_empty_offset", result.stats)
        self.assertEqual(result.stats.get("cold_wide_scope_cpu"), 1)
        self.assertNotIn("cuda_scope_bitmaps", result.stats)

        def verify_scope(fact, keys, measures, limit, scope, stats):
            return fact._plane_group(keys, measures, limit, scope)

        with patch.object(sqlgpugroups, "group", side_effect=verify_scope) as grouped:
            sql = query(dense)
            result = db.sql(sql, engine="gpu")
            grouped.assert_called_once()
        self.assertEqual(result.rows, oracle(sql))
        self.assertEqual(result.stats.get("cold_wide_scope_cpu"), 1)
        self.assertNotIn("cold_wide_group_cpu", result.stats)
        self.assertEqual(result.stats.get("cold_wide_scope_rows"), 2_999)
        self.assertLess(result.stats["cold_wide_group_row_budget"], 2_999)

    def test_unsupported_aggregates_and_order_use_existing_path(self):
        queries = (
            "SELECT watch, COUNT(*) c, AVG(optional) FROM t GROUP BY watch ORDER BY c DESC LIMIT 6",
            "SELECT watch, COUNT(*) c, SUM(flag) FROM t GROUP BY watch ORDER BY c DESC, watch LIMIT 6",
            "SELECT watch, COUNT(*) c FROM t GROUP BY watch ORDER BY c DESC FETCH FIRST 6 ROWS WITH TIES",
            "SELECT ip, COUNT(*) c FROM t GROUP BY ip ORDER BY c DESC LIMIT 6",
        )
        for query in queries:
            with self.subTest(query=query):
                result = self.db.sql(query)
                with patch.object(algq, "_plane_group_topk", None):
                    expected = self.db.sql(query).rows
                self.assertEqual(result.rows, expected)
                self.assertNotIn("raw_plane_group_rows", result.stats)


if __name__ == "__main__":
    unittest.main()
