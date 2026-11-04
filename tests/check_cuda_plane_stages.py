# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Physical CUDA paths for packed SQL stages."""

import unittest

import numpy as np
import pandas as pd

import torbles
from torbles import algq, sqlalgebra


class CudaPlaneStages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(algq.gpu_lib(), "gpu_scalar_plane_moments"):
            raise unittest.SkipTest("native CUDA plane stages are not built")
        row = np.arange(135_013, dtype=np.int64)
        stamp = ((row % 1650 - 900) * 61).astype("datetime64[s]").astype("datetime64[us]")
        frame = pd.DataFrame(dict(
            id=row, watch=(row % 107 - 53) * (1 << 50) + row % 17,
            ip=row % 11, flag=row % 17 - 8, width=row % 73,
            stamp=stamp, phrase=np.array(["https://a.test/é", "https://b.test/x", ""])[row % 3],
        ))
        cls.db = torbles.from_ddl(
            "CREATE TABLE t(id BIGINT, watch BIGINT, ip INTEGER, flag INTEGER, width INTEGER, "
            "stamp TIMESTAMP, phrase VARCHAR)", {"t": frame}, groups=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def exact(self, sql, route):
        result = self.db.sql(sql, engine="gpu")
        self.assertIn(route, result.stats)
        self.assertEqual(result.rows, self.db.con.execute(sql).fetchall())

    def counted_groups(self, sql, order, route, offset):
        result = self.db.sql(sql + order, engine="gpu")
        self.assertIn(route, result.stats)
        groups = set(self.db.con.execute(sql).fetchall())
        self.assertTrue(set(result.rows) <= groups)
        counts = sorted((row[-1] for row in groups), reverse=True)[offset:offset + 10]
        self.assertEqual([row[-1] for row in result.rows], counts)

    def test_gpu_basis_can_finalize_on_the_host(self):
        catalog = sqlalgebra.Catalog(engine="gpu")
        catalog.learn(self.db, "SELECT SUM(watch), SUM(flag) FROM t WHERE width > 21")
        sql = "SELECT SUM(watch + flag) FROM t WHERE width > 21"
        result = catalog.query(self.db, sql)
        self.assertEqual(result.rows, self.db.con.execute(sql).fetchall())
        self.assertEqual(result.stats["engine_queries"], 0)
        self.assertEqual(result.stats["algebra_derivation"], "basis")

    def test_wide_scalar_moments(self):
        self.exact("SELECT COUNT(*), SUM(watch), AVG(flag) FROM t WHERE width > 21", "cuda_scalar_moments")

    def test_selective_wide_point(self):
        self.exact("SELECT watch FROM t WHERE watch = 56294995342131202", "cuda_plane_point_lookup")

    def test_packed_scope_from_dictionary_semijoin(self):
        self.exact("SELECT COUNT(*) FROM t WHERE phrase LIKE '%a.test%'", "cuda_scope_bitmaps")

    def test_exact_wide_and_dictionary_distinct(self):
        self.exact("SELECT COUNT(DISTINCT watch) FROM t WHERE width > 21", "cuda_encoded_distinct")
        self.exact("SELECT COUNT(DISTINCT phrase) FROM t", "cuda_encoded_distinct")

    def test_multi_key_topk_with_split_timestamp_and_offset(self):
        self.exact("SELECT id,stamp,phrase FROM t WHERE width > 21 ORDER BY stamp DESC, phrase, id "
                   "LIMIT 10 OFFSET 2", "cuda_raw_topk_rows")

    def test_encoded_group_measures_and_case_offsets(self):
        self.counted_groups("SELECT watch, ip, SUM(flag), AVG(width), COUNT(*) c FROM t "
                            "WHERE width > 21 GROUP BY watch, ip",
                            " ORDER BY c DESC LIMIT 10 OFFSET 3", "cuda_encoded_group", 3)
        self.counted_groups("SELECT watch,ip,flag,width,CASE WHEN flag > 0 THEN phrase ELSE '' END k,COUNT(*) c "
                            "FROM t GROUP BY watch,ip,flag,width,k",
                            " ORDER BY c DESC LIMIT 10 OFFSET 1000", "cuda_encoded_group", 1000)

    def test_time_buckets_and_dictionary_reductions(self):
        self.exact("SELECT DATE_TRUNC('minute',stamp) m,COUNT(*) FROM t WHERE width > 21 "
                   "GROUP BY m ORDER BY m LIMIT 10 OFFSET 1000", "cuda_minute_groups")
        self.exact("SELECT ip, AVG(STRLEN(phrase)), COUNT(*) FROM t WHERE phrase <> '' "
                   "GROUP BY ip HAVING COUNT(*) > 5000 ORDER BY ip", "gpu_text_lookup_sums")
        self.exact("SELECT regexp_replace(phrase,'/[^/]*$','') k,COUNT(*),AVG(STRLEN(phrase)),MIN(phrase) "
                   "FROM t WHERE phrase <> '' GROUP BY k ORDER BY k", "gpu_text_histograms")
