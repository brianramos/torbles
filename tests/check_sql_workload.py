# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Cross-statement sharing of exact moment states."""

import unittest

import duckdb
import torbles
from torbles.sqltypes import SqlError


class WorkloadChecks(unittest.TestCase):
    def setUp(self):
        con = duckdb.connect(config={"threads": "2"})
        self.addCleanup(con.close)
        con.execute("CREATE TABLE t AS SELECT i % 7 g, i % 431 x, i % 23 y FROM range(100003) s(i)")
        self.db = torbles.from_duckdb(con, tables=["t"], threads=2)
        self.addCleanup(self.db.close)

    def test_fused_scope_keeps_statement_outputs_and_counts_shared_work_once(self):
        sql = [
            "SELECT count(*) n FROM t WHERE g < 4",
            "SELECT count(*) other FROM t WHERE g < 2",
            "SELECT sum(a.x) s, avg(a.x) a FROM t a WHERE a.g < 4",
            "SELECT avg(y) b, count(*) n FROM t WHERE g < 4",
            "SELECT x FROM t ORDER BY x DESC LIMIT 2",
        ]
        results = self.db.sql_batch(";".join(sql))
        for statement, result in zip(sql, results):
            independent = self.db.sql(statement)
            self.assertEqual((result.columns, result.types, result.rows),
                             (independent.columns, independent.types, independent.rows))
        self.assertEqual(results[0].stats["workload_shared_statements"], 3)
        self.assertEqual(results[0].stats["workload_output_slots"], 4)
        self.assertEqual(results[0].stats["scalar_plane_moments"], 2)
        self.assertEqual(results[2].stats["engine_queries"], 0)
        self.assertEqual(results[3].stats["engine_queries"], 0)
        self.assertNotIn("workload_shared_statements", results[1].stats)
        self.assertNotIn("workload_shared_statements", results[4].stats)

    def test_empty_scope_and_read_only_boundary(self):
        sql = "SELECT sum(x) FROM t WHERE g < 0; SELECT count(*) FROM t WHERE g < 0"
        self.assertEqual([r.rows for r in self.db.sql_batch(sql)], [[(None,)], [(0,)]])
        self.assertEqual(self.db.sql_batch(""), [])
        for statement in ("SET TIME ZONE 'America/New_York'", "DELETE FROM t"):
            with self.assertRaises(SqlError):
                self.db.sql_batch("SELECT count(*) FROM t; " + statement)
        self.assertIsNone(getattr(self.db, "_sql_zone", None))
