# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""SQL statement execution and atomicity for writable bitplane tables."""

import unittest

import duckdb
import torbles


class SQLDMLChecks(unittest.TestCase):
    def setUp(self):
        self.db = torbles.Database()
        self.addCleanup(self.db.close)
        self.duck = duckdb.connect()
        self.addCleanup(self.duck.close)
        self.register("fact", {"id": [1, 2, 3, 4], "v": [100, 200, 300, 400], "dk": [1, 1, 2, 2]})
        self.register("dim", {"id": [1, 2], "region": [0, 1]})

    def register(self, name, columns):
        self.db.add_writable_table(name, columns)
        schema = ", ".join(f"{k} UINTEGER" + (" PRIMARY KEY" if k == "id" else " NOT NULL") for k in columns)
        self.duck.execute(f"CREATE TABLE {name} ({schema})")
        placeholders = ",".join("?" for _ in columns)
        if len(columns["id"]):
            self.duck.executemany(f"INSERT INTO {name} VALUES ({placeholders})", list(zip(*columns.values())))

    def match(self, sql, engine="alg", keyed=None):
        got = self.db.sql(sql, engine)
        self.assertEqual(got.rows, self.duck.execute(sql).fetchall())
        if keyed is not None:
            self.assertEqual(bool(got.stats["writable_key_lookup"]), keyed)
        return got

    def test_four_statements_and_native_reads(self):
        self.register("stage", {"id": [2, 5], "v": [201, 230], "dk": [1, 2]})
        self.match("INSERT INTO fact (dk, id, v) VALUES (1, 6, 111)", keyed=False)
        self.match("INSERT INTO fact SELECT 7, 171, 2", keyed=False)
        self.match("SELECT COUNT(*) FROM fact WHERE v BETWEEN 100 AND 200", "gpu")
        self.match("UPDATE fact AS f SET v = f.v + u.v FROM stage AS u WHERE u.id = f.id", "gpu", True)
        self.match("UPDATE fact SET v = v + 1 WHERE id = 3", keyed=True)
        self.match("DELETE FROM fact AS f USING stage AS u WHERE f.id = u.id", keyed=True)
        self.match("MERGE INTO fact f USING stage s ON f.id = s.id "
                   "WHEN MATCHED THEN UPDATE SET v=s.v "
                   "WHEN NOT MATCHED THEN INSERT (id,v,dk) VALUES (s.id,s.v,s.dk)", keyed=True)
        self.match("MERGE INTO fact f USING stage s ON f.id=s.id WHEN MATCHED THEN UPDATE SET v=f.v+1", keyed=True)
        self.match("UPDATE dim SET region=1 WHERE id=1", keyed=True)
        result = self.match("SELECT SUM(f.v) FROM fact f JOIN dim d ON f.dk=d.id WHERE d.region=1 AND f.v<400", "gpu")
        self.assertEqual(result.stats.get("writable_native_joins"), 1)
        self.match("SELECT dk, COUNT(*), SUM(v) FROM fact WHERE v<450 GROUP BY dk ORDER BY dk")
        self.match("SELECT * FROM fact ORDER BY id")
        self.assertEqual(self.db["fact"].store.stats["plane_update_batches"], 3)

    def test_standard_correlated_forms_and_merge_delete(self):
        self.register("stage", {"id": [2, 5], "v": [205, 405], "dk": [2, 1]})
        self.match("UPDATE fact AS f SET v=(SELECT s.v FROM stage s WHERE s.id=f.id) "
                   "WHERE EXISTS (SELECT 1 FROM stage s WHERE s.id=f.id)")
        self.match("DELETE FROM fact WHERE EXISTS (SELECT 1 FROM stage s WHERE s.id=fact.id)")
        self.match("INSERT INTO fact SELECT s.id,s.v,s.dk FROM stage s WHERE s.id=2")
        self.match("MERGE INTO fact AS f USING stage AS s ON f.id=s.id "
                   "WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT (id,v,dk) VALUES (s.id,s.v,s.dk)",
                   keyed=True)
        self.match("SELECT * FROM fact ORDER BY id")
        self.match("INSERT INTO fact SELECT * FROM stage WHERE id=999")

    def test_invalid_statements_preserve_visible_data(self):
        self.register("stage", {"id": [1, 8], "v": [110, 180], "dk": [1, 1]})
        before = self.db.sql("SELECT * FROM fact ORDER BY id").rows
        for sql in (
            "INSERT INTO fact VALUES (8, 10, 1), (1, 20, 1)",
            "INSERT INTO fact (id,v) VALUES (8,10)",
            "UPDATE fact SET v=CASE WHEN id=1 THEN 111 ELSE NULL END",
            "UPDATE fact SET v=-1 WHERE id=1",
            "UPDATE fact SET id=8 WHERE id=1",
            "UPDATE fact f SET v=s.v FROM (VALUES (1, 90), (1, 91)) s(id,v) WHERE f.id=s.id",
            "MERGE INTO fact f USING stage s ON f.id=s.id WHEN MATCHED THEN UPDATE SET v=s.v "
            "WHEN NOT MATCHED THEN INSERT (id,v,dk) VALUES (2,s.v,s.dk)",
            "MERGE INTO fact f USING stage s ON f.v=s.v WHEN MATCHED THEN DELETE",
        ):
            with self.subTest(sql=sql), self.assertRaises(torbles.SqlError):
                self.db.sql(sql)
            self.assertEqual(self.db.sql("SELECT * FROM fact ORDER BY id").rows, before)

    def test_assignment_reports_the_original_expression_error(self):
        self.register("stage", {"id": [1], "v": [5]})
        with self.assertRaisesRegex(torbles.SqlError, "5x"):
            self.db.sql("UPDATE fact f SET v=CAST(CAST(s.v AS VARCHAR) || 'x' AS INTEGER) "
                        "FROM stage s WHERE f.id=s.id")
        self.assertEqual(self.db.sql("SELECT v FROM fact WHERE id=1").rows, [(100,)])

    def test_duplicate_delete_keys_and_sequential_statements(self):
        result = self.db.sql("DELETE FROM fact f USING (VALUES (1), (1), (99)) s(id) WHERE f.id=s.id")
        self.assertEqual(result.rows, [(1,)])
        result = self.db.sql("INSERT INTO fact VALUES (1, 9, 1); UPDATE fact SET v=10 WHERE id=1; "
                             "SELECT v FROM fact WHERE id=1")
        self.assertEqual(result.rows, [(10,)])


if __name__ == "__main__":
    unittest.main()
