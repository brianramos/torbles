# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""SQL correctness and routing for ordering directly from stored bitplanes."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles


class RawPlaneOrderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Cross a word and chunk boundary, with ties and holes after filtering.
        i = np.arange(131_117, dtype=np.int64)
        a = ((i * 11) % 7 - 3).astype(np.int32)
        b = ((i * 17) % 23 - 11).astype(np.int32)
        p = ((i * 5) % 19 - 9).astype(np.int32)
        keep = (i % 17 != 0) & (i % 31 != 0)
        nullable = pd.array(b, dtype="Int32")
        nullable[i % 97 == 0] = None
        cls.frame = pd.DataFrame(dict(id=i, a=a, b=b, p=p, keep=keep, nb=nullable))
        cls.db = torbles.from_ddl(
            "CREATE TABLE t(id INTEGER, a INTEGER, b INTEGER, p INTEGER, keep BOOLEAN, nb INTEGER)",
            {"t": cls.frame}, keep_plain=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def check(self, sql, *, order=False, window=False, expected=None):
        if expected is None:
            expected = self.db.con.execute(sql).fetchall()
        for engine in ("alg", "gpu", "plain"):
            with self.subTest(engine=engine, sql=sql):
                result = self.db.sql(sql, engine=engine)
                self.assertEqual(result.rows, expected)
                native = engine != "plain"
                self.assertEqual(bool(result.stats.get("raw_plane_order_rows")), native and order, result.stats)
                self.assertEqual(bool(result.stats.get("raw_plane_row_number_rows")), native and window, result.stats)

    def test_filtered_order_mixed_directions_with_offset(self):
        self.check("SELECT id, a, b FROM t WHERE keep AND p >= 0 "
                   "ORDER BY a DESC, b ASC, id DESC LIMIT 100001 OFFSET 119", order=True)

    def test_equal_sort_keys_keep_input_order(self):
        frame = self.frame
        filtered = np.flatnonzero((frame.keep & (frame.p >= 0)).to_numpy())
        a, b = frame.a.to_numpy(), frame.b.to_numpy()
        expected_ids = filtered[np.lexsort((b[filtered], -a[filtered]))][120:]
        expected = [(int(i), int(a[i]), int(b[i])) for i in expected_ids]
        self.check("SELECT id, a, b FROM t WHERE keep AND p >= 0 "
                   "ORDER BY a DESC, b ASC OFFSET 120", order=True, expected=expected)

    def test_partitioned_row_number_with_filtered_rows_and_ties(self):
        frame = self.frame
        a, b, p = (frame[name].to_numpy() for name in ("a", "b", "p"))
        filtered = np.flatnonzero((frame.keep & (frame.p >= -6)).to_numpy())
        ordered = filtered[np.lexsort((a[filtered], -b[filtered], p[filtered]))]
        counts = {}
        ranks = {}
        for i in ordered:
            partition = p[i]
            counts[partition] = counts.get(partition, 0) + 1
            ranks[int(i)] = counts[partition]
        expected = [(int(i), int(a[i]) + int(b[i]), ranks[int(i)]) for i in filtered]
        self.check("SELECT id, a + b, row_number() OVER (PARTITION BY p ORDER BY b DESC, a ASC) "
                   "FROM t WHERE keep AND p >= -6 ORDER BY id", window=True, expected=expected)

    def test_unfiltered_row_number_in_fact_order(self):
        self.check("SELECT id, row_number() OVER (PARTITION BY p ORDER BY b DESC, id) "
                   "FROM t ORDER BY id LIMIT 53", window=True)

    def test_sort_only_keys_do_not_decode(self):
        table = self.db["t"]
        decode = table._decode_column
        queries = (
            ("SELECT id FROM t ORDER BY p, b DESC, id", "raw_plane_order_rows"),
            ("SELECT id, row_number() OVER (PARTITION BY p ORDER BY b DESC, id) "
             "FROM t ORDER BY id", "raw_plane_row_number_rows"),
        )
        def payload_only(name):
            self.assertEqual(name, "id")
            return decode(name)
        for sql, route in queries:
            with self.subTest(route=route), patch.object(table, "_decode_column", side_effect=payload_only):
                result = self.db.sql(sql)
                self.assertEqual(result.rows, self.db.con.execute(sql).fetchall())
                self.assertEqual(result.stats[route], len(self.frame))

    def test_nullable_and_computed_sort_keys_use_general_ordering(self):
        self.check("SELECT id, nb FROM t WHERE keep ORDER BY nb DESC NULLS FIRST, id OFFSET 14")
        self.check("SELECT id, a, b FROM t WHERE keep ORDER BY a + b DESC, id OFFSET 14")

    def test_unsupported_window_shapes_use_general_ordering(self):
        self.check("SELECT id, row_number() OVER (PARTITION BY p ORDER BY nb DESC NULLS FIRST, id) "
                   "FROM t WHERE keep ORDER BY id LIMIT 53")
        self.check("SELECT id, row_number() OVER (PARTITION BY p ORDER BY b DESC, id), "
                   "rank() OVER (PARTITION BY p ORDER BY a, id) FROM t WHERE keep ORDER BY id LIMIT 53")


if __name__ == "__main__":
    unittest.main()
