# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""Stored FK trees collapse filter-only ancestors, preserving SQL join semantics."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torbles
from torbles import algq


DDL = """
CREATE TABLE section(id INTEGER PRIMARY KEY, label VARCHAR);
CREATE TABLE seller(id INTEGER PRIMARY KEY, zone VARCHAR);
CREATE TABLE category(id INTEGER PRIMARY KEY, section_id INTEGER REFERENCES section(id), active INTEGER);
CREATE TABLE item(id INTEGER PRIMARY KEY, category_id INTEGER REFERENCES category(id),
                  seller_id INTEGER REFERENCES seller(id), family VARCHAR);
CREATE TABLE sale(item_id INTEGER REFERENCES item(id), amount INTEGER);
"""

CHAIN = """
FROM sale f JOIN item i ON f.item_id = i.id
JOIN category c ON i.category_id = c.id
JOIN section z ON c.section_id = z.id
"""
BRANCHES = CHAIN + "JOIN seller s ON i.seller_id = s.id\n"
SECTION_FILTER = "z.label = 'keep' AND c.active = 1"
SELLER_FILTER = "(s.zone = 'north' OR s.zone IS NULL)"


def integers(values):
    return pd.array(values, dtype="Int32")


def frames():
    # NULL and orphan keys occur at both the fact and dimension levels.
    # A matching seller with a NULL zone qualifies. A missing seller cannot.
    item_ids = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    facts = [*item_ids, None, 999, 101, 110]
    amounts = [8, 12, 7, 5, 6, 3, 2, 1, 4, 10, 20, 9, 11, None, -3]
    return {
        "section": pd.DataFrame({"id": integers([31, 32]), "label": ["keep", "drop"]}),
        "seller": pd.DataFrame({"id": integers([21, 22, 23]), "zone": ["north", "south", None]}),
        "category": pd.DataFrame({"id": integers([11, 12, 13, 14, 15]),
                                  "section_id": integers([31, 32, None, 99, 31]),
                                  "active": integers([1, 1, 1, 1, 0])}),
        "item": pd.DataFrame({"id": integers(item_ids),
                              "category_id": integers([11, 11, 12, 13, 14, None, 999, 11, 11, 11, 15]),
                              "seller_id": integers([21, 22, 21, 21, 21, 21, 21, None, 999, 23, 21]),
                              "family": ["a", "b", "a", "b", "c", "c", "b", "c", "b", "b", "c"]}),
        "sale": pd.DataFrame({"item_id": integers(facts * 274), "amount": integers(amounts * 274)}),
    }


class SnowflakeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = torbles.from_ddl(DDL, frames(), keep_plain=True)
        cls.engines = [os.environ["TORBLES_SQL_ENGINE"]] if os.environ.get("TORBLES_SQL_ENGINE") else ["alg", "gpu"]

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def check_query(self, sql, *, edges=0, projected_ancestor=False, expected=None):
        want = self.db.con.execute(sql).fetchall()
        if expected is not None:
            self.assertEqual(want, expected, sql)
        for engine in self.engines:
            with self.subTest(engine=engine, sql=sql):
                fact_devices = []
                execute = algq.Query._exec

                def observe(query, device, table, plan):
                    ran = execute(query, device, table, plan)
                    if ran and query.fact.name == "sale":
                        fact_devices.append(query.device)
                    return ran

                with patch.object(algq.Query, "_exec", observe):
                    result = self.db.sql(sql, engine=engine)
                self.assertEqual(result.rows, want)
                stats = result.stats
                self.assertEqual(stats.get("snowflake_blocks", 0), bool(edges), stats)
                self.assertEqual(stats.get("snowflake_edges", 0), edges, stats)
                if edges and not projected_ancestor:
                    self.assertEqual(stats["star_blocks"], 1, stats)
                    self.assertGreater(stats["engine_queries"], 0, stats)
                    self.assertEqual(stats["general_blocks"], 0, stats)
                    wanted_device = "cpu" if engine == "alg" else engine
                    self.assertTrue(any(d.startswith(wanted_device) for d in fact_devices), fact_devices)
                else:
                    self.assertGreater(stats["general_blocks"], 0, stats)
                    if projected_ancestor:
                        self.assertGreater(stats["engine_queries"], 0, stats)

    def test_multi_level_filter_keeps_inner_join_semantics(self):
        self.check_query("SELECT count(*), sum(f.amount) " + CHAIN + "WHERE " + SECTION_FILTER,
                         edges=2, expected=[(7 * 274, 32 * 274)])

    def test_branches_keep_direct_dimension_output(self):
        self.check_query("SELECT i.family, count(*), sum(f.amount) " + BRANCHES
                         + f"WHERE {SECTION_FILTER} AND {SELLER_FILTER} GROUP BY i.family ORDER BY i.family",
                         edges=3, expected=[("a", 2 * 274, 8 * 274), ("b", 2 * 274, 7 * 274)])

    def test_outer_join_keeps_unmatched_rows_on_general_path(self):
        self.check_query("SELECT count(*), sum(f.amount) " + BRANCHES.replace("JOIN seller", "LEFT JOIN seller")
                         + f"WHERE {SECTION_FILTER} AND {SELLER_FILTER}", expected=[(6 * 274, 20 * 274)])

    def test_deep_output_keeps_general_path(self):
        self.check_query("SELECT z.label, count(*), sum(f.amount) " + CHAIN
                         + "GROUP BY z.label ORDER BY z.label")

    def test_projected_ancestor_uses_native_prefilter(self):
        self.check_query("SELECT z.label, count(*), sum(f.amount) " + CHAIN
                         + f"WHERE {SECTION_FILTER} GROUP BY z.label ORDER BY z.label",
                         edges=2, projected_ancestor=True, expected=[("keep", 7 * 274, 32 * 274)])

    def test_casted_fk_equality_keeps_general_path(self):
        casted = CHAIN.replace("c.section_id = z.id", "CAST(c.section_id AS TINYINT) = z.id")
        self.check_query("SELECT count(*), sum(f.amount) " + casted + "WHERE " + SECTION_FILTER,
                         expected=[(7 * 274, 32 * 274)])


if __name__ == "__main__":
    unittest.main()
